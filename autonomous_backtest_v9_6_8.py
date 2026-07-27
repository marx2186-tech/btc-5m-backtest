from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v9_6_8"
REQUEST_PATH = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(ROOT / "request.v9_6_8.json")))
REQUEST = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
STAT = REQUEST["statistical_validation"]
SNAP = REQUEST["snapshot_reproduction"]
WATCH_GATE = REQUEST["watch_gate"]
QUALIFIED_GATE = REQUEST["qualified_gate"]
CANDIDATE_GATE = REQUEST["candidate_gate"]
BLOCK = REQUEST["block_validation"]
TRACK = REQUEST["quad_track_validation"]
SHRINK = REQUEST["evidence_shrinkage_lane"]
GATE = REQUEST["expert_level_gating"]
ORIGINAL = REQUEST["v968_no_monthly_budget_lane"]
RISK_BUDGET = REQUEST["risk_budget_dual_track"]
RATING_SOURCE = REQUEST["rating_data_source"]
ENGINE_NAME = "BTC 5m unified rating source and dual risk-budget validation V9.6.8"

spec = importlib.util.spec_from_file_location("v968_strategy_core", ROOT / "_v968_strategy_core.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load _v968_strategy_core.py")
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


def file_sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def event_fingerprint(events: pd.DataFrame | list[dict[str,Any]]) -> str:
    if isinstance(events,pd.DataFrame):
        rows=events.to_dict("records")
    else: rows=list(events)
    norm=[]
    for t in rows:
        norm.append({
            "signal_i":int(t.get("signal_i",-1)),"exit_i":int(t.get("exit_i",-1)),
            "direction":int(t.get("direction",0)),"policy_key":str(t.get("policy_key","")),
            "month":str(t.get("month","")),"net_r":round(float(t.get("net_r",0.0)),12),
            "reason":str(t.get("reason",t.get("exit_reason","")))
        })
    norm=sorted(norm,key=lambda x:(x["month"],x["signal_i"],x["exit_i"],x["policy_key"]))
    return hashlib.sha256(json.dumps(norm,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def base_metrics(df: pd.DataFrame) -> dict[str,float]:
    if df.empty:
        return {"trades":0,"wins":0,"win_rate":0.0,"avg_win_R":0.0,"avg_loss_R":0.0,"avg_win_loss_ratio":0.0,"profit_factor":0.0,"net_R":0.0,"max_drawdown_R":0.0,"expectancy_R":0.0}
    r=df["net_r"].astype(float).to_numpy(); wins=r[r>0]; losses=-r[r<=0]
    curve=np.cumsum(r); peak=np.maximum.accumulate(np.r_[0.0,curve]); dd=peak[1:]-curve
    aw=float(wins.mean()) if len(wins) else 0.0; al=float(losses.mean()) if len(losses) else 0.0
    return {"trades":int(len(r)),"wins":int(len(wins)),"win_rate":float(len(wins)/len(r)),"avg_win_R":aw,"avg_loss_R":al,
            "avg_win_loss_ratio":float(aw/al) if al>0 else (999.0 if len(wins) else 0.0),
            "profit_factor":float(wins.sum()/losses.sum()) if losses.sum()>0 else (999.0 if wins.sum()>0 else 0.0),
            "net_R":float(r.sum()),"max_drawdown_R":float(dd.max()) if len(dd) else 0.0,"expectancy_R":float(r.mean())}


def wilson_lower(wins:int,n:int)->float:
    if n<=0:return 0.0
    z=float(STAT["wilson_z"]);p=wins/n;den=1+z*z/n
    return float((p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den)


def month_groups(months:list[str],size:int)->list[list[str]]:
    return [months[i:i+size] for i in range(0,len(months),size)]


def evidence_for_expert(df:pd.DataFrame, development_months:list[str], cluster_robust:bool)->tuple[dict[str,Any],list[dict[str,Any]],list[dict[str,Any]]]:
    df=df.copy()
    if not df.empty:
        df["month"]=df["month"].astype(str)
    m=base_metrics(df)
    a=float(STAT["beta_prior_wins"]);b=float(STAT["beta_prior_losses"])
    shrunk=float((m["wins"]+a)/(m["trades"]+a+b))
    monthly=[]
    month_net={}
    positive_month_profit=[]
    for month in development_months:
        q=df[df["month"]==month] if not df.empty else df
        mm=base_metrics(q);n=mm["trades"]
        if n==0: state="NO_SAMPLE"
        elif n<int(STAT["monthly_low_confidence_min_trades"]): state="INSUFFICIENT_SAMPLE"
        elif n<int(STAT["monthly_valid_min_trades"]): state="LOW_CONFIDENCE_OBSERVATION"
        else: state="VALID_MONTH_SAMPLE"
        monthly.append({"month":month,**mm,"sample_state":state,"effective_win_rate":mm["win_rate"] if state=="VALID_MONTH_SAMPLE" else None,"month_validation_passed":bool(state=="VALID_MONTH_SAMPLE")})
        month_net[month]=mm["net_R"];positive_month_profit.append(max(0.0,mm["net_R"]))
    blocks=[]
    valid_blocks=positive_blocks=positive_expectancy_blocks=high_win_rate_blocks=0
    for bi,months in enumerate(month_groups(development_months,int(STAT["block_months"])),1):
        complete=len(months)==int(STAT["block_months"])
        q=df[df["month"].isin(months)] if not df.empty else df
        bm=base_metrics(q)
        valid=complete and bm["trades"]>=int(STAT["block_min_trades"])
        state="VALID_BLOCK" if valid else ("INCOMPLETE_CALENDAR_BLOCK" if not complete else "INSUFFICIENT_BLOCK_SAMPLE")
        positive_expectancy_pass = bool(valid and bm["net_R"]>0 and bm["expectancy_R"]>float(BLOCK["positive_expectancy_min_expectancy_r"]) and bm["profit_factor"]>=float(BLOCK["positive_expectancy_min_profit_factor"]))
        high_win_rate_pass = bool(valid and bm["win_rate"]>=float(BLOCK["high_win_rate_min"]) and bm["wins"]>=int(BLOCK["high_win_rate_min_wins"]))
        if valid:
            valid_blocks+=1
            positive_blocks+=int(bm["net_R"]>0)
            positive_expectancy_blocks+=int(positive_expectancy_pass)
            high_win_rate_blocks+=int(high_win_rate_pass)
        blocks.append({"block_id":bi,"start_month":months[0],"end_month":months[-1],"months":"|".join(months),"calendar_complete":complete,"block_state":state,"block_valid":valid,"positive_expectancy_pass":positive_expectancy_pass,"high_win_rate_pass":high_win_rate_pass,**bm})
    active_months=sum(x["trades"]>0 for x in monthly);positive_months=sum(x["net_R"]>0 for x in monthly)
    worst_month=min((x["net_R"] for x in monthly),default=0.0)
    total_positive=sum(positive_month_profit);max_month_share=max(positive_month_profit,default=0.0)/(total_positive or 1.0)
    positives=df.loc[df["net_r"]>0,"net_r"].astype(float).sort_values(ascending=False).to_numpy() if not df.empty else np.array([])
    max_trade_share=float(positives[0]/positives.sum()) if len(positives) and positives.sum()>0 else 0.0
    sorted_df=df.sort_values("net_r",ascending=False) if not df.empty else df
    best_removed=base_metrics(sorted_df.iloc[1:])["net_R"] if len(sorted_df)>0 else 0.0
    best_two_removed=base_metrics(sorted_df.iloc[2:])["net_R"] if len(sorted_df)>1 else best_removed
    active=[x["month"] for x in monthly if x["trades"]>0]
    loo=[]
    for omitted in active:
        lm=base_metrics(df[df["month"]!=omitted]);loo.append({"omitted_month":omitted,**lm})
    loo_min=min((x["net_R"] for x in loo),default=0.0);loo_share=float(sum(x["net_R"]>0 for x in loo)/len(loo)) if loo else 0.0
    summary={**m,"shrunk_win_rate":shrunk,"wilson_lower":wilson_lower(int(m["wins"]),int(m["trades"])),
             "active_months":active_months,"positive_months":positive_months,"valid_blocks":valid_blocks,"positive_blocks":positive_blocks,"positive_expectancy_blocks":positive_expectancy_blocks,"high_win_rate_blocks":high_win_rate_blocks,
             "worst_month_R":float(worst_month),"max_single_month_profit_share":float(max_month_share),"max_single_trade_profit_share":max_trade_share,
             "best_trade_removed_net_R":float(best_removed),"best_two_trades_removed_net_R":float(best_two_removed),
             "loo_month_min_net_R":float(loo_min),"loo_month_positive_share":loo_share,
             "max_trades_single_month":max((x["trades"] for x in monthly),default=0),"cluster_robust":bool(cluster_robust)}
    return summary,monthly,blocks


def gate_reasons(s:dict[str,Any],gate:dict[str,Any])->list[str]:
    tests=[
      ("min_total_trades",s["trades"]>=int(gate.get("min_total_trades",0)),"总交易不足"),
      ("min_active_months",s["active_months"]>=int(gate.get("min_active_months",0)),"活跃月份不足"),
      ("min_valid_blocks",s["valid_blocks"]>=int(gate.get("min_valid_blocks",0)),"有效三个月验证块不足"),
      ("min_positive_blocks",s["positive_blocks"]>=int(gate.get("min_positive_blocks",0)),"正收益验证块不足"),
      ("min_positive_expectancy_blocks",s["positive_expectancy_blocks"]>=int(gate.get("min_positive_expectancy_blocks",0)),"正期望验证块不足"),
      ("min_high_win_rate_blocks",s["high_win_rate_blocks"]>=int(gate.get("min_high_win_rate_blocks",0)),"高胜率验证块不足"),
      ("min_raw_win_rate",s["win_rate"]>=float(gate.get("min_raw_win_rate",0)),"原始胜率不足"),
      ("min_shrunk_win_rate",s["shrunk_win_rate"]>=float(gate.get("min_shrunk_win_rate",0)),"收缩后胜率不足"),
      ("min_wilson_lower",s["wilson_lower"]>=float(gate.get("min_wilson_lower",0)),"95%胜率可信下界不足"),
      ("min_profit_factor",s["profit_factor"]>=float(gate.get("min_profit_factor",0)),"盈利因子不足"),
      ("min_avg_win_loss_ratio",s["avg_win_loss_ratio"]>=float(gate.get("min_avg_win_loss_ratio",0)),"实际盈亏比不足"),
      ("min_net_r",s["net_R"]>float(gate.get("min_net_r",-1e9)),"累计净R不足"),
      ("max_drawdown_r",s["max_drawdown_R"]<=float(gate.get("max_drawdown_r",1e9)),"最大回撤过大"),
      ("max_worst_month_loss_r",s["worst_month_R"]>=-float(gate.get("max_worst_month_loss_r",1e9)),"最差月份亏损过大"),
      ("min_best_trade_removed_net_r",s["best_trade_removed_net_R"]>float(gate.get("min_best_trade_removed_net_r",-1e9)),"删除最佳交易后不再盈利"),
      ("min_loo_month_positive_share",s["loo_month_positive_share"]>=float(gate.get("min_loo_month_positive_share",0)),"逐月删除稳定性不足"),
      ("min_loo_month_net_r",s["loo_month_min_net_R"]>=float(gate.get("min_loo_month_net_r",-1e9)),"删除单月后的最差净R不足"),
      ("max_single_trade_profit_share",s["max_single_trade_profit_share"]<=float(gate.get("max_single_trade_profit_share",1)),"利润过度依赖单笔"),
      ("max_single_month_profit_share",s["max_single_month_profit_share"]<=float(gate.get("max_single_month_profit_share",1)),"利润过度依赖单月"),
      ("max_trades_per_month",s["max_trades_single_month"]<=int(gate.get("max_trades_per_month",999)),"单月交易过多"),
      ("require_cluster_robust",(not gate.get("require_cluster_robust")) or bool(s["cluster_robust"]),"参数簇不稳健")]
    ignored=set(str(x) for x in GATE.get("rating_gate_ignored_fields", []))
    r=[msg for key,ok,msg in tests if key in gate and key not in ignored and not ok]
    if "max_total_trades" in gate and "max_total_trades" not in ignored and s["trades"]>int(gate["max_total_trades"]):r.append("总交易不再稀疏")
    return r


def tier_for(s:dict[str,Any])->tuple[str,list[str]]:
    q=gate_reasons(s,QUALIFIED_GATE)
    if not q:return "QUALIFIED",[]
    w=gate_reasons(s,WATCH_GATE)
    if not w:return "WATCH",q
    c=gate_reasons(s,CANDIDATE_GATE)
    if not c:return "CANDIDATE",w
    return "REJECTED",c


def empty_trade_frame_like(path:Path)->pd.DataFrame:
    if path.exists():
        try:return pd.read_csv(path).iloc[0:0]
        except Exception:pass
    return pd.DataFrame(columns=["portfolio_type","signal_time_utc","entry_time_utc","exit_time_utc","month","direction","expert","family","setup_group","source_experts","agreement_count","policy_key","calibrated_probability","online_utility_percentile","meta_probability","meta_decision","expected_utility_R","conservative_utility_R","net_R","win","exit_reason","bars"])


def import_reference(request_file:Path):
    old=os.environ.get("BACKTEST_REQUEST_FILE");os.environ["BACKTEST_REQUEST_FILE"]=str(request_file)
    try:
        sp=importlib.util.spec_from_file_location("v961_reference_strategy",ROOT/"_v961_reference_strategy.py")
        if sp is None or sp.loader is None:raise RuntimeError("Cannot load V9.6.1 reference")
        mod=importlib.util.module_from_spec(sp);sys.modules[sp.name]=mod;sp.loader.exec_module(mod);return mod
    finally:
        if old is None:os.environ.pop("BACKTEST_REQUEST_FILE",None)
        else:os.environ["BACKTEST_REQUEST_FILE"]=old


def trace_policy_lane(mod: Any, x: pd.DataFrame, expert: Any, policy: Any, lane: str, months: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild every raw candidate and label the first filter stage that rejected it."""
    idx = mod.expert_indices(x, expert)
    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for eval_month in months:
        if eval_month not in mod.MONTHS:
            continue
        eval_pos = list(mod.MONTHS).index(eval_month)
        train_months = set(mod.MONTHS[:eval_pos])
        model = mod.fit_model(x, idx, expert, policy, train_months)
        labels, exits, net_r, outcome_reasons = mod.outcome_arrays(x, idx, expert, policy.risk)
        candidate_months = x["month"].to_numpy()[idx]
        raw_pos = np.flatnonzero((candidate_months == eval_month) & (exits >= 0))
        if model is None:
            for k in raw_pos:
                signal_i = int(idx[k])
                all_rows.append({
                    "lane": lane, "month": eval_month, "signal_i": signal_i,
                    "exit_i": int(exits[k]), "direction": int(expert.direction),
                    "policy_key": str(policy.key), "day": str(x.index[signal_i].date()),
                    "net_r": float(net_r[k]), "win": bool(net_r[k] > 0),
                    "outcome_reason": int(outcome_reasons[k]), "reason": int(outcome_reasons[k]), "model_available": False,
                    "history_ready": False, "rank_pass": False, "meta_pass": False,
                    "utility_pass": False, "cap_pass": False, "selected": False,
                    "first_reject_stage": "MODEL_UNAVAILABLE"
                })
            continue
        features = list(mod.FEATURE_GROUPS[expert.feature_group])
        history = list(np.asarray(model.calibration_utility_scores, float)[-int(mod.MODEL["online_rank_window"]):])
        history_ready = len(history) >= int(mod.MODEL["online_rank_min_history"])
        month_rows: list[dict[str, Any]] = []
        if len(raw_pos) and history_ready:
            rows = idx[raw_pos]
            base_p = model.base_model.predict_proba(x.iloc[rows][features].to_numpy(float))[:, 1]
            p = mod.calibrate(model, base_p)
            router = mod.router_score(x, rows, expert)
            micro = mod.micro_score(x, rows, expert.direction)
            meta_x = np.column_stack([p, router, micro, p * router, p * micro, router * micro])
            meta_base = model.meta_model.predict_proba(meta_x)[:, 1] if model.meta_model is not None else p
            blend = float(mod.MODEL["meta_blend_weight"])
            brier_penalty = float(mod.MODEL["utility_uncertainty_penalty"]) * np.sqrt(max(float(model.calibration_brier), 0.0))
            sample_penalty = float(mod.MODEL.get("sample_uncertainty_penalty", 0.0)) / np.sqrt(max(int(model.calibration_rows), 1))
            total_penalty = float(brier_penalty + sample_penalty)
            for j, k in enumerate(raw_pos):
                final_p = float(np.clip((1 - blend) * p[j] + blend * meta_base[j], 0.01, 0.99))
                expected = float(final_p * model.avg_win_r - (1 - final_p) * model.avg_loss_r)
                conservative = float(expected - total_penalty)
                hist = np.asarray(history, float)
                pct = float((np.sum(hist <= conservative) + 1) / (len(hist) + 1))
                history.append(conservative)
                history = history[-int(mod.MODEL["online_rank_window"]):]
                severe_router = bool(router[j] < float(mod.MODEL["minimum_router_confidence"]))
                meta_probability_reject = bool(meta_base[j] < float(mod.MODEL["meta_hard_reject_probability"]))
                hard_negative = bool(conservative < float(mod.MODEL["hard_negative_utility_r"]))
                hard_reject = meta_probability_reject or severe_router or hard_negative
                if hard_reject:
                    decision = "REJECT"
                elif meta_base[j] >= float(mod.MODEL["meta_support_probability"]) and micro[j] >= float(mod.MODEL["meta_support_micro"]):
                    decision = "SUPPORT"
                else:
                    decision = "NEUTRAL"
                signal_i = int(idx[k])
                rank_pass = bool(pct >= float(policy.min_percentile))
                meta_pass = bool(rank_pass and decision != "REJECT")
                utility_pass = bool(meta_pass and conservative >= float(policy.min_expected_utility_r))
                if not rank_pass:
                    reject = "ONLINE_RANK"
                elif not meta_pass:
                    if meta_probability_reject:
                        reject = "META_PROBABILITY"
                    elif severe_router:
                        reject = "ROUTER_CONFIDENCE"
                    else:
                        reject = "HARD_NEGATIVE_UTILITY"
                elif not utility_pass:
                    reject = "MINIMUM_UTILITY"
                else:
                    reject = "PENDING_CAP"
                month_rows.append({
                    "lane": lane, "month": eval_month, "signal_i": signal_i,
                    "exit_i": int(exits[k]), "direction": int(expert.direction),
                    "policy_key": str(policy.key), "day": str(x.index[signal_i].date()),
                    "net_r": float(net_r[k]), "win": bool(net_r[k] > 0),
                    "outcome_reason": int(outcome_reasons[k]), "reason": int(outcome_reasons[k]), "model_available": True,
                    "history_ready": True, "base_probability": float(base_p[j]),
                    "probability": float(p[j]), "online_percentile": pct,
                    "rank_threshold": float(policy.min_percentile), "rank_pass": rank_pass,
                    "router": float(router[j]), "router_threshold": float(mod.MODEL["minimum_router_confidence"]),
                    "micro": float(micro[j]), "meta_probability": float(meta_base[j]),
                    "meta_decision": decision, "meta_pass": meta_pass,
                    "expected_utility": expected, "brier_penalty": float(brier_penalty),
                    "sample_uncertainty_penalty": float(sample_penalty), "total_penalty": total_penalty,
                    "utility": conservative, "minimum_utility": float(policy.min_expected_utility_r),
                    "utility_pass": utility_pass, "cap_pass": False, "selected": False,
                    "first_reject_stage": reject
                })
        elif len(raw_pos):
            for k in raw_pos:
                signal_i = int(idx[k])
                month_rows.append({
                    "lane": lane, "month": eval_month, "signal_i": signal_i,
                    "exit_i": int(exits[k]), "direction": int(expert.direction),
                    "policy_key": str(policy.key), "day": str(x.index[signal_i].date()),
                    "net_r": float(net_r[k]), "win": bool(net_r[k] > 0),
                    "outcome_reason": int(outcome_reasons[k]), "reason": int(outcome_reasons[k]), "model_available": True,
                    "history_ready": False, "rank_pass": False, "meta_pass": False,
                    "utility_pass": False, "cap_pass": False, "selected": False,
                    "first_reject_stage": "ONLINE_HISTORY_NOT_READY"
                })
        eligible = [r for r in month_rows if r.get("utility_pass", False)]
        last_exit = -1
        day_count: dict[str, int] = {}
        selected_count = 0
        for row in sorted(eligible, key=lambda z: (int(z["signal_i"]), -float(z["utility"]))):
            if selected_count >= int(policy.monthly_target):
                row["first_reject_stage"] = "MONTHLY_TARGET_CAP"
                continue
            if int(row["signal_i"]) <= last_exit:
                row["first_reject_stage"] = "POSITION_OVERLAP"
                continue
            if day_count.get(str(row["day"]), 0) >= 1:
                row["first_reject_stage"] = "DAILY_CAP"
                continue
            row["cap_pass"] = True
            row["selected"] = True
            row["first_reject_stage"] = "SELECTED"
            selected_count += 1
            day_count[str(row["day"])] = 1
            last_exit = int(row["exit_i"])
            selected_rows.append(dict(row))
        all_rows.extend(month_rows)
    return pd.DataFrame(all_rows), pd.DataFrame(selected_rows)




def allocate_original_rows(month_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Allocate original-quality signals without a monthly first-come quota.

    The allocator is causal and only uses information available when the signal arrives.
    It blocks an overlapping position, an exact same-bar duplicate and excess same-day
    entries. A signal arriving after the previous trade has exited is a new opportunity;
    no cross-trade market-cluster rule is applied.
    """
    rows = sorted(month_rows, key=lambda z: (int(z.get("signal_i", -1)), -float(z.get("utility", -1e9))))
    selected: list[dict[str, Any]] = []
    last_exit = -1
    selected_same_bar: set[tuple[int, int]] = set()
    day_count: dict[str, int] = {}
    for row in rows:
        if not bool(row.get("utility_pass", False)):
            continue
        sig = int(row.get("signal_i", -1))
        direction = int(row.get("direction", 0))
        day = str(row.get("day", ""))
        same_bar_key = (sig, direction)
        if bool(cfg.get("same_bar_dedup", True)) and same_bar_key in selected_same_bar:
            row["first_reject_stage"] = "SAME_BAR_DUPLICATE"
            row["allocation_rule"] = "SAME_BAR_ONLY"
            continue
        if bool(cfg.get("position_overlap_block", True)) and sig <= last_exit:
            row["first_reject_stage"] = "POSITION_OVERLAP"
            row["allocation_rule"] = "POSITION_OVERLAP_ONLY"
            continue
        if day_count.get(day, 0) >= int(cfg.get("max_trades_per_day", 1)):
            row["first_reject_stage"] = "DAILY_CAP"
            row["allocation_rule"] = "DAILY_RISK_ONLY"
            continue
        row["cap_pass"] = True
        row["selected"] = True
        row["first_reject_stage"] = "SELECTED"
        row["allocation_rule"] = "NO_MONTHLY_CAP_NO_CROSS_TRADE_CLUSTER"
        selected.append(dict(row))
        selected_same_bar.add(same_bar_key)
        day_count[day] = day_count.get(day, 0) + 1
        last_exit = int(row.get("exit_i", sig))
    return rows, selected


def trace_no_monthly_budget_lane(mod: Any, x: pd.DataFrame, expert: Any, policy: Any, months: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the frozen original signal-quality definition, then gate at expert level.

    Small-sample uncertainty does not change probability, rank, utility or the decision
    of an individual signal. It is applied later through the expert tier/risk multiplier.
    """
    sentinel = object()
    old_penalty = mod.MODEL.get("sample_uncertainty_penalty", sentinel)
    mod.MODEL["sample_uncertainty_penalty"] = 0.0
    try:
        trace, _ = trace_policy_lane(mod, x, expert, policy, "V9.6.7_NO_MONTHLY_BUDGET_ORIGINAL", months)
    finally:
        if old_penalty is sentinel:
            mod.MODEL.pop("sample_uncertainty_penalty", None)
        else:
            mod.MODEL["sample_uncertainty_penalty"] = old_penalty
    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    if trace.empty:
        return trace, trace.copy()
    for month, q in trace.groupby("month", sort=False):
        month_rows = q.to_dict("records")
        for row in month_rows:
            if bool(row.get("utility_pass", False)):
                row["selected"] = False
                row["cap_pass"] = False
                row["first_reject_stage"] = "PENDING_EXPERT_GATE_ALLOCATION"
            row["sample_uncertainty_changes_signal_selection"] = False
            row["monthly_cap_enabled"] = False
            row["cross_trade_cluster_dedup"] = False
        allocated, chosen = allocate_original_rows(month_rows, ORIGINAL)
        all_rows.extend(allocated)
        selected_rows.extend(chosen)
    return pd.DataFrame(all_rows), pd.DataFrame(selected_rows)


def annotate_expert_gate(events: pd.DataFrame, tier: str) -> pd.DataFrame:
    out = events.copy()
    multiplier = float(GATE["tier_risk_multiplier"][tier])
    mode = str(GATE["tier_execution_mode"][tier])
    out["expert_tier"] = tier
    out["risk_multiplier"] = multiplier
    out["execution_mode"] = mode
    out["portfolio_eligible"] = tier in {"WATCH", "QUALIFIED"}
    out["formal_eligible"] = tier == "QUALIFIED"
    out["risk_adjusted_net_r"] = out["net_r"].astype(float) * multiplier if not out.empty else pd.Series(dtype=float)
    return out


def build_unified_rating_shadow(
    ref: Any,
    rx: pd.DataFrame,
    selected_policy_rows: pd.DataFrame,
    months: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay every selected expert policy with the frozen original signal rules.

    The returned selected trades are the only admissible source for final expert
    evidence tiers in V9.6.8. Core fixed-penalty shadow trades are retained only
    for policy-selection comparison and are forbidden from final tier assignment.
    """
    selected_frames: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    missing: list[str] = []
    for _, row in selected_policy_rows.sort_values("expert_id").iterrows():
        eid = int(row["expert_id"])
        expert_name = str(row["expert"])
        policy_key = str(row["policy_key"])
        rexp = ref.EXPERT_BY_ID[eid]
        matches = [p for p in ref.policy_grid(rexp) if str(p.key) == policy_key]
        if not matches:
            missing.append(f"{eid}:{policy_key}")
            coverage.append({
                "expert_id": eid, "expert": expert_name, "policy_key": policy_key,
                "policy_found_in_reference": False, "raw_candidates": 0,
                "selected_trades": 0, "signal_fingerprint": ""
            })
            continue
        policy = matches[0]
        trace, selected = trace_no_monthly_budget_lane(ref, rx, rexp, policy, months)
        if not selected.empty:
            selected = selected.copy()
            selected["expert_id"] = eid
            selected["expert"] = expert_name
            selected["family"] = str(getattr(rexp, "family", row.get("family", "")))
            selected["setup_group"] = str(getattr(rexp, "setup_group", getattr(rexp, "family", "")))
            selected["rating_source_lane"] = str(RATING_SOURCE["source_lane"])
            selected_frames.append(selected)
        coverage.append({
            "expert_id": eid, "expert": expert_name, "policy_key": policy_key,
            "policy_found_in_reference": True, "raw_candidates": int(len(trace)),
            "selected_trades": int(len(selected)),
            "signal_fingerprint": event_fingerprint(selected)
        })
    coverage_df = pd.DataFrame(coverage)
    if missing and bool(RATING_SOURCE.get("policy_key_coverage_must_be_complete", True)):
        raise RuntimeError("Reference policy coverage incomplete: " + ",".join(missing))
    shadow = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    if not shadow.empty:
        shadow = shadow.sort_values(["month", "signal_i", "expert_id"]).reset_index(drop=True)
    return shadow, coverage_df


def attach_final_expert_gate(shadow: pd.DataFrame, tiers: pd.DataFrame) -> pd.DataFrame:
    if shadow.empty:
        return shadow.copy()
    gate_cols = tiers[["expert_id", "policy_key", "tier"]].copy()
    gate_cols["policy_key"] = gate_cols["policy_key"].astype(str)
    out = shadow.copy()
    out["policy_key"] = out["policy_key"].astype(str)
    out = out.merge(gate_cols, on=["expert_id", "policy_key"], how="left", validate="many_to_one")
    if out["tier"].isna().any():
        bad = out.loc[out["tier"].isna(), ["expert_id", "policy_key"]].drop_duplicates().to_dict("records")
        raise RuntimeError(f"Missing final tier for rating shadow rows: {bad}")
    out["expert_tier"] = out["tier"].astype(str)
    out["risk_multiplier"] = out["expert_tier"].map(lambda t: float(GATE["tier_risk_multiplier"][t]))
    out["execution_mode"] = out["expert_tier"].map(lambda t: str(GATE["tier_execution_mode"][t]))
    out["portfolio_eligible"] = out["expert_tier"].isin(["WATCH", "QUALIFIED"])
    out["formal_eligible"] = out["expert_tier"].eq("QUALIFIED")
    out["risk_adjusted_net_r"] = out["net_r"].astype(float) * out["risk_multiplier"].astype(float)
    return out


def build_unified_portfolio(gated_shadow: pd.DataFrame, allowed_tiers: set[str], risk_adjusted: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if gated_shadow.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    source = gated_shadow[gated_shadow["expert_tier"].isin(allowed_tiers)].copy()
    if source.empty:
        return source, pd.DataFrame(), pd.DataFrame()
    all_trades: list[dict[str, Any]] = []
    all_dedup: list[dict[str, Any]] = []
    all_conflicts: list[dict[str, Any]] = []
    for month, month_df in source.groupby("month", sort=True):
        events_by_expert: dict[int, list[dict[str, Any]]] = {}
        for eid, q in month_df.groupby("expert_id", sort=False):
            records = q.to_dict("records")
            for rec in records:
                rec["raw_net_r"] = float(rec["net_r"])
                if risk_adjusted:
                    rec["net_r"] = float(rec["risk_adjusted_net_r"])
            events_by_expert[int(eid)] = records
        trades, dedup, conflicts = core.combine_month(events_by_expert)
        for rec in trades:
            rec["month"] = str(month)
        for rec in dedup:
            rec["month"] = str(month)
        for rec in conflicts:
            rec["month"] = str(month)
        all_trades.extend(trades); all_dedup.extend(dedup); all_conflicts.extend(conflicts)
    return pd.DataFrame(all_trades), pd.DataFrame(all_dedup), pd.DataFrame(all_conflicts)

def allocate_evidence_rows(month_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chronological online allocation with separate strong/neutral quotas.

    Strong signals never consume the neutral quota. The cluster rule is backward-looking
    only, so the allocator does not use future candidates or outcomes.
    """
    rows = sorted(month_rows, key=lambda z: (int(z.get("signal_i", -1)), -float(z.get("evidence_utility", -1e9))))
    selected: list[dict[str, Any]] = []
    last_exit = -1
    selected_signals: list[int] = []
    strong_month = neutral_month = 0
    strong_day: dict[str, int] = {}
    neutral_day: dict[str, int] = {}
    total_day: dict[str, int] = {}
    window = int(cfg["market_cluster_window_bars"])
    for row in rows:
        if not bool(row.get("utility_pass", False)):
            continue
        day = str(row.get("day", ""))
        sig = int(row["signal_i"])
        signal_class = str(row.get("signal_class", "NEUTRAL"))
        if sig <= last_exit:
            row["first_reject_stage"] = "POSITION_OVERLAP"
            row["allocation_bucket"] = signal_class
            continue
        if any(0 <= sig - prior <= window for prior in selected_signals):
            row["first_reject_stage"] = "MARKET_CLUSTER_DUPLICATE"
            row["allocation_bucket"] = signal_class
            continue
        if total_day.get(day, 0) >= int(cfg["total_daily_cap"]):
            row["first_reject_stage"] = "TOTAL_DAILY_CAP"
            row["allocation_bucket"] = signal_class
            continue
        if signal_class == "STRONG":
            if strong_month >= int(cfg["strong_monthly_safety_cap"]):
                row["first_reject_stage"] = "STRONG_SAFETY_CAP"
                row["allocation_bucket"] = "STRONG"
                continue
            if strong_day.get(day, 0) >= int(cfg["strong_daily_cap"]):
                row["first_reject_stage"] = "STRONG_DAILY_CAP"
                row["allocation_bucket"] = "STRONG"
                continue
            strong_month += 1
            strong_day[day] = strong_day.get(day, 0) + 1
            row["allocation_bucket"] = "STRONG"
        else:
            if neutral_month >= int(cfg["neutral_monthly_cap"]):
                row["first_reject_stage"] = "NEUTRAL_MONTHLY_CAP"
                row["allocation_bucket"] = "NEUTRAL"
                continue
            if neutral_day.get(day, 0) >= int(cfg["neutral_daily_cap"]):
                row["first_reject_stage"] = "NEUTRAL_DAILY_CAP"
                row["allocation_bucket"] = "NEUTRAL"
                continue
            neutral_month += 1
            neutral_day[day] = neutral_day.get(day, 0) + 1
            row["allocation_bucket"] = "NEUTRAL"
        total_day[day] = total_day.get(day, 0) + 1
        row["cap_pass"] = True
        row["selected"] = True
        row["first_reject_stage"] = "SELECTED"
        selected.append(dict(row))
        selected_signals.append(sig)
        last_exit = int(row["exit_i"])
    return rows, selected


def trace_evidence_shrinkage_lane(mod: Any, x: pd.DataFrame, expert: Any, policy: Any, lane: str, months: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Third frozen lane: shrink expected utility toward zero using evidence reliability.

    Unlike the fixed-penalty lane, no constant sample penalty is subtracted from each
    candidate. The lane uses separate strong and neutral quotas and online cluster dedup.
    """
    idx = mod.expert_indices(x, expert)
    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for eval_month in months:
        if eval_month not in mod.MONTHS:
            continue
        eval_pos = list(mod.MONTHS).index(eval_month)
        train_months = set(mod.MONTHS[:eval_pos])
        model = mod.fit_model(x, idx, expert, policy, train_months)
        labels, exits, net_r, outcome_reasons = mod.outcome_arrays(x, idx, expert, policy.risk)
        candidate_months = x["month"].to_numpy()[idx]
        raw_pos = np.flatnonzero((candidate_months == eval_month) & (exits >= 0))
        if model is None:
            for k in raw_pos:
                signal_i = int(idx[k])
                all_rows.append({
                    "lane": lane, "month": eval_month, "signal_i": signal_i, "exit_i": int(exits[k]),
                    "direction": int(expert.direction), "policy_key": str(policy.key),
                    "day": str(x.index[signal_i].date()), "net_r": float(net_r[k]),
                    "win": bool(net_r[k] > 0), "outcome_reason": int(outcome_reasons[k]),
                    "reason": int(outcome_reasons[k]), "model_available": False,
                    "history_ready": False, "rank_pass": False, "meta_pass": False,
                    "utility_pass": False, "cap_pass": False, "selected": False,
                    "first_reject_stage": "MODEL_UNAVAILABLE"
                })
            continue
        features = list(mod.FEATURE_GROUPS[expert.feature_group])
        evidence_weight = float(model.calibration_rows / (model.calibration_rows + float(SHRINK["evidence_prior_rows"])))
        brier_penalty = float(mod.MODEL["utility_uncertainty_penalty"]) * np.sqrt(max(float(model.calibration_brier), 0.0))
        calibration_weight = float(np.clip(1.0 - brier_penalty / max(float(SHRINK["brier_penalty_reference"]), 1e-9), float(SHRINK["calibration_weight_floor"]), 1.0))
        reliability = float(evidence_weight * calibration_weight)
        raw_history = np.asarray(model.calibration_utility_scores, float)[-int(mod.MODEL["online_rank_window"]):]
        history = list(raw_history * reliability)
        history_ready = len(history) >= int(mod.MODEL["online_rank_min_history"])
        month_rows: list[dict[str, Any]] = []
        if len(raw_pos) and history_ready:
            rows = idx[raw_pos]
            base_p = model.base_model.predict_proba(x.iloc[rows][features].to_numpy(float))[:, 1]
            p = mod.calibrate(model, base_p)
            router = mod.router_score(x, rows, expert)
            micro = mod.micro_score(x, rows, expert.direction)
            meta_x = np.column_stack([p, router, micro, p * router, p * micro, router * micro])
            meta_base = model.meta_model.predict_proba(meta_x)[:, 1] if model.meta_model is not None else p
            blend = float(mod.MODEL["meta_blend_weight"])
            for j, k in enumerate(raw_pos):
                final_p = float(np.clip((1 - blend) * p[j] + blend * meta_base[j], 0.01, 0.99))
                raw_expected = float(final_p * model.avg_win_r - (1 - final_p) * model.avg_loss_r)
                evidence_utility = float(raw_expected * reliability)
                hist = np.asarray(history, float)
                pct = float((np.sum(hist <= evidence_utility) + 1) / (len(hist) + 1))
                history.append(evidence_utility)
                history = history[-int(mod.MODEL["online_rank_window"]):]
                severe_router = bool(router[j] < float(mod.MODEL["minimum_router_confidence"]))
                meta_probability_reject = bool(meta_base[j] < float(mod.MODEL["meta_hard_reject_probability"]))
                hard_negative = bool(raw_expected < float(mod.MODEL["hard_negative_utility_r"]))
                if meta_probability_reject or severe_router or hard_negative:
                    decision = "REJECT"
                elif meta_base[j] >= float(mod.MODEL["meta_support_probability"]) and micro[j] >= float(mod.MODEL["meta_support_micro"]):
                    decision = "SUPPORT"
                else:
                    decision = "NEUTRAL"
                signal_i = int(idx[k])
                rank_pass = bool(pct >= float(policy.min_percentile))
                meta_pass = bool(rank_pass and decision != "REJECT")
                utility_pass = bool(meta_pass and evidence_utility >= float(SHRINK["minimum_evidence_utility_r"]))
                strong = bool(utility_pass and (
                    decision == "SUPPORT" or (
                        pct >= float(SHRINK["strong_min_percentile"])
                        and meta_base[j] >= float(SHRINK["strong_min_meta_probability"])
                        and evidence_utility >= float(SHRINK["strong_min_evidence_utility_r"])
                    )
                ))
                if not rank_pass:
                    reject = "ONLINE_RANK"
                elif not meta_pass:
                    if meta_probability_reject: reject = "META_PROBABILITY"
                    elif severe_router: reject = "ROUTER_CONFIDENCE"
                    else: reject = "HARD_NEGATIVE_UTILITY"
                elif not utility_pass:
                    reject = "EVIDENCE_UTILITY"
                else:
                    reject = "PENDING_ALLOCATION"
                month_rows.append({
                    "lane": lane, "month": eval_month, "signal_i": signal_i,
                    "exit_i": int(exits[k]), "direction": int(expert.direction),
                    "policy_key": str(policy.key), "day": str(x.index[signal_i].date()),
                    "net_r": float(net_r[k]), "win": bool(net_r[k] > 0),
                    "outcome_reason": int(outcome_reasons[k]), "reason": int(outcome_reasons[k]),
                    "model_available": True, "history_ready": True,
                    "base_probability": float(base_p[j]), "probability": float(p[j]),
                    "online_percentile": pct, "rank_threshold": float(policy.min_percentile),
                    "rank_pass": rank_pass, "router": float(router[j]),
                    "router_threshold": float(mod.MODEL["minimum_router_confidence"]),
                    "micro": float(micro[j]), "meta_probability": float(meta_base[j]),
                    "meta_decision": decision, "meta_pass": meta_pass,
                    "raw_expected_utility": raw_expected, "expected_utility": raw_expected,
                    "evidence_weight": evidence_weight, "calibration_weight": calibration_weight,
                    "reliability_weight": reliability, "brier_penalty_diagnostic": brier_penalty,
                    "sample_uncertainty_penalty": 0.0, "fixed_sample_penalty": 0.0,
                    "evidence_utility": evidence_utility, "utility": evidence_utility,
                    "minimum_utility": float(SHRINK["minimum_evidence_utility_r"]),
                    "signal_class": "STRONG" if strong else "NEUTRAL",
                    "utility_pass": utility_pass, "cap_pass": False, "selected": False,
                    "first_reject_stage": reject
                })
        elif len(raw_pos):
            for k in raw_pos:
                signal_i = int(idx[k])
                month_rows.append({
                    "lane": lane, "month": eval_month, "signal_i": signal_i,
                    "exit_i": int(exits[k]), "direction": int(expert.direction),
                    "policy_key": str(policy.key), "day": str(x.index[signal_i].date()),
                    "net_r": float(net_r[k]), "win": bool(net_r[k] > 0),
                    "outcome_reason": int(outcome_reasons[k]), "reason": int(outcome_reasons[k]),
                    "model_available": True, "history_ready": False, "rank_pass": False,
                    "meta_pass": False, "utility_pass": False, "cap_pass": False,
                    "selected": False, "first_reject_stage": "ONLINE_HISTORY_NOT_READY"
                })
        allocated, selected = allocate_evidence_rows(month_rows, SHRINK)
        all_rows.extend(allocated)
        selected_rows.extend(selected)
    return pd.DataFrame(all_rows), pd.DataFrame(selected_rows)

def block_quality_rows(events: pd.DataFrame, lane: str, months: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    size = int(STAT["block_months"])
    for block_id, block_months in enumerate(month_groups(months, size), 1):
        complete = len(block_months) == size
        q = events[events["month"].astype(str).isin(block_months)].copy() if not events.empty else pd.DataFrame()
        m = base_metrics(q)
        valid = bool(complete and m["trades"] >= int(STAT["block_min_trades"]))
        positive_expectancy = bool(valid and m["net_R"] > 0 and m["expectancy_R"] > float(BLOCK["positive_expectancy_min_expectancy_r"]) and m["profit_factor"] >= float(BLOCK["positive_expectancy_min_profit_factor"]))
        high_win_rate = bool(valid and m["win_rate"] >= float(BLOCK["high_win_rate_min"]) and m["wins"] >= int(BLOCK["high_win_rate_min_wins"]))
        rows.append({
            "lane": lane, "block_id": block_id, "start_month": block_months[0],
            "end_month": block_months[-1], "months": "|".join(block_months),
            "calendar_complete": complete, "block_valid": valid,
            "positive_expectancy_pass": positive_expectancy,
            "high_win_rate_pass": high_win_rate, **m
        })
    return rows


def attribution_self_test() -> None:
    sample = pd.DataFrame([
        {"month": "2025-05", "net_r": 1.5},
        {"month": "2025-06", "net_r": -0.5},
        {"month": "2025-07", "net_r": 1.5},
    ])
    b = block_quality_rows(sample, "TEST", ["2025-05", "2025-06", "2025-07"])[0]
    assert b["block_valid"] and b["positive_expectancy_pass"] and b["high_win_rate_pass"]
    assert TRACK["winner_selection_enabled"] is False and TRACK["seen_oos_must_not_select_winner"] is True
    print("V968_QUAD_TRACK_SELF_TEST_OK")



def statistical_self_test()->None:
    rows=[]
    for i,r in enumerate([1.7,-0.3,1.6,-0.4,1.5,-0.2,1.4,-0.2]):
        rows.append({"month":["2025-05","2025-06","2025-07","2025-08"][i//2],"net_r":r})
    s,monthly,blocks=evidence_for_expert(pd.DataFrame(rows),list(core.DEVELOPMENT_MONTHS),True)
    assert s["trades"]==8 and s["best_trade_removed_net_R"]>0
    assert monthly[0]["sample_state"]=="INSUFFICIENT_SAMPLE"
    one=pd.DataFrame([{"month":"2025-05","net_r":1.5}]);o,_,_=evidence_for_expert(one,list(core.DEVELOPMENT_MONTHS),True)
    assert o["shrunk_win_rate"]<0.60 and o["wilson_lower"]<0.30
    print("V968_STATISTICAL_VALIDATION_SELF_TEST_OK")



def shrinkage_allocation_self_test() -> None:
    cfg=dict(SHRINK)
    rows=[
      {"signal_i":10,"exit_i":11,"day":"2026-01-01","utility_pass":True,"evidence_utility":0.03,"signal_class":"NEUTRAL","selected":False,"cap_pass":False,"first_reject_stage":"PENDING_ALLOCATION"},
      {"signal_i":20,"exit_i":21,"day":"2026-01-02","utility_pass":True,"evidence_utility":0.04,"signal_class":"NEUTRAL","selected":False,"cap_pass":False,"first_reject_stage":"PENDING_ALLOCATION"},
      {"signal_i":30,"exit_i":31,"day":"2026-01-03","utility_pass":True,"evidence_utility":0.05,"signal_class":"NEUTRAL","selected":False,"cap_pass":False,"first_reject_stage":"PENDING_ALLOCATION"},
      {"signal_i":40,"exit_i":41,"day":"2026-01-04","utility_pass":True,"evidence_utility":0.12,"signal_class":"STRONG","selected":False,"cap_pass":False,"first_reject_stage":"PENDING_ALLOCATION"},
      {"signal_i":43,"exit_i":44,"day":"2026-01-04","utility_pass":True,"evidence_utility":0.15,"signal_class":"STRONG","selected":False,"cap_pass":False,"first_reject_stage":"PENDING_ALLOCATION"},
    ]
    allocated, selected=allocate_evidence_rows(rows,cfg)
    assert [r["signal_i"] for r in selected]==[10,20,40]
    assert next(r for r in allocated if r["signal_i"]==30)["first_reject_stage"]=="NEUTRAL_MONTHLY_CAP"
    assert next(r for r in allocated if r["signal_i"]==43)["first_reject_stage"] in {"MARKET_CLUSTER_DUPLICATE","STRONG_DAILY_CAP"}
    n=24.0; w=n/(n+float(cfg["evidence_prior_rows"]))
    assert 0.0 < w < 1.0 and float(cfg["fixed_sample_penalty_r"])==0.0
    assert TRACK["winner_selection_enabled"] is False and TRACK["seen_oos_must_not_select_winner"] is True
    print("V968_EVIDENCE_SHRINKAGE_AND_ALLOCATION_OK")


def original_gate_self_test() -> None:
    rows = [
        {"signal_i": 10, "exit_i": 12, "direction": -1, "day": "2026-01-01", "utility": 0.05, "utility_pass": True, "net_r": 1.0, "selected": False, "cap_pass": False},
        {"signal_i": 10, "exit_i": 11, "direction": -1, "day": "2026-01-01", "utility": 0.04, "utility_pass": True, "net_r": -0.2, "selected": False, "cap_pass": False},
        {"signal_i": 11, "exit_i": 13, "direction": -1, "day": "2026-01-01", "utility": 0.06, "utility_pass": True, "net_r": -0.3, "selected": False, "cap_pass": False},
        {"signal_i": 14, "exit_i": 15, "direction": -1, "day": "2026-01-01", "utility": 0.07, "utility_pass": True, "net_r": 1.2, "selected": False, "cap_pass": False},
        {"signal_i": 20, "exit_i": 21, "direction": -1, "day": "2026-01-02", "utility": 0.08, "utility_pass": True, "net_r": 1.3, "selected": False, "cap_pass": False},
        {"signal_i": 30, "exit_i": 31, "direction": -1, "day": "2026-01-03", "utility": 0.09, "utility_pass": True, "net_r": 1.4, "selected": False, "cap_pass": False},
        {"signal_i": 40, "exit_i": 41, "direction": -1, "day": "2026-01-04", "utility": 0.10, "utility_pass": True, "net_r": 1.5, "selected": False, "cap_pass": False},
    ]
    allocated, selected = allocate_original_rows(rows, ORIGINAL)
    assert [r["signal_i"] for r in selected] == [10, 20, 30, 40]
    assert next(r for r in allocated if r["signal_i"] == 11)["first_reject_stage"] == "POSITION_OVERLAP"
    assert next(r for r in allocated if r["signal_i"] == 14)["first_reject_stage"] == "DAILY_CAP"
    assert not any(r.get("first_reject_stage") in {"MONTHLY_TARGET_CAP", "MARKET_CLUSTER_DUPLICATE"} for r in allocated)
    base = pd.DataFrame(selected)
    fps = []
    for tier in ["CANDIDATE", "WATCH", "QUALIFIED"]:
        fps.append(event_fingerprint(annotate_expert_gate(base, tier)))
    assert len(set(fps)) == 1
    assert GATE["signal_selection_independent_of_tier"] is True
    assert ORIGINAL["monthly_cap_enabled"] is False and ORIGINAL["cross_trade_cluster_dedup"] is False
    print("V968_ORIGINAL_SELECTION_AND_EXPERT_GATE_OK")


def rating_source_self_test() -> None:
    months=list(core.DEVELOPMENT_MONTHS)
    rating_rows=[]
    for i,r in enumerate([1.8,1.7,-1.2,-0.1,1.9,1.8,-1.1,-0.2,-1.2,-0.1]):
        rating_rows.append({"month":["2025-12","2026-01","2026-04"][min(i//4,2)],"net_r":r})
    rating_summary,_,_=evidence_for_expert(pd.DataFrame(rating_rows),months,True)
    core_summary,_,_=evidence_for_expert(pd.DataFrame(rating_rows[:6]),months,True)
    assert rating_summary["trades"]==10 and core_summary["trades"]==6
    test_gate={"min_total_trades":8,"max_trades_per_month":1}
    reasons=gate_reasons(rating_summary,test_gate)
    assert "总交易不足" not in reasons and "单月交易过多" not in reasons
    assert GATE["rating_ignores_execution_budget"] is True
    assert "max_trades_per_month" in set(GATE["rating_gate_ignored_fields"])
    assert RATING_SOURCE["forbid_core_fixed_penalty_shadow_for_final_tier"] is True
    assert RISK_BUDGET["winner_selection_enabled"] is False
    print("V968_UNIFIED_RATING_SOURCE_AND_DUAL_BUDGET_OK")

def main()->None:
    core.main()
    RESULTS.mkdir(exist_ok=True)
    core_shadow_path = RESULTS / "expert_shadow_trades.csv"
    core_shadow = pd.read_csv(core_shadow_path) if core_shadow_path.exists() and core_shadow_path.stat().st_size else pd.DataFrame()
    initial_tiers = pd.read_csv(RESULTS / "expert_tier_audit.csv")
    dev_months = list(core.DEVELOPMENT_MONTHS)
    trace_months = dev_months + [core.OOS_MONTH]

    # Load the common official dataset and the exact V9.6.1 reference runtime once.
    raw,audit=core.base.load_official_data();eth,ea=core.base.load_auxiliary_kline("ETHUSDT","klines");premium,pa=core.base.load_auxiliary_kline(core.SYMBOL,"premiumIndexKlines");funding,fa=core.base.load_funding_rate()
    x,align=core.base.add_features(raw,eth,premium,funding);x=core.add_sparse_masks(x)
    ref_request=ROOT/str(SNAP["reference_request_file"]);ref_request.write_text(json.dumps(REQUEST["legacy_v961_reference_request"],ensure_ascii=False,indent=2),encoding="utf-8")
    ref=import_reference(ref_request)
    rx,ralign=ref.base.add_features(raw,eth,premium,funding);rx=ref.add_sparse_masks(rx)

    # V9.6.8 invariant: final expert evidence is rebuilt from the complete no-monthly-budget
    # original-selection shadow for every selected expert policy. The fixed-penalty core
    # shadow remains policy-selection evidence only and cannot assign the final tier.
    rating_shadow, coverage_df = build_unified_rating_shadow(ref, rx, initial_tiers, trace_months)
    rating_shadow.to_csv(RESULTS / "v968_unified_rating_shadow_trades.csv", index=False)
    coverage_df.to_csv(RESULTS / "rating_source_policy_coverage.csv", index=False)

    source_compare=[]
    for _, row in initial_tiers.iterrows():
        eid=int(row["expert_id"]); policy=str(row["policy_key"]); expert=str(row["expert"])
        cq=core_shadow[(core_shadow.get("expert_id",pd.Series(dtype=int))==eid)&(core_shadow.get("policy_key",pd.Series(dtype=str)).astype(str)==policy)&(core_shadow.get("month",pd.Series(dtype=str)).astype(str).isin(dev_months))].copy() if not core_shadow.empty else pd.DataFrame()
        rq=rating_shadow[(rating_shadow.get("expert_id",pd.Series(dtype=int))==eid)&(rating_shadow.get("policy_key",pd.Series(dtype=str)).astype(str)==policy)&(rating_shadow.get("month",pd.Series(dtype=str)).astype(str).isin(dev_months))].copy() if not rating_shadow.empty else pd.DataFrame()
        cm=base_metrics(cq); rm=base_metrics(rq)
        source_compare.append({
            "expert_id":eid,"expert":expert,"policy_key":policy,
            "core_fixed_penalty_trades":cm["trades"],"core_fixed_penalty_wins":cm["wins"],"core_fixed_penalty_net_R":cm["net_R"],
            "unified_rating_trades":rm["trades"],"unified_rating_wins":rm["wins"],"unified_rating_net_R":rm["net_R"],
            "trade_count_delta":rm["trades"]-cm["trades"],"net_R_delta":rm["net_R"]-cm["net_R"],
            "core_signal_fingerprint":event_fingerprint(cq),"rating_signal_fingerprint":event_fingerprint(rq),
            "final_rating_source":str(RATING_SOURCE["source_lane"])
        })
    source_compare_df=pd.DataFrame(source_compare)
    source_compare_df.to_csv(RESULTS / "rating_source_comparison.csv", index=False)

    # Recompute all statistical evidence and final tiers from the unified rating shadow.
    new_rows=[];monthly_rows=[];block_rows=[];bayes_rows=[];remove_rows=[];loo_rows=[];concentration_rows=[]
    summaries={}
    for _,row in initial_tiers.iterrows():
        eid=int(row["expert_id"]);expert=str(row["expert"]);policy=str(row["policy_key"])
        q=rating_shadow[(rating_shadow.get("expert_id",pd.Series(dtype=int))==eid)&(rating_shadow.get("policy_key",pd.Series(dtype=str)).astype(str)==policy)&(rating_shadow.get("month",pd.Series(dtype=str)).astype(str).isin(dev_months))].copy() if not rating_shadow.empty else pd.DataFrame()
        summary,months,blocks=evidence_for_expert(q,dev_months,bool(row.get("selected_cluster_robust",False)))
        tier,reasons=tier_for(summary);summaries[eid]=summary
        base=row.to_dict();base.update(summary);base["core_policy_selection_tier"]=base.get("tier");base["legacy_v963_tier"]=base.get("tier");base["tier"]=tier;base["rejection_reason"]="|".join(reasons);base["candidate_shadow_only"]=tier=="CANDIDATE";base["selected_research"]=tier in {"WATCH","QUALIFIED"};base["selected_qualified"]=tier=="QUALIFIED";base["final_rating_source"]=str(RATING_SOURCE["source_lane"]);base["rating_ignores_execution_budget"]=bool(GATE["rating_ignores_execution_budget"]);new_rows.append(base)
        for m in months:monthly_rows.append({"expert_id":eid,"expert":expert,"policy_key":policy,"tier":tier,"rating_source":str(RATING_SOURCE["source_lane"]),**m})
        for b in blocks:block_rows.append({"expert_id":eid,"expert":expert,"policy_key":policy,"tier":tier,"rating_source":str(RATING_SOURCE["source_lane"]),**b})
        bayes_rows.append({"expert_id":eid,"expert":expert,"policy_key":policy,"tier":tier,"trades":summary["trades"],"wins":summary["wins"],"raw_win_rate":summary["win_rate"],"beta_prior_wins":STAT["beta_prior_wins"],"beta_prior_losses":STAT["beta_prior_losses"],"posterior_mean_win_rate":summary["shrunk_win_rate"],"wilson_95_lower":summary["wilson_lower"]})
        remove_rows.append({"expert_id":eid,"expert":expert,"policy_key":policy,"tier":tier,"net_R":summary["net_R"],"best_trade_removed_net_R":summary["best_trade_removed_net_R"],"best_two_trades_removed_net_R":summary["best_two_trades_removed_net_R"]})
        concentration_rows.append({"expert_id":eid,"expert":expert,"policy_key":policy,"tier":tier,"max_single_trade_profit_share":summary["max_single_trade_profit_share"],"max_single_month_profit_share":summary["max_single_month_profit_share"]})
        active=[m for m in dev_months if not q[q["month"].astype(str)==m].empty] if not q.empty else []
        for omitted in active:
            lm=base_metrics(q[q["month"].astype(str)!=omitted]);loo_rows.append({"expert_id":eid,"expert":expert,"policy_key":policy,"tier":tier,"omitted_month":omitted,**lm})
    new_tiers=pd.DataFrame(new_rows);new_tiers.to_csv(RESULTS/"expert_tier_audit.csv",index=False);new_tiers.to_csv(RESULTS/"selection_audit.csv",index=False)
    pd.DataFrame(monthly_rows).to_csv(RESULTS/"monthly_sample_validity.csv",index=False)
    pd.DataFrame(block_rows).to_csv(RESULTS/"time_block_validation.csv",index=False)
    pd.DataFrame(bayes_rows).to_csv(RESULTS/"bayesian_win_rate_audit.csv",index=False)
    pd.DataFrame(remove_rows).to_csv(RESULTS/"best_trade_removal_audit.csv",index=False)
    pd.DataFrame(loo_rows).to_csv(RESULTS/"leave_one_month_out_audit.csv",index=False)
    pd.DataFrame(concentration_rows).to_csv(RESULTS/"profit_concentration_audit.csv",index=False)

    # Attach final tiers to exactly the same rating shadow. Risk changes, signals do not.
    gated_rating_shadow = attach_final_expert_gate(rating_shadow, new_tiers)
    gated_rating_shadow.to_csv(RESULTS / "v968_unified_gated_shadow_trades.csv", index=False)
    research_execution = gated_rating_shadow[gated_rating_shadow.get("risk_multiplier",pd.Series(dtype=float)).astype(float)>0].copy() if not gated_rating_shadow.empty else gated_rating_shadow.copy()
    formal_ok = int((new_tiers["tier"] == "QUALIFIED").sum()) >= int(GATE["minimum_qualified_experts_for_formal"])
    formal_execution = gated_rating_shadow[gated_rating_shadow.get("formal_eligible",pd.Series(dtype=bool)).fillna(False).astype(bool)].copy() if formal_ok and not gated_rating_shadow.empty else gated_rating_shadow.iloc[0:0].copy()
    research_execution.to_csv(RESULTS / "v968_research_execution_trades.csv", index=False)
    formal_execution.to_csv(RESULTS / "v968_formal_execution_trades.csv", index=False)

    research_portfolio, research_dedup, research_conflicts = build_unified_portfolio(gated_rating_shadow, {"WATCH","QUALIFIED"}, risk_adjusted=True)
    qualified_portfolio, qualified_dedup, qualified_conflicts = build_unified_portfolio(gated_rating_shadow, {"QUALIFIED"}, risk_adjusted=False)
    fallback_cols=list(gated_rating_shadow.columns) if len(gated_rating_shadow.columns) else ["month","net_r","expert","expert_id","policy_key"]
    if research_portfolio.empty: research_portfolio=pd.DataFrame(columns=fallback_cols)
    if qualified_portfolio.empty: qualified_portfolio=pd.DataFrame(columns=fallback_cols)
    for name in ["watch_portfolio_trades.csv","research_portfolio_trades.csv","portfolio_trades.csv","trades.csv"]:
        research_portfolio.to_csv(RESULTS/name,index=False)
    qualified_portfolio.to_csv(RESULTS/"qualified_portfolio_trades.csv",index=False)
    research_dedup.to_csv(RESULTS/"v968_research_portfolio_dedup.csv",index=False)
    research_conflicts.to_csv(RESULTS/"v968_research_portfolio_conflicts.csv",index=False)
    qualified_dedup.to_csv(RESULTS/"v968_qualified_portfolio_dedup.csv",index=False)
    qualified_conflicts.to_csv(RESULTS/"v968_qualified_portfolio_conflicts.csv",index=False)

    # Blind evidence uses the same unified shadow source, never the core fixed-penalty ledger.
    blind=[]
    for _,row in new_tiers.iterrows():
        eid=int(row["expert_id"]);policy=str(row["policy_key"]);expert=str(row["expert"])
        q=rating_shadow[(rating_shadow.get("expert_id",pd.Series(dtype=int))==eid)&(rating_shadow.get("policy_key",pd.Series(dtype=str)).astype(str)==policy)&(rating_shadow.get("month",pd.Series(dtype=str)).astype(str)==core.OOS_MONTH)].copy() if not rating_shadow.empty else pd.DataFrame()
        bm=base_metrics(q);months_elapsed=1
        blind.append({"expert_id":eid,"expert":expert,"policy_key":policy,"development_tier":row["tier"],"rating_source":str(RATING_SOURCE["source_lane"]),"blind_months_elapsed":months_elapsed,"blind_trades":bm["trades"],"blind_wins":bm["wins"],"blind_raw_win_rate":bm["win_rate"],"watch_evidence_complete":months_elapsed>=int(STAT["blind_min_calendar_months_watch"]) and bm["trades"]>=int(STAT["blind_min_trades_watch"]),"qualified_evidence_complete":months_elapsed>=int(STAT["blind_min_calendar_months_qualified"]) and bm["trades"]>=int(STAT["blind_min_trades_qualified"])})
    blind_payload={"diagnostic_oos_months":[core.OOS_MONTH],"rule":"minimum calendar months AND minimum trades must both be met","rating_source":str(RATING_SOURCE["source_lane"]),"experts":blind}
    (RESULTS/"blind_evidence_progress.json").write_text(json.dumps(blind_payload,ensure_ascii=False,indent=2),encoding="utf-8")

    shadow = core_shadow

    # Build deterministic full snapshot and replay the seed policy.
    seed_id=int(SNAP["seed_expert_id"]);seed_row=new_tiers[new_tiers["expert_id"]==seed_id].iloc[0];seed_key=str(seed_row["policy_key"])
    expert=core.EXPERT_BY_ID[seed_id];policies=core.policy_grid(expert);policy=next(p for p in policies if p.key==seed_key)
    replay=core.evaluate_candidate(x,expert,policy)
    replay_events=[dict(t,month=m) for m in core.DEVELOPMENT_MONTHS for t in replay.monthly_events.get(m,[])]
    original=shadow[(shadow.get("expert_id",pd.Series(dtype=int))==seed_id)&(shadow.get("policy_key",pd.Series(dtype=str)).astype(str)==seed_key)&(shadow.get("month",pd.Series(dtype=str)).astype(str).isin(dev_months))].copy() if not shadow.empty else pd.DataFrame()
    original_fp=event_fingerprint(original);replay_fp=event_fingerprint(replay_events)
    data_cols=[c for c in ["open","high","low","close","volume"] if c in raw.columns]
    data_fp=hashlib.sha256(pd.util.hash_pandas_object(raw[data_cols],index=True).values.tobytes()).hexdigest()
    snapshot={"engine":ENGINE_NAME,"engine_version":"V9.6.8","seed_expert_id":seed_id,"seed_expert":expert.name,"policy_key":seed_key,"policy":asdict(policy),"feature_group":expert.feature_group,"feature_list":list(core.FEATURE_GROUPS[expert.feature_group]),"training_months":list(core.DEVELOPMENT_MONTHS),"diagnostic_oos_month":core.OOS_MONTH,"execution":{"fee_rate_per_side":core.FEE_RATE,"slippage_abs":core.SLIPPAGE_ABS,"next_bar_open":True,"same_candle_stop_before_tp":True},"model_runtime":REQUEST["model"],"statistical_validation":STAT,"code_hashes":{"entrypoint":file_sha256(ROOT/"autonomous_backtest_v9_6_8.py"),"strategy_core":file_sha256(ROOT/"_v968_strategy_core.py"),"base_engine":file_sha256(ROOT/"_v968_base_engine.py"),"request":file_sha256(REQUEST_PATH)},"official_data_fingerprint":data_fp,"signal_fingerprint":original_fp,"random_seed":REQUEST["model"]["base_seed"]}
    (RESULTS/"strategy_snapshot.json").write_text(json.dumps(snapshot,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    replay_audit={"policy_key":seed_key,"original_signal_fingerprint":original_fp,"replay_signal_fingerprint":replay_fp,"exact_match":original_fp==replay_fp,"original_metrics":base_metrics(original),"replay_metrics":core.metrics(replay_events)}
    (RESULTS/"snapshot_replay_audit.json").write_text(json.dumps(replay_audit,ensure_ascii=False,indent=2),encoding="utf-8")

    # Exact V9.6.1 reference lane and per-signal diff.
    rexp=ref.EXPERT_BY_ID[seed_id];rpolicy=next(p for p in ref.policy_grid(rexp) if p.key==str(SNAP["historical_policy_key"]))
    rc=ref.evaluate_candidate(rx,rexp,rpolicy)
    ref_events=[dict(t,month=m) for m in ref.DEVELOPMENT_MONTHS for t in rc.monthly_events.get(m,[])]
    ref_metrics=ref.metrics(ref_events);expected=SNAP["expected_v961"]
    exact_expected=(int(ref_metrics["trades"])==int(expected["trades"]) and int(ref_metrics["wins"])==int(expected["wins"]) and abs(float(ref_metrics["net_R"])-float(expected["net_R"]))<1e-9)
    ref_payload={"reference_engine":"V9.6.1 exact code lane","policy_key":rpolicy.key,"expected_historical_summary":expected,"reproduced_summary":ref_metrics,"exact_historical_summary_match":exact_expected,"reference_signal_fingerprint":event_fingerprint(ref_events),"current_v964_signal_fingerprint":original_fp}
    (RESULTS/"v961_reference_reproduction.json").write_text(json.dumps(ref_payload,ensure_ascii=False,indent=2),encoding="utf-8")
    old={(int(t["signal_i"]),int(t["direction"])):t for t in ref_events};new={(int(t["signal_i"]),int(t["direction"])):t for t in original.to_dict("records")}
    diff=[]
    for k in sorted(set(old)|set(new)):
        a=old.get(k);b=new.get(k);status="BOTH" if a and b else ("V961_ONLY" if a else "V964_ONLY")
        diff.append({"signal_i":k[0],"direction":k[1],"status":status,"v961_month":a.get("month") if a else None,"v964_month":b.get("month") if b else None,"v961_net_r":a.get("net_r") if a else None,"v964_net_r":b.get("net_r") if b else None,"v961_meta":a.get("meta_decision") if a else None,"v964_meta":b.get("meta_decision") if b else None,"v961_probability":a.get("probability") if a else None,"v964_probability":b.get("probability") if b else None,"v961_utility":a.get("utility") if a else None,"v964_utility":b.get("utility") if b else None})
    pd.DataFrame(diff).to_csv(RESULTS/"v961_vs_v964_signal_diff.csv",index=False)

    # Full per-candidate attribution for frozen comparison lanes plus V9.6.7.
    trace_months = list(core.DEVELOPMENT_MONTHS) + [core.OOS_MONTH]
    fixed_trace, fixed_selected = trace_policy_lane(core, x, expert, policy, "V9.6.4_FIXED_PENALTY", trace_months)
    reference_trace, reference_selected = trace_policy_lane(ref, rx, rexp, rpolicy, "V9.6.1_REFERENCE", trace_months)
    shrink_trace, shrink_selected = trace_evidence_shrinkage_lane(core, x, expert, policy, "V9.6.6_EVIDENCE_SHRINKAGE", trace_months)
    _, shrink_selected_replay = trace_evidence_shrinkage_lane(core, x, expert, policy, "V9.6.6_EVIDENCE_SHRINKAGE", trace_months)
    v967_trace, v967_selected = trace_no_monthly_budget_lane(ref, rx, rexp, rpolicy, trace_months)
    _, v967_selected_replay = trace_no_monthly_budget_lane(ref, rx, rexp, rpolicy, trace_months)

    seed_tier = str(new_tiers.loc[new_tiers["expert_id"] == seed_id, "tier"].iloc[0])
    v967_gated = annotate_expert_gate(v967_selected, seed_tier)
    v967_gated.to_csv(RESULTS / "v968_seed_no_monthly_budget_shadow_trades.csv", index=False)
    v967_gated[v967_gated["risk_multiplier"] > 0].to_csv(RESULTS / "v968_seed_research_execution_trades.csv", index=False)
    formal_ok = int((new_tiers["tier"] == "QUALIFIED").sum()) >= int(GATE["minimum_qualified_experts_for_formal"])
    if formal_ok:
        v967_gated[v967_gated["formal_eligible"]].to_csv(RESULTS / "v968_seed_formal_execution_trades.csv", index=False)
    else:
        v967_gated.iloc[0:0].to_csv(RESULTS / "v968_seed_formal_execution_trades.csv", index=False)

    gate_rows = []
    for _, gate_row in new_tiers.iterrows():
        tier = str(gate_row["tier"])
        gate_rows.append({
            "expert_id": int(gate_row["expert_id"]), "expert": str(gate_row["expert"]),
            "policy_key": str(gate_row["policy_key"]), "tier": tier,
            "risk_multiplier": float(GATE["tier_risk_multiplier"][tier]),
            "execution_mode": str(GATE["tier_execution_mode"][tier]),
            "signal_selection_independent_of_tier": bool(GATE["signal_selection_independent_of_tier"]),
            "portfolio_eligible": tier in {"WATCH", "QUALIFIED"},
            "formal_eligible": tier == "QUALIFIED" and formal_ok,
            "candidate_shadow_only": tier == "CANDIDATE"
        })
    pd.DataFrame(gate_rows).to_csv(RESULTS / "expert_level_gate_audit.csv", index=False)

    trace_all = pd.concat([reference_trace, fixed_trace, shrink_trace, v967_trace], ignore_index=True, sort=False)
    trace_all.to_csv(RESULTS / "signal_stage_attribution.csv", index=False)

    fixed_dev_selected = fixed_selected[fixed_selected["month"].astype(str).isin(dev_months)].copy() if not fixed_selected.empty else fixed_selected
    reference_dev_selected = reference_selected[reference_selected["month"].astype(str).isin(dev_months)].copy() if not reference_selected.empty else reference_selected
    shrink_dev_selected = shrink_selected[shrink_selected["month"].astype(str).isin(dev_months)].copy() if not shrink_selected.empty else shrink_selected
    v967_dev_selected = v967_selected[v967_selected["month"].astype(str).isin(dev_months)].copy() if not v967_selected.empty else v967_selected
    shrink_dev_replay = shrink_selected_replay[shrink_selected_replay["month"].astype(str).isin(dev_months)].copy() if not shrink_selected_replay.empty else shrink_selected_replay
    v967_dev_replay = v967_selected_replay[v967_selected_replay["month"].astype(str).isin(dev_months)].copy() if not v967_selected_replay.empty else v967_selected_replay
    fixed_trace_exact = event_fingerprint(fixed_dev_selected) == original_fp
    reference_trace_exact = event_fingerprint(reference_dev_selected) == event_fingerprint(ref_events)
    shrink_trace_exact = event_fingerprint(shrink_dev_selected) == event_fingerprint(shrink_dev_replay)
    v967_trace_exact = event_fingerprint(v967_dev_selected) == event_fingerprint(v967_dev_replay)
    if not fixed_trace_exact or not reference_trace_exact or not shrink_trace_exact or not v967_trace_exact:
        raise RuntimeError(f"Stage tracer mismatch: fixed={fixed_trace_exact} reference={reference_trace_exact} shrink={shrink_trace_exact} no_budget={v967_trace_exact}")

    seed_rating_dev = rating_shadow[(rating_shadow.get("expert_id",pd.Series(dtype=int))==seed_id)&(rating_shadow.get("policy_key",pd.Series(dtype=str)).astype(str)==str(rpolicy.key))&(rating_shadow.get("month",pd.Series(dtype=str)).astype(str).isin(dev_months))].copy() if not rating_shadow.empty else pd.DataFrame()
    rating_source_matches_no_budget = event_fingerprint(seed_rating_dev) == event_fingerprint(v967_dev_selected)
    rating_metrics_match_no_budget = base_metrics(seed_rating_dev) == base_metrics(v967_dev_selected)
    if bool(RATING_SOURCE.get("seed_source_trade_count_must_match_lane", True)) and not (rating_source_matches_no_budget and rating_metrics_match_no_budget):
        raise RuntimeError("Seed final rating source does not match the frozen no-monthly-budget lane")

    # The expert gate may change risk, but it must not alter the selected signal set.
    gate_fingerprints = {}
    for tier in ["CANDIDATE", "WATCH", "QUALIFIED"]:
        gate_fingerprints[tier] = event_fingerprint(annotate_expert_gate(v967_dev_selected, tier))
    gate_independent = len(set(gate_fingerprints.values())) == 1
    gate_audit = {
        "signal_selection_independent_of_tier": gate_independent,
        "gate_fingerprints": gate_fingerprints,
        "seed_development_tier": seed_tier,
        "seed_risk_multiplier": float(GATE["tier_risk_multiplier"][seed_tier]),
        "seed_execution_mode": str(GATE["tier_execution_mode"][seed_tier]),
        "candidate_enters_portfolio": bool(GATE["candidate_enters_portfolio"]),
        "formal_gate_open": formal_ok,
        "signal_selection_rules": ORIGINAL,
        "final_rating_source": str(RATING_SOURCE["source_lane"]),
        "rating_source_matches_no_budget_lane": rating_source_matches_no_budget,
        "rating_metrics_match_no_budget_lane": rating_metrics_match_no_budget,
        "rating_ignores_execution_budget": bool(GATE["rating_ignores_execution_budget"]),
        "rating_gate_ignored_fields": list(GATE.get("rating_gate_ignored_fields", []))
    }
    if not gate_independent:
        raise RuntimeError("Expert tier changed the selected signal fingerprint")
    (RESULTS / "signal_selection_independence_audit.json").write_text(json.dumps(gate_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    key_cols = ["month", "signal_i", "direction"]
    def lane_view(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=key_cols)
        keep = key_cols + [
            "exit_i", "day", "net_r", "win", "probability", "online_percentile",
            "rank_pass", "router", "micro", "meta_probability", "meta_decision",
            "meta_pass", "expected_utility", "raw_expected_utility", "brier_penalty",
            "brier_penalty_diagnostic", "sample_uncertainty_penalty", "total_penalty",
            "evidence_weight", "calibration_weight", "reliability_weight", "evidence_utility",
            "utility", "utility_pass", "signal_class", "allocation_bucket", "allocation_rule",
            "cap_pass", "selected", "first_reject_stage"
        ]
        q = df[[c for c in keep if c in df.columns]].copy()
        return q.rename(columns={c: f"{prefix}_{c}" for c in q.columns if c not in key_cols})

    cmp = lane_view(reference_trace, "v961").merge(lane_view(fixed_trace, "v964"), on=key_cols, how="outer")
    cmp = cmp.merge(lane_view(shrink_trace, "v966"), on=key_cols, how="outer")
    cmp = cmp.merge(lane_view(v967_trace, "v967"), on=key_cols, how="outer")
    for prefix in ["v961", "v964", "v966", "v967"]:
        col = f"{prefix}_selected"
        if col not in cmp: cmp[col] = False
        cmp[col] = cmp[col].fillna(False).astype(bool)
    cmp["selection_pattern"] = (
        cmp["v961_selected"].astype(int).astype(str)
        + cmp["v964_selected"].astype(int).astype(str)
        + cmp["v966_selected"].astype(int).astype(str)
        + cmp["v967_selected"].astype(int).astype(str)
    )
    outcome = None
    for c in ["v961_net_r", "v964_net_r", "v966_net_r", "v967_net_r"]:
        if c in cmp:
            outcome = cmp[c] if outcome is None else outcome.combine_first(cmp[c])
    cmp["outcome_net_r"] = outcome
    outwin = None
    for c in ["v961_win", "v964_win", "v966_win", "v967_win"]:
        if c in cmp:
            outwin = cmp[c] if outwin is None else outwin.combine_first(cmp[c])
    cmp["outcome_win"] = outwin
    for prefix in ["v961", "v964", "v966", "v967"]:
        cmp[f"profit_rejected_{prefix}"] = (~cmp[f"{prefix}_selected"]) & (cmp["outcome_net_r"] > 0)
    cmp.to_csv(RESULTS / "quad_lane_candidate_diff.csv", index=False)
    cmp[[c for c in cmp.columns if not c.startswith("v967_")]].to_csv(RESULTS / "triple_lane_candidate_diff.csv", index=False)
    focus = cmp[cmp["month"].astype(str) == str(TRACK["focus_month"])].copy()
    focus.to_csv(RESULTS / "january_signal_case_study.csv", index=False)

    impact_rows = []
    for (lane, month, stage), q in trace_all.groupby(["lane", "month", "first_reject_stage"], dropna=False):
        impact_rows.append({
            "lane": lane, "month": month, "first_reject_stage": stage,
            "candidates": int(len(q)), "wins_if_taken": int(q["win"].fillna(False).sum()),
            "losses_if_taken": int((~q["win"].fillna(False)).sum()),
            "counterfactual_net_R": float(q["net_r"].fillna(0).sum()),
            "selected_count": int(q["selected"].fillna(False).sum())
        })
    pd.DataFrame(impact_rows).to_csv(RESULTS / "filter_impact_summary.csv", index=False)

    shrink_cols = [c for c in ["month","signal_i","exit_i","day","net_r","win","probability","online_percentile","meta_probability","meta_decision","raw_expected_utility","evidence_weight","calibration_weight","reliability_weight","evidence_utility","signal_class","allocation_bucket","selected","first_reject_stage"] if c in shrink_trace.columns]
    shrink_trace[shrink_cols].to_csv(RESULTS / "evidence_shrinkage_attribution.csv", index=False)
    v967_cols = [c for c in ["month","signal_i","exit_i","day","net_r","win","probability","online_percentile","meta_probability","meta_decision","expected_utility","brier_penalty","utility","utility_pass","selected","allocation_rule","first_reject_stage","monthly_cap_enabled","cross_trade_cluster_dedup"] if c in v967_trace.columns]
    v967_trace[v967_cols].to_csv(RESULTS / "original_selection_attribution.csv", index=False)
    v967_trace[v967_trace["utility_pass"].fillna(False).astype(bool)][v967_cols].to_csv(RESULTS / "allocation_rule_audit.csv", index=False)

    selected_all = pd.concat([reference_selected, fixed_selected, shrink_selected, v967_selected], ignore_index=True, sort=False)
    selected_all[selected_all["month"].astype(str).isin(dev_months)].to_csv(RESULTS / "quad_track_development_trades.csv", index=False)
    selected_all[selected_all["month"].astype(str) == core.OOS_MONTH].to_csv(RESULTS / "quad_track_oos_shadow.csv", index=False)

    block_rows_quad = block_quality_rows(reference_dev_selected, "V9.6.1_REFERENCE", dev_months)
    block_rows_quad += block_quality_rows(fixed_dev_selected, "V9.6.4_FIXED_PENALTY", dev_months)
    block_rows_quad += block_quality_rows(shrink_dev_selected, "V9.6.6_EVIDENCE_SHRINKAGE", dev_months)
    block_rows_quad += block_quality_rows(v967_dev_selected, "V9.6.7_NO_MONTHLY_BUDGET_ORIGINAL", dev_months)
    pd.DataFrame(block_rows_quad).to_csv(RESULTS / "block_quality_validation.csv", index=False)

    reference_oos = reference_selected[reference_selected["month"].astype(str) == core.OOS_MONTH].copy() if not reference_selected.empty else reference_selected
    fixed_oos = fixed_selected[fixed_selected["month"].astype(str) == core.OOS_MONTH].copy() if not fixed_selected.empty else fixed_selected
    shrink_oos = shrink_selected[shrink_selected["month"].astype(str) == core.OOS_MONTH].copy() if not shrink_selected.empty else shrink_selected
    v967_oos = v967_selected[v967_selected["month"].astype(str) == core.OOS_MONTH].copy() if not v967_selected.empty else v967_selected
    quad_payload = {
        "winner_selection_enabled": False,
        "winner_decision": "NO_WINNER_SELECTION_FROM_SEEN_OOS",
        "reason": "2026-06 has already been inspected and cannot select or tune a winning lane",
        "lanes": {
            "V9.6.1_REFERENCE": {"development_metrics": base_metrics(reference_dev_selected), "diagnostic_oos_metrics": base_metrics(reference_oos), "development_signal_fingerprint": event_fingerprint(reference_dev_selected), "stage_trace_exact": reference_trace_exact},
            "V9.6.4_FIXED_PENALTY": {"development_metrics": base_metrics(fixed_dev_selected), "diagnostic_oos_metrics": base_metrics(fixed_oos), "development_signal_fingerprint": event_fingerprint(fixed_dev_selected), "stage_trace_exact": fixed_trace_exact},
            "V9.6.6_EVIDENCE_SHRINKAGE": {"development_metrics": base_metrics(shrink_dev_selected), "diagnostic_oos_metrics": base_metrics(shrink_oos), "development_signal_fingerprint": event_fingerprint(shrink_dev_selected), "stage_trace_exact": shrink_trace_exact},
            "V9.6.7_NO_MONTHLY_BUDGET_ORIGINAL": {"development_metrics": base_metrics(v967_dev_selected), "diagnostic_oos_metrics": base_metrics(v967_oos), "development_signal_fingerprint": event_fingerprint(v967_dev_selected), "stage_trace_exact": v967_trace_exact, "seed_tier": seed_tier, "risk_multiplier": float(GATE["tier_risk_multiplier"][seed_tier]), "execution_mode": str(GATE["tier_execution_mode"][seed_tier])}
        },
        "future_decision_rule": "Keep all four lanes frozen until a new blind window meets minimum calendar-month and trade-count evidence",
        "future_blind_min_calendar_months": int(TRACK["future_blind_min_calendar_months"]),
        "future_blind_min_trades_per_lane": int(TRACK["future_blind_min_trades_per_lane"]),
        "focus_month": TRACK["focus_month"]
    }
    (RESULTS / "quad_track_validation.json").write_text(json.dumps(quad_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Risk-budget dual track: same original signal-quality family, different execution budget.
    monthly_budget_dev = reference_dev_selected.copy()
    no_budget_dev = v967_dev_selected.copy()
    monthly_budget_oos = reference_oos.copy()
    no_budget_oos = v967_oos.copy()
    if not monthly_budget_dev.empty: monthly_budget_dev["risk_budget_lane"] = "V9.6.1_MONTHLY_BUDGET"
    if not no_budget_dev.empty: no_budget_dev["risk_budget_lane"] = "V9.6.7_NO_MONTHLY_BUDGET"
    if not monthly_budget_oos.empty: monthly_budget_oos["risk_budget_lane"] = "V9.6.1_MONTHLY_BUDGET"
    if not no_budget_oos.empty: no_budget_oos["risk_budget_lane"] = "V9.6.7_NO_MONTHLY_BUDGET"
    pd.concat([monthly_budget_dev,no_budget_dev],ignore_index=True,sort=False).to_csv(RESULTS/"risk_budget_dual_track_development_trades.csv",index=False)
    pd.concat([monthly_budget_oos,no_budget_oos],ignore_index=True,sort=False).to_csv(RESULTS/"risk_budget_dual_track_oos_shadow.csv",index=False)
    mb={(int(r["signal_i"]),int(r["direction"])):r for r in monthly_budget_dev.to_dict("records")}
    nb={(int(r["signal_i"]),int(r["direction"])):r for r in no_budget_dev.to_dict("records")}
    budget_diff=[]
    for key in sorted(set(mb)|set(nb)):
        a=mb.get(key);b=nb.get(key)
        budget_diff.append({
            "signal_i":key[0],"direction":key[1],
            "selection_status":"BOTH" if a and b else ("MONTHLY_BUDGET_ONLY" if a else "NO_MONTHLY_BUDGET_ONLY"),
            "month":str((a or b).get("month","")),
            "monthly_budget_net_r":a.get("net_r") if a else None,
            "no_monthly_budget_net_r":b.get("net_r") if b else None,
            "outcome_net_r":(a or b).get("net_r"),
            "monthly_budget_selected":bool(a),"no_monthly_budget_selected":bool(b)
        })
    pd.DataFrame(budget_diff).to_csv(RESULTS/"risk_budget_signal_diff.csv",index=False)
    risk_budget_payload={
        "winner_selection_enabled":False,
        "winner_decision":"NO_WINNER_SELECTION_FROM_SEEN_OOS",
        "reason":"2026-06 has already been inspected; both risk budgets remain frozen until fresh blind evidence",
        "rating_source":str(RATING_SOURCE["source_lane"]),
        "rating_is_independent_of_risk_budget":True,
        "lanes":{
            "V9.6.1_MONTHLY_BUDGET":{
                "development_metrics":base_metrics(monthly_budget_dev),
                "diagnostic_oos_metrics":base_metrics(monthly_budget_oos),
                "signal_fingerprint":event_fingerprint(monthly_budget_dev),
                "monthly_budget_enabled":True
            },
            "V9.6.7_NO_MONTHLY_BUDGET":{
                "development_metrics":base_metrics(no_budget_dev),
                "diagnostic_oos_metrics":base_metrics(no_budget_oos),
                "signal_fingerprint":event_fingerprint(no_budget_dev),
                "monthly_budget_enabled":False
            }
        },
        "future_blind_min_calendar_months":int(RISK_BUDGET["future_blind_min_calendar_months"]),
        "future_blind_min_trades_per_lane":int(RISK_BUDGET["future_blind_min_trades_per_lane"])
    }
    (RESULTS/"risk_budget_dual_track_validation.json").write_text(json.dumps(risk_budget_payload,ensure_ascii=False,indent=2),encoding="utf-8")

    seed_compare = source_compare_df[source_compare_df["expert_id"]==seed_id].iloc[0].to_dict()
    rating_audit={
        "final_rating_source":str(RATING_SOURCE["source_lane"]),
        "core_fixed_penalty_shadow_for_final_tier":False,
        "all_selected_experts_replayed":bool(RATING_SOURCE["all_selected_experts_replayed"]),
        "policy_coverage_complete":bool(coverage_df["policy_found_in_reference"].all()) if not coverage_df.empty else False,
        "rating_gate_ignored_fields":list(GATE.get("rating_gate_ignored_fields",[])),
        "rating_ignores_execution_budget":bool(GATE["rating_ignores_execution_budget"]),
        "seed_rating_source_matches_no_budget_lane":rating_source_matches_no_budget,
        "seed_rating_metrics_match_no_budget_lane":rating_metrics_match_no_budget,
        "seed_source_comparison":seed_compare,
        "seed_final_summary":summaries[seed_id],
        "seed_final_tier":seed_tier,
        "rating_shadow_fingerprint":event_fingerprint(rating_shadow[rating_shadow["month"].astype(str).isin(dev_months)]) if not rating_shadow.empty else event_fingerprint(pd.DataFrame())
    }
    (RESULTS/"rating_data_source_audit.json").write_text(json.dumps(rating_audit,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    separation_audit={
        "signal_layer":"frozen original probability/rank/meta/utility",
        "statistics_layer":str(RATING_SOURCE["source_lane"]),
        "risk_layer":"expert tier multiplier plus frozen dual risk budgets",
        "rating_ignores_execution_budget":bool(GATE["rating_ignores_execution_budget"]),
        "ignored_gate_fields":list(GATE.get("rating_gate_ignored_fields",[])),
        "monthly_budget_winner_selected":False,
        "seen_oos_used_for_selection":False
    }
    (RESULTS/"rating_execution_separation_audit.json").write_text(json.dumps(separation_audit,ensure_ascii=False,indent=2),encoding="utf-8")

    # Compatibility output keeps the first three frozen lanes available to old readers.
    (RESULTS / "triple_track_validation.json").write_text(json.dumps({**quad_payload, "lanes": {k:v for k,v in quad_payload["lanes"].items() if k != "V9.6.7_NO_MONTHLY_BUDGET_ORIGINAL"}}, ensure_ascii=False, indent=2), encoding="utf-8")

    rule_delta = {
        "V9.6.1_REFERENCE": {"base_seed": int(ref.MODEL["base_seed"]), "sample_uncertainty_penalty": float(ref.MODEL.get("sample_uncertainty_penalty", 0.0)), "policy": asdict(rpolicy)},
        "V9.6.4_FIXED_PENALTY": {"base_seed": int(core.MODEL["base_seed"]), "sample_uncertainty_penalty": float(core.MODEL.get("sample_uncertainty_penalty", 0.0)), "policy": asdict(policy)},
        "V9.6.6_EVIDENCE_SHRINKAGE": {"base_seed": int(core.MODEL["base_seed"]), "sample_uncertainty_mode": SHRINK["sample_uncertainty_mode"], "allocation": SHRINK, "policy": asdict(policy)},
        "V9.6.7_NO_MONTHLY_BUDGET_ORIGINAL": {"base_seed": int(ref.MODEL["base_seed"]), "signal_selection": ORIGINAL, "expert_level_gate": GATE, "policy": asdict(rpolicy)},
        "winner_selection_from_seen_oos": False
    }
    (RESULTS / "selection_rule_delta.json").write_text(json.dumps(rule_delta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Update status and report after the stricter evidence validation.
    status_path=RESULTS/"status.json";status=json.loads(status_path.read_text(encoding="utf-8"))
    counts={t:int((new_tiers["tier"]==t).sum()) for t in ["QUALIFIED","WATCH","CANDIDATE","REJECTED"]}
    def portfolio_stats(path:Path)->dict[str,dict[str,float]]:
        try:df=pd.read_csv(path)
        except Exception:df=pd.DataFrame()
        if not df.empty and "net_R" in df.columns and "net_r" not in df.columns:df=df.rename(columns={"net_R":"net_r"})
        return {m:base_metrics(df[df["month"].astype(str)==m]) if not df.empty and "month" in df.columns else base_metrics(pd.DataFrame()) for m in ["2026-05",core.OOS_MONTH]}
    research_stats=portfolio_stats(RESULTS/"research_portfolio_trades.csv");qualified_stats=portfolio_stats(RESULTS/"qualified_portfolio_trades.csv")
    research_selected_payload={str(r["expert"]):{"expert_id":int(r["expert_id"]),"tier":str(r["tier"]),"policy_key":str(r["policy_key"]),"development_summary":summaries[int(r["expert_id"])]} for _,r in new_tiers[new_tiers["tier"].isin(["WATCH","QUALIFIED"])].iterrows()}
    qualified_selected_payload={str(r["expert"]):{"expert_id":int(r["expert_id"]),"tier":str(r["tier"]),"policy_key":str(r["policy_key"]),"development_summary":summaries[int(r["expert_id"])]} for _,r in new_tiers[new_tiers["tier"]=="QUALIFIED"].iterrows()}
    status.update({
        "engine": ENGINE_NAME,
        "qualified": False,
        "not_for_live_trading": True,
        "tier_counts": counts,
        "statistical_validation_enabled": True,
        "monthly_raw_win_rate_is_not_validation": True,
        "selected_research_expert_count": counts["WATCH"] + counts["QUALIFIED"],
        "selected_qualified_expert_count": counts["QUALIFIED"],
        "selected_experts": {"research_watch_and_qualified_only": research_selected_payload, "qualified": qualified_selected_payload},
        "watch_portfolio_monthly_stats": research_stats,
        "research_portfolio_monthly_stats": research_stats,
        "qualified_portfolio_monthly_stats": qualified_stats,
        "snapshot_replay_exact": replay_audit["exact_match"],
        "v961_reference_exact_summary_match": exact_expected,
        "signal_stage_attribution_enabled": True,
        "quad_track_validation_enabled": True,
        "evidence_shrinkage_lane_enabled": True,
        "expert_level_gating_enabled": True,
        "rating_data_source_unified": True,
        "rating_source_lane": str(RATING_SOURCE["source_lane"]),
        "rating_source_policy_coverage_complete": bool(coverage_df["policy_found_in_reference"].all()) if not coverage_df.empty else False,
        "rating_source_matches_no_budget_lane": rating_source_matches_no_budget,
        "rating_metrics_match_no_budget_lane": rating_metrics_match_no_budget,
        "rating_ignores_execution_budget": bool(GATE["rating_ignores_execution_budget"]),
        "rating_gate_ignored_fields": list(GATE.get("rating_gate_ignored_fields", [])),
        "seed_core_fixed_penalty_trades": int(seed_compare["core_fixed_penalty_trades"]),
        "seed_unified_rating_trades": int(seed_compare["unified_rating_trades"]),
        "risk_budget_dual_track_enabled": True,
        "risk_budget_winner_selection_enabled": False,
        "signal_selection_independent_of_tier": gate_independent,
        "v968_no_monthly_budget_lane_trace_exact": v967_trace_exact,
        "v968_seed_tier": seed_tier,
        "v968_seed_risk_multiplier": float(GATE["tier_risk_multiplier"][seed_tier]),
        "v968_seed_execution_mode": str(GATE["tier_execution_mode"][seed_tier]),
        "v968_no_monthly_budget_enabled": not bool(ORIGINAL["monthly_cap_enabled"]),
        "v968_cross_trade_cluster_dedup": bool(ORIGINAL["cross_trade_cluster_dedup"]),
        "winner_selection_enabled": False,
        "winner_decision": "NO_WINNER_SELECTION_FROM_SEEN_OOS",
        "fixed_stage_trace_exact": fixed_trace_exact,
        "reference_stage_trace_exact": reference_trace_exact,
        "shrinkage_stage_trace_exact": shrink_trace_exact,
        "blind_evidence_complete": False,
        "constraints": {
            **status.get("constraints", {}),
            "watch_gate": WATCH_GATE,
            "qualified_gate": QUALIFIED_GATE,
            "statistical_validation": STAT,
            "block_validation": BLOCK,
            "quad_track_validation": TRACK,
            "evidence_shrinkage_lane": SHRINK,
            "expert_level_gating": GATE,
            "v968_no_monthly_budget_lane": ORIGINAL,
            "risk_budget_dual_track": RISK_BUDGET,
            "rating_data_source": RATING_SOURCE
        }
    })
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    seed_summary = summaries[seed_id]
    lanes = quad_payload["lanes"]
    report = f"""# BTCUSDT 5分钟 门控统计数据源统一与风险预算双轨 V9.6.8 报告

- 架构：所有已选专家用V9.6.7无月度预算原始影子统一重放 → 统计评级只读取该影子账本 → 专家等级只控制风险 → 月度预算与无月度预算双轨冻结验证。
- 专家等级：正式 {counts['QUALIFIED']}；观察 {counts['WATCH']}；候选 {counts['CANDIDATE']}；淘汰 {counts['REJECTED']}。
- 当前策略快照重放：**{'通过' if replay_audit['exact_match'] else '失败'}**。
- V9.6.1历史参考通道复现：**{'通过' if exact_expected else '失败'}**。
- 无月度预算原始选单通道重放：**{'通过' if v967_trace_exact else '失败'}**。
- 专家等级是否改变信号集合：**{'否' if gate_independent else '是'}**。
- 最终评级数据源：**V9.6.7无月度预算原始影子**；评级忽略执行预算字段。
- 风险预算双轨：**V9.6.1原始月度预算** 与 **V9.6.7无月度预算**；跨交易行情簇去重保持关闭。
- 胜出通道选择：**关闭**。2026年6月已经被查看，只能诊断，不能用于选通道或调参数。
- 实盘资格：**不合格**。

## 当前种子专家统一评级状态

| 指标 | 结果 |
|---|---:|
| 专家等级 | {seed_tier} |
| 执行模式 | {GATE['tier_execution_mode'][seed_tier]} |
| 风险倍数 | {float(GATE['tier_risk_multiplier'][seed_tier]):.2f} |
| 累计交易 | {seed_summary['trades']} |
| 原始胜率 | {seed_summary['win_rate']:.2%} |
| 贝叶斯收缩胜率 | {seed_summary['shrunk_win_rate']:.2%} |
| 95% Wilson下界 | {seed_summary['wilson_lower']:.2%} |
| 有效三个月块 | {seed_summary['valid_blocks']} |
| 正期望验证块 | {seed_summary['positive_expectancy_blocks']} |
| 高胜率验证块 | {seed_summary['high_win_rate_blocks']} |
| 删除最佳交易后净R | {seed_summary['best_trade_removed_net_R']:.3f} |

## 评级数据源接线核对

| 数据账本 | 交易 | 盈利 | 净R | 是否用于最终评级 |
|---|---:|---:|---:|---|
| V9.6.4固定惩罚核心账本 | {int(seed_compare['core_fixed_penalty_trades'])} | {int(seed_compare['core_fixed_penalty_wins'])} | {float(seed_compare['core_fixed_penalty_net_R']):.3f} | 否，仅用于策略参数选择审计 |
| V9.6.7无月度预算原始影子 | {int(seed_compare['unified_rating_trades'])} | {int(seed_compare['unified_rating_wins'])} | {float(seed_compare['unified_rating_net_R']):.3f} | **是，唯一最终评级数据源** |

- 种子专家评级账本与无月度预算通道逐信号一致：**{'是' if rating_source_matches_no_budget else '否'}**。
- 评级忽略执行预算字段：**{'是' if GATE['rating_ignores_execution_budget'] else '否'}**；忽略字段：`{'|'.join(GATE.get('rating_gate_ignored_fields', []))}`。

## 四通道开发期对照

| 通道 | 交易 | 胜率 | 盈亏比 | 净R | 最大回撤R |
|---|---:|---:|---:|---:|---:|
| V9.6.1参考 | {lanes['V9.6.1_REFERENCE']['development_metrics']['trades']} | {lanes['V9.6.1_REFERENCE']['development_metrics']['win_rate']:.2%} | {lanes['V9.6.1_REFERENCE']['development_metrics']['avg_win_loss_ratio']:.3f} | {lanes['V9.6.1_REFERENCE']['development_metrics']['net_R']:.3f} | {lanes['V9.6.1_REFERENCE']['development_metrics']['max_drawdown_R']:.3f} |
| V9.6.4固定惩罚 | {lanes['V9.6.4_FIXED_PENALTY']['development_metrics']['trades']} | {lanes['V9.6.4_FIXED_PENALTY']['development_metrics']['win_rate']:.2%} | {lanes['V9.6.4_FIXED_PENALTY']['development_metrics']['avg_win_loss_ratio']:.3f} | {lanes['V9.6.4_FIXED_PENALTY']['development_metrics']['net_R']:.3f} | {lanes['V9.6.4_FIXED_PENALTY']['development_metrics']['max_drawdown_R']:.3f} |
| V9.6.6证据收缩 | {lanes['V9.6.6_EVIDENCE_SHRINKAGE']['development_metrics']['trades']} | {lanes['V9.6.6_EVIDENCE_SHRINKAGE']['development_metrics']['win_rate']:.2%} | {lanes['V9.6.6_EVIDENCE_SHRINKAGE']['development_metrics']['avg_win_loss_ratio']:.3f} | {lanes['V9.6.6_EVIDENCE_SHRINKAGE']['development_metrics']['net_R']:.3f} | {lanes['V9.6.6_EVIDENCE_SHRINKAGE']['development_metrics']['max_drawdown_R']:.3f} |
| V9.6.7无月度预算原始选单 | {lanes['V9.6.7_NO_MONTHLY_BUDGET_ORIGINAL']['development_metrics']['trades']} | {lanes['V9.6.7_NO_MONTHLY_BUDGET_ORIGINAL']['development_metrics']['win_rate']:.2%} | {lanes['V9.6.7_NO_MONTHLY_BUDGET_ORIGINAL']['development_metrics']['avg_win_loss_ratio']:.3f} | {lanes['V9.6.7_NO_MONTHLY_BUDGET_ORIGINAL']['development_metrics']['net_R']:.3f} | {lanes['V9.6.7_NO_MONTHLY_BUDGET_ORIGINAL']['development_metrics']['max_drawdown_R']:.3f} |

## 风险预算双轨开发期对照

| 风险预算 | 交易 | 胜率 | 净R | 月度预算 |
|---|---:|---:|---:|---|
| V9.6.1原始月度预算 | {risk_budget_payload['lanes']['V9.6.1_MONTHLY_BUDGET']['development_metrics']['trades']} | {risk_budget_payload['lanes']['V9.6.1_MONTHLY_BUDGET']['development_metrics']['win_rate']:.2%} | {risk_budget_payload['lanes']['V9.6.1_MONTHLY_BUDGET']['development_metrics']['net_R']:.3f} | 开启 |
| V9.6.7无月度预算 | {risk_budget_payload['lanes']['V9.6.7_NO_MONTHLY_BUDGET']['development_metrics']['trades']} | {risk_budget_payload['lanes']['V9.6.7_NO_MONTHLY_BUDGET']['development_metrics']['win_rate']:.2%} | {risk_budget_payload['lanes']['V9.6.7_NO_MONTHLY_BUDGET']['development_metrics']['net_R']:.3f} | 关闭 |

两条风险预算保持冻结，2026年6月已查看，不从该月选择胜者。

`rating_source_comparison.csv`显示旧固定惩罚账本与统一评级账本的差异；`v968_unified_rating_shadow_trades.csv`是最终评级唯一数据源；`risk_budget_dual_track_validation.json`比较两种风险预算；`signal_selection_independence_audit.json`证明等级变化不改变信号集合。已查看OOS不选择胜者。
"""
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    (RESULTS / "run_identity.txt").write_text(
        f"{ENGINE_NAME}\noutput=results_v9_6_8\noos={core.OOS_MONTH}\n"
        f"snapshot_replay_exact={replay_audit['exact_match']}\nv961_reference_exact={exact_expected}\n"
        f"fixed_stage_trace_exact={fixed_trace_exact}\nreference_stage_trace_exact={reference_trace_exact}\n"
        f"shrinkage_stage_trace_exact={shrink_trace_exact}\nno_monthly_budget_stage_trace_exact={v967_trace_exact}\n"
        f"signal_selection_independent_of_tier={gate_independent}\nrating_source={RATING_SOURCE['source_lane']}\n"
        f"rating_source_matches_no_budget={rating_source_matches_no_budget}\nrating_ignores_execution_budget={GATE['rating_ignores_execution_budget']}\n"
        f"monthly_cap_enabled={ORIGINAL['monthly_cap_enabled']}\ncross_trade_cluster_dedup={ORIGINAL['cross_trade_cluster_dedup']}\n"
        f"risk_budget_winner_selection_enabled=False\nwinner_selection_enabled=False\n",
        encoding="utf-8"
    )
    print(json.dumps({
        "tier_counts": counts,
        "snapshot_replay_exact": replay_audit["exact_match"],
        "v961_reference_exact": exact_expected,
        "no_monthly_budget_trace_exact": v967_trace_exact,
        "signal_selection_independent_of_tier": gate_independent,
        "rating_source": str(RATING_SOURCE["source_lane"]),
        "rating_source_matches_no_budget": rating_source_matches_no_budget,
        "seed_core_fixed_penalty_trades": int(seed_compare["core_fixed_penalty_trades"]),
        "seed_unified_rating_trades": int(seed_compare["unified_rating_trades"]),
        "seed_tier": seed_tier,
        "seed_risk_multiplier": float(GATE["tier_risk_multiplier"][seed_tier]),
        "seed_evidence": seed_summary,
        "no_monthly_budget_development_metrics": lanes["V9.6.7_NO_MONTHLY_BUDGET_ORIGINAL"]["development_metrics"]
    }, ensure_ascii=False, indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--self-test",action="store_true");parser.add_argument("--pipeline-smoke",action="store_true");args=parser.parse_args()
    if args.self_test:
        core.synthetic_smoke();statistical_self_test();attribution_self_test();shrinkage_allocation_self_test();original_gate_self_test();rating_source_self_test()
    elif args.pipeline_smoke:
        core.pipeline_smoke();statistical_self_test();attribution_self_test();shrinkage_allocation_self_test();original_gate_self_test();rating_source_self_test()
    else:main()
