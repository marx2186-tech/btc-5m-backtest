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
RESULTS = ROOT / "results_v9_7_2"
REQUEST_PATH = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(ROOT / "request.v9_7_2.json")))
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
ORIGINAL = REQUEST["v969_no_monthly_budget_lane"]
RISK_BUDGET = REQUEST["risk_budget_dual_track"]
RATING_SOURCE = REQUEST["rating_data_source"]
EXPERT_COVERAGE = REQUEST["expert_coverage"]
DIAGNOSTIC = REQUEST["historical_diagnostic_window"]
MONTHLY_DATA_QUALITY = DIAGNOSTIC["monthly_data_quality"]
ENGINE_NAME = "BTC 5m 2026 Q2 historical diagnostic backtest V9.7.2"

spec = importlib.util.spec_from_file_location("v972_strategy_core", ROOT / "_v972_strategy_core.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load _v972_strategy_core.py")
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
    extended_x: pd.DataFrame,
    selected_policy_rows: pd.DataFrame,
    months: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay every selected expert policy with the frozen original signal rules.

    The returned selected trades are the only admissible source for final expert
    evidence tiers in V9.6.9. Core fixed-penalty shadow trades are retained only
    for policy-selection comparison and are forbidden from final tier assignment.
    """
    selected_frames: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    missing: list[str] = []
    for _, row in selected_policy_rows.sort_values("expert_id").iterrows():
        eid = int(row["expert_id"])
        expert_name = str(row["expert"])
        policy_key = str(row["policy_key"])

        # V9.7.2 formalizes expert coverage as a request-level contract.
        # Historical experts 0-31 replay through the exact V9.6.1 reference;
        # later experts 32-39 replay through the current core while retaining
        # the frozen original probability/rank/meta/utility rules.
        reference_ids = {int(x) for x in EXPERT_COVERAGE["v961_reference_ids"]}
        extended_ids = {int(x) for x in EXPERT_COVERAGE["extended_original_ids"]}
        if eid in reference_ids:
            if eid not in ref.EXPERT_BY_ID:
                missing.append(f"{eid}:{policy_key}:REFERENCE_EXPERT_NOT_FOUND")
                coverage.append({
                    "expert_id": eid, "expert": expert_name, "policy_key": policy_key,
                    "rating_engine": "V9.6.1_REFERENCE", "found_in_v961_reference": False,
                    "expert_found_in_rating_engine": False,
                    "policy_found_in_rating_engine": False, "raw_candidates": 0,
                    "selected_trades": 0, "signal_fingerprint": ""
                })
                continue
            source_mod = ref
            source_x = rx
            source_engine = "V9.6.1_REFERENCE"
            found_in_v961_reference = True
        elif eid in extended_ids:
            if eid not in core.EXPERT_BY_ID:
                missing.append(f"{eid}:{policy_key}:EXTENDED_EXPERT_NOT_FOUND")
                coverage.append({
                    "expert_id": eid, "expert": expert_name, "policy_key": policy_key,
                    "rating_engine": "V9.6.9_FROZEN_EXTENDED_ORIGINAL", "found_in_v961_reference": False,
                    "expert_found_in_rating_engine": False,
                    "policy_found_in_rating_engine": False, "raw_candidates": 0,
                    "selected_trades": 0, "signal_fingerprint": ""
                })
                continue
            source_mod = core
            source_x = extended_x
            source_engine = "V9.6.9_FROZEN_EXTENDED_ORIGINAL"
            found_in_v961_reference = False
        else:
            missing.append(f"{eid}:{policy_key}:EXPERT_ROUTE_NOT_CONFIGURED")
            coverage.append({
                "expert_id": eid, "expert": expert_name, "policy_key": policy_key,
                "rating_engine": "UNCONFIGURED", "found_in_v961_reference": False,
                "expert_found_in_rating_engine": False,
                "policy_found_in_rating_engine": False, "raw_candidates": 0,
                "selected_trades": 0, "signal_fingerprint": ""
            })
            continue

        rexp = source_mod.EXPERT_BY_ID[eid]
        matches = [p for p in source_mod.policy_grid(rexp) if str(p.key) == policy_key]
        if not matches:
            missing.append(f"{eid}:{policy_key}:POLICY_NOT_FOUND")
            coverage.append({
                "expert_id": eid, "expert": expert_name, "policy_key": policy_key,
                "rating_engine": source_engine,
                "found_in_v961_reference": found_in_v961_reference,
                "expert_found_in_rating_engine": True,
                "policy_found_in_rating_engine": False, "raw_candidates": 0,
                "selected_trades": 0, "signal_fingerprint": ""
            })
            continue

        policy = matches[0]
        mask_column = f"sparse_{rexp.key}"
        feature_frame_column_found = mask_column in source_x.columns
        if not feature_frame_column_found:
            missing.append(f"{eid}:{policy_key}:FEATURE_MASK_NOT_FOUND:{source_engine}:{mask_column}")
            coverage.append({
                "expert_id": eid, "expert": expert_name, "policy_key": policy_key,
                "rating_engine": source_engine,
                "feature_frame": "V9.6.1_REFERENCE_FEATURES" if source_mod is ref else "V9.6.9_FROZEN_EXTENDED_FEATURES",
                "found_in_v961_reference": found_in_v961_reference,
                "expert_found_in_rating_engine": True,
                "policy_found_in_rating_engine": True,
                "feature_frame_column_found": False,
                "required_mask_column": mask_column,
                "raw_candidates": 0, "selected_trades": 0, "signal_fingerprint": ""
            })
            continue
        trace, selected = trace_no_monthly_budget_lane(source_mod, source_x, rexp, policy, months)
        if not selected.empty:
            selected = selected.copy()
            selected["expert_id"] = eid
            selected["expert"] = expert_name
            selected["family"] = str(getattr(rexp, "family", row.get("family", "")))
            selected["setup_group"] = str(getattr(rexp, "setup_group", getattr(rexp, "family", "")))
            selected["rating_source_lane"] = str(RATING_SOURCE["source_lane"])
            selected["rating_engine"] = source_engine
            selected_frames.append(selected)
        coverage.append({
            "expert_id": eid, "expert": expert_name, "policy_key": policy_key,
            "rating_engine": source_engine,
            "feature_frame": "V9.6.1_REFERENCE_FEATURES" if source_mod is ref else "V9.6.9_FROZEN_EXTENDED_FEATURES",
            "found_in_v961_reference": found_in_v961_reference,
            "expert_found_in_rating_engine": True,
            "policy_found_in_rating_engine": True,
            "feature_frame_column_found": True,
            "required_mask_column": mask_column,
            "raw_candidates": int(len(trace)),
            "selected_trades": int(len(selected)),
            "signal_fingerprint": event_fingerprint(selected)
        })
    coverage_df = pd.DataFrame(coverage)
    if not coverage_df.empty:
        if "feature_frame_column_found" not in coverage_df:
            coverage_df["feature_frame_column_found"] = False
        coverage_df["feature_frame_column_found"] = coverage_df["feature_frame_column_found"].fillna(False).astype(bool)
        if "required_mask_column" not in coverage_df:
            coverage_df["required_mask_column"] = ""
        if "feature_frame" not in coverage_df:
            coverage_df["feature_frame"] = ""
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
    print("V972_QUAD_TRACK_SELF_TEST_OK")



def statistical_self_test()->None:
    rows=[]
    for i,r in enumerate([1.7,-0.3,1.6,-0.4,1.5,-0.2,1.4,-0.2]):
        rows.append({"month":["2025-05","2025-06","2025-07","2025-08"][i//2],"net_r":r})
    s,monthly,blocks=evidence_for_expert(pd.DataFrame(rows),list(core.DEVELOPMENT_MONTHS),True)
    assert s["trades"]==8 and s["best_trade_removed_net_R"]>0
    assert monthly[0]["sample_state"]=="INSUFFICIENT_SAMPLE"
    one=pd.DataFrame([{"month":"2025-05","net_r":1.5}]);o,_,_=evidence_for_expert(one,list(core.DEVELOPMENT_MONTHS),True)
    assert o["shrunk_win_rate"]<0.60 and o["wilson_lower"]<0.30
    print("V972_STATISTICAL_VALIDATION_SELF_TEST_OK")



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
    print("V972_EVIDENCE_SHRINKAGE_AND_ALLOCATION_OK")


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
    print("V972_ORIGINAL_SELECTION_AND_EXPERT_GATE_OK")


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
    # Regression for the production failure: the coverage contract must
    # resolve every current expert exactly once across the two rating engines.
    required_ids = {int(x) for x in EXPERT_COVERAGE["required_expert_ids"]}
    reference_ids = {int(x) for x in EXPERT_COVERAGE["v961_reference_ids"]}
    extended_ids = {int(x) for x in EXPERT_COVERAGE["extended_original_ids"]}
    assert reference_ids.isdisjoint(extended_ids)
    assert reference_ids | extended_ids == required_ids
    assert required_ids == set(core.EXPERT_BY_ID)
    selftest_ref_request = ROOT / ".v972_selftest_reference_request.json"
    selftest_ref_request.write_text(json.dumps(REQUEST["legacy_v961_reference_request"], ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        ref_mod = import_reference(selftest_ref_request)
        assert reference_ids == set(ref_mod.EXPERT_BY_ID)
        for eid in sorted(reference_ids):
            ref_keys = {str(p.key) for p in ref_mod.policy_grid(ref_mod.EXPERT_BY_ID[eid])}
            current_keys = {str(p.key) for p in core.policy_grid(core.EXPERT_BY_ID[eid])}
            assert ref_keys == current_keys and ref_keys
        for eid in sorted(extended_ids):
            assert {str(p.key) for p in core.policy_grid(core.EXPERT_BY_ID[eid])}

        # Production regression: reference experts must replay against the reference
        # feature frame, while extended experts must replay against the current frame.
        # Passing ref_x to expert 32 previously raised:
        # KeyError: sparse_cross_premium_revert_short.
        raw, eth, premium, funding = core.base.synthetic_inputs(5000)
        current_x, _ = core.base.add_features(raw.copy(), eth.copy(), premium.copy(), funding.copy())
        current_x = core.add_sparse_masks(current_x)
        ref_x, _ = ref_mod.base.add_features(raw.copy(), eth.copy(), premium.copy(), funding.copy())
        ref_x = ref_mod.add_sparse_masks(ref_x)
        assert current_x.index.equals(ref_x.index)
        for eid in sorted(reference_ids):
            assert f"sparse_{ref_mod.EXPERT_BY_ID[eid].key}" in ref_x.columns
        for eid in sorted(extended_ids):
            assert f"sparse_{core.EXPERT_BY_ID[eid].key}" in current_x.columns

        route_rows = pd.DataFrame([
            {
                "expert_id": eid,
                "expert": core.EXPERT_BY_ID[eid].name,
                "family": core.EXPERT_BY_ID[eid].family,
                "policy_key": str(core.policy_grid(core.EXPERT_BY_ID[eid])[0].key),
            }
            for eid in sorted(required_ids)
        ])
        _, route_coverage = build_unified_rating_shadow(ref_mod, ref_x, current_x, route_rows, [])
        assert len(route_coverage) == len(required_ids) == 40
        assert route_coverage["expert_id"].astype(int).nunique() == 40
        assert route_coverage["feature_frame_column_found"].astype(bool).all()
        assert route_coverage["expert_found_in_rating_engine"].astype(bool).all()
        assert route_coverage["policy_found_in_rating_engine"].astype(bool).all()
        assert set(route_coverage["rating_engine"]) == {"V9.6.1_REFERENCE", "V9.6.9_FROZEN_EXTENDED_ORIGINAL"}
        assert set(route_coverage["feature_frame"]) == {"V9.6.1_REFERENCE_FEATURES", "V9.6.9_FROZEN_EXTENDED_FEATURES"}
    finally:
        selftest_ref_request.unlink(missing_ok=True)
    assert int(EXPERT_COVERAGE["required_expert_count"]) == len(required_ids) == 40
    print("V972_UNIFIED_RATING_SOURCE_AND_DUAL_BUDGET_OK")




DIAGNOSTIC_LEDGER_COLUMNS = [
    "event_key","lane","month","expert_id","expert","family","setup_group","policy_key","direction",
    "signal_i","exit_i","signal_time_utc","entry_time_utc","exit_time_utc","day","net_r","win","reason",
    "probability","online_percentile","meta_probability","meta_decision","expected_utility","utility",
    "first_reject_stage","rating_engine"
]


def _read_csv_or_empty(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])


def _periods(start_month: str, end_month: str) -> list[str]:
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    if end < start:
        return []
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def _calendar_latest_complete_month(now: pd.Timestamp | None = None) -> str:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    current = pd.Period(now.strftime("%Y-%m"), freq="M")
    grace = int(DIAGNOSTIC.get("archive_grace_days", 5))
    latest = current - 1 if int(now.day) >= grace else current - 2
    return str(latest)


def _add_event_times(df: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for c in ["signal_time_utc","entry_time_utc","exit_time_utc"]:
            out[c] = pd.Series(dtype=str)
        return out
    def stamp(i: Any) -> str:
        j = int(i)
        if j < 0 or j >= len(x):
            return ""
        return pd.Timestamp(x.index[j]).isoformat()
    out["signal_time_utc"] = out["signal_i"].map(stamp)
    out["entry_time_utc"] = (out["signal_i"].astype(int) + 1).map(stamp)
    out["exit_time_utc"] = out["exit_i"].map(stamp)
    return out


def _event_key(row: pd.Series | dict[str, Any]) -> str:
    get = row.get
    return "|".join([
        str(get("lane", "")), str(get("expert_id", "")), str(get("policy_key", "")),
        str(get("signal_time_utc", "")), str(get("direction", ""))
    ])


def _merge_append_only(previous: pd.DataFrame, current: pd.DataFrame, ledger_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    previous = previous.copy(); current = current.copy()
    if previous.empty and len(previous.columns)==0: previous=pd.DataFrame(columns=DIAGNOSTIC_LEDGER_COLUMNS)
    if current.empty and len(current.columns)==0: current=pd.DataFrame(columns=DIAGNOSTIC_LEDGER_COLUMNS)
    for frame in (previous, current):
        if not frame.empty:
            frame["event_key"] = frame.apply(_event_key, axis=1)
    if previous.empty:
        merged = current.copy()
        return merged, {"ledger":ledger_name,"previous_rows":0,"new_rows":int(len(current)),"unchanged_rows":0,"mutations":0,"append_only_pass":True}
    if current.empty:
        return previous.copy(), {"ledger":ledger_name,"previous_rows":int(len(previous)),"new_rows":0,"unchanged_rows":int(len(previous)),"mutations":0,"append_only_pass":True}
    immutable = list(DIAGNOSTIC.get("immutable_event_fields", []))
    prev_by = {str(r["event_key"]): r for _,r in previous.iterrows()}
    mutations=[]; new=[]; unchanged=0
    for _,r in current.iterrows():
        key=str(r["event_key"])
        if key not in prev_by:
            new.append(r.to_dict()); continue
        old=prev_by[key]; unchanged += 1
        for c in immutable:
            a=old.get(c,""); b=r.get(c,"")
            if c=="net_r":
                same=abs(float(a)-float(b))<=1e-10
            else:
                same=str(a)==str(b)
            if not same:
                mutations.append({"event_key":key,"field":c,"previous":a,"current":b})
    if mutations:
        raise RuntimeError(f"Append-only diagnostic ledger mutation in {ledger_name}: {mutations[:5]}")
    merged=pd.concat([previous,pd.DataFrame(new)],ignore_index=True,sort=False) if new else previous.copy()
    if not merged.empty:
        merged=merged.drop_duplicates("event_key",keep="first").sort_values([c for c in ["month","signal_time_utc","lane","expert_id"] if c in merged.columns]).reset_index(drop=True)
    return merged,{"ledger":ledger_name,"previous_rows":int(len(previous)),"new_rows":int(len(new)),"unchanged_rows":int(unchanged),"mutations":0,"append_only_pass":True}


def _policy_for(mod: Any, eid: int, key: str) -> Any:
    expert=mod.EXPERT_BY_ID[eid]
    return expert,next(p for p in mod.policy_grid(expert) if str(p.key)==str(key))


def _with_blind_months(mod: Any, months: list[str]):
    class Ctx:
        def __enter__(self_nonlocal):
            self_nonlocal.old_months=mod.MONTHS; self_nonlocal.old_oos=mod.OOS_MONTH
            mod.MONTHS=tuple(months); mod.OOS_MONTH="__DIAGNOSTIC_CURRENT_MONTH_GUARD__"
            for name in ["MODEL_CACHE","EVAL_CACHE","OUTCOME_CACHE"]:
                cache=getattr(mod,name,None)
                if hasattr(cache,"clear"): cache.clear()
            return mod
        def __exit__(self_nonlocal,exc_type,exc,tb):
            mod.MONTHS=self_nonlocal.old_months; mod.OOS_MONTH=self_nonlocal.old_oos
    return Ctx()


def _window_start() -> pd.Timestamp:
    return pd.Timestamp(str(DIAGNOSTIC["window_start_utc"])).tz_convert("UTC")


def _window_cutoff() -> pd.Timestamp:
    return pd.Timestamp(str(DIAGNOSTIC["window_end_utc"])).tz_convert("UTC")


def _daily_window_dates() -> list[str]:
    start = _window_start().normalize()
    end = _window_cutoff().normalize()
    if end < start:
        raise ValueError("fixed_cutoff_utc is earlier than window_start_utc")
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]


def _audit_monthly_klines(
    data: pd.DataFrame,
    label: str,
    *,
    max_missing_rows: int = 0,
    max_missing_ratio: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "open_time" not in data.columns:
        raise RuntimeError(f"V9.7.2 monthly kline audit missing open_time for {label}")

    raw_open_time = pd.to_numeric(data["open_time"], errors="coerce")
    invalid_timestamps = int(raw_open_time.isna().sum())
    valid_open_time = raw_open_time.dropna().astype("int64")
    duplicates = int(valid_open_time.duplicated().sum())

    data = data.loc[raw_open_time.notna()].copy()
    data["open_time"] = valid_open_time.to_numpy()
    data = data.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    start = pd.Timestamp("2025-01-01T00:00:00Z")
    cutoff = _window_cutoff()
    last_open = cutoff.floor("5min")
    expected = np.arange(
        int(start.timestamp() * 1000),
        int(last_open.timestamp() * 1000) + core.base.STEP_MS,
        core.base.STEP_MS,
        dtype=np.int64,
    )
    actual = data["open_time"].astype("int64").to_numpy()
    unique_actual = np.unique(actual)
    missing = np.setdiff1d(expected, unique_actual)
    extra = np.setdiff1d(unique_actual, expected)
    missing_rows = int(len(missing))
    missing_ratio = float(missing_rows / len(expected)) if len(expected) else 0.0

    structural_integrity_passed = bool(
        invalid_timestamps == 0 and len(extra) == 0 and duplicates == 0
    )
    gap_budget_passed = bool(
        missing_rows <= int(max_missing_rows)
        and missing_ratio <= float(max_missing_ratio)
    )
    passed = bool(structural_integrity_passed and gap_budget_passed)

    audit = {
        "label": label,
        "source_mode": "VERIFIED_MONTHLY_ARCHIVES_ONLY",
        "start_utc": start.isoformat(),
        "cutoff_utc": cutoff.isoformat(),
        "last_included_bar_utc": last_open.isoformat(),
        "expected_rows": int(len(expected)),
        "actual_rows": int(len(data)),
        "missing_rows": missing_rows,
        "missing_ratio": missing_ratio,
        "extra_rows": int(len(extra)),
        "duplicate_timestamps": duplicates,
        "invalid_timestamps": invalid_timestamps,
        "max_missing_rows": int(max_missing_rows),
        "max_missing_ratio": float(max_missing_ratio),
        "structural_integrity_passed": structural_integrity_passed,
        "gap_budget_passed": gap_budget_passed,
        "first_missing_utc": (
            pd.to_datetime(missing[0], unit="ms", utc=True).isoformat()
            if missing_rows else None
        ),
        "last_missing_utc": (
            pd.to_datetime(missing[-1], unit="ms", utc=True).isoformat()
            if missing_rows else None
        ),
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(
            f"V9.7.2 monthly kline audit failed for {label}: "
            f"{json.dumps(audit, ensure_ascii=False)}"
        )
    return data, audit


def _expected_grid(start: pd.Timestamp, last_open: pd.Timestamp) -> np.ndarray:
    return np.arange(
        int(start.timestamp() * 1000),
        int(last_open.timestamp() * 1000) + core.base.STEP_MS,
        core.base.STEP_MS,
        dtype=np.int64,
    )


def _missing_grid_times(data: pd.DataFrame, start: pd.Timestamp, last_open: pd.Timestamp) -> np.ndarray:
    expected = _expected_grid(start, last_open)
    actual = pd.to_numeric(data["open_time"], errors="coerce").dropna().astype("int64").to_numpy()
    return np.setdiff1d(expected, np.unique(actual))


def _repair_premium_from_verified_daily_archives(
    premium: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Patch historical premium gaps with Binance verified DAILY archives only.

    No REST endpoint is used. The daily ZIP and its CHECKSUM are both verified by
    the frozen base-engine downloader. This is a source repair, not imputation.
    """
    repair_cfg = DIAGNOSTIC.get("premium_gap_repair", {})
    enabled = bool(repair_cfg.get("enabled", False))
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    last_open = _window_cutoff().floor("5min")
    missing_before = _missing_grid_times(premium, start, last_open)
    days = sorted({pd.to_datetime(v, unit="ms", utc=True).strftime("%Y-%m-%d") for v in missing_before})
    audit: dict[str, Any] = {
        "enabled": enabled,
        "source": "Binance USDⓈ-M Futures verified daily premiumIndexKlines archives",
        "uses_rest_fallback": False,
        "missing_rows_before": int(len(missing_before)),
        "missing_days_before": days,
        "max_repair_days": int(repair_cfg.get("max_repair_days", 0)),
        "files": [],
    }
    if not missing_before.size:
        audit.update({"missing_rows_after": 0, "missing_days_after": [], "repair_passed": True})
        return premium, audit
    if not enabled:
        raise RuntimeError(
            "PREMIUM_DAILY_GAP_REPAIR_DISABLED: verified monthly premium archive has "
            f"{len(missing_before)} missing rows across {days}"
        )
    max_days = int(repair_cfg.get("max_repair_days", 1))
    if len(days) > max_days:
        raise RuntimeError(
            f"PREMIUM_DAILY_GAP_REPAIR_LIMIT: missing days={days}, max_repair_days={max_days}"
        )

    frames = [premium.copy()]
    for day in days:
        name = f"{core.SYMBOL}-{core.INTERVAL}-{day}.zip"
        base = (
            "https://data.binance.vision/data/futures/um/daily/"
            f"premiumIndexKlines/{core.SYMBOL}/{core.INTERVAL}"
        )
        try:
            raw, digest = core.base.read_verified_zip(
                f"{base}/{name}",
                f"{base}/{name}.CHECKSUM",
                f"daily-premiumIndexKlines-{name}",
            )
            frame = core.base.parse_kline_csv(core.base.read_single_csv_zip(raw, name))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "PREMIUM_VERIFIED_DAILY_ARCHIVE_UNAVAILABLE: "
                f"{day}; monthly gap cannot be repaired without changing frozen features; "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        day_start = int(pd.Timestamp(f"{day}T00:00:00Z").timestamp() * 1000)
        day_end = day_start + 24 * 60 * 60 * 1000
        frame = frame[
            (pd.to_numeric(frame["open_time"], errors="coerce") >= day_start)
            & (pd.to_numeric(frame["open_time"], errors="coerce") < day_end)
        ].copy()
        if frame.empty:
            raise RuntimeError(f"PREMIUM_DAILY_ARCHIVE_EMPTY: {day}")
        frames.append(frame)
        audit["files"].append({"file": name, "sha256": digest, "rows": int(len(frame))})

    repaired = (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values("open_time")
        .drop_duplicates("open_time", keep="last")
        .reset_index(drop=True)
    )
    missing_after = _missing_grid_times(repaired, start, last_open)
    days_after = sorted({pd.to_datetime(v, unit="ms", utc=True).strftime("%Y-%m-%d") for v in missing_after})
    audit.update({
        "missing_rows_after": int(len(missing_after)),
        "missing_days_after": days_after,
        "repaired_rows_added": int(len(repaired) - len(premium)),
        "repair_passed": bool(len(missing_after) == 0),
    })
    if missing_after.size:
        raise RuntimeError(
            "PREMIUM_DAILY_GAP_REPAIR_INCOMPLETE: "
            f"remaining_rows={len(missing_after)}, remaining_days={days_after}"
        )
    return repaired, audit


def _monthly_quality_budget(label: str) -> tuple[int, float]:
    rule = MONTHLY_DATA_QUALITY.get(label)
    if not isinstance(rule, dict):
        raise RuntimeError(f"V9.7.2 missing monthly_data_quality rule for {label}")
    return int(rule["max_missing_rows"]), float(rule["max_missing_ratio"])


def _load_q2_monthly_frames(ref: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    monthly_months = _periods("2025-01", str(DIAGNOSTIC["end_month"]))
    old_core_months = core.base.MONTHS
    old_ref_months = ref.base.MONTHS
    core.base.MONTHS = tuple(monthly_months)
    ref.base.MONTHS = tuple(monthly_months)
    try:
        raw, raw_source_audit = core.base.load_official_data()
        eth, eth_source_audit = core.base.load_auxiliary_kline("ETHUSDT", "klines")
        premium, premium_source_audit = core.base.load_auxiliary_kline(core.SYMBOL, "premiumIndexKlines")
        funding, funding_source_audit = core.base.load_funding_rate()
    finally:
        core.base.MONTHS = old_core_months
        ref.base.MONTHS = old_ref_months

    premium, premium_daily_repair_audit = _repair_premium_from_verified_daily_archives(premium)

    raw_budget = _monthly_quality_budget("BTCUSDT_5m")
    eth_budget = _monthly_quality_budget("ETHUSDT_5m")
    premium_budget = _monthly_quality_budget("BTCUSDT_premiumIndexKlines_5m")
    raw, raw_audit = _audit_monthly_klines(
        raw, "BTCUSDT_5m", max_missing_rows=raw_budget[0], max_missing_ratio=raw_budget[1]
    )
    eth, eth_audit = _audit_monthly_klines(
        eth, "ETHUSDT_5m", max_missing_rows=eth_budget[0], max_missing_ratio=eth_budget[1]
    )
    premium, premium_audit = _audit_monthly_klines(
        premium, "BTCUSDT_premiumIndexKlines_5m",
        max_missing_rows=premium_budget[0], max_missing_ratio=premium_budget[1],
    )
    cutoff_ms = int(_window_cutoff().timestamp()*1000)
    funding = funding[pd.to_numeric(funding["calc_time"], errors="coerce") <= cutoff_ms].reset_index(drop=True)

    x, align = core.base.add_features(raw, eth, premium, funding)
    x = core.add_sparse_masks(x)
    rx, ralign = ref.base.add_features(raw, eth, premium, funding)
    rx = ref.add_sparse_masks(rx)
    if not x.index.equals(rx.index):
        raise RuntimeError("V9.7.2 Q2 feature-frame index mismatch")
    if x.empty:
        raise RuntimeError("V9.7.2 Q2 feature frame is empty")
    target_last_bar = _window_cutoff().floor("5min")
    feature_tail_gap = target_last_bar - pd.Timestamp(x.index[-1])
    if feature_tail_gap < pd.Timedelta(0) or feature_tail_gap > pd.Timedelta(minutes=60):
        raise RuntimeError(
            f"Q2 usable feature frame ended at {x.index[-1]}; allowed tail gap is 0-60 minutes from {target_last_bar}"
        )
    return rx, x, {
        "window_type": DIAGNOSTIC["window_type"],
        "window_start_utc": _window_start().isoformat(),
        "window_end_utc": _window_cutoff().isoformat(),
        "last_raw_5m_bar_utc": target_last_bar.isoformat(),
        "last_usable_feature_bar_utc": pd.Timestamp(x.index[-1]).isoformat(),
        "feature_tail_gap_minutes": float(feature_tail_gap / pd.Timedelta(minutes=1)),
        "monthly_months": monthly_months,
        "uses_monthly_archives_only": False,
        "uses_monthly_archives_as_primary": True,
        "uses_daily_archives": bool(premium_daily_repair_audit.get("files")),
        "daily_archives_for_premium_gap_repair_only": True,
        "uses_rest_fallback": False,
        "official_source_audit": raw_source_audit, "eth_source_audit": eth_source_audit,
        "premium_source_audit": premium_source_audit,
        "premium_daily_repair_audit": premium_daily_repair_audit,
        "funding_source_audit": funding_source_audit,
        "official_monthly_audit": raw_audit, "eth_monthly_audit": eth_audit,
        "premium_monthly_audit": premium_audit, "funding_rows": int(len(funding)),
        "extended_alignment": align, "reference_alignment": ralign,
    }


def _diagnostic_selected_for_experts(ref: Any, rx: pd.DataFrame, x: pd.DataFrame, tiers: pd.DataFrame, blind_months: list[str], expanded_months: list[str]) -> pd.DataFrame:
    frames=[]
    reference_ids={int(v) for v in EXPERT_COVERAGE["v961_reference_ids"]}
    for _,row in tiers.iterrows():
        eid=int(row["expert_id"]); key=str(row["policy_key"])
        mod,frame=(ref,rx) if eid in reference_ids else (core,x)
        expert,policy=_policy_for(mod,eid,key)
        with _with_blind_months(mod,expanded_months):
            _,selected=trace_no_monthly_budget_lane(mod,frame,expert,policy,blind_months)
        if selected.empty: continue
        selected=selected.copy(); selected["expert_id"]=eid; selected["expert"]=expert.name; selected["family"]=expert.family; selected["setup_group"]=expert.setup_group
        selected["lane"]="V9.7.2_DIAGNOSTIC_NO_MONTHLY_BUDGET_ORIGINAL"; selected["rating_engine"]="V9.6.1_REFERENCE" if mod is ref else "V9.6.9_FROZEN_EXTENDED_ORIGINAL"
        selected=_add_event_times(selected,frame); frames.append(selected)
    return pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame()


def _diagnostic_seed_risk_lanes(ref: Any, rx: pd.DataFrame, tiers: pd.DataFrame, blind_months: list[str], expanded_months: list[str]) -> pd.DataFrame:
    seed_id=int(REQUEST["robustness"]["seed_expert_id"])
    row=tiers.loc[tiers["expert_id"].astype(int)==seed_id].iloc[0]
    expert,policy=_policy_for(ref,seed_id,str(row["policy_key"]))
    frames=[]
    with _with_blind_months(ref,expanded_months):
        _,monthly=trace_policy_lane(ref,rx,expert,policy,"V9.6.1_MONTHLY_BUDGET",blind_months)
        _,no_month=trace_no_monthly_budget_lane(ref,rx,expert,policy,blind_months)
    for lane,frame in [("V9.6.1_MONTHLY_BUDGET",monthly),("V9.6.7_NO_MONTHLY_BUDGET",no_month)]:
        if frame.empty: continue
        q=frame.copy();q["lane"]=lane;q["expert_id"]=seed_id;q["expert"]=expert.name;q["family"]=expert.family;q["setup_group"]=expert.setup_group;q=_add_event_times(q,rx);frames.append(q)
    return pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame()


def _q2_outputs(previous_q2_expert_ledger: pd.DataFrame, previous_q2_risk_ledger: pd.DataFrame, ref: Any, tiers: pd.DataFrame, rating_shadow: pd.DataFrame, dev_months: list[str]) -> dict[str,Any]:
    window_months = _periods(str(DIAGNOSTIC["start_month"]), str(DIAGNOSTIC["end_month"]))
    rx, x, data_audit = _load_q2_monthly_frames(ref)
    expanded = _periods("2025-01", str(DIAGNOSTIC["end_month"]))

    registry = [{
        "month": month, "state": "HISTORICAL_DIAGNOSTIC_EVALUATED",
        "calendar_complete": True, "archive_available": True,
        "window_type": str(DIAGNOSTIC["window_type"]),
        "used_for_evaluation": True, "used_for_tuning": False,
        "used_for_winner_selection": False,
    } for month in window_months]
    pd.DataFrame(registry).to_csv(RESULTS/"q2_month_registry.csv", index=False)

    expert_ledger = _diagnostic_selected_for_experts(ref, rx, x, tiers, window_months, expanded)
    risk_ledger = _diagnostic_seed_risk_lanes(ref, rx, tiers, window_months, expanded)
    cutoff = _window_cutoff()
    for label, frame in (("expert", expert_ledger), ("risk", risk_ledger)):
        if not frame.empty:
            exit_times = pd.to_datetime(frame["exit_time_utc"], utc=True, errors="coerce")
            if exit_times.isna().any() or bool((exit_times > cutoff).any()):
                raise RuntimeError(f"{label} Q2 ledger contains unresolved or post-window exit")
    expert_ledger.to_csv(RESULTS/"q2_expert_trade_ledger.csv", index=False)
    risk_ledger.to_csv(RESULTS/"q2_risk_budget_trade_ledger.csv", index=False)

    expert_rows=[]
    for _, row in tiers.iterrows():
        eid=int(row["expert_id"]); key=str(row["policy_key"])
        q=expert_ledger[(expert_ledger.get("expert_id",pd.Series(dtype=int)).astype(int)==eid) & (expert_ledger.get("policy_key",pd.Series(dtype=str)).astype(str)==key)].copy() if not expert_ledger.empty else pd.DataFrame()
        expert_rows.append({
            "expert_id":eid,"expert":row["expert"],"policy_key":key,"frozen_tier":row["tier"],
            "window_start_utc":_window_start().isoformat(),"window_end_utc":_window_cutoff().isoformat(),
            "historical_diagnostic_only":True,"automatic_activation":False,**base_metrics(q),
        })
    expert_audit=pd.DataFrame(expert_rows)
    expert_audit.to_csv(RESULTS/"q2_expert_diagnostic_audit.csv",index=False)

    lane_progress={}
    for lane in ["V9.6.1_MONTHLY_BUDGET","V9.6.7_NO_MONTHLY_BUDGET"]:
        q=risk_ledger[risk_ledger.get("lane",pd.Series(dtype=str)).astype(str)==lane] if not risk_ledger.empty else pd.DataFrame()
        lane_progress[lane]={**base_metrics(q),"complete_calendar_months":len(window_months),"historical_diagnostic_only":True}
    progress={
        "window_type":str(DIAGNOSTIC["window_type"]),"window_start_utc":_window_start().isoformat(),
        "window_end_utc":_window_cutoff().isoformat(),"window_months":window_months,
        "complete_month_count":len(window_months),"window_evaluated":True,"risk_lanes":lane_progress,
        "decision_ready":False,"winner_selection_enabled":False,
        "winner_decision":"HISTORICAL_DIAGNOSTIC_ONLY_NO_WINNER",
        "automatic_tier_activation_enabled":False,
    }
    (RESULTS/"q2_diagnostic_progress.json").write_text(json.dumps(progress,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    readiness={
        "decision_ready":False,"window_type":str(DIAGNOSTIC["window_type"]),
        "winner_selection_enabled":False,"automatic_tier_activation_enabled":False,
        "action":"report historical Q2 diagnostics only; do not tune or select a winner",
    }
    (RESULTS/"q2_decision_readiness.json").write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding="utf-8")
    summary={
        "window_type":str(DIAGNOSTIC["window_type"]),"window_start_utc":_window_start().isoformat(),
        "window_end_utc":_window_cutoff().isoformat(),"months_included":window_months,
        "days_included":len(_daily_window_dates()),"last_raw_5m_bar_utc":data_audit["last_raw_5m_bar_utc"],
        "last_usable_feature_bar_utc":data_audit["last_usable_feature_bar_utc"],
        "feature_tail_gap_minutes":data_audit["feature_tail_gap_minutes"],
        "expert_shadow_trades":int(len(expert_ledger)),"risk_budget_trades":int(len(risk_ledger)),
        "uses_monthly_archives_only":False,"uses_monthly_archives_as_primary":True,
        "uses_daily_archives":bool(data_audit.get("premium_daily_repair_audit",{}).get("files")),
        "daily_archives_for_premium_gap_repair_only":True,"uses_rest_fallback":False,
        "historical_diagnostic_only":True,
    }
    (RESULTS/"q2_diagnostic_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    integrity={
        "static_historical_diagnostic":True,"expert_ledger_rows":int(len(expert_ledger)),
        "risk_budget_ledger_rows":int(len(risk_ledger)),"monthly_data_audit":data_audit,
        "window_start_utc":_window_start().isoformat(),"window_end_utc":_window_cutoff().isoformat(),
        "all_recorded_exits_at_or_before_window_end":True,
        "uses_monthly_archives_only":False,"uses_monthly_archives_as_primary":True,
        "uses_daily_archives":bool(data_audit.get("premium_daily_repair_audit",{}).get("files")),
        "daily_archives_for_premium_gap_repair_only":True,"uses_rest_fallback":False,
        "data_used_for_policy_search":False,"data_used_for_threshold_tuning":False,
        "data_used_for_signal_formula_changes":False,
    }
    (RESULTS/"q2_integrity_audit.json").write_text(json.dumps(integrity,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    return {"progress":progress,"integrity":integrity,"expert_audit":expert_audit}

def premium_daily_gap_repair_self_test() -> None:
    start = pd.Timestamp("2026-06-01T00:00:00Z")
    end = pd.Timestamp("2026-06-02T23:55:00Z")
    full = _expected_grid(start, end)
    gap_start = int(pd.Timestamp("2026-06-02T00:00:00Z").timestamp() * 1000)
    gap_end = gap_start + 24 * 60 * 60 * 1000
    partial = full[(full < gap_start) | (full >= gap_end)]
    frame = pd.DataFrame({"open_time": partial})
    missing = _missing_grid_times(frame, start, end)
    assert len(missing) == 288
    patch = pd.DataFrame({"open_time": full[(full >= gap_start) & (full < gap_end)]})
    merged = pd.concat([frame, patch], ignore_index=True).drop_duplicates("open_time")
    assert len(_missing_grid_times(merged, start, end)) == 0
    print("V972_PREMIUM_VERIFIED_DAILY_GAP_REPAIR_SELF_TEST_OK")


def q2_diagnostic_self_test() -> None:
    assert _periods("2026-04", "2026-06") == ["2026-04", "2026-05", "2026-06"]
    assert _window_start() == pd.Timestamp("2026-04-01T00:00:00Z")
    assert _window_cutoff() == pd.Timestamp("2026-06-30T23:59:59Z")
    assert len(_daily_window_dates()) == 91
    assert _daily_window_dates()[0] == "2026-04-01"
    assert _daily_window_dates()[-1] == "2026-06-30"
    assert DIAGNOSTIC["window_type"] == "FULL_MONTH_HISTORICAL_DIAGNOSTIC"
    assert DIAGNOSTIC["uses_monthly_archives_only"] is False
    assert DIAGNOSTIC["uses_monthly_archives_as_primary"] is True
    assert DIAGNOSTIC["uses_daily_archives"] is True
    assert DIAGNOSTIC["daily_archives_for_premium_gap_repair_only"] is True
    assert DIAGNOSTIC["uses_rest_fallback"] is False
    assert DIAGNOSTIC["premium_gap_repair"]["enabled"] is True
    assert DIAGNOSTIC["premium_gap_repair"]["checksum_required"] is True
    assert DIAGNOSTIC["premium_gap_repair"]["rest_fallback_allowed"] is False
    assert DIAGNOSTIC["premium_gap_repair"]["imputation_allowed"] is False
    start_ms=int(pd.Timestamp("2025-01-01T00:00:00Z").timestamp()*1000)
    last_ms=int(pd.Timestamp("2026-06-30T23:55:00Z").timestamp()*1000)
    times=np.arange(start_ms,last_ms+core.base.STEP_MS,core.base.STEP_MS,dtype=np.int64)
    combined,audit=_audit_monthly_klines(
        pd.DataFrame({"open_time":times}), "SELF_TEST",
        max_missing_rows=0, max_missing_ratio=0.0,
    )
    assert audit["passed"] and len(combined)==len(times)

    # Main BTC/ETH bars remain strict: one missing 5-minute bar must fail.
    try:
        _audit_monthly_klines(
            pd.DataFrame({"open_time":times[:-1]}), "STRICT_GAP_SELF_TEST",
            max_missing_rows=0, max_missing_ratio=0.0,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Strict BTC/ETH audit must reject one missing bar")

    # Premium index is auxiliary data. One verified missing day (288 bars) is allowed,
    # matching the frozen engine's past-only <=60-minute fill and feature cleaning.
    premium_gap = pd.DataFrame({"open_time":times[:-288]})
    _, premium_audit = _audit_monthly_klines(
        premium_gap, "PREMIUM_GAP_SELF_TEST",
        max_missing_rows=288, max_missing_ratio=0.002,
    )
    assert premium_audit["passed"] is True
    assert premium_audit["missing_rows"] == 288
    assert premium_audit["gap_budget_passed"] is True

    try:
        _audit_monthly_klines(
            pd.DataFrame({"open_time":times[:-289]}), "PREMIUM_EXCESSIVE_GAP_SELF_TEST",
            max_missing_rows=288, max_missing_ratio=0.002,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Premium audit must reject gaps above 288 bars")

    assert _monthly_quality_budget("BTCUSDT_5m") == (0, 0.0)
    assert _monthly_quality_budget("ETHUSDT_5m") == (0, 0.0)
    assert _monthly_quality_budget("BTCUSDT_premiumIndexKlines_5m") == (288, 0.002)
    assert DIAGNOSTIC["winner_selection_enabled"] is False
    assert DIAGNOSTIC["automatic_tier_activation_enabled"] is False
    print("V972_Q2_HISTORICAL_DIAGNOSTIC_SELF_TEST_OK")


def main()->None:
    previous_q2_expert_ledger = _read_csv_or_empty(RESULTS / "q2_expert_trade_ledger.csv", DIAGNOSTIC_LEDGER_COLUMNS)
    previous_q2_risk_ledger = _read_csv_or_empty(RESULTS / "q2_risk_budget_trade_ledger.csv", DIAGNOSTIC_LEDGER_COLUMNS)
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
    if not x.index.equals(rx.index):
        raise RuntimeError(
            "V9.7.2 rating feature-frame index mismatch: "
            f"extended_rows={len(x)}, reference_rows={len(rx)}, "
            f"extended_start={x.index.min() if len(x) else None}, reference_start={rx.index.min() if len(rx) else None}"
        )

    # V9.7.2 invariant: final expert evidence is rebuilt from the complete no-monthly-budget
    # original-selection shadow for every selected expert policy. The fixed-penalty core
    # shadow remains policy-selection evidence only and cannot assign the final tier.
    rating_shadow, coverage_df = build_unified_rating_shadow(ref, rx, x, initial_tiers, trace_months)
    rating_shadow.to_csv(RESULTS / "v969_unified_rating_shadow_trades.csv", index=False)
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
    gated_rating_shadow.to_csv(RESULTS / "v969_unified_gated_shadow_trades.csv", index=False)
    research_execution = gated_rating_shadow[gated_rating_shadow.get("risk_multiplier",pd.Series(dtype=float)).astype(float)>0].copy() if not gated_rating_shadow.empty else gated_rating_shadow.copy()
    formal_ok = int((new_tiers["tier"] == "QUALIFIED").sum()) >= int(GATE["minimum_qualified_experts_for_formal"])
    formal_execution = gated_rating_shadow[gated_rating_shadow.get("formal_eligible",pd.Series(dtype=bool)).fillna(False).astype(bool)].copy() if formal_ok and not gated_rating_shadow.empty else gated_rating_shadow.iloc[0:0].copy()
    research_execution.to_csv(RESULTS / "v969_research_execution_trades.csv", index=False)
    formal_execution.to_csv(RESULTS / "v969_formal_execution_trades.csv", index=False)

    research_portfolio, research_dedup, research_conflicts = build_unified_portfolio(gated_rating_shadow, {"WATCH","QUALIFIED"}, risk_adjusted=True)
    qualified_portfolio, qualified_dedup, qualified_conflicts = build_unified_portfolio(gated_rating_shadow, {"QUALIFIED"}, risk_adjusted=False)
    fallback_cols=list(gated_rating_shadow.columns) if len(gated_rating_shadow.columns) else ["month","net_r","expert","expert_id","policy_key"]
    if research_portfolio.empty: research_portfolio=pd.DataFrame(columns=fallback_cols)
    if qualified_portfolio.empty: qualified_portfolio=pd.DataFrame(columns=fallback_cols)
    for name in ["watch_portfolio_trades.csv","research_portfolio_trades.csv","portfolio_trades.csv","trades.csv"]:
        research_portfolio.to_csv(RESULTS/name,index=False)
    qualified_portfolio.to_csv(RESULTS/"qualified_portfolio_trades.csv",index=False)
    research_dedup.to_csv(RESULTS/"v969_research_portfolio_dedup.csv",index=False)
    research_conflicts.to_csv(RESULTS/"v969_research_portfolio_conflicts.csv",index=False)
    qualified_dedup.to_csv(RESULTS/"v969_qualified_portfolio_dedup.csv",index=False)
    qualified_conflicts.to_csv(RESULTS/"v969_qualified_portfolio_conflicts.csv",index=False)

    # Blind evidence uses the same unified shadow source, never the core fixed-penalty ledger.
    blind=[]
    for _,row in new_tiers.iterrows():
        eid=int(row["expert_id"]);policy=str(row["policy_key"]);expert=str(row["expert"])
        q=rating_shadow[(rating_shadow.get("expert_id",pd.Series(dtype=int))==eid)&(rating_shadow.get("policy_key",pd.Series(dtype=str)).astype(str)==policy)&(rating_shadow.get("month",pd.Series(dtype=str)).astype(str)==core.OOS_MONTH)].copy() if not rating_shadow.empty else pd.DataFrame()
        bm=base_metrics(q);months_elapsed=1
        blind.append({"expert_id":eid,"expert":expert,"policy_key":policy,"development_tier":row["tier"],"rating_source":str(RATING_SOURCE["source_lane"]),"blind_months_elapsed":months_elapsed,"blind_trades":bm["trades"],"blind_wins":bm["wins"],"blind_raw_win_rate":bm["win_rate"],"watch_evidence_complete":months_elapsed>=int(STAT["blind_min_calendar_months_watch"]) and bm["trades"]>=int(STAT["blind_min_trades_watch"]),"qualified_evidence_complete":months_elapsed>=int(STAT["blind_min_calendar_months_qualified"]) and bm["trades"]>=int(STAT["blind_min_trades_qualified"])})
    blind_payload={"diagnostic_oos_months":[core.OOS_MONTH],"rule":"minimum calendar months AND minimum trades must both be met","rating_source":str(RATING_SOURCE["source_lane"]),"experts":blind}
    (RESULTS/"baseline_seen_oos_status.json").write_text(json.dumps(blind_payload,ensure_ascii=False,indent=2),encoding="utf-8")

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
    snapshot={"engine":ENGINE_NAME,"engine_version":"V9.7.2","seed_expert_id":seed_id,"seed_expert":expert.name,"policy_key":seed_key,"policy":asdict(policy),"feature_group":expert.feature_group,"feature_list":list(core.FEATURE_GROUPS[expert.feature_group]),"training_months":list(core.DEVELOPMENT_MONTHS),"diagnostic_oos_month":core.OOS_MONTH,"execution":{"fee_rate_per_side":core.FEE_RATE,"slippage_abs":core.SLIPPAGE_ABS,"next_bar_open":True,"same_candle_stop_before_tp":True},"model_runtime":REQUEST["model"],"statistical_validation":STAT,"code_hashes":{"entrypoint":file_sha256(ROOT/"autonomous_backtest_v9_7_2.py"),"strategy_core":file_sha256(ROOT/"_v972_strategy_core.py"),"base_engine":file_sha256(ROOT/"_v972_base_engine.py"),"request":file_sha256(REQUEST_PATH)},"official_data_fingerprint":data_fp,"signal_fingerprint":original_fp,"random_seed":REQUEST["model"]["base_seed"]}
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
    v967_gated.to_csv(RESULTS / "v969_seed_no_monthly_budget_shadow_trades.csv", index=False)
    v967_gated[v967_gated["risk_multiplier"] > 0].to_csv(RESULTS / "v969_seed_research_execution_trades.csv", index=False)
    formal_ok = int((new_tiers["tier"] == "QUALIFIED").sum()) >= int(GATE["minimum_qualified_experts_for_formal"])
    if formal_ok:
        v967_gated[v967_gated["formal_eligible"]].to_csv(RESULTS / "v969_seed_formal_execution_trades.csv", index=False)
    else:
        v967_gated.iloc[0:0].to_csv(RESULTS / "v969_seed_formal_execution_trades.csv", index=False)

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
    required_expert_ids = {int(x) for x in EXPERT_COVERAGE["required_expert_ids"]}
    covered_expert_ids = set(coverage_df.loc[coverage_df["expert_found_in_rating_engine"].astype(bool), "expert_id"].astype(int)) if not coverage_df.empty else set()
    policy_coverage_complete = bool(coverage_df["policy_found_in_rating_engine"].all()) if not coverage_df.empty else False
    feature_frame_coverage_complete = bool(coverage_df["feature_frame_column_found"].all()) if not coverage_df.empty else False
    expert_id_coverage_complete = covered_expert_ids == required_expert_ids
    if bool(EXPERT_COVERAGE.get("require_complete_expert_coverage", True)) and not expert_id_coverage_complete:
        missing_ids = sorted(required_expert_ids - covered_expert_ids)
        extra_ids = sorted(covered_expert_ids - required_expert_ids)
        raise RuntimeError(f"V9.7.2 expert coverage incomplete: missing={missing_ids}, extra={extra_ids}")
    if bool(EXPERT_COVERAGE.get("require_complete_policy_coverage", True)) and not policy_coverage_complete:
        raise RuntimeError("V9.7.2 policy coverage incomplete; inspect rating_source_policy_coverage.csv")
    if not feature_frame_coverage_complete:
        bad = coverage_df.loc[
            ~coverage_df["feature_frame_column_found"].astype(bool),
            ["expert_id", "rating_engine", "feature_frame", "required_mask_column"],
        ].to_dict("records")
        raise RuntimeError(f"V9.7.2 feature-frame coverage incomplete: {bad}")
    rating_audit={
        "final_rating_source":str(RATING_SOURCE["source_lane"]),
        "core_fixed_penalty_shadow_for_final_tier":False,
        "all_selected_experts_replayed":bool(RATING_SOURCE["all_selected_experts_replayed"]),
        "policy_coverage_complete": policy_coverage_complete,
        "feature_frame_coverage_complete": feature_frame_coverage_complete,
        "feature_frame_index_match": True,
        "expert_id_coverage_complete": expert_id_coverage_complete,
        "required_expert_count": int(EXPERT_COVERAGE["required_expert_count"]),
        "covered_expert_count": len(covered_expert_ids),
        "required_expert_ids": sorted(required_expert_ids),
        "covered_expert_ids": sorted(covered_expert_ids),
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

    diagnostic_result = _q2_outputs(previous_q2_expert_ledger, previous_q2_risk_ledger, ref, new_tiers, rating_shadow, dev_months)

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
        "release_version": "9.7.2",
        "release_type": "q2_historical_diagnostic",
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
        "rating_source_policy_coverage_complete": policy_coverage_complete,
        "rating_source_feature_frame_coverage_complete": feature_frame_coverage_complete,
        "rating_source_feature_frame_index_match": True,
        "rating_source_expert_coverage_complete": expert_id_coverage_complete,
        "rating_source_required_expert_count": int(EXPERT_COVERAGE["required_expert_count"]),
        "rating_source_covered_expert_count": len(covered_expert_ids),
        "rating_source_matches_no_budget_lane": rating_source_matches_no_budget,
        "rating_metrics_match_no_budget_lane": rating_metrics_match_no_budget,
        "rating_ignores_execution_budget": bool(GATE["rating_ignores_execution_budget"]),
        "rating_gate_ignored_fields": list(GATE.get("rating_gate_ignored_fields", [])),
        "seed_core_fixed_penalty_trades": int(seed_compare["core_fixed_penalty_trades"]),
        "seed_unified_rating_trades": int(seed_compare["unified_rating_trades"]),
        "risk_budget_dual_track_enabled": True,
        "risk_budget_winner_selection_enabled": False,
        "signal_selection_independent_of_tier": gate_independent,
        "v969_no_monthly_budget_lane_trace_exact": v967_trace_exact,
        "v969_seed_tier": seed_tier,
        "v969_seed_risk_multiplier": float(GATE["tier_risk_multiplier"][seed_tier]),
        "v969_seed_execution_mode": str(GATE["tier_execution_mode"][seed_tier]),
        "v969_no_monthly_budget_enabled": not bool(ORIGINAL["monthly_cap_enabled"]),
        "v969_cross_trade_cluster_dedup": bool(ORIGINAL["cross_trade_cluster_dedup"]),
        "winner_selection_enabled": False,
        "winner_decision": "NO_WINNER_SELECTION_FROM_SEEN_OOS",
        "fixed_stage_trace_exact": fixed_trace_exact,
        "reference_stage_trace_exact": reference_trace_exact,
        "shrinkage_stage_trace_exact": shrink_trace_exact,
        "historical_diagnostic_window_evaluated": bool(diagnostic_result["progress"]["window_evaluated"]),
        "diagnostic_window_type": str(DIAGNOSTIC["window_type"]),
        "diagnostic_window_start_utc": str(DIAGNOSTIC["window_start_utc"]),
        "diagnostic_window_end_utc": str(DIAGNOSTIC["window_end_utc"]),
        "diagnostic_complete_month_count": int(diagnostic_result["progress"]["complete_month_count"]),
        "diagnostic_winner_selection_enabled": False,
        "diagnostic_automatic_tier_activation_enabled": False,
        "constraints": {
            **status.get("constraints", {}),
            "watch_gate": WATCH_GATE,
            "qualified_gate": QUALIFIED_GATE,
            "statistical_validation": STAT,
            "block_validation": BLOCK,
            "quad_track_validation": TRACK,
            "evidence_shrinkage_lane": SHRINK,
            "expert_level_gating": GATE,
            "v969_no_monthly_budget_lane": ORIGINAL,
            "risk_budget_dual_track": RISK_BUDGET,
            "rating_data_source": RATING_SOURCE,
            "expert_coverage": EXPERT_COVERAGE,
            "historical_diagnostic_window": DIAGNOSTIC
        }
    })
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    seed_summary = summaries[seed_id]
    lanes = quad_payload["lanes"]
    q2 = diagnostic_result["progress"]["risk_lanes"]
    report = f"""# BTCUSDT 5分钟 2026年第二季度历史诊断回测 V9.7.2 报告

- 回测窗口：**2026-04-01至2026-06-30 UTC**。
- 数据：只使用Binance官方月度归档；不下载7月日度数据，不调用Binance REST备用接口。
- 历史数据从2025年1月开始仅用于指标预热与逐月模型训练；报告交易只统计2026年4月至6月。
- 信号公式、专家策略键、统计门槛和两条风险预算均保持冻结。
- 本窗口已经被查看，只能作为历史诊断，禁止用于宣称盲测、自动选胜者或自动晋级。
- 当前专家等级：正式 {counts['QUALIFIED']}；观察 {counts['WATCH']}；候选 {counts['CANDIDATE']}；淘汰 {counts['REJECTED']}。
- 实盘资格：**不合格**。

## 2026年4月至6月风险预算对照

| 风险预算 | 交易 | 胜率 | 盈亏比 | 盈利因子 | 净R | 最大回撤R |
|---|---:|---:|---:|---:|---:|---:|
| V9.6.1原始月度预算 | {q2['V9.6.1_MONTHLY_BUDGET']['trades']} | {q2['V9.6.1_MONTHLY_BUDGET']['win_rate']:.2%} | {q2['V9.6.1_MONTHLY_BUDGET']['avg_win_loss_ratio']:.3f} | {q2['V9.6.1_MONTHLY_BUDGET']['profit_factor']:.3f} | {q2['V9.6.1_MONTHLY_BUDGET']['net_R']:.3f} | {q2['V9.6.1_MONTHLY_BUDGET']['max_drawdown_R']:.3f} |
| V9.6.7无月度预算 | {q2['V9.6.7_NO_MONTHLY_BUDGET']['trades']} | {q2['V9.6.7_NO_MONTHLY_BUDGET']['win_rate']:.2%} | {q2['V9.6.7_NO_MONTHLY_BUDGET']['avg_win_loss_ratio']:.3f} | {q2['V9.6.7_NO_MONTHLY_BUDGET']['profit_factor']:.3f} | {q2['V9.6.7_NO_MONTHLY_BUDGET']['net_R']:.3f} | {q2['V9.6.7_NO_MONTHLY_BUDGET']['max_drawdown_R']:.3f} |

## 当前种子专家冻结评级

| 指标 | 结果 |
|---|---:|
| 专家等级 | {seed_tier} |
| 执行模式 | {GATE['tier_execution_mode'][seed_tier]} |
| 风险倍数 | {float(GATE['tier_risk_multiplier'][seed_tier]):.2f} |
| 冻结评级累计交易 | {seed_summary['trades']} |
| 冻结评级胜率 | {seed_summary['win_rate']:.2%} |
| Wilson下界 | {seed_summary['wilson_lower']:.2%} |
| 删除最佳交易后净R | {seed_summary['best_trade_removed_net_R']:.3f} |

## 数据完整性

| 指标 | 结果 |
|---|---:|
| 窗口类型 | {DIAGNOSTIC['window_type']} |
| 起始时间 | {DIAGNOSTIC['window_start_utc']} |
| 截止时间 | {DIAGNOSTIC['window_end_utc']} |
| 完整月份 | 2026-04、2026-05、2026-06 |
| 纳入自然日 | {len(_daily_window_dates())} |
| 最后一根原始K线 | {diagnostic_result['integrity']['monthly_data_audit']['last_raw_5m_bar_utc']} |
| 最后一根可用特征K线 | {diagnostic_result['integrity']['monthly_data_audit']['last_usable_feature_bar_utc']} |
| 月度归档主源 | 是 |
| 日度归档 | 仅修复Premium历史缺口 |
| REST备用接口 | 否 |

重点结果文件：`q2_risk_budget_trade_ledger.csv`、`q2_expert_trade_ledger.csv`、`q2_diagnostic_summary.json`、`q2_integrity_audit.json`。
"""
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    (RESULTS / "run_identity.txt").write_text(
        f"{ENGINE_NAME}\noutput=results_v9_7_2\nwindow=2026-04-01T00:00:00Z..2026-06-30T23:59:59Z\n"
        f"monthly_archives_primary=True\ndaily_archives=premium_gap_repair_only\nrest_fallback=False\n"
        f"snapshot_replay_exact={replay_audit['exact_match']}\nv961_reference_exact={exact_expected}\n"
        f"signal_selection_independent_of_tier={gate_independent}\nwinner_selection_enabled=False\n",
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
        "q2_progress": diagnostic_result["progress"],
        "no_monthly_budget_development_metrics": lanes["V9.6.7_NO_MONTHLY_BUDGET_ORIGINAL"]["development_metrics"]
    }, ensure_ascii=False, indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--self-test",action="store_true");parser.add_argument("--pipeline-smoke",action="store_true");args=parser.parse_args()
    if args.self_test:
        core.synthetic_smoke();statistical_self_test();attribution_self_test();shrinkage_allocation_self_test();original_gate_self_test();rating_source_self_test();premium_daily_gap_repair_self_test();q2_diagnostic_self_test()
    elif args.pipeline_smoke:
        core.pipeline_smoke();statistical_self_test();attribution_self_test();shrinkage_allocation_self_test();original_gate_self_test();rating_source_self_test();premium_daily_gap_repair_self_test();q2_diagnostic_self_test()
    else:main()
