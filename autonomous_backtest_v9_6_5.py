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
RESULTS = ROOT / "results_v9_6_5"
REQUEST_PATH = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(ROOT / "request.v9_6_5.json")))
REQUEST = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
STAT = REQUEST["statistical_validation"]
SNAP = REQUEST["snapshot_reproduction"]
WATCH_GATE = REQUEST["watch_gate"]
QUALIFIED_GATE = REQUEST["qualified_gate"]
CANDIDATE_GATE = REQUEST["candidate_gate"]
BLOCK = REQUEST["block_validation"]
DUAL = REQUEST["dual_track_validation"]
ENGINE_NAME = "BTC 5m signal attribution and dual-track validation V9.6.5"

spec = importlib.util.spec_from_file_location("v965_strategy_core", ROOT / "_v965_strategy_core.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load _v965_strategy_core.py")
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
    r=[msg for key,ok,msg in tests if key in gate and not ok]
    if "max_total_trades" in gate and s["trades"]>int(gate["max_total_trades"]):r.append("总交易不再稀疏")
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
    assert DUAL["winner_selection_enabled"] is False and DUAL["seen_oos_must_not_select_winner"] is True
    print("V965_SIGNAL_ATTRIBUTION_SELF_TEST_OK")



def statistical_self_test()->None:
    rows=[]
    for i,r in enumerate([1.7,-0.3,1.6,-0.4,1.5,-0.2,1.4,-0.2]):
        rows.append({"month":["2025-05","2025-06","2025-07","2025-08"][i//2],"net_r":r})
    s,monthly,blocks=evidence_for_expert(pd.DataFrame(rows),list(core.DEVELOPMENT_MONTHS),True)
    assert s["trades"]==8 and s["best_trade_removed_net_R"]>0
    assert monthly[0]["sample_state"]=="INSUFFICIENT_SAMPLE"
    one=pd.DataFrame([{"month":"2025-05","net_r":1.5}]);o,_,_=evidence_for_expert(one,list(core.DEVELOPMENT_MONTHS),True)
    assert o["shrunk_win_rate"]<0.60 and o["wilson_lower"]<0.30
    print("V965_STATISTICAL_VALIDATION_SELF_TEST_OK")


def main()->None:
    core.main()
    RESULTS.mkdir(exist_ok=True)
    shadow_path=RESULTS/"expert_shadow_trades.csv"
    shadow=pd.read_csv(shadow_path) if shadow_path.exists() and shadow_path.stat().st_size else pd.DataFrame()
    tiers=pd.read_csv(RESULTS/"expert_tier_audit.csv")
    dev_months=list(core.DEVELOPMENT_MONTHS)
    new_rows=[];monthly_rows=[];block_rows=[];bayes_rows=[];remove_rows=[];loo_rows=[];concentration_rows=[]
    summaries={}
    for _,row in tiers.iterrows():
        eid=int(row["expert_id"]);expert=str(row["expert"]);policy=str(row["policy_key"])
        q=shadow[(shadow.get("expert_id",pd.Series(dtype=int))==eid)&(shadow.get("policy_key",pd.Series(dtype=str)).astype(str)==policy)&(shadow.get("month",pd.Series(dtype=str)).astype(str).isin(dev_months))].copy() if not shadow.empty else pd.DataFrame()
        summary,months,blocks=evidence_for_expert(q,dev_months,bool(row.get("selected_cluster_robust",False)))
        tier,reasons=tier_for(summary);summaries[eid]=summary
        base=row.to_dict();base.update(summary);base["legacy_v963_tier"]=base.get("tier");base["tier"]=tier;base["rejection_reason"]="|".join(reasons);base["candidate_shadow_only"]=tier=="CANDIDATE";base["selected_research"]=tier in {"WATCH","QUALIFIED"};base["selected_qualified"]=tier=="QUALIFIED";new_rows.append(base)
        for m in months:monthly_rows.append({"expert_id":eid,"expert":expert,"policy_key":policy,"tier":tier,**m})
        for b in blocks:block_rows.append({"expert_id":eid,"expert":expert,"policy_key":policy,"tier":tier,**b})
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

    # Rebuild executable portfolio outputs under the stricter evidence tiers.
    allowed=set(new_tiers.loc[new_tiers["tier"].isin(["WATCH","QUALIFIED"]),"expert"].astype(str))
    qualified=set(new_tiers.loc[new_tiers["tier"]=="QUALIFIED","expert"].astype(str))
    for name,permit in [("watch_portfolio_trades.csv",allowed),("research_portfolio_trades.csv",allowed),("portfolio_trades.csv",allowed),("trades.csv",allowed),("qualified_portfolio_trades.csv",qualified)]:
        p=RESULTS/name
        try:df=pd.read_csv(p)
        except Exception:df=empty_trade_frame_like(p)
        if not df.empty and "expert" in df.columns:df=df[df["expert"].astype(str).isin(permit)]
        elif not permit:df=empty_trade_frame_like(p)
        df.to_csv(p,index=False)

    # Blind evidence is time-and-trade based, not one-month win-rate based.
    blind=[]
    for _,row in new_tiers.iterrows():
        eid=int(row["expert_id"]);policy=str(row["policy_key"]);expert=str(row["expert"])
        q=shadow[(shadow.get("expert_id",pd.Series(dtype=int))==eid)&(shadow.get("policy_key",pd.Series(dtype=str)).astype(str)==policy)&(shadow.get("month",pd.Series(dtype=str)).astype(str)==core.OOS_MONTH)].copy() if not shadow.empty else pd.DataFrame()
        bm=base_metrics(q);months_elapsed=1
        blind.append({"expert_id":eid,"expert":expert,"policy_key":policy,"development_tier":row["tier"],"blind_months_elapsed":months_elapsed,"blind_trades":bm["trades"],"blind_wins":bm["wins"],"blind_raw_win_rate":bm["win_rate"],"watch_evidence_complete":months_elapsed>=int(STAT["blind_min_calendar_months_watch"]) and bm["trades"]>=int(STAT["blind_min_trades_watch"]),"qualified_evidence_complete":months_elapsed>=int(STAT["blind_min_calendar_months_qualified"]) and bm["trades"]>=int(STAT["blind_min_trades_qualified"])})
    blind_payload={"diagnostic_oos_months":[core.OOS_MONTH],"rule":"minimum calendar months AND minimum trades must both be met","experts":blind}
    (RESULTS/"blind_evidence_progress.json").write_text(json.dumps(blind_payload,ensure_ascii=False,indent=2),encoding="utf-8")

    # Build deterministic full snapshot and replay the seed policy.
    seed_id=int(SNAP["seed_expert_id"]);seed_row=new_tiers[new_tiers["expert_id"]==seed_id].iloc[0];seed_key=str(seed_row["policy_key"])
    expert=core.EXPERT_BY_ID[seed_id];policies=core.policy_grid(expert);policy=next(p for p in policies if p.key==seed_key)
    raw,audit=core.base.load_official_data();eth,ea=core.base.load_auxiliary_kline("ETHUSDT","klines");premium,pa=core.base.load_auxiliary_kline(core.SYMBOL,"premiumIndexKlines");funding,fa=core.base.load_funding_rate()
    x,align=core.base.add_features(raw,eth,premium,funding);x=core.add_sparse_masks(x)
    replay=core.evaluate_candidate(x,expert,policy)
    replay_events=[dict(t,month=m) for m in core.DEVELOPMENT_MONTHS for t in replay.monthly_events.get(m,[])]
    original=shadow[(shadow.get("expert_id",pd.Series(dtype=int))==seed_id)&(shadow.get("policy_key",pd.Series(dtype=str)).astype(str)==seed_key)&(shadow.get("month",pd.Series(dtype=str)).astype(str).isin(dev_months))].copy() if not shadow.empty else pd.DataFrame()
    original_fp=event_fingerprint(original);replay_fp=event_fingerprint(replay_events)
    data_cols=[c for c in ["open","high","low","close","volume"] if c in raw.columns]
    data_fp=hashlib.sha256(pd.util.hash_pandas_object(raw[data_cols],index=True).values.tobytes()).hexdigest()
    snapshot={"engine":ENGINE_NAME,"engine_version":"V9.6.5","seed_expert_id":seed_id,"seed_expert":expert.name,"policy_key":seed_key,"policy":asdict(policy),"feature_group":expert.feature_group,"feature_list":list(core.FEATURE_GROUPS[expert.feature_group]),"training_months":list(core.DEVELOPMENT_MONTHS),"diagnostic_oos_month":core.OOS_MONTH,"execution":{"fee_rate_per_side":core.FEE_RATE,"slippage_abs":core.SLIPPAGE_ABS,"next_bar_open":True,"same_candle_stop_before_tp":True},"model_runtime":REQUEST["model"],"statistical_validation":STAT,"code_hashes":{"entrypoint":file_sha256(ROOT/"autonomous_backtest_v9_6_5.py"),"strategy_core":file_sha256(ROOT/"_v965_strategy_core.py"),"base_engine":file_sha256(ROOT/"_v965_base_engine.py"),"request":file_sha256(REQUEST_PATH)},"official_data_fingerprint":data_fp,"signal_fingerprint":original_fp,"random_seed":REQUEST["model"]["base_seed"]}
    (RESULTS/"strategy_snapshot.json").write_text(json.dumps(snapshot,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    replay_audit={"policy_key":seed_key,"original_signal_fingerprint":original_fp,"replay_signal_fingerprint":replay_fp,"exact_match":original_fp==replay_fp,"original_metrics":base_metrics(original),"replay_metrics":core.metrics(replay_events)}
    (RESULTS/"snapshot_replay_audit.json").write_text(json.dumps(replay_audit,ensure_ascii=False,indent=2),encoding="utf-8")

    # Exact V9.6.1 reference lane and per-signal diff.
    ref_request=ROOT/str(SNAP["reference_request_file"]);ref_request.write_text(json.dumps(REQUEST["legacy_v961_reference_request"],ensure_ascii=False,indent=2),encoding="utf-8")
    ref=import_reference(ref_request)
    rx,ralign=ref.base.add_features(raw,eth,premium,funding);rx=ref.add_sparse_masks(rx)
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
        a=old.get(k);b=new.get(k);status="BOTH" if a and b else ("V961_ONLY" if a else "V965_ONLY")
        diff.append({"signal_i":k[0],"direction":k[1],"status":status,"v961_month":a.get("month") if a else None,"v964_month":b.get("month") if b else None,"v961_net_r":a.get("net_r") if a else None,"v964_net_r":b.get("net_r") if b else None,"v961_meta":a.get("meta_decision") if a else None,"v964_meta":b.get("meta_decision") if b else None,"v961_probability":a.get("probability") if a else None,"v964_probability":b.get("probability") if b else None,"v961_utility":a.get("utility") if a else None,"v964_utility":b.get("utility") if b else None})
    pd.DataFrame(diff).to_csv(RESULTS/"v961_vs_v964_signal_diff.csv",index=False)

    # Full per-candidate stage attribution for both frozen lanes.
    trace_months = list(core.DEVELOPMENT_MONTHS) + [core.OOS_MONTH]
    current_trace, current_selected = trace_policy_lane(core, x, expert, policy, "V9.6.4_FROZEN", trace_months)
    reference_trace, reference_selected = trace_policy_lane(ref, rx, rexp, rpolicy, "V9.6.1_REFERENCE", trace_months)
    trace_all = pd.concat([reference_trace, current_trace], ignore_index=True, sort=False)
    trace_all.to_csv(RESULTS / "signal_stage_attribution.csv", index=False)

    current_dev_selected = current_selected[current_selected["month"].astype(str).isin(dev_months)].copy() if not current_selected.empty else current_selected
    reference_dev_selected = reference_selected[reference_selected["month"].astype(str).isin(dev_months)].copy() if not reference_selected.empty else reference_selected
    current_trace_exact = event_fingerprint(current_dev_selected) == original_fp
    reference_trace_exact = event_fingerprint(reference_dev_selected) == event_fingerprint(ref_events)
    if not current_trace_exact or not reference_trace_exact:
        raise RuntimeError(f"Stage tracer mismatch: current={current_trace_exact} reference={reference_trace_exact}")

    # Candidate-by-candidate comparison, including profitable opportunities rejected by either lane.
    key_cols = ["month", "signal_i", "direction"]
    def lane_view(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=key_cols)
        keep = key_cols + [
            "exit_i", "day", "net_r", "win", "probability", "online_percentile",
            "rank_pass", "router", "micro", "meta_probability", "meta_decision",
            "meta_pass", "expected_utility", "brier_penalty", "sample_uncertainty_penalty",
            "total_penalty", "utility", "utility_pass", "cap_pass", "selected",
            "first_reject_stage"
        ]
        q = df[[c for c in keep if c in df.columns]].copy()
        return q.rename(columns={c: f"{prefix}_{c}" for c in q.columns if c not in key_cols})
    cmp = lane_view(reference_trace, "v961").merge(lane_view(current_trace, "v964"), on=key_cols, how="outer")
    cmp["candidate_presence"] = np.select(
        [cmp.get("v961_exit_i").notna() & cmp.get("v964_exit_i").notna(), cmp.get("v961_exit_i").notna()],
        ["BOTH_RAW", "V961_RAW_ONLY"], default="V965_RAW_ONLY"
    )
    cmp["selection_status"] = np.select(
        [cmp.get("v961_selected", False).fillna(False) & cmp.get("v964_selected", False).fillna(False),
         cmp.get("v961_selected", False).fillna(False),
         cmp.get("v964_selected", False).fillna(False)],
        ["BOTH_SELECTED", "V961_SELECTED_ONLY", "V965_SELECTED_ONLY"], default="NEITHER_SELECTED"
    )
    cmp["selection_changed"] = cmp["selection_status"].isin(["V961_SELECTED_ONLY", "V965_SELECTED_ONLY"])
    cmp["outcome_net_r"] = cmp.get("v961_net_r").combine_first(cmp.get("v964_net_r"))
    cmp["outcome_win"] = cmp.get("v961_win").combine_first(cmp.get("v964_win"))
    cmp["profit_opportunity_rejected_by_v961"] = (~cmp.get("v961_selected", False).fillna(False)) & (cmp["outcome_net_r"] > 0)
    cmp["profit_opportunity_rejected_by_v964"] = (~cmp.get("v964_selected", False).fillna(False)) & (cmp["outcome_net_r"] > 0)
    cmp.to_csv(RESULTS / "v961_vs_v964_candidate_diff.csv", index=False)

    focus = cmp[cmp["month"].astype(str) == str(DUAL["focus_month"])].copy()
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

    selected_both = pd.concat([reference_selected, current_selected], ignore_index=True, sort=False)
    selected_both[selected_both["month"].astype(str).isin(dev_months)].to_csv(RESULTS / "dual_track_development_trades.csv", index=False)
    selected_both[selected_both["month"].astype(str) == core.OOS_MONTH].to_csv(RESULTS / "dual_track_oos_shadow.csv", index=False)

    block_rows_dual = block_quality_rows(reference_dev_selected, "V9.6.1_REFERENCE", dev_months)
    block_rows_dual += block_quality_rows(current_dev_selected, "V9.6.4_FROZEN", dev_months)
    pd.DataFrame(block_rows_dual).to_csv(RESULTS / "block_quality_validation.csv", index=False)

    reference_oos = reference_selected[reference_selected["month"].astype(str) == core.OOS_MONTH].copy() if not reference_selected.empty else reference_selected
    current_oos = current_selected[current_selected["month"].astype(str) == core.OOS_MONTH].copy() if not current_selected.empty else current_selected
    dual_payload = {
        "winner_selection_enabled": False,
        "winner_decision": "NO_WINNER_SELECTION_FROM_SEEN_OOS",
        "reason": "2026-06 has already been inspected and cannot select or tune the winning lane",
        "lanes": {
            "V9.6.1_REFERENCE": {
                "development_metrics": base_metrics(reference_dev_selected),
                "diagnostic_oos_metrics": base_metrics(reference_oos),
                "development_signal_fingerprint": event_fingerprint(reference_dev_selected),
                "stage_trace_exact": reference_trace_exact
            },
            "V9.6.4_FROZEN": {
                "development_metrics": base_metrics(current_dev_selected),
                "diagnostic_oos_metrics": base_metrics(current_oos),
                "development_signal_fingerprint": event_fingerprint(current_dev_selected),
                "stage_trace_exact": current_trace_exact
            }
        },
        "future_decision_rule": "Keep both lanes frozen until a new blind window meets minimum calendar-month and trade-count evidence",
        "focus_month": DUAL["focus_month"]
    }
    (RESULTS / "dual_track_validation.json").write_text(json.dumps(dual_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rule_delta = {
        "V9.6.1_REFERENCE": {
            "base_seed": int(ref.MODEL["base_seed"]),
            "sample_uncertainty_penalty": float(ref.MODEL.get("sample_uncertainty_penalty", 0.0)),
            "utility_uncertainty_penalty": float(ref.MODEL["utility_uncertainty_penalty"]),
            "minimum_router_confidence": float(ref.MODEL["minimum_router_confidence"]),
            "hard_negative_utility_r": float(ref.MODEL["hard_negative_utility_r"]),
            "policy": asdict(rpolicy)
        },
        "V9.6.4_FROZEN": {
            "base_seed": int(core.MODEL["base_seed"]),
            "sample_uncertainty_penalty": float(core.MODEL.get("sample_uncertainty_penalty", 0.0)),
            "utility_uncertainty_penalty": float(core.MODEL["utility_uncertainty_penalty"]),
            "minimum_router_confidence": float(core.MODEL["minimum_router_confidence"]),
            "hard_negative_utility_r": float(core.MODEL["hard_negative_utility_r"]),
            "policy": asdict(policy)
        },
        "known_rule_delta": {
            "sample_uncertainty_penalty_added": float(core.MODEL.get("sample_uncertainty_penalty", 0.0)),
            "winner_selection_from_seen_oos": False
        }
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
    status.update({"engine":ENGINE_NAME,"qualified":False,"not_for_live_trading":True,"tier_counts":counts,"statistical_validation_enabled":True,"monthly_raw_win_rate_is_not_validation":True,"selected_research_expert_count":counts["WATCH"]+counts["QUALIFIED"],"selected_qualified_expert_count":counts["QUALIFIED"],"selected_experts":{"research_watch_and_qualified_only":research_selected_payload,"qualified":qualified_selected_payload},"watch_portfolio_monthly_stats":research_stats,"research_portfolio_monthly_stats":research_stats,"qualified_portfolio_monthly_stats":qualified_stats,"snapshot_replay_exact":replay_audit["exact_match"],"v961_reference_exact_summary_match":exact_expected,"signal_stage_attribution_enabled":True,"dual_track_validation_enabled":True,"winner_selection_enabled":False,"winner_decision":"NO_WINNER_SELECTION_FROM_SEEN_OOS","current_stage_trace_exact":current_trace_exact,"reference_stage_trace_exact":reference_trace_exact,"blind_evidence_complete":False,"constraints":{**status.get("constraints",{}),"watch_gate":WATCH_GATE,"qualified_gate":QUALIFIED_GATE,"statistical_validation":STAT,"block_validation":BLOCK,"dual_track_validation":DUAL}})
    status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    seed_summary=summaries[seed_id]
    report=f"""# BTCUSDT 5分钟 信号筛选差异归因与双轨验证 V9.6.5 报告

- 架构：冻结V9.6.1参考通道 + 冻结V9.6.5当前通道 → 全部原始候选逐层追踪 → 在线排名/Meta/效用/样本惩罚/持仓与频率上限归因 → 开发期双轨对照 → 已查看OOS只做影子诊断。
- 专家等级：正式 {counts['QUALIFIED']}；观察 {counts['WATCH']}；候选 {counts['CANDIDATE']}；淘汰 {counts['REJECTED']}。
- 当前策略快照重放：**{'通过' if replay_audit['exact_match'] else '失败'}**。
- V9.6.1历史参考通道复现：**{'通过' if exact_expected else '失败'}**。
- 两条通道的逐阶段追踪与最终交易一致：当前 `{current_trace_exact}`；参考 `{reference_trace_exact}`。
- 胜出通道选择：**关闭**。2026年6月已经被查看，只能诊断，不能用于选通道或调参数。
- 实盘资格：**不合格**。

## 当前种子专家统计证据

| 指标 | 结果 |
|---|---:|
| 累计交易 | {seed_summary['trades']} |
| 原始胜率 | {seed_summary['win_rate']:.2%} |
| 贝叶斯收缩胜率 | {seed_summary['shrunk_win_rate']:.2%} |
| 95% Wilson下界 | {seed_summary['wilson_lower']:.2%} |
| 有效三个月块 | {seed_summary['valid_blocks']} |
| 正期望验证块 | {seed_summary['positive_expectancy_blocks']} |
| 高胜率验证块 | {seed_summary['high_win_rate_blocks']} |
| 删除最佳交易后净R | {seed_summary['best_trade_removed_net_R']:.3f} |
| 最大单笔利润占比 | {seed_summary['max_single_trade_profit_share']:.2%} |
| 最大单月利润占比 | {seed_summary['max_single_month_profit_share']:.2%} |

## 双轨开发期对照

| 通道 | 交易 | 胜率 | 盈亏比 | 净R | 最大回撤R |
|---|---:|---:|---:|---:|---:|
| V9.6.1参考 | {dual_payload['lanes']['V9.6.1_REFERENCE']['development_metrics']['trades']} | {dual_payload['lanes']['V9.6.1_REFERENCE']['development_metrics']['win_rate']:.2%} | {dual_payload['lanes']['V9.6.1_REFERENCE']['development_metrics']['avg_win_loss_ratio']:.3f} | {dual_payload['lanes']['V9.6.1_REFERENCE']['development_metrics']['net_R']:.3f} | {dual_payload['lanes']['V9.6.1_REFERENCE']['development_metrics']['max_drawdown_R']:.3f} |
| V9.6.4冻结 | {dual_payload['lanes']['V9.6.4_FROZEN']['development_metrics']['trades']} | {dual_payload['lanes']['V9.6.4_FROZEN']['development_metrics']['win_rate']:.2%} | {dual_payload['lanes']['V9.6.4_FROZEN']['development_metrics']['avg_win_loss_ratio']:.3f} | {dual_payload['lanes']['V9.6.4_FROZEN']['development_metrics']['net_R']:.3f} | {dual_payload['lanes']['V9.6.4_FROZEN']['development_metrics']['max_drawdown_R']:.3f} |

`signal_stage_attribution.csv` 与 `january_signal_case_study.csv` 会明确显示每个机会在哪一层被拒绝。三个月块同时区分“正期望通过”和“高胜率通过”，两者不再混为一谈。
"""
    (RESULTS/"report.md").write_text(report,encoding="utf-8")
    (RESULTS/"run_identity.txt").write_text(f"{ENGINE_NAME}\noutput=results_v9_6_5\noos={core.OOS_MONTH}\nsnapshot_replay_exact={replay_audit['exact_match']}\nv961_reference_exact={exact_expected}\ncurrent_stage_trace_exact={current_trace_exact}\nreference_stage_trace_exact={reference_trace_exact}\nwinner_selection_enabled=False\n",encoding="utf-8")
    print(json.dumps({"tier_counts":counts,"snapshot_replay_exact":replay_audit["exact_match"],"v961_reference_exact":exact_expected,"seed_evidence":seed_summary},ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--self-test",action="store_true");parser.add_argument("--pipeline-smoke",action="store_true");args=parser.parse_args()
    if args.self_test:
        core.synthetic_smoke();statistical_self_test();attribution_self_test()
    elif args.pipeline_smoke:
        core.pipeline_smoke();statistical_self_test();attribution_self_test()
    else:main()
