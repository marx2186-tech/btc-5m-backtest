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
RESULTS = ROOT / "results_v9_6_4"
REQUEST_PATH = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(ROOT / "request.v9_6_4.json")))
REQUEST = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
STAT = REQUEST["statistical_validation"]
SNAP = REQUEST["snapshot_reproduction"]
WATCH_GATE = REQUEST["watch_gate"]
QUALIFIED_GATE = REQUEST["qualified_gate"]
CANDIDATE_GATE = REQUEST["candidate_gate"]
ENGINE_NAME = "BTC 5m full snapshot reproduction and sparse evidence validation V9.6.4"

spec = importlib.util.spec_from_file_location("v964_strategy_core", ROOT / "_v964_strategy_core.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load _v964_strategy_core.py")
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
    valid_blocks=positive_blocks=0
    for bi,months in enumerate(month_groups(development_months,int(STAT["block_months"])),1):
        complete=len(months)==int(STAT["block_months"])
        q=df[df["month"].isin(months)] if not df.empty else df
        bm=base_metrics(q)
        valid=complete and bm["trades"]>=int(STAT["block_min_trades"])
        state="VALID_BLOCK" if valid else ("INCOMPLETE_CALENDAR_BLOCK" if not complete else "INSUFFICIENT_BLOCK_SAMPLE")
        if valid:
            valid_blocks+=1; positive_blocks+=int(bm["net_R"]>0)
        blocks.append({"block_id":bi,"start_month":months[0],"end_month":months[-1],"months":"|".join(months),"calendar_complete":complete,"block_state":state,"block_valid":valid,**bm})
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
             "active_months":active_months,"positive_months":positive_months,"valid_blocks":valid_blocks,"positive_blocks":positive_blocks,
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


def statistical_self_test()->None:
    rows=[]
    for i,r in enumerate([1.7,-0.3,1.6,-0.4,1.5,-0.2,1.4,-0.2]):
        rows.append({"month":["2025-05","2025-06","2025-07","2025-08"][i//2],"net_r":r})
    s,monthly,blocks=evidence_for_expert(pd.DataFrame(rows),list(core.DEVELOPMENT_MONTHS),True)
    assert s["trades"]==8 and s["best_trade_removed_net_R"]>0
    assert monthly[0]["sample_state"]=="INSUFFICIENT_SAMPLE"
    one=pd.DataFrame([{"month":"2025-05","net_r":1.5}]);o,_,_=evidence_for_expert(one,list(core.DEVELOPMENT_MONTHS),True)
    assert o["shrunk_win_rate"]<0.60 and o["wilson_lower"]<0.30
    print("V964_STATISTICAL_VALIDATION_SELF_TEST_OK")


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
    snapshot={"engine":ENGINE_NAME,"engine_version":"V9.6.4","seed_expert_id":seed_id,"seed_expert":expert.name,"policy_key":seed_key,"policy":asdict(policy),"feature_group":expert.feature_group,"feature_list":list(core.FEATURE_GROUPS[expert.feature_group]),"training_months":list(core.DEVELOPMENT_MONTHS),"diagnostic_oos_month":core.OOS_MONTH,"execution":{"fee_rate_per_side":core.FEE_RATE,"slippage_abs":core.SLIPPAGE_ABS,"next_bar_open":True,"same_candle_stop_before_tp":True},"model_runtime":REQUEST["model"],"statistical_validation":STAT,"code_hashes":{"entrypoint":file_sha256(ROOT/"autonomous_backtest_v9_6_4.py"),"strategy_core":file_sha256(ROOT/"_v964_strategy_core.py"),"base_engine":file_sha256(ROOT/"_v964_base_engine.py"),"request":file_sha256(REQUEST_PATH)},"official_data_fingerprint":data_fp,"signal_fingerprint":original_fp,"random_seed":REQUEST["model"]["base_seed"]}
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
        a=old.get(k);b=new.get(k);status="BOTH" if a and b else ("V961_ONLY" if a else "V964_ONLY")
        diff.append({"signal_i":k[0],"direction":k[1],"status":status,"v961_month":a.get("month") if a else None,"v964_month":b.get("month") if b else None,"v961_net_r":a.get("net_r") if a else None,"v964_net_r":b.get("net_r") if b else None,"v961_meta":a.get("meta_decision") if a else None,"v964_meta":b.get("meta_decision") if b else None,"v961_probability":a.get("probability") if a else None,"v964_probability":b.get("probability") if b else None,"v961_utility":a.get("utility") if a else None,"v964_utility":b.get("utility") if b else None})
    pd.DataFrame(diff).to_csv(RESULTS/"v961_vs_v964_signal_diff.csv",index=False)

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
    status.update({"engine":ENGINE_NAME,"qualified":False,"not_for_live_trading":True,"tier_counts":counts,"statistical_validation_enabled":True,"monthly_raw_win_rate_is_not_validation":True,"selected_research_expert_count":counts["WATCH"]+counts["QUALIFIED"],"selected_qualified_expert_count":counts["QUALIFIED"],"selected_experts":{"research_watch_and_qualified_only":research_selected_payload,"qualified":qualified_selected_payload},"watch_portfolio_monthly_stats":research_stats,"research_portfolio_monthly_stats":research_stats,"qualified_portfolio_monthly_stats":qualified_stats,"snapshot_replay_exact":replay_audit["exact_match"],"v961_reference_exact_summary_match":exact_expected,"blind_evidence_complete":False,"constraints":{**status.get("constraints",{}),"watch_gate":WATCH_GATE,"qualified_gate":QUALIFIED_GATE,"statistical_validation":STAT}})
    status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    seed_summary=summaries[seed_id]
    report=f"""# BTCUSDT 5分钟 完整策略快照复现与稀疏专家统计验证 V9.6.4 报告

- 架构：V9.6.3核心搜索 → 完整策略快照 → 当前策略逐信号重放 → V9.6.1独立参考代码重放 → 月度样本有效性 → 不重叠三个月验证块 → 贝叶斯收缩与95% Wilson下界 → 删除最佳交易与逐月删除 → 盲测时间和交易量双门槛。
- 专家等级：正式 {counts['QUALIFIED']}；观察 {counts['WATCH']}；候选 {counts['CANDIDATE']}；淘汰 {counts['REJECTED']}。
- 单月1至2笔即使全部盈利，也只标记为 `INSUFFICIENT_SAMPLE`，不会被视作有效100%月胜率。
- 当前快照逐信号复现：**{'通过' if replay_audit['exact_match'] else '失败'}**。
- V9.6.1历史摘要独立复现：**{'通过' if exact_expected else '失败'}**。
- 2026年6月只有1个盲测自然月，不满足至少3个月且至少5/8笔的证据要求。
- 实盘资格：**不合格**。

## 种子专家统计证据

| 指标 | 结果 |
|---|---:|
| 累计交易 | {seed_summary['trades']} |
| 原始胜率 | {seed_summary['win_rate']:.2%} |
| 贝叶斯收缩胜率 | {seed_summary['shrunk_win_rate']:.2%} |
| 95% Wilson下界 | {seed_summary['wilson_lower']:.2%} |
| 有效三个月块 | {seed_summary['valid_blocks']} |
| 正收益验证块 | {seed_summary['positive_blocks']} |
| 删除最佳交易后净R | {seed_summary['best_trade_removed_net_R']:.3f} |
| 逐月删除正收益比例 | {seed_summary['loo_month_positive_share']:.2%} |
| 最大单笔利润占比 | {seed_summary['max_single_trade_profit_share']:.2%} |
| 最大单月利润占比 | {seed_summary['max_single_month_profit_share']:.2%} |

月度原始胜率只用于描述发生了什么；专家晋级依据是累计证据、三个月验证块、收缩胜率、可信下界、利润集中度和稳健性测试。
"""
    (RESULTS/"report.md").write_text(report,encoding="utf-8")
    (RESULTS/"run_identity.txt").write_text(f"{ENGINE_NAME}\noutput=results_v9_6_4\noos={core.OOS_MONTH}\nsnapshot_replay_exact={replay_audit['exact_match']}\nv961_reference_exact={exact_expected}\n",encoding="utf-8")
    print(json.dumps({"tier_counts":counts,"snapshot_replay_exact":replay_audit["exact_match"],"v961_reference_exact":exact_expected,"seed_evidence":seed_summary},ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--self-test",action="store_true");parser.add_argument("--pipeline-smoke",action="store_true");args=parser.parse_args()
    if args.self_test:
        core.synthetic_smoke();statistical_self_test()
    elif args.pipeline_smoke:
        core.pipeline_smoke();statistical_self_test()
    else:main()
