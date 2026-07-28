from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import _v103_engine as engine

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v10_3"
DEFAULT_REQUEST = ROOT / "request.v10_3.json"


def load_request() -> dict[str, Any]:
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))
    if not path.exists():
        raise FileNotFoundError(f"Missing request file: {path}")
    request = json.loads(path.read_text(encoding="utf-8"))
    validate_request(request)
    return request


def validate_request(request: dict[str, Any]) -> None:
    release = request["release"]
    if release["version"] != "10.3":
        raise ValueError("V10.3 requires release.version=10.3")
    if request["symbol"].upper() != "BTCUSDT":
        raise ValueError("V10.3 is frozen to BTCUSDT")
    if request["source_interval"] != "5m" or request["signal_timeframe_minutes"] != 60:
        raise ValueError("V10.3 must aggregate one-hour signals from official 5m data")
    if request["channel_count"] != 9:
        raise ValueError("V10.3 requires exactly 9 channels")
    if request["baseline_channel_count"] != 3:
        raise ValueError("V10.3 requires exactly 3 frozen baselines")
    if request["directional_channel_count"] != 6 or request["portfolio_channel_count"] != 3:
        raise ValueError("V10.3 requires 6 directional and 3 portfolio channels")
    if not release["long_short_parameters_are_independent"]:
        raise ValueError("V10.3 must keep long and short parameters independent")
    if not release["portfolio_uses_shared_position"]:
        raise ValueError("V10.3 portfolio must use a shared position")
    if not request["historical_diagnostic_only"] or request["winner_selection_enabled"]:
        raise ValueError("Viewed history must remain diagnostic with automatic winner selection disabled")
    execution = request["execution"]
    if float(execution["fee_rate_per_side"]) != 0.0005:
        raise ValueError("fee_rate_per_side must remain 0.0005")
    if float(execution["tick_size"]) * int(execution["slippage_ticks_per_fill"]) != 0.2:
        raise ValueError("slippage must remain 0.2 USDT per fill")
    if execution["same_bar_stop_target_rule"] != "STOP_FIRST_CONSERVATIVE":
        raise ValueError("same-bar ambiguity must remain stop-first")
    if execution["entry_rule"] != "NEXT_5M_OPEN_AFTER_CLOSED_1H_SIGNAL_BAR":
        raise ValueError("entry must remain next 5m open after closed 1h signal")
    expected = [spec.channel_id for spec in engine.CHANNELS]
    actual = [row["channel_id"] for row in request["channels"]]
    if actual != expected:
        raise ValueError(f"Channel list mismatch: expected={expected}, actual={actual}")
    for row, spec in zip(request["channels"], engine.CHANNELS):
        if row["direction_scope"] != spec.direction_scope or row["profile"] != spec.profile:
            raise ValueError(f"Direction/profile mismatch for {spec.channel_id}")
    if sum(bool(row.get("is_baseline")) for row in request["channels"]) != 3:
        raise ValueError("Exactly three channels must be baselines")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def clear_results() -> None:
    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_report(
    request: dict[str, Any],
    summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    components: pd.DataFrame,
    leader: dict[str, Any] | None,
) -> None:
    lines = [
        "# BTCUSDT V10.3 多空策略完全分离回测报告",
        "",
        f"- 数据窗口：**{request['start_month']}至{request['end_month']} UTC**。",
        "- 只使用1小时信号，但多头与空头不再共用ADX、DI、EMA斜率、RSI、止损、目标或持仓时间。",
        "- 多头使用质量回踩/趋势恢复逻辑；空头使用趋势持续性/下跌延续逻辑。",
        "- 三条组合通道只合并信号，不改写组件参数，并使用同一个共享仓位，禁止多空同时持仓。",
        "- 同一时间若出现相反方向冲突，保守地同时丢弃；所有信号在1小时K线收盘后下一根5分钟开盘执行。",
        "- 本历史数据已被查看，本报告只用于方向诊断，不是盲测，不允许自动宣布实盘合格。",
        "",
        "## 总体排名",
        "",
        "| 排名 | 通道 | 方向 | 交易 | 胜率 | 实际盈亏比 | PF | 净R | 回撤R | Q2净R | 删最佳10%后 | 最差滚动季度 | 候选 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for index, row in summary.iterrows():
        lines.append(
            f"| {index + 1} | {row['channel_label']} | {row['direction_scope']} | {int(row['trades'])} | "
            f"{row['win_rate']:.2%} | {row['avg_win_loss_ratio']:.3f} | {row['profit_factor']:.3f} | "
            f"{row['net_R']:.3f} | {row['max_drawdown_R']:.3f} | {row['historical_test_net_R']:.3f} | "
            f"{row['best_10pct_removed_net_R']:.3f} | {row['worst_rolling_quarter_net_R']:.3f} | "
            f"{'是' if bool(row['research_candidate']) else '否'} |"
        )

    lines.extend([
        "",
        "## 与各自冻结基线对照",
        "",
        "| 通道 | 对照基线 | 信号保留率 | 胜率变化 | PF变化 | Q2净R变化 | 删最佳10%变化 | 回撤变化 | 三项改善 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in comparison.loc[~comparison["is_baseline"]].itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {row.baseline_channel_id} | {row.retention_rate_vs_baseline:.2%} | "
            f"{row.delta_win_rate:+.2%} | {row.delta_profit_factor:+.3f} | "
            f"{row.delta_historical_test_net_R:+.3f} | {row.delta_best_10pct_removed_net_R:+.3f} | "
            f"{row.delta_max_drawdown_R:+.3f} | {'是' if bool(row.improves_primary_goals) else '否'} |"
        )

    portfolio_components = components.loc[components["channel_id"].str.contains("split_portfolio", na=False)] if not components.empty else components
    lines.extend([
        "",
        "## 组合内部多空贡献",
        "",
        "| 组合 | 来源组件 | 交易 | 胜率 | PF | 净R | 回撤R |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in portfolio_components.itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {row.source_component_label} | {int(row.trades)} | {row.win_rate:.2%} | "
            f"{row.profit_factor:.3f} | {row.net_R:.3f} | {row.max_drawdown_R:.3f} |"
        )

    lines.extend([
        "",
        "## 三阶段表现",
        "",
        "| 通道 | 阶段 | 交易 | 胜率 | PF | 净R | 回撤R | 期望R |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in phase_summary.itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {row.phase} | {int(row.trades)} | {row.win_rate:.2%} | "
            f"{row.profit_factor:.3f} | {row.net_R:.3f} | {row.max_drawdown_R:.3f} | {row.expectancy_R:.3f} |"
        )

    lines.extend(["", "## 当前结论", ""])
    if leader is None:
        lines.append("没有非基线通道生成有效结果。")
    else:
        lines.append(
            f"- 当前非基线历史排名第一：**{leader['channel_label']}**，交易{int(leader['trades'])}笔，"
            f"胜率{float(leader['win_rate']):.2%}，PF {float(leader['profit_factor']):.3f}，"
            f"净R {float(leader['net_R']):.3f}，Q2净R {float(leader['historical_test_net_R']):.3f}。"
        )
    lines.extend([
        "- 多头、空头与组合分别使用自己的最低样本和稳健性门槛；某个方向合格不代表另一方向自动合格。",
        "- 组合必须证明两个组件都能贡献，而不是由单一方向垄断全部利润。",
        "- 即使出现历史候选，仍必须冻结参数并使用未来未见数据验证。",
        "",
        "重点文件：`channel_summary.csv`、`baseline_comparison.csv`、`direction_summary.csv`、"
        "`portfolio_component_summary.csv`、`portfolio_overlap_audit.csv`、`rolling_quarter_summary.csv`、"
        "`trade_ledger.csv`、`robustness_audit.json`。",
    ])
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pipeline_smoke(request: dict[str, Any]) -> None:
    synthetic = engine.synthetic_5m_data(100_000, seed=20261003)
    result = engine.run_benchmark(synthetic, request)
    if len(result["summary"]) != 9:
        raise AssertionError("Pipeline smoke did not return 9 channels")
    if len(result["baseline_comparison"]) != 9:
        raise AssertionError("Pipeline smoke did not create complete baseline comparison")
    if len(result["phases"]) != 27:
        raise AssertionError("Pipeline smoke did not create 3 phases per channel")
    if len(result["signal_audit"]["portfolio_overlap"]) != 3:
        raise AssertionError("Pipeline smoke did not create portfolio overlap audit")
    if result["trades"].empty:
        raise AssertionError("Pipeline smoke produced no trades")
    print(json.dumps({
        "channels": int(len(result["summary"])),
        "trades": int(len(result["trades"])),
        "leader": result["research_leader"]["channel_id"] if result["research_leader"] else None,
    }, ensure_ascii=False, indent=2))
    print("V103_PIPELINE_SMOKE_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT V10.3 direction-decoupled long/short hourly strategies")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pipeline-smoke", action="store_true")
    args = parser.parse_args()
    request = load_request()

    if args.self_test:
        engine.self_test(request)
        print("V103_SELF_TEST_OK")
        return
    if args.pipeline_smoke:
        pipeline_smoke(request)
        return

    clear_results()
    raw, data_audit = engine.load_official_5m_data(request)
    benchmark = engine.run_benchmark(raw, request)
    summary = benchmark["summary"]
    phases = benchmark["phases"]
    monthly = benchmark["monthly"]
    directions = benchmark["directions"]
    components = benchmark["components"]
    rolling = benchmark["rolling_quarters"]
    comparison = benchmark["baseline_comparison"]
    ledger = benchmark["trades"]
    leader = benchmark["research_leader"]

    summary.to_csv(RESULTS / "channel_summary.csv", index=False)
    comparison.to_csv(RESULTS / "baseline_comparison.csv", index=False)
    phases.to_csv(RESULTS / "channel_phase_summary.csv", index=False)
    monthly.to_csv(RESULTS / "channel_monthly_summary.csv", index=False)
    directions.to_csv(RESULTS / "direction_summary.csv", index=False)
    components.to_csv(RESULTS / "portfolio_component_summary.csv", index=False)
    rolling.to_csv(RESULTS / "rolling_quarter_summary.csv", index=False)
    ledger.to_csv(RESULTS / "trade_ledger.csv", index=False)
    pd.DataFrame(benchmark["signal_audit"]["channels"]).to_csv(RESULTS / "signal_counts.csv", index=False)
    pd.DataFrame(benchmark["signal_audit"]["portfolio_overlap"]).to_csv(RESULTS / "portfolio_overlap_audit.csv", index=False)
    historical = ledger.loc[ledger["phase"] == "historical_test"].copy() if not ledger.empty else ledger.copy()
    historical.to_csv(RESULTS / "historical_test_trade_diagnostics.csv", index=False)

    qualification = benchmark["qualification"]
    candidate_channels = [channel_id for channel_id, audit in qualification.items() if audit["research_candidate"]]
    status = {
        "engine": "BTCUSDT hourly direction-decoupled long/short strategies V10.3",
        "release_version": "10.3",
        "historical_diagnostic_only": True,
        "qualified_for_live_trading": False,
        "signal_timeframe_minutes": 60,
        "long_short_parameters_are_independent": True,
        "portfolio_uses_shared_position": True,
        "channel_count": 9,
        "directional_channel_count": 6,
        "portfolio_channel_count": 3,
        "research_candidate_count": len(candidate_channels),
        "research_candidate_channels": candidate_channels,
        "research_leader": leader,
        "winner_selection_enabled": False,
        "next_step": "Compare long, short and shared-position portfolios independently; freeze only components that pass their own robustness thresholds.",
    }
    benchmark_summary = {
        "data_window": {"start_month": request["start_month"], "end_month": request["end_month"]},
        "channels": summary.to_dict(orient="records"),
        "baseline_comparison": comparison.to_dict(orient="records"),
        "research_leader": leader,
        "research_candidate_channels": candidate_channels,
        "live_qualification": False,
        "winner_decision": "HISTORICAL_DIRECTION_DECOUPLING_ONLY_NO_AUTOMATIC_WINNER",
    }
    identity = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "request_sha256": file_sha256(Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))),
        "base_engine_sha256": file_sha256(ROOT / "_v102_engine.py"),
        "engine_sha256": file_sha256(ROOT / "_v103_engine.py"),
        "runner_sha256": file_sha256(ROOT / "autonomous_backtest_v10_3.py"),
    }
    json_dump(RESULTS / "status.json", status)
    json_dump(RESULTS / "benchmark_summary.json", benchmark_summary)
    json_dump(RESULTS / "data_audit.json", data_audit)
    json_dump(RESULTS / "signal_and_portfolio_audit.json", benchmark["signal_audit"])
    json_dump(RESULTS / "robustness_audit.json", qualification)
    json_dump(RESULTS / "strategy_spec.json", {
        "channels": request["channels"],
        "direction_specific_parameters": request["direction_specific_parameters"],
        "portfolio_rules": request["portfolio_rules"],
        "execution": request["execution"],
        "no_lookahead_rules": request["no_lookahead_rules"],
        "research_candidate_thresholds": request["research_candidate_thresholds"],
    })
    json_dump(RESULTS / "run_identity.json", identity)
    (RESULTS / "run_identity.txt").write_text("\n".join(f"{key}={value}" for key, value in identity.items()) + "\n", encoding="utf-8")
    write_report(request, summary, phases, comparison, components, leader)
    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    print("V103_REAL_BACKTEST_COMPLETE")


if __name__ == "__main__":
    main()
