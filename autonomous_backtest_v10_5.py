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

import _v105_engine as engine

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v10_5"
DEFAULT_REQUEST = ROOT / "request.v10_5.json"


def load_request() -> dict[str, Any]:
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))
    if not path.exists():
        raise FileNotFoundError(f"Missing request file: {path}")
    request = json.loads(path.read_text(encoding="utf-8"))
    validate_request(request)
    return request


def validate_request(request: dict[str, Any]) -> None:
    release = request["release"]
    if release["version"] != "10.5":
        raise ValueError("V10.5 requires release.version=10.5")
    if request["symbol"].upper() != "BTCUSDT":
        raise ValueError("V10.5 is frozen to BTCUSDT")
    if request["source_interval"] != "5m" or request["signal_timeframe_minutes"] != 60:
        raise ValueError("V10.5 must aggregate one-hour signals from official 5m data")
    if request["channel_count"] != 11:
        raise ValueError("V10.5 requires exactly 11 channels")
    if request["baseline_channel_count"] != 3:
        raise ValueError("V10.5 requires exactly 3 baselines")
    if request["directional_channel_count"] != 5 or request["portfolio_channel_count"] != 6:
        raise ValueError("V10.5 requires 5 directional and 6 portfolio channels")
    if request["parameter_change_count"] != 3 or request["interaction_effect_count"] != 7:
        raise ValueError("V10.5 audit counts are fixed")
    if not release["long_short_parameters_are_independent"]:
        raise ValueError("Long and short parameters must remain independent")
    if not release["portfolio_uses_shared_position"]:
        raise ValueError("Portfolios must use one shared position")
    if not release["frozen_interaction_grid_only"] or not release["no_new_indicator_or_strategy_family"]:
        raise ValueError("V10.5 permits only the frozen interaction grid")
    if not request["historical_diagnostic_only"] or request["winner_selection_enabled"]:
        raise ValueError("Viewed history must remain diagnostic and winner selection disabled")

    execution = request["execution"]
    if float(execution["fee_rate_per_side"]) != 0.0005:
        raise ValueError("fee_rate_per_side must remain 0.0005")
    if float(execution["tick_size"]) * int(execution["slippage_ticks_per_fill"]) != 0.2:
        raise ValueError("slippage must remain 0.2 USDT per fill")
    if execution["same_bar_stop_target_rule"] != "STOP_FIRST_CONSERVATIVE":
        raise ValueError("same-bar ambiguity must remain stop-first")
    if execution["entry_rule"] != "NEXT_5M_OPEN_AFTER_CLOSED_1H_SIGNAL_BAR":
        raise ValueError("entry must remain the next 5m open after the closed 1h signal")

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
    parameter_changes: pd.DataFrame,
    interactions: pd.DataFrame,
    grid: pd.DataFrame,
    leader: dict[str, Any] | None,
) -> None:
    lines = [
        "# BTCUSDT V10.5 多空冻结交互优化回测报告",
        "",
        f"- 数据窗口：**{request['start_month']}至{request['end_month']} UTC**。",
        "- 冻结V10.4有效多头结构：价格延伸≤1.20 ATR、RR2.5，只测试ADX上限45、47.5、50。",
        "- 冻结V10.4有效空头结构：趋势持续性RR2.5，只比较ADX三根变化下限-3与-4。",
        "- 使用3×2多空因子网格，所有组合共享一个仓位；同一时间相反信号全部丢弃。",
        "- 手续费、滑点、止损优先、下一根5分钟开盘执行和无未来数据规则保持不变。",
        "- 历史数据已经查看，本报告只能用于诊断，不是盲测，也不能自动宣称实盘合格。",
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
        "## 3×2组合因子网格",
        "",
        "| 组合 | 多头ADX上限 | 空头模式 | 交易 | 胜率 | PF | 净R | Q2净R | 删最佳10%后 | 回撤R |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in grid.sort_values(["long_adx_cap", "short_mode"]).itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {row.long_adx_cap:.1f} | {row.short_mode} | {int(row.trades)} | "
            f"{row.win_rate:.2%} | {row.profit_factor:.3f} | {row.net_R:.3f} | "
            f"{row.historical_test_net_R:.3f} | {row.best_10pct_removed_net_R:.3f} | {row.max_drawdown_R:.3f} |"
        )

    lines.extend([
        "",
        "## 交互效应审计",
        "",
        "| 效应 | 因子 | 对照 | 变体 | 交易变化 | 胜率变化 | PF变化 | 净R变化 | Q2变化 | 删最佳10%变化 | 回撤变化 | 三项不劣 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in interactions.itertuples(index=False):
        lines.append(
            f"| {row.effect_id} | {row.factor} | {row.control_channel_label} | {row.variant_channel_label} | "
            f"{int(row.delta_trades):+d} | {row.delta_win_rate:+.2%} | {row.delta_profit_factor:+.3f} | "
            f"{row.delta_net_R:+.3f} | {row.delta_historical_test_net_R:+.3f} | "
            f"{row.delta_best_10pct_removed_net_R:+.3f} | {row.delta_max_drawdown_R:+.3f} | "
            f"{'是' if bool(row.interaction_improves_q2_tail_and_drawdown) else '否'} |"
        )

    lines.extend([
        "",
        "## 参数变更审计",
        "",
        "| 通道 | 对照基线 | 方向 | 变更数量 | 变更内容 | 类型 |",
        "|---|---|---|---:|---|---|",
    ])
    for row in parameter_changes.itertuples(index=False):
        label = engine.CHANNEL_BY_ID[row.channel_id].label
        lines.append(
            f"| {label} | {row.baseline_channel_id} | {row.direction} | {int(row.changed_condition_count)} | "
            f"{row.changed_condition} | {row.change_type} |"
        )

    lines.extend([
        "",
        "## 与冻结基线对照",
        "",
        "| 通道 | 对照基线 | 信号保留率 | 交易变化 | 胜率变化 | PF变化 | Q2净R变化 | 删最佳10%变化 | 回撤变化 | 三项不劣 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in comparison.loc[~comparison["is_baseline"]].itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {row.baseline_channel_id} | {row.retention_rate_vs_baseline:.2%} | "
            f"{int(row.delta_trades):+d} | {row.delta_win_rate:+.2%} | {row.delta_profit_factor:+.3f} | "
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
        lines.append("没有非基线组合生成有效结果。")
    else:
        lines.append(
            f"- 当前非基线组合历史排名第一：**{leader['channel_label']}**，交易{int(leader['trades'])}笔，"
            f"胜率{float(leader['win_rate']):.2%}，PF {float(leader['profit_factor']):.3f}，"
            f"净R {float(leader['net_R']):.3f}，Q2净R {float(leader['historical_test_net_R']):.3f}。"
        )
    lines.extend([
        "- V10.5只允许多头ADX上限与空头ADX衰减两项已确认变化形成固定网格，不搜索其他条件。",
        "- 组合必须达到样本、三个阶段、Q2、删除最佳交易和滚动季度门槛，才可冻结等待未来数据。",
        "- 即使历史候选出现，也不代表实盘合格；参数必须冻结并接受未来未见数据验证。",
        "",
        "重点文件：`channel_summary.csv`、`factor_grid.csv`、`interaction_effect_audit.csv`、",
        "`baseline_comparison.csv`、`portfolio_component_summary.csv`、`portfolio_overlap_audit.csv`、",
        "`rolling_quarter_summary.csv`、`trade_ledger.csv`、`robustness_audit.json`。",
    ])
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pipeline_smoke(request: dict[str, Any]) -> None:
    synthetic = engine.synthetic_5m_data(115_000, seed=20261005)
    result = engine.run_benchmark(synthetic, request)
    if len(result["summary"]) != 11:
        raise AssertionError("Pipeline smoke did not return 11 channels")
    if len(result["baseline_comparison"]) != 11:
        raise AssertionError("Pipeline smoke did not create complete comparisons")
    if len(result["phases"]) != 33:
        raise AssertionError("Pipeline smoke did not create 3 phases per channel")
    if len(result["signal_audit"]["portfolio_overlap"]) != 6:
        raise AssertionError("Pipeline smoke did not create 6 portfolio overlap rows")
    if len(result["parameter_change_audit"]) != 3:
        raise AssertionError("Pipeline smoke did not create parameter change audit")
    if len(result["interaction_effect_audit"]) != 7:
        raise AssertionError("Pipeline smoke did not create interaction audit")
    if len(result["factor_grid"]) != 6:
        raise AssertionError("Pipeline smoke did not create factor grid")
    if result["trades"].empty:
        raise AssertionError("Pipeline smoke produced no trades")
    print(json.dumps({
        "channels": int(len(result["summary"])),
        "trades": int(len(result["trades"])),
        "leader": result["research_leader"]["channel_id"] if result["research_leader"] else None,
    }, ensure_ascii=False, indent=2))
    print("V105_PIPELINE_SMOKE_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT V10.5 frozen long-short interaction grid")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pipeline-smoke", action="store_true")
    args = parser.parse_args()
    request = load_request()

    if args.self_test:
        engine.self_test(request)
        print("V105_SELF_TEST_OK")
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
    parameter_changes = benchmark["parameter_change_audit"]
    interactions = benchmark["interaction_effect_audit"]
    grid = benchmark["factor_grid"]
    ledger = benchmark["trades"]
    leader = benchmark["research_leader"]

    summary.to_csv(RESULTS / "channel_summary.csv", index=False)
    comparison.to_csv(RESULTS / "baseline_comparison.csv", index=False)
    parameter_changes.to_csv(RESULTS / "parameter_change_audit.csv", index=False)
    interactions.to_csv(RESULTS / "interaction_effect_audit.csv", index=False)
    grid.to_csv(RESULTS / "factor_grid.csv", index=False)
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
        "engine": "BTCUSDT hourly frozen long-short interaction grid V10.5",
        "release_version": "10.5",
        "historical_diagnostic_only": True,
        "qualified_for_live_trading": False,
        "signal_timeframe_minutes": 60,
        "long_short_parameters_are_independent": True,
        "portfolio_uses_shared_position": True,
        "frozen_interaction_grid_only": True,
        "channel_count": 11,
        "directional_channel_count": 5,
        "portfolio_channel_count": 6,
        "parameter_change_count": 3,
        "interaction_effect_count": 7,
        "research_candidate_count": len(candidate_channels),
        "research_candidate_channels": candidate_channels,
        "research_leader": leader,
        "winner_selection_enabled": False,
        "next_step": "Freeze only a grid cell that passes sample, three-phase, Q2, tail-removal and rolling-quarter checks; otherwise stop expanding the grid.",
    }
    benchmark_summary = {
        "data_window": {"start_month": request["start_month"], "end_month": request["end_month"]},
        "channels": summary.to_dict(orient="records"),
        "factor_grid": grid.to_dict(orient="records"),
        "interaction_effect_audit": interactions.to_dict(orient="records"),
        "baseline_comparison": comparison.to_dict(orient="records"),
        "parameter_change_audit": parameter_changes.to_dict(orient="records"),
        "research_leader": leader,
        "research_candidate_channels": candidate_channels,
        "live_qualification": False,
        "winner_decision": "HISTORICAL_FROZEN_INTERACTION_GRID_DIAGNOSTIC_NO_AUTOMATIC_WINNER",
    }
    identity = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "request_sha256": file_sha256(Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))),
        "v102_engine_sha256": file_sha256(ROOT / "_v102_engine.py"),
        "v103_engine_sha256": file_sha256(ROOT / "_v103_engine.py"),
        "v104_engine_sha256": file_sha256(ROOT / "_v104_engine.py"),
        "engine_sha256": file_sha256(ROOT / "_v105_engine.py"),
        "runner_sha256": file_sha256(ROOT / "autonomous_backtest_v10_5.py"),
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
    write_report(request, summary, phases, comparison, components, parameter_changes, interactions, grid, leader)
    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    print("V105_REAL_BACKTEST_COMPLETE")


if __name__ == "__main__":
    main()
