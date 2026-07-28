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

import _v106_engine as engine


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v10_6"
DEFAULT_REQUEST = ROOT / "request.v10_6.json"


def load_request() -> dict[str, Any]:
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))
    if not path.exists():
        raise FileNotFoundError(f"Missing request file: {path}")
    request = json.loads(path.read_text(encoding="utf-8"))
    validate_request(request)
    return request


def validate_request(request: dict[str, Any]) -> None:
    release = request["release"]
    if release["version"] != "10.6":
        raise ValueError("V10.6 requires release.version=10.6")
    if request["symbol"].upper() != "BTCUSDT":
        raise ValueError("V10.6 is frozen to BTCUSDT")
    if request["source_interval"] != "5m" or request["signal_timeframe_minutes"] != 60:
        raise ValueError("V10.6 requires closed 1h signals aggregated from official 5m data")
    if request["channel_count"] != 3 or request["directional_channel_count"] != 2 or request["portfolio_channel_count"] != 1:
        raise ValueError("V10.6 requires exactly 2 directional channels and 1 portfolio channel")
    if not request["parameters_frozen"] or not release["parameters_fully_frozen"]:
        raise ValueError("V10.6 parameters must be fully frozen")
    if release["parameter_optimization_enabled"] or request["winner_selection_enabled"]:
        raise ValueError("V10.6 forbids optimization and winner selection")
    if not release["contains_historical_backcast"] or not release["contains_unseen_postperiod_holdout"]:
        raise ValueError("V10.6 requires both historical backcast and untouched post-period holdout")
    if release["contains_true_prospective_data"]:
        raise ValueError("V10.6 does not contain true prospective data; true forward observation starts after package freeze")
    if not release["long_short_parameters_are_independent"] or not release["portfolio_uses_shared_position"]:
        raise ValueError("V10.6 must retain independent direction parameters and one shared portfolio position")

    expected = [spec.channel_id for spec in engine.CHANNELS]
    actual = [row["channel_id"] for row in request["channels"]]
    if expected != actual:
        raise ValueError(f"Channel list mismatch: expected={expected}, actual={actual}")
    for row, spec in zip(request["channels"], engine.CHANNELS):
        if row["direction_scope"] != spec.direction_scope or row["profile"] != spec.profile:
            raise ValueError(f"Direction/profile mismatch for {spec.channel_id}")

    execution = request["execution"]
    if float(execution["fee_rate_per_side"]) != 0.0005:
        raise ValueError("Base fee must remain 0.0005 per side")
    if float(execution["tick_size"]) * int(execution["slippage_ticks_per_fill"]) != 0.2:
        raise ValueError("Base slippage must remain 0.2 USDT per fill")
    if execution["same_bar_stop_target_rule"] != "STOP_FIRST_CONSERVATIVE":
        raise ValueError("Same-bar ambiguity must remain stop-first")
    if execution["entry_rule"] != "NEXT_5M_OPEN_AFTER_CLOSED_1H_SIGNAL_BAR":
        raise ValueError("Entry must remain the next 5m open after the closed 1h signal")

    holdout = request["unseen_holdout"]
    if pd.Timestamp(holdout["start_utc"]) != pd.Timestamp("2026-07-01T00:00:00Z"):
        raise ValueError("V10.6 untouched July holdout must start 2026-07-01 UTC")
    if pd.Timestamp(holdout["end_utc"]) > pd.Timestamp("2026-07-26T23:59:59Z"):
        raise ValueError("V10.6 may only use complete verified UTC days through 2026-07-26")
    if holdout["parameters_frozen_before_start"]:
        raise ValueError("The July holdout had already begun before V10.6 freeze and must not be mislabeled prospective")
    if holdout["chronologically_prospective"]:
        raise ValueError("The July holdout is untouched but not chronologically prospective")
    if not holdout["not_used_for_parameter_selection"]:
        raise ValueError("The July holdout must not be used for parameter selection")
    if pd.Timestamp(request["true_forward_observation"]["start_utc"]) < pd.Timestamp("2026-07-29T00:00:00Z"):
        raise ValueError("True forward observation must begin after the V10.6 package freeze")
    if set(request["phases"]) != {
        "backcast_2020_2021",
        "backcast_2022",
        "backcast_2023_2024",
        "v10_development",
        "v10_validation",
        "v10_historical_test",
        "unseen_holdout",
    }:
        raise ValueError("V10.6 phase map is fixed")
    if [row["scenario_id"] for row in request["cost_stress_scenarios"]] != ["base_cost", "cost_1_5x", "cost_2x"]:
        raise ValueError("V10.6 cost stress scenarios are fixed")


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
    phases: pd.DataFrame,
    yearly: pd.DataFrame,
    stress: pd.DataFrame,
    replay: dict[str, Any],
    warmup: pd.DataFrame,
    qualification: dict[str, Any],
) -> None:
    portfolio_id = "60m_portfolio_frozen_v10_6"
    portfolio = summary.set_index("channel_id").loc[portfolio_id]
    portfolio_years = yearly.loc[yearly["channel_id"] == portfolio_id]
    portfolio_phases = phases.loc[phases["channel_id"] == portfolio_id]
    portfolio_stress = stress.loc[stress["channel_id"] == portfolio_id]
    portfolio_audit = qualification[portfolio_id]

    lines = [
        "# BTCUSDT V10.6 冻结跨年份与未见后置留出验证报告",
        "",
        f"- 月度历史数据：**{request['start_month']}至{request['end_month']} UTC**。",
        f"- 未用于调参的后置留出数据：**{request['unseen_holdout']['start_utc']}至{request['unseen_holdout']['end_utc']}**，仅使用完整UTC日。",
        "- 多头、空头、止损、目标、持仓时间和组合规则全部冻结，不进行参数搜索。",
        "- 2020—2024属于反向时间回测；2026年7月1—26日虽未用于调参，但在V10.6冻结时已经发生，因此也不是严格前向测试。",
        f"- 真正的前向观察从 **{request['true_forward_observation']['start_utc']}** 开始，本包不包含该日期之后的数据。",
        "- 即使全部检查通过，本报告也不会自动宣布实盘合格。",
        "",
        "## 冻结参数",
        "",
        "| 方向 | 核心结构 | 止损 | 目标 | 最长持仓 |",
        "|---|---|---:|---:|---:|",
        "| 多头 | 1小时质量回踩；价格距EMA20≤1.20 ATR；ADX≤45 | 1.25 ATR | 2.5R | 36小时 |",
        "| 空头 | 1小时趋势持续性；ADX三根变化≥-4 | 1.25 ATR | 2.5R | 48小时 |",
        "| 组合 | 多空参数独立；共享一个仓位；同时间反向信号全部丢弃 | 组件独立 | 组件独立 | 组件独立 |",
        "",
        "## V10.5精确重放",
        "",
        f"- 重放结论：**{'通过' if replay['passed'] else '失败'}**。",
        "",
        "| 通道 | 预期交易 | 实际交易 | 预期PF | 实际PF | 预期净R | 实际净R | 通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in replay["channels"]:
        lines.append(
            f"| {row['channel_label']} | {int(row['expected_trades'])} | {int(row['actual_trades'])} | "
            f"{float(row['expected_profit_factor']):.3f} | {float(row['actual_profit_factor']):.3f} | "
            f"{float(row['expected_net_R']):.3f} | {float(row['actual_net_R']):.3f} | {'是' if row['passed'] else '否'} |"
        )

    lines.extend([
        "",
        "## 全窗口结果",
        "",
        "| 通道 | 交易 | 胜率 | 实际盈亏比 | PF | 净R | 最大回撤 | 删最佳10%后 | 回溯净R | 未见留出交易 | 未见留出净R | 跨年通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {int(row.trades)} | {row.win_rate:.2%} | {row.avg_win_loss_ratio:.3f} | "
            f"{row.profit_factor:.3f} | {row.net_R:.3f} | {row.max_drawdown_R:.3f} | "
            f"{row.best_10pct_removed_net_R:.3f} | {row.backcast_net_R:.3f} | "
            f"{int(row.unseen_holdout_trades)} | {row.unseen_holdout_net_R:.3f} | "
            f"{'是' if bool(row.cross_year_validation_pass) else '否'} |"
        )

    lines.extend([
        "",
        "## 组合年度表现",
        "",
        "| 年份 | 交易 | 胜率 | PF | 净R | 回撤R |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in portfolio_years.itertuples(index=False):
        lines.append(
            f"| {row.year} | {int(row.trades)} | {row.win_rate:.2%} | {row.profit_factor:.3f} | "
            f"{row.net_R:.3f} | {row.max_drawdown_R:.3f} |"
        )

    lines.extend([
        "",
        "## 组合阶段表现",
        "",
        "| 阶段 | 交易 | 胜率 | PF | 净R | 回撤R |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in portfolio_phases.itertuples(index=False):
        lines.append(
            f"| {row.phase} | {int(row.trades)} | {row.win_rate:.2%} | {row.profit_factor:.3f} | "
            f"{row.net_R:.3f} | {row.max_drawdown_R:.3f} |"
        )

    lines.extend([
        "",
        "## 组合成本压力",
        "",
        "| 场景 | 手续费倍数 | 滑点倍数 | 交易 | PF | 净R | 回撤R | 未见留出净R |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in portfolio_stress.itertuples(index=False):
        lines.append(
            f"| {row.scenario_id} | {row.fee_multiplier:.1f} | {row.slippage_multiplier:.1f} | "
            f"{int(row.trades)} | {row.profit_factor:.3f} | {row.net_R:.3f} | "
            f"{row.max_drawdown_R:.3f} | {row.unseen_holdout_net_R:.3f} |"
        )

    lines.extend([
        "",
        "## 起始窗口敏感性",
        "",
        "连续历史会让2025年初的EMA/ADX拥有更长预热；因此单独比较连续历史与V10.5孤立窗口。",
        "",
        "| 通道 | 连续窗口交易 | 孤立窗口交易 | 共同入场 | 仅连续 | 仅孤立 | 净R差 | 入场完全一致 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in warmup.itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {int(row.continuous_history_trades)} | {int(row.isolated_window_trades)} | "
            f"{int(row.common_entry_times)} | {int(row.continuous_only_entry_times)} | {int(row.isolated_only_entry_times)} | "
            f"{row.delta_net_R_continuous_minus_isolated:+.3f} | {'是' if bool(row.exact_entry_set_match) else '否'} |"
        )

    failed = [name for name, passed in portfolio_audit["checks"].items() if not passed]
    lines.extend([
        "",
        "## 当前判断",
        "",
        f"- 冻结组合全窗口：交易 **{int(portfolio['trades'])}** 笔，胜率 **{portfolio['win_rate']:.2%}**，"
        f"PF **{portfolio['profit_factor']:.3f}**，净R **{portfolio['net_R']:.3f}**，最大回撤 **{portfolio['max_drawdown_R']:.3f}R**。",
        f"- 2020—2024回溯：交易 **{int(portfolio['backcast_trades'])}** 笔，PF **{portfolio['backcast_profit_factor']:.3f}**，净R **{portfolio['backcast_net_R']:.3f}R**。",
        f"- 2026年7月未见后置留出：交易 **{int(portfolio['unseen_holdout_trades'])}** 笔，PF **{portfolio['unseen_holdout_profit_factor']:.3f}**，净R **{portfolio['unseen_holdout_net_R']:.3f}R**。",
        f"- 跨年验证结论：**{'通过' if bool(portfolio['cross_year_validation_pass']) else '未通过'}**。",
        f"- 未通过项目：{', '.join(failed) if failed else '无'}。",
        "- 若未见留出样本不足，不得用该留出集调参；真正前向数据从冻结后的日期开始追加。",
        "",
        "重点文件：`yearly_summary.csv`、`phase_summary.csv`、`rolling_12m_summary.csv`、"
        "`cost_stress_summary.csv`、`year_removal_robustness.csv`、`v10_5_replay_audit.json`、"
        "`warmup_sensitivity_audit.csv`、`unseen_holdout_trade_diagnostics.csv`。",
    ])
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pipeline_smoke(request: dict[str, Any]) -> None:
    synthetic = engine.synthetic_5m_data(140_000, seed=20261006)
    result = engine.run_benchmark(synthetic, request, replay_audit={"passed": True})
    if len(result["summary"]) != 3:
        raise AssertionError("Pipeline smoke did not return 3 channels")
    if len(result["phases"]) != 3 * len(request["phases"]):
        raise AssertionError("Pipeline smoke phase rows mismatch")
    if len(result["cost_stress"]) != 3 * len(request["cost_stress_scenarios"]):
        raise AssertionError("Pipeline smoke cost stress rows mismatch")
    if len(result["signal_audit"]["portfolio_overlap"]) != 1:
        raise AssertionError("Pipeline smoke portfolio audit mismatch")
    if result["trades"].empty:
        raise AssertionError("Pipeline smoke produced no trades")
    print(json.dumps({
        "channels": int(len(result["summary"])),
        "trades": int(len(result["trades"])),
        "portfolio": result["research_leader"],
    }, ensure_ascii=False, indent=2, default=str))
    print("V106_PIPELINE_SMOKE_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT V10.6 frozen cross-year and untouched holdout validation")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pipeline-smoke", action="store_true")
    args = parser.parse_args()
    request = load_request()

    if args.self_test:
        engine.self_test(request)
        print("V106_SELF_TEST_OK")
        return
    if args.pipeline_smoke:
        pipeline_smoke(request)
        return

    clear_results()
    raw, data_audit = engine.load_official_5m_data(request)
    parameter_lock = engine.parameter_lock_payload(request)
    replay_audit, isolated_ledger = engine.replay_v10_5_reference(raw, request)
    benchmark = engine.run_benchmark(raw, request, replay_audit=replay_audit)
    warmup = engine.warmup_sensitivity_audit(benchmark["trades"], isolated_ledger, request)

    summary = benchmark["summary"]
    phases = benchmark["phases"]
    yearly = benchmark["yearly"]
    monthly = benchmark["monthly"]
    directions = benchmark["directions"]
    components = benchmark["components"]
    rolling_3m = benchmark["rolling_3m"]
    rolling_6m = benchmark["rolling_6m"]
    rolling_12m = benchmark["rolling_12m"]
    year_removal = benchmark["year_removal"]
    stress = benchmark["cost_stress"]
    ledger = benchmark["trades"]
    qualification = benchmark["qualification"]
    leader = benchmark["research_leader"]

    summary.to_csv(RESULTS / "channel_summary.csv", index=False)
    phases.to_csv(RESULTS / "phase_summary.csv", index=False)
    yearly.to_csv(RESULTS / "yearly_summary.csv", index=False)
    monthly.to_csv(RESULTS / "monthly_summary.csv", index=False)
    directions.to_csv(RESULTS / "direction_summary.csv", index=False)
    components.to_csv(RESULTS / "portfolio_component_summary.csv", index=False)
    rolling_3m.to_csv(RESULTS / "rolling_3m_summary.csv", index=False)
    rolling_6m.to_csv(RESULTS / "rolling_6m_summary.csv", index=False)
    rolling_12m.to_csv(RESULTS / "rolling_12m_summary.csv", index=False)
    year_removal.to_csv(RESULTS / "year_removal_robustness.csv", index=False)
    stress.to_csv(RESULTS / "cost_stress_summary.csv", index=False)
    ledger.to_csv(RESULTS / "trade_ledger.csv", index=False)
    warmup.to_csv(RESULTS / "warmup_sensitivity_audit.csv", index=False)
    pd.DataFrame(benchmark["signal_audit"]["channels"]).to_csv(RESULTS / "signal_counts.csv", index=False)
    pd.DataFrame(benchmark["signal_audit"]["portfolio_overlap"]).to_csv(RESULTS / "portfolio_overlap_audit.csv", index=False)
    unseen_holdout = ledger.loc[ledger["phase"] == "unseen_holdout"].copy() if not ledger.empty else ledger.copy()
    unseen_holdout.to_csv(RESULTS / "unseen_holdout_trade_diagnostics.csv", index=False)

    passed_channels = [channel_id for channel_id, audit in qualification.items() if audit["cross_year_validation_pass"]]
    portfolio_id = "60m_portfolio_frozen_v10_6"
    portfolio_audit = qualification[portfolio_id]
    status = {
        "engine": "BTCUSDT hourly frozen cross-year and untouched post-period holdout validation V10.6",
        "release_version": "10.6",
        "evaluation_mode": request["evaluation_mode"],
        "parameters_frozen": True,
        "parameter_optimization_enabled": False,
        "parameter_lock_sha256": parameter_lock["parameter_lock_sha256"],
        "long_short_parameters_are_independent": True,
        "portfolio_uses_shared_position": True,
        "channel_count": 3,
        "v10_5_reference_replay_passed": bool(replay_audit["passed"]),
        "cross_year_validation_pass_count": len(passed_channels),
        "cross_year_validation_pass_channels": passed_channels,
        "portfolio_cross_year_validation_pass": bool(portfolio_audit["cross_year_validation_pass"]),
        "unseen_holdout_start_utc": request["unseen_holdout"]["start_utc"],
        "unseen_holdout_end_utc": request["unseen_holdout"]["end_utc"],
        "unseen_holdout_portfolio_trades": portfolio_audit["unseen_holdout_metrics"]["trades"],
        "unseen_holdout_portfolio_net_R": portfolio_audit["unseen_holdout_metrics"]["net_R"],
        "qualified_for_live_trading": False,
        "winner_selection_enabled": False,
        "research_leader": leader,
        "true_forward_observation_start_utc": request["true_forward_observation"]["start_utc"],
        "next_step": "Keep parameters frozen. Begin append-only true forward observation from 2026-07-29 UTC; never retune on the July 1-26 untouched holdout.",
    }
    benchmark_summary = {
        "data_window": {
            "monthly_start": request["start_month"],
            "monthly_end": request["end_month"],
            "unseen_holdout_start": request["unseen_holdout"]["start_utc"],
            "unseen_holdout_end": request["unseen_holdout"]["end_utc"],
        },
        "parameter_lock_sha256": parameter_lock["parameter_lock_sha256"],
        "channels": summary.to_dict(orient="records"),
        "v10_5_reference_replay": replay_audit,
        "cross_year_validation_pass_channels": passed_channels,
        "live_qualification": False,
        "decision": "FROZEN_VALIDATION_ONLY_NO_AUTOMATIC_LIVE_APPROVAL",
    }
    future_manifest = {
        "strategy_version": "10.6",
        "parameters_frozen": True,
        "parameter_lock_sha256": parameter_lock["parameter_lock_sha256"],
        "holdout_start_utc": request["unseen_holdout"]["start_utc"],
        "holdout_end_utc": request["unseen_holdout"]["end_utc"],
        "holdout_used_for_parameter_selection": False,
        "holdout_chronologically_prospective": False,
        "true_forward_observation_start_utc": request["true_forward_observation"]["start_utc"],
        "append_only_rule": "Only data from the true forward observation start onward may be appended; frozen strategy parameters may not be changed.",
        "failure_rule": "A failed holdout is not repaired by retuning on the same holdout.",
    }
    identity = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "request_sha256": file_sha256(Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))),
        "v102_engine_sha256": file_sha256(ROOT / "_v102_engine.py"),
        "v103_engine_sha256": file_sha256(ROOT / "_v103_engine.py"),
        "v104_engine_sha256": file_sha256(ROOT / "_v104_engine.py"),
        "v105_engine_sha256": file_sha256(ROOT / "_v105_engine.py"),
        "v106_engine_sha256": file_sha256(ROOT / "_v106_engine.py"),
        "runner_sha256": file_sha256(ROOT / "autonomous_backtest_v10_6.py"),
        "parameter_lock_sha256": parameter_lock["parameter_lock_sha256"],
    }

    json_dump(RESULTS / "status.json", status)
    json_dump(RESULTS / "benchmark_summary.json", benchmark_summary)
    json_dump(RESULTS / "data_audit.json", data_audit)
    json_dump(RESULTS / "signal_and_portfolio_audit.json", benchmark["signal_audit"])
    json_dump(RESULTS / "robustness_audit.json", qualification)
    json_dump(RESULTS / "parameter_lock.json", parameter_lock)
    json_dump(RESULTS / "future_validation_manifest.json", future_manifest)
    json_dump(RESULTS / "v10_5_replay_audit.json", replay_audit)
    json_dump(RESULTS / "strategy_spec.json", {
        "channels": request["channels"],
        "direction_specific_parameters": request["direction_specific_parameters"],
        "portfolio_rules": request["portfolio_rules"],
        "execution": request["execution"],
        "phases": request["phases"],
        "unseen_holdout": request["unseen_holdout"],
        "cost_stress_scenarios": request["cost_stress_scenarios"],
        "cross_year_thresholds": request["cross_year_thresholds"],
        "no_lookahead_rules": request["no_lookahead_rules"],
    })
    json_dump(RESULTS / "run_identity.json", identity)
    (RESULTS / "run_identity.txt").write_text("\n".join(f"{key}={value}" for key, value in identity.items()) + "\n", encoding="utf-8")
    write_report(request, summary, phases, yearly, stress, replay_audit, warmup, qualification)
    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    print("V106_REAL_VALIDATION_COMPLETE")


if __name__ == "__main__":
    main()
