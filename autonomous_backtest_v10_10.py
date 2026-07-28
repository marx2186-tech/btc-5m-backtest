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

import _v110_engine as engine


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v10_10"
DEFAULT_REQUEST = ROOT / "request.v10_10.json"


def load_request() -> dict[str, Any]:
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))
    if not path.exists():
        raise FileNotFoundError(f"Missing request file: {path}")
    request = json.loads(path.read_text(encoding="utf-8"))
    validate_request(request)
    return request


def validate_request(request: dict[str, Any]) -> None:
    release = request["release"]
    if release["version"] != "10.10":
        raise ValueError("V10.10 requires release.version=10.10")
    if request["symbol"].upper() != "BTCUSDT" or request["source_interval"] != "5m":
        raise ValueError("V10.10 is frozen to BTCUSDT official 5m source data")
    if request["start_month"] != "2025-01" or request["end_month"] != "2026-06":
        raise ValueError("V10.10 must use 2025 warmup and end at 2026-06")
    if request["evaluation_window"]["start_utc"] != "2026-01-01T00:00:00Z":
        raise ValueError("V10.10 evaluation start is fixed")
    if request["evaluation_window"]["end_utc"] != "2026-06-30T23:59:59Z":
        raise ValueError("V10.10 evaluation end is fixed")
    if request["channel_count"] != 4 or len(request["channels"]) != 4:
        raise ValueError("V10.10 requires exactly four channels")
    expected_ids = [spec.channel_id for spec in engine.CHANNELS]
    actual_ids = [row["channel_id"] for row in request["channels"]]
    if actual_ids != expected_ids:
        raise ValueError(f"Channel list mismatch: expected={expected_ids}, actual={actual_ids}")
    if release["parameter_optimization_enabled"] or release["winner_selection_enabled"]:
        raise ValueError("V10.10 is a fixed comparison, not a parameter search")
    if not release["upper_state_machine_identical_to_v10_9"]:
        raise ValueError("V10.10 must preserve the V10.9 upper state machine")
    if not release["same_confirmation_pool_for_all_experiments"] or not release["one_setup_per_cycle"]:
        raise ValueError("All V10.10 experiments must use the same one-hour confirmation pool")

    cycle = request["state_machine_parameters"]["one_hour_cycle"]
    frozen_cycle = {
        "impulse_breakout_lookback_hours": 12,
        "minimum_impulse_body_atr": 0.45,
        "minimum_impulse_close_location_value": 0.65,
        "minimum_impulse_volume_ratio": 0.90,
        "minimum_impulse_extension_ema20_atr": 0.60,
        "maximum_impulse_extension_ema20_atr": 2.00,
        "minimum_pullback_retracement_atr": 0.50,
        "pullback_touch_ema20_zone_atr": 0.20,
        "ema50_structure_tolerance_atr": 0.15,
        "maximum_hours_impulse_to_pullback": 18,
        "maximum_hours_pullback_to_confirmation": 6,
        "minimum_confirmation_body_atr": 0.25,
        "minimum_confirmation_close_location_value": 0.65,
        "maximum_confirmation_extension_ema20_atr": 0.90,
        "cycle_cooldown_hours": 8,
        "maximum_cycles_per_4h_environment_leg": 2,
    }
    for key, expected in frozen_cycle.items():
        if float(cycle[key]) != float(expected):
            raise ValueError(f"V10.9 upper-state parameter changed: {key}")

    env = request["multi_timeframe_parameters"]["four_hour_environment"]
    if float(env["minimum_adx14"]) != 18.0 or float(env["minimum_abs_ema50_slope_3_atr"]) != 0.02:
        raise ValueError("V10.9 four-hour environment must remain frozen")

    execution = request["state_machine_parameters"]["execution"]
    if float(execution["stop_atr_multiple"]) != 1.25 or float(execution["reward_risk"]) != 2.0:
        raise ValueError("All experiment channels must remain 1.25 ATR stop and 2R target")
    if int(execution["max_holding_hours"]) != 48:
        raise ValueError("Maximum holding must remain 48 hours")

    triggers = request["trigger_comparison_parameters"]
    local = triggers["single_local_breakout"]
    reclaim = triggers["pullback_reclaim"]
    if int(local["local_breakout_lookback_15m_bars"]) != 4:
        raise ValueError("Local breakout lookback must remain four 15m bars")
    if int(local["minimum_delay_after_confirmation_minutes"]) != 15 or float(local["setup_validity_hours"]) != 3.0:
        raise ValueError("Local breakout timing changed")
    if bool(local["require_confirmation_extreme_break"]):
        raise ValueError("V10.10 local breakout must not require the one-hour confirmation extreme")
    if int(reclaim["minimum_delay_after_confirmation_minutes"]) != 15 or float(reclaim["setup_validity_hours"]) != 4.0:
        raise ValueError("Pullback reclaim timing changed")
    if float(reclaim["ema20_touch_zone_atr"]) != 0.15 or not bool(reclaim["reclaim_must_occur_after_touch_bar"]):
        raise ValueError("Pullback reclaim touch/reclaim rule changed")

    costs = request["execution"]
    if float(costs["fee_rate_per_side"]) != 0.0005:
        raise ValueError("fee_rate_per_side must remain 0.0005")
    if float(costs["tick_size"]) * int(costs["slippage_ticks_per_fill"]) != 0.2:
        raise ValueError("slippage must remain 0.2 USDT per fill")
    if costs["same_bar_stop_target_rule"] != "STOP_FIRST_CONSERVATIVE":
        raise ValueError("same-bar ambiguity must remain conservative")


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


def write_report(request: dict[str, Any], benchmark: dict[str, Any]) -> None:
    summary = benchmark["summary"]
    direction = benchmark["direction_summary"]
    funnel = benchmark["signal_funnel"]
    coverage = benchmark["trigger_coverage"]
    losses = benchmark["loss_diagnostics"]
    lines = [
        "# BTCUSDT V10.10 冻结状态机成交层对照报告",
        "",
        "- 统计区间：**2026-01-01至2026-06-30 UTC**；2025年只用于指标预热。",
        "- 4H趋势腿和1H推动—回踩—确认状态机完整继承V10.9，不修改任何上游参数。",
        "- 三个实验通道使用完全相同的1H确认池，只比较成交层。",
        "- 实验A：1H确认收盘后直接在下一根5M开盘成交。",
        "- 实验B：等待首次15M单一局部结构突破，不再同时要求突破1H确认极值。",
        "- 实验C：等待15M触及EMA20区域，再由后续完整15M K线重新顺势确认。",
        "- 全部实验统一1H ATR×1.25止损、2R目标、最长48小时；该已查看区间只用于诊断。",
        "",
        "## 总体结果",
        "",
        "| 通道 | 信号 | 交易 | 胜/负 | 胜率 | 实际盈亏比 | PF | 净R | 最大回撤 | 删最佳10%后 | 门槛通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    signal_counts = {row["channel_id"]: row["signals"] for row in benchmark["signal_audit"]["channels"]}
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {int(signal_counts.get(row.channel_id, 0))} | {int(row.trades)} | "
            f"{int(row.wins)}/{int(row.losses)} | {row.win_rate:.2%} | {row.avg_win_loss_ratio:.3f} | "
            f"{row.profit_factor:.3f} | {row.net_R:.3f} | {row.max_drawdown_R:.3f} | "
            f"{row.best_10pct_removed_net_R:.3f} | {int(row.checks_passed)}/{int(row.checks_total)} |"
        )

    lines.extend([
        "",
        "## 状态机与成交漏斗",
        "",
        "| 阶段 | 多头 | 空头 |",
        "|---|---:|---:|",
    ])
    for row in funnel.itertuples(index=False):
        lines.append(f"| {row.stage} | {int(row.long_count)} | {int(row.short_count)} |")

    lines.extend([
        "",
        "## 相同1H确认池的覆盖率",
        "",
        "| 通道 | 可用1H确认 | 发出信号 | 覆盖率 | 多头 | 空头 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in coverage.itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {int(row.available_1h_confirmations)} | {int(row.signals_emitted)} | "
            f"{row.coverage_rate:.2%} | {int(row.long_signals)} | {int(row.short_signals)} |"
        )

    lines.extend([
        "",
        "## 多空方向贡献",
        "",
        "| 通道 | 方向 | 交易 | 胜率 | PF | 净R | 回撤R |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    if direction.empty:
        lines.append("| 无 | 无 | 0 | 0.00% | 0.000 | 0.000 | 0.000 |")
    else:
        for row in direction.itertuples(index=False):
            label = "多头" if int(row.direction) == 1 else "空头"
            lines.append(
                f"| {row.channel_label} | {label} | {int(row.trades)} | {row.win_rate:.2%} | "
                f"{row.profit_factor:.3f} | {row.net_R:.3f} | {row.max_drawdown_R:.3f} |"
            )

    lines.extend([
        "",
        "## 亏损类型",
        "",
        "| 通道 | 类型 | 数量 | 占该通道亏损 | 平均净R | 平均MFE | 平均MAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    if losses.empty:
        lines.append("| 无 | 无亏损或无交易 | 0 | 0.00% | 0.000 | 0.000 | 0.000 |")
    else:
        for row in losses.itertuples(index=False):
            lines.append(
                f"| {row.channel_label} | {row.loss_class} | {int(row.trades)} | {row.share_of_channel_losses:.2%} | "
                f"{row.avg_net_R:.3f} | {row.avg_mfe_R:.3f} | {row.avg_mae_R:.3f} |"
            )

    by_id = summary.set_index("channel_id")
    lines.extend([
        "",
        "## 当前判断规则",
        "",
        "- 先看实验A：若1H确认后直接成交也没有正期望，说明问题主要在1H确认池，而不是15M触发。",
        "- 若实验A有效、实验B/C无效，说明等待15M造成延迟或错误过滤。",
        "- 若实验B优于A，说明单一局部突破能提高入场质量；若C优于A，说明等待15M回踩确认更合适。",
        "- 本版本禁止自动选优；任何看起来更好的通道仍必须在新的独立时间段验证。",
        "",
    ])
    for channel_id in [
        "baseline_1h_shared_v10_6",
        "state_direct_after_1h_confirmation_rr2_0",
        "state_15m_single_local_break_rr2_0",
        "state_15m_pullback_reclaim_rr2_0",
    ]:
        row = by_id.loc[channel_id]
        lines.append(
            f"- {row.channel_label}：{int(row.trades)}笔，胜率{row.win_rate:.2%}，PF {row.profit_factor:.3f}，"
            f"净R {row.net_R:.3f}，回撤{row.max_drawdown_R:.3f}R。"
        )
    lines.extend([
        "",
        "重点文件：`trigger_coverage.csv`、`setup_signal_matrix.csv`、`setup_trade_matrix.csv`、",
        "`state_machine_funnel.csv`、`channel_summary.csv`、`loss_diagnostics.csv`、`trade_ledger.csv`。",
    ])
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pipeline_smoke(request: dict[str, Any]) -> None:
    raw = engine.synthetic_5m_data(220_000, seed=20261010)
    result = engine.run_benchmark(raw, request)
    if len(result["summary"]) != 4:
        raise AssertionError("Pipeline smoke did not return four channels")
    if set(result["signals"]) != {spec.channel_id for spec in engine.CHANNELS}:
        raise AssertionError("Pipeline smoke signal channel mismatch")
    matrix = result["setup_signal_matrix"]
    if not matrix.empty and not matrix["direct_triggered"].all():
        raise AssertionError("Direct channel did not consume every evaluation-window confirmation")
    if not result["signal_audit"]["trigger_comparison"]["all_channels_use_same_confirmation_pool"]:
        raise AssertionError("A trigger channel referenced a setup outside the common pool")
    print(json.dumps({
        "channels": 4,
        "trades": int(len(result["trades"])),
        "confirmations": int(len(matrix)),
        "direct_signals": int(len(result["signals"]["state_direct_after_1h_confirmation_rr2_0"])),
        "local_break_signals": int(len(result["signals"]["state_15m_single_local_break_rr2_0"])),
        "pullback_reclaim_signals": int(len(result["signals"]["state_15m_pullback_reclaim_rr2_0"])),
    }, ensure_ascii=False, indent=2))
    print("V110_PIPELINE_SMOKE_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT V10.10 frozen state-machine entry-layer comparison")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pipeline-smoke", action="store_true")
    args = parser.parse_args()
    request = load_request()

    if args.self_test:
        engine.self_test(request)
        print("V110_SELF_TEST_OK")
        return
    if args.pipeline_smoke:
        pipeline_smoke(request)
        return

    clear_results()
    raw, data_audit = engine.load_official_5m_data(request)
    benchmark = engine.run_benchmark(raw, request)
    summary = benchmark["summary"]
    ledger = benchmark["trades"]

    summary.to_csv(RESULTS / "channel_summary.csv", index=False)
    benchmark["direction_summary"].to_csv(RESULTS / "direction_summary.csv", index=False)
    benchmark["monthly_summary"].to_csv(RESULTS / "monthly_summary.csv", index=False)
    benchmark["trigger_type_summary"].to_csv(RESULTS / "trigger_type_summary.csv", index=False)
    benchmark["exit_reason_summary"].to_csv(RESULTS / "exit_reason_summary.csv", index=False)
    benchmark["loss_diagnostics"].to_csv(RESULTS / "loss_diagnostics.csv", index=False)
    benchmark["signal_funnel"].to_csv(RESULTS / "state_machine_funnel.csv", index=False)
    benchmark["state_machine_events"].to_csv(RESULTS / "state_machine_event_ledger.csv", index=False)
    benchmark["trigger_coverage"].to_csv(RESULTS / "trigger_coverage.csv", index=False)
    benchmark["setup_signal_matrix"].to_csv(RESULTS / "setup_signal_matrix.csv", index=False)
    benchmark["setup_trade_matrix"].to_csv(RESULTS / "setup_trade_matrix.csv", index=False)
    pd.DataFrame(benchmark["signal_audit"]["channels"]).to_csv(RESULTS / "signal_counts.csv", index=False)
    ledger.to_csv(RESULTS / "trade_ledger.csv", index=False)

    qualification = benchmark["qualification"]
    gate_channels = [channel_id for channel_id, audit in qualification.items() if audit["historical_research_gate_pass"]]
    results_by_id = {
        channel_id: summary.loc[summary["channel_id"] == channel_id].iloc[0].to_dict()
        for channel_id in [spec.channel_id for spec in engine.CHANNELS]
    }
    status = {
        "engine": "BTCUSDT V10.10 frozen V10.9 upper state machine / three entry-layer comparison",
        "release_version": "10.10",
        "historical_diagnostic_only": True,
        "qualified_for_live_trading": False,
        "parameter_optimization_enabled": False,
        "winner_selection_enabled": False,
        "evaluation_start_utc": request["evaluation_window"]["start_utc"],
        "evaluation_end_utc": request["evaluation_window"]["end_utc"],
        "warmup_start_utc": request["warmup_window"]["start_utc"],
        "channel_count": 4,
        "state_machine_audit": benchmark["signal_audit"]["state_machine"],
        "trigger_comparison_audit": benchmark["signal_audit"]["trigger_comparison"],
        "historical_research_gate_pass_channels": gate_channels,
        "baseline_result": results_by_id["baseline_1h_shared_v10_6"],
        "direct_after_1h_confirmation_result": results_by_id["state_direct_after_1h_confirmation_rr2_0"],
        "single_local_break_result": results_by_id["state_15m_single_local_break_rr2_0"],
        "pullback_reclaim_result": results_by_id["state_15m_pullback_reclaim_rr2_0"],
        "next_step": "Use the paired confirmation matrix to determine whether the edge is in the frozen one-hour confirmation pool or in one entry method. Do not tune these triggers on the viewed 2026-H1 window.",
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
        "v108_engine_sha256": file_sha256(ROOT / "_v108_engine.py"),
        "v109_engine_sha256": file_sha256(ROOT / "_v109_engine.py"),
        "engine_sha256": file_sha256(ROOT / "_v110_engine.py"),
        "runner_sha256": file_sha256(ROOT / "autonomous_backtest_v10_10.py"),
    }

    json_dump(RESULTS / "status.json", status)
    json_dump(RESULTS / "data_audit.json", data_audit)
    json_dump(RESULTS / "signal_audit.json", benchmark["signal_audit"])
    json_dump(RESULTS / "qualification_audit.json", qualification)
    json_dump(RESULTS / "strategy_spec.json", engine.parameter_manifest(request))
    json_dump(RESULTS / "run_identity.json", identity)
    (RESULTS / "run_identity.txt").write_text("\n".join(f"{key}={value}" for key, value in identity.items()) + "\n", encoding="utf-8")
    write_report(request, benchmark)
    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    print("V110_REAL_BACKTEST_COMPLETE")


if __name__ == "__main__":
    main()
