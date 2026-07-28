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

import _v111_engine as engine


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v10_11"
DEFAULT_REQUEST = ROOT / "request.v10_11.json"


def load_request() -> dict[str, Any]:
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))
    if not path.exists():
        raise FileNotFoundError(f"Missing request file: {path}")
    request = json.loads(path.read_text(encoding="utf-8"))
    validate_request(request)
    return request


def validate_request(request: dict[str, Any]) -> None:
    release = request["release"]
    if release["version"] != "10.11":
        raise ValueError("V10.11 requires release.version=10.11")
    if request["symbol"].upper() != "BTCUSDT" or request["source_interval"] != "5m":
        raise ValueError("V10.11 is frozen to BTCUSDT official 5m source data")
    if request["start_month"] != "2025-01" or request["end_month"] != "2026-06":
        raise ValueError("V10.11 must use 2025 warmup and end at 2026-06")
    if request["evaluation_window"]["start_utc"] != "2026-01-01T00:00:00Z":
        raise ValueError("V10.11 evaluation start is fixed")
    if request["evaluation_window"]["end_utc"] != "2026-06-30T23:59:59Z":
        raise ValueError("V10.11 evaluation end is fixed")
    if request["channel_count"] != 34 or len(request["channels"]) != 34:
        raise ValueError("V10.11 requires exactly 34 channels")
    expected_ids = [spec.channel_id for spec in engine.CHANNELS]
    actual_ids = [row["channel_id"] for row in request["channels"]]
    if actual_ids != expected_ids:
        raise ValueError("V10.11 channel list or order does not match the frozen engine")
    if release["parameter_optimization_enabled"] or release["winner_selection_enabled"]:
        raise ValueError("V10.11 is a fixed batch comparison, not parameter optimization")
    for key in [
        "upper_state_machine_identical_to_v10_9",
        "same_confirmation_pool_for_all_experiments",
        "one_setup_per_cycle",
        "single_upload_batch_execution",
    ]:
        if not bool(release[key]):
            raise ValueError(f"Frozen release flag must be true: {key}")

    matrix = request["matrix_definition"]
    expected_matrix = {
        "baseline_channels": 1,
        "fixed_directional_channels": 18,
        "fixed_shared_channels": 9,
        "break_even_shadow_channels": 6,
    }
    for key, value in expected_matrix.items():
        if int(matrix[key]) != value:
            raise ValueError(f"Matrix count mismatch: {key}")
    if matrix["entry_modes"] != ["direct_1h", "local_break_15m", "pullback_reclaim_15m"]:
        raise ValueError("Entry mode matrix changed")
    if [float(v) for v in matrix["reward_risks_fixed"]] != [1.5, 2.0, 2.5]:
        raise ValueError("Reward-risk matrix changed")
    if float(matrix["break_even_reward_risk"]) != 2.0 or float(matrix["break_even_activation_R"]) != 1.0:
        raise ValueError("Break-even shadow definition changed")
    if not bool(matrix["shared_break_even_channels_excluded"]):
        raise ValueError("V10.11 intentionally excludes shared-position break-even shadows")

    for row, spec in zip(request["channels"], engine.CHANNELS):
        meta = engine.CHANNEL_META[spec.channel_id]
        if row["direction_scope"] != spec.direction_scope:
            raise ValueError(f"Direction scope mismatch: {spec.channel_id}")
        if row["entry_mode"] != meta.entry_mode or row["direction_variant"] != meta.direction_variant:
            raise ValueError(f"Channel metadata mismatch: {spec.channel_id}")
        if row["management_mode"] != meta.management_mode:
            raise ValueError(f"Management mode mismatch: {spec.channel_id}")
        if float(row["reward_risk"]) != float(meta.reward_risk):
            raise ValueError(f"Reward-risk mismatch: {spec.channel_id}")
        if row["base_signal_channel_id"] != meta.base_signal_channel_id:
            raise ValueError(f"Base signal mismatch: {spec.channel_id}")

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
    if float(execution["stop_atr_multiple"]) != 1.25 or int(execution["max_holding_hours"]) != 48:
        raise ValueError("Experiment stop/holding definition changed")
    if float(request["state_machine_parameters"]["shadow_management"]["break_even_activation_R"]) != 1.0:
        raise ValueError("Break-even activation must remain 1R")

    local = request["trigger_comparison_parameters"]["single_local_breakout"]
    reclaim = request["trigger_comparison_parameters"]["pullback_reclaim"]
    if int(local["local_breakout_lookback_15m_bars"]) != 4:
        raise ValueError("Local breakout lookback changed")
    if int(local["minimum_delay_after_confirmation_minutes"]) != 15 or float(local["setup_validity_hours"]) != 3.0:
        raise ValueError("Local breakout timing changed")
    if bool(local["require_confirmation_extreme_break"]):
        raise ValueError("Local breakout must not require the 1H confirmation extreme")
    if int(reclaim["minimum_delay_after_confirmation_minutes"]) != 15 or float(reclaim["setup_validity_hours"]) != 4.0:
        raise ValueError("Pullback-reclaim timing changed")
    if float(reclaim["ema20_touch_zone_atr"]) != 0.15 or not bool(reclaim["reclaim_must_occur_after_touch_bar"]):
        raise ValueError("Pullback-reclaim rule changed")

    costs = request["execution"]
    if float(costs["fee_rate_per_side"]) != 0.0005:
        raise ValueError("fee_rate_per_side must remain 0.0005")
    if float(costs["tick_size"]) * int(costs["slippage_ticks_per_fill"]) != 0.2:
        raise ValueError("Base slippage must remain 0.2 USDT per fill")
    if costs["same_bar_stop_target_rule"] != "STOP_FIRST_CONSERVATIVE":
        raise ValueError("Same-bar ambiguity must remain conservative")
    if float(request["cost_stress"]["multiplier"]) != 1.5:
        raise ValueError("Cost stress multiplier must remain 1.5x")


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


def _summary_subset(summary: pd.DataFrame, management: str, direction: str) -> pd.DataFrame:
    return summary.loc[
        (summary["management_mode"] == management)
        & (summary["direction_variant"] == direction)
    ].sort_values(["entry_mode", "reward_risk"])


def _append_summary_table(lines: list[str], title: str, frame: pd.DataFrame) -> None:
    lines.extend([
        "",
        f"## {title}",
        "",
        "| 入场 | RR | 交易 | 胜率 | 实际盈亏比 | PF | 净R | 回撤 | 连亏 | 1.5倍成本PF | 删最佳10%后 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    if frame.empty:
        lines.append("| 无 | 0 | 0 | 0.00% | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0.000 | 0.000 |")
        return
    entry_labels = {
        "direct_1h": "1H直接",
        "local_break_15m": "15M突破",
        "pullback_reclaim_15m": "15M回踩确认",
    }
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {entry_labels.get(row.entry_mode, row.entry_mode)} | {row.reward_risk:.1f} | {int(row.trades)} | "
            f"{row.win_rate:.2%} | {row.avg_win_loss_ratio:.3f} | {row.profit_factor:.3f} | {row.net_R:.3f} | "
            f"{row.max_drawdown_R:.3f} | {int(row.max_consecutive_losses)} | {row.cost_1_5x_profit_factor:.3f} | "
            f"{row.best_10pct_removed_net_R:.3f} |"
        )


def write_report(request: dict[str, Any], benchmark: dict[str, Any]) -> None:
    summary = benchmark["summary"]
    baseline = summary.loc[summary["is_baseline"]].iloc[0]
    lines = [
        "# BTCUSDT V10.11 一次上传批量实验矩阵报告",
        "",
        "- 统计区间：**2026-01-01至2026-06-30 UTC**；2025年只用于指标预热。",
        "- V10.9的4H趋势腿与1H推动—回踩状态机完整冻结。",
        "- 一次运行共34条通道：1条基准、18条多/空独立固定管理、9条多空共享固定管理、6条1R保本影子。",
        "- 固定测试三种入场：1H确认直接、15M局部突破、15M回踩EMA20后重确认。",
        "- 固定测试三种目标：1.5R、2.0R、2.5R；同时执行1.5倍成本压力。",
        "- 本区间已经查看，所有结果仅用于诊断；排序不等于自动选优或实盘资格。",
        "",
        "## 原始基准",
        "",
        f"- 交易：**{int(baseline.trades)}**；胜率：**{baseline.win_rate:.2%}**；PF：**{baseline.profit_factor:.3f}**；"
        f"净R：**{baseline.net_R:.3f}**；最大回撤：**{baseline.max_drawdown_R:.3f}R**。",
    ]
    _append_summary_table(lines, "多头固定管理9组", _summary_subset(summary, "FIXED_STOP_TARGET", "long"))
    _append_summary_table(lines, "空头固定管理9组", _summary_subset(summary, "FIXED_STOP_TARGET", "short"))
    _append_summary_table(lines, "多空共享固定管理9组", _summary_subset(summary, "FIXED_STOP_TARGET", "shared"))

    lines.extend([
        "",
        "## 1R保本影子对照（仅多头/空头，RR2.0）",
        "",
        "| 入场 | 方向 | 固定净R | 保本净R | 差值 | 固定PF | 保本PF | 固定回撤 | 保本回撤 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in benchmark["management_comparison"].itertuples(index=False):
        direction_label = "多头" if row.direction_variant == "long" else "空头"
        lines.append(
            f"| {row.entry_mode} | {direction_label} | {row.net_R_fixed:.3f} | {row.net_R_be1:.3f} | "
            f"{row.net_R_delta_be1_minus_fixed:+.3f} | {row.profit_factor_fixed:.3f} | {row.profit_factor_be1:.3f} | "
            f"{row.max_drawdown_R_fixed:.3f} | {row.max_drawdown_R_be1:.3f} |"
        )

    lines.extend([
        "",
        "## 结果纪律",
        "",
        "- 优先检查交易数量、方向稳定性、PF、删除最佳10%后净R、最大连续亏损和1.5倍成本结果。",
        "- 单笔或极少样本的100%胜率、PF=999只代表没有亏损样本，不能视为合格。",
        "- `diagnostic_rank.csv`只是方便阅读的诊断排序，不自动选择赢家。",
        "- 后续应从同一个结果包中集中淘汰路线，不再为单个小变化重复上传。",
        "",
        "重点文件：`channel_summary.csv`、`diagnostic_rank.csv`、`management_comparison.csv`、",
        "`cost_stress_summary.csv`、`setup_channel_matrix.csv`、`loss_diagnostics.csv`、`trade_ledger.csv`。",
    ])
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pipeline_smoke(request: dict[str, Any]) -> None:
    raw = engine.synthetic_5m_data(220_000, seed=20261011)
    result = engine.run_benchmark(raw, request)
    if len(result["summary"]) != 34:
        raise AssertionError("Pipeline smoke did not return 34 channels")
    if len(result["cost_stress_summary"]) != 34:
        raise AssertionError("Pipeline smoke cost stress channel mismatch")
    if len(result["management_comparison"]) != 6:
        raise AssertionError("Pipeline smoke management comparison mismatch")
    if set(result["signals"]) != {spec.channel_id for spec in engine.CHANNELS}:
        raise AssertionError("Pipeline smoke signal channel mismatch")
    print(json.dumps({
        "channels": 34,
        "trades": int(len(result["trades"])),
        "setup_channel_rows": int(len(result["setup_channel_matrix"])),
        "management_pairs": int(len(result["management_comparison"])),
    }, ensure_ascii=False, indent=2))
    print("V111_PIPELINE_SMOKE_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT V10.11 34-channel batch experiment matrix")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pipeline-smoke", action="store_true")
    args = parser.parse_args()
    request = load_request()

    if args.self_test:
        engine.self_test(request)
        print("V111_SELF_TEST_OK")
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
    benchmark["diagnostic_rank"].to_csv(RESULTS / "diagnostic_rank.csv", index=False)
    benchmark["direction_summary"].to_csv(RESULTS / "direction_summary.csv", index=False)
    benchmark["monthly_summary"].to_csv(RESULTS / "monthly_summary.csv", index=False)
    benchmark["trigger_type_summary"].to_csv(RESULTS / "trigger_type_summary.csv", index=False)
    benchmark["exit_reason_summary"].to_csv(RESULTS / "exit_reason_summary.csv", index=False)
    benchmark["loss_diagnostics"].to_csv(RESULTS / "loss_diagnostics.csv", index=False)
    benchmark["management_comparison"].to_csv(RESULTS / "management_comparison.csv", index=False)
    benchmark["cost_stress_summary"].to_csv(RESULTS / "cost_stress_summary.csv", index=False)
    benchmark["signal_funnel"].to_csv(RESULTS / "state_machine_funnel.csv", index=False)
    benchmark["state_machine_events"].to_csv(RESULTS / "state_machine_event_ledger.csv", index=False)
    benchmark["trigger_coverage"].to_csv(RESULTS / "trigger_coverage.csv", index=False)
    benchmark["setup_signal_matrix"].to_csv(RESULTS / "setup_signal_matrix.csv", index=False)
    benchmark["setup_channel_matrix"].to_csv(RESULTS / "setup_channel_matrix.csv", index=False)
    pd.DataFrame(benchmark["signal_audit"]["channels"]).to_csv(RESULTS / "signal_counts.csv", index=False)
    ledger.to_csv(RESULTS / "trade_ledger.csv", index=False)

    qualification = benchmark["qualification"]
    gate_channels = [channel_id for channel_id, audit in qualification.items() if audit["historical_research_gate_pass"]]
    baseline = summary.loc[summary["channel_id"] == "baseline_1h_shared_v10_6"].iloc[0].to_dict()
    status = {
        "engine": "BTCUSDT V10.11 frozen state-machine 34-channel batch matrix",
        "release_version": "10.11",
        "historical_diagnostic_only": True,
        "qualified_for_live_trading": False,
        "parameter_optimization_enabled": False,
        "winner_selection_enabled": False,
        "evaluation_start_utc": request["evaluation_window"]["start_utc"],
        "evaluation_end_utc": request["evaluation_window"]["end_utc"],
        "warmup_start_utc": request["warmup_window"]["start_utc"],
        "channel_count": 34,
        "matrix_counts": benchmark["signal_audit"]["matrix"],
        "state_machine_audit": benchmark["signal_audit"]["state_machine"],
        "trigger_comparison_audit": benchmark["signal_audit"]["trigger_comparison"],
        "historical_research_gate_pass_channels": gate_channels,
        "baseline_result": baseline,
        "next_step": "Read all 34 channels together, eliminate weak direction/entry/RR/management routes, and avoid another upload for small isolated changes.",
    }
    request_path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))
    identity = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "request_sha256": file_sha256(request_path),
        "v102_engine_sha256": file_sha256(ROOT / "_v102_engine.py"),
        "v103_engine_sha256": file_sha256(ROOT / "_v103_engine.py"),
        "v104_engine_sha256": file_sha256(ROOT / "_v104_engine.py"),
        "v105_engine_sha256": file_sha256(ROOT / "_v105_engine.py"),
        "v106_engine_sha256": file_sha256(ROOT / "_v106_engine.py"),
        "v108_engine_sha256": file_sha256(ROOT / "_v108_engine.py"),
        "v109_engine_sha256": file_sha256(ROOT / "_v109_engine.py"),
        "v110_engine_sha256": file_sha256(ROOT / "_v110_engine.py"),
        "engine_sha256": file_sha256(ROOT / "_v111_engine.py"),
        "runner_sha256": file_sha256(ROOT / "autonomous_backtest_v10_11.py"),
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
    print("V111_REAL_BACKTEST_COMPLETE")


if __name__ == "__main__":
    main()
