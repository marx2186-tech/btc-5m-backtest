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

import _v109_engine as engine


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v10_9"
DEFAULT_REQUEST = ROOT / "request.v10_9.json"


def load_request() -> dict[str, Any]:
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))
    if not path.exists():
        raise FileNotFoundError(f"Missing request file: {path}")
    request = json.loads(path.read_text(encoding="utf-8"))
    validate_request(request)
    return request


def validate_request(request: dict[str, Any]) -> None:
    release = request["release"]
    if release["version"] != "10.9":
        raise ValueError("V10.9 requires release.version=10.9")
    if request["symbol"].upper() != "BTCUSDT" or request["source_interval"] != "5m":
        raise ValueError("V10.9 is frozen to BTCUSDT official 5m source data")
    if request["start_month"] != "2025-01" or request["end_month"] != "2026-06":
        raise ValueError("V10.9 must use 2025 warmup and end at 2026-06")
    if request["evaluation_window"]["start_utc"] != "2026-01-01T00:00:00Z":
        raise ValueError("V10.9 evaluation start is fixed")
    if request["evaluation_window"]["end_utc"] != "2026-06-30T23:59:59Z":
        raise ValueError("V10.9 evaluation end is fixed")
    if request["channel_count"] != 3 or len(request["channels"]) != 3:
        raise ValueError("V10.9 requires exactly three channels")
    expected_ids = [spec.channel_id for spec in engine.CHANNELS]
    actual_ids = [row["channel_id"] for row in request["channels"]]
    if actual_ids != expected_ids:
        raise ValueError(f"Channel list mismatch: expected={expected_ids}, actual={actual_ids}")
    if release["parameter_optimization_enabled"] or release["winner_selection_enabled"]:
        raise ValueError("V10.9 is a predeclared diagnostic, not a parameter search")
    if not release["one_setup_per_cycle"] or not release["shadow_management_does_not_change_signals"]:
        raise ValueError("V10.9 requires one setup per cycle and identical shadow signals")

    p = request["state_machine_parameters"]
    cycle = p["one_hour_cycle"]
    trigger = p["fifteen_minute_trigger"]
    execution = p["execution"]
    shadow = p["shadow_management"]
    fixed_values = {
        "impulse_breakout_lookback_hours": 12,
        "maximum_hours_impulse_to_pullback": 18,
        "maximum_hours_pullback_to_confirmation": 6,
        "cycle_cooldown_hours": 8,
        "maximum_cycles_per_4h_environment_leg": 2,
    }
    for key, expected in fixed_values.items():
        if int(cycle[key]) != expected:
            raise ValueError(f"Frozen state-machine integer mismatch: {key}")
    if float(cycle["minimum_pullback_retracement_atr"]) != 0.50:
        raise ValueError("Minimum pullback retracement must remain 0.50 ATR")
    if float(cycle["maximum_confirmation_extension_ema20_atr"]) != 0.90:
        raise ValueError("Confirmation extension cap must remain 0.90 ATR")
    if int(trigger["minimum_delay_after_confirmation_minutes"]) != 15:
        raise ValueError("15m trigger must wait one closed bar")
    if float(trigger["setup_validity_hours"]) != 3.0:
        raise ValueError("Setup validity must remain three hours")
    if float(execution["stop_atr_multiple"]) != 1.25 or float(execution["reward_risk"]) != 2.0:
        raise ValueError("Execution must remain 1.25 ATR stop and 2R target")
    if int(execution["max_holding_hours"]) != 48:
        raise ValueError("Maximum holding must remain 48 hours")
    if float(shadow["break_even_activation_R"]) != 1.0:
        raise ValueError("Shadow break-even activation must remain 1R")
    if shadow["activation_effective_from"] != "NEXT_COMPLETE_5M_BAR":
        raise ValueError("Break-even activation must be delayed to next 5m bar")

    execution_costs = request["execution"]
    if float(execution_costs["fee_rate_per_side"]) != 0.0005:
        raise ValueError("fee_rate_per_side must remain 0.0005")
    if float(execution_costs["tick_size"]) * int(execution_costs["slippage_ticks_per_fill"]) != 0.2:
        raise ValueError("slippage must remain 0.2 USDT per fill")
    if execution_costs["same_bar_stop_target_rule"] != "STOP_FIRST_CONSERVATIVE":
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
    losses = benchmark["loss_diagnostics"]
    management = benchmark["management_comparison"]
    lines = [
        "# BTCUSDT V10.9 严格推动—回踩状态机回测报告",
        "",
        "- 统计区间：**2026-01-01至2026-06-30 UTC**；2025年只用于指标预热。",
        "- 原1小时V10.6组合作为不变基准。",
        "- 主实验必须依次完成：4H趋势腿 → 1H明确推动 → 真实回踩 → 收盘确认 → 首个严格15M结构突破。",
        "- 每个推动—回踩周期最多产生一个确认和一个15M触发；每个4H环境腿最多两个周期。",
        "- 影子管理通道与主实验使用完全相同的信号，只比较达到1R后从下一根5M开始启用保本的影响。",
        "- 本区间已经查看，只能诊断，不能据此自动宣称实盘合格，也不得运行参数网格。",
        "",
        "## 总体结果",
        "",
        "| 通道 | 交易 | 胜/负 | 胜率 | 实际盈亏比 | PF | 净R | 最大回撤 | 删最佳10%后 | 门槛通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {int(row.trades)} | {int(row.wins)}/{int(row.losses)} | "
            f"{row.win_rate:.2%} | {row.avg_win_loss_ratio:.3f} | {row.profit_factor:.3f} | "
            f"{row.net_R:.3f} | {row.max_drawdown_R:.3f} | {row.best_10pct_removed_net_R:.3f} | "
            f"{int(row.checks_passed)}/{int(row.checks_total)} |"
        )

    lines.extend([
        "",
        "## 状态机漏斗",
        "",
        "| 阶段 | 多头 | 空头 |",
        "|---|---:|---:|",
    ])
    for row in funnel.itertuples(index=False):
        lines.append(f"| {row.stage} | {int(row.long_count)} | {int(row.short_count)} |")

    lines.extend([
        "",
        "## 多空方向贡献",
        "",
        "| 通道 | 方向 | 交易 | 胜率 | PF | 净R | 回撤R |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in direction.itertuples(index=False):
        label = "多头" if int(row.direction) == 1 else "空头"
        lines.append(
            f"| {row.channel_label} | {label} | {int(row.trades)} | {row.win_rate:.2%} | "
            f"{row.profit_factor:.3f} | {row.net_R:.3f} | {row.max_drawdown_R:.3f} |"
        )

    lines.extend([
        "",
        "## 固定管理与1R保本影子对照",
        "",
        "| 指标 | 固定止损止盈 | 1R后下一根5M保本 | 影子-固定 |",
        "|---|---:|---:|---:|",
    ])
    if management.empty:
        lines.append("| 无交易 | 0 | 0 | 0 |")
    else:
        for row in management.itertuples(index=False):
            lines.append(
                f"| {row.metric} | {row.fixed_value:.4f} | {row.break_even_shadow_value:.4f} | "
                f"{row.delta_shadow_minus_fixed:+.4f} |"
            )

    lines.extend([
        "",
        "## 亏损类型",
        "",
        "| 通道 | 类型 | 数量 | 占亏损 | 平均净R | 平均MFE | 平均MAE |",
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
    baseline = by_id.loc["baseline_1h_shared_v10_6"]
    primary = by_id.loc["strict_state_4h_1h_15m_fixed_rr2_0"]
    shadow = by_id.loc["strict_state_4h_1h_15m_be1_shadow_rr2_0"]
    lines.extend([
        "",
        "## 当前判断",
        "",
        f"- 原1小时基准：{int(baseline.trades)}笔，胜率{baseline.win_rate:.2%}，PF {baseline.profit_factor:.3f}，净R {baseline.net_R:.3f}。",
        f"- 严格状态机固定管理：{int(primary.trades)}笔，胜率{primary.win_rate:.2%}，PF {primary.profit_factor:.3f}，净R {primary.net_R:.3f}。",
        f"- 同信号1R保本影子：{int(shadow.trades)}笔，胜率{shadow.win_rate:.2%}，PF {shadow.profit_factor:.3f}，净R {shadow.net_R:.3f}。",
        "- 先判断状态机是否把V10.8的93笔噪声压缩成有限、可解释的完整周期，再单独判断保本管理是否改善回吐。",
        "- 若交易仍过多，说明状态生命周期仍有错误；若交易极少，则说明条件过严，但不得在同一窗口继续网格调参。",
        "",
        "重点文件：`state_machine_funnel.csv`、`state_machine_event_ledger.csv`、`signal_counts.csv`、",
        "`channel_summary.csv`、`management_comparison.csv`、`loss_diagnostics.csv`、`trade_ledger.csv`。",
    ])
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pipeline_smoke(request: dict[str, Any]) -> None:
    raw = engine.synthetic_5m_data(220_000, seed=20261009)
    result = engine.run_benchmark(raw, request)
    if len(result["summary"]) != 3:
        raise AssertionError("Pipeline smoke did not return three channels")
    if set(result["signals"]) != {spec.channel_id for spec in engine.CHANNELS}:
        raise AssertionError("Pipeline smoke signal channel mismatch")
    fixed = result["signals"]["strict_state_4h_1h_15m_fixed_rr2_0"]
    shadow = result["signals"]["strict_state_4h_1h_15m_be1_shadow_rr2_0"]
    if not fixed[["signal_time", "direction", "setup_id"]].reset_index(drop=True).equals(
        shadow[["signal_time", "direction", "setup_id"]].reset_index(drop=True)
    ):
        raise AssertionError("Fixed and shadow signal ledgers differ")
    print(json.dumps({"channels": 3, "trades": int(len(result["trades"])), "strict_signals": int(len(fixed))}, ensure_ascii=False, indent=2))
    print("V109_PIPELINE_SMOKE_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT V10.9 strict impulse-pullback state-machine diagnostic")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pipeline-smoke", action="store_true")
    args = parser.parse_args()
    request = load_request()

    if args.self_test:
        engine.self_test(request)
        print("V109_SELF_TEST_OK")
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
    benchmark["setup_type_summary"].to_csv(RESULTS / "setup_type_summary.csv", index=False)
    benchmark["exit_reason_summary"].to_csv(RESULTS / "exit_reason_summary.csv", index=False)
    benchmark["loss_diagnostics"].to_csv(RESULTS / "loss_diagnostics.csv", index=False)
    benchmark["signal_funnel"].to_csv(RESULTS / "state_machine_funnel.csv", index=False)
    benchmark["state_machine_events"].to_csv(RESULTS / "state_machine_event_ledger.csv", index=False)
    benchmark["management_comparison"].to_csv(RESULTS / "management_comparison.csv", index=False)
    pd.DataFrame(benchmark["signal_audit"]["channels"]).to_csv(RESULTS / "signal_counts.csv", index=False)
    ledger.to_csv(RESULTS / "trade_ledger.csv", index=False)

    qualification = benchmark["qualification"]
    gate_channels = [channel_id for channel_id, audit in qualification.items() if audit["historical_research_gate_pass"]]
    baseline = summary.loc[summary["channel_id"] == "baseline_1h_shared_v10_6"].iloc[0].to_dict()
    primary = summary.loc[summary["channel_id"] == "strict_state_4h_1h_15m_fixed_rr2_0"].iloc[0].to_dict()
    shadow = summary.loc[summary["channel_id"] == "strict_state_4h_1h_15m_be1_shadow_rr2_0"].iloc[0].to_dict()
    status = {
        "engine": "BTCUSDT V10.9 strict 4h-leg / 1h impulse-pullback state machine / 15m trigger",
        "release_version": "10.9",
        "historical_diagnostic_only": True,
        "qualified_for_live_trading": False,
        "parameter_optimization_enabled": False,
        "winner_selection_enabled": False,
        "evaluation_start_utc": request["evaluation_window"]["start_utc"],
        "evaluation_end_utc": request["evaluation_window"]["end_utc"],
        "warmup_start_utc": request["warmup_window"]["start_utc"],
        "channel_count": 3,
        "state_machine_audit": benchmark["signal_audit"]["state_machine"],
        "historical_research_gate_pass_channels": gate_channels,
        "baseline_result": baseline,
        "primary_state_machine_result": primary,
        "shadow_break_even_result": shadow,
        "next_step": "Inspect whether the finite state machine reduces repeated pullback noise and whether the predeclared 1R break-even shadow improves giveback without changing signals. Do not grid-tune on this viewed window.",
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
        "engine_sha256": file_sha256(ROOT / "_v109_engine.py"),
        "runner_sha256": file_sha256(ROOT / "autonomous_backtest_v10_9.py"),
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
    print("V109_REAL_BACKTEST_COMPLETE")


if __name__ == "__main__":
    main()
