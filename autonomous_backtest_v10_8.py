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

import _v108_engine as engine


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v10_8"
DEFAULT_REQUEST = ROOT / "request.v10_8.json"


def load_request() -> dict[str, Any]:
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))
    if not path.exists():
        raise FileNotFoundError(f"Missing request file: {path}")
    request = json.loads(path.read_text(encoding="utf-8"))
    validate_request(request)
    return request


def validate_request(request: dict[str, Any]) -> None:
    release = request["release"]
    if release["version"] != "10.8":
        raise ValueError("V10.8 requires release.version=10.8")
    if request["symbol"].upper() != "BTCUSDT" or request["source_interval"] != "5m":
        raise ValueError("V10.8 is frozen to BTCUSDT official 5m source data")
    if request["start_month"] != "2025-01" or request["end_month"] != "2026-06":
        raise ValueError("V10.8 must use 2025 warmup and end at 2026-06")
    if request["evaluation_window"]["start_utc"] != "2026-01-01T00:00:00Z":
        raise ValueError("V10.8 evaluation start is fixed")
    if request["evaluation_window"]["end_utc"] != "2026-06-30T23:59:59Z":
        raise ValueError("V10.8 evaluation end is fixed")
    if request["channel_count"] != 3 or len(request["channels"]) != 3:
        raise ValueError("V10.8 requires exactly three comparison channels")
    expected_ids = [spec.channel_id for spec in engine.CHANNELS]
    actual_ids = [row["channel_id"] for row in request["channels"]]
    if actual_ids != expected_ids:
        raise ValueError(f"Channel list mismatch: {actual_ids}")
    if release["parameter_optimization_enabled"] or release["winner_selection_enabled"]:
        raise ValueError("V10.8 is a predeclared diagnostic, not a parameter search")

    mtf = request["multi_timeframe_parameters"]
    if float(mtf["four_hour_environment"]["minimum_adx14"]) != 18.0:
        raise ValueError("4h ADX threshold must remain 18")
    if float(mtf["one_hour_setup"]["maximum_price_extension_ema20_atr"]) != 1.20:
        raise ValueError("1h extension cap must remain 1.20 ATR")
    if int(mtf["fifteen_minute_trigger"]["setup_validity_hours"]) != 4:
        raise ValueError("15m setup validity must remain four hours")
    if int(mtf["fifteen_minute_trigger"]["minimum_delay_after_1h_setup_minutes"]) != 15:
        raise ValueError("15m trigger must wait at least one closed bar after the 1h setup")
    if float(mtf["execution"]["stop_atr_multiple"]) != 1.25:
        raise ValueError("MTF stop must remain 1.25x aligned 1h ATR")
    if float(mtf["execution"]["reward_risk"]) != 2.0:
        raise ValueError("MTF reward-risk must remain 2.0")
    if int(mtf["execution"]["max_holding_hours"]) != 48:
        raise ValueError("MTF max holding must remain 48 hours")

    execution = request["execution"]
    if float(execution["fee_rate_per_side"]) != 0.0005:
        raise ValueError("fee_rate_per_side must remain 0.0005")
    if float(execution["tick_size"]) * int(execution["slippage_ticks_per_fill"]) != 0.2:
        raise ValueError("slippage must remain 0.2 USDT per fill")
    if execution["same_bar_stop_target_rule"] != "STOP_FIRST_CONSERVATIVE":
        raise ValueError("same-bar ambiguity must remain stop-first")


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


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def write_report(request: dict[str, Any], benchmark: dict[str, Any]) -> None:
    summary = benchmark["summary"]
    direction = benchmark["direction_summary"]
    funnel = benchmark["signal_funnel"]
    losses = benchmark["loss_diagnostics"]
    lines = [
        "# BTCUSDT V10.8 多周期精确入场回测报告",
        "",
        "- 统计区间：**2026-01-01至2026-06-30 UTC**。",
        "- 2025年数据只负责4H/1H/15M指标预热，不计入交易和收益。",
        "- 基准通道完整保留V10.6原1小时共享仓位组合与RR2.5。",
        "- 实验通道采用4小时趋势环境、1小时回踩结束/重新突破、15分钟精确触发。",
        "- 实验通道统一使用1小时ATR×1.25止损、2R目标、最长48小时、4小时反向环境完整退出。",
        "- 本区间已查看，只能用于诊断；不进行参数搜索，不自动宣布实盘合格。",
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
        "## 信号漏斗",
        "",
        "| 阶段 | 多头数量 | 空头数量 |",
        "|---|---:|---:|",
    ])
    for row in funnel.itertuples(index=False):
        lines.append(f"| {row.stage} | {int(row.long_count)} | {int(row.short_count)} |")

    lines.extend([
        "",
        "## 亏损类型",
        "",
        "| 通道 | 类型 | 数量 | 占该通道亏损 | 平均净R | 平均MFE | 平均MAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    if losses.empty:
        lines.append("| 无 | 无亏损交易或无交易 | 0 | 0.00% | 0.000 | 0.000 | 0.000 |")
    else:
        for row in losses.itertuples(index=False):
            lines.append(
                f"| {row.channel_label} | {row.loss_class} | {int(row.trades)} | {row.share_of_channel_losses:.2%} | "
                f"{row.avg_net_R:.3f} | {row.avg_mfe_R:.3f} | {row.avg_mae_R:.3f} |"
            )

    by_id = summary.set_index("channel_id")
    baseline = by_id.loc["baseline_1h_shared_v10_6"]
    precision = by_id.loc["mtf_4h_1h_15m_precision_rr2_0"]
    direct = by_id.loc["mtf_4h_1h_direct_rr2_0"]
    lines.extend([
        "",
        "## 当前判断",
        "",
        f"- 原1小时基准：{int(baseline.trades)}笔，胜率{baseline.win_rate:.2%}，PF {baseline.profit_factor:.3f}，净R {baseline.net_R:.3f}。",
        f"- 4H+1H直接成交：{int(direct.trades)}笔，胜率{direct.win_rate:.2%}，PF {direct.profit_factor:.3f}，净R {direct.net_R:.3f}。",
        f"- 4H+1H+15M精确成交：{int(precision.trades)}笔，胜率{precision.win_rate:.2%}，PF {precision.profit_factor:.3f}，净R {precision.net_R:.3f}。",
        "- 重点不是单独看最高收益，而是判断15分钟确认是否在增加或保持交易数量的同时，改善快速反转亏损、胜率、PF和回撤。",
        "- 即使某实验通道达到门槛，也仍然是2026上半年历史诊断，必须再做独立时间段验证。",
        "",
        "重点文件：`channel_summary.csv`、`direction_summary.csv`、`signal_funnel.csv`、`setup_type_summary.csv`、",
        "`loss_diagnostics.csv`、`trade_ledger.csv`、`qualification_audit.json`、`strategy_spec.json`。",
    ])
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pipeline_smoke(request: dict[str, Any]) -> None:
    raw = engine.synthetic_5m_data(160_000, seed=20261008)
    result = engine.run_benchmark(raw, request)
    if len(result["summary"]) != 3:
        raise AssertionError("Pipeline smoke did not return three channels")
    if set(result["signals"]) != {spec.channel_id for spec in engine.CHANNELS}:
        raise AssertionError("Pipeline smoke signal channel mismatch")
    if result["signal_funnel"].empty:
        raise AssertionError("Pipeline smoke did not produce the signal funnel")
    print(json.dumps({"channels": 3, "trades": int(len(result["trades"]))}, ensure_ascii=False, indent=2))
    print("V108_PIPELINE_SMOKE_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT V10.8 4H/1H/15M precision-entry diagnostic")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pipeline-smoke", action="store_true")
    args = parser.parse_args()
    request = load_request()

    if args.self_test:
        engine.self_test(request)
        print("V108_SELF_TEST_OK")
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
    benchmark["signal_funnel"].to_csv(RESULTS / "signal_funnel.csv", index=False)
    pd.DataFrame(benchmark["signal_audit"]["channels"]).to_csv(RESULTS / "signal_counts.csv", index=False)
    ledger.to_csv(RESULTS / "trade_ledger.csv", index=False)

    qualification = benchmark["qualification"]
    gate_channels = [channel_id for channel_id, audit in qualification.items() if audit["historical_research_gate_pass"]]
    primary = summary.loc[summary["channel_id"] == "mtf_4h_1h_15m_precision_rr2_0"].iloc[0].to_dict()
    baseline = summary.loc[summary["channel_id"] == "baseline_1h_shared_v10_6"].iloc[0].to_dict()
    status = {
        "engine": "BTCUSDT V10.8 4h environment + 1h setup + 15m precision entry",
        "release_version": "10.8",
        "historical_diagnostic_only": True,
        "qualified_for_live_trading": False,
        "parameter_optimization_enabled": False,
        "winner_selection_enabled": False,
        "evaluation_start_utc": request["evaluation_window"]["start_utc"],
        "evaluation_end_utc": request["evaluation_window"]["end_utc"],
        "warmup_start_utc": request["warmup_window"]["start_utc"],
        "channel_count": 3,
        "historical_research_gate_pass_channels": gate_channels,
        "baseline_result": baseline,
        "primary_experiment_result": primary,
        "next_step": "Review whether the 15m confirmation improves frequency and loss quality without damaging PF, actual win/loss ratio and drawdown; do not tune on the same six-month window.",
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
        "engine_sha256": file_sha256(ROOT / "_v108_engine.py"),
        "runner_sha256": file_sha256(ROOT / "autonomous_backtest_v10_8.py"),
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
    print("V108_REAL_BACKTEST_COMPLETE")


if __name__ == "__main__":
    main()
