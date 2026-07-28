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

import _v102_engine as engine

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v10_2"
DEFAULT_REQUEST = ROOT / "request.v10_2.json"


def load_request() -> dict[str, Any]:
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))
    if not path.exists():
        raise FileNotFoundError(f"Missing request file: {path}")
    request = json.loads(path.read_text(encoding="utf-8"))
    validate_request(request)
    return request


def validate_request(request: dict[str, Any]) -> None:
    release = request["release"]
    if release["version"] != "10.2":
        raise ValueError("V10.2 requires release.version=10.2")
    if request["symbol"].upper() != "BTCUSDT":
        raise ValueError("V10.2 is frozen to BTCUSDT")
    if request["source_interval"] != "5m":
        raise ValueError("V10.2 must aggregate from official 5m data")
    if request["signal_timeframe_minutes"] != 60:
        raise ValueError("V10.2 only permits the 60-minute signal timeframe")
    if request["channel_count"] != 8:
        raise ValueError("V10.2 requires exactly 8 independent hourly channels")
    if request["baseline_channel_count"] != 2 or request["optimized_channel_count"] != 6:
        raise ValueError("V10.2 requires 2 frozen baselines and 6 robustness candidates")
    if not release["trend_pullback_only"] or not release["range_reversal_removed"]:
        raise ValueError("V10.2 must remain trend-pullback-only")
    execution = request["execution"]
    if float(execution["fee_rate_per_side"]) != 0.0005:
        raise ValueError("fee_rate_per_side must remain 0.0005")
    if float(execution["tick_size"]) * int(execution["slippage_ticks_per_fill"]) != 0.2:
        raise ValueError("slippage must remain 0.2 USDT per fill")
    if execution["same_bar_stop_target_rule"] != "STOP_FIRST_CONSERVATIVE":
        raise ValueError("same-bar ambiguity must use conservative stop-first")
    if execution["reward_risk_grid"] != [2.5, 3.0]:
        raise ValueError("V10.2 reward-risk grid must remain [2.5, 3.0]")
    if not request["historical_diagnostic_only"]:
        raise ValueError("Viewed historical data must remain diagnostic only")
    if request["winner_selection_enabled"]:
        raise ValueError("Automatic winner selection must remain disabled")
    expected_channels = [spec.channel_id for spec in engine.CHANNELS]
    actual_channels = [row["channel_id"] for row in request["channels"]]
    if actual_channels != expected_channels:
        raise ValueError(f"Channel list mismatch: expected={expected_channels}, actual={actual_channels}")
    if any(int(row["timeframe_minutes"]) != 60 for row in request["channels"]):
        raise ValueError("Every V10.2 channel must use the 60-minute signal timeframe")
    if any(row["family"] != "trend_pullback" for row in request["channels"]):
        raise ValueError("Every V10.2 channel must belong to trend_pullback")
    if sum(bool(row["is_baseline"]) for row in request["channels"]) != 2:
        raise ValueError("Exactly two channels must be frozen V10.1 baselines")


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
    leader: dict[str, Any] | None,
) -> None:
    lines = [
        "# BTCUSDT V10.2 单一1小时趋势回踩稳健性报告",
        "",
        f"- 数据窗口：**{request['start_month']}至{request['end_month']} UTC**。",
        "- 只保留1小时趋势回踩；区间反转与二次确认分支均已停止。",
        "- RR2.5与RR3.0的V10.1质量过滤结果作为冻结基线，不改写其信号规则。",
        "- 新增防趋势衰竭、均衡趋势区间、趋势持续性三类过滤，每类分别测试RR2.5和RR3.0。",
        "- 重点不是追求历史总净R最高，而是让2026年4月至6月、删除最佳10%交易和滚动季度稳定性同时改善。",
        "- 所有信号只使用已收盘1小时K线，下一根5分钟K线开盘执行，手续费与滑点保持不变。",
        "- 本数据已经查看，只能用于历史诊断，不是盲测，也不能自动宣称实盘合格。",
        "",
        "## 总体排名",
        "",
        "| 排名 | 通道 | 交易 | 胜率 | 实际盈亏比 | PF | 净R | 回撤R | Q2净R | Q2 PF | 删最佳10%后 | 最差滚动季度 | 候选 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for index, row in summary.iterrows():
        lines.append(
            f"| {index + 1} | {row['channel_label']} | {int(row['trades'])} | {row['win_rate']:.2%} | "
            f"{row['avg_win_loss_ratio']:.3f} | {row['profit_factor']:.3f} | {row['net_R']:.3f} | "
            f"{row['max_drawdown_R']:.3f} | {row['historical_test_net_R']:.3f} | "
            f"{row['historical_test_profit_factor']:.3f} | {row['best_10pct_removed_net_R']:.3f} | "
            f"{row['worst_rolling_quarter_net_R']:.3f} | {'是' if bool(row['research_candidate']) else '否'} |"
        )

    lines.extend(
        [
            "",
            "## 与同RR的V10.1冻结基线对照",
            "",
            "| 通道 | 信号保留率 | Q2净R变化 | 删最佳10%变化 | 总净R变化 | 回撤变化 | 三项稳健目标改善 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in comparison.loc[~comparison["is_baseline"]].itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {row.retention_rate_vs_baseline:.2%} | "
            f"{row.delta_historical_test_net_R:+.3f} | {row.delta_best_10pct_removed_net_R:+.3f} | "
            f"{row.delta_net_R:+.3f} | {row.delta_max_drawdown_R:+.3f} | "
            f"{'是' if bool(row.improves_primary_robustness_goals) else '否'} |"
        )

    lines.extend(
        [
            "",
            "## 三阶段表现",
            "",
            "| 通道 | 阶段 | 交易 | 胜率 | PF | 净R | 回撤R | 期望R |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in phase_summary.itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {row.phase} | {int(row.trades)} | {row.win_rate:.2%} | "
            f"{row.profit_factor:.3f} | {row.net_R:.3f} | {row.max_drawdown_R:.3f} | {row.expectancy_R:.3f} |"
        )

    lines.extend(["", "## 当前结论", ""])
    if leader is None:
        lines.append("没有任何非基线通道生成有效交易。")
    else:
        lines.append(
            f"- 当前非基线排名第一：**{leader['channel_label']}**，交易{int(leader['trades'])}笔，"
            f"胜率{float(leader['win_rate']):.2%}，PF {float(leader['profit_factor']):.3f}，"
            f"总净R {float(leader['net_R']):.3f}，Q2净R {float(leader['historical_test_net_R']):.3f}。"
        )
    improved = comparison.loc[(~comparison["is_baseline"]) & comparison["improves_primary_robustness_goals"]]
    if improved.empty:
        lines.append("- 没有通道同时改善Q2、删除最佳10%交易后的结果和最大回撤。")
    else:
        lines.append("- 同时改善三项稳健目标：" + "、".join(improved["channel_label"].tolist()) + "。")
    lines.extend(
        [
            "- 只有总样本、Q2、滚动季度、尾部稳健性和三阶段全部通过时，才标记为历史研究候选。",
            "- 即使成为历史研究候选，也必须冻结后等待未来未见数据验证。",
            "",
            "重点文件：`channel_summary.csv`、`baseline_comparison.csv`、`direction_summary.csv`、"
            "`rolling_quarter_summary.csv`、`historical_test_trade_diagnostics.csv`、`robustness_audit.json`。",
        ]
    )
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pipeline_smoke(request: dict[str, Any]) -> None:
    synthetic = engine.synthetic_5m_data(90_000, seed=20261002)
    result = engine.run_benchmark(synthetic, request)
    if len(result["summary"]) != 8:
        raise AssertionError("Pipeline smoke did not return 8 hourly channels")
    if len(result["baseline_comparison"]) != 8:
        raise AssertionError("Pipeline smoke did not create complete baseline comparison")
    if len(result["phases"]) != 24:
        raise AssertionError("Pipeline smoke did not create 3 phases for every channel")
    if result["rolling_quarters"].empty:
        raise AssertionError("Pipeline smoke did not create rolling-quarter diagnostics")
    if result["trades"].empty:
        raise AssertionError("Pipeline smoke produced no trades")
    print(
        json.dumps(
            {
                "channels": int(len(result["summary"])),
                "trades": int(len(result["trades"])),
                "leader": result["research_leader"]["channel_id"] if result["research_leader"] else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("V102_PIPELINE_SMOKE_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT V10.2 hourly trend-pullback regime robustness")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pipeline-smoke", action="store_true")
    args = parser.parse_args()
    request = load_request()

    if args.self_test:
        engine.self_test(request)
        print("V102_SELF_TEST_OK")
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
    ledger = benchmark["trades"]
    comparison = benchmark["baseline_comparison"]
    directions = benchmark["directions"]
    rolling_quarters = benchmark["rolling_quarters"]
    leader = benchmark["research_leader"]

    summary.to_csv(RESULTS / "channel_summary.csv", index=False)
    comparison.to_csv(RESULTS / "baseline_comparison.csv", index=False)
    phases.to_csv(RESULTS / "channel_phase_summary.csv", index=False)
    monthly.to_csv(RESULTS / "channel_monthly_summary.csv", index=False)
    directions.to_csv(RESULTS / "direction_summary.csv", index=False)
    rolling_quarters.to_csv(RESULTS / "rolling_quarter_summary.csv", index=False)
    ledger.to_csv(RESULTS / "trade_ledger.csv", index=False)
    historical = ledger.loc[ledger["phase"] == "historical_test"].copy() if not ledger.empty else ledger.copy()
    historical.to_csv(RESULTS / "historical_test_trade_diagnostics.csv", index=False)
    pd.DataFrame(benchmark["signal_audit"]["channels"]).to_csv(RESULTS / "signal_counts.csv", index=False)

    qualification = benchmark["qualification"]
    candidate_channels = [channel_id for channel_id, audit in qualification.items() if audit["research_candidate"]]
    robustness_improvements = comparison.loc[
        (~comparison["is_baseline"]) & comparison["improves_primary_robustness_goals"]
    ]["channel_id"].tolist()
    status = {
        "engine": "BTCUSDT one-hour trend-pullback regime robustness V10.2",
        "release_version": "10.2",
        "historical_diagnostic_only": True,
        "qualified_for_live_trading": False,
        "signal_timeframe_minutes": 60,
        "strategy_family": "trend_pullback_only",
        "channel_count": 8,
        "baseline_channel_count": 2,
        "optimized_channel_count": 6,
        "reward_risk_grid": [2.5, 3.0],
        "research_candidate_count": len(candidate_channels),
        "research_candidate_channels": candidate_channels,
        "channels_improving_q2_tail_and_drawdown": robustness_improvements,
        "research_leader": leader,
        "winner_selection_enabled": False,
        "next_step": "Freeze only a channel that passes recent-period, tail-removal and rolling-quarter robustness; then wait for future unseen data.",
    }
    benchmark_summary = {
        "data_window": {"start_month": request["start_month"], "end_month": request["end_month"]},
        "signal_timeframe_minutes": 60,
        "strategy_family": "trend_pullback_only",
        "channels": summary.to_dict(orient="records"),
        "baseline_comparison": comparison.to_dict(orient="records"),
        "research_leader": leader,
        "research_candidate_channels": candidate_channels,
        "live_qualification": False,
        "winner_decision": "HISTORICAL_HOURLY_ROBUSTNESS_ONLY_NO_AUTOMATIC_WINNER",
    }
    strategy_spec = {
        "channels": request["channels"],
        "frozen_v10_1_quality_profile": request["frozen_v10_1_quality_profile"],
        "robustness_profiles": request["robustness_profiles"],
        "execution": request["execution"],
        "no_lookahead_rules": request["no_lookahead_rules"],
        "research_candidate_thresholds": request["research_candidate_thresholds"],
    }
    identity = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "request_sha256": file_sha256(Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))),
        "engine_sha256": file_sha256(ROOT / "_v102_engine.py"),
        "runner_sha256": file_sha256(ROOT / "autonomous_backtest_v10_2.py"),
    }

    json_dump(RESULTS / "status.json", status)
    json_dump(RESULTS / "benchmark_summary.json", benchmark_summary)
    json_dump(RESULTS / "data_audit.json", data_audit)
    json_dump(RESULTS / "resample_and_signal_audit.json", benchmark["signal_audit"])
    json_dump(RESULTS / "robustness_audit.json", qualification)
    json_dump(RESULTS / "strategy_spec.json", strategy_spec)
    json_dump(RESULTS / "run_identity.json", identity)
    (RESULTS / "run_identity.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in identity.items()) + "\n", encoding="utf-8"
    )
    write_report(request, summary, phases, comparison, leader)

    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    print("V102_REAL_BACKTEST_COMPLETE")


if __name__ == "__main__":
    main()
