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

import _v101_engine as engine

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v10_1"
DEFAULT_REQUEST = ROOT / "request.v10_1.json"


def load_request() -> dict[str, Any]:
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))
    if not path.exists():
        raise FileNotFoundError(f"Missing request file: {path}")
    request = json.loads(path.read_text(encoding="utf-8"))
    validate_request(request)
    return request


def validate_request(request: dict[str, Any]) -> None:
    release = request["release"]
    if release["version"] != "10.1":
        raise ValueError("V10.1 requires release.version=10.1")
    if request["symbol"].upper() != "BTCUSDT":
        raise ValueError("V10.1 is frozen to BTCUSDT")
    if request["source_interval"] != "5m":
        raise ValueError("V10.1 must aggregate from official 5m data")
    if request["signal_timeframe_minutes"] != 60:
        raise ValueError("V10.1 only permits the 60-minute signal timeframe")
    if request["channel_count"] != 14:
        raise ValueError("V10.1 requires exactly 14 independent hourly channels")
    execution = request["execution"]
    if float(execution["fee_rate_per_side"]) != 0.0005:
        raise ValueError("fee_rate_per_side must remain 0.0005")
    if float(execution["tick_size"]) * int(execution["slippage_ticks_per_fill"]) != 0.2:
        raise ValueError("slippage must remain 0.2 USDT per fill")
    if execution["same_bar_stop_target_rule"] != "STOP_FIRST_CONSERVATIVE":
        raise ValueError("same-bar ambiguity must use conservative stop-first")
    if not request["historical_diagnostic_only"]:
        raise ValueError("Viewed historical data must remain diagnostic only")
    if request["winner_selection_enabled"]:
        raise ValueError("Automatic winner selection must remain disabled")
    expected_channels = [spec.channel_id for spec in engine.CHANNELS]
    actual_channels = [row["channel_id"] for row in request["channels"]]
    if actual_channels != expected_channels:
        raise ValueError(f"Channel list mismatch: expected={expected_channels}, actual={actual_channels}")
    if any(int(row["timeframe_minutes"]) != 60 for row in request["channels"]):
        raise ValueError("Every V10.1 channel must use the 60-minute signal timeframe")
    if sorted(set(float(row["reward_risk"]) for row in request["channels"])) != [2.0, 2.5, 3.0]:
        raise ValueError("V10.1 reward-risk grid must remain [2.0, 2.5, 3.0]")


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
        "# BTCUSDT V10.1 单一1小时周期胜率与盈亏比优化报告",
        "",
        f"- 数据窗口：**{request['start_month']}至{request['end_month']} UTC**。",
        "- 只使用1小时信号周期；所有1小时K线均由完整官方5分钟K线聚合。",
        "- 保留V10.0趋势回踩与区间反转基线各一条，新增12条质量过滤/二次确认候选。",
        "- 候选固定测试目标盈亏比2.0、2.5、3.0；每条通道独立持仓，下一根5分钟K线开盘执行。",
        "- 目标是同时提高胜率与实际平均盈亏比，不通过取消手续费、降低滑点或使用未来数据美化结果。",
        "- 本数据已经查看，只能作为历史研究，禁止自动选胜者或直接宣称实盘合格。",
        "",
        "## 总体排名",
        "",
        "| 排名 | 通道 | 交易 | 胜率 | 实际盈亏比 | PF | 净R | 最大回撤R | 期望R | 删除最佳10%后净R | 候选 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for index, row in summary.iterrows():
        lines.append(
            f"| {index + 1} | {row['channel_label']} | {int(row['trades'])} | {row['win_rate']:.2%} | "
            f"{row['avg_win_loss_ratio']:.3f} | {row['profit_factor']:.3f} | {row['net_R']:.3f} | "
            f"{row['max_drawdown_R']:.3f} | {row['expectancy_R']:.3f} | {row['best_10pct_removed_net_R']:.3f} | "
            f"{'是' if bool(row['research_candidate']) else '否'} |"
        )

    improved = comparison.loc[
        (~comparison["is_baseline"])
        & comparison["improves_both_primary_goals"]
    ].sort_values(["delta_win_rate", "delta_avg_win_loss_ratio"], ascending=False)
    lines.extend(
        [
            "",
            "## 与V10.0一小时基线对照",
            "",
            "| 通道 | 胜率变化 | 实际盈亏比变化 | PF变化 | 净R变化 | 回撤变化R | 两项目标同时改善 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in comparison.loc[~comparison["is_baseline"]].itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {row.delta_win_rate:+.2%} | {row.delta_avg_win_loss_ratio:+.3f} | "
            f"{row.delta_profit_factor:+.3f} | {row.delta_net_R:+.3f} | {row.delta_max_drawdown_R:+.3f} | "
            f"{'是' if bool(row.improves_both_primary_goals) else '否'} |"
        )

    lines.extend(
        [
            "",
            "## 三阶段表现",
            "",
            "| 通道 | 阶段 | 交易 | 胜率 | 实际盈亏比 | PF | 净R | 最大回撤R | 期望R |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in phase_summary.itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {row.phase} | {int(row.trades)} | {row.win_rate:.2%} | "
            f"{row.avg_win_loss_ratio:.3f} | {row.profit_factor:.3f} | {row.net_R:.3f} | "
            f"{row.max_drawdown_R:.3f} | {row.expectancy_R:.3f} |"
        )

    lines.extend(["", "## 当前结论", ""])
    if leader is None:
        lines.append("没有任何优化通道生成有效交易。")
    else:
        lines.append(
            f"- 当前非基线排名第一：**{leader['channel_label']}**，交易{int(leader['trades'])}笔，"
            f"胜率{float(leader['win_rate']):.2%}，实际盈亏比{float(leader['avg_win_loss_ratio']):.3f}，"
            f"PF {float(leader['profit_factor']):.3f}，净R {float(leader['net_R']):.3f}。"
        )
    if improved.empty:
        lines.append("- 没有优化通道同时超过对应基线的胜率和实际平均盈亏比。")
    else:
        names = "、".join(improved["channel_label"].head(5).tolist())
        lines.append(f"- 同时改善胜率和实际盈亏比的通道：{names}。")
    lines.extend(
        [
            "- 排名不等于可实盘；仍必须通过PF、期望、回撤、跨月份、跨阶段和删除最佳交易后的稳健性门槛。",
            "- 即使出现历史研究候选，也必须冻结后等待未来未见数据验证。",
            "",
            "重点文件：`channel_summary.csv`、`baseline_comparison.csv`、`channel_phase_summary.csv`、`channel_monthly_summary.csv`、`trade_ledger.csv`、`robustness_audit.json`。",
        ]
    )
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pipeline_smoke(request: dict[str, Any]) -> None:
    synthetic = engine.synthetic_5m_data(80_000, seed=20261001)
    result = engine.run_benchmark(synthetic, request)
    if len(result["summary"]) != 14:
        raise AssertionError("Pipeline smoke did not return 14 hourly channels")
    if len(result["baseline_comparison"]) != 14:
        raise AssertionError("Pipeline smoke did not create complete baseline comparison")
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
    print("V101_PIPELINE_SMOKE_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT V10.1 hourly quality and reward-risk optimization")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pipeline-smoke", action="store_true")
    args = parser.parse_args()
    request = load_request()

    if args.self_test:
        engine.self_test(request)
        print("V101_SELF_TEST_OK")
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
    leader = benchmark["research_leader"]

    summary.to_csv(RESULTS / "channel_summary.csv", index=False)
    comparison.to_csv(RESULTS / "baseline_comparison.csv", index=False)
    phases.to_csv(RESULTS / "channel_phase_summary.csv", index=False)
    monthly.to_csv(RESULTS / "channel_monthly_summary.csv", index=False)
    ledger.to_csv(RESULTS / "trade_ledger.csv", index=False)
    pd.DataFrame(benchmark["signal_audit"]["channels"]).to_csv(RESULTS / "signal_counts.csv", index=False)

    qualification = benchmark["qualification"]
    candidate_channels = [channel_id for channel_id, audit in qualification.items() if audit["research_candidate"]]
    simultaneous_improvement = comparison.loc[
        (~comparison["is_baseline"])
        & comparison["improves_both_primary_goals"]
    ]["channel_id"].tolist()
    status = {
        "engine": "BTCUSDT one-hour quality and reward-risk optimization V10.1",
        "release_version": "10.1",
        "historical_diagnostic_only": True,
        "qualified_for_live_trading": False,
        "signal_timeframe_minutes": 60,
        "channel_count": 14,
        "baseline_channel_count": 2,
        "optimized_channel_count": 12,
        "reward_risk_grid": [2.0, 2.5, 3.0],
        "research_candidate_count": len(candidate_channels),
        "research_candidate_channels": candidate_channels,
        "channels_improving_win_rate_and_avg_win_loss_ratio": simultaneous_improvement,
        "research_leader": leader,
        "winner_selection_enabled": False,
        "next_step": "Only freeze a structurally robust hourly channel; then validate on future unseen data.",
    }
    benchmark_summary = {
        "data_window": {"start_month": request["start_month"], "end_month": request["end_month"]},
        "signal_timeframe_minutes": 60,
        "channels": summary.to_dict(orient="records"),
        "baseline_comparison": comparison.to_dict(orient="records"),
        "research_leader": leader,
        "research_candidate_channels": candidate_channels,
        "live_qualification": False,
        "winner_decision": "HISTORICAL_HOURLY_OPTIMIZATION_ONLY_NO_AUTOMATIC_WINNER",
    }
    strategy_spec = {
        "channels": request["channels"],
        "quality_profiles": request["quality_profiles"],
        "execution": request["execution"],
        "no_lookahead_rules": request["no_lookahead_rules"],
        "research_candidate_thresholds": request["research_candidate_thresholds"],
    }
    identity = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "request_sha256": file_sha256(Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))),
        "engine_sha256": file_sha256(ROOT / "_v101_engine.py"),
        "runner_sha256": file_sha256(ROOT / "autonomous_backtest_v10_1.py"),
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
    print("V101_REAL_BACKTEST_COMPLETE")


if __name__ == "__main__":
    main()
