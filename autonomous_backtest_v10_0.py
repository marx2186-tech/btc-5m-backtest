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

import _v100_engine as engine

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v10_0"
DEFAULT_REQUEST = ROOT / "request.v10_0.json"


def load_request() -> dict[str, Any]:
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))
    if not path.exists():
        raise FileNotFoundError(f"Missing request file: {path}")
    request = json.loads(path.read_text(encoding="utf-8"))
    validate_request(request)
    return request


def validate_request(request: dict[str, Any]) -> None:
    release = request["release"]
    if release["version"] != "10.0":
        raise ValueError("V10.0 requires release.version=10.0")
    if request["symbol"].upper() != "BTCUSDT":
        raise ValueError("V10.0 benchmark is frozen to BTCUSDT")
    if request["source_interval"] != "5m":
        raise ValueError("V10.0 must aggregate from official 5m data")
    if request["timeframes_minutes"] != [15, 30, 60]:
        raise ValueError("V10.0 requires 15m, 30m and 60m in parallel")
    if request["channel_count"] != 10:
        raise ValueError("V10.0 requires exactly 10 independent channels")
    execution = request["execution"]
    if float(execution["fee_rate_per_side"]) != 0.0005:
        raise ValueError("fee_rate_per_side must remain 0.0005")
    if float(execution["tick_size"]) * int(execution["slippage_ticks_per_fill"]) != 0.2:
        raise ValueError("slippage must remain 0.2 USDT per fill")
    if execution["same_bar_stop_target_rule"] != "STOP_FIRST_CONSERVATIVE":
        raise ValueError("same-bar ambiguity must use conservative stop-first")
    if not request["historical_diagnostic_only"]:
        raise ValueError("V10.0 data has already been viewed and must remain historical diagnostic only")
    if request["winner_selection_enabled"]:
        raise ValueError("Automatic winner selection must remain disabled")
    expected_channels = [spec.channel_id for spec in engine.CHANNELS]
    actual_channels = [row["channel_id"] for row in request["channels"]]
    if actual_channels != expected_channels:
        raise ValueError(f"Channel list mismatch: expected={expected_channels}, actual={actual_channels}")


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


def write_report(request: dict[str, Any], summary: pd.DataFrame, phase_summary: pd.DataFrame, leader: dict[str, Any] | None) -> None:
    lines = [
        "# BTCUSDT V10.0 多周期简单策略基准回测报告",
        "",
        f"- 数据窗口：**{request['start_month']}至{request['end_month']} UTC**。",
        "- 原始数据：Binance USDⓈ-M Futures官方5分钟月度归档，并逐文件校验SHA-256。",
        "- 三个周期15分钟、30分钟、1小时在同一次运行中并行回测；另有1小时趋势＋15分钟结构＋5分钟执行通道。",
        "- 10条通道互相独立，不共享仓位，不混合信号，不使用未收盘的高周期K线。",
        "- 这是已查看历史数据上的方向筛选，不是新盲测，也不能直接宣称实盘合格。",
        "",
        "## 总体排名",
        "",
        "| 排名 | 通道 | 交易 | 胜率 | 盈亏比 | PF | 净R | 最大回撤R | 期望R | 删除最佳10%后净R | 研究候选 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for index, row in summary.iterrows():
        lines.append(
            f"| {index + 1} | {row['channel_label']} | {int(row['trades'])} | {row['win_rate']:.2%} | "
            f"{row['avg_win_loss_ratio']:.3f} | {row['profit_factor']:.3f} | {row['net_R']:.3f} | "
            f"{row['max_drawdown_R']:.3f} | {row['expectancy_R']:.3f} | {row['best_10pct_removed_net_R']:.3f} | "
            f"{'是' if bool(row['research_candidate']) else '否'} |"
        )
    lines.extend(["", "## 三阶段表现", "", "| 通道 | 阶段 | 交易 | 胜率 | PF | 净R | 最大回撤R | 期望R |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in phase_summary.itertuples(index=False):
        lines.append(
            f"| {row.channel_label} | {row.phase} | {int(row.trades)} | {row.win_rate:.2%} | "
            f"{row.profit_factor:.3f} | {row.net_R:.3f} | {row.max_drawdown_R:.3f} | {row.expectancy_R:.3f} |"
        )
    lines.extend(["", "## 当前结论", ""])
    if leader is None:
        lines.append("没有任何通道生成有效交易。")
    else:
        lines.append(
            f"- 当前历史研究排名第一：**{leader['channel_label']}**，交易{int(leader['trades'])}笔，"
            f"胜率{float(leader['win_rate']):.2%}，PF {float(leader['profit_factor']):.3f}，"
            f"净R {float(leader['net_R']):.3f}，最大回撤{float(leader['max_drawdown_R']):.3f}R。"
        )
        lines.append("- 该排名只用于确定下一轮研究方向，不自动成为实盘策略。")
    lines.extend(
        [
            "- 只有同时达到交易样本、盈利因子、期望、回撤、跨月份与删除最佳交易稳健性门槛，才标记为研究候选。",
            "- 即使出现研究候选，仍需要冻结后使用未来数据做真正盲测。",
            "",
            "重点文件：`channel_summary.csv`、`channel_phase_summary.csv`、`channel_monthly_summary.csv`、`trade_ledger.csv`、`robustness_audit.json`。",
        ]
    )
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pipeline_smoke(request: dict[str, Any]) -> None:
    synthetic = engine.synthetic_5m_data(60_000, seed=20261000)
    result = engine.run_benchmark(synthetic, request)
    if len(result["summary"]) != 10:
        raise AssertionError("Pipeline smoke did not return 10 channels")
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
    print("V100_PIPELINE_SMOKE_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT V10.0 multi-timeframe simple-strategy benchmark")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pipeline-smoke", action="store_true")
    args = parser.parse_args()
    request = load_request()

    if args.self_test:
        engine.self_test(request)
        print("V100_SELF_TEST_OK")
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
    leader = benchmark["research_leader"]

    summary.to_csv(RESULTS / "channel_summary.csv", index=False)
    phases.to_csv(RESULTS / "channel_phase_summary.csv", index=False)
    monthly.to_csv(RESULTS / "channel_monthly_summary.csv", index=False)
    ledger.to_csv(RESULTS / "trade_ledger.csv", index=False)
    pd.DataFrame(benchmark["signal_audit"]["channels"]).to_csv(RESULTS / "signal_counts.csv", index=False)

    qualification = benchmark["qualification"]
    candidate_channels = [channel_id for channel_id, audit in qualification.items() if audit["research_candidate"]]
    status = {
        "engine": "BTCUSDT multi-timeframe simple-strategy benchmark V10.0",
        "release_version": "10.0",
        "historical_diagnostic_only": True,
        "qualified_for_live_trading": False,
        "timeframes_tested_minutes": [15, 30, 60],
        "channel_count": 10,
        "research_candidate_count": len(candidate_channels),
        "research_candidate_channels": candidate_channels,
        "research_leader": leader,
        "winner_selection_enabled": False,
        "next_step": "Freeze the best structurally robust channel and validate on future unseen data.",
    }
    benchmark_summary = {
        "data_window": {"start_month": request["start_month"], "end_month": request["end_month"]},
        "channels": summary.to_dict(orient="records"),
        "research_leader": leader,
        "research_candidate_channels": candidate_channels,
        "live_qualification": False,
        "winner_decision": "HISTORICAL_DIRECTIONAL_RESEARCH_ONLY",
    }
    strategy_spec = {
        "channels": request["channels"],
        "execution": request["execution"],
        "no_lookahead_rules": request["no_lookahead_rules"],
        "research_candidate_thresholds": request["research_candidate_thresholds"],
    }
    identity = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "request_sha256": file_sha256(Path(os.environ.get("BACKTEST_REQUEST_FILE", str(DEFAULT_REQUEST)))),
        "engine_sha256": file_sha256(ROOT / "_v100_engine.py"),
        "runner_sha256": file_sha256(ROOT / "autonomous_backtest_v10_0.py"),
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
    write_report(request, summary, phases, leader)

    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    print("V100_REAL_BACKTEST_COMPLETE")


if __name__ == "__main__":
    main()
