from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import _v107_engine as engine


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v10_7_forward"
DEFAULT_REQUEST = ROOT / "request.v10_7.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_request() -> dict[str, Any]:
    path = Path(os.environ.get("FORWARD_REQUEST_FILE", str(DEFAULT_REQUEST)))
    if not path.exists():
        raise FileNotFoundError(f"Missing request file: {path}")
    request = json.loads(path.read_text(encoding="utf-8"))
    engine.validate_config(request)
    return request


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(
            path,
            dtype={
                "record_id": "string",
                "record_sha256": "string",
                "chain_sha256": "string",
                "recorded_at_utc": "string",
            },
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_report(
    request: dict[str, Any],
    status: dict[str, Any],
    summary: pd.DataFrame,
    primary_audit: dict[str, Any],
    open_trades: pd.DataFrame,
    append_audit: dict[str, Any],
) -> None:
    by_id = summary.set_index("channel_id") if not summary.empty else pd.DataFrame()
    primary = by_id.loc[engine.SHORT_PRIMARY_ID] if not summary.empty else None
    long_shadow = by_id.loc[engine.LONG_SHADOW_ID] if not summary.empty else None
    portfolio_shadow = by_id.loc[engine.PORTFOLIO_SHADOW_ID] if not summary.empty else None
    lines = [
        "# BTCUSDT V10.7 追加式真实前向观察报告",
        "",
        f"- 参数冻结时间线起点：**{request['forward_observation']['start_utc']}**。",
        f"- 本次数据截止：**{status['effective_data_end_utc']}**，仅包含完整UTC日。",
        "- 空头是唯一主观察通道；多头与原多空组合只记录影子结果，不参与参数选择。",
        "- 不搜索参数、不替换指标、不自动宣布实盘合格。",
        "",
        "## 当前状态",
        "",
        f"- 状态：**{status['observation_status']}**。",
        f"- 已积累完整前向日：**{status['complete_forward_days']}** 天。",
        f"- 空头已完成交易：**{int(primary_audit['metrics']['trades'])} / {request['forward_evidence_thresholds']['minimum_closed_trades']}** 笔。",
        f"- 样本进度：**{primary_audit['progress_to_minimum_sample']:.1%}**。",
        f"- 实盘资格：**否**。",
        "",
        "## 三条冻结观察通道",
        "",
        "| 通道 | 角色 | 完成交易 | 胜率 | 实际盈亏比 | PF | 净R | 最大回撤 | 删除最佳10%后 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (primary, long_shadow, portfolio_shadow):
        if row is None:
            continue
        lines.append(
            f"| {row['channel_label']} | {row['role']} | {int(row['trades'])} | {row['win_rate']:.2%} | "
            f"{row['avg_win_loss_ratio']:.3f} | {row['profit_factor']:.3f} | {row['net_R']:.3f} | "
            f"{row['max_drawdown_R']:.3f} | {row['best_10pct_removed_net_R']:.3f} |"
        )

    lines.extend([
        "",
        "## 空头主观察门槛",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ])
    for name, passed in primary_audit["checks"].items():
        lines.append(f"| {name} | {'通过' if passed else '未通过/样本不足'} |")

    lines.extend([
        "",
        "## 当前未完成持仓快照",
        "",
    ])
    if open_trades.empty:
        lines.append("当前没有尚未达到止损、止盈或完整时间退出条件的观察持仓。")
    else:
        lines.extend([
            "| 通道 | 方向 | 入场时间 | 临时净R |",
            "|---|---|---|---:|",
        ])
        for row in open_trades.itertuples(index=False):
            direction = "多" if int(row.direction) == 1 else "空"
            lines.append(f"| {row.channel_label} | {direction} | {row.entry_time_utc} | {row.net_r:.3f} |")
        lines.append("")
        lines.append("上表仅为期末快照，不会写入完成交易账本，也不会用于统计。")

    lines.extend([
        "",
        "## 追加式账本审计",
        "",
        f"- 完成交易账本新增：**{append_audit['trades']['new_records_appended']}** 条。",
        f"- 信号账本新增：**{append_audit['signals']['new_records_appended']}** 条。",
        f"- 日快照新增：**{append_audit['snapshots']['new_records_appended']}** 条。",
        f"- 既有记录被修改或删除：**0**。",
        "",
        "## 纪律",
        "",
        "在空头完成交易少于20笔前，只积累证据，不评价参数优劣。达到20笔后也只执行预先写死的门槛检查；V10.7不会根据前向结果自动调参。",
        "",
        "重点文件：`forward_trade_ledger.csv`、`forward_signal_ledger.csv`、`daily_observation_snapshots.csv`、"
        "`open_trade_snapshot.csv`、`forward_summary.csv`、`cost_stress_summary.csv`、"
        "`primary_forward_audit.json`、`append_only_audit.json`、`strategy_lock.json`。",
    ])
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pipeline_smoke(request: dict[str, Any]) -> None:
    raw = engine.synthetic_5m_data(180_000, seed=20261707)
    first_raw = raw.iloc[:165_000].copy().reset_index(drop=True)
    first = engine.run_observation(first_raw, request)
    second = engine.run_observation(raw, request)
    if len(second["summary"]) != 3:
        raise AssertionError("Pipeline smoke did not return three channels")
    if len(second["cost_stress"]) != 6:
        raise AssertionError("Pipeline smoke did not return 3 channels x 2 cost scenarios")

    first_trades = engine.add_record_identity(first["closed_trades"], "trade")
    second_trades = engine.add_record_identity(second["closed_trades"], "trade")
    first_signals = engine.add_record_identity(first["signals"], "signal")
    second_signals = engine.add_record_identity(second["signals"], "signal")

    trade_ledger_1, trade_audit_1 = engine.append_only_merge(
        pd.DataFrame(), first_trades, "2026-08-01T12:00:00+00:00"
    )
    trade_ledger_2, trade_audit_2 = engine.append_only_merge(
        trade_ledger_1, second_trades, "2026-09-01T12:00:00+00:00"
    )
    trade_ledger_3, trade_audit_3 = engine.append_only_merge(
        trade_ledger_2, second_trades, "2026-09-02T12:00:00+00:00"
    )
    signal_ledger_1, signal_audit_1 = engine.append_only_merge(
        pd.DataFrame(), first_signals, "2026-08-01T12:00:00+00:00"
    )
    signal_ledger_2, signal_audit_2 = engine.append_only_merge(
        signal_ledger_1, second_signals, "2026-09-01T12:00:00+00:00"
    )
    assert trade_audit_3["new_records_appended"] == 0
    assert len(trade_ledger_3) == len(second_trades)
    assert len(signal_ledger_2) == len(second_signals)
    print(json.dumps({
        "channels": len(second["summary"]),
        "first_closed_trades": len(first["closed_trades"]),
        "final_closed_trades": len(second["closed_trades"]),
        "first_signals": len(first["signals"]),
        "final_signals": len(second["signals"]),
        "first_trade_append": trade_audit_1,
        "second_trade_append": trade_audit_2,
        "same_end_trade_append": trade_audit_3,
        "first_signal_append": signal_audit_1,
        "second_signal_append": signal_audit_2,
    }, ensure_ascii=False, indent=2, default=str))
    print("V107_PIPELINE_SMOKE_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT V10.7 append-only true forward observer")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pipeline-smoke", action="store_true")
    args = parser.parse_args()
    request = load_request()

    if args.self_test:
        engine.self_test(request)
        print("V107_SELF_TEST_OK")
        return
    if args.pipeline_smoke:
        pipeline_smoke(request)
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    run_utc = datetime.now(timezone.utc).isoformat()
    requested_end = engine.resolve_requested_end_date(request)
    effective_end, unavailable_days = engine.discover_latest_available_day(request, requested_end)

    existing_state_path = RESULTS / "state.json"
    existing_state = json.loads(existing_state_path.read_text(encoding="utf-8")) if existing_state_path.exists() else {}
    previous_end = existing_state.get("effective_data_end_date")
    if previous_end and effective_end < pd.Timestamp(previous_end, tz="UTC"):
        raise RuntimeError(f"Forward data end cannot move backward: previous={previous_end}, current={effective_end.date()}")

    raw, data_audit = engine.load_official_forward_data(request, effective_end)
    observation = engine.run_observation(raw, request)
    lock = engine.strategy_lock_payload(request)

    trade_recomputed = engine.add_record_identity(observation["closed_trades"], "trade")
    signal_recomputed = engine.add_record_identity(observation["signals"], "signal")
    existing_trades = read_csv_or_empty(RESULTS / "forward_trade_ledger.csv")
    existing_signals = read_csv_or_empty(RESULTS / "forward_signal_ledger.csv")
    merged_trades, trade_append = engine.append_only_merge(existing_trades, trade_recomputed, run_utc)
    merged_signals, signal_append = engine.append_only_merge(existing_signals, signal_recomputed, run_utc)

    primary_metrics = observation["primary_audit"]["metrics"]
    snapshot = pd.DataFrame([{
        "channel_id": "V10_7_DAILY_SNAPSHOT",
        "channel_label": "V10.7每日前向观察快照",
        "signal_time_utc": (effective_end + pd.Timedelta(hours=23, minutes=59, seconds=59)).isoformat(),
        "direction": 0,
        "source_component_id": engine.SHORT_PRIMARY_ID,
        "effective_data_end_date": effective_end.strftime("%Y-%m-%d"),
        "primary_closed_trades": int(primary_metrics["trades"]),
        "primary_win_rate": float(primary_metrics["win_rate"]),
        "primary_profit_factor": float(primary_metrics["profit_factor"]),
        "primary_net_R": float(primary_metrics["net_R"]),
        "primary_max_drawdown_R": float(primary_metrics["max_drawdown_R"]),
        "evidence_status": observation["primary_audit"]["evidence_status"],
    }])
    snapshot = engine.add_record_identity(snapshot, "signal")
    existing_snapshots = read_csv_or_empty(RESULTS / "daily_observation_snapshots.csv")
    merged_snapshots, snapshot_append = engine.append_only_merge(existing_snapshots, snapshot, run_utc)

    # The recomputed closed ledger and append-only ledger must describe exactly the same immutable records.
    if set(merged_trades.get("record_id", pd.Series(dtype=str)).astype(str)) != set(trade_recomputed.get("record_id", pd.Series(dtype=str)).astype(str)):
        raise RuntimeError("Persistent trade ledger differs from recomputed frozen history")
    if set(merged_signals.get("record_id", pd.Series(dtype=str)).astype(str)) != set(signal_recomputed.get("record_id", pd.Series(dtype=str)).astype(str)):
        raise RuntimeError("Persistent signal ledger differs from recomputed frozen history")

    forward_start = pd.Timestamp(request["forward_observation"]["start_utc"]).normalize()
    complete_forward_days = max(0, int((effective_end - forward_start).days) + 1)
    observation_status = observation["primary_audit"]["evidence_status"]
    if effective_end < forward_start:
        observation_status = "WAITING_FOR_TRUE_FORWARD_START"
    elif primary_metrics["trades"] == 0:
        observation_status = "FORWARD_STARTED_WAITING_FOR_FIRST_CLOSED_TRADE"

    append_audit = {
        "run_utc": run_utc,
        "trades": trade_append,
        "signals": signal_append,
        "snapshots": snapshot_append,
        "previous_effective_data_end_date": previous_end,
        "current_effective_data_end_date": effective_end.strftime("%Y-%m-%d"),
        "strategy_lock_sha256": lock["strategy_lock_sha256"],
        "append_only_passed": True,
    }
    status = {
        "engine": "BTCUSDT V10.7 append-only true forward observation",
        "release_version": "10.7",
        "parameters_frozen": True,
        "parameter_optimization_enabled": False,
        "winner_selection_enabled": False,
        "qualified_for_live_trading": False,
        "primary_channel_id": engine.SHORT_PRIMARY_ID,
        "long_channel_role": "SHADOW_ONLY",
        "legacy_portfolio_role": "SHADOW_ONLY",
        "forward_start_utc": request["forward_observation"]["start_utc"],
        "requested_data_end_date": requested_end.strftime("%Y-%m-%d"),
        "effective_data_end_date": effective_end.strftime("%Y-%m-%d"),
        "effective_data_end_utc": (effective_end + pd.Timedelta(hours=23, minutes=55)).isoformat(),
        "unavailable_latest_archive_days": unavailable_days,
        "complete_forward_days": complete_forward_days,
        "primary_closed_trades": int(primary_metrics["trades"]),
        "minimum_primary_closed_trades": int(request["forward_evidence_thresholds"]["minimum_closed_trades"]),
        "observation_status": observation_status,
        "strategy_lock_sha256": lock["strategy_lock_sha256"],
        "parent_v10_6_parameter_lock_sha256": request["parent_v10_6_parameter_lock_sha256"],
        "append_only_passed": True,
        "next_step": "Keep parameters frozen and let the scheduled workflow append complete UTC days until the primary short channel reaches 20 closed trades.",
    }
    state = {
        "last_run_utc": run_utc,
        "effective_data_end_date": effective_end.strftime("%Y-%m-%d"),
        "complete_forward_days": complete_forward_days,
        "trade_records": int(len(merged_trades)),
        "signal_records": int(len(merged_signals)),
        "snapshot_records": int(len(merged_snapshots)),
        "trade_chain_sha256": trade_append["final_chain_sha256"],
        "signal_chain_sha256": signal_append["final_chain_sha256"],
        "snapshot_chain_sha256": snapshot_append["final_chain_sha256"],
        "strategy_lock_sha256": lock["strategy_lock_sha256"],
    }

    merged_trades.to_csv(RESULTS / "forward_trade_ledger.csv", index=False)
    merged_signals.to_csv(RESULTS / "forward_signal_ledger.csv", index=False)
    merged_snapshots.to_csv(RESULTS / "daily_observation_snapshots.csv", index=False)
    observation["open_trades"].to_csv(RESULTS / "open_trade_snapshot.csv", index=False)
    observation["summary"].to_csv(RESULTS / "forward_summary.csv", index=False)
    observation["cost_stress"].to_csv(RESULTS / "cost_stress_summary.csv", index=False)
    pd.DataFrame(observation["signal_audit"]["channels"]).to_csv(RESULTS / "signal_counts.csv", index=False)
    pd.DataFrame(observation["signal_audit"]["portfolio_overlap"]).to_csv(RESULTS / "portfolio_overlap_audit.csv", index=False)

    json_dump(RESULTS / "status.json", status)
    json_dump(RESULTS / "state.json", state)
    json_dump(RESULTS / "strategy_lock.json", lock)
    json_dump(RESULTS / "primary_forward_audit.json", observation["primary_audit"])
    json_dump(RESULTS / "append_only_audit.json", append_audit)
    json_dump(RESULTS / "data_audit.json", data_audit)
    json_dump(RESULTS / "signal_audit.json", observation["signal_audit"])
    json_dump(RESULTS / "strategy_spec.json", {
        "channels": request["channels"],
        "channel_roles": request["channel_roles"],
        "direction_specific_parameters": request["direction_specific_parameters"],
        "portfolio_rules": request["portfolio_rules"],
        "execution": request["execution"],
        "forward_evidence_thresholds": request["forward_evidence_thresholds"],
    })
    identity = {
        "run_utc": run_utc,
        "python": sys.version,
        "platform": platform.platform(),
        "request_sha256": file_sha256(Path(os.environ.get("FORWARD_REQUEST_FILE", str(DEFAULT_REQUEST)))),
        "engine_sha256": file_sha256(ROOT / "_v107_engine.py"),
        "runner_sha256": file_sha256(ROOT / "autonomous_forward_observer_v10_7.py"),
        "strategy_lock_sha256": lock["strategy_lock_sha256"],
    }
    json_dump(RESULTS / "run_identity.json", identity)
    (RESULTS / "run_identity.txt").write_text("\n".join(f"{key}={value}" for key, value in identity.items()) + "\n", encoding="utf-8")
    write_report(request, status, observation["summary"], observation["primary_audit"], observation["open_trades"], append_audit)

    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    print("V107_FORWARD_OBSERVATION_COMPLETE")


if __name__ == "__main__":
    main()
