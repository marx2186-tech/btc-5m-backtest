from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

import _v106_engine as v106


ChannelSpec = v106.ChannelSpec
BASE = v106.BASE
STEP_MS = v106.STEP_MS

LONG_SHADOW_ID = "60m_long_shadow_v10_7"
SHORT_PRIMARY_ID = "60m_short_primary_v10_7"
PORTFOLIO_SHADOW_ID = "60m_legacy_portfolio_shadow_v10_7"

CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec(
        LONG_SHADOW_ID,
        "V10.7多头影子观察·冻结ADX≤45 RR2.5",
        "long_frozen_adx45",
        "LONG",
        1.25,
        2.5,
        36,
        True,
    ),
    ChannelSpec(
        SHORT_PRIMARY_ID,
        "V10.7空头主观察·冻结ADX衰减≥-4 RR2.5",
        "short_frozen_adx_decay_minus4",
        "SHORT",
        1.25,
        2.5,
        48,
        True,
    ),
    ChannelSpec(
        PORTFOLIO_SHADOW_ID,
        "V10.7原多空组合影子观察·共享单一仓位",
        "portfolio_shadow_v10_7",
        "BOTH",
        0.0,
        0.0,
        0,
        True,
        LONG_SHADOW_ID,
        SHORT_PRIMARY_ID,
    ),
)
CHANNEL_BY_ID = {spec.channel_id: spec for spec in CHANNELS}

TRADE_ID_FIELDS = ("channel_id", "entry_time_utc", "direction", "source_component_id")
SIGNAL_ID_FIELDS = ("channel_id", "signal_time_utc", "direction", "source_component_id")


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def strategy_lock_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "release_version": "10.7",
        "parent_v10_6_parameter_lock_sha256": config["parent_v10_6_parameter_lock_sha256"],
        "source_strategy": "V10.6 frozen long/short parameters; short is primary, long and legacy portfolio are shadow-only",
        "signal_timeframe_minutes": 60,
        "channels": [asdict(spec) for spec in CHANNELS],
        "channel_roles": config["channel_roles"],
        "direction_specific_parameters": config["direction_specific_parameters"],
        "portfolio_rules": config["portfolio_rules"],
        "execution": config["execution"],
        "forward_start_utc": config["forward_observation"]["start_utc"],
        "parameter_optimization_enabled": False,
        "winner_selection_enabled": False,
    }
    payload["strategy_lock_sha256"] = canonical_sha256(payload)
    return payload


def validate_config(config: dict[str, Any]) -> None:
    release = config["release"]
    if release["version"] != "10.7":
        raise ValueError("V10.7 requires release.version=10.7")
    required_true = (
        "parameters_fully_frozen",
        "append_only_true_forward_observation",
        "long_shadow_only",
        "short_primary_observation",
        "portfolio_shadow_only",
        "no_parameter_search",
        "no_new_indicator_or_strategy_family",
    )
    for key in required_true:
        if release.get(key) is not True:
            raise ValueError(f"release.{key} must be true")
    if config["symbol"].upper() != "BTCUSDT" or config["source_interval"] != "5m":
        raise ValueError("V10.7 is frozen to BTCUSDT official 5m archives")
    if int(config["signal_timeframe_minutes"]) != 60:
        raise ValueError("V10.7 uses only closed one-hour signals")
    if int(config["channel_count"]) != 3:
        raise ValueError("V10.7 requires exactly three channels")
    actual = [row["channel_id"] for row in config["channels"]]
    expected = [spec.channel_id for spec in CHANNELS]
    if actual != expected:
        raise ValueError(f"Channel order mismatch: expected={expected}, actual={actual}")
    if config["channel_roles"][SHORT_PRIMARY_ID] != "PRIMARY_FORWARD_OBSERVATION":
        raise ValueError("Short channel must remain the only primary observation channel")
    if config["channel_roles"][LONG_SHADOW_ID] != "SHADOW_ONLY":
        raise ValueError("Long channel must remain shadow-only")
    if config["channel_roles"][PORTFOLIO_SHADOW_ID] != "SHADOW_ONLY":
        raise ValueError("Legacy portfolio must remain shadow-only")
    execution = config["execution"]
    if float(execution["fee_rate_per_side"]) != 0.0005:
        raise ValueError("fee_rate_per_side must remain 0.0005")
    if float(execution["tick_size"]) * int(execution["slippage_ticks_per_fill"]) != 0.2:
        raise ValueError("slippage must remain 0.2 USDT per fill")
    if execution["same_bar_stop_target_rule"] != "STOP_FIRST_CONSERVATIVE":
        raise ValueError("same-bar ambiguity must remain stop-first")
    if execution["entry_rule"] != "NEXT_5M_OPEN_AFTER_CLOSED_1H_SIGNAL_BAR":
        raise ValueError("entry must remain next 5m open")
    lock = strategy_lock_payload(config)
    expected_lock = config.get("expected_strategy_lock_sha256")
    if expected_lock and lock["strategy_lock_sha256"] != expected_lock:
        raise ValueError(
            f"Strategy lock mismatch: expected={expected_lock}, actual={lock['strategy_lock_sha256']}"
        )


def resolve_requested_end_date(config: dict[str, Any], now_utc: datetime | None = None) -> pd.Timestamp:
    override = os.environ.get("FORWARD_END_DATE", "").strip()
    if override:
        end = pd.Timestamp(override, tz="UTC")
    else:
        now = now_utc or datetime.now(timezone.utc)
        end = pd.Timestamp((now - timedelta(days=1)).date(), tz="UTC")
    maximum = config["forward_observation"].get("maximum_end_date")
    if maximum:
        end = min(end, pd.Timestamp(maximum, tz="UTC"))
    return end.normalize()


def _daily_checksum_url(config: dict[str, Any], day: pd.Timestamp) -> str:
    symbol = config["symbol"].upper()
    interval = config["source_interval"]
    day_text = day.strftime("%Y-%m-%d")
    name = f"{symbol}-{interval}-{day_text}.zip"
    return f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/{interval}/{name}.CHECKSUM"


def discover_latest_available_day(config: dict[str, Any], requested_end: pd.Timestamp) -> tuple[pd.Timestamp, list[str]]:
    daily_start = pd.Timestamp(config["data_source"]["daily_archive_start_utc"]).normalize()
    if requested_end < daily_start:
        return requested_end, []
    max_lag = int(config["data_source"].get("max_archive_lag_days", 7))
    unavailable: list[str] = []
    for lag in range(max_lag + 1):
        candidate = requested_end - pd.Timedelta(days=lag)
        if candidate < daily_start:
            break
        url = _daily_checksum_url(config, candidate)
        try:
            response = requests.get(url, timeout=30)
        except requests.RequestException as exc:
            raise RuntimeError(f"Unable to probe Binance daily archive availability: {exc}") from exc
        if response.status_code == 200 and response.text.strip():
            return candidate, unavailable
        if response.status_code == 404:
            unavailable.append(candidate.strftime("%Y-%m-%d"))
            continue
        response.raise_for_status()
    raise RuntimeError(
        "No Binance verified daily archive found within configured lag window; unavailable="
        + ",".join(unavailable)
    )


def load_official_forward_data(config: dict[str, Any], effective_end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily_start = pd.Timestamp(config["data_source"]["daily_archive_start_utc"]).normalize()
    if effective_end < daily_start:
        raise ValueError("Effective end precedes required daily warmup start")
    inherited = copy.deepcopy(config)
    inherited["start_month"] = config["data_source"]["warmup_monthly_start"]
    inherited["end_month"] = config["data_source"]["warmup_monthly_end"]
    inherited["unseen_holdout"] = {
        "start_utc": daily_start.isoformat().replace("+00:00", "Z"),
        "end_utc": (effective_end + pd.Timedelta(hours=23, minutes=59, seconds=59)).isoformat().replace("+00:00", "Z"),
    }
    raw, audit = v106.load_official_5m_data(inherited)
    audit["requested_forward_end_date"] = effective_end.strftime("%Y-%m-%d")
    audit["forward_start_utc"] = config["forward_observation"]["start_utc"]
    audit["official_archives_only"] = True
    return raw, audit


def synthetic_5m_data(rows: int = 180_000, seed: int = 20261007) -> pd.DataFrame:
    return v106.synthetic_5m_data(rows, seed)


def generate_directional_signals(hourly_features: pd.DataFrame, spec: ChannelSpec) -> pd.DataFrame:
    if spec.profile not in {"long_frozen_adx45", "short_frozen_adx_decay_minus4"}:
        raise ValueError(f"Unsupported V10.7 directional profile: {spec.profile}")
    return v106.generate_directional_signals(hourly_features, spec)


def build_forward_signals(raw_5m: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    hourly = BASE.resample_hourly(raw_5m)
    features = BASE.add_hourly_features(hourly)
    start = pd.Timestamp(config["forward_observation"]["start_utc"])
    signal_map: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []

    for spec in CHANNELS:
        if spec.is_portfolio:
            continue
        signals = generate_directional_signals(features, spec)
        if not signals.empty:
            signals = signals.loc[pd.to_datetime(signals["signal_time"], utc=True) >= start].copy()
            signals = signals.sort_values("signal_time").drop_duplicates("signal_time", keep="first").reset_index(drop=True)
        signal_map[spec.channel_id] = signals
        audit_rows.append({
            "channel_id": spec.channel_id,
            "channel_label": spec.label,
            "role": config["channel_roles"][spec.channel_id],
            "signals": int(len(signals)),
            "long_signals": int((signals["direction"] == 1).sum()) if not signals.empty else 0,
            "short_signals": int((signals["direction"] == -1).sum()) if not signals.empty else 0,
        })

    portfolio_spec = CHANNEL_BY_ID[PORTFOLIO_SHADOW_ID]
    portfolio, overlap = v106.v105.v104.v103.build_portfolio_signals(portfolio_spec, signal_map)
    if not portfolio.empty:
        portfolio = portfolio.loc[pd.to_datetime(portfolio["signal_time"], utc=True) >= start].copy()
        portfolio = portfolio.sort_values("signal_time").drop_duplicates("signal_time", keep="first").reset_index(drop=True)
    signal_map[portfolio_spec.channel_id] = portfolio
    audit_rows.append({
        "channel_id": portfolio_spec.channel_id,
        "channel_label": portfolio_spec.label,
        "role": config["channel_roles"][portfolio_spec.channel_id],
        "signals": int(len(portfolio)),
        "long_signals": int((portfolio["direction"] == 1).sum()) if not portfolio.empty else 0,
        "short_signals": int((portfolio["direction"] == -1).sum()) if not portfolio.empty else 0,
    })
    return signal_map, {
        "channels": audit_rows,
        "portfolio_overlap": [overlap],
        "resample": {
            "hourly_rows": int(len(hourly)),
            "first_hour_close_utc": hourly.index.min().isoformat() if not hourly.empty else None,
            "last_hour_close_utc": hourly.index.max().isoformat() if not hourly.empty else None,
        },
    }


def _split_closed_and_open(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades.copy(), trades.copy()
    frame = trades.copy()
    entry = pd.to_datetime(frame["entry_time_utc"], utc=True)
    exit_time = pd.to_datetime(frame["exit_time_utc"], utc=True)
    full_time_exit = entry + pd.to_timedelta(frame["max_holding_hours"].astype(float), unit="h")
    provisional = (frame["exit_reason"] == "TIME_EXIT") & (exit_time < full_time_exit)
    open_snapshot = frame.loc[provisional].copy()
    closed = frame.loc[~provisional].copy()
    if not open_snapshot.empty:
        open_snapshot["provisional_end_of_data_exit"] = True
    return closed.reset_index(drop=True), open_snapshot.reset_index(drop=True)


def execute_forward_channels(
    raw_5m: pd.DataFrame,
    signal_map: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    closed_frames: list[pd.DataFrame] = []
    open_frames: list[pd.DataFrame] = []
    start = pd.Timestamp(config["forward_observation"]["start_utc"])
    for spec in CHANNELS:
        trades = v106.execute_channel(raw_5m, signal_map[spec.channel_id], spec, config)
        if not trades.empty:
            trades = trades.loc[pd.to_datetime(trades["entry_time_utc"], utc=True) >= start].copy()
        closed, open_snapshot = _split_closed_and_open(trades)
        if not closed.empty:
            closed_frames.append(closed)
        if not open_snapshot.empty:
            open_frames.append(open_snapshot)
    empty_trade_template = v106.execute_channel(raw_5m, pd.DataFrame(), CHANNELS[0], config)
    closed_ledger = pd.concat(closed_frames, ignore_index=True) if closed_frames else empty_trade_template.copy()
    open_ledger = pd.concat(open_frames, ignore_index=True) if open_frames else empty_trade_template.copy()
    if not closed_ledger.empty:
        closed_ledger = closed_ledger.sort_values(["exit_time_utc", "channel_id", "entry_time_utc"]).reset_index(drop=True)
    if not open_ledger.empty:
        open_ledger = open_ledger.sort_values(["channel_id", "entry_time_utc"]).reset_index(drop=True)
    return closed_ledger, open_ledger


def metrics(trades: pd.DataFrame | Iterable[dict[str, Any]]) -> dict[str, float]:
    return v106.metrics(trades)


def remove_best_fraction(trades: pd.DataFrame, fraction: float = 0.10) -> tuple[pd.DataFrame, dict[str, Any]]:
    return v106.remove_best_fraction(trades, fraction)


def summary_table(closed_ledger: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        subset = closed_ledger.loc[closed_ledger["channel_id"] == spec.channel_id] if not closed_ledger.empty else pd.DataFrame()
        _, tail = remove_best_fraction(subset, 0.10)
        rows.append({
            "channel_id": spec.channel_id,
            "channel_label": spec.label,
            "role": config["channel_roles"][spec.channel_id],
            **metrics(subset),
            "best_10pct_removed_net_R": tail["net_R"],
        })
    return pd.DataFrame(rows)


def cost_stress_table(
    raw_5m: pd.DataFrame,
    signal_map: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_fee = float(config["execution"]["fee_rate_per_side"])
    base_ticks = int(config["execution"]["slippage_ticks_per_fill"])
    for scenario in config["cost_stress_scenarios"]:
        stressed = copy.deepcopy(config)
        stressed["execution"]["fee_rate_per_side"] = base_fee * float(scenario["fee_multiplier"])
        stressed["execution"]["slippage_ticks_per_fill"] = int(round(base_ticks * float(scenario["slippage_multiplier"])))
        closed, _ = execute_forward_channels(raw_5m, signal_map, stressed)
        for spec in CHANNELS:
            subset = closed.loc[closed["channel_id"] == spec.channel_id] if not closed.empty else pd.DataFrame()
            rows.append({
                "scenario_id": scenario["scenario_id"],
                "fee_multiplier": float(scenario["fee_multiplier"]),
                "slippage_multiplier": float(scenario["slippage_multiplier"]),
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                **metrics(subset),
            })
    return pd.DataFrame(rows)


def primary_forward_audit(
    closed_ledger: pd.DataFrame,
    cost_stress: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    trades = closed_ledger.loc[closed_ledger["channel_id"] == SHORT_PRIMARY_ID] if not closed_ledger.empty else pd.DataFrame()
    base = metrics(trades)
    _, tail = remove_best_fraction(trades, float(config["forward_evidence_thresholds"]["remove_best_fraction"]))
    stress = cost_stress.loc[
        (cost_stress["channel_id"] == SHORT_PRIMARY_ID)
        & (cost_stress["scenario_id"] == "cost_1_5x")
    ]
    stress_pf = float(stress.iloc[0]["profit_factor"]) if not stress.empty else 0.0
    threshold = config["forward_evidence_thresholds"]
    checks = {
        "minimum_closed_trades": base["trades"] >= int(threshold["minimum_closed_trades"]),
        "minimum_win_rate": base["win_rate"] >= float(threshold["minimum_win_rate"]),
        "minimum_avg_win_loss_ratio": base["avg_win_loss_ratio"] >= float(threshold["minimum_avg_win_loss_ratio"]),
        "minimum_profit_factor": base["profit_factor"] >= float(threshold["minimum_profit_factor"]),
        "positive_expectancy": base["expectancy_R"] > 0,
        "maximum_drawdown_R": base["max_drawdown_R"] <= float(threshold["maximum_drawdown_R"]),
        "best_10pct_removed_still_profitable": tail["net_R"] > 0,
        "minimum_cost_1_5x_profit_factor": stress_pf >= float(threshold["minimum_cost_1_5x_profit_factor"]),
    }
    minimum_reached = checks["minimum_closed_trades"]
    if not minimum_reached:
        evidence_status = "COLLECTING_FORWARD_EVIDENCE"
    elif all(checks.values()):
        evidence_status = "FORWARD_EVIDENCE_GATE_PASSED_NOT_LIVE_QUALIFIED"
    else:
        evidence_status = "FORWARD_EVIDENCE_GATE_FAILED"
    return {
        "primary_channel_id": SHORT_PRIMARY_ID,
        "metrics": base,
        "best_10pct_removed": tail,
        "cost_1_5x_profit_factor": stress_pf,
        "checks": checks,
        "minimum_sample_reached": minimum_reached,
        "progress_to_minimum_sample": min(1.0, base["trades"] / max(1, int(threshold["minimum_closed_trades"]))),
        "evidence_status": evidence_status,
        "qualified_for_live_trading": False,
        "parameters_may_be_changed": False,
    }


def signal_ledger_frame(signal_map: dict[str, pd.DataFrame], config: dict[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    desired = [
        "channel_id", "channel_label", "profile", "signal_time", "direction",
        "source_component_id", "source_component_label", "stop_atr_multiple", "reward_risk",
        "max_holding_hours", "adx14", "rsi14", "plus_di14", "minus_di14",
        "ema_separation_atr", "ema_slope_3_atr", "volume_ratio", "channel_width_atr",
        "adx_change_3", "ema50_slope_6_atr", "price_extension_ema20_atr", "atr_ratio",
    ]
    for channel_id, signals in signal_map.items():
        if signals.empty:
            continue
        frame = signals.copy()
        for column in ("source_component_id", "source_component_label"):
            if column not in frame.columns:
                frame[column] = frame["channel_id"] if column.endswith("_id") else frame["channel_label"]
        keep = [column for column in desired if column in frame.columns]
        frame = frame[keep].copy()
        frame = frame.rename(columns={"signal_time": "signal_time_utc"})
        frame["signal_time_utc"] = pd.to_datetime(frame["signal_time_utc"], utc=True).map(lambda value: value.isoformat())
        frame["role"] = config["channel_roles"][channel_id]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["channel_id", "channel_label", "signal_time_utc", "direction", "role"])
    return pd.concat(frames, ignore_index=True).sort_values(["signal_time_utc", "channel_id"]).reset_index(drop=True)


def _stable_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return format(float(value), ".15g")
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return str(value)


def add_record_identity(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    result = frame.copy()
    if frame.empty:
        if "record_id" not in result.columns:
            result.insert(0, "record_id", pd.Series(dtype="string"))
        if "record_sha256" not in result.columns:
            result.insert(1, "record_sha256", pd.Series(dtype="string"))
        return result
    id_fields = TRADE_ID_FIELDS if kind == "trade" else SIGNAL_ID_FIELDS
    for field in id_fields:
        if field not in result.columns:
            result[field] = ""
    ids: list[str] = []
    hashes: list[str] = []
    for row in result.to_dict(orient="records"):
        identity = "|".join(str(row.get(field, "")) for field in id_fields)
        record_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        canonical = {key: _stable_value(value) for key, value in sorted(row.items()) if key not in {"record_sha256", "chain_sha256", "recorded_at_utc"}}
        ids.append(record_id)
        hashes.append(canonical_sha256(canonical))
    result.insert(0, "record_id", ids)
    result.insert(1, "record_sha256", hashes)
    return result


def append_only_merge(
    existing: pd.DataFrame,
    recomputed: pd.DataFrame,
    run_utc: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if recomputed.empty:
        recomputed = pd.DataFrame(columns=existing.columns if not existing.empty else recomputed.columns)
    if existing.empty:
        existing = pd.DataFrame(columns=recomputed.columns)
    for integrity_column in ("record_id", "record_sha256", "chain_sha256", "recorded_at_utc"):
        if integrity_column not in existing.columns:
            existing[integrity_column] = pd.Series(dtype="string")
        if integrity_column not in recomputed.columns:
            recomputed[integrity_column] = pd.Series(dtype="string")
    if not existing.empty:
        required = {"record_id", "record_sha256", "chain_sha256"}
        if not required.issubset(existing.columns):
            raise RuntimeError("Existing append-only ledger is missing integrity columns")
        existing_by_id = existing.set_index("record_id")
        recomputed_by_id = recomputed.set_index("record_id") if not recomputed.empty else pd.DataFrame()
        for record_id, row in existing_by_id.iterrows():
            if recomputed.empty or record_id not in recomputed_by_id.index:
                raise RuntimeError(f"Append-only violation: existing record disappeared: {record_id}")
            candidate = recomputed_by_id.loc[record_id]
            if isinstance(candidate, pd.DataFrame):
                candidate = candidate.iloc[0]
            if str(row["record_sha256"]) != str(candidate["record_sha256"]):
                raise RuntimeError(f"Append-only violation: existing record changed: {record_id}")
    existing_ids = set(existing["record_id"].astype(str)) if not existing.empty else set()
    new_rows = recomputed.loc[~recomputed["record_id"].astype(str).isin(existing_ids)].copy() if not recomputed.empty else recomputed.copy()
    if not new_rows.empty:
        sort_cols = [column for column in ("exit_time_utc", "signal_time_utc", "channel_id", "entry_time_utc") if column in new_rows.columns]
        if sort_cols:
            new_rows = new_rows.sort_values(sort_cols).reset_index(drop=True)
        previous_chain = str(existing.iloc[-1]["chain_sha256"]) if not existing.empty else "GENESIS"
        chains: list[str] = []
        for row_hash in new_rows["record_sha256"].astype(str):
            previous_chain = hashlib.sha256(f"{previous_chain}|{row_hash}".encode("utf-8")).hexdigest()
            chains.append(previous_chain)
        new_rows["chain_sha256"] = chains
        new_rows["recorded_at_utc"] = run_utc
    columns = list(existing.columns)
    for column in new_rows.columns:
        if column not in columns:
            columns.append(column)
    for column in columns:
        if column not in existing.columns:
            existing[column] = pd.NA
        if column not in new_rows.columns:
            new_rows[column] = pd.NA
    if existing.empty:
        merged = new_rows[columns].copy().reset_index(drop=True)
    elif new_rows.empty:
        merged = existing[columns].copy().reset_index(drop=True)
    else:
        merged = pd.concat([existing[columns], new_rows[columns]], ignore_index=True)
    audit = {
        "existing_records": int(len(existing)),
        "recomputed_records": int(len(recomputed)),
        "new_records_appended": int(len(new_rows)),
        "final_records": int(len(merged)),
        "final_chain_sha256": str(merged.iloc[-1]["chain_sha256"]) if not merged.empty else "GENESIS",
        "append_only_passed": True,
    }
    return merged, audit


def run_observation(raw_5m: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    signal_map, signal_audit = build_forward_signals(raw_5m, config)
    closed, open_snapshot = execute_forward_channels(raw_5m, signal_map, config)
    summary = summary_table(closed, config)
    stress = cost_stress_table(raw_5m, signal_map, config)
    primary_audit = primary_forward_audit(closed, stress, config)
    signals = signal_ledger_frame(signal_map, config)
    return {
        "signal_map": signal_map,
        "signal_audit": signal_audit,
        "signals": signals,
        "closed_trades": closed,
        "open_trades": open_snapshot,
        "summary": summary,
        "cost_stress": stress,
        "primary_audit": primary_audit,
    }


def self_test(config: dict[str, Any]) -> None:
    validate_config(config)
    assert len(CHANNELS) == 3
    assert sum(spec.is_portfolio for spec in CHANNELS) == 1
    raw = synthetic_5m_data(180_000, seed=20261007)
    result = run_observation(raw, config)
    assert len(result["summary"]) == 3
    assert set(result["summary"]["channel_id"]) == set(CHANNEL_BY_ID)
    assert set(result["cost_stress"]["scenario_id"]) == {"base_cost", "cost_1_5x"}
    if not result["closed_trades"].empty:
        short = result["closed_trades"].loc[result["closed_trades"]["channel_id"] == SHORT_PRIMARY_ID]
        long = result["closed_trades"].loc[result["closed_trades"]["channel_id"] == LONG_SHADOW_ID]
        assert set(short["direction"]).issubset({-1})
        assert set(long["direction"]).issubset({1})
        for _, group in result["closed_trades"].groupby("channel_id"):
            entry = pd.to_datetime(group["entry_time_utc"], utc=True).reset_index(drop=True)
            exit_time = pd.to_datetime(group["exit_time_utc"], utc=True).reset_index(drop=True)
            if len(group) > 1:
                assert (entry.iloc[1:].reset_index(drop=True) >= exit_time.iloc[:-1].reset_index(drop=True)).all()
    trade_records = add_record_identity(result["closed_trades"], "trade")
    signal_records = add_record_identity(result["signals"], "signal")
    merged_trades, audit_1 = append_only_merge(pd.DataFrame(), trade_records, "2026-07-29T12:00:00+00:00")
    merged_trades_2, audit_2 = append_only_merge(merged_trades, trade_records, "2026-07-30T12:00:00+00:00")
    assert audit_1["new_records_appended"] == len(trade_records)
    assert audit_2["new_records_appended"] == 0
    assert len(merged_trades_2) == len(merged_trades)
    merged_signals, _ = append_only_merge(pd.DataFrame(), signal_records, "2026-07-29T12:00:00+00:00")
    assert len(merged_signals) == len(signal_records)
    print("V107_ENGINE_SELF_TEST_OK")


__all__ = [
    "CHANNELS", "CHANNEL_BY_ID", "LONG_SHADOW_ID", "SHORT_PRIMARY_ID", "PORTFOLIO_SHADOW_ID",
    "strategy_lock_payload", "validate_config", "resolve_requested_end_date", "discover_latest_available_day",
    "load_official_forward_data", "synthetic_5m_data", "run_observation", "add_record_identity",
    "append_only_merge", "metrics", "self_test",
]
