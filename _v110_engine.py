from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

import _v108_engine as v108
import _v109_engine as v109


ChannelSpec = v108.ChannelSpec
BASE = v108.BASE
V103 = v108.V103


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec(
        "baseline_1h_shared_v10_6",
        "基准·V10.6原1小时多空共享组合 RR2.5",
        "baseline_1h_shared_v10_6",
        "BOTH",
        0.0,
        0.0,
        0,
        True,
    ),
    ChannelSpec(
        "state_direct_after_1h_confirmation_rr2_0",
        "实验A·严格状态机1H确认后直接成交 RR2.0",
        "state_direct_after_1h_confirmation",
        "BOTH",
        1.25,
        2.0,
        48,
        False,
    ),
    ChannelSpec(
        "state_15m_single_local_break_rr2_0",
        "实验B·严格状态机+15M单一局部突破 RR2.0",
        "state_15m_single_local_break",
        "BOTH",
        1.25,
        2.0,
        48,
        False,
    ),
    ChannelSpec(
        "state_15m_pullback_reclaim_rr2_0",
        "实验C·严格状态机+15M回踩重新确认 RR2.0",
        "state_15m_pullback_reclaim",
        "BOTH",
        1.25,
        2.0,
        48,
        False,
    ),
)

CHANNEL_BY_ID = {spec.channel_id: spec for spec in CHANNELS}


def synthetic_5m_data(rows: int = 220_000, seed: int = 1020) -> pd.DataFrame:
    return v109.synthetic_5m_data(rows, seed)


def load_official_5m_data(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    return v109.load_official_5m_data(config)


def _evaluation_bounds(config: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp(config["evaluation_window"]["start_utc"]),
        pd.Timestamp(config["evaluation_window"]["end_utc"]),
    )


def _filter_evaluation_window(signals: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if signals.empty:
        return signals
    start, end = _evaluation_bounds(config)
    times = pd.to_datetime(signals["signal_time"], utc=True)
    return signals.loc[(times >= start) & (times <= end)].sort_values(["signal_time", "direction"]).reset_index(drop=True)


def _setup_value(setup: Any, name: str, default: Any = np.nan) -> Any:
    if isinstance(setup, pd.Series):
        return setup.get(name, default)
    return getattr(setup, name, default)


def _signal_record(
    setup: Any,
    signal_time: pd.Timestamp,
    spec: ChannelSpec,
    config: dict[str, Any],
    trigger_type: str,
    trigger_row: pd.Series | None = None,
) -> dict[str, Any]:
    execution = config["state_machine_parameters"]["execution"]
    direction = int(_setup_value(setup, "direction"))
    source_label = "多头组件" if direction == 1 else "空头组件"
    setup_time = pd.Timestamp(_setup_value(setup, "setup_time"))
    trigger_volume = float(trigger_row.get("volume_ratio", np.nan)) if trigger_row is not None else np.nan
    trigger_clv = (
        float(trigger_row.get("clv_long" if direction == 1 else "clv_short", np.nan))
        if trigger_row is not None else np.nan
    )
    trigger_extension = float(trigger_row.get("price_extension_ema20_atr", np.nan)) if trigger_row is not None else np.nan
    return {
        "signal_time": pd.Timestamp(signal_time),
        "direction": direction,
        "close": float(_setup_value(setup, "close")),
        "atr14": float(_setup_value(setup, "atr14")),
        "adx14": float(_setup_value(setup, "adx14")),
        "rsi14": float(_setup_value(setup, "rsi14")),
        "plus_di14": float(_setup_value(setup, "plus_di14")),
        "minus_di14": float(_setup_value(setup, "minus_di14")),
        "ema_separation_atr": float(_setup_value(setup, "ema_separation_atr")),
        "ema_slope_3_atr": float(_setup_value(setup, "ema_slope_3_atr")),
        "volume_ratio": float(_setup_value(setup, "volume_ratio")),
        "clv_long": float(_setup_value(setup, "clv_long")),
        "clv_short": float(_setup_value(setup, "clv_short")),
        "channel_width_atr": float(_setup_value(setup, "channel_width_atr")),
        "adx_change_3": float(_setup_value(setup, "adx_change_3")),
        "ema50_slope_6_atr": float(_setup_value(setup, "ema50_slope_6_atr")),
        "price_extension_ema20_atr": float(_setup_value(setup, "price_extension_ema20_atr")),
        "atr_ratio": float(_setup_value(setup, "atr_ratio")),
        "trend_age_long": float(_setup_value(setup, "trend_age_long")),
        "trend_age_short": float(_setup_value(setup, "trend_age_short")),
        "stop_atr_multiple": float(execution["stop_atr_multiple"]),
        "reward_risk": float(execution["reward_risk"]),
        "max_holding_hours": int(execution["max_holding_hours"]),
        "channel_id": spec.channel_id,
        "channel_label": spec.label,
        "profile": spec.profile,
        "direction_scope": spec.direction_scope,
        "is_baseline": False,
        "source_component_id": f"{spec.channel_id}_{'long' if direction == 1 else 'short'}",
        "source_component_label": f"{spec.label}·{source_label}",
        "setup_type": "IMPULSE_PULLBACK_CONFIRM",
        "trigger_type": trigger_type,
        "setup_time": setup_time,
        "setup_id": str(_setup_value(setup, "setup_id")),
        "environment_leg_id": int(_setup_value(setup, "environment_leg_id")),
        "cycle_number": int(_setup_value(setup, "cycle_number")),
        "impulse_time": pd.Timestamp(_setup_value(setup, "impulse_time")),
        "pullback_time": pd.Timestamp(_setup_value(setup, "pullback_time")),
        "confirmation_level": float(_setup_value(setup, "confirmation_level")),
        "trigger_volume_ratio": trigger_volume,
        "trigger_clv": trigger_clv,
        "trigger_extension_ema20_atr": trigger_extension,
        "management_mode": "FIXED_STOP_TARGET",
    }


def _finalize_signals(rows: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    signals = pd.DataFrame(rows)
    if signals.empty:
        return signals
    signals = signals.sort_values(["signal_time", "direction", "setup_id"]).drop_duplicates("setup_id", keep="first")
    conflicts = signals.groupby("signal_time")["direction"].nunique()
    conflict_times = set(conflicts.loc[conflicts > 1].index)
    if conflict_times:
        signals = signals.loc[~signals["signal_time"].isin(conflict_times)]
    return _filter_evaluation_window(signals.reset_index(drop=True), config)


def build_direct_confirmation_signals(
    setups: pd.DataFrame,
    spec: ChannelSpec,
    config: dict[str, Any],
) -> pd.DataFrame:
    if setups.empty:
        return pd.DataFrame()
    rows = [
        _signal_record(setup, pd.Timestamp(setup.setup_time), spec, config, "DIRECT_AFTER_CLOSED_1H_CONFIRMATION")
        for setup in setups.itertuples(index=False)
    ]
    return _finalize_signals(rows, config)


def build_single_local_break_signals(
    m15: pd.DataFrame,
    setups: pd.DataFrame,
    h4_env: pd.DataFrame,
    spec: ChannelSpec,
    config: dict[str, Any],
) -> pd.DataFrame:
    if setups.empty:
        return pd.DataFrame()
    p = config["trigger_comparison_parameters"]["single_local_breakout"]
    minimum_delay = pd.Timedelta(minutes=int(p["minimum_delay_after_confirmation_minutes"]))
    validity = pd.Timedelta(hours=float(p["setup_validity_hours"]))
    max_extension = float(p["maximum_price_extension_ema20_atr"])
    m15_joined = v108._asof_join(m15, h4_env, ["environment_direction"], "h4_")
    rows: list[dict[str, Any]] = []
    for setup in setups.itertuples(index=False):
        setup_time = pd.Timestamp(setup.setup_time)
        direction = int(setup.direction)
        candidates = m15_joined.loc[
            (m15_joined.index >= setup_time + minimum_delay)
            & (m15_joined.index <= setup_time + validity)
        ].copy()
        if candidates.empty:
            continue
        if direction == 1:
            ok = (
                (candidates["h4_environment_direction"] == 1)
                & candidates["break_high_4"].fillna(False)
                & (candidates["close"] > candidates["ema20"])
                & (candidates["close"] > candidates["open"])
                & (candidates["price_extension_ema20_atr"] <= max_extension)
            )
            trigger_type = "SINGLE_LOCAL_HIGH_BREAK"
        else:
            ok = (
                (candidates["h4_environment_direction"] == -1)
                & candidates["break_low_4"].fillna(False)
                & (candidates["close"] < candidates["ema20"])
                & (candidates["close"] < candidates["open"])
                & (candidates["price_extension_ema20_atr"] <= max_extension)
            )
            trigger_type = "SINGLE_LOCAL_LOW_BREAK"
        valid = candidates.loc[ok]
        if valid.empty:
            continue
        trigger_time = pd.Timestamp(valid.index[0])
        rows.append(_signal_record(setup, trigger_time, spec, config, trigger_type, valid.iloc[0]))
    return _finalize_signals(rows, config)


def build_pullback_reclaim_signals(
    m15: pd.DataFrame,
    setups: pd.DataFrame,
    h4_env: pd.DataFrame,
    spec: ChannelSpec,
    config: dict[str, Any],
) -> pd.DataFrame:
    if setups.empty:
        return pd.DataFrame()
    p = config["trigger_comparison_parameters"]["pullback_reclaim"]
    minimum_delay = pd.Timedelta(minutes=int(p["minimum_delay_after_confirmation_minutes"]))
    validity = pd.Timedelta(hours=float(p["setup_validity_hours"]))
    touch_zone = float(p["ema20_touch_zone_atr"])
    max_extension = float(p["maximum_reclaim_extension_ema20_atr"])
    require_next_bar = bool(p["reclaim_must_occur_after_touch_bar"])

    m15_joined = v108._asof_join(m15, h4_env, ["environment_direction"], "h4_").copy()
    m15_joined["previous_high"] = m15_joined["high"].shift(1)
    m15_joined["previous_low"] = m15_joined["low"].shift(1)
    rows: list[dict[str, Any]] = []

    for setup in setups.itertuples(index=False):
        setup_time = pd.Timestamp(setup.setup_time)
        direction = int(setup.direction)
        candidates = m15_joined.loc[
            (m15_joined.index >= setup_time + minimum_delay)
            & (m15_joined.index <= setup_time + validity)
        ].copy()
        if candidates.empty:
            continue
        touch_position: int | None = None
        for pos, (_, row) in enumerate(candidates.iterrows()):
            atr = float(row.get("atr14", np.nan))
            if not math.isfinite(atr) or atr <= 0:
                continue
            if direction == 1:
                touched = float(row["low"]) <= float(row["ema20"]) + touch_zone * atr
            else:
                touched = float(row["high"]) >= float(row["ema20"]) - touch_zone * atr
            if touched:
                touch_position = pos
                break
        if touch_position is None:
            continue
        start_pos = touch_position + 1 if require_next_bar else touch_position
        if start_pos >= len(candidates):
            continue
        post_touch = candidates.iloc[start_pos:].copy()
        if direction == 1:
            ok = (
                (post_touch["h4_environment_direction"] == 1)
                & (post_touch["close"] > post_touch["ema20"])
                & (post_touch["close"] > post_touch["open"])
                & (post_touch["close"] > post_touch["previous_high"])
                & (post_touch["price_extension_ema20_atr"] <= max_extension)
            )
            trigger_type = "PULLBACK_TOUCH_THEN_RECLAIM_HIGH"
        else:
            ok = (
                (post_touch["h4_environment_direction"] == -1)
                & (post_touch["close"] < post_touch["ema20"])
                & (post_touch["close"] < post_touch["open"])
                & (post_touch["close"] < post_touch["previous_low"])
                & (post_touch["price_extension_ema20_atr"] <= max_extension)
            )
            trigger_type = "PULLBACK_TOUCH_THEN_RECLAIM_LOW"
        valid = post_touch.loc[ok]
        if valid.empty:
            continue
        trigger_time = pd.Timestamp(valid.index[0])
        record = _signal_record(setup, trigger_time, spec, config, trigger_type, valid.iloc[0])
        record["trigger_touch_time"] = pd.Timestamp(candidates.index[touch_position])
        rows.append(record)
    return _finalize_signals(rows, config)


def build_setup_signal_matrix(
    setups: pd.DataFrame,
    signal_map: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "setup_id", "setup_time_utc", "direction", "environment_leg_id", "cycle_number",
        "direct_signal_time_utc", "local_break_signal_time_utc", "pullback_reclaim_signal_time_utc",
        "direct_triggered", "local_break_triggered", "pullback_reclaim_triggered",
    ]
    if setups.empty:
        return pd.DataFrame(columns=columns)
    start, end = _evaluation_bounds(config)
    subset = setups.loc[
        (pd.to_datetime(setups["setup_time"], utc=True) >= start)
        & (pd.to_datetime(setups["setup_time"], utc=True) <= end)
    ].copy()
    rows: list[dict[str, Any]] = []
    channel_columns = {
        "state_direct_after_1h_confirmation_rr2_0": "direct",
        "state_15m_single_local_break_rr2_0": "local_break",
        "state_15m_pullback_reclaim_rr2_0": "pullback_reclaim",
    }
    lookups: dict[str, dict[str, pd.Timestamp]] = {}
    for channel_id, prefix in channel_columns.items():
        frame = signal_map[channel_id]
        lookups[prefix] = (
            {str(row.setup_id): pd.Timestamp(row.signal_time) for row in frame.itertuples(index=False)}
            if not frame.empty else {}
        )
    for setup in subset.itertuples(index=False):
        setup_id = str(setup.setup_id)
        row = {
            "setup_id": setup_id,
            "setup_time_utc": pd.Timestamp(setup.setup_time).isoformat(),
            "direction": int(setup.direction),
            "environment_leg_id": int(setup.environment_leg_id),
            "cycle_number": int(setup.cycle_number),
        }
        for prefix in ("direct", "local_break", "pullback_reclaim"):
            signal_time = lookups[prefix].get(setup_id)
            row[f"{prefix}_signal_time_utc"] = signal_time.isoformat() if signal_time is not None else ""
            row[f"{prefix}_triggered"] = signal_time is not None
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def build_all_signals(
    raw_5m: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m15 = v108.add_features(v108.resample_ohlcv(raw_5m, 15))
    h1 = v108.add_features(v108.resample_ohlcv(raw_5m, 60))
    h4 = v108.four_hour_environment(v108.add_features(v108.resample_ohlcv(raw_5m, 240)), config)
    setups, state_events, environment_legs = v109.build_state_machine_setups(h1, h4, config)

    baseline_spec = CHANNEL_BY_ID["baseline_1h_shared_v10_6"]
    direct_spec = CHANNEL_BY_ID["state_direct_after_1h_confirmation_rr2_0"]
    local_spec = CHANNEL_BY_ID["state_15m_single_local_break_rr2_0"]
    reclaim_spec = CHANNEL_BY_ID["state_15m_pullback_reclaim_rr2_0"]

    baseline = v108.build_baseline_signals(raw_5m, baseline_spec, config)
    if not baseline.empty:
        baseline["management_mode"] = "V10_6_FROZEN"
        baseline["setup_id"] = baseline["signal_time"].astype(str)
    direct = build_direct_confirmation_signals(setups, direct_spec, config)
    local_break = build_single_local_break_signals(m15, setups, h4, local_spec, config)
    reclaim = build_pullback_reclaim_signals(m15, setups, h4, reclaim_spec, config)

    signal_map = {
        baseline_spec.channel_id: baseline,
        direct_spec.channel_id: direct,
        local_spec.channel_id: local_break,
        reclaim_spec.channel_id: reclaim,
    }

    audit_rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        signals = signal_map[spec.channel_id]
        audit_rows.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "profile": spec.profile,
                "signals": int(len(signals)),
                "long_signals": int((signals["direction"] == 1).sum()) if not signals.empty else 0,
                "short_signals": int((signals["direction"] == -1).sum()) if not signals.empty else 0,
                "unique_signal_times": int(signals["signal_time"].nunique()) if not signals.empty else 0,
                "unique_setup_ids": int(signals["setup_id"].nunique()) if not signals.empty and "setup_id" in signals else 0,
            }
        )

    events = state_events.copy()
    stage_rows: list[dict[str, Any]] = []
    for stage in ["ENVIRONMENT_LEG_START", "IMPULSE_QUALIFIED", "PULLBACK_QUALIFIED", "CONFIRMATION_EMITTED"]:
        subset = events.loc[events["stage"] == stage] if not events.empty else pd.DataFrame()
        stage_rows.append(
            {
                "stage": stage,
                "long_count": int((subset["direction"] == 1).sum()) if not subset.empty else 0,
                "short_count": int((subset["direction"] == -1).sum()) if not subset.empty else 0,
            }
        )
    for stage, frame in [
        ("DIRECT_AFTER_1H_CONFIRMATION", direct),
        ("SINGLE_LOCAL_15M_BREAK", local_break),
        ("PULLBACK_RECLAIM_15M", reclaim),
    ]:
        stage_rows.append(
            {
                "stage": stage,
                "long_count": int((frame["direction"] == 1).sum()) if not frame.empty else 0,
                "short_count": int((frame["direction"] == -1).sum()) if not frame.empty else 0,
            }
        )
    funnel = pd.DataFrame(stage_rows)
    setup_matrix = build_setup_signal_matrix(setups, signal_map, config)
    setup_count = int(len(setup_matrix))
    coverage_rows = []
    for channel_id, prefix in [
        (direct_spec.channel_id, "direct"),
        (local_spec.channel_id, "local_break"),
        (reclaim_spec.channel_id, "pullback_reclaim"),
    ]:
        frame = signal_map[channel_id]
        coverage_rows.append(
            {
                "channel_id": channel_id,
                "channel_label": CHANNEL_BY_ID[channel_id].label,
                "available_1h_confirmations": setup_count,
                "signals_emitted": int(len(frame)),
                "coverage_rate": float(len(frame) / setup_count) if setup_count else 0.0,
                "long_signals": int((frame["direction"] == 1).sum()) if not frame.empty else 0,
                "short_signals": int((frame["direction"] == -1).sum()) if not frame.empty else 0,
                "all_signals_reference_valid_setup": bool(
                    set(frame["setup_id"].astype(str)).issubset(set(setup_matrix["setup_id"].astype(str)))
                ) if not frame.empty else True,
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    audit = {
        "resample": {
            "15m_rows": int(len(m15)),
            "1h_rows": int(len(h1)),
            "4h_rows": int(len(h4)),
            "no_lookahead_alignment": "BACKWARD_ASOF_ONLY_CLOSED_BARS",
        },
        "channels": audit_rows,
        "state_machine": {
            "environment_legs": int(len(environment_legs)),
            "confirmation_setups_all_data": int(len(setups)),
            "confirmation_setups_evaluation_window": setup_count,
            "one_setup_per_cycle": bool(setups["setup_id"].is_unique) if not setups.empty else True,
            "upper_state_machine_identical_to_v10_9": True,
        },
        "trigger_comparison": {
            "direct_signals": int(len(direct)),
            "single_local_break_signals": int(len(local_break)),
            "pullback_reclaim_signals": int(len(reclaim)),
            "all_channels_use_same_confirmation_pool": bool(coverage["all_signals_reference_valid_setup"].all()) if not coverage.empty else True,
        },
        "evaluation_window": config["evaluation_window"],
    }
    return signal_map, audit, funnel, events, setup_matrix, coverage


def metrics(trades: pd.DataFrame | Iterable[dict[str, Any]]) -> dict[str, float]:
    return v109.metrics(trades)


def remove_best_fraction(trades: pd.DataFrame, fraction: float = 0.10) -> tuple[pd.DataFrame, dict[str, Any]]:
    return v109.remove_best_fraction(trades, fraction)


def summarize_channel(trades: pd.DataFrame, spec: ChannelSpec, config: dict[str, Any]) -> dict[str, Any]:
    return v109.summarize_channel(trades, spec, config)


def grouped_metrics(trades: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return v109.grouped_metrics(trades, columns)


def loss_diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    return v109.loss_diagnostics(trades)


def build_setup_trade_matrix(ledger: pd.DataFrame, setup_signal_matrix: pd.DataFrame) -> pd.DataFrame:
    if setup_signal_matrix.empty:
        return setup_signal_matrix.copy()
    output = setup_signal_matrix.copy()
    if ledger.empty or "setup_id" not in ledger.columns:
        for prefix in ("direct", "local_break", "pullback_reclaim"):
            output[f"{prefix}_trade_executed"] = False
            output[f"{prefix}_net_R"] = np.nan
            output[f"{prefix}_exit_reason"] = ""
        return output
    mapping = {
        "direct": "state_direct_after_1h_confirmation_rr2_0",
        "local_break": "state_15m_single_local_break_rr2_0",
        "pullback_reclaim": "state_15m_pullback_reclaim_rr2_0",
    }
    for prefix, channel_id in mapping.items():
        frame = ledger.loc[ledger["channel_id"] == channel_id, ["setup_id", "net_r", "exit_reason"]].copy()
        if not frame.empty:
            frame["setup_id"] = frame["setup_id"].astype(str)
            frame = frame.drop_duplicates("setup_id", keep="first").set_index("setup_id")
            output[f"{prefix}_trade_executed"] = output["setup_id"].astype(str).isin(frame.index)
            output[f"{prefix}_net_R"] = output["setup_id"].astype(str).map(frame["net_r"])
            output[f"{prefix}_exit_reason"] = output["setup_id"].astype(str).map(frame["exit_reason"]).fillna("")
        else:
            output[f"{prefix}_trade_executed"] = False
            output[f"{prefix}_net_R"] = np.nan
            output[f"{prefix}_exit_reason"] = ""
    return output


def run_benchmark(raw_5m: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    signal_map, signal_audit, funnel, state_events, setup_matrix, coverage = build_all_signals(raw_5m, config)
    h4 = v108.four_hour_environment(v108.add_features(v108.resample_ohlcv(raw_5m, 240)), config)
    trade_frames: list[pd.DataFrame] = []
    qualification: dict[str, Any] = {}
    for spec in CHANNELS:
        signals = signal_map[spec.channel_id]
        if spec.is_baseline:
            trades = V103.execute_channel(raw_5m, signals, spec, config)
            trades = v108.enrich_baseline_path_diagnostics(raw_5m, trades)
            if not trades.empty:
                trades["management_mode"] = "V10_6_FROZEN"
                trades["setup_id"] = trades["signal_time_utc"].astype(str)
        else:
            trades = v109.execute_strict_channel(
                raw_5m,
                signals,
                spec,
                config,
                h4,
                break_even_shadow=False,
            )
        trade_frames.append(trades)
        qualification[spec.channel_id] = summarize_channel(trades, spec, config)

    nonempty = [frame for frame in trade_frames if not frame.empty]
    ledger = pd.concat(nonempty, ignore_index=True, sort=False) if nonempty else pd.DataFrame()
    if not ledger.empty:
        ledger = ledger.sort_values(["channel_id", "entry_time_utc"]).reset_index(drop=True)
    summary = pd.DataFrame(
        [{key: value for key, value in audit.items() if key != "checks"} for audit in qualification.values()]
    ).sort_values(["historical_research_gate_pass", "profit_factor", "net_R"], ascending=[False, False, False]).reset_index(drop=True)
    setup_trade_matrix = build_setup_trade_matrix(ledger, setup_matrix)
    return {
        "signals": signal_map,
        "signal_audit": signal_audit,
        "signal_funnel": funnel,
        "state_machine_events": state_events,
        "setup_signal_matrix": setup_matrix,
        "trigger_coverage": coverage,
        "setup_trade_matrix": setup_trade_matrix,
        "trades": ledger,
        "summary": summary,
        "qualification": qualification,
        "direction_summary": grouped_metrics(ledger, ["channel_id", "channel_label", "direction"]),
        "monthly_summary": grouped_metrics(ledger, ["channel_id", "channel_label", "month"]),
        "trigger_type_summary": grouped_metrics(ledger, ["channel_id", "channel_label", "setup_type", "trigger_type"]),
        "exit_reason_summary": grouped_metrics(ledger, ["channel_id", "channel_label", "exit_reason"]),
        "loss_diagnostics": loss_diagnostics(ledger),
    }


def parameter_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "release": config["release"],
        "channels": [asdict(spec) for spec in CHANNELS],
        "multi_timeframe_parameters": config["multi_timeframe_parameters"],
        "state_machine_parameters": config["state_machine_parameters"],
        "trigger_comparison_parameters": config["trigger_comparison_parameters"],
        "execution": config["execution"],
        "evaluation_window": config["evaluation_window"],
        "no_lookahead_rules": config["no_lookahead_rules"],
    }


def _trigger_fixture_test(config: dict[str, Any]) -> None:
    setup_time = pd.Timestamp("2026-02-01T10:00:00Z")
    setup = pd.DataFrame([
        {
            "setup_id": "LEG0001-C01-20260201T1000",
            "setup_time": setup_time,
            "direction": 1,
            "environment_leg_id": 1,
            "cycle_number": 1,
            "impulse_time": setup_time - pd.Timedelta(hours=3),
            "pullback_time": setup_time - pd.Timedelta(hours=1),
            "confirmation_level": 102.0,
            "close": 101.8,
            "atr14": 1.0,
            "adx14": 25.0,
            "rsi14": 56.0,
            "plus_di14": 30.0,
            "minus_di14": 15.0,
            "ema_separation_atr": 1.0,
            "ema_slope_3_atr": 0.1,
            "volume_ratio": 1.0,
            "clv_long": 0.8,
            "clv_short": 0.2,
            "channel_width_atr": 4.0,
            "adx_change_3": 0.2,
            "ema50_slope_6_atr": 0.1,
            "price_extension_ema20_atr": 0.7,
            "atr_ratio": 0.01,
            "trend_age_long": 8.0,
            "trend_age_short": 0.0,
        }
    ])
    index = pd.date_range(setup_time + pd.Timedelta(minutes=15), periods=12, freq="15min")
    rows = []
    for i, ts in enumerate(index):
        close = 101.9 + 0.02 * i
        rows.append({
            "open": close - 0.05,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": 100.0,
            "ema9": 101.8,
            "ema20": 101.75,
            "ema50": 101.0,
            "atr14": 0.5,
            "plus_di14": 30.0,
            "minus_di14": 15.0,
            "adx14": 25.0,
            "rsi14": 55.0,
            "channel_high_8": 0.0,
            "channel_low_8": 0.0,
            "channel_high_20": 0.0,
            "channel_low_20": 0.0,
            "volume_mean_20": 100.0,
            "ema_separation_atr": 1.0,
            "ema_slope_3_atr": 0.1,
            "ema50_slope_3_atr": 0.05,
            "ema50_slope_6_atr": 0.1,
            "channel_width_atr": 3.0,
            "clv_long": 0.75,
            "clv_short": 0.25,
            "body_atr": 0.2,
            "volume_ratio": 1.0,
            "adx_change_3": 0.1,
            "price_extension_ema20_atr": 0.4,
            "atr_ratio": 0.005,
            "trend_age_long": 5.0,
            "trend_age_short": 0.0,
            "close_above_ema20_count_3": 3.0,
            "close_below_ema20_count_3": 0.0,
            "ema9_cross_up_ema20": False,
            "ema9_cross_down_ema20": False,
            "break_high_4": False,
            "break_low_4": False,
        })
    m15 = pd.DataFrame(rows, index=index)
    m15.index.name = "signal_time"
    # Local breakout at the second closed 15m bar.
    m15.iloc[1, m15.columns.get_loc("close")] = 102.15
    m15.iloc[1, m15.columns.get_loc("high")] = 102.25
    m15.iloc[1, m15.columns.get_loc("open")] = 101.95
    m15.iloc[1, m15.columns.get_loc("break_high_4")] = True
    # Later touch and next-bar reclaim for the pullback channel.
    m15.iloc[4, m15.columns.get_loc("low")] = 101.70
    m15.iloc[4, m15.columns.get_loc("close")] = 101.76
    m15.iloc[4, m15.columns.get_loc("open")] = 101.95
    m15.iloc[5, m15.columns.get_loc("open")] = 101.80
    m15.iloc[5, m15.columns.get_loc("close")] = 102.10
    m15.iloc[5, m15.columns.get_loc("high")] = 102.20
    m15.iloc[5, m15.columns.get_loc("low")] = 101.78
    h4_index = pd.date_range("2026-02-01T08:00:00Z", periods=3, freq="4h")
    h4 = pd.DataFrame({"environment_direction": [1, 1, 1]}, index=h4_index)
    h4.index.name = "signal_time"

    direct = build_direct_confirmation_signals(setup, CHANNEL_BY_ID["state_direct_after_1h_confirmation_rr2_0"], config)
    local = build_single_local_break_signals(m15, setup, h4, CHANNEL_BY_ID["state_15m_single_local_break_rr2_0"], config)
    reclaim = build_pullback_reclaim_signals(m15, setup, h4, CHANNEL_BY_ID["state_15m_pullback_reclaim_rr2_0"], config)
    assert len(direct) == 1
    assert len(local) == 1
    assert len(reclaim) == 1
    assert pd.Timestamp(direct.iloc[0]["signal_time"]) == setup_time
    assert pd.Timestamp(local.iloc[0]["signal_time"]) > setup_time
    assert pd.Timestamp(reclaim.iloc[0]["signal_time"]) > pd.Timestamp(reclaim.iloc[0]["trigger_touch_time"])
    assert set(direct["setup_id"]) == set(local["setup_id"]) == set(reclaim["setup_id"])


def self_test(config: dict[str, Any]) -> None:
    _trigger_fixture_test(config)
    assert len(CHANNELS) == 4
    assert sum(spec.is_baseline for spec in CHANNELS) == 1
    raw = synthetic_5m_data(220_000, seed=20261010)
    signal_map, audit, funnel, events, setup_matrix, coverage = build_all_signals(raw, config)
    assert set(signal_map) == set(CHANNEL_BY_ID)
    assert audit["resample"]["no_lookahead_alignment"] == "BACKWARD_ASOF_ONLY_CLOSED_BARS"
    assert audit["state_machine"]["upper_state_machine_identical_to_v10_9"] is True
    assert audit["trigger_comparison"]["all_channels_use_same_confirmation_pool"] is True
    assert set(funnel["stage"]) == {
        "ENVIRONMENT_LEG_START", "IMPULSE_QUALIFIED", "PULLBACK_QUALIFIED", "CONFIRMATION_EMITTED",
        "DIRECT_AFTER_1H_CONFIRMATION", "SINGLE_LOCAL_15M_BREAK", "PULLBACK_RECLAIM_15M",
    }
    assert len(coverage) == 3
    if not setup_matrix.empty:
        assert setup_matrix["setup_id"].is_unique
        assert setup_matrix["direct_triggered"].all()
    for channel_id in [
        "state_direct_after_1h_confirmation_rr2_0",
        "state_15m_single_local_break_rr2_0",
        "state_15m_pullback_reclaim_rr2_0",
    ]:
        frame = signal_map[channel_id]
        if not frame.empty:
            assert frame["setup_id"].is_unique
            assert set(frame["setup_id"].astype(str)).issubset(set(setup_matrix["setup_id"].astype(str)))
    result = run_benchmark(raw, config)
    assert len(result["summary"]) == 4
    if not result["trades"].empty:
        for channel_id, group in result["trades"].groupby("channel_id"):
            entries = pd.to_datetime(group["entry_time_utc"], utc=True).reset_index(drop=True)
            exits = pd.to_datetime(group["exit_time_utc"], utc=True).reset_index(drop=True)
            assert (exits >= entries).all(), channel_id
            if len(group) > 1:
                assert (entries.iloc[1:].reset_index(drop=True) >= exits.iloc[:-1].reset_index(drop=True)).all(), channel_id
    print("V110_ENGINE_SELF_TEST_OK")


__all__ = [
    "CHANNELS", "CHANNEL_BY_ID", "load_official_5m_data", "synthetic_5m_data", "run_benchmark",
    "parameter_manifest", "self_test", "metrics", "remove_best_fraction",
]
