from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

import _v108_engine as v108


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
        "strict_state_4h_1h_15m_fixed_rr2_0",
        "主实验·4H趋势腿+1H推动回踩状态机+15M突破 RR2.0",
        "strict_state_fixed_rr2_0",
        "BOTH",
        1.25,
        2.0,
        48,
        False,
    ),
    ChannelSpec(
        "strict_state_4h_1h_15m_be1_shadow_rr2_0",
        "影子管理·同信号达到1R后下一根5M启用保本 RR2.0",
        "strict_state_be1_shadow_rr2_0",
        "BOTH",
        1.25,
        2.0,
        48,
        False,
    ),
)

CHANNEL_BY_ID = {spec.channel_id: spec for spec in CHANNELS}


def synthetic_5m_data(rows: int = 220_000, seed: int = 1019) -> pd.DataFrame:
    return v108.synthetic_5m_data(rows, seed)


def load_official_5m_data(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    return v108.load_official_5m_data(config)


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
    return signals.loc[(times >= start) & (times <= end)].sort_values("signal_time").reset_index(drop=True)


def _directional_structure_valid(row: pd.Series, direction: int, tolerance_atr: float) -> bool:
    atr = float(row.get("atr14", np.nan))
    if not math.isfinite(atr) or atr <= 0:
        return False
    if direction == 1:
        return bool(
            row.get("h4_environment_direction", 0) == 1
            and row.get("ema20", np.nan) > row.get("ema50", np.nan)
            and row.get("close", np.nan) >= row.get("ema50", np.nan) - tolerance_atr * atr
        )
    return bool(
        row.get("h4_environment_direction", 0) == -1
        and row.get("ema20", np.nan) < row.get("ema50", np.nan)
        and row.get("close", np.nan) <= row.get("ema50", np.nan) + tolerance_atr * atr
    )


def build_state_machine_setups(
    h1: pd.DataFrame,
    h4_env: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = config["state_machine_parameters"]["one_hour_cycle"]
    h4_columns = ["environment_direction", "adx14", "ema50_slope_3_atr"]
    x = v108._asof_join(h1, h4_env, h4_columns, "h4_").copy()
    x["channel_high_12"] = x["high"].shift(1).rolling(int(p["impulse_breakout_lookback_hours"])).max()
    x["channel_low_12"] = x["low"].shift(1).rolling(int(p["impulse_breakout_lookback_hours"])).min()

    min_body = float(p["minimum_impulse_body_atr"])
    min_clv = float(p["minimum_impulse_close_location_value"])
    min_volume = float(p["minimum_impulse_volume_ratio"])
    min_extension = float(p["minimum_impulse_extension_ema20_atr"])
    max_extension = float(p["maximum_impulse_extension_ema20_atr"])
    min_retrace = float(p["minimum_pullback_retracement_atr"])
    touch_zone = float(p["pullback_touch_ema20_zone_atr"])
    structure_tolerance = float(p["ema50_structure_tolerance_atr"])
    max_wait_pullback = int(p["maximum_hours_impulse_to_pullback"])
    max_wait_confirm = int(p["maximum_hours_pullback_to_confirmation"])
    confirm_body = float(p["minimum_confirmation_body_atr"])
    confirm_clv = float(p["minimum_confirmation_close_location_value"])
    confirm_extension = float(p["maximum_confirmation_extension_ema20_atr"])
    cooldown_hours = int(p["cycle_cooldown_hours"])
    max_cycles_per_leg = int(p["maximum_cycles_per_4h_environment_leg"])

    current_env = 0
    environment_leg_id = 0
    cycle_number = 0
    state = "WAIT_IMPULSE"
    impulse_index = -1
    impulse_time: pd.Timestamp | None = None
    impulse_extreme = np.nan
    impulse_atr = np.nan
    pullback_time: pd.Timestamp | None = None
    pullback_extreme = np.nan
    cooldown_until_index = -1

    setup_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    environment_legs: list[dict[str, Any]] = []

    def event(timestamp: pd.Timestamp, direction: int, stage: str, reason: str = "", **extra: Any) -> None:
        event_rows.append(
            {
                "event_time_utc": timestamp.isoformat(),
                "direction": int(direction),
                "environment_leg_id": int(environment_leg_id),
                "cycle_number": int(cycle_number),
                "stage": stage,
                "reason": reason,
                **extra,
            }
        )

    for position, (timestamp, row) in enumerate(x.iterrows()):
        timestamp = pd.Timestamp(timestamp)
        env_value = row.get("h4_environment_direction", 0)
        env = 0 if pd.isna(env_value) else int(env_value)
        if env != current_env:
            if env != 0:
                environment_leg_id += 1
                environment_legs.append(
                    {
                        "environment_leg_id": environment_leg_id,
                        "direction": env,
                        "start_time_utc": timestamp.isoformat(),
                    }
                )
                event(timestamp, env, "ENVIRONMENT_LEG_START")
            current_env = env
            cycle_number = 0
            state = "WAIT_IMPULSE"
            impulse_index = -1
            impulse_time = None
            impulse_extreme = np.nan
            impulse_atr = np.nan
            pullback_time = None
            pullback_extreme = np.nan
            cooldown_until_index = -1

        if env == 0:
            continue
        direction = env
        atr = float(row.get("atr14", np.nan))
        if not math.isfinite(atr) or atr <= 0:
            continue

        structure_valid = _directional_structure_valid(row, direction, structure_tolerance)
        if not structure_valid:
            if state not in ("WAIT_IMPULSE", "COOLDOWN"):
                event(timestamp, direction, "CYCLE_INVALIDATED", "ONE_HOUR_STRUCTURE_BROKEN")
            state = "WAIT_IMPULSE"
            impulse_index = -1
            impulse_time = None
            pullback_time = None
            continue

        if state == "COOLDOWN":
            if position < cooldown_until_index:
                continue
            state = "WAIT_IMPULSE"

        if state == "WAIT_IMPULSE":
            if cycle_number >= max_cycles_per_leg:
                continue
            directional_body = row["close"] > row["open"] if direction == 1 else row["close"] < row["open"]
            breakout = row["close"] > row["channel_high_12"] if direction == 1 else row["close"] < row["channel_low_12"]
            clv = float(row["clv_long"] if direction == 1 else row["clv_short"])
            extension = float(row["price_extension_ema20_atr"])
            impulse_ok = bool(
                breakout
                and directional_body
                and float(row["body_atr"]) >= min_body
                and clv >= min_clv
                and float(row["volume_ratio"]) >= min_volume
                and min_extension <= extension <= max_extension
            )
            if not impulse_ok:
                continue
            cycle_number += 1
            impulse_index = position
            impulse_time = timestamp
            impulse_extreme = float(row["high"] if direction == 1 else row["low"])
            impulse_atr = atr
            pullback_time = None
            pullback_extreme = float(row["low"] if direction == 1 else row["high"])
            state = "WAIT_PULLBACK"
            event(
                timestamp,
                direction,
                "IMPULSE_QUALIFIED",
                impulse_price=float(row["close"]),
                impulse_extreme=impulse_extreme,
                impulse_atr=impulse_atr,
            )
            continue

        if state == "WAIT_PULLBACK":
            bars_from_impulse = position - impulse_index
            if bars_from_impulse > max_wait_pullback:
                event(timestamp, direction, "CYCLE_EXPIRED", "PULLBACK_NOT_FOUND_IN_TIME")
                state = "WAIT_IMPULSE"
                continue
            if direction == 1:
                impulse_extreme = max(float(impulse_extreme), float(row["high"]))
                pullback_extreme = min(float(pullback_extreme), float(row["low"]))
                retracement_atr = (impulse_extreme - float(row["low"])) / atr
                touched = float(row["low"]) <= float(row["ema20"]) + touch_zone * atr
                retracing_bar = float(row["close"]) <= float(row["close"] if position == 0 else x.iloc[position - 1]["close"])
            else:
                impulse_extreme = min(float(impulse_extreme), float(row["low"]))
                pullback_extreme = max(float(pullback_extreme), float(row["high"]))
                retracement_atr = (float(row["high"]) - impulse_extreme) / atr
                retracing_bar = float(row["close"]) >= float(row["close"] if position == 0 else x.iloc[position - 1]["close"])
                touched = float(row["high"]) >= float(row["ema20"]) - touch_zone * atr
            if bars_from_impulse >= 1 and touched and retracing_bar and retracement_atr >= min_retrace:
                pullback_time = timestamp
                state = "WAIT_CONFIRMATION"
                event(
                    timestamp,
                    direction,
                    "PULLBACK_QUALIFIED",
                    retracement_atr=float(retracement_atr),
                    impulse_extreme=float(impulse_extreme),
                    pullback_extreme=float(pullback_extreme),
                )
            continue

        if state == "WAIT_CONFIRMATION":
            assert pullback_time is not None
            bars_from_pullback = int((timestamp - pullback_time) / pd.Timedelta(hours=1))
            if bars_from_pullback > max_wait_confirm:
                event(timestamp, direction, "CYCLE_EXPIRED", "CONFIRMATION_NOT_FOUND_IN_TIME")
                state = "WAIT_IMPULSE"
                continue
            previous = x.iloc[position - 1] if position > 0 else row
            if direction == 1:
                pullback_extreme = min(float(pullback_extreme), float(row["low"]))
                confirmation = bool(
                    row["close"] > row["ema20"]
                    and row["close"] > previous["high"]
                    and row["close"] > row["open"]
                    and row["body_atr"] >= confirm_body
                    and row["clv_long"] >= confirm_clv
                    and row["price_extension_ema20_atr"] <= confirm_extension
                    and row["plus_di14"] >= row["minus_di14"]
                )
                confirmation_level = float(row["high"])
            else:
                pullback_extreme = max(float(pullback_extreme), float(row["high"]))
                confirmation = bool(
                    row["close"] < row["ema20"]
                    and row["close"] < previous["low"]
                    and row["close"] < row["open"]
                    and row["body_atr"] >= confirm_body
                    and row["clv_short"] >= confirm_clv
                    and row["price_extension_ema20_atr"] <= confirm_extension
                    and row["minus_di14"] >= row["plus_di14"]
                )
                confirmation_level = float(row["low"])
            if not confirmation:
                continue

            setup_id = f"LEG{environment_leg_id:04d}-C{cycle_number:02d}-{timestamp.strftime('%Y%m%dT%H%M')}"
            setup = row.to_dict()
            setup.update(
                {
                    "setup_id": setup_id,
                    "setup_time": timestamp,
                    "direction": direction,
                    "setup_type": "IMPULSE_PULLBACK_CONFIRM",
                    "environment_leg_id": environment_leg_id,
                    "cycle_number": cycle_number,
                    "impulse_time": impulse_time,
                    "pullback_time": pullback_time,
                    "impulse_extreme": float(impulse_extreme),
                    "pullback_extreme": float(pullback_extreme),
                    "confirmation_level": confirmation_level,
                    "h4_environment_direction": direction,
                }
            )
            setup_rows.append(setup)
            event(
                timestamp,
                direction,
                "CONFIRMATION_EMITTED",
                setup_id=setup_id,
                confirmation_level=confirmation_level,
            )
            state = "COOLDOWN"
            cooldown_until_index = position + cooldown_hours
            continue

    setup_frame = pd.DataFrame(setup_rows)
    if not setup_frame.empty:
        setup_frame = setup_frame.sort_values("setup_time").drop_duplicates("setup_id").reset_index(drop=True)
    event_frame = pd.DataFrame(event_rows)
    leg_frame = pd.DataFrame(environment_legs)
    return setup_frame, event_frame, leg_frame


def build_strict_precision_signals(
    m15: pd.DataFrame,
    setups: pd.DataFrame,
    h4_env: pd.DataFrame,
    spec: ChannelSpec,
    config: dict[str, Any],
) -> pd.DataFrame:
    if setups.empty:
        return pd.DataFrame()
    p = config["state_machine_parameters"]["fifteen_minute_trigger"]
    execution = config["state_machine_parameters"]["execution"]
    minimum_delay = pd.Timedelta(minutes=int(p["minimum_delay_after_confirmation_minutes"]))
    validity = pd.Timedelta(hours=float(p["setup_validity_hours"]))
    min_volume = float(p["minimum_volume_ratio"])
    min_clv = float(p["minimum_close_location_value"])
    max_extension = float(p["maximum_price_extension_ema20_atr"])

    m15_joined = v108._asof_join(m15, h4_env, ["environment_direction"], "h4_")
    rows: list[dict[str, Any]] = []
    for setup in setups.itertuples(index=False):
        setup_time = pd.Timestamp(setup.setup_time)
        direction = int(setup.direction)
        start = setup_time + minimum_delay
        end = setup_time + validity
        candidates = m15_joined.loc[(m15_joined.index >= start) & (m15_joined.index <= end)].copy()
        if candidates.empty:
            continue
        confirmation_level = float(setup.confirmation_level)
        if direction == 1:
            candidates["trigger_ok"] = (
                (candidates["h4_environment_direction"] == 1)
                & candidates["break_high_4"].fillna(False)
                & (candidates["close"] > confirmation_level)
                & (candidates["close"] > candidates["ema20"])
                & (candidates["close"] > candidates["open"])
                & (candidates["ema9"] > candidates["ema20"])
                & (candidates["volume_ratio"] >= min_volume)
                & (candidates["clv_long"] >= min_clv)
                & (candidates["price_extension_ema20_atr"] <= max_extension)
            )
            trigger_type = "CONFIRM_HIGH_AND_LOCAL_BREAK"
        else:
            candidates["trigger_ok"] = (
                (candidates["h4_environment_direction"] == -1)
                & candidates["break_low_4"].fillna(False)
                & (candidates["close"] < confirmation_level)
                & (candidates["close"] < candidates["ema20"])
                & (candidates["close"] < candidates["open"])
                & (candidates["ema9"] < candidates["ema20"])
                & (candidates["volume_ratio"] >= min_volume)
                & (candidates["clv_short"] >= min_clv)
                & (candidates["price_extension_ema20_atr"] <= max_extension)
            )
            trigger_type = "CONFIRM_LOW_AND_LOCAL_BREAK"
        valid = candidates.loc[candidates["trigger_ok"]]
        if valid.empty:
            continue
        trigger_time = pd.Timestamp(valid.index[0])
        trigger = valid.iloc[0]
        source_label = "多头组件" if direction == 1 else "空头组件"
        row = {
            "signal_time": trigger_time,
            "direction": direction,
            "close": float(setup.close),
            "atr14": float(setup.atr14),
            "adx14": float(setup.adx14),
            "rsi14": float(setup.rsi14),
            "plus_di14": float(setup.plus_di14),
            "minus_di14": float(setup.minus_di14),
            "ema_separation_atr": float(setup.ema_separation_atr),
            "ema_slope_3_atr": float(setup.ema_slope_3_atr),
            "volume_ratio": float(setup.volume_ratio),
            "clv_long": float(setup.clv_long),
            "clv_short": float(setup.clv_short),
            "channel_width_atr": float(setup.channel_width_atr),
            "adx_change_3": float(setup.adx_change_3),
            "ema50_slope_6_atr": float(setup.ema50_slope_6_atr),
            "price_extension_ema20_atr": float(setup.price_extension_ema20_atr),
            "atr_ratio": float(setup.atr_ratio),
            "trend_age_long": float(setup.trend_age_long),
            "trend_age_short": float(setup.trend_age_short),
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
            "setup_id": str(setup.setup_id),
            "environment_leg_id": int(setup.environment_leg_id),
            "cycle_number": int(setup.cycle_number),
            "impulse_time": pd.Timestamp(setup.impulse_time),
            "pullback_time": pd.Timestamp(setup.pullback_time),
            "confirmation_level": confirmation_level,
            "trigger_volume_ratio": float(trigger.volume_ratio),
            "trigger_clv": float(trigger.clv_long if direction == 1 else trigger.clv_short),
            "trigger_extension_ema20_atr": float(trigger.price_extension_ema20_atr),
            "management_mode": "FIXED_STOP_TARGET",
        }
        rows.append(row)

    signals = pd.DataFrame(rows)
    if signals.empty:
        return signals
    signals = signals.sort_values(["signal_time", "direction"]).drop_duplicates("setup_id", keep="first")
    conflicts = signals.groupby("signal_time")["direction"].nunique()
    conflict_times = set(conflicts.loc[conflicts > 1].index)
    if conflict_times:
        signals = signals.loc[~signals["signal_time"].isin(conflict_times)]
    return _filter_evaluation_window(signals.reset_index(drop=True), config)


def _clone_signals_for_channel(signals: pd.DataFrame, spec: ChannelSpec, management_mode: str) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    cloned = signals.copy()
    cloned["channel_id"] = spec.channel_id
    cloned["channel_label"] = spec.label
    cloned["profile"] = spec.profile
    cloned["source_component_id"] = np.where(
        cloned["direction"] == 1,
        f"{spec.channel_id}_long",
        f"{spec.channel_id}_short",
    )
    cloned["source_component_label"] = np.where(
        cloned["direction"] == 1,
        f"{spec.label}·多头组件",
        f"{spec.label}·空头组件",
    )
    cloned["management_mode"] = management_mode
    return cloned


def build_all_signals(
    raw_5m: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    m15 = v108.add_features(v108.resample_ohlcv(raw_5m, 15))
    h1 = v108.add_features(v108.resample_ohlcv(raw_5m, 60))
    h4 = v108.four_hour_environment(v108.add_features(v108.resample_ohlcv(raw_5m, 240)), config)
    setups, state_events, environment_legs = build_state_machine_setups(h1, h4, config)

    baseline_spec = CHANNEL_BY_ID["baseline_1h_shared_v10_6"]
    fixed_spec = CHANNEL_BY_ID["strict_state_4h_1h_15m_fixed_rr2_0"]
    shadow_spec = CHANNEL_BY_ID["strict_state_4h_1h_15m_be1_shadow_rr2_0"]
    baseline = v108.build_baseline_signals(raw_5m, baseline_spec, config)
    if not baseline.empty:
        baseline["management_mode"] = "V10_6_FROZEN"
        baseline["setup_id"] = baseline["signal_time"].astype(str)
    fixed = build_strict_precision_signals(m15, setups, h4, fixed_spec, config)
    shadow = _clone_signals_for_channel(fixed, shadow_spec, "BE_AT_1R_NEXT_5M_BAR")
    fixed = _clone_signals_for_channel(fixed, fixed_spec, "FIXED_STOP_TARGET")

    signal_map = {
        baseline_spec.channel_id: baseline,
        fixed_spec.channel_id: fixed,
        shadow_spec.channel_id: shadow,
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
    setup_direction_counts = setups.groupby("direction").size().to_dict() if not setups.empty else {}
    trigger_direction_counts = fixed.groupby("direction").size().to_dict() if not fixed.empty else {}
    stage_rows = []
    for stage in ["ENVIRONMENT_LEG_START", "IMPULSE_QUALIFIED", "PULLBACK_QUALIFIED", "CONFIRMATION_EMITTED"]:
        subset = events.loc[events["stage"] == stage] if not events.empty else pd.DataFrame()
        stage_rows.append(
            {
                "stage": stage,
                "long_count": int((subset["direction"] == 1).sum()) if not subset.empty else 0,
                "short_count": int((subset["direction"] == -1).sum()) if not subset.empty else 0,
            }
        )
    stage_rows.append(
        {
            "stage": "STRICT_15M_TRIGGER",
            "long_count": int(trigger_direction_counts.get(1, 0)),
            "short_count": int(trigger_direction_counts.get(-1, 0)),
        }
    )
    funnel = pd.DataFrame(stage_rows)
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
            "confirmation_setups": int(len(setups)),
            "strict_triggers": int(len(fixed)),
            "one_setup_per_cycle": bool(setups["setup_id"].is_unique) if not setups.empty else True,
            "fixed_and_shadow_signals_identical": bool(
                fixed[["signal_time", "direction", "setup_id"]].reset_index(drop=True).equals(
                    shadow[["signal_time", "direction", "setup_id"]].reset_index(drop=True)
                )
            ) if not fixed.empty or not shadow.empty else True,
        },
        "evaluation_window": config["evaluation_window"],
    }
    return signal_map, audit, funnel, events


def _h4_regime_at_raw_times(raw_5m: pd.DataFrame, h4_env: pd.DataFrame) -> np.ndarray:
    return v108._h4_regime_at_raw_times(raw_5m, h4_env)


def execute_strict_channel(
    raw_5m: pd.DataFrame,
    signals: pd.DataFrame,
    spec: ChannelSpec,
    config: dict[str, Any],
    h4_env: pd.DataFrame,
    break_even_shadow: bool,
) -> pd.DataFrame:
    columns = [
        "channel_id", "channel_label", "profile", "direction_scope", "source_component_id", "source_component_label",
        "reward_risk_target", "stop_atr_multiple", "max_holding_hours", "signal_time_utc", "entry_time_utc",
        "exit_time_utc", "direction", "entry_price", "stop_price", "target_price", "exit_price", "stop_distance",
        "gross_r", "fee_r", "net_r", "win", "exit_reason", "holding_5m_bars", "signal_atr", "signal_adx",
        "signal_rsi", "signal_plus_di", "signal_minus_di", "signal_ema_separation_atr", "signal_ema_slope_3_atr",
        "signal_volume_ratio", "signal_clv", "signal_channel_width_atr", "signal_adx_change_3", "signal_ema50_slope_6_atr",
        "signal_price_extension_ema20_atr", "signal_atr_ratio", "signal_trend_age", "month", "phase", "is_baseline",
        "setup_type", "trigger_type", "setup_time_utc", "setup_id", "environment_leg_id", "cycle_number",
        "management_mode", "break_even_activated", "break_even_activation_bar", "mfe_R", "mae_R", "bars_to_mfe", "bars_to_mae",
    ]
    if signals.empty:
        return pd.DataFrame(columns=columns)

    fee_rate = float(config["execution"]["fee_rate_per_side"])
    slippage = float(config["execution"]["tick_size"]) * int(config["execution"]["slippage_ticks_per_fill"])
    min_stop_pct = float(config["execution"]["minimum_stop_distance_pct"])
    be_activation_r = float(config["state_machine_parameters"]["shadow_management"]["break_even_activation_R"])
    times = raw_5m["time"].to_numpy(dtype="datetime64[ns]")
    opens = raw_5m["open"].to_numpy(float)
    highs = raw_5m["high"].to_numpy(float)
    lows = raw_5m["low"].to_numpy(float)
    closes = raw_5m["close"].to_numpy(float)
    regime = _h4_regime_at_raw_times(raw_5m, h4_env)
    rows: list[dict[str, Any]] = []
    next_free_index = 0

    for signal in signals.sort_values("signal_time").itertuples(index=False):
        signal_time = pd.Timestamp(signal.signal_time)
        entry_i = int(np.searchsorted(times, signal_time.to_datetime64(), side="left"))
        if entry_i < next_free_index or entry_i >= len(raw_5m):
            continue
        direction = int(signal.direction)
        if regime[entry_i] not in (0, direction):
            continue
        entry_price = float(opens[entry_i] + slippage * direction)
        signal_atr = float(signal.atr14)
        if not math.isfinite(signal_atr) or signal_atr <= 0:
            continue
        stop_multiple = float(signal.stop_atr_multiple)
        reward_risk = float(signal.reward_risk)
        max_holding_hours = int(signal.max_holding_hours)
        stop_distance = max(signal_atr * stop_multiple, entry_price * min_stop_pct)
        stop_price = entry_price - direction * stop_distance
        target_price = entry_price + direction * reward_risk * stop_distance
        final_i = min(entry_i + max_holding_hours * 12 - 1, len(raw_5m) - 1)
        exit_i = final_i
        exit_reason = "TIME_EXIT"
        raw_exit = float(closes[final_i])
        mfe_r = -np.inf
        mae_r = -np.inf
        bars_to_mfe = 0
        bars_to_mae = 0
        break_even_active = False
        break_even_activated = False
        break_even_activation_bar = 0

        for bar_i in range(entry_i, final_i + 1):
            favorable = (highs[bar_i] - entry_price) / stop_distance if direction == 1 else (entry_price - lows[bar_i]) / stop_distance
            adverse = (entry_price - lows[bar_i]) / stop_distance if direction == 1 else (highs[bar_i] - entry_price) / stop_distance
            if favorable > mfe_r:
                mfe_r = float(favorable)
                bars_to_mfe = int(bar_i - entry_i + 1)
            if adverse > mae_r:
                mae_r = float(adverse)
                bars_to_mae = int(bar_i - entry_i + 1)

            if bar_i > entry_i and regime[bar_i] == -direction:
                exit_i, exit_reason, raw_exit = bar_i, "H4_TREND_REVERSAL", float(opens[bar_i])
                break

            if direction == 1:
                initial_stop_hit = lows[bar_i] <= stop_price
                target_hit = highs[bar_i] >= target_price
                break_even_hit = break_even_active and lows[bar_i] <= entry_price
            else:
                initial_stop_hit = highs[bar_i] >= stop_price
                target_hit = lows[bar_i] <= target_price
                break_even_hit = break_even_active and highs[bar_i] >= entry_price

            if initial_stop_hit:
                exit_i, exit_reason, raw_exit = bar_i, "STOP", stop_price
                break
            if break_even_hit:
                exit_i, exit_reason, raw_exit = bar_i, "BREAK_EVEN_FIRST_CONSERVATIVE", entry_price
                break
            if target_hit:
                exit_i, exit_reason, raw_exit = bar_i, "TARGET", target_price
                break

            # The threshold may be touched intrabar, but the break-even stop becomes active only on the next 5m bar.
            if break_even_shadow and not break_even_active and favorable >= be_activation_r:
                break_even_active = True
                break_even_activated = True
                break_even_activation_bar = int(bar_i - entry_i + 2)

        exit_price = float(raw_exit - slippage * direction)
        gross_r = direction * (exit_price - entry_price) / stop_distance
        fee_r = fee_rate * (entry_price + exit_price) / stop_distance
        net_r = gross_r - fee_r
        entry_time = pd.Timestamp(raw_5m.iloc[entry_i]["time"])
        exit_time = pd.Timestamp(raw_5m.iloc[exit_i]["time"]) + pd.Timedelta(minutes=5)
        rows.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "profile": spec.profile,
                "direction_scope": spec.direction_scope,
                "source_component_id": getattr(signal, "source_component_id", spec.channel_id),
                "source_component_label": getattr(signal, "source_component_label", spec.label),
                "reward_risk_target": reward_risk,
                "stop_atr_multiple": stop_multiple,
                "max_holding_hours": max_holding_hours,
                "signal_time_utc": signal_time.isoformat(),
                "entry_time_utc": entry_time.isoformat(),
                "exit_time_utc": exit_time.isoformat(),
                "direction": direction,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "exit_price": exit_price,
                "stop_distance": stop_distance,
                "gross_r": gross_r,
                "fee_r": fee_r,
                "net_r": net_r,
                "win": bool(net_r > 0),
                "exit_reason": exit_reason,
                "holding_5m_bars": int(exit_i - entry_i + 1),
                "signal_atr": signal_atr,
                "signal_adx": float(signal.adx14),
                "signal_rsi": float(signal.rsi14),
                "signal_plus_di": float(signal.plus_di14),
                "signal_minus_di": float(signal.minus_di14),
                "signal_ema_separation_atr": float(signal.ema_separation_atr),
                "signal_ema_slope_3_atr": float(signal.ema_slope_3_atr),
                "signal_volume_ratio": float(signal.volume_ratio),
                "signal_clv": float(signal.clv_long if direction == 1 else signal.clv_short),
                "signal_channel_width_atr": float(signal.channel_width_atr),
                "signal_adx_change_3": float(signal.adx_change_3),
                "signal_ema50_slope_6_atr": float(signal.ema50_slope_6_atr),
                "signal_price_extension_ema20_atr": float(signal.price_extension_ema20_atr),
                "signal_atr_ratio": float(signal.atr_ratio),
                "signal_trend_age": float(signal.trend_age_long if direction == 1 else signal.trend_age_short),
                "month": entry_time.strftime("%Y-%m"),
                "phase": "evaluation_2026_h1",
                "is_baseline": False,
                "setup_type": getattr(signal, "setup_type", ""),
                "trigger_type": getattr(signal, "trigger_type", ""),
                "setup_time_utc": pd.Timestamp(getattr(signal, "setup_time", signal_time)).isoformat(),
                "setup_id": getattr(signal, "setup_id", ""),
                "environment_leg_id": int(getattr(signal, "environment_leg_id", 0)),
                "cycle_number": int(getattr(signal, "cycle_number", 0)),
                "management_mode": getattr(signal, "management_mode", ""),
                "break_even_activated": bool(break_even_activated),
                "break_even_activation_bar": int(break_even_activation_bar),
                "mfe_R": float(max(mfe_r, 0.0)),
                "mae_R": float(max(mae_r, 0.0)),
                "bars_to_mfe": bars_to_mfe,
                "bars_to_mae": bars_to_mae,
            }
        )
        next_free_index = exit_i + 1

    return pd.DataFrame(rows, columns=columns)


def metrics(trades: pd.DataFrame | Iterable[dict[str, Any]]) -> dict[str, float]:
    return v108.metrics(trades)


def remove_best_fraction(trades: pd.DataFrame, fraction: float = 0.10) -> tuple[pd.DataFrame, dict[str, Any]]:
    return v108.remove_best_fraction(trades, fraction)


def summarize_channel(trades: pd.DataFrame, spec: ChannelSpec, config: dict[str, Any]) -> dict[str, Any]:
    m = metrics(trades)
    _, tail = remove_best_fraction(trades, float(config["evaluation_thresholds"]["remove_best_fraction"]))
    thresholds = config["evaluation_thresholds"]
    checks = {
        "minimum_trades": m["trades"] >= int(thresholds["minimum_trades"]),
        "minimum_win_rate": m["win_rate"] >= float(thresholds["minimum_win_rate"]),
        "minimum_avg_win_loss_ratio": m["avg_win_loss_ratio"] >= float(thresholds["minimum_avg_win_loss_ratio"]),
        "minimum_profit_factor": m["profit_factor"] >= float(thresholds["minimum_profit_factor"]),
        "maximum_drawdown_R": m["max_drawdown_R"] <= float(thresholds["maximum_drawdown_R"]),
        "best_10pct_removed_still_profitable": tail["net_R"] > 0,
    }
    return {
        "channel_id": spec.channel_id,
        "channel_label": spec.label,
        "profile": spec.profile,
        "is_baseline": spec.is_baseline,
        **m,
        "best_10pct_removed_net_R": tail["net_R"],
        "checks_passed": int(sum(checks.values())),
        "checks_total": int(len(checks)),
        "historical_research_gate_pass": bool(all(checks.values())),
        "checks": checks,
    }


def grouped_metrics(trades: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return v108.grouped_metrics(trades, columns)


def loss_diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    return v108.loss_diagnostics(trades)


def management_comparison(ledger: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "metric", "fixed_value", "break_even_shadow_value", "delta_shadow_minus_fixed",
    ]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    fixed = ledger.loc[ledger["channel_id"] == "strict_state_4h_1h_15m_fixed_rr2_0"]
    shadow = ledger.loc[ledger["channel_id"] == "strict_state_4h_1h_15m_be1_shadow_rr2_0"]
    fm = metrics(fixed)
    sm = metrics(shadow)
    rows = []
    for key in ["trades", "win_rate", "avg_win_loss_ratio", "profit_factor", "net_R", "max_drawdown_R", "expectancy_R"]:
        rows.append(
            {
                "metric": key,
                "fixed_value": float(fm[key]),
                "break_even_shadow_value": float(sm[key]),
                "delta_shadow_minus_fixed": float(sm[key] - fm[key]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def run_benchmark(raw_5m: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    signal_map, signal_audit, funnel, state_events = build_all_signals(raw_5m, config)
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
            trades = execute_strict_channel(
                raw_5m,
                signals,
                spec,
                config,
                h4,
                break_even_shadow=spec.channel_id.endswith("be1_shadow_rr2_0"),
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
    return {
        "signals": signal_map,
        "signal_audit": signal_audit,
        "signal_funnel": funnel,
        "state_machine_events": state_events,
        "trades": ledger,
        "summary": summary,
        "qualification": qualification,
        "direction_summary": grouped_metrics(ledger, ["channel_id", "channel_label", "direction"]),
        "monthly_summary": grouped_metrics(ledger, ["channel_id", "channel_label", "month"]),
        "setup_type_summary": grouped_metrics(ledger, ["channel_id", "channel_label", "setup_type", "trigger_type"]),
        "exit_reason_summary": grouped_metrics(ledger, ["channel_id", "channel_label", "exit_reason"]),
        "loss_diagnostics": loss_diagnostics(ledger),
        "management_comparison": management_comparison(ledger),
    }


def parameter_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "release": config["release"],
        "channels": [asdict(spec) for spec in CHANNELS],
        "state_machine_parameters": config["state_machine_parameters"],
        "execution": config["execution"],
        "evaluation_window": config["evaluation_window"],
        "no_lookahead_rules": config["no_lookahead_rules"],
    }



def _state_machine_fixture_test(config: dict[str, Any]) -> None:
    index = pd.date_range("2026-01-01T01:00:00Z", periods=24, freq="1h")
    rows: list[dict[str, Any]] = []
    for i, ts in enumerate(index):
        base_close = 100.0 + 0.05 * i
        row = {
            "open": base_close - 0.05, "high": base_close + 0.15, "low": base_close - 0.15, "close": base_close,
            "volume": 100.0, "ema9": 100.0, "ema20": 100.0, "ema50": 99.0, "atr14": 1.0,
            "plus_di14": 30.0, "minus_di14": 15.0, "adx14": 25.0, "rsi14": 55.0,
            "ema_separation_atr": 1.0, "ema_slope_3_atr": 0.1, "ema50_slope_3_atr": 0.05,
            "ema50_slope_6_atr": 0.1, "channel_width_atr": 3.0, "clv_long": 0.7, "clv_short": 0.3,
            "body_atr": 0.3, "volume_ratio": 1.0, "adx_change_3": 0.5, "price_extension_ema20_atr": 0.4,
            "atr_ratio": 0.01, "trend_age_long": 10.0, "trend_age_short": 0.0,
            "close_above_ema20_count_3": 3.0, "close_below_ema20_count_3": 0.0,
        }
        rows.append(row)
    h1 = pd.DataFrame(rows, index=index)
    h1.index.name = "signal_time"
    # Bar 15 is the impulse, bar 17 is the real pullback, and bar 18 confirms the cycle.
    h1.iloc[15, h1.columns.get_loc("open")] = 101.0
    h1.iloc[15, h1.columns.get_loc("close")] = 102.0
    h1.iloc[15, h1.columns.get_loc("high")] = 102.2
    h1.iloc[15, h1.columns.get_loc("low")] = 100.9
    h1.iloc[15, h1.columns.get_loc("body_atr")] = 1.0
    h1.iloc[15, h1.columns.get_loc("clv_long")] = 0.85
    h1.iloc[15, h1.columns.get_loc("volume_ratio")] = 1.2
    h1.iloc[15, h1.columns.get_loc("price_extension_ema20_atr")] = 1.0
    h1.iloc[16, h1.columns.get_loc("high")] = 102.4
    h1.iloc[16, h1.columns.get_loc("close")] = 101.7
    h1.iloc[17, h1.columns.get_loc("open")] = 101.5
    h1.iloc[17, h1.columns.get_loc("close")] = 100.25
    h1.iloc[17, h1.columns.get_loc("high")] = 101.6
    h1.iloc[17, h1.columns.get_loc("low")] = 99.95
    h1.iloc[17, h1.columns.get_loc("body_atr")] = 1.25
    h1.iloc[17, h1.columns.get_loc("price_extension_ema20_atr")] = 0.25
    h1.iloc[18, h1.columns.get_loc("open")] = 100.2
    h1.iloc[18, h1.columns.get_loc("close")] = 101.8
    h1.iloc[18, h1.columns.get_loc("high")] = 102.0
    h1.iloc[18, h1.columns.get_loc("low")] = 100.1
    h1.iloc[18, h1.columns.get_loc("body_atr")] = 1.6
    h1.iloc[18, h1.columns.get_loc("clv_long")] = 0.85
    h1.iloc[18, h1.columns.get_loc("price_extension_ema20_atr")] = 0.8

    h4_index = pd.date_range("2025-12-31T20:00:00Z", periods=8, freq="4h")
    h4 = pd.DataFrame(
        {"environment_direction": 1, "adx14": 25.0, "ema50_slope_3_atr": 0.05},
        index=h4_index,
    )
    h4.index.name = "signal_time"
    setups, events, _ = build_state_machine_setups(h1, h4, config)
    assert len(setups) == 1, setups
    assert setups.iloc[0]["direction"] == 1
    assert events.loc[events["stage"] == "IMPULSE_QUALIFIED"].shape[0] == 1
    assert events.loc[events["stage"] == "PULLBACK_QUALIFIED"].shape[0] == 1
    assert events.loc[events["stage"] == "CONFIRMATION_EMITTED"].shape[0] == 1

    m15_index = pd.date_range(index[18] + pd.Timedelta(minutes=15), periods=8, freq="15min")
    m15_rows = []
    for j, ts in enumerate(m15_index):
        close = 101.7 + 0.05 * j
        m15_rows.append({
            "open": close - 0.05, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 30.0,
            "ema9": 101.5, "ema20": 101.4, "ema50": 101.0, "atr14": 0.3, "plus_di14": 30.0,
            "minus_di14": 15.0, "adx14": 22.0, "rsi14": 55.0, "channel_high_8": 0.0, "channel_low_8": 0.0,
            "channel_high_20": 0.0, "channel_low_20": 0.0, "volume_mean_20": 30.0, "ema_separation_atr": 1.0,
            "ema_slope_3_atr": 0.1, "ema50_slope_3_atr": 0.05, "ema50_slope_6_atr": 0.1,
            "channel_width_atr": 3.0, "clv_long": 0.75, "clv_short": 0.25, "body_atr": 0.3,
            "volume_ratio": 1.0, "adx_change_3": 0.2, "price_extension_ema20_atr": 0.8, "atr_ratio": 0.003,
            "trend_age_long": 10.0, "trend_age_short": 0.0, "close_above_ema20_count_3": 3.0,
            "close_below_ema20_count_3": 0.0, "ema9_cross_up_ema20": False, "ema9_cross_down_ema20": False,
            "break_high_4": False, "break_low_4": False,
        })
    m15 = pd.DataFrame(m15_rows, index=m15_index)
    m15.index.name = "signal_time"
    m15.iloc[2, m15.columns.get_loc("close")] = 102.25
    m15.iloc[2, m15.columns.get_loc("high")] = 102.35
    m15.iloc[2, m15.columns.get_loc("open")] = 101.9
    m15.iloc[2, m15.columns.get_loc("break_high_4")] = True
    m15.iloc[2, m15.columns.get_loc("clv_long")] = 0.8
    spec = CHANNEL_BY_ID["strict_state_4h_1h_15m_fixed_rr2_0"]
    signals = build_strict_precision_signals(m15, setups, h4, spec, config)
    assert len(signals) == 1, signals
    assert pd.Timestamp(signals.iloc[0]["signal_time"]) > pd.Timestamp(setups.iloc[0]["setup_time"])


def self_test(config: dict[str, Any]) -> None:
    _state_machine_fixture_test(config)
    assert len(CHANNELS) == 3
    assert sum(spec.is_baseline for spec in CHANNELS) == 1
    raw = synthetic_5m_data(220_000, seed=20261009)
    signal_map, audit, funnel, events = build_all_signals(raw, config)
    assert set(signal_map) == set(CHANNEL_BY_ID)
    assert audit["resample"]["no_lookahead_alignment"] == "BACKWARD_ASOF_ONLY_CLOSED_BARS"
    assert set(funnel["stage"]) == {
        "ENVIRONMENT_LEG_START", "IMPULSE_QUALIFIED", "PULLBACK_QUALIFIED", "CONFIRMATION_EMITTED", "STRICT_15M_TRIGGER"
    }
    fixed = signal_map["strict_state_4h_1h_15m_fixed_rr2_0"]
    shadow = signal_map["strict_state_4h_1h_15m_be1_shadow_rr2_0"]
    assert fixed[["signal_time", "direction", "setup_id"]].reset_index(drop=True).equals(
        shadow[["signal_time", "direction", "setup_id"]].reset_index(drop=True)
    )
    if not fixed.empty:
        assert fixed["setup_id"].is_unique
        signal_times = pd.to_datetime(fixed["signal_time"], utc=True)
        setup_times = pd.to_datetime(fixed["setup_time"], utc=True)
        assert (signal_times >= setup_times + pd.Timedelta(minutes=15)).all()
        assert (signal_times <= setup_times + pd.Timedelta(hours=3)).all()
    confirmations = events.loc[events["stage"] == "CONFIRMATION_EMITTED"] if not events.empty else events
    if not confirmations.empty:
        assert confirmations["setup_id"].is_unique
        assert not confirmations.duplicated(["environment_leg_id", "cycle_number"]).any()
    result = run_benchmark(raw, config)
    assert len(result["summary"]) == 3
    if not result["trades"].empty:
        for channel_id, group in result["trades"].groupby("channel_id"):
            entries = pd.to_datetime(group["entry_time_utc"], utc=True).reset_index(drop=True)
            exits = pd.to_datetime(group["exit_time_utc"], utc=True).reset_index(drop=True)
            assert (exits >= entries).all(), channel_id
            if len(group) > 1:
                assert (entries.iloc[1:].reset_index(drop=True) >= exits.iloc[:-1].reset_index(drop=True)).all(), channel_id
    print("V109_ENGINE_SELF_TEST_OK")


__all__ = [
    "CHANNELS", "CHANNEL_BY_ID", "load_official_5m_data", "synthetic_5m_data", "run_benchmark",
    "parameter_manifest", "self_test", "metrics", "remove_best_fraction",
]
