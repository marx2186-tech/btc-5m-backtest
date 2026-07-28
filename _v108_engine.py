from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

import _v106_engine as v106


ChannelSpec = v106.ChannelSpec
BASE = v106.BASE
V103 = v106.v105.v104.v103


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
        "mtf_4h_1h_direct_rr2_0",
        "实验A·4H趋势+1H回踩/突破直接成交 RR2.0",
        "mtf_4h_1h_direct",
        "BOTH",
        1.25,
        2.0,
        48,
        False,
    ),
    ChannelSpec(
        "mtf_4h_1h_15m_precision_rr2_0",
        "实验B·4H趋势+1H确认+15M精确成交 RR2.0",
        "mtf_4h_1h_15m_precision",
        "BOTH",
        1.25,
        2.0,
        48,
        False,
    ),
)

CHANNEL_BY_ID = {spec.channel_id: spec for spec in CHANNELS}


def synthetic_5m_data(rows: int = 160_000, seed: int = 1018) -> pd.DataFrame:
    return v106.synthetic_5m_data(rows, seed)


def load_official_5m_data(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    return BASE.load_official_5m_data(config)


def resample_ohlcv(raw_5m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes % 5 != 0 or minutes < 5:
        raise ValueError("minutes must be a positive multiple of 5")
    indexed = raw_5m.set_index("time")[["open", "high", "low", "close", "volume"]]
    rule = f"{minutes}min"
    frame = indexed.resample(rule, origin="start_day", closed="left", label="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    count = indexed["close"].resample(rule, origin="start_day", closed="left", label="right").count()
    frame["source_5m_rows"] = count
    frame = frame.loc[frame["source_5m_rows"] == minutes // 5].dropna().copy()
    frame.index.name = "signal_time"
    return frame


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["ema9"] = BASE.ema(x["close"], 9)
    x["ema20"] = BASE.ema(x["close"], 20)
    x["ema50"] = BASE.ema(x["close"], 50)
    x["atr14"] = BASE.atr(x, 14)
    x["plus_di14"], x["minus_di14"], x["adx14"] = BASE.directional_movement(x, 14)
    x["rsi14"] = BASE.rsi(x["close"], 14)
    x["channel_high_8"] = x["high"].shift(1).rolling(8).max()
    x["channel_low_8"] = x["low"].shift(1).rolling(8).min()
    x["channel_high_20"] = x["high"].shift(1).rolling(20).max()
    x["channel_low_20"] = x["low"].shift(1).rolling(20).min()
    x["volume_mean_20"] = x["volume"].shift(1).rolling(20).mean()
    atr_safe = x["atr14"].replace(0, np.nan)
    x["ema_separation_atr"] = (x["ema20"] - x["ema50"]).abs() / atr_safe
    x["ema_slope_3_atr"] = (x["ema20"] - x["ema20"].shift(3)) / atr_safe
    x["ema50_slope_3_atr"] = (x["ema50"] - x["ema50"].shift(3)) / atr_safe
    x["ema50_slope_6_atr"] = (x["ema50"] - x["ema50"].shift(6)) / atr_safe
    x["channel_width_atr"] = (x["channel_high_20"] - x["channel_low_20"]) / atr_safe
    candle_range = (x["high"] - x["low"]).replace(0, np.nan)
    x["clv_long"] = (x["close"] - x["low"]) / candle_range
    x["clv_short"] = (x["high"] - x["close"]) / candle_range
    x["body_atr"] = (x["close"] - x["open"]).abs() / atr_safe
    x["volume_ratio"] = x["volume"] / x["volume_mean_20"].replace(0, np.nan)
    x["adx_change_3"] = x["adx14"] - x["adx14"].shift(3)
    x["price_extension_ema20_atr"] = (x["close"] - x["ema20"]).abs() / atr_safe
    x["atr_ratio"] = x["atr14"] / x["close"].replace(0, np.nan)
    x["atr_ratio_q20_252"] = x["atr_ratio"].shift(1).rolling(252).quantile(0.20)
    x["atr_ratio_q90_252"] = x["atr_ratio"].shift(1).rolling(252).quantile(0.90)
    x["trend_age_long"] = BASE.consecutive_true_count((x["ema20"] > x["ema50"]) & (x["ema_slope_3_atr"] > 0))
    x["trend_age_short"] = BASE.consecutive_true_count((x["ema20"] < x["ema50"]) & (x["ema_slope_3_atr"] < 0))
    x["close_above_ema20_count_3"] = (x["close"] > x["ema20"]).astype(float).rolling(3).sum()
    x["close_below_ema20_count_3"] = (x["close"] < x["ema20"]).astype(float).rolling(3).sum()
    x["ema9_cross_up_ema20"] = (x["ema9"] > x["ema20"]) & (x["ema9"].shift(1) <= x["ema20"].shift(1))
    x["ema9_cross_down_ema20"] = (x["ema9"] < x["ema20"]) & (x["ema9"].shift(1) >= x["ema20"].shift(1))
    x["break_high_4"] = x["close"] > x["high"].shift(1).rolling(4).max()
    x["break_low_4"] = x["close"] < x["low"].shift(1).rolling(4).min()
    return x


def _asof_join(left: pd.DataFrame, right: pd.DataFrame, right_columns: list[str], prefix: str) -> pd.DataFrame:
    left_reset = left.reset_index().sort_values("signal_time")
    right_reset = right.reset_index().sort_values("signal_time")[["signal_time", *right_columns]]
    rename = {column: f"{prefix}{column}" for column in right_columns}
    right_reset = right_reset.rename(columns=rename)
    merged = pd.merge_asof(left_reset, right_reset, on="signal_time", direction="backward", allow_exact_matches=True)
    return merged.set_index("signal_time")


def four_hour_environment(h4: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    p = config["multi_timeframe_parameters"]["four_hour_environment"]
    x = h4.copy()
    min_adx = float(p["minimum_adx14"])
    min_slope = float(p["minimum_abs_ema50_slope_3_atr"])
    x["env_long"] = (
        (x["ema20"] > x["ema50"])
        & (x["close"] > x["ema50"])
        & (x["ema50_slope_3_atr"] >= min_slope)
        & (x["adx14"] >= min_adx)
        & (x["plus_di14"] > x["minus_di14"])
    )
    x["env_short"] = (
        (x["ema20"] < x["ema50"])
        & (x["close"] < x["ema50"])
        & (x["ema50_slope_3_atr"] <= -min_slope)
        & (x["adx14"] >= min_adx)
        & (x["minus_di14"] > x["plus_di14"])
    )
    x["environment_direction"] = np.select([x["env_long"], x["env_short"]], [1, -1], default=0).astype(int)
    return x


def one_hour_setups(h1: pd.DataFrame, h4_env: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    p = config["multi_timeframe_parameters"]["one_hour_setup"]
    h4_cols = ["env_long", "env_short", "environment_direction", "adx14", "ema50_slope_3_atr"]
    x = _asof_join(h1, h4_env, h4_cols, "h4_")
    atr = x["atr14"].replace(0, np.nan)
    near_atr = float(p["pullback_near_ema_atr"])
    structure_tolerance = float(p["ema50_structure_tolerance_atr"])
    max_extension = float(p["maximum_price_extension_ema20_atr"])
    min_volume = float(p["minimum_breakout_volume_ratio"])

    long_pullback = (
        x["h4_env_long"].eq(True)
        & (x["ema20"] > x["ema50"])
        & (x["low"] <= x["ema20"] + near_atr * atr)
        & (x["low"] >= x["ema50"] - structure_tolerance * atr)
        & (x["close"] > x["ema20"])
        & (x["close"] > x["open"])
        & ((x["close"] > x["high"].shift(1)) | ((x["close"] > x["ema20"]) & (x["close"].shift(1) <= x["ema20"].shift(1))))
    )
    short_pullback = (
        x["h4_env_short"].eq(True)
        & (x["ema20"] < x["ema50"])
        & (x["high"] >= x["ema20"] - near_atr * atr)
        & (x["high"] <= x["ema50"] + structure_tolerance * atr)
        & (x["close"] < x["ema20"])
        & (x["close"] < x["open"])
        & ((x["close"] < x["low"].shift(1)) | ((x["close"] < x["ema20"]) & (x["close"].shift(1) >= x["ema20"].shift(1))))
    )
    long_breakout = (
        x["h4_env_long"].eq(True)
        & (x["ema20"] > x["ema50"])
        & (x["close"] > x["channel_high_8"])
        & (x["close"] > x["open"])
        & (x["price_extension_ema20_atr"] <= max_extension)
        & (x["volume_ratio"] >= min_volume)
    )
    short_breakout = (
        x["h4_env_short"].eq(True)
        & (x["ema20"] < x["ema50"])
        & (x["close"] < x["channel_low_8"])
        & (x["close"] < x["open"])
        & (x["price_extension_ema20_atr"] <= max_extension)
        & (x["volume_ratio"] >= min_volume)
    )

    # One setup event per fresh transition; do not emit the same state every hour.
    long_setup = long_pullback | long_breakout
    short_setup = short_pullback | short_breakout
    x["long_setup_event"] = long_setup & ~long_setup.shift(1).eq(True)
    x["short_setup_event"] = short_setup & ~short_setup.shift(1).eq(True)
    x["long_setup_type"] = np.where(long_pullback, "PULLBACK_END", np.where(long_breakout, "REBREAK", ""))
    x["short_setup_type"] = np.where(short_pullback, "PULLBACK_END", np.where(short_breakout, "REBREAK", ""))
    return x


def _signal_records(
    selected: pd.DataFrame,
    spec: ChannelSpec,
    direction: int,
    setup_type_column: str,
    trigger_type_column: str | None,
    stop_atr_multiple: float,
    reward_risk: float,
    max_holding_hours: int,
) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for timestamp, row in selected.iterrows():
        source_label = "多头组件" if direction == 1 else "空头组件"
        record = {
            "signal_time": pd.Timestamp(timestamp),
            "direction": int(direction),
            "close": float(row["close"]),
            "atr14": float(row["atr14"]),
            "adx14": float(row["adx14"]),
            "rsi14": float(row["rsi14"]),
            "plus_di14": float(row["plus_di14"]),
            "minus_di14": float(row["minus_di14"]),
            "ema_separation_atr": float(row["ema_separation_atr"]),
            "ema_slope_3_atr": float(row["ema_slope_3_atr"]),
            "volume_ratio": float(row["volume_ratio"]),
            "clv_long": float(row["clv_long"]),
            "clv_short": float(row["clv_short"]),
            "channel_width_atr": float(row["channel_width_atr"]),
            "adx_change_3": float(row["adx_change_3"]),
            "ema50_slope_6_atr": float(row["ema50_slope_6_atr"]),
            "price_extension_ema20_atr": float(row["price_extension_ema20_atr"]),
            "atr_ratio": float(row["atr_ratio"]),
            "atr_ratio_q20_252": float(row.get("atr_ratio_q20_252", np.nan)),
            "atr_ratio_q90_252": float(row.get("atr_ratio_q90_252", np.nan)),
            "trend_age_long": float(row["trend_age_long"]),
            "trend_age_short": float(row["trend_age_short"]),
            "close_above_ema20_count_3": float(row["close_above_ema20_count_3"]),
            "close_below_ema20_count_3": float(row["close_below_ema20_count_3"]),
            "stop_atr_multiple": float(stop_atr_multiple),
            "reward_risk": float(reward_risk),
            "max_holding_hours": int(max_holding_hours),
            "channel_id": spec.channel_id,
            "channel_label": spec.label,
            "profile": spec.profile,
            "direction_scope": spec.direction_scope,
            "is_baseline": bool(spec.is_baseline),
            "source_component_id": f"{spec.channel_id}_{'long' if direction == 1 else 'short'}",
            "source_component_label": f"{spec.label}·{source_label}",
            "setup_type": str(row.get(setup_type_column, "")),
            "trigger_type": str(row.get(trigger_type_column, "")) if trigger_type_column else "DIRECT_1H_CLOSE",
            "setup_time": pd.Timestamp(row.get("setup_time", timestamp)),
            "h4_environment_direction": int(row.get("h4_environment_direction", row.get("h4_environment_direction", direction))),
            "h4_adx": float(row.get("h4_adx14", np.nan)),
            "h4_ema50_slope_3_atr": float(row.get("h4_ema50_slope_3_atr", np.nan)),
        }
        records.append(record)
    return pd.DataFrame(records)


def build_baseline_signals(raw_5m: pd.DataFrame, spec: ChannelSpec, config: dict[str, Any]) -> pd.DataFrame:
    signal_map, _ = v106.build_all_signals(raw_5m)
    signals = signal_map["60m_portfolio_frozen_v10_6"].copy()
    if signals.empty:
        return signals
    signals["channel_id"] = spec.channel_id
    signals["channel_label"] = spec.label
    signals["profile"] = spec.profile
    signals["direction_scope"] = spec.direction_scope
    signals["is_baseline"] = True
    signals["setup_type"] = "V10_6_FROZEN_1H"
    signals["trigger_type"] = "DIRECT_1H_CLOSE"
    signals["setup_time"] = signals["signal_time"]
    return _filter_evaluation_window(signals, config)


def _filter_evaluation_window(signals: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if signals.empty:
        return signals
    start = pd.Timestamp(config["evaluation_window"]["start_utc"])
    end = pd.Timestamp(config["evaluation_window"]["end_utc"])
    times = pd.to_datetime(signals["signal_time"], utc=True)
    return signals.loc[(times >= start) & (times <= end)].sort_values("signal_time").reset_index(drop=True)


def build_direct_signals(h1_setup: pd.DataFrame, spec: ChannelSpec, config: dict[str, Any]) -> pd.DataFrame:
    p = config["multi_timeframe_parameters"]["execution"]
    long_rows = h1_setup.loc[h1_setup["long_setup_event"].fillna(False)].copy()
    long_rows["setup_time"] = long_rows.index
    long_rows["h4_environment_direction"] = 1
    short_rows = h1_setup.loc[h1_setup["short_setup_event"].fillna(False)].copy()
    short_rows["setup_time"] = short_rows.index
    short_rows["h4_environment_direction"] = -1
    long_signals = _signal_records(
        long_rows,
        spec,
        1,
        "long_setup_type",
        None,
        float(p["stop_atr_multiple"]),
        float(p["reward_risk"]),
        int(p["max_holding_hours"]),
    )
    short_signals = _signal_records(
        short_rows,
        spec,
        -1,
        "short_setup_type",
        None,
        float(p["stop_atr_multiple"]),
        float(p["reward_risk"]),
        int(p["max_holding_hours"]),
    )
    frames = [frame for frame in (long_signals, short_signals) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    signals = pd.concat(frames, ignore_index=True).sort_values(["signal_time", "direction"])
    conflicts = signals.groupby("signal_time")["direction"].nunique()
    conflict_times = set(conflicts.loc[conflicts > 1].index)
    if conflict_times:
        signals = signals.loc[~signals["signal_time"].isin(conflict_times)]
    return _filter_evaluation_window(signals.drop_duplicates("signal_time", keep="first"), config)


def build_precision_signals(
    m15: pd.DataFrame,
    h1_setup: pd.DataFrame,
    h4_env: pd.DataFrame,
    spec: ChannelSpec,
    config: dict[str, Any],
) -> pd.DataFrame:
    p15 = config["multi_timeframe_parameters"]["fifteen_minute_trigger"]
    pexec = config["multi_timeframe_parameters"]["execution"]
    validity = pd.Timedelta(hours=float(p15["setup_validity_hours"]))

    events: list[dict[str, Any]] = []
    for timestamp, row in h1_setup.loc[h1_setup["long_setup_event"].fillna(False)].iterrows():
        events.append({"setup_time": timestamp, "setup_direction": 1, "setup_type": row["long_setup_type"]})
    for timestamp, row in h1_setup.loc[h1_setup["short_setup_event"].fillna(False)].iterrows():
        events.append({"setup_time": timestamp, "setup_direction": -1, "setup_type": row["short_setup_type"]})
    if not events:
        return pd.DataFrame()
    event_frame = pd.DataFrame(events).sort_values("setup_time")

    left = m15.reset_index().sort_values("signal_time")
    joined = pd.merge_asof(
        left,
        event_frame,
        left_on="signal_time",
        right_on="setup_time",
        direction="backward",
        tolerance=validity,
        allow_exact_matches=True,
    ).set_index("signal_time")
    joined = _asof_join(joined, h4_env, ["environment_direction", "adx14", "ema50_slope_3_atr"], "h4_")
    joined["setup_age_minutes"] = (joined.index.to_series() - pd.to_datetime(joined["setup_time"], utc=True)).dt.total_seconds() / 60.0

    min_volume = float(p15["minimum_volume_ratio"])
    min_clv = float(p15["minimum_close_location_value"])
    long_cross = joined["ema9_cross_up_ema20"].fillna(False) & (joined["close"] > joined["ema20"])
    short_cross = joined["ema9_cross_down_ema20"].fillna(False) & (joined["close"] < joined["ema20"])
    long_break = joined["break_high_4"].fillna(False) & (joined["close"] > joined["open"]) & (joined["volume_ratio"] >= min_volume) & (joined["clv_long"] >= min_clv)
    short_break = joined["break_low_4"].fillna(False) & (joined["close"] < joined["open"]) & (joined["volume_ratio"] >= min_volume) & (joined["clv_short"] >= min_clv)

    joined["long_trigger"] = (
        (joined["setup_direction"] == 1)
        & (joined["h4_environment_direction"] == 1)
        & (joined["setup_age_minutes"] >= 15.0)
        & (long_cross | long_break)
    )
    joined["short_trigger"] = (
        (joined["setup_direction"] == -1)
        & (joined["h4_environment_direction"] == -1)
        & (joined["setup_age_minutes"] >= 15.0)
        & (short_cross | short_break)
    )
    joined["long_trigger_type"] = np.where(long_cross, "EMA9_RECLAIM", np.where(long_break, "LOCAL_HIGH_BREAK", ""))
    joined["short_trigger_type"] = np.where(short_cross, "EMA9_REJECT", np.where(short_break, "LOCAL_LOW_BREAK", ""))

    # One precise entry per one-hour setup event: use the first valid 15m trigger only.
    long_rows = joined.loc[joined["long_trigger"].fillna(False)].copy()
    if not long_rows.empty:
        long_rows = long_rows.sort_index().reset_index().drop_duplicates("setup_time", keep="first").set_index("signal_time")
        long_rows["h4_environment_direction"] = 1
    short_rows = joined.loc[joined["short_trigger"].fillna(False)].copy()
    if not short_rows.empty:
        short_rows = short_rows.sort_index().reset_index().drop_duplicates("setup_time", keep="first").set_index("signal_time")
        short_rows["h4_environment_direction"] = -1

    # Stop distance always uses the aligned 1h ATR and feature state, not the 15m ATR.
    h1_feature_columns = [
        "atr14", "adx14", "rsi14", "plus_di14", "minus_di14", "ema_separation_atr", "ema_slope_3_atr",
        "volume_ratio", "clv_long", "clv_short", "channel_width_atr", "adx_change_3", "ema50_slope_6_atr",
        "price_extension_ema20_atr", "atr_ratio", "atr_ratio_q20_252", "atr_ratio_q90_252", "trend_age_long",
        "trend_age_short", "close_above_ema20_count_3", "close_below_ema20_count_3", "close",
    ]
    h1_aligned_long = _asof_join(long_rows, h1_setup, h1_feature_columns, "h1_") if not long_rows.empty else long_rows
    h1_aligned_short = _asof_join(short_rows, h1_setup, h1_feature_columns, "h1_") if not short_rows.empty else short_rows

    def replace_with_h1(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        for column in h1_feature_columns:
            frame[column] = frame[f"h1_{column}"]
        return frame

    long_rows = replace_with_h1(h1_aligned_long)
    short_rows = replace_with_h1(h1_aligned_short)

    long_signals = _signal_records(
        long_rows,
        spec,
        1,
        "setup_type",
        "long_trigger_type",
        float(pexec["stop_atr_multiple"]),
        float(pexec["reward_risk"]),
        int(pexec["max_holding_hours"]),
    )
    short_signals = _signal_records(
        short_rows,
        spec,
        -1,
        "setup_type",
        "short_trigger_type",
        float(pexec["stop_atr_multiple"]),
        float(pexec["reward_risk"]),
        int(pexec["max_holding_hours"]),
    )
    frames = [frame for frame in (long_signals, short_signals) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    signals = pd.concat(frames, ignore_index=True).sort_values(["signal_time", "direction"])
    conflicts = signals.groupby("signal_time")["direction"].nunique()
    conflict_times = set(conflicts.loc[conflicts > 1].index)
    if conflict_times:
        signals = signals.loc[~signals["signal_time"].isin(conflict_times)]
    return _filter_evaluation_window(signals.drop_duplicates("signal_time", keep="first"), config)


def build_all_signals(raw_5m: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame]:
    h15 = add_features(resample_ohlcv(raw_5m, 15))
    h1 = add_features(resample_ohlcv(raw_5m, 60))
    h4 = four_hour_environment(add_features(resample_ohlcv(raw_5m, 240)), config)
    h1_setup = one_hour_setups(h1, h4, config)

    baseline_spec = CHANNEL_BY_ID["baseline_1h_shared_v10_6"]
    direct_spec = CHANNEL_BY_ID["mtf_4h_1h_direct_rr2_0"]
    precision_spec = CHANNEL_BY_ID["mtf_4h_1h_15m_precision_rr2_0"]

    signal_map = {
        baseline_spec.channel_id: build_baseline_signals(raw_5m, baseline_spec, config),
        direct_spec.channel_id: build_direct_signals(h1_setup, direct_spec, config),
        precision_spec.channel_id: build_precision_signals(h15, h1_setup, h4, precision_spec, config),
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
            }
        )

    funnel = pd.DataFrame(
        [
            {
                "stage": "4h_environment_bars",
                "long_count": int(h4["env_long"].sum()),
                "short_count": int(h4["env_short"].sum()),
            },
            {
                "stage": "1h_setup_events",
                "long_count": int(h1_setup["long_setup_event"].sum()),
                "short_count": int(h1_setup["short_setup_event"].sum()),
            },
            {
                "stage": "15m_precision_signals",
                "long_count": int((signal_map[precision_spec.channel_id]["direction"] == 1).sum()) if not signal_map[precision_spec.channel_id].empty else 0,
                "short_count": int((signal_map[precision_spec.channel_id]["direction"] == -1).sum()) if not signal_map[precision_spec.channel_id].empty else 0,
            },
        ]
    )
    audit = {
        "resample": {
            "15m_rows": int(len(h15)),
            "1h_rows": int(len(h1)),
            "4h_rows": int(len(h4)),
            "no_lookahead_alignment": "BACKWARD_ASOF_ONLY_CLOSED_BARS",
        },
        "channels": audit_rows,
        "evaluation_window": config["evaluation_window"],
    }
    return signal_map, audit, funnel


def _h4_regime_at_raw_times(raw_5m: pd.DataFrame, h4_env: pd.DataFrame) -> np.ndarray:
    left = pd.DataFrame({"time": pd.to_datetime(raw_5m["time"], utc=True)}).sort_values("time")
    right = h4_env.reset_index()[["signal_time", "environment_direction"]].sort_values("signal_time")
    joined = pd.merge_asof(left, right, left_on="time", right_on="signal_time", direction="backward", allow_exact_matches=True)
    return joined["environment_direction"].fillna(0).astype(int).to_numpy()


def execute_mtf_channel(
    raw_5m: pd.DataFrame,
    signals: pd.DataFrame,
    spec: ChannelSpec,
    config: dict[str, Any],
    h4_env: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "channel_id", "channel_label", "profile", "direction_scope", "source_component_id", "source_component_label",
        "reward_risk_target", "stop_atr_multiple", "max_holding_hours", "signal_time_utc", "entry_time_utc",
        "exit_time_utc", "direction", "entry_price", "stop_price", "target_price", "exit_price", "stop_distance",
        "gross_r", "fee_r", "net_r", "win", "exit_reason", "holding_5m_bars", "signal_atr", "signal_adx",
        "signal_rsi", "signal_plus_di", "signal_minus_di", "signal_ema_separation_atr", "signal_ema_slope_3_atr",
        "signal_volume_ratio", "signal_clv", "signal_channel_width_atr", "signal_adx_change_3", "signal_ema50_slope_6_atr",
        "signal_price_extension_ema20_atr", "signal_atr_ratio", "signal_trend_age", "month", "phase", "is_baseline",
        "setup_type", "trigger_type", "setup_time_utc", "mfe_R", "mae_R", "bars_to_mfe", "bars_to_mae",
    ]
    if signals.empty:
        return pd.DataFrame(columns=columns)

    fee_rate = float(config["execution"]["fee_rate_per_side"])
    slippage = float(config["execution"]["tick_size"]) * int(config["execution"]["slippage_ticks_per_fill"])
    min_stop_pct = float(config["execution"]["minimum_stop_distance_pct"])
    times = raw_5m["time"].to_numpy(dtype="datetime64[ns]")
    opens = raw_5m["open"].to_numpy(float)
    highs = raw_5m["high"].to_numpy(float)
    lows = raw_5m["low"].to_numpy(float)
    closes = raw_5m["close"].to_numpy(float)
    regime = _h4_regime_at_raw_times(raw_5m, h4_env)
    rows: list[dict[str, Any]] = []
    next_free_index = 0

    for row in signals.sort_values("signal_time").itertuples(index=False):
        signal_time = pd.Timestamp(row.signal_time)
        entry_i = int(np.searchsorted(times, signal_time.to_datetime64(), side="left"))
        if entry_i < next_free_index or entry_i >= len(raw_5m):
            continue
        direction = int(row.direction)
        if regime[entry_i] not in (0, direction):
            continue
        entry_price = float(opens[entry_i] + slippage * direction)
        signal_atr = float(row.atr14)
        if not math.isfinite(signal_atr) or signal_atr <= 0:
            continue
        stop_multiple = float(row.stop_atr_multiple)
        reward_risk = float(row.reward_risk)
        max_holding_hours = int(row.max_holding_hours)
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

        for bar_i in range(entry_i, final_i + 1):
            favorable = (highs[bar_i] - entry_price) / stop_distance if direction == 1 else (entry_price - lows[bar_i]) / stop_distance
            adverse = (entry_price - lows[bar_i]) / stop_distance if direction == 1 else (highs[bar_i] - entry_price) / stop_distance
            if favorable > mfe_r:
                mfe_r = float(favorable)
                bars_to_mfe = int(bar_i - entry_i + 1)
            if adverse > mae_r:
                mae_r = float(adverse)
                bars_to_mae = int(bar_i - entry_i + 1)

            # A completed 4h opposite environment is known at this bar open and exits before intrabar checks.
            if bar_i > entry_i and regime[bar_i] == -direction:
                exit_i, exit_reason, raw_exit = bar_i, "H4_TREND_REVERSAL", float(opens[bar_i])
                break
            if direction == 1:
                stop_hit = lows[bar_i] <= stop_price
                target_hit = highs[bar_i] >= target_price
            else:
                stop_hit = highs[bar_i] >= stop_price
                target_hit = lows[bar_i] <= target_price
            if stop_hit and target_hit:
                exit_i, exit_reason, raw_exit = bar_i, "STOP_FIRST_CONSERVATIVE", stop_price
                break
            if stop_hit:
                exit_i, exit_reason, raw_exit = bar_i, "STOP", stop_price
                break
            if target_hit:
                exit_i, exit_reason, raw_exit = bar_i, "TARGET", target_price
                break

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
                "source_component_id": getattr(row, "source_component_id", spec.channel_id),
                "source_component_label": getattr(row, "source_component_label", spec.label),
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
                "signal_adx": float(row.adx14),
                "signal_rsi": float(row.rsi14),
                "signal_plus_di": float(row.plus_di14),
                "signal_minus_di": float(row.minus_di14),
                "signal_ema_separation_atr": float(row.ema_separation_atr),
                "signal_ema_slope_3_atr": float(row.ema_slope_3_atr),
                "signal_volume_ratio": float(row.volume_ratio),
                "signal_clv": float(row.clv_long if direction == 1 else row.clv_short),
                "signal_channel_width_atr": float(row.channel_width_atr),
                "signal_adx_change_3": float(row.adx_change_3),
                "signal_ema50_slope_6_atr": float(row.ema50_slope_6_atr),
                "signal_price_extension_ema20_atr": float(row.price_extension_ema20_atr),
                "signal_atr_ratio": float(row.atr_ratio),
                "signal_trend_age": float(row.trend_age_long if direction == 1 else row.trend_age_short),
                "month": entry_time.strftime("%Y-%m"),
                "phase": "evaluation_2026_h1",
                "is_baseline": spec.is_baseline,
                "setup_type": getattr(row, "setup_type", ""),
                "trigger_type": getattr(row, "trigger_type", ""),
                "setup_time_utc": pd.Timestamp(getattr(row, "setup_time", signal_time)).isoformat(),
                "mfe_R": float(max(mfe_r, 0.0)),
                "mae_R": float(max(mae_r, 0.0)),
                "bars_to_mfe": bars_to_mfe,
                "bars_to_mae": bars_to_mae,
            }
        )
        next_free_index = exit_i + 1

    return pd.DataFrame(rows, columns=columns)


def enrich_baseline_path_diagnostics(raw_5m: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        for column in ("setup_type", "trigger_type", "setup_time_utc", "mfe_R", "mae_R", "bars_to_mfe", "bars_to_mae"):
            trades[column] = pd.Series(dtype=float if column.endswith("_R") or column.startswith("bars") else str)
        return trades
    times = raw_5m["time"].to_numpy(dtype="datetime64[ns]")
    highs = raw_5m["high"].to_numpy(float)
    lows = raw_5m["low"].to_numpy(float)
    output = trades.copy()
    diagnostics: list[tuple[float, float, int, int]] = []
    for row in output.itertuples(index=False):
        entry_i = int(np.searchsorted(times, pd.Timestamp(row.entry_time_utc).to_datetime64(), side="left"))
        exit_open = pd.Timestamp(row.exit_time_utc) - pd.Timedelta(minutes=5)
        exit_i = int(np.searchsorted(times, exit_open.to_datetime64(), side="left"))
        exit_i = min(max(exit_i, entry_i), len(times) - 1)
        direction = int(row.direction)
        entry = float(row.entry_price)
        stop_distance = float(row.stop_distance)
        favorable = (highs[entry_i : exit_i + 1] - entry) / stop_distance if direction == 1 else (entry - lows[entry_i : exit_i + 1]) / stop_distance
        adverse = (entry - lows[entry_i : exit_i + 1]) / stop_distance if direction == 1 else (highs[entry_i : exit_i + 1] - entry) / stop_distance
        diagnostics.append((float(max(np.max(favorable), 0.0)), float(max(np.max(adverse), 0.0)), int(np.argmax(favorable) + 1), int(np.argmax(adverse) + 1)))
    output["setup_type"] = "V10_6_FROZEN_1H"
    output["trigger_type"] = "DIRECT_1H_CLOSE"
    output["setup_time_utc"] = output["signal_time_utc"]
    output[["mfe_R", "mae_R", "bars_to_mfe", "bars_to_mae"]] = pd.DataFrame(diagnostics, index=output.index)
    return output


def metrics(trades: pd.DataFrame | Iterable[dict[str, Any]]) -> dict[str, float]:
    return v106.metrics(trades)


def remove_best_fraction(trades: pd.DataFrame, fraction: float = 0.10) -> tuple[pd.DataFrame, dict[str, Any]]:
    return v106.remove_best_fraction(trades, fraction)


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
    base_columns = columns + list(metrics(pd.DataFrame()).keys())
    if trades.empty:
        return pd.DataFrame(columns=base_columns)
    rows: list[dict[str, Any]] = []
    for keys, group in trades.groupby(columns, dropna=False, sort=True):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(columns, keys_tuple))
        row.update(metrics(group))
        rows.append(row)
    return pd.DataFrame(rows, columns=base_columns)


def loss_diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "channel_id", "channel_label", "loss_class", "trades", "avg_net_R", "avg_mfe_R", "avg_mae_R",
        "median_holding_bars", "share_of_channel_losses",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    losses = trades.loc[trades["net_r"] <= 0].copy()
    if losses.empty:
        return pd.DataFrame(columns=columns)
    losses["loss_class"] = np.select(
        [
            losses["holding_5m_bars"] <= 24,
            losses["mfe_R"] >= 1.0,
            losses["mfe_R"] >= 0.5,
            losses["exit_reason"] == "H4_TREND_REVERSAL",
        ],
        ["FAST_REVERSAL_LE_2H", "GAVE_BACK_AT_LEAST_1R", "GAVE_BACK_0_5_TO_1R", "H4_ENVIRONMENT_REVERSAL"],
        default="NORMAL_FAILED_SETUP",
    )
    rows: list[dict[str, Any]] = []
    for (channel_id, channel_label), channel_losses in losses.groupby(["channel_id", "channel_label"]):
        total = len(channel_losses)
        for loss_class, group in channel_losses.groupby("loss_class"):
            rows.append(
                {
                    "channel_id": channel_id,
                    "channel_label": channel_label,
                    "loss_class": loss_class,
                    "trades": int(len(group)),
                    "avg_net_R": float(group["net_r"].mean()),
                    "avg_mfe_R": float(group["mfe_R"].mean()),
                    "avg_mae_R": float(group["mae_R"].mean()),
                    "median_holding_bars": float(group["holding_5m_bars"].median()),
                    "share_of_channel_losses": float(len(group) / total),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def run_benchmark(raw_5m: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    signal_map, signal_audit, funnel = build_all_signals(raw_5m, config)
    h4 = four_hour_environment(add_features(resample_ohlcv(raw_5m, 240)), config)
    trade_frames: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for spec in CHANNELS:
        signals = signal_map[spec.channel_id]
        if spec.is_baseline:
            trades = V103.execute_channel(raw_5m, signals, spec, config)
            trades = enrich_baseline_path_diagnostics(raw_5m, trades)
        else:
            trades = execute_mtf_channel(raw_5m, signals, spec, config, h4)
        trade_frames.append(trades)
        audits[spec.channel_id] = summarize_channel(trades, spec, config)
    nonempty = [frame for frame in trade_frames if not frame.empty]
    ledger = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    if not ledger.empty:
        ledger = ledger.sort_values(["channel_id", "entry_time_utc"]).reset_index(drop=True)
    summary = pd.DataFrame([{key: value for key, value in audit.items() if key != "checks"} for audit in audits.values()])
    summary = summary.sort_values(["historical_research_gate_pass", "profit_factor", "net_R"], ascending=[False, False, False]).reset_index(drop=True)
    return {
        "signals": signal_map,
        "signal_audit": signal_audit,
        "signal_funnel": funnel,
        "trades": ledger,
        "summary": summary,
        "qualification": audits,
        "direction_summary": grouped_metrics(ledger, ["channel_id", "channel_label", "direction"]),
        "monthly_summary": grouped_metrics(ledger, ["channel_id", "channel_label", "month"]),
        "setup_type_summary": grouped_metrics(ledger, ["channel_id", "channel_label", "setup_type", "trigger_type"]),
        "exit_reason_summary": grouped_metrics(ledger, ["channel_id", "channel_label", "exit_reason"]),
        "loss_diagnostics": loss_diagnostics(ledger),
    }


def parameter_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "release": config["release"],
        "channels": [asdict(spec) for spec in CHANNELS],
        "multi_timeframe_parameters": config["multi_timeframe_parameters"],
        "execution": config["execution"],
        "evaluation_window": config["evaluation_window"],
        "no_lookahead_rules": config["no_lookahead_rules"],
    }


def self_test(config: dict[str, Any]) -> None:
    assert len(CHANNELS) == 3
    assert sum(spec.is_baseline for spec in CHANNELS) == 1
    raw = synthetic_5m_data(160_000, seed=20261008)
    signal_map, audit, funnel = build_all_signals(raw, config)
    assert set(signal_map) == set(CHANNEL_BY_ID)
    assert audit["resample"]["no_lookahead_alignment"] == "BACKWARD_ASOF_ONLY_CLOSED_BARS"
    assert len(funnel) == 3
    for channel_id, signals in signal_map.items():
        if not signals.empty:
            assert pd.to_datetime(signals["signal_time"], utc=True).is_monotonic_increasing, channel_id
            assert signals["signal_time"].nunique() == len(signals), channel_id
    result = run_benchmark(raw, config)
    assert len(result["summary"]) == 3
    assert set(result["qualification"]) == set(CHANNEL_BY_ID)
    if not result["trades"].empty:
        for channel_id, group in result["trades"].groupby("channel_id"):
            entries = pd.to_datetime(group["entry_time_utc"], utc=True).reset_index(drop=True)
            exits = pd.to_datetime(group["exit_time_utc"], utc=True).reset_index(drop=True)
            assert (exits >= entries).all(), channel_id
            if len(group) > 1:
                assert (entries.iloc[1:].reset_index(drop=True) >= exits.iloc[:-1].reset_index(drop=True)).all(), channel_id
    print("V108_ENGINE_SELF_TEST_OK")


__all__ = [
    "CHANNELS", "CHANNEL_BY_ID", "load_official_5m_data", "synthetic_5m_data", "run_benchmark",
    "parameter_manifest", "self_test", "metrics", "remove_best_fraction",
]
