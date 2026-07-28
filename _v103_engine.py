from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

import _v102_engine as base


@dataclass(frozen=True)
class ChannelSpec:
    channel_id: str
    label: str
    profile: str
    direction_scope: str  # LONG, SHORT, BOTH
    stop_atr_multiple: float
    reward_risk: float
    max_holding_hours: int
    is_baseline: bool = False
    long_component_id: str | None = None
    short_component_id: str | None = None

    @property
    def family(self) -> str:
        return "trend_pullback"

    @property
    def is_portfolio(self) -> bool:
        return self.direction_scope == "BOTH"


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec(
        "60m_long_quality_baseline_rr2_5",
        "多头冻结基线·1小时质量回踩 RR2.5",
        "long_quality_baseline",
        "LONG",
        1.25,
        2.5,
        36,
        True,
    ),
    ChannelSpec(
        "60m_long_recovery_rr2_0",
        "多头独立策略·趋势恢复 RR2.0",
        "long_recovery",
        "LONG",
        1.20,
        2.0,
        48,
        False,
    ),
    ChannelSpec(
        "60m_long_recovery_rr2_5",
        "多头独立策略·趋势恢复 RR2.5",
        "long_recovery",
        "LONG",
        1.20,
        2.5,
        48,
        False,
    ),
    ChannelSpec(
        "60m_short_persistence_baseline_rr2_5",
        "空头冻结基线·趋势持续性 RR2.5",
        "short_persistence_baseline",
        "SHORT",
        1.25,
        2.5,
        48,
        True,
    ),
    ChannelSpec(
        "60m_short_continuation_rr2_5",
        "空头独立策略·下跌延续 RR2.5",
        "short_continuation",
        "SHORT",
        1.30,
        2.5,
        36,
        False,
    ),
    ChannelSpec(
        "60m_short_continuation_rr3_0",
        "空头独立策略·下跌延续 RR3.0",
        "short_continuation",
        "SHORT",
        1.30,
        3.0,
        36,
        False,
    ),
    ChannelSpec(
        "60m_split_portfolio_frozen",
        "多空分离组合·冻结基线",
        "split_portfolio_frozen",
        "BOTH",
        0.0,
        0.0,
        0,
        True,
        "60m_long_quality_baseline_rr2_5",
        "60m_short_persistence_baseline_rr2_5",
    ),
    ChannelSpec(
        "60m_split_portfolio_balanced",
        "多空分离组合·均衡版（多2.0/空2.5）",
        "split_portfolio_balanced",
        "BOTH",
        0.0,
        0.0,
        0,
        False,
        "60m_long_recovery_rr2_0",
        "60m_short_continuation_rr2_5",
    ),
    ChannelSpec(
        "60m_split_portfolio_growth",
        "多空分离组合·进取版（多2.5/空3.0）",
        "split_portfolio_growth",
        "BOTH",
        0.0,
        0.0,
        0,
        False,
        "60m_long_recovery_rr2_5",
        "60m_short_continuation_rr3_0",
    ),
)

CHANNEL_BY_ID = {spec.channel_id: spec for spec in CHANNELS}


def load_official_5m_data(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    return base.load_official_5m_data(config)


def synthetic_5m_data(rows: int = 60_000, seed: int = 1013) -> pd.DataFrame:
    return base.synthetic_5m_data(rows, seed)


def _empty_mask(x: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=x.index, dtype=bool)


def long_quality_baseline_mask(x: pd.DataFrame) -> pd.Series:
    long_mask, _ = base.trend_quality_masks(x)
    return long_mask


def short_quality_reference_mask(x: pd.DataFrame) -> pd.Series:
    _, short_mask = base.trend_quality_masks(x)
    return short_mask


def short_persistence_baseline_mask(x: pd.DataFrame) -> pd.Series:
    _, short_mask = base.trend_persistence_masks(x)
    return short_mask


def long_recovery_mask(x: pd.DataFrame) -> pd.Series:
    """Long-only continuation after a controlled pullback and renewed trend support."""
    base_long = long_quality_baseline_mask(x)
    plus_advantage = x["plus_di14"] - x["minus_di14"]
    return (
        base_long
        & x["adx14"].between(22.0, 42.0)
        & plus_advantage.between(4.0, 20.0)
        & x["ema_separation_atr"].between(0.65, 2.30)
        & x["ema_slope_3_atr"].between(0.10, 0.45)
        & (x["ema50_slope_6_atr"] >= 0.02)
        & (x["adx_change_3"] >= -4.0)
        & x["volume_ratio"].between(0.75, 2.20)
        & x["channel_width_atr"].between(2.75, 8.50)
        & (x["price_extension_ema20_atr"] <= 0.80)
        & x["trend_age_long"].between(5.0, 144.0)
        & (x["close_above_ema20_count_3"] >= 2.0)
        & x["rsi14"].between(50.0, 67.0)
    )


def short_continuation_mask(x: pd.DataFrame) -> pd.Series:
    """Short-only continuation allowing faster bearish expansion and wider squeeze risk."""
    base_short = short_quality_reference_mask(x)
    minus_advantage = x["minus_di14"] - x["plus_di14"]
    return (
        base_short
        & x["adx14"].between(22.0, 48.0)
        & minus_advantage.between(4.0, 24.0)
        & x["ema_separation_atr"].between(0.65, 2.80)
        & x["ema_slope_3_atr"].between(-0.55, -0.10)
        & (x["ema50_slope_6_atr"] <= -0.02)
        & (x["adx_change_3"] >= -5.0)
        & x["volume_ratio"].between(0.65, 2.60)
        & x["channel_width_atr"].between(2.50, 9.00)
        & (x["price_extension_ema20_atr"] <= 1.00)
        & x["trend_age_short"].between(5.0, 144.0)
        & (x["close_below_ema20_count_3"] >= 2.0)
        & x["rsi14"].between(30.0, 50.0)
    )


def _signal_rows(mask: pd.Series, direction: int, x: pd.DataFrame, spec: ChannelSpec) -> pd.DataFrame:
    # The inherited helper is deliberately reused so feature columns stay identical to V10.2.
    return base._signal_rows(mask, direction, x, spec)


def generate_directional_signals(hourly_features: pd.DataFrame, spec: ChannelSpec) -> pd.DataFrame:
    if spec.profile == "long_quality_baseline":
        mask, direction = long_quality_baseline_mask(hourly_features), 1
    elif spec.profile == "long_recovery":
        mask, direction = long_recovery_mask(hourly_features), 1
    elif spec.profile == "short_persistence_baseline":
        mask, direction = short_persistence_baseline_mask(hourly_features), -1
    elif spec.profile == "short_continuation":
        mask, direction = short_continuation_mask(hourly_features), -1
    else:
        raise ValueError(f"Unsupported directional profile: {spec.profile}")
    signals = _signal_rows(mask, direction, hourly_features, spec)
    if signals.empty:
        return signals
    return signals.sort_values("signal_time").drop_duplicates("signal_time", keep="first").reset_index(drop=True)


def build_portfolio_signals(
    spec: ChannelSpec,
    signal_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not spec.long_component_id or not spec.short_component_id:
        raise ValueError(f"Portfolio components missing for {spec.channel_id}")
    long_signals = signal_map[spec.long_component_id].copy()
    short_signals = signal_map[spec.short_component_id].copy()
    component_frames = [frame for frame in (long_signals, short_signals) if not frame.empty]
    if not component_frames:
        return pd.DataFrame(), {
            "channel_id": spec.channel_id,
            "long_component_id": spec.long_component_id,
            "short_component_id": spec.short_component_id,
            "long_component_signals": 0,
            "short_component_signals": 0,
            "conflict_timestamps": 0,
            "portfolio_signals": 0,
        }
    merged = pd.concat(component_frames, ignore_index=True).sort_values(["signal_time", "direction"])
    direction_counts = merged.groupby("signal_time")["direction"].nunique()
    conflict_times = set(direction_counts.loc[direction_counts > 1].index)
    if conflict_times:
        merged = merged.loc[~merged["signal_time"].isin(conflict_times)].copy()
    merged = merged.drop_duplicates("signal_time", keep="first").reset_index(drop=True)
    # Preserve component-level stop, target and holding rules; only ownership changes to the portfolio channel.
    merged["source_component_id"] = merged["channel_id"]
    merged["source_component_label"] = merged["channel_label"]
    merged["channel_id"] = spec.channel_id
    merged["channel_label"] = spec.label
    merged["profile"] = spec.profile
    merged["is_baseline"] = bool(spec.is_baseline)
    audit = {
        "channel_id": spec.channel_id,
        "long_component_id": spec.long_component_id,
        "short_component_id": spec.short_component_id,
        "long_component_signals": int(len(long_signals)),
        "short_component_signals": int(len(short_signals)),
        "conflict_timestamps": int(len(conflict_times)),
        "portfolio_signals": int(len(merged)),
    }
    return merged, audit


def build_all_signals(raw_5m: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    hourly = base.resample_hourly(raw_5m)
    x = base.add_hourly_features(hourly)
    signal_map: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    portfolio_audit: list[dict[str, Any]] = []

    for spec in CHANNELS:
        if spec.is_portfolio:
            continue
        signals = generate_directional_signals(x, spec)
        signal_map[spec.channel_id] = signals
        audit_rows.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "profile": spec.profile,
                "direction_scope": spec.direction_scope,
                "reward_risk": spec.reward_risk,
                "stop_atr_multiple": spec.stop_atr_multiple,
                "max_holding_hours": spec.max_holding_hours,
                "is_baseline": spec.is_baseline,
                "signals": int(len(signals)),
                "long_signals": int((signals["direction"] == 1).sum()) if not signals.empty else 0,
                "short_signals": int((signals["direction"] == -1).sum()) if not signals.empty else 0,
            }
        )

    for spec in CHANNELS:
        if not spec.is_portfolio:
            continue
        signals, p_audit = build_portfolio_signals(spec, signal_map)
        signal_map[spec.channel_id] = signals
        portfolio_audit.append(p_audit)
        audit_rows.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "profile": spec.profile,
                "direction_scope": spec.direction_scope,
                "reward_risk": "ASYMMETRIC",
                "stop_atr_multiple": "ASYMMETRIC",
                "max_holding_hours": "ASYMMETRIC",
                "is_baseline": spec.is_baseline,
                "signals": int(len(signals)),
                "long_signals": int((signals["direction"] == 1).sum()) if not signals.empty else 0,
                "short_signals": int((signals["direction"] == -1).sum()) if not signals.empty else 0,
            }
        )

    return signal_map, {
        "resample": {
            "60": {
                "rows": int(len(hourly)),
                "first_bar_close_utc": hourly.index.min().isoformat() if not hourly.empty else None,
                "last_bar_close_utc": hourly.index.max().isoformat() if not hourly.empty else None,
                "source_rows_per_bar": 12,
            }
        },
        "channels": audit_rows,
        "portfolio_overlap": portfolio_audit,
    }


def phase_for_time(timestamp: pd.Timestamp, config: dict[str, Any]) -> str:
    return base.phase_for_time(timestamp, config)


def execute_channel(
    raw_5m: pd.DataFrame,
    signals: pd.DataFrame,
    spec: ChannelSpec,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "channel_id", "channel_label", "profile", "direction_scope", "source_component_id",
        "source_component_label", "reward_risk_target", "stop_atr_multiple", "max_holding_hours",
        "signal_time_utc", "entry_time_utc", "exit_time_utc", "direction", "entry_price",
        "stop_price", "target_price", "exit_price", "stop_distance", "gross_r", "fee_r", "net_r",
        "win", "exit_reason", "holding_5m_bars", "signal_atr", "signal_adx", "signal_rsi",
        "signal_plus_di", "signal_minus_di", "signal_ema_separation_atr", "signal_ema_slope_3_atr",
        "signal_volume_ratio", "signal_clv", "signal_channel_width_atr", "signal_adx_change_3",
        "signal_ema50_slope_6_atr", "signal_price_extension_ema20_atr", "signal_atr_ratio",
        "signal_trend_age", "month", "phase", "is_baseline",
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
    trades: list[dict[str, Any]] = []
    next_free_index = 0

    for row in signals.sort_values("signal_time").itertuples(index=False):
        signal_time = pd.Timestamp(row.signal_time)
        entry_i = int(np.searchsorted(times, signal_time.to_datetime64(), side="left"))
        if entry_i < next_free_index or entry_i >= len(raw_5m):
            continue
        direction = int(row.direction)
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

        for bar_i in range(entry_i, final_i + 1):
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
        source_component_id = getattr(row, "source_component_id", spec.channel_id)
        source_component_label = getattr(row, "source_component_label", spec.label)
        trades.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "profile": spec.profile,
                "direction_scope": spec.direction_scope,
                "source_component_id": source_component_id,
                "source_component_label": source_component_label,
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
                "phase": phase_for_time(entry_time, config),
                "is_baseline": spec.is_baseline,
            }
        )
        next_free_index = exit_i + 1

    return pd.DataFrame(trades, columns=columns)


def metrics(trades: pd.DataFrame | Iterable[dict[str, Any]]) -> dict[str, float]:
    return base.metrics(trades)


def remove_best_fraction(trades: pd.DataFrame, fraction: float = 0.10) -> tuple[pd.DataFrame, dict[str, Any]]:
    return base.remove_best_fraction(trades, fraction)


def month_range(start_month: str, end_month: str) -> list[str]:
    return base.month_range(start_month, end_month)


def monthly_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["channel_id", "channel_label", "month"] + list(metrics(pd.DataFrame()).keys())
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (channel_id, channel_label, month), group in trades.groupby(["channel_id", "channel_label", "month"], sort=True):
        rows.append({"channel_id": channel_id, "channel_label": channel_label, "month": month, **metrics(group)})
    return pd.DataFrame(rows)


def phase_metrics(trades: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        channel_trades = trades.loc[trades["channel_id"] == spec.channel_id] if not trades.empty else pd.DataFrame()
        for phase_name in config["phases"]:
            phase_trades = channel_trades.loc[channel_trades["phase"] == phase_name] if not channel_trades.empty else pd.DataFrame()
            rows.append({"channel_id": spec.channel_id, "channel_label": spec.label, "phase": phase_name, **metrics(phase_trades)})
    return pd.DataFrame(rows)


def direction_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["channel_id", "channel_label", "direction"] + list(metrics(pd.DataFrame()).keys())
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (channel_id, channel_label, direction), group in trades.groupby(["channel_id", "channel_label", "direction"], sort=True):
        rows.append({
            "channel_id": channel_id,
            "channel_label": channel_label,
            "direction": "LONG" if int(direction) == 1 else "SHORT",
            **metrics(group),
        })
    return pd.DataFrame(rows)


def component_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["channel_id", "channel_label", "source_component_id", "source_component_label"] + list(metrics(pd.DataFrame()).keys())
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, group in trades.groupby(["channel_id", "channel_label", "source_component_id", "source_component_label"], sort=True):
        channel_id, channel_label, source_id, source_label = keys
        rows.append({
            "channel_id": channel_id,
            "channel_label": channel_label,
            "source_component_id": source_id,
            "source_component_label": source_label,
            **metrics(group),
        })
    return pd.DataFrame(rows)


def rolling_quarter_metrics(trades: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = ["channel_id", "channel_label", "window_start_month", "window_end_month"] + list(metrics(pd.DataFrame()).keys())
    all_months = month_range(str(config["start_month"]), str(config["end_month"]))
    rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        channel_trades = trades.loc[trades["channel_id"] == spec.channel_id] if not trades.empty else pd.DataFrame()
        for index in range(2, len(all_months)):
            window = all_months[index - 2:index + 1]
            subset = channel_trades.loc[channel_trades["month"].isin(window)] if not channel_trades.empty else pd.DataFrame()
            rows.append({
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "window_start_month": window[0],
                "window_end_month": window[-1],
                **metrics(subset),
            })
    return pd.DataFrame(rows, columns=columns)


def _thresholds_for_scope(spec: ChannelSpec, config: dict[str, Any]) -> dict[str, Any]:
    key = "portfolio" if spec.is_portfolio else "directional"
    return config["research_candidate_thresholds"][key]


def qualification_audit(channel_trades: pd.DataFrame, spec: ChannelSpec, config: dict[str, Any]) -> dict[str, Any]:
    threshold = _thresholds_for_scope(spec, config)
    base_metrics = metrics(channel_trades)
    _, robust = remove_best_fraction(channel_trades, float(threshold["remove_best_fraction"]))
    months = monthly_metrics(channel_trades)
    positive_months = int((months["net_R"] > 0).sum()) if not months.empty else 0
    active_months = int((months["trades"] > 0).sum()) if not months.empty else 0
    positive_total = float(months.loc[months["net_R"] > 0, "net_R"].sum()) if not months.empty else 0.0
    maximum_positive = float(months.loc[months["net_R"] > 0, "net_R"].max()) if positive_total > 0 else 0.0
    max_month_profit_share = maximum_positive / positive_total if positive_total > 0 else 1.0
    phase_values = channel_trades.groupby("phase")["net_r"].sum().to_dict() if not channel_trades.empty else {}
    positive_phases = int(sum(float(value) > 0 for value in phase_values.values()))
    historical = channel_trades.loc[channel_trades["phase"] == "historical_test"] if not channel_trades.empty else pd.DataFrame()
    historical_metrics = metrics(historical)
    quarters = rolling_quarter_metrics(channel_trades, config)
    quarters = quarters.loc[quarters["channel_id"] == spec.channel_id]
    positive_rolling_quarters = int((quarters["net_R"] > 0).sum()) if not quarters.empty else 0
    active_rolling_quarters = int((quarters["trades"] > 0).sum()) if not quarters.empty else 0
    worst_rolling_quarter_net_R = float(quarters["net_R"].min()) if not quarters.empty else 0.0
    checks = {
        "minimum_trades": base_metrics["trades"] >= int(threshold["minimum_trades"]),
        "minimum_win_rate": base_metrics["win_rate"] >= float(threshold["minimum_win_rate"]),
        "minimum_avg_win_loss_ratio": base_metrics["avg_win_loss_ratio"] >= float(threshold["minimum_avg_win_loss_ratio"]),
        "minimum_profit_factor": base_metrics["profit_factor"] >= float(threshold["minimum_profit_factor"]),
        "minimum_expectancy_R": base_metrics["expectancy_R"] >= float(threshold["minimum_expectancy_R"]),
        "maximum_drawdown_R": base_metrics["max_drawdown_R"] <= float(threshold["maximum_drawdown_R"]),
        "minimum_positive_months": positive_months >= int(threshold["minimum_positive_months"]),
        "best_trades_removed_still_profitable": robust["net_R"] > 0,
        "maximum_single_positive_month_share": max_month_profit_share <= float(threshold["maximum_single_positive_month_share"]),
        "minimum_positive_phases": positive_phases >= int(threshold["minimum_positive_phases"]),
        "historical_test_positive": historical_metrics["net_R"] > 0,
        "historical_test_profit_factor": historical_metrics["profit_factor"] >= float(threshold["minimum_historical_test_profit_factor"]),
        "minimum_positive_rolling_quarters": positive_rolling_quarters >= int(threshold["minimum_positive_rolling_quarters"]),
        "minimum_worst_rolling_quarter_net_R": worst_rolling_quarter_net_R >= float(threshold["minimum_worst_rolling_quarter_net_R"]),
    }
    return {
        "scope": "portfolio" if spec.is_portfolio else spec.direction_scope,
        "metrics": base_metrics,
        "best_fraction_removed": robust,
        "historical_test_metrics": historical_metrics,
        "active_months": active_months,
        "positive_months": positive_months,
        "positive_phases": positive_phases,
        "active_rolling_quarters": active_rolling_quarters,
        "positive_rolling_quarters": positive_rolling_quarters,
        "worst_rolling_quarter_net_R": worst_rolling_quarter_net_R,
        "max_single_positive_month_share": max_month_profit_share,
        "checks": checks,
        "research_candidate": bool(all(checks.values())),
        "qualified_for_live_trading": False,
        "historical_data_already_viewed": True,
    }


def benchmark_score(audit: dict[str, Any]) -> float:
    m = audit["metrics"]
    robust = audit["best_fraction_removed"]
    historical = audit["historical_test_metrics"]
    if m["trades"] == 0:
        return -1e9
    return float(
        16.0 * m["win_rate"]
        + 3.0 * min(m["avg_win_loss_ratio"], 4.0)
        + 4.0 * min(m["profit_factor"], 4.0)
        + m["net_R"]
        + 12.0 * m["expectancy_R"]
        + 1.5 * historical["net_R"]
        + 3.0 * min(historical["profit_factor"], 3.0)
        + 0.4 * audit["positive_months"]
        + 0.5 * audit["positive_rolling_quarters"]
        + 0.8 * audit["positive_phases"]
        + 0.8 * robust["net_R"]
        + 0.5 * audit["worst_rolling_quarter_net_R"]
        - 0.5 * m["max_drawdown_R"]
    )


def baseline_id_for_spec(spec: ChannelSpec) -> str:
    if spec.direction_scope == "LONG":
        return "60m_long_quality_baseline_rr2_5"
    if spec.direction_scope == "SHORT":
        return "60m_short_persistence_baseline_rr2_5"
    return "60m_split_portfolio_frozen"


def baseline_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    by_id = summary.set_index("channel_id")
    rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        spec = CHANNEL_BY_ID[row.channel_id]
        baseline_id = baseline_id_for_spec(spec)
        baseline = by_id.loc[baseline_id]
        baseline_trades = float(baseline["trades"])
        rows.append({
            "channel_id": row.channel_id,
            "channel_label": row.channel_label,
            "direction_scope": row.direction_scope,
            "profile": row.profile,
            "is_baseline": row.is_baseline,
            "baseline_channel_id": baseline_id,
            "trades": row.trades,
            "retention_rate_vs_baseline": float(row.trades / baseline_trades) if baseline_trades > 0 else 0.0,
            "win_rate": row.win_rate,
            "avg_win_loss_ratio": row.avg_win_loss_ratio,
            "profit_factor": row.profit_factor,
            "net_R": row.net_R,
            "max_drawdown_R": row.max_drawdown_R,
            "historical_test_net_R": row.historical_test_net_R,
            "best_10pct_removed_net_R": row.best_10pct_removed_net_R,
            "delta_trades": row.trades - baseline["trades"],
            "delta_win_rate": row.win_rate - baseline["win_rate"],
            "delta_avg_win_loss_ratio": row.avg_win_loss_ratio - baseline["avg_win_loss_ratio"],
            "delta_profit_factor": row.profit_factor - baseline["profit_factor"],
            "delta_net_R": row.net_R - baseline["net_R"],
            "delta_max_drawdown_R": row.max_drawdown_R - baseline["max_drawdown_R"],
            "delta_historical_test_net_R": row.historical_test_net_R - baseline["historical_test_net_R"],
            "delta_best_10pct_removed_net_R": row.best_10pct_removed_net_R - baseline["best_10pct_removed_net_R"],
            "improves_recent_period": row.historical_test_net_R > baseline["historical_test_net_R"],
            "improves_tail_robustness": row.best_10pct_removed_net_R > baseline["best_10pct_removed_net_R"],
            "improves_primary_goals": (
                row.historical_test_net_R > baseline["historical_test_net_R"]
                and row.best_10pct_removed_net_R > baseline["best_10pct_removed_net_R"]
                and row.max_drawdown_R <= baseline["max_drawdown_R"]
            ),
        })
    return pd.DataFrame(rows)


def run_benchmark(raw_5m: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    signal_map, signal_audit = build_all_signals(raw_5m)
    trade_frames: list[pd.DataFrame] = []
    qualification: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        trades = execute_channel(raw_5m, signal_map[spec.channel_id], spec, config)
        trade_frames.append(trades)
        audit = qualification_audit(trades, spec, config)
        audit["benchmark_score"] = benchmark_score(audit)
        qualification[spec.channel_id] = audit
        summary_rows.append({
            "channel_id": spec.channel_id,
            "channel_label": spec.label,
            "timeframe_minutes": 60,
            "profile": spec.profile,
            "direction_scope": spec.direction_scope,
            "is_portfolio": spec.is_portfolio,
            "is_baseline": spec.is_baseline,
            "configured_stop_atr_multiple": spec.stop_atr_multiple if not spec.is_portfolio else np.nan,
            "configured_reward_risk": spec.reward_risk if not spec.is_portfolio else np.nan,
            "configured_max_holding_hours": spec.max_holding_hours if not spec.is_portfolio else np.nan,
            **audit["metrics"],
            "active_months": audit["active_months"],
            "positive_months": audit["positive_months"],
            "positive_phases": audit["positive_phases"],
            "positive_rolling_quarters": audit["positive_rolling_quarters"],
            "worst_rolling_quarter_net_R": audit["worst_rolling_quarter_net_R"],
            "historical_test_trades": audit["historical_test_metrics"]["trades"],
            "historical_test_win_rate": audit["historical_test_metrics"]["win_rate"],
            "historical_test_profit_factor": audit["historical_test_metrics"]["profit_factor"],
            "historical_test_net_R": audit["historical_test_metrics"]["net_R"],
            "best_10pct_removed_net_R": audit["best_fraction_removed"]["net_R"],
            "max_single_positive_month_share": audit["max_single_positive_month_share"],
            "research_candidate": audit["research_candidate"],
            "benchmark_score": audit["benchmark_score"],
        })
    ledger = pd.concat([frame for frame in trade_frames if not frame.empty], ignore_index=True) if any(not f.empty for f in trade_frames) else pd.DataFrame()
    if not ledger.empty:
        ledger = ledger.sort_values(["channel_id", "entry_time_utc"]).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values(["research_candidate", "benchmark_score"], ascending=[False, False]).reset_index(drop=True)
    comparison = baseline_comparison(summary)
    leader = None
    candidates = summary.loc[~summary["is_baseline"]]
    if not candidates.empty:
        leader = {key: (value.item() if isinstance(value, np.generic) else value) for key, value in candidates.iloc[0].to_dict().items()}
    return {
        "signals": signal_map,
        "signal_audit": signal_audit,
        "trades": ledger,
        "summary": summary,
        "baseline_comparison": comparison,
        "monthly": monthly_metrics(ledger),
        "phases": phase_metrics(ledger, config),
        "directions": direction_metrics(ledger),
        "components": component_metrics(ledger),
        "rolling_quarters": rolling_quarter_metrics(ledger, config),
        "qualification": qualification,
        "research_leader": leader,
    }


def self_test(config: dict[str, Any]) -> None:
    assert len(CHANNELS) == 9
    assert sum(spec.is_baseline for spec in CHANNELS) == 3
    assert sum(spec.is_portfolio for spec in CHANNELS) == 3
    raw = synthetic_5m_data(45_000, seed=20261003)
    hourly = base.resample_hourly(raw)
    assert len(hourly) == len(raw) // 12
    signals, signal_audit = build_all_signals(raw)
    assert set(signals) == {spec.channel_id for spec in CHANNELS}
    assert len(signal_audit["channels"]) == 9
    assert len(signal_audit["portfolio_overlap"]) == 3

    long_baseline_times = set(pd.to_datetime(signals["60m_long_quality_baseline_rr2_5"].get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
    for channel_id in ("60m_long_recovery_rr2_0", "60m_long_recovery_rr2_5"):
        candidate = signals[channel_id]
        assert set(pd.to_datetime(candidate.get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True)).issubset(long_baseline_times)
        if not candidate.empty:
            assert set(candidate["direction"]) == {1}
    for channel_id in ("60m_short_persistence_baseline_rr2_5", "60m_short_continuation_rr2_5", "60m_short_continuation_rr3_0"):
        candidate = signals[channel_id]
        if not candidate.empty:
            assert set(candidate["direction"]) == {-1}
    assert list(signals["60m_long_recovery_rr2_0"].get("signal_time", [])) == list(signals["60m_long_recovery_rr2_5"].get("signal_time", []))
    assert list(signals["60m_short_continuation_rr2_5"].get("signal_time", [])) == list(signals["60m_short_continuation_rr3_0"].get("signal_time", []))

    result = run_benchmark(raw, config)
    assert len(result["summary"]) == 9
    assert len(result["baseline_comparison"]) == 9
    assert len(result["phases"]) == 27
    assert len(result["rolling_quarters"]) == 9 * 16
    assert not result["trades"].empty
    for channel_id, group in result["trades"].groupby("channel_id"):
        entries = pd.to_datetime(group["entry_time_utc"], utc=True)
        signals_at = pd.to_datetime(group["signal_time_utc"], utc=True)
        exits = pd.to_datetime(group["exit_time_utc"], utc=True)
        assert (entries >= signals_at).all(), channel_id
        assert (exits >= entries).all(), channel_id
        if len(group) > 1:
            assert (entries.iloc[1:].reset_index(drop=True) >= exits.iloc[:-1].reset_index(drop=True)).all(), channel_id
    for spec in CHANNELS:
        channel_trades = result["trades"].loc[result["trades"]["channel_id"] == spec.channel_id]
        if spec.direction_scope == "LONG" and not channel_trades.empty:
            assert set(channel_trades["direction"]) == {1}
        if spec.direction_scope == "SHORT" and not channel_trades.empty:
            assert set(channel_trades["direction"]) == {-1}
    sample = pd.DataFrame({"net_r": [2.0, -1.0, 2.0, -1.0]})
    assert abs(metrics(sample)["net_R"] - 2.0) < 1e-12
    print("V103_ENGINE_SELF_TEST_OK")


__all__ = [
    "CHANNELS",
    "ChannelSpec",
    "load_official_5m_data",
    "run_benchmark",
    "self_test",
    "synthetic_5m_data",
]
