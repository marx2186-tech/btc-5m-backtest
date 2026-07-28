from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

import _v103_engine as v103


ChannelSpec = v103.ChannelSpec


CHANNELS: tuple[ChannelSpec, ...] = (
    # Frozen directional baselines.
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
        "60m_long_extension_guard_rr2_5",
        "多头独立策略·回踩延伸保护 RR2.5",
        "long_extension_guard",
        "LONG",
        1.25,
        2.5,
        36,
        False,
    ),
    ChannelSpec(
        "60m_long_extension_adx_cap_rr2_5",
        "多头独立策略·延伸保护+ADX上限 RR2.5",
        "long_extension_adx_cap",
        "LONG",
        1.25,
        2.5,
        36,
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
        "60m_short_persistence_age_relax_rr2_5",
        "空头独立策略·仅放宽趋势年龄 RR2.5",
        "short_persistence_age_relax",
        "SHORT",
        1.25,
        2.5,
        48,
        False,
    ),
    ChannelSpec(
        "60m_short_persistence_slope_relax_rr2_5",
        "空头独立策略·仅放宽EMA50斜率 RR2.5",
        "short_persistence_slope_relax",
        "SHORT",
        1.25,
        2.5,
        48,
        False,
    ),
    ChannelSpec(
        "60m_short_persistence_adx_relax_rr2_5",
        "空头独立策略·仅放宽ADX衰减 RR2.5",
        "short_persistence_adx_relax",
        "SHORT",
        1.25,
        2.5,
        48,
        False,
    ),
    # Shared-position portfolios. Each candidate changes only one directional component.
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
        "60m_split_portfolio_long_extension",
        "多空分离组合·仅优化多头延伸保护",
        "split_portfolio_long_extension",
        "BOTH",
        0.0,
        0.0,
        0,
        False,
        "60m_long_extension_guard_rr2_5",
        "60m_short_persistence_baseline_rr2_5",
    ),
    ChannelSpec(
        "60m_split_portfolio_long_extension_adx",
        "多空分离组合·仅优化多头延伸+ADX",
        "split_portfolio_long_extension_adx",
        "BOTH",
        0.0,
        0.0,
        0,
        False,
        "60m_long_extension_adx_cap_rr2_5",
        "60m_short_persistence_baseline_rr2_5",
    ),
    ChannelSpec(
        "60m_split_portfolio_short_age",
        "多空分离组合·仅放宽空头趋势年龄",
        "split_portfolio_short_age",
        "BOTH",
        0.0,
        0.0,
        0,
        False,
        "60m_long_quality_baseline_rr2_5",
        "60m_short_persistence_age_relax_rr2_5",
    ),
    ChannelSpec(
        "60m_split_portfolio_short_slope",
        "多空分离组合·仅放宽空头EMA50斜率",
        "split_portfolio_short_slope",
        "BOTH",
        0.0,
        0.0,
        0,
        False,
        "60m_long_quality_baseline_rr2_5",
        "60m_short_persistence_slope_relax_rr2_5",
    ),
    ChannelSpec(
        "60m_split_portfolio_short_adx",
        "多空分离组合·仅放宽空头ADX衰减",
        "split_portfolio_short_adx",
        "BOTH",
        0.0,
        0.0,
        0,
        False,
        "60m_long_quality_baseline_rr2_5",
        "60m_short_persistence_adx_relax_rr2_5",
    ),
)

CHANNEL_BY_ID = {spec.channel_id: spec for spec in CHANNELS}

# Make inherited phase/rolling helpers iterate over the V10.4 channel set.
v103.CHANNELS = CHANNELS
v103.CHANNEL_BY_ID = CHANNEL_BY_ID


PARAMETER_CHANGE_AUDIT: tuple[dict[str, Any], ...] = (
    {
        "channel_id": "60m_long_extension_guard_rr2_5",
        "baseline_channel_id": "60m_long_quality_baseline_rr2_5",
        "direction": "LONG",
        "changed_condition_count": 1,
        "changed_condition": "price_extension_ema20_atr <= 1.20",
        "change_type": "ADD_FILTER",
    },
    {
        "channel_id": "60m_long_extension_adx_cap_rr2_5",
        "baseline_channel_id": "60m_long_quality_baseline_rr2_5",
        "direction": "LONG",
        "changed_condition_count": 2,
        "changed_condition": "price_extension_ema20_atr <= 1.20; adx14 <= 45.0",
        "change_type": "ADD_TWO_FILTERS",
    },
    {
        "channel_id": "60m_short_persistence_age_relax_rr2_5",
        "baseline_channel_id": "60m_short_persistence_baseline_rr2_5",
        "direction": "SHORT",
        "changed_condition_count": 1,
        "changed_condition": "trend_age_short 8..120 -> 6..144",
        "change_type": "RELAX_ONE_FILTER",
    },
    {
        "channel_id": "60m_short_persistence_slope_relax_rr2_5",
        "baseline_channel_id": "60m_short_persistence_baseline_rr2_5",
        "direction": "SHORT",
        "changed_condition_count": 1,
        "changed_condition": "ema50_slope_6_atr <= -0.04 -> <= -0.025",
        "change_type": "RELAX_ONE_FILTER",
    },
    {
        "channel_id": "60m_short_persistence_adx_relax_rr2_5",
        "baseline_channel_id": "60m_short_persistence_baseline_rr2_5",
        "direction": "SHORT",
        "changed_condition_count": 1,
        "changed_condition": "adx_change_3 >= -3.0 -> >= -4.0",
        "change_type": "RELAX_ONE_FILTER",
    },
)


def load_official_5m_data(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    return v103.load_official_5m_data(config)


def synthetic_5m_data(rows: int = 60_000, seed: int = 1014) -> pd.DataFrame:
    return v103.synthetic_5m_data(rows, seed)


def long_quality_baseline_mask(x: pd.DataFrame) -> pd.Series:
    return v103.long_quality_baseline_mask(x)


def long_extension_guard_mask(x: pd.DataFrame) -> pd.Series:
    return long_quality_baseline_mask(x) & (x["price_extension_ema20_atr"] <= 1.20)


def long_extension_adx_cap_mask(x: pd.DataFrame) -> pd.Series:
    return long_extension_guard_mask(x) & (x["adx14"] <= 45.0)


def _short_persistence_mask(
    x: pd.DataFrame,
    *,
    trend_age_min: float = 8.0,
    trend_age_max: float = 120.0,
    ema50_slope_max: float = -0.04,
    adx_change_min: float = -3.0,
) -> pd.Series:
    # Reconstruct the frozen V10.2 persistence short mask and alter only explicit arguments.
    base_short = v103.short_quality_reference_mask(x)
    common = (
        x["adx14"].between(23.0, 45.0)
        & (x["adx_change_3"] >= adx_change_min)
        & x["volume_ratio"].between(0.70, 2.20)
        & x["channel_width_atr"].between(3.0, 8.0)
        & (x["price_extension_ema20_atr"] <= 0.80)
    )
    return (
        base_short
        & common
        & x["trend_age_short"].between(trend_age_min, trend_age_max)
        & (x["ema50_slope_6_atr"] <= ema50_slope_max)
        & (x["close_below_ema20_count_3"] >= 2.0)
    )


def short_persistence_baseline_mask(x: pd.DataFrame) -> pd.Series:
    # Use inherited frozen implementation and verify equivalence in self-test.
    return v103.short_persistence_baseline_mask(x)


def short_persistence_age_relax_mask(x: pd.DataFrame) -> pd.Series:
    return _short_persistence_mask(x, trend_age_min=6.0, trend_age_max=144.0)


def short_persistence_slope_relax_mask(x: pd.DataFrame) -> pd.Series:
    return _short_persistence_mask(x, ema50_slope_max=-0.025)


def short_persistence_adx_relax_mask(x: pd.DataFrame) -> pd.Series:
    return _short_persistence_mask(x, adx_change_min=-4.0)


def _signal_rows(mask: pd.Series, direction: int, x: pd.DataFrame, spec: ChannelSpec) -> pd.DataFrame:
    return v103._signal_rows(mask, direction, x, spec)


def generate_directional_signals(hourly_features: pd.DataFrame, spec: ChannelSpec) -> pd.DataFrame:
    if spec.profile == "long_quality_baseline":
        mask, direction = long_quality_baseline_mask(hourly_features), 1
    elif spec.profile == "long_extension_guard":
        mask, direction = long_extension_guard_mask(hourly_features), 1
    elif spec.profile == "long_extension_adx_cap":
        mask, direction = long_extension_adx_cap_mask(hourly_features), 1
    elif spec.profile == "short_persistence_baseline":
        mask, direction = short_persistence_baseline_mask(hourly_features), -1
    elif spec.profile == "short_persistence_age_relax":
        mask, direction = short_persistence_age_relax_mask(hourly_features), -1
    elif spec.profile == "short_persistence_slope_relax":
        mask, direction = short_persistence_slope_relax_mask(hourly_features), -1
    elif spec.profile == "short_persistence_adx_relax":
        mask, direction = short_persistence_adx_relax_mask(hourly_features), -1
    else:
        raise ValueError(f"Unsupported V10.4 directional profile: {spec.profile}")
    signals = _signal_rows(mask, direction, hourly_features, spec)
    if signals.empty:
        return signals
    return signals.sort_values("signal_time").drop_duplicates("signal_time", keep="first").reset_index(drop=True)


def build_all_signals(raw_5m: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    hourly = v103.base.resample_hourly(raw_5m)
    x = v103.base.add_hourly_features(hourly)
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
        signals, audit = v103.build_portfolio_signals(spec, signal_map)
        signal_map[spec.channel_id] = signals
        portfolio_audit.append(audit)
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
        "parameter_change_audit": list(PARAMETER_CHANGE_AUDIT),
    }


def execute_channel(
    raw_5m: pd.DataFrame,
    signals: pd.DataFrame,
    spec: ChannelSpec,
    config: dict[str, Any],
) -> pd.DataFrame:
    return v103.execute_channel(raw_5m, signals, spec, config)


def metrics(trades: pd.DataFrame | Iterable[dict[str, Any]]) -> dict[str, float]:
    return v103.metrics(trades)


def remove_best_fraction(trades: pd.DataFrame, fraction: float = 0.10) -> tuple[pd.DataFrame, dict[str, Any]]:
    return v103.remove_best_fraction(trades, fraction)


def monthly_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    return v103.monthly_metrics(trades)


def phase_metrics(trades: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        channel_trades = trades.loc[trades["channel_id"] == spec.channel_id] if not trades.empty else pd.DataFrame()
        for phase_name in config["phases"]:
            phase_trades = channel_trades.loc[channel_trades["phase"] == phase_name] if not channel_trades.empty else pd.DataFrame()
            rows.append({"channel_id": spec.channel_id, "channel_label": spec.label, "phase": phase_name, **metrics(phase_trades)})
    return pd.DataFrame(rows)


def direction_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    return v103.direction_metrics(trades)


def component_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    return v103.component_metrics(trades)


def rolling_quarter_metrics(trades: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = ["channel_id", "channel_label", "window_start_month", "window_end_month"] + list(metrics(pd.DataFrame()).keys())
    all_months = v103.month_range(str(config["start_month"]), str(config["end_month"]))
    rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        channel_trades = trades.loc[trades["channel_id"] == spec.channel_id] if not trades.empty else pd.DataFrame()
        for index in range(2, len(all_months)):
            window = all_months[index - 2:index + 1]
            subset = channel_trades.loc[channel_trades["month"].isin(window)] if not channel_trades.empty else pd.DataFrame()
            rows.append(
                {
                    "channel_id": spec.channel_id,
                    "channel_label": spec.label,
                    "window_start_month": window[0],
                    "window_end_month": window[-1],
                    **metrics(subset),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _thresholds_for_scope(spec: ChannelSpec, config: dict[str, Any]) -> dict[str, Any]:
    if spec.is_portfolio:
        key = "portfolio"
    elif spec.direction_scope == "LONG":
        key = "long"
    elif spec.direction_scope == "SHORT":
        key = "short"
    else:
        raise ValueError(f"Unsupported scope: {spec.direction_scope}")
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
    return v103.benchmark_score(audit)


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
        rows.append(
            {
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
            }
        )
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
        summary_rows.append(
            {
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
            }
        )
    nonempty = [frame for frame in trade_frames if not frame.empty]
    ledger = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
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
        "parameter_change_audit": pd.DataFrame(PARAMETER_CHANGE_AUDIT),
    }


def self_test(config: dict[str, Any]) -> None:
    assert len(CHANNELS) == 13
    assert sum(spec.is_baseline for spec in CHANNELS) == 3
    assert sum(spec.is_portfolio for spec in CHANNELS) == 6
    raw = synthetic_5m_data(55_000, seed=20261004)
    hourly = v103.base.resample_hourly(raw)
    x = v103.base.add_hourly_features(hourly)

    inherited_short = short_persistence_baseline_mask(x).fillna(False)
    reconstructed_short = _short_persistence_mask(x).fillna(False)
    assert inherited_short.equals(reconstructed_short), "Reconstructed short baseline differs from frozen V10.2 mask"

    signals, signal_audit = build_all_signals(raw)
    assert set(signals) == {spec.channel_id for spec in CHANNELS}
    assert len(signal_audit["channels"]) == 13
    assert len(signal_audit["portfolio_overlap"]) == 6
    assert len(signal_audit["parameter_change_audit"]) == 5

    long_base_times = set(pd.to_datetime(signals["60m_long_quality_baseline_rr2_5"].get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
    long_guard_times = set(pd.to_datetime(signals["60m_long_extension_guard_rr2_5"].get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
    long_strict_times = set(pd.to_datetime(signals["60m_long_extension_adx_cap_rr2_5"].get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
    assert long_strict_times.issubset(long_guard_times)
    assert long_guard_times.issubset(long_base_times)

    short_base_times = set(pd.to_datetime(signals["60m_short_persistence_baseline_rr2_5"].get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
    for channel_id in (
        "60m_short_persistence_age_relax_rr2_5",
        "60m_short_persistence_slope_relax_rr2_5",
        "60m_short_persistence_adx_relax_rr2_5",
    ):
        candidate_times = set(pd.to_datetime(signals[channel_id].get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
        assert short_base_times.issubset(candidate_times), channel_id

    result = run_benchmark(raw, config)
    assert len(result["summary"]) == 13
    assert len(result["baseline_comparison"]) == 13
    assert len(result["phases"]) == 39
    assert len(result["rolling_quarters"]) == 13 * 16
    assert len(result["parameter_change_audit"]) == 5
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

    print("V104_ENGINE_SELF_TEST_OK")


__all__ = [
    "CHANNELS",
    "ChannelSpec",
    "PARAMETER_CHANGE_AUDIT",
    "load_official_5m_data",
    "run_benchmark",
    "self_test",
    "synthetic_5m_data",
]
