from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

import _v104_engine as v104


ChannelSpec = v104.ChannelSpec


CHANNELS: tuple[ChannelSpec, ...] = (
    # Directional controls and candidates.
    ChannelSpec(
        "60m_long_extension_adx45_rr2_5",
        "多头冻结基线·延伸保护+ADX≤45 RR2.5",
        "long_extension_adx45",
        "LONG",
        1.25,
        2.5,
        36,
        True,
    ),
    ChannelSpec(
        "60m_long_extension_adx47_5_rr2_5",
        "多头独立策略·延伸保护+ADX≤47.5 RR2.5",
        "long_extension_adx47_5",
        "LONG",
        1.25,
        2.5,
        36,
        False,
    ),
    ChannelSpec(
        "60m_long_extension_adx50_rr2_5",
        "多头独立策略·延伸保护+ADX≤50 RR2.5",
        "long_extension_adx50",
        "LONG",
        1.25,
        2.5,
        36,
        False,
    ),
    ChannelSpec(
        "60m_short_persistence_reference_rr2_5",
        "空头冻结参考·趋势持续性 RR2.5",
        "short_persistence_reference",
        "SHORT",
        1.25,
        2.5,
        48,
        True,
    ),
    ChannelSpec(
        "60m_short_persistence_adx_relax_frozen_rr2_5",
        "空头冻结优化·ADX衰减≥-4 RR2.5",
        "short_persistence_adx_relax_frozen",
        "SHORT",
        1.25,
        2.5,
        48,
        False,
    ),
    # 3 x 2 controlled portfolio grid: long ADX cap x short mode.
    ChannelSpec(
        "60m_split_portfolio_v104_leader",
        "多空组合冻结基线·多ADX45/空原始",
        "split_portfolio_v104_leader",
        "BOTH",
        0.0,
        0.0,
        0,
        True,
        "60m_long_extension_adx45_rr2_5",
        "60m_short_persistence_reference_rr2_5",
    ),
    ChannelSpec(
        "60m_split_portfolio_interaction_45",
        "多空交互组合·多ADX45/空ADX-4",
        "split_portfolio_interaction_45",
        "BOTH",
        0.0,
        0.0,
        0,
        False,
        "60m_long_extension_adx45_rr2_5",
        "60m_short_persistence_adx_relax_frozen_rr2_5",
    ),
    ChannelSpec(
        "60m_split_portfolio_long47_reference",
        "多空对照组合·多ADX47.5/空原始",
        "split_portfolio_long47_reference",
        "BOTH",
        0.0,
        0.0,
        0,
        False,
        "60m_long_extension_adx47_5_rr2_5",
        "60m_short_persistence_reference_rr2_5",
    ),
    ChannelSpec(
        "60m_split_portfolio_interaction_47_5",
        "多空交互组合·多ADX47.5/空ADX-4",
        "split_portfolio_interaction_47_5",
        "BOTH",
        0.0,
        0.0,
        0,
        False,
        "60m_long_extension_adx47_5_rr2_5",
        "60m_short_persistence_adx_relax_frozen_rr2_5",
    ),
    ChannelSpec(
        "60m_split_portfolio_long50_reference",
        "多空对照组合·多ADX50/空原始",
        "split_portfolio_long50_reference",
        "BOTH",
        0.0,
        0.0,
        0,
        False,
        "60m_long_extension_adx50_rr2_5",
        "60m_short_persistence_reference_rr2_5",
    ),
    ChannelSpec(
        "60m_split_portfolio_interaction_50",
        "多空交互组合·多ADX50/空ADX-4",
        "split_portfolio_interaction_50",
        "BOTH",
        0.0,
        0.0,
        0,
        False,
        "60m_long_extension_adx50_rr2_5",
        "60m_short_persistence_adx_relax_frozen_rr2_5",
    ),
)

CHANNEL_BY_ID = {spec.channel_id: spec for spec in CHANNELS}

# Make inherited helpers use the V10.5 universe where they inspect global channel sets.
v104.CHANNELS = CHANNELS
v104.CHANNEL_BY_ID = CHANNEL_BY_ID
v104.v103.CHANNELS = CHANNELS
v104.v103.CHANNEL_BY_ID = CHANNEL_BY_ID


PARAMETER_CHANGE_AUDIT: tuple[dict[str, Any], ...] = (
    {
        "channel_id": "60m_long_extension_adx47_5_rr2_5",
        "baseline_channel_id": "60m_long_extension_adx45_rr2_5",
        "direction": "LONG",
        "changed_condition_count": 1,
        "changed_condition": "adx14 upper cap 45.0 -> 47.5",
        "change_type": "RELAX_ONE_FILTER",
    },
    {
        "channel_id": "60m_long_extension_adx50_rr2_5",
        "baseline_channel_id": "60m_long_extension_adx45_rr2_5",
        "direction": "LONG",
        "changed_condition_count": 1,
        "changed_condition": "adx14 upper cap 45.0 -> 50.0",
        "change_type": "RELAX_ONE_FILTER",
    },
    {
        "channel_id": "60m_short_persistence_adx_relax_frozen_rr2_5",
        "baseline_channel_id": "60m_short_persistence_reference_rr2_5",
        "direction": "SHORT",
        "changed_condition_count": 1,
        "changed_condition": "adx_change_3 minimum -3.0 -> -4.0",
        "change_type": "FROZEN_EFFECTIVE_RELAXATION",
    },
)


PORTFOLIO_FACTOR_GRID: tuple[dict[str, Any], ...] = (
    {"channel_id": "60m_split_portfolio_v104_leader", "long_adx_cap": 45.0, "short_mode": "REFERENCE_-3", "is_v104_control": True},
    {"channel_id": "60m_split_portfolio_interaction_45", "long_adx_cap": 45.0, "short_mode": "RELAXED_-4", "is_v104_control": False},
    {"channel_id": "60m_split_portfolio_long47_reference", "long_adx_cap": 47.5, "short_mode": "REFERENCE_-3", "is_v104_control": False},
    {"channel_id": "60m_split_portfolio_interaction_47_5", "long_adx_cap": 47.5, "short_mode": "RELAXED_-4", "is_v104_control": False},
    {"channel_id": "60m_split_portfolio_long50_reference", "long_adx_cap": 50.0, "short_mode": "REFERENCE_-3", "is_v104_control": False},
    {"channel_id": "60m_split_portfolio_interaction_50", "long_adx_cap": 50.0, "short_mode": "RELAXED_-4", "is_v104_control": False},
)


INTERACTION_PAIRS: tuple[dict[str, str], ...] = (
    {
        "effect_id": "short_relax_effect_at_long45",
        "factor": "SHORT_ADX_CHANGE_MIN_-3_TO_-4",
        "control_channel_id": "60m_split_portfolio_v104_leader",
        "variant_channel_id": "60m_split_portfolio_interaction_45",
    },
    {
        "effect_id": "short_relax_effect_at_long47_5",
        "factor": "SHORT_ADX_CHANGE_MIN_-3_TO_-4",
        "control_channel_id": "60m_split_portfolio_long47_reference",
        "variant_channel_id": "60m_split_portfolio_interaction_47_5",
    },
    {
        "effect_id": "short_relax_effect_at_long50",
        "factor": "SHORT_ADX_CHANGE_MIN_-3_TO_-4",
        "control_channel_id": "60m_split_portfolio_long50_reference",
        "variant_channel_id": "60m_split_portfolio_interaction_50",
    },
    {
        "effect_id": "long_cap_47_5_effect_with_short_reference",
        "factor": "LONG_ADX_CAP_45_TO_47_5",
        "control_channel_id": "60m_split_portfolio_v104_leader",
        "variant_channel_id": "60m_split_portfolio_long47_reference",
    },
    {
        "effect_id": "long_cap_50_effect_with_short_reference",
        "factor": "LONG_ADX_CAP_45_TO_50",
        "control_channel_id": "60m_split_portfolio_v104_leader",
        "variant_channel_id": "60m_split_portfolio_long50_reference",
    },
    {
        "effect_id": "long_cap_47_5_effect_with_short_relaxed",
        "factor": "LONG_ADX_CAP_45_TO_47_5",
        "control_channel_id": "60m_split_portfolio_interaction_45",
        "variant_channel_id": "60m_split_portfolio_interaction_47_5",
    },
    {
        "effect_id": "long_cap_50_effect_with_short_relaxed",
        "factor": "LONG_ADX_CAP_45_TO_50",
        "control_channel_id": "60m_split_portfolio_interaction_45",
        "variant_channel_id": "60m_split_portfolio_interaction_50",
    },
)


def load_official_5m_data(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    return v104.load_official_5m_data(config)


def synthetic_5m_data(rows: int = 60_000, seed: int = 1015) -> pd.DataFrame:
    return v104.synthetic_5m_data(rows, seed)


def long_extension_mask(x: pd.DataFrame, adx_cap: float) -> pd.Series:
    return v104.long_extension_guard_mask(x) & (x["adx14"] <= float(adx_cap))


def short_reference_mask(x: pd.DataFrame) -> pd.Series:
    return v104.short_persistence_baseline_mask(x)


def short_adx_relax_frozen_mask(x: pd.DataFrame) -> pd.Series:
    return v104.short_persistence_adx_relax_mask(x)


def _signal_rows(mask: pd.Series, direction: int, x: pd.DataFrame, spec: ChannelSpec) -> pd.DataFrame:
    return v104._signal_rows(mask, direction, x, spec)


def generate_directional_signals(hourly_features: pd.DataFrame, spec: ChannelSpec) -> pd.DataFrame:
    if spec.profile == "long_extension_adx45":
        mask, direction = long_extension_mask(hourly_features, 45.0), 1
    elif spec.profile == "long_extension_adx47_5":
        mask, direction = long_extension_mask(hourly_features, 47.5), 1
    elif spec.profile == "long_extension_adx50":
        mask, direction = long_extension_mask(hourly_features, 50.0), 1
    elif spec.profile == "short_persistence_reference":
        mask, direction = short_reference_mask(hourly_features), -1
    elif spec.profile == "short_persistence_adx_relax_frozen":
        mask, direction = short_adx_relax_frozen_mask(hourly_features), -1
    else:
        raise ValueError(f"Unsupported V10.5 directional profile: {spec.profile}")
    signals = _signal_rows(mask, direction, hourly_features, spec)
    if signals.empty:
        return signals
    return signals.sort_values("signal_time").drop_duplicates("signal_time", keep="first").reset_index(drop=True)


def build_all_signals(raw_5m: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    hourly = v104.v103.base.resample_hourly(raw_5m)
    x = v104.v103.base.add_hourly_features(hourly)
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
        signals, audit = v104.v103.build_portfolio_signals(spec, signal_map)
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
        "portfolio_factor_grid": list(PORTFOLIO_FACTOR_GRID),
    }


def execute_channel(raw_5m: pd.DataFrame, signals: pd.DataFrame, spec: ChannelSpec, config: dict[str, Any]) -> pd.DataFrame:
    return v104.execute_channel(raw_5m, signals, spec, config)


def metrics(trades: pd.DataFrame | Iterable[dict[str, Any]]) -> dict[str, float]:
    return v104.metrics(trades)


def remove_best_fraction(trades: pd.DataFrame, fraction: float = 0.10) -> tuple[pd.DataFrame, dict[str, Any]]:
    return v104.remove_best_fraction(trades, fraction)


def monthly_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    return v104.monthly_metrics(trades)


def phase_metrics(trades: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        channel_trades = trades.loc[trades["channel_id"] == spec.channel_id] if not trades.empty else pd.DataFrame()
        for phase_name in config["phases"]:
            phase_trades = channel_trades.loc[channel_trades["phase"] == phase_name] if not channel_trades.empty else pd.DataFrame()
            rows.append({"channel_id": spec.channel_id, "channel_label": spec.label, "phase": phase_name, **metrics(phase_trades)})
    return pd.DataFrame(rows)


def direction_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    return v104.direction_metrics(trades)


def component_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    return v104.component_metrics(trades)


def rolling_quarter_metrics(trades: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = ["channel_id", "channel_label", "window_start_month", "window_end_month"] + list(metrics(pd.DataFrame()).keys())
    all_months = v104.v103.month_range(str(config["start_month"]), str(config["end_month"]))
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
    return v104.benchmark_score(audit)


def baseline_id_for_spec(spec: ChannelSpec) -> str:
    if spec.direction_scope == "LONG":
        return "60m_long_extension_adx45_rr2_5"
    if spec.direction_scope == "SHORT":
        return "60m_short_persistence_reference_rr2_5"
    return "60m_split_portfolio_v104_leader"


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
                    row.historical_test_net_R >= baseline["historical_test_net_R"]
                    and row.best_10pct_removed_net_R >= baseline["best_10pct_removed_net_R"]
                    and row.max_drawdown_R <= baseline["max_drawdown_R"]
                ),
            }
        )
    return pd.DataFrame(rows)


def interaction_effect_audit(summary: pd.DataFrame) -> pd.DataFrame:
    by_id = summary.set_index("channel_id")
    rows: list[dict[str, Any]] = []
    for pair in INTERACTION_PAIRS:
        control = by_id.loc[pair["control_channel_id"]]
        variant = by_id.loc[pair["variant_channel_id"]]
        rows.append(
            {
                **pair,
                "control_channel_label": control["channel_label"],
                "variant_channel_label": variant["channel_label"],
                "delta_trades": int(variant["trades"] - control["trades"]),
                "delta_win_rate": float(variant["win_rate"] - control["win_rate"]),
                "delta_avg_win_loss_ratio": float(variant["avg_win_loss_ratio"] - control["avg_win_loss_ratio"]),
                "delta_profit_factor": float(variant["profit_factor"] - control["profit_factor"]),
                "delta_net_R": float(variant["net_R"] - control["net_R"]),
                "delta_max_drawdown_R": float(variant["max_drawdown_R"] - control["max_drawdown_R"]),
                "delta_historical_test_net_R": float(variant["historical_test_net_R"] - control["historical_test_net_R"]),
                "delta_best_10pct_removed_net_R": float(variant["best_10pct_removed_net_R"] - control["best_10pct_removed_net_R"]),
                "interaction_improves_q2_tail_and_drawdown": bool(
                    variant["historical_test_net_R"] >= control["historical_test_net_R"]
                    and variant["best_10pct_removed_net_R"] >= control["best_10pct_removed_net_R"]
                    and variant["max_drawdown_R"] <= control["max_drawdown_R"]
                ),
            }
        )
    return pd.DataFrame(rows)


def factor_grid(summary: pd.DataFrame) -> pd.DataFrame:
    by_id = summary.set_index("channel_id")
    rows: list[dict[str, Any]] = []
    for item in PORTFOLIO_FACTOR_GRID:
        row = by_id.loc[item["channel_id"]]
        rows.append(
            {
                **item,
                "channel_label": row["channel_label"],
                "trades": int(row["trades"]),
                "win_rate": float(row["win_rate"]),
                "avg_win_loss_ratio": float(row["avg_win_loss_ratio"]),
                "profit_factor": float(row["profit_factor"]),
                "net_R": float(row["net_R"]),
                "max_drawdown_R": float(row["max_drawdown_R"]),
                "historical_test_net_R": float(row["historical_test_net_R"]),
                "best_10pct_removed_net_R": float(row["best_10pct_removed_net_R"]),
                "research_candidate": bool(row["research_candidate"]),
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
    interactions = interaction_effect_audit(summary)
    grid = factor_grid(summary)
    leader = None
    candidates = summary.loc[(~summary["is_baseline"]) & summary["is_portfolio"]]
    if not candidates.empty:
        leader = {key: (value.item() if isinstance(value, np.generic) else value) for key, value in candidates.iloc[0].to_dict().items()}
    return {
        "signals": signal_map,
        "signal_audit": signal_audit,
        "trades": ledger,
        "summary": summary,
        "baseline_comparison": comparison,
        "interaction_effect_audit": interactions,
        "factor_grid": grid,
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
    assert len(CHANNELS) == 11
    assert sum(spec.is_baseline for spec in CHANNELS) == 3
    assert sum(spec.is_portfolio for spec in CHANNELS) == 6
    raw = synthetic_5m_data(60_000, seed=20261005)
    signals, signal_audit = build_all_signals(raw)
    assert set(signals) == {spec.channel_id for spec in CHANNELS}
    assert len(signal_audit["channels"]) == 11
    assert len(signal_audit["portfolio_overlap"]) == 6
    assert len(signal_audit["parameter_change_audit"]) == 3
    assert len(signal_audit["portfolio_factor_grid"]) == 6

    long45 = set(pd.to_datetime(signals["60m_long_extension_adx45_rr2_5"].get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
    long47 = set(pd.to_datetime(signals["60m_long_extension_adx47_5_rr2_5"].get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
    long50 = set(pd.to_datetime(signals["60m_long_extension_adx50_rr2_5"].get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
    assert long45.issubset(long47)
    assert long47.issubset(long50)

    short_reference = set(pd.to_datetime(signals["60m_short_persistence_reference_rr2_5"].get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
    short_relaxed = set(pd.to_datetime(signals["60m_short_persistence_adx_relax_frozen_rr2_5"].get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
    assert short_reference.issubset(short_relaxed)

    result = run_benchmark(raw, config)
    assert len(result["summary"]) == 11
    assert len(result["baseline_comparison"]) == 11
    assert len(result["interaction_effect_audit"]) == 7
    assert len(result["factor_grid"]) == 6
    assert len(result["phases"]) == 33
    assert len(result["rolling_quarters"]) == 11 * 16
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

    print("V105_ENGINE_SELF_TEST_OK")


__all__ = [
    "CHANNELS",
    "CHANNEL_BY_ID",
    "ChannelSpec",
    "PARAMETER_CHANGE_AUDIT",
    "PORTFOLIO_FACTOR_GRID",
    "INTERACTION_PAIRS",
    "load_official_5m_data",
    "run_benchmark",
    "self_test",
    "synthetic_5m_data",
]
