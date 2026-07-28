from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

import _v105_engine as v105


ChannelSpec = v105.ChannelSpec


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec(
        "60m_long_frozen_adx45_rr2_5",
        "V10.6冻结多头·延伸保护+ADX≤45 RR2.5",
        "long_frozen_adx45",
        "LONG",
        1.25,
        2.5,
        36,
        True,
    ),
    ChannelSpec(
        "60m_short_frozen_adx_decay_minus4_rr2_5",
        "V10.6冻结空头·趋势持续性+ADX衰减≥-4 RR2.5",
        "short_frozen_adx_decay_minus4",
        "SHORT",
        1.25,
        2.5,
        48,
        True,
    ),
    ChannelSpec(
        "60m_portfolio_frozen_v10_6",
        "V10.6冻结多空组合·共享单一仓位",
        "portfolio_frozen_v10_6",
        "BOTH",
        0.0,
        0.0,
        0,
        True,
        "60m_long_frozen_adx45_rr2_5",
        "60m_short_frozen_adx_decay_minus4_rr2_5",
    ),
)

CHANNEL_BY_ID = {spec.channel_id: spec for spec in CHANNELS}

# Inherited helpers inspect module-level channel universes, so freeze every level to V10.6.
v105.CHANNELS = CHANNELS
v105.CHANNEL_BY_ID = CHANNEL_BY_ID
v105.v104.CHANNELS = CHANNELS
v105.v104.CHANNEL_BY_ID = CHANNEL_BY_ID
v105.v104.v103.CHANNELS = CHANNELS
v105.v104.v103.CHANNEL_BY_ID = CHANNEL_BY_ID

BASE = v105.v104.v103.base
STEP_MS = BASE.STEP_MS


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parameter_lock_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "release_version": "10.6",
        "source": "V10.5 historical leader 60m_split_portfolio_interaction_45",
        "signal_timeframe_minutes": 60,
        "channels": [asdict(spec) for spec in CHANNELS],
        "direction_specific_parameters": config["direction_specific_parameters"],
        "portfolio_rules": config["portfolio_rules"],
        "execution": config["execution"],
        "no_lookahead_rules": config["no_lookahead_rules"],
        "parameter_optimization_enabled": False,
        "unseen_holdout_start_utc": config["unseen_holdout"]["start_utc"],
        "true_forward_observation_start_utc": config["true_forward_observation"]["start_utc"],
    }
    payload["parameter_lock_sha256"] = canonical_sha256(payload)
    return payload


def load_official_5m_data(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load verified monthly history plus verified daily untouched post-period holdout files."""
    monthly_config = copy.deepcopy(config)
    monthly_config["start_month"] = str(config["start_month"])
    monthly_config["end_month"] = str(config["end_month"])
    monthly, monthly_audit = v105.load_official_5m_data(monthly_config)

    symbol = str(config["symbol"]).upper()
    interval = str(config["source_interval"])
    holdout_start = pd.Timestamp(config["unseen_holdout"]["start_utc"])
    holdout_end = pd.Timestamp(config["unseen_holdout"]["end_utc"])
    daily_dates = pd.date_range(holdout_start.normalize(), holdout_end.normalize(), freq="D", tz="UTC")
    daily_frames: list[pd.DataFrame] = []
    daily_files: list[dict[str, Any]] = []

    for day in daily_dates:
        day_text = day.strftime("%Y-%m-%d")
        name = f"{symbol}-{interval}-{day_text}.zip"
        base_url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/{interval}"
        raw, digest = BASE.read_verified_zip(
            f"{base_url}/{name}",
            f"{base_url}/{name}.CHECKSUM",
            name,
        )
        frame = BASE.parse_kline_csv(BASE.read_single_csv_zip(raw, name))
        daily_frames.append(frame)
        daily_files.append({"file": name, "sha256": digest, "rows": int(len(frame))})

    combined_parts = [monthly]
    if daily_frames:
        daily = pd.concat(daily_frames, ignore_index=True)
        daily["time"] = pd.to_datetime(pd.to_numeric(daily["open_time"], errors="raise").astype("int64"), unit="ms", utc=True)
        for column in ("open", "high", "low", "close", "volume"):
            daily[column] = pd.to_numeric(daily[column], errors="coerce").astype(float)
        combined_parts.append(daily)

    data = (
        pd.concat(combined_parts, ignore_index=True)
        .sort_values("open_time")
        .drop_duplicates("open_time", keep="last")
        .reset_index(drop=True)
    )

    start = pd.Timestamp(f"{config['start_month']}-01T00:00:00Z")
    exclusive_end = holdout_end.normalize() + pd.Timedelta(days=1)
    expected_times = np.arange(
        int(start.timestamp() * 1000),
        int(exclusive_end.timestamp() * 1000),
        STEP_MS,
        dtype=np.int64,
    )
    actual_times = pd.to_numeric(data["open_time"], errors="coerce").dropna().astype("int64").to_numpy()
    unique_times = np.unique(actual_times)
    missing = np.setdiff1d(expected_times, unique_times)
    extra = np.setdiff1d(unique_times, expected_times)
    duplicate_count = int(pd.Series(actual_times).duplicated().sum())

    o = data["open"].to_numpy(float)
    h = data["high"].to_numpy(float)
    l = data["low"].to_numpy(float)
    c = data["close"].to_numpy(float)
    finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    valid_ohlc = finite & (h >= np.maximum.reduce([o, c, l])) & (l <= np.minimum.reduce([o, c, h]))
    close_time = pd.to_numeric(data["close_time"], errors="coerce").fillna(-1).astype("int64").to_numpy()

    audit = {
        "source": "Binance USDⓈ-M Futures official verified monthly and daily 5m klines",
        "symbol": symbol,
        "interval": interval,
        "monthly_start": str(config["start_month"]),
        "monthly_end": str(config["end_month"]),
        "unseen_daily_start": daily_dates.min().strftime("%Y-%m-%d") if len(daily_dates) else None,
        "unseen_daily_end": daily_dates.max().strftime("%Y-%m-%d") if len(daily_dates) else None,
        "start_utc": start.isoformat(),
        "end_utc": pd.to_datetime(expected_times[-1], unit="ms", utc=True).isoformat(),
        "expected_rows": int(len(expected_times)),
        "actual_rows": int(len(data)),
        "unique_rows": int(len(unique_times)),
        "missing_rows": int(len(missing)),
        "extra_rows": int(len(extra)),
        "duplicate_timestamps": duplicate_count,
        "off_grid_rows": int(np.sum((actual_times - expected_times[0]) % STEP_MS != 0)),
        "invalid_close_time_rows": int(np.sum(close_time != actual_times + STEP_MS - 1)),
        "invalid_ohlc_rows": int(np.sum(~valid_ohlc)),
        "uses_rest_fallback": False,
        "monthly_audit": monthly_audit,
        "unseen_daily_files": daily_files,
    }
    audit["passed"] = bool(
        len(data) == len(expected_times)
        and len(unique_times) == len(expected_times)
        and len(missing) == 0
        and len(extra) == 0
        and duplicate_count == 0
        and audit["off_grid_rows"] == 0
        and audit["invalid_close_time_rows"] == 0
        and audit["invalid_ohlc_rows"] == 0
        and monthly_audit.get("passed") is True
    )
    if not audit["passed"]:
        raise RuntimeError("V10.6 official data audit failed: " + json.dumps(audit, ensure_ascii=False, default=str))

    data["time"] = pd.to_datetime(data["open_time"].astype("int64"), unit="ms", utc=True)
    return data, audit


def synthetic_5m_data(rows: int = 120_000, seed: int = 1016) -> pd.DataFrame:
    return v105.synthetic_5m_data(rows, seed)


def _signal_rows(mask: pd.Series, direction: int, x: pd.DataFrame, spec: ChannelSpec) -> pd.DataFrame:
    return v105._signal_rows(mask, direction, x, spec)


def generate_directional_signals(hourly_features: pd.DataFrame, spec: ChannelSpec) -> pd.DataFrame:
    if spec.profile == "long_frozen_adx45":
        mask, direction = v105.long_extension_mask(hourly_features, 45.0), 1
    elif spec.profile == "short_frozen_adx_decay_minus4":
        mask, direction = v105.short_adx_relax_frozen_mask(hourly_features), -1
    else:
        raise ValueError(f"Unsupported V10.6 directional profile: {spec.profile}")
    signals = _signal_rows(mask, direction, hourly_features, spec)
    if signals.empty:
        return signals
    return signals.sort_values("signal_time").drop_duplicates("signal_time", keep="first").reset_index(drop=True)


def build_all_signals(raw_5m: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    hourly = BASE.resample_hourly(raw_5m)
    x = BASE.add_hourly_features(hourly)
    signal_map: dict[str, pd.DataFrame] = {}
    channel_audit: list[dict[str, Any]] = []
    portfolio_audit: list[dict[str, Any]] = []

    for spec in CHANNELS:
        if spec.is_portfolio:
            continue
        signals = generate_directional_signals(x, spec)
        signal_map[spec.channel_id] = signals
        channel_audit.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "profile": spec.profile,
                "direction_scope": spec.direction_scope,
                "signals": int(len(signals)),
                "long_signals": int((signals["direction"] == 1).sum()) if not signals.empty else 0,
                "short_signals": int((signals["direction"] == -1).sum()) if not signals.empty else 0,
                "reward_risk": spec.reward_risk,
                "stop_atr_multiple": spec.stop_atr_multiple,
                "max_holding_hours": spec.max_holding_hours,
                "is_frozen": True,
            }
        )

    portfolio_spec = CHANNEL_BY_ID["60m_portfolio_frozen_v10_6"]
    portfolio_signals, audit = v105.v104.v103.build_portfolio_signals(portfolio_spec, signal_map)
    signal_map[portfolio_spec.channel_id] = portfolio_signals
    portfolio_audit.append(audit)
    channel_audit.append(
        {
            "channel_id": portfolio_spec.channel_id,
            "channel_label": portfolio_spec.label,
            "profile": portfolio_spec.profile,
            "direction_scope": portfolio_spec.direction_scope,
            "signals": int(len(portfolio_signals)),
            "long_signals": int((portfolio_signals["direction"] == 1).sum()) if not portfolio_signals.empty else 0,
            "short_signals": int((portfolio_signals["direction"] == -1).sum()) if not portfolio_signals.empty else 0,
            "reward_risk": "ASYMMETRIC_COMPONENT_FROZEN",
            "stop_atr_multiple": "ASYMMETRIC_COMPONENT_FROZEN",
            "max_holding_hours": "ASYMMETRIC_COMPONENT_FROZEN",
            "is_frozen": True,
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
        "channels": channel_audit,
        "portfolio_overlap": portfolio_audit,
    }


def execute_channel(raw_5m: pd.DataFrame, signals: pd.DataFrame, spec: ChannelSpec, config: dict[str, Any]) -> pd.DataFrame:
    return v105.execute_channel(raw_5m, signals, spec, config)


def metrics(trades: pd.DataFrame | Iterable[dict[str, Any]]) -> dict[str, float]:
    return v105.metrics(trades)


def remove_best_fraction(trades: pd.DataFrame, fraction: float = 0.10) -> tuple[pd.DataFrame, dict[str, Any]]:
    return v105.remove_best_fraction(trades, fraction)


def month_range(start_month: str, end_month: str) -> list[str]:
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    return [str(item) for item in pd.period_range(start, end, freq="M")]


def monthly_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["channel_id", "channel_label", "month"] + list(metrics(pd.DataFrame()).keys())
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (channel_id, channel_label, month), group in trades.groupby(["channel_id", "channel_label", "month"], sort=True):
        rows.append({"channel_id": channel_id, "channel_label": channel_label, "month": month, **metrics(group)})
    return pd.DataFrame(rows, columns=columns)


def yearly_metrics(trades: pd.DataFrame, years: list[str]) -> pd.DataFrame:
    columns = ["channel_id", "channel_label", "year"] + list(metrics(pd.DataFrame()).keys())
    rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        channel_trades = trades.loc[trades["channel_id"] == spec.channel_id].copy() if not trades.empty else pd.DataFrame()
        if not channel_trades.empty:
            channel_trades["year"] = pd.to_datetime(channel_trades["entry_time_utc"], utc=True).dt.strftime("%Y")
        for year in years:
            subset = channel_trades.loc[channel_trades["year"] == year] if not channel_trades.empty else pd.DataFrame()
            rows.append({"channel_id": spec.channel_id, "channel_label": spec.label, "year": year, **metrics(subset)})
    return pd.DataFrame(rows, columns=columns)


def phase_metrics(trades: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = ["channel_id", "channel_label", "phase"] + list(metrics(pd.DataFrame()).keys())
    rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        channel_trades = trades.loc[trades["channel_id"] == spec.channel_id] if not trades.empty else pd.DataFrame()
        for phase_name in config["phases"]:
            subset = channel_trades.loc[channel_trades["phase"] == phase_name] if not channel_trades.empty else pd.DataFrame()
            rows.append({"channel_id": spec.channel_id, "channel_label": spec.label, "phase": phase_name, **metrics(subset)})
    return pd.DataFrame(rows, columns=columns)


def direction_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    return v105.direction_metrics(trades)


def component_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    return v105.component_metrics(trades)


def rolling_window_metrics(trades: pd.DataFrame, config: dict[str, Any], window_months: int) -> pd.DataFrame:
    columns = ["channel_id", "channel_label", "window_months", "window_start_month", "window_end_month"] + list(metrics(pd.DataFrame()).keys())
    all_months = month_range(str(config["analysis_start_month"]), str(config["analysis_end_month"]))
    rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        channel_trades = trades.loc[trades["channel_id"] == spec.channel_id] if not trades.empty else pd.DataFrame()
        for index in range(window_months - 1, len(all_months)):
            window = all_months[index - window_months + 1 : index + 1]
            subset = channel_trades.loc[channel_trades["month"].isin(window)] if not channel_trades.empty else pd.DataFrame()
            rows.append(
                {
                    "channel_id": spec.channel_id,
                    "channel_label": spec.label,
                    "window_months": window_months,
                    "window_start_month": window[0],
                    "window_end_month": window[-1],
                    **metrics(subset),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def year_removal_robustness(trades: pd.DataFrame, years: list[str]) -> pd.DataFrame:
    columns = ["channel_id", "channel_label", "removed_year"] + list(metrics(pd.DataFrame()).keys())
    rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        channel_trades = trades.loc[trades["channel_id"] == spec.channel_id].copy() if not trades.empty else pd.DataFrame()
        if not channel_trades.empty:
            channel_trades["year"] = pd.to_datetime(channel_trades["entry_time_utc"], utc=True).dt.strftime("%Y")
        for year in years:
            subset = channel_trades.loc[channel_trades["year"] != year] if not channel_trades.empty else pd.DataFrame()
            rows.append({"channel_id": spec.channel_id, "channel_label": spec.label, "removed_year": year, **metrics(subset)})
    return pd.DataFrame(rows, columns=columns)


def cost_stress_summary(
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
        for spec in CHANNELS:
            trades = execute_channel(raw_5m, signal_map[spec.channel_id], spec, stressed)
            overall = metrics(trades)
            unseen_holdout = metrics(trades.loc[trades["phase"] == "unseen_holdout"] if not trades.empty else pd.DataFrame())
            rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "fee_multiplier": float(scenario["fee_multiplier"]),
                    "slippage_multiplier": float(scenario["slippage_multiplier"]),
                    "fee_rate_per_side": stressed["execution"]["fee_rate_per_side"],
                    "slippage_ticks_per_fill": stressed["execution"]["slippage_ticks_per_fill"],
                    "channel_id": spec.channel_id,
                    "channel_label": spec.label,
                    **overall,
                    "unseen_holdout_trades": unseen_holdout["trades"],
                    "unseen_holdout_profit_factor": unseen_holdout["profit_factor"],
                    "unseen_holdout_net_R": unseen_holdout["net_R"],
                }
            )
    return pd.DataFrame(rows)


def _scope_thresholds(spec: ChannelSpec, config: dict[str, Any]) -> dict[str, Any]:
    if spec.is_portfolio:
        return config["cross_year_thresholds"]["portfolio"]
    if spec.direction_scope == "LONG":
        return config["cross_year_thresholds"]["long"]
    return config["cross_year_thresholds"]["short"]


def qualification_audit(
    channel_trades: pd.DataFrame,
    spec: ChannelSpec,
    config: dict[str, Any],
    yearly: pd.DataFrame,
    phases: pd.DataFrame,
    rolling_12m: pd.DataFrame,
    stress: pd.DataFrame,
    replay_passed: bool,
) -> dict[str, Any]:
    threshold = _scope_thresholds(spec, config)
    overall = metrics(channel_trades)
    _, tail = remove_best_fraction(channel_trades, float(threshold["remove_best_fraction"]))
    y = yearly.loc[yearly["channel_id"] == spec.channel_id]
    p = phases.loc[phases["channel_id"] == spec.channel_id]
    r12 = rolling_12m.loc[rolling_12m["channel_id"] == spec.channel_id]
    active_r12 = r12.loc[r12["trades"] > 0]
    s15 = stress.loc[(stress["channel_id"] == spec.channel_id) & (stress["scenario_id"] == "cost_1_5x")]

    positive_years = int((y["net_R"] > 0).sum())
    active_years = int((y["trades"] > 0).sum())
    positive_year_profit = float(y.loc[y["net_R"] > 0, "net_R"].sum())
    largest_positive_year = float(y.loc[y["net_R"] > 0, "net_R"].max()) if positive_year_profit > 0 else 0.0
    max_positive_year_share = largest_positive_year / positive_year_profit if positive_year_profit > 0 else 1.0
    positive_phases = int((p["net_R"] > 0).sum())
    backcast = channel_trades.loc[channel_trades["phase"].str.startswith("backcast_", na=False)] if not channel_trades.empty else pd.DataFrame()
    reference = channel_trades.loc[channel_trades["phase"].str.startswith("v10_", na=False)] if not channel_trades.empty else pd.DataFrame()
    unseen_holdout = channel_trades.loc[channel_trades["phase"] == "unseen_holdout"] if not channel_trades.empty else pd.DataFrame()
    backcast_m = metrics(backcast)
    reference_m = metrics(reference)
    unseen_holdout_m = metrics(unseen_holdout)
    positive_r12_ratio = float((active_r12["net_R"] > 0).mean()) if not active_r12.empty else 0.0
    worst_r12 = float(active_r12["net_R"].min()) if not active_r12.empty else 0.0
    stress_15 = s15.iloc[0].to_dict() if not s15.empty else {"net_R": 0.0, "profit_factor": 0.0}

    checks = {
        "minimum_trades": overall["trades"] >= int(threshold["minimum_trades"]),
        "minimum_win_rate": overall["win_rate"] >= float(threshold["minimum_win_rate"]),
        "minimum_avg_win_loss_ratio": overall["avg_win_loss_ratio"] >= float(threshold["minimum_avg_win_loss_ratio"]),
        "minimum_profit_factor": overall["profit_factor"] >= float(threshold["minimum_profit_factor"]),
        "minimum_expectancy_R": overall["expectancy_R"] >= float(threshold["minimum_expectancy_R"]),
        "maximum_drawdown_R": overall["max_drawdown_R"] <= float(threshold["maximum_drawdown_R"]),
        "minimum_positive_years": positive_years >= int(threshold["minimum_positive_years"]),
        "minimum_positive_phases": positive_phases >= int(threshold["minimum_positive_phases"]),
        "best_10pct_removed_still_profitable": tail["net_R"] > 0,
        "maximum_single_positive_year_share": max_positive_year_share <= float(threshold["maximum_single_positive_year_share"]),
        "minimum_positive_rolling_12m_ratio": positive_r12_ratio >= float(threshold["minimum_positive_rolling_12m_ratio"]),
        "minimum_worst_rolling_12m_net_R": worst_r12 >= float(threshold["minimum_worst_rolling_12m_net_R"]),
        "backcast_positive": backcast_m["net_R"] > 0,
        "minimum_backcast_profit_factor": backcast_m["profit_factor"] >= float(threshold["minimum_backcast_profit_factor"]),
        "reference_window_positive": reference_m["net_R"] > 0,
        "minimum_unseen_holdout_trades": unseen_holdout_m["trades"] >= int(threshold["minimum_unseen_holdout_trades"]),
        "unseen_holdout_nonnegative": unseen_holdout_m["net_R"] >= 0,
        "cost_1_5x_positive": float(stress_15["net_R"]) > 0,
        "minimum_cost_1_5x_profit_factor": float(stress_15["profit_factor"]) >= float(threshold["minimum_cost_1_5x_profit_factor"]),
        "v10_5_reference_replay_passed": bool(replay_passed),
    }
    return {
        "scope": "portfolio" if spec.is_portfolio else spec.direction_scope,
        "metrics": overall,
        "best_fraction_removed": tail,
        "active_years": active_years,
        "positive_years": positive_years,
        "positive_phases": positive_phases,
        "max_single_positive_year_share": max_positive_year_share,
        "positive_rolling_12m_ratio": positive_r12_ratio,
        "worst_rolling_12m_net_R": worst_r12,
        "backcast_metrics": backcast_m,
        "reference_metrics": reference_m,
        "unseen_holdout_metrics": unseen_holdout_m,
        "cost_1_5x_metrics": {
            "trades": int(stress_15.get("trades", 0)),
            "profit_factor": float(stress_15.get("profit_factor", 0.0)),
            "net_R": float(stress_15.get("net_R", 0.0)),
            "max_drawdown_R": float(stress_15.get("max_drawdown_R", 0.0)),
        },
        "checks": checks,
        "cross_year_validation_pass": bool(all(checks.values())),
        "qualified_for_live_trading": False,
        "historical_backcast_is_not_true_forward_test": True,
        "unseen_holdout_is_not_chronologically_prospective": True,
    }


def replay_v10_5_reference(raw_5m: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    reference = config["v10_5_reference_replay"]
    start = pd.Timestamp(reference["start_utc"])
    end = pd.Timestamp(reference["end_utc"])
    isolated_raw = raw_5m.loc[(raw_5m["time"] >= start) & (raw_5m["time"] <= end)].copy().reset_index(drop=True)
    signal_map, _ = build_all_signals(isolated_raw)
    rows: list[dict[str, Any]] = []
    replay_trades: list[pd.DataFrame] = []
    tolerance = float(reference.get("numeric_tolerance", 1e-8))
    all_passed = True
    for spec in CHANNELS:
        trades = execute_channel(isolated_raw, signal_map[spec.channel_id], spec, config)
        replay_trades.append(trades)
        actual = metrics(trades)
        expected = reference["expected_metrics"][spec.channel_id]
        checks = {
            "trades": int(actual["trades"]) == int(expected["trades"]),
            "win_rate": abs(float(actual["win_rate"]) - float(expected["win_rate"])) <= tolerance,
            "profit_factor": abs(float(actual["profit_factor"]) - float(expected["profit_factor"])) <= tolerance,
            "net_R": abs(float(actual["net_R"]) - float(expected["net_R"])) <= tolerance,
            "max_drawdown_R": abs(float(actual["max_drawdown_R"]) - float(expected["max_drawdown_R"])) <= tolerance,
        }
        passed = bool(all(checks.values()))
        all_passed = all_passed and passed
        rows.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "passed": passed,
                "expected_trades": expected["trades"],
                "actual_trades": actual["trades"],
                "expected_win_rate": expected["win_rate"],
                "actual_win_rate": actual["win_rate"],
                "expected_profit_factor": expected["profit_factor"],
                "actual_profit_factor": actual["profit_factor"],
                "expected_net_R": expected["net_R"],
                "actual_net_R": actual["net_R"],
                "expected_max_drawdown_R": expected["max_drawdown_R"],
                "actual_max_drawdown_R": actual["max_drawdown_R"],
                "checks": checks,
            }
        )
    ledger = pd.concat([frame for frame in replay_trades if not frame.empty], ignore_index=True) if any(not frame.empty for frame in replay_trades) else pd.DataFrame()
    return {
        "passed": all_passed,
        "reference_start_utc": start.isoformat(),
        "reference_end_utc": end.isoformat(),
        "numeric_tolerance": tolerance,
        "channels": rows,
    }, ledger


def warmup_sensitivity_audit(full_ledger: pd.DataFrame, isolated_ledger: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    start = pd.Timestamp(config["v10_5_reference_replay"]["start_utc"])
    end = pd.Timestamp(config["v10_5_reference_replay"]["end_utc"])
    rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        continuous = full_ledger.loc[full_ledger["channel_id"] == spec.channel_id].copy() if not full_ledger.empty else pd.DataFrame()
        if not continuous.empty:
            entries = pd.to_datetime(continuous["entry_time_utc"], utc=True)
            continuous = continuous.loc[(entries >= start) & (entries <= end)].copy()
        isolated = isolated_ledger.loc[isolated_ledger["channel_id"] == spec.channel_id].copy() if not isolated_ledger.empty else pd.DataFrame()
        continuous_times = set(pd.to_datetime(continuous.get("entry_time_utc", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
        isolated_times = set(pd.to_datetime(isolated.get("entry_time_utc", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
        cm = metrics(continuous)
        im = metrics(isolated)
        rows.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "continuous_history_trades": cm["trades"],
                "isolated_window_trades": im["trades"],
                "common_entry_times": len(continuous_times & isolated_times),
                "continuous_only_entry_times": len(continuous_times - isolated_times),
                "isolated_only_entry_times": len(isolated_times - continuous_times),
                "delta_net_R_continuous_minus_isolated": cm["net_R"] - im["net_R"],
                "delta_profit_factor_continuous_minus_isolated": cm["profit_factor"] - im["profit_factor"],
                "exact_entry_set_match": continuous_times == isolated_times,
            }
        )
    return pd.DataFrame(rows)


def benchmark_score(audit: dict[str, Any]) -> float:
    m = audit["metrics"]
    tail = audit["best_fraction_removed"]
    unseen_holdout = audit["unseen_holdout_metrics"]
    return float(
        18.0 * m["win_rate"]
        + 4.0 * min(m["avg_win_loss_ratio"], 4.0)
        + 5.0 * min(m["profit_factor"], 4.0)
        + 0.35 * m["net_R"]
        + 12.0 * m["expectancy_R"]
        + 0.25 * tail["net_R"]
        + 1.0 * audit["positive_years"]
        + 1.0 * audit["positive_phases"]
        + 0.15 * unseen_holdout["net_R"]
        - 0.5 * m["max_drawdown_R"]
        + (20.0 if audit["cross_year_validation_pass"] else 0.0)
    )


def run_benchmark(
    raw_5m: pd.DataFrame,
    config: dict[str, Any],
    *,
    replay_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signal_map, signal_audit = build_all_signals(raw_5m)
    trade_frames: list[pd.DataFrame] = []
    for spec in CHANNELS:
        trade_frames.append(execute_channel(raw_5m, signal_map[spec.channel_id], spec, config))
    ledger = pd.concat([frame for frame in trade_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in trade_frames) else pd.DataFrame()
    if not ledger.empty:
        ledger = ledger.sort_values(["channel_id", "entry_time_utc"]).reset_index(drop=True)

    years = [str(year) for year in range(int(config["analysis_start_month"][:4]), int(config["analysis_end_month"][:4]) + 1)]
    monthly = monthly_metrics(ledger)
    yearly = yearly_metrics(ledger, years)
    phases = phase_metrics(ledger, config)
    rolling_3m = rolling_window_metrics(ledger, config, 3)
    rolling_6m = rolling_window_metrics(ledger, config, 6)
    rolling_12m = rolling_window_metrics(ledger, config, 12)
    year_removal = year_removal_robustness(ledger, years)
    stress = cost_stress_summary(raw_5m, signal_map, config)
    replay_passed = bool(replay_audit["passed"]) if replay_audit is not None else True

    qualification: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        channel_trades = ledger.loc[ledger["channel_id"] == spec.channel_id] if not ledger.empty else pd.DataFrame()
        audit = qualification_audit(channel_trades, spec, config, yearly, phases, rolling_12m, stress, replay_passed)
        audit["benchmark_score"] = benchmark_score(audit)
        qualification[spec.channel_id] = audit
        summary_rows.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "direction_scope": spec.direction_scope,
                "profile": spec.profile,
                "is_portfolio": spec.is_portfolio,
                "parameters_frozen": True,
                **audit["metrics"],
                "active_years": audit["active_years"],
                "positive_years": audit["positive_years"],
                "positive_phases": audit["positive_phases"],
                "positive_rolling_12m_ratio": audit["positive_rolling_12m_ratio"],
                "worst_rolling_12m_net_R": audit["worst_rolling_12m_net_R"],
                "best_10pct_removed_net_R": audit["best_fraction_removed"]["net_R"],
                "backcast_trades": audit["backcast_metrics"]["trades"],
                "backcast_profit_factor": audit["backcast_metrics"]["profit_factor"],
                "backcast_net_R": audit["backcast_metrics"]["net_R"],
                "reference_trades": audit["reference_metrics"]["trades"],
                "reference_profit_factor": audit["reference_metrics"]["profit_factor"],
                "reference_net_R": audit["reference_metrics"]["net_R"],
                "unseen_holdout_trades": audit["unseen_holdout_metrics"]["trades"],
                "unseen_holdout_win_rate": audit["unseen_holdout_metrics"]["win_rate"],
                "unseen_holdout_profit_factor": audit["unseen_holdout_metrics"]["profit_factor"],
                "unseen_holdout_net_R": audit["unseen_holdout_metrics"]["net_R"],
                "cost_1_5x_profit_factor": audit["cost_1_5x_metrics"]["profit_factor"],
                "cost_1_5x_net_R": audit["cost_1_5x_metrics"]["net_R"],
                "cross_year_validation_pass": audit["cross_year_validation_pass"],
                "benchmark_score": audit["benchmark_score"],
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["cross_year_validation_pass", "benchmark_score"], ascending=[False, False]
    ).reset_index(drop=True)
    leader = summary.loc[summary["is_portfolio"]].iloc[0].to_dict()
    leader = {key: (value.item() if isinstance(value, np.generic) else value) for key, value in leader.items()}
    return {
        "signals": signal_map,
        "signal_audit": signal_audit,
        "trades": ledger,
        "summary": summary,
        "monthly": monthly,
        "yearly": yearly,
        "phases": phases,
        "directions": direction_metrics(ledger),
        "components": component_metrics(ledger),
        "rolling_3m": rolling_3m,
        "rolling_6m": rolling_6m,
        "rolling_12m": rolling_12m,
        "year_removal": year_removal,
        "cost_stress": stress,
        "qualification": qualification,
        "research_leader": leader,
    }


def self_test(config: dict[str, Any]) -> None:
    assert len(CHANNELS) == 3
    assert sum(spec.is_portfolio for spec in CHANNELS) == 1
    assert all(spec.is_baseline for spec in CHANNELS)
    lock = parameter_lock_payload(config)
    assert len(lock["parameter_lock_sha256"]) == 64

    raw = synthetic_5m_data(120_000, seed=20261006)
    signals, audit = build_all_signals(raw)
    assert set(signals) == {spec.channel_id for spec in CHANNELS}
    assert len(audit["channels"]) == 3
    assert len(audit["portfolio_overlap"]) == 1

    long_times = set(pd.to_datetime(signals["60m_long_frozen_adx45_rr2_5"].get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
    short_times = set(pd.to_datetime(signals["60m_short_frozen_adx_decay_minus4_rr2_5"].get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
    portfolio = signals["60m_portfolio_frozen_v10_6"]
    portfolio_times = set(pd.to_datetime(portfolio.get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
    assert portfolio_times.issubset(long_times | short_times)

    result = run_benchmark(raw, config, replay_audit={"passed": True})
    assert len(result["summary"]) == 3
    assert len(result["phases"]) == 3 * len(config["phases"])
    assert len(result["rolling_3m"]) == 3 * (len(month_range(config["analysis_start_month"], config["analysis_end_month"])) - 2)
    assert len(result["cost_stress"]) == 3 * len(config["cost_stress_scenarios"])
    assert not result["trades"].empty
    for channel_id, group in result["trades"].groupby("channel_id"):
        entries = pd.to_datetime(group["entry_time_utc"], utc=True).reset_index(drop=True)
        signals_at = pd.to_datetime(group["signal_time_utc"], utc=True).reset_index(drop=True)
        exits = pd.to_datetime(group["exit_time_utc"], utc=True).reset_index(drop=True)
        assert (entries >= signals_at).all(), channel_id
        assert (exits >= entries).all(), channel_id
        if len(group) > 1:
            assert (entries.iloc[1:].reset_index(drop=True) >= exits.iloc[:-1].reset_index(drop=True)).all(), channel_id
    print("V106_ENGINE_SELF_TEST_OK")


__all__ = [
    "CHANNELS",
    "CHANNEL_BY_ID",
    "ChannelSpec",
    "build_all_signals",
    "canonical_sha256",
    "load_official_5m_data",
    "parameter_lock_payload",
    "replay_v10_5_reference",
    "run_benchmark",
    "self_test",
    "synthetic_5m_data",
    "warmup_sensitivity_audit",
]
