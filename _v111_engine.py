from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

import _v110_engine as v110


ChannelSpec = v110.ChannelSpec
BASE = v110.BASE
V103 = v110.V103
v108 = v110.v108
v109 = v110.v109


ENTRY_MODES: tuple[tuple[str, str, str], ...] = (
    ("direct_1h", "1H确认直接成交", "state_direct_after_1h_confirmation_rr2_0"),
    ("local_break_15m", "15M局部突破", "state_15m_single_local_break_rr2_0"),
    ("pullback_reclaim_15m", "15M回踩重确认", "state_15m_pullback_reclaim_rr2_0"),
)
DIRECTION_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("long", "多头", "LONG"),
    ("short", "空头", "SHORT"),
    ("shared", "多空共享", "BOTH"),
)
REWARD_RISKS: tuple[float, ...] = (1.5, 2.0, 2.5)


@dataclass(frozen=True)
class ChannelMeta:
    entry_mode: str
    direction_variant: str
    management_mode: str
    reward_risk: float
    base_signal_channel_id: str


def _rr_key(rr: float) -> str:
    return str(rr).replace(".", "_")


def _make_channels() -> tuple[tuple[ChannelSpec, ...], dict[str, ChannelMeta]]:
    channels: list[ChannelSpec] = [
        ChannelSpec(
            "baseline_1h_shared_v10_6",
            "基准·V10.6原1小时多空共享组合 RR2.5",
            "baseline_1h_shared_v10_6",
            "BOTH",
            0.0,
            0.0,
            0,
            True,
        )
    ]
    meta: dict[str, ChannelMeta] = {
        "baseline_1h_shared_v10_6": ChannelMeta(
            "baseline_v10_6", "shared", "V10_6_FROZEN", 2.5, "baseline_1h_shared_v10_6"
        )
    }

    # 27 fixed-management channels: 3 entries x (long/short/shared) x 3 reward-risk targets.
    for entry_mode, entry_label, base_signal_id in ENTRY_MODES:
        for direction_variant, direction_label, direction_scope in DIRECTION_VARIANTS:
            for rr in REWARD_RISKS:
                channel_id = f"matrix_{entry_mode}_{direction_variant}_fixed_rr{_rr_key(rr)}"
                label = f"批量·{entry_label}·{direction_label}·固定 RR{rr:.1f}"
                profile = f"matrix_{entry_mode}_{direction_variant}_fixed"
                channels.append(ChannelSpec(channel_id, label, profile, direction_scope, 1.25, rr, 48, False))
                meta[channel_id] = ChannelMeta(entry_mode, direction_variant, "FIXED_STOP_TARGET", rr, base_signal_id)

    # 6 break-even shadows: 3 entries x long/short at RR2.0. Shared BE is intentionally excluded.
    for entry_mode, entry_label, base_signal_id in ENTRY_MODES:
        for direction_variant, direction_label, direction_scope in DIRECTION_VARIANTS[:2]:
            channel_id = f"matrix_{entry_mode}_{direction_variant}_be1_rr2_0"
            label = f"影子·{entry_label}·{direction_label}·1R后保本 RR2.0"
            profile = f"matrix_{entry_mode}_{direction_variant}_be1"
            channels.append(ChannelSpec(channel_id, label, profile, direction_scope, 1.25, 2.0, 48, False))
            meta[channel_id] = ChannelMeta(entry_mode, direction_variant, "BE_AT_1R_NEXT_5M_BAR", 2.0, base_signal_id)

    if len(channels) != 34:
        raise AssertionError(f"V10.11 channel construction error: {len(channels)}")
    return tuple(channels), meta


CHANNELS, CHANNEL_META = _make_channels()
CHANNEL_BY_ID = {spec.channel_id: spec for spec in CHANNELS}


def synthetic_5m_data(rows: int = 220_000, seed: int = 10211) -> pd.DataFrame:
    return v110.synthetic_5m_data(rows, seed)


def load_official_5m_data(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    return v110.load_official_5m_data(config)


def _filter_direction(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if frame.empty or scope == "BOTH":
        return frame.copy()
    direction = 1 if scope == "LONG" else -1
    return frame.loc[frame["direction"] == direction].copy()


def _clone_signals(base: pd.DataFrame, spec: ChannelSpec, meta: ChannelMeta) -> pd.DataFrame:
    cloned = _filter_direction(base, spec.direction_scope)
    if cloned.empty:
        return cloned
    cloned = cloned.sort_values(["signal_time", "direction", "setup_id"]).reset_index(drop=True)
    cloned["channel_id"] = spec.channel_id
    cloned["channel_label"] = spec.label
    cloned["profile"] = spec.profile
    cloned["direction_scope"] = spec.direction_scope
    cloned["stop_atr_multiple"] = float(spec.stop_atr_multiple)
    cloned["reward_risk"] = float(spec.reward_risk)
    cloned["max_holding_hours"] = int(spec.max_holding_hours)
    cloned["management_mode"] = meta.management_mode
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
    return cloned


def build_all_signals(
    raw_5m: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_map, base_audit, funnel, state_events, setup_matrix, coverage = v110.build_all_signals(raw_5m, config)
    signal_map: dict[str, pd.DataFrame] = {
        "baseline_1h_shared_v10_6": base_map["baseline_1h_shared_v10_6"].copy()
    }
    if not signal_map["baseline_1h_shared_v10_6"].empty:
        signal_map["baseline_1h_shared_v10_6"]["management_mode"] = "V10_6_FROZEN"

    audit_rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        meta = CHANNEL_META[spec.channel_id]
        if spec.is_baseline:
            signals = signal_map[spec.channel_id]
        else:
            signals = _clone_signals(base_map[meta.base_signal_channel_id], spec, meta)
            signal_map[spec.channel_id] = signals
        audit_rows.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "profile": spec.profile,
                "entry_mode": meta.entry_mode,
                "direction_variant": meta.direction_variant,
                "management_mode": meta.management_mode,
                "reward_risk": meta.reward_risk,
                "signals": int(len(signals)),
                "long_signals": int((signals["direction"] == 1).sum()) if not signals.empty else 0,
                "short_signals": int((signals["direction"] == -1).sum()) if not signals.empty else 0,
                "unique_signal_times": int(signals["signal_time"].nunique()) if not signals.empty else 0,
                "unique_setup_ids": int(signals["setup_id"].nunique()) if not signals.empty and "setup_id" in signals else 0,
            }
        )

    audit = {
        **base_audit,
        "channels": audit_rows,
        "matrix": {
            "channel_count": len(CHANNELS),
            "fixed_directional_channels": 18,
            "fixed_shared_channels": 9,
            "break_even_shadow_channels": 6,
            "baseline_channels": 1,
            "entry_modes": [row[0] for row in ENTRY_MODES],
            "reward_risks": list(REWARD_RISKS),
            "same_confirmation_pool": True,
            "upper_state_machine_identical_to_v10_9": True,
        },
    }
    return signal_map, audit, funnel, state_events, setup_matrix, coverage


def metrics(trades: pd.DataFrame | Iterable[dict[str, Any]]) -> dict[str, float]:
    return v110.metrics(trades)


def remove_best_fraction(trades: pd.DataFrame, fraction: float = 0.10) -> tuple[pd.DataFrame, dict[str, Any]]:
    return v110.remove_best_fraction(trades, fraction)


def max_consecutive_losses(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    ordered = trades.sort_values("entry_time_utc")
    max_run = 0
    run = 0
    for is_win in ordered["win"].astype(bool):
        if is_win:
            run = 0
        else:
            run += 1
            max_run = max(max_run, run)
    return int(max_run)


def summarize_channel(trades: pd.DataFrame, spec: ChannelSpec, config: dict[str, Any]) -> dict[str, Any]:
    base = v110.summarize_channel(trades, spec, config)
    meta = CHANNEL_META[spec.channel_id]
    return {
        **base,
        "entry_mode": meta.entry_mode,
        "direction_variant": meta.direction_variant,
        "management_mode": meta.management_mode,
        "reward_risk": meta.reward_risk,
        "max_consecutive_losses": max_consecutive_losses(trades),
    }


def _run_one_channel(
    raw_5m: pd.DataFrame,
    signals: pd.DataFrame,
    spec: ChannelSpec,
    config: dict[str, Any],
    h4: pd.DataFrame,
    break_even_shadow: bool,
) -> pd.DataFrame:
    if spec.is_baseline:
        trades = V103.execute_channel(raw_5m, signals, spec, config)
        trades = v108.enrich_baseline_path_diagnostics(raw_5m, trades)
        if not trades.empty:
            trades["management_mode"] = "V10_6_FROZEN"
            trades["setup_id"] = trades["signal_time_utc"].astype(str)
        return trades
    return v109.execute_strict_channel(raw_5m, signals, spec, config, h4, break_even_shadow=break_even_shadow)


def _cost_stress_config(config: dict[str, Any], multiplier: float) -> dict[str, Any]:
    stressed = copy.deepcopy(config)
    stressed["execution"]["fee_rate_per_side"] = float(config["execution"]["fee_rate_per_side"]) * multiplier
    base_ticks = int(config["execution"]["slippage_ticks_per_fill"])
    stressed["execution"]["slippage_ticks_per_fill"] = max(1, int(round(base_ticks * multiplier)))
    return stressed


def build_setup_channel_matrix(
    signal_map: dict[str, pd.DataFrame],
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    trade_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if not ledger.empty:
        for row in ledger.itertuples(index=False):
            setup_id = str(getattr(row, "setup_id", ""))
            if setup_id:
                trade_lookup[(str(row.channel_id), setup_id)] = {
                    "trade_executed": True,
                    "net_R": float(row.net_r),
                    "win": bool(row.win),
                    "exit_reason": str(row.exit_reason),
                    "entry_time_utc": str(row.entry_time_utc),
                    "exit_time_utc": str(row.exit_time_utc),
                    "mfe_R": float(getattr(row, "mfe_R", np.nan)),
                    "mae_R": float(getattr(row, "mae_R", np.nan)),
                }
    rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        if spec.is_baseline:
            continue
        meta = CHANNEL_META[spec.channel_id]
        signals = signal_map[spec.channel_id]
        for signal in signals.itertuples(index=False):
            setup_id = str(signal.setup_id)
            trade = trade_lookup.get((spec.channel_id, setup_id), {})
            rows.append(
                {
                    "setup_id": setup_id,
                    "setup_time_utc": pd.Timestamp(signal.setup_time).isoformat(),
                    "signal_time_utc": pd.Timestamp(signal.signal_time).isoformat(),
                    "direction": int(signal.direction),
                    "environment_leg_id": int(signal.environment_leg_id),
                    "cycle_number": int(signal.cycle_number),
                    "channel_id": spec.channel_id,
                    "channel_label": spec.label,
                    "entry_mode": meta.entry_mode,
                    "direction_variant": meta.direction_variant,
                    "management_mode": meta.management_mode,
                    "reward_risk": meta.reward_risk,
                    "signal_emitted": True,
                    "trade_executed": bool(trade.get("trade_executed", False)),
                    "net_R": trade.get("net_R", np.nan),
                    "win": trade.get("win", np.nan),
                    "exit_reason": trade.get("exit_reason", ""),
                    "entry_time_utc": trade.get("entry_time_utc", ""),
                    "exit_time_utc": trade.get("exit_time_utc", ""),
                    "mfe_R": trade.get("mfe_R", np.nan),
                    "mae_R": trade.get("mae_R", np.nan),
                }
            )
    return pd.DataFrame(rows)


def management_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    fixed = summary.loc[
        (summary["management_mode"] == "FIXED_STOP_TARGET")
        & (summary["reward_risk"] == 2.0)
        & (summary["direction_variant"].isin(["long", "short"]))
    ].copy()
    shadow = summary.loc[summary["management_mode"] == "BE_AT_1R_NEXT_5M_BAR"].copy()
    keys = ["entry_mode", "direction_variant"]
    merged = fixed.merge(shadow, on=keys, suffixes=("_fixed", "_be1"), how="outer")
    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        output: dict[str, Any] = {
            "entry_mode": row.entry_mode,
            "direction_variant": row.direction_variant,
            "fixed_channel_id": getattr(row, "channel_id_fixed", ""),
            "be1_channel_id": getattr(row, "channel_id_be1", ""),
        }
        for metric_name in ["trades", "win_rate", "avg_win_loss_ratio", "profit_factor", "net_R", "max_drawdown_R", "max_consecutive_losses"]:
            fixed_value = float(getattr(row, f"{metric_name}_fixed", 0.0))
            be_value = float(getattr(row, f"{metric_name}_be1", 0.0))
            output[f"{metric_name}_fixed"] = fixed_value
            output[f"{metric_name}_be1"] = be_value
            output[f"{metric_name}_delta_be1_minus_fixed"] = be_value - fixed_value
        rows.append(output)
    return pd.DataFrame(rows)


def run_benchmark(raw_5m: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    signal_map, signal_audit, funnel, state_events, setup_matrix, coverage = build_all_signals(raw_5m, config)
    h4 = v108.four_hour_environment(v108.add_features(v108.resample_ohlcv(raw_5m, 240)), config)

    trade_frames: list[pd.DataFrame] = []
    qualification: dict[str, Any] = {}
    cost_rows: list[dict[str, Any]] = []
    stress_multiplier = float(config["cost_stress"]["multiplier"])
    stress_config = _cost_stress_config(config, stress_multiplier)

    for spec in CHANNELS:
        signals = signal_map[spec.channel_id]
        meta = CHANNEL_META[spec.channel_id]
        is_be = meta.management_mode == "BE_AT_1R_NEXT_5M_BAR"
        trades = _run_one_channel(raw_5m, signals, spec, config, h4, is_be)
        trade_frames.append(trades)
        qualification[spec.channel_id] = summarize_channel(trades, spec, config)

        stressed = _run_one_channel(raw_5m, signals, spec, stress_config, h4, is_be)
        stress_metrics = metrics(stressed)
        cost_rows.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "entry_mode": meta.entry_mode,
                "direction_variant": meta.direction_variant,
                "management_mode": meta.management_mode,
                "reward_risk": meta.reward_risk,
                "cost_multiplier": stress_multiplier,
                "trades": stress_metrics["trades"],
                "win_rate": stress_metrics["win_rate"],
                "avg_win_loss_ratio": stress_metrics["avg_win_loss_ratio"],
                "profit_factor": stress_metrics["profit_factor"],
                "net_R": stress_metrics["net_R"],
                "max_drawdown_R": stress_metrics["max_drawdown_R"],
            }
        )

    nonempty = [frame for frame in trade_frames if not frame.empty]
    ledger = pd.concat(nonempty, ignore_index=True, sort=False) if nonempty else pd.DataFrame()
    if not ledger.empty:
        ledger = ledger.sort_values(["channel_id", "entry_time_utc"]).reset_index(drop=True)

    summary = pd.DataFrame(
        [{key: value for key, value in audit.items() if key != "checks"} for audit in qualification.values()]
    )
    cost_stress = pd.DataFrame(cost_rows)
    summary = summary.merge(
        cost_stress[["channel_id", "profit_factor", "net_R", "max_drawdown_R"]].rename(
            columns={
                "profit_factor": "cost_1_5x_profit_factor",
                "net_R": "cost_1_5x_net_R",
                "max_drawdown_R": "cost_1_5x_max_drawdown_R",
            }
        ),
        on="channel_id",
        how="left",
    )
    summary = summary.sort_values(
        ["historical_research_gate_pass", "profit_factor", "net_R"], ascending=[False, False, False]
    ).reset_index(drop=True)

    setup_channel_matrix = build_setup_channel_matrix(signal_map, ledger)
    management = management_comparison(summary)
    matrix_only = summary.loc[~summary["is_baseline"]].copy()
    diagnostic_rank = matrix_only.sort_values(
        ["trades", "profit_factor", "net_R"], ascending=[False, False, False]
    ).reset_index(drop=True)
    diagnostic_rank.insert(0, "diagnostic_rank", np.arange(1, len(diagnostic_rank) + 1))

    return {
        "signals": signal_map,
        "signal_audit": signal_audit,
        "signal_funnel": funnel,
        "state_machine_events": state_events,
        "setup_signal_matrix": setup_matrix,
        "trigger_coverage": coverage,
        "setup_channel_matrix": setup_channel_matrix,
        "trades": ledger,
        "summary": summary,
        "diagnostic_rank": diagnostic_rank,
        "qualification": qualification,
        "direction_summary": v110.grouped_metrics(ledger, ["channel_id", "channel_label", "direction"]),
        "monthly_summary": v110.grouped_metrics(ledger, ["channel_id", "channel_label", "month"]),
        "trigger_type_summary": v110.grouped_metrics(ledger, ["channel_id", "channel_label", "setup_type", "trigger_type"]),
        "exit_reason_summary": v110.grouped_metrics(ledger, ["channel_id", "channel_label", "exit_reason"]),
        "loss_diagnostics": v110.loss_diagnostics(ledger),
        "management_comparison": management,
        "cost_stress_summary": cost_stress,
    }


def parameter_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "release": config["release"],
        "channels": [
            {**asdict(spec), **asdict(CHANNEL_META[spec.channel_id])}
            for spec in CHANNELS
        ],
        "matrix_definition": config["matrix_definition"],
        "multi_timeframe_parameters": config["multi_timeframe_parameters"],
        "state_machine_parameters": config["state_machine_parameters"],
        "trigger_comparison_parameters": config["trigger_comparison_parameters"],
        "execution": config["execution"],
        "cost_stress": config["cost_stress"],
        "evaluation_window": config["evaluation_window"],
        "no_lookahead_rules": config["no_lookahead_rules"],
    }


def self_test(config: dict[str, Any]) -> None:
    assert len(CHANNELS) == 34
    assert sum(spec.is_baseline for spec in CHANNELS) == 1
    fixed_directional = [
        spec for spec in CHANNELS
        if not spec.is_baseline
        and CHANNEL_META[spec.channel_id].management_mode == "FIXED_STOP_TARGET"
        and CHANNEL_META[spec.channel_id].direction_variant in {"long", "short"}
    ]
    fixed_shared = [
        spec for spec in CHANNELS
        if CHANNEL_META[spec.channel_id].management_mode == "FIXED_STOP_TARGET"
        and CHANNEL_META[spec.channel_id].direction_variant == "shared"
    ]
    be = [spec for spec in CHANNELS if CHANNEL_META[spec.channel_id].management_mode == "BE_AT_1R_NEXT_5M_BAR"]
    assert len(fixed_directional) == 18
    assert len(fixed_shared) == 9
    assert len(be) == 6

    raw = synthetic_5m_data(220_000, seed=20261011)
    signal_map, audit, funnel, _, setup_matrix, _ = build_all_signals(raw, config)
    assert set(signal_map) == set(CHANNEL_BY_ID)
    assert audit["matrix"]["channel_count"] == 34
    assert audit["matrix"]["same_confirmation_pool"] is True
    assert audit["state_machine"]["upper_state_machine_identical_to_v10_9"] is True
    if not setup_matrix.empty:
        assert setup_matrix["setup_id"].is_unique

    for entry_mode, _, base_signal_id in ENTRY_MODES:
        base = v110.CHANNEL_BY_ID[base_signal_id]
        del base  # structural existence check
        long_ids = [
            spec.channel_id for spec in CHANNELS
            if CHANNEL_META[spec.channel_id].entry_mode == entry_mode
            and CHANNEL_META[spec.channel_id].direction_variant == "long"
        ]
        for channel_id in long_ids:
            frame = signal_map[channel_id]
            if not frame.empty:
                assert (frame["direction"] == 1).all()
        short_ids = [
            spec.channel_id for spec in CHANNELS
            if CHANNEL_META[spec.channel_id].entry_mode == entry_mode
            and CHANNEL_META[spec.channel_id].direction_variant == "short"
        ]
        for channel_id in short_ids:
            frame = signal_map[channel_id]
            if not frame.empty:
                assert (frame["direction"] == -1).all()

    result = run_benchmark(raw, config)
    assert len(result["summary"]) == 34
    assert len(result["cost_stress_summary"]) == 34
    assert len(result["management_comparison"]) == 6
    if not result["trades"].empty:
        for channel_id, group in result["trades"].groupby("channel_id"):
            entries = pd.to_datetime(group["entry_time_utc"], utc=True).reset_index(drop=True)
            exits = pd.to_datetime(group["exit_time_utc"], utc=True).reset_index(drop=True)
            assert (exits >= entries).all(), channel_id
            if len(group) > 1:
                assert (entries.iloc[1:].reset_index(drop=True) >= exits.iloc[:-1].reset_index(drop=True)).all(), channel_id
    print("V111_ENGINE_SELF_TEST_OK")


__all__ = [
    "CHANNELS", "CHANNEL_BY_ID", "CHANNEL_META", "ENTRY_MODES", "DIRECTION_VARIANTS", "REWARD_RISKS",
    "load_official_5m_data", "synthetic_5m_data", "run_benchmark", "parameter_manifest", "self_test",
    "metrics", "remove_best_fraction", "max_consecutive_losses",
]
