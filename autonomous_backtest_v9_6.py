from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v9_6"
RESULTS.mkdir(exist_ok=True)
ENGINE_VERSION = "V9.6"
ENGINE_NAME = "BTC 5m high-precision sparse expert pool V9.6"
OOS_MONTH = "2026-06"

REQUEST_PATH = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(ROOT / "request.v9_6.json")))
REQUEST = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
SYMBOL = str(REQUEST["symbol"]).upper()
INTERVAL = str(REQUEST["interval"]).lower()
EVAL_MONTHS = tuple(str(x) for x in REQUEST["months"])
if EVAL_MONTHS != ("2026-05", "2026-06"):
    raise ValueError("V9.6 requires evaluation months 2026-05 and untouched OOS 2026-06")
MONTHS = ("2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06")
DEVELOPMENT_MONTHS = MONTHS[3:9]
FEE_RATE = float(REQUEST["fee_rate_per_side"])
SLIPPAGE_ABS = float(REQUEST["tick_size"]) * int(REQUEST["slippage_ticks_per_fill"])
if FEE_RATE != 0.0005 or abs(SLIPPAGE_ABS - 0.2) > 1e-12:
    raise ValueError("V9.6 fixes one-side fee at 0.050% and slippage at 0.2 USDT per fill")

FINAL = REQUEST["final_target"]
STAGE = REQUEST["research_stage"]
GATE = REQUEST["sparse_expert_gate"]
PORT = REQUEST["portfolio"]
MODEL = REQUEST["model"]
SEARCH = REQUEST["search"]
BASE_SEED = int(MODEL["base_seed"])

# Load the frozen V9.5.1 data/feature/outcome engine from this package. A private
# compatibility request prevents the base module from consuming V9.6 settings.
compat = ROOT / ".v96_base_request.json"
compat.write_text(json.dumps({
    "symbol": SYMBOL, "interval": INTERVAL, "months": ["2026-05", "2026-06"],
    "fee_rate_per_side": FEE_RATE, "tick_size": REQUEST["tick_size"],
    "slippage_ticks_per_fill": REQUEST["slippage_ticks_per_fill"],
    "min_trades_per_month": 15, "max_trades_per_month": 30,
    "min_win_rate": 0.70, "min_avg_win_loss_ratio": 1.50,
    "core_expert_ids": [0,1,2,3,4,5]
}, ensure_ascii=False), encoding="utf-8")
old_request = os.environ.get("BACKTEST_REQUEST_FILE")
os.environ["BACKTEST_REQUEST_FILE"] = str(compat)
spec = importlib.util.spec_from_file_location("v96_base_engine", ROOT / "_v96_base_engine.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load _v96_base_engine.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
if old_request is None:
    os.environ.pop("BACKTEST_REQUEST_FILE", None)
else:
    os.environ["BACKTEST_REQUEST_FILE"] = old_request


@dataclass(frozen=True)
class SparseExpert:
    id: int
    key: str
    name: str
    family: str
    direction: int
    feature_group: str
    setup_group: str


@dataclass(frozen=True)
class RiskConfig:
    rr: float
    sl_atr: float
    min_stop_pct: float
    max_hold: int
    breakeven_trigger: float
    breakeven_lock: float
    early_bars: int
    early_cut_r: float


@dataclass(frozen=True)
class ModelConfig:
    max_depth: int
    learning_rate: float
    max_iter: int
    l2_regularization: float
    min_samples_leaf: int


@dataclass(frozen=True)
class Policy:
    expert_id: int
    risk: RiskConfig
    model: ModelConfig
    monthly_target: int
    min_probability: float
    min_percentile: float

    @property
    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


@dataclass
class FittedModel:
    base_model: HistGradientBoostingClassifier
    calibrator: LogisticRegression | None
    calibration_month: str
    calibration_scores: np.ndarray
    calibration_labels: np.ndarray
    base_rate: float
    calibration_brier: float
    meta_model: LogisticRegression | None
    meta_brier: float
    core_rows: int
    calibration_rows: int


@dataclass
class Candidate:
    policy: Policy
    monthly_events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    monthly_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    monthly_audit: dict[str, dict[str, Any]] = field(default_factory=dict)
    aggregate: dict[str, float] = field(default_factory=dict)
    eligible: bool = False
    reasons: list[str] = field(default_factory=list)
    score: float = -1e12


EXPERTS: tuple[SparseExpert, ...] = (
    SparseExpert(0, "trend_pullback_long", "趋势回踩重新站稳多头", "趋势回踩", 1, "trend", "趋势回踩"),
    SparseExpert(1, "trend_pullback_short", "趋势反抽失败空头", "趋势回踩", -1, "trend", "趋势反抽"),
    SparseExpert(2, "trend_accel_long", "趋势整理再加速多头", "趋势加速", 1, "trend", "趋势延续"),
    SparseExpert(3, "trend_accel_short", "趋势整理再加速空头", "趋势加速", -1, "trend", "趋势延续"),
    SparseExpert(4, "break_retest_long", "结构突破首次回踩多头", "结构突破回踩", 1, "trend", "突破回踩"),
    SparseExpert(5, "break_retest_short", "结构跌破首次反抽空头", "结构突破回踩", -1, "trend", "跌破反抽"),
    SparseExpert(6, "range_sweep_long", "区间扫低收回多头", "流动性反转", 1, "range", "扫低收回"),
    SparseExpert(7, "range_sweep_short", "区间扫高跌回空头", "流动性反转", -1, "range", "扫高跌回"),
    SparseExpert(8, "range_second_long", "区间二次测试多头", "二次测试", 1, "range", "二次测试"),
    SparseExpert(9, "range_second_short", "区间二次测试空头", "二次测试", -1, "range", "二次测试"),
    SparseExpert(10, "squeeze_break_long", "压缩后放量突破多头", "波动扩张", 1, "breakout", "压缩突破"),
    SparseExpert(11, "squeeze_break_short", "压缩后放量突破空头", "波动扩张", -1, "breakout", "压缩突破"),
    SparseExpert(12, "eth_sync_long", "BTC与ETH同步确认多头", "跨资产同步", 1, "cross", "跨资产同步"),
    SparseExpert(13, "eth_sync_short", "BTC与ETH同步确认空头", "跨资产同步", -1, "cross", "跨资产同步"),
    SparseExpert(14, "derivative_exhaust_long", "衍生品极端回归多头", "衍生品反转", 1, "derivative", "极端回归"),
    SparseExpert(15, "derivative_exhaust_short", "衍生品极端回归空头", "衍生品反转", -1, "derivative", "极端回归"),
)
EXPERT_BY_ID = {e.id: e for e in EXPERTS}
OUTCOME_CACHE: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
MODEL_CACHE: dict[tuple[int, int, str, tuple[str, ...]], FittedModel | None] = {}
EVAL_CACHE: dict[tuple[int, int, str, str, tuple[str, ...]], tuple[list[dict[str, Any]], dict[str, Any]]] = {}

COMMON_FEATURES = (
    "ret1","ret3","ret6","ret12","ret24","ema8_gap","ema21_gap","ema55_gap","ema200_gap",
    "atr_pct","atr_rank","rsi","adx","di_gap","macd_hist_atr","macd_slope_atr","bb_pos","bb_rank","bb_width","vwap_dev",
    "rel_vol","vol_z","trade_z","taker_ratio","taker_z","body","close_loc","upper_wick","lower_wick","range_exp","eff12","eff24","chop",
    "m15_trend","m15_gap","m15_adx","m15_rsi","m15_eff","m15_atr_pct","h1_trend","h1_gap","h1_adx","h1_rsi","h1_eff","h1_atr_pct",
    "eth_ret3","eth_ret12","eth_gap","eth_h1_trend","btc_eth_corr","btc_eth_spread_12","premium_z","premium_delta","funding_z","funding_change","derivative_pressure",
    "hour_sin","hour_cos","weekday"
)
FEATURE_GROUPS = {
    "trend": COMMON_FEATURES + ("regime_trend","setup_reclaim_long","setup_reclaim_short","setup_continuation_long","setup_continuation_short"),
    "range": COMMON_FEATURES + ("regime_range","setup_range_reversal_long","setup_range_reversal_short","setup_range_second_test_long"),
    "breakout": COMMON_FEATURES + ("regime_high_vol","setup_high_break_long","setup_high_break_short"),
    "cross": COMMON_FEATURES + ("regime_trend","regime_high_vol","setup_continuation_long","setup_continuation_short","setup_high_break_long","setup_high_break_short"),
    "derivative": COMMON_FEATURES + ("regime_range","setup_reclaim_long","setup_reclaim_short","setup_range_reversal_long","setup_range_reversal_short"),
}


def assert_oos_isolation(months: Iterable[str], context: str) -> None:
    if OOS_MONTH in set(months):
        raise RuntimeError(f"OOS leakage blocked in {context}")


def clear_results() -> None:
    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    RESULTS.mkdir()


def add_sparse_masks(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    prev_low = y["low"].rolling(12).min().shift(1)
    prev_high = y["high"].rolling(12).max().shift(1)
    prev_low_2 = y["low"].rolling(24).min().shift(2)
    prev_high_2 = y["high"].rolling(24).max().shift(2)
    hour = y.index.hour
    liquid_session = ((hour >= 7) & (hour <= 21))
    masks: dict[int, pd.Series] = {
        0: (y["regime_trend"] > 0) & (y["h1_trend"] > 0) & (y["m15_trend"] >= 0) & (y["setup_reclaim_long"] > 0) & (y["rsi"].between(42, 62)),
        1: (y["regime_trend"] > 0) & (y["h1_trend"] < 0) & (y["m15_trend"] <= 0) & (y["setup_reclaim_short"] > 0) & (y["rsi"].between(38, 58)),
        2: (y["regime_trend"] > 0) & (y["h1_trend"] > 0) & (y["setup_continuation_long"] > 0) & (y["adx"] >= 18) & (y["rel_vol"] >= 0.85),
        3: (y["regime_trend"] > 0) & (y["h1_trend"] < 0) & (y["setup_continuation_short"] > 0) & (y["adx"] >= 18) & (y["rel_vol"] >= 0.85),
        4: (y["h1_trend"] >= 0) & (y["close"] > prev_high) & (y["low"] <= prev_high * 1.0015) & (y["close_loc"] > 0.58) & liquid_session,
        5: (y["h1_trend"] <= 0) & (((y["setup_break_retest_short"] > 0)) | ((y["close"] < prev_low) & (y["high"] >= prev_low * 0.9985))) & (y["close_loc"] < 0.42) & liquid_session,
        6: (y["regime_range"] > 0) & (y["setup_range_reversal_long"] > 0) & (y["lower_wick"] >= 0.24) & (y["close_loc"] >= 0.52),
        7: (y["regime_range"] > 0) & (y["setup_range_reversal_short"] > 0) & (y["upper_wick"] >= 0.24) & (y["close_loc"] <= 0.48),
        8: (y["regime_range"] > 0) & (y["setup_range_second_test_long"] > 0) & (y["low"] >= prev_low_2 * 0.997) & (y["vwap_dev"] > -0.006),
        9: (y["regime_range"] > 0) & (y["high"] <= prev_high_2 * 1.003) & (y["upper_wick"] >= 0.30) & (y["rsi"] >= 55) & (y["ret3"] < 0) & (y["vwap_dev"] < 0.006),
        10: (y["setup_high_break_long"] > 0) & (y["bb_rank"].shift(6) <= 0.42) & (y["rel_vol"] >= 1.15) & (y["eth_ret3"] >= -0.002),
        11: (y["setup_high_break_short"] > 0) & (y["bb_rank"].shift(6) <= 0.42) & (y["rel_vol"] >= 1.15) & (y["eth_ret3"] <= 0.002),
        12: (y["h1_trend"] > 0) & (y["eth_h1_trend"] > 0) & (y["btc_eth_corr"] >= 0.35) & ((y["setup_continuation_long"] > 0) | (y["setup_high_break_long"] > 0)) & (y["eth_ret3"] > 0),
        13: (y["h1_trend"] < 0) & (y["eth_h1_trend"] < 0) & (y["btc_eth_corr"] >= 0.35) & ((y["setup_continuation_short"] > 0) | (y["setup_high_break_short"] > 0)) & (y["eth_ret3"] < 0),
        14: (y["regime_range"] > 0) & (y["funding_z"] <= -1.25) & (y["premium_z"] <= -1.0) & ((y["setup_reclaim_long"] > 0) | (y["setup_range_reversal_long"] > 0)) & (y["close_loc"] > 0.52),
        15: (y["regime_range"] > 0) & (y["funding_z"] >= 1.25) & (y["premium_z"] >= 1.0) & ((y["setup_reclaim_short"] > 0) | (y["setup_range_reversal_short"] > 0)) & (y["close_loc"] < 0.48),
    }
    for expert in EXPERTS:
        mask = masks[expert.id].fillna(False)
        y[f"sparse_{expert.key}"] = mask.astype(float)
        structure = mask.rolling(3).max().fillna(0).astype(bool)
        rising = structure & (~structure.shift(1, fill_value=False))
        y[f"sparse_{expert.key}_cycle"] = rising.cumsum().astype(float)
    return y.replace([np.inf, -np.inf], np.nan).dropna()


def expert_indices(x: pd.DataFrame, expert: SparseExpert) -> np.ndarray:
    return np.flatnonzero(x[f"sparse_{expert.key}"].to_numpy(bool)).astype(np.int64)


def risk_grid(expert: SparseExpert) -> tuple[RiskConfig, ...]:
    if expert.family in {"流动性反转", "二次测试", "衍生品反转"}:
        return (
            RiskConfig(1.8, 1.05, 0.0030, 144, 0.85, 0.08, 42, 0.32),
            RiskConfig(2.0, 1.20, 0.0033, 180, 0.95, 0.10, 48, 0.34),
        )
    if expert.family in {"波动扩张", "跨资产同步"}:
        return (
            RiskConfig(2.0, 1.25, 0.0034, 180, 0.95, 0.08, 48, 0.38),
            RiskConfig(2.2, 1.40, 0.0038, 216, 1.05, 0.12, 60, 0.42),
        )
    return (
        RiskConfig(1.8, 1.15, 0.0031, 180, 0.90, 0.08, 48, 0.36),
        RiskConfig(2.0, 1.30, 0.0035, 216, 1.00, 0.12, 60, 0.40),
    )


def model_config(expert: SparseExpert) -> ModelConfig:
    if expert.feature_group in {"range", "derivative"}:
        return ModelConfig(2, 0.035, 220, 7.0, 28)
    if expert.feature_group == "breakout":
        return ModelConfig(2, 0.030, 240, 8.0, 34)
    return ModelConfig(3, 0.025, 260, 8.0, 40)


def policy_grid(expert: SparseExpert) -> list[Policy]:
    return [
        Policy(expert.id, risk, model_config(expert), int(target), float(prob), float(pct))
        for risk, prob, pct, target in itertools.product(
            risk_grid(expert), SEARCH["probability_thresholds"], SEARCH["online_percentiles"], SEARCH["monthly_targets"]
        )
    ]


def outcome_arrays(x: pd.DataFrame, idx: np.ndarray, expert: SparseExpert, risk: RiskConfig):
    cache_key = (id(x), expert.id, hashlib.sha256(json.dumps(asdict(risk), sort_keys=True).encode()).hexdigest()[:12])
    if cache_key in OUTCOME_CACHE:
        return OUTCOME_CACHE[cache_key]
    arrays = [x[k].to_numpy(float) for k in ("open","high","low","close","atr")]
    result = base.compute_outcomes(
        idx, expert.direction, *arrays, risk.rr, risk.sl_atr, risk.min_stop_pct, risk.max_hold,
        risk.breakeven_trigger, risk.breakeven_lock, risk.early_bars, risk.early_cut_r,
        FEE_RATE, SLIPPAGE_ABS
    )
    OUTCOME_CACHE[cache_key] = result
    return result


def enough_classes(y: np.ndarray) -> bool:
    return len(y) >= int(MODEL["calibration_min_rows"]) and len(np.unique(y)) == 2 and min(np.sum(y == 0), np.sum(y == 1)) >= int(MODEL["calibration_min_class"])


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrate(model: FittedModel, p: np.ndarray) -> np.ndarray:
    raw = model.calibrator.predict_proba(logit(p))[:, 1] if model.calibrator is not None else p
    s = float(MODEL["probability_shrinkage"])
    return np.clip((1-s) * raw + s * model.base_rate, 0.01, 0.99)


def router_score(x: pd.DataFrame, idx: np.ndarray, expert: SparseExpert) -> np.ndarray:
    direction = expert.direction
    if expert.feature_group == "trend":
        score = 0.34 + 0.20*(x.iloc[idx]["regime_trend"].to_numpy()>0) + 0.16*(x.iloc[idx]["h1_trend"].to_numpy()*direction>0) + 0.12*(x.iloc[idx]["m15_trend"].to_numpy()*direction>=0) + 0.10*np.clip(x.iloc[idx]["adx"].to_numpy()/35,0,1)
    elif expert.feature_group == "range":
        score = 0.38 + 0.25*(x.iloc[idx]["regime_range"].to_numpy()>0) + 0.12*np.clip((55-x.iloc[idx]["adx"].to_numpy())/35,0,1) + 0.10*np.clip(1-x.iloc[idx]["eff24"].to_numpy(),0,1)
    elif expert.feature_group == "breakout":
        score = 0.34 + 0.22*(x.iloc[idx]["regime_high_vol"].to_numpy()>0) + 0.15*np.clip(x.iloc[idx]["rel_vol"].to_numpy()/2,0,1) + 0.12*np.clip(x.iloc[idx]["range_exp"].to_numpy()/2,0,1)
    elif expert.feature_group == "cross":
        score = 0.36 + 0.20*np.clip(x.iloc[idx]["btc_eth_corr"].to_numpy(),0,1) + 0.16*(x.iloc[idx]["eth_h1_trend"].to_numpy()*direction>0) + 0.12*(x.iloc[idx]["h1_trend"].to_numpy()*direction>0)
    else:
        ext = np.clip((np.abs(x.iloc[idx]["funding_z"].to_numpy()) + np.abs(x.iloc[idx]["premium_z"].to_numpy()))/5,0,1)
        score = 0.38 + 0.24*(x.iloc[idx]["regime_range"].to_numpy()>0) + 0.22*ext
    return np.clip(score, 0.05, 0.99)


def micro_score(x: pd.DataFrame, idx: np.ndarray, direction: int) -> np.ndarray:
    taker = np.clip(0.5 + direction * (x.iloc[idx]["taker_ratio"].to_numpy() - 0.5), 0, 1)
    loc = x.iloc[idx]["close_loc"].to_numpy() if direction > 0 else 1 - x.iloc[idx]["close_loc"].to_numpy()
    vol = np.clip(x.iloc[idx]["rel_vol"].to_numpy()/2, 0, 1)
    eth = np.clip(0.5 + direction * x.iloc[idx]["eth_ret3"].to_numpy()*50, 0, 1)
    return np.clip(0.15 + 0.28*taker + 0.25*loc + 0.18*vol + 0.14*eth, 0.05, 0.99)


def fit_model(x: pd.DataFrame, idx: np.ndarray, expert: SparseExpert, policy: Policy, train_months: set[str]) -> FittedModel | None:
    assert_oos_isolation(train_months, f"fit {expert.name}")
    risk_key = hashlib.sha256(json.dumps(asdict(policy.risk), sort_keys=True).encode()).hexdigest()[:12]
    cache_key = (id(x), expert.id, risk_key, tuple(sorted(train_months)))
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    labels, exits, net_r, reasons = outcome_arrays(x, idx, expert, policy.risk)
    months = x["month"].to_numpy()[idx]
    valid = exits >= 0
    available = sorted(set(months[valid]) & train_months)
    if len(available) < 2:
        MODEL_CACHE[cache_key] = None
        return None
    cal_month = available[-1]
    core_months = set(available[:-1])
    core_mask = valid & np.isin(months, list(core_months))
    cal_mask = valid & (months == cal_month)
    y_core, y_cal = labels[core_mask], labels[cal_mask]
    if not enough_classes(y_core) or not enough_classes(y_cal):
        MODEL_CACHE[cache_key] = None
        return None
    features = list(FEATURE_GROUPS[expert.feature_group])
    core_rows, cal_rows = idx[core_mask], idx[cal_mask]
    cfg = policy.model
    model = HistGradientBoostingClassifier(
        max_depth=cfg.max_depth, learning_rate=cfg.learning_rate, max_iter=cfg.max_iter,
        l2_regularization=cfg.l2_regularization, min_samples_leaf=cfg.min_samples_leaf,
        random_state=BASE_SEED + expert.id * 1009
    )
    model.fit(x.iloc[core_rows][features].to_numpy(float), y_core, sample_weight=base.mild_class_weights(y_core))
    base_cal = model.predict_proba(x.iloc[cal_rows][features].to_numpy(float))[:,1]
    calibrator = None
    if enough_classes(y_cal):
        calibrator = LogisticRegression(C=0.45, solver="lbfgs", max_iter=400, random_state=BASE_SEED + expert.id)
        calibrator.fit(logit(base_cal), y_cal)
    base_rate = float(np.mean(y_cal))
    provisional = FittedModel(model, calibrator, cal_month, np.empty(0), y_cal, base_rate, 0.0, None, 0.0, int(core_mask.sum()), int(cal_mask.sum()))
    calibrated = calibrate(provisional, base_cal)
    router = router_score(x, cal_rows, expert)
    micro = micro_score(x, cal_rows, expert.direction)
    meta_x = np.column_stack([calibrated, router, micro, calibrated*router, calibrated*micro, router*micro])
    meta = None
    if enough_classes(y_cal):
        meta = LogisticRegression(C=0.55, solver="lbfgs", max_iter=400, random_state=BASE_SEED + expert.id*17)
        meta.fit(meta_x, y_cal)
    meta_p = meta.predict_proba(meta_x)[:,1] if meta is not None else calibrated
    provisional.calibration_scores = calibrated
    provisional.calibration_brier = float(np.mean((calibrated-y_cal)**2))
    provisional.meta_model = meta
    provisional.meta_brier = float(np.mean((meta_p-y_cal)**2))
    MODEL_CACHE[cache_key] = provisional
    return provisional


def evaluate_month(x: pd.DataFrame, idx: np.ndarray, expert: SparseExpert, policy: Policy, model: FittedModel, eval_month: str, train_months: set[str]):
    assert_oos_isolation(train_months, f"evaluate {eval_month}")
    risk_key = hashlib.sha256(json.dumps(asdict(policy.risk), sort_keys=True).encode()).hexdigest()[:12]
    cache_key = (id(x), expert.id, risk_key, eval_month, tuple(sorted(train_months)))
    cached = EVAL_CACHE.get(cache_key)
    if cached is None:
        labels, exits, net_r, reasons = outcome_arrays(x, idx, expert, policy.risk)
        months = x["month"].to_numpy()[idx]
        pos = np.flatnonzero((months == eval_month) & (exits >= 0))
        features = list(FEATURE_GROUPS[expert.feature_group])
        audit = {"raw_candidates": int(len(pos)), "model_available": True, "calibration_month": model.calibration_month, "calibration_rows": model.calibration_rows, "calibration_brier": model.calibration_brier, "meta_brier": model.meta_brier}
        history = list(np.asarray(model.calibration_scores,float)[-int(MODEL["online_rank_window"]):])
        if len(history) < int(MODEL["online_rank_min_history"]):
            prototypes=[]
            audit={**audit,"calibration_mean":float(np.mean(model.calibration_scores)) if len(model.calibration_scores) else 0.0,"eval_mean":0.0}
        else:
            rows = idx[pos]
            base_p = model.base_model.predict_proba(x.iloc[rows][features].to_numpy(float))[:,1]
            p = calibrate(model, base_p)
            router = router_score(x, rows, expert)
            micro = micro_score(x, rows, expert.direction)
            meta_x = np.column_stack([p,router,micro,p*router,p*micro,router*micro])
            meta_base = model.meta_model.predict_proba(meta_x)[:,1] if model.meta_model is not None else p
            prototypes=[]
            for j,k in enumerate(pos):
                hist=np.asarray(history,float)
                pct=float((np.sum(hist<=p[j])+1)/(len(hist)+1))
                history.append(float(p[j])); history=history[-int(MODEL["online_rank_window"]):]
                meta_p=float(np.clip(0.85*meta_base[j]+0.15*pct,0.01,0.99))
                utility=meta_p*policy.risk.rr-(1-meta_p)
                signal_i=int(idx[k]); exit_i=int(exits[k])
                prototypes.append({
                    "signal_i":signal_i,"exit_i":exit_i,"direction":expert.direction,"expert_id":expert.id,"expert":expert.name,"family":expert.family,
                    "setup_group":expert.setup_group,"base_probability":float(base_p[j]),"probability":float(p[j]),"online_percentile":pct,
                    "router":float(router[j]),"micro":float(micro[j]),"meta_probability":meta_p,"utility":float(utility),"net_r":float(net_r[k]),"win":bool(net_r[k]>0),
                    "reason":int(reasons[k]),"day":str(x.index[signal_i].date()),"cycle":int(x.iloc[signal_i][f"sparse_{expert.key}_cycle"])
                })
            audit={**audit,"calibration_mean":float(np.mean(model.calibration_scores)),"eval_mean":float(np.mean(p)) if len(p) else 0.0}
        EVAL_CACHE[cache_key]=(prototypes,audit)
    else:
        prototypes,audit=cached
    passed_probability=[e for e in prototypes if e["probability"]>=policy.min_probability and e["online_percentile"]>=policy.min_percentile]
    passed_meta=[e for e in passed_probability if e["meta_probability"]>=float(MODEL["minimum_meta_probability"]) and e["utility"]>=float(MODEL["minimum_expected_utility_r"])]
    capped=[]; last_exit=-1; day_count={}
    for proto in sorted(passed_meta,key=lambda z:(z["signal_i"],-z["utility"])):
        if len(capped)>=policy.monthly_target: break
        if proto["signal_i"]<=last_exit or day_count.get(proto["day"],0)>=1: continue
        e=dict(proto);e["policy_key"]=policy.key
        capped.append(e);day_count[e["day"]]=1;last_exit=e["exit_i"]
    return capped, {**audit,"after_probability":len(passed_probability),"after_meta":len(passed_meta),"after_cap":len(capped)}


def metrics(trades: list[dict[str, Any]]) -> dict[str,float]:
    if not trades:
        return {"trades":0,"wins":0,"win_rate":0.0,"avg_win_R":0.0,"avg_loss_R":0.0,"avg_win_loss_ratio":0.0,"profit_factor":0.0,"net_R":0.0,"max_drawdown_R":0.0,"expectancy_R":0.0}
    r=np.array([float(t["net_r"]) for t in trades]); wins=r[r>0]; losses=-r[r<=0]
    curve=np.cumsum(r); peak=np.maximum.accumulate(np.r_[0.0,curve]); dd=peak[1:]-curve
    avgw=float(wins.mean()) if len(wins) else 0.0; avgl=float(losses.mean()) if len(losses) else 0.0
    return {"trades":int(len(r)),"wins":int(len(wins)),"win_rate":float(len(wins)/len(r)),"avg_win_R":avgw,"avg_loss_R":avgl,"avg_win_loss_ratio":float(avgw/avgl) if avgl>0 else 0.0,"profit_factor":float(wins.sum()/losses.sum()) if losses.sum()>0 else (999.0 if wins.sum()>0 else 0.0),"net_R":float(r.sum()),"max_drawdown_R":float(dd.max()) if len(dd) else 0.0,"expectancy_R":float(r.mean())}


def wilson_lower(wins:int,n:int,z:float=1.0)->float:
    if n<=0:return 0.0
    p=wins/n; d=1+z*z/n
    return float((p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d)


def candidate_summary(candidate: Candidate) -> dict[str,float]:
    all_trades=[t for m in DEVELOPMENT_MONTHS for t in candidate.monthly_events.get(m,[])]
    agg=metrics(all_trades); prior=int(MODEL["stability_prior_trades"])
    shrunk=(agg["wins"]+0.5*prior)/(agg["trades"]+prior) if agg["trades"]+prior else 0
    active=sum(candidate.monthly_metrics.get(m,{}).get("trades",0)>0 for m in DEVELOPMENT_MONTHS)
    positive=sum(candidate.monthly_metrics.get(m,{}).get("net_R",0)>0 for m in DEVELOPMENT_MONTHS)
    worst=min((candidate.monthly_metrics.get(m,{}).get("net_R",0.0) for m in DEVELOPMENT_MONTHS),default=0.0)
    positives=[max(0,candidate.monthly_metrics.get(m,{}).get("net_R",0.0)) for m in DEVELOPMENT_MONTHS]
    share=max(positives)/(sum(positives) or 1.0)
    return {**agg,"shrunk_win_rate":float(shrunk),"wilson_lower":wilson_lower(int(agg["wins"]),int(agg["trades"])),"active_months":active,"positive_months":positive,"worst_month_R":float(worst),"max_single_month_profit_share":float(share)}


def eligibility(summary: dict[str,float], candidate: Candidate)->tuple[bool,list[str]]:
    reasons=[]
    checks=[
        (summary["trades"]>=int(GATE["min_total_trades"]),"总交易不足"),(summary["trades"]<=int(GATE["max_total_trades"]),"交易过多不再稀疏"),
        (summary["active_months"]>=int(GATE["min_active_months"]),"活跃月份不足"),(summary["positive_months"]>=int(GATE["min_positive_months"]),"正收益月份不足"),
        (summary["win_rate"]>=float(GATE["min_raw_win_rate"]),"原始胜率不足"),(summary["shrunk_win_rate"]>=float(GATE["min_shrunk_win_rate"]),"收缩后胜率不足"),
        (summary["wilson_lower"]>=float(GATE["min_wilson_lower"]),"胜率可信下界不足"),(summary["profit_factor"]>=float(GATE["min_profit_factor"]),"盈利因子不足"),
        (summary["avg_win_loss_ratio"]>=float(GATE["min_avg_win_loss_ratio"]),"实际盈亏比不足"),(summary["net_R"]>float(GATE["min_net_r"]),"累计净R不足"),
        (summary["worst_month_R"]>=-float(GATE["max_worst_month_loss_r"]),"最差月份亏损过大"),(summary["max_single_month_profit_share"]<=float(GATE["max_single_month_profit_share"]),"利润过度集中")]
    for ok,msg in checks:
        if not ok: reasons.append(msg)
    if any(candidate.monthly_metrics.get(m,{}).get("trades",0)>int(GATE["max_trades_per_month"]) for m in DEVELOPMENT_MONTHS): reasons.append("单月交易不够稀疏")
    return not reasons,reasons


def candidate_score(s:dict[str,float])->float:
    return float(7*s["net_R"]+20*s["shrunk_win_rate"]+9*s["wilson_lower"]+2*min(s["profit_factor"],4)+2*min(s["avg_win_loss_ratio"],4)+1.5*s["active_months"]-3*s["max_drawdown_R"]-4*s["max_single_month_profit_share"])


def evaluate_candidate(x:pd.DataFrame,expert:SparseExpert,policy:Policy)->Candidate:
    c=Candidate(policy); idx=expert_indices(x,expert)
    for eval_pos in range(3,9):
        month=MONTHS[eval_pos]; train=set(MONTHS[:eval_pos]); model=fit_model(x,idx,expert,policy,train)
        if model is None:
            c.monthly_events[month]=[]; c.monthly_metrics[month]=metrics([]); c.monthly_audit[month]={"model_available":False,"raw_candidates":int(np.sum(x.iloc[idx]["month"].to_numpy()==month))}; continue
        events,audit=evaluate_month(x,idx,expert,policy,model,month,train)
        c.monthly_events[month]=events;c.monthly_metrics[month]=metrics(events);c.monthly_audit[month]=audit
    c.aggregate=candidate_summary(c);c.eligible,c.reasons=eligibility(c.aggregate,c);c.score=candidate_score(c.aggregate)
    return c


def event_overlap(a:list[dict[str,Any]],b:list[dict[str,Any]],window:int=1)->dict[str,float]:
    if not a or not b:return {"signal_overlap":0.0,"loss_overlap":0.0,"return_correlation":0.0}
    am={int(t["signal_i"]):t for t in a}; bm={int(t["signal_i"]):t for t in b}; matched=[]
    for i,t in am.items():
        choices=[j for j in bm if abs(j-i)<=window and bm[j]["direction"]==t["direction"]]
        if choices: matched.append((t,bm[min(choices,key=lambda j:abs(j-i))]))
    overlap=len(matched)/max(1,min(len(a),len(b)))
    loss=sum((not x["win"]) and (not y["win"]) for x,y in matched)/max(1,len(matched))
    corr=0.0
    if len(matched)>=3:
        av=np.array([x["net_r"] for x,y in matched]);bv=np.array([y["net_r"] for x,y in matched])
        if av.std()>1e-9 and bv.std()>1e-9:corr=float(np.corrcoef(av,bv)[0,1])
    return {"signal_overlap":float(overlap),"loss_overlap":float(loss),"return_correlation":float(corr)}


def all_dev_events(c:Candidate)->list[dict[str,Any]]:
    return [t for m in DEVELOPMENT_MONTHS for t in c.monthly_events.get(m,[])]


def compatible(a:Candidate,b:Candidate)->bool:
    o=event_overlap(all_dev_events(a),all_dev_events(b),int(PORT["dedup_window_bars"]))
    return o["signal_overlap"]<=float(PORT["max_pair_signal_overlap"]) and o["loss_overlap"]<=float(PORT["max_pair_loss_overlap"]) and abs(o["return_correlation"])<=float(PORT["max_pair_return_correlation"])


def select_experts(best_by_expert:dict[int,Candidate])->tuple[list[Candidate],str]:
    eligible=sorted([c for c in best_by_expert.values() if c.eligible],key=lambda c:c.score,reverse=True)
    pool=eligible if eligible else sorted([c for c in best_by_expert.values() if c.aggregate.get("trades",0)>0],key=lambda c:c.score,reverse=True)
    selected=[]
    for c in pool:
        if len(selected)>=int(PORT["max_experts"]):break
        if all(compatible(c,s) for s in selected):selected.append(c)
    status="SPARSE_EXPERT_POOL_QUALIFIED" if len(eligible)>=int(PORT["min_experts"]) and len(selected)>=int(PORT["min_experts"]) else ("INSUFFICIENT_HIGH_PRECISION_EXPERTS" if selected else "ZERO_EXECUTABLE_SPARSE_EXPERTS")
    return selected,status


def combine_month(events_by_expert:dict[int,list[dict[str,Any]]])->tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    all_events=sorted([dict(t) for ev in events_by_expert.values() for t in ev],key=lambda t:(t["signal_i"],-t["utility"]))
    trades=[];dedup=[];conflicts=[];i=0;busy_until=-1;day_count={};expert_day={};setup_day={};losses={}
    while i<len(all_events):
        start=all_events[i]["signal_i"];group=[]
        while i<len(all_events) and all_events[i]["signal_i"]<=start+int(PORT["dedup_window_bars"]):group.append(all_events[i]);i+=1
        by_dir={d:[e for e in group if e["direction"]==d] for d in (-1,1)}
        direction=None
        if by_dir[-1] and by_dir[1]:
            best_l=max(by_dir[1],key=lambda e:e["utility"]);best_s=max(by_dir[-1],key=lambda e:e["utility"])
            gap=abs(best_l["utility"]-best_s["utility"])
            if gap<float(PORT["direction_conflict_score_margin"]):
                conflicts.append({"signal_i":start,"long_experts":"|".join(e["expert"] for e in by_dir[1]),"short_experts":"|".join(e["expert"] for e in by_dir[-1]),"decision":"skip","utility_gap":gap});continue
            direction=1 if best_l["utility"]>best_s["utility"] else -1
            conflicts.append({"signal_i":start,"long_experts":"|".join(e["expert"] for e in by_dir[1]),"short_experts":"|".join(e["expert"] for e in by_dir[-1]),"decision":"long" if direction>0 else "short","utility_gap":gap})
        else: direction=1 if by_dir[1] else -1
        candidates=by_dir[direction];primary=max(candidates,key=lambda e:e["utility"]);primary=dict(primary)
        primary["source_experts"]=[e["expert"] for e in candidates];primary["agreement_count"]=len(candidates)
        dedup.append({"signal_i":start,"direction":direction,"candidate_count":len(candidates),"source_experts":"|".join(primary["source_experts"]),"chosen":primary["expert"],"deduplicated_count":max(0,len(candidates)-1)})
        day=primary["day"];eid=primary["expert_id"];setup=primary["setup_group"]
        if primary["signal_i"]<=busy_until or day_count.get(day,0)>=int(PORT["max_trades_per_day"]) or expert_day.get((eid,day),0)>=int(PORT["max_trades_per_expert_day"]) or setup_day.get((setup,day),0)>=int(PORT["max_trades_per_setup_day"]) or losses.get((eid,day),0)>=int(PORT["max_consecutive_losses_expert_day"]):continue
        trades.append(primary);busy_until=primary["exit_i"];day_count[day]=day_count.get(day,0)+1;expert_day[(eid,day)]=expert_day.get((eid,day),0)+1;setup_day[(setup,day)]=setup_day.get((setup,day),0)+1
        losses[(eid,day)]=0 if primary["win"] else losses.get((eid,day),0)+1
    return trades,dedup,conflicts


def stage_pass(m:dict[str,float],trades:list[dict[str,Any]])->bool:
    shares=pd.Series([t["expert"] for t in trades]).value_counts(normalize=True) if trades else pd.Series(dtype=float)
    return int(STAGE["min_trades"])<=m["trades"]<=int(STAGE["max_trades"]) and m["win_rate"]>=float(STAGE["min_win_rate"]) and m["avg_win_loss_ratio"]>=float(STAGE["min_avg_win_loss_ratio"]) and m["profit_factor"]>=float(STAGE["min_profit_factor"]) and m["max_drawdown_R"]<=float(STAGE["max_drawdown_r"]) and len(shares)>=int(STAGE["min_active_experts"]) and (shares.max() if len(shares) else 0)<=float(STAGE["max_single_expert_trade_share"])


def final_pass(m:dict[str,float])->bool:
    return int(FINAL["min_trades"])<=m["trades"]<=int(FINAL["max_trades"]) and m["win_rate"]>=float(FINAL["min_win_rate"]) and m["avg_win_loss_ratio"]>=float(FINAL["min_avg_win_loss_ratio"])


def detailed_frame(x:pd.DataFrame,trades:list[dict[str,Any]],month:str)->pd.DataFrame:
    rows=[]
    reason={1:"TP",2:"PROTECTED_STOP",3:"TIME_EXIT",4:"EARLY_CUT"}
    for t in trades:
        si=t["signal_i"];ei=si+1;xi=t["exit_i"]
        rows.append({"signal_time_utc":x.index[si].isoformat(),"entry_time_utc":x.index[ei].isoformat(),"exit_time_utc":x.index[xi].isoformat(),"month":month,"direction":"LONG" if t["direction"]>0 else "SHORT","expert":t["expert"],"family":t["family"],"setup_group":t["setup_group"],"source_experts":"|".join(t.get("source_experts",[t["expert"]])),"agreement_count":t.get("agreement_count",1),"policy_key":t["policy_key"],"calibrated_probability":t["probability"],"online_percentile":t["online_percentile"],"meta_probability":t["meta_probability"],"expected_utility_R":t["utility"],"net_R":t["net_r"],"win":t["win"],"exit_reason":reason.get(t["reason"],"UNKNOWN"),"bars":xi-ei+1})
    return pd.DataFrame(rows)


def synthetic_smoke() -> None:
    raw,eth,premium,funding=base.synthetic_inputs(26000)
    x,_=base.add_features(raw,eth,premium,funding);x=add_sparse_masks(x)
    assert len(x)>10000
    counts={e.name:len(expert_indices(x,e)) for e in EXPERTS}
    assert len(counts)==16 and any(v>0 for v in counts.values())
    sample={"net_r":1.5,"win":True,"signal_i":10,"exit_i":15,"direction":1,"expert_id":0,"expert":EXPERTS[0].name,"family":EXPERTS[0].family,"setup_group":"x","policy_key":"x","utility":0.5,"day":"2026-05-01","probability":0.6,"online_percentile":0.9,"meta_probability":0.6}
    trades,dedup,conflicts=combine_month({0:[sample]})
    assert len(trades)==1
    print("V96_SELF_TEST_OK",json.dumps(counts,ensure_ascii=False))



def pipeline_smoke() -> None:
    raw, eth, premium, funding = base.synthetic_inputs(88000)
    start = int(pd.Timestamp("2025-09-01", tz="UTC").timestamp() * 1000)
    times = start + (raw["open_time"].to_numpy() - int(raw["open_time"].iloc[0]))
    for frame in (raw, eth, premium):
        frame["open_time"] = times
        frame["close_time"] = times + 299999
    funding["calc_time"] = times[::96][:len(funding)]
    x, _ = base.add_features(raw, eth, premium, funding)
    x = add_sparse_masks(x)
    representatives = [EXPERTS[1], EXPERTS[6], EXPERTS[10]]
    candidates = []
    for expert in representatives:
        candidate = evaluate_candidate(x, expert, policy_grid(expert)[0])
        candidates.append(candidate)
    selected = [candidate for candidate in candidates if candidate.aggregate.get("trades", 0) > 0]
    events = {c.policy.expert_id: c.monthly_events.get("2026-05", []) for c in selected}
    trades, dedup, conflicts = combine_month(events)
    smoke_dir = ROOT / ".v96_pipeline_smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir()
    pd.DataFrame([{"expert": EXPERT_BY_ID[c.policy.expert_id].name, **c.aggregate} for c in candidates]).to_csv(smoke_dir / "candidate_summary.csv", index=False)
    detailed_frame(x, trades, "2026-05").to_csv(smoke_dir / "portfolio_trades.csv", index=False)
    pd.DataFrame(dedup).to_csv(smoke_dir / "dedup.csv", index=False)
    pd.DataFrame(conflicts).to_csv(smoke_dir / "conflicts.csv", index=False)
    payload = {"representative_experts": len(representatives), "selected_nonzero": len(selected), "portfolio_metrics": metrics(trades)}
    (smoke_dir / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    required = ["candidate_summary.csv", "portfolio_trades.csv", "dedup.csv", "conflicts.csv", "status.json"]
    assert all((smoke_dir / name).exists() for name in required)
    print("V96_PIPELINE_SMOKE_OK", json.dumps(payload, ensure_ascii=False))

def main()->None:
    clear_results()
    raw,audit=base.load_official_data();eth,ea=base.load_auxiliary_kline("ETHUSDT","klines");premium,pa=base.load_auxiliary_kline(SYMBOL,"premiumIndexKlines");funding,fa=base.load_funding_rate()
    audit["auxiliary_sources"]={"eth":ea,"premium":pa,"funding":fa}
    x,align=base.add_features(raw,eth,premium,funding);x=add_sparse_masks(x);audit["alignment"]=align
    all_candidates:dict[int,list[Candidate]]={}
    leaderboard=[];monthly_rows=[];funnel=[]
    for expert in EXPERTS:
        print(f"SEARCH_SPARSE_EXPERT={expert.id}:{expert.name}:{expert.family}")
        candidates=[evaluate_candidate(x,expert,p) for p in policy_grid(expert)]
        candidates.sort(key=lambda c:c.score,reverse=True);all_candidates[expert.id]=candidates
        for rank,c in enumerate(candidates,1):
            leaderboard.append({"expert_id":expert.id,"expert":expert.name,"family":expert.family,"rank":rank,"policy_key":c.policy.key,"eligible":c.eligible,"reasons":"|".join(c.reasons),"score":c.score,**c.aggregate})
        best=candidates[0]
        for m in DEVELOPMENT_MONTHS:
            monthly_rows.append({"expert":expert.name,"family":expert.family,"policy_key":best.policy.key,"month":m,**best.monthly_metrics[m]})
            funnel.append({"expert":expert.name,"family":expert.family,"month":m,**best.monthly_audit[m]})
    best_by={eid:cands[0] for eid,cands in all_candidates.items()}
    selected,status=select_experts(best_by)
    # Development portfolio uses the frozen best policy of each selected expert.
    development_portfolio={};development_metrics={};development_dedup=[];development_conflicts=[]
    for m in DEVELOPMENT_MONTHS:
        events={c.policy.expert_id:c.monthly_events.get(m,[]) for c in selected}
        tr,dd,cc=combine_month(events);development_portfolio[m]=tr;development_metrics[m]=metrics(tr)
        for r in dd:r["month"]=m
        for r in cc:r["month"]=m
        development_dedup.extend(dd);development_conflicts.extend(cc)
    # Fresh OOS: policies are frozen before June, then refit only on Sep-May.
    june_events={};june_audit=[];shadow=[]
    for c in selected:
        expert=EXPERT_BY_ID[c.policy.expert_id];idx=expert_indices(x,expert);model=fit_model(x,idx,expert,c.policy,set(MONTHS[:9]))
        if model is None:events=[];faudit={"model_available":False,"raw_candidates":int(np.sum(x.iloc[idx]["month"].to_numpy()==OOS_MONTH))}
        else:events,faudit=evaluate_month(x,idx,expert,c.policy,model,OOS_MONTH,set(MONTHS[:9]))
        june_events[expert.id]=events;june_audit.append({"expert":expert.name,"family":expert.family,"month":OOS_MONTH,**faudit})
        for t in events:shadow.append(dict(t,month=OOS_MONTH))
    june_trades,june_dedup,june_conflicts=combine_month(june_events);june_metric=metrics(june_trades)
    may_trades=development_portfolio["2026-05"];may_metric=development_metrics["2026-05"]
    # Pairwise overlap for selected and all best experts.
    overlap_rows=[]
    for a,b in itertools.combinations(best_by.values(),2):
        ov=event_overlap(all_dev_events(a),all_dev_events(b),int(PORT["dedup_window_bars"]))
        overlap_rows.append({"expert_a":EXPERT_BY_ID[a.policy.expert_id].name,"expert_b":EXPERT_BY_ID[b.policy.expert_id].name,**ov,"compatible":compatible(a,b)})
    selected_ids={c.policy.expert_id for c in selected}
    selection_rows=[]
    for eid,c in best_by.items():
        selection_rows.append({"expert_id":eid,"expert":EXPERT_BY_ID[eid].name,"family":EXPERT_BY_ID[eid].family,"selected":eid in selected_ids,"eligible":c.eligible,"policy_key":c.policy.key,"selection_status":status,"rejection_reason":"selected" if eid in selected_ids else ("|".join(c.reasons) or "correlation_or_pool_limit"),**c.aggregate})
    selected_payload={EXPERT_BY_ID[c.policy.expert_id].name:{"expert_id":c.policy.expert_id,"family":EXPERT_BY_ID[c.policy.expert_id].family,"policy_key":c.policy.key,"policy":asdict(c.policy),"development_summary":c.aggregate,"eligible":c.eligible} for c in selected}
    # Outputs
    pd.DataFrame(leaderboard).to_csv(RESULTS/"sparse_expert_leaderboard.csv",index=False)
    pd.DataFrame(monthly_rows).to_csv(RESULTS/"expert_monthly_stats.csv",index=False)
    pd.DataFrame(funnel+june_audit).to_csv(RESULTS/"signal_funnel.csv",index=False)
    pd.DataFrame(overlap_rows).to_csv(RESULTS/"expert_overlap.csv",index=False)
    pd.DataFrame(selection_rows).to_csv(RESULTS/"selection_audit.csv",index=False)
    pd.DataFrame(development_dedup+[dict(r,month=OOS_MONTH) for r in june_dedup]).to_csv(RESULTS/"signal_deduplication.csv",index=False)
    pd.DataFrame(development_conflicts+[dict(r,month=OOS_MONTH) for r in june_conflicts]).to_csv(RESULTS/"signal_conflicts.csv",index=False)
    portfolio_frame=pd.concat([detailed_frame(x,may_trades,"2026-05"),detailed_frame(x,june_trades,OOS_MONTH)],ignore_index=True)
    portfolio_frame.to_csv(RESULTS/"portfolio_trades.csv",index=False);portfolio_frame.to_csv(RESULTS/"trades.csv",index=False)
    shadow_rows=[]
    for c in selected:
        for m in DEVELOPMENT_MONTHS:
            for t in c.monthly_events.get(m,[]):shadow_rows.append(dict(t,month=m))
    shadow_rows.extend(shadow)
    pd.DataFrame(shadow_rows).to_csv(RESULTS/"expert_shadow_trades.csv",index=False)
    coverage=[]
    for m in ("2026-05",OOS_MONTH):
        tr=may_trades if m=="2026-05" else june_trades
        vc=pd.Series([t["expert"] for t in tr]).value_counts() if tr else pd.Series(dtype=int)
        for name,count in vc.items():coverage.append({"month":m,"expert":name,"trades":int(count),"trade_share":float(count/len(tr))})
    pd.DataFrame(coverage).to_csv(RESULTS/"opportunity_coverage.csv",index=False)
    (RESULTS/"selected_policy.json").write_text(json.dumps(selected_payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    (RESULTS/"data_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    qualified_pool=status=="SPARSE_EXPERT_POOL_QUALIFIED"
    stage_ok=qualified_pool and stage_pass(may_metric,may_trades) and stage_pass(june_metric,june_trades)
    final_ok=qualified_pool and final_pass(may_metric) and final_pass(june_metric)
    status_payload={
        "qualified":False,"research_stage_qualified":stage_ok,"final_hard_metrics_passed":final_ok,"not_for_live_trading":True,"fresh_blind_month_required":True,
        "selection_status":status,"engine":ENGINE_NAME,"architecture":"16 sparse setup experts -> independent holdout calibration -> high precision gate -> low overlap pool -> one-position opportunity coverage portfolio",
        "selected_expert_count":len(selected),"eligible_sparse_expert_count":sum(c.eligible for c in best_by.values()),"selected_experts":selected_payload,
        "monthly_stats":{"2026-05":may_metric,OOS_MONTH:june_metric},"development_monthly_stats":development_metrics,
        "constraints":{"sparse_expert_gate":GATE,"research_stage":STAGE,"final_target":FINAL,"portfolio":PORT},
        "oos_isolation":{"used_for_training":False,"used_for_thresholds":False,"used_for_expert_selection":False,"used_for_portfolio_selection":False,"evaluation_occurs_after_policy_freeze":True},
        "searched_experts":len(EXPERTS),"searched_policies":sum(len(v) for v in all_candidates.values())
    }
    (RESULTS/"status.json").write_text(json.dumps(status_payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    report=f"""# BTCUSDT 5分钟 高精度稀疏专家池与机会覆盖组合 V9.6 报告

- 架构：16个具体形态低频专家 → 独立滚动训练与留出月校准 → 原始/收缩/Wilson胜率筛选 → 低重合专家池 → 全局单仓组合。
- 选择状态：**{status}**。
- 正式合格稀疏专家：{sum(c.eligible for c in best_by.values())}；最终选入：{len(selected)}。
- 选入专家：{'、'.join(EXPERT_BY_ID[c.policy.expert_id].name for c in selected) or '无'}。
- 阶段验收：**{'通过' if stage_ok else '未通过'}**；原最终指标：**{'通过' if final_ok else '未通过'}**。
- 实盘资格：**不合格**；5月和6月已经被查看，后续仍需新的完整盲测月。

## 组合账户月度结果

| 月份 | 交易 | 胜率 | 实际盈亏比 | 盈利因子 | 净R | 最大回撤R | 阶段通过 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-05 | {may_metric['trades']} | {may_metric['win_rate']:.2%} | {may_metric['avg_win_loss_ratio']:.3f} | {may_metric['profit_factor']:.3f} | {may_metric['net_R']:.3f} | {may_metric['max_drawdown_R']:.3f} | {'是' if stage_pass(may_metric,may_trades) else '否'} |
| 2026-06 | {june_metric['trades']} | {june_metric['win_rate']:.2%} | {june_metric['avg_win_loss_ratio']:.3f} | {june_metric['profit_factor']:.3f} | {june_metric['net_R']:.3f} | {june_metric['max_drawdown_R']:.3f} | {'是' if stage_pass(june_metric,june_trades) else '否'} |

## 研究原则

单个专家允许低频，但必须以跨月总样本、收缩后胜率、Wilson可信下界、实际盈亏比、盈利因子、最差月份和利润集中度共同晋级。组合不要求多个专家同时确认；独立合格信号可执行，但高度重合专家不会同时进入专家池。
"""
    (RESULTS/"report.md").write_text(report,encoding="utf-8")
    (RESULTS/"run_identity.txt").write_text(f"{ENGINE_NAME}\noos={OOS_MONTH}\noutput=results_v9_6\nselection_status={status}\n",encoding="utf-8")
    print(json.dumps(status_payload["monthly_stats"],ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--self-test",action="store_true");parser.add_argument("--pipeline-smoke",action="store_true");args=parser.parse_args()
    if args.self_test:synthetic_smoke()
    elif args.pipeline_smoke:pipeline_smoke()
    else:main()
