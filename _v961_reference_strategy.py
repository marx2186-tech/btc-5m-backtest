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
RESULTS = ROOT / "results_v9_6_1_reference"
RESULTS.mkdir(exist_ok=True)
ENGINE_VERSION = "V9.6.1"
ENGINE_NAME = "BTC 5m sparse expert discovery and soft-meta pool V9.6.1"
OOS_MONTH = "2026-06"

REQUEST_PATH = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(ROOT / "request.v9_6_1_reference.runtime.json")))
REQUEST = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
SYMBOL = str(REQUEST["symbol"]).upper()
INTERVAL = str(REQUEST["interval"]).lower()
EVAL_MONTHS = tuple(str(x) for x in REQUEST["months"])
if EVAL_MONTHS != ("2026-05", "2026-06"):
    raise ValueError("V9.6.1 requires evaluation months 2026-05 and diagnostic OOS 2026-06")
MONTHS = tuple(str(p) for p in pd.period_range("2025-01", "2026-06", freq="M"))
DEVELOPMENT_START_INDEX = 4
DEVELOPMENT_MONTHS = MONTHS[DEVELOPMENT_START_INDEX:-1]
FEE_RATE = float(REQUEST["fee_rate_per_side"])
SLIPPAGE_ABS = float(REQUEST["tick_size"]) * int(REQUEST["slippage_ticks_per_fill"])
if FEE_RATE != 0.0005 or abs(SLIPPAGE_ABS - 0.2) > 1e-12:
    raise ValueError("V9.6.1 fixes one-side fee at 0.050% and slippage at 0.2 USDT per fill")

FINAL = REQUEST["final_target"]
STAGE = REQUEST["research_stage"]

CANDIDATE_GATE = REQUEST["candidate_gate"]
WATCH_GATE = REQUEST["watch_gate"]
QUALIFIED_GATE = REQUEST["qualified_gate"]
PORT = REQUEST["portfolio"]
MODEL = REQUEST["model"]
SEARCH = REQUEST["search"]
BASE_SEED = int(MODEL["base_seed"])

# Load the frozen V9.5.1 data/feature/outcome engine from this package. A private
# compatibility request prevents the base module from consuming V9.6 settings.
compat = ROOT / ".v961_reference_base_request.json"
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
spec = importlib.util.spec_from_file_location("v961_reference_base_engine", ROOT / "_v964_base_engine.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load _v961_base_engine.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
base.MONTHS = MONTHS
base.DEVELOPMENT_MONTHS = DEVELOPMENT_MONTHS
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
    min_percentile: float
    min_expected_utility_r: float

    @property
    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


@dataclass
class FittedModel:
    base_model: HistGradientBoostingClassifier
    calibrator: LogisticRegression | None
    calibration_month: str
    calibration_scores: np.ndarray
    calibration_utility_scores: np.ndarray
    calibration_labels: np.ndarray
    base_rate: float
    calibration_brier: float
    meta_model: LogisticRegression | None
    meta_brier: float
    avg_win_r: float
    avg_loss_r: float
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
    tier: str = "REJECTED"
    reasons: list[str] = field(default_factory=list)
    score: float = -1e12


EXPERTS: tuple[SparseExpert, ...] = (
    SparseExpert(0, "trend_reclaim_long", "趋势回踩重新站稳多头", "趋势回踩", 1, "trend", "趋势回踩"),
    SparseExpert(1, "trend_reject_short", "趋势反抽失败空头", "趋势回踩", -1, "trend", "趋势反抽"),
    SparseExpert(2, "ema21_reclaim_long", "EMA21回踩收回多头", "均线回踩", 1, "trend", "EMA21收回"),
    SparseExpert(3, "ema21_reject_short", "EMA21反抽受阻空头", "均线回踩", -1, "trend", "EMA21受阻"),
    SparseExpert(4, "vwap_reclaim_long", "VWAP重新站回多头", "VWAP回归", 1, "trend", "VWAP收回"),
    SparseExpert(5, "vwap_reject_short", "VWAP反抽失败空头", "VWAP回归", -1, "trend", "VWAP受阻"),
    SparseExpert(6, "sr_flip_long", "阻力转支撑确认多头", "结构角色转换", 1, "trend", "阻力转支撑"),
    SparseExpert(7, "sr_flip_short", "支撑转阻力确认空头", "结构角色转换", -1, "trend", "支撑转阻力"),
    SparseExpert(8, "contract_pullback_long", "缩量回踩恢复多头", "量价回踩", 1, "trend", "缩量回踩"),
    SparseExpert(9, "contract_pullback_short", "缩量反抽恢复空头", "量价回踩", -1, "trend", "缩量反抽"),
    SparseExpert(10, "session_continue_long", "欧美时段顺势延续多头", "时段延续", 1, "session", "时段延续"),
    SparseExpert(11, "session_continue_short", "欧美时段顺势延续空头", "时段延续", -1, "session", "时段延续"),
    SparseExpert(12, "sweep_immediate_long", "区间扫低立即收回多头", "流动性反转", 1, "range", "扫低立即收回"),
    SparseExpert(13, "sweep_immediate_short", "区间扫高立即跌回空头", "流动性反转", -1, "range", "扫高立即跌回"),
    SparseExpert(14, "sweep_confirm_long", "扫低后确认K多头", "流动性确认", 1, "range", "扫低确认"),
    SparseExpert(15, "sweep_confirm_short", "扫高后确认K空头", "流动性确认", -1, "range", "扫高确认"),
    SparseExpert(16, "second_test_long", "区间二次测试不破多头", "二次测试", 1, "range", "二次测试"),
    SparseExpert(17, "second_test_short", "区间二次测试不破空头", "二次测试", -1, "range", "二次测试"),
    SparseExpert(18, "sweep_vwap_long", "扫低后站回VWAP多头", "流动性VWAP", 1, "range", "扫低VWAP"),
    SparseExpert(19, "sweep_vwap_short", "扫高后跌破VWAP空头", "流动性VWAP", -1, "range", "扫高VWAP"),
    SparseExpert(20, "eth_divergence_long", "BTC弱于ETH背离修复多头", "跨资产背离", 1, "cross", "BTCETH背离"),
    SparseExpert(21, "eth_divergence_short", "BTC强于ETH背离修复空头", "跨资产背离", -1, "cross", "BTCETH背离"),
    SparseExpert(22, "squeeze_break_long", "压缩后首次放量突破多头", "波动扩张", 1, "breakout", "压缩突破"),
    SparseExpert(23, "squeeze_break_short", "压缩后首次放量突破空头", "波动扩张", -1, "breakout", "压缩突破"),
    SparseExpert(24, "failed_break_long", "向下假突破回区间多头", "假突破反转", 1, "range", "假突破回归"),
    SparseExpert(25, "failed_break_short", "向上假突破回区间空头", "假突破反转", -1, "range", "假突破回归"),
    SparseExpert(26, "funding_extreme_long", "资金费率极端回归多头", "衍生品反转", 1, "derivative", "资金费率极端"),
    SparseExpert(27, "funding_extreme_short", "资金费率极端回归空头", "衍生品反转", -1, "derivative", "资金费率极端"),
    SparseExpert(28, "premium_extreme_long", "合约溢价极端修复多头", "基差反转", 1, "derivative", "溢价极端"),
    SparseExpert(29, "premium_extreme_short", "合约溢价极端修复空头", "基差反转", -1, "derivative", "溢价极端"),
    SparseExpert(30, "opening_break_long", "欧美开盘区间突破多头", "开盘突破", 1, "session", "开盘突破"),
    SparseExpert(31, "opening_break_short", "欧美开盘区间突破空头", "开盘突破", -1, "session", "开盘突破"),
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
    "cross": COMMON_FEATURES + ("regime_trend","regime_range","setup_reclaim_long","setup_reclaim_short"),
    "derivative": COMMON_FEATURES + ("regime_range","setup_reclaim_long","setup_reclaim_short","setup_range_reversal_long","setup_range_reversal_short"),
    "session": COMMON_FEATURES + ("regime_trend","regime_high_vol","setup_continuation_long","setup_continuation_short","setup_high_break_long","setup_high_break_short"),
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
    low12 = y["low"].rolling(12).min().shift(1)
    high12 = y["high"].rolling(12).max().shift(1)
    low24 = y["low"].rolling(24).min().shift(2)
    high24 = y["high"].rolling(24).max().shift(2)
    low36 = y["low"].rolling(36).min().shift(1)
    high36 = y["high"].rolling(36).max().shift(1)
    prev_sweep_low = y["low"].shift(1) < y["low"].rolling(12).min().shift(2)
    prev_sweep_high = y["high"].shift(1) > y["high"].rolling(12).max().shift(2)
    hour = y.index.hour
    europe_us = ((hour >= 7) & (hour <= 21))
    open_window = ((hour >= 7) & (hour <= 10)) | ((hour >= 13) & (hour <= 16))
    masks: dict[int, pd.Series] = {
        0: (y["regime_trend"]>0)&(y["h1_trend"]>0)&(y["m15_trend"]>=0)&(y["setup_reclaim_long"]>0)&y["rsi"].between(40,64),
        1: (y["regime_trend"]>0)&(y["h1_trend"]<0)&(y["m15_trend"]<=0)&(y["setup_reclaim_short"]>0)&y["rsi"].between(36,60),
        2: (y["h1_trend"]>0)&(y["m15_trend"]>=0)&y["ema21_gap"].between(-0.005,0.0025)&(y["ret3"]>0)&(y["close_loc"]>0.54),
        3: (y["h1_trend"]<0)&(y["m15_trend"]<=0)&y["ema21_gap"].between(-0.0025,0.005)&(y["ret3"]<0)&(y["close_loc"]<0.46),
        4: (y["h1_trend"]>=0)&(y["vwap_dev"].shift(1)<0)&(y["vwap_dev"]>=0)&(y["ret3"]>0)&(y["close_loc"]>0.54),
        5: (y["h1_trend"]<=0)&(y["vwap_dev"].shift(1)>0)&(y["vwap_dev"]<=0)&(y["ret3"]<0)&(y["close_loc"]<0.46),
        6: (y["h1_trend"]>=0)&(y["close"]>high12)&(y["low"]<=high12*1.002)&(y["close_loc"]>0.55)&europe_us,
        7: (y["h1_trend"]<=0)&(y["close"]<low12)&(y["high"]>=low12*0.998)&(y["close_loc"]<0.45)&europe_us,
        8: (y["h1_trend"]>0)&(y["m15_trend"]>=0)&(y["ret12"]>0)&(y["rel_vol"]<1.05)&y["ema21_gap"].between(-0.004,0.004)&(y["ret3"]>0),
        9: (y["h1_trend"]<0)&(y["m15_trend"]<=0)&(y["ret12"]<0)&(y["rel_vol"]<1.05)&y["ema21_gap"].between(-0.004,0.004)&(y["ret3"]<0),
        10: europe_us&(y["h1_trend"]>0)&(y["m15_trend"]>0)&(y["ret12"]>0)&((y["setup_continuation_long"]>0)|(y["setup_reclaim_long"]>0)),
        11: europe_us&(y["h1_trend"]<0)&(y["m15_trend"]<0)&(y["ret12"]<0)&((y["setup_continuation_short"]>0)|(y["setup_reclaim_short"]>0)),
        12: (y["regime_range"]>0)&(y["setup_range_reversal_long"]>0)&(y["lower_wick"]>=0.22)&(y["close_loc"]>=0.52),
        13: (y["regime_range"]>0)&(y["setup_range_reversal_short"]>0)&(y["upper_wick"]>=0.22)&(y["close_loc"]<=0.48),
        14: (y["regime_range"]>0)&prev_sweep_low&(y["close"]>y["close"].shift(1))&(y["ret1"]>0)&(y["close_loc"]>0.52),
        15: (y["regime_range"]>0)&prev_sweep_high&(y["close"]<y["close"].shift(1))&(y["ret1"]<0)&(y["close_loc"]<0.48),
        16: (y["regime_range"]>0)&(y["setup_range_second_test_long"]>0)&(y["low"]>=low24*0.996)&(y["vwap_dev"]>-0.007),
        17: (y["regime_range"]>0)&(y["high"]<=high24*1.004)&(y["upper_wick"]>=0.25)&(y["rsi"]>=52)&(y["ret3"]<0),
        18: (y["regime_range"]>0)&(y["setup_range_reversal_long"]>0)&(y["vwap_dev"]>=0)&(y["ret3"]>0),
        19: (y["regime_range"]>0)&(y["setup_range_reversal_short"]>0)&(y["vwap_dev"]<=0)&(y["ret3"]<0),
        20: (y["ret12"]<-0.003)&(y["eth_ret12"]>y["ret12"]+0.0025)&(y["lower_wick"]>0.20)&(y["ret3"]>0)&(y["btc_eth_corr"]>0.15),
        21: (y["ret12"]>0.003)&(y["eth_ret12"]<y["ret12"]-0.0025)&(y["upper_wick"]>0.20)&(y["ret3"]<0)&(y["btc_eth_corr"]>0.15),
        22: (y["setup_high_break_long"]>0)&(y["bb_rank"].shift(6)<=0.48)&(y["rel_vol"]>=1.05)&(y["close_loc"]>0.58),
        23: (y["setup_high_break_short"]>0)&(y["bb_rank"].shift(6)<=0.48)&(y["rel_vol"]>=1.05)&(y["close_loc"]<0.42),
        24: (y["low"]<low12)&(y["close"]>low12)&(y["close_loc"]>0.56)&(y["eff24"]<0.55),
        25: (y["high"]>high12)&(y["close"]<high12)&(y["close_loc"]<0.44)&(y["eff24"]<0.55),
        26: (y["funding_z"]<=-1.0)&((y["setup_reclaim_long"]>0)|(y["setup_range_reversal_long"]>0))&(y["close_loc"]>0.52),
        27: (y["funding_z"]>=1.0)&((y["setup_reclaim_short"]>0)|(y["setup_range_reversal_short"]>0))&(y["close_loc"]<0.48),
        28: (y["premium_z"]<=-1.15)&(y["ret3"]>0)&(y["close_loc"]>0.54)&(y["lower_wick"]>0.18),
        29: (y["premium_z"]>=1.15)&(y["ret3"]<0)&(y["close_loc"]<0.46)&(y["upper_wick"]>0.18),
        30: open_window&(y["close"]>high36)&(y["rel_vol"]>=1.0)&(y["close_loc"]>0.58)&(y["eth_ret3"]>=-0.001),
        31: open_window&(y["close"]<low36)&(y["rel_vol"]>=1.0)&(y["close_loc"]<0.42)&(y["eth_ret3"]<=0.001),
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
    if expert.family in {"流动性反转","流动性确认","二次测试","流动性VWAP","假突破反转","衍生品反转","基差反转"}:
        return (
            RiskConfig(1.8,1.05,0.0030,144,0.85,0.08,42,0.32),
            RiskConfig(2.0,1.18,0.0033,180,0.95,0.10,48,0.34),
        )
    if expert.family in {"波动扩张","开盘突破","跨资产背离"}:
        return (
            RiskConfig(1.8,1.20,0.0033,168,0.90,0.08,48,0.36),
            RiskConfig(2.1,1.35,0.0037,216,1.00,0.10,60,0.40),
        )
    return (
        RiskConfig(1.8,1.12,0.0031,180,0.90,0.08,48,0.34),
        RiskConfig(2.0,1.28,0.0035,216,1.00,0.10,60,0.38),
    )

def model_config(expert: SparseExpert) -> ModelConfig:
    if expert.feature_group in {"range","derivative"}:
        return ModelConfig(2,0.032,210,7.5,20)
    if expert.feature_group in {"breakout","session"}:
        return ModelConfig(2,0.028,230,8.0,24)
    if expert.feature_group == "cross":
        return ModelConfig(2,0.030,220,7.5,22)
    return ModelConfig(2,0.028,240,8.0,28)

def policy_grid(expert: SparseExpert) -> list[Policy]:
    return [
        Policy(expert.id,risk,model_config(expert),int(target),float(pct),float(util))
        for risk,pct,target,util in itertools.product(
            risk_grid(expert),SEARCH["online_percentiles"],SEARCH["monthly_targets"],SEARCH["minimum_expected_utility_r"]
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
    if len(available) < 3:
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
    model = HistGradientBoostingClassifier(max_depth=cfg.max_depth,learning_rate=cfg.learning_rate,max_iter=cfg.max_iter,l2_regularization=cfg.l2_regularization,min_samples_leaf=cfg.min_samples_leaf,random_state=BASE_SEED+expert.id*1009)
    model.fit(x.iloc[core_rows][features].to_numpy(float),y_core,sample_weight=base.mild_class_weights(y_core))
    base_cal = model.predict_proba(x.iloc[cal_rows][features].to_numpy(float))[:,1]
    calibrator = LogisticRegression(C=0.40,solver="lbfgs",max_iter=400,random_state=BASE_SEED+expert.id)
    calibrator.fit(logit(base_cal),y_cal)
    base_rate=float(np.mean(y_cal))
    cal_net=np.asarray(net_r[cal_mask],float)
    wins=cal_net[cal_net>0]; losses=-cal_net[cal_net<=0]
    avg_win=float(wins.mean()) if len(wins) else max(0.5,float(policy.risk.rr)*0.75)
    avg_loss=float(losses.mean()) if len(losses) else 1.0
    provisional=FittedModel(model,calibrator,cal_month,np.empty(0),np.empty(0),y_cal,base_rate,0.0,None,0.0,avg_win,avg_loss,int(core_mask.sum()),int(cal_mask.sum()))
    calibrated=calibrate(provisional,base_cal)
    router=router_score(x,cal_rows,expert); micro=micro_score(x,cal_rows,expert.direction)
    meta_x=np.column_stack([calibrated,router,micro,calibrated*router,calibrated*micro,router*micro])
    meta=LogisticRegression(C=0.35,solver="lbfgs",max_iter=400,random_state=BASE_SEED+expert.id*17)
    meta.fit(meta_x,y_cal)
    meta_p=meta.predict_proba(meta_x)[:,1]
    blend=float(MODEL["meta_blend_weight"])
    final_p=np.clip((1-blend)*calibrated+blend*meta_p,0.01,0.99)
    utility=final_p*avg_win-(1-final_p)*avg_loss
    penalty=float(MODEL["utility_uncertainty_penalty"])*math.sqrt(max(float(np.mean((calibrated-y_cal)**2)),0.0))
    conservative=utility-penalty
    provisional.calibration_scores=calibrated
    provisional.calibration_utility_scores=conservative
    provisional.calibration_brier=float(np.mean((calibrated-y_cal)**2))
    provisional.meta_model=meta
    provisional.meta_brier=float(np.mean((meta_p-y_cal)**2))
    MODEL_CACHE[cache_key]=provisional
    return provisional

def evaluate_month(x: pd.DataFrame, idx: np.ndarray, expert: SparseExpert, policy: Policy, model: FittedModel, eval_month: str, train_months: set[str]):
    assert_oos_isolation(train_months, f"evaluate {eval_month}")
    risk_key=hashlib.sha256(json.dumps(asdict(policy.risk),sort_keys=True).encode()).hexdigest()[:12]
    cache_key=(id(x),expert.id,risk_key,eval_month,tuple(sorted(train_months)))
    cached=EVAL_CACHE.get(cache_key)
    if cached is None:
        labels,exits,net_r,reasons=outcome_arrays(x,idx,expert,policy.risk)
        months=x["month"].to_numpy()[idx]
        pos=np.flatnonzero((months==eval_month)&(exits>=0))
        features=list(FEATURE_GROUPS[expert.feature_group])
        audit={"raw_candidates":int(len(pos)),"model_available":True,"calibration_month":model.calibration_month,"calibration_rows":model.calibration_rows,"calibration_brier":model.calibration_brier,"meta_brier":model.meta_brier,"avg_win_r":model.avg_win_r,"avg_loss_r":model.avg_loss_r}
        history=list(np.asarray(model.calibration_utility_scores,float)[-int(MODEL["online_rank_window"]):])
        prototypes=[]
        if len(pos)>0 and len(history)>=int(MODEL["online_rank_min_history"]):
            rows=idx[pos]
            base_p=model.base_model.predict_proba(x.iloc[rows][features].to_numpy(float))[:,1]
            p=calibrate(model,base_p);router=router_score(x,rows,expert);micro=micro_score(x,rows,expert.direction)
            meta_x=np.column_stack([p,router,micro,p*router,p*micro,router*micro])
            meta_base=model.meta_model.predict_proba(meta_x)[:,1] if model.meta_model is not None else p
            blend=float(MODEL["meta_blend_weight"]);penalty=float(MODEL["utility_uncertainty_penalty"])*math.sqrt(max(model.calibration_brier,0.0))
            for j,k in enumerate(pos):
                final_p=float(np.clip((1-blend)*p[j]+blend*meta_base[j],0.01,0.99))
                expected=float(final_p*model.avg_win_r-(1-final_p)*model.avg_loss_r)
                conservative=float(expected-penalty)
                hist=np.asarray(history,float);pct=float((np.sum(hist<=conservative)+1)/(len(hist)+1))
                history.append(conservative);history=history[-int(MODEL["online_rank_window"]):]
                severe_router=router[j]<float(MODEL["minimum_router_confidence"])
                hard_reject=(meta_base[j]<float(MODEL["meta_hard_reject_probability"])) or severe_router or (conservative<float(MODEL["hard_negative_utility_r"]))
                if hard_reject: decision="REJECT"
                elif meta_base[j]>=float(MODEL["meta_support_probability"]) and micro[j]>=float(MODEL["meta_support_micro"]): decision="SUPPORT"
                else: decision="NEUTRAL"
                signal_i=int(idx[k]);exit_i=int(exits[k])
                prototypes.append({"signal_i":signal_i,"exit_i":exit_i,"direction":expert.direction,"expert_id":expert.id,"expert":expert.name,"family":expert.family,"setup_group":expert.setup_group,"base_probability":float(base_p[j]),"probability":float(p[j]),"online_percentile":pct,"router":float(router[j]),"micro":float(micro[j]),"meta_probability":float(meta_base[j]),"meta_decision":decision,"expected_utility":expected,"utility":conservative,"net_r":float(net_r[k]),"win":bool(net_r[k]>0),"reason":int(reasons[k]),"day":str(x.index[signal_i].date()),"cycle":int(x.iloc[signal_i][f"sparse_{expert.key}_cycle"])})
        audit={**audit,"calibration_mean":float(np.mean(model.calibration_scores)) if len(model.calibration_scores) else 0.0,"calibration_utility_mean":float(np.mean(model.calibration_utility_scores)) if len(model.calibration_utility_scores) else 0.0,"eval_mean":float(np.mean([p["probability"] for p in prototypes])) if prototypes else 0.0,"eval_utility_mean":float(np.mean([p["utility"] for p in prototypes])) if prototypes else 0.0}
        EVAL_CACHE[cache_key]=(prototypes,audit)
    else:
        prototypes,audit=cached
    ranked=[e for e in prototypes if e["online_percentile"]>=policy.min_percentile]
    non_rejected=[e for e in ranked if e["meta_decision"]!="REJECT"]
    positive=[e for e in non_rejected if e["utility"]>=policy.min_expected_utility_r]
    capped=[];last_exit=-1;day_count={}
    for proto in sorted(positive,key=lambda z:(z["signal_i"],-z["utility"])):
        if len(capped)>=policy.monthly_target:break
        if proto["signal_i"]<=last_exit or day_count.get(proto["day"],0)>=1:continue
        e=dict(proto);e["policy_key"]=policy.key;capped.append(e);day_count[e["day"]]=1;last_exit=e["exit_i"]
    decisions={k:sum(e["meta_decision"]==k for e in ranked) for k in ("SUPPORT","NEUTRAL","REJECT")}
    return capped,{**audit,"after_rank":len(ranked),"meta_support":decisions["SUPPORT"],"meta_neutral":decisions["NEUTRAL"],"meta_reject":decisions["REJECT"],"after_soft_meta":len(non_rejected),"after_positive_utility":len(positive),"after_cap":len(capped)}

def metrics(trades: list[dict[str, Any]]) -> dict[str,float]:
    if not trades:
        return {"trades":0,"wins":0,"win_rate":0.0,"avg_win_R":0.0,"avg_loss_R":0.0,"avg_win_loss_ratio":0.0,"profit_factor":0.0,"net_R":0.0,"max_drawdown_R":0.0,"expectancy_R":0.0}
    r=np.array([float(t["net_r"]) for t in trades]); wins=r[r>0]; losses=-r[r<=0]
    curve=np.cumsum(r); peak=np.maximum.accumulate(np.r_[0.0,curve]); dd=peak[1:]-curve
    avgw=float(wins.mean()) if len(wins) else 0.0; avgl=float(losses.mean()) if len(losses) else 0.0
    return {"trades":int(len(r)),"wins":int(len(wins)),"win_rate":float(len(wins)/len(r)),"avg_win_R":avgw,"avg_loss_R":avgl,"avg_win_loss_ratio":float(avgw/avgl) if avgl>0 else (999.0 if len(wins)>0 else 0.0),"profit_factor":float(wins.sum()/losses.sum()) if losses.sum()>0 else (999.0 if wins.sum()>0 else 0.0),"net_R":float(r.sum()),"max_drawdown_R":float(dd.max()) if len(dd) else 0.0,"expectancy_R":float(r.mean())}


def wilson_lower(wins:int,n:int,z:float=1.0)->float:
    if n<=0:return 0.0
    p=wins/n; d=1+z*z/n
    return float((p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d)


def candidate_summary(candidate: Candidate) -> dict[str,float]:
    all_trades=[t for m in DEVELOPMENT_MONTHS for t in candidate.monthly_events.get(m,[])]
    agg=metrics(all_trades);prior=int(MODEL["stability_prior_trades"])
    shrunk=(agg["wins"]+0.5*prior)/(agg["trades"]+prior) if agg["trades"]+prior else 0.5
    active=sum(candidate.monthly_metrics.get(m,{}).get("trades",0)>0 for m in DEVELOPMENT_MONTHS)
    positive=sum(candidate.monthly_metrics.get(m,{}).get("net_R",0)>0 for m in DEVELOPMENT_MONTHS)
    worst=min((candidate.monthly_metrics.get(m,{}).get("net_R",0.0) for m in DEVELOPMENT_MONTHS),default=0.0)
    positives=[max(0,candidate.monthly_metrics.get(m,{}).get("net_R",0.0)) for m in DEVELOPMENT_MONTHS]
    share=max(positives)/(sum(positives) or 1.0)
    max_month=max((candidate.monthly_metrics.get(m,{}).get("trades",0) for m in DEVELOPMENT_MONTHS),default=0)
    return {**agg,"shrunk_win_rate":float(shrunk),"wilson_lower":wilson_lower(int(agg["wins"]),int(agg["trades"])),"active_months":active,"positive_months":positive,"worst_month_R":float(worst),"max_single_month_profit_share":float(share),"max_trades_single_month":int(max_month)}

def _gate_reasons(summary:dict[str,float],gate:dict[str,Any],level:str)->list[str]:
    reasons=[]
    mapping=[("min_total_trades",summary["trades"]>=int(gate.get("min_total_trades",0)),"总交易不足"),("min_active_months",summary["active_months"]>=int(gate.get("min_active_months",0)),"活跃月份不足"),("min_positive_months",summary["positive_months"]>=int(gate.get("min_positive_months",0)),"正收益月份不足"),("min_raw_win_rate",summary["win_rate"]>=float(gate.get("min_raw_win_rate",0)),"原始胜率不足"),("min_shrunk_win_rate",summary["shrunk_win_rate"]>=float(gate.get("min_shrunk_win_rate",0)),"收缩后胜率不足"),("min_wilson_lower",summary["wilson_lower"]>=float(gate.get("min_wilson_lower",0)),"胜率可信下界不足"),("min_profit_factor",summary["profit_factor"]>=float(gate.get("min_profit_factor",0)),"盈利因子不足"),("min_avg_win_loss_ratio",summary["avg_win_loss_ratio"]>=float(gate.get("min_avg_win_loss_ratio",0)),"实际盈亏比不足"),("min_net_r",summary["net_R"]>float(gate.get("min_net_r",-1e9)),"累计净R不足"),("max_drawdown_r",summary["max_drawdown_R"]<=float(gate.get("max_drawdown_r",1e9)),"最大回撤过大"),("max_worst_month_loss_r",summary["worst_month_R"]>=-float(gate.get("max_worst_month_loss_r",1e9)),"最差月份亏损过大"),("max_single_month_profit_share",summary["max_single_month_profit_share"]<=float(gate.get("max_single_month_profit_share",1.0)),"利润过度集中"),("max_trades_per_month",summary["max_trades_single_month"]<=int(gate.get("max_trades_per_month",999)),"单月交易过多")]
    for key,ok,msg in mapping:
        if key in gate and not ok:reasons.append(msg)
    if "max_total_trades" in gate and summary["trades"]>int(gate["max_total_trades"]):reasons.append("总交易不再稀疏")
    return reasons

def eligibility(summary: dict[str,float], candidate: Candidate)->tuple[str,list[str]]:
    q=_gate_reasons(summary,QUALIFIED_GATE,"QUALIFIED")
    if not q:return "QUALIFIED",[]
    w=_gate_reasons(summary,WATCH_GATE,"WATCH")
    if not w:return "WATCH",q
    c=_gate_reasons(summary,CANDIDATE_GATE,"CANDIDATE")
    if not c:return "CANDIDATE",w
    return "REJECTED",c

def candidate_score(s:dict[str,float])->float:
    tier_bonus={"QUALIFIED":300.0,"WATCH":180.0,"CANDIDATE":80.0,"REJECTED":0.0}
    return float(7*s["net_R"]+22*s["shrunk_win_rate"]+10*s["wilson_lower"]+2*min(s["profit_factor"],5)+2*min(s["avg_win_loss_ratio"],5)+2*s["active_months"]+1.5*s["positive_months"]-3*s["max_drawdown_R"]-4*s["max_single_month_profit_share"])

def evaluate_candidate(x:pd.DataFrame,expert:SparseExpert,policy:Policy)->Candidate:
    c=Candidate(policy);idx=expert_indices(x,expert)
    for eval_pos in range(DEVELOPMENT_START_INDEX,len(MONTHS)-1):
        month=MONTHS[eval_pos];train=set(MONTHS[:eval_pos]);model=fit_model(x,idx,expert,policy,train)
        if model is None:
            c.monthly_events[month]=[];c.monthly_metrics[month]=metrics([]);c.monthly_audit[month]={"model_available":False,"raw_candidates":int(np.sum(x.iloc[idx]["month"].to_numpy()==month))};continue
        events,audit=evaluate_month(x,idx,expert,policy,model,month,train)
        c.monthly_events[month]=events;c.monthly_metrics[month]=metrics(events);c.monthly_audit[month]=audit
    c.aggregate=candidate_summary(c);c.tier,c.reasons=eligibility(c.aggregate,c);c.eligible=c.tier=="QUALIFIED";c.score=candidate_score(c.aggregate)+{"QUALIFIED":300,"WATCH":180,"CANDIDATE":80,"REJECTED":0}[c.tier]
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


def _select_compatible(pool:list[Candidate],max_experts:int)->list[Candidate]:
    selected=[]
    for c in sorted(pool,key=lambda z:z.score,reverse=True):
        if len(selected)>=max_experts:break
        if all(compatible(c,s) for s in selected):selected.append(c)
    return selected

def select_experts(best_by_expert:dict[int,Candidate])->tuple[list[Candidate],list[Candidate],str]:
    qualified=[c for c in best_by_expert.values() if c.tier=="QUALIFIED"]
    watch=[c for c in best_by_expert.values() if c.tier=="WATCH"]
    candidate=[c for c in best_by_expert.values() if c.tier=="CANDIDATE"]
    qualified_selected=_select_compatible(qualified,int(PORT["max_experts"]))
    research_pool=qualified+watch
    if len(research_pool)<int(PORT["research_min_experts"]):
        research_pool+=candidate
    research_selected=_select_compatible(research_pool,int(PORT["max_experts"]))
    if len(qualified_selected)>=int(PORT["qualified_min_experts"]):status="QUALIFIED_EXPERT_POOL_AVAILABLE"
    elif research_selected:status="RESEARCH_EXPERT_POOL_ONLY"
    else:status="ZERO_POSITIVE_SPARSE_EXPERTS"
    return research_selected,qualified_selected,status

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


def detailed_frame(x:pd.DataFrame,trades:list[dict[str,Any]],month:str,portfolio_type:str="research")->pd.DataFrame:
    rows=[];reason={1:"TP",2:"PROTECTED_STOP",3:"TIME_EXIT",4:"EARLY_CUT"}
    for t in trades:
        si=t["signal_i"];ei=si+1;xi=t["exit_i"]
        rows.append({"portfolio_type":portfolio_type,"signal_time_utc":x.index[si].isoformat(),"entry_time_utc":x.index[ei].isoformat(),"exit_time_utc":x.index[xi].isoformat(),"month":month,"direction":"LONG" if t["direction"]>0 else "SHORT","expert":t["expert"],"family":t["family"],"setup_group":t["setup_group"],"source_experts":"|".join(t.get("source_experts",[t["expert"]])),"agreement_count":t.get("agreement_count",1),"policy_key":t["policy_key"],"calibrated_probability":t["probability"],"online_utility_percentile":t["online_percentile"],"meta_probability":t["meta_probability"],"meta_decision":t.get("meta_decision","NEUTRAL"),"expected_utility_R":t.get("expected_utility",t["utility"]),"conservative_utility_R":t["utility"],"net_R":t["net_r"],"win":t["win"],"exit_reason":reason.get(t["reason"],"UNKNOWN"),"bars":xi-ei+1})
    return pd.DataFrame(rows)

def synthetic_smoke() -> None:
    raw,eth,premium,funding=base.synthetic_inputs(30000)
    x,_=base.add_features(raw,eth,premium,funding);x=add_sparse_masks(x)
    assert len(x)>10000
    counts={e.name:len(expert_indices(x,e)) for e in EXPERTS}
    assert len(counts)==32 and any(v>0 for v in counts.values())
    sample={"net_r":1.5,"win":True,"signal_i":10,"exit_i":15,"direction":1,"expert_id":0,"expert":EXPERTS[0].name,"family":EXPERTS[0].family,"setup_group":"x","policy_key":"x","utility":0.5,"expected_utility":0.55,"day":"2026-05-01","probability":0.6,"online_percentile":0.9,"meta_probability":0.5,"meta_decision":"NEUTRAL"}
    trades,dedup,conflicts=combine_month({0:[sample]});assert len(trades)==1
    fake=Candidate(policy_grid(EXPERTS[0])[0]);fake.aggregate={"trades":12,"wins":8,"win_rate":8/12,"avg_win_R":1.6,"avg_loss_R":0.8,"avg_win_loss_ratio":2.0,"profit_factor":4.0,"net_R":8.0,"max_drawdown_R":1.2,"expectancy_R":0.66,"shrunk_win_rate":0.60,"wilson_lower":0.52,"active_months":5,"positive_months":4,"worst_month_R":-0.8,"max_single_month_profit_share":0.4,"max_trades_single_month":3}
    tier,_=eligibility(fake.aggregate,fake);assert tier=="QUALIFIED"
    print("V961_SELF_TEST_OK",json.dumps({"experts":len(EXPERTS),"nonzero_masks":sum(v>0 for v in counts.values()),"tier":tier},ensure_ascii=False))

def pipeline_smoke() -> None:
    raw,eth,premium,funding=base.synthetic_inputs(120000)
    start=int(pd.Timestamp("2025-01-01",tz="UTC").timestamp()*1000);times=start+(raw["open_time"].to_numpy()-int(raw["open_time"].iloc[0]))
    for frame in (raw,eth,premium):frame["open_time"]=times;frame["close_time"]=times+299999
    funding["calc_time"]=times[::96][:len(funding)]
    x,_=base.add_features(raw,eth,premium,funding);x=add_sparse_masks(x)
    reps=[EXPERTS[1],EXPERTS[12],EXPERTS[22]];candidates=[evaluate_candidate(x,e,policy_grid(e)[0]) for e in reps]
    smoke_dir=ROOT/".v961_pipeline_smoke"
    if smoke_dir.exists():shutil.rmtree(smoke_dir)
    smoke_dir.mkdir();pd.DataFrame([{"expert":EXPERT_BY_ID[c.policy.expert_id].name,"tier":c.tier,**c.aggregate} for c in candidates]).to_csv(smoke_dir/"candidate_summary.csv",index=False)
    (smoke_dir/"status.json").write_text(json.dumps({"representatives":3,"tiers":[c.tier for c in candidates]},ensure_ascii=False,indent=2),encoding="utf-8")
    assert (smoke_dir/"candidate_summary.csv").exists();print("V961_PIPELINE_SMOKE_OK")

def main()->None:
    clear_results()
    raw,audit=base.load_official_data();eth,ea=base.load_auxiliary_kline("ETHUSDT","klines");premium,pa=base.load_auxiliary_kline(SYMBOL,"premiumIndexKlines");funding,fa=base.load_funding_rate()
    audit["auxiliary_sources"]={"eth":ea,"premium":pa,"funding":fa};audit["research_months"]={"all":list(MONTHS),"development":list(DEVELOPMENT_MONTHS),"diagnostic_oos":OOS_MONTH}
    x,align=base.add_features(raw,eth,premium,funding);x=add_sparse_masks(x);audit["alignment"]=align
    all_candidates={};leaderboard=[];monthly_rows=[];funnel=[]
    tier_priority={"QUALIFIED":3,"WATCH":2,"CANDIDATE":1,"REJECTED":0}
    for expert in EXPERTS:
        print(f"SEARCH_SPARSE_EXPERT={expert.id}:{expert.name}:{expert.family}",flush=True)
        candidates=[evaluate_candidate(x,expert,p) for p in policy_grid(expert)]
        candidates.sort(key=lambda c:(tier_priority[c.tier],c.score),reverse=True);all_candidates[expert.id]=candidates
        for rank,c in enumerate(candidates,1):leaderboard.append({"expert_id":expert.id,"expert":expert.name,"family":expert.family,"rank":rank,"policy_key":c.policy.key,"tier":c.tier,"eligible":c.eligible,"reasons":"|".join(c.reasons),"score":c.score,**c.aggregate})
        best=candidates[0]
        for m in DEVELOPMENT_MONTHS:
            monthly_rows.append({"expert":expert.name,"family":expert.family,"policy_key":best.policy.key,"tier":best.tier,"month":m,**best.monthly_metrics[m]})
            funnel.append({"expert":expert.name,"family":expert.family,"policy_key":best.policy.key,"tier":best.tier,"month":m,**best.monthly_audit[m]})
    best_by={eid:cands[0] for eid,cands in all_candidates.items()}
    research_selected,qualified_selected,status=select_experts(best_by)
    def build_dev(selected):
        portfolios={};stats={};dedup=[];conflicts=[]
        for m in DEVELOPMENT_MONTHS:
            tr,dd,cc=combine_month({c.policy.expert_id:c.monthly_events.get(m,[]) for c in selected});portfolios[m]=tr;stats[m]=metrics(tr)
            dedup.extend([dict(r,month=m) for r in dd]);conflicts.extend([dict(r,month=m) for r in cc])
        return portfolios,stats,dedup,conflicts
    research_dev,research_dev_stats,research_dedup,research_conflicts=build_dev(research_selected)
    qualified_dev,qualified_dev_stats,qualified_dedup,qualified_conflicts=build_dev(qualified_selected)
    # Evaluate every non-rejected best expert in diagnostic June; pool selection remains frozen before June.
    oos_events={};oos_audit=[];shadow_oos=[]
    for eid,c in best_by.items():
        if c.tier=="REJECTED":continue
        expert=EXPERT_BY_ID[eid];idx=expert_indices(x,expert);model=fit_model(x,idx,expert,c.policy,set(MONTHS[:-1]))
        if model is None:events=[];faudit={"model_available":False,"raw_candidates":int(np.sum(x.iloc[idx]["month"].to_numpy()==OOS_MONTH))}
        else:events,faudit=evaluate_month(x,idx,expert,c.policy,model,OOS_MONTH,set(MONTHS[:-1]))
        oos_events[eid]=events;oos_audit.append({"expert":expert.name,"family":expert.family,"policy_key":c.policy.key,"tier":c.tier,"month":OOS_MONTH,**faudit});shadow_oos.extend([dict(t,month=OOS_MONTH,tier=c.tier) for t in events])
    research_june,research_june_dd,research_june_cc=combine_month({c.policy.expert_id:oos_events.get(c.policy.expert_id,[]) for c in research_selected})
    qualified_june,qualified_june_dd,qualified_june_cc=combine_month({c.policy.expert_id:oos_events.get(c.policy.expert_id,[]) for c in qualified_selected})
    research_may=research_dev.get("2026-05",[]);qualified_may=qualified_dev.get("2026-05",[])
    research_metrics={"2026-05":metrics(research_may),OOS_MONTH:metrics(research_june)};qualified_metrics={"2026-05":metrics(qualified_may),OOS_MONTH:metrics(qualified_june)}
    overlap_rows=[]
    for a,b in itertools.combinations(best_by.values(),2):
        ov=event_overlap(all_dev_events(a),all_dev_events(b),int(PORT["dedup_window_bars"]));overlap_rows.append({"expert_a":EXPERT_BY_ID[a.policy.expert_id].name,"tier_a":a.tier,"expert_b":EXPERT_BY_ID[b.policy.expert_id].name,"tier_b":b.tier,**ov,"compatible":compatible(a,b)})
    research_ids={c.policy.expert_id for c in research_selected};qualified_ids={c.policy.expert_id for c in qualified_selected}
    selection=[]
    for eid,c in best_by.items():selection.append({"expert_id":eid,"expert":EXPERT_BY_ID[eid].name,"family":EXPERT_BY_ID[eid].family,"tier":c.tier,"selected_research":eid in research_ids,"selected_qualified":eid in qualified_ids,"policy_key":c.policy.key,"selection_status":status,"rejection_reason":"|".join(c.reasons),**c.aggregate})
    selected_payload=lambda selected:{EXPERT_BY_ID[c.policy.expert_id].name:{"expert_id":c.policy.expert_id,"family":EXPERT_BY_ID[c.policy.expert_id].family,"tier":c.tier,"policy_key":c.policy.key,"policy":asdict(c.policy),"development_summary":c.aggregate} for c in selected}
    pd.DataFrame(leaderboard).to_csv(RESULTS/"sparse_expert_leaderboard.csv",index=False);pd.DataFrame(monthly_rows).to_csv(RESULTS/"expert_monthly_stats.csv",index=False);pd.DataFrame(funnel+oos_audit).to_csv(RESULTS/"signal_funnel.csv",index=False);pd.DataFrame(funnel+oos_audit).to_csv(RESULTS/"soft_meta_audit.csv",index=False);pd.DataFrame(overlap_rows).to_csv(RESULTS/"expert_overlap.csv",index=False);pd.DataFrame(selection).to_csv(RESULTS/"expert_tier_audit.csv",index=False);pd.DataFrame(selection).to_csv(RESULTS/"selection_audit.csv",index=False)
    drift=[{"expert":r.get("expert"),"family":r.get("family"),"tier":r.get("tier"),"month":r.get("month"),"calibration_mean":r.get("calibration_mean",0),"eval_mean":r.get("eval_mean",0),"probability_drift":r.get("eval_mean",0)-r.get("calibration_mean",0),"calibration_utility_mean":r.get("calibration_utility_mean",0),"eval_utility_mean":r.get("eval_utility_mean",0),"utility_drift":r.get("eval_utility_mean",0)-r.get("calibration_utility_mean",0)} for r in funnel+oos_audit]
    pd.DataFrame(drift).to_csv(RESULTS/"cross_month_score_drift.csv",index=False)
    all_research_dd=research_dedup+[dict(r,month=OOS_MONTH) for r in research_june_dd];all_research_cc=research_conflicts+[dict(r,month=OOS_MONTH) for r in research_june_cc]
    pd.DataFrame(all_research_dd).to_csv(RESULTS/"signal_deduplication.csv",index=False);pd.DataFrame(all_research_cc).to_csv(RESULTS/"signal_conflicts.csv",index=False)
    research_frame=pd.concat([detailed_frame(x,research_may,"2026-05","research"),detailed_frame(x,research_june,OOS_MONTH,"research")],ignore_index=True);qualified_frame=pd.concat([detailed_frame(x,qualified_may,"2026-05","qualified"),detailed_frame(x,qualified_june,OOS_MONTH,"qualified")],ignore_index=True)
    research_frame.to_csv(RESULTS/"research_portfolio_trades.csv",index=False);qualified_frame.to_csv(RESULTS/"qualified_portfolio_trades.csv",index=False);research_frame.to_csv(RESULTS/"portfolio_trades.csv",index=False);research_frame.to_csv(RESULTS/"trades.csv",index=False)
    shadow=[]
    for eid,c in best_by.items():
        for m in DEVELOPMENT_MONTHS:
            shadow.extend([dict(t,month=m,tier=c.tier) for t in c.monthly_events.get(m,[])])
    shadow.extend(shadow_oos);pd.DataFrame(shadow).to_csv(RESULTS/"expert_shadow_trades.csv",index=False)
    coverage=[]
    for ptype,months_map in (("research",{"2026-05":research_may,OOS_MONTH:research_june}),("qualified",{"2026-05":qualified_may,OOS_MONTH:qualified_june})):
        for m,tr in months_map.items():
            vc=pd.Series([t["expert"] for t in tr]).value_counts() if tr else pd.Series(dtype=int)
            for name,count in vc.items():coverage.append({"portfolio_type":ptype,"month":m,"expert":name,"trades":int(count),"trade_share":float(count/len(tr))})
    pd.DataFrame(coverage).to_csv(RESULTS/"opportunity_coverage.csv",index=False)
    selected_json={"research":selected_payload(research_selected),"qualified":selected_payload(qualified_selected)};(RESULTS/"selected_policy.json").write_text(json.dumps(selected_json,ensure_ascii=False,indent=2,default=str),encoding="utf-8");(RESULTS/"data_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    qualified_pool_available=len(qualified_selected)>=int(PORT["qualified_min_experts"]);research_stage_ok=qualified_pool_available and stage_pass(qualified_metrics["2026-05"],qualified_may) and stage_pass(qualified_metrics[OOS_MONTH],qualified_june);final_ok=qualified_pool_available and final_pass(qualified_metrics["2026-05"]) and final_pass(qualified_metrics[OOS_MONTH])
    tier_counts={tier:sum(c.tier==tier for c in best_by.values()) for tier in ("QUALIFIED","WATCH","CANDIDATE","REJECTED")}
    status_payload={"qualified":False,"research_stage_qualified":research_stage_ok,"final_hard_metrics_passed":final_ok,"not_for_live_trading":True,"fresh_blind_month_required":True,"selection_status":status,"engine":ENGINE_NAME,"architecture":"32 specific sparse experts -> 13 rolling development months -> expert-specific conservative expected-R rank -> soft meta support/neutral/reject -> tiered pool -> separate research and qualified portfolios","tier_counts":tier_counts,"selected_research_expert_count":len(research_selected),"selected_qualified_expert_count":len(qualified_selected),"selected_experts":selected_json,"research_portfolio_monthly_stats":research_metrics,"qualified_portfolio_monthly_stats":qualified_metrics,"development_months":list(DEVELOPMENT_MONTHS),"constraints":{"candidate_gate":CANDIDATE_GATE,"watch_gate":WATCH_GATE,"qualified_gate":QUALIFIED_GATE,"research_stage":STAGE,"final_target":FINAL,"portfolio":PORT,"model":MODEL},"oos_isolation":{"used_for_training":False,"used_for_thresholds":False,"used_for_expert_selection":False,"used_for_portfolio_selection":False,"evaluation_occurs_after_policy_freeze":True},"searched_experts":len(EXPERTS),"searched_policies":sum(len(v) for v in all_candidates.values())}
    (RESULTS/"status.json").write_text(json.dumps(status_payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    rm=research_metrics["2026-05"];rj=research_metrics[OOS_MONTH];qm=qualified_metrics["2026-05"];qj=qualified_metrics[OOS_MONTH]
    report=f"""# BTCUSDT 5分钟 稀疏专家扩容、跨月校准与软Meta过滤 V9.6.1 报告

- 架构：32个具体稀疏专家 → 13个滚动开发月 → 专家独立保守预期R与在线排名 → Meta支持/中性/拒绝 → 候选/观察/正式三级晋级。
- 选择状态：**{status}**。
- 专家等级：正式 {tier_counts['QUALIFIED']}；观察 {tier_counts['WATCH']}；候选 {tier_counts['CANDIDATE']}；淘汰 {tier_counts['REJECTED']}。
- 研究组合选入：{len(research_selected)}；正式组合选入：{len(qualified_selected)}。
- 实盘资格：**不合格**；2026年6月仅作已查看的诊断月，正式资格仍需新的完整盲测月。

## 研究组合

| 月份 | 交易 | 胜率 | 实际盈亏比 | 盈利因子 | 净R | 最大回撤R |
|---|---:|---:|---:|---:|---:|---:|
| 2026-05 | {rm['trades']} | {rm['win_rate']:.2%} | {rm['avg_win_loss_ratio']:.3f} | {rm['profit_factor']:.3f} | {rm['net_R']:.3f} | {rm['max_drawdown_R']:.3f} |
| 2026-06 | {rj['trades']} | {rj['win_rate']:.2%} | {rj['avg_win_loss_ratio']:.3f} | {rj['profit_factor']:.3f} | {rj['net_R']:.3f} | {rj['max_drawdown_R']:.3f} |

## 正式组合（仅正式专家）

| 月份 | 交易 | 胜率 | 实际盈亏比 | 盈利因子 | 净R | 最大回撤R |
|---|---:|---:|---:|---:|---:|---:|
| 2026-05 | {qm['trades']} | {qm['win_rate']:.2%} | {qm['avg_win_loss_ratio']:.3f} | {qm['profit_factor']:.3f} | {qm['net_R']:.3f} | {qm['max_drawdown_R']:.3f} |
| 2026-06 | {qj['trades']} | {qj['win_rate']:.2%} | {qj['avg_win_loss_ratio']:.3f} | {qj['profit_factor']:.3f} | {qj['net_R']:.3f} | {qj['max_drawdown_R']:.3f} |

研究组合允许观察专家积累证据；正式组合绝不使用候选或观察专家。Meta中性信号可以执行，但明显负预期、路由冲突或Meta强烈反对的信号会被拒绝。
"""
    (RESULTS/"report.md").write_text(report,encoding="utf-8");(RESULTS/"run_identity.txt").write_text(f"{ENGINE_NAME}\nmonths={','.join(MONTHS)}\ndevelopment={','.join(DEVELOPMENT_MONTHS)}\noos={OOS_MONTH}\noutput=results_v9_6_1\nselection_status={status}\n",encoding="utf-8")
    print(json.dumps({"tier_counts":tier_counts,"research":research_metrics,"qualified":qualified_metrics},ensure_ascii=False,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--self-test",action="store_true");parser.add_argument("--pipeline-smoke",action="store_true");args=parser.parse_args()
    if args.self_test:synthetic_smoke()
    elif args.pipeline_smoke:pipeline_smoke()
    else:main()
