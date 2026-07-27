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
RESULTS = ROOT / "results_v9_6_3"
RESULTS.mkdir(exist_ok=True)
ENGINE_VERSION = "V9.6.3"
ENGINE_NAME = "BTC 5m evidence-first policy cluster validation pool V9.6.3"
OOS_MONTH = "2026-06"

REQUEST_PATH = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(ROOT / "request.v9_6_3.json")))
REQUEST = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
SYMBOL = str(REQUEST["symbol"]).upper()
INTERVAL = str(REQUEST["interval"]).lower()
EVAL_MONTHS = tuple(str(x) for x in REQUEST["months"])
if EVAL_MONTHS != ("2026-05", "2026-06"):
    raise ValueError("V9.6.3 requires evaluation months 2026-05 and diagnostic OOS 2026-06")
MONTHS = tuple(str(p) for p in pd.period_range("2025-01", "2026-06", freq="M"))
DEVELOPMENT_START_INDEX = 4
DEVELOPMENT_MONTHS = MONTHS[DEVELOPMENT_START_INDEX:-1]
FEE_RATE = float(REQUEST["fee_rate_per_side"])
SLIPPAGE_ABS = float(REQUEST["tick_size"]) * int(REQUEST["slippage_ticks_per_fill"])
if FEE_RATE != 0.0005 or abs(SLIPPAGE_ABS - 0.2) > 1e-12:
    raise ValueError("V9.6.3 fixes one-side fee at 0.050% and slippage at 0.2 USDT per fill")

FINAL = REQUEST["final_target"]
STAGE = REQUEST["research_stage"]

CANDIDATE_GATE = REQUEST["candidate_gate"]
WATCH_GATE = REQUEST["watch_gate"]
QUALIFIED_GATE = REQUEST["qualified_gate"]
PORT = REQUEST["portfolio"]
MODEL = REQUEST["model"]
SEARCH = REQUEST["search"]
BASE_SEED = int(MODEL["base_seed"])
ROBUSTNESS = REQUEST["robustness"]
EVIDENCE = REQUEST["evidence_selection"]

# Load the frozen V9.5.1 data/feature/outcome engine from this package. A private
# compatibility request prevents the base module from consuming V9.6 settings.
compat = ROOT / ".v963_base_request.json"
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
spec = importlib.util.spec_from_file_location("v963_base_engine", ROOT / "_v963_base_engine.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load _v963_base_engine.py")
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
    SparseExpert(32, "cross_premium_revert_short", "BTC强于ETH且溢价回落空头", "跨资产衍生品", -1, "cross", "跨资产溢价回落"),
    SparseExpert(33, "cross_taker_revert_short", "BTC强于ETH且主动卖盘空头", "跨资产订单流", -1, "cross", "跨资产主动卖盘"),
    SparseExpert(34, "cross_session_revert_short", "BTC强于ETH欧美盘修复空头", "跨资产时段", -1, "session", "跨资产欧美修复"),
    SparseExpert(35, "eth_lead_down_short", "ETH先跌BTC滞后补跌空头", "跨资产领先滞后", -1, "cross", "ETH领先下跌"),
    SparseExpert(36, "cross_premium_revert_long", "BTC弱于ETH且溢价回升多头", "跨资产衍生品", 1, "cross", "跨资产溢价回升"),
    SparseExpert(37, "eth_lead_up_long", "ETH先涨BTC滞后补涨多头", "跨资产领先滞后", 1, "cross", "ETH领先上涨"),
    SparseExpert(38, "cross_vwap_fail_short", "跨资产背离后VWAP失守空头", "跨资产VWAP", -1, "cross", "背离VWAP失守"),
    SparseExpert(39, "cross_vwap_reclaim_long", "跨资产背离后VWAP收回多头", "跨资产VWAP", 1, "cross", "背离VWAP收回"),
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
        32: (y["ret12"]>0.003)&(y["eth_ret12"]<y["ret12"]-0.0022)&(y["premium_delta"]<0)&(y["premium_z"]>0.25)&(y["ret3"]<0)&(y["upper_wick"]>0.16),
        33: (y["ret12"]>0.0025)&(y["eth_ret12"]<y["ret12"]-0.0020)&(y["taker_ratio"]<0.49)&(y["taker_z"]<0)&(y["ret3"]<0)&(y["close_loc"]<0.48),
        34: europe_us&(y["ret12"]>0.0025)&(y["eth_ret12"]<y["ret12"]-0.0020)&(y["ret3"]<0)&(y["m15_trend"]<=0)&(y["upper_wick"]>0.16),
        35: (y["eth_ret3"]<-0.002)&(y["ret3"]>y["eth_ret3"]+0.0015)&(y["btc_eth_corr"]>0.20)&(y["ret1"]<0)&(y["close_loc"]<0.48),
        36: (y["ret12"]<-0.003)&(y["eth_ret12"]>y["ret12"]+0.0022)&(y["premium_delta"]>0)&(y["premium_z"]<-0.25)&(y["ret3"]>0)&(y["lower_wick"]>0.16),
        37: (y["eth_ret3"]>0.002)&(y["ret3"]<y["eth_ret3"]-0.0015)&(y["btc_eth_corr"]>0.20)&(y["ret1"]>0)&(y["close_loc"]>0.52),
        38: (y["ret12"]>0.0025)&(y["eth_ret12"]<y["ret12"]-0.0020)&(y["vwap_dev"].shift(1)>0)&(y["vwap_dev"]<=0)&(y["ret3"]<0),
        39: (y["ret12"]<-0.0025)&(y["eth_ret12"]>y["ret12"]+0.0020)&(y["vwap_dev"].shift(1)<0)&(y["vwap_dev"]>=0)&(y["ret3"]>0),
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
    brier_penalty=float(MODEL["utility_uncertainty_penalty"])*math.sqrt(max(float(np.mean((calibrated-y_cal)**2)),0.0))
    sample_penalty=float(MODEL["sample_uncertainty_penalty"])/math.sqrt(max(len(y_cal),1))
    penalty=brier_penalty+sample_penalty
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
        audit={"raw_candidates":int(len(pos)),"model_available":True,"calibration_month":model.calibration_month,"calibration_rows":model.calibration_rows,"calibration_brier":model.calibration_brier,"meta_brier":model.meta_brier,"avg_win_r":model.avg_win_r,"avg_loss_r":model.avg_loss_r,"sample_uncertainty_penalty":float(MODEL["sample_uncertainty_penalty"])/math.sqrt(max(model.calibration_rows,1))}
        history=list(np.asarray(model.calibration_utility_scores,float)[-int(MODEL["online_rank_window"]):])
        prototypes=[]
        if len(pos)>0 and len(history)>=int(MODEL["online_rank_min_history"]):
            rows=idx[pos]
            base_p=model.base_model.predict_proba(x.iloc[rows][features].to_numpy(float))[:,1]
            p=calibrate(model,base_p);router=router_score(x,rows,expert);micro=micro_score(x,rows,expert.direction)
            meta_x=np.column_stack([p,router,micro,p*router,p*micro,router*micro])
            meta_base=model.meta_model.predict_proba(meta_x)[:,1] if model.meta_model is not None else p
            blend=float(MODEL["meta_blend_weight"]);penalty=float(MODEL["utility_uncertainty_penalty"])*math.sqrt(max(model.calibration_brier,0.0))+float(MODEL["sample_uncertainty_penalty"])/math.sqrt(max(model.calibration_rows,1))
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


def _policy_vector(policy: Policy) -> tuple[float, ...]:
    """Numeric vector used only to identify local parameter neighbours."""
    r=policy.risk; m=policy.model
    return (
        float(r.rr),float(r.sl_atr),float(r.min_stop_pct)*1000,float(r.max_hold)/100,
        float(r.breakeven_trigger),float(r.breakeven_lock),float(r.early_bars)/50,float(r.early_cut_r),
        float(m.max_depth),float(m.learning_rate)*100,float(m.max_iter)/100,float(m.l2_regularization)/5,float(m.min_samples_leaf)/20,
        float(policy.monthly_target),float(policy.min_percentile)*10,float(policy.min_expected_utility_r)*20,
    )


def policy_distance(a: Policy, b: Policy) -> float:
    av=np.asarray(_policy_vector(a),dtype=float);bv=np.asarray(_policy_vector(b),dtype=float)
    scale=np.maximum(np.maximum(np.abs(av),np.abs(bv)),0.25)
    return float(np.mean(np.abs(av-bv)/scale))


def _loo_stats(c: Candidate) -> tuple[float,float]:
    active=[m for m in DEVELOPMENT_MONTHS if c.monthly_events.get(m)]
    vals=[]
    for omitted in active:
        vals.append(metrics([t for m in DEVELOPMENT_MONTHS if m!=omitted for t in c.monthly_events.get(m,[])])["net_R"])
    return (float(min(vals)) if vals else 0.0,float(sum(v>0 for v in vals)/len(vals)) if vals else 0.0)


def _signal_consensus(anchor: Candidate, neighbours: list[Candidate]) -> float:
    anchor_events=all_dev_events(anchor)
    peers=[n for n in neighbours if n.policy.key!=anchor.policy.key]
    if not anchor_events or not peers:return 0.0
    window=int(PORT["dedup_window_bars"])
    peer_events=[all_dev_events(p) for p in peers]
    ratios=[]
    for t in anchor_events:
        hits=0
        for events in peer_events:
            if any(abs(int(u["signal_i"])-int(t["signal_i"]))<=window and int(u["direction"])==int(t["direction"]) for u in events):hits+=1
        ratios.append(hits/len(peers))
    return float(np.mean(ratios)) if ratios else 0.0


def parameter_cluster_audit(expert: SparseExpert, anchor: Candidate, candidates: list[Candidate]) -> dict[str,Any]:
    size=max(1,int(EVIDENCE["cluster_size"]))
    nearest=sorted(candidates,key=lambda c:(policy_distance(anchor.policy,c.policy),-c.aggregate.get("trades",0)))[:size]
    positives=[c.aggregate.get("net_R",0)>0 for c in nearest]
    mature=[c.tier in {"WATCH","QUALIFIED"} for c in nearest]
    trades=[float(c.aggregate.get("trades",0)) for c in nearest]
    active=[float(c.aggregate.get("active_months",0)) for c in nearest]
    nets=[float(c.aggregate.get("net_R",0)) for c in nearest]
    consensus=_signal_consensus(anchor,nearest)
    positive_share=float(sum(positives)/len(nearest)) if nearest else 0.0
    mature_share=float(sum(mature)/len(nearest)) if nearest else 0.0
    median_trades=float(np.median(trades)) if trades else 0.0
    median_active=float(np.median(active)) if active else 0.0
    robust=(positive_share>=float(EVIDENCE["cluster_min_positive_share"]) and
            median_trades>=float(EVIDENCE["cluster_min_median_trades"]) and
            consensus>=float(EVIDENCE["cluster_min_signal_consensus"]))
    return {
        "expert_id":expert.id,"expert":expert.name,"family":expert.family,
        "anchor_policy_key":anchor.policy.key,"anchor_tier":anchor.tier,"anchor_trades":anchor.aggregate.get("trades",0),
        "cluster_size":len(nearest),"cluster_policy_keys":"|".join(c.policy.key for c in nearest),
        "cluster_positive_share":positive_share,"cluster_mature_share":mature_share,
        "cluster_median_trades":median_trades,"cluster_median_active_months":median_active,
        "cluster_median_net_R":float(np.median(nets)) if nets else 0.0,"cluster_min_net_R":min(nets) if nets else 0.0,
        "cluster_signal_consensus":consensus,"cluster_robust":bool(robust),
    }


def evidence_key(c: Candidate, cluster: dict[str,Any]) -> tuple[Any,...]:
    s=c.aggregate;tier_rank={"QUALIFIED":3,"WATCH":2,"CANDIDATE":1,"REJECTED":0}[c.tier]
    loo_min,loo_share=_loo_stats(c)
    cap=float(EVIDENCE["evidence_trade_cap"])
    return (
        tier_rank,
        int(bool(cluster.get("cluster_robust",False))),
        min(float(s.get("trades",0)),cap),
        float(s.get("active_months",0)),
        float(s.get("positive_months",0)),
        int(loo_min>0),loo_share,
        float(cluster.get("cluster_positive_share",0)),
        float(cluster.get("cluster_signal_consensus",0)),
        float(s.get("shrunk_win_rate",0)),float(s.get("wilson_lower",0)),
        float(s.get("net_R",0)),-float(s.get("max_drawdown_R",0)),-float(s.get("max_single_month_profit_share",1)),
    )


def select_policy_for_expert(expert: SparseExpert, candidates: list[Candidate]) -> tuple[Candidate,list[dict[str,Any]],dict[str,Any]]:
    """Evidence-first selection. A mature policy cannot be displaced by a prettier low-sample candidate."""
    audits=[parameter_cluster_audit(expert,c,candidates) for c in candidates]
    audit_by_key={a["anchor_policy_key"]:a for a in audits}
    mature=[c for c in candidates if c.tier in {"WATCH","QUALIFIED"}]
    candidate_pool=[c for c in candidates if c.tier=="CANDIDATE"]
    pool=mature or candidate_pool or candidates
    preferred_keys=set(str(k) for k in EVIDENCE.get("preferred_seed_policy_keys",[])) if expert.id==int(EVIDENCE["seed_expert_id"]) else set()
    preferred=[c for c in pool if c.policy.key in preferred_keys]
    chosen=max(preferred or pool,key=lambda c:evidence_key(c,audit_by_key[c.policy.key]))
    # Explicit mature-policy protection: sparse candidates never replace WATCH/QUALIFIED.
    protected=bool(mature and chosen.tier in {"WATCH","QUALIFIED"})
    reason="mature_policy_protected" if protected else ("preferred_seed_anchor" if chosen.policy.key in preferred_keys else "evidence_first")
    meta={"selection_reason":reason,"mature_policy_available":bool(mature),"mature_policy_protected":protected,
          "preferred_seed_anchor_available":bool(preferred),"selected_cluster_robust":bool(audit_by_key[chosen.policy.key]["cluster_robust"])}
    ordered=sorted(candidates,key=lambda c:evidence_key(c,audit_by_key[c.policy.key]),reverse=True)
    if ordered[0].policy.key!=chosen.policy.key:
        ordered=[chosen]+[c for c in ordered if c.policy.key!=chosen.policy.key]
    return chosen,audits,meta


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
    qualified_selected=_select_compatible(qualified,int(PORT["max_experts"]))
    # V9.6.3 invariant: CANDIDATE experts remain pure shadow and can never
    # enter the research portfolio. Research is WATCH + QUALIFIED only.
    research_selected=_select_compatible(qualified+watch,int(PORT["max_experts"]))
    if len(qualified_selected)>=int(PORT["qualified_min_experts"]):status="QUALIFIED_EXPERT_POOL_AVAILABLE"
    elif research_selected:status="WATCH_EXPERT_POOL_AVAILABLE"
    else:status="NO_WATCH_OR_QUALIFIED_EXPERTS"
    return research_selected,qualified_selected,status


def executable_events(candidate: Candidate, events: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """Apply tier-specific live-research gates after expert tier is frozen."""
    if candidate.tier=="QUALIFIED":
        return [e for e in events if e.get("meta_decision")=="SUPPORT" or (
            e.get("meta_decision")=="NEUTRAL" and
            float(e.get("meta_probability",0))>=float(MODEL["qualified_neutral_min_meta"]) and
            float(e.get("utility",-999))>=float(MODEL["qualified_neutral_min_utility_r"])
        )]
    if candidate.tier=="WATCH":
        return [e for e in events if e.get("meta_decision")=="SUPPORT" or (
            e.get("meta_decision")=="NEUTRAL" and
            float(e.get("meta_probability",0))>=float(MODEL["watch_neutral_min_meta"]) and
            float(e.get("utility",-999))>=float(MODEL["watch_neutral_min_utility_r"])
        )]
    return []


def candidate_support_events(candidate: Candidate, events: list[dict[str,Any]]) -> list[dict[str,Any]]:
    if candidate.tier!="CANDIDATE": return []
    return [e for e in events if e.get("meta_decision")=="SUPPORT" and
            float(e.get("meta_probability",0))>=float(MODEL["candidate_support_min_meta"]) and
            float(e.get("utility",-999))>=float(MODEL["candidate_support_min_utility_r"]) and
            float(e.get("online_percentile",0))>=float(MODEL["candidate_support_min_percentile"])]


def robustness_row(expert: SparseExpert, candidates: list[Candidate]) -> dict[str,Any]:
    best=candidates[0]
    active=[m for m in DEVELOPMENT_MONTHS if best.monthly_events.get(m)]
    loo=[]
    for omitted in active:
        trades=[t for m in DEVELOPMENT_MONTHS if m!=omitted for t in best.monthly_events.get(m,[])]
        mm=metrics(trades);loo.append(mm["net_R"])
    neighbors=candidates[:max(int(ROBUSTNESS["minimum_policy_neighbors"]),1)]
    positive=sum(c.aggregate.get("net_R",0)>0 for c in neighbors)
    watch_plus=sum(c.tier in {"WATCH","QUALIFIED"} for c in neighbors)
    return {"expert_id":expert.id,"expert":expert.name,"family":expert.family,"tier":best.tier,
            "policy_key":best.policy.key,"active_months":len(active),
            "leave_one_active_month_out_min_net_R":min(loo) if loo else 0.0,
            "leave_one_active_month_out_positive_share":sum(v>0 for v in loo)/len(loo) if loo else 0.0,
            "neighbor_policies":len(neighbors),"positive_neighbor_share":positive/max(1,len(neighbors)),
            "watch_or_qualified_neighbor_share":watch_plus/max(1,len(neighbors)),
            "seed_expert":expert.id==int(ROBUSTNESS["seed_expert_id"])}

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
    assert len(counts)==40 and any(v>0 for v in counts.values())
    # A 7-trade WATCH must beat a prettier 4-trade CANDIDATE from the same expert.
    policies=policy_grid(EXPERTS[21]);watch=Candidate(next(p for p in policies if p.key=="fca571982b7c88a0"));sparse=Candidate(next(p for p in policies if p.key=="9a19b379445c8be4"))
    base_summary={"wins":5,"avg_win_R":1.8,"avg_loss_R":1.0,"avg_win_loss_ratio":1.8,"profit_factor":3.0,"net_R":6.0,"max_drawdown_R":1.2,"expectancy_R":0.8,"shrunk_win_rate":0.60,"wilson_lower":0.52,"positive_months":3,"worst_month_R":0.0,"max_single_month_profit_share":0.40,"max_trades_single_month":3}
    watch.aggregate={"trades":7,"win_rate":5/7,"active_months":3,**base_summary};watch.tier="WATCH"
    sparse.aggregate={"trades":4,"wins":3,"win_rate":0.75,"avg_win_R":1.9,"avg_loss_R":0.08,"avg_win_loss_ratio":23.0,"profit_factor":60.0,"net_R":5.5,"max_drawdown_R":0.08,"expectancy_R":1.3,"shrunk_win_rate":0.58,"wilson_lower":0.50,"active_months":3,"positive_months":2,"worst_month_R":-0.08,"max_single_month_profit_share":0.66,"max_trades_single_month":2};sparse.tier="CANDIDATE"
    watch.monthly_events={m:[] for m in DEVELOPMENT_MONTHS};sparse.monthly_events={m:[] for m in DEVELOPMENT_MONTHS}
    watch.monthly_events["2025-12"]=[{"signal_i":1,"direction":-1,"net_r":1.8,"win":True}]
    watch.monthly_events["2026-01"]=[{"signal_i":2,"direction":-1,"net_r":1.8,"win":True}]
    watch.monthly_events["2026-04"]=[{"signal_i":3,"direction":-1,"net_r":1.8,"win":True}]
    sparse.monthly_events["2025-12"]=[{"signal_i":1,"direction":-1,"net_r":1.9,"win":True}]
    sparse.monthly_events["2026-04"]=[{"signal_i":3,"direction":-1,"net_r":1.9,"win":True}]
    chosen,_,meta=select_policy_for_expert(EXPERTS[21],[sparse,watch])
    assert chosen.policy.key==watch.policy.key and meta["mature_policy_protected"]
    fake_candidate=Candidate(policy_grid(EXPERTS[1])[0]);fake_candidate.tier="CANDIDATE";fake_candidate.score=999
    research,qualified,status=select_experts({21:watch,1:fake_candidate})
    assert [c.tier for c in research]==["WATCH"] and not qualified and status=="WATCH_EXPERT_POOL_AVAILABLE"
    neutral={"meta_decision":"NEUTRAL","meta_probability":0.46,"utility":0.10,"online_percentile":0.95}
    support={"meta_decision":"SUPPORT","meta_probability":0.62,"utility":0.20,"online_percentile":0.95}
    assert len(executable_events(watch,[neutral]))==1 and not executable_events(fake_candidate,[support]) and len(candidate_support_events(fake_candidate,[support]))==1
    fake=Candidate(policy_grid(EXPERTS[0])[0]);fake.aggregate={"trades":12,"wins":8,"win_rate":8/12,"avg_win_R":1.6,"avg_loss_R":0.8,"avg_win_loss_ratio":2.0,"profit_factor":4.0,"net_R":8.0,"max_drawdown_R":1.2,"expectancy_R":0.66,"shrunk_win_rate":0.60,"wilson_lower":0.52,"active_months":5,"positive_months":4,"worst_month_R":-0.8,"max_single_month_profit_share":0.4,"max_trades_single_month":3}
    tier,_=eligibility(fake.aggregate,fake);assert tier=="QUALIFIED"
    print("V963_SELF_TEST_OK",json.dumps({"experts":len(EXPERTS),"nonzero_masks":sum(v>0 for v in counts.values()),"evidence_first":True,"mature_protected":True,"tier":tier},ensure_ascii=False))


def pipeline_smoke() -> None:
    raw,eth,premium,funding=base.synthetic_inputs(120000)
    start=int(pd.Timestamp("2025-01-01",tz="UTC").timestamp()*1000);times=start+(raw["open_time"].to_numpy()-int(raw["open_time"].iloc[0]))
    for frame in (raw,eth,premium):frame["open_time"]=times;frame["close_time"]=times+299999
    funding["calc_time"]=times[::96][:len(funding)]
    x,_=base.add_features(raw,eth,premium,funding);x=add_sparse_masks(x)
    reps=[EXPERTS[1],EXPERTS[21],EXPERTS[32],EXPERTS[39]];candidates=[evaluate_candidate(x,e,policy_grid(e)[0]) for e in reps]
    smoke_dir=ROOT/".v963_pipeline_smoke"
    if smoke_dir.exists():shutil.rmtree(smoke_dir)
    smoke_dir.mkdir();pd.DataFrame([{"expert":EXPERT_BY_ID[c.policy.expert_id].name,"tier":c.tier,**c.aggregate} for c in candidates]).to_csv(smoke_dir/"candidate_summary.csv",index=False)
    (smoke_dir/"status.json").write_text(json.dumps({"representatives":4,"tiers":[c.tier for c in candidates]},ensure_ascii=False,indent=2),encoding="utf-8")
    assert (smoke_dir/"candidate_summary.csv").exists();print("V963_PIPELINE_SMOKE_OK")


def _concat_frames(frames:list[pd.DataFrame])->pd.DataFrame:
    nonempty=[f for f in frames if f is not None and (len(f)>0 or len(f.columns)>0)]
    return pd.concat(nonempty,ignore_index=True) if nonempty else pd.DataFrame()


def main()->None:
    clear_results()
    raw,audit=base.load_official_data();eth,ea=base.load_auxiliary_kline("ETHUSDT","klines");premium,pa=base.load_auxiliary_kline(SYMBOL,"premiumIndexKlines");funding,fa=base.load_funding_rate()
    audit["auxiliary_sources"]={"eth":ea,"premium":pa,"funding":fa};audit["research_months"]={"all":list(MONTHS),"development":list(DEVELOPMENT_MONTHS),"diagnostic_oos":OOS_MONTH}
    x,align=base.add_features(raw,eth,premium,funding);x=add_sparse_masks(x);audit["alignment"]=align
    all_candidates={};best_by={};leaderboard=[];monthly_rows=[];funnel=[];cluster_rows=[];selection_comparison=[]
    selection_meta={}
    for expert in EXPERTS:
        print(f"SEARCH_SPARSE_EXPERT={expert.id}:{expert.name}:{expert.family}",flush=True)
        raw_candidates=[evaluate_candidate(x,expert,p) for p in policy_grid(expert)]
        chosen,audits,meta=select_policy_for_expert(expert,raw_candidates)
        audit_by_key={a["anchor_policy_key"]:a for a in audits}
        candidates=sorted(raw_candidates,key=lambda c:evidence_key(c,audit_by_key[c.policy.key]),reverse=True)
        if candidates[0].policy.key!=chosen.policy.key:candidates=[chosen]+[c for c in candidates if c.policy.key!=chosen.policy.key]
        all_candidates[expert.id]=candidates;best_by[expert.id]=chosen;selection_meta[expert.id]=meta;cluster_rows.extend(audits)
        for rank,c in enumerate(candidates,1):
            ca=audit_by_key[c.policy.key]
            leaderboard.append({"expert_id":expert.id,"expert":expert.name,"family":expert.family,"rank":rank,"selected":c.policy.key==chosen.policy.key,"policy_key":c.policy.key,"tier":c.tier,"eligible":c.eligible,"reasons":"|".join(c.reasons),"score":c.score,"cluster_robust":ca["cluster_robust"],"cluster_positive_share":ca["cluster_positive_share"],"cluster_signal_consensus":ca["cluster_signal_consensus"],**c.aggregate})
        score_winner=max(raw_candidates,key=lambda c:c.score)
        selection_comparison.append({"expert_id":expert.id,"expert":expert.name,"family":expert.family,"selected_policy_key":chosen.policy.key,"selected_tier":chosen.tier,"selected_trades":chosen.aggregate.get("trades",0),"selected_active_months":chosen.aggregate.get("active_months",0),"legacy_score_winner_key":score_winner.policy.key,"legacy_score_winner_tier":score_winner.tier,"legacy_score_winner_trades":score_winner.aggregate.get("trades",0),"policy_changed_by_evidence_rule":chosen.policy.key!=score_winner.policy.key,**meta})
        for m in DEVELOPMENT_MONTHS:
            monthly_rows.append({"expert":expert.name,"family":expert.family,"policy_key":chosen.policy.key,"tier":chosen.tier,"month":m,**chosen.monthly_metrics[m]})
            funnel.append({"expert":expert.name,"family":expert.family,"policy_key":chosen.policy.key,"tier":chosen.tier,"month":m,**chosen.monthly_audit[m]})
    research_selected,qualified_selected,status=select_experts(best_by)

    def build_dev(selected):
        portfolios={};stats={};dedup=[];conflicts=[]
        for m in DEVELOPMENT_MONTHS:
            source={c.policy.expert_id:executable_events(c,c.monthly_events.get(m,[])) for c in selected}
            tr,dd,cc=combine_month(source);portfolios[m]=tr;stats[m]=metrics(tr)
            dedup.extend([dict(r,month=m) for r in dd]);conflicts.extend([dict(r,month=m) for r in cc])
        return portfolios,stats,dedup,conflicts

    research_dev,research_dev_stats,research_dedup,research_conflicts=build_dev(research_selected)
    qualified_dev,qualified_dev_stats,qualified_dedup,qualified_conflicts=build_dev(qualified_selected)

    # Candidate SUPPORT-only records are experimental and never feed either portfolio.
    candidate_support_dev=[]
    for c in best_by.values():
        if c.tier!="CANDIDATE":continue
        for m in DEVELOPMENT_MONTHS:
            candidate_support_dev.extend([dict(t,month=m,tier=c.tier) for t in candidate_support_events(c,c.monthly_events.get(m,[]))])

    # Evaluate every non-rejected best expert in diagnostic June; selection is frozen before June.
    oos_events={};oos_audit=[];shadow_oos=[]
    for eid,c in best_by.items():
        if c.tier=="REJECTED":continue
        expert=EXPERT_BY_ID[eid];idx=expert_indices(x,expert);model=fit_model(x,idx,expert,c.policy,set(MONTHS[:-1]))
        if model is None:events=[];faudit={"model_available":False,"raw_candidates":int(np.sum(x.iloc[idx]["month"].to_numpy()==OOS_MONTH))}
        else:events,faudit=evaluate_month(x,idx,expert,c.policy,model,OOS_MONTH,set(MONTHS[:-1]))
        oos_events[eid]=events;oos_audit.append({"expert":expert.name,"family":expert.family,"policy_key":c.policy.key,"tier":c.tier,"month":OOS_MONTH,**faudit});shadow_oos.extend([dict(t,month=OOS_MONTH,tier=c.tier) for t in events])
    research_june,research_june_dd,research_june_cc=combine_month({c.policy.expert_id:executable_events(c,oos_events.get(c.policy.expert_id,[])) for c in research_selected})
    qualified_june,qualified_june_dd,qualified_june_cc=combine_month({c.policy.expert_id:executable_events(c,oos_events.get(c.policy.expert_id,[])) for c in qualified_selected})
    candidate_support_oos=[]
    for c in best_by.values():candidate_support_oos.extend([dict(t,month=OOS_MONTH,tier=c.tier) for t in candidate_support_events(c,oos_events.get(c.policy.expert_id,[]))])

    research_may=research_dev.get("2026-05",[]);qualified_may=qualified_dev.get("2026-05",[])
    research_metrics={"2026-05":metrics(research_may),OOS_MONTH:metrics(research_june)};qualified_metrics={"2026-05":metrics(qualified_may),OOS_MONTH:metrics(qualified_june)}
    overlap_rows=[]
    for a,b in itertools.combinations(best_by.values(),2):
        ov=event_overlap(all_dev_events(a),all_dev_events(b),int(PORT["dedup_window_bars"]));overlap_rows.append({"expert_a":EXPERT_BY_ID[a.policy.expert_id].name,"tier_a":a.tier,"expert_b":EXPERT_BY_ID[b.policy.expert_id].name,"tier_b":b.tier,**ov,"compatible":compatible(a,b)})
    research_ids={c.policy.expert_id for c in research_selected};qualified_ids={c.policy.expert_id for c in qualified_selected}
    selection=[]
    for eid,c in best_by.items():selection.append({"expert_id":eid,"expert":EXPERT_BY_ID[eid].name,"family":EXPERT_BY_ID[eid].family,"tier":c.tier,"selected_research":eid in research_ids,"selected_qualified":eid in qualified_ids,"candidate_shadow_only":c.tier=="CANDIDATE","policy_key":c.policy.key,"selection_status":status,"selection_reason":selection_meta[eid]["selection_reason"],"mature_policy_available":selection_meta[eid]["mature_policy_available"],"mature_policy_protected":selection_meta[eid]["mature_policy_protected"],"selected_cluster_robust":selection_meta[eid]["selected_cluster_robust"],"rejection_reason":"|".join(c.reasons),**c.aggregate})
    robustness=[robustness_row(EXPERT_BY_ID[eid],cands) for eid,cands in all_candidates.items()]
    selected_payload=lambda selected:{EXPERT_BY_ID[c.policy.expert_id].name:{"expert_id":c.policy.expert_id,"family":EXPERT_BY_ID[c.policy.expert_id].family,"tier":c.tier,"policy_key":c.policy.key,"policy":asdict(c.policy),"development_summary":c.aggregate} for c in selected}

    pd.DataFrame(leaderboard).to_csv(RESULTS/"sparse_expert_leaderboard.csv",index=False);pd.DataFrame(monthly_rows).to_csv(RESULTS/"expert_monthly_stats.csv",index=False);pd.DataFrame(funnel+oos_audit).to_csv(RESULTS/"signal_funnel.csv",index=False);pd.DataFrame(funnel+oos_audit).to_csv(RESULTS/"soft_meta_audit.csv",index=False);pd.DataFrame(overlap_rows).to_csv(RESULTS/"expert_overlap.csv",index=False);pd.DataFrame(selection).to_csv(RESULTS/"expert_tier_audit.csv",index=False);pd.DataFrame(selection).to_csv(RESULTS/"selection_audit.csv",index=False);pd.DataFrame(robustness).to_csv(RESULTS/"seed_robustness_audit.csv",index=False)
    pd.DataFrame(cluster_rows).to_csv(RESULTS/"parameter_cluster_audit.csv",index=False);pd.DataFrame(selection_comparison).to_csv(RESULTS/"policy_selection_comparison.csv",index=False)
    seed_keys=set(str(k) for k in EVIDENCE.get("preferred_seed_policy_keys",[]));seed_rows=[r for r in leaderboard if r["expert_id"]==int(EVIDENCE["seed_expert_id"]) and r["policy_key"] in seed_keys];pd.DataFrame(seed_rows).to_csv(RESULTS/"seed_policy_comparison.csv",index=False)
    retention=[r for r in selection_comparison if r["mature_policy_available"] or r["expert_id"]==int(EVIDENCE["seed_expert_id"])];pd.DataFrame(retention).to_csv(RESULTS/"historical_policy_retention_audit.csv",index=False)
    drift=[{"expert":r.get("expert"),"family":r.get("family"),"tier":r.get("tier"),"month":r.get("month"),"calibration_mean":r.get("calibration_mean",0),"eval_mean":r.get("eval_mean",0),"probability_drift":r.get("eval_mean",0)-r.get("calibration_mean",0),"calibration_utility_mean":r.get("calibration_utility_mean",0),"eval_utility_mean":r.get("eval_utility_mean",0),"utility_drift":r.get("eval_utility_mean",0)-r.get("calibration_utility_mean",0)} for r in funnel+oos_audit]
    pd.DataFrame(drift).to_csv(RESULTS/"cross_month_score_drift.csv",index=False)
    all_research_dd=research_dedup+[dict(r,month=OOS_MONTH) for r in research_june_dd];all_research_cc=research_conflicts+[dict(r,month=OOS_MONTH) for r in research_june_cc]
    pd.DataFrame(all_research_dd).to_csv(RESULTS/"signal_deduplication.csv",index=False);pd.DataFrame(all_research_cc).to_csv(RESULTS/"signal_conflicts.csv",index=False)
    research_frame=_concat_frames([detailed_frame(x,research_may,"2026-05","watch_research"),detailed_frame(x,research_june,OOS_MONTH,"watch_research")]);qualified_frame=_concat_frames([detailed_frame(x,qualified_may,"2026-05","qualified"),detailed_frame(x,qualified_june,OOS_MONTH,"qualified")])
    candidate_support_frame=_concat_frames([detailed_frame(x,[t for t in candidate_support_dev if t["month"]=="2026-05"],"2026-05","candidate_support_experimental"),detailed_frame(x,candidate_support_oos,OOS_MONTH,"candidate_support_experimental")])
    research_frame.to_csv(RESULTS/"watch_portfolio_trades.csv",index=False);research_frame.to_csv(RESULTS/"research_portfolio_trades.csv",index=False);qualified_frame.to_csv(RESULTS/"qualified_portfolio_trades.csv",index=False);candidate_support_frame.to_csv(RESULTS/"candidate_support_only_trades.csv",index=False);research_frame.to_csv(RESULTS/"portfolio_trades.csv",index=False);research_frame.to_csv(RESULTS/"trades.csv",index=False)
    shadow=[]
    for eid,c in best_by.items():
        for m in DEVELOPMENT_MONTHS:shadow.extend([dict(t,month=m,tier=c.tier) for t in c.monthly_events.get(m,[])])
    shadow.extend(shadow_oos);pd.DataFrame(shadow).to_csv(RESULTS/"expert_shadow_trades.csv",index=False)
    candidate_shadow=[t for t in shadow if t.get("tier")=="CANDIDATE"];pd.DataFrame(candidate_shadow).to_csv(RESULTS/"candidate_shadow_trades.csv",index=False)
    coverage=[]
    for ptype,months_map in (("watch_research",{"2026-05":research_may,OOS_MONTH:research_june}),("qualified",{"2026-05":qualified_may,OOS_MONTH:qualified_june})):
        for m,tr in months_map.items():
            vc=pd.Series([t["expert"] for t in tr]).value_counts() if tr else pd.Series(dtype=int)
            for name,count in vc.items():coverage.append({"portfolio_type":ptype,"month":m,"expert":name,"trades":int(count),"trade_share":float(count/len(tr))})
    pd.DataFrame(coverage).to_csv(RESULTS/"opportunity_coverage.csv",index=False)
    selected_json={"research_watch_and_qualified_only":selected_payload(research_selected),"qualified":selected_payload(qualified_selected)};(RESULTS/"selected_policy.json").write_text(json.dumps(selected_json,ensure_ascii=False,indent=2,default=str),encoding="utf-8");(RESULTS/"data_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    qualified_pool_available=len(qualified_selected)>=int(PORT["qualified_min_experts"]);research_stage_ok=qualified_pool_available and stage_pass(qualified_metrics["2026-05"],qualified_may) and stage_pass(qualified_metrics[OOS_MONTH],qualified_june);final_ok=qualified_pool_available and final_pass(qualified_metrics["2026-05"]) and final_pass(qualified_metrics[OOS_MONTH])
    tier_counts={tier:sum(c.tier==tier for c in best_by.values()) for tier in ("QUALIFIED","WATCH","CANDIDATE","REJECTED")}
    support_metrics={"2026-05":metrics([t for t in candidate_support_dev if t["month"]=="2026-05"]),OOS_MONTH:metrics(candidate_support_oos)}
    status_payload={"qualified":False,"research_stage_qualified":research_stage_ok,"final_hard_metrics_passed":final_ok,"not_for_live_trading":True,"fresh_blind_month_required":True,"selection_status":status,"engine":ENGINE_NAME,"architecture":"40 sparse experts -> evidence-first policy selection -> local parameter clusters -> mature-policy retention -> candidate pure shadow -> WATCH/QUALIFIED research execution","tier_counts":tier_counts,"selected_research_expert_count":len(research_selected),"selected_qualified_expert_count":len(qualified_selected),"candidate_experts_excluded_from_research":True,"evidence_first_policy_selection":True,"mature_policy_protected_count":sum(bool(v["mature_policy_protected"]) for v in selection_meta.values()),"cluster_robust_selected_expert_count":sum(bool(v["selected_cluster_robust"]) for v in selection_meta.values()),"seed_selected_policy_key":best_by[int(EVIDENCE["seed_expert_id"])].policy.key,"seed_preferred_anchor_selected":best_by[int(EVIDENCE["seed_expert_id"])].policy.key in set(EVIDENCE.get("preferred_seed_policy_keys",[])),"selected_experts":selected_json,"watch_portfolio_monthly_stats":research_metrics,"research_portfolio_monthly_stats":research_metrics,"qualified_portfolio_monthly_stats":qualified_metrics,"candidate_support_only_monthly_stats":support_metrics,"development_months":list(DEVELOPMENT_MONTHS),"constraints":{"candidate_gate":CANDIDATE_GATE,"watch_gate":WATCH_GATE,"qualified_gate":QUALIFIED_GATE,"research_stage":STAGE,"final_target":FINAL,"portfolio":PORT,"model":MODEL,"robustness":ROBUSTNESS,"evidence_selection":EVIDENCE},"oos_isolation":{"used_for_training":False,"used_for_thresholds":False,"used_for_expert_selection":False,"used_for_portfolio_selection":False,"evaluation_occurs_after_policy_freeze":True},"searched_experts":len(EXPERTS),"searched_policies":sum(len(v) for v in all_candidates.values())}
    (RESULTS/"status.json").write_text(json.dumps(status_payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    rm=research_metrics["2026-05"];rj=research_metrics[OOS_MONTH];qm=qualified_metrics["2026-05"];qj=qualified_metrics[OOS_MONTH];cm=support_metrics["2026-05"];cj=support_metrics[OOS_MONTH]
    report=f"""# BTCUSDT 5分钟 证据优先参数选择与专家簇验证 V9.6.3 报告

- 架构：40个稀疏专家 → 13个滚动开发月 → 证据优先参数选择 → 局部参数簇验证 → 成熟参数保护 → 候选纯影子 → WATCH/QUALIFIED执行。
- 选择状态：**{status}**。
- 专家等级：正式 {tier_counts['QUALIFIED']}；观察 {tier_counts['WATCH']}；候选 {tier_counts['CANDIDATE']}；淘汰 {tier_counts['REJECTED']}。
- WATCH/正式研究组合选入：{len(research_selected)}；正式组合选入：{len(qualified_selected)}。
- 候选专家已从研究组合完全排除；候选SUPPORT交易只写入独立实验文件。
- 参数选择：优先保留QUALIFIED/WATCH，再比较交易样本、活跃月份、正收益月份、留一月稳健性和参数簇一致性。
- 种子专家选中参数：`{best_by[int(EVIDENCE["seed_expert_id"])].policy.key}`；历史锚点是否保留：`{best_by[int(EVIDENCE["seed_expert_id"])].policy.key in set(EVIDENCE.get("preferred_seed_policy_keys",[]))}`。
- 实盘资格：**不合格**；2026年6月仅作已查看诊断月，仍需新的完整盲测月。

## WATCH/QUALIFIED研究组合

| 月份 | 交易 | 胜率 | 实际盈亏比 | 盈利因子 | 净R | 最大回撤R |
|---|---:|---:|---:|---:|---:|---:|
| 2026-05 | {rm['trades']} | {rm['win_rate']:.2%} | {rm['avg_win_loss_ratio']:.3f} | {rm['profit_factor']:.3f} | {rm['net_R']:.3f} | {rm['max_drawdown_R']:.3f} |
| 2026-06 | {rj['trades']} | {rj['win_rate']:.2%} | {rj['avg_win_loss_ratio']:.3f} | {rj['profit_factor']:.3f} | {rj['net_R']:.3f} | {rj['max_drawdown_R']:.3f} |

## 候选SUPPORT-only实验（不计入组合）

| 月份 | 交易 | 胜率 | 净R | 最大回撤R |
|---|---:|---:|---:|---:|
| 2026-05 | {cm['trades']} | {cm['win_rate']:.2%} | {cm['net_R']:.3f} | {cm['max_drawdown_R']:.3f} |
| 2026-06 | {cj['trades']} | {cj['win_rate']:.2%} | {cj['net_R']:.3f} | {cj['max_drawdown_R']:.3f} |

## 正式组合（仅QUALIFIED）

| 月份 | 交易 | 胜率 | 实际盈亏比 | 盈利因子 | 净R | 最大回撤R |
|---|---:|---:|---:|---:|---:|---:|
| 2026-05 | {qm['trades']} | {qm['win_rate']:.2%} | {qm['avg_win_loss_ratio']:.3f} | {qm['profit_factor']:.3f} | {qm['net_R']:.3f} | {qm['max_drawdown_R']:.3f} |
| 2026-06 | {qj['trades']} | {qj['win_rate']:.2%} | {qj['avg_win_loss_ratio']:.3f} | {qj['profit_factor']:.3f} | {qj['net_R']:.3f} | {qj['max_drawdown_R']:.3f} |

V9.6.3的主要目标是防止低样本漂亮参数覆盖更有证据的成熟参数，并验证专家优势是否存在于一组邻近参数中。候选专家仍不进入组合。
"""
    (RESULTS/"report.md").write_text(report,encoding="utf-8");(RESULTS/"run_identity.txt").write_text(f"{ENGINE_NAME}\nmonths={','.join(MONTHS)}\ndevelopment={','.join(DEVELOPMENT_MONTHS)}\noos={OOS_MONTH}\noutput=results_v9_6_3\nselection_status={status}\n",encoding="utf-8")
    print(json.dumps({"tier_counts":tier_counts,"watch_research":research_metrics,"candidate_support_only":support_metrics,"qualified":qualified_metrics},ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--self-test",action="store_true");parser.add_argument("--pipeline-smoke",action="store_true");args=parser.parse_args()
    if args.self_test:synthetic_smoke()
    elif args.pipeline_smoke:pipeline_smoke()
    else:main()
