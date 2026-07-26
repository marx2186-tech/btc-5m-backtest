from __future__ import annotations

import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from numba import njit
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:  # GitHub runner fallback; keeps deployment to one file.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.5.1"])
    from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
RESULTS = ROOT / "results_v6_1"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

ENGINE_VERSION = "V6.1"
ENGINE_NAME = "BTC 5m regime walk-forward ensemble V6.1"

DEFAULT_REQUEST: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "interval": "5m",
    "months": ["2026-05", "2026-06"],
    "fee_rate_per_side": 0.0005,
    "tick_size": 0.1,
    "slippage_ticks_per_fill": 2,
    "min_trades_per_month": 15,
    "max_trades_per_month": 30,
    "min_win_rate": 0.70,
    "min_avg_win_loss_ratio": 1.50,
    "base_seed": 20260801,
}


def load_request() -> dict[str, Any]:
    req = dict(DEFAULT_REQUEST)
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(ROOT / "request.v6_1.json")))
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("request.v6_1.json must contain a JSON object")
        req.update(loaded)
    return req


REQUEST = load_request()
SYMBOL = str(REQUEST["symbol"]).upper()
INTERVAL = str(REQUEST["interval"]).lower()
EVAL_MONTHS = tuple(str(x) for x in REQUEST["months"])
if len(EVAL_MONTHS) != 2:
    raise ValueError("V6 requires exactly two evaluation months")
first_eval = pd.Period(EVAL_MONTHS[0], freq="M")
last_eval = pd.Period(EVAL_MONTHS[1], freq="M")
if last_eval != first_eval + 1:
    raise ValueError("Evaluation months must be consecutive")
MONTHS = tuple(str(first_eval - offset) for offset in range(8, 0, -1)) + EVAL_MONTHS
FEE_RATE = float(REQUEST["fee_rate_per_side"])
TICK_SIZE = float(REQUEST["tick_size"])
SLIPPAGE_ABS = TICK_SIZE * int(REQUEST["slippage_ticks_per_fill"])
MIN_TRADES = int(REQUEST["min_trades_per_month"])
MAX_TRADES = int(REQUEST["max_trades_per_month"])
MIN_WIN_RATE = float(REQUEST["min_win_rate"])
MIN_RATIO = float(REQUEST["min_avg_win_loss_ratio"])
BASE_SEED = int(REQUEST["base_seed"])

COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote", "ignore",
]


def interval_to_ms(interval: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    value, unit = int(interval[:-1]), interval[-1]
    if unit not in units:
        raise ValueError(f"Unsupported interval: {interval}")
    return value * units[unit]


STEP_MS = interval_to_ms(INTERVAL)


def download(url: str, path: Path, attempts: int = 6) -> bytes:
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=90, headers={"User-Agent": "btc-regime-walkforward-v6/1.0"})
            response.raise_for_status()
            path.write_bytes(response.content)
            return response.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** min(attempt, 4))
    raise RuntimeError(f"Download failed: {url}: {last}")


def load_official_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    files: list[dict[str, Any]] = []
    for month in MONTHS:
        name = f"{SYMBOL}-{INTERVAL}-{month}.zip"
        base = f"https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{INTERVAL}"
        raw = download(f"{base}/{name}", CACHE / name)
        checksum = download(f"{base}/{name}.CHECKSUM", CACHE / f"{name}.CHECKSUM").decode("utf-8").strip()
        expected_hash = checksum.split()[0].lower()
        actual_hash = hashlib.sha256(raw).hexdigest().lower()
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch for {name}")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
            if len(members) != 1:
                raise RuntimeError(f"Unexpected ZIP contents for {name}: {members}")
            content = archive.read(members[0])
        first = content.splitlines()[0].decode("utf-8", errors="ignore").lower()
        has_header = "open_time" in first or "open time" in first
        frame = pd.read_csv(io.BytesIO(content), header=0 if has_header else None).iloc[:, :12]
        frame.columns = COLS
        frames.append(frame)
        files.append({"file": name, "sha256": actual_hash, "rows": int(len(frame))})

    data = pd.concat(frames, ignore_index=True)
    for col in [c for c in COLS if c != "ignore"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.sort_values("open_time").reset_index(drop=True)

    start = pd.Timestamp(f"{MONTHS[0]}-01T00:00:00Z")
    end = pd.Timestamp(f"{MONTHS[-1]}-01T00:00:00Z") + pd.offsets.MonthBegin(1)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    expected_times = np.arange(start_ms, end_ms, STEP_MS, dtype=np.int64)
    times = data["open_time"].astype("int64").to_numpy()
    unique_times = np.unique(times)
    duplicated = int(pd.Series(times).duplicated().sum())
    missing = np.setdiff1d(expected_times, unique_times)
    extra = np.setdiff1d(unique_times, expected_times)
    off_grid = int(np.sum((times - start_ms) % STEP_MS != 0))
    bad_close = int(np.sum(data["close_time"].astype("int64").to_numpy() != times + STEP_MS - 1))
    o = data["open"].to_numpy(float)
    h = data["high"].to_numpy(float)
    l = data["low"].to_numpy(float)
    c = data["close"].to_numpy(float)
    finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    valid_ohlc = finite & (h >= np.maximum.reduce([o, c, l])) & (l <= np.minimum.reduce([o, c, h]))
    audit = {
        "source": "Binance USDⓈ-M Futures official monthly klines",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "start_utc": start.isoformat(),
        "end_utc": pd.to_datetime(end_ms - STEP_MS, unit="ms", utc=True).isoformat(),
        "expected_rows": int(len(expected_times)),
        "actual_rows": int(len(data)),
        "unique_rows": int(len(unique_times)),
        "duplicate_timestamps": duplicated,
        "missing_rows": int(len(missing)),
        "extra_rows": int(len(extra)),
        "off_grid_rows": off_grid,
        "invalid_close_time_rows": bad_close,
        "invalid_ohlc_rows": int(np.sum(~valid_ohlc)),
        "files": files,
    }
    audit["passed"] = bool(
        len(data) == len(expected_times)
        and len(unique_times) == len(expected_times)
        and duplicated == 0
        and len(missing) == 0
        and len(extra) == 0
        and off_grid == 0
        and bad_close == 0
        and np.all(valid_ohlc)
    )
    if not audit["passed"]:
        raise RuntimeError("Data audit failed: " + json.dumps(audit, ensure_ascii=False, indent=2))
    return data, audit


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def zscore(series: pd.Series, length: int) -> pd.Series:
    mean = series.rolling(length).mean()
    std = series.rolling(length).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    x = data.copy()
    x.index = pd.to_datetime(x["open_time"], unit="ms", utc=True)
    o, h, l, c, v = (x[k].astype(float) for k in ("open", "high", "low", "close", "volume"))

    for length in (8, 21, 55, 200):
        x[f"ema{length}"] = c.ewm(span=length, adjust=False).mean()

    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    x["atr"] = rma(tr, 14)
    x["atr_pct"] = x["atr"] / c
    x["atr_rank"] = x["atr_pct"].rolling(288).rank(pct=True)

    up = h.diff()
    down = -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_di = 100 * rma(plus_dm, 14) / x["atr"].replace(0, np.nan)
    minus_di = 100 * rma(minus_dm, 14) / x["atr"].replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["plus_di"] = plus_di
    x["minus_di"] = minus_di
    x["di_gap"] = (plus_di - minus_di) / 100.0
    x["adx"] = rma(dx, 14)

    change = c.diff()
    gain, loss = change.clip(lower=0), -change.clip(upper=0)
    rs = rma(gain, 14) / rma(loss, 14).replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + rs)

    fast = c.ewm(span=12, adjust=False).mean()
    slow = c.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    macd_hist = macd - macd.ewm(span=9, adjust=False).mean()
    x["macd_hist_atr"] = macd_hist / x["atr"].replace(0, np.nan)
    x["macd_slope_atr"] = macd_hist.diff() / x["atr"].replace(0, np.nan)

    basis = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    lower, upper = basis - 2 * sd, basis + 2 * sd
    x["bb_pos"] = (c - lower) / (upper - lower).replace(0, np.nan)
    width = (upper - lower) / basis.replace(0, np.nan)
    x["bb_rank"] = width.rolling(288).rank(pct=True)

    utc_day = x.index.floor("D")
    typical = (h + l + c) / 3
    x["vwap"] = (typical * v).groupby(utc_day).cumsum() / v.groupby(utc_day).cumsum().replace(0, np.nan)
    x["vwap_dev"] = (c - x["vwap"]) / x["atr"].replace(0, np.nan)

    x["rel_vol"] = v / v.rolling(48).mean().replace(0, np.nan)
    x["vol_z"] = zscore(v, 48)
    x["trade_z"] = zscore(x["trade_count"].astype(float), 48)
    x["taker_ratio"] = x["taker_buy_volume"].astype(float) / v.replace(0, np.nan)
    x["taker_z"] = zscore(x["taker_ratio"], 48)

    candle_range = (h - l).replace(0, np.nan)
    x["body"] = (c - o).abs() / candle_range
    x["close_loc"] = (c - l) / candle_range
    x["upper_wick"] = (h - np.maximum(o, c)) / candle_range
    x["lower_wick"] = (np.minimum(o, c) - l) / candle_range
    x["range_exp"] = candle_range / candle_range.rolling(24).mean().replace(0, np.nan)

    for n in (1, 3, 6, 12, 24):
        x[f"ret{n}"] = c.pct_change(n)
    abs_change = c.diff().abs()
    for n in (12, 24):
        x[f"eff{n}"] = (c - c.shift(n)).abs() / abs_change.rolling(n).sum().replace(0, np.nan)

    hh14, ll14 = h.rolling(14).max(), l.rolling(14).min()
    x["chop"] = 100 * np.log10(tr.rolling(14).sum() / (hh14 - ll14).replace(0, np.nan)) / math.log10(14)

    for length in (8, 21, 55, 200):
        x[f"ema{length}_gap"] = (c - x[f"ema{length}"]) / x["atr"].replace(0, np.nan)

    for window in (6, 12, 24):
        x[f"don{window}h"] = h.shift(1).rolling(window).max()
        x[f"don{window}l"] = l.shift(1).rolling(window).min()

    def htf(rule: str, prefix: str) -> None:
        bars = x[["open", "high", "low", "close", "volume"]].resample(
            rule, label="right", closed="left"
        ).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        e20 = bars["close"].ewm(span=20, adjust=False).mean()
        e50 = bars["close"].ewm(span=50, adjust=False).mean()
        trend = np.where((bars["close"] > e20) & (e20 > e50), 1.0,
                         np.where((bars["close"] < e20) & (e20 < e50), -1.0, 0.0))
        gap = (e20 - e50) / bars["close"].replace(0, np.nan)
        trend_s = pd.Series(trend, index=bars.index).shift(1)
        gap_s = gap.shift(1)
        x[f"{prefix}_trend"] = trend_s.reindex(x.index, method="ffill")
        x[f"{prefix}_gap"] = gap_s.reindex(x.index, method="ffill")

    htf("15min", "m15")
    htf("60min", "h1")
    htf("240min", "h4")

    pull_long = (
        (x["h1_trend"] >= 0) & (x["m15_trend"] > 0)
        & (l <= x["ema21"] + 0.35 * x["atr"]) & (c > x["ema8"])
        & (c > o) & (x["macd_slope_atr"] > 0)
    )
    pull_short = (
        (x["h1_trend"] <= 0) & (x["m15_trend"] < 0)
        & (h >= x["ema21"] - 0.35 * x["atr"]) & (c < x["ema8"])
        & (c < o) & (x["macd_slope_atr"] < 0)
    )
    sweep_long = (
        (l < x["don12l"]) & (c > x["don12l"]) & (c > o)
        & (x["lower_wick"] >= 0.28) & (x["taker_ratio"] >= 0.48)
    )
    sweep_short = (
        (h > x["don12h"]) & (c < x["don12h"]) & (c < o)
        & (x["upper_wick"] >= 0.28) & (x["taker_ratio"] <= 0.52)
    )
    breakout_long = (
        (c > x["don12h"]) & (c > x["ema55"]) & (x["h1_trend"] >= 0)
        & (x["body"] >= 0.42) & (x["close_loc"] >= 0.65) & (x["rel_vol"] >= 0.85)
    )
    breakout_short = (
        (c < x["don12l"]) & (c < x["ema55"]) & (x["h1_trend"] <= 0)
        & (x["body"] >= 0.42) & (x["close_loc"] <= 0.35) & (x["rel_vol"] >= 0.85)
    )
    momentum_long = (
        (x["h1_trend"] > 0) & (x["m15_trend"] > 0) & (c > x["don6h"])
        & (x["taker_ratio"] >= 0.53) & (x["range_exp"] >= 1.0)
    )
    momentum_short = (
        (x["h1_trend"] < 0) & (x["m15_trend"] < 0) & (c < x["don6l"])
        & (x["taker_ratio"] <= 0.47) & (x["range_exp"] >= 1.0)
    )

    x["setup_pull_long"] = pull_long.astype(float)
    x["setup_pull_short"] = pull_short.astype(float)
    x["setup_sweep_long"] = sweep_long.astype(float)
    x["setup_sweep_short"] = sweep_short.astype(float)
    x["setup_break_long"] = breakout_long.astype(float)
    x["setup_break_short"] = breakout_short.astype(float)
    x["setup_mom_long"] = momentum_long.astype(float)
    x["setup_mom_short"] = momentum_short.astype(float)
    x["candidate_long"] = pull_long | sweep_long | breakout_long | momentum_long
    x["candidate_short"] = pull_short | sweep_short | breakout_short | momentum_short

    # Exclusive market regimes. These are computed only from current/past bars.
    high_vol = x["atr_rank"] >= 0.80
    trend_regime = (
        ~high_vol
        & (x["h1_trend"] != 0)
        & (x["h1_trend"] == x["h4_trend"])
        & (x["adx"] >= 22)
        & (x["chop"] <= 56)
    )
    range_regime = (~high_vol) & (~trend_regime) & ((x["adx"] <= 20) | (x["chop"] >= 60))
    neutral_regime = ~(high_vol | trend_regime | range_regime)
    x["regime"] = np.select(
        [trend_regime, range_regime, high_vol, neutral_regime],
        [0, 1, 2, 3],
        default=3,
    ).astype(float)
    x["regime_trend"] = (x["regime"] == 0).astype(float)
    x["regime_range"] = (x["regime"] == 1).astype(float)
    x["regime_high_vol"] = (x["regime"] == 2).astype(float)
    x["regime_neutral"] = (x["regime"] == 3).astype(float)

    hours = x.index.hour + x.index.minute / 60.0
    x["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    x["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    x["weekday"] = x.index.weekday.astype(float)
    x["month"] = x.index.to_period("M").astype(str)

    return x.replace([np.inf, -np.inf], np.nan).dropna().copy()


FEATURES = [
    "ret1", "ret3", "ret6", "ret12", "ret24",
    "ema8_gap", "ema21_gap", "ema55_gap", "ema200_gap",
    "atr_pct", "atr_rank", "rsi", "adx", "di_gap",
    "macd_hist_atr", "macd_slope_atr", "bb_pos", "bb_rank", "vwap_dev",
    "rel_vol", "vol_z", "trade_z", "taker_ratio", "taker_z",
    "body", "close_loc", "upper_wick", "lower_wick", "range_exp",
    "eff12", "eff24", "chop",
    "m15_trend", "h1_trend", "h4_trend", "m15_gap", "h1_gap", "h4_gap",
    "hour_sin", "hour_cos", "weekday",
    "regime_trend", "regime_range", "regime_high_vol", "regime_neutral",
    "setup_pull_long", "setup_pull_short", "setup_sweep_long", "setup_sweep_short",
    "setup_break_long", "setup_break_short", "setup_mom_long", "setup_mom_short",
]


@dataclass(frozen=True)
class RiskConfig:
    rr: float
    sl_atr: float
    min_stop_pct: float
    max_hold: int


@dataclass(frozen=True)
class ModelConfig:
    max_depth: int
    learning_rate: float
    max_iter: int
    l2_regularization: float
    min_samples_leaf: int


@dataclass
class Policy:
    risk: RiskConfig
    model: ModelConfig
    quantile: float
    cooldown: int
    direction_mode: int  # 0 both, 1 long, 2 short
    regime_mask: int = 15  # bits: trend, range, high-vol, neutral
    setup_mask: int = 15   # bits: pullback, sweep, breakout, momentum
    validation_metrics: dict[str, dict[str, float]] | None = None
    validation_scores: list[float] | None = None
    aggregate_score: float = -1e12

    def clone(self, *, regime_mask: int | None = None, setup_mask: int | None = None) -> "Policy":
        return Policy(
            self.risk,
            self.model,
            self.quantile,
            self.cooldown,
            self.direction_mode,
            self.regime_mask if regime_mask is None else regime_mask,
            self.setup_mask if setup_mask is None else setup_mask,
            dict(self.validation_metrics or {}),
            list(self.validation_scores or []),
            self.aggregate_score,
        )


@njit(cache=True)
def compute_outcomes(
    indices: np.ndarray,
    direction: int,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    rr: float,
    sl_atr: float,
    min_stop_pct: float,
    max_hold: int,
    fee_rate: float,
    slippage_abs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(indices)
    labels = np.full(n, -1, dtype=np.int8)
    exits = np.full(n, -1, dtype=np.int64)
    net_r = np.zeros(n, dtype=np.float64)
    reasons = np.zeros(n, dtype=np.int8)  # 1 TP, 2 SL, 3 TIME
    for k in range(n):
        signal_i = indices[k]
        entry_i = signal_i + 1
        if entry_i >= len(close):
            continue
        entry = open_[entry_i] + direction * slippage_abs
        risk = max(atr[signal_i] * sl_atr, close[signal_i] * min_stop_pct)
        stop = entry - direction * risk
        target = entry + direction * risk * rr
        end_i = min(len(close) - 1, entry_i + max_hold)
        exit_price = close[end_i] - direction * slippage_abs
        reason = 3
        exit_i = end_i
        for j in range(entry_i, end_i + 1):
            if direction > 0:
                if low[j] <= stop:
                    exit_price = stop - slippage_abs
                    reason = 2
                    exit_i = j
                    break
                if high[j] >= target:
                    exit_price = target - slippage_abs
                    reason = 1
                    exit_i = j
                    break
            else:
                if high[j] >= stop:
                    exit_price = stop + slippage_abs
                    reason = 2
                    exit_i = j
                    break
                if low[j] <= target:
                    exit_price = target + slippage_abs
                    reason = 1
                    exit_i = j
                    break
        gross = (exit_price - entry) * direction
        fees = fee_rate * (entry + exit_price)
        value = (gross - fees) / risk
        labels[k] = 1 if reason == 1 else (0 if reason == 2 else -1)
        exits[k] = exit_i
        net_r[k] = value
        reasons[k] = reason
    return labels, exits, net_r, reasons


def balanced_weights(y: np.ndarray) -> np.ndarray:
    positives = max(1, int(np.sum(y == 1)))
    negatives = max(1, int(np.sum(y == 0)))
    total = positives + negatives
    w_pos = total / (2.0 * positives)
    w_neg = total / (2.0 * negatives)
    return np.where(y == 1, w_pos, w_neg)


def fit_models(
    x: pd.DataFrame,
    long_data: dict[str, np.ndarray],
    short_data: dict[str, np.ndarray],
    risk: RiskConfig,
    model_cfg: ModelConfig,
    train_months: set[str],
) -> tuple[HistGradientBoostingClassifier | None, HistGradientBoostingClassifier | None, float, float]:
    models: list[HistGradientBoostingClassifier | None] = []
    thresholds: list[float] = []
    for side_data in (long_data, short_data):
        idx = side_data["idx"]
        labels, _, _, _ = compute_outcomes(
            idx, int(side_data["direction"]),
            x["open"].to_numpy(float), x["high"].to_numpy(float), x["low"].to_numpy(float),
            x["close"].to_numpy(float), x["atr"].to_numpy(float),
            risk.rr, risk.sl_atr, risk.min_stop_pct, risk.max_hold, FEE_RATE, SLIPPAGE_ABS,
        )
        months = x["month"].to_numpy()[idx]
        mask = np.isin(months, list(train_months)) & (labels >= 0)
        y = labels[mask].astype(int)
        if len(y) < 120 or len(np.unique(y)) < 2 or min(np.sum(y == 0), np.sum(y == 1)) < 25:
            models.append(None)
            thresholds.append(float("nan"))
            continue
        X = x.iloc[idx[mask]][FEATURES].to_numpy(np.float64)
        model = HistGradientBoostingClassifier(
            max_depth=model_cfg.max_depth,
            learning_rate=model_cfg.learning_rate,
            max_iter=model_cfg.max_iter,
            l2_regularization=model_cfg.l2_regularization,
            min_samples_leaf=model_cfg.min_samples_leaf,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=BASE_SEED,
        )
        base_w = balanced_weights(y)
        train_month_values = months[mask]
        ordered_months = sorted(train_months)
        recency = {m: 0.82 + 0.36 * (i / max(1, len(ordered_months) - 1)) for i, m in enumerate(ordered_months)}
        time_w = np.array([recency.get(str(m), 1.0) for m in train_month_values], dtype=float)
        model.fit(X, y, sample_weight=base_w * time_w)
        train_prob = model.predict_proba(X)[:, 1]
        models.append(model)
        thresholds.append(float(np.nanmedian(train_prob)))
    return models[0], models[1], thresholds[0], thresholds[1]


def make_side_data(x: pd.DataFrame, column: str, direction: int) -> dict[str, np.ndarray]:
    idx = np.flatnonzero(x[column].to_numpy(bool)).astype(np.int64)
    return {"idx": idx, "direction": np.array(direction, dtype=np.int64)}



def build_prediction_bundle(
    x: pd.DataFrame,
    side_data: dict[str, np.ndarray],
    model: HistGradientBoostingClassifier | None,
    risk: RiskConfig,
    train_months: set[str],
) -> dict[str, Any] | None:
    if model is None:
        return None
    idx = side_data["idx"]
    direction = int(side_data["direction"])
    labels, exits, net_r, reasons = compute_outcomes(
        idx, direction,
        x["open"].to_numpy(float), x["high"].to_numpy(float), x["low"].to_numpy(float),
        x["close"].to_numpy(float), x["atr"].to_numpy(float),
        risk.rr, risk.sl_atr, risk.min_stop_pct, risk.max_hold, FEE_RATE, SLIPPAGE_ABS,
    )
    prob = model.predict_proba(x.iloc[idx][FEATURES].to_numpy(np.float64))[:, 1]
    months = x["month"].to_numpy()[idx]
    train_mask = np.isin(months, list(train_months)) & (labels >= 0)
    train_prob = prob[train_mask]
    if len(train_prob) < 50:
        return None
    suffix = "long" if direction > 0 else "short"
    setup_mask = (
        (x.iloc[idx][f"setup_pull_{suffix}"].to_numpy(bool).astype(np.int16) * 1)
        | (x.iloc[idx][f"setup_sweep_{suffix}"].to_numpy(bool).astype(np.int16) * 2)
        | (x.iloc[idx][f"setup_break_{suffix}"].to_numpy(bool).astype(np.int16) * 4)
        | (x.iloc[idx][f"setup_mom_{suffix}"].to_numpy(bool).astype(np.int16) * 8)
    )
    return {
        "idx": idx, "direction": direction, "labels": labels, "exits": exits,
        "net_r": net_r, "reasons": reasons, "prob": prob, "months": months,
        "train_prob": train_prob,
        "regime": x.iloc[idx]["regime"].to_numpy(np.int16),
        "setup_mask": setup_mask,
    }


def events_from_bundle(bundle: dict[str, Any] | None, eval_month: str, quantile: float) -> list[dict[str, Any]]:
    if bundle is None:
        return []
    threshold = float(np.quantile(bundle["train_prob"], quantile))
    mask = (bundle["months"] == eval_month) & (bundle["prob"] >= threshold) & (bundle["exits"] >= 0)
    events: list[dict[str, Any]] = []
    for k in np.flatnonzero(mask):
        events.append({
            "signal_i": int(bundle["idx"][k]),
            "exit_i": int(bundle["exits"][k]),
            "direction": int(bundle["direction"]),
            "prob": float(bundle["prob"][k]),
            "net_r": float(bundle["net_r"][k]),
            "reason": int(bundle["reasons"][k]),
            "regime": int(bundle["regime"][k]),
            "setup_mask": int(bundle["setup_mask"][k]),
        })
    return events

def predict_events(
    x: pd.DataFrame,
    side_data: dict[str, np.ndarray],
    model: HistGradientBoostingClassifier | None,
    risk: RiskConfig,
    eval_month: str,
    train_months: set[str],
    quantile: float,
) -> list[dict[str, Any]]:
    if model is None:
        return []
    idx = side_data["idx"]
    direction = int(side_data["direction"])
    labels, exits, net_r, reasons = compute_outcomes(
        idx, direction,
        x["open"].to_numpy(float), x["high"].to_numpy(float), x["low"].to_numpy(float),
        x["close"].to_numpy(float), x["atr"].to_numpy(float),
        risk.rr, risk.sl_atr, risk.min_stop_pct, risk.max_hold, FEE_RATE, SLIPPAGE_ABS,
    )
    X = x.iloc[idx][FEATURES].to_numpy(np.float64)
    prob = model.predict_proba(X)[:, 1]
    months = x["month"].to_numpy()[idx]
    train_prob = prob[np.isin(months, list(train_months)) & (labels >= 0)]
    if len(train_prob) < 50:
        return []
    threshold = float(np.quantile(train_prob, quantile))
    mask = (months == eval_month) & (prob >= threshold) & (exits >= 0)

    suffix = "long" if direction > 0 else "short"
    setup_values = (
        (x.iloc[idx][f"setup_pull_{suffix}"].to_numpy(bool).astype(np.int16) * 1)
        | (x.iloc[idx][f"setup_sweep_{suffix}"].to_numpy(bool).astype(np.int16) * 2)
        | (x.iloc[idx][f"setup_break_{suffix}"].to_numpy(bool).astype(np.int16) * 4)
        | (x.iloc[idx][f"setup_mom_{suffix}"].to_numpy(bool).astype(np.int16) * 8)
    )
    regime_values = x.iloc[idx]["regime"].to_numpy(np.int16)

    events: list[dict[str, Any]] = []
    for k in np.flatnonzero(mask):
        events.append({
            "signal_i": int(idx[k]),
            "exit_i": int(exits[k]),
            "direction": direction,
            "prob": float(prob[k]),
            "net_r": float(net_r[k]),
            "reason": int(reasons[k]),
            "regime": int(regime_values[k]),
            "setup_mask": int(setup_values[k]),
        })
    return events


def select_trades(
    events: list[dict[str, Any]],
    cooldown: int,
    direction_mode: int,
    regime_mask: int = 15,
    setup_mask: int = 15,
) -> list[dict[str, Any]]:
    if direction_mode == 1:
        events = [e for e in events if e["direction"] > 0]
    elif direction_mode == 2:
        events = [e for e in events if e["direction"] < 0]
    events = [
        e for e in events
        if ((regime_mask >> int(e.get("regime", 3))) & 1) != 0
        and (int(e.get("setup_mask", 15)) & setup_mask) != 0
    ]
    events = sorted(events, key=lambda e: (e["signal_i"], -e["prob"]))
    selected: list[dict[str, Any]] = []
    last_exit = -10**9
    i = 0
    while i < len(events):
        signal_i = events[i]["signal_i"]
        same: list[dict[str, Any]] = []
        while i < len(events) and events[i]["signal_i"] == signal_i:
            same.append(events[i])
            i += 1
        best = max(same, key=lambda e: e["prob"])
        if signal_i <= last_exit + cooldown:
            continue
        selected.append(best)
        last_exit = best["exit_i"]
    return selected


def metrics(trades: list[dict[str, Any]]) -> dict[str, float]:
    values = np.array([t["net_r"] for t in trades], dtype=float)
    count = int(len(values))
    wins = values[values > 0]
    losses = -values[values <= 0]
    wr = float(len(wins) / count) if count else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    ratio = avg_win / avg_loss if avg_win > 0 and avg_loss > 0 else 0.0
    pf = float(wins.sum() / losses.sum()) if len(losses) and losses.sum() > 0 else 0.0
    net = float(values.sum()) if count else 0.0
    cum = np.cumsum(values) if count else np.array([0.0])
    peak = np.maximum.accumulate(np.r_[0.0, cum])
    dd = peak[1:] - cum if count else np.array([0.0])
    return {
        "trades": count,
        "wins": int(len(wins)),
        "win_rate": wr,
        "avg_win_R": avg_win,
        "avg_loss_R": avg_loss,
        "avg_win_loss_ratio": ratio,
        "profit_factor": pf,
        "net_R": net,
        "max_drawdown_R": float(dd.max()) if len(dd) else 0.0,
    }


def month_score(m: dict[str, float], *, final_month: bool = False) -> float:
    count = int(m["trades"])
    target_min = MIN_TRADES if final_month else max(10, MIN_TRADES - 3)
    target_max = MAX_TRADES if final_month else MAX_TRADES + 8
    target = (target_min + target_max) / 2
    count_error = abs(count - target)
    score = (
        m["win_rate"] * 1050.0
        + min(m["avg_win_loss_ratio"], 4.0) * 220.0
        + min(m["profit_factor"], 8.0) * 45.0
        + m["net_R"] * 9.0
        - m["max_drawdown_R"] * 5.0
        - count_error * 10.0
    )
    if count < target_min or count > target_max:
        score -= 4200.0 + abs(count - np.clip(count, target_min, target_max)) * 260.0
    if m["win_rate"] < MIN_WIN_RATE:
        score -= (MIN_WIN_RATE - m["win_rate"]) * 7200.0
    if m["avg_win_loss_ratio"] < MIN_RATIO:
        score -= (MIN_RATIO - m["avg_win_loss_ratio"]) * 3200.0
    if m["net_R"] <= 0:
        score -= 900.0 + abs(m["net_R"]) * 40.0
    return float(score)


def refresh_policy_score(policy: Policy) -> None:
    scores = np.asarray(policy.validation_scores or [], dtype=float)
    if len(scores) == 0:
        policy.aggregate_score = -1e12
        return
    recent = scores[-1]
    policy.aggregate_score = float(
        0.42 * np.min(scores)
        + 0.33 * np.mean(scores)
        + 0.25 * recent
        - 0.18 * np.std(scores)
    )


def add_validation(policy: Policy, month: str, value: dict[str, float], *, final_month: bool = False) -> None:
    if policy.validation_metrics is None:
        policy.validation_metrics = {}
    if policy.validation_scores is None:
        policy.validation_scores = []
    policy.validation_metrics[month] = value
    policy.validation_scores.append(month_score(value, final_month=final_month))
    refresh_policy_score(policy)


def policy_base_key(policy: Policy) -> tuple[RiskConfig, ModelConfig]:
    return policy.risk, policy.model


def evaluate_policy_batch(
    x: pd.DataFrame,
    long_data: dict[str, np.ndarray],
    short_data: dict[str, np.ndarray],
    policies: list[Policy],
    train_months: set[str],
    eval_month: str,
    *,
    final_month: bool = False,
) -> None:
    groups: dict[tuple[RiskConfig, ModelConfig], list[Policy]] = {}
    for policy in policies:
        groups.setdefault(policy_base_key(policy), []).append(policy)
    for (risk, model_cfg), group in groups.items():
        long_model, short_model, _, _ = fit_models(x, long_data, short_data, risk, model_cfg, train_months)
        if long_model is None and short_model is None:
            for policy in group:
                add_validation(policy, eval_month, metrics([]), final_month=final_month)
            continue
        long_bundle = build_prediction_bundle(x, long_data, long_model, risk, train_months)
        short_bundle = build_prediction_bundle(x, short_data, short_model, risk, train_months)
        event_cache: dict[float, list[dict[str, Any]]] = {}
        for policy in group:
            if policy.quantile not in event_cache:
                events = events_from_bundle(long_bundle, eval_month, policy.quantile)
                events += events_from_bundle(short_bundle, eval_month, policy.quantile)
                event_cache[policy.quantile] = events
            chosen = select_trades(
                event_cache[policy.quantile], policy.cooldown, policy.direction_mode,
                policy.regime_mask, policy.setup_mask,
            )
            add_validation(policy, eval_month, metrics(chosen), final_month=final_month)


def evaluate_policy(
    x: pd.DataFrame,
    long_data: dict[str, np.ndarray],
    short_data: dict[str, np.ndarray],
    policy: Policy,
    train_months: set[str],
    eval_month: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    long_model, short_model, _, _ = fit_models(x, long_data, short_data, policy.risk, policy.model, train_months)
    events = predict_events(x, long_data, long_model, policy.risk, eval_month, train_months, policy.quantile)
    events += predict_events(x, short_data, short_model, policy.risk, eval_month, train_months, policy.quantile)
    selected = select_trades(events, policy.cooldown, policy.direction_mode, policy.regime_mask, policy.setup_mask)
    return metrics(selected), selected


def setup_name(x: pd.DataFrame, signal_i: int, direction: int) -> str:
    suffix = "long" if direction > 0 else "short"
    names = [
        (f"setup_pull_{suffix}", "趋势回踩"),
        (f"setup_sweep_{suffix}", "流动性扫单反转"),
        (f"setup_break_{suffix}", "突破确认"),
        (f"setup_mom_{suffix}", "动量延续"),
    ]
    active = [label for col, label in names if x.iloc[signal_i][col] > 0.5]
    return "+".join(active) if active else "候选信号"


def detailed_trades(x: pd.DataFrame, trades: list[dict[str, Any]], risk: RiskConfig, month: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for t in trades:
        i = int(t["signal_i"])
        entry_i = i + 1
        direction = int(t["direction"])
        entry = float(x.iloc[entry_i]["open"] + direction * SLIPPAGE_ABS)
        risk_abs = max(float(x.iloc[i]["atr"]) * risk.sl_atr, float(x.iloc[i]["close"]) * risk.min_stop_pct)
        exit_i = int(t["exit_i"])
        reason = {1: "TP", 2: "SL", 3: "TIME"}.get(int(t["reason"]), "UNKNOWN")
        rows.append({
            "signal_time_utc": x.index[i].isoformat(),
            "entry_time_utc": x.index[entry_i].isoformat(),
            "exit_time_utc": x.index[exit_i].isoformat(),
            "month": month,
            "direction": "LONG" if direction > 0 else "SHORT",
            "setup": setup_name(x, i, direction),
            "probability": float(t["prob"]),
            "entry": entry,
            "risk_abs": risk_abs,
            "target_R": risk.rr,
            "net_R": float(t["net_r"]),
            "win": bool(t["net_r"] > 0),
            "exit_reason": reason,
            "bars": exit_i - entry_i,
        })
    return pd.DataFrame(rows)


def main() -> None:
    # The existing GitHub workflow uploads and commits results_v4/. Clear stale V4/V5 files first.
    if RESULTS.exists():
        for item in RESULTS.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    RESULTS.mkdir(exist_ok=True)

    raw, audit = load_official_data()
    x = add_features(raw)
    long_data = make_side_data(x, "candidate_long", 1)
    short_data = make_side_data(x, "candidate_short", -1)

    # Higher nominal RR and longer holding windows preserve the realised win/loss ratio.
    risk_grid = [
        RiskConfig(rr, sl, min_stop, hold)
        for rr in (2.4, 2.7, 3.0)
        for sl in (1.1, 1.7)
        for min_stop in (0.0030, 0.0045)
        for hold in (288, 576)
    ]
    model_grid = [
        ModelConfig(2, 0.045, 170, 2.0, 35),
        ModelConfig(3, 0.035, 210, 5.0, 50),
    ]
    quantiles = (0.88, 0.91, 0.94, 0.965, 0.985)
    cooldowns = (3, 8, 16)
    directions = (0, 1, 2)
    regime_masks = (15, 1, 2, 4, 8, 5, 9)
    setup_masks = (15, 1, 2, 4, 8, 3, 5, 9, 12)

    # Months: Sep-Apr development, May final selection, June untouched OOS.
    # Stage 1: Sep-Nov train -> Dec broad prefilter.
    stage1: list[Policy] = []
    train0 = {MONTHS[0], MONTHS[1], MONTHS[2]}
    for risk in risk_grid:
        for model_cfg in model_grid:
            long_model, short_model, _, _ = fit_models(x, long_data, short_data, risk, model_cfg, train0)
            if long_model is None and short_model is None:
                continue
            long_bundle = build_prediction_bundle(x, long_data, long_model, risk, train0)
            short_bundle = build_prediction_bundle(x, short_data, short_model, risk, train0)
            for q in quantiles:
                events = events_from_bundle(long_bundle, MONTHS[3], q)
                events += events_from_bundle(short_bundle, MONTHS[3], q)
                for cooldown in cooldowns:
                    for direction_mode in directions:
                        policy = Policy(risk, model_cfg, q, cooldown, direction_mode)
                        m = metrics(select_trades(events, cooldown, direction_mode))
                        add_validation(policy, MONTHS[3], m)
                        stage1.append(policy)
    stage1.sort(key=lambda p: p.aggregate_score, reverse=True)
    base_shortlist = stage1[:60]

    # Stage 2: Expand only the strongest bases with regime/setup filters; validate January.
    expanded: list[Policy] = []
    for base in base_shortlist:
        for regime_mask in regime_masks:
            for setup_mask in setup_masks:
                expanded.append(base.clone(regime_mask=regime_mask, setup_mask=setup_mask))
    evaluate_policy_batch(
        x, long_data, short_data, expanded,
        set(MONTHS[:4]), MONTHS[4],
    )
    expanded.sort(key=lambda p: p.aggregate_score, reverse=True)
    active = expanded[:54]

    # Rolling tournament: each month must remain useful before the next stage.
    rolling = [
        (5, set(MONTHS[:5]), 44),
        (6, set(MONTHS[:6]), 34),
        (7, set(MONTHS[:7]), 24),
    ]
    for eval_idx, train_months, keep in rolling:
        evaluate_policy_batch(x, long_data, short_data, active, train_months, MONTHS[eval_idx])
        active.sort(key=lambda p: p.aggregate_score, reverse=True)
        active = active[:keep]

    # May is the last selection month and uses the exact hard trade-count constraint.
    evaluate_policy_batch(x, long_data, short_data, active, set(MONTHS[:8]), MONTHS[8], final_month=True)
    active.sort(key=lambda p: p.aggregate_score, reverse=True)
    may_qualified = [
        p for p in active
        if p.validation_metrics
        and MIN_TRADES <= p.validation_metrics[MONTHS[8]]["trades"] <= MAX_TRADES
        and p.validation_metrics[MONTHS[8]]["win_rate"] >= MIN_WIN_RATE
        and p.validation_metrics[MONTHS[8]]["avg_win_loss_ratio"] >= MIN_RATIO
    ]
    selected = (may_qualified or active)[0]

    # June remains completely untouched until the selected policy is frozen.
    june_metrics, june_trades = evaluate_policy(
        x, long_data, short_data, selected,
        set(MONTHS[:9]), MONTHS[9],
    )
    may_metrics, may_trades = evaluate_policy(
        x, long_data, short_data, selected,
        set(MONTHS[:8]), MONTHS[8],
    )

    qualified = bool(
        MIN_TRADES <= may_metrics["trades"] <= MAX_TRADES
        and MIN_TRADES <= june_metrics["trades"] <= MAX_TRADES
        and may_metrics["win_rate"] >= MIN_WIN_RATE
        and june_metrics["win_rate"] >= MIN_WIN_RATE
        and may_metrics["avg_win_loss_ratio"] >= MIN_RATIO
        and june_metrics["avg_win_loss_ratio"] >= MIN_RATIO
    )

    all_trades = pd.concat([
        detailed_trades(x, may_trades, selected.risk, MONTHS[8]),
        detailed_trades(x, june_trades, selected.risk, MONTHS[9]),
    ], ignore_index=True)
    all_trades.to_csv(RESULTS / "trades.csv", index=False)

    regime_labels = {1: "趋势", 2: "震荡", 4: "高波动", 8: "中性", 15: "全部"}
    setup_labels = {1: "趋势回踩", 2: "扫单反转", 4: "突破确认", 8: "动量延续", 15: "全部"}
    status = {
        "qualified": qualified,
        "engine": ENGINE_NAME,
        "method": "Sep-Nov train→Dec prefilter; rolling Jan-Apr validation; May selection; June untouched OOS",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "months": list(MONTHS),
        "constraints": {
            "min_trades": MIN_TRADES,
            "max_trades": MAX_TRADES,
            "min_win_rate": MIN_WIN_RATE,
            "min_avg_win_loss_ratio": MIN_RATIO,
        },
        "selected_policy": {
            "risk": asdict(selected.risk),
            "model": asdict(selected.model),
            "probability_quantile": selected.quantile,
            "cooldown_bars": selected.cooldown,
            "direction_mode": {0: "双向", 1: "只做多", 2: "只做空"}[selected.direction_mode],
            "regime_mask": selected.regime_mask,
            "regime": regime_labels.get(selected.regime_mask, f"组合{selected.regime_mask}"),
            "setup_mask": selected.setup_mask,
            "setup": setup_labels.get(selected.setup_mask, f"组合{selected.setup_mask}"),
        },
        "development_monthly_stats": selected.validation_metrics,
        "monthly_stats": {MONTHS[8]: may_metrics, MONTHS[9]: june_metrics},
        "stage1_policies": len(stage1),
        "expanded_policies": len(expanded),
        "may_hard_qualified_candidates": len(may_qualified),
    }
    (RESULTS / "engine_version.txt").write_text(
        f"{ENGINE_NAME}\nmonths={','.join(MONTHS)}\noutput=results_v6_1\n", encoding="utf-8"
    )
    (RESULTS / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "selected_policy.json").write_text(
        json.dumps(status["selected_policy"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    top_rows: list[dict[str, Any]] = []
    for rank, policy in enumerate(active[:20], start=1):
        row: dict[str, Any] = {
            "rank": rank,
            "aggregate_score": policy.aggregate_score,
            "rr": policy.risk.rr,
            "sl_atr": policy.risk.sl_atr,
            "min_stop_pct": policy.risk.min_stop_pct,
            "max_hold": policy.risk.max_hold,
            "quantile": policy.quantile,
            "cooldown": policy.cooldown,
            "direction": {0: "双向", 1: "只做多", 2: "只做空"}[policy.direction_mode],
            "regime_mask": policy.regime_mask,
            "setup_mask": policy.setup_mask,
        }
        for month, value in (policy.validation_metrics or {}).items():
            for key, metric_value in value.items():
                row[f"{month}_{key}"] = metric_value
        top_rows.append(row)
    pd.DataFrame(top_rows).to_csv(RESULTS / "top_policies.csv", index=False)

    report = f"""# BTCUSDT 5分钟 市场状态 Walk-Forward V6.1 回测报告

- 数据：Binance USDⓈ-M 永续官方5分钟K线，{audit['actual_rows']:,}根；缺失{audit['missing_rows']}、重复{audit['duplicate_timestamps']}。
- 方法：9–11月训练→12月预筛；1–4月逐月滚动验证；5月最终选择；6月完全样本外。
- 引擎版本：V6.1；结果写入 results_v6_1，使用独立 V6.1 工作流。
- 模型：多空独立 HistGradientBoosting；趋势、震荡、高波动、中性状态过滤；四类价格行为机会。
- 风控：只搜索2.40R–3.00R目标和1–3天持仓窗口，以保护实际平均盈亏比。
- 成本：单边手续费{FEE_RATE*100:.3f}%；每次成交滑点{SLIPPAGE_ABS:.1f} USDT。
- 最终验收：**{'达标' if qualified else '未达到全部要求'}**。

## 选择策略

- 方向：{ {0:'双向',1:'只做多',2:'只做空'}[selected.direction_mode] }
- 市场状态：{regime_labels.get(selected.regime_mask, f'组合{selected.regime_mask}')}
- 机会结构：{setup_labels.get(selected.setup_mask, f'组合{selected.setup_mask}')}
- 固定目标：{selected.risk.rr:.2f}R
- 止损：max({selected.risk.sl_atr:.2f}×ATR, {selected.risk.min_stop_pct*100:.3f}%价格)
- 最长持仓：{selected.risk.max_hold}根5分钟K线
- 概率分位：训练候选前{(1-selected.quantile)*100:.2f}%
- 冷却：{selected.cooldown}根K线
- 5月硬条件候选数量：{len(may_qualified)}

## 月度结果

| 月份 | 交易 | 胜率 | 平均盈利/平均亏损 | 盈利因子 | 净R | 最大回撤R |
|---|---:|---:|---:|---:|---:|---:|
| {MONTHS[8]} | {may_metrics['trades']} | {may_metrics['win_rate']*100:.2f}% | {may_metrics['avg_win_loss_ratio']:.3f} | {may_metrics['profit_factor']:.3f} | {may_metrics['net_R']:.3f} | {may_metrics['max_drawdown_R']:.3f} |
| {MONTHS[9]} | {june_metrics['trades']} | {june_metrics['win_rate']*100:.2f}% | {june_metrics['avg_win_loss_ratio']:.3f} | {june_metrics['profit_factor']:.3f} | {june_metrics['net_R']:.3f} | {june_metrics['max_drawdown_R']:.3f} |

## 保守假设

信号在K线收盘后确认，下一根K线开盘成交；同一根K线同时触及止损和止盈时按止损优先；超时按市价退出。6月没有参与模型、阈值、市场状态或结构筛选。
"""
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
