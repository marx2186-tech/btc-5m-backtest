from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from numba import njit, prange

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
RESULTS = ROOT / "results_v4"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

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
    "search_rounds": 10,
    "search_batch": 30000,
    "seed_count": 5,
    "base_seed": 20260730,
}


def load_request() -> dict[str, Any]:
    req = dict(DEFAULT_REQUEST)
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(ROOT / "request.v4.json")))
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("request.v4.json must contain a JSON object")
        req.update(loaded)
    return req


def interval_to_ms(interval: str) -> int:
    match = re.fullmatch(r"(\d+)([mhd])", interval.strip().lower())
    if not match:
        raise ValueError(f"Unsupported interval: {interval}")
    value, unit = int(match.group(1)), match.group(2)
    return value * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]


REQUEST = load_request()
SYMBOL = str(REQUEST["symbol"]).upper()
INTERVAL = str(REQUEST["interval"]).lower()
MONTHS = tuple(str(v) for v in REQUEST["months"])
if len(MONTHS) != 2:
    raise ValueError("V4 requires exactly two complete calendar months")
STEP_MS = interval_to_ms(INTERVAL)
START_TS = pd.Timestamp(f"{MONTHS[0]}-01T00:00:00Z")
END_EXCLUSIVE_TS = pd.Timestamp(f"{MONTHS[1]}-01T00:00:00Z") + pd.offsets.MonthBegin(1)
START_MS = int(START_TS.timestamp() * 1000)
END_EXCLUSIVE_MS = int(END_EXCLUSIVE_TS.timestamp() * 1000)
EXPECTED = (END_EXCLUSIVE_MS - START_MS) // STEP_MS
FEE_RATE = float(REQUEST["fee_rate_per_side"])
TICK_SIZE = float(REQUEST["tick_size"])
SLIPPAGE_TICKS = int(REQUEST["slippage_ticks_per_fill"])
SLIPPAGE_ABS = TICK_SIZE * SLIPPAGE_TICKS
MIN_TRADES = int(REQUEST["min_trades_per_month"])
MAX_TRADES = int(REQUEST["max_trades_per_month"])
TARGET_TRADES = (MIN_TRADES + MAX_TRADES) / 2.0
MIN_WIN_RATE = float(REQUEST["min_win_rate"])
MIN_RATIO = float(REQUEST["min_avg_win_loss_ratio"])
BASE_SEED = int(REQUEST["base_seed"])
SEED_COUNT = int(REQUEST["seed_count"])

COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote", "ignore",
]


def download(url: str, path: Path, attempts: int = 6) -> bytes:
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    last: Exception | None = None
    headers = {"User-Agent": "btc-price-action-v4/1.0"}
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=90, headers=headers)
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

    expected_times = np.arange(START_MS, END_EXCLUSIVE_MS, STEP_MS, dtype=np.int64)
    times = data["open_time"].astype("int64").to_numpy()
    unique_times = np.unique(times)
    duplicated = int(pd.Series(times).duplicated().sum())
    missing = np.setdiff1d(expected_times, unique_times)
    extra = np.setdiff1d(unique_times, expected_times)
    off_grid = int(np.sum((times - START_MS) % STEP_MS != 0))
    bad_close = int(np.sum(data["close_time"].astype("int64").to_numpy() != times + STEP_MS - 1))
    o = data["open"].to_numpy(float)
    h = data["high"].to_numpy(float)
    l = data["low"].to_numpy(float)
    c = data["close"].to_numpy(float)
    finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    valid_ohlc = finite & (h >= np.maximum.reduce([o, c, l])) & (l <= np.minimum.reduce([o, c, h]))
    invalid_ohlc = int(np.sum(~valid_ohlc))
    audit = {
        "source": "Binance USDⓈ-M Futures official monthly klines",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "start_utc": START_TS.isoformat(),
        "end_utc": pd.to_datetime(END_EXCLUSIVE_MS - STEP_MS, unit="ms", utc=True).isoformat(),
        "expected_rows": int(EXPECTED),
        "actual_rows": int(len(data)),
        "unique_rows": int(len(unique_times)),
        "duplicate_timestamps": duplicated,
        "missing_rows": int(len(missing)),
        "extra_rows": int(len(extra)),
        "off_grid_rows": off_grid,
        "invalid_close_time_rows": bad_close,
        "invalid_ohlc_rows": invalid_ohlc,
        "files": files,
    }
    audit["passed"] = bool(
        len(data) == EXPECTED
        and len(unique_times) == EXPECTED
        and duplicated == 0
        and len(missing) == 0
        and len(extra) == 0
        and off_grid == 0
        and bad_close == 0
        and invalid_ohlc == 0
        and times[0] == START_MS
        and times[-1] == END_EXCLUSIVE_MS - STEP_MS
    )
    if not audit["passed"]:
        raise RuntimeError("Data audit failed: " + json.dumps(audit, ensure_ascii=False, indent=2))
    return data, audit


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    x = data.copy()
    x.index = pd.to_datetime(x["open_time"], unit="ms", utc=True)
    o, h, l, c, v = (x[k].astype(float) for k in ("open", "high", "low", "close", "volume"))

    for length in (8, 21, 55, 200):
        x[f"ema{length}"] = c.ewm(span=length, adjust=False).mean()

    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    x["tr"] = tr
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
    x["plus_di"], x["minus_di"], x["adx"] = plus_di, minus_di, rma(dx, 14)

    change = c.diff()
    gain, loss = change.clip(lower=0), -change.clip(upper=0)
    rs = rma(gain, 14) / rma(loss, 14).replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + rs)
    fast = c.ewm(span=12, adjust=False).mean()
    slow = c.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    x["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    x["macd_slope"] = x["macd_hist"].diff()

    basis = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    lower, upper = basis - 2 * sd, basis + 2 * sd
    x["bb_pos"] = (c - lower) / (upper - lower).replace(0, np.nan)
    x["bb_width"] = (upper - lower) / basis.replace(0, np.nan)
    x["bb_rank"] = x["bb_width"].rolling(288).rank(pct=True)

    utc_day = x.index.floor("D")
    typical = (h + l + c) / 3
    x["vwap"] = (typical * v).groupby(utc_day).cumsum() / v.groupby(utc_day).cumsum().replace(0, np.nan)
    x["vwap_dev"] = (c - x["vwap"]) / x["atr"].replace(0, np.nan)
    x["rel_vol"] = v / v.rolling(48).mean().replace(0, np.nan)

    candle_range = (h - l).replace(0, np.nan)
    body_abs = (c - o).abs()
    x["body"] = body_abs / candle_range
    x["close_loc"] = (c - l) / candle_range
    x["upper_wick"] = (h - np.maximum(o, c)) / candle_range
    x["lower_wick"] = (np.minimum(o, c) - l) / candle_range
    x["range_exp"] = candle_range / candle_range.rolling(24).mean().replace(0, np.nan)

    for n in (1, 3, 6, 12, 24):
        x[f"ret{n}"] = c.pct_change(n)
    abs_change = c.diff().abs()
    for n in (12, 24):
        x[f"eff{n}"] = (c - c.shift(n)).abs() / abs_change.rolling(n).sum().replace(0, np.nan)

    hh14 = h.rolling(14).max()
    ll14 = l.rolling(14).min()
    x["chop"] = 100 * np.log10(tr.rolling(14).sum() / (hh14 - ll14).replace(0, np.nan)) / math.log10(14)

    windows = (6, 12, 24)
    for wi, window in enumerate(windows):
        prev_high = h.shift(1).rolling(window).max()
        prev_low = l.shift(1).rolling(window).min()
        x[f"don{window}h"] = prev_high
        x[f"don{window}l"] = prev_low
        x[f"sweep_l_{wi}"] = (l < prev_low) & (c > prev_low) & (c > o)
        x[f"sweep_s_{wi}"] = (h > prev_high) & (c < prev_high) & (c < o)

    def htf_features(rule: str, prefix: str) -> None:
        bars = x[["open", "high", "low", "close", "volume"]].resample(rule, label="right", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        e20 = bars["close"].ewm(span=20, adjust=False).mean()
        e50 = bars["close"].ewm(span=50, adjust=False).mean()
        long_regime = (bars["close"] > e20) & (e20 > e50) & (e50 > e50.shift(2))
        short_regime = (bars["close"] < e20) & (e20 < e50) & (e50 < e50.shift(2))
        x[f"{prefix}_long"] = long_regime.shift(1).reindex(x.index, method="ffill").fillna(False)
        x[f"{prefix}_short"] = short_regime.shift(1).reindex(x.index, method="ffill").fillna(False)

    htf_features("15min", "m15")
    htf_features("60min", "h1")
    htf_features("240min", "h4")

    pull_tols = (0.15, 0.30, 0.50)
    pull_bars = (3, 6, 12)
    for ti, tol in enumerate(pull_tols):
        touch_l = (l <= x["ema21"] + tol * x["atr"]) & (l >= x["ema55"] - x["atr"])
        touch_s = (h >= x["ema21"] - tol * x["atr"]) & (h <= x["ema55"] + x["atr"])
        for bi, bars in enumerate(pull_bars):
            x[f"pl_{ti}_{bi}"] = touch_l.rolling(bars).max().fillna(0).astype(bool) & (c > x["ema8"])
            x[f"ps_{ti}_{bi}"] = touch_s.rolling(bars).max().fillna(0).astype(bool) & (c < x["ema8"])

    retest_windows = (12, 24)
    retest_bars = (3, 6, 12)
    retest_tols = (0.10, 0.25, 0.40)
    for wi, window in enumerate(retest_windows):
        high_level = h.shift(1).rolling(window).max()
        low_level = l.shift(1).rolling(window).min()
        break_l = c > high_level
        break_s = c < low_level
        for bi, bars in enumerate(retest_bars):
            last_high = high_level.where(break_l).shift(1).ffill(limit=bars)
            last_low = low_level.where(break_s).shift(1).ffill(limit=bars)
            for ti, tol in enumerate(retest_tols):
                idx = wi * 9 + bi * 3 + ti
                x[f"retest_l_{idx}"] = (
                    last_high.notna() & (l <= last_high + tol * x["atr"]) & (c > last_high) & (c > o)
                )
                x[f"retest_s_{idx}"] = (
                    last_low.notna() & (h >= last_low - tol * x["atr"]) & (c < last_low) & (c < o)
                )

    return x.replace([np.inf, -np.inf], np.nan).dropna().copy()


@dataclass
class Config:
    family_mask: int
    direction: int
    rr: float
    sl_atr: float
    min_stop_pct: float
    max_hold: int
    cooldown: int
    session_start: int
    session_end: int
    weekday_mask: int
    htf_mode: int
    adx_min: float
    rsi_long_min: float
    rsi_long_max: float
    relvol_min: float
    body_min: float
    close_loc_min: float
    wick_min: float
    eff_min: float
    chop_max: float
    atr_rank_min: float
    vwap_dev_max: float
    pull_idx: int
    sweep_idx: int
    retest_idx: int
    trail_trigger: float
    trail_lock: float


FAMILY_NAMES = {
    1: "多周期趋势回踩",
    2: "流动性扫单反转",
    3: "趋势回踩+流动性扫单",
    4: "突破回踩确认",
    5: "趋势回踩+突破回踩",
    6: "扫单反转+突破回踩",
    7: "三结构组合",
}


def random_config(rng: random.Random) -> Config:
    if rng.random() < 0.58:
        start, end = 0, 24
    else:
        start = rng.randrange(0, 17)
        end = min(24, start + rng.randrange(6, 13))
    rr = rng.choice((2.0, 2.2, 2.4, 2.6, 2.8, 3.0))
    trigger = rng.uniform(1.70, min(2.45, rr - 0.10))
    lock_hi = max(1.56, min(trigger - 0.05, rr - 0.15))
    lock = rng.uniform(1.50, lock_hi)
    return Config(
        family_mask=rng.choices((1, 2, 3, 4, 5, 6, 7), weights=(7, 7, 9, 7, 10, 8, 6))[0],
        direction=rng.choices((0, 1, 2), weights=(8, 3, 3))[0],
        rr=rr,
        sl_atr=rng.uniform(0.80, 2.40),
        min_stop_pct=rng.uniform(0.0025, 0.0080),
        max_hold=rng.choice((216, 288, 432, 576)),
        cooldown=rng.randrange(6, 61),
        session_start=start,
        session_end=end,
        weekday_mask=rng.choice((127, 127, 31, 62, 124)),
        htf_mode=rng.randrange(3),
        adx_min=rng.uniform(15, 38),
        rsi_long_min=rng.uniform(42, 56),
        rsi_long_max=rng.uniform(58, 74),
        relvol_min=rng.uniform(0.75, 1.80),
        body_min=rng.uniform(0.25, 0.72),
        close_loc_min=rng.uniform(0.58, 0.90),
        wick_min=rng.uniform(0.25, 0.70),
        eff_min=rng.uniform(0.16, 0.62),
        chop_max=rng.uniform(45, 66),
        atr_rank_min=rng.uniform(0.10, 0.75),
        vwap_dev_max=rng.uniform(0.60, 3.00),
        pull_idx=rng.randrange(9),
        sweep_idx=rng.randrange(3),
        retest_idx=rng.randrange(18),
        trail_trigger=trigger,
        trail_lock=lock,
    )


def build_arrays(x: pd.DataFrame) -> dict[str, np.ndarray]:
    keys = [
        "open", "high", "low", "close", "atr", "ema8", "ema21", "ema55", "ema200",
        "vwap", "vwap_dev", "plus_di", "minus_di", "adx", "rsi", "macd_hist", "macd_slope",
        "rel_vol", "body", "close_loc", "upper_wick", "lower_wick", "range_exp",
        "ret1", "ret3", "ret12", "eff12", "eff24", "chop", "atr_rank",
    ]
    arrays = {key: x[key].to_numpy(np.float64) for key in keys}
    for key in ("m15_long", "m15_short", "h1_long", "h1_short", "h4_long", "h4_short"):
        arrays[key] = x[key].to_numpy(np.bool_)
    arrays["pull_l"] = np.column_stack([x[f"pl_{ti}_{bi}"].to_numpy(np.bool_) for ti in range(3) for bi in range(3)])
    arrays["pull_s"] = np.column_stack([x[f"ps_{ti}_{bi}"].to_numpy(np.bool_) for ti in range(3) for bi in range(3)])
    arrays["sweep_l"] = np.column_stack([x[f"sweep_l_{i}"].to_numpy(np.bool_) for i in range(3)])
    arrays["sweep_s"] = np.column_stack([x[f"sweep_s_{i}"].to_numpy(np.bool_) for i in range(3)])
    arrays["retest_l"] = np.column_stack([x[f"retest_l_{i}"].to_numpy(np.bool_) for i in range(18)])
    arrays["retest_s"] = np.column_stack([x[f"retest_s_{i}"].to_numpy(np.bool_) for i in range(18)])
    index = x.index
    period_codes = index.to_period("M").astype(str)
    arrays["period"] = np.where(period_codes == MONTHS[0], 0, 1).astype(np.int16)
    arrays["hour"] = index.hour.to_numpy(np.int16)
    arrays["weekday"] = index.weekday.to_numpy(np.int16)
    arrays["day"] = index.day.to_numpy(np.int16)
    arrays["timestamp"] = (index.view("int64") // 1_000_000).astype(np.int64)
    return arrays


def pack_configs(configs: list[Config]) -> np.ndarray:
    p = np.zeros((len(configs), 27), dtype=np.float64)
    for i, cfg in enumerate(configs):
        p[i] = [
            cfg.family_mask, cfg.direction, cfg.rr, cfg.sl_atr, cfg.min_stop_pct,
            cfg.max_hold, cfg.cooldown, cfg.session_start, cfg.session_end, cfg.weekday_mask,
            cfg.htf_mode, cfg.adx_min, cfg.rsi_long_min, cfg.rsi_long_max, cfg.relvol_min,
            cfg.body_min, cfg.close_loc_min, cfg.wick_min, cfg.eff_min, cfg.chop_max,
            cfg.atr_rank_min, cfg.vwap_dev_max, cfg.pull_idx, cfg.sweep_idx, cfg.retest_idx,
            cfg.trail_trigger, cfg.trail_lock,
        ]
    return p


@njit(cache=True)
def allowed_time(hour: int, weekday: int, start: int, end: int, mask: int) -> bool:
    if ((mask >> weekday) & 1) == 0:
        return False
    if start == 0 and end == 24:
        return True
    return start <= hour < end


@njit(cache=True)
def htf_ok(long_side, mode, m15_long, m15_short, h1_long, h1_short, h4_long, h4_short, i):
    if long_side:
        if mode == 0:
            return h1_long[i] and m15_long[i] and not h4_short[i]
        if mode == 1:
            return h4_long[i] and h1_long[i]
        return h4_long[i] and h1_long[i] and m15_long[i]
    if mode == 0:
        return h1_short[i] and m15_short[i] and not h4_long[i]
    if mode == 1:
        return h4_short[i] and h1_short[i]
    return h4_short[i] and h1_short[i] and m15_short[i]


@njit(cache=True)
def family_signal(
    i, long_side, p, o, h, l, c, atr, ema8, ema21, ema55, ema200, vwap, vwap_dev,
    plus_di, minus_di, adx, rsi, macd_hist, macd_slope, rel_vol, body, close_loc,
    upper_wick, lower_wick, range_exp, ret1, ret3, ret12, eff12, eff24, chop, atr_rank,
    m15_long, m15_short, h1_long, h1_short, h4_long, h4_short,
    pull_l, pull_s, sweep_l, sweep_s, retest_l, retest_s,
):
    mask = int(p[0])
    mode = int(p[10])
    adx_min = p[11]
    rsi_lo = p[12]
    rsi_hi = p[13]
    rv_min = p[14]
    body_min = p[15]
    loc_min = p[16]
    wick_min = p[17]
    eff_min = p[18]
    chop_max = p[19]
    atr_min = p[20]
    vwap_max = p[21]
    pull_idx = int(p[22])
    sweep_idx = int(p[23])
    retest_idx = int(p[24])

    trend_ok = htf_ok(long_side, mode, m15_long, m15_short, h1_long, h1_short, h4_long, h4_short, i)
    trend_sig = False
    sweep_sig = False
    retest_sig = False

    if long_side:
        candle_ok = c[i] > o[i] and body[i] >= body_min and close_loc[i] >= loc_min
        trend_sig = (
            (mask & 1) != 0 and trend_ok and pull_l[i, pull_idx]
            and c[i] > ema8[i] and ema8[i] > ema21[i] and ema21[i] > ema55[i] and c[i] > ema200[i]
            and adx[i] >= adx_min and plus_di[i] > minus_di[i]
            and rsi[i] >= rsi_lo and rsi[i] <= rsi_hi
            and macd_hist[i] > 0 and macd_slope[i] > 0
            and eff12[i] >= eff_min and chop[i] <= chop_max
            and rel_vol[i] >= rv_min and atr_rank[i] >= atr_min
            and vwap_dev[i] >= -0.30 and vwap_dev[i] <= vwap_max
            and candle_ok
        )
        sweep_sig = (
            (mask & 2) != 0 and sweep_l[i, sweep_idx]
            and not h4_short[i] and not h1_short[i]
            and lower_wick[i] >= wick_min and close_loc[i] >= loc_min
            and rsi[i] <= min(54.0, rsi_lo + 8.0) and rsi[i] > rsi[i - 1]
            and macd_slope[i] > 0 and rel_vol[i] >= rv_min * 0.80
            and range_exp[i] >= 0.90 and abs(vwap_dev[i]) <= vwap_max
        )
        retest_sig = (
            (mask & 4) != 0 and trend_ok and retest_l[i, retest_idx]
            and c[i] > ema21[i] and ema21[i] > ema55[i] and c[i] > ema200[i]
            and adx[i] >= adx_min * 0.85 and plus_di[i] > minus_di[i]
            and rsi[i] >= max(48.0, rsi_lo - 4.0) and rsi[i] <= rsi_hi + 4.0
            and macd_hist[i] > 0 and eff24[i] >= eff_min * 0.85
            and rel_vol[i] >= rv_min * 0.85 and candle_ok
        )
    else:
        candle_ok = c[i] < o[i] and body[i] >= body_min and close_loc[i] <= 1.0 - loc_min
        trend_sig = (
            (mask & 1) != 0 and trend_ok and pull_s[i, pull_idx]
            and c[i] < ema8[i] and ema8[i] < ema21[i] and ema21[i] < ema55[i] and c[i] < ema200[i]
            and adx[i] >= adx_min and minus_di[i] > plus_di[i]
            and rsi[i] <= 100.0 - rsi_lo and rsi[i] >= 100.0 - rsi_hi
            and macd_hist[i] < 0 and macd_slope[i] < 0
            and eff12[i] >= eff_min and chop[i] <= chop_max
            and rel_vol[i] >= rv_min and atr_rank[i] >= atr_min
            and vwap_dev[i] <= 0.30 and vwap_dev[i] >= -vwap_max
            and candle_ok
        )
        sweep_sig = (
            (mask & 2) != 0 and sweep_s[i, sweep_idx]
            and not h4_long[i] and not h1_long[i]
            and upper_wick[i] >= wick_min and close_loc[i] <= 1.0 - loc_min
            and rsi[i] >= max(46.0, 92.0 - rsi_lo) and rsi[i] < rsi[i - 1]
            and macd_slope[i] < 0 and rel_vol[i] >= rv_min * 0.80
            and range_exp[i] >= 0.90 and abs(vwap_dev[i]) <= vwap_max
        )
        retest_sig = (
            (mask & 4) != 0 and trend_ok and retest_s[i, retest_idx]
            and c[i] < ema21[i] and ema21[i] < ema55[i] and c[i] < ema200[i]
            and adx[i] >= adx_min * 0.85 and minus_di[i] > plus_di[i]
            and rsi[i] <= min(52.0, 104.0 - rsi_lo) and rsi[i] >= 96.0 - rsi_hi
            and macd_hist[i] < 0 and eff24[i] >= eff_min * 0.85
            and rel_vol[i] >= rv_min * 0.85 and candle_ok
        )

    if trend_sig:
        return 1
    if sweep_sig:
        return 2
    if retest_sig:
        return 3
    return 0


@njit(parallel=True, cache=True)
def evaluate_many(
    params, o, h, l, c, atr, ema8, ema21, ema55, ema200, vwap, vwap_dev,
    plus_di, minus_di, adx, rsi, macd_hist, macd_slope, rel_vol, body, close_loc,
    upper_wick, lower_wick, range_exp, ret1, ret3, ret12, eff12, eff24, chop, atr_rank,
    m15_long, m15_short, h1_long, h1_short, h4_long, h4_short,
    pull_l, pull_s, sweep_l, sweep_s, retest_l, retest_s, period, hour, weekday, day,
):
    ncfg = params.shape[0]
    out = np.zeros((ncfg, 26), dtype=np.float64)
    n = len(c)
    for q in prange(ncfg):
        p = params[q]
        pos = 0
        entry = stop = target = risk = 0.0
        entry_i = last_exit = -100000
        entry_period = 0
        trail_armed = False

        count0 = wins0 = count1 = wins1 = 0
        win_sum0 = loss_sum0 = win_sum1 = loss_sum1 = 0.0
        win_n0 = loss_n0 = win_n1 = loss_n1 = 0
        half_count_a = half_count_b = half_wins_a = half_wins_b = 0
        half_win_sum_a = half_loss_sum_a = half_win_sum_b = half_loss_sum_b = 0.0
        half_win_n_a = half_loss_n_a = half_win_n_b = half_loss_n_b = 0
        half_net_a = half_net_b = 0.0
        total_r = 0.0
        train_cum = train_peak = train_max_dd = 0.0
        long_count = short_count = 0

        for i in range(1200, n):
            if pos != 0:
                exit_price = 0.0
                done = False
                if pos > 0:
                    if l[i] <= stop:
                        exit_price, done = stop - SLIPPAGE_ABS, True
                    elif h[i] >= target:
                        exit_price, done = target - SLIPPAGE_ABS, True
                else:
                    if h[i] >= stop:
                        exit_price, done = stop + SLIPPAGE_ABS, True
                    elif l[i] <= target:
                        exit_price, done = target + SLIPPAGE_ABS, True
                if not done and i - entry_i >= int(p[5]):
                    exit_price = c[i] - SLIPPAGE_ABS if pos > 0 else c[i] + SLIPPAGE_ABS
                    done = True

                if done:
                    gross = (exit_price - entry) * pos
                    fees = FEE_RATE * (entry + exit_price)
                    net_r = (gross - fees) / risk
                    total_r += net_r
                    if entry_period == 0:
                        train_cum += net_r
                        train_peak = max(train_peak, train_cum)
                        train_max_dd = max(train_max_dd, train_peak - train_cum)
                        count0 += 1
                        first_half = day[entry_i] <= 15
                        if first_half:
                            half_count_a += 1
                            half_net_a += net_r
                        else:
                            half_count_b += 1
                            half_net_b += net_r
                        if net_r > 0:
                            wins0 += 1
                            win_sum0 += net_r
                            win_n0 += 1
                            if first_half:
                                half_wins_a += 1
                                half_win_sum_a += net_r
                                half_win_n_a += 1
                            else:
                                half_wins_b += 1
                                half_win_sum_b += net_r
                                half_win_n_b += 1
                        else:
                            loss_sum0 += -net_r
                            loss_n0 += 1
                            if first_half:
                                half_loss_sum_a += -net_r
                                half_loss_n_a += 1
                            else:
                                half_loss_sum_b += -net_r
                                half_loss_n_b += 1
                    else:
                        count1 += 1
                        if net_r > 0:
                            wins1 += 1
                            win_sum1 += net_r
                            win_n1 += 1
                        else:
                            loss_sum1 += -net_r
                            loss_n1 += 1
                    pos = 0
                    last_exit = i
                    trail_armed = False
                else:
                    if not trail_armed:
                        favorable = h[i] - entry if pos > 0 else entry - l[i]
                        if favorable >= p[25] * risk:
                            trail_armed = True
                    if trail_armed:
                        locked = entry + p[26] * risk if pos > 0 else entry - p[26] * risk
                        if pos > 0:
                            stop = max(stop, locked)
                        else:
                            stop = min(stop, locked)
                continue

            if i - last_exit < int(p[6]):
                continue
            if not allowed_time(int(hour[i]), int(weekday[i]), int(p[7]), int(p[8]), int(p[9])):
                continue

            direction = int(p[1])
            long_family = 0
            short_family = 0
            if direction != 2:
                long_family = family_signal(
                    i, True, p, o, h, l, c, atr, ema8, ema21, ema55, ema200, vwap, vwap_dev,
                    plus_di, minus_di, adx, rsi, macd_hist, macd_slope, rel_vol, body, close_loc,
                    upper_wick, lower_wick, range_exp, ret1, ret3, ret12, eff12, eff24, chop, atr_rank,
                    m15_long, m15_short, h1_long, h1_short, h4_long, h4_short,
                    pull_l, pull_s, sweep_l, sweep_s, retest_l, retest_s,
                )
            if direction != 1:
                short_family = family_signal(
                    i, False, p, o, h, l, c, atr, ema8, ema21, ema55, ema200, vwap, vwap_dev,
                    plus_di, minus_di, adx, rsi, macd_hist, macd_slope, rel_vol, body, close_loc,
                    upper_wick, lower_wick, range_exp, ret1, ret3, ret12, eff12, eff24, chop, atr_rank,
                    m15_long, m15_short, h1_long, h1_short, h4_long, h4_short,
                    pull_l, pull_s, sweep_l, sweep_s, retest_l, retest_s,
                )
            if (long_family > 0) == (short_family > 0):
                continue

            pos = 1 if long_family > 0 else -1
            entry = c[i] + SLIPPAGE_ABS if pos > 0 else c[i] - SLIPPAGE_ABS
            risk = max(atr[i] * p[3], c[i] * p[4])
            stop = entry - risk if pos > 0 else entry + risk
            target = entry + risk * p[2] if pos > 0 else entry - risk * p[2]
            entry_i = i
            entry_period = int(period[i])
            trail_armed = False
            if entry_period == 0:
                if pos > 0:
                    long_count += 1
                else:
                    short_count += 1

        wr0 = wins0 / count0 if count0 else 0.0
        wr1 = wins1 / count1 if count1 else 0.0
        ratio0 = (win_sum0 / win_n0) / (loss_sum0 / loss_n0) if win_n0 and loss_n0 else 0.0
        ratio1 = (win_sum1 / win_n1) / (loss_sum1 / loss_n1) if win_n1 and loss_n1 else 0.0
        pf0 = win_sum0 / loss_sum0 if loss_sum0 > 0 else 0.0
        pf1 = win_sum1 / loss_sum1 if loss_sum1 > 0 else 0.0
        half_wr_a = half_wins_a / half_count_a if half_count_a else 0.0
        half_wr_b = half_wins_b / half_count_b if half_count_b else 0.0
        half_ratio_a = (half_win_sum_a / half_win_n_a) / (half_loss_sum_a / half_loss_n_a) if half_win_n_a and half_loss_n_a else 0.0
        half_ratio_b = (half_win_sum_b / half_win_n_b) / (half_loss_sum_b / half_loss_n_b) if half_win_n_b and half_loss_n_b else 0.0
        train_net = win_sum0 - loss_sum0

        qualified = (
            MIN_TRADES <= count0 <= MAX_TRADES
            and MIN_TRADES <= count1 <= MAX_TRADES
            and wr0 >= MIN_WIN_RATE and wr1 >= MIN_WIN_RATE
            and ratio0 >= MIN_RATIO and ratio1 >= MIN_RATIO
        )

        count_error = abs(count0 - TARGET_TRADES)
        if count0 < MIN_TRADES or count0 > MAX_TRADES:
            score = -100000.0 - count_error * 1000.0
        else:
            min_half_wr = min(half_wr_a, half_wr_b)
            min_half_ratio = min(half_ratio_a, half_ratio_b)
            score = (
                wr0 * 800.0
                + min_half_wr * 900.0
                + min(ratio0, 4.0) * 150.0
                + min(min_half_ratio, 4.0) * 110.0
                + min(pf0, 8.0) * 30.0
                + train_net * 5.0
                - train_max_dd * 1.5
                - count_error * 7.0
            )
            if wr0 < MIN_WIN_RATE:
                score -= (MIN_WIN_RATE - wr0) * 4200.0
            if ratio0 < MIN_RATIO:
                score -= (MIN_RATIO - ratio0) * 1500.0
            half_min_count = max(5, MIN_TRADES // 3)
            if half_count_a < half_min_count or half_count_b < half_min_count:
                score -= 2000.0
            if half_net_a <= 0 or half_net_b <= 0:
                score -= 1600.0
            if min_half_wr < 0.60:
                score -= (0.60 - min_half_wr) * 3600.0
            if min_half_ratio < 1.20:
                score -= (1.20 - min_half_ratio) * 1200.0
            total_entries = long_count + short_count
            if int(p[1]) == 0 and total_entries > 0:
                concentration = max(long_count, short_count) / total_entries
                if concentration > 0.82:
                    score -= (concentration - 0.82) * 1800.0

        out[q, 0] = score
        out[q, 1] = 1.0 if qualified else 0.0
        out[q, 2] = count0
        out[q, 3] = wr0
        out[q, 4] = ratio0
        out[q, 5] = pf0
        out[q, 6] = count1
        out[q, 7] = wr1
        out[q, 8] = ratio1
        out[q, 9] = pf1
        out[q, 10] = total_r
        out[q, 11] = train_max_dd
        out[q, 12] = wins0
        out[q, 13] = wins1
        out[q, 14] = half_count_a
        out[q, 15] = half_wr_a
        out[q, 16] = half_ratio_a
        out[q, 17] = half_net_a
        out[q, 18] = half_count_b
        out[q, 19] = half_wr_b
        out[q, 20] = half_ratio_b
        out[q, 21] = half_net_b
        out[q, 22] = long_count
        out[q, 23] = short_count
        out[q, 24] = train_net
        out[q, 25] = min(half_wr_a, half_wr_b)
    return out


def run_signal(cfg: Config, i: int, long_side: bool, a: dict[str, np.ndarray]) -> int:
    p = pack_configs([cfg])[0]
    return int(family_signal(
        i, long_side, p, a["open"], a["high"], a["low"], a["close"], a["atr"],
        a["ema8"], a["ema21"], a["ema55"], a["ema200"], a["vwap"], a["vwap_dev"],
        a["plus_di"], a["minus_di"], a["adx"], a["rsi"], a["macd_hist"], a["macd_slope"],
        a["rel_vol"], a["body"], a["close_loc"], a["upper_wick"], a["lower_wick"],
        a["range_exp"], a["ret1"], a["ret3"], a["ret12"], a["eff12"], a["eff24"],
        a["chop"], a["atr_rank"], a["m15_long"], a["m15_short"], a["h1_long"],
        a["h1_short"], a["h4_long"], a["h4_short"], a["pull_l"], a["pull_s"],
        a["sweep_l"], a["sweep_s"], a["retest_l"], a["retest_s"],
    ))


def simulate_detail(cfg: Config, a: dict[str, np.ndarray], x: pd.DataFrame) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    pos = 0
    family = 0
    entry = stop = target = risk = 0.0
    entry_i = last_exit = -100000
    entry_period = 0
    trail_armed = False
    n = len(a["close"])
    for i in range(1200, n):
        if pos:
            exit_price: float | None = None
            reason = ""
            if pos > 0:
                if a["low"][i] <= stop:
                    exit_price, reason = stop - SLIPPAGE_ABS, "TRAIL" if trail_armed else "SL"
                elif a["high"][i] >= target:
                    exit_price, reason = target - SLIPPAGE_ABS, "TP"
            else:
                if a["high"][i] >= stop:
                    exit_price, reason = stop + SLIPPAGE_ABS, "TRAIL" if trail_armed else "SL"
                elif a["low"][i] <= target:
                    exit_price, reason = target + SLIPPAGE_ABS, "TP"
            if exit_price is None and i - entry_i >= cfg.max_hold:
                exit_price = a["close"][i] - SLIPPAGE_ABS if pos > 0 else a["close"][i] + SLIPPAGE_ABS
                reason = "TIME"
            if exit_price is not None:
                net_r = (((exit_price - entry) * pos) - FEE_RATE * (entry + exit_price)) / risk
                trades.append({
                    "entry_time_utc": x.index[entry_i].isoformat(),
                    "exit_time_utc": x.index[i].isoformat(),
                    "month": MONTHS[entry_period],
                    "direction": "LONG" if pos > 0 else "SHORT",
                    "family": ("多周期趋势回踩", "流动性扫单反转", "突破回踩确认")[family - 1],
                    "entry": entry,
                    "exit": exit_price,
                    "stop": stop,
                    "target": target,
                    "risk_abs": risk,
                    "net_R": net_r,
                    "win": bool(net_r > 0),
                    "exit_reason": reason,
                    "bars": i - entry_i,
                })
                pos = 0
                last_exit = i
                trail_armed = False
            else:
                if not trail_armed:
                    favorable = a["high"][i] - entry if pos > 0 else entry - a["low"][i]
                    if favorable >= cfg.trail_trigger * risk:
                        trail_armed = True
                if trail_armed:
                    locked = entry + cfg.trail_lock * risk if pos > 0 else entry - cfg.trail_lock * risk
                    stop = max(stop, locked) if pos > 0 else min(stop, locked)
            continue

        if i - last_exit < cfg.cooldown:
            continue
        if not allowed_time(int(a["hour"][i]), int(a["weekday"][i]), cfg.session_start, cfg.session_end, cfg.weekday_mask):
            continue
        long_family = run_signal(cfg, i, True, a) if cfg.direction != 2 else 0
        short_family = run_signal(cfg, i, False, a) if cfg.direction != 1 else 0
        if (long_family > 0) == (short_family > 0):
            continue
        pos = 1 if long_family > 0 else -1
        family = long_family if long_family > 0 else short_family
        entry = a["close"][i] + SLIPPAGE_ABS if pos > 0 else a["close"][i] - SLIPPAGE_ABS
        risk = max(a["atr"][i] * cfg.sl_atr, a["close"][i] * cfg.min_stop_pct)
        stop = entry - risk if pos > 0 else entry + risk
        target = entry + risk * cfg.rr if pos > 0 else entry - risk * cfg.rr
        entry_i = i
        entry_period = int(a["period"][i])
        trail_armed = False
    return trades


def monthly_stats(trades: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for month in MONTHS:
        z = trades[trades["month"] == month] if not trades.empty else trades
        wins = z[z["net_R"] > 0]["net_R"] if not z.empty else pd.Series(dtype=float)
        losses = -z[z["net_R"] <= 0]["net_R"] if not z.empty else pd.Series(dtype=float)
        result[month] = {
            "trades": int(len(z)),
            "wins": int((z["net_R"] > 0).sum()) if not z.empty else 0,
            "win_rate": float((z["net_R"] > 0).mean()) if len(z) else 0.0,
            "avg_win_R": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss_R": float(losses.mean()) if len(losses) else 0.0,
            "avg_win_loss_ratio": float(wins.mean() / losses.mean()) if len(wins) and len(losses) else 0.0,
            "profit_factor": float(wins.sum() / losses.sum()) if losses.sum() > 0 else 0.0,
            "net_R": float(z["net_R"].sum()) if len(z) else 0.0,
        }
    return result


def call_evaluator(params: np.ndarray, a: dict[str, np.ndarray]) -> np.ndarray:
    return evaluate_many(
        params, a["open"], a["high"], a["low"], a["close"], a["atr"], a["ema8"],
        a["ema21"], a["ema55"], a["ema200"], a["vwap"], a["vwap_dev"], a["plus_di"],
        a["minus_di"], a["adx"], a["rsi"], a["macd_hist"], a["macd_slope"], a["rel_vol"],
        a["body"], a["close_loc"], a["upper_wick"], a["lower_wick"], a["range_exp"],
        a["ret1"], a["ret3"], a["ret12"], a["eff12"], a["eff24"], a["chop"], a["atr_rank"],
        a["m15_long"], a["m15_short"], a["h1_long"], a["h1_short"], a["h4_long"], a["h4_short"],
        a["pull_l"], a["pull_s"], a["sweep_l"], a["sweep_s"], a["retest_l"], a["retest_s"],
        a["period"], a["hour"], a["weekday"], a["day"],
    )


def main() -> None:
    raw, audit = load_official_data()
    x = add_indicators(raw)
    a = build_arrays(x)
    rounds = int(os.environ.get("SEARCH_ROUNDS", str(REQUEST["search_rounds"])))
    batch = int(os.environ.get("SEARCH_BATCH", str(REQUEST["search_batch"])))

    all_configs: list[Config] = []
    all_metrics: list[np.ndarray] = []
    for seed_index in range(SEED_COUNT):
        seed = BASE_SEED + seed_index * 100003
        rng = random.Random(seed)
        for round_index in range(rounds):
            configs = [random_config(rng) for _ in range(batch)]
            metrics = call_evaluator(pack_configs(configs), a)
            keep_n = min(700, len(configs))
            keep = np.argsort(metrics[:, 0])[-keep_n:]
            all_configs.extend(configs[int(i)] for i in keep)
            all_metrics.extend(metrics[int(i)].copy() for i in keep)
            print(
                f"seed {seed_index + 1}/{SEED_COUNT}, round {round_index + 1}/{rounds}, "
                f"best={metrics[keep[-1], 0]:.3f}, qualified={int(metrics[:, 1].sum())}"
            )

    matrix = np.vstack(all_metrics)
    order = np.argsort(matrix[:, 0])[::-1]
    best_idx = int(order[0])
    cfg = all_configs[best_idx]
    trades = pd.DataFrame(simulate_detail(cfg, a, x))
    stats = monthly_stats(trades)
    qualified = all(
        MIN_TRADES <= stats[m]["trades"] <= MAX_TRADES
        and stats[m]["win_rate"] >= MIN_WIN_RATE
        and stats[m]["avg_win_loss_ratio"] >= MIN_RATIO
        for m in MONTHS
    )

    (RESULTS / "request.json").write_text(json.dumps(REQUEST, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "best_config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    trades.to_csv(RESULTS / "trades.csv", index=False)

    rows = []
    for rank, ix in enumerate(order[:200], 1):
        metric = matrix[int(ix)]
        c0 = all_configs[int(ix)]
        rows.append({
            "rank": rank,
            "score_train_only": metric[0],
            "diagnostic_qualified": bool(metric[1]),
            "family": FAMILY_NAMES[c0.family_mask],
            "direction": ("双向", "只做多", "只做空")[c0.direction],
            "may_trades": metric[2],
            "may_win_rate": metric[3],
            "may_ratio": metric[4],
            "june_trades": metric[6],
            "june_win_rate": metric[7],
            "june_ratio": metric[8],
            "total_net_R": metric[10],
            "may_max_drawdown_R": metric[11],
            "first_half_trades": metric[14],
            "first_half_win_rate": metric[15],
            "first_half_ratio": metric[16],
            "first_half_net_R": metric[17],
            "second_half_trades": metric[18],
            "second_half_win_rate": metric[19],
            "second_half_ratio": metric[20],
            "second_half_net_R": metric[21],
            "config": json.dumps(asdict(c0), ensure_ascii=False),
        })
    pd.DataFrame(rows).to_csv(RESULTS / "top_candidates.csv", index=False)

    status = {
        "qualified": qualified,
        "engine": "BTC 5m price-action structure V4",
        "selected_by": f"{MONTHS[0]} only, with half-month stability; {MONTHS[1]} untouched OOS",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "months": list(MONTHS),
        "constraints": {
            "min_trades": MIN_TRADES,
            "max_trades": MAX_TRADES,
            "min_win_rate": MIN_WIN_RATE,
            "min_avg_win_loss_ratio": MIN_RATIO,
        },
        "family": FAMILY_NAMES[cfg.family_mask],
        "direction": ("双向", "只做多", "只做空")[cfg.direction],
        "monthly_stats": stats,
        "search_size": rounds * batch * SEED_COUNT,
        "config": asdict(cfg),
    }
    (RESULTS / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# BTCUSDT 5m 价格行为结构 V4 回测报告",
        "",
        "- 引擎：多周期趋势回踩、流动性扫单反转、突破回踩确认。",
        f"- 数据审计：{'通过' if audit['passed'] else '失败'}，{audit['actual_rows']:,}根，缺失{audit['missing_rows']}，重复{audit['duplicate_timestamps']}。",
        f"- 搜索规模：{rounds * batch * SEED_COUNT:,}组可复现参数。",
        f"- 选择方法：只使用{MONTHS[0]}及其前后半月稳定性；{MONTHS[1]}保持样本外。",
        f"- 成本：单边手续费{FEE_RATE * 100:.3f}%；每次成交滑点{SLIPPAGE_ABS:.1f} USDT。",
        f"- 最终验收：**{'达到全部要求' if qualified else '未达到全部要求'}**。",
        "",
        "## 策略结构",
        "",
        f"- 启用结构：{FAMILY_NAMES[cfg.family_mask]}",
        f"- 交易方向：{('双向', '只做多', '只做空')[cfg.direction]}",
        f"- 固定目标：{cfg.rr:.2f}R；达到{cfg.trail_trigger:.2f}R后锁定{cfg.trail_lock:.2f}R。",
        "",
        "## 月度结果",
        "",
        "| 月份 | 交易 | 胜率 | 平均盈利/平均亏损 | 盈利因子 | 净R |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for month in MONTHS:
        s = stats[month]
        report.append(
            f"| {month} | {s['trades']} | {s['win_rate'] * 100:.2f}% | "
            f"{s['avg_win_loss_ratio']:.3f} | {s['profit_factor']:.3f} | {s['net_R']:.3f} |"
        )
    report += [
        "",
        "## 说明",
        "",
        "回测只使用当前及历史K线；15分钟、1小时和4小时过滤均读取上一根已完成K线。",
        "同一根K线同时触及止损和止盈时按止损优先；移动止损在触发后的下一根K线生效。",
        "参数排名不使用样本外月份。历史通过不代表未来收益保证。",
    ]
    (RESULTS / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
