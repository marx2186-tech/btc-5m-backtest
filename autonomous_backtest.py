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
RESULTS = ROOT / "results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

DEFAULT_REQUEST: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "interval": "5m",
    "months": ["2026-05", "2026-06"],
    "fee_rate_per_side": 0.0005,
    "tick_size": 0.1,
    "slippage_ticks_per_fill": 2,
    "min_trades_per_month": 20,
    "max_trades_per_month": 30,
    "min_win_rate": 0.70,
    "min_avg_win_loss_ratio": 1.50,
    "search_rounds": 6,
    "search_batch": 25000,
    "seed_count": 4,
    "base_seed": 20260727,
}


def load_request() -> dict[str, Any]:
    req = dict(DEFAULT_REQUEST)
    request_file = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(ROOT / "request.json")))
    if request_file.exists():
        loaded = json.loads(request_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("request.json must contain a JSON object")
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
MONTHS = tuple(str(x) for x in REQUEST["months"])
if len(MONTHS) != 2:
    raise ValueError("Exactly two complete calendar months are required")
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
    headers = {"User-Agent": "btc-regime-backtest/3.0"}
    for k in range(attempts):
        try:
            response = requests.get(url, timeout=90, headers=headers)
            response.raise_for_status()
            path.write_bytes(response.content)
            return response.content
        except Exception as exc:
            last = exc
            time.sleep(2 ** min(k, 4))
    raise RuntimeError(f"Download failed: {url}: {last}")


def load_official_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    files: list[dict[str, Any]] = []
    for month in MONTHS:
        name = f"{SYMBOL}-{INTERVAL}-{month}.zip"
        base = f"https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{INTERVAL}"
        raw = download(f"{base}/{name}", CACHE / name)
        checksum_text = download(f"{base}/{name}.CHECKSUM", CACHE / f"{name}.CHECKSUM").decode("utf-8").strip()
        expected_hash = checksum_text.split()[0].lower()
        actual_hash = hashlib.sha256(raw).hexdigest().lower()
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch for {name}")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = [x for x in archive.namelist() if x.lower().endswith(".csv")]
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
    duplicated = int(pd.Series(times).duplicated().sum())
    unique_times = np.unique(times)
    missing = np.setdiff1d(expected_times, unique_times)
    extra = np.setdiff1d(unique_times, expected_times)
    off_grid = int(np.sum((times - START_MS) % STEP_MS != 0))
    bad_close_time = int(np.sum(data["close_time"].astype("int64").to_numpy() != times + STEP_MS - 1))
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
        "invalid_close_time_rows": bad_close_time,
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
        and bad_close_time == 0
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

    basis = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    lower, upper = basis - 2 * sd, basis + 2 * sd
    x["bb_pos"] = (c - lower) / (upper - lower).replace(0, np.nan)
    x["bb_width"] = (upper - lower) / basis.replace(0, np.nan)
    x["bb_rank"] = x["bb_width"].rolling(288).rank(pct=True)
    x["squeeze_release"] = (x["bb_rank"].shift(1) <= 0.35) & (x["bb_rank"] > x["bb_rank"].shift(1))

    utc_day = x.index.floor("D")
    typical = (h + l + c) / 3
    x["vwap"] = (typical * v).groupby(utc_day).cumsum() / v.groupby(utc_day).cumsum().replace(0, np.nan)
    x["vwap_dev"] = (c - x["vwap"]) / x["atr"].replace(0, np.nan)
    x["vol_z"] = (v - v.rolling(48).mean()) / v.rolling(48).std(ddof=0).replace(0, np.nan)

    candle_range = (h - l).replace(0, np.nan)
    x["body"] = (c - o).abs() / candle_range
    x["close_loc"] = (c - l) / candle_range
    for n in (1, 3, 6, 12, 24):
        x[f"ret{n}"] = c.pct_change(n)

    abs_change = c.diff().abs()
    for n in (12, 24):
        x[f"eff{n}"] = (c - c.shift(n)).abs() / abs_change.rolling(n).sum().replace(0, np.nan)

    x["don12h"] = h.shift(1).rolling(12).max()
    x["don12l"] = l.shift(1).rolling(12).min()
    x["don24h"] = h.shift(1).rolling(24).max()
    x["don24l"] = l.shift(1).rolling(24).min()

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

    pull_tols = (0.15, 0.30, 0.50)
    pull_bars = (3, 6, 12)
    for ti, tol in enumerate(pull_tols):
        touch_l = (l <= x["ema21"] + tol * x["atr"]) & (l >= x["ema55"] - x["atr"])
        touch_s = (h >= x["ema21"] - tol * x["atr"]) & (h <= x["ema55"] + x["atr"])
        for bi, bars in enumerate(pull_bars):
            x[f"pl_{ti}_{bi}"] = touch_l.rolling(bars).max().fillna(0).astype(bool) & (c > x["ema8"])
            x[f"ps_{ti}_{bi}"] = touch_s.rolling(bars).max().fillna(0).astype(bool) & (c < x["ema8"])

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
    adx_trend_min: float
    adx_range_max: float
    rsi_long_min: float
    rsi_long_max: float
    volz_break: float
    body_min: float
    close_loc_min: float
    eff_min: float
    atr_rank_min: float
    squeeze_rank_max: float
    vwap_dev_entry: float
    pull_idx: int
    don_idx: int
    trail_trigger: float
    trail_lock: float


FAMILY_NAMES = {
    1: "趋势回踩",
    2: "压缩突破",
    3: "趋势回踩+压缩突破",
    4: "震荡反转",
    5: "趋势回踩+震荡反转",
    6: "压缩突破+震荡反转",
    7: "三策略组合",
}


def random_config(rng: random.Random) -> Config:
    if rng.random() < 0.72:
        start, end = 0, 24
    else:
        start = rng.randrange(0, 17)
        end = min(24, start + rng.randrange(6, 13))
    rr = rng.choice((2.2, 2.4, 2.6, 2.8, 3.0))
    trigger = rng.uniform(1.90, min(2.60, rr - 0.10))
    lock = rng.uniform(1.75, max(1.76, min(trigger - 0.05, rr - 0.15)))
    return Config(
        family_mask=rng.choices((1, 2, 3, 4, 5, 6, 7), weights=(5, 5, 8, 1, 2, 2, 2))[0],
        direction=rng.choices((0, 1, 2), weights=(14, 1, 1))[0],
        rr=rr,
        sl_atr=rng.uniform(0.90, 2.80),
        min_stop_pct=rng.uniform(0.0030, 0.0090),
        max_hold=rng.choice((144, 216, 288, 432, 576)),
        cooldown=rng.randrange(3, 37),
        session_start=start,
        session_end=end,
        weekday_mask=rng.choice((127, 127, 127, 31, 62, 124)),
        adx_trend_min=rng.uniform(14, 36),
        adx_range_max=rng.uniform(12, 25),
        rsi_long_min=rng.uniform(38, 55),
        rsi_long_max=rng.uniform(58, 76),
        volz_break=rng.uniform(-0.30, 1.80),
        body_min=rng.uniform(0.22, 0.68),
        close_loc_min=rng.uniform(0.58, 0.88),
        eff_min=rng.uniform(0.18, 0.72),
        atr_rank_min=rng.uniform(0.15, 0.78),
        squeeze_rank_max=rng.uniform(0.12, 0.55),
        vwap_dev_entry=rng.uniform(0.55, 2.40),
        pull_idx=rng.randrange(9),
        don_idx=rng.randrange(2),
        trail_trigger=trigger,
        trail_lock=lock,
    )


def build_arrays(x: pd.DataFrame) -> dict[str, np.ndarray]:
    keys = [
        "open", "high", "low", "close", "atr", "ema8", "ema21", "ema55", "ema200",
        "vwap", "vwap_dev", "plus_di", "minus_di", "adx", "rsi", "bb_pos", "bb_rank",
        "vol_z", "body", "close_loc", "ret1", "ret3", "ret12", "eff12", "eff24",
        "atr_rank", "don12h", "don12l", "don24h", "don24l",
    ]
    arrays = {key: x[key].to_numpy(np.float64) for key in keys}
    for key in ("squeeze_release", "m15_long", "m15_short", "h1_long", "h1_short"):
        arrays[key] = x[key].to_numpy(np.bool_)
    arrays["pull_l"] = np.column_stack(
        [x[f"pl_{ti}_{bi}"].to_numpy(np.bool_) for ti in range(3) for bi in range(3)]
    )
    arrays["pull_s"] = np.column_stack(
        [x[f"ps_{ti}_{bi}"].to_numpy(np.bool_) for ti in range(3) for bi in range(3)]
    )
    index = x.index
    period_codes = index.to_period("M").astype(str)
    arrays["period"] = np.where(period_codes == MONTHS[0], 0, 1).astype(np.int16)
    arrays["hour"] = index.hour.to_numpy(np.int16)
    arrays["weekday"] = index.weekday.to_numpy(np.int16)
    arrays["day"] = index.day.to_numpy(np.int16)
    arrays["timestamp"] = (index.view("int64") // 1_000_000).astype(np.int64)
    return arrays


def pack_configs(configs: list[Config]) -> np.ndarray:
    p = np.zeros((len(configs), 25), dtype=np.float64)
    for i, cfg in enumerate(configs):
        p[i] = [
            cfg.family_mask, cfg.direction, cfg.rr, cfg.sl_atr, cfg.min_stop_pct,
            cfg.max_hold, cfg.cooldown, cfg.session_start, cfg.session_end, cfg.weekday_mask,
            cfg.adx_trend_min, cfg.adx_range_max, cfg.rsi_long_min, cfg.rsi_long_max,
            cfg.volz_break, cfg.body_min, cfg.close_loc_min, cfg.eff_min, cfg.atr_rank_min,
            cfg.squeeze_rank_max, cfg.vwap_dev_entry, cfg.pull_idx, cfg.don_idx,
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
def family_signal(
    i, long_side, p, o, h, l, c, atr, ema8, ema21, ema55, ema200, vwap, vwap_dev,
    plus_di, minus_di, adx, rsi, bb_pos, bb_rank, squeeze_release, vol_z, body,
    close_loc, ret1, ret3, ret12, eff12, eff24, atr_rank, don12h, don12l, don24h,
    don24l, m15_long, m15_short, h1_long, h1_short, pull_l, pull_s,
):
    mask = int(p[0])
    adx_trend = p[10]
    adx_range = p[11]
    rsi_lo = p[12]
    rsi_hi = p[13]
    vz = p[14]
    bmin = p[15]
    locmin = p[16]
    effmin = p[17]
    atrmin = p[18]
    squeeze_max = p[19]
    vdev = p[20]
    pull_idx = int(p[21])
    don_idx = int(p[22])

    trend_sig = False
    break_sig = False
    range_sig = False

    if long_side:
        candle_ok = c[i] > o[i] and body[i] >= bmin and close_loc[i] >= locmin
        trend_sig = (
            (mask & 1) != 0
            and h1_long[i] and m15_long[i]
            and pull_l[i, pull_idx]
            and c[i] > ema8[i] and ema21[i] > ema55[i] and c[i] > ema200[i]
            and adx[i] >= adx_trend and plus_di[i] > minus_di[i]
            and rsi[i] >= rsi_lo and rsi[i] <= rsi_hi
            and eff12[i] >= effmin * 0.70
            and candle_ok
        )
        don = don12h[i] if don_idx == 0 else don24h[i]
        break_sig = (
            (mask & 2) != 0
            and (h1_long[i] or m15_long[i]) and not h1_short[i]
            and c[i] > don and c[i] > vwap[i] and c[i] > ema55[i]
            and adx[i] >= adx_trend and plus_di[i] > minus_di[i]
            and vol_z[i] >= vz and atr_rank[i] >= atrmin
            and (squeeze_release[i] or bb_rank[i - 1] <= squeeze_max)
            and eff24[i] >= effmin and ret3[i] > 0 and ret12[i] > 0
            and candle_ok
        )
        range_sig = (
            (mask & 4) != 0
            and not h1_short[i] and not m15_short[i]
            and adx[i] <= adx_range and eff12[i] <= max(0.18, effmin * 0.75)
            and vwap_dev[i] <= -vdev and bb_pos[i] <= 0.20
            and rsi[i] <= min(47.0, rsi_lo + 4.0)
            and c[i] > o[i] and body[i] >= bmin * 0.75 and close_loc[i] >= locmin
        )
    else:
        candle_ok = c[i] < o[i] and body[i] >= bmin and close_loc[i] <= 1.0 - locmin
        trend_sig = (
            (mask & 1) != 0
            and h1_short[i] and m15_short[i]
            and pull_s[i, pull_idx]
            and c[i] < ema8[i] and ema21[i] < ema55[i] and c[i] < ema200[i]
            and adx[i] >= adx_trend and minus_di[i] > plus_di[i]
            and rsi[i] <= 100.0 - rsi_lo and rsi[i] >= 100.0 - rsi_hi
            and eff12[i] >= effmin * 0.70
            and candle_ok
        )
        don = don12l[i] if don_idx == 0 else don24l[i]
        break_sig = (
            (mask & 2) != 0
            and (h1_short[i] or m15_short[i]) and not h1_long[i]
            and c[i] < don and c[i] < vwap[i] and c[i] < ema55[i]
            and adx[i] >= adx_trend and minus_di[i] > plus_di[i]
            and vol_z[i] >= vz and atr_rank[i] >= atrmin
            and (squeeze_release[i] or bb_rank[i - 1] <= squeeze_max)
            and eff24[i] >= effmin and ret3[i] < 0 and ret12[i] < 0
            and candle_ok
        )
        range_sig = (
            (mask & 4) != 0
            and not h1_long[i] and not m15_long[i]
            and adx[i] <= adx_range and eff12[i] <= max(0.18, effmin * 0.75)
            and vwap_dev[i] >= vdev and bb_pos[i] >= 0.80
            and rsi[i] >= max(53.0, 96.0 - rsi_lo)
            and c[i] < o[i] and body[i] >= bmin * 0.75 and close_loc[i] <= 1.0 - locmin
        )

    if trend_sig:
        return 1
    if break_sig:
        return 2
    if range_sig:
        return 3
    return 0


@njit(parallel=True, cache=True)
def evaluate_many(
    params, o, h, l, c, atr, ema8, ema21, ema55, ema200, vwap, vwap_dev,
    plus_di, minus_di, adx, rsi, bb_pos, bb_rank, squeeze_release, vol_z, body,
    close_loc, ret1, ret3, ret12, eff12, eff24, atr_rank, don12h, don12l, don24h,
    don24l, m15_long, m15_short, h1_long, h1_short, pull_l, pull_s, period, hour,
    weekday, day,
):
    ncfg = params.shape[0]
    out = np.zeros((ncfg, 24), dtype=np.float64)
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
        half_count_a = half_count_b = 0
        half_net_a = half_net_b = 0.0
        half_wins_a = half_wins_b = 0
        total_r = 0.0
        train_cum = train_peak = train_max_dd = 0.0
        long_count = short_count = 0
        family1 = family2 = family3 = 0

        for i in range(600, n):
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
                        if day[entry_i] <= 15:
                            half_count_a += 1
                            half_net_a += net_r
                            if net_r > 0:
                                half_wins_a += 1
                        else:
                            half_count_b += 1
                            half_net_b += net_r
                            if net_r > 0:
                                half_wins_b += 1
                        if net_r > 0:
                            wins0 += 1
                            win_sum0 += net_r
                            win_n0 += 1
                        else:
                            loss_sum0 += -net_r
                            loss_n0 += 1
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
                        if favorable >= p[23] * risk:
                            trail_armed = True
                    if trail_armed:
                        locked = entry + p[24] * risk if pos > 0 else entry - p[24] * risk
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
                    i, True, p, o, h, l, c, atr, ema8, ema21, ema55, ema200, vwap,
                    vwap_dev, plus_di, minus_di, adx, rsi, bb_pos, bb_rank,
                    squeeze_release, vol_z, body, close_loc, ret1, ret3, ret12,
                    eff12, eff24, atr_rank, don12h, don12l, don24h, don24l,
                    m15_long, m15_short, h1_long, h1_short, pull_l, pull_s,
                )
            if direction != 1:
                short_family = family_signal(
                    i, False, p, o, h, l, c, atr, ema8, ema21, ema55, ema200, vwap,
                    vwap_dev, plus_di, minus_di, adx, rsi, bb_pos, bb_rank,
                    squeeze_release, vol_z, body, close_loc, ret1, ret3, ret12,
                    eff12, eff24, atr_rank, don12h, don12l, don24h, don24l,
                    m15_long, m15_short, h1_long, h1_short, pull_l, pull_s,
                )
            if (long_family > 0) == (short_family > 0):
                continue

            pos = 1 if long_family > 0 else -1
            family = long_family if long_family > 0 else short_family
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
            if family == 1:
                family1 += 1
            elif family == 2:
                family2 += 1
            else:
                family3 += 1

        wr0 = wins0 / count0 if count0 else 0.0
        wr1 = wins1 / count1 if count1 else 0.0
        ratio0 = (win_sum0 / win_n0) / (loss_sum0 / loss_n0) if win_n0 and loss_n0 else 0.0
        ratio1 = (win_sum1 / win_n1) / (loss_sum1 / loss_n1) if win_n1 and loss_n1 else 0.0
        pf0 = win_sum0 / loss_sum0 if loss_sum0 > 0 else 0.0
        pf1 = win_sum1 / loss_sum1 if loss_sum1 > 0 else 0.0
        half_wr_a = half_wins_a / half_count_a if half_count_a else 0.0
        half_wr_b = half_wins_b / half_count_b if half_count_b else 0.0
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
            score = (
                wr0 * 700.0
                + min(ratio0, 4.0) * 130.0
                + min(pf0, 8.0) * 35.0
                + train_net * 5.0
                - train_max_dd * 1.2
                - count_error * 10.0
            )
            if wr0 < MIN_WIN_RATE:
                score -= (MIN_WIN_RATE - wr0) * 3200.0
            if ratio0 < MIN_RATIO:
                score -= (MIN_RATIO - ratio0) * 1200.0
            if win_n0 == 0 or loss_n0 == 0:
                score -= 6000.0
            if half_count_a < 7 or half_count_b < 7:
                score -= 900.0
            if half_net_a <= 0 or half_net_b <= 0:
                score -= 700.0
            if half_wr_a < 0.55:
                score -= (0.55 - half_wr_a) * 900.0
            if half_wr_b < 0.55:
                score -= (0.55 - half_wr_b) * 900.0
            total_entries = long_count + short_count
            if int(p[1]) == 0 and total_entries > 0:
                concentration = max(long_count, short_count) / total_entries
                if concentration > 0.88:
                    score -= (concentration - 0.88) * 900.0

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
        out[q, 16] = half_net_a
        out[q, 17] = half_count_b
        out[q, 18] = half_wr_b
        out[q, 19] = half_net_b
        out[q, 20] = long_count
        out[q, 21] = short_count
        out[q, 22] = family1
        out[q, 23] = family2 + family3
    return out


def run_signal(cfg: Config, i: int, long_side: bool, a: dict[str, np.ndarray]) -> int:
    p = pack_configs([cfg])[0]
    return int(family_signal(
        i, long_side, p, a["open"], a["high"], a["low"], a["close"], a["atr"],
        a["ema8"], a["ema21"], a["ema55"], a["ema200"], a["vwap"], a["vwap_dev"],
        a["plus_di"], a["minus_di"], a["adx"], a["rsi"], a["bb_pos"], a["bb_rank"],
        a["squeeze_release"], a["vol_z"], a["body"], a["close_loc"], a["ret1"],
        a["ret3"], a["ret12"], a["eff12"], a["eff24"], a["atr_rank"], a["don12h"],
        a["don12l"], a["don24h"], a["don24l"], a["m15_long"], a["m15_short"],
        a["h1_long"], a["h1_short"], a["pull_l"], a["pull_s"],
    ))


def simulate_detail(cfg: Config, a: dict[str, np.ndarray], x: pd.DataFrame) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    pos = 0
    entry = stop = target = risk = 0.0
    entry_i = last_exit = -100000
    entry_period = 0
    family = 0
    trail_armed = False
    n = len(a["close"])

    for i in range(600, n):
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
                    "family": ("趋势回踩", "压缩突破", "震荡反转")[family - 1],
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
        subset = trades[trades["month"] == month] if len(trades) else trades
        wins = subset[subset["net_R"] > 0]["net_R"] if len(subset) else pd.Series(dtype=float)
        losses = -subset[subset["net_R"] <= 0]["net_R"] if len(subset) else pd.Series(dtype=float)
        result[month] = {
            "trades": int(len(subset)),
            "wins": int((subset["net_R"] > 0).sum()) if len(subset) else 0,
            "win_rate": float((subset["net_R"] > 0).mean()) if len(subset) else 0.0,
            "avg_win_R": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss_R": float(losses.mean()) if len(losses) else 0.0,
            "avg_win_loss_ratio": float(wins.mean() / losses.mean()) if len(wins) and len(losses) else 0.0,
            "profit_factor": float(wins.sum() / losses.sum()) if losses.sum() > 0 else 0.0,
            "net_R": float(subset["net_R"].sum()) if len(subset) else 0.0,
        }
    return result


def pine(cfg: Config) -> str:
    pull_tols = (0.15, 0.30, 0.50)
    pull_bars = (3, 6, 12)
    tol = pull_tols[cfg.pull_idx // 3]
    pbars = pull_bars[cfg.pull_idx % 3]
    don_len = 12 if cfg.don_idx == 0 else 24
    direction = ("双向", "只做多", "只做空")[cfg.direction]
    mask = cfg.family_mask
    use_trend = str(bool(mask & 1)).lower()
    use_break = str(bool(mask & 2)).lower()
    use_range = str(bool(mask & 4)).lower()
    return f'''//@version=6
strategy("{SYMBOL} {INTERVAL} BTC短线多状态策略", overlay=true, pyramiding=0, initial_capital=100000, default_qty_type=strategy.percent_of_equity, default_qty_value=20, commission_type=strategy.commission.percent, commission_value={FEE_RATE*100:.6f}, slippage={SLIPPAGE_TICKS}, process_orders_on_close=true)

// 训练月 {MONTHS[0]}；样本外验证月 {MONTHS[1]}。只使用已完成的高周期K线。
g="自动优化参数"
rr=input.float({cfg.rr:.4f},"目标盈亏比",minval=1.50,step=0.05,group=g)
slAtr=input.float({cfg.sl_atr:.6f},"止损ATR倍数",minval=0.5,step=0.05,group=g)
minStop=input.float({cfg.min_stop_pct*100:.6f},"最小止损百分比",minval=0.1,step=0.05,group=g)/100
startTime=input.time({START_MS},"开始时间",group=g)
endTime=input.time({END_EXCLUSIVE_MS-STEP_MS},"结束时间",group=g)

ema8=ta.ema(close,8)
ema21=ta.ema(close,21)
ema55=ta.ema(close,55)
ema200=ta.ema(close,200)
atr=ta.atr(14)
atrPct=atr/close
atrRank=ta.percentrank(atrPct,288)/100
[pdi,mdi,adx]=ta.dmi(14,14)
rsi=ta.rsi(close,14)
basis=ta.sma(close,20)
sd=ta.stdev(close,20)
lower=basis-2*sd
upper=basis+2*sd
bbPos=(close-lower)/math.max(upper-lower,syminfo.mintick)
bbWidth=(upper-lower)/math.max(basis,syminfo.mintick)
bbRank=ta.percentrank(bbWidth,288)/100
squeezeRelease=bbRank[1]<={cfg.squeeze_rank_max:.6f} and bbRank>bbRank[1]
vwapValue=ta.vwap(hlc3)
vwapDev=(close-vwapValue)/math.max(atr,syminfo.mintick)
volMean=ta.sma(volume,48)
volSd=ta.stdev(volume,48)
volZ=(volume-volMean)/math.max(volSd,syminfo.mintick)
rng=math.max(high-low,syminfo.mintick)
body=math.abs(close-open)/rng
closeLoc=(close-low)/rng
ret3=close/close[3]-1
ret12=close/close[12]-1
eff12=math.abs(close-close[12])/math.max(ta.sum(math.abs(ta.change(close)),12),syminfo.mintick)
eff24=math.abs(close-close[24])/math.max(ta.sum(math.abs(ta.change(close)),24),syminfo.mintick)
donH=ta.highest(high[1],{don_len})
donL=ta.lowest(low[1],{don_len})

m15Long=request.security(syminfo.tickerid,"15",close[1]>ta.ema(close,20)[1] and ta.ema(close,20)[1]>ta.ema(close,50)[1] and ta.ema(close,50)[1]>ta.ema(close,50)[3],lookahead=barmerge.lookahead_on)
m15Short=request.security(syminfo.tickerid,"15",close[1]<ta.ema(close,20)[1] and ta.ema(close,20)[1]<ta.ema(close,50)[1] and ta.ema(close,50)[1]<ta.ema(close,50)[3],lookahead=barmerge.lookahead_on)
h1Long=request.security(syminfo.tickerid,"60",close[1]>ta.ema(close,20)[1] and ta.ema(close,20)[1]>ta.ema(close,50)[1] and ta.ema(close,50)[1]>ta.ema(close,50)[3],lookahead=barmerge.lookahead_on)
h1Short=request.security(syminfo.tickerid,"60",close[1]<ta.ema(close,20)[1] and ta.ema(close,20)[1]<ta.ema(close,50)[1] and ta.ema(close,50)[1]<ta.ema(close,50)[3],lookahead=barmerge.lookahead_on)

longTouch=low<=ema21+atr*{tol:.4f} and low>=ema55-atr
shortTouch=high>=ema21-atr*{tol:.4f} and high<=ema55+atr
pullLong=ta.barssince(longTouch)<={pbars} and close>ema8
pullShort=ta.barssince(shortTouch)<={pbars} and close<ema8
longCandle=close>open and body>={cfg.body_min:.6f} and closeLoc>={cfg.close_loc_min:.6f}
shortCandle=close<open and body>={cfg.body_min:.6f} and closeLoc<={1-cfg.close_loc_min:.6f}

trendLong={use_trend} and h1Long and m15Long and pullLong and close>ema8 and ema21>ema55 and close>ema200 and adx>={cfg.adx_trend_min:.6f} and pdi>mdi and rsi>={cfg.rsi_long_min:.6f} and rsi<={cfg.rsi_long_max:.6f} and eff12>={cfg.eff_min*0.70:.6f} and longCandle
trendShort={use_trend} and h1Short and m15Short and pullShort and close<ema8 and ema21<ema55 and close<ema200 and adx>={cfg.adx_trend_min:.6f} and mdi>pdi and rsi<={100-cfg.rsi_long_min:.6f} and rsi>={100-cfg.rsi_long_max:.6f} and eff12>={cfg.eff_min*0.70:.6f} and shortCandle

breakLong={use_break} and (h1Long or m15Long) and not h1Short and close>donH and close>vwapValue and close>ema55 and adx>={cfg.adx_trend_min:.6f} and pdi>mdi and volZ>={cfg.volz_break:.6f} and atrRank>={cfg.atr_rank_min:.6f} and (squeezeRelease or bbRank[1]<={cfg.squeeze_rank_max:.6f}) and eff24>={cfg.eff_min:.6f} and ret3>0 and ret12>0 and longCandle
breakShort={use_break} and (h1Short or m15Short) and not h1Long and close<donL and close<vwapValue and close<ema55 and adx>={cfg.adx_trend_min:.6f} and mdi>pdi and volZ>={cfg.volz_break:.6f} and atrRank>={cfg.atr_rank_min:.6f} and (squeezeRelease or bbRank[1]<={cfg.squeeze_rank_max:.6f}) and eff24>={cfg.eff_min:.6f} and ret3<0 and ret12<0 and shortCandle

rangeLong={use_range} and not h1Short and not m15Short and adx<={cfg.adx_range_max:.6f} and eff12<=math.max(0.18,{cfg.eff_min*0.75:.6f}) and vwapDev<=-{cfg.vwap_dev_entry:.6f} and bbPos<=0.20 and rsi<=math.min(47,{cfg.rsi_long_min+4:.6f}) and close>open and body>={cfg.body_min*0.75:.6f} and closeLoc>={cfg.close_loc_min:.6f}
rangeShort={use_range} and not h1Long and not m15Long and adx<={cfg.adx_range_max:.6f} and eff12<=math.max(0.18,{cfg.eff_min*0.75:.6f}) and vwapDev>={cfg.vwap_dev_entry:.6f} and bbPos>=0.80 and rsi>=math.max(53,{96-cfg.rsi_long_min:.6f}) and close<open and body>={cfg.body_min*0.75:.6f} and closeLoc<={1-cfg.close_loc_min:.6f}

hourOk=hour(time,"UTC")>={cfg.session_start} and hour(time,"UTC")<{cfg.session_end}
dow=dayofweek(time,"UTC")
wd=dow==dayofweek.monday?0:dow==dayofweek.tuesday?1:dow==dayofweek.wednesday?2:dow==dayofweek.thursday?3:dow==dayofweek.friday?4:dow==dayofweek.saturday?5:6
weekdayOk=int(math.floor({cfg.weekday_mask}/math.pow(2,wd)))%2==1
inRange=time>=startTime and time<=endTime
var int lastExit=na
if strategy.position_size==0 and strategy.position_size[1]!=0
    lastExit:=bar_index
cooldownOk=na(lastExit) or bar_index-lastExit>={cfg.cooldown}
allowLong="{direction}"!="只做空"
allowShort="{direction}"!="只做多"
longSignal=inRange and hourOk and weekdayOk and cooldownOk and strategy.position_size==0 and allowLong and (trendLong or breakLong or rangeLong)
shortSignal=inRange and hourOk and weekdayOk and cooldownOk and strategy.position_size==0 and allowShort and (trendShort or breakShort or rangeShort)

var float risk=na
var float stopPrice=na
var float targetPrice=na
var int entryBar=na
var bool trailArmed=false
if longSignal and not shortSignal
    risk:=math.max(atr*slAtr,close*minStop)
    trailArmed:=false
    strategy.entry("L",strategy.long)
if shortSignal and not longSignal
    risk:=math.max(atr*slAtr,close*minStop)
    trailArmed:=false
    strategy.entry("S",strategy.short)
if strategy.position_size!=0 and strategy.position_size[1]==0
    entryBar:=bar_index
    stopPrice:=strategy.position_size>0?strategy.position_avg_price-risk:strategy.position_avg_price+risk
    targetPrice:=strategy.position_size>0?strategy.position_avg_price+risk*rr:strategy.position_avg_price-risk*rr
if strategy.position_size>0
    strategy.exit("LX","L",stop=stopPrice,limit=targetPrice)
    if not trailArmed and high-strategy.position_avg_price>=risk*{cfg.trail_trigger:.6f}
        trailArmed:=true
    if trailArmed
        stopPrice:=math.max(stopPrice,strategy.position_avg_price+risk*{cfg.trail_lock:.6f})
if strategy.position_size<0
    strategy.exit("SX","S",stop=stopPrice,limit=targetPrice)
    if not trailArmed and strategy.position_avg_price-low>=risk*{cfg.trail_trigger:.6f}
        trailArmed:=true
    if trailArmed
        stopPrice:=math.min(stopPrice,strategy.position_avg_price-risk*{cfg.trail_lock:.6f})
if strategy.position_size!=0 and bar_index-entryBar>={cfg.max_hold}
    strategy.close_all(comment="超时退出")

plot(ema8,"EMA8",color=color.aqua)
plot(ema21,"EMA21",color=color.orange)
plot(ema55,"EMA55",color=color.blue)
plotshape(longSignal and not shortSignal,style=shape.triangleup,location=location.belowbar,color=color.lime,size=size.tiny,text="多")
plotshape(shortSignal and not longSignal,style=shape.triangledown,location=location.abovebar,color=color.red,size=size.tiny,text="空")
'''


def main() -> None:
    raw, audit = load_official_data()
    x = add_indicators(raw)
    arrays = build_arrays(x)

    all_configs: list[Config] = []
    all_metrics: list[np.ndarray] = []
    rounds = int(os.environ.get("SEARCH_ROUNDS", str(REQUEST["search_rounds"])))
    batch = int(os.environ.get("SEARCH_BATCH", str(REQUEST["search_batch"])))

    for seed_index in range(SEED_COUNT):
        seed = BASE_SEED + seed_index * 100003
        rng = random.Random(seed)
        for round_index in range(rounds):
            configs = [random_config(rng) for _ in range(batch)]
            params = pack_configs(configs)
            metrics = evaluate_many(
                params, arrays["open"], arrays["high"], arrays["low"], arrays["close"],
                arrays["atr"], arrays["ema8"], arrays["ema21"], arrays["ema55"],
                arrays["ema200"], arrays["vwap"], arrays["vwap_dev"], arrays["plus_di"],
                arrays["minus_di"], arrays["adx"], arrays["rsi"], arrays["bb_pos"],
                arrays["bb_rank"], arrays["squeeze_release"], arrays["vol_z"],
                arrays["body"], arrays["close_loc"], arrays["ret1"], arrays["ret3"],
                arrays["ret12"], arrays["eff12"], arrays["eff24"], arrays["atr_rank"],
                arrays["don12h"], arrays["don12l"], arrays["don24h"], arrays["don24l"],
                arrays["m15_long"], arrays["m15_short"], arrays["h1_long"],
                arrays["h1_short"], arrays["pull_l"], arrays["pull_s"], arrays["period"],
                arrays["hour"], arrays["weekday"], arrays["day"],
            )
            keep = np.argsort(metrics[:, 0])[-700:]
            all_configs.extend(configs[i] for i in keep)
            all_metrics.extend(metrics[i].copy() for i in keep)
            print(
                f"seed {seed_index+1}/{SEED_COUNT}, round {round_index+1}/{rounds}, "
                f"best={metrics[keep[-1],0]:.3f}, diagnostic_qualified={int(metrics[:,1].sum())}"
            )

    metrics = np.vstack(all_metrics)
    order = np.argsort(metrics[:, 0])[::-1]
    best_index = int(order[0])
    cfg = all_configs[best_index]
    best_metrics = metrics[best_index]
    trades = pd.DataFrame(simulate_detail(cfg, arrays, x))
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
    (RESULTS / f"{SYMBOL}_{INTERVAL}_optimized_strategy.pine").write_text(pine(cfg), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for rank, index in enumerate(order[:120], 1):
        candidate = all_configs[int(index)]
        m = metrics[int(index)]
        rows.append({
            "rank": rank,
            "score_train_only": m[0],
            "diagnostic_qualified": bool(m[1]),
            "family": FAMILY_NAMES[candidate.family_mask],
            "direction": ("双向", "只做多", "只做空")[candidate.direction],
            "train_trades": m[2],
            "train_win_rate": m[3],
            "train_ratio": m[4],
            "test_trades": m[6],
            "test_win_rate": m[7],
            "test_ratio": m[8],
            "total_net_R": m[10],
            "max_drawdown_R": m[11],
            "first_half_trades": m[14],
            "first_half_win_rate": m[15],
            "first_half_net_R": m[16],
            "second_half_trades": m[17],
            "second_half_win_rate": m[18],
            "second_half_net_R": m[19],
            "config": json.dumps(asdict(candidate), ensure_ascii=False),
        })
    pd.DataFrame(rows).to_csv(RESULTS / "top_candidates.csv", index=False)

    status = {
        "qualified": qualified,
        "engine": "BTC 5m regime-aware short-term V3",
        "selected_by": f"{MONTHS[0]} train-only score with half-month stability checks",
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
        "search_size": int(rounds * batch * SEED_COUNT),
        "train_half_stability": {
            "first_half_trades": int(best_metrics[14]),
            "first_half_win_rate": float(best_metrics[15]),
            "first_half_net_R": float(best_metrics[16]),
            "second_half_trades": int(best_metrics[17]),
            "second_half_win_rate": float(best_metrics[18]),
            "second_half_net_R": float(best_metrics[19]),
        },
    }
    (RESULTS / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        f"# {SYMBOL} {INTERVAL} BTC短线多状态回测报告",
        "",
        "- 引擎：趋势回踩、波动压缩突破、低ADX震荡反转三类策略分离。",
        f"- 数据审计：通过，{audit['actual_rows']:,}根，缺失{audit['missing_rows']}，重复{audit['duplicate_timestamps']}。",
        f"- 搜索规模：{rounds * batch * SEED_COUNT:,}组可复现参数。",
        f"- 选择方法：只按{MONTHS[0]}训练成绩与前后半月稳定性排名；{MONTHS[1]}保持样本外验证。",
        f"- 成本：单边手续费{FEE_RATE*100:.3f}%；每次成交滑点{SLIPPAGE_ABS:.1f} USDT。",
        f"- 最终验收：**{'达到全部要求' if qualified else '未达到全部要求，以下为训练选择后的样本外结果'}**。",
        "",
        "## 策略结构",
        "",
        f"- 启用策略：{FAMILY_NAMES[cfg.family_mask]}",
        f"- 交易方向：{('双向', '只做多', '只做空')[cfg.direction]}",
        f"- 固定目标：{cfg.rr:.2f}R；达到{cfg.trail_trigger:.2f}R后，下一根K线起锁定{cfg.trail_lock:.2f}R。",
        "",
        "## 月度结果",
        "",
        "| 月份 | 交易 | 胜率 | 平均盈利/平均亏损 | 盈利因子 | 净R |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for month in MONTHS:
        s = stats[month]
        report_lines.append(
            f"| {month} | {s['trades']} | {s['win_rate']:.2%} | "
            f"{s['avg_win_loss_ratio']:.3f} | {s['profit_factor']:.3f} | {s['net_R']:.3f} |"
        )
    report_lines += [
        "",
        "## 训练月稳定性",
        "",
        f"- 前半月：{int(best_metrics[14])}笔，胜率{best_metrics[15]:.2%}，净{best_metrics[16]:.3f}R。",
        f"- 后半月：{int(best_metrics[17])}笔，胜率{best_metrics[18]:.2%}，净{best_metrics[19]:.3f}R。",
        "",
        "## 说明",
        "",
        "回测只使用当前及历史K线；高周期过滤只读取上一根已完成的15分钟/1小时K线。",
        "同一根K线同时触及止损和止盈时按止损优先；移动止损在触发后的下一根K线生效，属于保守处理。",
        "参数排名不使用样本外月份。历史通过也不代表未来收益保证。",
    ]
    (RESULTS / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
