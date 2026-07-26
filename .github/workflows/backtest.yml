from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
RESULTS = ROOT / "results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

SYMBOL = "BTCUSDT"
INTERVAL = "5m"
MONTHS = ("2026-05", "2026-06")
STEP_MS = 5 * 60 * 1000
START_MS = int(pd.Timestamp("2026-05-01T00:00:00Z").timestamp() * 1000)
END_EXCLUSIVE_MS = int(pd.Timestamp("2026-07-01T00:00:00Z").timestamp() * 1000)
EXPECTED = (END_EXCLUSIVE_MS - START_MS) // STEP_MS
FEE_RATE = 0.0005
SLIPPAGE_ABS = 0.2  # Binance BTCUSDT tick 0.1, two ticks each fill.
SEED = 20260726

COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote", "ignore",
]


def download(url: str, path: Path, attempts: int = 6) -> bytes:
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    last: Exception | None = None
    headers = {"User-Agent": "btc-5m-backtest/1.0"}
    for k in range(attempts):
        try:
            r = requests.get(url, timeout=90, headers=headers)
            r.raise_for_status()
            path.write_bytes(r.content)
            return r.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** min(k, 4))
    raise RuntimeError(f"Download failed: {url}: {last}")


def load_official_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    files: list[dict[str, Any]] = []
    for month in MONTHS:
        name = f"{SYMBOL}-{INTERVAL}-{month}.zip"
        base = f"https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{INTERVAL}"
        zpath = CACHE / name
        cpath = CACHE / f"{name}.CHECKSUM"
        raw = download(f"{base}/{name}", zpath)
        checksum_text = download(f"{base}/{name}.CHECKSUM", cpath).decode("utf-8").strip()
        expected_hash = checksum_text.split()[0].lower()
        actual_hash = hashlib.sha256(raw).hexdigest().lower()
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual_hash} != {expected_hash}")
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            members = [x for x in zf.namelist() if x.lower().endswith(".csv")]
            if len(members) != 1:
                raise RuntimeError(f"Unexpected ZIP members in {name}: {members}")
            content = zf.read(members[0])
        first = content.splitlines()[0].decode("utf-8", errors="ignore").lower()
        has_header = "open_time" in first or "open time" in first
        df = pd.read_csv(io.BytesIO(content), header=0 if has_header else None)
        if df.shape[1] < 12:
            raise RuntimeError(f"Unexpected CSV columns in {name}: {df.shape[1]}")
        df = df.iloc[:, :12]
        df.columns = COLS
        frames.append(df)
        files.append({"file": name, "sha256": actual_hash, "rows": int(len(df))})

    df = pd.concat(frames, ignore_index=True)
    numeric = [c for c in COLS if c != "ignore"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("open_time").reset_index(drop=True)

    expected_times = np.arange(START_MS, END_EXCLUSIVE_MS, STEP_MS, dtype=np.int64)
    times = df["open_time"].astype("int64").to_numpy()
    duplicated = int(pd.Series(times).duplicated().sum())
    unique_times = np.unique(times)
    missing = np.setdiff1d(expected_times, unique_times)
    extra = np.setdiff1d(unique_times, expected_times)
    off_grid = int(np.sum((times - START_MS) % STEP_MS != 0))
    bad_close_time = int(np.sum(df["close_time"].astype("int64").to_numpy() != times + STEP_MS - 1))
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    valid_ohlc = finite & (h >= np.maximum.reduce([o, c, l])) & (l <= np.minimum.reduce([o, c, h])) & (l <= h)
    invalid_ohlc = int(np.sum(~valid_ohlc))
    audit = {
        "source": "Binance USDⓈ-M Futures official monthly klines",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "start_utc": "2026-05-01T00:00:00Z",
        "end_utc": "2026-06-30T23:55:00Z",
        "expected_rows": int(EXPECTED),
        "actual_rows": int(len(df)),
        "unique_rows": int(len(unique_times)),
        "duplicate_timestamps": duplicated,
        "missing_rows": int(len(missing)),
        "extra_rows": int(len(extra)),
        "off_grid_rows": off_grid,
        "invalid_close_time_rows": bad_close_time,
        "invalid_ohlc_rows": invalid_ohlc,
        "first_open_time": int(times[0]),
        "last_open_time": int(times[-1]),
        "files": files,
        "passed": bool(
            len(df) == EXPECTED
            and len(unique_times) == EXPECTED
            and duplicated == 0
            and len(missing) == 0
            and len(extra) == 0
            and off_grid == 0
            and bad_close_time == 0
            and invalid_ohlc == 0
            and times[0] == START_MS
            and times[-1] == END_EXCLUSIVE_MS - STEP_MS
        ),
        "missing_preview": [pd.to_datetime(x, unit="ms", utc=True).isoformat() for x in missing[:20]],
    }
    if not audit["passed"]:
        raise RuntimeError("Data audit failed: " + json.dumps(audit, ensure_ascii=False, indent=2))
    return df, audit


def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.index = pd.to_datetime(x["open_time"], unit="ms", utc=True)
    o, h, l, c, v = (x[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    for n in (9, 21, 50, 200):
        x[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    x["atr"] = rma(tr, 14)
    up = h.diff()
    down = -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_di = 100 * rma(plus_dm, 14) / x["atr"].replace(0, np.nan)
    minus_di = 100 * rma(minus_dm, 14) / x["atr"].replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["plus_di"], x["minus_di"], x["adx"] = plus_di, minus_di, rma(dx, 14)
    d = c.diff()
    gain, loss = d.clip(lower=0), -d.clip(upper=0)
    rs = rma(gain, 14) / rma(loss, 14).replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + rs)
    fast = c.ewm(span=12, adjust=False).mean()
    slow = c.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    x["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    basis = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    x["bb_pos"] = (c - (basis - 2 * sd)) / (4 * sd).replace(0, np.nan)
    utc_day = x.index.floor("D")
    pv = ((h + l + c) / 3) * v
    x["vwap"] = pv.groupby(utc_day).cumsum() / v.groupby(utc_day).cumsum().replace(0, np.nan)
    x["vol_z"] = (v - v.rolling(50).mean()) / v.rolling(50).std(ddof=0).replace(0, np.nan)
    rng = (h - l).replace(0, np.nan)
    x["body"] = (c - o).abs() / rng
    x["close_loc"] = (c - l) / rng
    for n in (1, 3, 6, 12, 24):
        x[f"ret{n}"] = c.pct_change(n)
    x["don3h"] = h.shift(1).rolling(3).max()
    x["don3l"] = l.shift(1).rolling(3).min()
    x["don6h"] = h.shift(1).rolling(6).max()
    x["don6l"] = l.shift(1).rolling(6).min()
    x["cross_up_ema9"] = (c > x["ema9"]) & (c.shift(1) <= x["ema9"].shift(1))
    x["cross_dn_ema9"] = (c < x["ema9"]) & (c.shift(1) >= x["ema9"].shift(1))
    lower = basis - 2 * sd
    upper = basis + 2 * sd
    x["meanrev_long"] = (c > lower) & (c.shift(1) <= lower.shift(1)) & (x["rsi"] < 42)
    x["meanrev_short"] = (c < upper) & (c.shift(1) >= upper.shift(1)) & (x["rsi"] > 58)

    def htf_features(rule: str, prefix: str) -> None:
        bars = x[["open", "high", "low", "close", "volume"]].resample(rule, label="right", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        e20 = bars["close"].ewm(span=20, adjust=False).mean()
        e50 = bars["close"].ewm(span=50, adjust=False).mean()
        long = (bars["close"] > e20) & (e20 > e50) & (e50 > e50.shift(2))
        short = (bars["close"] < e20) & (e20 < e50) & (e50 < e50.shift(2))
        x[f"{prefix}_long"] = long.reindex(x.index, method="ffill").fillna(False)
        x[f"{prefix}_short"] = short.reindex(x.index, method="ffill").fillna(False)

    htf_features("15min", "m15")
    htf_features("60min", "h1")

    pull_tols = (0.2, 0.4, 0.6)
    pull_bars = (3, 6, 12)
    for ti, tol in enumerate(pull_tols):
        touch_l = (l <= x["ema21"] + tol * x["atr"]) & (l >= x["ema50"] - x["atr"])
        touch_s = (h >= x["ema21"] - tol * x["atr"]) & (h <= x["ema50"] + x["atr"])
        for bi, bars in enumerate(pull_bars):
            x[f"pl_{ti}_{bi}"] = touch_l.rolling(bars).max().fillna(0).astype(bool) & (c > x["ema9"])
            x[f"ps_{ti}_{bi}"] = touch_s.rolling(bars).max().fillna(0).astype(bool) & (c < x["ema9"])
    return x.replace([np.inf, -np.inf], np.nan).dropna().copy()


@dataclass
class Config:
    direction: int
    trigger: int
    rr: float
    sl_atr: float
    min_stop_pct: float
    max_hold: int
    cooldown: int
    session_start: int
    session_end: int
    weekday_mask: int
    adx_min: float
    rsi_long_min: float
    rsi_long_max: float
    volz_min: float
    body_min: float
    close_loc_min: float
    ret3_min: float
    bb_min: float
    bb_max: float
    pull_idx: int
    don_idx: int
    score_need: int
    weights: list[int]


def random_config(rng: random.Random) -> Config:
    weights = [rng.choice((0, 0, 1, 1, 2, 3)) for _ in range(16)]
    if sum(weights) < 7:
        for j in rng.sample(range(16), 5):
            weights[j] = rng.choice((1, 2, 3))
    total = sum(weights)
    direction = rng.choices((0, 1, 2), weights=(5, 3, 3))[0]
    if rng.random() < 0.25:
        start, end = 0, 24
    else:
        start = rng.randrange(0, 20)
        end = min(24, start + rng.randrange(5, 15))
    mask_choices = (31, 127, 62, 124, 30, 60, 120)  # Mon-Fri, all, Tue-Sat, Wed-Sun, etc.
    return Config(
        direction=direction,
        trigger=rng.randrange(5),
        rr=rng.choice((1.5, 1.6, 1.75, 2.0, 2.25)),
        sl_atr=rng.uniform(0.9, 3.0),
        min_stop_pct=rng.uniform(0.0025, 0.009),
        max_hold=rng.choice((12, 18, 24, 36, 48, 72, 96)),
        cooldown=rng.randrange(4, 73),
        session_start=start,
        session_end=end,
        weekday_mask=rng.choice(mask_choices),
        adx_min=rng.uniform(12, 36),
        rsi_long_min=rng.uniform(35, 55),
        rsi_long_max=rng.uniform(58, 78),
        volz_min=rng.uniform(-0.8, 1.8),
        body_min=rng.uniform(0.22, 0.72),
        close_loc_min=rng.uniform(0.55, 0.88),
        ret3_min=rng.uniform(-0.001, 0.004),
        bb_min=rng.uniform(0.15, 0.55),
        bb_max=rng.uniform(0.55, 1.15),
        pull_idx=rng.randrange(9),
        don_idx=rng.randrange(2),
        score_need=rng.randint(max(2, int(total * 0.35)), max(3, int(total * 0.78))),
        weights=weights,
    )


def build_arrays(x: pd.DataFrame) -> dict[str, np.ndarray]:
    keys = [
        "open", "high", "low", "close", "atr", "ema9", "ema21", "ema50", "vwap",
        "plus_di", "minus_di", "adx", "rsi", "macd_hist", "bb_pos", "vol_z", "body",
        "close_loc", "ret1", "ret3", "ret12", "don3h", "don3l", "don6h", "don6l",
    ]
    a = {k: x[k].to_numpy(np.float64) for k in keys}
    for k in ("cross_up_ema9", "cross_dn_ema9", "meanrev_long", "meanrev_short", "m15_long", "m15_short", "h1_long", "h1_short"):
        a[k] = x[k].to_numpy(np.bool_)
    a["pull_l"] = np.column_stack([x[f"pl_{ti}_{bi}"].to_numpy(np.bool_) for ti in range(3) for bi in range(3)])
    a["pull_s"] = np.column_stack([x[f"ps_{ti}_{bi}"].to_numpy(np.bool_) for ti in range(3) for bi in range(3)])
    idx = x.index
    a["month"] = idx.month.to_numpy(np.int16)
    a["hour"] = idx.hour.to_numpy(np.int16)
    a["weekday"] = idx.weekday.to_numpy(np.int16)
    a["timestamp"] = (idx.view("int64") // 1_000_000).astype(np.int64)
    return a


def pack_configs(configs: list[Config]) -> tuple[np.ndarray, np.ndarray]:
    p = np.zeros((len(configs), 22), dtype=np.float64)
    w = np.zeros((len(configs), 16), dtype=np.int16)
    for i, c in enumerate(configs):
        p[i] = [
            c.direction, c.trigger, c.rr, c.sl_atr, c.min_stop_pct, c.max_hold, c.cooldown,
            c.session_start, c.session_end, c.weekday_mask, c.adx_min, c.rsi_long_min,
            c.rsi_long_max, c.volz_min, c.body_min, c.close_loc_min, c.ret3_min,
            c.bb_min, c.bb_max, c.pull_idx, c.don_idx, c.score_need,
        ]
        w[i] = np.asarray(c.weights, dtype=np.int16)
    return p, w


@njit(cache=True)
def allowed_time(hour: int, weekday: int, start: int, end: int, mask: int) -> bool:
    if ((mask >> weekday) & 1) == 0:
        return False
    if start == 0 and end == 24:
        return True
    return start <= hour < end


@njit(cache=True)
def calc_signal(i, long_side, p, w, o, h, l, c, ema9, ema21, ema50, vwap, plus_di, minus_di,
                adx, rsi, macdh, bb, volz, body, cloc, ret1, ret3, ret12, don3h, don3l,
                don6h, don6l, cross_up, cross_dn, mean_l, mean_s, m15_l, m15_s, h1_l, h1_s,
                pull_l, pull_s):
    adx_min, rlo, rhi, vz, bmin, locmin, r3min, bbmin, bbmax = p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18]
    if long_side:
        comp = np.empty(16, dtype=np.int16)
        comp[0] = ema9[i] > ema21[i]
        comp[1] = ema21[i] > ema50[i]
        comp[2] = c[i] > vwap[i]
        comp[3] = adx[i] >= adx_min
        comp[4] = plus_di[i] > minus_di[i]
        comp[5] = rsi[i] >= rlo and rsi[i] <= rhi
        comp[6] = macdh[i] > 0
        comp[7] = ret3[i] >= r3min
        comp[8] = ret12[i] > 0
        comp[9] = volz[i] >= vz
        comp[10] = c[i] > o[i] and body[i] >= bmin and cloc[i] >= locmin
        comp[11] = c[i] > (don3h[i] if int(p[20]) == 0 else don6h[i])
        comp[12] = pull_l[i, int(p[19])]
        comp[13] = m15_l[i]
        comp[14] = h1_l[i]
        comp[15] = bb[i] >= bbmin and bb[i] <= bbmax
        trigger = int(p[1])
        base = (comp[11] and comp[10]) if trigger == 0 else (comp[12] and comp[10]) if trigger == 1 else (cross_up[i] and comp[10]) if trigger == 2 else (mean_l[i] and comp[10]) if trigger == 3 else (c[i] > o[i] and ret1[i] > 0)
    else:
        comp = np.empty(16, dtype=np.int16)
        comp[0] = ema9[i] < ema21[i]
        comp[1] = ema21[i] < ema50[i]
        comp[2] = c[i] < vwap[i]
        comp[3] = adx[i] >= adx_min
        comp[4] = minus_di[i] > plus_di[i]
        comp[5] = rsi[i] <= 100 - rlo and rsi[i] >= 100 - rhi
        comp[6] = macdh[i] < 0
        comp[7] = ret3[i] <= -r3min
        comp[8] = ret12[i] < 0
        comp[9] = volz[i] >= vz
        comp[10] = c[i] < o[i] and body[i] >= bmin and cloc[i] <= 1 - locmin
        comp[11] = c[i] < (don3l[i] if int(p[20]) == 0 else don6l[i])
        comp[12] = pull_s[i, int(p[19])]
        comp[13] = m15_s[i]
        comp[14] = h1_s[i]
        comp[15] = bb[i] <= 1 - bbmin and bb[i] >= 1 - bbmax
        trigger = int(p[1])
        base = (comp[11] and comp[10]) if trigger == 0 else (comp[12] and comp[10]) if trigger == 1 else (cross_dn[i] and comp[10]) if trigger == 2 else (mean_s[i] and comp[10]) if trigger == 3 else (c[i] < o[i] and ret1[i] < 0)
    if not base:
        return False
    score = 0
    for j in range(16):
        if comp[j]:
            score += w[j]
    return score >= int(p[21])


@njit(parallel=True, cache=True)
def evaluate_many(params, weights, o, h, l, c, atr, ema9, ema21, ema50, vwap, plus_di,
                  minus_di, adx, rsi, macdh, bb, volz, body, cloc, ret1, ret3, ret12,
                  don3h, don3l, don6h, don6l, cross_up, cross_dn, mean_l, mean_s, m15_l,
                  m15_s, h1_l, h1_s, pull_l, pull_s, month, hour, weekday):
    ncfg = params.shape[0]
    out = np.zeros((ncfg, 14), dtype=np.float64)
    n = len(c)
    for q in prange(ncfg):
        p = params[q]
        w = weights[q]
        pos = 0
        entry = stop = target = risk = 0.0
        entry_i = last_exit = -100000
        entry_month = 0
        count5 = wins5 = count6 = wins6 = 0
        win_sum5 = loss_sum5 = win_sum6 = loss_sum6 = 0.0
        win_n5 = loss_n5 = win_n6 = loss_n6 = 0
        total_r = 0.0
        peak = 0.0
        max_dd = 0.0
        for i in range(250, n):
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
                    if total_r > peak:
                        peak = total_r
                    dd = peak - total_r
                    if dd > max_dd:
                        max_dd = dd
                    if entry_month == 5:
                        count5 += 1
                        if net_r > 0:
                            wins5 += 1; win_sum5 += net_r; win_n5 += 1
                        else:
                            loss_sum5 += -net_r; loss_n5 += 1
                    else:
                        count6 += 1
                        if net_r > 0:
                            wins6 += 1; win_sum6 += net_r; win_n6 += 1
                        else:
                            loss_sum6 += -net_r; loss_n6 += 1
                    pos = 0
                    last_exit = i
                continue
            if i - last_exit < int(p[6]):
                continue
            if not allowed_time(int(hour[i]), int(weekday[i]), int(p[7]), int(p[8]), int(p[9])):
                continue
            direction = int(p[0])
            go_long = direction != 2 and calc_signal(i, True, p, w, o, h, l, c, ema9, ema21, ema50, vwap, plus_di, minus_di, adx, rsi, macdh, bb, volz, body, cloc, ret1, ret3, ret12, don3h, don3l, don6h, don6l, cross_up, cross_dn, mean_l, mean_s, m15_l, m15_s, h1_l, h1_s, pull_l, pull_s)
            go_short = direction != 1 and calc_signal(i, False, p, w, o, h, l, c, ema9, ema21, ema50, vwap, plus_di, minus_di, adx, rsi, macdh, bb, volz, body, cloc, ret1, ret3, ret12, don3h, don3l, don6h, don6l, cross_up, cross_dn, mean_l, mean_s, m15_l, m15_s, h1_l, h1_s, pull_l, pull_s)
            if go_long == go_short:
                continue
            pos = 1 if go_long else -1
            entry = c[i] + SLIPPAGE_ABS if pos > 0 else c[i] - SLIPPAGE_ABS
            risk = max(atr[i] * p[3], c[i] * p[4])
            stop = entry - risk if pos > 0 else entry + risk
            target = entry + risk * p[2] if pos > 0 else entry - risk * p[2]
            entry_i, entry_month = i, int(month[i])
        wr5 = wins5 / count5 if count5 else 0.0
        wr6 = wins6 / count6 if count6 else 0.0
        ratio5 = (win_sum5 / win_n5) / (loss_sum5 / loss_n5) if win_n5 and loss_n5 else 0.0
        ratio6 = (win_sum6 / win_n6) / (loss_sum6 / loss_n6) if win_n6 and loss_n6 else 0.0
        pf5 = win_sum5 / loss_sum5 if loss_sum5 > 0 else (99.0 if win_sum5 > 0 else 0.0)
        pf6 = win_sum6 / loss_sum6 if loss_sum6 > 0 else (99.0 if win_sum6 > 0 else 0.0)
        qualified = count5 >= 20 and count5 <= 30 and count6 >= 20 and count6 <= 30 and wr5 >= .70 and wr6 >= .70 and ratio5 >= 1.5 and ratio6 >= 1.5
        count_pen = abs(count5 - 25) + abs(count6 - 25)
        score = min(wr5, wr6) * 180 + min(ratio5, ratio6) * 18 + min(pf5, pf6) * 5 + total_r - count_pen * 2.5 - max_dd * 0.4
        if qualified:
            score += 10000
        out[q] = (score, qualified, count5, wr5, ratio5, pf5, count6, wr6, ratio6, pf6, total_r, max_dd, wins5, wins6)
    return out


def simulate_detail(cfg: Config, a: dict[str, np.ndarray], x: pd.DataFrame) -> list[dict[str, Any]]:
    p, w = pack_configs([cfg]); p, w = p[0], w[0]
    trades: list[dict[str, Any]] = []
    pos = 0
    entry = stop = target = risk = 0.0
    entry_i = last_exit = -100000
    entry_month = 0
    n = len(a["close"])
    for i in range(250, n):
        if pos:
            exit_price = None; reason = ""
            if pos > 0:
                if a["low"][i] <= stop: exit_price, reason = stop - SLIPPAGE_ABS, "SL"
                elif a["high"][i] >= target: exit_price, reason = target - SLIPPAGE_ABS, "TP"
            else:
                if a["high"][i] >= stop: exit_price, reason = stop + SLIPPAGE_ABS, "SL"
                elif a["low"][i] <= target: exit_price, reason = target + SLIPPAGE_ABS, "TP"
            if exit_price is None and i - entry_i >= cfg.max_hold:
                exit_price = a["close"][i] - SLIPPAGE_ABS if pos > 0 else a["close"][i] + SLIPPAGE_ABS
                reason = "TIME"
            if exit_price is not None:
                net_r = (((exit_price - entry) * pos) - FEE_RATE * (entry + exit_price)) / risk
                trades.append({
                    "entry_time_utc": x.index[entry_i].isoformat(), "exit_time_utc": x.index[i].isoformat(),
                    "month": f"2026-{entry_month:02d}", "direction": "LONG" if pos > 0 else "SHORT",
                    "entry": entry, "exit": exit_price, "stop": stop, "target": target,
                    "risk_abs": risk, "net_R": net_r, "win": bool(net_r > 0), "exit_reason": reason,
                    "bars": i - entry_i,
                })
                pos = 0; last_exit = i
            continue
        if i - last_exit < cfg.cooldown or not allowed_time(int(a["hour"][i]), int(a["weekday"][i]), cfg.session_start, cfg.session_end, cfg.weekday_mask):
            continue
        long_sig = cfg.direction != 2 and calc_signal(i, True, p, w, a["open"], a["high"], a["low"], a["close"], a["ema9"], a["ema21"], a["ema50"], a["vwap"], a["plus_di"], a["minus_di"], a["adx"], a["rsi"], a["macd_hist"], a["bb_pos"], a["vol_z"], a["body"], a["close_loc"], a["ret1"], a["ret3"], a["ret12"], a["don3h"], a["don3l"], a["don6h"], a["don6l"], a["cross_up_ema9"], a["cross_dn_ema9"], a["meanrev_long"], a["meanrev_short"], a["m15_long"], a["m15_short"], a["h1_long"], a["h1_short"], a["pull_l"], a["pull_s"])
        short_sig = cfg.direction != 1 and calc_signal(i, False, p, w, a["open"], a["high"], a["low"], a["close"], a["ema9"], a["ema21"], a["ema50"], a["vwap"], a["plus_di"], a["minus_di"], a["adx"], a["rsi"], a["macd_hist"], a["bb_pos"], a["vol_z"], a["body"], a["close_loc"], a["ret1"], a["ret3"], a["ret12"], a["don3h"], a["don3l"], a["don6h"], a["don6l"], a["cross_up_ema9"], a["cross_dn_ema9"], a["meanrev_long"], a["meanrev_short"], a["m15_long"], a["m15_short"], a["h1_long"], a["h1_short"], a["pull_l"], a["pull_s"])
        if long_sig == short_sig: continue
        pos = 1 if long_sig else -1
        entry = a["close"][i] + SLIPPAGE_ABS if pos > 0 else a["close"][i] - SLIPPAGE_ABS
        risk = max(a["atr"][i] * cfg.sl_atr, a["close"][i] * cfg.min_stop_pct)
        stop = entry - risk if pos > 0 else entry + risk
        target = entry + risk * cfg.rr if pos > 0 else entry - risk * cfg.rr
        entry_i, entry_month = i, int(a["month"][i])
    return trades


def monthly_stats(trades: pd.DataFrame) -> dict[str, dict[str, float]]:
    result = {}
    for m in ("2026-05", "2026-06"):
        z = trades[trades["month"] == m]
        wins = z[z["net_R"] > 0]["net_R"]
        losses = -z[z["net_R"] <= 0]["net_R"]
        result[m] = {
            "trades": int(len(z)), "wins": int((z["net_R"] > 0).sum()),
            "win_rate": float((z["net_R"] > 0).mean()) if len(z) else 0.0,
            "avg_win_R": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss_R": float(losses.mean()) if len(losses) else 0.0,
            "avg_win_loss_ratio": float(wins.mean() / losses.mean()) if len(wins) and len(losses) else 0.0,
            "profit_factor": float(wins.sum() / losses.sum()) if losses.sum() > 0 else 99.0 if wins.sum() > 0 else 0.0,
            "net_R": float(z["net_R"].sum()),
        }
    return result


def pine(cfg: Config) -> str:
    weights = ",".join(str(v) for v in cfg.weights)
    pull_tols = (0.2, 0.4, 0.6); pull_bars = (3, 6, 12)
    tol = pull_tols[cfg.pull_idx // 3]; pbars = pull_bars[cfg.pull_idx % 3]
    direction = ("双向", "只做多", "只做空")[cfg.direction]
    return f'''//@version=6
strategy("BTC 5分钟自动优化策略", overlay=true, pyramiding=0, initial_capital=100000, default_qty_type=strategy.percent_of_equity, default_qty_value=20, commission_type=strategy.commission.percent, commission_value=0.05, slippage=2, process_orders_on_close=true)

// 自动搜索结果；回测区间默认2026-05-01至2026-06-30。仅建议用于5分钟图。
g="自动优化参数"
rr=input.float({cfg.rr:.4f},"盈亏比",minval=1.5,step=0.05,group=g)
slAtr=input.float({cfg.sl_atr:.6f},"止损ATR倍数",minval=0.5,step=0.05,group=g)
minStop=input.float({cfg.min_stop_pct*100:.6f},"最小止损百分比",minval=0.1,step=0.05,group=g)/100
startTime=input.time(1777593600000,"开始时间",group=g)
endTime=input.time(1782863700000,"结束时间",group=g)

ema9=ta.ema(close,9)
ema21=ta.ema(close,21)
ema50=ta.ema(close,50)
atr=ta.atr(14)
[pdi,mdi,adx]=ta.dmi(14,14)
rsi=ta.rsi(close,14)
macd=ta.ema(close,12)-ta.ema(close,26)
macdHist=macd-ta.ema(macd,9)
basis=ta.sma(close,20)
sd=ta.stdev(close,20)
bbPos=(close-(basis-2*sd))/math.max(4*sd,syminfo.mintick)
vwapValue=ta.vwap(hlc3)
volMean=ta.sma(volume,50), volSd=ta.stdev(volume,50)
volZ=(volume-volMean)/math.max(volSd,syminfo.mintick)
rng=math.max(high-low,syminfo.mintick)
body=math.abs(close-open)/rng
closeLoc=(close-low)/rng
ret1=close/close[1]-1
ret3=close/close[3]-1
ret12=close/close[12]-1
donH={('ta.highest(high[1],3)' if cfg.don_idx==0 else 'ta.highest(high[1],6)')}
donL={('ta.lowest(low[1],3)' if cfg.don_idx==0 else 'ta.lowest(low[1],6)')}

m15Long=request.security(syminfo.tickerid,"15",close[1]>ta.ema(close,20)[1] and ta.ema(close,20)[1]>ta.ema(close,50)[1] and ta.ema(close,50)[1]>ta.ema(close,50)[3],lookahead=barmerge.lookahead_on)
m15Short=request.security(syminfo.tickerid,"15",close[1]<ta.ema(close,20)[1] and ta.ema(close,20)[1]<ta.ema(close,50)[1] and ta.ema(close,50)[1]<ta.ema(close,50)[3],lookahead=barmerge.lookahead_on)
h1Long=request.security(syminfo.tickerid,"60",close[1]>ta.ema(close,20)[1] and ta.ema(close,20)[1]>ta.ema(close,50)[1] and ta.ema(close,50)[1]>ta.ema(close,50)[3],lookahead=barmerge.lookahead_on)
h1Short=request.security(syminfo.tickerid,"60",close[1]<ta.ema(close,20)[1] and ta.ema(close,20)[1]<ta.ema(close,50)[1] and ta.ema(close,50)[1]<ta.ema(close,50)[3],lookahead=barmerge.lookahead_on)

longTouch=low<=ema21+atr*{tol:.4f} and low>=ema50-atr
shortTouch=high>=ema21-atr*{tol:.4f} and high<=ema50+atr
pullLong=ta.barssince(longTouch)<={pbars} and close>ema9
pullShort=ta.barssince(shortTouch)<={pbars} and close<ema9
meanLong=ta.crossover(close,basis-2*sd) and rsi<42
meanShort=ta.crossunder(close,basis+2*sd) and rsi>58

var weights=array.from({weights})
longComp=array.from(ema9>ema21,ema21>ema50,close>vwapValue,adx>={cfg.adx_min:.6f},pdi>mdi,rsi>={cfg.rsi_long_min:.6f} and rsi<={cfg.rsi_long_max:.6f},macdHist>0,ret3>={cfg.ret3_min:.8f},ret12>0,volZ>={cfg.volz_min:.6f},close>open and body>={cfg.body_min:.6f} and closeLoc>={cfg.close_loc_min:.6f},close>donH,pullLong,m15Long,h1Long,bbPos>={cfg.bb_min:.6f} and bbPos<={cfg.bb_max:.6f})
shortComp=array.from(ema9<ema21,ema21<ema50,close<vwapValue,adx>={cfg.adx_min:.6f},mdi>pdi,rsi<={100-cfg.rsi_long_min:.6f} and rsi>={100-cfg.rsi_long_max:.6f},macdHist<0,ret3<=-{cfg.ret3_min:.8f},ret12<0,volZ>={cfg.volz_min:.6f},close<open and body>={cfg.body_min:.6f} and closeLoc<={1-cfg.close_loc_min:.6f},close<donL,pullShort,m15Short,h1Short,bbPos<={1-cfg.bb_min:.6f} and bbPos>={1-cfg.bb_max:.6f})
f_score(arr)=>
    int s=0
    for i=0 to 15
        if array.get(arr,i)
            s+=array.get(weights,i)
    s
longBase={('array.get(longComp,11) and array.get(longComp,10)' if cfg.trigger==0 else 'array.get(longComp,12) and array.get(longComp,10)' if cfg.trigger==1 else 'ta.crossover(close,ema9) and array.get(longComp,10)' if cfg.trigger==2 else 'meanLong and array.get(longComp,10)' if cfg.trigger==3 else 'close>open and ret1>0')}
shortBase={('array.get(shortComp,11) and array.get(shortComp,10)' if cfg.trigger==0 else 'array.get(shortComp,12) and array.get(shortComp,10)' if cfg.trigger==1 else 'ta.crossunder(close,ema9) and array.get(shortComp,10)' if cfg.trigger==2 else 'meanShort and array.get(shortComp,10)' if cfg.trigger==3 else 'close<open and ret1<0')}

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
longSignal=inRange and hourOk and weekdayOk and cooldownOk and strategy.position_size==0 and allowLong and longBase and f_score(longComp)>={cfg.score_need}
shortSignal=inRange and hourOk and weekdayOk and cooldownOk and strategy.position_size==0 and allowShort and shortBase and f_score(shortComp)>={cfg.score_need}

var float risk=na
var int entryBar=na
if longSignal
    risk:=math.max(atr*slAtr,close*minStop)
    strategy.entry("L",strategy.long)
if shortSignal
    risk:=math.max(atr*slAtr,close*minStop)
    strategy.entry("S",strategy.short)
if strategy.position_size!=0 and strategy.position_size[1]==0
    entryBar:=bar_index
if strategy.position_size>0
    strategy.exit("LX","L",stop=strategy.position_avg_price-risk,limit=strategy.position_avg_price+risk*rr)
if strategy.position_size<0
    strategy.exit("SX","S",stop=strategy.position_avg_price+risk,limit=strategy.position_avg_price-risk*rr)
if strategy.position_size!=0 and bar_index-entryBar>={cfg.max_hold}
    strategy.close_all(comment="时间退出")
plot(ema9,"EMA9",color=color.aqua)
plot(ema21,"EMA21",color=color.orange)
plot(ema50,"EMA50",color=color.blue)
plotshape(longSignal,style=shape.triangleup,location=location.belowbar,color=color.lime,size=size.tiny,text="多")
plotshape(shortSignal,style=shape.triangledown,location=location.abovebar,color=color.red,size=size.tiny,text="空")
'''


def main() -> None:
    raw, audit = load_official_data()
    x = add_indicators(raw)
    a = build_arrays(x)
    rng = random.Random(SEED)
    all_configs: list[Config] = []
    all_metrics: list[np.ndarray] = []
    rounds = int(os.environ.get("SEARCH_ROUNDS", "4"))
    batch = int(os.environ.get("SEARCH_BATCH", "18000"))
    for r in range(rounds):
        configs = [random_config(rng) for _ in range(batch)]
        params, weights = pack_configs(configs)
        m = evaluate_many(params, weights, a["open"], a["high"], a["low"], a["close"], a["atr"], a["ema9"], a["ema21"], a["ema50"], a["vwap"], a["plus_di"], a["minus_di"], a["adx"], a["rsi"], a["macd_hist"], a["bb_pos"], a["vol_z"], a["body"], a["close_loc"], a["ret1"], a["ret3"], a["ret12"], a["don3h"], a["don3l"], a["don6h"], a["don6l"], a["cross_up_ema9"], a["cross_dn_ema9"], a["meanrev_long"], a["meanrev_short"], a["m15_long"], a["m15_short"], a["h1_long"], a["h1_short"], a["pull_l"], a["pull_s"], a["month"], a["hour"], a["weekday"])
        keep = np.argsort(m[:, 0])[-250:]
        all_configs.extend(configs[i] for i in keep)
        all_metrics.extend(m[i].copy() for i in keep)
        print(f"round {r+1}/{rounds}, best={m[keep[-1],0]:.3f}, qualified={int(m[:,1].sum())}")
    metrics = np.vstack(all_metrics)
    order = np.argsort(metrics[:, 0])[::-1]
    best_idx = int(order[0])
    cfg = all_configs[best_idx]
    bestm = metrics[best_idx]
    trades = pd.DataFrame(simulate_detail(cfg, a, x))
    stats = monthly_stats(trades)
    qualified = all(
        20 <= stats[m]["trades"] <= 30
        and stats[m]["win_rate"] >= .70
        and stats[m]["avg_win_loss_ratio"] >= 1.5
        for m in stats
    )
    pd.DataFrame([asdict(cfg)]).to_json(RESULTS / "best_config.json", orient="records", indent=2, force_ascii=False)
    trades.to_csv(RESULTS / "trades.csv", index=False)
    (RESULTS / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "BTC_5m_optimized_strategy.pine").write_text(pine(cfg), encoding="utf-8")
    rows = []
    for rank, ix in enumerate(order[:100], 1):
        c0 = all_configs[int(ix)]; mm = metrics[int(ix)]
        rows.append({"rank": rank, "score": mm[0], "qualified": bool(mm[1]), "may_trades": mm[2], "may_win_rate": mm[3], "may_ratio": mm[4], "june_trades": mm[6], "june_win_rate": mm[7], "june_ratio": mm[8], "net_R": mm[10], "max_drawdown_R": mm[11], "config": json.dumps(asdict(c0), ensure_ascii=False)})
    pd.DataFrame(rows).to_csv(RESULTS / "top_candidates.csv", index=False)
    report = [
        "# BTCUSDT 5分钟自动优化回测报告", "",
        f"- 数据审计：{'通过' if audit['passed'] else '失败'}，{audit['actual_rows']:,}根，缺失{audit['missing_rows']}，重复{audit['duplicate_timestamps']}。",
        f"- 搜索规模：{rounds*batch:,}组可复现参数。",
        f"- 手续费：单边{FEE_RATE*100:.3f}%；滑点：每次成交{SLIPPAGE_ABS:.1f} USDT。",
        f"- 最终验收：**{'达到全部要求' if qualified else '未达到全部要求，以下为最接近候选'}**。", "",
        "## 月度结果", "", "| 月份 | 交易 | 胜率 | 平均盈利/平均亏损 | 盈利因子 | 净R |", "|---|---:|---:|---:|---:|---:|",
    ]
    for m, s in stats.items():
        report.append(f"| {m} | {s['trades']} | {s['win_rate']:.2%} | {s['avg_win_loss_ratio']:.3f} | {s['profit_factor']:.3f} | {s['net_R']:.3f} |")
    report += ["", "## 最优参数", "", "```json", json.dumps(asdict(cfg), indent=2, ensure_ascii=False), "```", "", "## 说明", "", "回测只使用当前及历史K线生成信号；同一根K线同时触及止损和止盈时按止损优先，属于保守处理。两个自然月都参与参数筛选，因此属于短窗口历史拟合结果，不代表未来保证。"]
    (RESULTS / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("QUALIFIED", qualified)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
