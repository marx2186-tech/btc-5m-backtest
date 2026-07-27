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
RESULTS = ROOT / "results_v8"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

ENGINE_VERSION = "V8.1"
ENGINE_NAME = "BTC 5m multi-timeframe expert rejection ensemble V8.1"

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
    "base_seed": 20260808,
}


def load_request() -> dict[str, Any]:
    req = dict(DEFAULT_REQUEST)
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(ROOT / "request.v8.json")))
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("request.v8.json must contain a JSON object")
        req.update(loaded)
    return req


REQUEST = load_request()
SYMBOL = str(REQUEST["symbol"]).upper()
INTERVAL = str(REQUEST["interval"]).lower()
EVAL_MONTHS = tuple(str(x) for x in REQUEST["months"])
if len(EVAL_MONTHS) != 2:
    raise ValueError("V8.1 requires exactly two evaluation months")
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
            response = requests.get(url, timeout=90, headers={"User-Agent": "btc-mtf-expert-v8/1.0"})
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




def read_verified_zip(url: str, checksum_url: str, cache_name: str) -> tuple[bytes, str]:
    raw = download(url, CACHE / cache_name)
    checksum = download(checksum_url, CACHE / f"{cache_name}.CHECKSUM").decode("utf-8").strip()
    expected_hash = checksum.split()[0].lower()
    actual_hash = hashlib.sha256(raw).hexdigest().lower()
    if actual_hash != expected_hash:
        raise RuntimeError(f"SHA-256 mismatch for {cache_name}")
    return raw, actual_hash


def read_single_csv_zip(raw: bytes, name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
        if len(members) != 1:
            raise RuntimeError(f"Unexpected ZIP contents for {name}: {members}")
        return archive.read(members[0])


def parse_kline_csv(content: bytes) -> pd.DataFrame:
    first = content.splitlines()[0].decode("utf-8", errors="ignore").lower()
    has_header = "open_time" in first or "open time" in first
    frame = pd.read_csv(io.BytesIO(content), header=0 if has_header else None).iloc[:, :12]
    frame.columns = COLS
    for col in [c for c in COLS if c != "ignore"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def load_auxiliary_kline(symbol: str, data_type: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    files: list[dict[str, Any]] = []
    for month in MONTHS:
        name = f"{symbol}-{INTERVAL}-{month}.zip"
        base = f"https://data.binance.vision/data/futures/um/monthly/{data_type}/{symbol}/{INTERVAL}"
        raw, digest = read_verified_zip(f"{base}/{name}", f"{base}/{name}.CHECKSUM", f"{data_type}-{name}")
        frame = parse_kline_csv(read_single_csv_zip(raw, name))
        frames.append(frame)
        files.append({"file": name, "sha256": digest, "rows": int(len(frame))})
    data = pd.concat(frames, ignore_index=True).sort_values("open_time").drop_duplicates("open_time")
    expected = sum(int(f["rows"]) for f in files)
    if len(data) != expected:
        raise RuntimeError(f"Duplicate timestamps in {data_type}/{symbol}")
    return data.reset_index(drop=True), {"symbol": symbol, "data_type": data_type, "files": files, "rows": int(len(data))}


def load_funding_rate() -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    files: list[dict[str, Any]] = []
    for month in MONTHS:
        name = f"{SYMBOL}-fundingRate-{month}.zip"
        base = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{SYMBOL}"
        raw, digest = read_verified_zip(f"{base}/{name}", f"{base}/{name}.CHECKSUM", f"funding-{name}")
        content = read_single_csv_zip(raw, name)
        first = content.splitlines()[0].decode("utf-8", errors="ignore").lower()
        has_header = "funding" in first or "calc_time" in first
        frame = pd.read_csv(io.BytesIO(content), header=0 if has_header else None)
        lower = {str(c).strip().lower(): c for c in frame.columns}
        time_col = next((lower[k] for k in ("calc_time", "fundingtime", "funding_time", "timestamp", "time") if k in lower), None)
        rate_col = next((lower[k] for k in ("last_funding_rate", "fundingrate", "funding_rate", "rate") if k in lower), None)
        if time_col is None or rate_col is None:
            if frame.shape[1] < 2:
                raise RuntimeError(f"Unexpected funding-rate schema for {name}: {list(frame.columns)}")
            numeric = frame.apply(pd.to_numeric, errors="coerce")
            time_col = numeric.columns[0]
            rate_col = numeric.columns[-1]
            frame = numeric
        out = pd.DataFrame({
            "calc_time": pd.to_numeric(frame[time_col], errors="coerce"),
            "funding_rate": pd.to_numeric(frame[rate_col], errors="coerce"),
        }).dropna()
        frames.append(out)
        files.append({"file": name, "sha256": digest, "rows": int(len(out))})
    data = pd.concat(frames, ignore_index=True).sort_values("calc_time").drop_duplicates("calc_time")
    if data.empty:
        raise RuntimeError("Funding-rate archive is empty")
    return data.reset_index(drop=True), {"symbol": SYMBOL, "data_type": "fundingRate", "files": files, "rows": int(len(data))}


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def zscore(series: pd.Series, length: int) -> pd.Series:
    mean = series.rolling(length).mean()
    std = series.rolling(length).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def add_features(
    data: pd.DataFrame,
    eth_data: pd.DataFrame,
    premium_data: pd.DataFrame,
    funding_data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    x = data.copy()
    x.index = pd.to_datetime(x["open_time"], unit="ms", utc=True)
    o, h, l, c, v = (x[k].astype(float) for k in ("open", "high", "low", "close", "volume"))

    def normalized_series(frame: pd.DataFrame, time_col: str, value_col: str) -> pd.Series:
        raw_time = pd.to_numeric(frame[time_col], errors="coerce")
        raw_value = pd.to_numeric(frame[value_col], errors="coerce")
        valid = raw_time.notna() & raw_value.notna()
        raw_time = raw_time.loc[valid].astype("float64")
        raw_value = raw_value.loc[valid].astype("float64")
        if raw_time.empty:
            return pd.Series(dtype="float64")

        median_ts = float(raw_time.median())
        # Futures archives normally use milliseconds. The guards also handle
        # seconds, microseconds or nanoseconds without silently shifting dates.
        if median_ts >= 1e17:
            millis = raw_time / 1_000_000.0
        elif median_ts >= 1e14:
            millis = raw_time / 1_000.0
        elif median_ts < 1e11:
            millis = raw_time * 1_000.0
        else:
            millis = raw_time

        idx = pd.to_datetime(np.rint(millis).astype("int64"), unit="ms", utc=True)
        # Snap sub-second archive irregularities to the nearest 5-minute open.
        idx = idx.round("5min")
        series = pd.Series(raw_value.to_numpy(float), index=idx).sort_index()
        return series.groupby(level=0).last()

    eth_series = normalized_series(eth_data, "open_time", "close")
    premium_series = normalized_series(premium_data, "open_time", "close")
    funding = funding_data.copy()
    funding.index = pd.to_datetime(funding["calc_time"], unit="ms", utc=True)

    eth_exact = eth_series.reindex(x.index)
    premium_exact = premium_series.reindex(x.index)

    # Only carry information forward from an already published observation.
    # No backward fill is permitted, so this cannot introduce look-ahead bias.
    eth_close = eth_series.reindex(
        x.index, method="ffill", tolerance=pd.Timedelta(minutes=15)
    )
    premium_close = premium_series.reindex(
        x.index, method="ffill", tolerance=pd.Timedelta(minutes=60)
    )

    alignment_audit = {
        "btc_rows": int(len(x)),
        "eth_source_rows": int(len(eth_series)),
        "premium_source_rows": int(len(premium_series)),
        "eth_exact_missing": int(eth_exact.isna().sum()),
        "premium_exact_missing": int(premium_exact.isna().sum()),
        "eth_missing_after_past_fill": int(eth_close.isna().sum()),
        "premium_missing_after_past_fill": int(premium_close.isna().sum()),
        "eth_coverage_after_past_fill": float(eth_close.notna().mean()),
        "premium_coverage_after_past_fill": float(premium_close.notna().mean()),
        "alignment_rule": "nearest-5m normalization; past-only forward fill; ETH<=15m; premium<=60m",
    }
    print("AUX_ALIGNMENT=" + json.dumps(alignment_audit, ensure_ascii=False, sort_keys=True))

    if alignment_audit["eth_coverage_after_past_fill"] < 0.99:
        raise RuntimeError("ETH auxiliary coverage is below 99% after past-only alignment")
    if alignment_audit["premium_coverage_after_past_fill"] < 0.95:
        raise RuntimeError("Premium-index coverage is below 95% after past-only alignment")

    x["eth_close"] = eth_close
    for n in (1, 3, 6, 12, 24, 72):
        x[f"eth_ret{n}"] = eth_close.pct_change(n)
    eth_e20 = eth_close.ewm(span=20, adjust=False).mean()
    eth_e50 = eth_close.ewm(span=50, adjust=False).mean()
    x["eth_gap"] = (eth_e20 - eth_e50) / eth_close.replace(0, np.nan)
    x["btc_eth_corr"] = c.pct_change().rolling(72).corr(eth_close.pct_change())

    x["premium"] = premium_close
    x["premium_z"] = zscore(premium_close, 288)
    x["premium_delta"] = premium_close.diff(3)
    x["premium_abs_rank"] = premium_close.abs().rolling(288).rank(pct=True)

    funding_series = funding["funding_rate"].astype(float).sort_index()
    # One-bar delay prevents using a funding publication before it is observable.
    aligned_funding = funding_series.reindex(x.index, method="ffill").shift(1).fillna(0.0)
    x["funding_rate"] = aligned_funding
    x["funding_z"] = zscore(aligned_funding, 90)
    x["funding_change"] = aligned_funding.diff()
    x["derivative_pressure"] = x["premium_z"].clip(-4, 4) + x["funding_z"].clip(-4, 4)

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

    for n in (1, 3, 6, 12, 24, 72):
        x[f"ret{n}"] = c.pct_change(n)
    x["btc_eth_spread_12"] = x["ret12"] - x["eth_ret12"]
    x["btc_eth_spread_72"] = x["ret72"] - x["eth_ret72"]
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
        prev_close = bars["close"].shift(1)
        htr = pd.concat([
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        hatr = rma(htr, 14)
        hup = bars["high"].diff()
        hdown = -bars["low"].diff()
        hpdm = hup.where((hup > hdown) & (hup > 0), 0.0)
        hmdm = hdown.where((hdown > hup) & (hdown > 0), 0.0)
        hpdi = 100 * rma(hpdm, 14) / hatr.replace(0, np.nan)
        hmdi = 100 * rma(hmdm, 14) / hatr.replace(0, np.nan)
        hdx = 100 * (hpdi - hmdi).abs() / (hpdi + hmdi).replace(0, np.nan)
        hadx = rma(hdx, 14)
        delta = bars["close"].diff()
        hrs = rma(delta.clip(lower=0), 14) / rma(-delta.clip(upper=0), 14).replace(0, np.nan)
        hrsi = 100 - 100 / (1 + hrs)
        heff = (bars["close"] - bars["close"].shift(12)).abs() / bars["close"].diff().abs().rolling(12).sum().replace(0, np.nan)
        trend = np.where((bars["close"] > e20) & (e20 > e50), 1.0,
                         np.where((bars["close"] < e20) & (e20 < e50), -1.0, 0.0))
        values = {
            f"{prefix}_trend": pd.Series(trend, index=bars.index),
            f"{prefix}_gap": (e20 - e50) / bars["close"].replace(0, np.nan),
            f"{prefix}_adx": hadx,
            f"{prefix}_rsi": hrsi,
            f"{prefix}_eff": heff,
            f"{prefix}_atr_pct": hatr / bars["close"].replace(0, np.nan),
        }
        for name, series in values.items():
            x[name] = series.shift(1).reindex(x.index, method="ffill")

    htf("15min", "m15")
    htf("60min", "h1")
    htf("240min", "h4")

    # ETH one-hour trend is also shifted, so the current 5-minute signal never sees an unfinished hour.
    eth_hour = eth_close.resample("60min", label="right", closed="left").last().dropna()
    eth20 = eth_hour.ewm(span=20, adjust=False).mean()
    eth50 = eth_hour.ewm(span=50, adjust=False).mean()
    eth_trend = pd.Series(np.where((eth_hour > eth20) & (eth20 > eth50), 1.0,
                                   np.where((eth_hour < eth20) & (eth20 < eth50), -1.0, 0.0)), index=eth_hour.index)
    x["eth_h1_trend"] = eth_trend.shift(1).reindex(x.index, method="ffill")

    pull_long = (
        (x["h1_trend"] > 0) & (x["m15_trend"] > 0)
        & (l <= x["ema21"] + 0.30 * x["atr"]) & (c > x["ema8"])
        & (c > o) & (x["macd_slope_atr"] > 0)
    )
    pull_short = (
        (x["h1_trend"] < 0) & (x["m15_trend"] < 0)
        & (h >= x["ema21"] - 0.30 * x["atr"]) & (c < x["ema8"])
        & (c < o) & (x["macd_slope_atr"] < 0)
    )
    sweep_long = (
        (l < x["don12l"]) & (c > x["don12l"]) & (c > o)
        & (x["lower_wick"] >= 0.30) & (x["taker_ratio"] >= 0.48)
    )
    sweep_short = (
        (h > x["don12h"]) & (c < x["don12h"]) & (c < o)
        & (x["upper_wick"] >= 0.30) & (x["taker_ratio"] <= 0.52)
    )
    breakout_long = (
        (c > x["don12h"]) & (c > x["ema55"])
        & (x["body"] >= 0.45) & (x["close_loc"] >= 0.68) & (x["rel_vol"] >= 0.90)
    )
    breakout_short = (
        (c < x["don12l"]) & (c < x["ema55"])
        & (x["body"] >= 0.45) & (x["close_loc"] <= 0.32) & (x["rel_vol"] >= 0.90)
    )
    momentum_long = (
        (x["h1_trend"] > 0) & (x["m15_trend"] > 0) & (c > x["don6h"])
        & (x["taker_ratio"] >= 0.54) & (x["range_exp"] >= 1.05)
    )
    momentum_short = (
        (x["h1_trend"] < 0) & (x["m15_trend"] < 0) & (c < x["don6l"])
        & (x["taker_ratio"] <= 0.46) & (x["range_exp"] >= 1.05)
    )

    x["setup_pull_long"] = pull_long.astype(float)
    x["setup_pull_short"] = pull_short.astype(float)
    x["setup_sweep_long"] = sweep_long.astype(float)
    x["setup_sweep_short"] = sweep_short.astype(float)
    x["setup_break_long"] = breakout_long.astype(float)
    x["setup_break_short"] = breakout_short.astype(float)
    x["setup_mom_long"] = momentum_long.astype(float)
    x["setup_mom_short"] = momentum_short.astype(float)

    high_vol = x["atr_rank"] >= 0.80
    trend_regime = (
        ~high_vol & (x["h1_trend"] != 0) & (x["h1_trend"] == x["h4_trend"])
        & (x["h1_adx"] >= 20) & (x["chop"] <= 57)
    )
    range_regime = (~high_vol) & (~trend_regime) & ((x["h1_adx"] <= 22) | (x["chop"] >= 59))
    neutral_regime = ~(high_vol | trend_regime | range_regime)
    x["regime"] = np.select(
        [trend_regime, range_regime, high_vol, neutral_regime],
        [0, 1, 2, 3], default=3,
    ).astype(float)
    x["regime_trend"] = (x["regime"] == 0).astype(float)
    x["regime_range"] = (x["regime"] == 1).astype(float)
    x["regime_high_vol"] = (x["regime"] == 2).astype(float)
    x["regime_neutral"] = (x["regime"] == 3).astype(float)

    context_long = (x["eth_h1_trend"] >= 0) & (x["derivative_pressure"] <= 4.5)
    context_short = (x["eth_h1_trend"] <= 0) & (x["derivative_pressure"] >= -4.5)
    x["expert_trend_long"] = (
        trend_regime & context_long & (x["h1_trend"] > 0) & (pull_long | breakout_long | momentum_long)
    ).astype(float)
    x["expert_trend_short"] = (
        trend_regime & context_short & (x["h1_trend"] < 0) & (pull_short | breakout_short | momentum_short)
    ).astype(float)
    x["expert_range_long"] = (
        range_regime & sweep_long & (x["m15_rsi"] <= 48) & (x["premium_z"] <= 1.5)
    ).astype(float)
    x["expert_range_short"] = (
        range_regime & sweep_short & (x["m15_rsi"] >= 52) & (x["premium_z"] >= -1.5)
    ).astype(float)
    x["expert_high_long"] = (
        high_vol & context_long & (x["h1_trend"] >= 0) & (breakout_long | momentum_long)
        & (x["premium_delta"] >= -0.00005)
    ).astype(float)
    x["expert_high_short"] = (
        high_vol & context_short & (x["h1_trend"] <= 0) & (breakout_short | momentum_short)
        & (x["premium_delta"] <= 0.00005)
    ).astype(float)

    hours = x.index.hour + x.index.minute / 60.0
    x["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    x["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    x["weekday"] = x.index.weekday.astype(float)
    x["month"] = x.index.to_period("M").astype(str)

    clean = x.replace([np.inf, -np.inf], np.nan).dropna().copy()
    alignment_audit["rows_after_feature_dropna"] = int(len(clean))
    alignment_audit["rows_removed_by_feature_dropna"] = int(len(x) - len(clean))
    return clean, alignment_audit


FEATURES = [
    "ret1", "ret3", "ret6", "ret12", "ret24", "ret72",
    "ema8_gap", "ema21_gap", "ema55_gap", "ema200_gap",
    "atr_pct", "atr_rank", "rsi", "adx", "di_gap",
    "macd_hist_atr", "macd_slope_atr", "bb_pos", "bb_rank", "vwap_dev",
    "rel_vol", "vol_z", "trade_z", "taker_ratio", "taker_z",
    "body", "close_loc", "upper_wick", "lower_wick", "range_exp",
    "eff12", "eff24", "chop",
    "m15_trend", "m15_gap", "m15_adx", "m15_rsi", "m15_eff", "m15_atr_pct",
    "h1_trend", "h1_gap", "h1_adx", "h1_rsi", "h1_eff", "h1_atr_pct",
    "h4_trend", "h4_gap", "h4_adx", "h4_rsi", "h4_eff", "h4_atr_pct",
    "eth_ret1", "eth_ret3", "eth_ret6", "eth_ret12", "eth_ret24", "eth_ret72",
    "eth_gap", "eth_h1_trend", "btc_eth_corr", "btc_eth_spread_12", "btc_eth_spread_72",
    "premium", "premium_z", "premium_delta", "premium_abs_rank",
    "funding_rate", "funding_z", "funding_change", "derivative_pressure",
    "hour_sin", "hour_cos", "weekday",
    "regime_trend", "regime_range", "regime_high_vol", "regime_neutral",
    "setup_pull_long", "setup_pull_short", "setup_sweep_long", "setup_sweep_short",
    "setup_break_long", "setup_break_short", "setup_mom_long", "setup_mom_short",
]

EXPERTS: tuple[dict[str, Any], ...] = (
    {"id": 0, "name": "趋势多头", "column": "expert_trend_long", "direction": 1},
    {"id": 1, "name": "趋势空头", "column": "expert_trend_short", "direction": -1},
    {"id": 2, "name": "震荡多头反转", "column": "expert_range_long", "direction": 1},
    {"id": 3, "name": "震荡空头反转", "column": "expert_range_short", "direction": -1},
    {"id": 4, "name": "高波动多头突破", "column": "expert_high_long", "direction": 1},
    {"id": 5, "name": "高波动空头突破", "column": "expert_high_short", "direction": -1},
)
EXPERT_NAME = {int(e["id"]): str(e["name"]) for e in EXPERTS}


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


@dataclass
class Policy:
    risk: RiskConfig
    model: ModelConfig
    expert_target: int
    min_probability: float
    min_percentile: float
    min_margin: float
    cooldown: int
    expert_mask: int
    validation_metrics: dict[str, dict[str, float]] | None = None
    validation_scores: list[float] | None = None
    aggregate_score: float = -1e12


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
    breakeven_trigger: float,
    breakeven_lock: float,
    early_bars: int,
    early_cut_r: float,
    fee_rate: float,
    slippage_abs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(indices)
    labels = np.full(n, -1, dtype=np.int8)
    exits = np.full(n, -1, dtype=np.int64)
    net_r = np.zeros(n, dtype=np.float64)
    reasons = np.zeros(n, dtype=np.int8)
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
        max_favourable_r = 0.0
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
                favourable_r = (high[j] - entry) / risk
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
                favourable_r = (entry - low[j]) / risk
            if favourable_r > max_favourable_r:
                max_favourable_r = favourable_r
            if breakeven_trigger > 0.0 and max_favourable_r >= breakeven_trigger:
                protected = entry + direction * risk * breakeven_lock
                if direction > 0:
                    if protected > stop:
                        stop = protected
                else:
                    if protected < stop:
                        stop = protected
            bars_held = j - entry_i + 1
            if early_bars > 0 and bars_held >= early_bars and max_favourable_r < 0.55:
                current_r = ((close[j] - entry) * direction) / risk
                if current_r <= -early_cut_r:
                    exit_price = close[j] - direction * slippage_abs
                    reason = 4
                    exit_i = j
                    break
        gross = (exit_price - entry) * direction
        fees = fee_rate * (entry + exit_price)
        value = (gross - fees) / risk
        labels[k] = 1 if value > 0.0 else 0
        exits[k] = exit_i
        net_r[k] = value
        reasons[k] = reason
    return labels, exits, net_r, reasons


def mild_class_weights(y: np.ndarray) -> np.ndarray:
    positives = max(1, int(np.sum(y == 1)))
    negatives = max(1, int(np.sum(y == 0)))
    total = positives + negatives
    w_pos = math.sqrt(total / (2.0 * positives))
    w_neg = math.sqrt(total / (2.0 * negatives))
    return np.where(y == 1, w_pos, w_neg)


def make_expert_data(x: pd.DataFrame) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for expert in EXPERTS:
        idx = np.flatnonzero(x[str(expert["column"])].to_numpy(bool)).astype(np.int64)
        result[int(expert["id"])] = {
            "idx": idx,
            "direction": int(expert["direction"]),
            "name": str(expert["name"]),
        }
    return result


def fit_expert_models(
    x: pd.DataFrame,
    expert_data: dict[int, dict[str, Any]],
    risk: RiskConfig,
    model_cfg: ModelConfig,
    train_months: set[str],
) -> dict[int, HistGradientBoostingClassifier | None]:
    models: dict[int, HistGradientBoostingClassifier | None] = {}
    open_ = x["open"].to_numpy(float)
    high = x["high"].to_numpy(float)
    low = x["low"].to_numpy(float)
    close = x["close"].to_numpy(float)
    atr = x["atr"].to_numpy(float)
    all_months = x["month"].to_numpy()
    ordered_months = sorted(train_months)
    recency = {m: 0.88 + 0.24 * (i / max(1, len(ordered_months) - 1)) for i, m in enumerate(ordered_months)}
    for expert_id, data in expert_data.items():
        idx = data["idx"]
        labels, _, _, _ = compute_outcomes(
            idx, int(data["direction"]), open_, high, low, close, atr,
            risk.rr, risk.sl_atr, risk.min_stop_pct, risk.max_hold,
            risk.breakeven_trigger, risk.breakeven_lock, risk.early_bars, risk.early_cut_r,
            FEE_RATE, SLIPPAGE_ABS,
        )
        months = all_months[idx]
        mask = np.isin(months, list(train_months)) & (labels >= 0)
        y = labels[mask].astype(int)
        if len(y) < 70 or len(np.unique(y)) < 2 or min(np.sum(y == 0), np.sum(y == 1)) < 12:
            models[expert_id] = None
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
            n_iter_no_change=18,
            random_state=BASE_SEED + expert_id * 97,
        )
        time_w = np.array([recency.get(str(m), 1.0) for m in months[mask]], dtype=float)
        model.fit(X, y, sample_weight=mild_class_weights(y) * time_w)
        models[expert_id] = model
    return models


def build_expert_bundles(
    x: pd.DataFrame,
    expert_data: dict[int, dict[str, Any]],
    models: dict[int, HistGradientBoostingClassifier | None],
    risk: RiskConfig,
) -> dict[int, dict[str, Any]]:
    bundles: dict[int, dict[str, Any]] = {}
    open_ = x["open"].to_numpy(float)
    high = x["high"].to_numpy(float)
    low = x["low"].to_numpy(float)
    close = x["close"].to_numpy(float)
    atr = x["atr"].to_numpy(float)
    all_months = x["month"].to_numpy()
    for expert_id, data in expert_data.items():
        model = models.get(expert_id)
        if model is None:
            continue
        idx = data["idx"]
        labels, exits, net_r, reasons = compute_outcomes(
            idx, int(data["direction"]), open_, high, low, close, atr,
            risk.rr, risk.sl_atr, risk.min_stop_pct, risk.max_hold,
            risk.breakeven_trigger, risk.breakeven_lock, risk.early_bars, risk.early_cut_r,
            FEE_RATE, SLIPPAGE_ABS,
        )
        prob = model.predict_proba(x.iloc[idx][FEATURES].to_numpy(np.float64))[:, 1]
        bundles[expert_id] = {
            "idx": idx,
            "direction": int(data["direction"]),
            "name": str(data["name"]),
            "labels": labels,
            "exits": exits,
            "net_r": net_r,
            "reasons": reasons,
            "prob": prob,
            "months": all_months[idx],
        }
    return bundles


def events_from_expert(
    expert_id: int,
    bundle: dict[str, Any],
    eval_month: str,
    train_months: set[str],
    expert_target: int,
    min_probability: float,
    min_percentile: float,
) -> list[dict[str, Any]]:
    train_mask = np.isin(bundle["months"], list(train_months)) & (bundle["exits"] >= 0)
    train_prob = bundle["prob"][train_mask]
    if len(train_prob) < 55:
        return []
    desired_total = max(1.0, expert_target * max(1, len(train_months)))
    keep_fraction = min(0.22, max(0.002, desired_total / len(train_prob)))
    quantile = max(min_percentile, 1.0 - keep_fraction)
    percentile_threshold = float(np.quantile(train_prob, quantile))
    threshold = max(min_probability, percentile_threshold)
    sorted_train = np.sort(train_prob)
    eval_mask = (
        (bundle["months"] == eval_month)
        & (bundle["exits"] >= 0)
        & (bundle["prob"] >= threshold)
    )
    events: list[dict[str, Any]] = []
    for k in np.flatnonzero(eval_mask):
        score = float(np.searchsorted(sorted_train, bundle["prob"][k], side="right") / len(sorted_train))
        events.append({
            "signal_i": int(bundle["idx"][k]),
            "exit_i": int(bundle["exits"][k]),
            "direction": int(bundle["direction"]),
            "expert_id": int(expert_id),
            "expert": str(bundle["name"]),
            "prob": float(bundle["prob"][k]),
            "score": score,
            "net_r": float(bundle["net_r"][k]),
            "reason": int(bundle["reasons"][k]),
        })
    return events


def policy_events(
    bundles: dict[int, dict[str, Any]],
    policy: Policy,
    eval_month: str,
    train_months: set[str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for expert_id, bundle in bundles.items():
        if ((policy.expert_mask >> expert_id) & 1) == 0:
            continue
        events.extend(events_from_expert(
            expert_id, bundle, eval_month, train_months,
            policy.expert_target, policy.min_probability, policy.min_percentile,
        ))
    return events


def select_trades(events: list[dict[str, Any]], cooldown: int, min_margin: float) -> list[dict[str, Any]]:
    events = sorted(events, key=lambda e: (e["signal_i"], -e["score"], -e["prob"]))
    selected: list[dict[str, Any]] = []
    last_exit = -10**9
    i = 0
    while i < len(events):
        signal_i = events[i]["signal_i"]
        same: list[dict[str, Any]] = []
        while i < len(events) and events[i]["signal_i"] == signal_i:
            same.append(events[i])
            i += 1
        best_long = max((e for e in same if e["direction"] > 0), key=lambda e: e["score"], default=None)
        best_short = max((e for e in same if e["direction"] < 0), key=lambda e: e["score"], default=None)
        if best_long is not None and best_short is not None:
            if abs(best_long["score"] - best_short["score"]) < min_margin:
                continue
            best = best_long if best_long["score"] > best_short["score"] else best_short
        else:
            best = best_long if best_long is not None else best_short
        if best is None or signal_i <= last_exit + cooldown:
            continue
        selected.append(best)
        last_exit = int(best["exit_i"])
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


def wilson_lower_bound(wins: int, count: int, z: float = 1.0) -> float:
    if count <= 0:
        return 0.0
    p = wins / count
    denom = 1.0 + z * z / count
    centre = p + z * z / (2.0 * count)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * count)) / count)
    return (centre - margin) / denom


def month_score(m: dict[str, float], *, final_month: bool = False) -> float:
    count = int(m["trades"])
    wins = int(m["wins"])
    target_min = MIN_TRADES if final_month else max(9, MIN_TRADES - 6)
    target_max = MAX_TRADES if final_month else MAX_TRADES + 6
    target = (MIN_TRADES + MAX_TRADES) / 2.0
    lower = wilson_lower_bound(wins, count)
    score = (
        lower * 3200.0
        + m["win_rate"] * 1250.0
        + min(m["avg_win_loss_ratio"], 3.5) * 210.0
        + min(m["profit_factor"], 6.0) * 100.0
        + m["net_R"] * 14.0
        - m["max_drawdown_R"] * 12.0
        - abs(count - target) * 9.0
    )
    if count < target_min or count > target_max:
        score -= 3600.0 + abs(count - np.clip(count, target_min, target_max)) * 220.0
    if m["win_rate"] < MIN_WIN_RATE:
        score -= (MIN_WIN_RATE - m["win_rate"]) * 9200.0
    if m["avg_win_loss_ratio"] < MIN_RATIO:
        score -= (MIN_RATIO - m["avg_win_loss_ratio"]) * 3800.0
    if m["net_R"] <= 0:
        score -= 1000.0 + abs(m["net_R"]) * 55.0
    return float(score)


def refresh_policy_score(policy: Policy) -> None:
    scores = np.asarray(policy.validation_scores or [], dtype=float)
    if len(scores) == 0:
        policy.aggregate_score = -1e12
        return
    policy.aggregate_score = float(
        0.62 * np.min(scores)
        + 0.23 * np.median(scores)
        + 0.15 * scores[-1]
        - 0.35 * np.std(scores)
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
    expert_data: dict[int, dict[str, Any]],
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
        models = fit_expert_models(x, expert_data, risk, model_cfg, train_months)
        bundles = build_expert_bundles(x, expert_data, models, risk)
        event_cache: dict[tuple[int, float, float, int], list[dict[str, Any]]] = {}
        for policy in group:
            key = (policy.expert_target, policy.min_probability, policy.min_percentile, policy.expert_mask)
            if key not in event_cache:
                event_cache[key] = policy_events(bundles, policy, eval_month, train_months)
            chosen = select_trades(event_cache[key], policy.cooldown, policy.min_margin)
            add_validation(policy, eval_month, metrics(chosen), final_month=final_month)


def evaluate_policy(
    x: pd.DataFrame,
    expert_data: dict[int, dict[str, Any]],
    policy: Policy,
    train_months: set[str],
    eval_month: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    models = fit_expert_models(x, expert_data, policy.risk, policy.model, train_months)
    bundles = build_expert_bundles(x, expert_data, models, policy.risk)
    events = policy_events(bundles, policy, eval_month, train_months)
    selected = select_trades(events, policy.cooldown, policy.min_margin)
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
    return "+".join(active) if active else "专家候选"


def detailed_trades(x: pd.DataFrame, trades: list[dict[str, Any]], risk: RiskConfig, month: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for t in trades:
        i = int(t["signal_i"])
        entry_i = i + 1
        direction = int(t["direction"])
        entry = float(x.iloc[entry_i]["open"] + direction * SLIPPAGE_ABS)
        risk_abs = max(float(x.iloc[i]["atr"]) * risk.sl_atr, float(x.iloc[i]["close"]) * risk.min_stop_pct)
        exit_i = int(t["exit_i"])
        reason = {1: "TP", 2: "PROTECTED_STOP", 3: "TIME", 4: "EARLY_CUT"}.get(int(t["reason"]), "UNKNOWN")
        rows.append({
            "signal_time_utc": x.index[i].isoformat(),
            "entry_time_utc": x.index[entry_i].isoformat(),
            "exit_time_utc": x.index[exit_i].isoformat(),
            "month": month,
            "direction": "LONG" if direction > 0 else "SHORT",
            "expert": str(t.get("expert", EXPERT_NAME.get(int(t.get("expert_id", -1)), "未知专家"))),
            "setup": setup_name(x, i, direction),
            "probability": float(t["prob"]),
            "training_percentile": float(t["score"]),
            "entry": entry,
            "risk_abs": risk_abs,
            "target_R": risk.rr,
            "net_R": float(t["net_r"]),
            "win": bool(t["net_r"] > 0),
            "exit_reason": reason,
            "bars": exit_i - entry_i,
        })
    return pd.DataFrame(rows)


def expert_breakdown(trades: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for expert_id, name in EXPERT_NAME.items():
        subset = [t for t in trades if int(t.get("expert_id", -1)) == expert_id]
        if subset:
            result[name] = metrics(subset)
    return result


def synthetic_inputs(rows: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260808)
    times = np.arange(1_700_000_000_000, 1_700_000_000_000 + rows * 300_000, 300_000, dtype=np.int64)
    returns = rng.normal(0.0, 0.0012, rows) + 0.00015 * np.sin(np.arange(rows) / 80.0)
    close = 50000.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(15.0, close * rng.uniform(0.0004, 0.0022, rows))
    high = np.maximum(open_, close) + spread * rng.uniform(0.2, 1.0, rows)
    low = np.minimum(open_, close) - spread * rng.uniform(0.2, 1.0, rows)
    volume = rng.lognormal(5.0, 0.6, rows)
    taker = volume * np.clip(0.5 + rng.normal(0, 0.08, rows), 0.05, 0.95)
    base = pd.DataFrame({
        "open_time": times, "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "close_time": times + 299_999, "quote_volume": volume * close,
        "trade_count": rng.integers(200, 2000, rows), "taker_buy_volume": taker,
        "taker_buy_quote": taker * close, "ignore": 0,
    })
    eth_close = 3000.0 * np.exp(np.cumsum(returns * 1.15 + rng.normal(0, 0.0005, rows)))
    eth = base.copy()
    eth["open"] = np.r_[eth_close[0], eth_close[:-1]]
    eth["close"] = eth_close
    eth["high"] = np.maximum(eth["open"], eth["close"]) * 1.001
    eth["low"] = np.minimum(eth["open"], eth["close"]) * 0.999
    premium = base.copy()
    pclose = rng.normal(0, 0.00025, rows).cumsum() * 0.02
    premium["open"] = np.r_[pclose[0], pclose[:-1]]
    premium["close"] = pclose
    premium["high"] = np.maximum(premium["open"], premium["close"]) + 0.00002
    premium["low"] = np.minimum(premium["open"], premium["close"]) - 0.00002
    fund_idx = np.arange(0, rows, 96)
    funding = pd.DataFrame({"calc_time": times[fund_idx], "funding_rate": rng.normal(0.00005, 0.00008, len(fund_idx))})
    return base, eth, premium, funding


def self_test() -> None:
    base, eth, premium, funding = synthetic_inputs()
    x, alignment_audit = add_features(base, eth, premium, funding)
    if x.empty:
        raise RuntimeError("V8.1 self-test produced no feature rows")
    missing = [f for f in FEATURES if f not in x.columns]
    if missing:
        raise RuntimeError(f"V8.1 self-test missing features: {missing}")
    # Regression test for the real GitHub failure: auxiliary archives may
    # contain sparse missing bars or a different timestamp precision.
    eth_gap = eth.drop(index=[250, 251]).reset_index(drop=True)
    premium_gap = premium.drop(index=[100, 101, 102, 900]).reset_index(drop=True)
    premium_gap["open_time"] = premium_gap["open_time"].astype("int64") * 1000
    x_gap, gap_audit = add_features(base, eth_gap, premium_gap, funding)
    if x_gap.empty:
        raise RuntimeError("V8.1 alignment regression test produced no feature rows")
    if gap_audit["eth_exact_missing"] < 2 or gap_audit["premium_exact_missing"] < 4:
        raise RuntimeError("V8.1 alignment regression test did not exercise missing timestamps")
    if gap_audit["eth_missing_after_past_fill"] != 0 or gap_audit["premium_missing_after_past_fill"] != 0:
        raise RuntimeError("V8.1 past-only alignment regression test failed")

    experts = make_expert_data(x)
    if set(experts) != set(range(6)):
        raise RuntimeError("V8.1 self-test expert registry mismatch")
    indices = np.arange(100, min(130, len(x) - 2), dtype=np.int64)
    compute_outcomes(
        indices, 1, x["open"].to_numpy(float), x["high"].to_numpy(float),
        x["low"].to_numpy(float), x["close"].to_numpy(float), x["atr"].to_numpy(float),
        2.0, 1.2, 0.004, 144, 1.0, 0.1, 48, 0.4, FEE_RATE, SLIPPAGE_ABS,
    )
    print(f"V8.1 self-test passed: rows={len(x)}, features={len(FEATURES)}, experts={len(experts)}")


def main() -> None:
    if RESULTS.exists():
        for item in RESULTS.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    RESULTS.mkdir(exist_ok=True)

    raw, audit = load_official_data()
    eth, eth_audit = load_auxiliary_kline("ETHUSDT", "klines")
    premium, premium_audit = load_auxiliary_kline(SYMBOL, "premiumIndexKlines")
    funding, funding_audit = load_funding_rate()
    audit["auxiliary_sources"] = {
        "eth_klines": eth_audit,
        "premium_index_klines": premium_audit,
        "funding_rate": funding_audit,
    }
    x, alignment_audit = add_features(raw, eth, premium, funding)
    audit["auxiliary_alignment"] = alignment_audit
    expert_data = make_expert_data(x)

    risk_grid: list[RiskConfig] = []
    for rr in (1.8, 2.0, 2.2):
        for sl in (1.1, 1.4):
            for min_stop in (0.0036, 0.0045):
                hold = 144 if rr <= 2.0 else 288
                management = (0.95, 0.10, 48, 0.40) if rr <= 2.0 else (1.10, 0.15, 72, 0.35)
                risk_grid.append(RiskConfig(rr, sl, min_stop, hold, *management))
    model_grid = [
        ModelConfig(2, 0.035, 210, 4.0, 45),
        ModelConfig(3, 0.025, 260, 8.0, 65),
    ]
    expert_targets = (3, 5, 7)
    min_probabilities = (0.52, 0.57, 0.62)
    min_percentiles = (0.88, 0.92, 0.95)
    min_margins = (0.03, 0.07)
    cooldowns = (3, 8)
    expert_masks = (63, 3, 12, 48, 15, 51, 21, 42)

    stage1: list[Policy] = []
    train0 = {MONTHS[0], MONTHS[1], MONTHS[2]}
    for risk in risk_grid:
        for model_cfg in model_grid:
            models = fit_expert_models(x, expert_data, risk, model_cfg, train0)
            bundles = build_expert_bundles(x, expert_data, models, risk)
            if not bundles:
                continue
            for target in expert_targets:
                for min_prob in min_probabilities:
                    for min_pct in min_percentiles:
                        for margin in min_margins:
                            for cooldown in cooldowns:
                                for expert_mask in expert_masks:
                                    policy = Policy(risk, model_cfg, target, min_prob, min_pct, margin, cooldown, expert_mask)
                                    events = policy_events(bundles, policy, MONTHS[3], train0)
                                    add_validation(policy, MONTHS[3], metrics(select_trades(events, cooldown, margin)))
                                    stage1.append(policy)
    stage1.sort(key=lambda p: p.aggregate_score, reverse=True)
    active = stage1[:120]

    rolling = [
        (4, set(MONTHS[:4]), 90),
        (5, set(MONTHS[:5]), 68),
        (6, set(MONTHS[:6]), 50),
        (7, set(MONTHS[:7]), 36),
    ]
    for eval_idx, train_months, keep in rolling:
        evaluate_policy_batch(x, expert_data, active, train_months, MONTHS[eval_idx])
        active.sort(key=lambda p: p.aggregate_score, reverse=True)
        active = active[:keep]

    evaluate_policy_batch(x, expert_data, active, set(MONTHS[:8]), MONTHS[8], final_month=True)
    active.sort(key=lambda p: p.aggregate_score, reverse=True)
    may_qualified = [
        p for p in active
        if p.validation_metrics
        and MIN_TRADES <= p.validation_metrics[MONTHS[8]]["trades"] <= MAX_TRADES
        and p.validation_metrics[MONTHS[8]]["win_rate"] >= MIN_WIN_RATE
        and p.validation_metrics[MONTHS[8]]["avg_win_loss_ratio"] >= MIN_RATIO
    ]
    selected = (may_qualified or active)[0]

    june_metrics, june_trades = evaluate_policy(x, expert_data, selected, set(MONTHS[:9]), MONTHS[9])
    may_metrics, may_trades = evaluate_policy(x, expert_data, selected, set(MONTHS[:8]), MONTHS[8])
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

    active_experts = [EXPERT_NAME[i] for i in range(6) if ((selected.expert_mask >> i) & 1) != 0]
    development = selected.validation_metrics or {}
    development_win_rates = [float(v["win_rate"]) for v in development.values()]
    status = {
        "qualified": qualified,
        "engine": ENGINE_NAME,
        "method": "Sep-Nov train→Dec prefilter; rolling Jan-Apr validation; May selection; June untouched OOS",
        "architecture": "1h regime gate → 15m structure expert → 5m entry; six independent experts; long/short/no-trade rejection",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "months": list(MONTHS),
        "data_context": ["BTCUSDT perpetual klines", "ETHUSDT perpetual klines", "BTC premium-index klines", "BTC funding rate"],
        "constraints": {
            "min_trades": MIN_TRADES,
            "max_trades": MAX_TRADES,
            "min_win_rate": MIN_WIN_RATE,
            "min_avg_win_loss_ratio": MIN_RATIO,
        },
        "selected_policy": {
            "risk": asdict(selected.risk),
            "model": asdict(selected.model),
            "expert_target_per_month": selected.expert_target,
            "min_probability": selected.min_probability,
            "min_training_percentile": selected.min_percentile,
            "long_short_margin": selected.min_margin,
            "cooldown_bars": selected.cooldown,
            "expert_mask": selected.expert_mask,
            "active_experts": active_experts,
        },
        "development_monthly_stats": development,
        "development_floor_win_rate": min(development_win_rates) if development_win_rates else 0.0,
        "monthly_stats": {MONTHS[8]: may_metrics, MONTHS[9]: june_metrics},
        "expert_stats": {
            MONTHS[8]: expert_breakdown(may_trades),
            MONTHS[9]: expert_breakdown(june_trades),
        },
        "may_hard_qualified_candidates": len(may_qualified),
        "search_policies": len(stage1),
    }
    (RESULTS / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "selected_policy.json").write_text(json.dumps(status["selected_policy"], ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for rank, policy in enumerate(active[:25], start=1):
        row: dict[str, Any] = {
            "rank": rank,
            "aggregate_score": policy.aggregate_score,
            "rr": policy.risk.rr,
            "sl_atr": policy.risk.sl_atr,
            "min_stop_pct": policy.risk.min_stop_pct,
            "max_hold": policy.risk.max_hold,
            "expert_target": policy.expert_target,
            "min_probability": policy.min_probability,
            "min_percentile": policy.min_percentile,
            "min_margin": policy.min_margin,
            "cooldown": policy.cooldown,
            "expert_mask": policy.expert_mask,
        }
        for month, value in (policy.validation_metrics or {}).items():
            for key, val in value.items():
                row[f"{month}_{key}"] = val
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "candidate_leaderboard.csv", index=False)

    report = f"""# BTCUSDT 5分钟 多周期专家拒绝交易 V8.1 回测报告

- 数据：Binance USDⓈ-M 永续官方5分钟K线，BTC主数据{audit['actual_rows']:,}根；并加入ETHUSDT、BTC溢价指数和资金费率。
- 方法：9–11月训练→12月预筛；1–4月逐月滚动验证；5月最终选择；6月完全样本外。
- 架构：1小时判断市场状态，15分钟识别结构，5分钟确认入场；趋势多、趋势空、震荡多、震荡空、高波动多、高波动空分别训练。
- 拒绝机制：多空同时出现时必须拉开训练分位优势；概率或训练分位不足时输出“不交易”。
- 筛选原则：优先提高开发月份最低分和最低胜率，不按总收益单独选型。
- 成本：单边手续费{FEE_RATE*100:.3f}%；每次成交滑点{SLIPPAGE_ABS:.1f} USDT。
- 最终验收：**{'达到全部要求' if qualified else '未达到全部要求'}**。

## 选择策略

- 活跃专家：{'、'.join(active_experts)}
- 每位专家历史月度候选目标：{selected.expert_target}
- 最低盈利概率：{selected.min_probability:.2f}
- 最低训练分位：{selected.min_percentile:.2%}
- 多空优势阈值：{selected.min_margin:.2%}
- 固定目标：{selected.risk.rr:.2f}R
- 止损：max({selected.risk.sl_atr:.2f}×ATR, {selected.risk.min_stop_pct*100:.3f}%价格)
- 5月硬条件候选数量：{len(may_qualified)}

## 月度结果

| 月份 | 交易 | 胜率 | 平均盈利/平均亏损 | 盈利因子 | 净R | 最大回撤R |
|---|---:|---:|---:|---:|---:|---:|
| {MONTHS[8]} | {may_metrics['trades']} | {may_metrics['win_rate']*100:.2f}% | {may_metrics['avg_win_loss_ratio']:.3f} | {may_metrics['profit_factor']:.3f} | {may_metrics['net_R']:.3f} | {may_metrics['max_drawdown_R']:.3f} |
| {MONTHS[9]} | {june_metrics['trades']} | {june_metrics['win_rate']*100:.2f}% | {june_metrics['avg_win_loss_ratio']:.3f} | {june_metrics['profit_factor']:.3f} | {june_metrics['net_R']:.3f} | {june_metrics['max_drawdown_R']:.3f} |

## 保守假设

信号在5分钟K线收盘后确认，下一根K线开盘成交；15分钟、1小时和4小时特征只使用上一根完整高周期K线；同一根K线同时触及止损和止盈时按止损优先；资金费率延迟一根5分钟K线后使用；6月没有参与模型、阈值、专家组合或策略筛选。
"""
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    (RESULTS / "run_identity.txt").write_text(
        f"{ENGINE_NAME}\nmonths={','.join(MONTHS)}\noutput=results_v8\n",
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
