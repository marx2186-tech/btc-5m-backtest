from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from numba import njit

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:  # One-file runner fallback.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.5.1"])
    from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
RESULTS = ROOT / "results_v9"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

ENGINE_VERSION = "V9.0"
ENGINE_NAME = "BTC 5m expert-specific walk-forward ensemble V9.0"
OOS_MONTH = "2026-06"

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
    "base_seed": 20260909,
    "combination_top_per_expert": 2,
    "combination_beam_width": 160,
    "expert_elimination": {
        "max_consecutive_zero_trade_months": 2,
        "min_floor_win_rate": 0.40,
        "min_cumulative_win_rate": 0.48,
        "max_negative_expectancy_months": 3,
        "min_active_months": 2,
        "min_total_trades": 8
    }
}


def load_request() -> dict[str, Any]:
    request = dict(DEFAULT_REQUEST)
    request["expert_elimination"] = dict(DEFAULT_REQUEST["expert_elimination"])
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(ROOT / "request.v9.json")))
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("request.v9.json must contain a JSON object")
        elimination = loaded.pop("expert_elimination", None)
        request.update(loaded)
        if elimination is not None:
            if not isinstance(elimination, dict):
                raise ValueError("expert_elimination must be an object")
            request["expert_elimination"].update(elimination)
    return request


REQUEST = load_request()
SYMBOL = str(REQUEST["symbol"]).upper()
INTERVAL = str(REQUEST["interval"]).lower()
EVAL_MONTHS = tuple(str(x) for x in REQUEST["months"])
if len(EVAL_MONTHS) != 2:
    raise ValueError("V9 requires exactly two evaluation months")
first_eval = pd.Period(EVAL_MONTHS[0], freq="M")
last_eval = pd.Period(EVAL_MONTHS[1], freq="M")
if last_eval != first_eval + 1:
    raise ValueError("Evaluation months must be consecutive")
if str(last_eval) != OOS_MONTH:
    raise ValueError(f"V9 hard-codes {OOS_MONTH} as the untouched out-of-sample month")
MONTHS = tuple(str(first_eval - offset) for offset in range(8, 0, -1)) + EVAL_MONTHS
DEVELOPMENT_MONTHS = MONTHS[3:9]  # Dec through May.
FEE_RATE = float(REQUEST["fee_rate_per_side"])
TICK_SIZE = float(REQUEST["tick_size"])
SLIPPAGE_ABS = TICK_SIZE * int(REQUEST["slippage_ticks_per_fill"])
MIN_TRADES = int(REQUEST["min_trades_per_month"])
MAX_TRADES = int(REQUEST["max_trades_per_month"])
MIN_WIN_RATE = float(REQUEST["min_win_rate"])
MIN_RATIO = float(REQUEST["min_avg_win_loss_ratio"])
BASE_SEED = int(REQUEST["base_seed"])
TOP_PER_EXPERT = int(REQUEST["combination_top_per_expert"])
BEAM_WIDTH = int(REQUEST["combination_beam_width"])
ELIMINATION = dict(REQUEST["expert_elimination"])

if FEE_RATE != 0.0005:
    raise ValueError("V9 fixed requirement: fee_rate_per_side must be 0.0005")
if abs(SLIPPAGE_ABS - 0.2) > 1e-12:
    raise ValueError("V9 fixed requirement: each fill slippage must equal 0.2 USDT")

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


def assert_oos_isolation(train_months: Iterable[str], context: str) -> None:
    months = {str(m) for m in train_months}
    if OOS_MONTH in months:
        raise RuntimeError(f"OOS leakage blocked: {OOS_MONTH} appeared in {context}")


def walk_forward_plan() -> list[tuple[str, set[str]]]:
    plan: list[tuple[str, set[str]]] = []
    for eval_idx in range(3, 9):
        eval_month = MONTHS[eval_idx]
        train_months = set(MONTHS[:eval_idx])
        assert_oos_isolation(train_months, f"training plan for {eval_month}")
        if eval_month == OOS_MONTH:
            raise RuntimeError("June must not appear in development plan")
        plan.append((eval_month, train_months))
    return plan


def download(url: str, path: Path, attempts: int = 6) -> bytes:
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                timeout=90,
                headers={"User-Agent": "btc-expert-specific-v9/1.0"},
            )
            response.raise_for_status()
            path.write_bytes(response.content)
            return response.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** min(attempt, 4))
    raise RuntimeError(f"Download failed: {url}: {last}")


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


def load_official_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    files: list[dict[str, Any]] = []
    for month in MONTHS:
        name = f"{SYMBOL}-{INTERVAL}-{month}.zip"
        base = f"https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{INTERVAL}"
        raw, digest = read_verified_zip(f"{base}/{name}", f"{base}/{name}.CHECKSUM", name)
        frame = parse_kline_csv(read_single_csv_zip(raw, name))
        frames.append(frame)
        files.append({"file": name, "sha256": digest, "rows": int(len(frame))})

    data = pd.concat(frames, ignore_index=True).sort_values("open_time").reset_index(drop=True)
    start = pd.Timestamp(f"{MONTHS[0]}-01T00:00:00Z")
    end = pd.Timestamp(f"{MONTHS[-1]}-01T00:00:00Z") + pd.offsets.MonthBegin(1)
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    expected_times = np.arange(start_ms, end_ms, STEP_MS, dtype=np.int64)
    times = data["open_time"].astype("int64").to_numpy()
    unique_times = np.unique(times)
    missing = np.setdiff1d(expected_times, unique_times)
    extra = np.setdiff1d(unique_times, expected_times)
    o, h, l, c = (data[k].to_numpy(float) for k in ("open", "high", "low", "close"))
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
        "duplicate_timestamps": int(pd.Series(times).duplicated().sum()),
        "missing_rows": int(len(missing)),
        "extra_rows": int(len(extra)),
        "off_grid_rows": int(np.sum((times - start_ms) % STEP_MS != 0)),
        "invalid_close_time_rows": int(np.sum(data["close_time"].astype("int64").to_numpy() != times + STEP_MS - 1)),
        "invalid_ohlc_rows": int(np.sum(~valid_ohlc)),
        "files": files,
    }
    audit["passed"] = bool(
        len(data) == len(expected_times)
        and len(unique_times) == len(expected_times)
        and audit["duplicate_timestamps"] == 0
        and len(missing) == 0
        and len(extra) == 0
        and audit["off_grid_rows"] == 0
        and audit["invalid_close_time_rows"] == 0
        and np.all(valid_ohlc)
    )
    if not audit["passed"]:
        raise RuntimeError("Data audit failed: " + json.dumps(audit, ensure_ascii=False, indent=2))
    return data, audit


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
    return data.reset_index(drop=True), {
        "symbol": symbol,
        "data_type": data_type,
        "files": files,
        "rows": int(len(data)),
    }


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
            time_col, rate_col, frame = numeric.columns[0], numeric.columns[-1], numeric
        out = pd.DataFrame({
            "calc_time": pd.to_numeric(frame[time_col], errors="coerce"),
            "funding_rate": pd.to_numeric(frame[rate_col], errors="coerce"),
        }).dropna()
        frames.append(out)
        files.append({"file": name, "sha256": digest, "rows": int(len(out))})
    data = pd.concat(frames, ignore_index=True).sort_values("calc_time").drop_duplicates("calc_time")
    if data.empty:
        raise RuntimeError("Funding-rate archive is empty")
    return data.reset_index(drop=True), {
        "symbol": SYMBOL,
        "data_type": "fundingRate",
        "files": files,
        "rows": int(len(data)),
    }


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def zscore(series: pd.Series, length: int) -> pd.Series:
    mean = series.rolling(length).mean()
    std = series.rolling(length).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def normalized_series(frame: pd.DataFrame, time_col: str, value_col: str) -> pd.Series:
    raw_time = pd.to_numeric(frame[time_col], errors="coerce")
    raw_value = pd.to_numeric(frame[value_col], errors="coerce")
    valid = raw_time.notna() & raw_value.notna()
    raw_time = raw_time.loc[valid].astype("float64")
    raw_value = raw_value.loc[valid].astype("float64")
    if raw_time.empty:
        return pd.Series(dtype="float64")
    median_ts = float(raw_time.median())
    if median_ts >= 1e17:
        millis = raw_time / 1_000_000.0
    elif median_ts >= 1e14:
        millis = raw_time / 1_000.0
    elif median_ts < 1e11:
        millis = raw_time * 1_000.0
    else:
        millis = raw_time
    idx = pd.to_datetime(np.rint(millis).astype("int64"), unit="ms", utc=True).round("5min")
    series = pd.Series(raw_value.to_numpy(float), index=idx).sort_index()
    return series.groupby(level=0).last()


def add_features(
    data: pd.DataFrame,
    eth_data: pd.DataFrame,
    premium_data: pd.DataFrame,
    funding_data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    x = data.copy()
    x.index = pd.to_datetime(x["open_time"], unit="ms", utc=True)
    o, h, l, c, v = (x[k].astype(float) for k in ("open", "high", "low", "close", "volume"))

    eth_series = normalized_series(eth_data, "open_time", "close")
    premium_series = normalized_series(premium_data, "open_time", "close")
    eth_exact = eth_series.reindex(x.index)
    premium_exact = premium_series.reindex(x.index)
    eth_close = eth_series.reindex(x.index, method="ffill", tolerance=pd.Timedelta(minutes=15))
    premium_close = premium_series.reindex(x.index, method="ffill", tolerance=pd.Timedelta(minutes=60))
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
        "alignment_rule": "nearest-5m normalization; past-only ffill; ETH<=15m; premium<=60m",
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

    funding = funding_data.copy()
    funding.index = pd.to_datetime(funding["calc_time"], unit="ms", utc=True)
    funding_series = funding["funding_rate"].astype(float).sort_index()
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

    up, down = h.diff(), -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_di = 100 * rma(plus_dm, 14) / x["atr"].replace(0, np.nan)
    minus_di = 100 * rma(minus_dm, 14) / x["atr"].replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["plus_di"], x["minus_di"] = plus_di, minus_di
    x["di_gap"] = (plus_di - minus_di) / 100.0
    x["adx"] = rma(dx, 14)

    change = c.diff()
    rs = rma(change.clip(lower=0), 14) / rma(-change.clip(upper=0), 14).replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + rs)
    fast, slow = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    macd_hist = macd - macd.ewm(span=9, adjust=False).mean()
    x["macd_hist_atr"] = macd_hist / x["atr"].replace(0, np.nan)
    x["macd_slope_atr"] = macd_hist.diff() / x["atr"].replace(0, np.nan)

    basis = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    lower, upper = basis - 2 * sd, basis + 2 * sd
    width = (upper - lower) / basis.replace(0, np.nan)
    x["bb_pos"] = (c - lower) / (upper - lower).replace(0, np.nan)
    x["bb_rank"] = width.rolling(288).rank(pct=True)
    x["bb_width"] = width

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
    for window in (6, 12, 24, 48):
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
        hup, hdown = bars["high"].diff(), -bars["low"].diff()
        hpdm = hup.where((hup > hdown) & (hup > 0), 0.0)
        hmdm = hdown.where((hdown > hup) & (hdown > 0), 0.0)
        hpdi = 100 * rma(hpdm, 14) / hatr.replace(0, np.nan)
        hmdi = 100 * rma(hmdm, 14) / hatr.replace(0, np.nan)
        hdx = 100 * (hpdi - hmdi).abs() / (hpdi + hmdi).replace(0, np.nan)
        delta = bars["close"].diff()
        hrs = rma(delta.clip(lower=0), 14) / rma(-delta.clip(upper=0), 14).replace(0, np.nan)
        heff = (bars["close"] - bars["close"].shift(12)).abs() / bars["close"].diff().abs().rolling(12).sum().replace(0, np.nan)
        trend = np.where((bars["close"] > e20) & (e20 > e50), 1.0,
                         np.where((bars["close"] < e20) & (e20 < e50), -1.0, 0.0))
        values = {
            f"{prefix}_trend": pd.Series(trend, index=bars.index),
            f"{prefix}_gap": (e20 - e50) / bars["close"].replace(0, np.nan),
            f"{prefix}_adx": rma(hdx, 14),
            f"{prefix}_rsi": 100 - 100 / (1 + hrs),
            f"{prefix}_eff": heff,
            f"{prefix}_atr_pct": hatr / bars["close"].replace(0, np.nan),
        }
        for name, series in values.items():
            x[name] = series.shift(1).reindex(x.index, method="ffill")

    htf("15min", "m15")
    htf("60min", "h1")
    htf("240min", "h4")

    eth_hour = eth_close.resample("60min", label="right", closed="left").last().dropna()
    eth20, eth50 = eth_hour.ewm(span=20, adjust=False).mean(), eth_hour.ewm(span=50, adjust=False).mean()
    eth_trend = pd.Series(np.where((eth_hour > eth20) & (eth20 > eth50), 1.0,
                                   np.where((eth_hour < eth20) & (eth20 < eth50), -1.0, 0.0)), index=eth_hour.index)
    x["eth_h1_trend"] = eth_trend.shift(1).reindex(x.index, method="ffill")

    # V9 entry structures. Trend-short preserves V8's strongest family; the other
    # five structures are rebuilt to require a two-stage context + 5m confirmation.
    reclaim_long = (
        (l <= x["ema21"] + 0.35 * x["atr"]) & (c > x["ema8"]) & (c > o)
        & (x["close_loc"] >= 0.60) & (x["macd_slope_atr"] > 0)
    )
    reclaim_short = (
        (h >= x["ema21"] - 0.35 * x["atr"]) & (c < x["ema8"]) & (c < o)
        & (x["close_loc"] <= 0.40) & (x["macd_slope_atr"] < 0)
    )
    continuation_long = (
        (c > x["don6h"]) & (x["body"] >= 0.42) & (x["close_loc"] >= 0.70)
        & (x["taker_ratio"] >= 0.53) & (x["rel_vol"] >= 0.85)
    )
    continuation_short = (
        (c < x["don6l"]) & (x["body"] >= 0.42) & (x["close_loc"] <= 0.30)
        & (x["taker_ratio"] <= 0.47) & (x["rel_vol"] >= 0.85)
    )
    range_reversal_long = (
        (l < x["don12l"]) & (c > x["don12l"]) & (c > o)
        & (x["lower_wick"] >= 0.34) & (x["close_loc"] >= 0.58)
        & (x["vwap_dev"] <= 0.25) & (x["rsi"] > x["rsi"].shift(1))
        & (x["taker_ratio"] >= 0.48)
    )
    range_reversal_short = (
        (h > x["don12h"]) & (c < x["don12h"]) & (c < o)
        & (x["upper_wick"] >= 0.34) & (x["close_loc"] <= 0.42)
        & (x["vwap_dev"] >= -0.25) & (x["rsi"] < x["rsi"].shift(1))
        & (x["taker_ratio"] <= 0.52)
    )
    squeeze = x["bb_rank"].shift(3).rolling(6).min() <= 0.28
    high_break_long = (
        squeeze & (c > x["don24h"]) & (x["range_exp"] >= 1.25)
        & (x["body"] >= 0.55) & (x["close_loc"] >= 0.76)
        & (x["rel_vol"] >= 1.05) & (x["taker_ratio"] >= 0.55)
    )
    high_break_short = (
        squeeze & (c < x["don24l"]) & (x["range_exp"] >= 1.25)
        & (x["body"] >= 0.55) & (x["close_loc"] <= 0.24)
        & (x["rel_vol"] >= 1.05) & (x["taker_ratio"] <= 0.45)
    )

    high_vol = (x["atr_rank"] >= 0.78) | ((x["range_exp"] >= 1.45) & (x["atr_rank"] >= 0.62))
    trend_regime = (
        ~high_vol & (x["h1_trend"] != 0) & (x["h1_trend"] == x["h4_trend"])
        & (x["h1_adx"] >= 18) & (x["chop"] <= 59)
    )
    range_regime = (~high_vol) & (~trend_regime) & ((x["h1_adx"] <= 24) | (x["chop"] >= 57))
    neutral_regime = ~(high_vol | trend_regime | range_regime)
    x["regime"] = np.select([trend_regime, range_regime, high_vol, neutral_regime], [0, 1, 2, 3], default=3).astype(float)
    x["regime_trend"] = (x["regime"] == 0).astype(float)
    x["regime_range"] = (x["regime"] == 1).astype(float)
    x["regime_high_vol"] = (x["regime"] == 2).astype(float)
    x["regime_neutral"] = (x["regime"] == 3).astype(float)

    trend_long_context = (
        trend_regime & (x["h1_trend"] > 0) & (x["m15_trend"] >= 0)
        & (x["eth_h1_trend"] >= 0) & (x["derivative_pressure"] <= 3.2)
        & (x["premium_z"] <= 2.0)
    )
    trend_short_context = (
        trend_regime & (x["h1_trend"] < 0) & (x["m15_trend"] <= 0)
        & (x["eth_h1_trend"] <= 0) & (x["derivative_pressure"] >= -4.2)
    )
    range_long_context = (
        range_regime & (x["m15_rsi"] <= 51) & (x["bb_pos"] <= 0.42)
        & (x["premium_z"] <= 1.3) & (x["derivative_pressure"] <= 2.8)
    )
    range_short_context = (
        range_regime & (x["m15_rsi"] >= 49) & (x["bb_pos"] >= 0.58)
        & (x["premium_z"] >= -1.3) & (x["derivative_pressure"] >= -2.8)
    )
    high_long_context = (
        high_vol & (x["h1_trend"] >= 0) & (x["m15_trend"] >= 0)
        & (x["eth_h1_trend"] >= 0) & (x["eth_ret12"] >= -0.004)
        & (x["premium_delta"] >= -0.00008) & (x["funding_z"] <= 2.8)
    )
    high_short_context = (
        high_vol & (x["h1_trend"] <= 0) & (x["m15_trend"] <= 0)
        & (x["eth_h1_trend"] <= 0) & (x["eth_ret12"] <= 0.004)
        & (x["premium_delta"] <= 0.00008) & (x["funding_z"] >= -2.8)
    )

    x["setup_reclaim_long"] = reclaim_long.astype(float)
    x["setup_reclaim_short"] = reclaim_short.astype(float)
    x["setup_continuation_long"] = continuation_long.astype(float)
    x["setup_continuation_short"] = continuation_short.astype(float)
    x["setup_range_reversal_long"] = range_reversal_long.astype(float)
    x["setup_range_reversal_short"] = range_reversal_short.astype(float)
    x["setup_high_break_long"] = high_break_long.astype(float)
    x["setup_high_break_short"] = high_break_short.astype(float)

    x["expert_trend_long"] = (trend_long_context & (reclaim_long | continuation_long)).astype(float)
    x["expert_trend_short"] = (trend_short_context & (reclaim_short | continuation_short)).astype(float)
    x["expert_range_long"] = (range_long_context & range_reversal_long).astype(float)
    x["expert_range_short"] = (range_short_context & range_reversal_short).astype(float)
    x["expert_high_long"] = (high_long_context & high_break_long).astype(float)
    x["expert_high_short"] = (high_short_context & high_break_short).astype(float)

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
    "macd_hist_atr", "macd_slope_atr", "bb_pos", "bb_rank", "bb_width", "vwap_dev",
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
    "setup_reclaim_long", "setup_reclaim_short",
    "setup_continuation_long", "setup_continuation_short",
    "setup_range_reversal_long", "setup_range_reversal_short",
    "setup_high_break_long", "setup_high_break_short",
]

EXPERTS: tuple[dict[str, Any], ...] = (
    {"id": 0, "key": "trend_long", "name": "趋势多头", "column": "expert_trend_long", "direction": 1},
    {"id": 1, "key": "trend_short", "name": "趋势空头", "column": "expert_trend_short", "direction": -1},
    {"id": 2, "key": "range_long", "name": "震荡多头反转", "column": "expert_range_long", "direction": 1},
    {"id": 3, "key": "range_short", "name": "震荡空头反转", "column": "expert_range_short", "direction": -1},
    {"id": 4, "key": "high_long", "name": "高波动多头突破", "column": "expert_high_long", "direction": 1},
    {"id": 5, "key": "high_short", "name": "高波动空头突破", "column": "expert_high_short", "direction": -1},
)
EXPERT_BY_ID = {int(e["id"]): e for e in EXPERTS}
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


@dataclass(frozen=True)
class ExpertPolicy:
    expert_id: int
    risk: RiskConfig
    model: ModelConfig
    monthly_target: int
    min_probability: float
    min_percentile: float
    cooldown: int

    @property
    def key(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class ExpertCandidate:
    policy: ExpertPolicy
    monthly_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    monthly_events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    aggregate_score: float = -1e12
    eligible: bool = False
    elimination_reasons: list[str] = field(default_factory=list)


@dataclass
class CombinationCandidate:
    policies: dict[int, ExpertPolicy]
    monthly_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    monthly_expert_metrics: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    monthly_trades: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    aggregate_score: float = -1e12

    @property
    def key(self) -> str:
        raw = "|".join(f"{k}:{v.key}" for k, v in sorted(self.policies.items()))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


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
            # Conservative same-candle ordering: stop is always checked first.
            if direction > 0:
                if low[j] <= stop:
                    exit_price, reason, exit_i = stop - slippage_abs, 2, j
                    break
                if high[j] >= target:
                    exit_price, reason, exit_i = target - slippage_abs, 1, j
                    break
                favourable_r = (high[j] - entry) / risk
            else:
                if high[j] >= stop:
                    exit_price, reason, exit_i = stop + slippage_abs, 2, j
                    break
                if low[j] <= target:
                    exit_price, reason, exit_i = target + slippage_abs, 1, j
                    break
                favourable_r = (entry - low[j]) / risk
            if favourable_r > max_favourable_r:
                max_favourable_r = favourable_r
            if breakeven_trigger > 0.0 and max_favourable_r >= breakeven_trigger:
                protected = entry + direction * risk * breakeven_lock
                if direction > 0 and protected > stop:
                    stop = protected
                elif direction < 0 and protected < stop:
                    stop = protected
            bars_held = j - entry_i + 1
            if early_bars > 0 and bars_held >= early_bars and max_favourable_r < 0.55:
                current_r = ((close[j] - entry) * direction) / risk
                if current_r <= -early_cut_r:
                    exit_price, reason, exit_i = close[j] - direction * slippage_abs, 4, j
                    break
        gross = (exit_price - entry) * direction
        fees = fee_rate * (entry + exit_price)
        value = (gross - fees) / risk
        labels[k] = 1 if value > 0.0 else 0
        exits[k], net_r[k], reasons[k] = exit_i, value, reason
    return labels, exits, net_r, reasons


def mild_class_weights(y: np.ndarray) -> np.ndarray:
    positives = max(1, int(np.sum(y == 1)))
    negatives = max(1, int(np.sum(y == 0)))
    total = positives + negatives
    return np.where(y == 1, math.sqrt(total / (2.0 * positives)), math.sqrt(total / (2.0 * negatives)))


def make_expert_data(x: pd.DataFrame) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for expert in EXPERTS:
        idx = np.flatnonzero(x[str(expert["column"])].to_numpy(bool)).astype(np.int64)
        result[int(expert["id"])] = {
            "idx": idx,
            "direction": int(expert["direction"]),
            "name": str(expert["name"]),
            "key": str(expert["key"]),
        }
    return result


def fit_one_expert_model(
    x: pd.DataFrame,
    expert_data: dict[int, dict[str, Any]],
    policy: ExpertPolicy,
    train_months: set[str],
) -> HistGradientBoostingClassifier | None:
    assert_oos_isolation(train_months, f"model training expert={policy.expert_id}")
    data = expert_data[policy.expert_id]
    idx = data["idx"]
    open_, high, low, close, atr = (x[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr"))
    labels, _, _, _ = compute_outcomes(
        idx, int(data["direction"]), open_, high, low, close, atr,
        policy.risk.rr, policy.risk.sl_atr, policy.risk.min_stop_pct, policy.risk.max_hold,
        policy.risk.breakeven_trigger, policy.risk.breakeven_lock,
        policy.risk.early_bars, policy.risk.early_cut_r,
        FEE_RATE, SLIPPAGE_ABS,
    )
    months = x["month"].to_numpy()[idx]
    mask = np.isin(months, list(train_months)) & (labels >= 0)
    y = labels[mask].astype(int)
    if len(y) < 60 or len(np.unique(y)) < 2 or min(np.sum(y == 0), np.sum(y == 1)) < 10:
        return None
    ordered = sorted(train_months)
    recency = {m: 0.86 + 0.28 * (i / max(1, len(ordered) - 1)) for i, m in enumerate(ordered)}
    weights = np.array([recency.get(str(m), 1.0) for m in months[mask]], dtype=float)
    model = HistGradientBoostingClassifier(
        max_depth=policy.model.max_depth,
        learning_rate=policy.model.learning_rate,
        max_iter=policy.model.max_iter,
        l2_regularization=policy.model.l2_regularization,
        min_samples_leaf=policy.model.min_samples_leaf,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=18,
        random_state=BASE_SEED + policy.expert_id * 101,
    )
    model.fit(x.iloc[idx[mask]][FEATURES].to_numpy(np.float64), y, sample_weight=mild_class_weights(y) * weights)
    return model


def build_one_expert_bundle(
    x: pd.DataFrame,
    expert_data: dict[int, dict[str, Any]],
    policy: ExpertPolicy,
    model: HistGradientBoostingClassifier,
) -> dict[str, Any]:
    data = expert_data[policy.expert_id]
    idx = data["idx"]
    open_, high, low, close, atr = (x[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr"))
    labels, exits, net_r, reasons = compute_outcomes(
        idx, int(data["direction"]), open_, high, low, close, atr,
        policy.risk.rr, policy.risk.sl_atr, policy.risk.min_stop_pct, policy.risk.max_hold,
        policy.risk.breakeven_trigger, policy.risk.breakeven_lock,
        policy.risk.early_bars, policy.risk.early_cut_r,
        FEE_RATE, SLIPPAGE_ABS,
    )
    prob = model.predict_proba(x.iloc[idx][FEATURES].to_numpy(np.float64))[:, 1]
    return {
        "idx": idx,
        "direction": int(data["direction"]),
        "expert_id": policy.expert_id,
        "expert": str(data["name"]),
        "labels": labels,
        "exits": exits,
        "net_r": net_r,
        "reasons": reasons,
        "prob": prob,
        "months": x["month"].to_numpy()[idx],
    }


def events_from_expert(
    bundle: dict[str, Any],
    policy: ExpertPolicy,
    eval_month: str,
    train_months: set[str],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    assert_oos_isolation(train_months, f"threshold calibration expert={policy.expert_id} eval={eval_month}")
    train_mask = np.isin(bundle["months"], list(train_months)) & (bundle["exits"] >= 0)
    train_prob = bundle["prob"][train_mask]
    if len(train_prob) < 45:
        return [], {"probability_threshold": 1.0, "quantile": 1.0, "training_candidates": int(len(train_prob))}
    desired_total = max(1.0, policy.monthly_target * max(1, len(train_months)))
    keep_fraction = min(0.30, max(0.003, desired_total / len(train_prob)))
    quantile = max(policy.min_percentile, 1.0 - keep_fraction)
    probability_threshold = max(policy.min_probability, float(np.quantile(train_prob, quantile)))
    sorted_train = np.sort(train_prob)
    eval_mask = (
        (bundle["months"] == eval_month)
        & (bundle["exits"] >= 0)
        & (bundle["prob"] >= probability_threshold)
    )
    events: list[dict[str, Any]] = []
    for k in np.flatnonzero(eval_mask):
        score = float(np.searchsorted(sorted_train, bundle["prob"][k], side="right") / len(sorted_train))
        events.append({
            "signal_i": int(bundle["idx"][k]),
            "exit_i": int(bundle["exits"][k]),
            "direction": int(bundle["direction"]),
            "expert_id": int(bundle["expert_id"]),
            "expert": str(bundle["expert"]),
            "prob": float(bundle["prob"][k]),
            "score": score,
            "net_r": float(bundle["net_r"][k]),
            "reason": int(bundle["reasons"][k]),
            "policy_key": policy.key,
        })
    # Monthly target is expert-specific. Keep the strongest candidate signals,
    # then restore time order before cooldown/position selection.
    if len(events) > policy.monthly_target:
        events = sorted(events, key=lambda e: (e["score"], e["prob"]), reverse=True)[:policy.monthly_target]
    events = sorted(events, key=lambda e: (e["signal_i"], -e["score"], -e["prob"]))
    return events, {
        "probability_threshold": float(probability_threshold),
        "quantile": float(quantile),
        "training_candidates": int(len(train_prob)),
        "raw_eval_candidates": int(np.sum(eval_mask)),
        "capped_eval_candidates": int(len(events)),
    }


def apply_expert_cooldown(events: list[dict[str, Any]], cooldown: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    last_exit = -10**9
    for event in sorted(events, key=lambda e: (e["signal_i"], -e["score"], -e["prob"])):
        if event["signal_i"] <= last_exit + cooldown:
            continue
        selected.append(event)
        last_exit = int(event["exit_i"])
    return selected


def combine_events(
    events_by_expert: dict[int, list[dict[str, Any]]],
    policies: dict[int, ExpertPolicy],
    min_direction_margin: float = 0.025,
) -> list[dict[str, Any]]:
    prefiltered: list[dict[str, Any]] = []
    for expert_id, events in events_by_expert.items():
        prefiltered.extend(apply_expert_cooldown(events, policies[expert_id].cooldown))
    prefiltered.sort(key=lambda e: (e["signal_i"], -e["score"], -e["prob"]))
    selected: list[dict[str, Any]] = []
    global_last_exit = -10**9
    i = 0
    while i < len(prefiltered):
        signal_i = int(prefiltered[i]["signal_i"])
        same: list[dict[str, Any]] = []
        while i < len(prefiltered) and int(prefiltered[i]["signal_i"]) == signal_i:
            same.append(prefiltered[i])
            i += 1
        if signal_i <= global_last_exit:
            continue
        best_long = max((e for e in same if e["direction"] > 0), key=lambda e: (e["score"], e["prob"]), default=None)
        best_short = max((e for e in same if e["direction"] < 0), key=lambda e: (e["score"], e["prob"]), default=None)
        if best_long is not None and best_short is not None:
            long_strength = 0.65 * best_long["score"] + 0.35 * best_long["prob"]
            short_strength = 0.65 * best_short["score"] + 0.35 * best_short["prob"]
            if abs(long_strength - short_strength) < min_direction_margin:
                continue
            best = best_long if long_strength > short_strength else best_short
        else:
            best = best_long if best_long is not None else best_short
        if best is None:
            continue
        selected.append(best)
        global_last_exit = int(best["exit_i"])
    return selected


def metrics(trades: list[dict[str, Any]]) -> dict[str, float]:
    values = np.array([t["net_r"] for t in trades], dtype=float)
    count = int(len(values))
    wins = values[values > 0]
    losses = -values[values <= 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    cum = np.cumsum(values) if count else np.array([0.0])
    peak = np.maximum.accumulate(np.r_[0.0, cum])
    dd = peak[1:] - cum if count else np.array([0.0])
    return {
        "trades": count,
        "wins": int(len(wins)),
        "win_rate": float(len(wins) / count) if count else 0.0,
        "avg_win_R": avg_win,
        "avg_loss_R": avg_loss,
        "avg_win_loss_ratio": avg_win / avg_loss if avg_win > 0 and avg_loss > 0 else 0.0,
        "profit_factor": float(wins.sum() / losses.sum()) if len(losses) and losses.sum() > 0 else 0.0,
        "net_R": float(values.sum()) if count else 0.0,
        "max_drawdown_R": float(dd.max()) if len(dd) else 0.0,
        "expectancy_R": float(values.mean()) if count else 0.0,
    }


def expert_breakdown(trades: list[dict[str, Any]], include_empty: bool = True) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for expert_id, name in EXPERT_NAME.items():
        subset = [t for t in trades if int(t.get("expert_id", -1)) == expert_id]
        if subset or include_empty:
            output[name] = metrics(subset)
    return output


def wilson_lower_bound(wins: int, count: int, z: float = 1.0) -> float:
    if count <= 0:
        return 0.0
    p = wins / count
    denom = 1.0 + z * z / count
    centre = p + z * z / (2.0 * count)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * count)) / count)
    return (centre - margin) / denom


def active_month_values(monthly: dict[str, dict[str, float]], key: str) -> list[float]:
    return [float(v[key]) for v in monthly.values() if int(v["trades"]) > 0]


def consecutive_zero_months(monthly: dict[str, dict[str, float]]) -> int:
    maximum = current = 0
    for month in DEVELOPMENT_MONTHS:
        if int(monthly.get(month, {}).get("trades", 0)) == 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def evaluate_elimination(candidate: ExpertCandidate) -> tuple[bool, list[str]]:
    monthly = candidate.monthly_metrics
    reasons: list[str] = []
    active = [m for m in DEVELOPMENT_MONTHS if int(monthly.get(m, {}).get("trades", 0)) > 0]
    total_trades = sum(int(monthly.get(m, {}).get("trades", 0)) for m in DEVELOPMENT_MONTHS)
    total_wins = sum(int(monthly.get(m, {}).get("wins", 0)) for m in DEVELOPMENT_MONTHS)
    if consecutive_zero_months(monthly) >= int(ELIMINATION["max_consecutive_zero_trade_months"]):
        reasons.append("连续两个开发月无交易")
    meaningful_wr = [
        float(monthly[m]["win_rate"])
        for m in active
        if int(monthly[m]["trades"]) >= 3
    ]
    if meaningful_wr and min(meaningful_wr) < float(ELIMINATION["min_floor_win_rate"]):
        reasons.append("开发月最低胜率过低")
    cumulative_wr = total_wins / total_trades if total_trades else 0.0
    if total_trades >= int(ELIMINATION["min_total_trades"]) and cumulative_wr < float(ELIMINATION["min_cumulative_win_rate"]):
        reasons.append("累计胜率过低")
    negative_months = sum(float(monthly.get(m, {}).get("expectancy_R", 0.0)) < 0 for m in active)
    if negative_months >= int(ELIMINATION["max_negative_expectancy_months"]):
        reasons.append("多月负期望")
    if len(active) < int(ELIMINATION["min_active_months"]):
        reasons.append("有效开发月份不足")
    if total_trades < int(ELIMINATION["min_total_trades"]):
        reasons.append("开发交易样本不足")
    return len(reasons) == 0, reasons


def expert_candidate_score(candidate: ExpertCandidate) -> float:
    monthly = candidate.monthly_metrics
    active = [monthly[m] for m in DEVELOPMENT_MONTHS if int(monthly.get(m, {}).get("trades", 0)) > 0]
    if not active:
        return -1e12
    floor_wr = min(float(m["win_rate"]) for m in active)
    floor_pf = min(float(m["profit_factor"]) for m in active)
    cumulative_values: list[float] = []
    for month in DEVELOPMENT_MONTHS:
        cumulative_values.extend([float(t["net_r"]) for t in candidate.monthly_events.get(month, [])])
    cumulative = metrics([{"net_r": v} for v in cumulative_values]) if cumulative_values else metrics([])
    trade_counts = [int(monthly[m]["trades"]) for m in DEVELOPMENT_MONTHS]
    target_gap = float(np.mean([abs(c - candidate.policy.monthly_target) for c in trade_counts]))
    return float(
        floor_wr * 3600
        + min(floor_pf, 5.0) * 900
        + cumulative["win_rate"] * 1500
        + min(cumulative["profit_factor"], 6.0) * 420
        + cumulative["net_R"] * 24
        - cumulative["max_drawdown_R"] * 22
        - target_gap * 28
        - consecutive_zero_months(monthly) * 800
    )


def combination_score(candidate: CombinationCandidate) -> float:
    monthly = candidate.monthly_metrics
    stats = [monthly[m] for m in DEVELOPMENT_MONTHS]
    floor_wr = min(float(m["win_rate"]) for m in stats)
    floor_pf = min(float(m["profit_factor"]) for m in stats)
    floor_ratio = min(float(m["avg_win_loss_ratio"]) for m in stats)
    median_wr = float(np.median([m["win_rate"] for m in stats]))
    median_pf = float(np.median([m["profit_factor"] for m in stats]))
    may = monthly[MONTHS[8]]
    count_penalty = sum(
        abs(int(m["trades"]) - np.clip(int(m["trades"]), max(10, MIN_TRADES - 5), MAX_TRADES + 5))
        for m in stats
    )
    hard_may_penalty = 0.0
    if not (MIN_TRADES <= may["trades"] <= MAX_TRADES):
        hard_may_penalty += 7000 + abs(may["trades"] - np.clip(may["trades"], MIN_TRADES, MAX_TRADES)) * 500
    if may["win_rate"] < MIN_WIN_RATE:
        hard_may_penalty += (MIN_WIN_RATE - may["win_rate"]) * 16000
    if may["avg_win_loss_ratio"] < MIN_RATIO:
        hard_may_penalty += (MIN_RATIO - may["avg_win_loss_ratio"]) * 7000
    # Explicitly prioritize the weakest development month's win rate and PF.
    return float(
        floor_wr * 10500
        + min(floor_pf, 5.0) * 2600
        + min(floor_ratio, 4.0) * 900
        + median_wr * 2200
        + min(median_pf, 6.0) * 700
        + sum(float(m["net_R"]) for m in stats) * 20
        - sum(float(m["max_drawdown_R"]) for m in stats) * 14
        - count_penalty * 220
        - hard_may_penalty
    )


def risk_grid_for_expert(expert_id: int) -> list[RiskConfig]:
    if expert_id == 1:  # Keep and optimize trend-short separately.
        return [
            RiskConfig(2.0, 1.25, 0.0034, 216, 1.00, 0.12, 60, 0.38),
            RiskConfig(2.2, 1.40, 0.0036, 288, 1.10, 0.15, 72, 0.35),
            RiskConfig(2.4, 1.55, 0.0038, 288, 1.20, 0.18, 72, 0.32),
        ]
    if expert_id == 0:
        return [
            RiskConfig(1.9, 1.20, 0.0032, 216, 0.95, 0.10, 54, 0.38),
            RiskConfig(2.2, 1.35, 0.0035, 288, 1.05, 0.14, 66, 0.35),
        ]
    if expert_id in (2, 3):
        return [
            RiskConfig(1.8, 1.05, 0.0030, 144, 0.85, 0.08, 42, 0.32),
            RiskConfig(2.0, 1.20, 0.0033, 180, 0.95, 0.10, 48, 0.30),
        ]
    return [
        RiskConfig(2.2, 1.45, 0.0040, 216, 1.10, 0.12, 48, 0.35),
        RiskConfig(2.5, 1.65, 0.0045, 288, 1.25, 0.18, 60, 0.32),
    ]


def model_grid_for_expert(expert_id: int) -> list[ModelConfig]:
    if expert_id in (2, 3):
        return [ModelConfig(2, 0.035, 220, 6.0, 42)]
    if expert_id in (4, 5):
        return [ModelConfig(2, 0.030, 240, 7.0, 38)]
    return [
        ModelConfig(2, 0.035, 220, 5.0, 45),
        ModelConfig(3, 0.025, 270, 8.0, 60),
    ]


def policy_grid_for_expert(expert_id: int) -> list[ExpertPolicy]:
    # All six experts own their probability, percentile, target, risk and cooldown.
    if expert_id == 1:
        targets, probs, pcts, cooldowns = (5, 7, 9), (0.51, 0.55, 0.59), (0.82, 0.87, 0.91), (2, 5, 9)
    elif expert_id == 0:
        targets, probs, pcts, cooldowns = (4, 6, 8), (0.50, 0.54, 0.58), (0.80, 0.86, 0.90), (2, 5)
    elif expert_id in (2, 3):
        targets, probs, pcts, cooldowns = (3, 5, 7), (0.50, 0.54, 0.58), (0.80, 0.86, 0.91), (4, 8)
    else:
        targets, probs, pcts, cooldowns = (3, 5, 7), (0.51, 0.56, 0.61), (0.82, 0.88, 0.93), (5, 10)
    return [
        ExpertPolicy(expert_id, risk, model, target, prob, pct, cooldown)
        for risk, model, target, prob, pct, cooldown in itertools.product(
            risk_grid_for_expert(expert_id),
            model_grid_for_expert(expert_id),
            targets,
            probs,
            pcts,
            cooldowns,
        )
    ]


def evaluate_expert_candidates(
    x: pd.DataFrame,
    expert_data: dict[int, dict[str, Any]],
    expert_id: int,
) -> list[ExpertCandidate]:
    candidates = [ExpertCandidate(policy=p) for p in policy_grid_for_expert(expert_id)]
    plan = walk_forward_plan()
    groups: dict[tuple[RiskConfig, ModelConfig], list[ExpertCandidate]] = {}
    for candidate in candidates:
        groups.setdefault((candidate.policy.risk, candidate.policy.model), []).append(candidate)

    for eval_month, train_months in plan:
        for (_, _), group in groups.items():
            representative = group[0].policy
            model = fit_one_expert_model(x, expert_data, representative, train_months)
            if model is None:
                for candidate in group:
                    candidate.monthly_events[eval_month] = []
                    candidate.monthly_metrics[eval_month] = metrics([])
                continue
            bundle = build_one_expert_bundle(x, expert_data, representative, model)
            event_cache: dict[tuple[int, float, float], list[dict[str, Any]]] = {}
            for candidate in group:
                key = (
                    candidate.policy.monthly_target,
                    candidate.policy.min_probability,
                    candidate.policy.min_percentile,
                )
                if key not in event_cache:
                    event_cache[key], _ = events_from_expert(bundle, candidate.policy, eval_month, train_months)
                chosen = apply_expert_cooldown(event_cache[key], candidate.policy.cooldown)
                candidate.monthly_events[eval_month] = chosen
                candidate.monthly_metrics[eval_month] = metrics(chosen)

    for candidate in candidates:
        candidate.eligible, candidate.elimination_reasons = evaluate_elimination(candidate)
        candidate.aggregate_score = expert_candidate_score(candidate)
    candidates.sort(key=lambda c: c.aggregate_score, reverse=True)
    return candidates


def candidate_options_by_expert(all_candidates: dict[int, list[ExpertCandidate]]) -> dict[int, list[ExpertCandidate]]:
    options: dict[int, list[ExpertCandidate]] = {}
    for expert_id, candidates in all_candidates.items():
        eligible = [c for c in candidates if c.eligible]
        options[expert_id] = eligible[:TOP_PER_EXPERT]
    return options


def build_combinations(
    options: dict[int, list[ExpertCandidate]],
) -> list[CombinationCandidate]:
    active_ids = [expert_id for expert_id in range(6) if options.get(expert_id)]
    if not active_ids:
        return [CombinationCandidate(policies={})]
    # Beam construction avoids a blind full Cartesian explosion while allowing
    # each surviving expert to be disabled or use one of its own top policies.
    beam: list[dict[int, ExpertPolicy]] = [{}]
    for expert_id in active_ids:
        expanded: list[dict[int, ExpertPolicy]] = []
        for current in beam:
            expanded.append(dict(current))
            for candidate in options[expert_id]:
                item = dict(current)
                item[expert_id] = candidate.policy
                expanded.append(item)
        expanded.sort(key=lambda policies: sum(
            next(c.aggregate_score for c in options[eid] if c.policy.key == policy.key)
            for eid, policy in policies.items()
        ), reverse=True)
        beam = expanded[:BEAM_WIDTH]
    return [CombinationCandidate(policies=p) for p in beam if p]


def evaluate_development_combinations(
    combinations: list[CombinationCandidate],
    all_candidates: dict[int, list[ExpertCandidate]],
) -> list[CombinationCandidate]:
    lookup = {
        (candidate.policy.expert_id, candidate.policy.key): candidate
        for candidates in all_candidates.values()
        for candidate in candidates
    }
    for combo in combinations:
        for month in DEVELOPMENT_MONTHS:
            events_by_expert = {
                expert_id: lookup[(expert_id, policy.key)].monthly_events.get(month, [])
                for expert_id, policy in combo.policies.items()
            }
            selected = combine_events(events_by_expert, combo.policies)
            combo.monthly_trades[month] = selected
            combo.monthly_metrics[month] = metrics(selected)
            combo.monthly_expert_metrics[month] = expert_breakdown(selected)
        combo.aggregate_score = combination_score(combo)
    combinations.sort(key=lambda c: c.aggregate_score, reverse=True)
    return combinations


def evaluate_selected_month(
    x: pd.DataFrame,
    expert_data: dict[int, dict[str, Any]],
    selected: CombinationCandidate,
    eval_month: str,
    train_months: set[str],
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, dict[str, float]], dict[str, Any]]:
    if eval_month == OOS_MONTH:
        # This is the only permitted June entry point, after all policy selection.
        if selected.aggregate_score <= -1e11:
            raise RuntimeError("Cannot evaluate June before selecting a development policy")
    assert_oos_isolation(train_months, f"final evaluation training for {eval_month}")
    events_by_expert: dict[int, list[dict[str, Any]]] = {}
    thresholds: dict[str, Any] = {}
    for expert_id, policy in selected.policies.items():
        model = fit_one_expert_model(x, expert_data, policy, train_months)
        if model is None:
            events_by_expert[expert_id] = []
            thresholds[EXPERT_NAME[expert_id]] = {"model_available": False}
            continue
        bundle = build_one_expert_bundle(x, expert_data, policy, model)
        events, threshold = events_from_expert(bundle, policy, eval_month, train_months)
        events_by_expert[expert_id] = events
        thresholds[EXPERT_NAME[expert_id]] = {"model_available": True, **threshold}
    selected_trades = combine_events(events_by_expert, selected.policies)
    return metrics(selected_trades), selected_trades, expert_breakdown(selected_trades), thresholds


def setup_name(x: pd.DataFrame, signal_i: int, direction: int) -> str:
    suffix = "long" if direction > 0 else "short"
    mapping = [
        (f"setup_reclaim_{suffix}", "趋势回踩再确认"),
        (f"setup_continuation_{suffix}", "趋势延续突破"),
        (f"setup_range_reversal_{suffix}", "流动性扫单反转"),
        (f"setup_high_break_{suffix}", "压缩后高波动突破"),
    ]
    active = [label for col, label in mapping if col in x.columns and float(x.iloc[signal_i][col]) > 0.5]
    return "+".join(active) if active else "专家候选"


def detailed_trades(
    x: pd.DataFrame,
    trades: list[dict[str, Any]],
    policies: dict[int, ExpertPolicy],
    month: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        signal_i = int(trade["signal_i"])
        entry_i = signal_i + 1
        expert_id = int(trade["expert_id"])
        policy = policies[expert_id]
        direction = int(trade["direction"])
        entry = float(x.iloc[entry_i]["open"] + direction * SLIPPAGE_ABS)
        risk_abs = max(
            float(x.iloc[signal_i]["atr"]) * policy.risk.sl_atr,
            float(x.iloc[signal_i]["close"]) * policy.risk.min_stop_pct,
        )
        exit_i = int(trade["exit_i"])
        reason = {1: "TP", 2: "PROTECTED_STOP", 3: "TIME", 4: "EARLY_CUT"}.get(int(trade["reason"]), "UNKNOWN")
        rows.append({
            "signal_time_utc": x.index[signal_i].isoformat(),
            "entry_time_utc": x.index[entry_i].isoformat(),
            "exit_time_utc": x.index[exit_i].isoformat(),
            "month": month,
            "direction": "LONG" if direction > 0 else "SHORT",
            "expert": EXPERT_NAME[expert_id],
            "setup": setup_name(x, signal_i, direction),
            "policy_key": policy.key,
            "probability": float(trade["prob"]),
            "training_percentile": float(trade["score"]),
            "entry": entry,
            "risk_abs": risk_abs,
            "target_R": policy.risk.rr,
            "net_R": float(trade["net_r"]),
            "win": bool(trade["net_r"] > 0),
            "exit_reason": reason,
            "bars": exit_i - entry_i,
        })
    return pd.DataFrame(rows)


def selected_policy_payload(selected: CombinationCandidate) -> dict[str, Any]:
    return {
        EXPERT_NAME[expert_id]: {
            "expert_id": expert_id,
            "policy_key": policy.key,
            "minimum_profit_probability": policy.min_probability,
            "minimum_training_percentile": policy.min_percentile,
            "monthly_candidate_target": policy.monthly_target,
            "cooldown_bars": policy.cooldown,
            "risk": asdict(policy.risk),
            "model": asdict(policy.model),
        }
        for expert_id, policy in sorted(selected.policies.items())
    }


def write_candidate_leaderboards(
    all_candidates: dict[int, list[ExpertCandidate]],
    combinations: list[CombinationCandidate],
) -> None:
    expert_rows: list[dict[str, Any]] = []
    for expert_id, candidates in all_candidates.items():
        for rank, candidate in enumerate(candidates[:30], start=1):
            row: dict[str, Any] = {
                "expert_id": expert_id,
                "expert": EXPERT_NAME[expert_id],
                "rank": rank,
                "policy_key": candidate.policy.key,
                "aggregate_score": candidate.aggregate_score,
                "eligible": candidate.eligible,
                "elimination_reasons": "|".join(candidate.elimination_reasons),
                "monthly_target": candidate.policy.monthly_target,
                "min_probability": candidate.policy.min_probability,
                "min_percentile": candidate.policy.min_percentile,
                "cooldown": candidate.policy.cooldown,
                **{f"risk_{k}": v for k, v in asdict(candidate.policy.risk).items()},
                **{f"model_{k}": v for k, v in asdict(candidate.policy.model).items()},
            }
            for month, value in candidate.monthly_metrics.items():
                for key, val in value.items():
                    row[f"{month}_{key}"] = val
            expert_rows.append(row)
    pd.DataFrame(expert_rows).to_csv(RESULTS / "expert_candidate_leaderboard.csv", index=False)

    combo_rows: list[dict[str, Any]] = []
    for rank, combo in enumerate(combinations[:40], start=1):
        row = {
            "rank": rank,
            "combination_key": combo.key,
            "aggregate_score": combo.aggregate_score,
            "active_experts": "|".join(EXPERT_NAME[i] for i in sorted(combo.policies)),
            "expert_policy_keys": "|".join(f"{EXPERT_NAME[i]}={p.key}" for i, p in sorted(combo.policies.items())),
        }
        for month, value in combo.monthly_metrics.items():
            for key, val in value.items():
                row[f"{month}_{key}"] = val
        combo_rows.append(row)
    pd.DataFrame(combo_rows).to_csv(RESULTS / "candidate_leaderboard.csv", index=False)


def synthetic_inputs(rows: int = 6500) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BASE_SEED)
    times = np.arange(1_700_000_000_000, 1_700_000_000_000 + rows * 300_000, 300_000, dtype=np.int64)
    regime = np.sin(np.arange(rows) / 420.0) * 0.00022
    returns = rng.normal(0.0, 0.00125, rows) + regime
    close = 50_000.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(15.0, close * rng.uniform(0.0004, 0.0024, rows))
    high = np.maximum(open_, close) + spread * rng.uniform(0.2, 1.0, rows)
    low = np.minimum(open_, close) - spread * rng.uniform(0.2, 1.0, rows)
    volume = rng.lognormal(5.0, 0.65, rows)
    taker = volume * np.clip(0.5 + rng.normal(0, 0.09, rows), 0.05, 0.95)
    base = pd.DataFrame({
        "open_time": times, "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "close_time": times + 299_999, "quote_volume": volume * close,
        "trade_count": rng.integers(200, 2200, rows), "taker_buy_volume": taker,
        "taker_buy_quote": taker * close, "ignore": 0,
    })
    eth_close = 3000.0 * np.exp(np.cumsum(returns * 1.12 + rng.normal(0, 0.00055, rows)))
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
    funding = pd.DataFrame({
        "calc_time": times[fund_idx],
        "funding_rate": rng.normal(0.00005, 0.00008, len(fund_idx)),
    })
    return base, eth, premium, funding


def threshold_independence_test() -> None:
    probabilities = np.linspace(0.40, 0.90, 120)
    months = np.array(["2025-09"] * 80 + ["2025-12"] * 40)
    bundle = {
        "idx": np.arange(120),
        "direction": -1,
        "expert_id": 1,
        "expert": "趋势空头",
        "labels": np.ones(120, dtype=np.int8),
        "exits": np.arange(120) + 1,
        "net_r": np.ones(120),
        "reasons": np.ones(120, dtype=np.int8),
        "prob": probabilities,
        "months": months,
    }
    risk = RiskConfig(2.0, 1.2, 0.0035, 144, 1.0, 0.1, 48, 0.35)
    model = ModelConfig(2, 0.03, 100, 3.0, 20)
    loose = ExpertPolicy(1, risk, model, 12, 0.50, 0.70, 2)
    strict = ExpertPolicy(1, risk, model, 3, 0.75, 0.95, 9)
    loose_events, loose_meta = events_from_expert(bundle, loose, "2025-12", {"2025-09"})
    strict_events, strict_meta = events_from_expert(bundle, strict, "2025-12", {"2025-09"})
    if loose.key == strict.key:
        raise RuntimeError("Expert policy keys are not independent")
    if loose_meta["probability_threshold"] >= strict_meta["probability_threshold"]:
        raise RuntimeError("Independent probability/percentile thresholds were not honored")
    if len(loose_events) <= len(strict_events):
        raise RuntimeError("Independent monthly candidate targets were not honored")


def self_test() -> None:
    plan = walk_forward_plan()
    if any(OOS_MONTH in train for _, train in plan):
        raise RuntimeError("June leakage in walk-forward plan")
    try:
        assert_oos_isolation({"2026-05", OOS_MONTH}, "negative self-test")
    except RuntimeError:
        pass
    else:
        raise RuntimeError("OOS leakage guard failed")

    base, eth, premium, funding = synthetic_inputs()
    x, alignment_audit = add_features(base, eth, premium, funding)
    if x.empty:
        raise RuntimeError("V9 self-test produced no feature rows")
    missing = [feature for feature in FEATURES if feature not in x.columns]
    if missing:
        raise RuntimeError(f"V9 self-test missing features: {missing}")

    # Missing auxiliary-data regression: sparse gaps and microsecond timestamps
    # must be handled with past-only filling and no backward-looking fill.
    eth_gap = eth.drop(index=[250, 251]).reset_index(drop=True)
    premium_gap = premium.drop(index=[100, 101, 102, 900]).reset_index(drop=True)
    premium_gap["open_time"] = premium_gap["open_time"].astype("int64") * 1000
    x_gap, gap_audit = add_features(base, eth_gap, premium_gap, funding)
    if x_gap.empty:
        raise RuntimeError("V9 missing-auxiliary regression produced no rows")
    if gap_audit["eth_exact_missing"] < 2 or gap_audit["premium_exact_missing"] < 4:
        raise RuntimeError("Missing-auxiliary regression did not exercise missing timestamps")
    if gap_audit["eth_missing_after_past_fill"] != 0 or gap_audit["premium_missing_after_past_fill"] != 0:
        raise RuntimeError("Past-only auxiliary alignment regression failed")

    experts = make_expert_data(x)
    if set(experts) != set(range(6)):
        raise RuntimeError("V9 expert registry mismatch")
    indices = np.arange(100, min(140, len(x) - 2), dtype=np.int64)
    compute_outcomes(
        indices, 1,
        x["open"].to_numpy(float), x["high"].to_numpy(float), x["low"].to_numpy(float),
        x["close"].to_numpy(float), x["atr"].to_numpy(float),
        2.0, 1.2, 0.004, 144, 1.0, 0.1, 48, 0.4, FEE_RATE, SLIPPAGE_ABS,
    )
    threshold_independence_test()
    if alignment_audit["eth_coverage_after_past_fill"] < 0.99:
        raise RuntimeError("Synthetic alignment unexpectedly failed")
    print(
        f"V9.0 self-test passed: rows={len(x)}, features={len(FEATURES)}, "
        f"experts={len(experts)}, oos={OOS_MONTH} isolated"
    )


def clear_results() -> None:
    if RESULTS.exists():
        for item in RESULTS.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    RESULTS.mkdir(exist_ok=True)


def main() -> None:
    clear_results()
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

    all_candidates: dict[int, list[ExpertCandidate]] = {}
    for expert_id in range(6):
        print(f"SEARCH_EXPERT={expert_id}:{EXPERT_NAME[expert_id]}")
        all_candidates[expert_id] = evaluate_expert_candidates(x, expert_data, expert_id)

    options = candidate_options_by_expert(all_candidates)
    combinations = evaluate_development_combinations(build_combinations(options), all_candidates)
    selected = combinations[0]

    # May is recomputed from its proper training set for final output. It remains
    # part of development selection; June is evaluated only after `selected` exists.
    may_metrics, may_trades, may_experts, may_thresholds = evaluate_selected_month(
        x, expert_data, selected, MONTHS[8], set(MONTHS[:8])
    )
    june_metrics, june_trades, june_experts, june_thresholds = evaluate_selected_month(
        x, expert_data, selected, OOS_MONTH, set(MONTHS[:9])
    )

    qualified = bool(
        MIN_TRADES <= may_metrics["trades"] <= MAX_TRADES
        and MIN_TRADES <= june_metrics["trades"] <= MAX_TRADES
        and may_metrics["win_rate"] >= MIN_WIN_RATE
        and june_metrics["win_rate"] >= MIN_WIN_RATE
        and may_metrics["avg_win_loss_ratio"] >= MIN_RATIO
        and june_metrics["avg_win_loss_ratio"] >= MIN_RATIO
    )

    frames = [
        detailed_trades(x, may_trades, selected.policies, MONTHS[8]),
        detailed_trades(x, june_trades, selected.policies, OOS_MONTH),
    ]
    nonempty = [frame for frame in frames if not frame.empty]
    all_trades = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=[
        "signal_time_utc", "entry_time_utc", "exit_time_utc", "month", "direction",
        "expert", "setup", "policy_key", "probability", "training_percentile",
        "entry", "risk_abs", "target_R", "net_R", "win", "exit_reason", "bars",
    ])
    all_trades.to_csv(RESULTS / "trades.csv", index=False)

    elimination_status = {
        EXPERT_NAME[expert_id]: {
            "eligible_candidates": sum(c.eligible for c in candidates),
            "searched_candidates": len(candidates),
            "best_candidate_eligible": bool(candidates[0].eligible),
            "best_candidate_reasons": candidates[0].elimination_reasons,
            "selected": expert_id in selected.policies,
        }
        for expert_id, candidates in all_candidates.items()
    }
    status = {
        "qualified": qualified,
        "engine": ENGINE_NAME,
        "method": "Sep-Nov initial train; Dec-May rolling development and policy selection; June evaluated once after freeze",
        "architecture": "completed 1h regime → completed 15m structure → 5m close signal → next 5m open fill; six expert-specific policies",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "months": list(MONTHS),
        "oos_month": OOS_MONTH,
        "oos_isolation": {
            "used_for_training": False,
            "used_for_thresholds": False,
            "used_for_expert_elimination": False,
            "used_for_combination_selection": False,
            "evaluation_occurs_after_policy_freeze": True,
        },
        "data_context": [
            "BTCUSDT perpetual klines",
            "ETHUSDT perpetual klines",
            "BTC premium-index klines",
            "BTC funding rate",
        ],
        "constraints": {
            "min_trades": MIN_TRADES,
            "max_trades": MAX_TRADES,
            "min_win_rate": MIN_WIN_RATE,
            "min_avg_win_loss_ratio": MIN_RATIO,
            "fee_rate_per_side": FEE_RATE,
            "slippage_per_fill_usdt": SLIPPAGE_ABS,
        },
        "selected_combination_key": selected.key,
        "selected_expert_policies": selected_policy_payload(selected),
        "expert_elimination": elimination_status,
        "development_monthly_stats": selected.monthly_metrics,
        "development_expert_stats": selected.monthly_expert_metrics,
        "monthly_stats": {MONTHS[8]: may_metrics, OOS_MONTH: june_metrics},
        "expert_stats": {MONTHS[8]: may_experts, OOS_MONTH: june_experts},
        "threshold_audit": {MONTHS[8]: may_thresholds, OOS_MONTH: june_thresholds},
        "may_hard_qualified_combinations": sum(
            MIN_TRADES <= combo.monthly_metrics[MONTHS[8]]["trades"] <= MAX_TRADES
            and combo.monthly_metrics[MONTHS[8]]["win_rate"] >= MIN_WIN_RATE
            and combo.monthly_metrics[MONTHS[8]]["avg_win_loss_ratio"] >= MIN_RATIO
            for combo in combinations
        ),
        "searched_expert_policies": sum(len(v) for v in all_candidates.values()),
        "searched_combinations": len(combinations),
    }
    (RESULTS / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "selected_policy.json").write_text(
        json.dumps(status["selected_expert_policies"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_candidate_leaderboards(all_candidates, combinations)

    selected_names = list(status["selected_expert_policies"].keys())
    report = f"""# BTCUSDT 5分钟 专家独立参数 V9.0 回测报告

- 数据：Binance USDⓈ-M 永续官方5分钟K线；加入 ETHUSDT、BTC 溢价指数与资金费率。
- 成本：单边手续费 {FEE_RATE*100:.3f}%；每次成交滑点 {SLIPPAGE_ABS:.1f} USDT。
- 架构：已完成的1小时市场状态 → 已完成的15分钟结构 → 5分钟收盘确认 → 下一根5分钟K线开盘成交。
- 专家参数：每个专家独立使用盈利概率、训练分位、月度候选目标、止盈止损与冷却时间。
- 专家淘汰：连续两个开发月无交易、最低胜率过低或多月负期望的候选不得进入最终组合。
- 组合筛选：优先比较开发月份最低胜率与最低盈利因子；5月参与开发筛选。
- 样本外：2026年6月未参与训练、阈值、专家淘汰或组合筛选，只在策略冻结后评估一次。
- 最终验收：**{'达到全部要求' if qualified else '未达到全部要求'}**。

## 最终专家组合

- 活跃专家：{'、'.join(selected_names) if selected_names else '无'}
- 组合键：`{selected.key}`
- 5月满足全部硬条件的开发组合：{status['may_hard_qualified_combinations']}

## 月度结果

| 月份 | 交易 | 胜率 | 实际平均盈利/平均亏损 | 盈利因子 | 净R | 最大回撤R |
|---|---:|---:|---:|---:|---:|---:|
| {MONTHS[8]} | {may_metrics['trades']} | {may_metrics['win_rate']*100:.2f}% | {may_metrics['avg_win_loss_ratio']:.3f} | {may_metrics['profit_factor']:.3f} | {may_metrics['net_R']:.3f} | {may_metrics['max_drawdown_R']:.3f} |
| {OOS_MONTH} | {june_metrics['trades']} | {june_metrics['win_rate']*100:.2f}% | {june_metrics['avg_win_loss_ratio']:.3f} | {june_metrics['profit_factor']:.3f} | {june_metrics['net_R']:.3f} | {june_metrics['max_drawdown_R']:.3f} |

## 保守执行假设

高周期特征只使用上一根已经完成的高周期K线；资金费率延迟一根5分钟K线使用；同一根K线同时触及止损和止盈时先按止损成交；策略同一时刻最多持有一笔仓位。
"""
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    (RESULTS / "run_identity.txt").write_text(
        f"{ENGINE_NAME}\nmonths={','.join(MONTHS)}\noos={OOS_MONTH}\noutput=results_v9\n",
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=ENGINE_NAME)
    parser.add_argument("--self-test", action="store_true", help="run offline synthetic regression tests")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
