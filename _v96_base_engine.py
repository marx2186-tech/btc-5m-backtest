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
    from sklearn.linear_model import LogisticRegression
except ImportError:  # One-file runner fallback.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.5.1"])
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
RESULTS = ROOT / "results_v9_5_1"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

ENGINE_VERSION = "V9.5.1"
ENGINE_NAME = "BTC 5m multi-model routed shadow ensemble V9.5.1"
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
    "research_stage": {
        "min_trades": 8,
        "max_trades": 15,
        "min_win_rate": 0.60,
        "min_avg_win_loss_ratio": 1.50,
        "min_profit_factor": 1.50,
        "max_drawdown_r": 3.0,
        "min_active_models": 3,
        "max_single_model_trade_share": 0.60
    },
    "core_expert_ids": [0, 1, 2, 3, 4, 5],
    "max_trades_per_expert_per_day": 2,
    "max_trades_per_setup_per_day": 1,
    "max_consecutive_losses_per_expert_day": 2,
    "loss_reentry_min_bars": 12,
    "require_new_structure_cycle_after_loss": True,
    "dedup_window_bars": 1,
    "direction_conflict_utility_margin": 0.08,
    "minimum_meta_probability": 0.50,
    "minimum_expected_utility_r": 0.03,
    "dynamic_weight_min": 0.10,
    "dynamic_weight_max": 0.50,
    "dynamic_weight_shadow_window": 20,
    "base_seed": 20260921,
    "combination_top_per_expert": 3,
    "combination_beam_width": 220,
    "diagnostic_top_per_expert": 2,
    "funnel_min_opportunity_candidates": 6,
    "calibration_min_rows": 24,
    "calibration_min_class": 3,
    "online_rank_window": 500,
    "online_rank_min_history": 35,
    "probability_shrinkage": 0.15,
    "stability_prior_trades": 10,
    "expert_elimination": {
        "max_consecutive_zero_opportunity_months": 2,
        "min_floor_win_rate": 0.35,
        "min_cumulative_win_rate": 0.45,
        "max_negative_expectancy_months": 2,
        "min_active_months": 3,
        "min_positive_months": 3,
        "min_total_trades": 12,
        "min_signal_months": 4,
        "min_raw_candidates_total": 60,
        "min_cumulative_profit_factor": 1.30,
        "min_cumulative_avg_win_loss_ratio": 1.30,
        "min_cumulative_net_r": 0.0,
        "max_worst_month_loss_r": 2.5,
        "max_single_month_profit_share": 0.55
    }
}


def load_request() -> dict[str, Any]:
    request = dict(DEFAULT_REQUEST)
    request["expert_elimination"] = dict(DEFAULT_REQUEST["expert_elimination"])
    request["research_stage"] = dict(DEFAULT_REQUEST["research_stage"])
    path = Path(os.environ.get("BACKTEST_REQUEST_FILE", str(ROOT / "request.v9_5_1.json")))
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("request.v9_5_1.json must contain a JSON object")
        elimination = loaded.pop("expert_elimination", None)
        research_stage = loaded.pop("research_stage", None)
        request.update(loaded)
        if elimination is not None:
            if not isinstance(elimination, dict):
                raise ValueError("expert_elimination must be an object")
            request["expert_elimination"].update(elimination)
        if research_stage is not None:
            if not isinstance(research_stage, dict):
                raise ValueError("research_stage must be an object")
            request["research_stage"].update(research_stage)
    return request


REQUEST = load_request()
SYMBOL = str(REQUEST["symbol"]).upper()
INTERVAL = str(REQUEST["interval"]).lower()
EVAL_MONTHS = tuple(str(x) for x in REQUEST["months"])
if len(EVAL_MONTHS) != 2:
    raise ValueError("V9.5.1 requires exactly two evaluation months")
first_eval = pd.Period(EVAL_MONTHS[0], freq="M")
last_eval = pd.Period(EVAL_MONTHS[1], freq="M")
if last_eval != first_eval + 1:
    raise ValueError("Evaluation months must be consecutive")
if str(last_eval) != OOS_MONTH:
    raise ValueError(f"V9.5.1 hard-codes {OOS_MONTH} as the untouched out-of-sample month")
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
DIAGNOSTIC_TOP_PER_EXPERT = int(REQUEST["diagnostic_top_per_expert"])
FUNNEL_MIN_OPPORTUNITY = int(REQUEST["funnel_min_opportunity_candidates"])
ELIMINATION = dict(REQUEST["expert_elimination"])
CALIBRATION_MIN_ROWS = int(REQUEST["calibration_min_rows"])
CALIBRATION_MIN_CLASS = int(REQUEST["calibration_min_class"])
ONLINE_RANK_WINDOW = int(REQUEST["online_rank_window"])
ONLINE_RANK_MIN_HISTORY = int(REQUEST["online_rank_min_history"])
PROBABILITY_SHRINKAGE = float(REQUEST["probability_shrinkage"])
STABILITY_PRIOR_TRADES = int(REQUEST["stability_prior_trades"])
RESEARCH_STAGE = dict(REQUEST["research_stage"])
STAGE_MIN_TRADES = int(RESEARCH_STAGE["min_trades"])
STAGE_MAX_TRADES = int(RESEARCH_STAGE["max_trades"])
STAGE_MIN_WIN_RATE = float(RESEARCH_STAGE["min_win_rate"])
STAGE_MIN_RATIO = float(RESEARCH_STAGE["min_avg_win_loss_ratio"])
STAGE_MAX_DRAWDOWN = float(RESEARCH_STAGE["max_drawdown_r"])
STAGE_MIN_PROFIT_FACTOR = float(RESEARCH_STAGE["min_profit_factor"])
STAGE_MIN_ACTIVE_MODELS = int(RESEARCH_STAGE["min_active_models"])
STAGE_MAX_MODEL_SHARE = float(RESEARCH_STAGE["max_single_model_trade_share"])
DEDUP_WINDOW_BARS = int(REQUEST["dedup_window_bars"])
DIRECTION_CONFLICT_MARGIN = float(REQUEST["direction_conflict_utility_margin"])
MIN_META_PROBABILITY = float(REQUEST["minimum_meta_probability"])
MIN_EXPECTED_UTILITY_R = float(REQUEST["minimum_expected_utility_r"])
DYNAMIC_WEIGHT_MIN = float(REQUEST["dynamic_weight_min"])
DYNAMIC_WEIGHT_MAX = float(REQUEST["dynamic_weight_max"])
DYNAMIC_WEIGHT_WINDOW = int(REQUEST["dynamic_weight_shadow_window"])
CORE_EXPERT_IDS = tuple(sorted({int(x) for x in REQUEST["core_expert_ids"]}))
MAX_TRADES_PER_EXPERT_DAY = int(REQUEST["max_trades_per_expert_per_day"])
MAX_TRADES_PER_SETUP_DAY = int(REQUEST["max_trades_per_setup_per_day"])
MAX_CONSECUTIVE_LOSSES_EXPERT_DAY = int(REQUEST["max_consecutive_losses_per_expert_day"])
LOSS_REENTRY_MIN_BARS = int(REQUEST["loss_reentry_min_bars"])
REQUIRE_NEW_STRUCTURE_CYCLE = bool(REQUEST["require_new_structure_cycle_after_loss"])

if CORE_EXPERT_IDS != (0, 1, 2, 3, 4, 5):
    raise ValueError("V9.5.1 requires six isolated directional specialists [0..5]")
if not (1 <= STAGE_MIN_TRADES <= STAGE_MAX_TRADES <= MIN_TRADES):
    raise ValueError("V9.5.1 research stage must not exceed the final minimum trade target")

if FEE_RATE != 0.0005:
    raise ValueError("V9.5.1 fixed requirement: fee_rate_per_side must be 0.0005")
if abs(SLIPPAGE_ABS - 0.2) > 1e-12:
    raise ValueError("V9.5.1 fixed requirement: each fill slippage must equal 0.2 USDT")

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
                headers={"User-Agent": "btc-multi-model-router-arbitrated-v9-5-1/1.0"},
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

    # V9.5.1 signal funnel: 1h market state and 15m structure remain hard gates.
    # ETH, premium, funding and microstructure are diagnostic/model features rather
    # than a chain of mandatory filters. This preserves context without starving
    # the expert candidate pool.
    reclaim_long_core = (
        (l <= x["ema21"] + 0.50 * x["atr"]) & (c > x["ema8"]) & (c > o)
    )
    reclaim_short_core = (
        (h >= x["ema21"] - 0.50 * x["atr"]) & (c < x["ema8"]) & (c < o)
    )
    reclaim_long_confirm = (
        (x["close_loc"] >= 0.55).astype(int)
        + (x["macd_slope_atr"] > 0).astype(int)
        + (x["taker_ratio"] >= 0.50).astype(int)
        + (x["rel_vol"] >= 0.75).astype(int)
    )
    reclaim_short_confirm = (
        (x["close_loc"] <= 0.45).astype(int)
        + (x["macd_slope_atr"] < 0).astype(int)
        + (x["taker_ratio"] <= 0.50).astype(int)
        + (x["rel_vol"] >= 0.75).astype(int)
    )
    reclaim_long = reclaim_long_core & (reclaim_long_confirm >= 1)
    reclaim_short = reclaim_short_core & (reclaim_short_confirm >= 1)

    continuation_long_core = (c > x["don6h"]) & (c > x["ema21"])
    continuation_short_core = (c < x["don6l"]) & (c < x["ema21"])
    continuation_long_confirm = (
        (x["body"] >= 0.34).astype(int)
        + (x["close_loc"] >= 0.62).astype(int)
        + (x["taker_ratio"] >= 0.51).astype(int)
        + (x["rel_vol"] >= 0.75).astype(int)
        + (x["range_exp"] >= 0.95).astype(int)
    )
    continuation_short_confirm = (
        (x["body"] >= 0.34).astype(int)
        + (x["close_loc"] <= 0.38).astype(int)
        + (x["taker_ratio"] <= 0.49).astype(int)
        + (x["rel_vol"] >= 0.75).astype(int)
        + (x["range_exp"] >= 0.95).astype(int)
    )
    continuation_long = continuation_long_core & (continuation_long_confirm >= 2)
    continuation_short = continuation_short_core & (continuation_short_confirm >= 2)

    # V9.5.1 adds a second, independent trend-short path: the first failed retest
    # after a completed downside break. It does not loosen the calibrated score
    # gate; it only gives the core short expert another structurally distinct setup.
    prior_down_break = (c.shift(1) < x["don12l"].shift(1)) & (c.shift(1) < x["ema21"].shift(1))
    break_retest_short = (
        prior_down_break
        & (h >= x["don12l"] - 0.20 * x["atr"])
        & (c < x["don12l"])
        & (c < o)
        & ((x["close_loc"] <= 0.42).astype(int)
           + (x["taker_ratio"] <= 0.49).astype(int)
           + (x["macd_slope_atr"] < 0).astype(int) >= 2)
    )

    range_sweep_long = (l < x["don12l"]) & (c > x["don12l"]) & (c > o)
    range_sweep_short = (h > x["don12h"]) & (c < x["don12h"]) & (c < o)
    range_long_confirm = (
        (x["lower_wick"] >= 0.26).astype(int)
        + (x["close_loc"] >= 0.54).astype(int)
        + (x["vwap_dev"] <= 0.45).astype(int)
        + (x["rsi"] > x["rsi"].shift(1)).astype(int)
        + (x["taker_ratio"] >= 0.48).astype(int)
        + (x["rel_vol"] >= 0.75).astype(int)
    )
    range_short_confirm = (
        (x["upper_wick"] >= 0.26).astype(int)
        + (x["close_loc"] <= 0.46).astype(int)
        + (x["vwap_dev"] >= -0.45).astype(int)
        + (x["rsi"] < x["rsi"].shift(1)).astype(int)
        + (x["taker_ratio"] <= 0.52).astype(int)
        + (x["rel_vol"] >= 0.75).astype(int)
    )
    range_reversal_long = range_sweep_long & (range_long_confirm >= 2)
    range_reversal_short = range_sweep_short & (range_short_confirm >= 2)

    # A second independent range-long path: after a prior liquidity sweep, price
    # retests the low without making a materially lower low and reclaims VWAP.
    recent_sweep_long = range_sweep_long.shift(1).rolling(24, min_periods=1).max().fillna(0).astype(bool)
    recent_sweep_low = l.where(range_sweep_long).shift(1).rolling(24, min_periods=1).min()
    range_second_test_long = (
        recent_sweep_long
        & recent_sweep_low.notna()
        & (l >= recent_sweep_low - 0.12 * x["atr"])
        & (l <= x["don12l"] + 0.35 * x["atr"])
        & (c > x["vwap"])
        & (c > o)
        & (x["rsi"] > x["rsi"].shift(1))
    )

    squeeze = x["bb_rank"].shift(2).rolling(8).min() <= 0.36
    squeeze_break_long = (
        squeeze & (c > x["don12h"]) & (x["range_exp"] >= 1.05)
        & ((x["body"] >= 0.40).astype(int) + (x["close_loc"] >= 0.68).astype(int)
           + (x["rel_vol"] >= 0.90).astype(int) + (x["taker_ratio"] >= 0.52).astype(int) >= 2)
    )
    squeeze_break_short = (
        squeeze & (c < x["don12l"]) & (x["range_exp"] >= 1.05)
        & ((x["body"] >= 0.40).astype(int) + (x["close_loc"] <= 0.32).astype(int)
           + (x["rel_vol"] >= 0.90).astype(int) + (x["taker_ratio"] <= 0.48).astype(int) >= 2)
    )
    impulse_break_long = (
        (c > x["don24h"]) & (x["range_exp"] >= 1.35)
        & (x["body"] >= 0.48) & (x["close_loc"] >= 0.70)
    )
    impulse_break_short = (
        (c < x["don24l"]) & (x["range_exp"] >= 1.35)
        & (x["body"] >= 0.48) & (x["close_loc"] <= 0.30)
    )
    high_break_long = squeeze_break_long | impulse_break_long
    high_break_short = squeeze_break_short | impulse_break_short

    high_vol = (x["atr_rank"] >= 0.72) | ((x["range_exp"] >= 1.30) & (x["atr_rank"] >= 0.55))
    trend_regime = (~high_vol) & (x["h1_trend"] != 0) & (x["h1_adx"] >= 16) & (x["chop"] <= 61)
    range_regime = (~high_vol) & (~trend_regime) & ((x["h1_adx"] <= 27) | (x["chop"] >= 54))
    neutral_regime = ~(high_vol | trend_regime | range_regime)
    x["regime"] = np.select([trend_regime, range_regime, high_vol, neutral_regime], [0, 1, 2, 3], default=3).astype(float)
    x["regime_trend"] = (x["regime"] == 0).astype(float)
    x["regime_range"] = (x["regime"] == 1).astype(float)
    x["regime_high_vol"] = (x["regime"] == 2).astype(float)
    x["regime_neutral"] = (x["regime"] == 3).astype(float)

    trend_long_market = trend_regime & (x["h1_trend"] > 0)
    trend_short_market = trend_regime & (x["h1_trend"] < 0)
    trend_long_structure = trend_long_market & (x["m15_trend"] >= 0) & (x["m15_gap"] >= -0.0015)
    trend_short_structure = trend_short_market & (x["m15_trend"] <= 0) & (x["m15_gap"] <= 0.0015)
    range_long_market = range_regime
    range_short_market = range_regime
    range_long_structure = range_long_market & ((x["m15_rsi"] <= 55) | (x["bb_pos"] <= 0.50))
    range_short_structure = range_short_market & ((x["m15_rsi"] >= 45) | (x["bb_pos"] >= 0.50))
    high_long_market = high_vol & (x["h1_trend"] >= 0)
    high_short_market = high_vol & (x["h1_trend"] <= 0)
    high_long_structure = high_long_market & (x["m15_trend"] >= 0)
    high_short_structure = high_short_market & (x["m15_trend"] <= 0)

    aux_long_support = (
        (x["eth_h1_trend"] >= 0).astype(int)
        + (x["derivative_pressure"] <= 3.5).astype(int)
        + (x["premium_z"] <= 2.2).astype(int)
        + (x["funding_z"] <= 3.0).astype(int)
    ) >= 2
    aux_short_support = (
        (x["eth_h1_trend"] <= 0).astype(int)
        + (x["derivative_pressure"] >= -3.5).astype(int)
        + (x["premium_z"] >= -2.2).astype(int)
        + (x["funding_z"] >= -3.0).astype(int)
    ) >= 2

    signal_columns: dict[str, pd.Series] = {
        "setup_reclaim_long": reclaim_long.astype(float),
        "setup_reclaim_short": reclaim_short.astype(float),
        "setup_continuation_long": continuation_long.astype(float),
        "setup_continuation_short": continuation_short.astype(float),
        "setup_break_retest_short": break_retest_short.astype(float),
        "setup_range_reversal_long": range_reversal_long.astype(float),
        "setup_range_second_test_long": range_second_test_long.astype(float),
        "setup_range_reversal_short": range_reversal_short.astype(float),
        "setup_high_break_long": high_break_long.astype(float),
        "setup_high_break_short": high_break_short.astype(float),
        "expert_trend_long": (trend_long_structure & (reclaim_long | continuation_long)).astype(float),
        "expert_trend_short": (trend_short_structure & (reclaim_short | continuation_short | break_retest_short)).astype(float),
        "expert_range_long": (range_long_structure & (range_reversal_long | range_second_test_long)).astype(float),
        "expert_range_short": (range_short_structure & range_reversal_short).astype(float),
        "expert_high_long": (high_long_structure & high_break_long).astype(float),
        "expert_high_short": (high_short_structure & high_break_short).astype(float),
    }

    funnel_definitions = {
        "trend_long": (trend_long_market, trend_long_structure, reclaim_long | continuation_long, aux_long_support),
        "trend_short": (trend_short_market, trend_short_structure, reclaim_short | continuation_short | break_retest_short, aux_short_support),
        "range_long": (range_long_market, range_long_structure, range_reversal_long | range_second_test_long, aux_long_support),
        "range_short": (range_short_market, range_short_structure, range_reversal_short, aux_short_support),
        "high_long": (high_long_market, high_long_structure, high_break_long, aux_long_support),
        "high_short": (high_short_market, high_short_structure, high_break_short, aux_short_support),
    }
    for key, (market_mask, structure_mask, setup_mask, aux_mask) in funnel_definitions.items():
        signal_columns[f"funnel_{key}_market"] = market_mask.astype(float)
        signal_columns[f"funnel_{key}_structure"] = structure_mask.astype(float)
        signal_columns[f"funnel_{key}_setup"] = setup_mask.astype(float)
        signal_columns[f"funnel_{key}_structural_candidate"] = (structure_mask & setup_mask).astype(float)
        signal_columns[f"funnel_{key}_aux_support"] = (structure_mask & setup_mask & aux_mask).astype(float)
        rising = structure_mask & (~structure_mask.shift(1, fill_value=False))
        signal_columns[f"funnel_{key}_structure_cycle"] = rising.cumsum().astype(float)
    x = pd.concat([x, pd.DataFrame(signal_columns, index=x.index)], axis=1)

    hours = x.index.hour + x.index.minute / 60.0
    x["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    x["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    x["weekday"] = x.index.weekday.astype(float)
    x["month"] = x.index.strftime("%Y-%m")

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
    "setup_break_retest_short",
    "setup_range_reversal_long", "setup_range_reversal_short",
    "setup_range_second_test_long",
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


@dataclass
class CalibratedExpertModel:
    base_model: HistGradientBoostingClassifier
    calibrator: LogisticRegression | None
    calibration_month: str
    core_months: tuple[str, ...]
    calibration_scores: np.ndarray
    calibration_labels: np.ndarray
    calibration_base_rate: float
    calibration_brier: float
    core_rows: int
    calibration_rows: int


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
    monthly_thresholds: dict[str, dict[str, Any]] = field(default_factory=dict)
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


def _enough_classes(y: np.ndarray, minimum_rows: int, minimum_class: int) -> bool:
    return bool(
        len(y) >= minimum_rows
        and len(np.unique(y)) == 2
        and min(int(np.sum(y == 0)), int(np.sum(y == 1))) >= minimum_class
    )


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(float), 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def calibrate_probabilities(model: CalibratedExpertModel, base_probability: np.ndarray) -> np.ndarray:
    base_probability = np.asarray(base_probability, dtype=float)
    if model.calibrator is not None:
        raw = model.calibrator.predict_proba(_logit(base_probability))[:, 1]
    else:
        raw = base_probability
    # Shrink every estimate slightly toward the observed holdout base rate. This
    # prevents tiny calibration sets from producing unstable 0/1 probabilities.
    calibrated = (
        (1.0 - PROBABILITY_SHRINKAGE) * raw
        + PROBABILITY_SHRINKAGE * model.calibration_base_rate
    )
    return np.clip(calibrated, 0.01, 0.99)


def fit_one_expert_model(
    x: pd.DataFrame,
    expert_data: dict[int, dict[str, Any]],
    policy: ExpertPolicy,
    train_months: set[str],
) -> CalibratedExpertModel | None:
    assert_oos_isolation(train_months, f"model training expert={policy.expert_id}")
    data = expert_data[policy.expert_id]
    idx = data["idx"]
    open_, high, low, close, atr = (x[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr"))
    labels, _, net_r, _ = compute_outcomes(
        idx, int(data["direction"]), open_, high, low, close, atr,
        policy.risk.rr, policy.risk.sl_atr, policy.risk.min_stop_pct, policy.risk.max_hold,
        policy.risk.breakeven_trigger, policy.risk.breakeven_lock,
        policy.risk.early_bars, policy.risk.early_cut_r,
        FEE_RATE, SLIPPAGE_ABS,
    )
    months = x["month"].to_numpy()[idx]
    valid_train = np.isin(months, list(train_months)) & (labels >= 0)
    ordered_months = sorted(str(m) for m in train_months)
    if len(ordered_months) < 2:
        return None

    calibration_month = ordered_months[-1]
    core_months = tuple(ordered_months[:-1])
    core_mask = valid_train & np.isin(months, list(core_months))
    calibration_mask = valid_train & (months == calibration_month)
    y_core = labels[core_mask].astype(int)
    y_calibration = labels[calibration_mask].astype(int)

    base_min_rows = 45 if policy.expert_id in (0, 1) else 36
    base_min_class = 8 if policy.expert_id in (0, 1) else 6

    # Fallback to a strictly chronological tail holdout if a complete month is
    # too small. The tail remains excluded from base-model fitting.
    if not _enough_classes(y_core, base_min_rows, base_min_class) or not _enough_classes(
        y_calibration, CALIBRATION_MIN_ROWS, CALIBRATION_MIN_CLASS
    ):
        positions = np.flatnonzero(valid_train)
        if len(positions) < base_min_rows + CALIBRATION_MIN_ROWS:
            return None
        split = max(base_min_rows, int(len(positions) * 0.75))
        split = min(split, len(positions) - CALIBRATION_MIN_ROWS)
        core_positions = positions[:split]
        calibration_positions = positions[split:]
        core_mask = np.zeros(len(labels), dtype=bool)
        calibration_mask = np.zeros(len(labels), dtype=bool)
        core_mask[core_positions] = True
        calibration_mask[calibration_positions] = True
        y_core = labels[core_mask].astype(int)
        y_calibration = labels[calibration_mask].astype(int)
        if not _enough_classes(y_core, base_min_rows, base_min_class) or not _enough_classes(
            y_calibration, CALIBRATION_MIN_ROWS, CALIBRATION_MIN_CLASS
        ):
            return None
        calibration_month = "chronological_tail"
        core_months = tuple(sorted(set(str(m) for m in months[core_mask])))

    ordered_core = sorted(set(str(m) for m in months[core_mask]))
    recency = {m: 0.86 + 0.28 * (i / max(1, len(ordered_core) - 1)) for i, m in enumerate(ordered_core)}
    weights = np.array([recency.get(str(m), 1.0) for m in months[core_mask]], dtype=float)
    magnitude = 1.0 + 0.15 * np.minimum(np.abs(net_r[core_mask]), 2.0)

    base_model = HistGradientBoostingClassifier(
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
    base_model.fit(
        x.iloc[idx[core_mask]][FEATURES].to_numpy(np.float64),
        y_core,
        sample_weight=mild_class_weights(y_core) * weights * magnitude,
    )

    calibration_base_probability = base_model.predict_proba(
        x.iloc[idx[calibration_mask]][FEATURES].to_numpy(np.float64)
    )[:, 1]
    calibrator: LogisticRegression | None = None
    if _enough_classes(y_calibration, CALIBRATION_MIN_ROWS, CALIBRATION_MIN_CLASS):
        calibrator = LogisticRegression(
            C=0.75,
            solver="lbfgs",
            max_iter=500,
            random_state=BASE_SEED + policy.expert_id * 211,
        )
        calibrator.fit(_logit(calibration_base_probability), y_calibration)

    base_rate = float(np.mean(y_calibration))
    provisional = CalibratedExpertModel(
        base_model=base_model,
        calibrator=calibrator,
        calibration_month=calibration_month,
        core_months=core_months,
        calibration_scores=np.empty(0, dtype=float),
        calibration_labels=y_calibration,
        calibration_base_rate=base_rate,
        calibration_brier=0.0,
        core_rows=int(np.sum(core_mask)),
        calibration_rows=int(np.sum(calibration_mask)),
    )
    calibration_scores = calibrate_probabilities(provisional, calibration_base_probability)
    provisional.calibration_scores = calibration_scores
    provisional.calibration_brier = float(np.mean((calibration_scores - y_calibration) ** 2))
    return provisional


def setup_group_array(x: pd.DataFrame, idx: np.ndarray, expert_id: int) -> np.ndarray:
    frame = x.iloc[idx]
    if expert_id == 1:
        return np.where(
            frame["setup_break_retest_short"].to_numpy(float) > 0.5,
            "趋势破位首次反抽",
            np.where(
                frame["setup_reclaim_short"].to_numpy(float) > 0.5,
                "趋势回踩失败",
                "趋势延续突破",
            ),
        ).astype(object)
    if expert_id == 2:
        return np.where(
            frame["setup_range_second_test_long"].to_numpy(float) > 0.5,
            "震荡二次测试",
            "震荡扫低收回",
        ).astype(object)
    return np.full(len(idx), "诊断形态", dtype=object)


def build_one_expert_bundle(
    x: pd.DataFrame,
    expert_data: dict[int, dict[str, Any]],
    policy: ExpertPolicy,
    model: CalibratedExpertModel,
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
    base_probability = model.base_model.predict_proba(
        x.iloc[idx][FEATURES].to_numpy(np.float64)
    )[:, 1]
    probability = calibrate_probabilities(model, base_probability)
    expert_key = str(data["key"])
    return {
        "idx": idx,
        "direction": int(data["direction"]),
        "expert_id": policy.expert_id,
        "expert": str(data["name"]),
        "labels": labels,
        "exits": exits,
        "net_r": net_r,
        "reasons": reasons,
        "base_prob": base_probability,
        "prob": probability,
        "months": x["month"].to_numpy()[idx],
        "days": x.index.strftime("%Y-%m-%d").to_numpy()[idx],
        "structure_cycle": x[f"funnel_{expert_key}_structure_cycle"].to_numpy(np.int64)[idx],
        "setup_group": setup_group_array(x, idx, policy.expert_id),
        "calibration_month": model.calibration_month,
        "calibration_scores": model.calibration_scores,
        "calibration_labels": model.calibration_labels,
        "calibration_base_rate": model.calibration_base_rate,
        "calibration_brier": model.calibration_brier,
        "core_rows": model.core_rows,
        "calibration_rows": model.calibration_rows,
    }


def events_from_expert(
    bundle: dict[str, Any],
    policy: ExpertPolicy,
    eval_month: str,
    train_months: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert_oos_isolation(train_months, f"threshold calibration expert={policy.expert_id} eval={eval_month}")
    eval_positions = np.flatnonzero((bundle["months"] == eval_month) & (bundle["exits"] >= 0))
    calibration_scores = np.asarray(bundle.get("calibration_scores", []), dtype=float)
    calibration_scores = calibration_scores[np.isfinite(calibration_scores)]
    base_audit: dict[str, Any] = {
        "model_available": True,
        "training_candidates": int(bundle.get("core_rows", 0)),
        "calibration_month": str(bundle.get("calibration_month", "unknown")),
        "calibration_candidates": int(len(calibration_scores)),
        "calibration_base_rate": float(bundle.get("calibration_base_rate", 0.0)),
        "calibration_brier": float(bundle.get("calibration_brier", 0.0)),
        "raw_structural_candidates": int(len(eval_positions)),
    }
    if len(calibration_scores) < ONLINE_RANK_MIN_HISTORY:
        return [], {
            **base_audit,
            "probability_threshold": float(policy.min_probability),
            "quantile": float(policy.min_percentile),
            "raw_eval_after_probability": 0,
            "capped_eval_candidates": 0,
            "after_cooldown": 0,
            "score_drift": 0.0,
        }

    history = list(calibration_scores[-ONLINE_RANK_WINDOW:])
    events: list[dict[str, Any]] = []
    passed_gate = 0
    eval_probabilities: list[float] = []
    for k in eval_positions:
        probability = float(bundle["prob"][k])
        base_probability = float(bundle.get("base_prob", bundle["prob"])[k])
        hist = np.asarray(history, dtype=float)
        percentile = float((np.sum(hist <= probability) + 1.0) / (len(hist) + 1.0))
        eval_probabilities.append(probability)
        passes = probability >= policy.min_probability and percentile >= policy.min_percentile
        # Append every observed candidate score after making the decision, so the
        # percentile adapts online without seeing future candidates in the month.
        history.append(probability)
        if len(history) > ONLINE_RANK_WINDOW:
            del history[: len(history) - ONLINE_RANK_WINDOW]
        if not passes:
            continue
        passed_gate += 1
        if len(events) >= policy.monthly_target:
            continue
        events.append({
            "signal_i": int(bundle["idx"][k]),
            "exit_i": int(bundle["exits"][k]),
            "direction": int(bundle["direction"]),
            "expert_id": int(bundle["expert_id"]),
            "expert": str(bundle["expert"]),
            "base_prob": base_probability,
            "prob": probability,
            "score": percentile,
            "net_r": float(bundle["net_r"][k]),
            "reason": int(bundle["reasons"][k]),
            "policy_key": policy.key,
            "day": str(bundle.get("days", np.full(len(bundle["idx"]), f"{eval_month}-01", dtype=object))[k]),
            "structure_cycle": int(bundle.get("structure_cycle", np.arange(len(bundle["idx"]), dtype=np.int64))[k]),
            "setup_group": str(bundle.get("setup_group", np.full(len(bundle["idx"]), "self-test", dtype=object))[k]),
        })

    events = sorted(events, key=lambda e: (e["signal_i"], -e["score"], -e["prob"]))
    after_cooldown = len(apply_expert_cooldown(events, policy.cooldown))
    eval_mean = float(np.mean(eval_probabilities)) if eval_probabilities else 0.0
    calibration_mean = float(np.mean(calibration_scores)) if len(calibration_scores) else 0.0
    return events, {
        **base_audit,
        "probability_threshold": float(policy.min_probability),
        "quantile": float(policy.min_percentile),
        "raw_eval_after_probability": int(passed_gate),
        "capped_eval_candidates": int(len(events)),
        "after_cooldown": int(after_cooldown),
        "calibration_score_mean": calibration_mean,
        "eval_score_mean": eval_mean,
        "score_drift": float(eval_mean - calibration_mean),
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
    """Combine experts with causal quality-protection controls.

    V9.5.1 deliberately does not relax probability or percentile gates. Frequency
    can only increase through independent setups. After a loss, the same expert
    and direction must wait at least LOSS_REENTRY_MIN_BARS and, when enabled,
    a newly activated completed-15m structure cycle. Daily caps and consecutive
    loss shutdowns prevent one mistaken regime from creating a cluster of trades.
    """
    prefiltered: list[dict[str, Any]] = []
    for expert_id, events in events_by_expert.items():
        if expert_id not in policies:
            continue
        prefiltered.extend(apply_expert_cooldown(events, policies[expert_id].cooldown))
    prefiltered.sort(key=lambda e: (e["signal_i"], -e["score"], -e["prob"]))

    selected: list[dict[str, Any]] = []
    global_last_exit = -10**9
    expert_day_count: dict[tuple[int, str], int] = {}
    setup_day_count: dict[tuple[int, str, str], int] = {}
    consecutive_losses: dict[tuple[int, str], int] = {}
    last_loss: dict[int, dict[str, Any]] = {}

    def allowed(event: dict[str, Any]) -> bool:
        expert_id = int(event["expert_id"])
        day = str(event.get("day", "unknown"))
        setup_group = str(event.get("setup_group", "unknown"))
        if expert_day_count.get((expert_id, day), 0) >= MAX_TRADES_PER_EXPERT_DAY:
            return False
        if setup_day_count.get((expert_id, day, setup_group), 0) >= MAX_TRADES_PER_SETUP_DAY:
            return False
        if consecutive_losses.get((expert_id, day), 0) >= MAX_CONSECUTIVE_LOSSES_EXPERT_DAY:
            return False
        previous = last_loss.get(expert_id)
        if previous is not None and int(previous["direction"]) == int(event["direction"]):
            if int(event["signal_i"]) <= int(previous["exit_i"]) + LOSS_REENTRY_MIN_BARS:
                return False
            if REQUIRE_NEW_STRUCTURE_CYCLE and int(event.get("structure_cycle", 0)) <= int(previous["structure_cycle"]):
                return False
        return True

    i = 0
    while i < len(prefiltered):
        signal_i = int(prefiltered[i]["signal_i"])
        same: list[dict[str, Any]] = []
        while i < len(prefiltered) and int(prefiltered[i]["signal_i"]) == signal_i:
            same.append(prefiltered[i])
            i += 1
        if signal_i <= global_last_exit:
            continue
        same = [event for event in same if allowed(event)]
        if not same:
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

        chosen = dict(best)
        chosen["quality_protection"] = "passed"
        selected.append(chosen)
        global_last_exit = int(chosen["exit_i"])
        expert_id = int(chosen["expert_id"])
        day = str(chosen.get("day", "unknown"))
        setup_group = str(chosen.get("setup_group", "unknown"))
        expert_day_count[(expert_id, day)] = expert_day_count.get((expert_id, day), 0) + 1
        setup_day_count[(expert_id, day, setup_group)] = setup_day_count.get((expert_id, day, setup_group), 0) + 1
        if float(chosen["net_r"]) <= 0.0:
            consecutive_losses[(expert_id, day)] = consecutive_losses.get((expert_id, day), 0) + 1
            last_loss[expert_id] = {
                "exit_i": int(chosen["exit_i"]),
                "structure_cycle": int(chosen.get("structure_cycle", 0)),
                "direction": int(chosen["direction"]),
            }
        else:
            consecutive_losses[(expert_id, day)] = 0
            last_loss.pop(expert_id, None)
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


def raw_structural_candidates(candidate: ExpertCandidate, month: str) -> int:
    return int(candidate.monthly_thresholds.get(month, {}).get("raw_structural_candidates", 0))


def consecutive_zero_opportunity_months(candidate: ExpertCandidate) -> int:
    maximum = current = 0
    for month in DEVELOPMENT_MONTHS:
        has_opportunity = raw_structural_candidates(candidate, month) >= FUNNEL_MIN_OPPORTUNITY
        has_trade = int(candidate.monthly_metrics.get(month, {}).get("trades", 0)) > 0
        if has_opportunity and not has_trade:
            current += 1
            maximum = max(maximum, current)
        elif has_trade:
            current = 0
        else:
            # No corresponding market/setup opportunity: do not count as failure.
            current = 0
    return maximum


def cumulative_candidate_metrics(candidate: ExpertCandidate) -> dict[str, float]:
    values = [
        event
        for month in DEVELOPMENT_MONTHS
        for event in candidate.monthly_events.get(month, [])
    ]
    return metrics(values)


def evaluate_elimination(candidate: ExpertCandidate) -> tuple[bool, list[str]]:
    monthly = candidate.monthly_metrics
    reasons: list[str] = []
    active = [m for m in DEVELOPMENT_MONTHS if int(monthly.get(m, {}).get("trades", 0)) > 0]
    signal_months = [m for m in DEVELOPMENT_MONTHS if raw_structural_candidates(candidate, m) > 0]
    total_raw = sum(raw_structural_candidates(candidate, m) for m in DEVELOPMENT_MONTHS)
    cumulative = cumulative_candidate_metrics(candidate)
    total_trades = int(cumulative["trades"])
    if consecutive_zero_opportunity_months(candidate) >= int(ELIMINATION["max_consecutive_zero_opportunity_months"]):
        reasons.append("连续两个有机会开发月无交易")
    meaningful_wr = [
        float(monthly[m]["win_rate"])
        for m in active
        if int(monthly[m]["trades"]) >= 3
    ]
    if meaningful_wr and min(meaningful_wr) < float(ELIMINATION["min_floor_win_rate"]):
        reasons.append("开发月最低胜率过低")
    if total_trades >= int(ELIMINATION["min_total_trades"]) and cumulative["win_rate"] < float(ELIMINATION["min_cumulative_win_rate"]):
        reasons.append("累计胜率过低")
    negative_months = sum(float(monthly.get(m, {}).get("expectancy_R", 0.0)) < 0 for m in active)
    if negative_months >= int(ELIMINATION["max_negative_expectancy_months"]):
        reasons.append("多月负期望")
    if len(active) < int(ELIMINATION["min_active_months"]):
        reasons.append("有效开发月份不足")
    if len(signal_months) < int(ELIMINATION["min_signal_months"]):
        reasons.append("原始信号月份不足")
    if total_raw < int(ELIMINATION["min_raw_candidates_total"]):
        reasons.append("原始候选不足")
    if total_trades < int(ELIMINATION["min_total_trades"]):
        reasons.append("开发交易样本不足")
    if total_trades >= int(ELIMINATION["min_total_trades"]):
        if cumulative["profit_factor"] < float(ELIMINATION["min_cumulative_profit_factor"]):
            reasons.append("累计盈利因子不足")
        if cumulative["avg_win_loss_ratio"] < float(ELIMINATION["min_cumulative_avg_win_loss_ratio"]):
            reasons.append("累计实际盈亏比不足")
        if cumulative["net_R"] <= float(ELIMINATION["min_cumulative_net_r"]):
            reasons.append("累计净R非正")
        positive_months = sum(float(monthly.get(m, {}).get("net_R", 0.0)) > 0 for m in DEVELOPMENT_MONTHS)
        if positive_months < int(ELIMINATION["min_positive_months"]):
            reasons.append("正收益月份不足")
        worst_month = min(float(monthly.get(m, {}).get("net_R", 0.0)) for m in DEVELOPMENT_MONTHS)
        if worst_month < -float(ELIMINATION["max_worst_month_loss_r"]):
            reasons.append("最差月份亏损过大")
        positive_values = [max(0.0, float(monthly.get(m, {}).get("net_R", 0.0))) for m in DEVELOPMENT_MONTHS]
        positive_total = sum(positive_values)
        concentration = max(positive_values) / positive_total if positive_total > 0 else 1.0
        if concentration > float(ELIMINATION["max_single_month_profit_share"]):
            reasons.append("利润过度集中单月")
    return len(reasons) == 0, reasons

def shrunk_win_rate(metric: dict[str, float], prior_rate: float = 0.50) -> float:
    trades = int(metric.get("trades", 0))
    wins = int(metric.get("wins", 0))
    return float((wins + STABILITY_PRIOR_TRADES * prior_rate) / (trades + STABILITY_PRIOR_TRADES))


def expert_candidate_score(candidate: ExpertCandidate) -> float:
    monthly = candidate.monthly_metrics
    active = [monthly[m] for m in DEVELOPMENT_MONTHS if int(monthly.get(m, {}).get("trades", 0)) > 0]
    if not active:
        total_raw = sum(raw_structural_candidates(candidate, m) for m in DEVELOPMENT_MONTHS)
        return -1e9 + total_raw
    meaningful = [m for m in active if int(m["trades"]) >= 3] or active
    shrunk_rates = [shrunk_win_rate(m) for m in meaningful]
    floor_wr = min(shrunk_rates)
    median_wr = float(np.median(shrunk_rates))
    positive_pf = [float(m["profit_factor"]) for m in meaningful if float(m["profit_factor"]) > 0]
    floor_pf = min(positive_pf) if positive_pf else 0.0
    cumulative = cumulative_candidate_metrics(candidate)
    trade_counts = np.array([int(monthly[m]["trades"]) for m in DEVELOPMENT_MONTHS], dtype=float)
    target_gap = float(np.mean(np.abs(trade_counts - candidate.policy.monthly_target)))
    total_raw = sum(raw_structural_candidates(candidate, m) for m in DEVELOPMENT_MONTHS)
    coverage = len(active) / len(DEVELOPMENT_MONTHS)
    monthly_net = np.array([float(monthly[m]["net_R"]) for m in DEVELOPMENT_MONTHS], dtype=float)
    dispersion = float(np.std(monthly_net))
    concentration = float(np.max(trade_counts) / max(1.0, np.sum(trade_counts)))
    drift = np.mean([
        abs(float(candidate.monthly_thresholds.get(m, {}).get("score_drift", 0.0)))
        for m in DEVELOPMENT_MONTHS
    ])
    return float(
        floor_wr * 3200
        + median_wr * 1800
        + min(floor_pf, 5.0) * 650
        + cumulative["win_rate"] * 1600
        + min(cumulative["profit_factor"], 6.0) * 480
        + cumulative["net_R"] * 32
        + coverage * 900
        + min(total_raw, 360) * 1.0
        - cumulative["max_drawdown_R"] * 24
        - target_gap * 14
        - dispersion * 180
        - concentration * 650
        - drift * 1200
        - consecutive_zero_opportunity_months(candidate) * 650
    )


def combination_score(candidate: CombinationCandidate) -> float:
    monthly = candidate.monthly_metrics
    stats = [monthly[m] for m in DEVELOPMENT_MONTHS]
    meaningful = [m for m in stats if int(m["trades"]) >= 3]
    if len(meaningful) < 2:
        return -1e12
    shrunk_rates = [shrunk_win_rate(m) for m in meaningful]
    floor_wr = min(shrunk_rates)
    median_wr = float(np.median(shrunk_rates))
    positive_pf = [float(m["profit_factor"]) for m in meaningful if float(m["profit_factor"]) > 0]
    floor_pf = min(positive_pf) if positive_pf else 0.0
    ratios = [float(m["avg_win_loss_ratio"]) for m in meaningful if float(m["avg_win_loss_ratio"]) > 0]
    floor_ratio = min(ratios) if ratios else 0.0
    coverage = len(meaningful) / len(DEVELOPMENT_MONTHS)
    may = monthly[MONTHS[8]]
    counts = np.array([int(m["trades"]) for m in stats], dtype=float)
    count_penalty = sum(
        abs(int(m["trades"]) - np.clip(int(m["trades"]), STAGE_MIN_TRADES, STAGE_MAX_TRADES))
        for m in stats
    )
    net_values = np.array([float(m["net_R"]) for m in stats], dtype=float)
    net_dispersion = float(np.std(net_values))
    monthly_concentration = float(np.max(counts) / max(1.0, np.sum(counts)))
    negative_months = int(np.sum(net_values < 0))

    expert_totals: list[int] = []
    for expert_id in CORE_EXPERT_IDS:
        name = EXPERT_NAME[expert_id]
        expert_totals.append(sum(
            int(candidate.monthly_expert_metrics.get(month, {}).get(name, {}).get("trades", 0))
            for month in DEVELOPMENT_MONTHS
        ))
    total_core = sum(expert_totals)
    expert_concentration = max(expert_totals) / total_core if total_core else 1.0

    may_penalty = 0.0
    if not (STAGE_MIN_TRADES <= may["trades"] <= STAGE_MAX_TRADES):
        may_penalty += 5000 + abs(may["trades"] - np.clip(may["trades"], STAGE_MIN_TRADES, STAGE_MAX_TRADES)) * 450
    if may["win_rate"] < STAGE_MIN_WIN_RATE:
        may_penalty += (STAGE_MIN_WIN_RATE - may["win_rate"]) * 18000
    if may["avg_win_loss_ratio"] < STAGE_MIN_RATIO:
        may_penalty += (STAGE_MIN_RATIO - may["avg_win_loss_ratio"]) * 7000
    if may["max_drawdown_R"] > STAGE_MAX_DRAWDOWN:
        may_penalty += (may["max_drawdown_R"] - STAGE_MAX_DRAWDOWN) * 1800

    return float(
        floor_wr * 12000
        + median_wr * 3200
        + min(floor_pf, 5.0) * 2200
        + min(floor_ratio, 4.0) * 1000
        + coverage * 2600
        + sum(net_values) * 26
        - sum(float(m["max_drawdown_R"]) for m in stats) * 22
        - count_penalty * 110
        - net_dispersion * 480
        - monthly_concentration * 2400
        - max(0.0, expert_concentration - 0.70) * 5000
        - negative_months * 650
        - may_penalty
    )


def risk_grid_for_expert(expert_id: int) -> list[RiskConfig]:
    if expert_id == 1:
        return [
            RiskConfig(1.8, 1.15, 0.0030, 180, 0.90, 0.08, 48, 0.38),
            RiskConfig(2.0, 1.25, 0.0034, 216, 1.00, 0.12, 60, 0.38),
            RiskConfig(2.2, 1.40, 0.0036, 288, 1.10, 0.15, 72, 0.35),
        ]
    if expert_id == 0:
        return [
            RiskConfig(1.8, 1.10, 0.0030, 180, 0.90, 0.08, 48, 0.38),
            RiskConfig(2.0, 1.20, 0.0032, 216, 0.95, 0.10, 54, 0.38),
            RiskConfig(2.2, 1.35, 0.0035, 288, 1.05, 0.14, 66, 0.35),
        ]
    if expert_id in (2, 3):
        return [
            RiskConfig(1.6, 0.95, 0.0028, 120, 0.80, 0.06, 36, 0.32),
            RiskConfig(1.8, 1.05, 0.0030, 144, 0.85, 0.08, 42, 0.32),
            RiskConfig(2.0, 1.20, 0.0033, 180, 0.95, 0.10, 48, 0.30),
        ]
    return [
        RiskConfig(1.9, 1.25, 0.0035, 180, 0.95, 0.10, 42, 0.36),
        RiskConfig(2.2, 1.45, 0.0040, 216, 1.10, 0.12, 48, 0.35),
        RiskConfig(2.5, 1.65, 0.0045, 288, 1.25, 0.18, 60, 0.32),
    ]


def model_grid_for_expert(expert_id: int) -> list[ModelConfig]:
    if expert_id in (2, 3, 4, 5):
        return [
            ModelConfig(2, 0.035, 220, 6.0, 32),
            ModelConfig(3, 0.025, 260, 8.0, 42),
        ]
    return [
        ModelConfig(2, 0.035, 220, 5.0, 38),
        ModelConfig(3, 0.025, 270, 8.0, 50),
    ]


def policy_grid_for_expert(expert_id: int) -> list[ExpertPolicy]:
    # The score gate uses calibrated probability plus an online percentile over
    # past candidates. Lower absolute floors are intentional because the honest
    # holdout calibrator no longer emits the overconfident in-sample scores used
    # by V9.1.
    if expert_id == 1:
        targets, probs, pcts, cooldowns = (6, 8, 10), (0.38, 0.42, 0.46), (0.70, 0.78, 0.86), (1, 3, 6)
    elif expert_id == 2:
        targets, probs, pcts, cooldowns = (6, 8, 10), (0.38, 0.42, 0.46), (0.75, 0.82, 0.88), (2, 5, 8)
    elif expert_id in (0, 3):
        targets, probs, pcts, cooldowns = (4, 7, 10), (0.34, 0.38, 0.42), (0.65, 0.75, 0.85), (2, 5, 8)
    else:
        targets, probs, pcts, cooldowns = (4, 7, 10), (0.35, 0.40, 0.45), (0.68, 0.78, 0.86), (2, 5, 8)
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
                raw_count = int(np.sum(x["month"].to_numpy()[expert_data[expert_id]["idx"]] == eval_month))
                for candidate in group:
                    candidate.monthly_events[eval_month] = []
                    candidate.monthly_metrics[eval_month] = metrics([])
                    candidate.monthly_thresholds[eval_month] = {
                        "model_available": False,
                        "training_candidates": 0,
                        "calibration_month": "unavailable",
                        "calibration_candidates": 0,
                        "calibration_base_rate": 0.0,
                        "calibration_brier": 0.0,
                        "raw_structural_candidates": raw_count,
                        "probability_threshold": float(candidate.policy.min_probability),
                        "quantile": float(candidate.policy.min_percentile),
                        "raw_eval_after_probability": 0,
                        "capped_eval_candidates": 0,
                        "after_cooldown": 0,
                        "score_drift": 0.0,
                    }
                continue
            bundle = build_one_expert_bundle(x, expert_data, representative, model)
            event_cache: dict[tuple[int, float, float], tuple[list[dict[str, Any]], dict[str, Any]]] = {}
            for candidate in group:
                key = (
                    candidate.policy.monthly_target,
                    candidate.policy.min_probability,
                    candidate.policy.min_percentile,
                )
                if key not in event_cache:
                    event_cache[key] = events_from_expert(bundle, candidate.policy, eval_month, train_months)
                cached_events, threshold_audit = event_cache[key]
                chosen = apply_expert_cooldown(cached_events, candidate.policy.cooldown)
                candidate.monthly_events[eval_month] = chosen
                candidate.monthly_metrics[eval_month] = metrics(chosen)
                candidate.monthly_thresholds[eval_month] = {
                    **threshold_audit,
                    "after_cooldown": int(len(chosen)),
                }

    for candidate in candidates:
        candidate.eligible, candidate.elimination_reasons = evaluate_elimination(candidate)
        candidate.aggregate_score = expert_candidate_score(candidate)
    candidates.sort(key=lambda c: c.aggregate_score, reverse=True)
    return candidates


def candidate_options_by_expert(all_candidates: dict[int, list[ExpertCandidate]]) -> dict[int, list[ExpertCandidate]]:
    options: dict[int, list[ExpertCandidate]] = {}
    for expert_id, candidates in all_candidates.items():
        if expert_id not in CORE_EXPERT_IDS:
            options[expert_id] = []
            continue
        eligible = [c for c in candidates if c.eligible]
        options[expert_id] = eligible[:TOP_PER_EXPERT]
    return options


def build_combinations(
    options: dict[int, list[ExpertCandidate]],
) -> list[CombinationCandidate]:
    active_ids = [expert_id for expert_id in range(6) if options.get(expert_id)]
    if not active_ids:
        return []
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


def diagnostic_options_by_expert(all_candidates: dict[int, list[ExpertCandidate]]) -> dict[int, list[ExpertCandidate]]:
    options: dict[int, list[ExpertCandidate]] = {}
    for expert_id, candidates in all_candidates.items():
        if expert_id not in CORE_EXPERT_IDS:
            options[expert_id] = []
            continue
        usable = [
            c for c in candidates
            if sum(raw_structural_candidates(c, m) for m in DEVELOPMENT_MONTHS) > 0
            and sum(int(c.monthly_metrics.get(m, {}).get("trades", 0)) for m in DEVELOPMENT_MONTHS) > 0
        ]
        options[expert_id] = usable[:DIAGNOSTIC_TOP_PER_EXPERT]
    return options


def static_signal_funnel(
    x: pd.DataFrame,
    months: Iterable[str],
) -> dict[str, dict[str, dict[str, int]]]:
    output: dict[str, dict[str, dict[str, int]]] = {}
    for month in tuple(str(m) for m in months):
        frame = x.loc[x["month"] == month]
        output[month] = {}
        for expert in EXPERTS:
            key = str(expert["key"])
            output[month][str(expert["name"])] = {
                "bars": int(len(frame)),
                "market_state": int(frame[f"funnel_{key}_market"].sum()),
                "structure": int(frame[f"funnel_{key}_structure"].sum()),
                "five_minute_setup": int(frame[f"funnel_{key}_setup"].sum()),
                "raw_structural_candidates": int(frame[f"funnel_{key}_structural_candidate"].sum()),
                "aux_support_observed": int(frame[f"funnel_{key}_aux_support"].sum()),
            }
    return output


def selected_funnel_payload(
    selected: CombinationCandidate,
    all_candidates: dict[int, list[ExpertCandidate]],
    static_funnel: dict[str, dict[str, dict[str, int]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    lookup = {
        (candidate.policy.expert_id, candidate.policy.key): candidate
        for candidates in all_candidates.values()
        for candidate in candidates
    }
    output = json.loads(json.dumps(static_funnel, ensure_ascii=False))
    for month in DEVELOPMENT_MONTHS:
        for expert_id, policy in selected.policies.items():
            candidate = lookup[(expert_id, policy.key)]
            audit = candidate.monthly_thresholds.get(month, {})
            output[month][EXPERT_NAME[expert_id]].update({
                "model_available": bool(audit.get("model_available", False)),
                "training_candidates": int(audit.get("training_candidates", 0)),
                "calibration_month": str(audit.get("calibration_month", "unknown")),
                "calibration_candidates": int(audit.get("calibration_candidates", 0)),
                "calibration_base_rate": float(audit.get("calibration_base_rate", 0.0)),
                "calibration_brier": float(audit.get("calibration_brier", 0.0)),
                "probability_threshold": float(audit.get("probability_threshold", 1.0)),
                "online_rank_percentile": float(audit.get("quantile", 1.0)),
                "score_drift": float(audit.get("score_drift", 0.0)),
                "after_probability": int(audit.get("raw_eval_after_probability", 0)),
                "after_monthly_cap": int(audit.get("capped_eval_candidates", 0)),
                "after_cooldown": int(audit.get("after_cooldown", 0)),
                "selected_policy": True,
            })
    return output


def write_signal_funnel_csv(payload: dict[str, dict[str, dict[str, Any]]]) -> None:
    rows: list[dict[str, Any]] = []
    for month, expert_map in payload.items():
        for expert, values in expert_map.items():
            rows.append({"month": month, "expert": expert, **values})
    pd.DataFrame(rows).to_csv(RESULTS / "signal_funnel.csv", index=False)


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
        # A diagnostic fallback may have a poor score, but it must still be a
        # non-empty policy frozen entirely from development data.
        if not selected.policies:
            raise RuntimeError("Cannot evaluate June without a frozen non-empty development policy")
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
        ("setup_break_retest_short", "趋势破位首次反抽"),
        (f"setup_range_reversal_{suffix}", "流动性扫单反转"),
        ("setup_range_second_test_long", "区间二次测试收回"),
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
            "base_model_probability": float(trade.get("base_prob", trade["prob"])),
            "calibrated_probability": float(trade["prob"]),
            "online_score_percentile": float(trade["score"]),
            "setup_group": str(trade.get("setup_group", "unknown")),
            "structure_cycle": int(trade.get("structure_cycle", 0)),
            "quality_protection": str(trade.get("quality_protection", "passed")),
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
            "minimum_calibrated_probability": policy.min_probability,
            "minimum_online_rank_percentile": policy.min_percentile,
            "monthly_candidate_target": policy.monthly_target,
            "cooldown_bars": policy.cooldown,
            "quality_protection": {
                "max_trades_per_expert_per_day": MAX_TRADES_PER_EXPERT_DAY,
                "max_trades_per_setup_per_day": MAX_TRADES_PER_SETUP_DAY,
                "max_consecutive_losses_per_expert_day": MAX_CONSECUTIVE_LOSSES_EXPERT_DAY,
                "loss_reentry_min_bars": LOSS_REENTRY_MIN_BARS,
                "require_new_structure_cycle_after_loss": REQUIRE_NEW_STRUCTURE_CYCLE,
            },
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
                audit = candidate.monthly_thresholds.get(month, {})
                for key in (
                    "model_available", "training_candidates", "calibration_month",
                    "calibration_candidates", "calibration_base_rate", "calibration_brier",
                    "raw_structural_candidates", "probability_threshold", "quantile",
                    "score_drift", "raw_eval_after_probability",
                    "capped_eval_candidates", "after_cooldown",
                ):
                    row[f"{month}_funnel_{key}"] = audit.get(key)
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
    probabilities = np.linspace(0.25, 0.85, 120)
    months = np.array(["2025-11"] * 80 + ["2025-12"] * 40)
    bundle = {
        "idx": np.arange(120),
        "direction": -1,
        "expert_id": 1,
        "expert": "趋势空头",
        "labels": np.ones(120, dtype=np.int8),
        "exits": np.arange(120) + 1,
        "net_r": np.ones(120),
        "reasons": np.ones(120, dtype=np.int8),
        "base_prob": probabilities,
        "prob": probabilities,
        "months": months,
        "calibration_month": "2025-11",
        "calibration_scores": probabilities[:80],
        "calibration_labels": np.r_[np.zeros(40), np.ones(40)],
        "calibration_base_rate": 0.5,
        "calibration_brier": 0.20,
        "core_rows": 200,
        "calibration_rows": 80,
    }
    risk = RiskConfig(2.0, 1.2, 0.0035, 144, 1.0, 0.1, 48, 0.35)
    model = ModelConfig(2, 0.03, 100, 3.0, 20)
    loose = ExpertPolicy(1, risk, model, 12, 0.34, 0.65, 2)
    strict = ExpertPolicy(1, risk, model, 3, 0.70, 0.95, 9)
    loose_events, loose_meta = events_from_expert(bundle, loose, "2025-12", {"2025-09", "2025-10", "2025-11"})
    strict_events, strict_meta = events_from_expert(bundle, strict, "2025-12", {"2025-09", "2025-10", "2025-11"})
    if loose.key == strict.key:
        raise RuntimeError("Expert policy keys are not independent")
    if loose_meta["probability_threshold"] >= strict_meta["probability_threshold"]:
        raise RuntimeError("Independent calibrated probability floors were not honored")
    if len(loose_events) <= len(strict_events):
        raise RuntimeError("Independent online rank gates and monthly targets were not honored")
    # Online adaptation must not inspect future evaluation scores.
    first_half = {**bundle, "idx": bundle["idx"][:100], "labels": bundle["labels"][:100],
                  "exits": bundle["exits"][:100], "net_r": bundle["net_r"][:100],
                  "reasons": bundle["reasons"][:100], "base_prob": bundle["base_prob"][:100],
                  "prob": bundle["prob"][:100], "months": bundle["months"][:100]}
    early_events, _ = events_from_expert(first_half, loose, "2025-12", {"2025-09", "2025-10", "2025-11"})
    if [e["signal_i"] for e in early_events] != [e["signal_i"] for e in loose_events if e["signal_i"] < 100]:
        raise RuntimeError("Online score gate changed earlier decisions after seeing future candidates")


def quality_protection_test() -> None:
    policy = policy_grid_for_expert(1)[0]
    policies = {1: policy}

    def event(signal_i: int, exit_i: int, day: str, cycle: int, setup: str, net_r: float) -> dict[str, Any]:
        return {
            "signal_i": signal_i,
            "exit_i": exit_i,
            "direction": -1,
            "expert_id": 1,
            "expert": EXPERT_NAME[1],
            "base_prob": 0.70,
            "prob": 0.55,
            "score": 0.90,
            "net_r": net_r,
            "reason": 1 if net_r > 0 else 2,
            "policy_key": policy.key,
            "day": day,
            "structure_cycle": cycle,
            "setup_group": setup,
        }

    # A loss locks the same structure cycle even after the minimum bar wait;
    # a later newly activated structure cycle is allowed.
    locked = combine_events({1: [
        event(10, 11, "2026-01-01", 1, "A", -1.0),
        event(30, 31, "2026-01-01", 1, "B", 1.0),
        event(50, 51, "2026-01-01", 2, "B", 1.0),
    ]}, policies)
    if [e["signal_i"] for e in locked] != [10, 50]:
        raise RuntimeError("V9.5.1 structure-cycle loss lock failed")

    # The same concrete setup can trade at most once per expert per day.
    setup_capped = combine_events({1: [
        event(100, 101, "2026-01-02", 3, "A", 1.0),
        event(120, 121, "2026-01-02", 4, "A", 1.0),
    ]}, policies)
    if len(setup_capped) != 1:
        raise RuntimeError("V9.5.1 per-setup daily cap failed")

    # Different setups may trade, but the expert daily cap remains binding.
    expert_capped = combine_events({1: [
        event(200, 201, "2026-01-03", 5, "A", 1.0),
        event(220, 221, "2026-01-03", 6, "B", 1.0),
        event(240, 241, "2026-01-03", 7, "C", 1.0),
    ]}, policies)
    if len(expert_capped) != MAX_TRADES_PER_EXPERT_DAY:
        raise RuntimeError("V9.5.1 per-expert daily cap failed")


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
        raise RuntimeError("V9.5.1 self-test produced no feature rows")
    missing = [feature for feature in FEATURES if feature not in x.columns]
    if missing:
        raise RuntimeError(f"V9.5.1 self-test missing features: {missing}")

    # Missing auxiliary-data regression: sparse gaps and microsecond timestamps
    # must be handled with past-only filling and no backward-looking fill.
    eth_gap = eth.drop(index=[250, 251]).reset_index(drop=True)
    premium_gap = premium.drop(index=[100, 101, 102, 900]).reset_index(drop=True)
    premium_gap["open_time"] = premium_gap["open_time"].astype("int64") * 1000
    x_gap, gap_audit = add_features(base, eth_gap, premium_gap, funding)
    if x_gap.empty:
        raise RuntimeError("V9.5.1 missing-auxiliary regression produced no rows")
    if gap_audit["eth_exact_missing"] < 2 or gap_audit["premium_exact_missing"] < 4:
        raise RuntimeError("Missing-auxiliary regression did not exercise missing timestamps")
    if gap_audit["eth_missing_after_past_fill"] != 0 or gap_audit["premium_missing_after_past_fill"] != 0:
        raise RuntimeError("Past-only auxiliary alignment regression failed")

    experts = make_expert_data(x)
    if set(experts) != set(range(6)):
        raise RuntimeError("V9.5.1 expert registry mismatch")
    indices = np.arange(100, min(140, len(x) - 2), dtype=np.int64)
    compute_outcomes(
        indices, 1,
        x["open"].to_numpy(float), x["high"].to_numpy(float), x["low"].to_numpy(float),
        x["close"].to_numpy(float), x["atr"].to_numpy(float),
        2.0, 1.2, 0.004, 144, 1.0, 0.1, 48, 0.4, FEE_RATE, SLIPPAGE_ABS,
    )
    threshold_independence_test()
    quality_protection_test()
    funnel = static_signal_funnel(x, DEVELOPMENT_MONTHS)
    if not funnel or not all(EXPERT_NAME[i] in funnel[next(iter(funnel))] for i in range(6)):
        raise RuntimeError("V9.5.1 signal funnel registry failed")
    if build_combinations({}) != []:
        raise RuntimeError("V9.5.1 empty combination must not be treated as a valid policy")
    dummy = {i: [ExpertCandidate(policy=policy_grid_for_expert(i)[0], eligible=True)] for i in range(6)}
    core_options = candidate_options_by_expert(dummy)
    if any(core_options[i] for i in range(6) if i not in CORE_EXPERT_IDS):
        raise RuntimeError("V9.5.1 diagnostic-only expert leaked into combination selection")
    probe = ExpertCandidate(policy=policy_grid_for_expert(0)[0])
    for month in DEVELOPMENT_MONTHS:
        probe.monthly_metrics[month] = metrics([])
        probe.monthly_thresholds[month] = {"raw_structural_candidates": 0}
    if consecutive_zero_opportunity_months(probe) != 0:
        raise RuntimeError("V9.5.1 no-opportunity month was incorrectly treated as failure")
    if alignment_audit["eth_coverage_after_past_fill"] < 0.99:
        raise RuntimeError("Synthetic alignment unexpectedly failed")
    print(
        f"V9.5.1 self-test passed: rows={len(x)}, features={len(FEATURES)}, "
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


def prefer_complete_core_combinations(combinations: list[CombinationCandidate]) -> list[CombinationCandidate]:
    complete = [c for c in combinations if set(c.policies) == set(CORE_EXPERT_IDS)]
    return complete if complete else combinations


def stage_pass(metric: dict[str, float]) -> bool:
    return bool(
        STAGE_MIN_TRADES <= metric["trades"] <= STAGE_MAX_TRADES
        and metric["win_rate"] >= STAGE_MIN_WIN_RATE
        and metric["avg_win_loss_ratio"] >= STAGE_MIN_RATIO
        and metric["max_drawdown_R"] <= STAGE_MAX_DRAWDOWN
    )


def final_metric_pass(metric: dict[str, float]) -> bool:
    return bool(
        MIN_TRADES <= metric["trades"] <= MAX_TRADES
        and metric["win_rate"] >= MIN_WIN_RATE
        and metric["avg_win_loss_ratio"] >= MIN_RATIO
    )


def quality_protection_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "max_trades_same_day": 0,
            "max_trades_same_expert_day": 0,
            "max_trades_same_setup_day": 0,
            "expert_trade_share": {},
        }
    day_counts: dict[str, int] = {}
    expert_day: dict[tuple[int, str], int] = {}
    setup_day: dict[tuple[int, str, str], int] = {}
    expert_counts: dict[str, int] = {}
    for trade in trades:
        day = str(trade.get("day", "unknown"))
        expert_id = int(trade["expert_id"])
        setup = str(trade.get("setup_group", "unknown"))
        day_counts[day] = day_counts.get(day, 0) + 1
        expert_day[(expert_id, day)] = expert_day.get((expert_id, day), 0) + 1
        setup_day[(expert_id, day, setup)] = setup_day.get((expert_id, day, setup), 0) + 1
        name = EXPERT_NAME[expert_id]
        expert_counts[name] = expert_counts.get(name, 0) + 1
    total = len(trades)
    return {
        "max_trades_same_day": max(day_counts.values(), default=0),
        "max_trades_same_expert_day": max(expert_day.values(), default=0),
        "max_trades_same_setup_day": max(setup_day.values(), default=0),
        "expert_trade_share": {name: count / total for name, count in sorted(expert_counts.items())},
    }



# ---------------------------------------------------------------------------
# V9.5.1 architecture overrides: isolated specialists, router, micro confirmation,
# holdout meta filter, causal shadow accounts, deduplication and arbitration.
# ---------------------------------------------------------------------------

MODEL_FAMILY = {
    0: "趋势延续",
    1: "趋势延续",
    2: "区间流动性反转",
    3: "区间流动性反转",
    4: "波动扩张突破",
    5: "波动扩张突破",
}
MODEL_ROLE = {
    0: "趋势多头回踩与确认",
    1: "趋势空头反抽与确认",
    2: "区间扫低与二次测试多头",
    3: "区间扫高与假突破空头",
    4: "压缩后向上波动扩张",
    5: "压缩后向下波动扩张",
}

COMMON_FEATURES = [
    "ret1", "ret3", "ret6", "ret12", "ret24", "atr_pct", "atr_rank",
    "body", "close_loc", "upper_wick", "lower_wick", "range_exp",
    "rel_vol", "vol_z", "trade_z", "taker_ratio", "taker_z",
    "m15_trend", "m15_gap", "m15_adx", "m15_rsi", "m15_eff", "m15_atr_pct",
    "h1_trend", "h1_gap", "h1_adx", "h1_rsi", "h1_eff", "h1_atr_pct",
    "eth_ret3", "eth_ret12", "eth_gap", "eth_h1_trend", "btc_eth_corr",
    "premium", "premium_z", "premium_delta", "funding_rate", "funding_z",
    "derivative_pressure", "hour_sin", "hour_cos", "weekday",
]
TREND_FEATURES = COMMON_FEATURES + [
    "ema8_gap", "ema21_gap", "ema55_gap", "ema200_gap", "adx", "di_gap",
    "macd_hist_atr", "macd_slope_atr", "eff12", "eff24", "chop",
    "h4_trend", "h4_gap", "h4_adx", "h4_eff",
    "setup_reclaim_long", "setup_reclaim_short",
    "setup_continuation_long", "setup_continuation_short", "setup_break_retest_short",
]
RANGE_FEATURES = COMMON_FEATURES + [
    "rsi", "bb_pos", "bb_rank", "bb_width", "vwap_dev", "chop",
    "eff12", "eff24", "premium_abs_rank", "btc_eth_spread_12",
    "setup_range_reversal_long", "setup_range_reversal_short",
    "setup_range_second_test_long",
]
BREAKOUT_FEATURES = COMMON_FEATURES + [
    "bb_width", "bb_rank", "atr_rank", "range_exp", "rel_vol", "vol_z",
    "trade_z", "taker_z", "macd_slope_atr", "premium_delta",
    "btc_eth_spread_12", "btc_eth_spread_72",
    "setup_high_break_long", "setup_high_break_short",
]
FEATURES_BY_EXPERT = {
    0: tuple(dict.fromkeys(TREND_FEATURES)),
    1: tuple(dict.fromkeys(TREND_FEATURES)),
    2: tuple(dict.fromkeys(RANGE_FEATURES)),
    3: tuple(dict.fromkeys(RANGE_FEATURES)),
    4: tuple(dict.fromkeys(BREAKOUT_FEATURES)),
    5: tuple(dict.fromkeys(BREAKOUT_FEATURES)),
}


def features_for_expert(expert_id: int) -> tuple[str, ...]:
    return FEATURES_BY_EXPERT[int(expert_id)]


def router_confidence_array(x: pd.DataFrame, idx: np.ndarray, expert_id: int) -> np.ndarray:
    frame = x.iloc[idx]
    direction = int(EXPERT_BY_ID[expert_id]["direction"])
    h1_dir = np.clip(direction * frame["h1_trend"].to_numpy(float), -1.0, 1.0)
    m15_dir = np.clip(direction * frame["m15_trend"].to_numpy(float), -1.0, 1.0)
    if expert_id in (0, 1):
        raw = (
            0.28
            + 0.26 * frame["regime_trend"].to_numpy(float)
            + 0.16 * np.maximum(h1_dir, 0.0)
            + 0.14 * np.maximum(m15_dir, 0.0)
            + 0.08 * np.clip(frame["h1_adx"].to_numpy(float) / 35.0, 0.0, 1.0)
            + 0.08 * np.clip(frame["m15_adx"].to_numpy(float) / 35.0, 0.0, 1.0)
        )
    elif expert_id in (2, 3):
        raw = (
            0.30
            + 0.30 * frame["regime_range"].to_numpy(float)
            + 0.16 * np.clip(frame["chop"].to_numpy(float), 0.0, 1.0)
            + 0.12 * (1.0 - np.clip(frame["h1_adx"].to_numpy(float) / 35.0, 0.0, 1.0))
            + 0.12 * (1.0 - np.abs(m15_dir))
        )
    else:
        raw = (
            0.26
            + 0.30 * frame["regime_high_vol"].to_numpy(float)
            + 0.16 * frame["atr_rank"].to_numpy(float)
            + 0.12 * frame["bb_rank"].to_numpy(float)
            + 0.08 * np.clip(frame["rel_vol"].to_numpy(float) / 2.5, 0.0, 1.0)
            + 0.08 * np.maximum(m15_dir, 0.0)
        )
    return np.clip(raw, 0.01, 0.99)


def micro_confirmation_array(x: pd.DataFrame, idx: np.ndarray, expert_id: int) -> np.ndarray:
    frame = x.iloc[idx]
    direction = int(EXPERT_BY_ID[expert_id]["direction"])
    taker = direction * (frame["taker_ratio"].to_numpy(float) - 0.5) * 2.0
    premium = direction * np.tanh(frame["premium_z"].to_numpy(float) / 2.0)
    eth = direction * np.tanh(frame["eth_ret3"].to_numpy(float) * 250.0)
    flow = direction * np.tanh(frame["derivative_pressure"].to_numpy(float) / 2.0)
    volume = np.clip(frame["rel_vol"].to_numpy(float) / 2.0, 0.0, 1.0)
    raw = 0.42 + 0.16 * taker + 0.11 * premium + 0.11 * eth + 0.10 * flow + 0.10 * volume
    return np.clip(raw, 0.01, 0.99)


def meta_feature_matrix(
    calibrated_probability: np.ndarray,
    router_confidence: np.ndarray,
    micro_confirmation: np.ndarray,
    online_percentile: np.ndarray | None = None,
) -> np.ndarray:
    p = np.asarray(calibrated_probability, dtype=float)
    router = np.asarray(router_confidence, dtype=float)
    micro = np.asarray(micro_confirmation, dtype=float)
    if online_percentile is None:
        online = np.full(len(p), 0.5, dtype=float)
    else:
        online = np.asarray(online_percentile, dtype=float)
    return np.column_stack([
        p,
        router,
        micro,
        online,
        p * router,
        p * micro,
        router * micro,
    ])


def meta_probabilities(model: CalibratedExpertModel, matrix: np.ndarray) -> np.ndarray:
    meta_model = getattr(model, "meta_model", None)
    if meta_model is not None:
        raw = meta_model.predict_proba(matrix)[:, 1]
    else:
        raw = 0.55 * matrix[:, 0] + 0.25 * matrix[:, 1] + 0.20 * matrix[:, 2]
    base_rate = float(getattr(model, "calibration_base_rate", 0.5))
    return np.clip(0.90 * raw + 0.10 * base_rate, 0.01, 0.99)


def fit_one_expert_model(
    x: pd.DataFrame,
    expert_data: dict[int, dict[str, Any]],
    policy: ExpertPolicy,
    train_months: set[str],
) -> CalibratedExpertModel | None:
    assert_oos_isolation(train_months, f"specialist/meta training model={policy.expert_id}")
    data = expert_data[policy.expert_id]
    idx = data["idx"]
    open_, high, low, close, atr = (x[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr"))
    labels, _, net_r, _ = compute_outcomes(
        idx, int(data["direction"]), open_, high, low, close, atr,
        policy.risk.rr, policy.risk.sl_atr, policy.risk.min_stop_pct, policy.risk.max_hold,
        policy.risk.breakeven_trigger, policy.risk.breakeven_lock,
        policy.risk.early_bars, policy.risk.early_cut_r,
        FEE_RATE, SLIPPAGE_ABS,
    )
    months = x["month"].to_numpy()[idx]
    valid_train = np.isin(months, list(train_months)) & (labels >= 0)
    ordered_months = sorted(str(m) for m in train_months)
    if len(ordered_months) < 2:
        return None
    calibration_month = ordered_months[-1]
    core_months = tuple(ordered_months[:-1])
    core_mask = valid_train & np.isin(months, list(core_months))
    calibration_mask = valid_train & (months == calibration_month)
    y_core = labels[core_mask].astype(int)
    y_calibration = labels[calibration_mask].astype(int)
    base_min_rows = 45 if policy.expert_id in (0, 1) else 36
    base_min_class = 8 if policy.expert_id in (0, 1) else 6
    if not _enough_classes(y_core, base_min_rows, base_min_class) or not _enough_classes(
        y_calibration, CALIBRATION_MIN_ROWS, CALIBRATION_MIN_CLASS
    ):
        positions = np.flatnonzero(valid_train)
        if len(positions) < base_min_rows + CALIBRATION_MIN_ROWS:
            return None
        split = max(base_min_rows, int(len(positions) * 0.75))
        split = min(split, len(positions) - CALIBRATION_MIN_ROWS)
        core_positions = positions[:split]
        calibration_positions = positions[split:]
        core_mask = np.zeros(len(labels), dtype=bool)
        calibration_mask = np.zeros(len(labels), dtype=bool)
        core_mask[core_positions] = True
        calibration_mask[calibration_positions] = True
        y_core = labels[core_mask].astype(int)
        y_calibration = labels[calibration_mask].astype(int)
        if not _enough_classes(y_core, base_min_rows, base_min_class) or not _enough_classes(
            y_calibration, CALIBRATION_MIN_ROWS, CALIBRATION_MIN_CLASS
        ):
            return None
        calibration_month = "chronological_tail"
        core_months = tuple(sorted(set(str(m) for m in months[core_mask])))

    feature_names = features_for_expert(policy.expert_id)
    ordered_core = sorted(set(str(m) for m in months[core_mask]))
    recency = {m: 0.86 + 0.28 * (i / max(1, len(ordered_core) - 1)) for i, m in enumerate(ordered_core)}
    weights = np.array([recency.get(str(m), 1.0) for m in months[core_mask]], dtype=float)
    magnitude = 1.0 + 0.15 * np.minimum(np.abs(net_r[core_mask]), 2.0)
    base_model = HistGradientBoostingClassifier(
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
    base_model.fit(
        x.iloc[idx[core_mask]][list(feature_names)].to_numpy(np.float64),
        y_core,
        sample_weight=mild_class_weights(y_core) * weights * magnitude,
    )
    calibration_rows = idx[calibration_mask]
    base_cal = base_model.predict_proba(
        x.iloc[calibration_rows][list(feature_names)].to_numpy(np.float64)
    )[:, 1]
    calibrator: LogisticRegression | None = None
    if _enough_classes(y_calibration, CALIBRATION_MIN_ROWS, CALIBRATION_MIN_CLASS):
        calibrator = LogisticRegression(C=0.75, solver="lbfgs", max_iter=500, random_state=BASE_SEED + policy.expert_id * 211)
        calibrator.fit(_logit(base_cal), y_calibration)
    base_rate = float(np.mean(y_calibration))
    provisional = CalibratedExpertModel(
        base_model=base_model,
        calibrator=calibrator,
        calibration_month=calibration_month,
        core_months=core_months,
        calibration_scores=np.empty(0, dtype=float),
        calibration_labels=y_calibration,
        calibration_base_rate=base_rate,
        calibration_brier=0.0,
        core_rows=int(np.sum(core_mask)),
        calibration_rows=int(np.sum(calibration_mask)),
    )
    calibrated = calibrate_probabilities(provisional, base_cal)
    router = router_confidence_array(x, calibration_rows, policy.expert_id)
    micro = micro_confirmation_array(x, calibration_rows, policy.expert_id)
    # Calibration rows are chronological holdout rows. Meta fitting therefore
    # remains separated from specialist fitting and never sees the evaluation month.
    meta_x = meta_feature_matrix(calibrated, router, micro)
    meta_model: LogisticRegression | None = None
    if _enough_classes(y_calibration, CALIBRATION_MIN_ROWS, CALIBRATION_MIN_CLASS):
        meta_model = LogisticRegression(C=0.60, solver="lbfgs", max_iter=500, random_state=BASE_SEED + policy.expert_id * 307)
        meta_model.fit(meta_x, y_calibration)
    provisional.calibration_scores = calibrated
    provisional.calibration_brier = float(np.mean((calibrated - y_calibration) ** 2))
    provisional.meta_model = meta_model
    provisional.meta_feature_names = ["specialist_p", "router_p", "micro_p", "online_rank", "p_x_router", "p_x_micro", "router_x_micro"]
    provisional.feature_names = list(feature_names)
    meta_cal = meta_probabilities(provisional, meta_x)
    provisional.meta_calibration_scores = meta_cal
    provisional.meta_brier = float(np.mean((meta_cal - y_calibration) ** 2))
    return provisional


def build_one_expert_bundle(
    x: pd.DataFrame,
    expert_data: dict[int, dict[str, Any]],
    policy: ExpertPolicy,
    model: CalibratedExpertModel,
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
    feature_names = features_for_expert(policy.expert_id)
    base_probability = model.base_model.predict_proba(x.iloc[idx][list(feature_names)].to_numpy(np.float64))[:, 1]
    probability = calibrate_probabilities(model, base_probability)
    router = router_confidence_array(x, idx, policy.expert_id)
    micro = micro_confirmation_array(x, idx, policy.expert_id)
    meta = meta_probabilities(model, meta_feature_matrix(probability, router, micro))
    expert_key = str(data["key"])
    return {
        "idx": idx,
        "direction": int(data["direction"]),
        "expert_id": policy.expert_id,
        "expert": str(data["name"]),
        "family": MODEL_FAMILY[policy.expert_id],
        "role": MODEL_ROLE[policy.expert_id],
        "labels": labels,
        "exits": exits,
        "net_r": net_r,
        "reasons": reasons,
        "base_prob": base_probability,
        "prob": probability,
        "router": router,
        "micro": micro,
        "meta_prob": meta,
        "months": x["month"].to_numpy()[idx],
        "days": x.index.strftime("%Y-%m-%d").to_numpy()[idx],
        "structure_cycle": x[f"funnel_{expert_key}_structure_cycle"].to_numpy(np.int64)[idx],
        "setup_group": setup_group_array(x, idx, policy.expert_id),
        "calibration_month": model.calibration_month,
        "calibration_scores": model.calibration_scores,
        "calibration_labels": model.calibration_labels,
        "calibration_base_rate": model.calibration_base_rate,
        "calibration_brier": model.calibration_brier,
        "meta_brier": float(getattr(model, "meta_brier", 0.0)),
        "core_rows": model.core_rows,
        "calibration_rows": model.calibration_rows,
        "feature_count": len(feature_names),
    }


def events_from_expert(
    bundle: dict[str, Any],
    policy: ExpertPolicy,
    eval_month: str,
    train_months: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert_oos_isolation(train_months, f"threshold/meta evaluation model={policy.expert_id} eval={eval_month}")
    eval_positions = np.flatnonzero((bundle["months"] == eval_month) & (bundle["exits"] >= 0))
    calibration_scores = np.asarray(bundle.get("calibration_scores", []), dtype=float)
    calibration_scores = calibration_scores[np.isfinite(calibration_scores)]
    base_audit = {
        "model_available": True,
        "training_candidates": int(bundle.get("core_rows", 0)),
        "calibration_month": str(bundle.get("calibration_month", "unknown")),
        "calibration_candidates": int(len(calibration_scores)),
        "calibration_base_rate": float(bundle.get("calibration_base_rate", 0.0)),
        "calibration_brier": float(bundle.get("calibration_brier", 0.0)),
        "meta_brier": float(bundle.get("meta_brier", 0.0)),
        "feature_count": int(bundle.get("feature_count", 0)),
        "raw_structural_candidates": int(len(eval_positions)),
    }
    if len(calibration_scores) < ONLINE_RANK_MIN_HISTORY:
        return [], {
            **base_audit,
            "probability_threshold": float(policy.min_probability),
            "quantile": float(policy.min_percentile),
            "meta_probability_threshold": MIN_META_PROBABILITY,
            "expected_utility_threshold": MIN_EXPECTED_UTILITY_R,
            "raw_eval_after_probability": 0,
            "after_meta_filter": 0,
            "capped_eval_candidates": 0,
            "after_cooldown": 0,
            "score_drift": 0.0,
        }
    history = list(calibration_scores[-ONLINE_RANK_WINDOW:])
    events: list[dict[str, Any]] = []
    passed_specialist = 0
    passed_meta = 0
    eval_probabilities: list[float] = []
    for k in eval_positions:
        probability = float(bundle["prob"][k])
        base_probability = float(bundle["base_prob"][k])
        router = float(bundle["router"][k])
        micro = float(bundle["micro"][k])
        hist = np.asarray(history, dtype=float)
        percentile = float((np.sum(hist <= probability) + 1.0) / (len(hist) + 1.0))
        meta_matrix = meta_feature_matrix(
            np.array([probability]), np.array([router]), np.array([micro]), np.array([percentile])
        )
        # The bundle meta score was produced without the online percentile. Blend
        # the causal online rank at decision time without refitting on eval data.
        meta_probability = float(np.clip(0.85 * bundle["meta_prob"][k] + 0.15 * percentile, 0.01, 0.99))
        expected_utility = meta_probability * policy.risk.rr - (1.0 - meta_probability)
        eval_probabilities.append(probability)
        specialist_pass = probability >= policy.min_probability and percentile >= policy.min_percentile
        history.append(probability)
        if len(history) > ONLINE_RANK_WINDOW:
            del history[: len(history) - ONLINE_RANK_WINDOW]
        if not specialist_pass:
            continue
        passed_specialist += 1
        if meta_probability < MIN_META_PROBABILITY or expected_utility < MIN_EXPECTED_UTILITY_R:
            continue
        passed_meta += 1
        if len(events) >= policy.monthly_target:
            continue
        events.append({
            "signal_i": int(bundle["idx"][k]),
            "exit_i": int(bundle["exits"][k]),
            "direction": int(bundle["direction"]),
            "expert_id": int(bundle["expert_id"]),
            "expert": str(bundle["expert"]),
            "family": str(bundle["family"]),
            "role": str(bundle["role"]),
            "base_prob": base_probability,
            "prob": probability,
            "router_confidence": router,
            "micro_confirmation": micro,
            "meta_probability": meta_probability,
            "expected_utility": expected_utility,
            "score": percentile,
            "net_r": float(bundle["net_r"][k]),
            "reason": int(bundle["reasons"][k]),
            "policy_key": policy.key,
            "day": str(bundle["days"][k]),
            "structure_cycle": int(bundle["structure_cycle"][k]),
            "setup_group": str(bundle["setup_group"][k]),
        })
    events = sorted(events, key=lambda e: (e["signal_i"], -e["expected_utility"], -e["score"]))
    chosen = apply_expert_cooldown(events, policy.cooldown)
    eval_mean = float(np.mean(eval_probabilities)) if eval_probabilities else 0.0
    calibration_mean = float(np.mean(calibration_scores)) if len(calibration_scores) else 0.0
    return events, {
        **base_audit,
        "probability_threshold": float(policy.min_probability),
        "quantile": float(policy.min_percentile),
        "meta_probability_threshold": MIN_META_PROBABILITY,
        "expected_utility_threshold": MIN_EXPECTED_UTILITY_R,
        "raw_eval_after_probability": int(passed_specialist),
        "after_meta_filter": int(passed_meta),
        "capped_eval_candidates": int(len(events)),
        "after_cooldown": int(len(chosen)),
        "calibration_score_mean": calibration_mean,
        "eval_score_mean": eval_mean,
        "score_drift": float(eval_mean - calibration_mean),
    }


def _bounded_weights(raw: dict[int, float], active_ids: Iterable[int]) -> dict[int, float]:
    ids = list(active_ids)
    if not ids:
        return {}
    values = np.array([max(1e-6, raw.get(i, 1.0)) for i in ids], dtype=float)
    values /= values.sum()
    for _ in range(12):
        values = np.clip(values, DYNAMIC_WEIGHT_MIN, DYNAMIC_WEIGHT_MAX)
        values /= values.sum()
    return {i: float(v) for i, v in zip(ids, values)}


def causal_model_weights(history: dict[int, list[dict[str, Any]]], active_ids: Iterable[int]) -> dict[int, float]:
    raw: dict[int, float] = {}
    for expert_id in active_ids:
        events = sorted(history.get(expert_id, []), key=lambda e: int(e["signal_i"]))[-DYNAMIC_WEIGHT_WINDOW:]
        m = metrics(events)
        if not events:
            raw[expert_id] = 1.0
            continue
        pf = min(float(m["profit_factor"]), 4.0)
        net = float(m["net_R"])
        dd = float(m["max_drawdown_R"])
        wr = float(m["win_rate"])
        raw[expert_id] = max(
            0.15,
            1.0
            + 0.35 * math.tanh(net / 4.0)
            + 0.20 * math.tanh((pf - 1.0) / 1.5)
            + 0.20 * (wr - 0.5)
            - 0.12 * min(dd, 5.0),
        )
    return _bounded_weights(raw, active_ids)


def portfolio_arbitrate(
    events_by_expert: dict[int, list[dict[str, Any]]],
    policies: dict[int, ExpertPolicy],
    weights: dict[int, float],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    prefiltered: list[dict[str, Any]] = []
    for expert_id, events in events_by_expert.items():
        if expert_id in policies:
            prefiltered.extend(apply_expert_cooldown(events, policies[expert_id].cooldown))
    prefiltered.sort(key=lambda e: (int(e["signal_i"]), -float(e["expected_utility"])))
    selected: list[dict[str, Any]] = []
    dedup_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    global_last_exit = -10**9
    expert_day_count: dict[tuple[int, str], int] = {}
    setup_day_count: dict[tuple[int, str, str], int] = {}
    consecutive_losses: dict[tuple[int, str], int] = {}
    last_loss: dict[int, dict[str, Any]] = {}

    def allowed(event: dict[str, Any]) -> tuple[bool, str]:
        expert_id = int(event["expert_id"])
        day = str(event.get("day", "unknown"))
        setup = str(event.get("setup_group", "unknown"))
        if expert_day_count.get((expert_id, day), 0) >= MAX_TRADES_PER_EXPERT_DAY:
            return False, "model_daily_cap"
        if setup_day_count.get((expert_id, day, setup), 0) >= MAX_TRADES_PER_SETUP_DAY:
            return False, "setup_daily_cap"
        if consecutive_losses.get((expert_id, day), 0) >= MAX_CONSECUTIVE_LOSSES_EXPERT_DAY:
            return False, "daily_loss_lock"
        previous = last_loss.get(expert_id)
        if previous is not None and int(previous["direction"]) == int(event["direction"]):
            if int(event["signal_i"]) <= int(previous["exit_i"]) + LOSS_REENTRY_MIN_BARS:
                return False, "loss_cooldown"
            if REQUIRE_NEW_STRUCTURE_CYCLE and int(event.get("structure_cycle", 0)) <= int(previous["structure_cycle"]):
                return False, "same_structure_after_loss"
        return True, "passed"

    i = 0
    while i < len(prefiltered):
        cluster_start = int(prefiltered[i]["signal_i"])
        cluster: list[dict[str, Any]] = []
        while i < len(prefiltered) and int(prefiltered[i]["signal_i"]) <= cluster_start + DEDUP_WINDOW_BARS:
            cluster.append(prefiltered[i])
            i += 1
        if cluster_start <= global_last_exit:
            conflict_rows.append({
                "signal_i": cluster_start, "decision": "global_position_skip",
                "candidate_models": "|".join(sorted({str(e["expert"]) for e in cluster})),
            })
            continue
        allowed_cluster: list[dict[str, Any]] = []
        for event in cluster:
            ok, reason = allowed(event)
            meta_rows.append({
                "signal_i": int(event["signal_i"]), "expert": event["expert"],
                "family": event["family"], "direction": int(event["direction"]),
                "specialist_probability": float(event["prob"]),
                "meta_probability": float(event["meta_probability"]),
                "expected_utility": float(event["expected_utility"]),
                "router_confidence": float(event["router_confidence"]),
                "micro_confirmation": float(event["micro_confirmation"]),
                "model_weight": float(weights.get(int(event["expert_id"]), 0.0)),
                "quality_decision": reason,
            })
            if ok:
                allowed_cluster.append(event)
        if not allowed_cluster:
            continue

        direction_candidates: dict[int, list[dict[str, Any]]] = {1: [], -1: []}
        for event in allowed_cluster:
            direction_candidates[int(event["direction"])].append(event)

        aggregated: dict[int, dict[str, Any]] = {}
        for direction, events in direction_candidates.items():
            if not events:
                continue
            families = sorted({str(e["family"]) for e in events})
            source_models = sorted({str(e["expert"]) for e in events})
            primary = max(
                events,
                key=lambda e: (
                    float(e["expected_utility"]) * float(weights.get(int(e["expert_id"]), 0.0)),
                    float(e["meta_probability"]), float(e["score"]),
                ),
            )
            agreement = len(families)
            weighted_utility = sum(
                float(e["expected_utility"]) * float(weights.get(int(e["expert_id"]), 0.0))
                for e in events
            )
            weighted_utility *= 1.0 + 0.08 * max(0, agreement - 1)
            aggregated[direction] = {
                "primary": primary,
                "utility": weighted_utility,
                "source_models": source_models,
                "source_families": families,
                "agreement_count": agreement,
                "candidate_count": len(events),
            }
            dedup_rows.append({
                "signal_i": cluster_start,
                "direction": "LONG" if direction > 0 else "SHORT",
                "candidate_count": len(events),
                "independent_family_count": agreement,
                "source_models": "|".join(source_models),
                "chosen_primary": str(primary["expert"]),
                "deduplicated_count": max(0, len(events) - 1),
            })

        chosen_side: dict[str, Any] | None = None
        if 1 in aggregated and -1 in aggregated:
            long_u = float(aggregated[1]["utility"])
            short_u = float(aggregated[-1]["utility"])
            margin = abs(long_u - short_u)
            if margin < DIRECTION_CONFLICT_MARGIN:
                conflict_rows.append({
                    "signal_i": cluster_start,
                    "decision": "direction_conflict_skip",
                    "long_utility": long_u,
                    "short_utility": short_u,
                    "utility_margin": margin,
                    "candidate_models": "|".join(sorted({str(e["expert"]) for e in allowed_cluster})),
                })
                continue
            chosen_side = aggregated[1] if long_u > short_u else aggregated[-1]
            conflict_rows.append({
                "signal_i": cluster_start,
                "decision": "direction_conflict_resolved",
                "long_utility": long_u,
                "short_utility": short_u,
                "utility_margin": margin,
                "chosen_direction": "LONG" if long_u > short_u else "SHORT",
                "candidate_models": "|".join(sorted({str(e["expert"]) for e in allowed_cluster})),
            })
        elif aggregated:
            chosen_side = next(iter(aggregated.values()))
        if chosen_side is None:
            continue

        chosen = dict(chosen_side["primary"])
        chosen["source_models"] = list(chosen_side["source_models"])
        chosen["source_families"] = list(chosen_side["source_families"])
        chosen["agreement_count"] = int(chosen_side["agreement_count"])
        chosen["portfolio_utility"] = float(chosen_side["utility"])
        chosen["model_weight"] = float(weights.get(int(chosen["expert_id"]), 0.0))
        chosen["quality_protection"] = "passed"
        selected.append(chosen)
        global_last_exit = int(chosen["exit_i"])
        expert_id = int(chosen["expert_id"])
        day = str(chosen["day"])
        setup = str(chosen["setup_group"])
        expert_day_count[(expert_id, day)] = expert_day_count.get((expert_id, day), 0) + 1
        setup_day_count[(expert_id, day, setup)] = setup_day_count.get((expert_id, day, setup), 0) + 1
        if float(chosen["net_r"]) <= 0.0:
            consecutive_losses[(expert_id, day)] = consecutive_losses.get((expert_id, day), 0) + 1
            last_loss[expert_id] = {
                "exit_i": int(chosen["exit_i"]),
                "structure_cycle": int(chosen.get("structure_cycle", 0)),
                "direction": int(chosen["direction"]),
            }
        else:
            consecutive_losses[(expert_id, day)] = 0
            last_loss.pop(expert_id, None)
    return selected, {"dedup": dedup_rows, "conflicts": conflict_rows, "meta": meta_rows}


def combine_events(
    events_by_expert: dict[int, list[dict[str, Any]]],
    policies: dict[int, ExpertPolicy],
    min_direction_margin: float = 0.025,
) -> list[dict[str, Any]]:
    weights = _bounded_weights({i: 1.0 for i in policies}, policies)
    return portfolio_arbitrate(events_by_expert, policies, weights)[0]


def active_model_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for trade in trades:
        name = str(trade.get("expert", "unknown"))
        family = str(trade.get("family", MODEL_FAMILY.get(int(trade.get("expert_id", -1)), "unknown")))
        counts[name] = counts.get(name, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
    total = len(trades)
    return {
        "active_models": sum(v > 0 for v in counts.values()),
        "active_families": sum(v > 0 for v in family_counts.values()),
        "max_single_model_trade_share": max(counts.values(), default=0) / total if total else 0.0,
        "model_trade_share": {k: v / total for k, v in sorted(counts.items())} if total else {},
        "family_trade_share": {k: v / total for k, v in sorted(family_counts.items())} if total else {},
    }


def stage_pass(metric: dict[str, float], trades: list[dict[str, Any]] | None = None) -> bool:
    diversity = active_model_summary(trades or [])
    return bool(
        STAGE_MIN_TRADES <= int(metric["trades"]) <= STAGE_MAX_TRADES
        and float(metric["win_rate"]) >= STAGE_MIN_WIN_RATE
        and float(metric["avg_win_loss_ratio"]) >= STAGE_MIN_RATIO
        and float(metric["profit_factor"]) >= STAGE_MIN_PROFIT_FACTOR
        and float(metric["max_drawdown_R"]) <= STAGE_MAX_DRAWDOWN
        and diversity["active_models"] >= STAGE_MIN_ACTIVE_MODELS
        and diversity["max_single_model_trade_share"] <= STAGE_MAX_MODEL_SHARE
    )


def final_metric_pass(metric: dict[str, float]) -> bool:
    return bool(
        MIN_TRADES <= int(metric["trades"]) <= MAX_TRADES
        and float(metric["win_rate"]) >= MIN_WIN_RATE
        and float(metric["avg_win_loss_ratio"]) >= MIN_RATIO
    )


def combination_score(candidate: CombinationCandidate) -> float:
    stats = [candidate.monthly_metrics[m] for m in DEVELOPMENT_MONTHS]
    meaningful = [m for m in stats if int(m["trades"]) >= 3]
    if len(meaningful) < 2:
        return -1e12
    shadow = getattr(candidate, "monthly_shadow_trades", {})
    all_portfolio = [t for month in DEVELOPMENT_MONTHS for t in candidate.monthly_trades.get(month, [])]
    diversity = active_model_summary(all_portfolio)
    wrs = [shrunk_win_rate(m) for m in meaningful]
    pfs = [float(m["profit_factor"]) for m in meaningful if float(m["profit_factor"]) > 0]
    ratios = [float(m["avg_win_loss_ratio"]) for m in meaningful if float(m["avg_win_loss_ratio"]) > 0]
    net = np.array([float(m["net_R"]) for m in stats], dtype=float)
    count = np.array([int(m["trades"]) for m in stats], dtype=float)
    may_metric = candidate.monthly_metrics[MONTHS[8]]
    may_trades = candidate.monthly_trades.get(MONTHS[8], [])
    may_penalty = 0.0 if stage_pass(may_metric, may_trades) else 4500.0
    shadow_total = {
        expert_id: metrics([t for month in DEVELOPMENT_MONTHS for t in shadow.get(month, {}).get(expert_id, [])])
        for expert_id in candidate.policies
    }
    profitable_shadow_models = sum(float(m["net_R"]) > 0 for m in shadow_total.values())
    return float(
        min(wrs) * 9000
        + float(np.median(wrs)) * 3500
        + min(min(pfs) if pfs else 0.0, 5.0) * 1700
        + min(min(ratios) if ratios else 0.0, 4.0) * 900
        + sum(net) * 30
        + diversity["active_models"] * 500
        + diversity["active_families"] * 700
        + profitable_shadow_models * 350
        - np.std(net) * 420
        - np.sum(net < 0) * 700
        - max(0.0, diversity["max_single_model_trade_share"] - STAGE_MAX_MODEL_SHARE) * 7000
        - np.mean(np.abs(count - np.clip(count, STAGE_MIN_TRADES, STAGE_MAX_TRADES))) * 180
        - may_penalty
    )


def candidate_options_by_expert(all_candidates: dict[int, list[ExpertCandidate]]) -> dict[int, list[ExpertCandidate]]:
    return {
        expert_id: [c for c in candidates if c.eligible][:TOP_PER_EXPERT]
        for expert_id, candidates in all_candidates.items()
    }


def diagnostic_options_by_expert(all_candidates: dict[int, list[ExpertCandidate]]) -> dict[int, list[ExpertCandidate]]:
    result: dict[int, list[ExpertCandidate]] = {}
    for expert_id, candidates in all_candidates.items():
        usable = [
            c for c in candidates
            if sum(raw_structural_candidates(c, m) for m in DEVELOPMENT_MONTHS) > 0
            and sum(int(c.monthly_metrics.get(m, {}).get("trades", 0)) for m in DEVELOPMENT_MONTHS) > 0
        ]
        result[expert_id] = usable[:DIAGNOSTIC_TOP_PER_EXPERT]
    return result


def build_combinations(options: dict[int, list[ExpertCandidate]]) -> list[CombinationCandidate]:
    active_ids = [i for i in range(6) if options.get(i)]
    if not active_ids:
        return []
    beam: list[dict[int, ExpertPolicy]] = [{}]
    for expert_id in active_ids:
        expanded: list[dict[int, ExpertPolicy]] = []
        for current in beam:
            expanded.append(dict(current))
            for candidate in options[expert_id]:
                item = dict(current)
                item[expert_id] = candidate.policy
                expanded.append(item)
        expanded.sort(
            key=lambda policies: sum(
                next(c.aggregate_score for c in options[eid] if c.policy.key == policy.key)
                for eid, policy in policies.items()
            ),
            reverse=True,
        )
        beam = expanded[:BEAM_WIDTH]
    combos: list[CombinationCandidate] = []
    for policies in beam:
        families = {MODEL_FAMILY[i] for i in policies}
        if len(policies) >= 3 and len(families) >= 2:
            combos.append(CombinationCandidate(policies=policies))
    return combos


def _history_before_month(
    selected: CombinationCandidate,
    all_candidates: dict[int, list[ExpertCandidate]],
    month: str,
    include_month: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    lookup = {
        (c.policy.expert_id, c.policy.key): c
        for candidates in all_candidates.values() for c in candidates
    }
    result = {i: [] for i in selected.policies}
    for m in DEVELOPMENT_MONTHS:
        if m == month and not include_month:
            break
        for expert_id, policy in selected.policies.items():
            result[expert_id].extend(lookup[(expert_id, policy.key)].monthly_events.get(m, []))
        if m == month:
            break
    return result


def evaluate_development_combinations(
    combinations: list[CombinationCandidate],
    all_candidates: dict[int, list[ExpertCandidate]],
) -> list[CombinationCandidate]:
    lookup = {
        (c.policy.expert_id, c.policy.key): c
        for candidates in all_candidates.values() for c in candidates
    }
    for combo in combinations:
        history: dict[int, list[dict[str, Any]]] = {i: [] for i in combo.policies}
        combo.monthly_shadow_trades = {}
        combo.monthly_weights = {}
        combo.monthly_arbitration_audit = {}
        for month in DEVELOPMENT_MONTHS:
            events_by_expert = {
                expert_id: lookup[(expert_id, policy.key)].monthly_events.get(month, [])
                for expert_id, policy in combo.policies.items()
            }
            shadow = {
                expert_id: apply_expert_cooldown(events, combo.policies[expert_id].cooldown)
                for expert_id, events in events_by_expert.items()
            }
            weights = causal_model_weights(history, combo.policies)
            selected, audits = portfolio_arbitrate(events_by_expert, combo.policies, weights)
            combo.monthly_shadow_trades[month] = shadow
            combo.monthly_weights[month] = weights
            combo.monthly_arbitration_audit[month] = audits
            combo.monthly_trades[month] = selected
            combo.monthly_metrics[month] = metrics(selected)
            combo.monthly_expert_metrics[month] = expert_breakdown(selected)
            for expert_id, events in shadow.items():
                history[expert_id].extend(events)
        combo.aggregate_score = combination_score(combo)
    combinations.sort(key=lambda c: c.aggregate_score, reverse=True)
    return combinations


def evaluate_selected_month(
    x: pd.DataFrame,
    expert_data: dict[int, dict[str, Any]],
    selected: CombinationCandidate,
    eval_month: str,
    train_months: set[str],
    shadow_history: dict[int, list[dict[str, Any]]] | None = None,
) -> tuple[
    dict[str, float], list[dict[str, Any]], dict[str, dict[str, float]], dict[str, Any],
    dict[int, list[dict[str, Any]]], dict[int, float], dict[str, list[dict[str, Any]]],
]:
    if not selected.policies:
        assert_oos_isolation(train_months, f"empty safe-degradation evaluation for {eval_month}")
        return metrics([]), [], {}, {}, {}, {}, {"conflicts": [], "dedup": [], "meta": []}
    assert_oos_isolation(train_months, f"final multi-model evaluation training for {eval_month}")
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
    shadow = {
        expert_id: apply_expert_cooldown(events, selected.policies[expert_id].cooldown)
        for expert_id, events in events_by_expert.items()
    }
    weights = causal_model_weights(shadow_history or {i: [] for i in selected.policies}, selected.policies)
    portfolio, audits = portfolio_arbitrate(events_by_expert, selected.policies, weights)
    return metrics(portfolio), portfolio, expert_breakdown(portfolio), thresholds, shadow, weights, audits


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
        risk_abs = max(float(x.iloc[signal_i]["atr"]) * policy.risk.sl_atr, float(x.iloc[signal_i]["close"]) * policy.risk.min_stop_pct)
        exit_i = int(trade["exit_i"])
        reason = {1: "TP", 2: "PROTECTED_STOP", 3: "TIME", 4: "EARLY_CUT"}.get(int(trade["reason"]), "UNKNOWN")
        rows.append({
            "signal_time_utc": x.index[signal_i].isoformat(),
            "entry_time_utc": x.index[entry_i].isoformat(),
            "exit_time_utc": x.index[exit_i].isoformat(),
            "month": month,
            "direction": "LONG" if direction > 0 else "SHORT",
            "primary_model": EXPERT_NAME[expert_id],
            "model_family": MODEL_FAMILY[expert_id],
            "model_role": MODEL_ROLE[expert_id],
            "source_models": "|".join(trade.get("source_models", [EXPERT_NAME[expert_id]])),
            "source_families": "|".join(trade.get("source_families", [MODEL_FAMILY[expert_id]])),
            "independent_agreement_count": int(trade.get("agreement_count", 1)),
            "setup": setup_name(x, signal_i, direction),
            "setup_group": str(trade.get("setup_group", "unknown")),
            "policy_key": policy.key,
            "base_model_probability": float(trade.get("base_prob", trade["prob"])),
            "calibrated_probability": float(trade["prob"]),
            "online_score_percentile": float(trade["score"]),
            "router_confidence": float(trade.get("router_confidence", 0.0)),
            "micro_confirmation": float(trade.get("micro_confirmation", 0.0)),
            "meta_probability": float(trade.get("meta_probability", 0.0)),
            "expected_utility_R": float(trade.get("expected_utility", 0.0)),
            "portfolio_utility": float(trade.get("portfolio_utility", 0.0)),
            "model_weight": float(trade.get("model_weight", 0.0)),
            "structure_cycle": int(trade.get("structure_cycle", 0)),
            "quality_protection": str(trade.get("quality_protection", "passed")),
            "entry": entry,
            "risk_abs": risk_abs,
            "target_R": policy.risk.rr,
            "net_R": float(trade["net_r"]),
            "win": bool(trade["net_r"] > 0),
            "exit_reason": reason,
            "bars": exit_i - entry_i,
        })
    return pd.DataFrame(rows)


def shadow_trade_frame(
    x: pd.DataFrame,
    shadow: dict[int, list[dict[str, Any]]],
    policies: dict[int, ExpertPolicy],
    month: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for expert_id, events in shadow.items():
        frame = detailed_trades(x, events, policies, month)
        if not frame.empty:
            frame["shadow_account"] = EXPERT_NAME[expert_id]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def model_monthly_rows(shadow_by_month: dict[str, dict[int, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for month, mapping in shadow_by_month.items():
        for expert_id in sorted(mapping):
            rows.append({
                "month": month,
                "model_id": expert_id,
                "model": EXPERT_NAME[expert_id],
                "family": MODEL_FAMILY[expert_id],
                **metrics(mapping[expert_id]),
            })
    return rows


def model_correlation_rows(shadow_by_month: dict[str, dict[int, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    events_by_model = {
        i: [e for mapping in shadow_by_month.values() for e in mapping.get(i, [])]
        for i in range(6)
    }
    rows: list[dict[str, Any]] = []
    for a, b in itertools.combinations(range(6), 2):
        ea, eb = events_by_model[a], events_by_model[b]
        sa = {int(e["signal_i"]): float(e["net_r"]) for e in ea}
        sb = {int(e["signal_i"]): float(e["net_r"]) for e in eb}
        overlap = sorted(set(sa) & set(sb))
        union = set(sa) | set(sb)
        if len(overlap) >= 2:
            corr = float(np.corrcoef([sa[i] for i in overlap], [sb[i] for i in overlap])[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        else:
            corr = 0.0
        rows.append({
            "model_a": EXPERT_NAME[a], "model_b": EXPERT_NAME[b],
            "family_a": MODEL_FAMILY[a], "family_b": MODEL_FAMILY[b],
            "signal_overlap_count": len(overlap),
            "signal_union_count": len(union),
            "signal_jaccard": len(overlap) / len(union) if union else 0.0,
            "overlap_outcome_correlation": corr,
        })
    return rows


def selected_policy_payload(selected: CombinationCandidate) -> dict[str, Any]:
    return {
        EXPERT_NAME[expert_id]: {
            "model_id": expert_id,
            "family": MODEL_FAMILY[expert_id],
            "role": MODEL_ROLE[expert_id],
            "policy_key": policy.key,
            "feature_count": len(features_for_expert(expert_id)),
            "minimum_calibrated_probability": policy.min_probability,
            "minimum_online_rank_percentile": policy.min_percentile,
            "minimum_meta_probability": MIN_META_PROBABILITY,
            "minimum_expected_utility_R": MIN_EXPECTED_UTILITY_R,
            "monthly_candidate_target": policy.monthly_target,
            "cooldown_bars": policy.cooldown,
            "risk": asdict(policy.risk),
            "model": asdict(policy.model),
        }
        for expert_id, policy in sorted(selected.policies.items())
    }


def prefer_complete_core_combinations(combinations: list[CombinationCandidate]) -> list[CombinationCandidate]:
    # V9.5.1 does not require all six models. It prioritizes true family diversity
    # and avoids a portfolio that is merely multiple variants of one regime.
    return sorted(
        combinations,
        key=lambda c: (
            len({MODEL_FAMILY[i] for i in c.policies}),
            len(c.policies),
            c.aggregate_score,
        ),
        reverse=True,
    )


def quality_protection_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    base = active_model_summary(trades)
    if not trades:
        return {
            "max_trades_same_day": 0,
            "max_trades_same_expert_day": 0,
            "max_trades_same_setup_day": 0,
            **base,
        }
    day_counts: dict[str, int] = {}
    expert_day: dict[tuple[int, str], int] = {}
    setup_day: dict[tuple[int, str, str], int] = {}
    for trade in trades:
        day = str(trade.get("day", "unknown"))
        expert_id = int(trade["expert_id"])
        setup = str(trade.get("setup_group", "unknown"))
        day_counts[day] = day_counts.get(day, 0) + 1
        expert_day[(expert_id, day)] = expert_day.get((expert_id, day), 0) + 1
        setup_day[(expert_id, day, setup)] = setup_day.get((expert_id, day, setup), 0) + 1
    return {
        "max_trades_same_day": max(day_counts.values(), default=0),
        "max_trades_same_expert_day": max(expert_day.values(), default=0),
        "max_trades_same_setup_day": max(setup_day.values(), default=0),
        **base,
    }


def self_test() -> None:
    try:
        assert_oos_isolation({"2026-05", "2026-06"}, "intentional self-test")
        raise AssertionError("OOS leakage guard did not trigger")
    except RuntimeError:
        pass
    raw, eth, premium, funding = synthetic_inputs(8500)
    x, _ = add_features(raw, eth, premium, funding)
    assert len(x) > 1000
    for expert_id in range(6):
        idx = np.arange(min(250, len(x)), dtype=np.int64)
        router = router_confidence_array(x, idx, expert_id)
        micro = micro_confirmation_array(x, idx, expert_id)
        assert np.all((router >= 0.0) & (router <= 1.0))
        assert np.all((micro >= 0.0) & (micro <= 1.0))
        assert len(features_for_expert(expert_id)) >= 35
    policies = {i: policy_grid_for_expert(i)[0] for i in (0, 2, 5)}
    mock: dict[int, list[dict[str, Any]]] = {}
    for i, direction in ((0, 1), (2, 1), (5, -1)):
        mock[i] = [{
            "signal_i": 100, "exit_i": 104, "direction": direction,
            "expert_id": i, "expert": EXPERT_NAME[i], "family": MODEL_FAMILY[i], "role": MODEL_ROLE[i],
            "prob": 0.62, "base_prob": 0.60, "score": 0.91,
            "router_confidence": 0.78, "micro_confirmation": 0.70,
            "meta_probability": 0.64, "expected_utility": 0.64 * policies[i].risk.rr - 0.36,
            "net_r": 1.2 if direction > 0 else -0.4, "reason": 1 if direction > 0 else 2,
            "policy_key": policies[i].key, "day": "2026-05-01",
            "structure_cycle": 10 + i, "setup_group": f"setup-{i}",
        }]
    weights = _bounded_weights({i: 1.0 for i in policies}, policies)
    trades, audit = portfolio_arbitrate(mock, policies, weights)
    assert len(trades) == 1
    assert trades[0]["direction"] == 1
    assert trades[0]["agreement_count"] == 2
    assert audit["dedup"] and audit["conflicts"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9

    def mock_candidate(expert_id: int, trades: int, net_r: float) -> ExpertCandidate:
        candidate = ExpertCandidate(policy=policy_grid_for_expert(expert_id)[0])
        candidate.aggregate_score = net_r + trades * 0.1
        candidate.monthly_metrics = {m: metrics([]) for m in DEVELOPMENT_MONTHS}
        candidate.monthly_events = {m: [] for m in DEVELOPMENT_MONTHS}
        if trades > 0:
            month = DEVELOPMENT_MONTHS[-1]
            candidate.monthly_metrics[month] = {**metrics([]), "trades": trades, "net_R": net_r}
            candidate.monthly_events[month] = [{"net_r": net_r / trades}] * trades
        return candidate

    zero_map = {i: [mock_candidate(i, 0, 0.0)] for i in range(6)}
    zero_combo, zero_status = _fallback_multi_model_policy(zero_map)
    assert zero_status == "ZERO_EXECUTABLE_MODELS" and not zero_combo.policies

    two_map = {i: [mock_candidate(i, 0, 0.0)] for i in range(6)}
    two_map[1] = [mock_candidate(1, 2, 1.0)]
    two_map[2] = [mock_candidate(2, 3, 1.5)]
    two_combo, two_status = _fallback_multi_model_policy(two_map)
    assert two_status == "INSUFFICIENT_MODEL_DIVERSITY" and len(two_combo.policies) == 2

    three_map = {i: [mock_candidate(i, 0, 0.0)] for i in range(6)}
    three_map[0] = [mock_candidate(0, 2, 0.8)]
    three_map[2] = [mock_candidate(2, 3, 1.2)]
    three_map[4] = [mock_candidate(4, 2, 0.7)]
    three_combo, three_status = _fallback_multi_model_policy(three_map)
    assert three_status == "EXPERIMENTAL_FALLBACK" and len(three_combo.policies) == 3
    audit_rows = model_availability_audit_rows(two_map, two_combo, two_status)
    assert len(audit_rows) == 6
    assert sum(bool(row["fallback_selected"]) for row in audit_rows) == 2
    print("V951_SELF_TEST_OK")


def _development_trade_count(candidate: ExpertCandidate) -> int:
    return sum(int(candidate.monthly_metrics.get(m, {}).get("trades", 0)) for m in DEVELOPMENT_MONTHS)


def _development_active_months(candidate: ExpertCandidate) -> int:
    return sum(int(candidate.monthly_metrics.get(m, {}).get("trades", 0)) > 0 for m in DEVELOPMENT_MONTHS)


def _development_positive_months(candidate: ExpertCandidate) -> int:
    return sum(float(candidate.monthly_metrics.get(m, {}).get("net_R", 0.0)) > 0.0 for m in DEVELOPMENT_MONTHS)


def _development_net_r(candidate: ExpertCandidate) -> float:
    return sum(float(candidate.monthly_metrics.get(m, {}).get("net_R", 0.0)) for m in DEVELOPMENT_MONTHS)


def _fallback_rank(candidate: ExpertCandidate) -> tuple[int, int, float, float]:
    return (
        _development_active_months(candidate),
        _development_trade_count(candidate),
        _development_net_r(candidate),
        float(candidate.aggregate_score),
    )


def _fallback_multi_model_policy(
    all_candidates: dict[int, list[ExpertCandidate]],
) -> tuple[CombinationCandidate, str]:
    """Select every genuinely executable development model without crashing.

    V9.5.1 does not relax any probability, rank, Meta, diversity or final
    qualification threshold. It only changes failure handling: fewer than three
    executable models are evaluated as an explicitly insufficient diagnostic
    portfolio; zero executable models produce an empty frozen policy and full
    diagnostics instead of raising an exception.
    """
    usable_by_expert: dict[int, list[ExpertCandidate]] = {}
    for expert_id, candidates in all_candidates.items():
        usable = [c for c in candidates if _development_trade_count(c) > 0]
        usable.sort(key=_fallback_rank, reverse=True)
        usable_by_expert[expert_id] = usable

    chosen: dict[int, ExpertPolicy] = {}
    # First preserve true family diversity using the complete candidate search,
    # not only the first 10/20 leaderboard rows.
    for family in ("趋势延续", "区间流动性反转", "波动扩张突破"):
        family_candidates = [
            candidate
            for expert_id, candidates in usable_by_expert.items()
            if MODEL_FAMILY[expert_id] == family
            for candidate in candidates
        ]
        if family_candidates:
            best = max(family_candidates, key=_fallback_rank)
            chosen[best.policy.expert_id] = best.policy

    # Then include the best remaining executable directional specialists.
    remaining = [
        candidates[0]
        for expert_id, candidates in usable_by_expert.items()
        if candidates and expert_id not in chosen
    ]
    remaining.sort(key=_fallback_rank, reverse=True)
    for candidate in remaining:
        chosen[candidate.policy.expert_id] = candidate.policy

    if not chosen:
        return CombinationCandidate(policies={}), "ZERO_EXECUTABLE_MODELS"

    families = {MODEL_FAMILY[expert_id] for expert_id in chosen}
    if len(chosen) < STAGE_MIN_ACTIVE_MODELS or len(families) < 2:
        return CombinationCandidate(policies=chosen), "INSUFFICIENT_MODEL_DIVERSITY"
    return CombinationCandidate(policies=chosen), "EXPERIMENTAL_FALLBACK"


def model_availability_audit_rows(
    all_candidates: dict[int, list[ExpertCandidate]],
    selected: CombinationCandidate,
    selection_status: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_keys = {expert_id: policy.key for expert_id, policy in selected.policies.items()}
    for expert_id in range(6):
        candidates = all_candidates.get(expert_id, [])
        executable = [c for c in candidates if _development_trade_count(c) > 0]
        executable.sort(key=_fallback_rank, reverse=True)
        best = executable[0] if executable else (candidates[0] if candidates else None)
        selected_key = selected_keys.get(expert_id)
        selected_candidate = next((c for c in candidates if c.policy.key == selected_key), None)
        selected_development_executable = bool(selected_candidate and _development_trade_count(selected_candidate) > 0)
        if selected_key is not None:
            rejection_reason = "selected_executable_safe_fallback" if selected_development_executable else "selected_diagnostic_probe"
        elif not executable:
            rejection_reason = "no_policy_produced_development_trades"
        else:
            rejection_reason = "not_selected_by_family_and_stability_ranking"
        raw_candidates = 0
        if best is not None:
            raw_candidates = sum(
                int(best.monthly_thresholds.get(month, {}).get("raw_structural_candidates", 0))
                for month in DEVELOPMENT_MONTHS
            )
        rows.append({
            "model_id": expert_id,
            "model": EXPERT_NAME[expert_id],
            "family": MODEL_FAMILY[expert_id],
            "role": MODEL_ROLE[expert_id],
            "searched_policies": len(candidates),
            "eligible_policies": sum(bool(c.eligible) for c in candidates),
            "policies_with_development_trades": len(executable),
            "raw_structural_candidates_best_policy": raw_candidates,
            "best_policy_key": best.policy.key if best is not None else "",
            "best_total_trades": _development_trade_count(best) if best is not None else 0,
            "best_active_months": _development_active_months(best) if best is not None else 0,
            "best_positive_months": _development_positive_months(best) if best is not None else 0,
            "best_net_R": _development_net_r(best) if best is not None else 0.0,
            "best_aggregate_score": float(best.aggregate_score) if best is not None else -1e12,
            "fallback_selected": selected_key is not None,
            "selected_policy_key": selected_key or "",
            "selected_development_executable": selected_development_executable,
            "selection_status": selection_status,
            "rejection_reason": rejection_reason,
        })
    return rows


def _flatten_audit(audits_by_month: dict[str, dict[str, list[dict[str, Any]]]], key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, audit in audits_by_month.items():
        for item in audit.get(key, []):
            rows.append({"month": month, **item})
    return pd.DataFrame(rows)


def _router_report(x: pd.DataFrame, months: Iterable[str], portfolio_by_month: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month in months:
        frame = x.loc[x["month"] == month]
        trades = portfolio_by_month.get(month, [])
        rows.append({
            "month": month,
            "bars": len(frame),
            "trend_regime_bars": int(frame["regime_trend"].sum()),
            "range_regime_bars": int(frame["regime_range"].sum()),
            "high_vol_regime_bars": int(frame["regime_high_vol"].sum()),
            "neutral_regime_bars": int(frame["regime_neutral"].sum()),
            "portfolio_trades": len(trades),
            "trend_family_trades": sum(str(t.get("family")) == "趋势延续" for t in trades),
            "range_family_trades": sum(str(t.get("family")) == "区间流动性反转" for t in trades),
            "breakout_family_trades": sum(str(t.get("family")) == "波动扩张突破" for t in trades),
        })
    return pd.DataFrame(rows)


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
    development_static_funnel = static_signal_funnel(x, DEVELOPMENT_MONTHS)

    all_candidates: dict[int, list[ExpertCandidate]] = {}
    for expert_id in range(6):
        print(f"SEARCH_MODEL={expert_id}:{EXPERT_NAME[expert_id]}:{MODEL_FAMILY[expert_id]}")
        all_candidates[expert_id] = evaluate_expert_candidates(x, expert_data, expert_id)

    eligible_options = candidate_options_by_expert(all_candidates)
    valid_combinations = build_combinations(eligible_options)
    valid_combinations = evaluate_development_combinations(valid_combinations, all_candidates) if valid_combinations else []
    valid_combinations = prefer_complete_core_combinations(valid_combinations)
    selection_status = "MULTI_MODEL_RESEARCH_POLICY"
    combinations = valid_combinations
    if valid_combinations:
        selected = valid_combinations[0]
    else:
        diagnostic_options = diagnostic_options_by_expert(all_candidates)
        diagnostic_combinations = build_combinations(diagnostic_options)
        diagnostic_combinations = evaluate_development_combinations(diagnostic_combinations, all_candidates) if diagnostic_combinations else []
        diagnostic_combinations = prefer_complete_core_combinations(diagnostic_combinations)
        if diagnostic_combinations:
            selected = diagnostic_combinations[0]
            combinations = diagnostic_combinations
        else:
            selected, fallback_status = _fallback_multi_model_policy(all_candidates)
            selected = evaluate_development_combinations([selected], all_candidates)[0]
            combinations = [selected]
            selection_status = fallback_status
        if diagnostic_combinations:
            selection_status = "EXPERIMENTAL_FALLBACK"

    lookup = {
        (c.policy.expert_id, c.policy.key): c
        for candidates in all_candidates.values() for c in candidates
    }
    may_history = {i: [] for i in selected.policies}
    for month in DEVELOPMENT_MONTHS:
        if month == MONTHS[8]:
            break
        for expert_id, policy in selected.policies.items():
            may_history[expert_id].extend(lookup[(expert_id, policy.key)].monthly_events.get(month, []))
    june_history = {i: list(v) for i, v in may_history.items()}
    for expert_id, policy in selected.policies.items():
        june_history[expert_id].extend(lookup[(expert_id, policy.key)].monthly_events.get(MONTHS[8], []))

    may_metrics, may_trades, may_experts, may_thresholds, may_shadow, may_weights, may_audits = evaluate_selected_month(
        x, expert_data, selected, MONTHS[8], set(MONTHS[:8]), may_history
    )
    june_metrics, june_trades, june_experts, june_thresholds, june_shadow, june_weights, june_audits = evaluate_selected_month(
        x, expert_data, selected, OOS_MONTH, set(MONTHS[:9]), june_history
    )

    model_diversity_requirement_met = bool(
        len(selected.policies) >= STAGE_MIN_ACTIVE_MODELS
        and len({MODEL_FAMILY[i] for i in selected.policies}) >= 2
    )
    safe_degradation_applied = selection_status in {"INSUFFICIENT_MODEL_DIVERSITY", "ZERO_EXECUTABLE_MODELS"}
    research_stage_qualified = bool(
        model_diversity_requirement_met
        and stage_pass(may_metrics, may_trades)
        and stage_pass(june_metrics, june_trades)
    )
    final_hard_metrics_passed = bool(final_metric_pass(may_metrics) and final_metric_pass(june_metrics))
    qualified = False
    not_for_live_trading = True

    portfolio_frames = [
        detailed_trades(x, may_trades, selected.policies, MONTHS[8]),
        detailed_trades(x, june_trades, selected.policies, OOS_MONTH),
    ]
    portfolio_nonempty = [f for f in portfolio_frames if not f.empty]
    portfolio_df = pd.concat(portfolio_nonempty, ignore_index=True) if portfolio_nonempty else pd.DataFrame()
    portfolio_df.to_csv(RESULTS / "portfolio_trades.csv", index=False)
    portfolio_df.to_csv(RESULTS / "trades.csv", index=False)

    shadow_by_month = {MONTHS[8]: may_shadow, OOS_MONTH: june_shadow}
    shadow_frames = [
        shadow_trade_frame(x, may_shadow, selected.policies, MONTHS[8]),
        shadow_trade_frame(x, june_shadow, selected.policies, OOS_MONTH),
    ]
    shadow_nonempty = [f for f in shadow_frames if not f.empty]
    shadow_df = pd.concat(shadow_nonempty, ignore_index=True) if shadow_nonempty else pd.DataFrame()
    shadow_df.to_csv(RESULTS / "model_shadow_trades.csv", index=False)
    pd.DataFrame(model_monthly_rows(shadow_by_month)).to_csv(RESULTS / "model_monthly_stats.csv", index=False)
    pd.DataFrame(model_correlation_rows(shadow_by_month)).to_csv(RESULTS / "model_correlation.csv", index=False)

    audits_by_month = {MONTHS[8]: may_audits, OOS_MONTH: june_audits}
    _flatten_audit(audits_by_month, "conflicts").to_csv(RESULTS / "signal_conflicts.csv", index=False)
    _flatten_audit(audits_by_month, "dedup").to_csv(RESULTS / "signal_deduplication.csv", index=False)
    _flatten_audit(audits_by_month, "meta").to_csv(RESULTS / "meta_filter_audit.csv", index=False)

    portfolio_by_month = {MONTHS[8]: may_trades, OOS_MONTH: june_trades}
    _router_report(x, (MONTHS[8], OOS_MONTH), portfolio_by_month).to_csv(RESULTS / "regime_router_report.csv", index=False)

    contribution_rows: list[dict[str, Any]] = []
    for month, trades in portfolio_by_month.items():
        total = len(trades)
        for expert_id in sorted(selected.policies):
            subset = [t for t in trades if int(t["expert_id"]) == expert_id]
            contribution_rows.append({
                "month": month,
                "model": EXPERT_NAME[expert_id],
                "family": MODEL_FAMILY[expert_id],
                "primary_trades": len(subset),
                "primary_trade_share": len(subset) / total if total else 0.0,
                "primary_net_R": sum(float(t["net_r"]) for t in subset),
                "confirmation_appearances": sum(EXPERT_NAME[expert_id] in t.get("source_models", []) for t in trades),
            })
    pd.DataFrame(contribution_rows).to_csv(RESULTS / "model_contribution.csv", index=False)

    availability_rows = model_availability_audit_rows(all_candidates, selected, selection_status)
    availability_df = pd.DataFrame(availability_rows)
    availability_df.to_csv(RESULTS / "model_availability_audit.csv", index=False)
    executable_development_models = int(sum(row["policies_with_development_trades"] > 0 for row in availability_rows))
    selected_executable_development_models = int(sum(row["selected_development_executable"] for row in availability_rows))

    elimination_status = {
        EXPERT_NAME[expert_id]: {
            "family": MODEL_FAMILY[expert_id],
            "role": MODEL_ROLE[expert_id],
            "feature_count": len(features_for_expert(expert_id)),
            "eligible_candidates": sum(c.eligible for c in candidates),
            "searched_candidates": len(candidates),
            "best_candidate_eligible": bool(candidates[0].eligible),
            "best_candidate_reasons": candidates[0].elimination_reasons,
            "selected": expert_id in selected.policies,
        }
        for expert_id, candidates in all_candidates.items()
    }

    static_funnel = {**development_static_funnel, **static_signal_funnel(x, (OOS_MONTH,))}
    funnel_payload = selected_funnel_payload(selected, all_candidates, static_funnel)
    for month, thresholds in ((MONTHS[8], may_thresholds), (OOS_MONTH, june_thresholds)):
        for expert_name, values in thresholds.items():
            if expert_name not in funnel_payload[month]:
                continue
            funnel_payload[month][expert_name].update({
                "model_available": bool(values.get("model_available", False)),
                "training_candidates": int(values.get("training_candidates", 0)),
                "calibration_month": str(values.get("calibration_month", "unknown")),
                "calibration_candidates": int(values.get("calibration_candidates", 0)),
                "calibration_base_rate": float(values.get("calibration_base_rate", 0.0)),
                "calibration_brier": float(values.get("calibration_brier", 0.0)),
                "meta_brier": float(values.get("meta_brier", 0.0)),
                "probability_threshold": float(values.get("probability_threshold", 1.0)),
                "online_rank_percentile": float(values.get("quantile", 1.0)),
                "after_probability": int(values.get("raw_eval_after_probability", 0)),
                "after_meta_filter": int(values.get("after_meta_filter", 0)),
                "after_monthly_cap": int(values.get("capped_eval_candidates", 0)),
                "after_cooldown": int(values.get("after_cooldown", 0)),
                "score_drift": float(values.get("score_drift", 0.0)),
                "selected_policy": True,
            })
    (RESULTS / "signal_funnel.json").write_text(json.dumps(funnel_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_signal_funnel_csv(funnel_payload)

    may_diversity = quality_protection_summary(may_trades)
    june_diversity = quality_protection_summary(june_trades)
    best_shadow_by_month: dict[str, dict[str, Any]] = {}
    for month, mapping in shadow_by_month.items():
        ranked = sorted(
            ((EXPERT_NAME[i], metrics(events)) for i, events in mapping.items()),
            key=lambda item: item[1]["net_R"], reverse=True,
        )
        best_shadow_by_month[month] = {"model": ranked[0][0], **ranked[0][1]} if ranked else {}

    selected_policy_data = selected_policy_payload(selected)
    availability_by_name = {row["model"]: row for row in availability_rows}
    for model_name, payload in selected_policy_data.items():
        row = availability_by_name.get(model_name, {})
        payload["development_executable"] = bool(row.get("selected_development_executable", False))
        payload["selection_role"] = "executable_safe_fallback" if payload["development_executable"] else "diagnostic_probe"

    status = {
        "qualified": qualified,
        "research_stage_qualified": research_stage_qualified,
        "final_hard_metrics_passed": final_hard_metrics_passed,
        "safe_degradation_applied": safe_degradation_applied,
        "model_diversity_requirement_met": model_diversity_requirement_met,
        "executable_development_models": executable_development_models,
        "selected_executable_development_models": selected_executable_development_models,
        "fresh_blind_month_required": True,
        "selection_status": selection_status,
        "not_for_live_trading": not_for_live_trading,
        "engine": ENGINE_NAME,
        "method": "Regime router plus isolated directional specialists, holdout calibration, micro confirmation, holdout meta filter, causal shadow weighting, deduplication and one-position arbitration",
        "architecture": "completed 1h/15m router → six isolated specialists → specialist calibration → micro confirmation → meta filter → independent shadow accounts → dedup/conflict arbitration → next 5m open fill",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "months": list(MONTHS),
        "oos_month": OOS_MONTH,
        "oos_isolation": {
            "used_for_training": False,
            "used_for_thresholds": False,
            "used_for_model_selection": False,
            "used_for_portfolio_selection": False,
            "evaluation_occurs_after_policy_freeze": True,
        },
        "constraints": {
            "final": {"min_trades": MIN_TRADES, "max_trades": MAX_TRADES, "min_win_rate": MIN_WIN_RATE, "min_avg_win_loss_ratio": MIN_RATIO},
            "research_stage": RESEARCH_STAGE,
            "fee_rate_per_side": FEE_RATE,
            "slippage_per_fill_usdt": SLIPPAGE_ABS,
            "dedup_window_bars": DEDUP_WINDOW_BARS,
            "direction_conflict_utility_margin": DIRECTION_CONFLICT_MARGIN,
            "minimum_meta_probability": MIN_META_PROBABILITY,
            "minimum_expected_utility_R": MIN_EXPECTED_UTILITY_R,
            "dynamic_weight_bounds": [DYNAMIC_WEIGHT_MIN, DYNAMIC_WEIGHT_MAX],
        },
        "selected_combination_key": selected.key,
        "selected_model_policies": selected_policy_data,
        "model_elimination": elimination_status,
        "development_portfolio_monthly_stats": selected.monthly_metrics,
        "monthly_stats": {MONTHS[8]: may_metrics, OOS_MONTH: june_metrics},
        "model_primary_stats": {MONTHS[8]: may_experts, OOS_MONTH: june_experts},
        "model_weights": {MONTHS[8]: {EXPERT_NAME[i]: w for i, w in may_weights.items()}, OOS_MONTH: {EXPERT_NAME[i]: w for i, w in june_weights.items()}},
        "diversity_audit": {MONTHS[8]: may_diversity, OOS_MONTH: june_diversity},
        "best_shadow_model": best_shadow_by_month,
        "portfolio_vs_best_shadow": {
            month: {
                "portfolio_net_R": portfolio_by_month and metrics(portfolio_by_month[month])["net_R"],
                "best_shadow_net_R": best_shadow_by_month[month].get("net_R", 0.0),
                "portfolio_beats_best_shadow_net_R": metrics(portfolio_by_month[month])["net_R"] > best_shadow_by_month[month].get("net_R", 0.0),
                "portfolio_drawdown": metrics(portfolio_by_month[month])["max_drawdown_R"],
                "best_shadow_drawdown": best_shadow_by_month[month].get("max_drawdown_R", 0.0),
            }
            for month in (MONTHS[8], OOS_MONTH)
        },
        "threshold_audit": {MONTHS[8]: may_thresholds, OOS_MONTH: june_thresholds},
        "output_files": [
            "portfolio_trades.csv", "model_shadow_trades.csv", "model_monthly_stats.csv",
            "model_correlation.csv", "model_contribution.csv", "signal_conflicts.csv",
            "signal_deduplication.csv", "meta_filter_audit.csv", "regime_router_report.csv",
            "signal_funnel.csv", "candidate_leaderboard.csv", "expert_candidate_leaderboard.csv",
            "model_availability_audit.csv",
        ],
        "searched_expert_policies": sum(len(v) for v in all_candidates.values()),
        "searched_valid_combinations": len(valid_combinations),
        "searched_output_combinations": len(combinations),
    }
    (RESULTS / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "selected_policy.json").write_text(json.dumps(status["selected_model_policies"], ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_candidate_leaderboards(all_candidates, combinations)

    selected_names = list(status["selected_model_policies"].keys())
    selected_names_text = "、".join(selected_names) if selected_names else "无（零可执行模型，仅输出完整诊断）"
    selected_families_text = "、".join(sorted({MODEL_FAMILY[i] for i in selected.policies})) if selected.policies else "无"
    report = f"""# BTCUSDT 5分钟 多模型不足安全降级与完整诊断 V9.5.1 报告

- 架构：市场状态路由器 → 六个多空隔离专家 → 独立留出月校准 → 微观确认 → Meta过滤 → 独立影子账户 → 信号去重与方向冲突仲裁 → 全局单仓执行。
- 三类模型：趋势延续、区间流动性反转、波动扩张突破；同一模型的多空方向分别训练。
- 模型隔离：每个模型拥有独立特征集、校准器、Meta过滤、在线排名、冷却和影子交易记录。
- 成本：单边手续费 {FEE_RATE*100:.3f}%；每次成交滑点 {SLIPPAGE_ABS:.1f} USDT。
- 阶段目标：每月{STAGE_MIN_TRADES}–{STAGE_MAX_TRADES}笔、胜率≥{STAGE_MIN_WIN_RATE*100:.0f}%、实际盈亏比≥{STAGE_MIN_RATIO:.1f}、盈利因子≥{STAGE_MIN_PROFIT_FACTOR:.1f}、最大回撤≤{STAGE_MAX_DRAWDOWN:.1f}R、至少{STAGE_MIN_ACTIVE_MODELS}个模型成交、单模型占比≤{STAGE_MAX_MODEL_SHARE*100:.0f}%。
- 原最终目标：每月{MIN_TRADES}–{MAX_TRADES}笔、胜率≥{MIN_WIN_RATE*100:.0f}%、实际盈亏比≥{MIN_RATIO:.1f}。
- 选择状态：**{selection_status}**；安全降级：**{'已启用' if safe_degradation_applied else '未触发'}**。
- 可执行开发期模型：{executable_development_models}个；实际选入可执行模型：{selected_executable_development_models}个；多模型多样性要求：**{'满足' if model_diversity_requirement_met else '不满足'}**。
- 安全降级只保证程序完整输出，不降低概率、在线排名、Meta、至少3模型或单模型占比等验收标准。
- 无论数值是否达到，5月和6月已被查看，因此仍不可实盘。
- V9.5.1阶段验收：**{'通过' if research_stage_qualified else '未通过'}**。
- 原最终指标：**{'数值通过但需新盲测月' if final_hard_metrics_passed else '未通过'}**。

## 最终模型组合

- 活跃模型：{selected_names_text}
- 覆盖模型家族：{selected_families_text}
- 组合键：`{selected.key}`

## 组合账户月度结果

| 月份 | 交易 | 胜率 | 实际盈亏比 | 盈利因子 | 净R | 最大回撤R | 活跃模型 | 最大单模型占比 | 阶段通过 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| {MONTHS[8]} | {may_metrics['trades']} | {may_metrics['win_rate']*100:.2f}% | {may_metrics['avg_win_loss_ratio']:.3f} | {may_metrics['profit_factor']:.3f} | {may_metrics['net_R']:.3f} | {may_metrics['max_drawdown_R']:.3f} | {may_diversity['active_models']} | {may_diversity['max_single_model_trade_share']*100:.1f}% | {'是' if stage_pass(may_metrics, may_trades) else '否'} |
| {OOS_MONTH} | {june_metrics['trades']} | {june_metrics['win_rate']*100:.2f}% | {june_metrics['avg_win_loss_ratio']:.3f} | {june_metrics['profit_factor']:.3f} | {june_metrics['net_R']:.3f} | {june_metrics['max_drawdown_R']:.3f} | {june_diversity['active_models']} | {june_diversity['max_single_model_trade_share']*100:.1f}% | {'是' if stage_pass(june_metrics, june_trades) else '否'} |

## 关键诊断文件

- `model_shadow_trades.csv`：每个模型即使未被组合采用，也保留独立影子交易结果。
- `portfolio_trades.csv`：组合仲裁后真实单仓交易。
- `model_monthly_stats.csv`：模型独立月度胜率、盈亏比、净R和回撤。
- `model_correlation.csv`：模型信号重合度及重合交易收益相关性。
- `signal_deduplication.csv`：同方向重复候选的来源、去重数量和主模型。
- `signal_conflicts.csv`：多空冲突的放弃或解决记录。
- `meta_filter_audit.csv`：专业模型、路由、微观确认、Meta概率和效用审计。
- `model_contribution.csv`：各模型对最终组合交易和净R的贡献。
- `model_availability_audit.csv`：逐模型记录完整候选搜索、开发期可执行策略数量、最佳覆盖月份、净R、是否选入及拒绝原因。

## 保守执行假设

高周期特征只使用已经完成的K线；资金费率延迟一根5分钟K线使用；同K线同时触及止损和止盈时先按止损；真实组合同一时刻最多持有一笔仓位，影子账户仅用于研究统计。
"""
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    (RESULTS / "run_identity.txt").write_text(
        f"{ENGINE_NAME}\nmonths={','.join(MONTHS)}\noos={OOS_MONTH}\noutput=results_v9_5_1\nselection_status={selection_status}\nresearch_stage_qualified={research_stage_qualified}\nqualified=false\n",
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=ENGINE_NAME)
    parser.add_argument("--self-test", action="store_true", help="run offline V9.5.1 architecture regression tests")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
