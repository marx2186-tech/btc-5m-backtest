from __future__ import annotations

import hashlib
import io
import json
import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache_v10_2"
CACHE.mkdir(exist_ok=True)

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote",
    "ignore",
]
STEP_MS = 5 * 60 * 1000


@dataclass(frozen=True)
class ChannelSpec:
    channel_id: str
    family: str
    profile: str
    label: str
    stop_atr_multiple: float
    reward_risk: float
    max_holding_hours: int
    is_baseline: bool = False


def _rr_key(value: float) -> str:
    return str(value).replace(".", "_")


BASELINE_CHANNELS: tuple[ChannelSpec, ...] = tuple(
    ChannelSpec(
        f"60m_trend_pullback_quality_rr{_rr_key(rr)}",
        "trend_pullback",
        "quality",
        f"V10.1冻结基线·1小时趋势回踩质量过滤 RR{rr:.1f}",
        1.25,
        rr,
        36,
        True,
    )
    for rr in (2.5, 3.0)
)


def _optimized_channels() -> tuple[ChannelSpec, ...]:
    rows: list[ChannelSpec] = []
    for profile, profile_label, stop_multiple, hold_hours in (
        ("anti_exhaustion", "防趋势衰竭", 1.25, 36),
        ("balanced", "均衡趋势区间", 1.20, 36),
        ("persistence", "趋势持续性", 1.25, 48),
    ):
        for rr in (2.5, 3.0):
            rows.append(
                ChannelSpec(
                    f"60m_trend_pullback_{profile}_rr{_rr_key(rr)}",
                    "trend_pullback",
                    profile,
                    f"1小时趋势回踩·{profile_label} RR{rr:.1f}",
                    stop_multiple,
                    rr,
                    hold_hours,
                    False,
                )
            )
    return tuple(rows)


CHANNELS: tuple[ChannelSpec, ...] = BASELINE_CHANNELS + _optimized_channels()


def month_range(start_month: str, end_month: str) -> list[str]:
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    if end < start:
        raise ValueError("end_month must not be earlier than start_month")
    return [str(x) for x in pd.period_range(start, end, freq="M")]


def download(url: str, path: Path, attempts: int = 6) -> bytes:
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                timeout=90,
                headers={"User-Agent": "btc-hourly-regime-robustness-v10-2/1.0"},
            )
            response.raise_for_status()
            path.write_bytes(response.content)
            return response.content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 ** min(attempt, 4))
    raise RuntimeError(f"Download failed: {url}: {last_error}")


def read_verified_zip(url: str, checksum_url: str, cache_name: str) -> tuple[bytes, str]:
    raw = download(url, CACHE / cache_name)
    checksum_text = download(checksum_url, CACHE / f"{cache_name}.CHECKSUM").decode("utf-8").strip()
    expected = checksum_text.split()[0].lower()
    actual = hashlib.sha256(raw).hexdigest().lower()
    if expected != actual:
        raise RuntimeError(f"SHA-256 mismatch for {cache_name}: expected={expected}, actual={actual}")
    return raw, actual


def read_single_csv_zip(raw: bytes, name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = [member for member in archive.namelist() if member.lower().endswith(".csv")]
        if len(members) != 1:
            raise RuntimeError(f"Unexpected ZIP contents for {name}: {members}")
        return archive.read(members[0])


def parse_kline_csv(content: bytes) -> pd.DataFrame:
    first_line = content.splitlines()[0].decode("utf-8", errors="ignore").lower()
    has_header = "open_time" in first_line or "open time" in first_line
    frame = pd.read_csv(io.BytesIO(content), header=0 if has_header else None).iloc[:, :12]
    frame.columns = KLINE_COLUMNS
    for column in [x for x in KLINE_COLUMNS if x != "ignore"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_official_5m_data(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    symbol = str(config["symbol"]).upper()
    interval = "5m"
    months = month_range(str(config["start_month"]), str(config["end_month"]))
    frames: list[pd.DataFrame] = []
    files: list[dict[str, Any]] = []

    for month in months:
        name = f"{symbol}-{interval}-{month}.zip"
        base = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}"
        raw, digest = read_verified_zip(f"{base}/{name}", f"{base}/{name}.CHECKSUM", name)
        frame = parse_kline_csv(read_single_csv_zip(raw, name))
        frames.append(frame)
        files.append({"file": name, "sha256": digest, "rows": int(len(frame))})

    data = (
        pd.concat(frames, ignore_index=True)
        .sort_values("open_time")
        .drop_duplicates("open_time", keep="last")
        .reset_index(drop=True)
    )
    start = pd.Timestamp(f"{months[0]}-01T00:00:00Z")
    end = pd.Timestamp(f"{months[-1]}-01T00:00:00Z") + pd.offsets.MonthBegin(1)
    expected_times = np.arange(
        int(start.timestamp() * 1000),
        int(end.timestamp() * 1000),
        STEP_MS,
        dtype=np.int64,
    )
    actual_times = pd.to_numeric(data["open_time"], errors="coerce").dropna().astype("int64").to_numpy()
    unique_times = np.unique(actual_times)
    missing = np.setdiff1d(expected_times, unique_times)
    extra = np.setdiff1d(unique_times, expected_times)
    duplicate_count = int(pd.Series(actual_times).duplicated().sum())

    o = data["open"].to_numpy(float)
    h = data["high"].to_numpy(float)
    l = data["low"].to_numpy(float)
    c = data["close"].to_numpy(float)
    finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    valid_ohlc = finite & (h >= np.maximum.reduce([o, c, l])) & (l <= np.minimum.reduce([o, c, h]))
    close_time = pd.to_numeric(data["close_time"], errors="coerce").fillna(-1).astype("int64").to_numpy()

    audit = {
        "source": "Binance USDⓈ-M Futures official verified monthly 5m klines",
        "symbol": symbol,
        "interval": interval,
        "months": months,
        "start_utc": start.isoformat(),
        "end_utc": pd.to_datetime(expected_times[-1], unit="ms", utc=True).isoformat(),
        "expected_rows": int(len(expected_times)),
        "actual_rows": int(len(data)),
        "unique_rows": int(len(unique_times)),
        "missing_rows": int(len(missing)),
        "extra_rows": int(len(extra)),
        "duplicate_timestamps": duplicate_count,
        "off_grid_rows": int(np.sum((actual_times - expected_times[0]) % STEP_MS != 0)),
        "invalid_close_time_rows": int(np.sum(close_time != actual_times + STEP_MS - 1)),
        "invalid_ohlc_rows": int(np.sum(~valid_ohlc)),
        "uses_rest_fallback": False,
        "files": files,
    }
    audit["passed"] = bool(
        len(data) == len(expected_times)
        and len(unique_times) == len(expected_times)
        and len(missing) == 0
        and len(extra) == 0
        and duplicate_count == 0
        and audit["off_grid_rows"] == 0
        and audit["invalid_close_time_rows"] == 0
        and audit["invalid_ohlc_rows"] == 0
    )
    if not audit["passed"]:
        raise RuntimeError("V10.2 official 5m data audit failed: " + json.dumps(audit, ensure_ascii=False))

    data["time"] = pd.to_datetime(data["open_time"].astype("int64"), unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce").astype(float)
    return data, audit


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return rma(true_range, length)


def directional_movement(frame: pd.DataFrame, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=frame.index)
    atr_value = atr(frame, length).replace(0, np.nan)
    plus_di = 100.0 * rma(plus_dm, length) / atr_value
    minus_di = 100.0 * rma(minus_dm, length) / atr_value
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_value = rma(dx, length)
    return plus_di, minus_di, adx_value


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length).replace(0, np.nan)
    relative_strength = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)


def resample_hourly(raw_5m: pd.DataFrame) -> pd.DataFrame:
    indexed = raw_5m.set_index("time")[["open", "high", "low", "close", "volume"]]
    frame = indexed.resample(
        "60min",
        origin="start_day",
        closed="left",
        label="right",
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    count = indexed["close"].resample(
        "60min",
        origin="start_day",
        closed="left",
        label="right",
    ).count()
    frame["source_5m_rows"] = count
    frame = frame.loc[frame["source_5m_rows"] == 12].dropna().copy()
    frame.index.name = "signal_time"
    return frame


def consecutive_true_count(mask: pd.Series) -> pd.Series:
    values = mask.fillna(False).astype(bool).to_numpy()
    counts = np.zeros(len(values), dtype=np.int64)
    running = 0
    for index, value in enumerate(values):
        running = running + 1 if value else 0
        counts[index] = running
    return pd.Series(counts, index=mask.index, dtype=float)


def add_hourly_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["atr14"] = atr(x, 14)
    x["plus_di14"], x["minus_di14"], x["adx14"] = directional_movement(x, 14)
    x["rsi14"] = rsi(x["close"], 14)
    x["channel_high_20"] = x["high"].shift(1).rolling(20).max()
    x["channel_low_20"] = x["low"].shift(1).rolling(20).min()
    x["volume_mean_20"] = x["volume"].shift(1).rolling(20).mean()
    x["ema_separation_atr"] = (x["ema20"] - x["ema50"]).abs() / x["atr14"].replace(0, np.nan)
    x["ema_slope_3_atr"] = (x["ema20"] - x["ema20"].shift(3)) / x["atr14"].replace(0, np.nan)
    x["di_spread"] = (x["plus_di14"] - x["minus_di14"]).abs()
    x["channel_width_atr"] = (x["channel_high_20"] - x["channel_low_20"]) / x["atr14"].replace(0, np.nan)
    candle_range = (x["high"] - x["low"]).replace(0, np.nan)
    x["clv_long"] = (x["close"] - x["low"]) / candle_range
    x["clv_short"] = (x["high"] - x["close"]) / candle_range
    x["body_atr"] = (x["close"] - x["open"]).abs() / x["atr14"].replace(0, np.nan)
    x["lower_wick"] = np.minimum(x["open"], x["close"]) - x["low"]
    x["upper_wick"] = x["high"] - np.maximum(x["open"], x["close"])
    x["body_abs"] = (x["close"] - x["open"]).abs().replace(0, np.nan)
    x["lower_wick_body"] = x["lower_wick"] / x["body_abs"]
    x["upper_wick_body"] = x["upper_wick"] / x["body_abs"]
    x["volume_ratio"] = x["volume"] / x["volume_mean_20"].replace(0, np.nan)
    x["adx_change_3"] = x["adx14"] - x["adx14"].shift(3)
    x["ema50_slope_6_atr"] = (x["ema50"] - x["ema50"].shift(6)) / x["atr14"].replace(0, np.nan)
    x["price_extension_ema20_atr"] = (x["close"] - x["ema20"]).abs() / x["atr14"].replace(0, np.nan)
    x["atr_ratio"] = x["atr14"] / x["close"].replace(0, np.nan)
    x["atr_ratio_q20_252"] = x["atr_ratio"].shift(1).rolling(252).quantile(0.20)
    x["atr_ratio_q90_252"] = x["atr_ratio"].shift(1).rolling(252).quantile(0.90)
    x["trend_age_long"] = consecutive_true_count((x["ema20"] > x["ema50"]) & (x["ema_slope_3_atr"] > 0))
    x["trend_age_short"] = consecutive_true_count((x["ema20"] < x["ema50"]) & (x["ema_slope_3_atr"] < 0))
    x["close_above_ema20_count_3"] = (x["close"] > x["ema20"]).astype(float).rolling(3).sum()
    x["close_below_ema20_count_3"] = (x["close"] < x["ema20"]).astype(float).rolling(3).sum()
    return x


def _signal_rows(mask: pd.Series, direction: int, x: pd.DataFrame, spec: ChannelSpec) -> pd.DataFrame:
    columns = [
        "close",
        "atr14",
        "adx14",
        "rsi14",
        "plus_di14",
        "minus_di14",
        "ema_separation_atr",
        "ema_slope_3_atr",
        "volume_ratio",
        "clv_long",
        "clv_short",
        "channel_width_atr",
        "adx_change_3",
        "ema50_slope_6_atr",
        "price_extension_ema20_atr",
        "atr_ratio",
        "atr_ratio_q20_252",
        "atr_ratio_q90_252",
        "trend_age_long",
        "trend_age_short",
        "close_above_ema20_count_3",
        "close_below_ema20_count_3",
    ]
    selected = x.loc[mask.fillna(False), columns].copy()
    if selected.empty:
        return pd.DataFrame()
    selected = selected.reset_index()
    selected["direction"] = int(direction)
    selected["channel_id"] = spec.channel_id
    selected["channel_label"] = spec.label
    selected["family"] = spec.family
    selected["profile"] = spec.profile
    selected["stop_atr_multiple"] = float(spec.stop_atr_multiple)
    selected["reward_risk"] = float(spec.reward_risk)
    selected["max_holding_hours"] = int(spec.max_holding_hours)
    selected["is_baseline"] = bool(spec.is_baseline)
    return selected


def baseline_masks(x: pd.DataFrame, family: str) -> tuple[pd.Series, pd.Series]:
    if family == "trend_pullback":
        long_mask = (
            (x["ema20"] > x["ema50"])
            & (x["ema20"].diff(3) > 0)
            & (x["adx14"] >= 18.0)
            & (x["low"] <= x["ema20"])
            & (x["close"] > x["ema20"])
            & (x["close"] > x["open"])
        )
        short_mask = (
            (x["ema20"] < x["ema50"])
            & (x["ema20"].diff(3) < 0)
            & (x["adx14"] >= 18.0)
            & (x["high"] >= x["ema20"])
            & (x["close"] < x["ema20"])
            & (x["close"] < x["open"])
        )
        return long_mask, short_mask
    if family == "range_reversal":
        range_regime = (x["adx14"] <= 22.0) & (x["ema_separation_atr"] <= 1.0)
        long_mask = (
            range_regime
            & (x["low"] < x["channel_low_20"])
            & (x["close"] > x["channel_low_20"])
            & (x["rsi14"] <= 45.0)
        )
        short_mask = (
            range_regime
            & (x["high"] > x["channel_high_20"])
            & (x["close"] < x["channel_high_20"])
            & (x["rsi14"] >= 55.0)
        )
        return long_mask, short_mask
    raise ValueError(f"Unsupported baseline family: {family}")


def trend_quality_masks(x: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    common = (
        (x["adx14"] >= 23.0)
        & (x["ema_separation_atr"] >= 0.65)
        & (x["volume_ratio"] >= 0.80)
        & (x["body_atr"].between(0.12, 1.35))
    )
    long_mask = (
        common
        & (x["ema20"] > x["ema50"])
        & (x["ema_slope_3_atr"] >= 0.10)
        & (x["plus_di14"] >= x["minus_di14"] + 4.0)
        & (x["low"] <= x["ema20"] + 0.10 * x["atr14"])
        & (x["low"] >= x["ema50"] - 0.25 * x["atr14"])
        & (x["close"] > x["ema20"])
        & (x["close"] > x["open"])
        & (x["clv_long"] >= 0.65)
        & (x["rsi14"].between(50.0, 69.0))
    )
    short_mask = (
        common
        & (x["ema20"] < x["ema50"])
        & (x["ema_slope_3_atr"] <= -0.10)
        & (x["minus_di14"] >= x["plus_di14"] + 4.0)
        & (x["high"] >= x["ema20"] - 0.10 * x["atr14"])
        & (x["high"] <= x["ema50"] + 0.25 * x["atr14"])
        & (x["close"] < x["ema20"])
        & (x["close"] < x["open"])
        & (x["clv_short"] >= 0.65)
        & (x["rsi14"].between(31.0, 50.0))
    )
    return long_mask, short_mask


def trend_confirmation_masks(x: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    prior_long_pullback = (
        (x["ema20"].shift(1) > x["ema50"].shift(1))
        & (x["low"].shift(1) <= x["ema20"].shift(1) + 0.10 * x["atr14"].shift(1))
        & (x["low"].shift(1) >= x["ema50"].shift(1) - 0.20 * x["atr14"].shift(1))
        & (x["close"].shift(1) > x["ema50"].shift(1))
    )
    prior_short_pullback = (
        (x["ema20"].shift(1) < x["ema50"].shift(1))
        & (x["high"].shift(1) >= x["ema20"].shift(1) - 0.10 * x["atr14"].shift(1))
        & (x["high"].shift(1) <= x["ema50"].shift(1) + 0.20 * x["atr14"].shift(1))
        & (x["close"].shift(1) < x["ema50"].shift(1))
    )
    common = (
        (x["adx14"] >= 25.0)
        & (x["ema_separation_atr"] >= 0.80)
        & (x["volume_ratio"] >= 0.90)
        & (x["body_atr"].between(0.15, 1.50))
    )
    long_mask = (
        common
        & prior_long_pullback
        & (x["ema20"] > x["ema50"])
        & (x["ema_slope_3_atr"] >= 0.14)
        & (x["plus_di14"] >= x["minus_di14"] + 6.0)
        & (x["close"] > x["high"].shift(1))
        & (x["close"] > x["open"])
        & (x["clv_long"] >= 0.70)
        & (x["rsi14"].between(52.0, 70.0))
    )
    short_mask = (
        common
        & prior_short_pullback
        & (x["ema20"] < x["ema50"])
        & (x["ema_slope_3_atr"] <= -0.14)
        & (x["minus_di14"] >= x["plus_di14"] + 6.0)
        & (x["close"] < x["low"].shift(1))
        & (x["close"] < x["open"])
        & (x["clv_short"] >= 0.70)
        & (x["rsi14"].between(30.0, 48.0))
    )
    return long_mask, short_mask


def range_quality_masks(x: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    range_regime = (
        (x["adx14"] <= 18.0)
        & (x["di_spread"] <= 8.0)
        & (x["ema_separation_atr"] <= 0.55)
        & (x["channel_width_atr"].between(4.0, 10.0))
        & (x["volume_ratio"] >= 0.80)
    )
    long_mask = (
        range_regime
        & (x["low"] <= x["channel_low_20"] - 0.10 * x["atr14"])
        & (x["close"] >= x["channel_low_20"] + 0.05 * x["atr14"])
        & (x["rsi14"] <= 36.0)
        & (x["clv_long"] >= 0.68)
        & (x["lower_wick_body"] >= 1.10)
    )
    short_mask = (
        range_regime
        & (x["high"] >= x["channel_high_20"] + 0.10 * x["atr14"])
        & (x["close"] <= x["channel_high_20"] - 0.05 * x["atr14"])
        & (x["rsi14"] >= 64.0)
        & (x["clv_short"] >= 0.68)
        & (x["upper_wick_body"] >= 1.10)
    )
    return long_mask, short_mask


def range_confirmation_masks(x: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    prior_range_regime = (
        (x["adx14"].shift(1) <= 19.0)
        & (x["di_spread"].shift(1) <= 10.0)
        & (x["ema_separation_atr"].shift(1) <= 0.65)
        & (x["channel_width_atr"].shift(1).between(4.0, 11.0))
    )
    prior_long_sweep = (
        prior_range_regime
        & (x["low"].shift(1) <= x["channel_low_20"].shift(1) - 0.08 * x["atr14"].shift(1))
        & (x["close"].shift(1) > x["channel_low_20"].shift(1))
        & (x["rsi14"].shift(1) <= 39.0)
        & (x["clv_long"].shift(1) >= 0.58)
    )
    prior_short_sweep = (
        prior_range_regime
        & (x["high"].shift(1) >= x["channel_high_20"].shift(1) + 0.08 * x["atr14"].shift(1))
        & (x["close"].shift(1) < x["channel_high_20"].shift(1))
        & (x["rsi14"].shift(1) >= 61.0)
        & (x["clv_short"].shift(1) >= 0.58)
    )
    current_regime = (
        (x["adx14"] <= 20.0)
        & (x["ema_separation_atr"] <= 0.70)
        & (x["volume_ratio"] >= 0.70)
    )
    long_mask = (
        current_regime
        & prior_long_sweep
        & (x["close"] > x["high"].shift(1))
        & (x["close"] > x["open"])
        & (x["clv_long"] >= 0.65)
        & (x["rsi14"] <= 52.0)
    )
    short_mask = (
        current_regime
        & prior_short_sweep
        & (x["close"] < x["low"].shift(1))
        & (x["close"] < x["open"])
        & (x["clv_short"] >= 0.65)
        & (x["rsi14"] >= 48.0)
    )
    return long_mask, short_mask


def trend_anti_exhaustion_masks(x: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    base_long, base_short = trend_quality_masks(x)
    volatility_regime = x["atr_ratio"].between(x["atr_ratio_q20_252"], x["atr_ratio_q90_252"])
    common = (
        x["adx14"].between(23.0, 42.0)
        & x["volume_ratio"].between(0.75, 1.80)
        & x["channel_width_atr"].between(3.0, 7.0)
        & (x["price_extension_ema20_atr"] <= 0.85)
        & volatility_regime
    )
    long_advantage = x["plus_di14"] - x["minus_di14"]
    short_advantage = x["minus_di14"] - x["plus_di14"]
    long_mask = (
        base_long
        & common
        & long_advantage.between(4.0, 18.0)
        & x["ema_slope_3_atr"].between(0.10, 0.36)
        & x["rsi14"].between(52.0, 64.5)
        & x["trend_age_long"].between(5.0, 96.0)
    )
    short_mask = (
        base_short
        & common
        & short_advantage.between(4.0, 18.0)
        & x["ema_slope_3_atr"].between(-0.36, -0.10)
        & x["rsi14"].between(35.5, 48.0)
        & x["trend_age_short"].between(5.0, 96.0)
    )
    return long_mask, short_mask


def trend_balanced_masks(x: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    base_long, base_short = trend_quality_masks(x)
    volatility_regime = x["atr_ratio"].between(x["atr_ratio_q20_252"], x["atr_ratio_q90_252"])
    common = (
        x["adx14"].between(23.0, 36.0)
        & x["ema_separation_atr"].between(0.65, 1.60)
        & x["volume_ratio"].between(0.75, 1.60)
        & x["channel_width_atr"].between(3.5, 6.25)
        & x["adx_change_3"].between(-2.5, 8.0)
        & (x["price_extension_ema20_atr"] <= 0.70)
        & volatility_regime
    )
    long_advantage = x["plus_di14"] - x["minus_di14"]
    short_advantage = x["minus_di14"] - x["plus_di14"]
    long_mask = (
        base_long
        & common
        & long_advantage.between(5.0, 14.5)
        & x["ema_slope_3_atr"].between(0.10, 0.28)
        & x["rsi14"].between(52.0, 63.5)
        & x["trend_age_long"].between(6.0, 72.0)
    )
    short_mask = (
        base_short
        & common
        & short_advantage.between(5.0, 14.5)
        & x["ema_slope_3_atr"].between(-0.28, -0.10)
        & x["rsi14"].between(36.5, 48.0)
        & x["trend_age_short"].between(6.0, 72.0)
    )
    return long_mask, short_mask


def trend_persistence_masks(x: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    base_long, base_short = trend_quality_masks(x)
    common = (
        x["adx14"].between(23.0, 45.0)
        & (x["adx_change_3"] >= -3.0)
        & x["volume_ratio"].between(0.70, 2.20)
        & x["channel_width_atr"].between(3.0, 8.0)
        & (x["price_extension_ema20_atr"] <= 0.80)
    )
    long_mask = (
        base_long
        & common
        & x["trend_age_long"].between(8.0, 120.0)
        & (x["ema50_slope_6_atr"] >= 0.04)
        & (x["close_above_ema20_count_3"] >= 2.0)
    )
    short_mask = (
        base_short
        & common
        & x["trend_age_short"].between(8.0, 120.0)
        & (x["ema50_slope_6_atr"] <= -0.04)
        & (x["close_below_ema20_count_3"] >= 2.0)
    )
    return long_mask, short_mask


def masks_for_spec(x: pd.DataFrame, spec: ChannelSpec) -> tuple[pd.Series, pd.Series]:
    if spec.family != "trend_pullback":
        raise ValueError(f"V10.2 only permits trend_pullback channels: {spec}")
    if spec.profile == "quality":
        return trend_quality_masks(x)
    if spec.profile == "anti_exhaustion":
        return trend_anti_exhaustion_masks(x)
    if spec.profile == "balanced":
        return trend_balanced_masks(x)
    if spec.profile == "persistence":
        return trend_persistence_masks(x)
    raise ValueError(f"Unsupported V10.2 channel spec: {spec}")


def generate_signals(hourly: pd.DataFrame, spec: ChannelSpec) -> pd.DataFrame:
    x = add_hourly_features(hourly)
    long_mask, short_mask = masks_for_spec(x, spec)
    frames = [_signal_rows(long_mask, 1, x, spec), _signal_rows(short_mask, -1, x, spec)]
    nonempty = [part for part in frames if not part.empty]
    if not nonempty:
        return pd.DataFrame()
    signals = pd.concat(nonempty, ignore_index=True)
    return signals.sort_values(["signal_time", "direction"]).drop_duplicates("signal_time", keep="first").reset_index(drop=True)


def build_all_signals(raw_5m: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    hourly = resample_hourly(raw_5m)
    signal_map: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        signals = generate_signals(hourly, spec)
        signal_map[spec.channel_id] = signals
        audit_rows.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "family": spec.family,
                "profile": spec.profile,
                "reward_risk": spec.reward_risk,
                "stop_atr_multiple": spec.stop_atr_multiple,
                "max_holding_hours": spec.max_holding_hours,
                "is_baseline": spec.is_baseline,
                "signals": int(len(signals)),
                "long_signals": int((signals["direction"] == 1).sum()) if not signals.empty else 0,
                "short_signals": int((signals["direction"] == -1).sum()) if not signals.empty else 0,
            }
        )
    resample_audit = {
        "60": {
            "rows": int(len(hourly)),
            "first_bar_close_utc": hourly.index.min().isoformat() if not hourly.empty else None,
            "last_bar_close_utc": hourly.index.max().isoformat() if not hourly.empty else None,
            "source_rows_per_bar": 12,
        }
    }
    return signal_map, {"resample": resample_audit, "channels": audit_rows}


def phase_for_time(timestamp: pd.Timestamp, config: dict[str, Any]) -> str:
    ts = pd.Timestamp(timestamp)
    for phase_name, phase in config["phases"].items():
        start = pd.Timestamp(phase["start_utc"])
        end = pd.Timestamp(phase["end_utc"])
        if start <= ts <= end:
            return phase_name
    return "outside_phase"


def execute_channel(
    raw_5m: pd.DataFrame,
    signals: pd.DataFrame,
    spec: ChannelSpec,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "channel_id",
        "channel_label",
        "family",
        "profile",
        "reward_risk_target",
        "stop_atr_multiple",
        "signal_time_utc",
        "entry_time_utc",
        "exit_time_utc",
        "direction",
        "entry_price",
        "stop_price",
        "target_price",
        "exit_price",
        "stop_distance",
        "gross_r",
        "fee_r",
        "net_r",
        "win",
        "exit_reason",
        "holding_5m_bars",
        "signal_atr",
        "signal_adx",
        "signal_rsi",
        "signal_plus_di",
        "signal_minus_di",
        "signal_ema_separation_atr",
        "signal_ema_slope_3_atr",
        "signal_volume_ratio",
        "signal_clv",
        "signal_channel_width_atr",
        "signal_adx_change_3",
        "signal_ema50_slope_6_atr",
        "signal_price_extension_ema20_atr",
        "signal_atr_ratio",
        "signal_trend_age",
        "month",
        "phase",
        "is_baseline",
    ]
    if signals.empty:
        return pd.DataFrame(columns=columns)

    fee_rate = float(config["execution"]["fee_rate_per_side"])
    slippage = float(config["execution"]["tick_size"]) * int(config["execution"]["slippage_ticks_per_fill"])
    min_stop_pct = float(config["execution"]["minimum_stop_distance_pct"])

    times = raw_5m["time"].to_numpy(dtype="datetime64[ns]")
    opens = raw_5m["open"].to_numpy(float)
    highs = raw_5m["high"].to_numpy(float)
    lows = raw_5m["low"].to_numpy(float)
    closes = raw_5m["close"].to_numpy(float)
    max_holding_5m = spec.max_holding_hours * 12
    trades: list[dict[str, Any]] = []
    next_free_index = 0

    for row in signals.sort_values("signal_time").itertuples(index=False):
        signal_time = pd.Timestamp(row.signal_time)
        entry_i = int(np.searchsorted(times, signal_time.to_datetime64(), side="left"))
        if entry_i < next_free_index or entry_i >= len(raw_5m):
            continue
        direction = int(row.direction)
        entry_price = float(opens[entry_i] + slippage * direction)
        signal_atr = float(row.atr14)
        if not math.isfinite(signal_atr) or signal_atr <= 0:
            continue
        stop_distance = max(signal_atr * spec.stop_atr_multiple, entry_price * min_stop_pct)
        stop_price = entry_price - direction * stop_distance
        target_price = entry_price + direction * spec.reward_risk * stop_distance
        final_i = min(entry_i + max_holding_5m - 1, len(raw_5m) - 1)
        exit_i = final_i
        exit_reason = "TIME_EXIT"
        raw_exit = float(closes[final_i])

        for bar_i in range(entry_i, final_i + 1):
            if direction == 1:
                stop_hit = lows[bar_i] <= stop_price
                target_hit = highs[bar_i] >= target_price
            else:
                stop_hit = highs[bar_i] >= stop_price
                target_hit = lows[bar_i] <= target_price
            if stop_hit and target_hit:
                exit_i = bar_i
                exit_reason = "STOP_FIRST_CONSERVATIVE"
                raw_exit = stop_price
                break
            if stop_hit:
                exit_i = bar_i
                exit_reason = "STOP"
                raw_exit = stop_price
                break
            if target_hit:
                exit_i = bar_i
                exit_reason = "TARGET"
                raw_exit = target_price
                break

        exit_price = float(raw_exit - slippage * direction)
        gross_r = direction * (exit_price - entry_price) / stop_distance
        fee_r = fee_rate * (entry_price + exit_price) / stop_distance
        net_r = gross_r - fee_r
        entry_time = pd.Timestamp(raw_5m.iloc[entry_i]["time"])
        exit_time = pd.Timestamp(raw_5m.iloc[exit_i]["time"]) + pd.Timedelta(minutes=5)
        clv = float(row.clv_long if direction == 1 else row.clv_short)
        trades.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "family": spec.family,
                "profile": spec.profile,
                "reward_risk_target": spec.reward_risk,
                "stop_atr_multiple": spec.stop_atr_multiple,
                "signal_time_utc": signal_time.isoformat(),
                "entry_time_utc": entry_time.isoformat(),
                "exit_time_utc": exit_time.isoformat(),
                "direction": direction,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "exit_price": exit_price,
                "stop_distance": stop_distance,
                "gross_r": gross_r,
                "fee_r": fee_r,
                "net_r": net_r,
                "win": bool(net_r > 0),
                "exit_reason": exit_reason,
                "holding_5m_bars": int(exit_i - entry_i + 1),
                "signal_atr": signal_atr,
                "signal_adx": float(row.adx14),
                "signal_rsi": float(row.rsi14),
                "signal_plus_di": float(row.plus_di14),
                "signal_minus_di": float(row.minus_di14),
                "signal_ema_separation_atr": float(row.ema_separation_atr),
                "signal_ema_slope_3_atr": float(row.ema_slope_3_atr),
                "signal_volume_ratio": float(row.volume_ratio),
                "signal_clv": clv,
                "signal_channel_width_atr": float(row.channel_width_atr),
                "signal_adx_change_3": float(row.adx_change_3),
                "signal_ema50_slope_6_atr": float(row.ema50_slope_6_atr),
                "signal_price_extension_ema20_atr": float(row.price_extension_ema20_atr),
                "signal_atr_ratio": float(row.atr_ratio),
                "signal_trend_age": float(row.trend_age_long if direction == 1 else row.trend_age_short),
                "month": entry_time.strftime("%Y-%m"),
                "phase": phase_for_time(entry_time, config),
                "is_baseline": spec.is_baseline,
            }
        )
        next_free_index = exit_i + 1

    return pd.DataFrame(trades, columns=columns)


def metrics(trades: pd.DataFrame | Iterable[dict[str, Any]]) -> dict[str, float]:
    if isinstance(trades, pd.DataFrame):
        values = pd.to_numeric(trades.get("net_r", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(float)
    else:
        values = np.asarray([float(row["net_r"]) for row in trades], dtype=float)
    if len(values) == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_win_R": 0.0,
            "avg_loss_R": 0.0,
            "avg_win_loss_ratio": 0.0,
            "profit_factor": 0.0,
            "net_R": 0.0,
            "max_drawdown_R": 0.0,
            "expectancy_R": 0.0,
            "median_R": 0.0,
        }
    winners = values[values > 0]
    losers = values[values <= 0]
    win_sum = float(winners.sum())
    loss_sum = float(-losers.sum())
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    drawdown = peaks - equity
    avg_win = float(winners.mean()) if len(winners) else 0.0
    avg_loss = float(-losers.mean()) if len(losers) else 0.0
    return {
        "trades": int(len(values)),
        "wins": int(len(winners)),
        "losses": int(len(losers)),
        "win_rate": float(len(winners) / len(values)),
        "avg_win_R": avg_win,
        "avg_loss_R": avg_loss,
        "avg_win_loss_ratio": float(avg_win / avg_loss) if avg_loss > 0 else (999.0 if avg_win > 0 else 0.0),
        "profit_factor": float(win_sum / loss_sum) if loss_sum > 0 else (999.0 if win_sum > 0 else 0.0),
        "net_R": float(values.sum()),
        "max_drawdown_R": float(drawdown.max()) if len(drawdown) else 0.0,
        "expectancy_R": float(values.mean()),
        "median_R": float(np.median(values)),
    }


def remove_best_fraction(trades: pd.DataFrame, fraction: float = 0.10) -> tuple[pd.DataFrame, dict[str, Any]]:
    if trades.empty:
        return trades.copy(), {"removed_trades": 0, **metrics(trades)}
    remove_count = max(1, int(math.ceil(len(trades) * fraction)))
    ordered = trades.sort_values("net_r", ascending=False)
    remaining = ordered.iloc[remove_count:].copy()
    if "entry_time_utc" in remaining.columns:
        remaining = remaining.sort_values("entry_time_utc")
    remaining = remaining.reset_index(drop=True)
    removed = ordered.iloc[:remove_count]
    return remaining, {
        "removed_trades": int(remove_count),
        "removed_net_R": float(removed["net_r"].sum()),
        **metrics(remaining),
    }


def monthly_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["channel_id", "channel_label", "month"] + list(metrics(pd.DataFrame()).keys())
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (channel_id, channel_label, month), group in trades.groupby(["channel_id", "channel_label", "month"], sort=True):
        rows.append({"channel_id": channel_id, "channel_label": channel_label, "month": month, **metrics(group)})
    return pd.DataFrame(rows)


def phase_metrics(trades: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        channel_trades = trades.loc[trades["channel_id"] == spec.channel_id] if not trades.empty else pd.DataFrame()
        for phase_name in config["phases"]:
            phase_trades = channel_trades.loc[channel_trades["phase"] == phase_name] if not channel_trades.empty else pd.DataFrame()
            rows.append(
                {
                    "channel_id": spec.channel_id,
                    "channel_label": spec.label,
                    "phase": phase_name,
                    **metrics(phase_trades),
                }
            )
    return pd.DataFrame(rows)


def direction_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["channel_id", "channel_label", "direction"] + list(metrics(pd.DataFrame()).keys())
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (channel_id, channel_label, direction), group in trades.groupby(
        ["channel_id", "channel_label", "direction"], sort=True
    ):
        rows.append(
            {
                "channel_id": channel_id,
                "channel_label": channel_label,
                "direction": "LONG" if int(direction) == 1 else "SHORT",
                **metrics(group),
            }
        )
    return pd.DataFrame(rows)


def rolling_quarter_metrics(trades: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = ["channel_id", "channel_label", "window_start_month", "window_end_month"] + list(
        metrics(pd.DataFrame()).keys()
    )
    if trades.empty:
        return pd.DataFrame(columns=columns)
    all_months = month_range(str(config["start_month"]), str(config["end_month"]))
    rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        channel_trades = trades.loc[trades["channel_id"] == spec.channel_id]
        for index in range(2, len(all_months)):
            window = all_months[index - 2 : index + 1]
            subset = channel_trades.loc[channel_trades["month"].isin(window)]
            rows.append(
                {
                    "channel_id": spec.channel_id,
                    "channel_label": spec.label,
                    "window_start_month": window[0],
                    "window_end_month": window[-1],
                    **metrics(subset),
                }
            )
    return pd.DataFrame(rows)


def qualification_audit(channel_trades: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    threshold = config["research_candidate_thresholds"]
    base = metrics(channel_trades)
    _, robust = remove_best_fraction(channel_trades, float(threshold["remove_best_fraction"]))
    months = monthly_metrics(channel_trades)
    positive_months = int((months["net_R"] > 0).sum()) if not months.empty else 0
    active_months = int((months["trades"] > 0).sum()) if not months.empty else 0
    positive_total = float(months.loc[months["net_R"] > 0, "net_R"].sum()) if not months.empty else 0.0
    maximum_positive = float(months.loc[months["net_R"] > 0, "net_R"].max()) if positive_total > 0 else 0.0
    max_month_profit_share = maximum_positive / positive_total if positive_total > 0 else 1.0
    phase = channel_trades.groupby("phase")["net_r"].sum().to_dict() if not channel_trades.empty else {}
    positive_phases = int(sum(float(value) > 0 for value in phase.values()))
    historical = channel_trades.loc[channel_trades["phase"] == "historical_test"] if not channel_trades.empty else pd.DataFrame()
    historical_metrics = metrics(historical)
    quarters = rolling_quarter_metrics(channel_trades, config)
    if not quarters.empty and not channel_trades.empty:
        current_channel_id = str(channel_trades.iloc[0]["channel_id"])
        quarters = quarters.loc[quarters["channel_id"] == current_channel_id].reset_index(drop=True)
    positive_rolling_quarters = int((quarters["net_R"] > 0).sum()) if not quarters.empty else 0
    active_rolling_quarters = int((quarters["trades"] > 0).sum()) if not quarters.empty else 0
    worst_rolling_quarter_net_R = float(quarters["net_R"].min()) if not quarters.empty else 0.0

    checks = {
        "minimum_trades": base["trades"] >= int(threshold["minimum_trades"]),
        "minimum_win_rate": base["win_rate"] >= float(threshold["minimum_win_rate"]),
        "minimum_avg_win_loss_ratio": base["avg_win_loss_ratio"] >= float(threshold["minimum_avg_win_loss_ratio"]),
        "minimum_profit_factor": base["profit_factor"] >= float(threshold["minimum_profit_factor"]),
        "minimum_expectancy_R": base["expectancy_R"] >= float(threshold["minimum_expectancy_R"]),
        "maximum_drawdown_R": base["max_drawdown_R"] <= float(threshold["maximum_drawdown_R"]),
        "minimum_positive_months": positive_months >= int(threshold["minimum_positive_months"]),
        "best_trades_removed_still_profitable": robust["net_R"] > 0,
        "maximum_single_positive_month_share": max_month_profit_share <= float(threshold["maximum_single_positive_month_share"]),
        "minimum_positive_phases": positive_phases >= int(threshold["minimum_positive_phases"]),
        "historical_test_positive": historical_metrics["net_R"] > 0,
        "historical_test_profit_factor": historical_metrics["profit_factor"] >= float(threshold["minimum_historical_test_profit_factor"]),
        "minimum_positive_rolling_quarters": positive_rolling_quarters >= int(threshold["minimum_positive_rolling_quarters"]),
        "minimum_worst_rolling_quarter_net_R": worst_rolling_quarter_net_R >= float(threshold["minimum_worst_rolling_quarter_net_R"]),
    }
    return {
        "metrics": base,
        "best_fraction_removed": robust,
        "historical_test_metrics": historical_metrics,
        "active_months": active_months,
        "positive_months": positive_months,
        "positive_phases": positive_phases,
        "active_rolling_quarters": active_rolling_quarters,
        "positive_rolling_quarters": positive_rolling_quarters,
        "worst_rolling_quarter_net_R": worst_rolling_quarter_net_R,
        "max_single_positive_month_share": max_month_profit_share,
        "checks": checks,
        "research_candidate": bool(all(checks.values())),
        "qualified_for_live_trading": False,
        "historical_data_already_viewed": True,
    }


def benchmark_score(audit: dict[str, Any]) -> float:
    m = audit["metrics"]
    robust = audit["best_fraction_removed"]
    historical = audit["historical_test_metrics"]
    if m["trades"] == 0:
        return -1e9
    score = (
        18.0 * float(m["win_rate"])
        + 3.0 * min(float(m["avg_win_loss_ratio"]), 4.0)
        + 4.0 * min(float(m["profit_factor"]), 4.0)
        + float(m["net_R"])
        + 14.0 * float(m["expectancy_R"])
        + 1.25 * float(historical["net_R"])
        + 4.0 * min(float(historical["profit_factor"]), 3.0)
        + 0.35 * float(audit["positive_months"])
        + 0.50 * float(audit["positive_rolling_quarters"])
        + 0.75 * float(audit["positive_phases"])
        + 0.75 * float(robust["net_R"])
        + 0.50 * float(audit["worst_rolling_quarter_net_R"])
        - 0.45 * float(m["max_drawdown_R"])
    )
    if m["trades"] < 45:
        score -= (45 - m["trades"]) * 0.30
    return float(score)


def baseline_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    by_id = summary.set_index("channel_id")
    rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        rr_key = _rr_key(float(row.reward_risk_target))
        baseline_id = f"60m_trend_pullback_quality_rr{rr_key}"
        baseline = by_id.loc[baseline_id]
        baseline_trades = float(baseline["trades"])
        rows.append(
            {
                "channel_id": row.channel_id,
                "channel_label": row.channel_label,
                "profile": row.profile,
                "reward_risk_target": row.reward_risk_target,
                "is_baseline": row.is_baseline,
                "baseline_channel_id": baseline_id,
                "trades": row.trades,
                "retention_rate_vs_baseline": float(row.trades / baseline_trades) if baseline_trades > 0 else 0.0,
                "win_rate": row.win_rate,
                "avg_win_loss_ratio": row.avg_win_loss_ratio,
                "profit_factor": row.profit_factor,
                "net_R": row.net_R,
                "max_drawdown_R": row.max_drawdown_R,
                "historical_test_net_R": row.historical_test_net_R,
                "historical_test_profit_factor": row.historical_test_profit_factor,
                "best_10pct_removed_net_R": row.best_10pct_removed_net_R,
                "delta_trades": row.trades - baseline["trades"],
                "delta_win_rate": row.win_rate - baseline["win_rate"],
                "delta_avg_win_loss_ratio": row.avg_win_loss_ratio - baseline["avg_win_loss_ratio"],
                "delta_profit_factor": row.profit_factor - baseline["profit_factor"],
                "delta_net_R": row.net_R - baseline["net_R"],
                "delta_max_drawdown_R": row.max_drawdown_R - baseline["max_drawdown_R"],
                "delta_historical_test_net_R": row.historical_test_net_R - baseline["historical_test_net_R"],
                "delta_best_10pct_removed_net_R": row.best_10pct_removed_net_R - baseline["best_10pct_removed_net_R"],
                "improves_recent_period": row.historical_test_net_R > baseline["historical_test_net_R"],
                "improves_tail_robustness": row.best_10pct_removed_net_R > baseline["best_10pct_removed_net_R"],
                "improves_primary_robustness_goals": (
                    row.historical_test_net_R > baseline["historical_test_net_R"]
                    and row.best_10pct_removed_net_R > baseline["best_10pct_removed_net_R"]
                    and row.max_drawdown_R <= baseline["max_drawdown_R"]
                ),
            }
        )
    return pd.DataFrame(rows)


def run_benchmark(raw_5m: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    signal_map, signal_audit = build_all_signals(raw_5m)
    trade_frames: list[pd.DataFrame] = []
    qualification: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for spec in CHANNELS:
        trades = execute_channel(raw_5m, signal_map[spec.channel_id], spec, config)
        trade_frames.append(trades)
        audit = qualification_audit(trades, config)
        audit["benchmark_score"] = benchmark_score(audit)
        qualification[spec.channel_id] = audit
        summary_rows.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "timeframe_minutes": 60,
                "family": spec.family,
                "profile": spec.profile,
                "reward_risk_target": spec.reward_risk,
                "stop_atr_multiple": spec.stop_atr_multiple,
                "max_holding_hours": spec.max_holding_hours,
                "is_baseline": spec.is_baseline,
                **audit["metrics"],
                "active_months": audit["active_months"],
                "positive_months": audit["positive_months"],
                "positive_phases": audit["positive_phases"],
                "positive_rolling_quarters": audit["positive_rolling_quarters"],
                "worst_rolling_quarter_net_R": audit["worst_rolling_quarter_net_R"],
                "historical_test_trades": audit["historical_test_metrics"]["trades"],
                "historical_test_win_rate": audit["historical_test_metrics"]["win_rate"],
                "historical_test_profit_factor": audit["historical_test_metrics"]["profit_factor"],
                "historical_test_net_R": audit["historical_test_metrics"]["net_R"],
                "best_10pct_removed_net_R": audit["best_fraction_removed"]["net_R"],
                "max_single_positive_month_share": audit["max_single_positive_month_share"],
                "research_candidate": audit["research_candidate"],
                "benchmark_score": audit["benchmark_score"],
            }
        )

    nonempty_trade_frames = [frame for frame in trade_frames if not frame.empty]
    ledger = pd.concat(nonempty_trade_frames, ignore_index=True) if nonempty_trade_frames else pd.DataFrame()
    if not ledger.empty:
        ledger = ledger.sort_values(["channel_id", "entry_time_utc"]).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["research_candidate", "benchmark_score"], ascending=[False, False]
    ).reset_index(drop=True)
    comparison = baseline_comparison(summary)
    leader = None
    non_baseline = summary.loc[~summary["is_baseline"]]
    if not non_baseline.empty:
        leader = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in non_baseline.iloc[0].to_dict().items()
        }
    return {
        "signals": signal_map,
        "signal_audit": signal_audit,
        "trades": ledger,
        "summary": summary,
        "baseline_comparison": comparison,
        "monthly": monthly_metrics(ledger),
        "phases": phase_metrics(ledger, config),
        "directions": direction_metrics(ledger),
        "rolling_quarters": rolling_quarter_metrics(ledger, config),
        "qualification": qualification,
        "research_leader": leader,
    }


def synthetic_5m_data(rows: int = 60_000, seed: int = 1010) -> pd.DataFrame:
    if rows < 5000:
        raise ValueError("synthetic_5m_data requires at least 5000 rows")
    rng = np.random.default_rng(seed)
    time_index = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    regime_length = 1800
    drift = np.zeros(rows, dtype=float)
    for start in range(0, rows, regime_length):
        regime = (start // regime_length) % 4
        if regime == 0:
            drift[start : start + regime_length] = 0.00018
        elif regime == 1:
            drift[start : start + regime_length] = 0.0
        elif regime == 2:
            drift[start : start + regime_length] = -0.00018
        else:
            drift[start : start + regime_length] = 0.0
    volatility = np.where((np.arange(rows) // 600) % 3 == 0, 0.00045, 0.00095)
    returns = drift + rng.normal(0.0, volatility, rows)
    close = 40_000.0 * np.exp(np.cumsum(returns))
    open_price = np.r_[close[0], close[:-1]]
    spread = np.maximum(close * np.abs(rng.normal(0.00038, 0.00017, rows)), 1.0)
    high = np.maximum(open_price, close) + spread
    low = np.minimum(open_price, close) - spread
    volume = rng.lognormal(mean=8.0, sigma=0.45, size=rows)
    volume[(np.arange(rows) % 997) < 4] *= 3.0
    open_ms = (time_index.view("int64") // 1_000_000).astype(np.int64)
    return pd.DataFrame(
        {
            "open_time": open_ms,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": open_ms + STEP_MS - 1,
            "quote_volume": volume * close,
            "trade_count": np.full(rows, 100),
            "taker_buy_volume": volume * 0.5,
            "taker_buy_quote": volume * close * 0.5,
            "ignore": 0,
            "time": time_index,
        }
    )


def self_test(config: dict[str, Any]) -> None:
    assert len(CHANNELS) == 8
    assert all(spec.reward_risk in {2.5, 3.0} for spec in CHANNELS)
    assert all(spec.max_holding_hours > 0 for spec in CHANNELS)
    assert sum(spec.is_baseline for spec in CHANNELS) == 2
    raw = synthetic_5m_data(30_000, seed=321)
    hourly = resample_hourly(raw)
    assert len(hourly) == len(raw) // 12
    assert hourly.index[0] == raw.iloc[0]["time"] + pd.Timedelta(hours=1)
    signals, signal_audit = build_all_signals(raw)
    assert set(signals) == {spec.channel_id for spec in CHANNELS}
    assert len(signal_audit["channels"]) == 8
    frozen_times = set(pd.to_datetime(signals["60m_trend_pullback_quality_rr2_5"]["signal_time"], utc=True))
    for spec in CHANNELS:
        candidate_times = set(pd.to_datetime(signals[spec.channel_id].get("signal_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True))
        if not spec.is_baseline:
            assert candidate_times.issubset(frozen_times), spec.channel_id
    for profile in ("quality", "anti_exhaustion", "balanced", "persistence"):
        left = signals[f"60m_trend_pullback_{profile}_rr2_5"]
        right = signals[f"60m_trend_pullback_{profile}_rr3_0"]
        assert list(left.get("signal_time", [])) == list(right.get("signal_time", [])), profile
        assert list(left.get("direction", [])) == list(right.get("direction", [])), profile
    result = run_benchmark(raw, config)
    assert len(result["summary"]) == 8
    assert set(result["summary"]["channel_id"]) == {spec.channel_id for spec in CHANNELS}
    assert len(result["baseline_comparison"]) == 8
    assert len(result["rolling_quarters"]) == 8 * 16
    assert not result["trades"].empty, "Synthetic smoke should produce at least one trade"
    for channel_id, group in result["trades"].groupby("channel_id"):
        entries = pd.to_datetime(group["entry_time_utc"], utc=True)
        signals_at = pd.to_datetime(group["signal_time_utc"], utc=True)
        exits = pd.to_datetime(group["exit_time_utc"], utc=True)
        assert (entries >= signals_at).all(), channel_id
        assert (exits >= entries).all(), channel_id
        if len(group) > 1:
            assert (entries.iloc[1:].reset_index(drop=True) >= exits.iloc[:-1].reset_index(drop=True)).all(), channel_id
    sample = pd.DataFrame({"net_r": [2.0, -1.0, 2.0, -1.0]})
    m = metrics(sample)
    assert m["trades"] == 4 and m["wins"] == 2
    assert abs(m["net_R"] - 2.0) < 1e-12
    remaining, robust = remove_best_fraction(sample, 0.10)
    assert len(remaining) == 3 and robust["removed_trades"] == 1
    print("V102_ENGINE_SELF_TEST_OK")


__all__ = [
    "CHANNELS",
    "ChannelSpec",
    "load_official_5m_data",
    "run_benchmark",
    "self_test",
    "synthetic_5m_data",
]
