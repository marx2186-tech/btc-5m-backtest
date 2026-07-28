from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache_v10_0"
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
    timeframe_minutes: int
    strategy: str
    label: str
    stop_atr_multiple: float
    reward_risk: float
    max_holding_signal_bars: int


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("15m_trend_pullback", 15, "trend_pullback", "15分钟趋势回踩", 1.20, 2.0, 24),
    ChannelSpec("15m_range_reversal", 15, "range_reversal", "15分钟区间反转", 1.10, 2.0, 24),
    ChannelSpec("15m_volatility_breakout", 15, "volatility_breakout", "15分钟波动突破", 1.25, 2.0, 24),
    ChannelSpec("30m_trend_pullback", 30, "trend_pullback", "30分钟趋势回踩", 1.20, 2.0, 24),
    ChannelSpec("30m_range_reversal", 30, "range_reversal", "30分钟区间反转", 1.10, 2.0, 24),
    ChannelSpec("30m_volatility_breakout", 30, "volatility_breakout", "30分钟波动突破", 1.25, 2.0, 24),
    ChannelSpec("60m_trend_pullback", 60, "trend_pullback", "1小时趋势回踩", 1.20, 2.0, 24),
    ChannelSpec("60m_range_reversal", 60, "range_reversal", "1小时区间反转", 1.10, 2.0, 24),
    ChannelSpec("60m_volatility_breakout", 60, "volatility_breakout", "1小时波动突破", 1.25, 2.0, 24),
    ChannelSpec("1h_15m_5m_trend", 15, "mtf_trend_pullback", "1小时趋势＋15分钟结构＋5分钟执行", 1.20, 2.0, 24),
)


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
                headers={"User-Agent": "btc-multitimeframe-benchmark-v10-0/1.0"},
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
        raise RuntimeError("V10.0 official 5m data audit failed: " + json.dumps(audit, ensure_ascii=False))

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


def adx(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=frame.index)
    atr_value = atr(frame, length).replace(0, np.nan)
    plus_di = 100.0 * rma(plus_dm, length) / atr_value
    minus_di = 100.0 * rma(minus_dm, length) / atr_value
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return rma(dx, length)


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length).replace(0, np.nan)
    relative_strength = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)


def resample_ohlcv(raw_5m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes not in {15, 30, 60}:
        raise ValueError(f"Unsupported benchmark timeframe: {minutes}")
    indexed = raw_5m.set_index("time")[["open", "high", "low", "close", "volume"]]
    frame = indexed.resample(
        f"{minutes}min",
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
    expected_count = minutes // 5
    count = indexed["close"].resample(
        f"{minutes}min",
        origin="start_day",
        closed="left",
        label="right",
    ).count()
    frame["source_5m_rows"] = count
    frame = frame.loc[frame["source_5m_rows"] == expected_count].dropna().copy()
    frame.index.name = "signal_time"
    return frame


def add_timeframe_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["atr14"] = atr(x, 14)
    x["adx14"] = adx(x, 14)
    x["rsi14"] = rsi(x["close"], 14)
    x["channel_high_20"] = x["high"].shift(1).rolling(20).max()
    x["channel_low_20"] = x["low"].shift(1).rolling(20).min()
    x["volume_mean_20"] = x["volume"].shift(1).rolling(20).mean()
    x["atr_ratio"] = x["atr14"] / x["close"].replace(0, np.nan)
    x["atr_ratio_q30"] = x["atr_ratio"].shift(1).rolling(100).quantile(0.30)
    x["ema_separation_atr"] = (x["ema20"] - x["ema50"]).abs() / x["atr14"].replace(0, np.nan)
    return x


def _signal_rows(mask: pd.Series, direction: int, x: pd.DataFrame, spec: ChannelSpec) -> pd.DataFrame:
    selected = x.loc[mask.fillna(False), ["close", "atr14", "adx14", "rsi14", "volume"]].copy()
    if selected.empty:
        return pd.DataFrame()
    selected = selected.reset_index()
    selected["direction"] = int(direction)
    selected["channel_id"] = spec.channel_id
    selected["channel_label"] = spec.label
    selected["strategy"] = spec.strategy
    selected["timeframe_minutes"] = int(spec.timeframe_minutes)
    selected["stop_atr_multiple"] = float(spec.stop_atr_multiple)
    selected["reward_risk"] = float(spec.reward_risk)
    selected["max_holding_signal_bars"] = int(spec.max_holding_signal_bars)
    return selected


def generate_single_timeframe_signals(frame: pd.DataFrame, spec: ChannelSpec) -> pd.DataFrame:
    x = add_timeframe_features(frame)
    if spec.strategy == "trend_pullback":
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
    elif spec.strategy == "range_reversal":
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
    elif spec.strategy == "volatility_breakout":
        squeeze = x["atr_ratio"].shift(1) <= x["atr_ratio_q30"]
        volume_expansion = x["volume"] >= x["volume_mean_20"] * 1.15
        long_mask = squeeze & volume_expansion & (x["close"] > x["channel_high_20"])
        short_mask = squeeze & volume_expansion & (x["close"] < x["channel_low_20"])
    else:
        raise ValueError(f"Unsupported single-timeframe strategy: {spec.strategy}")

    frames = [_signal_rows(long_mask, 1, x, spec), _signal_rows(short_mask, -1, x, spec)]
    signals = pd.concat([part for part in frames if not part.empty], ignore_index=True) if any(not p.empty for p in frames) else pd.DataFrame()
    if signals.empty:
        return signals
    return signals.sort_values(["signal_time", "direction"]).drop_duplicates("signal_time", keep="first").reset_index(drop=True)


def generate_mtf_signals(frame_15m: pd.DataFrame, frame_60m: pd.DataFrame, spec: ChannelSpec) -> pd.DataFrame:
    x15 = add_timeframe_features(frame_15m)
    x60 = add_timeframe_features(frame_60m)
    trend = x60[["ema20", "ema50", "adx14"]].copy()
    trend["trend_direction"] = np.where(
        (trend["ema20"] > trend["ema50"]) & (trend["adx14"] >= 18.0),
        1,
        np.where((trend["ema20"] < trend["ema50"]) & (trend["adx14"] >= 18.0), -1, 0),
    )
    left = x15.reset_index().sort_values("signal_time")
    right = trend.reset_index().rename(columns={"signal_time": "hour_close_time"}).sort_values("hour_close_time")
    merged = pd.merge_asof(
        left,
        right[["hour_close_time", "trend_direction"]],
        left_on="signal_time",
        right_on="hour_close_time",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("signal_time")

    long_mask = (
        (merged["trend_direction"] == 1)
        & (merged["low"] <= merged["ema20"])
        & (merged["close"] > merged["ema20"])
        & (merged["close"] > merged["open"])
    )
    short_mask = (
        (merged["trend_direction"] == -1)
        & (merged["high"] >= merged["ema20"])
        & (merged["close"] < merged["ema20"])
        & (merged["close"] < merged["open"])
    )
    frames = [_signal_rows(long_mask, 1, merged, spec), _signal_rows(short_mask, -1, merged, spec)]
    signals = pd.concat([part for part in frames if not part.empty], ignore_index=True) if any(not p.empty for p in frames) else pd.DataFrame()
    if signals.empty:
        return signals
    return signals.sort_values(["signal_time", "direction"]).drop_duplicates("signal_time", keep="first").reset_index(drop=True)


def build_all_signals(raw_5m: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames = {minutes: resample_ohlcv(raw_5m, minutes) for minutes in (15, 30, 60)}
    signal_map: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    for spec in CHANNELS:
        if spec.strategy == "mtf_trend_pullback":
            signals = generate_mtf_signals(frames[15], frames[60], spec)
        else:
            signals = generate_single_timeframe_signals(frames[spec.timeframe_minutes], spec)
        signal_map[spec.channel_id] = signals
        audit_rows.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "timeframe_minutes": spec.timeframe_minutes,
                "strategy": spec.strategy,
                "signals": int(len(signals)),
                "long_signals": int((signals["direction"] == 1).sum()) if not signals.empty else 0,
                "short_signals": int((signals["direction"] == -1).sum()) if not signals.empty else 0,
            }
        )
    resample_audit = {
        str(minutes): {
            "rows": int(len(frame)),
            "first_bar_close_utc": frame.index.min().isoformat() if not frame.empty else None,
            "last_bar_close_utc": frame.index.max().isoformat() if not frame.empty else None,
            "source_rows_per_bar": minutes // 5,
        }
        for minutes, frame in frames.items()
    }
    return signal_map, {"resample": resample_audit, "channels": audit_rows}


def execute_channel(
    raw_5m: pd.DataFrame,
    signals: pd.DataFrame,
    spec: ChannelSpec,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "channel_id",
        "channel_label",
        "strategy",
        "timeframe_minutes",
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
        "month",
        "phase",
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
    max_holding_5m = spec.max_holding_signal_bars * (spec.timeframe_minutes // 5)
    trades: list[dict[str, Any]] = []
    next_free_index = 0

    for row in signals.sort_values("signal_time").itertuples(index=False):
        signal_time = pd.Timestamp(row.signal_time)
        signal_time64 = signal_time.to_datetime64()
        entry_i = int(np.searchsorted(times, signal_time64, side="left"))
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
        trades.append(
            {
                "channel_id": spec.channel_id,
                "channel_label": spec.label,
                "strategy": spec.strategy,
                "timeframe_minutes": spec.timeframe_minutes,
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
                "signal_adx": float(row.adx14) if math.isfinite(float(row.adx14)) else np.nan,
                "signal_rsi": float(row.rsi14) if math.isfinite(float(row.rsi14)) else np.nan,
                "month": entry_time.strftime("%Y-%m"),
                "phase": phase_for_time(entry_time, config),
            }
        )
        next_free_index = exit_i + 1

    return pd.DataFrame(trades, columns=columns)


def phase_for_time(timestamp: pd.Timestamp, config: dict[str, Any]) -> str:
    ts = pd.Timestamp(timestamp)
    for phase_name, phase in config["phases"].items():
        start = pd.Timestamp(phase["start_utc"])
        end = pd.Timestamp(phase["end_utc"])
        if start <= ts <= end:
            return phase_name
    return "outside_phase"


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
    channel_lookup = {spec.channel_id: spec.label for spec in CHANNELS}
    for spec in CHANNELS:
        channel_trades = trades.loc[trades["channel_id"] == spec.channel_id] if not trades.empty else pd.DataFrame()
        for phase_name in config["phases"]:
            phase_trades = channel_trades.loc[channel_trades["phase"] == phase_name] if not channel_trades.empty else pd.DataFrame()
            rows.append(
                {
                    "channel_id": spec.channel_id,
                    "channel_label": channel_lookup[spec.channel_id],
                    "phase": phase_name,
                    **metrics(phase_trades),
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

    checks = {
        "minimum_trades": base["trades"] >= int(threshold["minimum_trades"]),
        "minimum_profit_factor": base["profit_factor"] >= float(threshold["minimum_profit_factor"]),
        "minimum_expectancy_R": base["expectancy_R"] >= float(threshold["minimum_expectancy_R"]),
        "maximum_drawdown_R": base["max_drawdown_R"] <= float(threshold["maximum_drawdown_R"]),
        "minimum_positive_months": positive_months >= int(threshold["minimum_positive_months"]),
        "best_trades_removed_still_profitable": robust["net_R"] > 0,
        "maximum_single_positive_month_share": max_month_profit_share <= float(threshold["maximum_single_positive_month_share"]),
        "minimum_positive_phases": positive_phases >= int(threshold["minimum_positive_phases"]),
    }
    return {
        "metrics": base,
        "best_fraction_removed": robust,
        "active_months": active_months,
        "positive_months": positive_months,
        "positive_phases": positive_phases,
        "max_single_positive_month_share": max_month_profit_share,
        "checks": checks,
        "research_candidate": bool(all(checks.values())),
        "qualified_for_live_trading": False,
        "historical_data_already_viewed": True,
    }


def benchmark_score(audit: dict[str, Any]) -> float:
    m = audit["metrics"]
    robust = audit["best_fraction_removed"]
    if m["trades"] == 0:
        return -1e9
    pf_component = min(float(m["profit_factor"]), 4.0)
    score = (
        float(m["net_R"])
        + 12.0 * float(m["expectancy_R"])
        + 2.0 * pf_component
        + 0.25 * float(audit["positive_months"])
        + 0.50 * float(audit["positive_phases"])
        + 0.50 * float(robust["net_R"])
        - 0.35 * float(m["max_drawdown_R"])
    )
    if m["trades"] < 30:
        score -= (30 - m["trades"]) * 0.20
    return float(score)


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
                "timeframe_minutes": spec.timeframe_minutes,
                "strategy": spec.strategy,
                **audit["metrics"],
                "active_months": audit["active_months"],
                "positive_months": audit["positive_months"],
                "positive_phases": audit["positive_phases"],
                "best_10pct_removed_net_R": audit["best_fraction_removed"]["net_R"],
                "max_single_positive_month_share": audit["max_single_positive_month_share"],
                "research_candidate": audit["research_candidate"],
                "benchmark_score": audit["benchmark_score"],
            }
        )

    ledger = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    if not ledger.empty:
        ledger = ledger.sort_values(["channel_id", "entry_time_utc"]).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["research_candidate", "benchmark_score"], ascending=[False, False]
    ).reset_index(drop=True)
    if summary.empty:
        leader = None
    else:
        leader = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in summary.iloc[0].to_dict().items()
        }
    return {
        "signals": signal_map,
        "signal_audit": signal_audit,
        "trades": ledger,
        "summary": summary,
        "monthly": monthly_metrics(ledger),
        "phases": phase_metrics(ledger, config),
        "qualification": qualification,
        "research_leader": leader,
    }


def synthetic_5m_data(rows: int = 60_000, seed: int = 1000) -> pd.DataFrame:
    if rows < 5000:
        raise ValueError("synthetic_5m_data requires at least 5000 rows")
    rng = np.random.default_rng(seed)
    time_index = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    regime_length = 1800
    drift = np.zeros(rows, dtype=float)
    for start in range(0, rows, regime_length):
        regime = (start // regime_length) % 4
        if regime == 0:
            drift[start : start + regime_length] = 0.00016
        elif regime == 1:
            drift[start : start + regime_length] = 0.0
        elif regime == 2:
            drift[start : start + regime_length] = -0.00016
        else:
            drift[start : start + regime_length] = 0.0
    volatility = np.where((np.arange(rows) // 600) % 3 == 0, 0.00045, 0.0009)
    returns = drift + rng.normal(0.0, volatility, rows)
    close = 40_000.0 * np.exp(np.cumsum(returns))
    open_price = np.r_[close[0], close[:-1]]
    spread = np.maximum(close * np.abs(rng.normal(0.00035, 0.00015, rows)), 1.0)
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
    raw = synthetic_5m_data(20_000, seed=321)
    frame_15 = resample_ohlcv(raw, 15)
    assert len(frame_15) == len(raw) // 3
    assert frame_15.index[0] == raw.iloc[0]["time"] + pd.Timedelta(minutes=15)
    signals, signal_audit = build_all_signals(raw)
    assert set(signals) == {spec.channel_id for spec in CHANNELS}
    assert len(signal_audit["channels"]) == 10
    result = run_benchmark(raw, config)
    assert len(result["summary"]) == 10
    assert set(result["summary"]["channel_id"]) == {spec.channel_id for spec in CHANNELS}
    assert not result["trades"].empty, "Synthetic smoke should produce at least one trade"
    for channel_id, group in result["trades"].groupby("channel_id"):
        entries = pd.to_datetime(group["entry_time_utc"], utc=True)
        exits = pd.to_datetime(group["exit_time_utc"], utc=True)
        assert (exits >= entries).all(), channel_id
        if len(group) > 1:
            assert (entries.iloc[1:].reset_index(drop=True) >= exits.iloc[:-1].reset_index(drop=True)).all(), channel_id
    sample = pd.DataFrame({"net_r": [2.0, -1.0, 2.0, -1.0]})
    m = metrics(sample)
    assert m["trades"] == 4 and m["wins"] == 2
    assert abs(m["net_R"] - 2.0) < 1e-12
    remaining, robust = remove_best_fraction(sample, 0.10)
    assert len(remaining) == 3 and robust["removed_trades"] == 1
    print("V100_ENGINE_SELF_TEST_OK")


__all__ = [
    "CHANNELS",
    "ChannelSpec",
    "load_official_5m_data",
    "run_benchmark",
    "self_test",
    "synthetic_5m_data",
]
