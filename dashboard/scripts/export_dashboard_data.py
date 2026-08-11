#!/usr/bin/env python3
"""Export a recent, read-only dashboard snapshot from Gate and bot state."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from indicators import atr as atr_indicator  # noqa: E402
from indicators import ma  # noqa: E402
from indicators import regression_channel as rc  # noqa: E402
from core import candles as candle_utils  # noqa: E402


GATE_CANDLES = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
TIMEFRAMES = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fetch_candles(client: httpx.Client, timeframe: str, limit: int = 420):
    response = client.get(
        GATE_CANDLES,
        params={"contract": "BTC_USDT", "interval": timeframe, "limit": limit},
    )
    response.raise_for_status()
    frame = candle_utils.to_dataframe(response.json())
    now = int(time.time())
    step = TIMEFRAMES[timeframe]
    return frame.loc[frame["ts"] + step <= now].tail(360).reset_index(drop=True)


def enrich(frame):
    channel = rc.channel(frame, length=100, width_mult=2.0)
    ema20 = ma.ema(frame, 20)
    ema60 = ma.ema(frame, 60)
    atr14 = atr_indicator.atr(frame, 14)

    candles = []
    for index, row in frame.iterrows():
        candles.append(
            {
                "time": int(row["ts"]),
                "open": finite(row["open"]),
                "high": finite(row["high"]),
                "low": finite(row["low"]),
                "close": finite(row["close"]),
                "volume": finite(row["quote_volume"]),
                "mid": finite(channel["mid"].iloc[index]),
                "upper": finite(channel["upper"].iloc[index]),
                "lower": finite(channel["lower"].iloc[index]),
                "slope": finite(channel["slope"].iloc[index]),
                "r2": finite(channel["r2"].iloc[index]),
                "ema20": finite(ema20.iloc[index]),
                "ema60": finite(ema60.iloc[index]),
                "atr": finite(atr14.iloc[index]),
            }
        )
    return candles


def strategy_snapshot(candles_by_tf: dict[str, list[dict]]) -> dict:
    hourly = candles_by_tf["1h"]
    major = candles_by_tf["4h"]
    latest = hourly[-1]
    previous = hourly[-2]
    latest_major = major[-1]

    slope = latest["slope"] or 0.0
    signal = "waiting"
    side = None
    entry = None
    target = None

    if (
        slope < 0
        and latest["upper"] is not None
        and previous["upper"] is not None
        and latest["high"] > latest["upper"]
        and previous["high"] <= previous["upper"]
    ):
        signal = "short_watch"
        side = "short"
        entry = latest["upper"] * 1.02
        target = latest["upper"]
    elif (
        slope > 0
        and latest["lower"] is not None
        and previous["lower"] is not None
        and latest["low"] < latest["lower"]
        and previous["low"] >= previous["lower"]
    ):
        signal = "long_watch"
        side = "long"
        entry = latest["lower"] * 0.98
        target = latest["lower"]

    close = latest_major["close"]
    fast = latest_major["ema20"]
    slow = latest_major["ema60"]
    regime = "none"
    if all(value is not None for value in (close, fast, slow)):
        if close > fast > slow:
            regime = "bull"
        elif close < fast < slow:
            regime = "bear"

    return {
        "signal": signal,
        "side": side,
        "entry": finite(entry),
        "target": finite(target),
        "slope": finite(slope),
        "r2": latest["r2"],
        "atr": latest["atr"],
        "upper": latest["upper"],
        "lower": latest["lower"],
        "majorRegime": regime,
        "entryOffsetPct": 0.02,
        "channelLength": 100,
        "channelWidthMult": 2.0,
        "executionTimeframe": "1h",
        "contextTimeframe": "4h",
    }


def load_state(path: Path) -> dict:
    if not path.exists():
        return {
            "balance": 1000.0,
            "day_start_balance": 1000.0,
            "positions": [],
            "day_key": "",
        }
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-file",
        type=Path,
        default=ROOT / "data" / "state.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public" / "btc-dashboard.json",
    )
    args = parser.parse_args()

    with httpx.Client(timeout=20.0, headers={"User-Agent": "xrp-paper-bot-dashboard/0.1"}) as client:
        candles_by_tf = {
            timeframe: enrich(fetch_candles(client, timeframe))
            for timeframe in TIMEFRAMES
        }

    state = load_state(args.state_file)
    minute = candles_by_tf["1m"]
    latest = minute[-1]
    previous = minute[-2]
    change_pct = (latest["close"] / previous["close"] - 1.0) * 100

    positions = [
        position
        for position in state.get("positions", [])
        if position.get("symbol") == "BTC_USDT"
        and position.get("strategy") == "btc_regression_channel"
    ]

    payload = {
        "generatedAt": int(time.time()),
        "source": "Gate.io public candles + paper-bot state snapshot",
        "market": {
            "symbol": "BTC_USDT",
            "displaySymbol": "BTC / USDT",
            "price": latest["close"],
            "changePct": change_pct,
        },
        "strategy": strategy_snapshot(candles_by_tf),
        "account": {
            "balance": finite(state.get("balance")) or 0.0,
            "dayStartBalance": finite(state.get("day_start_balance")) or 0.0,
            "dayKey": state.get("day_key", ""),
        },
        "positions": positions,
        "candles": candles_by_tf,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
