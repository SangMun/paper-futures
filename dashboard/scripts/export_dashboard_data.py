#!/usr/bin/env python3
"""Export a recent, read-only dashboard snapshot from Gate and bot state."""

from __future__ import annotations

import argparse
import csv
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
from indicators import oscillators  # noqa: E402
from indicators import regression_channel as rc  # noqa: E402
from indicators import volume as volume_indicator  # noqa: E402
from indicators import vwap as vwap_indicator  # noqa: E402
from core import candles as candle_utils  # noqa: E402
from config.settings import BTC_REGIME_25X, FEES  # noqa: E402
from engine.loop import build_strategies  # noqa: E402


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


def fetch_candles(
    client: httpx.Client,
    timeframe: str,
    contract: str = "BTC_USDT",
    limit: int = 560,
):
    response = client.get(
        GATE_CANDLES,
        params={"contract": contract, "interval": timeframe, "limit": limit},
    )
    response.raise_for_status()
    frame = candle_utils.to_dataframe(response.json())
    now = int(time.time())
    step = TIMEFRAMES[timeframe]
    return frame.loc[frame["ts"] + step <= now].tail(500).reset_index(drop=True)


def enrich(frame):
    channel = rc.channel(frame, length=100, width_mult=2.0)
    ema20 = ma.ema(frame, 20)
    ema60 = ma.ema(frame, 60)
    atr14 = atr_indicator.atr(frame, 14)
    rsi14 = oscillators.rsi(frame, 14)
    macd = oscillators.macd(frame, fast=12, slow=26, signal=9)
    sma20 = ma.sma(frame, 20)
    sma25 = ma.sma(frame, 25)
    sma50 = ma.sma(frame, 50)
    session_vwap = vwap_indicator.vwap(frame)
    rvol20 = volume_indicator.rvol(frame, 20)

    candles = []
    for index, row in frame.iterrows():
        candles.append(
            {
                "time": int(row["ts"]),
                "open": finite(row["open"]),
                "high": finite(row["high"]),
                "low": finite(row["low"]),
                "close": finite(row["close"]),
                "volume": finite(row["volume"]),
                "quoteVolume": finite(row["quote_volume"]),
                "mid": finite(channel["mid"].iloc[index]),
                "upper": finite(channel["upper"].iloc[index]),
                "lower": finite(channel["lower"].iloc[index]),
                "slope": finite(channel["slope"].iloc[index]),
                "r2": finite(channel["r2"].iloc[index]),
                "ema20": finite(ema20.iloc[index]),
                "ema60": finite(ema60.iloc[index]),
                "atr": finite(atr14.iloc[index]),
                "rsi14": finite(rsi14.iloc[index]),
                "macd": finite(macd["macd"].iloc[index]),
                "macdSignal": finite(macd["signal"].iloc[index]),
                "macdHist": finite(macd["hist"].iloc[index]),
                "sma20": finite(sma20.iloc[index]),
                "sma25": finite(sma25.iloc[index]),
                "sma50": finite(sma50.iloc[index]),
                "vwap": finite(session_vwap.iloc[index]),
                "rvol": finite(rvol20.iloc[index]),
            }
        )
    return candles


def channel_entry_offset() -> float:
    """실시간 봇에 등록된 회귀채널 전략의 진입 이격. 대시보드가 다른 값을 보여주면 안 된다."""

    for strategy in build_strategies()["BTC_USDT"]:
        if strategy.name == "btc_regression_channel":
            return float(strategy.entry_offset_pct)
    return 0.01


def strategy_snapshot(candles_by_tf: dict[str, list[dict]]) -> dict:
    offset = channel_entry_offset()
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
        entry = latest["upper"] * (1 + offset)
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
        entry = latest["lower"] * (1 - offset)
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
        "entryOffsetPct": offset,
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


def _number(row: dict, key: str, default: float = 0.0) -> float:
    value = finite(row.get(key))
    return default if value is None else value


def _entry_signals(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if record.get("action") == "entry":
            records.append(record)
    return records


def load_signal_events(path: Path, limit: int = 120) -> list[dict]:
    """최근 XRP 신호와 차단 사유를 대시보드용으로 정규화한다."""

    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if row.get("symbol") != "XRP_USDT" or row.get("action") not in {"entry", "skip"}:
            continue
        notes = row.get("notes") or []
        if isinstance(notes, str):
            notes = [notes]
        events.append(
            {
                "time": int(row.get("ts") or 0),
                "action": row.get("action"),
                "strategy": row.get("strategy") or "",
                "side": row.get("side"),
                "skipReason": row.get("skip_reason"),
                "entry": finite(row.get("entry")),
                "stop": finite(row.get("stop")),
                "target": finite(row.get("tp1")),
                "score": finite(row.get("score")),
                "reasons": row.get("reasons") or [],
                "indicators": row.get("indicators") or {},
                "detail": row.get("detail"),
                "notes": notes,
            }
        )
    return sorted(events, key=lambda event: event["time"], reverse=True)[:limit]


def xrp_pattern_monitors() -> list[dict]:
    """실제 엔진에 등록된 XRP 패턴 인스턴스만 노출한다."""

    monitors = []
    for strategy in build_strategies()["XRP_USDT"]:
        tf = strategy.tf
        family = strategy.name.removeprefix("xrp_patterns.").rsplit(".", 1)[0]
        monitors.append(
            {
                "name": strategy.name,
                "family": family,
                "timeframe": tf.entry,
                "contextTimeframe": tf.context,
                "confirmTimeframe": tf.confirm,
                "orderType": "limit" if getattr(strategy, "limit_entry", False) else "market",
                "trendAlign": bool(getattr(strategy, "trend_align", False)),
                "blockedUtcHours": sorted(getattr(strategy, "block_hours", ())),
            }
        )
    return monitors


def _matching_signal(row: dict, signals: list[dict], used: set[int]) -> dict | None:
    price = _number(row, "price")
    candidates = []
    for index, signal in enumerate(signals):
        if index in used:
            continue
        if any(signal.get(key) != row.get(key) for key in ("strategy", "symbol", "side")):
            continue
        signal_price = finite(signal.get("entry"))
        price_error = abs((signal_price or 0.0) - price) / max(abs(price), 1e-12)
        time_error = abs(int(signal.get("ts", 0)) - int(row.get("ts", 0)))
        if price_error <= 1e-5:
            candidates.append((price_error, time_error, index, signal))
    if not candidates:
        return None
    _, _, index, signal = min(candidates)
    used.add(index)
    return signal


def _exit_mode(events: list[dict]) -> str:
    reasons = [event["reason"] for event in events]
    if "tp1" in reasons and reasons[-1] == "trail":
        return "tp1_trail"
    if reasons[-1:] == ["tp1"]:
        return "take_profit"
    if reasons[-1:] == ["trail"]:
        return "trailing"
    if reasons[-1:] == ["stop"]:
        return "stop_loss"
    if reasons[-1:] == ["time"]:
        return "time_exit"
    return reasons[-1] if reasons else "open"


def load_trade_history(trade_path: Path, signal_path: Path, limit: int = 100) -> list[dict]:
    if not trade_path.exists():
        return []
    with trade_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    signals = _entry_signals(signal_path)
    used_signals: set[int] = set()
    open_trades: list[dict] = []
    completed: list[dict] = []

    for row in rows:
        if row.get("action") == "entry":
            signal = _matching_signal(row, signals, used_signals) or {}
            entry_order = "limit" if row.get("reason") == "limit" or row.get("note") == "maker" else "market"
            entry_fee_rate = FEES.maker_rate if entry_order == "limit" else FEES.taker_rate
            notional = _number(row, "notional")
            open_trades.append(
                {
                    "id": f"{row.get('symbol')}-{row.get('strategy')}-{row.get('ts')}",
                    "entryTime": int(row.get("ts") or 0),
                    "strategy": row.get("strategy", ""),
                    "symbol": row.get("symbol", ""),
                    "side": row.get("side", ""),
                    "entryPrice": _number(row, "price"),
                    "contracts": int(_number(row, "contracts")),
                    "notional": notional,
                    "leverage": int(_number(row, "leverage", 1.0)),
                    "stop": _number(row, "stop"),
                    "tp1": _number(row, "tp1"),
                    "entryOrder": entry_order,
                    "entryFee": notional * entry_fee_rate,
                    "score": _number(signal, "score"),
                    "reasons": signal.get("reasons") or [],
                    "context": signal.get("indicators") or {},
                    "exitEvents": [],
                    "closedRatio": 0.0,
                }
            )
            continue

        if row.get("action") != "exit":
            continue
        candidates = [
            trade for trade in open_trades
            if trade["strategy"] == row.get("strategy")
            and trade["symbol"] == row.get("symbol")
            and trade["side"] == row.get("side")
            and trade["contracts"] == int(_number(row, "contracts"))
        ]
        if not candidates:
            continue
        trade = candidates[0]
        ratio = _number(row, "ratio", 1.0)
        event = {
            "time": int(row.get("ts") or 0),
            "reason": row.get("reason") or "exit",
            "price": _number(row, "price"),
            "ratio": ratio,
            "fee": _number(row, "fee"),
            "pnl": _number(row, "pnl"),
            "note": row.get("note") or "",
        }
        trade["exitEvents"].append(event)
        trade["closedRatio"] += ratio
        if trade["closedRatio"] < 0.999999:
            continue

        events = trade["exitEvents"]
        ratio_sum = sum(item["ratio"] for item in events) or 1.0
        trade["exitTime"] = events[-1]["time"]
        trade["exitPrice"] = sum(item["price"] * item["ratio"] for item in events) / ratio_sum
        trade["exitMode"] = _exit_mode(events)
        trade["fees"] = trade["entryFee"] + sum(item["fee"] for item in events)
        trade["pnl"] = sum(item["pnl"] for item in events) - trade["entryFee"]
        trade["balanceAfter"] = _number(row, "balance")
        trade["holdingSeconds"] = max(0, trade["exitTime"] - trade["entryTime"])
        trade.pop("closedRatio", None)
        open_trades.remove(trade)
        completed.append(trade)

    return sorted(completed, key=lambda trade: trade["exitTime"], reverse=True)[:limit]


# 계좌 이름 -> 대시보드 라벨. 폴더 이름은 deploy/paper-futures@.service 인스턴스와 같다.
ACCOUNTS = {
    "btcchan": "BTC 회귀채널",
    "btc25x": "BTC 25×",
    "xrp": "XRP 패턴",
}
INITIAL_BALANCE = 1000.0


# 계좌 -> 그 계좌가 돌리는 전략 이름 접두사. 분리 전 합산 계좌(2026-08-07~09-15)의
# 체결을 전략별로 갈라 각 계좌 이력에 이어 붙이는 데 쓴다.
ACCOUNT_STRATEGY_PREFIX = {
    "btcchan": "btc_regression_channel",
    "btc25x": "btc_regime_25x",
    "xrp": "xrp_patterns",
}


def load_account(root: Path, name: str) -> dict:
    """계좌 폴더가 있으면 그것, 없으면 루트(단일 계좌 시절)로 대신한다.

    계좌 폴더가 있을 때 루트의 trades.csv는 분리 전 합산 계좌 기록이다.
    이 계좌 전략의 체결만 골라 뒤에 붙인다 — 잔고는 새 계좌 것이지만
    전략의 성적은 분리 전부터 이어서 봐야 한다.
    """

    directory = root / name if (root / name).is_dir() else root
    trades = load_trade_history(directory / "trades.csv", directory / "signals.jsonl")
    if directory != root:
        prefix = ACCOUNT_STRATEGY_PREFIX[name]
        legacy = [
            dict(trade, legacy=True)
            for trade in load_trade_history(root / "trades.csv", root / "signals.jsonl", limit=1000)
            if trade["strategy"].startswith(prefix)
        ]
        trades = sorted(trades + legacy, key=lambda t: t["exitTime"], reverse=True)
    return {
        "name": name,
        "dir": directory,
        "state": load_state(directory / "state.json"),
        "trades": trades,
    }


def account_summary(acct: dict) -> dict:
    """계좌 카드 요약. 잔고와 짝이 맞도록 분리 후 체결만 센다 — 분리 전 이력은
    trades 목록에만 남긴다."""

    state = acct["state"]
    balance = finite(state.get("balance")) or 0.0
    current = [t for t in acct["trades"] if not t.get("legacy")]
    return {
        "name": acct["name"],
        "label": ACCOUNTS[acct["name"]],
        "balance": balance,
        "dayStartBalance": finite(state.get("day_start_balance")) or 0.0,
        "dayKey": state.get("day_key", ""),
        "initialBalance": INITIAL_BALANCE,
        "returnPct": (balance / INITIAL_BALANCE - 1.0) * 100.0,
        "closedTrades": len(current),
        "openPositions": len(state.get("positions", [])),
        "realizedPnl": sum(t.get("pnl", 0.0) for t in current),
        "legacyTrades": len(acct["trades"]) - len(current),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public" / "btc-dashboard.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="계좌별 폴더(btcchan/btc25x/xrp)가 있는 데이터 루트. "
             "폴더가 없으면 루트의 단일 계좌 파일로 대신한다.",
    )
    args = parser.parse_args()

    with httpx.Client(timeout=20.0, headers={"User-Agent": "paper-futures-dashboard/0.1"}) as client:
        candles_by_tf = {
            timeframe: enrich(fetch_candles(client, timeframe, "BTC_USDT"))
            for timeframe in TIMEFRAMES
        }
        xrp_candles_by_tf = {
            timeframe: enrich(fetch_candles(client, timeframe, "XRP_USDT"))
            for timeframe in TIMEFRAMES
            if timeframe != "1m"
        }

    accounts = {name: load_account(args.data_dir, name) for name in ACCOUNTS}
    state = accounts["btcchan"]["state"]
    state_25x = accounts["btc25x"]["state"]
    state_xrp = accounts["xrp"]["state"]
    minute = candles_by_tf["1m"]
    latest = minute[-1]
    previous = minute[-2]
    change_pct = (latest["close"] / previous["close"] - 1.0) * 100
    xrp_minute = xrp_candles_by_tf["5m"]
    xrp_latest = xrp_minute[-1]
    xrp_previous = xrp_minute[-2]
    xrp_change_pct = (xrp_latest["close"] / xrp_previous["close"] - 1.0) * 100

    positions = [
        position
        for position in state.get("positions", [])
        if position.get("symbol") == "BTC_USDT"
        and position.get("strategy") == "btc_regression_channel"
    ]
    positions_25x = [
        position
        for position in state_25x.get("positions", [])
        if position.get("symbol") == "BTC_USDT"
        and position.get("strategy") == "btc_regime_25x"
    ]
    xrp_positions = [
        position
        for position in state_xrp.get("positions", [])
        if position.get("symbol") == "XRP_USDT"
        and str(position.get("strategy", "")).startswith("xrp_patterns.")
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
        "account": account_summary(accounts["btcchan"]),
        # 2026-09-15부터 계좌를 전략별 프로세스 셋으로 나눴다. 페이지마다
        # 자기 계좌를 보여주고, 세 계좌 요약은 모든 페이지 상단에 같이 띄운다.
        "accounts": {name: account_summary(acct) for name, acct in accounts.items()},
        "positions": positions,
        "positions25x": positions_25x,
        "xrpMarket": {
            "symbol": "XRP_USDT",
            "displaySymbol": "XRP / USDT",
            "price": xrp_latest["close"],
            "changePct": xrp_change_pct,
        },
        "xrpPositions": xrp_positions,
        "xrpCandles": xrp_candles_by_tf,
        "xrpPatterns": xrp_pattern_monitors(),
        "xrpSignals": load_signal_events(accounts["xrp"]["dir"] / "signals.jsonl"),
        "strategy25x": {
            "channelLength": BTC_REGIME_25X.channel_length,
            "channelWidthMult": BTC_REGIME_25X.channel_width_mult,
            "channelR2Min": BTC_REGIME_25X.channel_r2_min,
            "channelMinMovePct": BTC_REGIME_25X.channel_min_move_pct,
            "maxEntryDistancePct": BTC_REGIME_25X.max_entry_distance_pct,
            "limitTtlBars": BTC_REGIME_25X.limit_ttl_bars,
            "leverage": BTC_REGIME_25X.leverage,
            "marginFraction": BTC_REGIME_25X.margin_fraction,
            "accountTakeProfitPct": BTC_REGIME_25X.account_take_profit_pct,
            "targetMarginRoe": (
                BTC_REGIME_25X.account_take_profit_pct
                / BTC_REGIME_25X.margin_fraction
            ),
            "makerRate": FEES.maker_rate,
            "maxDailyEntries": BTC_REGIME_25X.max_daily_entries,
            "consecutiveLossLimit": BTC_REGIME_25X.consecutive_loss_limit,
            "dailyEntries": state_25x.get("daily_entries", {}).get("btc_regime_25x", 0),
            "consecutiveLosses": state_25x.get("consecutive_losses", {}).get("btc_regime_25x", 0),
            "pausedUntil": state_25x.get("paused_until", {}).get("btc_regime_25x", 0),
        },
        "trades": sorted(
            (t for acct in accounts.values() for t in acct["trades"]),
            key=lambda t: t["exitTime"], reverse=True,
        ),
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
