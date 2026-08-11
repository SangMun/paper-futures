"""BTC 반전·추세 지속 후보의 미래참조와 체결 집계 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import FEES, LEVERAGE_CAP
from strategies.base import Prepared, Signal
from strategies.btc_pullback_reversal import BtcPullbackReversal
from strategies.btc_trend_continuation import BtcTrendContinuation
from tools.replay_btc_candidates import completed_positions


T0 = 1_700_000_000
HOUR = 3600


def bars(rows) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [T0 + i * HOUR for i in range(len(rows))],
            "dt": pd.to_datetime(
                [T0 + i * HOUR for i in range(len(rows))], unit="s", utc=True
            ),
            "open": [float(r[0]) for r in rows],
            "high": [float(r[1]) for r in rows],
            "low": [float(r[2]) for r in rows],
            "close": [float(r[3]) for r in rows],
            "volume": [100.0] * len(rows),
            "quote_volume": [10_000.0] * len(rows),
        }
    )


def prepared(df: pd.DataFrame, regime: str = "bull") -> Prepared:
    n = len(df)
    return Prepared(
        frames={"1h": df},
        series={
            "1h": {
                "ctx_regime": pd.Series([regime] * n),
                "atr": pd.Series([1.0] * n),
            }
        },
    )


def setup_signal(side: str = "long") -> Signal:
    return Signal(
        strategy="btc_trend",
        symbol="BTC_USDT",
        side=side,
        entry=100.0,
        stop_hint=98.0 if side == "long" else 102.0,
        target_hint=105.0 if side == "long" else 95.0,
        score=3.0,
        reasons=["confirmed"],
        context={"atr": 1.0},
    )


def test_반전후보는_돌파봉마감뒤_리테스트가격에_지정가를_둔다(monkeypatch):
    df = bars(
        [
            (99, 100, 98, 99),
            (99, 100, 98, 99),
            (99, 100, 98, 99),
            (99, 101, 98.5, 100),       # setup
            (100, 103, 99.5, 102),      # setup 고점 101 종가 돌파
        ]
    )
    strategy = BtcPullbackReversal()
    monkeypatch.setattr(
        strategy,
        "_strict_setups",
        lambda _p, at: [(setup_signal(), "btc_trend")] if at == 3 else [],
    )
    monkeypatch.setattr(strategy._trend, "_stop_hint", lambda *_: 98.0)
    monkeypatch.setattr(strategy._trend, "_target_hint", lambda *_: 105.0)

    signals = strategy.generate(prepared(df), 4)

    assert len(signals) == 1
    assert signals[0].strategy == "btc_pullback_reversal"
    assert signals[0].entry == pytest.approx(101.0)
    assert signals[0].order_type == "limit"
    assert signals[0].ttl_bars == 3
    assert "limit_at_break_level" in signals[0].reasons
    assert signals[0].max_tranches == 1


def test_과거_돌파를_다음봉에서_다시_주문하지_않는다(monkeypatch):
    df = bars(
        [
            (99, 100, 98, 99),
            (99, 100, 98, 99),
            (99, 100, 98, 99),
            (99, 101, 98.5, 100),
            (100, 103, 99.5, 102),
            (102, 103, 101.2, 102.5),
        ]
    )
    strategy = BtcPullbackReversal()
    monkeypatch.setattr(
        strategy,
        "_strict_setups",
        lambda _p, at: [(setup_signal(), "btc_trend")] if at == 3 else [],
    )

    assert strategy.generate(prepared(df), 5) == []


def test_추세후보는_현재봉을_제외한_20봉_고점을_종가로_돌파한다(monkeypatch):
    rows = [(100, 101 + i * 0.1, 98, 100) for i in range(20)]
    rows.append((100, 104, 99, 103.1))  # 직전 고점 102.9 위에서 마감
    df = bars(rows)
    strategy = BtcTrendContinuation()
    monkeypatch.setattr(strategy._base, "_stop_hint", lambda *_: 98.0)
    monkeypatch.setattr(strategy._base, "_target_hint", lambda *_: 106.0)

    signals = strategy.generate(prepared(df), 20)

    assert len(signals) == 1
    assert signals[0].side == "long"
    assert signals[0].context["channel_level"] == pytest.approx(102.9)
    assert signals[0].entry == pytest.approx(102.9)
    assert signals[0].order_type == "limit"
    assert signals[0].ttl_bars == 3


def test_추세후보는_고가만_넘고_종가가_못넘으면_진입하지_않는다():
    rows = [(100, 101 + i * 0.1, 98, 100) for i in range(20)]
    rows.append((100, 104, 99, 102.8))
    df = bars(rows)

    assert BtcTrendContinuation().generate(prepared(df), 20) == []


def test_후보전략도_레버리지캡이_있다():
    assert LEVERAGE_CAP["btc_pullback_reversal"] == 3.0
    assert LEVERAGE_CAP["btc_trend_continuation"] == 3.0


def test_부분익절은_포지션승률에서_한건으로_합친다():
    trades = [
        {
            "ts": "1", "strategy": "btc_trend_continuation", "side": "long",
            "action": "entry", "notional": "1000", "pnl": "", "reason": "",
        },
        {
            "ts": "2", "strategy": "btc_trend_continuation", "side": "long",
            "action": "exit", "notional": "", "pnl": "5", "reason": "tp1",
        },
        {
            "ts": "3", "strategy": "btc_trend_continuation", "side": "long",
            "action": "exit", "notional": "", "pnl": "2", "reason": "trail",
        },
    ]

    result = completed_positions(trades)

    assert len(result) == 1
    assert result[0].pnl == pytest.approx(7 - 1000 * FEES.taker_rate)
