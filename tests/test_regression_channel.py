"""롤링 회귀채널 계산과 BTC 지정가 신호 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators import regression_channel as rc
from indicators.base import IndicatorError
from strategies.base import Prepared
from strategies.btc_regression_channel import BtcRegressionChannel


T0 = 1_700_000_000
HOUR = 3600


def frame(closes) -> pd.DataFrame:
    closes = [float(x) for x in closes]
    return pd.DataFrame(
        {
            "ts": [T0 + i * HOUR for i in range(len(closes))],
            "dt": pd.to_datetime(
                [T0 + i * HOUR for i in range(len(closes))], unit="s", utc=True
            ),
            "open": closes,
            "high": [x + 0.5 for x in closes],
            "low": [x - 0.5 for x in closes],
            "close": closes,
            "volume": [100.0] * len(closes),
            "quote_volume": [10_000.0] * len(closes),
        }
    )


def signal_prepared(side: str) -> Prepared:
    n = 101
    closes = [100.0] * n
    series = {
        "reg_slope": [np.nan] * n,
        "reg_upper": [np.nan] * n,
        "reg_lower": [np.nan] * n,
        "reg_r2": [np.nan] * n,
        "atr": [1.0] * n,
    }
    if side == "short":
        closes[99], closes[100] = 100.0, 101.2
        series["reg_slope"][99:] = [-0.1, -0.1]
        series["reg_upper"][99:] = [101.0, 101.0]
        series["reg_lower"][99:] = [95.0, 95.0]
    else:
        closes[99], closes[100] = 100.0, 98.8
        series["reg_slope"][99:] = [0.1, 0.1]
        series["reg_upper"][99:] = [105.0, 105.0]
        series["reg_lower"][99:] = [99.0, 99.0]
    series["reg_r2"][99:] = [0.7, 0.7]
    candles = frame(closes)
    if side == "short":
        candles.loc[99, "high"] = 100.5
        candles.loc[100, "high"] = 101.5
    else:
        candles.loc[99, "low"] = 99.5
        candles.loc[100, "low"] = 98.5
    return Prepared(
        frames={"1h": candles},
        series={"1h": {k: pd.Series(v) for k, v in series.items()}},
    )


def test_직선은_기울기와_r2가_정확하다():
    df = frame(5.0 + 2.0 * np.arange(140))
    out = rc.channel(df, length=100, width_mult=2.0)

    assert out["slope"].iloc[-1] == pytest.approx(2.0)
    assert out["mid"].iloc[-1] == pytest.approx(df["close"].iloc[-1])
    assert out["r2"].iloc[-1] == pytest.approx(1.0)
    assert out["deviation"].iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_채널은_미래봉을_보지_않는다():
    df = frame(100 + np.sin(np.arange(180) / 7))
    changed = df.copy()
    changed.loc[151:, "close"] = changed.loc[151:, "close"] * 10

    original = rc.channel(df, length=100)
    future_changed = rc.channel(changed, length=100)

    for key in original.columns:
        assert future_changed[key].iloc[150] == pytest.approx(original[key].iloc[150])


def test_회귀기간과_폭은_검증한다():
    df = frame(range(20))
    with pytest.raises(IndicatorError):
        rc.channel(df, length=1)
    with pytest.raises(IndicatorError):
        rc.channel(df, length=10, width_mult=0)


def test_하락채널_상단이탈은_1퍼위_숏지정가():
    signal = BtcRegressionChannel(entry_offset_pct=0.01).generate(
        signal_prepared("short"), 100
    )[0]

    assert signal.side == "short"
    assert signal.entry == pytest.approx(102.01)
    assert signal.target_hint == pytest.approx(101.0)
    assert signal.order_type == "limit"
    assert signal.max_tranches == 1


def test_상승채널_하단이탈은_1퍼아래_롱지정가():
    signal = BtcRegressionChannel(entry_offset_pct=0.01).generate(
        signal_prepared("long"), 100
    )[0]

    assert signal.side == "long"
    assert signal.entry == pytest.approx(98.01)
    assert signal.target_hint == pytest.approx(99.0)
    assert signal.order_type == "limit"


def test_이미_지정가보다_멀리_가버렸으면_시장가로_쫓지_않는다():
    prepared = signal_prepared("short")
    prepared.frames["1h"].loc[100, "close"] = 104.0

    assert BtcRegressionChannel().generate(prepared, 100) == []


def test_진입이격_범위를_검증한다():
    with pytest.raises(ValueError):
        BtcRegressionChannel(entry_offset_pct=-0.01)
    with pytest.raises(ValueError):
        BtcRegressionChannel(entry_offset_pct=0.2)


def test_이격_0이면_이탈봉_종가_시장가():
    signal = BtcRegressionChannel(entry_offset_pct=0).generate(
        signal_prepared("long"), 100
    )[0]
    assert signal.order_type == "market"
    assert signal.ttl_bars == 0
    assert signal.entry == float(signal_prepared("long").frame("1h")["close"].iloc[100])


def test_방향을_따로_끌수있다():
    assert BtcRegressionChannel(direction="long").generate(
        signal_prepared("short"), 100
    ) == []
    assert BtcRegressionChannel(direction="short").generate(
        signal_prepared("long"), 100
    ) == []
    with pytest.raises(ValueError):
        BtcRegressionChannel(direction="up")


def test_동적목표는_지정한_확정봉까지만_본다():
    df = frame(100 + np.sin(np.arange(140) / 5))
    changed = df.copy()
    changed.loc[121:, "close"] = changed.loc[121:, "close"] * 5
    strategy = BtcRegressionChannel()

    original = strategy.target_before_bar({"1h": df}, 120, "short")
    future_changed = strategy.target_before_bar({"1h": changed}, 120, "short")

    assert future_changed == pytest.approx(original)


# ------------------------------------------------------------------ 국면 필터


def test_국면_필터는_대세_역행과_횡보숏을_막는다():
    """949일 실측: 상승숏 -49.58, 하락롱 -35.91, 횡보숏 -89.16.
    셋을 빼면 103건 -6.53이 41건 +168.12가 된다."""

    from strategies.btc_regression_channel import BtcRegressionChannel

    s = BtcRegressionChannel(regime_filter=True)
    assert s.regime_filter

    # 필터가 없으면 통과시킨다 (기본값)
    assert not BtcRegressionChannel().regime_filter


def test_추세_일치_자리는_리스크를_키운다():
    """건당 기대값이 +25.5 vs 횡보장 +0.37로 약 69배 차이였다."""

    from engine.loop import build_strategies

    s = build_strategies()["BTC_USDT"][0]
    assert s.aligned_risk_mult == 2.0
    assert s.regime_filter


def test_신호_리스크_배수는_상한이_있다():
    """표본이 적은 구간이 우연이었을 때 손실도 같은 배수로 커진다."""

    from risk import sizing as sz

    assert sz.MAX_SIGNAL_RISK_MULT == 2.0
    with pytest.raises(sz.SizingError, match="신호 리스크 배수"):
        sz.size_position("BTC_USDT", "long", 60_000.0, 58_500.0, 1_000.0,
                         "btc_regression_channel", risk_multiplier=5.0)
