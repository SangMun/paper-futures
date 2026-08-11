"""BTC 25배 단기 회귀채널 전략 회귀 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import FEES
from config.symbols import BTC
from strategies.base import Prepared
from strategies.btc_regime_25x import BtcRegime25x, _maker_target_move


T0 = 1_700_000_000
FIVE_MIN = 300


def _frame(n: int, base: float = 100.0) -> pd.DataFrame:
    close = np.linspace(base, base + n - 1, n)
    ts = np.arange(n, dtype="int64") * FIVE_MIN + T0
    return pd.DataFrame(
        {
            "ts": ts,
            "dt": pd.to_datetime(ts, unit="s", utc=True),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 100.0),
            "quote_volume": close * 100.0,
        }
    )


def _prepared(
    side: str,
    *,
    r2: float = 0.8,
    distance: float = 0.002,
    regime: str | None = None,
) -> Prepared:
    n = 80
    df = _frame(n)
    close = float(df["close"].iloc[-1])
    slope = 0.03 if side == "long" else -0.03
    entry = close * (1.0 - distance) if side == "long" else close * (1.0 + distance)
    mid = close
    lower = entry if side == "long" else close * 0.997
    upper = close * 1.003 if side == "long" else entry

    def full(value):
        return pd.Series([value] * n, index=df.index, dtype="float64")

    # 4시간봉 국면. 기본은 진입 방향과 일치시켜 국면 필터를 통과시킨다.
    if regime is None:
        regime = "bull" if side == "long" else "bear"

    return Prepared(
        frames={"5m": df},
        series={
            "5m": {
                "major_regime": pd.Series([regime] * n, index=df.index, dtype="object"),
                "atr": full(0.8),
                "channel_mid": full(mid),
                "channel_upper": full(upper),
                "channel_lower": full(lower),
                "channel_slope": full(slope),
                "channel_r2": full(r2),
                "channel_deviation": full(0.5),
                "channel_move_pct": full(0.008),
            }
        },
    )


def test_BTC_엔진은_5분마다_판단한다():
    assert BTC.driver_tf == "5m"


@pytest.mark.parametrize("side", ["long", "short"])
def test_25배_증거금20퍼_목표폭은_메이커비용을_포함한다(side):
    move = _maker_target_move(side, 0.20, 25)
    m = FEES.maker_rate
    if side == "long":
        net_per_notional = move - m - m * (1 + move)
    else:
        net_per_notional = move - m - m * (1 - move)

    assert move == pytest.approx(0.0078, rel=0.001)
    assert net_per_notional * 25 == pytest.approx(0.20)


@pytest.mark.parametrize(
    ("side", "reason"),
    [("long", "rising_15m_channel"), ("short", "falling_15m_channel")],
)
def test_채널끝에_메이커지정가를_둔다(side, reason):
    signal = BtcRegime25x().generate(_prepared(side), 79)[0]

    assert signal.side == side
    assert signal.order_type == "limit"
    assert signal.ttl_bars == 3
    assert reason in signal.reasons
    assert signal.position_policy.fixed_leverage == 25
    assert signal.position_policy.margin_fraction == pytest.approx(0.10)
    assert signal.position_policy.tp1_is_maker
    assert signal.position_policy.max_hold_bars == 12
    assert not signal.position_policy.time_exit_enabled
    assert signal.position_policy.partial_exit_ratio == pytest.approx(1.0)
    target_distance = abs(signal.target_hint - signal.entry) / signal.entry
    stop_distance = abs(signal.stop_hint - signal.entry) / signal.entry
    assert signal.position_policy.stop_pct_from_entry == pytest.approx(target_distance)
    assert stop_distance == pytest.approx(target_distance)
    assert signal.position_policy.max_daily_entries == 12
    assert signal.position_policy.consecutive_loss_limit == 3
    assert signal.context["target_account_profit_pct"] == pytest.approx(0.02)
    assert signal.context["target_margin_roe"] == pytest.approx(0.20)
    if side == "long":
        assert signal.entry == pytest.approx(signal.context["channel_lower"])
        assert signal.target_hint > signal.entry
    else:
        assert signal.entry == pytest.approx(signal.context["channel_upper"])
        assert signal.target_hint < signal.entry


def test_횡보채널은_거래하지_않는다():
    assert BtcRegime25x().generate(_prepared("long", r2=0.4), 79) == []


def test_채널끝이_현재가에서_너무멀면_주문하지_않는다():
    assert BtcRegime25x().generate(
        _prepared("long", distance=0.006), 79
    ) == []


def test_잘못된_목표입력은_거절한다():
    with pytest.raises(ValueError):
        _maker_target_move("up", 0.20, 25)
    with pytest.raises(ValueError):
        _maker_target_move("long", 0.0, 25)


# ------------------------------------------------------- 4시간봉 대세 국면 필터


@pytest.mark.parametrize("side", ["long", "short"])
def test_대세_역행_진입은_막는다(side):
    """상승장 숏·하락장 롱은 949일 실측 217건 -2,962였다."""

    counter = "bear" if side == "long" else "bull"
    assert BtcRegime25x().generate(_prepared(side, regime=counter), 79) == []


@pytest.mark.parametrize("side", ["long", "short"])
def test_횡보장은_기본적으로_제외하고_옵션으로_연다(side):
    """486건 합계 +3.8에 앞뒤 절반 부호가 갈린다 — 낙폭만 늘린다."""

    prepared = _prepared(side, regime="range")
    assert BtcRegime25x().generate(prepared, 79) == []
    assert BtcRegime25x(allow_range=True).generate(prepared, 79)


@pytest.mark.parametrize("side", ["long", "short"])
def test_상위봉이_없으면_진입하지_않는다(side):
    """국면 필터 없는 원본은 실측 1,260건 -377이다.

    4시간봉 수집이 실패했을 때 조용히 그 동작으로 돌아가면 안 된다.
    """

    assert BtcRegime25x().generate(_prepared(side, regime="none"), 79) == []


@pytest.mark.parametrize("side", ["long", "short"])
def test_국면_필터를_끄면_역행도_통과한다(side):
    """필터의 효과를 재려면 끌 수 있어야 한다."""

    counter = "bear" if side == "long" else "bull"
    prepared = _prepared(side, regime=counter)
    assert BtcRegime25x(regime_filter=False).generate(prepared, 79)


@pytest.mark.parametrize("side", ["long", "short"])
def test_흐릿한_채널은_진입하지_않는다(side):
    """R² 하한(0.68) 아래는 잡음에 그은 직선이라 지정가가 역선택된다."""

    assert BtcRegime25x().generate(_prepared(side, r2=0.60), 79) == []
    assert BtcRegime25x().generate(_prepared(side, r2=0.75), 79)
