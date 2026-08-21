"""틱 단위 청산 판정.

봉 마감을 기다리지 않고 진행 중인 봉의 누적 고저로 판정하는 경로.
확정봉 경로(process_bar)와 결과가 어긋나지 않는지도 여기서 본다.
"""

from __future__ import annotations

import pytest

from core import live
from risk import exits as ex


ATR = 1.0


def plan(stop=90.0, tp1=120.0, partial=0.5, trail=1.5, max_bars=40, time_exit=True):
    return ex.ExitPlan(
        stop=stop,
        tp1=tp1,
        partial_ratio=partial,
        trail_atr_mult=trail,
        max_hold_bars=max_bars,
        time_exit_enabled=time_exit,
    )


def long_state(**kw):
    return ex.PositionState(side="long", entry=100.0, plan=plan(**kw))


def short_state(**kw):
    return ex.PositionState(
        side="short", entry=100.0, plan=plan(stop=110.0, tp1=80.0, **kw)
    )


# ------------------------------------------------------------------ 손절 감지


def test_봉_마감_전에_손절을_잡는다():
    """핵심. 저가가 손절을 찍으면 봉이 안 끝나도 청산한다."""

    state = long_state()

    event = ex.process_tick(
        state, open_=100.0, high=100.5, low=89.0, last=95.0, atr=ATR
    )

    assert event is not None
    assert event.reason == ex.EXIT_STOP
    # 종가(95)가 아니라 손절가(90)에 체결. 거래소 스탑 주문과 같은 취급.
    assert event.price == pytest.approx(90.0 * (1 - ex.FEES.slippage_rate))


def test_현재가만_보면_놓치는_꼬리를_잡는다():
    """10초 전 100, 지금 99인데 그 사이 89를 찍은 경우.

    현재가만 보면 아무 일도 없었던 것처럼 보인다.
    누적 저가를 넘기기 때문에 잡힌다.
    """

    state = long_state()

    event = ex.process_tick(
        state, open_=100.0, high=100.0, low=89.0, last=99.0, atr=ATR
    )

    assert event is not None
    assert event.reason == ex.EXIT_STOP


def test_손절_안_닿으면_None():
    state = long_state()

    event = ex.process_tick(
        state, open_=100.0, high=101.0, low=95.0, last=97.0, atr=ATR
    )

    assert event is None


def test_숏_손절():
    state = short_state()

    event = ex.process_tick(
        state, open_=100.0, high=111.0, low=99.0, last=105.0, atr=ATR
    )

    assert event is not None
    assert event.reason == ex.EXIT_STOP
    assert event.price == pytest.approx(110.0 * (1 + ex.FEES.slippage_rate))


# ------------------------------------------------------------------ 멱등성


def test_같은_봉을_여러_번_넘겨도_한_번만_청산된다():
    """폴링은 같은 진행 중인 봉을 반복해서 본다.

    누적 극값이 손절에 안 닿는 한 몇 번을 넘겨도 청산이 생기면 안 된다.
    """

    state = long_state()

    for _ in range(30):
        event = ex.process_tick(
            state, open_=100.0, high=102.0, low=95.0, last=101.0, atr=ATR
        )
        assert event is None


def test_누적_고저가_커져도_손절_전까지는_조용하다():
    state = long_state()
    lows = [99.0, 97.0, 95.0, 93.0, 91.0]

    for low in lows:
        assert ex.process_tick(
            state, open_=100.0, high=102.0, low=low, last=low + 0.5, atr=ATR
        ) is None

    # 이제 손절을 넘어선다
    event = ex.process_tick(
        state, open_=100.0, high=102.0, low=89.9, last=90.5, atr=ATR
    )
    assert event is not None
    assert event.reason == ex.EXIT_STOP


# ------------------------------------------------------------------ 시간 손절


def test_틱에서는_시간_손절이_나지_않는다():
    """bars_held는 봉 시계만 센다.

    틱에서도 판정하면 같은 봉 안에서 수십 번 평가되어
    시간 손절이 봉 경계와 무관하게 터진다.
    """

    state = long_state(max_bars=10)
    state.bars_held = 50            # 이미 한도를 넘긴 상태

    event = ex.process_tick(
        state, open_=100.0, high=101.0, low=99.0, last=100.5, atr=ATR
    )

    assert event is None


def test_봉_시계에서는_시간_손절이_난다():
    """같은 상태라도 process_bar는 시간 손절을 낸다. 경로 차이 확인."""

    state = long_state(max_bars=10)
    state.bars_held = 50

    event = ex.process_bar(
        state, open_=100.0, high=101.0, low=99.0, close=100.5, atr=ATR
    )

    assert event is not None
    assert event.reason == ex.EXIT_TIME


def test_bars_held는_틱에서_증가하지_않는다():
    state = long_state()
    before = state.bars_held

    for _ in range(10):
        ex.process_tick(state, open_=100.0, high=101.0, low=99.0, last=100.0, atr=ATR)

    assert state.bars_held == before


# ------------------------------------------------------------------ 1차 익절


def test_틱에서_1차_익절():
    state = long_state()

    event = ex.process_tick(
        state, open_=100.0, high=121.0, low=99.0, last=119.0, atr=ATR
    )

    assert event is not None
    assert event.reason == ex.EXIT_TP1
    assert event.price == pytest.approx(120.0)
    assert event.ratio == pytest.approx(0.5)


def test_손절과_익절이_같은_구간에_있으면_손절_우선():
    """봉 안의 순서를 모르므로 불리한 쪽을 택한다. 봉 경로와 같은 규칙."""

    state = long_state()

    event = ex.process_tick(
        state, open_=100.0, high=121.0, low=89.0, last=100.0, atr=ATR
    )

    assert event.reason == ex.EXIT_STOP


# ------------------------------------------------------------------ 트레일링


def test_트레일은_틱마다_갱신된다():
    """봉 경로는 5분에 한 번 움직인다. 틱 경로는 매번 따라간다."""

    state = long_state()
    state.tp1_filled = True
    state.stop = 100.0

    # 저가는 손절 위에 둔다. 닿으면 트레일을 갱신하기 전에 청산돼버린다.
    ex.process_tick(state, open_=101.0, high=110.0, low=100.5, last=109.0, atr=ATR)
    first = state.stop

    ex.process_tick(state, open_=109.0, high=115.0, low=108.6, last=114.0, atr=ATR)
    second = state.stop

    assert first == pytest.approx(110.0 - 1.5 * ATR)
    assert second == pytest.approx(115.0 - 1.5 * ATR)
    assert second > first


def test_새_손절은_다음_틱부터_유효하다():
    """같은 틱에서 트레일을 올리고 그 손절로 청산하면 룩어헤드다."""

    state = long_state()
    state.tp1_filled = True
    state.stop = 100.0

    # 고가 110 -> 트레일 108.5. 저가 109는 108.5에 안 닿는다.
    event = ex.process_tick(
        state, open_=109.0, high=110.0, low=109.0, last=109.5, atr=ATR
    )

    assert event is None
    assert state.stop == pytest.approx(108.5)

    # 다음 틱에서야 108.5가 적용된다
    event = ex.process_tick(
        state, open_=109.5, high=110.0, low=108.0, last=108.2, atr=ATR
    )
    assert event is not None
    assert event.reason == ex.EXIT_TRAIL


def test_트레일은_되돌아가지_않는다():
    state = long_state()
    state.tp1_filled = True
    state.stop = 100.0

    ex.process_tick(state, open_=101.0, high=115.0, low=100.5, last=114.0, atr=ATR)
    high_water = state.stop

    # 가격이 밀려도 손절은 그대로 (새 손절 113.5에는 닿지 않는 선까지)
    ex.process_tick(state, open_=114.0, high=115.0, low=113.6, last=113.8, atr=ATR)

    assert state.stop == pytest.approx(high_water)


def test_tp1_전에는_트레일이_작동하지_않는다():
    state = long_state()
    original = state.stop

    ex.process_tick(state, open_=100.0, high=118.0, low=99.0, last=117.0, atr=ATR)

    assert state.stop == pytest.approx(original)


# ------------------------------------------------------------------ 갭


def test_봉_시가가_이미_손절을_넘었으면_갭_체결():
    state = long_state()

    event = ex.process_tick(
        state, open_=85.0, high=86.0, low=84.0, last=85.5, atr=ATR
    )

    assert event.note == "갭 손절"
    assert event.price == pytest.approx(85.0 * (1 - ex.FEES.slippage_rate))


# ------------------------------------------------------------------ 입력 검증


def test_누적_고저가_모순이면_예외():
    state = long_state()

    with pytest.raises(ex.ExitError):
        ex.process_tick(state, open_=100.0, high=99.0, low=101.0, last=100.0, atr=ATR)


def test_현재가가_누적_범위_밖이면_예외():
    """last가 bar_high를 넘으면 누적값이 아니라는 뜻이다."""

    state = long_state()

    with pytest.raises(ex.ExitError):
        ex.process_tick(state, open_=100.0, high=101.0, low=99.0, last=105.0, atr=ATR)


# ------------------------------------------------------------------ 봉/틱 일치


def test_지나간_저점으로_트레일_손절이_터지지_않는다():
    """회귀 테스트. 누적 고저를 그대로 넘기면 여기서 잘못 청산된다.

    봉 시가 100 -> 99까지 눌림 -> 115까지 상승.
    99는 트레일이 올라가기 전에 찍힌 값이라 체결될 수 없다.
    """

    state = long_state()
    state.tp1_filled = True
    state.stop = 98.0

    q1 = live.LiveQuote(
        bar_ts=1000, observed_ts=1010,
        bar_open=100.0, bar_high=100.5, bar_low=99.0, last=100.0,
    )
    w1 = live.window_since(q1, prev_bar_ts=0, prev_high=0, prev_low=0, prev_price=0)
    assert ex.process_tick(
        state, open_=w1.open, high=w1.high, low=w1.low, last=w1.last, atr=ATR
    ) is None
    assert state.stop == pytest.approx(99.0)        # 트레일이 99로 올라감

    # 저가는 그대로 99, 가격은 114로 급등
    q2 = live.LiveQuote(
        bar_ts=1000, observed_ts=1020,
        bar_open=100.0, bar_high=115.0, bar_low=99.0, last=114.0,
    )
    w2 = live.window_since(
        q2, prev_bar_ts=1000, prev_high=q1.bar_high, prev_low=q1.bar_low,
        prev_price=q1.last,
    )
    event = ex.process_tick(
        state, open_=w2.open, high=w2.high, low=w2.low, last=w2.last, atr=ATR
    )

    assert event is None, "이미 지나간 저점으로 청산되면 안 된다"
    assert state.stop == pytest.approx(113.5)


def test_틱을_다_흘려도_봉_하나와_결론이_같다():
    """5분봉 하나 = 틱 여러 개. 같은 가격 흐름이면 같은 청산이 나와야 한다.

    이게 어긋나면 리플레이(봉)와 실시간(틱)의 성과가 갈린다.
    """

    flow = [                    # (누적high, 누적low, last)
        (100.5, 99.5, 100.0),
        (101.0, 99.0, 100.8),
        (101.0, 95.0, 96.0),
        (101.0, 89.5, 90.2),    # 여기서 손절
    ]

    tick_state = long_state()
    tick_event = None
    for high, low, last in flow:
        tick_event = ex.process_tick(
            tick_state, open_=100.0, high=high, low=low, last=last, atr=ATR
        )
        if tick_event:
            break

    bar_state = long_state()
    bar_event = ex.process_bar(
        bar_state, open_=100.0, high=101.0, low=89.5, close=90.2, atr=ATR
    )

    assert tick_event is not None and bar_event is not None
    assert tick_event.reason == bar_event.reason == ex.EXIT_STOP
    assert tick_event.price == pytest.approx(bar_event.price)
