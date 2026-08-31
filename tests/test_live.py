"""진행 중인 봉 스냅샷과 틱 구간 계산."""

from __future__ import annotations

import pytest

from core import candles as cd
from core import live
from tests.test_candles import STEP, T0, raw_candle


def frame(*rows):
    return cd.to_dataframe(list(rows))


def quote(bar_ts=T0, o=100.0, h=101.0, l=99.0, last=100.5, observed=None):
    return live.LiveQuote(
        bar_ts=bar_ts,
        observed_ts=observed if observed is not None else bar_ts + 60,
        bar_open=o, bar_high=h, bar_low=l, last=last,
    )


# ------------------------------------------------------------------ 추출


def test_진행_중인_봉을_꺼낸다():
    df = frame(
        raw_candle(T0),
        raw_candle(T0 + STEP, o=1.0, h=1.2, l=0.95, c=1.1),
    )

    q = live.from_frame(df, "5m", now_ts=T0 + STEP + 60)

    assert q is not None
    assert q.bar_ts == T0 + STEP
    assert (q.bar_open, q.bar_high, q.bar_low, q.last) == (1.0, 1.2, 0.95, 1.1)


def test_마지막_봉이_이미_마감됐으면_None():
    """봉이 막 넘어간 직후. 그 구간은 봉 시계가 확정봉으로 처리한다."""

    df = frame(raw_candle(T0), raw_candle(T0 + STEP))

    assert live.from_frame(df, "5m", now_ts=T0 + 2 * STEP + 1) is None


def test_경계_시각은_마감으로_본다():
    df = frame(raw_candle(T0))

    assert live.from_frame(df, "5m", now_ts=T0 + STEP) is None
    assert live.from_frame(df, "5m", now_ts=T0 + STEP - 1) is not None


def test_빈_캔들():
    assert live.from_frame(cd.to_dataframe([]), "5m", now_ts=T0) is None
    assert live.from_frame(None, "5m", now_ts=T0) is None


# ------------------------------------------------------------------ 첫 관측


def test_첫_관측은_봉_전체가_구간():
    q = quote(o=100.0, h=102.0, l=98.0, last=101.0)

    w = live.window_since(q, prev_bar_ts=0, prev_high=0.0, prev_low=0.0, prev_price=0.0)

    assert (w.open, w.high, w.low, w.last) == (100.0, 102.0, 98.0, 101.0)


def test_봉이_바뀌면_새_봉_전체가_구간():
    q = quote(bar_ts=T0 + STEP, o=100.0, h=102.0, l=98.0, last=101.0)

    w = live.window_since(
        q, prev_bar_ts=T0, prev_high=105.0, prev_low=95.0, prev_price=100.0
    )

    assert (w.open, w.high, w.low, w.last) == (100.0, 102.0, 98.0, 101.0)


# ------------------------------------------------------------------ 핵심: 지나간 극값 제외


def test_이미_지나간_저점은_구간에_들어오지_않는다():
    """이 테스트가 설계의 핵심이다.

    봉 전반에 99를 찍고 지금은 114다. 누적 저가는 계속 99로 남아 있지만
    그 99는 직전 관측에서 이미 확인했다. 이번 구간에 다시 넣으면
    그 사이에 올라간 트레일 손절에 잘못 걸린다.
    """

    q = quote(o=100.0, h=115.0, l=99.0, last=114.0)

    w = live.window_since(
        q, prev_bar_ts=T0, prev_high=100.5, prev_low=99.0, prev_price=100.0
    )

    assert w.low == 100.0           # 99가 아니라 관측 가격 사이의 최저
    assert w.high == 115.0          # 신고점은 이번 구간에서 찍혔다


def test_누적_저가가_새로_깨지면_구간에_들어온다():
    q = quote(o=100.0, h=101.0, l=95.0, last=96.0)

    w = live.window_since(
        q, prev_bar_ts=T0, prev_high=101.0, prev_low=99.0, prev_price=99.5
    )

    assert w.low == 95.0            # 신저점이므로 이번 구간


def test_움직임이_없으면_구간도_한_점():
    q = quote(o=100.0, h=101.0, l=99.0, last=100.0)

    w = live.window_since(
        q, prev_bar_ts=T0, prev_high=101.0, prev_low=99.0, prev_price=100.0
    )

    assert w.high == w.low == 100.0


def test_구간_양_끝은_반드시_포함된다():
    q = quote(o=100.0, h=101.0, l=99.0, last=100.9)

    w = live.window_since(
        q, prev_bar_ts=T0, prev_high=101.0, prev_low=99.0, prev_price=99.2
    )

    assert w.low <= 99.2 and w.high >= 100.9


# ------------------------------------------------------------------ 봉 마감 꼬리


def test_봉_마감_직전_구간을_버리지_않는다():
    """폴링 간격 때문에 봉의 마지막 몇 초는 어떤 틱도 보지 못한다."""

    row = {"open": 100.0, "high": 103.0, "low": 94.0, "close": 95.0}

    w = live.tail_window(row, prev_high=101.0, prev_low=99.0, prev_price=100.0)

    assert w.low == 94.0            # 마감 직전에 깨진 저점이 잡힌다
    assert w.last == 95.0


def test_마감_꼬리도_지나간_극값은_제외():
    row = {"open": 100.0, "high": 103.0, "low": 94.0, "close": 102.5}

    w = live.tail_window(row, prev_high=101.0, prev_low=94.0, prev_price=102.0)

    assert w.low == 102.0           # 94는 이미 봤다
    assert w.high == 103.0


# ------------------------------------------------------------------ 피드 지연


def test_확정봉이_멈추면_지연으로_본다():
    df = frame(raw_candle(T0), raw_candle(T0 + STEP))

    assert not live.is_feed_lagging(df, "5m", now_ts=T0 + 2 * STEP + 10)
    assert live.is_feed_lagging(df, "5m", now_ts=T0 + 10 * STEP)


def test_빈_프레임은_지연():
    assert live.is_feed_lagging(cd.to_dataframe([]), "5m", now_ts=T0)


# ------------------------------------------------------------------ 검증


def test_고저_모순은_생성_시점에_막는다():
    with pytest.raises(live.LiveError):
        live.LiveQuote(
            bar_ts=T0, observed_ts=T0,
            bar_open=1.0, bar_high=0.9, bar_low=1.1, last=1.0,
        )


def test_현재가가_범위_밖이면_예외():
    with pytest.raises(live.LiveError):
        live.LiveQuote(
            bar_ts=T0, observed_ts=T0,
            bar_open=1.0, bar_high=1.05, bar_low=0.95, last=1.5,
        )


def test_0_이하_가격():
    with pytest.raises(live.LiveError):
        live.TickWindow(open=0.0, high=1.0, low=0.0, last=1.0)
