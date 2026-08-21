"""실행 시계용 실시간 시세.

**이 모듈은 실행 경로 전용이다.** `indicators/`와 `strategies/`는 여기를
import하지 않는다. 진행 중인 봉으로 지표를 계산하면 같은 시점을 다시 계산할
때 값이 달라져서, 과거 검증과 실시간의 결과가 어긋난다. 그 경계를 지키려고
캔들 정제(candles.py)와 파일을 나눴다.

경계는 tests/test_layering.py가 강제한다.

---

거래소가 주는 진행 중인 봉은 **누적** 고저다(봉 시작 이후 전체). 청산 판정에
이 값을 그대로 쓰면 안 된다. 누적값은 순서 정보를 잃기 때문이다.

    봉 시가 100 -> 99까지 눌림(누적저가 99) -> 115까지 상승
      틱A: 저가 99 vs 손절 98  -> 청산 없음. 트레일이 손절을 99로 올림
      틱B: 저가 99 vs 손절 99  -> 청산. 하지만 가격은 114다

99는 손절이 올라가기 *전에* 찍힌 값이라 실제로는 체결되지 않았다. 그래서
누적값을 직전 관측과 비교해 **그 사이에 실제로 지나온 구간**(TickWindow)으로
바꾼 뒤 판정한다.

현재가만 찍어보는 방식을 쓰지 않는 이유도 같다. 10초 사이에 손절을 찍고
되돌아온 꼬리를 놓친다. 누적 고저의 변화량은 그 꼬리를 잡아낸다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from core import candles as cd


# 마지막 확정봉이 이 배수만큼 오래되면 시세 피드가 밀린 것으로 본다.
# 멈춘 값으로 손절을 판정하면 실제와 다른 가격에서 청산된다.
MAX_FEED_LAG_BARS = 3


class LiveError(ValueError):
    pass


def _check_ohlc(prefix: str, open_: float, high: float, low: float, last: float) -> None:
    for name, value in (
        ("시가", open_), ("고가", high), ("저가", low), ("현재가", last)
    ):
        if not math.isfinite(value) or value <= 0:
            raise LiveError(f"{prefix} {name}는 0보다 큰 유한값이어야 한다: {value}")

    if low > high:
        raise LiveError(f"{prefix} 고저 역전: high={high}, low={low}")
    if high < max(open_, last) or low > min(open_, last):
        raise LiveError(
            f"{prefix} 범위 오류: open={open_}, high={high}, low={low}, last={last}"
        )


@dataclass(frozen=True)
class LiveQuote:
    """진행 중인 봉 하나의 누적 스냅샷.

    bar_high / bar_low는 bar_ts부터 observed_ts까지의 누적값이다.
    청산 판정에 직접 쓰지 말고 window_since()로 구간을 만들어 쓴다.
    """

    bar_ts: int             # 진행 중인 봉의 시작 시각
    observed_ts: int        # 이 값을 관측한 시각
    bar_open: float
    bar_high: float
    bar_low: float
    last: float

    def __post_init__(self):
        _check_ohlc("진행 중인 봉", self.bar_open, self.bar_high, self.bar_low, self.last)


@dataclass(frozen=True)
class TickWindow:
    """직전 관측 이후 실제로 지나온 구간. 청산 판정의 입력.

    봉이 아니라 구간이다. open은 직전에 관측한 가격이고
    last는 지금 가격이다.
    """

    open: float
    high: float
    low: float
    last: float

    def __post_init__(self):
        _check_ohlc("구간", self.open, self.high, self.low, self.last)


def from_frame(
    df: pd.DataFrame,
    interval: str,
    now_ts: float | None = None,
) -> LiveQuote | None:
    """미마감 봉을 포함한 캔들에서 진행 중인 봉을 꺼낸다.

    `GateClient.fetch(..., drop_unclosed=False)` 결과를 그대로 넘긴다.
    마지막 봉이 이미 마감됐으면(봉이 막 넘어간 직후) None을 돌려준다.
    그 구간은 봉 시계가 확정봉으로 처리한다.
    """

    if df is None or df.empty:
        return None

    step = cd.interval_seconds(interval)
    if now_ts is None:
        now_ts = pd.Timestamp.now(tz="UTC").timestamp()

    row = df.iloc[-1]
    bar_ts = int(row["ts"])
    if bar_ts + step <= now_ts:
        return None                     # 마감된 봉. 진행 중인 봉이 아직 없다.

    return LiveQuote(
        bar_ts=bar_ts,
        observed_ts=int(now_ts),
        bar_open=float(row["open"]),
        bar_high=float(row["high"]),
        bar_low=float(row["low"]),
        last=float(row["close"]),
    )


def window_since(
    quote: LiveQuote,
    prev_bar_ts: int,
    prev_high: float,
    prev_low: float,
    prev_price: float,
) -> TickWindow:
    """직전 관측 이후의 구간을 만든다.

    prev_bar_ts가 지금 봉과 다르면(첫 관측이거나 봉이 바뀜) 봉 시작 이후
    전체를 구간으로 본다.

    같은 봉이면 누적값의 **변화**만 이번 구간에 속한다.
      - 누적 고가가 커졌으면 그 신고점은 이번 구간에서 찍힌 것이다
      - 안 커졌으면 이번 구간의 고점은 관측 가격 두 개 사이에 있다

    후자에는 사각지대가 있다. 두 관측 사이에 봉의 기존 고점을 넘지 않는
    선에서 움직인 부분은 알 수 없다. 틱 데이터 없이는 복원 불가능하고,
    폴링 간격(10초)만큼으로 한정된다. 이미 지나간 극값을 이번 구간에
    넣는 것보다는 이쪽이 안전하다 — 없던 청산을 만들지는 않는다.
    """

    if prev_bar_ts != quote.bar_ts or prev_price <= 0:
        return TickWindow(
            open=quote.bar_open,
            high=quote.bar_high,
            low=quote.bar_low,
            last=quote.last,
        )

    high = quote.bar_high if quote.bar_high > prev_high else max(prev_price, quote.last)
    low = quote.bar_low if quote.bar_low < prev_low else min(prev_price, quote.last)

    # 구간 자체의 양 끝은 반드시 포함한다.
    high = max(high, prev_price, quote.last)
    low = min(low, prev_price, quote.last)

    return TickWindow(open=prev_price, high=high, low=low, last=quote.last)


def tail_window(
    closed_row,
    prev_high: float,
    prev_low: float,
    prev_price: float,
) -> TickWindow:
    """봉이 넘어갈 때, 직전 관측부터 그 봉 마감까지의 남은 구간.

    폴링 간격 때문에 봉의 마지막 몇 초는 어떤 틱도 보지 못한다.
    이 구간을 버리면 봉 끝에서 손절이 닿아도 놓친다.
    """

    close = float(closed_row["close"])
    bar_high = float(closed_row["high"])
    bar_low = float(closed_row["low"])

    if prev_price <= 0:
        return TickWindow(
            open=float(closed_row["open"]), high=bar_high, low=bar_low, last=close
        )

    high = bar_high if bar_high > prev_high else max(prev_price, close)
    low = bar_low if bar_low < prev_low else min(prev_price, close)
    high = max(high, prev_price, close)
    low = min(low, prev_price, close)

    return TickWindow(open=prev_price, high=high, low=low, last=close)


def feed_lag_seconds(df: pd.DataFrame, interval: str, now_ts: float) -> float:
    """마지막 **확정봉**이 얼마나 오래됐는가.

    진행 중인 봉은 거래가 없어도 갱신되므로 지연 판단에 쓸 수 없다.
    """

    if df is None or df.empty:
        return float("inf")

    step = cd.interval_seconds(interval)
    closed = df[df["ts"] + step <= now_ts]
    if closed.empty:
        return float("inf")

    return float(now_ts) - int(closed["ts"].iloc[-1]) - step


def is_feed_lagging(
    df: pd.DataFrame,
    interval: str,
    now_ts: float,
    max_bars: int = MAX_FEED_LAG_BARS,
) -> bool:
    return feed_lag_seconds(df, interval, now_ts) > cd.interval_seconds(interval) * max_bars
