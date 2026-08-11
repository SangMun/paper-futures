"""스윙 고저점 탐지.

명세 §2. 피보나치의 입력이자, RSI 다이버전스와 이중 바닥도 같은 함수를 쓴다.

**확정 지연이 이 모듈의 존재 이유다.**
피벗은 우측 n개 캔들이 만들어진 뒤에야 판정할 수 있다. i번 봉이 고점인지는
i+n까지 봐야 알 수 있으므로, t 시점의 매매 판단에는 `confirmed_at <= t`인
피벗만 써야 한다. 이걸 빠뜨리면 과거 검증이 실제보다 좋게 나오는데,
결과가 "좋아지기만" 해서 눈에 띄지 않는다.

그래서 `as_of`를 선택 인자로 두지 않는다. 호출부가 반드시 명시해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from config.settings import FIB
from indicators.base import IndicatorError, check_period, column


Kind = Literal["high", "low"]
Direction = Literal["up", "down"]


@dataclass(frozen=True)
class Pivot:
    index: int
    timestamp: int
    price: float
    kind: Kind
    confirmed_at: int       # 이 인덱스부터 판정 가능 (= index + n)

    def is_confirmed(self, as_of: int) -> bool:
        return self.confirmed_at <= as_of


@dataclass(frozen=True)
class Swing:
    """임펄스 한 구간. 피보나치를 그리는 단위."""

    start: Pivot
    end: Pivot
    direction: Direction    # up = 저점->고점

    @property
    def low(self) -> float:
        return self.start.price if self.direction == "up" else self.end.price

    @property
    def high(self) -> float:
        return self.end.price if self.direction == "up" else self.start.price

    @property
    def size(self) -> float:
        return self.high - self.low

    @property
    def size_pct(self) -> float:
        return self.size / self.low if self.low > 0 else 0.0

    @property
    def bars(self) -> int:
        return self.end.index - self.start.index


def all_pivots(df: pd.DataFrame, n: int) -> list[Pivot]:
    """프레임 전체의 피벗. 확정 시점(`confirmed_at`)을 함께 들고 있다.

    **한 번만 계산하고 재사용한다.** 봉마다 이 함수를 부르면 매번 전 구간
    롤링 최대/최소를 다시 돌게 되어 리플레이가 O(n^2)가 된다.
    실측으로 9,600봉 리플레이가 63초였고, 사전 계산하면 그게 사라진다.

    반환값에는 미확정 피벗도 들어 있다. 걸러내는 건 `usable_pivots`의 몫이고,
    그래야 as_of 규율이 한 곳에만 남는다.

    고점: high[i] == max(high[i-n : i+n+1])
    저점: low[i]  == min(low[i-n : i+n+1])

    같은 값이 연속하면 여러 봉이 동시에 피벗이 될 수 있다. 첫 번째만 채택한다 —
    같은 가격대의 피벗이 여러 개면 스윙 짝짓기가 어긋난다.
    """

    n = check_period(n, "n")
    high = column(df, "high")
    low = column(df, "low")
    ts = df["ts"].to_numpy()

    window = 2 * n + 1
    if len(df) < window:
        return []

    is_high = high == high.rolling(window, center=True).max()
    is_low = low == low.rolling(window, center=True).min()

    # 같은 봉이 고점이자 저점이면(완전 평탄 구간) 둘 다 버린다.
    both = is_high & is_low
    is_high &= ~both
    is_low &= ~both

    out: list[Pivot] = []
    prev_price: dict[str, float] = {}

    for i in np.flatnonzero((is_high | is_low).to_numpy()):
        i = int(i)
        confirmed = i + n

        kind: Kind = "high" if bool(is_high.iloc[i]) else "low"
        price = float(high.iloc[i] if kind == "high" else low.iloc[i])

        # 같은 종류가 같은 가격으로 연속하면 첫 번째만 남긴다
        if out and out[-1].kind == kind and prev_price.get(kind) == price:
            continue
        prev_price[kind] = price

        out.append(Pivot(i, int(ts[i]), price, kind, confirmed))

    return out


def usable_pivots(pivots: list[Pivot], as_of: int) -> list[Pivot]:
    """as_of 시점에 이미 확정된 피벗만.

    확정 지연을 강제하는 자리는 여기 하나뿐이다. 여러 곳에서 각자
    거르면 한 곳만 빠뜨려도 미래를 보게 되고, 결과가 좋아지기만 해서
    눈에 띄지 않는다.
    """

    return [p for p in pivots if p.confirmed_at <= as_of]


def find_pivots(df: pd.DataFrame, n: int, as_of: int) -> list[Pivot]:
    """확정된 피벗만 시간순으로. 도구·테스트용 단축 경로.

    리플레이에서는 `all_pivots`를 한 번 부르고 `usable_pivots`로 거를 것.
    """

    if as_of < 0 or as_of >= len(df):
        raise IndicatorError(f"as_of가 범위를 벗어남: {as_of} (길이 {len(df)})")
    return usable_pivots(all_pivots(df, n), as_of)


def build_swings(
    pivots: list[Pivot],
    min_swing_pct: float = FIB.min_swing_pct,
    as_of: int | None = None,
    max_lookback: int = FIB.max_lookback,
) -> list[Swing]:
    """임펄스 구간을 만든다. **인접 피벗만 짝짓지 않는다.**

    명세 §2.3은 "직전 반대 피벗 대비"라고만 적혀 있어 처음엔 인접 피벗을
    이었는데, 차트로 확인해 보니(§10-3) 그 방식은 추세를 잘게 쪼갠다.
    되돌림을 낀 80봉짜리 상승이 작은 다리 여러 개로 나뉘어, 상위 후보에는
    급락 구간만 남았다. pivot_n을 3~12로 바꿔도 해결되지 않았다.

    사람이 피보나치를 그릴 때는 "의미 있는 저점에서 그 뒤 최고점까지"를
    한 다리로 본다. 중간 되돌림은 다리를 끊지 않고, **시작점을 되돌파할 때**
    비로소 끊긴다. 그 정의를 그대로 구현한다.

    새 극값이 나올 때마다 후보를 남기므로 같은 시작점에서 여러 스케일의
    스윙이 나온다. 합류 판정(§4)은 어차피 여러 스케일의 겹침을 보는 것이라
    이쪽이 오히려 목적에 맞는다.
    """

    if not 0 < min_swing_pct < 1:
        raise IndicatorError(f"min_swing_pct는 0~1 사이여야 한다: {min_swing_pct}")

    swings: list[Swing] = []

    for i, origin in enumerate(pivots):
        direction: Direction = "up" if origin.kind == "low" else "down"
        best: Pivot | None = None

        for nxt in pivots[i + 1:]:
            if nxt.kind == origin.kind:
                # 같은 종류가 시작점을 넘어서면 이 다리는 끝난 것이다.
                if direction == "up" and nxt.price <= origin.price:
                    break
                if direction == "down" and nxt.price >= origin.price:
                    break
                continue

            improved = (
                best is None
                or (direction == "up" and nxt.price > best.price)
                or (direction == "down" and nxt.price < best.price)
            )
            if not improved:
                continue

            best = nxt
            swing = Swing(origin, best, direction)
            if swing.low > 0 and swing.size_pct >= min_swing_pct:
                swings.append(swing)

    if as_of is not None:
        swings = [
            s for s in swings
            if s.end.confirmed_at <= as_of and as_of - s.end.index <= max_lookback
        ]

    return swings


def swings_at(
    swings: list[Swing],
    as_of: int,
    max_lookback: int = FIB.max_lookback,
    limit: int = FIB.max_swing_candidates,
) -> list[Swing]:
    """사전 계산된 스윙 중 as_of 시점에 쓸 수 있는 것. 큰 것부터 limit개.

    크기순으로 고르는 이유는 명세 §4가 "스윙 크기 기준 상위 K개"를
    요구하기 때문이다. 어떤 스윙이 정답인지는 스케일 선택 문제라
    유일해가 없고, 그래서 하나를 고르는 대신 여러 개의 겹침을 본다.
    """

    usable = [
        s
        for s in swings
        if s.end.confirmed_at <= as_of
        and 0 <= as_of - s.end.index <= max_lookback
        # **시작점도** 룩백 안에 있어야 한다.
        #
        # 끝만 검사하면 아주 오래전에 시작해 최근에 끝난 거대 스윙이 통과한다.
        # 크기순으로 상위 K개를 고르므로 그런 스윙이 항상 이기고, 그 되돌림
        # 레벨은 현재가에서 수십 ATR 떨어져 있어 가격이 닿을 일이 없다.
        # 실측: as_of=9000에서 6,654봉 전에 시작한 118% 스윙이 1~3위를 차지했고,
        # 그 결과 400일 리플레이에서 신호가 3건만 나왔다.
        #
        # 명세 §2.3의 "상대적으로 최근에 나온 변곡점"은 다리 전체를 뜻한다.
        and as_of - s.start.index <= max_lookback
    ]
    return sorted(usable, key=lambda s: s.size_pct, reverse=True)[:limit]


def all_swings(
    df: pd.DataFrame,
    n: int = FIB.pivot_n,
    min_swing_pct: float = FIB.min_swing_pct,
) -> list[Swing]:
    """프레임 전체의 스윙. prepare()에서 한 번만 부른다."""

    return build_swings(all_pivots(df, n), min_swing_pct)


def recent_swings(
    df: pd.DataFrame,
    as_of: int,
    n: int = FIB.pivot_n,
    min_swing_pct: float = FIB.min_swing_pct,
    max_lookback: int = FIB.max_lookback,
    limit: int = FIB.max_swing_candidates,
) -> list[Swing]:
    """합류 계산에 쓸 스윙 후보. 도구·테스트용 단축 경로.

    리플레이에서는 `all_swings` + `swings_at`을 쓸 것.
    """

    return swings_at(all_swings(df, n, min_swing_pct), as_of, max_lookback, limit)


def pivot_series(df: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series]:
    """봉마다 "그 시점까지 확정된 마지막 고점/저점 가격".

    수평 지지저항 판정에 쓴다. 봉마다 find_pivots를 다시 부르면 O(n^2)라
    한 번에 계산해 둔다. shift(n)으로 확정 지연을 반영하므로 미래를 보지 않는다.
    """

    n = check_period(n, "n")
    high = column(df, "high")
    low = column(df, "low")
    window = 2 * n + 1

    is_high = high == high.rolling(window, center=True).max()
    is_low = low == low.rolling(window, center=True).min()

    # 확정 시점으로 n칸 밀어낸 뒤 앞으로 채운다.
    last_high = high.where(is_high).shift(n).ffill()
    last_low = low.where(is_low).shift(n).ffill()
    return last_high, last_low
