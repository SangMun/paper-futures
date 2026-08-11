"""RSI 다이버전스와 이중 바닥.

명세 §10의 백로그 항목. `pivots.py`를 그대로 재사용하므로 추가 비용이 거의 없다 —
가격 대신 RSI 시리즈에 같은 피벗 탐지를 돌리면 다이버전스, 두 저점의 가격
근접도를 보면 이중 바닥이다.

두 패턴 모두 확정 지연을 그대로 물려받는다. 피벗이 확정돼야 비교할 저점이
생기므로, 다이버전스는 본질적으로 늦은 신호다. 그게 정상이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.settings import INDICATORS
from indicators import oscillators as osc
from indicators.base import IndicatorError
from indicators.pivots import Pivot, all_pivots, usable_pivots


@dataclass(frozen=True)
class Divergence:
    kind: str               # "bullish" / "bearish"
    price_a: Pivot
    price_b: Pivot
    osc_a: float
    osc_b: float

    @property
    def confirmed_at(self) -> int:
        return self.price_b.confirmed_at


def _osc_frame(values: pd.Series) -> pd.DataFrame:
    """피벗 탐지는 OHLC를 요구한다. 오실레이터를 그 형태로 감싼다."""

    return pd.DataFrame(
        {
            "ts": range(len(values)),
            "high": values,
            "low": values,
            "close": values,
            "open": values,
        }
    )


def find_divergences(
    df: pd.DataFrame,
    as_of: int,
    n: int = 5,
    period: int = INDICATORS.rsi_period,
    max_gap: int = 60,
) -> list[Divergence]:
    """가격과 RSI의 방향이 어긋난 구간.

    강세: 가격 저점은 낮아졌는데 RSI 저점은 높아짐
    약세: 가격 고점은 높아졌는데 RSI 고점은 낮아짐

    두 피벗이 max_gap봉 이상 떨어지면 비교하지 않는다. 너무 멀면
    "추세가 바뀐 것"이지 다이버전스가 아니다.
    """

    if as_of < 0 or as_of >= len(df):
        raise IndicatorError(f"as_of가 범위를 벗어남: {as_of} (길이 {len(df)})")

    rsi = osc.rsi(df, period)
    if rsi.notna().sum() < n * 2 + 1:
        return []

    price_pivots = usable_pivots(all_pivots(df, n), as_of)
    out: list[Divergence] = []

    for kind, want in (("bullish", "low"), ("bearish", "high")):
        same = [p for p in price_pivots if p.kind == want]
        for a, b in zip(same, same[1:]):
            if b.index - a.index > max_gap:
                continue

            ra, rb = rsi.iloc[a.index], rsi.iloc[b.index]
            if pd.isna(ra) or pd.isna(rb):
                continue

            if kind == "bullish" and b.price < a.price and rb > ra:
                out.append(Divergence(kind, a, b, float(ra), float(rb)))
            elif kind == "bearish" and b.price > a.price and rb < ra:
                out.append(Divergence(kind, a, b, float(ra), float(rb)))

    return sorted(out, key=lambda d: d.price_b.index)


def divergence_series(
    df: pd.DataFrame,
    n: int = 5,
    period: int = INDICATORS.rsi_period,
    max_gap: int = 60,
) -> pd.DataFrame:
    """봉마다 "이 시점에 확정된 다이버전스가 있는가". 컬럼: bullish, bearish

    전 구간을 한 번에 계산해 prepare()에서 쓴다. 봉마다 `find_divergences`를
    부르면 매번 피벗을 다시 찾아 리플레이가 O(n^2)가 된다.

    확정 시점(`confirmed_at`)에 표시하므로 미래를 보지 않는다 —
    두 번째 피벗이 확정돼야 비교할 저점이 생긴다.
    """

    found = find_divergences(df, len(df) - 1, n=n, period=period, max_gap=max_gap)

    out = pd.DataFrame(
        {"bullish": False, "bearish": False},
        index=df.index,
    )
    for d in found:
        at = d.confirmed_at
        if 0 <= at < len(df):
            out.iloc[at, out.columns.get_loc(d.kind)] = True
    return out


def double_series(
    df: pd.DataFrame,
    n: int = 5,
    price_tolerance: float = 0.005,
    min_gap: int = 5,
    max_gap: int = 60,
) -> pd.DataFrame:
    """봉마다 확정된 이중 바닥/천장. 컬럼: bottom, top

    두 번째 저점의 가격도 함께 돌려준다 — 손절 기준으로 쓴다.
    """

    found = find_double_bottoms(
        df, len(df) - 1, n=n, price_tolerance=price_tolerance,
        min_gap=min_gap, max_gap=max_gap,
    )

    out = pd.DataFrame(
        {"bottom": False, "top": False, "level": float("nan")},
        index=df.index,
    )
    for d in found:
        at = d.confirmed_at
        if 0 <= at < len(df):
            out.iloc[at, out.columns.get_loc(d.kind)] = True
            # 두 저점 중 낮은 쪽(천장이면 높은 쪽)이 무효화 기준이다.
            out.iloc[at, out.columns.get_loc("level")] = (
                min(d.first.price, d.second.price)
                if d.kind == "bottom"
                else max(d.first.price, d.second.price)
            )
    return out


@dataclass(frozen=True)
class DoubleBottom:
    kind: str               # "bottom" / "top"
    first: Pivot
    second: Pivot
    tolerance_used: float

    @property
    def confirmed_at(self) -> int:
        return self.second.confirmed_at


def find_double_bottoms(
    df: pd.DataFrame,
    as_of: int,
    n: int = 5,
    price_tolerance: float = 0.005,
    min_gap: int = 5,
    max_gap: int = 60,
) -> list[DoubleBottom]:
    """두 저점(고점)의 가격이 price_tolerance 이내로 붙은 구간.

    min_gap을 두는 이유는 바로 이웃한 피벗 두 개는 같은 저점의 노이즈일
    뿐이지 이중 바닥이 아니기 때문이다.
    """

    if not 0 < price_tolerance < 1:
        raise IndicatorError(f"price_tolerance는 0~1 사이여야 한다: {price_tolerance}")

    pivots = usable_pivots(all_pivots(df, n), as_of)
    out: list[DoubleBottom] = []

    for kind, want in (("bottom", "low"), ("top", "high")):
        same = [p for p in pivots if p.kind == want]
        for a, b in zip(same, same[1:]):
            gap = b.index - a.index
            if not (min_gap <= gap <= max_gap):
                continue
            if a.price <= 0:
                continue

            diff = abs(b.price - a.price) / a.price
            if diff <= price_tolerance:
                out.append(DoubleBottom(kind, a, b, diff))

    return sorted(out, key=lambda d: d.second.index)
