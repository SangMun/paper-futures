"""True Range와 ATR.

손절 버퍼, 트레일링 폭, 합류 존 허용 오차, XRP 변동성 서킷브레이커가
전부 이 값을 입력으로 쓴다. 지표 중 의존하는 곳이 가장 많다.
"""

from __future__ import annotations

import pandas as pd

from config.settings import INDICATORS
from indicators.base import check_period, column, empty_like, wilder


def true_range(df: pd.DataFrame) -> pd.Series:
    """max(고-저, |고-전종가|, |저-전종가|).

    전종가를 쓰는 두 항이 있는 이유는 갭 때문이다. 갭 상승 후 좁은 봉이
    만들어지면 고저폭만으로는 변동이 작아 보이지만, 실제로 가격은
    전날 종가에서 그만큼 뛴 것이다.

    첫 봉은 전종가가 없어 NaN.
    """

    high = column(df, "high")
    low = column(df, "low")
    prev_close = column(df, "close").shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    tr.iloc[:1] = float("nan")
    return tr


def atr(df: pd.DataFrame, period: int = INDICATORS.atr_period) -> pd.Series:
    """Wilder 평활 ATR. 차트 프로그램 기본값과 같은 계산이다."""

    period = check_period(period)
    if len(df) < period + 1:
        return empty_like(df)
    return wilder(true_range(df), period)


def atr_pct(df: pd.DataFrame, period: int = INDICATORS.atr_period) -> pd.Series:
    """종가 대비 ATR 비율.

    종목 간 변동성을 비교하거나, 손절 거리를 비율로 환산할 때 쓴다.
    BTC 6만 달러의 ATR 500과 XRP 1달러의 ATR 0.008은 절대값으로 비교되지 않는다.
    """

    return atr(df, period) / column(df, "close")
