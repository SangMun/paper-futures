"""볼린저 밴드와 스퀴즈 판정.

BTC 추세 전략의 진입 조합, XRP 스퀴즈 돌파 패턴이 쓴다.
"""

from __future__ import annotations

import pandas as pd

from config.settings import INDICATORS
from indicators.base import IndicatorError, check_period, column, empty_like


def bollinger(
    df: pd.DataFrame,
    period: int = INDICATORS.bb_period,
    std_mult: float = INDICATORS.bb_std,
    col: str = "close",
) -> pd.DataFrame:
    """중심선 / 상단 / 하단 / 밴드폭.

    컬럼: mid, upper, lower, bandwidth

    bandwidth는 (상단 - 하단) / 중심선. 절대폭이 아니라 비율인 이유는
    가격 수준이 다른 구간끼리, 그리고 BTC와 XRP를 같은 기준으로
    비교해야 하기 때문이다.

    표준편차는 모표준편차(ddof=0)를 쓴다. 차트 프로그램 기본값이 이쪽이다.
    """

    period = check_period(period)
    src = column(df, col)

    if len(src) < period:
        e = empty_like(df)
        return pd.DataFrame({"mid": e, "upper": e, "lower": e, "bandwidth": e})

    mid = src.rolling(period, min_periods=period).mean()
    sd = src.rolling(period, min_periods=period).std(ddof=0)

    upper = mid + sd * std_mult
    lower = mid - sd * std_mult

    return pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "bandwidth": (upper - lower) / mid,
        }
    )


def squeeze(
    df: pd.DataFrame,
    period: int = INDICATORS.bb_period,
    std_mult: float = INDICATORS.bb_std,
    percentile: float = INDICATORS.bb_squeeze_percentile,
    lookback: int = INDICATORS.bb_squeeze_lookback,
    col: str = "close",
) -> pd.Series:
    """밴드폭이 최근 lookback 구간의 하위 percentile 이하인가 (bool Series).

    고정 임계값을 쓰지 않고 백분위로 판정하는 이유는, "좁다"의 기준이
    종목과 시기마다 다르기 때문이다. XRP 5분봉의 평소 밴드폭과
    BTC 1시간봉의 평소 밴드폭은 애초에 자릿수가 다르다.

    백분위는 **현재 봉을 포함한 과거 lookback 구간** 안에서만 계산한다.
    전체 구간의 분위수를 쓰면 미래 데이터를 보게 된다.
    """

    if not 0.0 < percentile < 100.0:
        raise IndicatorError(f"percentile은 0~100 사이여야 한다: {percentile}")

    lookback = check_period(lookback, "lookback")
    bw = bollinger(df, period, std_mult, col)["bandwidth"]

    out = pd.Series(False, index=df.index, dtype="bool")
    if bw.notna().sum() < 2:
        return out

    # 각 시점에서 자기 자신이 과거 구간 중 몇 번째 분위인지.
    # 룩백이 다 찰 때까지는 판정하지 않는다. 20봉짜리 창에서 낸 순위를
    # "최근 100봉 중 하위 20%"라고 부를 수 없고, 그렇게 하면 초기 구간에서
    # 스퀴즈가 과다 발생한다(실측 24% vs 기준 20%).
    rank = bw.rolling(lookback, min_periods=lookback).rank(pct=True) * 100.0
    return (rank <= percentile).fillna(False).astype("bool")


def position(
    df: pd.DataFrame,
    period: int = INDICATORS.bb_period,
    std_mult: float = INDICATORS.bb_std,
    col: str = "close",
) -> pd.Series:
    """밴드 내 상대 위치. 0 = 하단, 1 = 상단 (%B).

    밴드를 벗어나면 0 미만 / 1 초과가 나온다. 그 자체가 신호다.
    """

    b = bollinger(df, period, std_mult, col)
    width = b["upper"] - b["lower"]
    return (column(df, col) - b["lower"]) / width.where(width != 0)
