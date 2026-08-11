"""이동평균과 국면 판정.

BTC 추세 전략은 일봉 MA50/200으로 국면을 정하고, 같은 선을 눌림목 타점으로도
쓴다. 두 용도가 같은 함수를 쓰도록 여기 모아 둔다.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from config.settings import INDICATORS
from indicators.base import check_period, column, ema_series, empty_like


Regime = Literal["bull", "bear", "none"]


def sma(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    period = check_period(period)
    src = column(df, col)
    if len(src) < period:
        return empty_like(df)
    return src.rolling(period, min_periods=period).mean()


def ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    """지수이동평균. 앞 period개는 SMA로 시드를 잡고 그 앞은 NaN.

    pandas의 ewm을 그대로 쓰면 첫 봉부터 값이 나오는데, 그건 사실상
    첫 종가 자체라 평균이 아니다. 워밍업 구간을 NaN으로 남겨야
    전략이 "아직 못 믿는 값"을 구분할 수 있다.

    EMA는 경로 의존적이라 데이터 시작 지점이 다르면 값도 미세하게 다르다.
    수렴하려면 period의 3배 이상 앞선 캔들이 필요하다. 500봉을 받는 이유 중 하나다.
    """

    return ema_series(column(df, col), period)


def regime(
    df: pd.DataFrame,
    fast: int = INDICATORS.ma_fast,
    slow: int = INDICATORS.ma_slow,
    col: str = "close",
) -> pd.Series:
    """MA 국면. 'bull' / 'bear' / 'none'.

    정배열(가격 > 빠른선 > 느린선)이면 bull, 역배열이면 bear, 나머지는 none.
    단순 골든크로스보다 조건이 하나 더 붙는다. 크로스만 보면 두 선이
    붙어서 오르내리는 횡보 구간에서 국면이 계속 뒤집히기 때문이다.

    워밍업 구간도 'none'이다. 전략은 bull/bear일 때만 진입하므로
    NaN 대신 none을 돌려주는 편이 호출부가 단순해진다.
    """

    f = sma(df, fast, col)
    s = sma(df, slow, col)
    price = column(df, col)

    out = pd.Series("none", index=df.index, dtype="object")
    out[(price > f) & (f > s)] = "bull"
    out[(price < f) & (f < s)] = "bear"
    out[f.isna() | s.isna()] = "none"
    return out


def distance_pct(price: pd.Series | float, line: pd.Series | float) -> pd.Series | float:
    """이평선까지의 거리 비율. 눌림목 근접 판정에 쓴다.

    양수면 가격이 선 위에 있다는 뜻.
    """

    return (price - line) / line


def touched(
    df: pd.DataFrame,
    line: pd.Series,
    tolerance: float = 0.005,
) -> pd.Series:
    """해당 봉이 이평선을 건드렸는지 (bool Series).

    종가가 아니라 고저 범위로 판정한다. 눌림목은 꼬리로만 닿고
    종가는 위에서 마감하는 경우가 오히려 전형적이기 때문이다.
    tolerance는 선 근처를 스친 경우까지 포함하기 위한 여유.
    """

    high = column(df, "high")
    low = column(df, "low")
    band = line.abs() * tolerance
    return (low <= line + band) & (high >= line - band)
