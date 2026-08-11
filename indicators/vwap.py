"""세션 기준 VWAP과 표준편차 밴드.

거래량 가중 평균가. 기관 체결 기준선으로 널리 쓰여서, 다수가 같은 선을
본다는 점에서 이평선과 성격이 비슷하다. 다만 거래량이 들어가므로
"얼마나 많은 돈이 어느 가격에 실렸는가"를 반영한다는 게 다르다.

세션마다 리셋한다. 누적을 무한정 이어가면 며칠 전 거래가 지금 기준선을
붙잡고 있어서, 5분봉 단타의 되돌림 기준으로 쓸 수 없다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.base import IndicatorError, column


def session_id(df: pd.DataFrame, hours: int = 24) -> pd.Series:
    """봉을 세션 단위로 묶는 정수 라벨.

    기본은 UTC 자정 기준 하루. 암호화폐는 24시간 거래되지만 일 단위
    구분은 여전히 의미가 있다 — 파생 정산과 지표 기준일이 그 경계다.
    """

    if hours <= 0:
        raise IndicatorError(f"세션 길이는 0보다 커야 한다: {hours}")
    return (df["ts"] // (hours * 3600)).astype("int64")


def vwap(df: pd.DataFrame, hours: int = 24) -> pd.Series:
    """세션 누적 VWAP.

    전형가(고+저+종)/3에 거래대금을 가중한다. 종가만 쓰면 봉 안에서
    실제로 체결된 가격대를 무시하게 된다.
    """

    typical = (column(df, "high") + column(df, "low") + column(df, "close")) / 3.0
    volume = column(df, "volume")
    session = session_id(df, hours)

    pv = (typical * volume).groupby(session).cumsum()
    vol = volume.groupby(session).cumsum()

    return pv / vol.where(vol > 0)


def bands(
    df: pd.DataFrame,
    std_mult: float = 2.0,
    hours: int = 24,
) -> pd.DataFrame:
    """VWAP과 ±std_mult 밴드. 컬럼: vwap, upper, lower, deviation

    표준편차는 VWAP 대비 편차를 세션 안에서 누적해 계산한다.
    볼린저처럼 고정 창을 쓰지 않는 이유는, VWAP 자체가 세션 시작부터의
    누적값이라 기준선과 밴드의 관측 구간이 어긋나면 해석이 안 되기 때문이다.

    deviation은 (가격 - VWAP) / 표준편차. 몇 시그마 떨어졌는지다.
    """

    if std_mult <= 0:
        raise IndicatorError(f"표준편차 배수는 0보다 커야 한다: {std_mult}")

    center = vwap(df, hours)
    close = column(df, "close")
    session = session_id(df, hours)

    diff_sq = (close - center) ** 2
    # 세션 내 누적 평균 제곱편차 -> 표준편차
    count = pd.Series(1.0, index=df.index).groupby(session).cumsum()
    var = diff_sq.groupby(session).cumsum() / count
    sd = np.sqrt(var)

    return pd.DataFrame(
        {
            "vwap": center,
            "upper": center + sd * std_mult,
            "lower": center - sd * std_mult,
            "deviation": (close - center) / sd.where(sd > 0),
        }
    )
