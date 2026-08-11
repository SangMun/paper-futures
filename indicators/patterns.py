"""캔들 형태 판정.

명세 §5.3의 확인 캔들 규칙을 구현한다. 레벨에 닿자마자 진입하면 그대로
관통하는 경우를 걸러내지 못하므로, 마감된 캔들이 되돌림을 거부했는지 본다.

세 전략이 전부 쓴다.
  btc_trend: MA 눌림목 확인
  fib_retracement: 합류 존 확인 (§5.3)
  xrp_patterns: 스윕 회복 확인

`core/candles.py`가 아니라 여기 있는 이유는, 저쪽은 거래소 응답을 정제하는
데이터 레이어이고 이건 계산 레이어이기 때문이다.
"""

from __future__ import annotations

import pandas as pd

from indicators.base import column


# 몸통이 이 비율보다 작으면 도지로 보고 꼬리 비율을 계산하지 않는다.
# 0으로 나누는 걸 막으면서, 몸통이 거의 없는 캔들의 꼬리 비율이
# 무한대로 튀어 아무 캔들이나 확인 캔들이 되는 것도 함께 막는다.
_MIN_BODY_RATIO = 0.05


def body(df: pd.DataFrame) -> pd.Series:
    return (column(df, "close") - column(df, "open")).abs()


def candle_range(df: pd.DataFrame) -> pd.Series:
    return column(df, "high") - column(df, "low")


def lower_wick(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1).astype("float64") - column(df, "low")


def upper_wick(df: pd.DataFrame) -> pd.Series:
    return column(df, "high") - df[["open", "close"]].max(axis=1).astype("float64")


def long_lower_wick(df: pd.DataFrame, mult: float = 2.0) -> pd.Series:
    """아래꼬리가 몸통의 mult배 이상 (롱 확인).

    몸통이 캔들 범위의 5% 미만이면 제외한다. 도지는 방향을 말해주지 않는데,
    비율만 보면 꼬리/몸통이 폭발적으로 커져 전부 확인 캔들로 잡힌다.
    """

    b = body(df)
    rng = candle_range(df)
    meaningful = b >= rng * _MIN_BODY_RATIO
    return (lower_wick(df) >= b * mult) & meaningful


def long_upper_wick(df: pd.DataFrame, mult: float = 2.0) -> pd.Series:
    b = body(df)
    rng = candle_range(df)
    meaningful = b >= rng * _MIN_BODY_RATIO
    return (upper_wick(df) >= b * mult) & meaningful


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """직전 음봉을 감싸는 양봉."""

    o, c = column(df, "open"), column(df, "close")
    prev_o, prev_c = o.shift(1), c.shift(1)

    return (c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c)


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    o, c = column(df, "open"), column(df, "close")
    prev_o, prev_c = o.shift(1), c.shift(1)

    return (c < o) & (prev_c > prev_o) & (c <= prev_o) & (o >= prev_c)


def rejection(df: pd.DataFrame, level: pd.Series | float, side: str) -> pd.Series:
    """되돌림 거부: 레벨을 건드렸다가 반대편에서 마감.

    롱이면 저가가 레벨 아래로 내려갔지만 종가는 레벨 위.
    이게 "지지가 작동했다"의 가장 단순한 정의다.
    """

    if side == "long":
        return (column(df, "low") <= level) & (column(df, "close") > level)
    if side == "short":
        return (column(df, "high") >= level) & (column(df, "close") < level)
    raise ValueError(f"side는 long/short: {side!r}")


def confirmation(
    df: pd.DataFrame,
    level: pd.Series | float,
    side: str,
    wick_mult: float = 2.0,
) -> pd.Series:
    """명세 §5.3의 확인 캔들. 셋 중 하나면 성립.

      되돌림 거부 / 긴 꼬리 / 인걸핑

    셋을 OR로 묶는 이유는 원문이 "다음 중 하나"라고 규정하기 때문이다.
    AND로 묶으면 신호가 거의 발생하지 않는다.
    """

    if side == "long":
        return rejection(df, level, "long") | long_lower_wick(df, wick_mult) | bullish_engulfing(df)
    if side == "short":
        return rejection(df, level, "short") | long_upper_wick(df, wick_mult) | bearish_engulfing(df)
    raise ValueError(f"side는 long/short: {side!r}")
