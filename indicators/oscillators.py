"""RSI와 MACD.

BTC 추세 전략의 진입 판정, 피보나치 전략의 방향 필터(§5.1),
XRP 다이버전스 패턴이 여기 값을 쓴다.
"""

from __future__ import annotations

import pandas as pd

from config.settings import INDICATORS
from indicators.base import (
    IndicatorError,
    check_period,
    column,
    ema_series,
    empty_like,
    wilder,
)
from indicators.ma import ema


def rsi(df: pd.DataFrame, period: int = INDICATORS.rsi_period, col: str = "close") -> pd.Series:
    """Wilder RSI.

    상승분/하락분을 각각 Wilder 평활한 뒤 비율을 낸다.
    단순이동평균으로 계산하는 변형도 있지만 차트 프로그램 기본값은 Wilder다.

    하락분이 0인 구간(연속 상승)에서는 RS가 무한대가 되므로 100으로 둔다.
    """

    period = check_period(period)
    src = column(df, col)
    if len(src) < period + 1:
        return empty_like(df)

    delta = src.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # diff의 첫 값은 NaN이다. wilder가 앞쪽 NaN을 건너뛰고 시작한다.
    gain.iloc[0] = float("nan")
    loss.iloc[0] = float("nan")

    avg_gain = wilder(gain, period)
    avg_loss = wilder(loss, period)

    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 이면 rs가 inf 또는 NaN(0/0)이 된다.
    out = out.where(avg_loss != 0.0, 100.0)
    out[avg_gain.isna() | avg_loss.isna()] = float("nan")
    return out


def macd(
    df: pd.DataFrame,
    fast: int = INDICATORS.macd_fast,
    slow: int = INDICATORS.macd_slow,
    signal: int = INDICATORS.macd_signal,
    col: str = "close",
) -> pd.DataFrame:
    """MACD선 / 시그널선 / 히스토그램.

    컬럼: macd, signal, hist

    시그널선은 MACD선의 EMA다. MACD선 자체가 워밍업 구간에서 NaN이므로
    시그널선은 그보다 signal개만큼 더 늦게 시작한다. 즉 slow + signal개
    이전 구간은 판정에 쓸 수 없다.
    """

    fast = check_period(fast, "fast")
    slow = check_period(slow, "slow")
    signal = check_period(signal, "signal")
    if fast >= slow:
        raise IndicatorError(f"fast({fast})는 slow({slow})보다 작아야 한다")

    empty = empty_like(df)
    if len(df) < slow:
        return pd.DataFrame({"macd": empty, "signal": empty, "hist": empty})

    line = ema(df, fast, col) - ema(df, slow, col)
    # ema_series가 선행 NaN을 건너뛰므로 MACD선을 그대로 넘겨도 된다.
    sig = ema_series(line, signal)

    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def macd_bullish(df: pd.DataFrame, **kw) -> pd.Series:
    """피보나치 명세 §5.1의 롱 방향 조건: MACD > 시그널 **그리고** MACD > 0.

    0 위 조건이 붙는 이유는, 하락 추세 중 반등에서도 크로스는 나오기 때문이다.
    """

    m = macd(df, **kw)
    return (m["macd"] > m["signal"]) & (m["macd"] > 0)


def macd_bearish(df: pd.DataFrame, **kw) -> pd.Series:
    m = macd(df, **kw)
    return (m["macd"] < m["signal"]) & (m["macd"] < 0)
