"""EXP-015의 진입 시점 후보.

양방향 hedge는 여기서 방향을 고르지 않는다. 이 모듈은 새 pair를 여는 시장
상태만 판단한다. 실제 체결·청산은 ``engine.dual_hedge``가 맡는다.
"""

from __future__ import annotations

import pandas as pd

from config.settings import DUAL_HEDGE, INDICATORS, DualHedgeConfig
from indicators import bollinger as bb


def squeeze_breakout_ready(
    closed: pd.DataFrame,
    config: DualHedgeConfig = DUAL_HEDGE,
) -> bool:
    """확정된 마지막 봉이 스퀴즈 상태에서 직전 범위를 이탈했는지 판정한다.

    ``closed``에는 진행 중인 봉을 넣지 않는다. 직전 봉까지 스퀴즈였고 마지막
    확정봉이 그 범위를 이탈했을 때만 True다. 돌파 범위는 마지막 봉 직전
    ``lookback`` 봉으로 한정해 미래 참조와 자기 자신 포함을 막는다.
    """

    lookback = config.squeeze_breakout_lookback_bars
    # 마지막 확정봉은 돌파 판정에만 쓰고, 직전 봉의 밴드폭을 그보다 앞선
    # ``squeeze_lookback_bars``와 비교한다. 이렇게 해야 돌파봉의 큰 폭이
    # 스퀴즈 기준 자체를 바꾸지 않으며, 동점(완전한 횡보)도 스퀴즈로 인정된다.
    minimum = config.squeeze_lookback_bars + INDICATORS.bb_period + 1
    if len(closed) < minimum:
        return False

    bandwidth = bb.bollinger(closed)["bandwidth"]
    pre_breakout_width = float(bandwidth.iloc[-2])
    reference_widths = bandwidth.iloc[-config.squeeze_lookback_bars - 2:-2]
    if reference_widths.isna().any() or pre_breakout_width != pre_breakout_width:
        return False
    threshold = float(reference_widths.quantile(config.squeeze_percentile / 100.0))
    if pre_breakout_width > threshold:
        return False

    previous = closed.iloc[-lookback - 1:-1]
    close = float(closed["close"].iloc[-1])
    return close > float(previous["high"].max()) or close < float(previous["low"].min())
