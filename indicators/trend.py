"""추세 강도 지표.

방향은 이동평균 배열로 정하고, ADX는 그 방향성이 충분히 강한지를 확인한다.
ADX 자체에는 상승/하락 방향이 없으므로 +DI/-DI를 함께 돌려준다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.atr import true_range
from indicators.base import check_period, column, wilder


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder 방식 ADX와 +DI/-DI.

    반환 열은 ``adx``, ``plus_di``, ``minus_di``다. 첫 ADX는 DM/TR을 한 번
    평활한 뒤 DX를 다시 평활해야 하므로 DI보다 늦게 시작한다.
    """

    period = check_period(period)
    index = df.index
    if len(df) < period + 1:
        empty = pd.Series(np.nan, index=index, dtype="float64")
        return pd.DataFrame(
            {"adx": empty.copy(), "plus_di": empty.copy(), "minus_di": empty.copy()},
            index=index,
        )

    high = column(df, "high")
    low = column(df, "low")
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=index, dtype="float64"
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=index, dtype="float64"
    )
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan

    tr_smoothed = wilder(true_range(df), period)
    plus_di = 100.0 * wilder(plus_dm, period) / tr_smoothed
    minus_di = 100.0 * wilder(minus_dm, period) / tr_smoothed

    denominator = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / denominator.replace(0.0, np.nan)
    dx = dx.mask(denominator == 0.0, 0.0)
    adx_series = wilder(dx, period)

    return pd.DataFrame(
        {"adx": adx_series, "plus_di": plus_di, "minus_di": minus_di},
        index=index,
    )
