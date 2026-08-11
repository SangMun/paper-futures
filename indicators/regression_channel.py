# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Rolling-channel method adapted from the user-provided
# "Gaussian Channel Strategy [불스토리]" © bullstory1, which credits
# DonovanWall's "Gaussian Channel [DW]". This file only implements the
# rolling linear-regression channel portion.

"""과거 N봉만 사용하는 롤링 선형회귀 채널."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from indicators.base import IndicatorError, check_period, column


def channel(
    df: pd.DataFrame,
    length: int = 100,
    width_mult: float = 2.0,
    col: str = "close",
) -> pd.DataFrame:
    """회귀선 현재값, 상·하단, 기울기, R²를 돌려준다.

    Pine 코드의 표준편차 채널과 같은 정의를 쓴다.

      잔차 폭 = 가격 표준편차 × sqrt(1 - R²) × width_mult

    각 행은 그 행을 포함한 직전 ``length``개 확정 봉만 본다. 앞
    ``length - 1``개는 워밍업이라 NaN이다.
    """

    length = check_period(length, "length")
    if length < 2:
        raise IndicatorError(f"회귀 기간은 2 이상이어야 한다: {length}")
    if not math.isfinite(width_mult) or width_mult <= 0:
        raise IndicatorError(f"채널 폭 배수는 0보다 큰 유한값이어야 한다: {width_mult}")

    y = column(df, col).to_numpy(dtype="float64")
    names = ("mid", "upper", "lower", "slope", "r2", "deviation")
    values = {name: np.full(len(y), np.nan, dtype="float64") for name in names}
    if len(y) < length:
        return pd.DataFrame(values, index=df.index)

    windows = np.lib.stride_tricks.sliding_window_view(y, length)
    valid = np.isfinite(windows).all(axis=1)
    if not valid.any():
        return pd.DataFrame(values, index=df.index)

    x = np.arange(length, dtype="float64")
    x_mean = float(x.mean())
    x_centered = x - x_mean
    ss_x = float(np.dot(x_centered, x_centered))

    w = windows[valid]
    y_mean = w.mean(axis=1)
    y_centered = w - y_mean[:, None]
    slopes = (y_centered @ x_centered) / ss_x
    mids = y_mean + slopes * (length - 1 - x_mean)

    ss_y = np.square(y_centered).sum(axis=1)
    covariance_numerator = y_centered @ x_centered
    denom = ss_x * ss_y
    r2 = np.zeros(len(w), dtype="float64")
    nonflat = denom > 0
    r2[nonflat] = np.square(covariance_numerator[nonflat]) / denom[nonflat]
    r2 = np.clip(r2, 0.0, 1.0)

    sd_y = w.std(axis=1, ddof=0)
    deviation = sd_y * np.sqrt(np.maximum(1.0 - r2, 0.0)) * width_mult

    target_rows = np.flatnonzero(valid) + length - 1
    values["mid"][target_rows] = mids
    values["upper"][target_rows] = mids + deviation
    values["lower"][target_rows] = mids - deviation
    values["slope"][target_rows] = slopes
    values["r2"][target_rows] = r2
    values["deviation"][target_rows] = deviation
    return pd.DataFrame(values, index=df.index)
