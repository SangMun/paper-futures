"""거래량 판정.

차트 패턴 다섯 개가 전부 같은 두 가지를 본다.
  형성 중  — 거래량이 줄어드는가 (에너지 압축)
  돌파 시  — 거래량이 늘어나는가 (실제 참여)

"평균 대비 몇 배"를 절대 배수로 박지 않고 RVOL(상대 거래량)로 보는 이유는,
주식에서 쓰는 "평균의 50~60%" 같은 수치를 암호화폐에 그대로 옮길 근거가
없기 때문이다. 같은 종목의 최근 구간과 비교하는 편이 안전하다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.base import check_period, column


def rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """상대 거래량 = 이번 봉 / 직전 lookback봉 평균.

    자기 자신을 평균에 넣지 않는다. 넣으면 급증한 봉이 기준선을 함께
    끌어올려 배수가 작아진다.
    """

    lookback = check_period(lookback, "lookback")
    volume = column(df, "volume")
    baseline = volume.rolling(lookback, min_periods=lookback).mean().shift(1)
    return volume / baseline.where(baseline > 0)


def volume_trend(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """직전 window봉 거래량의 기울기. 음수면 감소 추세.

    평균 거래량으로 나눠 정규화하므로 종목·시기와 무관하게 비교된다.
    선형회귀 기울기를 쓰는 이유는 "처음보다 끝이 작다"만 보면
    중간에 급증했다 꺼진 경우를 감지하지 못하기 때문이다.
    """

    window = check_period(window, "window")
    volume = column(df, "volume").to_numpy(dtype="float64")
    n = len(volume)
    out = np.full(n, np.nan)

    if n < window:
        return pd.Series(out, index=df.index)

    x = np.arange(window, dtype="float64")
    x_centered = x - x.mean()
    denom = (x_centered**2).sum()

    for i in range(window - 1, n):
        seg = volume[i - window + 1 : i + 1]
        mean = seg.mean()
        if mean <= 0:
            continue
        slope = (x_centered * (seg - mean)).sum() / denom
        out[i] = slope / mean          # 평균 대비 봉당 변화율

    return pd.Series(out, index=df.index)


def declining_between(df: pd.DataFrame, start: int, end: int) -> bool:
    """[start, end] 구간에서 거래량이 줄었는가.

    패턴 형성 구간처럼 길이가 매번 다른 곳에 쓴다. 전반부 평균과
    후반부 평균을 비교한다 — 구간이 짧아 회귀선을 그리기 어렵다.
    """

    if end <= start or start < 0 or end >= len(df):
        return False

    seg = column(df, "volume").iloc[start : end + 1].to_numpy(dtype="float64")
    if len(seg) < 4:
        return False

    half = len(seg) // 2
    first, second = seg[:half].mean(), seg[half:].mean()
    return bool(first > 0 and second < first)
