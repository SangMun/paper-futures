"""지표 공통 규약.

모든 지표는 입력 DataFrame과 **같은 인덱스**를 갖는 Series/DataFrame을 돌려준다.
전략이 `as_of` 위치 인덱스로 값을 꺼내 쓰기 때문에, 길이가 달라지거나
인덱스가 밀리면 다른 봉의 값을 읽게 된다.

워밍업 구간은 NaN으로 남긴다. 채워 넣지 않는 이유는 계산이 안정되지 않은
구간을 전략이 구분할 수 없게 되기 때문이다. 그 구간에서 나온 신호는
실전에서 재현되지 않는다.

데이터가 모자랄 때 예외를 던지지 않고 전부 NaN을 돌려주는 것도 같은 이유다.
캔들 수집이 잠깐 모자란다고 루프 전체가 죽으면 안 되고, NaN은 전략 쪽에서
자연스럽게 "신호 없음"으로 처리된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class IndicatorError(ValueError):
    """입력 자체가 잘못된 경우. 데이터 부족은 여기에 해당하지 않는다."""


def column(df: pd.DataFrame, name: str) -> pd.Series:
    """지정 컬럼을 float Series로. 없으면 예외."""

    if name not in df.columns:
        raise IndicatorError(f"컬럼 없음: {name!r} (보유: {list(df.columns)})")
    return df[name].astype("float64")


def check_period(period: int, name: str = "period") -> int:
    if not isinstance(period, (int, np.integer)) or period < 1:
        raise IndicatorError(f"{name}는 1 이상의 정수여야 한다: {period!r}")
    return int(period)


def empty_like(df: pd.DataFrame) -> pd.Series:
    """전부 NaN인 Series. 데이터 부족 시 반환값."""

    return pd.Series(np.nan, index=df.index, dtype="float64")


def ema_series(values: pd.Series, period: int) -> pd.Series:
    """Series 단위 EMA. 앞 period개의 SMA를 시드로 잡는다.

    pandas의 ewm을 그대로 쓰면 첫 봉부터 값이 나오는데 그건 첫 값 자체라
    평균이 아니다. 워밍업을 NaN으로 남겨야 전략이 "아직 못 믿는 값"을 거른다.

    선행 NaN은 건너뛰고 그 뒤부터 시작한다. MACD 시그널선처럼
    이미 NaN이 앞에 붙은 Series를 다시 평활할 때 필요하다.
    """

    period = check_period(period)
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    arr = values.to_numpy(dtype="float64")

    valid = np.flatnonzero(~np.isnan(arr))
    if len(valid) < period:
        return out

    start = valid[0]
    seed_end = start + period
    if seed_end > len(arr):
        return out

    # numpy 배열에 채운 뒤 마지막에 한 번만 Series로 감싼다.
    # `out.iloc[i] = ...`로 봉마다 대입하면 pandas 인덱싱이 호출당
    # 수 마이크로초씩 붙어, 5분봉 51,835봉 리플레이에서 이 한 줄이
    # 전체 시간의 대부분을 차지한다(실측 300만 회 대입에 55초).
    # 계산 순서와 산술은 그대로라 결과가 비트 단위로 같다.
    alpha = 2.0 / (period + 1)
    prev = float(np.mean(arr[start:seed_end]))
    res = np.full(len(arr), np.nan)
    res[seed_end - 1] = prev

    for i in range(seed_end, len(arr)):
        x = arr[i]
        if not np.isnan(x):
            prev = prev + alpha * (x - prev)
        res[i] = prev

    return pd.Series(res, index=values.index, dtype="float64")


def wilder(values: pd.Series, period: int) -> pd.Series:
    """Wilder 평활 (RMA). ATR과 RSI가 공유한다.

    일반 EMA(span=n)와 다르다. Wilder의 alpha는 1/n이고 EMA는 2/(n+1)이라
    같은 period를 넣어도 값이 다르게 나온다. 차트 프로그램이 보여주는
    ATR/RSI는 Wilder 기준이므로, 눈으로 검증할 때 값이 맞으려면 이쪽이어야 한다.

    첫 값은 앞 period개의 단순평균으로 시작한다. 그 앞은 NaN.
    """

    period = check_period(period)
    if len(values) < period:
        return pd.Series(np.nan, index=values.index, dtype="float64")

    out = pd.Series(np.nan, index=values.index, dtype="float64")
    arr = values.to_numpy(dtype="float64")

    # 앞쪽에 NaN이 섞여 있으면 그 뒤부터 시작한다.
    # (true_range의 첫 값처럼 구조적으로 비는 경우)
    valid = np.flatnonzero(~np.isnan(arr))
    if len(valid) < period:
        return out

    start = valid[0]
    seed_end = start + period
    if seed_end > len(arr):
        return out

    # ema_series와 같은 이유로 numpy에 채운다.
    prev = float(np.mean(arr[start:seed_end]))
    res = np.full(len(arr), np.nan)
    res[seed_end - 1] = prev

    alpha = 1.0 / period
    for i in range(seed_end, len(arr)):
        x = arr[i]
        if not np.isnan(x):
            prev = prev + alpha * (x - prev)
        res[i] = prev

    return pd.Series(res, index=values.index, dtype="float64")
