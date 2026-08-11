"""캔들 정제 단위 테스트.

네트워크를 타지 않는다. 거래소 응답 형태만 흉내 낸 딕셔너리로 검사한다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core import candles as cd


STEP = 300  # 5분
T0 = 1_700_000_000


def raw_candle(t, o=1.0, h=1.1, l=0.9, c=1.05, v=100, s=None):
    return {
        "t": t,
        "o": str(o),
        "h": str(h),
        "l": str(l),
        "c": str(c),
        "v": v,
        "sum": str(v * c if s is None else s),
    }


def make_raw(n, start=T0, step=STEP):
    return [raw_candle(start + i * step) for i in range(n)]


# ------------------------------------------------------------------ 파싱


def test_to_dataframe_기본_형태():
    df = cd.to_dataframe(make_raw(3))

    assert list(df.columns) == list(cd.COLUMNS)
    assert len(df) == 3
    assert df["ts"].tolist() == [T0, T0 + STEP, T0 + 2 * STEP]
    assert df["open"].dtype == "float64"
    # as_of가 위치 인덱스이므로 RangeIndex를 유지해야 한다
    assert df.index.tolist() == [0, 1, 2]


def test_to_dataframe_시간순_정렬():
    raw = make_raw(3)
    df = cd.to_dataframe([raw[2], raw[0], raw[1]])

    assert df["ts"].is_monotonic_increasing


def test_to_dataframe_중복은_나중_것_채택():
    """같은 봉이 두 번 오면 최신 값이 이긴다. 미마감 봉이 갱신될 때의 상황."""

    old = raw_candle(T0, c=1.00)
    new = raw_candle(T0, c=1.50)
    df = cd.to_dataframe([old, new])

    assert len(df) == 1
    assert df["close"].iloc[0] == 1.50


def test_to_dataframe_빈_리스트():
    df = cd.to_dataframe([])

    assert df.empty
    assert list(df.columns) == list(cd.COLUMNS)
    assert df["ts"].dtype == "int64"


def test_to_dataframe_sum_없어도_통과():
    item = raw_candle(T0)
    del item["sum"]

    df = cd.to_dataframe([item])

    assert df["quote_volume"].iloc[0] == 0.0


def test_to_dataframe_잘못된_형식은_예외():
    with pytest.raises(cd.CandleError):
        cd.to_dataframe([{"t": T0, "o": "1.0"}])          # 필드 누락
    with pytest.raises(cd.CandleError):
        cd.to_dataframe(["not a dict"])
    with pytest.raises(cd.CandleError):
        cd.to_dataframe({"t": T0})                        # 리스트가 아님


# ------------------------------------------------------------------ 미마감 제거


def test_drop_unclosed_마지막봉_제거():
    """마지막 봉이 아직 진행 중이면 잘라낸다."""

    df = cd.to_dataframe(make_raw(3))
    now = T0 + 2 * STEP + 100          # 마지막 봉 시작 후 100초

    out = cd.drop_unclosed(df, "5m", now_ts=now)

    assert len(out) == 2
    assert out["ts"].iloc[-1] == T0 + STEP


def test_drop_unclosed_경계_시각은_마감으로_본다():
    df = cd.to_dataframe(make_raw(3))
    now = T0 + 3 * STEP                # 마지막 봉이 막 닫힌 순간

    out = cd.drop_unclosed(df, "5m", now_ts=now)

    assert len(out) == 3


def test_drop_unclosed_인덱스_재설정():
    df = cd.to_dataframe(make_raw(3))
    out = cd.drop_unclosed(df, "5m", now_ts=T0 + 2 * STEP + 1)

    assert out.index.tolist() == [0, 1]


def test_drop_unclosed_빈_입력():
    assert cd.drop_unclosed(cd.to_dataframe([]), "5m", now_ts=T0).empty


def test_interval_seconds_미지원_타임프레임():
    with pytest.raises(cd.CandleError):
        cd.interval_seconds("3m")


# ------------------------------------------------------------------ 검증


def test_validate_정상_데이터는_문제없음():
    df = cd.to_dataframe(make_raw(10))

    assert cd.validate(df, "5m") == []


def test_validate_결측_캔들_탐지():
    raw = make_raw(5)
    del raw[2]                          # 한 봉 통째로 누락

    problems = cd.validate(cd.to_dataframe(raw), "5m")

    assert any("결측 캔들 1봉" in p for p in problems)


def test_validate_결측_개수_집계():
    raw = [raw_candle(T0), raw_candle(T0 + 4 * STEP)]   # 사이 3봉 누락

    problems = cd.validate(cd.to_dataframe(raw), "5m")

    assert any("결측 캔들 3봉" in p for p in problems)


def test_validate_고가_저가_모순():
    df = cd.to_dataframe([raw_candle(T0, o=1.0, h=0.5, l=0.9, c=1.05)])

    problems = cd.validate(df, "5m")

    assert any("고가" in p for p in problems)


def test_validate_음수_가격():
    df = cd.to_dataframe([raw_candle(T0, o=-1.0)])

    assert any("0 이하" in p for p in cd.validate(df, "5m"))


def test_validate_빈_데이터():
    assert cd.validate(cd.to_dataframe([]), "5m") == ["캔들이 비어 있음"]


# ------------------------------------------------------------------ 병합


def test_merge_이어붙이기():
    old = cd.to_dataframe(make_raw(3))
    new = cd.to_dataframe(make_raw(3, start=T0 + 3 * STEP))

    out = cd.merge(old, new)

    assert len(out) == 6
    assert out["ts"].is_monotonic_increasing
    assert out.index.tolist() == list(range(6))


def test_merge_겹치는_구간은_신규_우선():
    old = cd.to_dataframe([raw_candle(T0, c=1.0), raw_candle(T0 + STEP, c=1.0)])
    new = cd.to_dataframe([raw_candle(T0 + STEP, c=2.0), raw_candle(T0 + 2 * STEP, c=2.0)])

    out = cd.merge(old, new)

    assert len(out) == 3
    assert out.loc[out["ts"] == T0 + STEP, "close"].iloc[0] == 2.0


def test_merge_한쪽이_비어도_동작():
    empty = cd.to_dataframe([])
    df = cd.to_dataframe(make_raw(2))

    assert len(cd.merge(empty, df)) == 2
    assert len(cd.merge(df, empty)) == 2
    assert cd.merge(empty, empty).empty


# ------------------------------------------------------------------ 합성


ALIGNED = T0 // 900 * 900 + 900      # 15분 경계에 맞춘 시작점


def test_resample_5분봉을_15분봉으로():
    """Gate 아카이브에 15분봉이 없어 5분봉에서 만든다."""

    raw = [
        raw_candle(ALIGNED + 0 * STEP, o=1.0, h=1.5, l=0.9, c=1.2, v=10),
        raw_candle(ALIGNED + 1 * STEP, o=1.2, h=1.8, l=1.1, c=1.6, v=20),
        raw_candle(ALIGNED + 2 * STEP, o=1.6, h=1.7, l=1.3, c=1.4, v=30),
    ]
    out = cd.resample(cd.to_dataframe(raw), "5m", "15m")

    assert len(out) == 1
    row = out.iloc[0]
    assert row["open"] == 1.0          # 첫 시가
    assert row["high"] == 1.8          # 구간 최고
    assert row["low"] == 0.9           # 구간 최저
    assert row["close"] == 1.4         # 마지막 종가
    assert row["volume"] == 60         # 합계


def test_resample_미완성_구간은_버린다():
    """3개짜리 구간에 2개만 있으면 고저가 실제보다 좁아진다."""

    raw = [raw_candle(ALIGNED + i * STEP) for i in range(5)]   # 15분봉 1개 + 5분봉 2개
    out = cd.resample(cd.to_dataframe(raw), "5m", "15m")

    assert len(out) == 1


def test_resample_하위봉으로는_못_간다():
    df = cd.to_dataframe([raw_candle(T0)])

    with pytest.raises(cd.CandleError):
        cd.resample(df, "1h", "5m")     # 없는 정보를 만들 수 없다
    with pytest.raises(cd.CandleError):
        cd.resample(df, "1h", "1h")     # 같은 봉도 의미 없다
