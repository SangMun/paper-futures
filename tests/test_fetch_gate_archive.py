"""Gate 월별 캔들 아카이브 변환 테스트. 네트워크는 사용하지 않는다."""

from __future__ import annotations

import gzip
from datetime import datetime, timezone

import pytest

from core import candles as cd
from tools import fetch_gate_archive as arc


def payload(*rows: str) -> bytes:
    return gzip.compress(("\n".join(rows) + "\n").encode())


def test_archive_url():
    month = datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert arc.archive_url("XRP_USDT", "5m", month) == (
        "https://download.gatedata.org/futures_usdt/"
        "candlesticks_5m/202607/XRP_USDT-202607.csv.gz"
    )


def test_month_starts는_현재_미완결월을_제외():
    start = int(datetime(2026, 2, 6, tzinfo=timezone.utc).timestamp())
    end = int(datetime(2026, 8, 5, tzinfo=timezone.utc).timestamp())

    months = arc.month_starts(start, end)

    assert [m.strftime("%Y%m") for m in months] == [
        "202602", "202603", "202604", "202605", "202606", "202607"
    ]


def test_parse_archive_컬럼_순서는_OHLC가_아니다():
    """아카이브는 t,v,**c**,h,l,**o** 순이다. OHLC로 읽으면 시가와 종가가 뒤바뀐다.

    이 테스트가 없으면 순서를 틀려도 아무것도 잡히지 않는다.
    고가·저가는 제자리라 `candles.validate()`를 통과하고, 차트도 정상으로 보인다.
    깨지는 건 양봉/음봉 판정뿐이다 — 인걸핑, 확인 캔들(명세 §5.3), 아래꼬리,
    XRP 패턴이 전부 조용히 반대로 동작한다.

    2026-07 XRP_USDT 5분봉 실제 첫 줄을 그대로 쓴다.
    네 값이 서로 달라야 어떤 순열로 틀려도 걸린다.
    """

    df = arc.parse_archive(payload("1782864000,10655,1.0404,1.0410,1.0392,1.0397"), "XRP_USDT")
    row = df.iloc[0]

    assert row["open"] == pytest.approx(1.0397)      # 5번째 필드
    assert row["high"] == pytest.approx(1.0410)
    assert row["low"] == pytest.approx(1.0392)
    assert row["close"] == pytest.approx(1.0404)     # 3번째 필드
    assert row["open"] != row["close"]               # 뒤바꿈이 보이는 값인가


def test_parse_archive_표준컬럼과_거래대금_근사():
    df = arc.parse_archive(
        payload(
            "1782864000,100,1.04,1.05,1.03,1.04",
            "1782864300,200,1.05,1.06,1.04,1.04",
        ),
        "XRP_USDT",
    )

    assert list(df.columns) == list(cd.COLUMNS)
    assert df["ts"].tolist() == [1782864000, 1782864300]
    assert df["volume"].tolist() == [100.0, 200.0]
    # 아카이브에는 quote_volume이 없어 재구성한다.
    # XRP 계약 승수 10 * HLC3 1.04 * 100계약
    assert df["quote_volume"].iloc[0] == pytest.approx(1040.0)


def test_parse_archive_깨진_gzip은_예외():
    with pytest.raises(arc.ArchiveError, match="파싱 실패"):
        arc.parse_archive(b"not-gzip", "XRP_USDT")


def test_결측_캔들은_경고이고_가격_NaN은_치명적():
    problems = ["결측 캔들 4봉 (첫 구간 ...)", "가격에 NaN 포함"]

    assert arc.critical_problems(problems) == ["가격에 NaN 포함"]
