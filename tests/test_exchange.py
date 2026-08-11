"""거래소 클라이언트 단위 테스트.

가짜 transport를 끼워 실제 네트워크 없이 재시도·페이지네이션을 검사한다.
실통신 확인은 tools/check_candles.py 쪽에서 따로 한다.
"""

from __future__ import annotations

import httpx
import pytest

from config.symbols import XRP
from core import exchange as ex
from tests.test_candles import STEP, T0, raw_candle


def stub(handler, **kwargs) -> ex.GateClient:
    """backoff 0으로 두어 재시도 테스트가 지연 없이 끝나게 한다."""

    return ex.GateClient(
        transport=httpx.MockTransport(handler),
        backoff=0.0,
        **kwargs,
    )


# ------------------------------------------------------------------ 재시도


def test_5xx는_재시도_후_성공():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, json=[raw_candle(T0)])

    with stub(handler) as client:
        result = client.fetch("XRP_USDT", "5m", limit=1, drop_unclosed=False)

    assert len(calls) == 3
    assert len(result) == 1


def test_429도_재시도_대상():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, text="too many requests")

    with stub(handler, max_retries=2) as client:
        with pytest.raises(ex.ExchangeError):
            client.fetch("XRP_USDT", "5m")

    assert len(calls) == 2


def test_400은_재시도하지_않는다():
    """파라미터 오류는 몇 번을 보내도 같은 답이 온다."""

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400, text="invalid contract")

    with stub(handler) as client:
        with pytest.raises(ex.ExchangeError, match="400"):
            client.fetch("NOPE_USDT", "5m")

    assert len(calls) == 1


def test_네트워크_오류도_재시도():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("boom")

    with stub(handler, max_retries=3) as client:
        with pytest.raises(ex.ExchangeError):
            client.fetch("XRP_USDT", "5m")

    assert len(calls) == 3


# ------------------------------------------------------------------ fetch


def test_fetch_미마감봉_제거_기본동작():
    def handler(request):
        return httpx.Response(200, json=[raw_candle(T0 + i * STEP) for i in range(3)])

    with stub(handler) as client:
        result = client.fetch(
            "XRP_USDT", "5m", now_ts=T0 + 2 * STEP + 10
        )

    assert len(result) == 2
    assert result.ok


def test_fetch_문제_보고():
    """결측 캔들이 있으면 problems에 남고 ok가 False가 된다."""

    def handler(request):
        return httpx.Response(200, json=[raw_candle(T0), raw_candle(T0 + 3 * STEP)])

    with stub(handler) as client:
        result = client.fetch("XRP_USDT", "5m", drop_unclosed=False)

    assert not result.ok
    assert any("결측" in p for p in result.problems)


def test_fetch_limit_상한_초과는_사전_차단():
    with stub(lambda r: httpx.Response(200, json=[])) as client:
        with pytest.raises(ValueError, match="상한"):
            client.fetch("XRP_USDT", "5m", limit=ex.MAX_LIMIT + 1)


def test_fetch_요청_파라미터_확인():
    seen = {}

    def handler(request):
        seen.update(dict(request.url.params))
        return httpx.Response(200, json=[])

    with stub(handler) as client:
        client.fetch("XRP_USDT", "1h", limit=250)

    assert seen == {"contract": "XRP_USDT", "interval": "1h", "limit": "250"}


# ------------------------------------------------------------------ fetch_range


def test_fetch_range_페이지네이션(monkeypatch):
    """2000봉 제한을 넘는 구간은 나눠서 받고 하나로 합친다."""

    monkeypatch.setattr(ex, "MAX_LIMIT", 3)
    pages = []

    def handler(request):
        p = request.url.params
        start, end = int(p["from"]), int(p["to"])
        pages.append((start, end))
        ts = range(start, end + 1, STEP)
        return httpx.Response(200, json=[raw_candle(t) for t in ts])

    with stub(handler) as client:
        result = client.fetch_range(
            "XRP_USDT", "5m", T0, T0 + 7 * STEP, drop_unclosed=False
        )

    assert len(pages) > 1
    assert len(result) == 8                        # 0..7봉, 중복 없이
    assert result.df["ts"].is_monotonic_increasing
    assert result.df["ts"].duplicated().sum() == 0


def test_fetch_range_빈_구간_건너뛰기(monkeypatch):
    """거래가 없어 캔들이 안 오는 페이지가 있어도 멈추지 않는다."""

    monkeypatch.setattr(ex, "MAX_LIMIT", 2)

    def handler(request):
        start = int(request.url.params["from"])
        if start < T0 + 4 * STEP:
            return httpx.Response(200, json=[])
        end = int(request.url.params["to"])
        return httpx.Response(
            200, json=[raw_candle(t) for t in range(start, end + 1, STEP)]
        )

    with stub(handler) as client:
        result = client.fetch_range(
            "XRP_USDT", "5m", T0, T0 + 5 * STEP, drop_unclosed=False
        )

    assert len(result) == 2


def test_fetch_range_뒤집힌_구간은_예외():
    with stub(lambda r: httpx.Response(200, json=[])) as client:
        with pytest.raises(ValueError):
            client.fetch_range("XRP_USDT", "5m", T0 + STEP, T0)


# ------------------------------------------------------------------ fetch_frames


def test_fetch_frames_필요_타임프레임_전부():
    """XRP는 5분~1일봉을 전부 받는다.

    패턴마다 잘 맞는 봉이 달라(실측 double_top 5분 -0.21% / 1시간 +0.40%)
    인스턴스마다 다른 봉을 쓴다. 어느 패턴이든 동작하려면 전부 있어야 한다.
    """

    seen = []

    def handler(request):
        seen.append(request.url.params["interval"])
        return httpx.Response(200, json=[raw_candle(T0)])

    with stub(handler) as client:
        frames = client.fetch_frames(XRP, limit=1, now_ts=T0 + 10 * STEP)

    # 패턴마다 다른 봉을 쓰므로 전부 받는다 (build_strategies 참조).
    assert set(frames) == XRP.required_timeframes == {"5m", "15m", "1h", "4h", "1d"}
    assert seen == ["5m", "15m", "1h", "4h", "1d"]     # 짧은 것부터
    assert all(isinstance(v, ex.FetchResult) for v in frames.values())
