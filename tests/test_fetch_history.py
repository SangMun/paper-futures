"""fetch_history.backfill의 최근분 요청 범위."""

import pandas as pd

from tools import fetch_history


class _Client:
    def __init__(self):
        self.calls = []

    def fetch_range(self, contract, interval, start, end):
        self.calls.append((start, end))
        return type("R", (), {"df": pd.DataFrame()})()


def test_stale_cache_requests_only_within_exchange_limit(monkeypatch):
    now = 1_800_000_000
    days = 32
    last = now - 50 * 86400  # 캐시 끝이 상한(32일)보다 오래됨
    monkeypatch.setattr(fetch_history.cache, "coverage", lambda c, i: (last - 86400, last))
    monkeypatch.setattr(fetch_history.cache, "load", lambda c, i: pd.DataFrame())

    client = _Client()
    fetch_history.backfill(client, "XRP_USDT", "5m", days, now=now)

    want_start = now - days * 86400
    assert (want_start, now) in client.calls
    assert all(start >= want_start for start, _ in client.calls)


def test_fresh_cache_resumes_from_cache_end(monkeypatch):
    now = 1_800_000_000
    last = now - 3 * 86400
    monkeypatch.setattr(fetch_history.cache, "coverage", lambda c, i: (now - 32 * 86400, last))
    monkeypatch.setattr(fetch_history.cache, "load", lambda c, i: pd.DataFrame())

    client = _Client()
    fetch_history.backfill(client, "XRP_USDT", "5m", 32, now=now)

    assert client.calls == [(last, now)]
