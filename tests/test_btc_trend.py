"""BTC 추세 전략과 확인 캔들 테스트.

가장 중요한 건 `test_상위봉_정렬은_마감된_봉만_본다`이다.
1시간봉에서 일봉을 참조할 때 미래를 보기 가장 쉬운 자리이고,
틀려도 결과가 "좋아지기만" 해서 눈에 띄지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators import patterns
from strategies.base import Signal
from strategies.btc_trend import BtcTrend, context_index


T0 = 1_700_000_000      # UTC 자정 정렬은 아니지만 상대 간격만 쓴다
HOUR = 3600
DAY = 86400


def ohlc(opens, highs, lows, closes, ts=None, step=HOUR):
    n = len(closes)
    ts = ts if ts is not None else [T0 + i * step for i in range(n)]
    return pd.DataFrame(
        {
            "ts": ts,
            "dt": pd.to_datetime(ts, unit="s", utc=True),
            "open": [float(x) for x in opens],
            "high": [float(x) for x in highs],
            "low": [float(x) for x in lows],
            "close": [float(x) for x in closes],
            "volume": [100.0] * n,
            "quote_volume": [100.0 * c for c in closes],
        }
    )


def _price(t: float, base: float = 60_000.0) -> float:
    """두 타임프레임이 공유하는 가격 함수.

    추세에 진동을 얹는다. 순수 직선이면 RSI가 100에 붙고 볼린저 폭이 0이라
    되돌림 진입 조건이 영원히 성립하지 않는다.
    """

    days = t / DAY
    return base + days * 120.0 + 900.0 * np.sin(days / 2.3) + 250.0 * np.sin(days * 3.7)


def trend_frames(n_hours=900, n_days=340):
    """상승 추세 1시간봉 + 일봉.

    일봉은 MA200을 채워야 하므로 1시간봉보다 훨씬 뒤에서 시작한다.
    실제 운용과 같은 구성이다 — 일봉 800봉, 1시간봉 500봉.
    """

    h_start = T0
    d_start = T0 + (n_hours - 1) * HOUR - (n_days - 1) * DAY

    h_ts = [h_start + i * HOUR for i in range(n_hours)]
    d_ts = [d_start + i * DAY for i in range(n_days)]

    return {
        "1h": _bars([_price(t - d_start) for t in h_ts], h_ts, HOUR, wick=60.0),
        "1d": _bars([_price(t - d_start) for t in d_ts], d_ts, DAY, wick=400.0),
    }


def _bars(closes, ts, step, wick):
    """종가 목록을 실제 캔들로.

    시가를 직전 종가로 잡아 몸통을 만든다. open == close로 두면 전부 도지가
    되어 확인 캔들 판정(꼬리 비율·인걸핑)이 구조적으로 성립하지 않는다.
    """

    opens = [closes[0]] + list(closes[:-1])
    highs = [max(o, c) + wick for o, c in zip(opens, closes)]
    lows = [min(o, c) - wick for o, c in zip(opens, closes)]
    return ohlc(opens, highs, lows, closes, ts=ts, step=step)


# ------------------------------------------------------------------ 확인 캔들


def test_되돌림_거부():
    """저가는 레벨 아래로 갔지만 종가는 위에서 마감."""

    df = ohlc([100], [102], [97], [101])
    assert bool(patterns.rejection(df, 98.0, "long").iloc[0])
    assert not bool(patterns.rejection(df, 102.5, "long").iloc[0])


def test_아래꼬리는_몸통의_2배_이상():
    long_wick = ohlc([100], [101], [94], [100.5])     # 몸통 0.5, 아래꼬리 6
    short_wick = ohlc([100], [105], [99.5], [104])    # 몸통 4, 아래꼬리 0.5

    assert bool(patterns.long_lower_wick(long_wick).iloc[0])
    assert not bool(patterns.long_lower_wick(short_wick).iloc[0])


def test_도지는_꼬리_비율로_통과시키지_않는다():
    """몸통이 0에 가까우면 비율이 무한대가 되어 아무 캔들이나 확인 캔들이 된다."""

    doji = ohlc([100], [103], [97], [100.0001])
    assert not bool(patterns.long_lower_wick(doji).iloc[0])


def test_인걸핑():
    df = ohlc([102, 99], [103, 104], [98, 98.5], [99, 103])
    assert bool(patterns.bullish_engulfing(df).iloc[1])


def test_확인캔들은_셋_중_하나면_성립():
    """원문이 '다음 중 하나'라 OR로 묶는다. AND면 신호가 거의 없다."""

    only_wick = ohlc([100], [101], [94], [100.5])
    assert bool(patterns.confirmation(only_wick, 200.0, "long").iloc[0])


def test_숏_확인캔들은_대칭():
    df = ohlc([100], [106], [99], [99.5])
    assert bool(patterns.rejection(df, 103.0, "short").iloc[0])


def test_잘못된_side는_예외():
    with pytest.raises(ValueError):
        patterns.confirmation(ohlc([1], [2], [0.5], [1.5]), 1.0, "up")


# ------------------------------------------------------------------ 상위봉 정렬


def test_상위봉_정렬은_마감된_봉만_본다():
    """일봉은 ts+86400이 지나야 마감된다. 같은 날짜 일봉을 쓰면 미래 참조다."""

    # 일봉 0: [0, 86400),  일봉 1: [86400, 172800)
    context_ts = np.array([0, DAY, 2 * DAY])

    # 1시간봉이 일봉 0 구간 한복판(12시)에 있으면 쓸 수 있는 일봉이 없다
    mid_day0 = np.array([12 * HOUR])
    assert context_index(mid_day0, context_ts)[0] == -1

    # 22:00~23:00 봉은 23:00에 마감된다. 일봉 0(24:00 마감)은 아직 못 쓴다.
    before = np.array([DAY - 2 * HOUR])
    assert context_index(before, context_ts)[0] == -1

    # 23:00~24:00 봉은 일봉 0과 **동시에** 마감된다. 이 시점부터 쓸 수 있다.
    simultaneous = np.array([DAY - HOUR])
    assert context_index(simultaneous, context_ts)[0] == 0

    # 그 다음 봉부터도 당연히 일봉 0
    after = np.array([DAY])
    assert context_index(after, context_ts)[0] == 0


def test_상위봉_정렬은_단조증가():
    entry_ts = np.arange(0, 10 * DAY, HOUR)
    context_ts = np.arange(0, 12 * DAY, DAY)

    idx = context_index(entry_ts, context_ts)
    assert (np.diff(idx) >= 0).all()


# ------------------------------------------------------------------ 전략


def test_국면이_아니면_신호가_없다():
    """일봉 MA200이 채워지기 전에는 진입 후보를 만들지 않는다."""

    frames = trend_frames(n_hours=300, n_days=60)     # MA200 불가
    s = BtcTrend()
    p = s.prepare(frames)

    assert (p.series["1h"]["ctx_regime"] == "none").all()
    assert [sig for i in range(200, 300) for sig in s.generate(p, i)] == []


def test_prepare가_일봉_값을_1시간봉_축에_올린다():
    frames = trend_frames()
    p = BtcTrend().prepare(frames)

    for key in ("ctx_regime", "ctx_ma_fast", "ctx_ma_slow", "ctx_index"):
        assert key in p.series["1h"]
        assert len(p.series["1h"][key]) == len(frames["1h"])


def test_ctx_index는_마감된_일봉을_가리킨다():
    """1시간봉 ts + 3600 이후에 마감되는 일봉을 가리키면 안 된다."""

    frames = trend_frames()
    p = BtcTrend().prepare(frames)
    h_ts = frames["1h"]["ts"].to_numpy()
    d_ts = frames["1d"]["ts"].to_numpy()

    for i in (0, 100, 500, len(h_ts) - 1):
        idx = int(p.at("1h", "ctx_index", i))
        assert idx >= 0
        assert d_ts[idx] + DAY <= h_ts[i] + HOUR       # 이미 마감됨
        if idx + 1 < len(d_ts):
            assert d_ts[idx + 1] + DAY > h_ts[i] + HOUR  # 다음 건 아직


def test_상승추세에서_롱_신호가_나온다():
    frames = trend_frames(n_hours=900)
    s = BtcTrend()
    p = s.prepare(frames)

    found = [sig for i in range(700, 900) for sig in s.generate(p, i)]

    assert found, "상승 추세 900봉에서 신호가 하나도 없다"
    assert all(sig.side == "long" for sig in found)
    assert all(sig.strategy == "btc_trend" for sig in found)


def test_신호는_손절이_진입가_아래():
    frames = trend_frames(n_hours=900)
    s = BtcTrend()
    p = s.prepare(frames)

    for i in range(700, 900):
        for sig in s.generate(p, i):
            assert sig.stop_hint < sig.entry
            assert sig.risk_distance() > 0


def test_신호는_근거와_지표를_남긴다():
    frames = trend_frames(n_hours=900)
    s = BtcTrend()
    p = s.prepare(frames)

    sigs = [sig for i in range(700, 900) for sig in s.generate(p, i)]
    sig = sigs[0]

    assert "confirmed" in sig.reasons
    assert any(r.startswith("regime_") for r in sig.reasons)
    assert sig.context["atr"] > 0
    assert sig.max_tranches == 3


def test_범위를_벗어난_as_of는_빈_목록():
    frames = trend_frames(n_hours=300)
    s = BtcTrend()
    p = s.prepare(frames)

    assert s.generate(p, 0) == []
    assert s.generate(p, 10_000) == []


def test_generate는_미래를_보지_않는다():
    """그 시점에 실제로 존재하던 데이터만 줘도 같은 신호가 나와야 한다.

    전체 프레임으로 낸 신호와, as_of 시점까지만 잘라낸 프레임으로 낸 신호를
    비교한다. 일봉은 "그때 이미 마감된 것"만 남긴다 — 실시간 루프가
    보게 되는 것과 정확히 같은 상태다.
    """

    full = trend_frames(n_hours=900)
    s = BtcTrend()
    p_full = s.prepare(full)

    h_ts = full["1h"]["ts"].to_numpy()
    d_ts = full["1d"]["ts"].to_numpy()

    for i in range(750, 900, 7):
        closed = (d_ts + DAY) <= (h_ts[i] + HOUR)
        past = {
            "1h": full["1h"].iloc[: i + 1].reset_index(drop=True),
            "1d": full["1d"].loc[closed].reset_index(drop=True),
        }
        p_past = s.prepare(past)

        a = s.generate(p_past, i)
        b = s.generate(p_full, i)

        assert len(a) == len(b), f"as_of={i}: 과거만 {len(a)}건 vs 전체 {len(b)}건"
        for x, y in zip(a, b):
            assert x.entry == pytest.approx(y.entry)
            assert x.stop_hint == pytest.approx(y.stop_hint)
            assert x.reasons == y.reasons
            assert x.score == pytest.approx(y.score)
