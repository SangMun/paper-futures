"""지표 단위 테스트.

손으로 계산 가능한 고정 입력만 쓴다. 실데이터로 "그럴듯한 값이 나온다"를
확인하는 건 검증이 아니다 — 부호가 뒤집혀도 그럴듯해 보인다.

인덱스 정렬과 워밍업 NaN을 매 지표마다 확인하는 이유는, 이 둘이 깨지면
전략이 다른 봉의 값을 읽거나 안정되지 않은 값으로 진입하게 되는데
둘 다 조용히 일어나기 때문이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators import atr as atr_mod
from indicators import bollinger as bb
from indicators import ma
from indicators import oscillators as osc
from indicators import trend
from indicators import vpvr
from indicators.base import IndicatorError, ema_series, wilder


T0 = 1_700_000_000
STEP = 3600


def frame(closes, highs=None, lows=None, opens=None, volumes=None) -> pd.DataFrame:
    """종가 목록으로 최소한의 OHLCV DataFrame을 만든다."""

    n = len(closes)
    closes = [float(c) for c in closes]
    return pd.DataFrame(
        {
            "ts": [T0 + i * STEP for i in range(n)],
            "dt": pd.to_datetime([T0 + i * STEP for i in range(n)], unit="s", utc=True),
            "open": [float(x) for x in (opens or closes)],
            "high": [float(x) for x in (highs or [c * 1.01 for c in closes])],
            "low": [float(x) for x in (lows or [c * 0.99 for c in closes])],
            "close": closes,
            "volume": [float(x) for x in (volumes or [100.0] * n)],
            "quote_volume": [
                float(v) * c for v, c in zip(volumes or [100.0] * n, closes)
            ],
        }
    )


# ------------------------------------------------------------------ 평활 함수


def test_wilder_시드는_단순평균():
    """첫 유효값은 앞 period개의 평균이어야 한다."""

    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = wilder(s, 3)

    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(2.0)          # (1+2+3)/3
    assert out.iloc[3] == pytest.approx(2.0 + (4.0 - 2.0) / 3)


def test_wilder_와_ema는_다르다():
    """alpha가 1/n vs 2/(n+1)이라 같은 period여도 값이 갈린다."""

    s = pd.Series([float(i) for i in range(1, 21)])
    assert wilder(s, 5).iloc[-1] != pytest.approx(ema_series(s, 5).iloc[-1])


def test_ema_선행_NaN을_건너뛴다():
    """MACD 시그널선처럼 앞이 비어 있는 Series에도 그대로 쓸 수 있어야 한다."""

    s = pd.Series([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0])
    out = ema_series(s, 3)

    assert out.iloc[:4].isna().all()
    assert out.iloc[4] == pytest.approx(2.0)          # (1+2+3)/3


# ------------------------------------------------------------------ 이동평균


def test_sma_워밍업은_NaN():
    df = frame([1, 2, 3, 4, 5])
    out = ma.sma(df, 3)

    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)
    assert out.index.equals(df.index)


def test_ema_데이터_부족하면_전부_NaN():
    df = frame([1, 2])
    assert ma.ema(df, 5).isna().all()


def test_regime_정배열과_역배열():
    up = frame(list(range(1, 61)))
    down = frame(list(range(60, 0, -1)))

    assert ma.regime(up, fast=3, slow=10).iloc[-1] == "bull"
    assert ma.regime(down, fast=3, slow=10).iloc[-1] == "bear"


def test_regime_워밍업은_none():
    df = frame(list(range(1, 21)))
    assert (ma.regime(df, fast=5, slow=50) == "none").all()


def test_regime_크로스만으로는_bull이_아니다():
    """빠른선이 느린선 위여도 가격이 빠른선 아래면 none.

    두 선이 붙어 오르내리는 횡보에서 국면이 계속 뒤집히는 걸 막는 조건이다.
    """

    closes = list(range(1, 41)) + [20.0]     # 마지막에 급락
    df = frame(closes)
    r = ma.regime(df, fast=5, slow=20)

    assert r.iloc[-2] == "bull"
    assert r.iloc[-1] == "none"


def test_touched_꼬리로만_닿아도_인정():
    """종가가 위에서 마감해도 저가가 선을 건드렸으면 눌림목이다."""

    df = frame([10.0, 10.0], highs=[11.0, 11.0], lows=[9.0, 9.5])
    line = pd.Series([9.2, 9.2])

    out = ma.touched(df, line, tolerance=0.0)
    assert bool(out.iloc[0]) is True      # 저가 9.0 <= 9.2
    assert bool(out.iloc[1]) is False     # 저가 9.5 > 9.2


def test_ADX는_방향DI와_같은_인덱스를_돌려준다():
    up = frame(list(range(1, 80)))
    out = trend.adx(up, 14)

    assert list(out.columns) == ["adx", "plus_di", "minus_di"]
    assert out.index.equals(up.index)
    assert out["adx"].iloc[-1] > 20
    assert out["plus_di"].iloc[-1] > out["minus_di"].iloc[-1]


def test_ADX_데이터가_모자라면_전부_NaN():
    out = trend.adx(frame([1, 2, 3]), 14)

    assert out.isna().all().all()


# ------------------------------------------------------------------ ATR


def test_true_range_갭을_잡는다():
    """고저폭은 좁지만 전종가에서 뛴 봉의 TR은 갭만큼 커야 한다."""

    df = frame(
        closes=[10.0, 20.0],
        highs=[10.0, 20.5],
        lows=[10.0, 19.5],
    )
    tr = atr_mod.true_range(df)

    assert np.isnan(tr.iloc[0])
    # max(20.5-19.5, |20.5-10|, |19.5-10|) = 10.5
    assert tr.iloc[1] == pytest.approx(10.5)


def test_atr_일정한_봉이면_고저폭과_같다():
    df = frame([10.0] * 20, highs=[11.0] * 20, lows=[9.0] * 20)
    out = atr_mod.atr(df, 14)

    assert out.iloc[:13].isna().all()
    assert out.iloc[-1] == pytest.approx(2.0)


def test_atr_데이터_부족():
    assert atr_mod.atr(frame([1, 2, 3]), 14).isna().all()


def test_atr_pct_가격_수준에_무관():
    """자릿수가 다른 두 종목이 같은 변동성이면 같은 값이 나와야 한다."""

    small = frame([1.0] * 20, highs=[1.02] * 20, lows=[0.98] * 20)
    large = frame([1000.0] * 20, highs=[1020.0] * 20, lows=[980.0] * 20)

    assert atr_mod.atr_pct(small, 14).iloc[-1] == pytest.approx(
        atr_mod.atr_pct(large, 14).iloc[-1]
    )


# ------------------------------------------------------------------ RSI / MACD


def test_rsi_연속_상승이면_100():
    df = frame([float(i) for i in range(1, 31)])
    assert osc.rsi(df, 14).iloc[-1] == pytest.approx(100.0)


def test_rsi_연속_하락이면_0():
    df = frame([float(i) for i in range(30, 0, -1)])
    assert osc.rsi(df, 14).iloc[-1] == pytest.approx(0.0)


def test_rsi_등락이_같으면_50_부근():
    """상승분과 하락분의 크기가 같으면 50 근처.

    정확히 50이 되지는 않는다. Wilder 평활은 마지막 봉의 방향이 직전 값에
    바로 반영되므로, 하락으로 끝나면 50보다 조금 아래에서 마감한다.
    그래서 상승/하락으로 끝나는 두 경우를 같이 확인한다.
    """

    closes = [10.0]
    for i in range(40):
        closes.append(closes[-1] + (1.0 if i % 2 == 0 else -1.0))

    ends_down = osc.rsi(frame(closes), 14).iloc[-1]
    ends_up = osc.rsi(frame(closes + [closes[-1] + 1.0]), 14).iloc[-1]

    assert ends_down < 50.0 < ends_up
    assert ends_down == pytest.approx(50.0, abs=3.0)
    assert ends_up == pytest.approx(50.0, abs=3.0)


def test_rsi_워밍업_구간():
    df = frame([float(i) for i in range(1, 31)])
    out = osc.rsi(df, 14)

    assert out.iloc[:14].isna().all()
    assert out.notna().iloc[14]


def test_macd_상승추세에서_양수_히스토그램():
    df = frame([float(i) for i in range(1, 121)])
    m = osc.macd(df)

    assert m.iloc[-1]["macd"] > 0
    assert list(m.columns) == ["macd", "signal", "hist"]


def test_macd_시그널선이_더_늦게_시작():
    """MACD선이 유효해진 뒤에야 시그널선을 굴릴 수 있다."""

    df = frame([float(i) for i in range(1, 121)])
    m = osc.macd(df, fast=12, slow=26, signal=9)

    first_macd = m["macd"].first_valid_index()
    first_signal = m["signal"].first_valid_index()
    assert first_macd == 25                      # slow - 1
    assert first_signal == first_macd + 8        # signal - 1 만큼 더


def test_macd_fast가_slow보다_크면_예외():
    with pytest.raises(IndicatorError):
        osc.macd(frame([1.0] * 60), fast=26, slow=12)


def test_macd_bullish는_0선_위를_요구():
    """하락 추세 중 반등에서 크로스가 나와도 롱 조건이 되면 안 된다."""

    closes = [float(i) for i in range(200, 60, -1)] + [70.0 + i for i in range(15)]
    df = frame(closes)
    m = osc.macd(df)
    bull = osc.macd_bullish(df)

    crossed = m["macd"] > m["signal"]
    below_zero = m["macd"] < 0
    mask = crossed & below_zero
    assert mask.any()                       # 그런 구간이 실제로 존재하고
    assert not bull[mask].any()             # 거기서는 전부 False


# ------------------------------------------------------------------ 볼린저


def test_bollinger_변동_없으면_밴드폭_0():
    df = frame([10.0] * 30)
    b = bb.bollinger(df, 20, 2.0)

    assert b["mid"].iloc[-1] == pytest.approx(10.0)
    assert b["bandwidth"].iloc[-1] == pytest.approx(0.0)


def test_bollinger_표준편차_배수가_반영된다():
    closes = [10.0, 12.0] * 15
    df = frame(closes)
    b1 = bb.bollinger(df, 20, 1.0)
    b2 = bb.bollinger(df, 20, 2.0)

    w1 = b1["upper"].iloc[-1] - b1["mid"].iloc[-1]
    w2 = b2["upper"].iloc[-1] - b2["mid"].iloc[-1]
    assert w2 == pytest.approx(w1 * 2)


def test_bollinger_워밍업():
    df = frame([10.0] * 30)
    assert bb.bollinger(df, 20)["mid"].iloc[:19].isna().all()


def test_squeeze_수축_구간을_찾는다():
    """넓게 흔들리다가 좁아지면 뒤쪽이 스퀴즈로 잡혀야 한다."""

    wide = [10.0 + (3.0 if i % 2 else -3.0) for i in range(80)]
    tight = [10.0 + (0.05 if i % 2 else -0.05) for i in range(40)]
    df = frame(wide + tight)

    sq = bb.squeeze(df, period=20, percentile=20.0, lookback=100)
    assert not sq.iloc[:80].any()
    assert bool(sq.iloc[-1]) is True


def test_squeeze_미래를_보지_않는다():
    """뒤에 데이터를 더 붙여도 앞 구간의 판정이 바뀌면 안 된다."""

    rng = np.random.default_rng(0)
    closes = list(10.0 + rng.normal(0, 0.3, 300).cumsum())

    a = bb.squeeze(frame(closes[:200]), period=20, lookback=100)
    b = bb.squeeze(frame(closes), period=20, lookback=100)

    assert a.tolist() == b.iloc[:200].tolist()


def test_squeeze_percentile_범위_검사():
    with pytest.raises(IndicatorError):
        bb.squeeze(frame([10.0] * 50), percentile=0.0)


def test_position_밴드_이탈은_0_1_밖():
    closes = [10.0, 12.0] * 14 + [30.0]
    df = frame(closes)
    assert bb.position(df, 20).iloc[-1] > 1.0


# ------------------------------------------------------------------ VPVR


def test_profile_거래량이_몰린_가격대가_POC():
    """대부분의 봉이 머문 가격대가 POC로 잡혀야 한다."""

    closes = [10.0] * 90 + [float(11 + i * 0.1) for i in range(30)]
    highs = [c + 0.05 for c in closes]
    lows = [c - 0.05 for c in closes]
    df = frame(closes, highs=highs, lows=lows)

    p = vpvr.profile(df, as_of=len(df) - 1, lookback=200, num_bins=50)
    assert p is not None
    assert p.poc == pytest.approx(10.0, abs=0.3)


def test_profile_as_of_이후를_보지_않는다():
    """as_of 뒤에 전혀 다른 가격대를 붙여도 결과가 같아야 한다."""

    base = [10.0] * 120
    df_short = frame(base)
    df_long = frame(base + [50.0] * 60)

    a = vpvr.profile(df_short, as_of=119, lookback=100, num_bins=20)
    b = vpvr.profile(df_long, as_of=119, lookback=100, num_bins=20)

    assert a is not None and b is not None
    assert a.poc == pytest.approx(b.poc)
    assert np.allclose(a.volumes, b.volumes)


def test_profile_연속_HVN은_하나로_묶인다():
    """붙어 있는 빈들이 각각 레벨로 나오면 익절 목표가 중복된다."""

    closes = [10.0] * 150
    df = frame(closes, highs=[10.02] * 150, lows=[9.98] * 150)

    p = vpvr.profile(df, as_of=149, lookback=150, num_bins=50)
    assert p is not None
    assert len(p.hvn) <= 3


def test_최근접_노드_조회():
    """탐색 자체의 동작. 레벨 목록을 직접 넘겨 확인한다."""

    p = vpvr.profile(frame([10.0] * 150), as_of=149, lookback=150, num_bins=30)
    assert p is not None

    levels = (5.0, 12.0, 18.0)
    assert p.nearest_above(15.0, levels) == 18.0
    assert p.nearest_below(15.0, levels) == 12.0
    assert p.nearest_above(1e9, levels) is None
    assert p.nearest_below(0.0, levels) is None


def test_profile_빈_구간이_많아도_HVN이_퍼지지_않는다():
    """거래가 없던 빈까지 넣어 백분위를 내면 임계값이 0으로 내려가
    모든 빈이 HVN이 된다. 그러면 존 전체가 하나로 묶여 무의미해진다."""

    closes = [10.0] * 60 + [20.0] * 60
    df = frame(closes, highs=[c + 0.05 for c in closes], lows=[c - 0.05 for c in closes])

    p = vpvr.profile(df, as_of=119, lookback=120, num_bins=50)
    assert p is not None

    # 실제로 대부분의 빈이 비어 있는 상황이어야 테스트가 의미를 갖는다
    assert (p.volumes == 0).sum() > 40
    assert len(p.hvn) <= 2
    # 가격이 머문 적 없는 중간 지대가 HVN으로 잡히면 안 된다
    assert not any(11.0 < x < 19.0 for x in p.hvn)


def test_profile_밸류에어리어가_구간을_감싼다():
    df = frame([10.0 + (i % 5) * 0.1 for i in range(150)])
    p = vpvr.profile(df, as_of=149, lookback=150, num_bins=30)

    assert p is not None
    val, vah = p.value_area
    assert val <= p.poc <= vah


def test_profile_데이터_부족하면_None():
    df = frame([10.0] * 10)
    assert vpvr.profile(df, as_of=9, lookback=100, num_bins=50) is None


def test_profile_as_of_범위_초과는_예외():
    df = frame([10.0] * 100)
    with pytest.raises(IndicatorError):
        vpvr.profile(df, as_of=200)


def test_profile_거래대금을_쓴다():
    """가격이 다른 두 구간에 같은 계약 수가 있으면 높은 가격대가 POC.

    계약 수로 계산하면 두 구간이 동률이 되어 이 테스트가 깨진다.
    """

    closes = [10.0] * 60 + [20.0] * 60
    df = frame(
        closes,
        highs=[c + 0.05 for c in closes],
        lows=[c - 0.05 for c in closes],
        volumes=[100.0] * 120,
    )

    p = vpvr.profile(df, as_of=119, lookback=120, num_bins=50)
    assert p is not None
    assert p.poc == pytest.approx(20.0, abs=0.5)


# ------------------------------------------------------------------ 벡터화 회귀


def _wilder_reference(values, period):
    """최적화 전 구현. 봉마다 pandas에 대입하던 방식.

    저장 방식만 바꿨고 산술과 순서는 그대로이므로 결과가 완전히 같아야 한다.
    다르면 최적화가 계산을 바꾼 것이다.
    """

    out = pd.Series(np.nan, index=values.index, dtype="float64")
    arr = values.to_numpy(dtype="float64")
    valid = np.flatnonzero(~np.isnan(arr))
    if len(valid) < period:
        return out

    start = valid[0]
    seed_end = start + period
    if seed_end > len(arr):
        return out

    prev = float(np.mean(arr[start:seed_end]))
    out.iloc[seed_end - 1] = prev
    alpha = 1.0 / period
    for i in range(seed_end, len(arr)):
        x = arr[i]
        if not np.isnan(x):
            prev = prev + alpha * (x - prev)
        out.iloc[i] = prev
    return out


def _ema_reference(values, period):
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    arr = values.to_numpy(dtype="float64")
    valid = np.flatnonzero(~np.isnan(arr))
    if len(valid) < period:
        return out

    start = valid[0]
    seed_end = start + period
    if seed_end > len(arr):
        return out

    alpha = 2.0 / (period + 1)
    prev = float(np.mean(arr[start:seed_end]))
    out.iloc[seed_end - 1] = prev
    for i in range(seed_end, len(arr)):
        x = arr[i]
        if not np.isnan(x):
            prev = prev + alpha * (x - prev)
        out.iloc[i] = prev
    return out


@pytest.mark.parametrize("period", [3, 14, 26])
def test_wilder_벡터화가_계산을_바꾸지_않는다(period):
    rng = np.random.default_rng(3)
    s = pd.Series(100 + rng.normal(0, 1, 500).cumsum())

    assert np.array_equal(
        wilder(s, period).to_numpy(),
        _wilder_reference(s, period).to_numpy(),
        equal_nan=True,
    )


@pytest.mark.parametrize("period", [3, 14, 26])
def test_ema_벡터화가_계산을_바꾸지_않는다(period):
    rng = np.random.default_rng(4)
    s = pd.Series(100 + rng.normal(0, 1, 500).cumsum())

    assert np.array_equal(
        ema_series(s, period).to_numpy(),
        _ema_reference(s, period).to_numpy(),
        equal_nan=True,
    )


def test_선행_NaN이_있어도_동일():
    """MACD 시그널선처럼 앞이 비어 있는 입력."""

    rng = np.random.default_rng(5)
    s = pd.Series([np.nan] * 30 + list(100 + rng.normal(0, 1, 200).cumsum()))

    assert np.array_equal(
        ema_series(s, 9).to_numpy(), _ema_reference(s, 9).to_numpy(), equal_nan=True
    )
