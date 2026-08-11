"""피벗·피보나치·합류 테스트.

가장 중요한 건 확정 지연이다. 이걸 빠뜨리면 과거 검증이 실제보다 좋게
나오는데, 결과가 "좋아지기만" 해서 눈에 띄지 않는다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from indicators import confluence, divergence, fib
from indicators import pivots as pv
from indicators.base import IndicatorError


T0 = 1_700_000_000
HOUR = 3600


def bars(closes, highs=None, lows=None):
    n = len(closes)
    ts = [T0 + i * HOUR for i in range(n)]
    opens = [closes[0]] + list(closes[:-1])
    return pd.DataFrame(
        {
            "ts": ts,
            "dt": pd.to_datetime(ts, unit="s", utc=True),
            "open": [float(x) for x in opens],
            "high": [float(x) for x in (highs if highs is not None else [max(o, c) for o, c in zip(opens, closes)])],
            "low": [float(x) for x in (lows if lows is not None else [min(o, c) for o, c in zip(opens, closes)])],
            "close": [float(x) for x in closes],
            "volume": [100.0] * n,
            "quote_volume": [100.0 * float(c) for c in closes],
        }
    )


def vshape(depth=10.0, width=11, base=100.0):
    """가운데가 최저인 V자. 저점 피벗 하나가 확실히 잡힌다."""

    half = width // 2
    return [base - depth * (1 - abs(i - half) / half) for i in range(width)]


def zigzag(points, leg=10):
    """꼭짓점 목록을 직선으로 이어 붙인다.

    평탄 구간을 만들지 않는 게 요점이다. 값이 같은 봉이 이어지면 롤링
    최대/최소가 여러 봉에서 동시에 성립해 퇴화 피벗이 무더기로 생긴다.
    """

    out = [float(points[0])]
    for a, b in zip(points, points[1:]):
        for k in range(1, leg + 1):
            out.append(float(a) + (float(b) - float(a)) * k / leg)
    return out


# ------------------------------------------------------------------ 피벗


def test_저점_피벗을_찾는다():
    df = bars(vshape())
    pivots = pv.find_pivots(df, n=3, as_of=len(df) - 1)

    lows = [p for p in pivots if p.kind == "low"]
    assert lows
    assert lows[0].index == 5              # V의 바닥


def test_확정_지연():
    """i번 봉이 피벗인지는 i+n까지 봐야 안다."""

    df = bars(vshape())
    n = 3

    assert pv.find_pivots(df, n, as_of=5) == []          # 바닥 당일엔 모른다
    assert pv.find_pivots(df, n, as_of=7) == []          # 아직
    assert pv.find_pivots(df, n, as_of=8)                # i+n = 8 부터


def test_confirmed_at은_index_더하기_n():
    df = bars(vshape())
    p = pv.find_pivots(df, n=3, as_of=len(df) - 1)[0]

    assert p.confirmed_at == p.index + 3


def test_as_of가_범위를_벗어나면_예외():
    df = bars(vshape())
    with pytest.raises(IndicatorError):
        pv.find_pivots(df, 3, as_of=999)


def test_usable_pivots가_확정_지연을_강제():
    df = bars(vshape())
    every = pv.all_pivots(df, 3)

    assert every                                          # 전체에는 있다
    assert pv.usable_pivots(every, as_of=4) == []          # 아직 확정 안 됨


def test_사전계산과_직접호출이_같은_결과():
    """리플레이는 사전 계산 경로를 쓴다. 두 경로가 갈리면 안 된다."""

    df = bars(vshape(width=41))
    as_of = len(df) - 1

    direct = pv.find_pivots(df, 5, as_of)
    cached = pv.usable_pivots(pv.all_pivots(df, 5), as_of)

    assert [p.index for p in direct] == [p.index for p in cached]


# ------------------------------------------------------------------ 스윙


def test_스윙은_인접_피벗에_갇히지_않는다():
    """되돌림을 낀 추세가 잘게 쪼개지면 안 된다 (§10-3 눈검증에서 발견).

    올랐다-조금내렸다-더올랐다 구조에서, 전체 상승이 한 다리로 잡혀야 한다.
    """

    closes = (
        [100.0] * 8
        + [100 + i for i in range(1, 21)]      # 100 -> 120
        + [120 - i for i in range(1, 6)]       # 120 -> 115 (되돌림)
        + [115 + i for i in range(1, 26)]      # 115 -> 140
        + [140.0] * 8
    )
    df = bars(closes)
    swings = pv.build_swings(pv.all_pivots(df, 3), min_swing_pct=0.05)

    ups = [s for s in swings if s.direction == "up"]
    assert ups
    biggest = max(ups, key=lambda s: s.size_pct)
    # 100 -> 140 전체가 한 다리로 잡혀야 한다 (인접 짝짓기면 최대 25/115=22%)
    assert biggest.size_pct > 0.35


def test_시작점을_되돌파하면_다리가_끊긴다():
    closes = (
        [100.0] * 8
        + [100 + i for i in range(1, 16)]      # 100 -> 115
        + [115 - i for i in range(1, 26)]      # 115 -> 90 (시작점 이탈)
        + [90 + i for i in range(1, 16)]       # 90 -> 105
        + [105.0] * 8
    )
    df = bars(closes)
    swings = pv.build_swings(pv.all_pivots(df, 3), min_swing_pct=0.03)

    # 100에서 시작한 상승 다리가 105까지 이어지면 안 된다 (중간에 90을 찍음)
    for s in swings:
        if s.direction == "up" and abs(s.low - 100.0) < 1.0:
            assert s.high <= 116.0


def test_최소_스윙_크기_필터():
    """1% 스윙은 5% 문턱을 못 넘고 0.5% 문턱은 넘는다."""

    df = bars(zigzag([100, 99, 100, 99, 100]))      # 진폭 1%

    assert pv.build_swings(pv.all_pivots(df, 3), min_swing_pct=0.05) == []
    assert pv.build_swings(pv.all_pivots(df, 3), min_swing_pct=0.005)


def test_잘못된_min_swing_pct는_예외():
    with pytest.raises(IndicatorError):
        pv.build_swings([], min_swing_pct=1.5)


def test_swings_at은_확정과_룩백을_거른다():
    df = bars(vshape(width=41))
    every = pv.all_swings(df, n=3, min_swing_pct=0.01)

    assert pv.swings_at(every, as_of=5) == []                       # 아직 미확정
    assert pv.swings_at(every, as_of=len(df) - 1, max_lookback=2) == []   # 너무 오래됨


# ------------------------------------------------------------------ 피보나치


def swing_up(low=100.0, high=200.0):
    a = pv.Pivot(0, T0, low, "low", 5)
    b = pv.Pivot(50, T0 + 50 * HOUR, high, "high", 55)
    return pv.Swing(a, b, "up")


def swing_down(high=200.0, low=100.0):
    a = pv.Pivot(0, T0, high, "high", 5)
    b = pv.Pivot(50, T0 + 50 * HOUR, low, "low", 55)
    return pv.Swing(a, b, "down")


def test_상승_되돌림은_고점에서_내려온다():
    s = swing_up(100, 200)

    assert fib.retracement(s, 0.0) == pytest.approx(200)
    assert fib.retracement(s, 0.5) == pytest.approx(150)
    assert fib.retracement(s, 1.0) == pytest.approx(100)


def test_하락_되돌림은_저점에서_올라간다():
    s = swing_down(200, 100)

    assert fib.retracement(s, 0.5) == pytest.approx(150)
    assert fib.retracement(s, 0.0) == pytest.approx(100)


def test_확장은_임펄스_방향으로():
    assert fib.extension(swing_up(100, 200), 1.272) == pytest.approx(227.2)
    assert fib.extension(swing_down(200, 100), 1.272) == pytest.approx(72.8)


def test_확장_비율은_1보다_커야():
    with pytest.raises(IndicatorError):
        fib.extension(swing_up(), 0.5)


def test_핵심_레벨은_4개():
    """레벨을 늘릴수록 아무 가격에나 걸려 변별력이 떨어진다 (§3.1)."""

    assert len(fib.core_levels(swing_up())) == 4


def test_골든포켓_표시():
    golden = [x for x in fib.core_levels(swing_up()) if x.is_golden]
    assert {x.ratio for x in golden} == {0.618, 0.65}


def test_100퍼센트_이탈이면_무효():
    s = swing_up(100, 200)

    assert not fib.invalidated(s, 105.0)
    assert fib.invalidated(s, 99.0)


def test_진입_방향():
    assert fib.entry_side(swing_up()) == "long"
    assert fib.entry_side(swing_down()) == "short"


def test_다음_하위_레벨이_손절_기준():
    s = swing_up(100, 200)
    below = fib.next_level_below(s, fib.retracement(s, 0.5), "long")

    assert below < 150
    # 다음 레벨이 없으면 100% 지점
    assert fib.next_level_below(s, 100.0, "long") == pytest.approx(100.0)


# ------------------------------------------------------------------ 합류


def test_합류_존은_점수순():
    df = bars(vshape(depth=20, width=201, base=100.0))
    zones = confluence.find_zones(df, len(df) - 1, atr=1.0)

    scores = [z.score for z in zones]
    assert scores == sorted(scores, reverse=True)


def test_ATR이_0이면_빈_목록():
    """허용 오차를 정할 수 없으면 존을 만들 수 없다."""

    df = bars(vshape(width=101))
    assert confluence.find_zones(df, len(df) - 1, atr=0.0) == []


def test_존_범위_판정은_고저로():
    z = confluence.Zone(price=100.0, low=99.0, high=101.0, side="long")

    assert z.touched_by(bar_low=98.0, bar_high=99.5)      # 아래에서 스침
    assert not z.touched_by(bar_low=102.0, bar_high=103.0)


def test_사전계산_스윙을_쓰면_결과가_같다():
    df = bars(vshape(depth=20, width=201, base=100.0))
    as_of = len(df) - 1

    direct = confluence.find_zones(df, as_of, atr=1.0)
    cached = confluence.find_zones(
        df, as_of, atr=1.0, precomputed_swings=pv.all_swings(df)
    )

    assert [round(z.price, 6) for z in direct] == [round(z.price, 6) for z in cached]


# ------------------------------------------------------------------ 다이버전스


def test_강세_다이버전스():
    """가격 저점은 낮아졌는데 RSI 저점은 높아진 구간.

    첫 하락을 급하게, 두 번째를 완만하게 만들면 두 번째 저점이 더 낮아도
    RSI는 덜 눌린다.
    """

    closes = (
        zigzag([100, 100], leg=20)[:20]
        + zigzag([100, 70], leg=6)[1:]        # 급락
        + zigzag([70, 92], leg=14)[1:]        # 반등
        + zigzag([92, 68], leg=40)[1:]        # 완만한 재하락, 더 낮은 저점
        + zigzag([68, 85], leg=14)[1:]
    )
    df = bars(closes)
    found = divergence.find_divergences(df, len(df) - 1, n=3, max_gap=120)

    assert any(d.kind == "bullish" for d in found)


def test_다이버전스도_확정_지연을_따른다():
    df = bars(vshape(width=41))
    early = divergence.find_divergences(df, as_of=3, n=5)

    assert early == []


def test_이중_바닥():
    """두 저점이 0.3% 차이로 붙어 있다."""

    df = bars(zigzag([100, 90, 100, 90.3, 100], leg=8))
    found = divergence.find_double_bottoms(
        df, len(df) - 1, n=3, price_tolerance=0.01, max_gap=60
    )

    assert any(d.kind == "bottom" for d in found)


def test_너무_가까운_저점은_이중바닥이_아니다():
    """바로 이웃한 피벗은 같은 저점의 노이즈다."""

    df = bars(zigzag([100, 90, 95, 90.2, 100], leg=2))
    found = divergence.find_double_bottoms(df, len(df) - 1, n=1, min_gap=8)

    assert [d for d in found if d.kind == "bottom"] == []


def test_이중바닥_허용오차_검증():
    with pytest.raises(IndicatorError):
        divergence.find_double_bottoms(bars(vshape()), 5, price_tolerance=2.0)
