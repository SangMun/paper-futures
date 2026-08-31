"""고전 차트 패턴 5종 테스트.

가장 중요한 건 **완성 확인**이다. 모양이 보인다는 것과 완성됐다는 건 다르고,
목선을 종가로 넘기 전에 진입하는 게 이 패턴들의 대표적 실수다.
꼬리로만 넘은 경우를 걸러내는지도 함께 본다.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from indicators import volume as vol_mod
from strategies.xrp_patterns import chart_base as cb
from strategies.xrp_patterns.cup_handle import CupHandle
from strategies.xrp_patterns.double_top import DoubleTop
from strategies.xrp_patterns.head_shoulders import HeadShoulders
from strategies.xrp_patterns.triangle import Triangle
from strategies.xrp_patterns.wedge import Wedge


T0 = 1_700_000_000
MIN5 = 300
HOUR = 3600

PATTERNS = [HeadShoulders, Wedge, Triangle, CupHandle, DoubleTop]


def bars(closes, ts=None, step=MIN5, volume=None, wick=0.0005):
    n = len(closes)
    ts = ts if ts is not None else [T0 + i * step for i in range(n)]
    opens = [closes[0]] + list(closes[:-1])
    return pd.DataFrame(
        {
            "ts": ts,
            "dt": pd.to_datetime(ts, unit="s", utc=True),
            "open": [float(o) for o in opens],
            "high": [max(o, c) * (1 + wick) for o, c in zip(opens, closes)],
            "low": [min(o, c) * (1 - wick) for o, c in zip(opens, closes)],
            "close": [float(c) for c in closes],
            "volume": [float(v) for v in (volume if volume is not None else [100.0] * n)],
            "quote_volume": [100.0 * float(c) for c in closes],
        }
    )


def leg(a, b, n):
    """a에서 b까지 n봉 직선. 평탄 구간을 만들지 않는다."""

    return [a + (b - a) * (k + 1) / n for k in range(n)]


def frames_of(closes, volume=None, context="flat"):
    """5분봉 + 1시간봉 프레임.

    context는 상위봉 방향이다. 숏 패턴을 시험하려면 상위봉이 상승이면 안 된다 —
    `xb.blocked`가 방향 불일치로 막는 게 정상 동작이기 때문이다.
    """

    df5 = bars(closes, volume=volume)
    n1h = len(closes) // 12 + 80
    h_ts = [T0 - 70 * HOUR + i * HOUR for i in range(n1h)]

    step = {"bull": 0.0008, "bear": -0.0008, "flat": 0.0}[context]
    rng = np.random.default_rng(7)
    hc = [1.0 + i * step + rng.normal(0, 0.0001) for i in range(n1h)]

    return {"5m": df5, "1h": bars(hc, ts=h_ts, step=HOUR)}


# ------------------------------------------------------------------ 공통 판정


def test_종가_돌파만_인정한다():
    """꼬리로 잠깐 넘는 건 이탈이 아니다."""

    df = bars([1.00, 1.00, 0.98])
    assert cb.close_break(df, 2, level=0.99, side="short")

    # 종가가 레벨 위에 남으면 이탈이 아니다
    df2 = bars([1.00, 1.00, 0.995])
    assert not cb.close_break(df2, 2, level=0.99, side="short")


def test_이미_넘어간_자리는_새_신호가_아니다():
    """직전 봉도 레벨 밖이면 '방금 이탈'이 아니다. 같은 이탈을 여러 번 세면 안 된다."""

    df = bars([1.00, 0.98, 0.97])
    assert not cb.close_break(df, 2, level=0.99, side="short")


def test_리테스트_판정():
    """돌파 후 되돌아 왔지만 패턴 내부로 복귀하지 않아야 성공이다."""

    # 리테스트 봉의 고가가 실제로 레벨에 닿아야 한다
    df = bars([1.00, 0.98, 0.9895, 0.985], wick=0.002)
    assert cb.retest_ok(df, break_at=1, as_of=3, level=0.99, side="short")

    back_inside = bars([1.00, 0.98, 0.995, 0.996], wick=0.002)
    assert not cb.retest_ok(back_inside, break_at=1, as_of=3, level=0.99, side="short")


def test_RVOL은_자기_자신을_평균에_넣지_않는다():
    v = [100.0] * 20 + [300.0]
    df = bars([1.0] * 21, volume=v)

    assert vol_mod.rvol(df, 20).iloc[-1] == pytest.approx(3.0)


def test_거래량_감소_판정():
    rising = bars([1.0] * 20, volume=list(range(1, 21)))
    falling = bars([1.0] * 20, volume=list(range(20, 0, -1)))

    assert not vol_mod.declining_between(rising, 0, 19)
    assert vol_mod.declining_between(falling, 0, 19)


def test_패턴_높이_목표는_참고치로만_계산된다():
    assert cb.pattern_target(100.0, 10.0, "short") == pytest.approx(90.0)
    assert cb.pattern_target(100.0, 10.0, "long") == pytest.approx(110.0)


def test_피벗_색인은_확정된_것만_돌려준다():
    from indicators.pivots import Pivot

    view = cb.PivotView([
        Pivot(10, T0, 1.0, "high", 15),
        Pivot(20, T0, 1.1, "low", 25),
    ])

    assert view.upto(as_of=14) == []
    assert len(view.upto(as_of=15)) == 1
    assert len(view.upto(as_of=25)) == 2


# ------------------------------------------------------------------ 계약


@pytest.mark.parametrize("cls", PATTERNS)
def test_태그는_패턴별로_분리된다(cls):
    """합쳐서 집계하면 어떤 패턴이 기여했는지 가려낼 수 없다."""

    name = cls().name
    assert name.startswith("xrp_patterns.")
    assert name != "xrp_patterns"


@pytest.mark.parametrize("cls", PATTERNS)
def test_평탄한_구간에서는_신호가_없다(cls):
    rng = np.random.default_rng(0)
    frames = frames_of(list(1.0 + rng.normal(0, 0.0002, 400)))

    s = cls()
    p = s.prepare(frames)
    assert [g for i in range(200, 400) for g in s.generate(p, i)] == []


@pytest.mark.parametrize("cls", PATTERNS)
def test_범위를_벗어난_as_of는_빈_목록(cls):
    frames = frames_of([1.0 + i * 0.0001 for i in range(400)])
    s = cls()
    p = s.prepare(frames)

    assert s.generate(p, 0) == []
    assert s.generate(p, 99_999) == []


@pytest.mark.parametrize("cls", PATTERNS)
def test_기본은_시장가_주문(cls):
    """완성 확인 후 진입이 기본이다.

    다만 돌파 봉 종가는 이미 움직임의 절반 이상이 끝난 자리라 추격이 된다.
    실측에서 레벨 지정가가 나은 패턴이 있어(wedge.5m: 시장가 -0.074% vs
    지정가 +0.091%, 앞뒤 절반 모두 플러스) 선택 가능하게 열어 두었다.
    """

    import inspect

    src = inspect.getsource(cls)
    assert 'order_type="market"' in src or 'order_type="limit" if' in src


def test_지정가_전환은_선택이다():
    """켠 전략만 지정가를 쓴다. 나머지는 기본값(시장가) 그대로."""

    from engine.loop import build_strategies

    by_name = {s.name: s for s in build_strategies()["XRP_USDT"]}
    assert by_name["xrp_patterns.wedge.5m"].limit_entry
    assert not by_name["xrp_patterns.wedge.15m"].limit_entry


def test_시간대_차단():
    """16~24시 UTC(한국 새벽)는 앞뒤 절반 모두 -0.57% / -0.42%로 최악이었다."""

    from engine.loop import build_strategies

    w = {s.name: s for s in build_strategies()["XRP_USDT"]}["xrp_patterns.wedge.5m"]
    assert w.block_hours == frozenset(range(16, 24))


@pytest.mark.parametrize("cls", PATTERNS)
def test_품질_문턱을_넘어야_신호가_된다(cls, monkeypatch):
    """모양이 맞아도 판정표를 못 채우면 진입하지 않는다."""

    monkeypatch.setattr(cb, "MIN_QUALITY", 99)
    rng = np.random.default_rng(1)
    closes = list(1.0 + rng.normal(0, 0.002, 500).cumsum())
    frames = frames_of(closes)

    s = cls()
    p = s.prepare(frames)
    assert [g for i in range(250, 500) for g in s.generate(p, i)] == []


# ------------------------------------------------------------------ 헤드앤숄더


def hs_closes():
    """왼쪽 어깨 → 머리 → 오른쪽 어깨 → 목선 이탈."""

    return (
        leg(0.95, 1.00, 40)      # 선행 상승
        + leg(1.00, 1.03, 12)    # 왼쪽 어깨
        + leg(1.03, 1.00, 10)    # 골 1 (목선)
        + leg(1.00, 1.06, 14)    # 머리
        + leg(1.06, 1.005, 12)   # 골 2
        + leg(1.005, 1.032, 12)  # 오른쪽 어깨
        + leg(1.032, 0.985, 14)  # 목선 이탈
    )


def test_헤드앤숄더_탐지():
    n = len(hs_closes())
    # 형성 중 거래량 감소 + 돌파 봉 급증
    volume = list(np.linspace(200, 60, n - 6)) + [60] * 3 + [400] * 3
    frames = frames_of(hs_closes(), volume=volume, context="bear")

    s = HeadShoulders()
    p = s.prepare(frames)
    sigs = [(i, g) for i in range(60, n) for g in s.generate(p, i)]

    assert sigs, "헤드앤숄더 신호가 없다"
    _, g = sigs[0]
    assert g.side == "short"
    assert g.stop_hint > g.entry              # 손절은 오른쪽 어깨 위
    assert "neckline" in g.context


def test_목선_이탈_전에는_신호가_없다():
    """머리 모양만 보고 진입하지 않는다."""

    closes = hs_closes()[:-14]                # 목선 이탈 구간을 잘라냄
    frames = frames_of(closes)

    s = HeadShoulders()
    p = s.prepare(frames)
    assert [g for i in range(60, len(closes)) for g in s.generate(p, i)] == []


# ------------------------------------------------------------------ 이중천장


def test_이중천장은_목선_돌파로_확인한다():
    closes = (
        leg(0.95, 1.00, 40)
        + leg(1.00, 1.05, 16)     # 첫 고점
        + leg(1.05, 1.005, 14)    # 목선
        + leg(1.005, 1.048, 16)   # 두 번째 고점
        + leg(1.048, 0.99, 14)    # 목선 이탈
    )
    n = len(closes)
    volume = list(np.linspace(200, 70, n - 4)) + [400] * 4
    frames = frames_of(closes, volume=volume, context="bear")

    s = DoubleTop()
    p = s.prepare(frames)
    sigs = [g for i in range(60, n) for g in s.generate(p, i)]

    assert sigs, "이중천장 신호가 없다"
    assert sigs[0].side == "short"
    assert sigs[0].stop_hint > sigs[0].entry


def test_두_번째_고점에서는_진입하지_않는다():
    """이전 구현의 결함. 두 번 닿았다는 사실만으로는 확률이 낮다."""

    closes = (
        leg(0.95, 1.00, 40)
        + leg(1.00, 1.05, 16)
        + leg(1.05, 1.005, 14)
        + leg(1.005, 1.048, 16)   # 두 번째 고점까지만
    )
    frames = frames_of(closes)

    s = DoubleTop()
    p = s.prepare(frames)
    assert [g for i in range(60, len(closes)) for g in s.generate(p, i)] == []


# ------------------------------------------------------------------ 삼각·쐐기


def test_삼각형은_방향을_미리_정하지_않는다():
    """상단 돌파면 롱, 하단 이탈이면 숏. 둘 다 가능해야 한다."""

    import inspect

    src = inspect.getsource(Triangle)
    assert 'side, level, opposite = "long"' in src
    assert 'side, level, opposite = "short"' in src


def test_쐐기는_내부_거래량이_늘면_제외():
    """거래량이 늘고 있으면 추세 강화지 쐐기가 아니다."""

    import inspect

    src = inspect.getsource(Wedge)
    assert "vol_trend" in src and "trend > 0" in src


def test_쐐기_기울기_판정():
    from indicators.pivots import Pivot

    # 직선의 기울기는 점 순서와 무관하다. 부호는 가격 방향이 정한다.
    up1 = Pivot(0, T0, 1.00, "high", 5)
    up2 = Pivot(20, T0, 1.02, "high", 25)
    assert cb.slope_of(up1, up2) > 0

    down2 = Pivot(20, T0, 0.98, "high", 25)
    assert cb.slope_of(up1, down2) < 0


def test_추세선_보간():
    from indicators.pivots import Pivot

    a = Pivot(0, T0, 1.0, "high", 5)
    b = Pivot(10, T0, 1.1, "high", 15)

    assert cb.line_at(a, b, 5) == pytest.approx(1.05)
    assert cb.line_at(a, b, 20) == pytest.approx(1.2)


# ------------------------------------------------------------------ 컵앤핸들


def test_V자는_컵이_아니다():
    """급락 후 급반등은 매물 소화가 없어 컵의 근거가 사라진다."""

    closes = (
        leg(0.95, 1.00, 40)
        + leg(1.00, 0.96, 6)      # 급락
        + leg(0.96, 1.00, 6)      # 급반등 = V자
        + leg(1.00, 0.995, 5)
        + leg(0.995, 1.02, 8)
    )
    frames = frames_of(closes)

    s = CupHandle()
    p = s.prepare(frames)
    assert [g for i in range(60, len(closes)) for g in s.generate(p, i)] == []


def test_핸들이_너무_깊으면_제외():
    """컵 상승분의 1/3 이상 되돌리면 핸들이 아니다."""

    from strategies.xrp_patterns.cup_handle import MAX_HANDLE_RETRACE

    assert MAX_HANDLE_RETRACE == pytest.approx(1 / 3)


def test_컵_최소_길이를_강제한다():
    """5분봉에서 작은 요철을 컵앤핸들로 해석하지 않는다."""

    from strategies.xrp_patterns.cup_handle import MIN_CUP_BARS

    assert MIN_CUP_BARS >= 30


# ------------------------------------------------------------------ OI


def test_OI가_없어도_동작한다():
    """수집 상한이 달라 커버리지가 어긋날 수 있다. 없으면 항목만 빠진다."""

    frames = frames_of([1.0 + i * 0.0001 for i in range(300)])
    series = cb.oi_change(frames, frames["5m"])

    assert len(series) == 300
    assert series.isna().all()


def test_OI_변화율이_봉에_정렬된다():
    closes = [1.0 + i * 0.0001 for i in range(300)]
    frames = frames_of(closes)
    ts = frames["5m"]["ts"]

    frames["OI"] = pd.DataFrame({
        "ts": ts,
        "open_interest": [1000.0 + i * 10 for i in range(300)],
    })
    series = cb.oi_change(frames, frames["5m"])

    assert series.index.equals(frames["5m"].index)
    assert series.iloc[-1] == pytest.approx(10.0 / (1000.0 + 298 * 10))


def test_OI_증가는_가점_감소는_아니다():
    """OI 감소는 청산성 스퀴즈다. 강해 보여도 소진되기 쉬워 가점하지 않는다."""

    import inspect

    src = inspect.getsource(cb.assess)
    assert "oi >= OI_EXPANSION" in src
    assert "oi_up_" in src


# ------------------------------------------------------------------ 고테가와


def test_고테가와는_이격도가_벌어져야_신호():
    from strategies.xrp_patterns.kotegawa import MIN_DEVIATION, Kotegawa

    # 이평선에 붙어 움직이면 신호가 없다
    frames = frames_of([1.0 + i * 0.00002 for i in range(600)])
    s = Kotegawa()
    p = s.prepare(frames)

    assert [g for i in range(450, 600) for g in s.generate(p, i)] == []
    assert MIN_DEVIATION > 0


def test_고테가와_목표는_이평선_회귀():
    """다른 패턴과 달리 VPVR이 아니라 MA를 목표로 쓴다 — 평균 회귀가 논리다."""

    import inspect
    from strategies.xrp_patterns.kotegawa import Kotegawa

    src = inspect.getsource(Kotegawa)
    assert "target_hint=target" in src
    assert "vpvr_target" not in src


def test_고테가와_문턱은_고정비율이_아니다():
    """-20%는 일본 주식 일봉 수치다. 5분봉 XRP에는 발생하지 않는다."""

    import inspect
    from strategies.xrp_patterns import kotegawa

    src = inspect.getsource(kotegawa)
    assert "rank" in src and "DEVIATION_PERCENTILE" in src


def test_캔들이_아닌_프레임은_지표_계산에서_빠진다():
    """OI 프레임에는 OHLC가 없다. 넣으면 표준 지표 계산이 예외로 죽는다."""

    from strategies.base import is_candle_frame, prepare_standard

    frames = frames_of([1.0 + i * 0.0001 for i in range(300)])
    frames["OI"] = pd.DataFrame({
        "ts": frames["5m"]["ts"],
        "open_interest": [1000.0] * 300,
    })

    assert not is_candle_frame(frames["OI"])
    p = prepare_standard(frames)

    assert "OI" not in p.series          # 지표는 계산하지 않고
    assert "OI" in p.frames              # 원본은 남는다


# ------------------------------------------------------------------ 요구사항 감사
# 아래 4개는 처음 구현에서 빠져 있던 요구사항의 회귀 테스트다.


def test_VWAP_확인이_판정표에_있다():
    """'돌파 거래량, VWAP, BTC 방향, OI 변화를 함께 확인' 중 VWAP."""

    import inspect

    src = inspect.getsource(cb.assess)
    assert "vwap" in src and "vwap_aligned" in src


def test_15분봉_종가_확인이_판정표에_있다():
    """'꼬리로 잠깐 넘는 것보다 5분·15분봉 종가 돌파를 우선'."""

    import inspect

    assert cb.CONFIRM_TF == "15m"
    src = inspect.getsource(cb.assess)
    assert "confirm_break_long" in src and "confirm_break_short" in src


def test_상위봉_지지저항_확인이_판정표에_있다():
    """'1시간·4시간봉의 의미 있는 지지·저항에서 발생'."""

    import inspect

    src = inspect.getsource(cb.assess)
    assert "htf_level" in src


def test_수렴패턴은_리스크를_줄여_잡는다():
    """'삼각형·쐐기·오른쪽 어깨 내부에서는 과도한 레버리지 금지'.

    레버리지 캡으로는 강제되지 않는다 — 필요 배수 최대값이 약 1.75x라
    캡(2~5x)에 닿지 않기 때문이다. 포지션 크기를 줄이는 쪽으로 강제한다.
    """

    from risk import sizing as sz

    tri = sz.size_position("XRP_USDT", "long", 1.0, 0.994, 1_000.0, "xrp_patterns.triangle")
    cup = sz.size_position("XRP_USDT", "long", 1.0, 0.994, 1_000.0, "xrp_patterns.cup_handle")

    assert tri.ok and cup.ok
    assert tri.notional < cup.notional * 0.6, "수렴 패턴 명목이 절반 수준이어야 한다"
    assert tri.risk_amount < cup.risk_amount
    assert tri.leverage <= cup.leverage


def test_세부_태그_캡이_전략군보다_우선한다():
    from config.settings import LEVERAGE_CAP

    family = LEVERAGE_CAP["xrp_patterns"]
    for tag in ("xrp_patterns.triangle", "xrp_patterns.wedge", "xrp_patterns.head_shoulders"):
        assert tag in LEVERAGE_CAP, f"{tag} 캡이 없다"
        assert LEVERAGE_CAP[tag] < family


def test_패턴이_쓰는_봉을_전부_받는다():
    """패턴마다 잘 맞는 봉이 다르다. 하나라도 빠지면 그 패턴이 죽는다."""

    from config.symbols import XRP
    from engine.loop import build_strategies

    needed = {s.tf.entry for s in build_strategies()["XRP_USDT"]}
    needed |= {s.tf.context for s in build_strategies()["XRP_USDT"]}
    assert needed <= XRP.required_timeframes, (
        f"빠진 봉: {needed - XRP.required_timeframes}"
    )


def test_OI가_낡으면_0이_아니라_NaN이_된다():
    """무제한 ffill이면 변화율이 0이 되어 '변화 없음'으로 조용히 읽힌다.

    실제로는 '값을 모른다'가 맞다. 한도를 넘으면 NaN으로 두어
    판정 항목이 건너뛰어지게 한다.
    """

    ts = [1_700_000_000 + i * 300 for i in range(300)]
    entry = pd.DataFrame({
        "ts": ts, "open": [1.0] * 300, "high": [1.0] * 300,
        "low": [1.0] * 300, "close": [1.0] * 300, "volume": [100.0] * 300,
    })
    # OI는 250봉까지만 존재
    oi = pd.DataFrame({"ts": ts[:250], "open_interest": [1000.0 + i * 5 for i in range(250)]})

    s = cb.oi_change({"5m": entry, "OI": oi}, entry)

    assert pd.notna(s.iloc[240]), "OI가 있는 구간은 값이 나와야 한다"
    assert pd.notna(s.iloc[251]), f"한도({cb.OI_STALE_LIMIT}봉) 안은 채워진다"
    assert pd.isna(s.iloc[260]), "한도를 넘으면 NaN — 0이면 '변화 없음'으로 오독된다"
    assert pd.isna(s.iloc[299])


def test_엔진에_OI_증분_갱신이_있다():
    """캐시만 읽으면 기동 시점 이후 OI 판정이 조용히 죽는다."""

    from engine.loop import Engine

    assert hasattr(Engine, "refresh_oi")

    import inspect

    src = inspect.getsource(Engine.refresh_oi)
    assert "fetch_open_interest" in src
    assert "save_oi" in src
    assert "except Exception" in src, "실패가 루프를 막으면 안 된다"


# ------------------------------------------------------------------ 대세 필터


def test_대세_역행_판정():
    """횡보는 역행으로 보지 않는다 — 실측에서 횡보는 양방향 다 플러스였다."""

    from strategies.xrp_patterns import base as xb

    class FakePrepared:
        def __init__(self, reg): self.reg = reg
        def at(self, *a): return self.reg

    tf = xb.LADDER["5m"]
    assert xb.counter_trend(FakePrepared("bull"), 0, "short", tf)
    assert xb.counter_trend(FakePrepared("bear"), 0, "long", tf)
    assert not xb.counter_trend(FakePrepared("bull"), 0, "long", tf)
    assert not xb.counter_trend(FakePrepared("none"), 0, "short", tf)


def test_trend_align이_켜진_전략만_막힌다():
    """이익 나는 전략은 건드리지 않는다."""

    from engine.loop import build_strategies

    by_name = {s.name: s for s in build_strategies()["XRP_USDT"]}
    assert by_name["xrp_patterns.wedge.5m"].trend_align
    assert by_name["xrp_patterns.head_shoulders.5m"].trend_align
    assert not by_name["xrp_patterns.wedge.15m"].trend_align


def test_대세_국면은_4시간봉을_본다():
    """한 칸 위 봉만으로는 큰 흐름을 못 본다. 1시간봉이 반등 중이어도
    4시간봉은 하락일 수 있다."""

    from strategies.xrp_patterns import base as xb

    assert xb.MAJOR_TF == "4h"
    frames = frames_of([1.0 + i * 0.0001 for i in range(300)])
    out = xb.major_regime(frames, frames["5m"].index, xb.LADDER["5m"])
    assert len(out) == 300          # 4h 프레임이 없으면 전부 none


def test_리테스트는_돌파_이후_봉으로만_판정한다():
    """`break_at`에 `as_of - 1`을 넘기면 검사 구간이 돌파 봉 한 줄이 되고,
    `close < level`이 돌파의 정의와 같아져 항상 참이 된다. 실거래 15건이
    전부 이 항목을 켰던 원인이다. 리테스트는 돌파 **이후** 봉에서만 성립한다.
    """

    level = 100.0
    # 목선을 종가로 하향 이탈한 뒤 되돌림이 전혀 없는 흐름.
    df = pd.DataFrame({
        "high":  [102, 101, 98.5, 98.0],
        "low":   [99, 97, 96, 95],
        "close": [101, 98, 97, 96],
    })

    assert cb.close_break(df, 1, level, "short")
    # 돌파 봉(index 1) 이후 구간에는 되돌림이 없으므로 리테스트 아님.
    assert not cb.retest_ok(df, 1, 3, level, "short")

    # 되돌아 올라와 목선을 건드렸지만 종가로 회복하지 못하면 리테스트다.
    retested = df.copy()
    retested.loc[2, ["high", "close"]] = [100.5, 99.0]
    assert cb.retest_ok(retested, 1, 3, level, "short")


def test_어떤_패턴도_retest_ok를_부르지_않는다():
    """진입이 돌파 봉 종가에서 일어나므로 그 시점에 리테스트는 존재할 수
    없다. 되돌림 진입을 만들려면 돌파를 기억해 두는 별도 경로가 필요하다."""

    import inspect
    from strategies.xrp_patterns import (
        cup_handle, double_top, head_shoulders, triangle, wedge,
    )

    for module in (cup_handle, double_top, head_shoulders, triangle, wedge):
        assert "retest_ok" not in inspect.getsource(module), module.__name__


def test_매물벽_판정은_ATR_기준이라_항상_켜지지_않는다():
    """기준 거리를 종가-레벨 간격으로 두면 돌파 봉에서 거의 0이 되어 어떤
    벽이든 통과한다. ATR 배수로 재야 '한 봉이 갈 거리'라는 의도가 남는다."""

    assert cb.WALL_CLEARANCE_ATR > 0
    source = inspect.getsource(cb.assess)
    assert "WALL_CLEARANCE_ATR" in source
    # 매물대를 못 찾은 경우(None)를 가점으로 세면 모르는 걸 합격으로 읽는다.
    assert "nearest_hvn is not None" in source
