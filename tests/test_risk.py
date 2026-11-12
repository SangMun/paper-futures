"""리스크 레이어 단위 테스트.

손으로 검산 가능한 값만 쓴다. 이 레이어가 틀리면 손실이 계획보다 커지는데,
지표와 달리 차트로 눈에 띄지 않는다.
"""

from __future__ import annotations

import pytest

from config.settings import FEES, RISK
from risk import limits as lim
from risk import sizing as sz
from risk.exits import (
    EXIT_STOP,
    EXIT_TIME,
    EXIT_TP1,
    EXIT_TRAIL,
    ExitError,
    PositionState,
    apply_tp1,
    build_plan,
    check_exit,
    process_bar,
    update_trailing,
)


BTC = "BTC_USDT"
XRP = "XRP_USDT"


# ------------------------------------------------------------------ 크기 산정


def test_손실액이_잔고의_1퍼센트():
    """어떤 손절 거리든 1회 손실은 계좌의 1%로 고정돼야 한다."""

    for stop_pct in (0.005, 0.01, 0.025, 0.04):
        p = sz.size_position(
            XRP, "long", 1.0, 1.0 - stop_pct, balance=1_000.0, strategy="xrp_patterns"
        )
        assert p.ok, (stop_pct, p.rejected, p.notes)
        # 계약 단위 반올림으로 예산보다 작아질 수는 있어도 넘지는 않는다
        assert p.risk_amount <= 10.0 + 1e-9
        assert p.risk_amount == pytest.approx(10.0, rel=0.05)


def test_손절이_좁을수록_포지션이_커진다():
    a = sz.size_position(XRP, "long", 1.0, 0.99, 1_000.0, "xrp_patterns")
    b = sz.size_position(XRP, "long", 1.0, 0.98, 1_000.0, "xrp_patterns")

    assert a.notional > b.notional
    # 고정 비율인 왕복 비용도 계약당 손실에 들어가므로 정확히 2배는 아니다.
    assert 1.8 < a.notional / b.notional < 1.9


def test_수수료와_슬리피지를_포함해도_리스크_예산을_넘지_않는다():
    p = sz.size_position(XRP, "long", 1.0, 0.995, 1_000.0, "xrp_patterns")

    assert p.ok
    assert p.risk_amount <= 10.0
    assert p.risk_amount == pytest.approx(p.price_risk_amount + p.estimated_cost)
    assert p.estimated_cost > 0
    assert p.entry_fill_price > 1.0
    assert p.stop_fill_price < 0.995


def test_슬리피지_스트레스에서는_포지션이_더_작다():
    base = sz.size_position(XRP, "long", 1.0, 0.995, 1_000.0, "xrp_patterns")
    stress = sz.size_position(
        XRP,
        "long",
        1.0,
        0.995,
        1_000.0,
        "xrp_patterns",
        execution=sz.ExecutionAssumption.stress(),
    )

    assert stress.contracts < base.contracts
    assert stress.risk_amount <= 10.0


def test_수수료보다_좁은_손절은_거절():
    """0.38% 하한. 이보다 좁으면 이겨도 수수료가 수익의 절반을 가져간다."""

    p = sz.size_position(XRP, "long", 1.0, 0.998, 1_000.0, "xrp_patterns")

    assert not p.ok
    assert p.rejected == sz.REJECT_STOP_TOO_TIGHT
    assert "왕복 비용" in p.notes[0]


def test_하한은_수수료에서_유도된다():
    """수수료가 내려가면 하한도 같이 내려가야 한다."""

    cheap = FEES.__class__(taker_rate=0.0002, slippage_rate=0.0001)
    stop = 0.998                                     # 0.2%

    assert not sz.size_position(XRP, "long", 1.0, stop, 1_000.0, "xrp_patterns").ok
    assert sz.size_position(
        XRP, "long", 1.0, stop, 1_000.0, "xrp_patterns", fees=cheap
    ).ok


def test_너무_넓은_손절은_거절():
    p = sz.size_position(BTC, "long", 60_000.0, 55_000.0, 1_000.0, "btc_trend")

    assert not p.ok
    assert p.rejected == sz.REJECT_STOP_TOO_WIDE


def test_레버리지는_역산_결과다():
    """비용 포함 후 명목은 줄지만 잔고 1,000에서 1x면 충분하다."""

    p = sz.size_position(BTC, "long", 60_000.0, 58_500.0, 1_000.0, "btc_trend")

    assert p.ok
    assert p.notional == pytest.approx(372.0, rel=0.02)
    assert p.leverage == 1


def test_XRP_단타는_2배_이상_필요():
    """손절 0.5%에 왕복 비용을 포함해도 1x를 넘어 2x가 필요하다."""

    p = sz.size_position(XRP, "long", 1.0, 0.995, 1_000.0, "xrp_patterns")

    assert p.ok
    assert p.notional == pytest.approx(1_450.0, rel=0.02)
    assert p.leverage == 2


def test_필요_레버리지는_잔고와_무관하다():
    """필요 레버리지 = risk_per_trade / 손절거리. 잔고가 약분되어 사라진다.

    명목 = 잔고 * 리스크 / 손절,  필요배수 = 명목 / 잔고 = 리스크 / 손절.
    계좌가 커도 작아도 같은 배수가 나온다.
    """

    small = sz.size_position(XRP, "long", 1.0, 0.995, 200.0, "xrp_patterns")
    large = sz.size_position(XRP, "long", 1.0, 0.995, 50_000.0, "xrp_patterns")

    assert small.leverage == large.leverage == 2       # 0.01 / 0.005


def test_전략별_레버리지_캡():
    """캡을 넘으면 거절한다.

    기본 설정에서는 이 경로에 도달하지 않는다. 필요 배수의 최대값이
    risk_per_trade / min_stop_pct = 0.01 / 0.0038 = 2.63x라 캡(5x) 아래다.
    캡은 리스크 비율을 올리거나 손절 하한을 낮췄을 때를 대비한 안전장치다.
    """

    aggressive = RISK.__class__(risk_per_trade=0.05)
    p = sz.size_position(
        XRP, "long", 1.0, 0.995, 1_000.0, "xrp_patterns", risk=aggressive
    )

    assert not p.ok                                    # 0.05 / 0.005 = 10x > 5x
    assert p.rejected == sz.REJECT_LEVERAGE_INFEASIBLE
    assert "전략캡" in p.notes[0]


def test_청산가는_손절보다_3배_이상_멀다():
    for stop_pct in (0.005, 0.01, 0.02):
        p = sz.size_position(
            XRP, "long", 1.0, 1 - stop_pct, 5_000.0, "xrp_patterns"
        )
        assert p.ok
        assert p.liquidation_multiple() >= RISK.min_liquidation_multiple


def test_청산가에_유지증거금률이_반영된다():
    """1/L만 쓰면 청산 거리를 실제보다 멀게 본다."""

    naive = 1.0 * (1 - 1 / 10)                  # 0.9
    actual = sz.liquidation_price(1.0, "long", 10, maintenance_rate=0.005)

    assert actual > naive                        # 더 가깝다
    assert actual == pytest.approx(0.905)


def test_숏_청산가는_위쪽():
    assert sz.liquidation_price(1.0, "short", 10, 0.005) == pytest.approx(1.095)


def test_격리청산가에는_실제배정증거금과_절사된계약수량을_쓴다():
    # 100x를 요청했더라도 76계약(760 XRP)으로 절사되면 실제 명목가치는
    # 1,000 USDT보다 작다. 레버리지 역수 근사 대신 MMR 등식으로 풀어야 한다.
    assert sz.isolated_liquidation_price(
        1.3045, "long", margin=10.0, contracts=76, multiplier=10.0,
        maintenance_rate=0.005,
    ) == pytest.approx((1.3045 - 10.0 / 760.0) / 0.995)
    assert sz.isolated_liquidation_price(
        1.3039, "short", margin=10.0, contracts=76, multiplier=10.0,
        maintenance_rate=0.005,
    ) == pytest.approx((1.3039 + 10.0 / 760.0) / 1.005)


def test_격리청산가는_Gate_화면의_100x_예상청산가와_맞는다():
    # 2026-10-24 Gate 화면: 100x 격리 숏, 진입 1.2831, 예상 청산가 1.2894.
    # 증거금 = 명목가 1%. 청산 수수료를 넣으면 1.2885가 되어 화면과 다르다.
    liq = sz.isolated_liquidation_price(
        1.2831, "short", margin=1.2831 * 1000 / 100, contracts=100, multiplier=10.0,
        maintenance_rate=0.005,
    )
    assert 1.2894 <= liq < 1.2895


def test_최소_주문_단위_미달은_거절():
    """BTC 1계약 = 6 USDT. 잔고가 작고 손절이 넓으면 1계약도 못 산다."""

    p = sz.size_position(BTC, "long", 60_000.0, 57_600.0, balance=10.0, strategy="btc_trend")

    assert not p.ok
    assert p.rejected == sz.REJECT_SIZE_BELOW_MIN


def test_계약수는_내림():
    """올림하면 리스크가 예산을 넘는다."""

    p = sz.size_position(BTC, "long", 60_000.0, 58_500.0, 1_000.0, "btc_trend")

    assert p.contracts == int(p.contracts)
    assert p.notional <= 400.0 + 1e-9


def test_분할진입은_총량을_먼저_쪼갠다():
    """3분할 합계가 1회분과 같아야 한다. 아니면 3배 리스크가 된다."""

    parts = [
        sz.size_position(
            BTC, "long", 60_000.0, 58_500.0, 1_000.0, "btc_trend",
            tranche=t, max_tranches=3,
        )
        for t in (1, 2, 3)
    ]
    single = sz.size_position(BTC, "long", 60_000.0, 58_500.0, 1_000.0, "btc_trend")

    assert all(p.ok for p in parts)
    combined = sum(p.risk_amount for p in parts)
    loss_per_contract = single.risk_amount / single.contracts
    assert combined <= single.risk_amount
    assert single.risk_amount - combined < loss_per_contract * 3
    assert parts[0].notional > parts[1].notional      # 0.4 > 0.3


def test_트랜치_가중치_정규화():
    """2분할을 원하면 앞 두 개를 잘라 쓰고 합이 1이 되게 맞춘다."""

    a = sz.tranche_risk(0.01, 1, 2)
    b = sz.tranche_risk(0.01, 2, 2)

    assert a + b == pytest.approx(0.01)
    assert a > b


def test_잘못된_입력은_예외():
    with pytest.raises(sz.SizingError):
        sz.size_position(XRP, "long", 1.0, 1.1, 1_000.0, "xrp_patterns")   # 롱인데 손절이 위
    with pytest.raises(sz.SizingError):
        sz.size_position("ETH_USDT", "long", 1.0, 0.99, 1_000.0, "x")      # 스펙 없음


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_사이징은_NaN과_무한대를_차단한다(bad):
    with pytest.raises(sz.SizingError):
        sz.size_position(XRP, "long", bad, 0.99, 1_000.0, "xrp_patterns")
    with pytest.raises(sz.SizingError):
        sz.size_position(XRP, "long", 1.0, 0.99, bad, "xrp_patterns")


def test_정수로_실현할_수_없는_분수_레버리지는_거절한다():
    """허용 상한 2.5x인데 3x가 필요하면 2x로 낮춰 통과시키면 안 된다."""

    strict_liquidation = RISK.__class__(min_liquidation_multiple=9.875)
    p = sz.size_position(
        XRP,
        "long",
        1.0,
        0.96,
        1_000.0,
        "xrp_patterns",
        available_margin=104.17,
        risk=strict_liquidation,
    )

    assert not p.ok
    assert p.rejected == sz.REJECT_LEVERAGE_INFEASIBLE


def test_세부_전략_태그는_전략군_레버리지_캡을_쓴다():
    p = sz.size_position(XRP, "long", 1.0, 0.995, 1_000.0, "xrp_patterns.flag")

    assert p.ok
    assert p.leverage <= 5


def test_BTC_고정25배는_잔고10퍼를_증거금으로_쓴다():
    p = sz.size_position(
        BTC,
        "long",
        60_000.0,
        58_800.0,
        1_000.0,
        "btc_regime_25x",
        fixed_leverage=25,
        margin_fraction=0.10,
        min_liquidation_multiple=1.25,
    )

    assert p.ok, p.notes
    assert p.leverage == 25
    assert p.margin <= 100.0
    # 계약 단위 내림 때문에 정확히 2,500 USDT보다 조금 작을 수 있다.
    assert p.notional == pytest.approx(2_500.0, rel=0.005)
    assert p.risk_amount == pytest.approx(54.75, rel=0.02)
    assert p.liquidation_multiple() >= 1.25


def test_BTC_25배는_명시적_청산거리_완화_없이는_거절된다():
    p = sz.size_position(
        BTC,
        "long",
        60_000.0,
        58_800.0,
        1_000.0,
        "btc_regime_25x",
        fixed_leverage=25,
        margin_fraction=0.10,
    )

    assert not p.ok
    assert p.rejected == sz.REJECT_LEVERAGE_INFEASIBLE


def test_BTC_25배_1퍼_손절의_계좌예상손실은_약3퍼센트():
    p = sz.size_position(
        BTC,
        "long",
        60_000.0,
        59_400.0,
        1_000.0,
        "btc_regime_25x",
        fixed_leverage=25,
        margin_fraction=0.10,
        min_liquidation_multiple=3.0,
    )

    assert p.ok, p.notes
    # BTC 가격 -1%와 왕복 비용 약 0.19%가 명목 2.5배에 적용된다.
    assert p.risk_amount == pytest.approx(29.75, rel=0.03)


# ------------------------------------------------------------------ 청산 계획


def test_손절에_ATR_버퍼가_붙는다():
    plan = build_plan("long", 100.0, 98.0, atr=1.0)

    assert plan.stop == pytest.approx(97.5)      # 98 - 0.5 * 1


def test_단타는_R_배수로_목표를_잡는다():
    plan = build_plan("long", 100.0, 99.0, atr=1.0, scalp=True)

    stop_dist = 100.0 - plan.stop                # 99 - 0.3 = 98.7 -> 1.3
    assert plan.tp1 == pytest.approx(100.0 + stop_dist * 1.5)
    assert plan.reward_risk(100.0) == pytest.approx(1.5)


def test_BTC는_구조적_목표를_쓴다():
    plan = build_plan("long", 100.0, 98.0, atr=1.0, target_hint=110.0)

    assert plan.tp1 == 110.0


def test_목표가_반대편이면_예외():
    with pytest.raises(ExitError):
        build_plan("long", 100.0, 98.0, atr=1.0, target_hint=95.0)


def test_손절_힌트가_진입가_반대편이면_예외():
    """전략이 잘못된 stop_hint를 냈을 때 조용히 넘어가지 않는다.

    ATR 버퍼는 손절을 항상 진입가에서 **멀어지는** 쪽으로 민다.
    따라서 이 경로는 stop_hint 자체가 틀렸을 때만 걸린다.
    """

    with pytest.raises(ExitError):
        build_plan("long", 100.0, 101.0, atr=1.0)
    with pytest.raises(ExitError):
        build_plan("short", 100.0, 99.0, atr=1.0)


def test_ATR이_커도_손절은_멀어지기만_한다():
    near = build_plan("long", 100.0, 98.0, atr=1.0).stop
    far = build_plan("long", 100.0, 98.0, atr=10.0).stop

    assert far < near < 100.0


def test_고정손절_분할익절_트레일링_비용본전_덮어쓰기():
    plan = build_plan(
        "long",
        100.0,
        98.0,
        atr=1.0,
        target_hint=104.0,
        stop_buffer_atr_mult=0.0,
        partial_exit_ratio=0.5,
        trail_atr_mult=2.5,
        max_hold_bars=720,
        breakeven_offset_pct=0.0019,
    )

    assert plan.stop == pytest.approx(98.0)
    assert plan.tp1 == pytest.approx(104.0)
    assert plan.trail_atr_mult == pytest.approx(2.5)
    assert plan.max_hold_bars == 720
    state = PositionState("long", 100.0, plan)
    apply_tp1(state, check_exit(state, 100.0, 105.0, 99.0, 104.5, 0.0))
    assert state.stop == pytest.approx(100.19)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_청산계획은_NaN과_무한대를_차단한다(bad):
    with pytest.raises(ExitError):
        build_plan("long", 100.0, 98.0, atr=bad)
    with pytest.raises(ExitError):
        build_plan("long", 100.0, bad, atr=1.0)


# ------------------------------------------------------------------ 청산 판정


def state(side="long", entry=100.0, stop_hint=98.0, atr=1.0, **kw):
    return PositionState(side=side, entry=entry, plan=build_plan(side, entry, stop_hint, atr, **kw))


def bar_exit(s, high, low, open_=100.0, close=100.0, slippage_rate=0.0):
    return check_exit(s, open_, high, low, close, slippage_rate)


def test_같은_봉에서_손절이_익절보다_우선():
    """봉 안의 순서를 모르므로 불리한 쪽을 택한다. 아니면 결과가 부풀려진다."""

    s = state(target_hint=110.0)
    ev = bar_exit(s, high=115.0, low=90.0)

    assert ev.reason == EXIT_STOP


def test_1차_익절은_절반만():
    s = state(target_hint=110.0)
    ev = bar_exit(s, high=111.0, low=99.0)

    assert ev.reason == EXIT_TP1
    assert ev.ratio == pytest.approx(0.5)


def test_1차_익절_뒤_손절이_본전으로():
    s = state(target_hint=110.0)
    apply_tp1(s, bar_exit(s, high=111.0, low=99.0))

    assert s.stop == pytest.approx(100.0)
    assert s.remaining_ratio == pytest.approx(0.5)
    assert s.tp1_filled


def test_트레일링은_익절_전에는_작동하지_않는다():
    """진입 직후부터 따라붙이면 정상적인 되돌림에 털린다."""

    s = state(target_hint=110.0)
    before = s.stop
    update_trailing(s, high=108.0, low=100.0, atr=1.0)

    assert s.stop == before


def test_트레일링은_유리한_방향으로만():
    s = state(target_hint=110.0)
    apply_tp1(s, bar_exit(s, high=111.0, low=99.0))

    update_trailing(s, high=120.0, low=110.0, atr=1.0)
    raised = s.stop
    assert raised == pytest.approx(120.0 - 1.5)

    update_trailing(s, high=112.0, low=111.0, atr=1.0)   # 되돌림
    assert s.stop == pytest.approx(raised)               # 내려가지 않는다


def test_트레일링_손절은_trail로_기록된다():
    """원 손절과 트레일링 청산을 구분해야 성과 분석이 된다."""

    s = state(target_hint=110.0)
    apply_tp1(s, bar_exit(s, high=111.0, low=99.0))
    update_trailing(s, high=120.0, low=110.0, atr=1.0)

    ev = bar_exit(s, high=119.0, low=118.0, open_=119.0, close=118.5)
    assert ev.reason == EXIT_TRAIL


def test_시간_손절():
    s = state(target_hint=110.0)
    s.bars_held = s.plan.max_hold_bars
    ev = bar_exit(s, high=101.0, low=99.5, close=100.5)

    assert ev.reason == EXIT_TIME
    assert ev.ratio == pytest.approx(1.0)
    assert ev.price == pytest.approx(100.5)


def test_시간청산을_끄면_보유한도를_넘어도_종가청산하지_않는다():
    plan = build_plan(
        "long",
        100.0,
        99.0,
        atr=1.0,
        target_hint=102.0,
        max_hold_bars=12,
        time_exit_enabled=False,
    )
    s = PositionState("long", 100.0, plan)
    s.bars_held = 100

    assert check_exit(s, 100.0, 101.0, 99.5, 100.5) is None


def test_숏도_대칭으로_동작():
    s = state(side="short", entry=100.0, stop_hint=102.0, target_hint=90.0)

    assert s.plan.stop == pytest.approx(102.5)
    assert bar_exit(s, high=103.0, low=95.0).reason == EXIT_STOP
    assert bar_exit(s, high=101.0, low=89.0).reason == EXIT_TP1


def test_롱_갭_손절은_손절가가_아니라_시가에_체결된다():
    s = state()
    ev = bar_exit(s, open_=90.0, high=92.0, low=89.0, close=91.0)

    assert ev.reason == EXIT_STOP
    assert ev.price == pytest.approx(90.0)
    assert ev.note == "갭 손절"


def test_숏_갭_손절도_시가에_체결된다():
    s = state(side="short", stop_hint=102.0, target_hint=90.0)
    ev = bar_exit(s, open_=110.0, high=111.0, low=108.0, close=109.0)

    assert ev.reason == EXIT_STOP
    assert ev.price == pytest.approx(110.0)
    assert ev.note == "갭 손절"


def test_트레일링_갱신은_다음_봉부터_유효하다():
    s = state(target_hint=110.0)
    apply_tp1(s, bar_exit(s, high=111.0, low=99.0))

    # 기존 손절(100)은 닿지 않았지만 새 트레일(118.5)이라면 같은 봉 저가에 닿는다.
    ev = process_bar(
        s, open_=110.0, high=120.0, low=110.0, close=119.0, atr=1.0,
        slippage_rate=0.0,
    )
    assert ev is None
    assert s.stop == pytest.approx(118.5)

    next_ev = process_bar(
        s, open_=119.0, high=119.0, low=118.0, close=118.2, atr=1.0,
        slippage_rate=0.0,
    )
    assert next_ev.reason == EXIT_TRAIL
    assert next_ev.price == pytest.approx(118.5)


def test_청산_봉의_NaN은_차단한다():
    with pytest.raises(ExitError):
        bar_exit(state(), high=float("nan"), low=99.0)


def test_ATR이_NaN이어도_기존_손절은_먼저_실행한다():
    s = state()
    ev = process_bar(
        s,
        open_=100.0,
        high=101.0,
        low=97.0,
        close=98.0,
        atr=float("nan"),
        slippage_rate=0.0,
    )

    assert ev.reason == EXIT_STOP
    assert ev.price == pytest.approx(s.plan.stop)


# ------------------------------------------------------------------ 계좌 한도


def account(balance=1_000.0, **kw):
    return lim.AccountState(balance=balance, day_start_balance=balance, **kw)


def test_기본은_허용():
    assert lim.check_entry(account(), XRP, "long", 10.0).allowed


def test_일일_손실_한도():
    a = account(balance=940.0)
    a.day_start_balance = 1_000.0
    v = lim.check_entry(a, XRP, "long", 10.0)

    assert not v.allowed
    assert v.reason == lim.REJECT_DAILY_LOSS


def test_연속_손절_한도에_도달하면_정지된다():
    """카운터 자체가 아니라 정지 만료 시각이 진입을 막는다."""

    now = 1_700_000_000
    a = account(consecutive_losses={"A": 4})
    a.now_ts = now

    lim.record_result(a, -5.0, strategy="A")     # 5회째 손절

    assert a.consecutive_losses["A"] == 5
    assert a.paused_until["A"] > now, "만료 시각이 세워져야 한다"

    v = lim.check_entry(a, XRP, "long", 10.0, strategy="A")
    assert not v.allowed
    assert v.reason == lim.REJECT_CONSECUTIVE
    assert "잔여" in v.detail


def test_정지는_전략별이다():
    """한 전략의 연패가 다른 전략을 멈추면 전략별 성과 측정이 오염된다.

    실측으로 kotegawa를 빼자 double_top 손실이 -221에서 -270으로 늘었다 —
    정지가 우연히 브레이크 역할을 하고 있었던 것이다.
    """

    now = 1_700_000_000
    a = account(consecutive_losses={"A": 4})
    a.now_ts = now
    lim.record_result(a, -5.0, strategy="A")

    assert not lim.check_entry(a, XRP, "long", 10.0, strategy="A").allowed
    assert lim.check_entry(a, XRP, "long", 10.0, strategy="B").allowed, (
        "다른 전략은 영향을 받으면 안 된다"
    )


def test_측정_모드는_계좌_브레이크를_끈다():
    """전략별 맨얼굴을 보려면 계좌 단위 브레이크가 꺼져 있어야 한다."""

    from config.settings import MEASUREMENT_RISK

    a = account(balance=500.0)          # 당일 -50%
    a.day_start_balance = 1_000.0

    assert not lim.check_entry(a, XRP, "long", 10.0).allowed
    assert lim.check_entry(a, XRP, "long", 10.0, risk=MEASUREMENT_RISK).allowed


def test_정지는_영구가_아니다():
    """가장 위험했던 결함. 카운터는 이익 거래로만 초기화되는데 진입이
    막혀 있으면 이익 거래가 나올 수 없어, 만료가 없으면 봇이 영구히 멈춘다.

    실측으로 180일 리플레이가 5,000봉에서 멈추고 963건이 거절됐다.
    """

    from config.settings import RISK

    now = 1_700_000_000
    a = account(consecutive_losses={"A": 4})
    a.now_ts = now
    lim.record_result(a, -5.0, strategy="A")
    assert not lim.check_entry(a, XRP, "long", 10.0, strategy="A").allowed

    # 만료 시각을 지나면 풀린다
    a.now_ts = now + int(RISK.pause_hours * 3600) + 1
    assert lim.clear_expired_pause(a) == ["A"]

    assert a.paused_until["A"] == 0
    assert a.consecutive_losses["A"] == 0, "카운터도 함께 풀려야 한다 — 안 그러면 다음 손절 한 번에 재정지"
    assert lim.check_entry(a, XRP, "long", 10.0, strategy="A").allowed


def test_이익이_나면_정지도_풀린다():
    now = 1_700_000_000
    a = account(consecutive_losses={"A": 4})
    a.now_ts = now
    lim.record_result(a, -5.0, strategy="A")
    assert a.paused_until["A"] > 0

    lim.record_result(a, 10.0, strategy="A")
    assert a.consecutive_losses["A"] == 0
    assert a.paused_until["A"] == 0


def test_종목별_리스크_상한():
    """BTC 추세 롱과 BTC 피보 롱이 겹치면 사실상 하나의 베팅이다."""

    a = account()
    a.open_positions = [
        lim.OpenRisk(BTC, "long", "btc_trend", 20.0),
        lim.OpenRisk(BTC, "long", "btc_fib", 20.0),
    ]
    v = lim.check_entry(a, BTC, "long", 10.0)

    assert not v.allowed
    assert v.reason == lim.REJECT_SYMBOL_RISK


def test_상관이_검증되기_전에는_방향과_무관하게_전액_합산():

    a = account()
    a.open_positions = [lim.OpenRisk(BTC, "long", "btc_trend", 100.0)]

    same = lim.weighted_open_risk(a.open_positions, a.balance, side="long")
    opposite = lim.weighted_open_risk(a.open_positions, a.balance, side="short")

    assert same == pytest.approx(0.10)
    assert opposite == pytest.approx(0.10)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_유효하지_않은_신규_리스크는_거절한다(bad):
    verdict = lim.check_entry(account(), XRP, "long", bad)

    assert not verdict.allowed
    assert verdict.reason == lim.REJECT_INVALID_RISK


def test_전체_리스크_상한():
    a = account()
    a.open_positions = [
        lim.OpenRisk(f"S{i}_USDT", "long", "x", 20.0) for i in range(8)
    ]
    v = lim.check_entry(a, XRP, "long", 10.0)

    assert not v.allowed
    assert v.reason == lim.REJECT_TOTAL_RISK


def test_본전은_연패로_세지_않는다():
    """트레일링이 본전에서 끊는 일이 잦다. 그걸 손절로 세면 카운터가 빨리 찬다."""

    a = account(consecutive_losses={"A": 2})
    lim.record_result(a, 0.0, strategy="A")

    assert a.consecutive_losses["A"] == 2


def test_이익이_나면_연패_초기화():
    a = account(consecutive_losses={"A": 3})
    lim.record_result(a, 5.0, strategy="A")

    assert a.consecutive_losses["A"] == 0
    assert a.balance == pytest.approx(1_005.0)


def test_새_날은_손실_기준점만_옮긴다():
    a = account(balance=950.0, consecutive_losses={"A": 3})
    lim.start_new_day(a)

    assert a.day_start_balance == 950.0
    assert a.consecutive_losses["A"] == 3     # 연패는 날짜와 무관하게 이어진다


def test_열린_포지션_증거금은_가용에서_빠진다():
    """잔고에서는 수수료와 실현손익만 빠지므로, 열린 포지션이 묶은 증거금을
    따로 빼지 않으면 사이징이 매번 잔고 전액을 가용으로 본다. 실측으로
    2026-08-24에 XRP 3포지션이 잔고 1,096 USDT에 증거금 1,794 USDT를
    요구했다 — 실거래소면 세 번째 주문이 거부됐을 자리다."""

    from engine import state as st

    position = st.OpenPosition(
        symbol="XRP_USDT", side="long", strategy="xrp_patterns",
        entry=1.0, contracts=100, leverage=2,
        stop=0.99, tp1=1.02, partial_ratio=0.5, trail_atr_mult=1.5,
        max_hold_bars=100, risk_amount=10.0,
    )
    # 명목 = 100계약 * 1.0 * 배수 10 = 1,000, 레버리지 2배 -> 증거금 500
    assert st.used_margin([position]) == pytest.approx(500.0)
    assert st.used_margin([]) == 0.0

    # 가용이 줄면 같은 신호라도 더 작은 포지션이 나오거나 거절된다.
    full = sz.size_position(XRP, "long", 1.0, 0.99, 1_000.0, "xrp_patterns")
    squeezed = sz.size_position(
        XRP, "long", 1.0, 0.99, 1_000.0, "xrp_patterns", available_margin=50.0
    )
    assert full.ok
    assert not squeezed.ok or squeezed.notional <= full.notional


def test_엔진이_가용_증거금을_사이징에_넘긴다():
    """파라미터는 있는데 호출부가 안 넘겨서 오래 죽어 있던 경로다."""

    import inspect

    from engine import loop

    source = inspect.getsource(loop.Engine)
    assert "available_margin=" in source
    assert "used_margin" in source
