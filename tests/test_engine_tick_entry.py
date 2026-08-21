"""실행 시계의 진입 처리.

지정가 체결이 봉 마감을 기다리지 않는지, 그리고 메이커 전환 설정이
켜졌을 때만 동작하는지를 본다.
"""

from __future__ import annotations

import pytest

from config import settings
from engine import loop as loop_mod
from engine import state as st
from strategies.base import Signal
from tests.test_engine_tick import (
    STEP, TickClient, XRP, live_now, make_engine, raw,
)


class FakeStrategy:
    """진입봉만 알려주면 되는 최소 전략."""

    def __init__(self, name="test", tf="5m"):
        self.name = name
        self.tf = type("TF", (), {"entry": tf})()

    def generate(self, prepared, as_of):
        return []


def pending_item(spec, engine, price=0.98, side="long", placed_ts=0, strategy="test"):
    """대기 지정가 주문 한 건을 직접 밀어 넣는다.

    _consider 전 경로를 타면 리스크·사이징까지 얽혀 이 테스트의 관심사인
    "언제 체결되는가"가 가려진다.
    """

    from risk import exits as ex
    from risk import sizing as sz

    # 손절·목표는 방향에 맞춰 놓는다. 숏인데 손절이 아래 있으면
    # 체결 즉시 손절되어 이 테스트의 관심사가 가려진다.
    stop = price * (0.99 if side == "long" else 1.01)
    tp1 = price * (1.02 if side == "long" else 0.98)
    signal = Signal(
        strategy=strategy, symbol=XRP, side=side, entry=price,
        stop_hint=stop, order_type="limit", ttl_bars=3,
    )
    plan = ex.ExitPlan(
        stop=stop, tp1=tp1, partial_ratio=0.5,
        trail_atr_mult=1.5, max_hold_bars=40,
    )
    position = sz.Position(
        contracts=10, notional=100.0, margin=100.0, leverage=1,
        stop_distance=0.01, risk_amount=1.0, price_risk_amount=1.0,
        estimated_cost=0.05, entry_fill_price=price,
        stop_fill_price=stop,
        liquidation_price=price * 0.5, liquidation_distance=0.5,
    )
    return {
        "spec": spec, "signal": signal, "position": position, "plan": plan,
        "tranche": 1, "placed_ts": placed_ts,
    }


def setup(tmp_path, price=0.98, side="long", placed_offset=-STEP):
    client = TickClient()
    e = make_engine(tmp_path, client)
    spec = next(s for s in e.symbols if s.contract == XRP)
    e.strategies = {XRP: [FakeStrategy()]}

    live_ts = int(client.closed[-1]["t"]) + STEP
    e.pending_signals.append(
        pending_item(spec, e, price=price, side=side, placed_ts=live_ts + placed_offset)
    )
    return client, e


# ------------------------------------------------------------------ 체결


def test_봉_마감_전에_지정가가_체결된다(tmp_path):
    client, e = setup(tmp_path, price=0.98)
    client.set_live(1.0, 1.0, 0.975, 0.99)      # 저가가 지정가를 지나감

    e.run_tick(now_ts=live_now(client))

    assert len(e.bot_state.positions) == 1
    assert e.pending_signals == []
    assert e.limit_stats["filled"] == 1


def test_가격이_안_닿으면_대기_유지(tmp_path):
    client, e = setup(tmp_path, price=0.95)
    client.set_live(1.0, 1.01, 0.99, 1.0)

    e.run_tick(now_ts=live_now(client))

    assert e.bot_state.positions == []
    assert len(e.pending_signals) == 1


def test_숏_지정가는_위로_올라와야_체결(tmp_path):
    client, e = setup(tmp_path, price=1.02, side="short")

    client.set_live(1.0, 1.01, 0.99, 1.0)
    assert e.run_tick(now_ts=live_now(client)) == 0
    assert len(e.pending_signals) == 1

    client.set_live(1.0, 1.03, 0.99, 1.02)
    e.run_tick(now_ts=live_now(client, 70))
    assert len(e.bot_state.positions) == 1


def test_주문을_낸_봉으로는_체결하지_않는다(tmp_path):
    """신호는 봉이 마감된 뒤 나온다. 그 봉 안의 가격에는 주문이 없었다."""

    client, e = setup(tmp_path, price=0.98, placed_offset=0)   # 진행 중인 봉에 주문
    client.set_live(1.0, 1.0, 0.975, 0.99)

    e.run_tick(now_ts=live_now(client))

    assert e.bot_state.positions == []
    assert len(e.pending_signals) == 1


def test_포지션이_없어도_대기_주문이_있으면_감시한다(tmp_path):
    client, e = setup(tmp_path, price=0.98)
    client.set_live(1.0, 1.0, 0.975, 0.99)

    assert e.bot_state.positions == []
    e.run_tick(now_ts=live_now(client))

    assert client.calls == 1
    assert len(e.bot_state.positions) == 1


def test_포지션도_주문도_없으면_거래소를_부르지_않는다(tmp_path):
    client = TickClient()
    e = make_engine(tmp_path, client)

    e.run_tick(now_ts=int(client.closed[-1]["t"]) + STEP + 30)

    assert client.calls == 0


def test_체결은_메이커_수수료(tmp_path):
    """지정가는 리베이트라 잔고가 늘어난다."""

    client, e = setup(tmp_path, price=0.98)
    before = e.bot_state.balance
    client.set_live(1.0, 1.0, 0.975, 0.99)

    e.run_tick(now_ts=live_now(client))

    assert e.bot_state.balance > before


# ------------------------------------------------------------------ 메이커 전환


def test_전환은_기본적으로_꺼져_있다():
    """켜면 일부 신호가 체결되지 않아 전략 성격이 바뀐다.

    기존 성과 기록과 비교가 끊기므로 기본값은 꺼짐이어야 한다.
    """

    assert settings.ENTRY.prefer_maker_entries is False


def maker_on(monkeypatch, ttl=2):
    cfg = settings.EntryConfig(prefer_maker_entries=True, maker_entry_ttl_bars=ttl)
    monkeypatch.setattr(settings, "ENTRY", cfg)
    monkeypatch.setattr(loop_mod, "ENTRY", cfg)


def market_signal(**kw):
    base = dict(
        strategy="test", symbol=XRP, side="long", entry=1.0,
        stop_hint=0.99, order_type="market",
    )
    base.update(kw)
    return Signal(**base)


def test_켜면_시장가가_지정가로_바뀐다(tmp_path, monkeypatch):
    maker_on(monkeypatch)
    e = make_engine(tmp_path, TickClient())

    out = e._as_maker_entry(market_signal())

    assert out.order_type == "limit"
    assert out.ttl_bars == 2
    assert out.entry == 1.0            # 가격은 그대로 — 쫓아가면 리베이트가 사라진다


def test_꺼져_있으면_시장가_그대로(tmp_path):
    e = make_engine(tmp_path, TickClient())

    out = e._as_maker_entry(market_signal())

    assert out.order_type == "market"
    assert out.ttl_bars == 0


def test_이미_지정가면_손대지_않는다(tmp_path, monkeypatch):
    """전략이 정한 TTL이 우선이다. BTC 회귀채널은 12봉짜리를 쓴다."""

    maker_on(monkeypatch, ttl=2)
    e = make_engine(tmp_path, TickClient())

    out = e._as_maker_entry(market_signal(order_type="limit", ttl_bars=12))

    assert out.order_type == "limit"
    assert out.ttl_bars == 12


# ------------------------------------------------------------------ 체결 후 상태


def test_체결봉은_보유_봉_수에_세지_않는다(tmp_path):
    """틱 체결은 벽시계 시각을 남긴다.

    그 값을 last_bar_ts에 그대로 넣으면 어떤 봉 ts와도 같지 않아
    체결봉까지 세어지고, 시간 손절이 한 봉 일찍 터진다.
    """

    client, e = setup(tmp_path, price=0.98)
    client.set_live(1.0, 1.0, 0.975, 0.99)

    e.run_tick(now_ts=live_now(client))

    position = e.bot_state.positions[0]
    assert position.last_bar_ts == int(client.live_bar["t"])


def test_체결_직후_관측_기준점이_지금으로_맞춰진다(tmp_path):
    """안 맞추면 다음 틱의 구간이 봉 시작까지 거슬러 올라간다.

    체결 전 가격 범위로 손절·트레일을 판정하게 된다.
    """

    client, e = setup(tmp_path, price=0.98)
    client.set_live(1.0, 1.0, 0.975, 0.99)

    e.run_tick(now_ts=live_now(client))

    p = e.bot_state.positions[0]
    assert p.ticked_bar_ts == int(client.live_bar["t"])
    assert p.last_tick_price == 0.99
    assert p.last_tick_low == 0.975


def test_체결_후_되돌아와도_체결_전_저점으로_손절되지_않는다(tmp_path):
    """봉 전반에 크게 빠졌다가 회복한 뒤 지정가가 체결된 경우."""

    client, e = setup(tmp_path, price=0.995)
    # 저가 0.975까지 빠졌다가 0.995로 회복 -> 지정가 체결
    client.set_live(1.0, 1.0, 0.975, 0.995)
    e.run_tick(now_ts=live_now(client, 60))

    assert len(e.bot_state.positions) == 1
    stop = e.bot_state.positions[0].stop
    assert stop < 0.99, "이 테스트는 손절이 봉 저가보다 위일 때만 의미가 있다"

    # 같은 봉에서 가격이 안정적으로 유지된다. 누적 저가는 여전히 0.975다.
    client.set_live(1.0, 1.0, 0.975, 0.996)
    e.run_tick(now_ts=live_now(client, 120))

    assert len(e.bot_state.positions) == 1


def test_체결_직후_손절이_닿으면_다음_틱이_잡는다(tmp_path):
    """체결봉 검사를 건너뛴 대신 누락이 생기지 않는지 확인한다."""

    client, e = setup(tmp_path, price=0.995)
    client.set_live(1.0, 1.0, 0.975, 0.995)
    e.run_tick(now_ts=live_now(client, 60))
    assert len(e.bot_state.positions) == 1

    # 체결 뒤 실제로 손절까지 밀린다
    client.set_live(1.0, 1.0, 0.97, 0.978)
    e.run_tick(now_ts=live_now(client, 120))

    assert e.bot_state.positions == []
