"""실행 시계(run_tick) 통합 테스트.

봉 마감을 기다리지 않고 청산이 나는지, 그리고 그 때문에 봉 시계가
같은 봉을 중복 판정하지 않는지를 본다. 네트워크는 타지 않는다.
"""

from __future__ import annotations

import pytest

from core import candles as cd
from core.exchange import ExchangeError, FetchResult
from engine import state as st
from engine.loop import Engine
from reporting import logs
from reporting.notify import Notifier


XRP = "XRP_USDT"
STEP = 300
T0 = 1_786_000_000          # 5분 경계에 맞는 값


def raw(ts, o, h, l, c, v=1000):
    return {"t": ts, "o": str(o), "h": str(h), "l": str(l), "c": str(c),
            "v": v, "sum": str(v * c)}


def history(n=40, price=1.0, start=T0):
    """ATR이 계산될 만큼의 확정봉. 폭을 일정하게 둬 ATR을 예측 가능하게 한다."""

    return [
        raw(start + i * STEP, price, price + 0.01, price - 0.01, price)
        for i in range(n)
    ]


class TickClient:
    """진행 중인 봉을 붙여 돌려주는 가짜 거래소.

    live_bar를 바꿔 가며 run_tick을 여러 번 부르면 폴링을 흉내낼 수 있다.
    """

    def __init__(self, closed=None):
        self.closed = closed if closed is not None else history()
        self.live_bar = None
        self.calls = 0
        self.fail = False

    def set_live(self, o, h, l, c, ts=None):
        bar_ts = ts if ts is not None else int(self.closed[-1]["t"]) + STEP
        self.live_bar = raw(bar_ts, o, h, l, c)

    def roll_live_into_closed(self):
        """진행 중이던 봉을 마감 처리한다."""

        self.closed = self.closed + [self.live_bar]
        self.live_bar = None

    def fetch(self, contract, interval, limit=500, drop_unclosed=True, now_ts=None):
        self.calls += 1
        if self.fail:
            raise ExchangeError("연결 실패")
        rows = list(self.closed)
        if self.live_bar is not None and not drop_unclosed:
            rows = rows + [self.live_bar]
        df = cd.to_dataframe(rows)
        return FetchResult(contract, interval, df, [])

    def close(self):
        pass


def make_engine(tmp_path, client, position=None, balance=1_000.0):
    state = st.BotState(balance=balance, day_start_balance=balance)
    if position is not None:
        state.positions.append(position)
    return Engine(
        strategies={},
        client=client,
        trade_log=logs.TradeLog(tmp_path / "t.csv"),
        signal_log=logs.SignalLog(tmp_path / "s.jsonl"),
        notifier=Notifier(url=""),
        bot_state=state,
    )


def long_position(stop=0.98, tp1=1.05, **kw):
    base = dict(
        symbol=XRP, side="long", strategy="test", entry=1.0, contracts=100,
        leverage=1, stop=stop, tp1=tp1, partial_ratio=0.5, trail_atr_mult=1.5,
        max_hold_bars=40, risk_amount=10.0, entry_tf="5m",
    )
    base.update(kw)
    return st.OpenPosition(**base)


def live_now(client, offset=60):
    """진행 중인 봉이 시작된 뒤 offset초 지난 시각."""

    return int(client.live_bar["t"]) + offset


# ------------------------------------------------------------------ 핵심


def test_봉_마감_전에_손절이_나간다(tmp_path):
    """이게 이번 작업의 목적이다.

    5분봉이면 예전에는 최대 5분을 기다렸다. 이제 폴링 간격 안에 나간다.
    """

    client = TickClient()
    e = make_engine(tmp_path, client, long_position(stop=0.98))
    client.set_live(1.0, 1.0, 0.975, 0.99)      # 저가가 손절을 뚫음

    closed = e.run_tick(now_ts=live_now(client))

    assert closed == 1
    assert e.bot_state.positions == []
    rows = logs.read_trades(tmp_path / "t.csv")
    assert rows[-1]["reason"] == "stop"


def test_손절에_안_닿으면_유지된다(tmp_path):
    client = TickClient()
    e = make_engine(tmp_path, client, long_position(stop=0.98))
    client.set_live(1.0, 1.01, 0.99, 1.005)

    assert e.run_tick(now_ts=live_now(client)) == 0
    assert len(e.bot_state.positions) == 1


def test_봉_마감_전에_1차_익절이_나간다(tmp_path):
    client = TickClient()
    e = make_engine(tmp_path, client, long_position(tp1=1.05))
    client.set_live(1.0, 1.06, 0.995, 1.055)

    e.run_tick(now_ts=live_now(client))

    p = e.bot_state.positions[0]
    assert p.tp1_filled                     # 절반만 나가고 포지션은 남는다
    assert p.remaining_ratio == pytest.approx(0.5)


# ------------------------------------------------------------------ 거짓 청산 방지


def test_이미_지나간_저점으로_청산되지_않는다(tmp_path):
    """봉 전반의 저점 + 봉 후반에 올라간 트레일 = 없던 청산.

    core/live.py의 window_since가 막는다. 엔진까지 이어지는지 확인한다.
    """

    client = TickClient()
    position = long_position(stop=0.98, tp1=1.05)
    position.tp1_filled = True              # 트레일 작동 조건
    e = make_engine(tmp_path, client, position)

    # 1) 0.99까지 눌렸다가 1.04까지 회복.
    #    트레일이 저점 0.99 **위로** 올라가야 결함이 드러난다.
    client.set_live(1.0, 1.05, 0.99, 1.04)
    assert e.run_tick(now_ts=live_now(client, 60)) == 0
    stop_after_first = e.bot_state.positions[0].stop
    assert stop_after_first > 0.99, "트레일이 지나간 저점 위로 올라가야 검증이 성립한다"

    # 2) 같은 봉에서 계속 상승. 누적 저가는 여전히 0.99다.
    #    누적값을 그대로 넘기면 여기서 0.99 <= 손절이 되어 잘못 청산된다.
    client.set_live(1.0, 1.10, 0.99, 1.09)
    closed = e.run_tick(now_ts=live_now(client, 70))

    assert closed == 0, "지나간 저점으로 청산되면 안 된다"
    assert len(e.bot_state.positions) == 1
    assert e.bot_state.positions[0].stop > stop_after_first     # 트레일은 올라감


def test_트레일이_틱마다_따라_올라간다(tmp_path):
    client = TickClient()
    position = long_position(stop=0.98)
    position.tp1_filled = True
    e = make_engine(tmp_path, client, position)

    stops = []
    for high, last in [(1.05, 1.045), (1.08, 1.075), (1.12, 1.115)]:
        client.set_live(1.0, high, 0.995, last)
        e.run_tick(now_ts=live_now(client))
        stops.append(e.bot_state.positions[0].stop)

    assert stops[0] < stops[1] < stops[2]


# ------------------------------------------------------------------ 봉 넘어감


def test_봉이_넘어가면_꼬리_구간을_처리한다(tmp_path):
    """폴링 간격 때문에 봉의 마지막 몇 초는 어떤 틱도 보지 못한다.

    그 구간에서 손절이 닿았으면 다음 틱이 잡아야 한다.
    """

    client = TickClient()
    e = make_engine(tmp_path, client, long_position(stop=0.98))

    # 봉 중간까지는 멀쩡하다
    client.set_live(1.0, 1.01, 0.995, 1.005)
    assert e.run_tick(now_ts=live_now(client, 60)) == 0

    # 아무도 못 본 마지막 몇 초에 0.97까지 빠지고 봉이 마감됐다
    client.live_bar = raw(int(client.live_bar["t"]), 1.0, 1.01, 0.97, 0.985)
    client.roll_live_into_closed()
    client.set_live(0.985, 0.99, 0.984, 0.988)

    closed = e.run_tick(now_ts=live_now(client, 30))

    assert closed == 1, "봉 꼬리에서 닿은 손절을 놓쳤다"


def test_틱이_다_본_봉은_봉_시계가_중복_판정하지_않는다(tmp_path):
    """covered_bar_ts가 올라가면 _advance는 시간 손절만 본다."""

    client = TickClient()
    position = long_position(stop=0.98)
    position.tp1_filled = True
    e = make_engine(tmp_path, client, position)

    client.set_live(1.0, 1.10, 0.99, 1.09)
    e.run_tick(now_ts=live_now(client, 60))

    bar_ts = int(client.live_bar["t"])
    client.roll_live_into_closed()
    client.set_live(1.09, 1.095, 1.085, 1.09)
    e.run_tick(now_ts=live_now(client, 30))

    p = e.bot_state.positions[0]
    assert p.covered_bar_ts == bar_ts

    # 그 봉을 봉 시계에 넣어도 청산되지 않는다.
    # 누적 저가 0.99와 올라간 트레일을 비교하면 잘못 청산된다.
    import pandas as pd

    bar = pd.Series({"ts": bar_ts, "open": 1.0, "high": 1.10, "low": 0.99, "close": 1.09})
    handled = e._advance(p, bar, atr=0.01)

    assert not handled
    assert len(e.bot_state.positions) == 1


def test_틱에서는_시간_손절이_나지_않는다(tmp_path):
    client = TickClient()
    position = long_position(stop=0.98, max_hold_bars=10)
    position.bars_held = 50
    e = make_engine(tmp_path, client, position)
    client.set_live(1.0, 1.01, 0.99, 1.005)

    assert e.run_tick(now_ts=live_now(client)) == 0
    assert len(e.bot_state.positions) == 1


# ------------------------------------------------------------------ 방어


def test_포지션이_없으면_거래소를_부르지_않는다(tmp_path):
    client = TickClient()
    e = make_engine(tmp_path, client)

    assert e.run_tick(now_ts=T0) == 0
    assert client.calls == 0


def test_수집이_실패해도_죽지_않는다(tmp_path):
    client = TickClient()
    e = make_engine(tmp_path, client, long_position())
    client.set_live(1.0, 1.01, 0.99, 1.0)
    client.fail = True

    assert e.run_tick(now_ts=live_now(client)) == 0
    assert len(e.bot_state.positions) == 1


def test_진행_중인_봉이_없으면_건너뛴다(tmp_path):
    """봉이 막 넘어간 순간. 그 구간은 봉 시계가 처리한다."""

    client = TickClient()
    e = make_engine(tmp_path, client, long_position())

    assert e.run_tick(now_ts=int(client.closed[-1]["t"]) + STEP + 1) == 0
    assert len(e.bot_state.positions) == 1


def test_시세_피드가_밀리면_판정하지_않는다(tmp_path):
    """멈춘 값으로 손절을 판정하면 실제와 다른 가격에서 청산된다."""

    client = TickClient()
    e = make_engine(tmp_path, client, long_position(stop=0.98))
    client.set_live(1.0, 1.0, 0.90, 0.95)       # 손절을 한참 뚫은 값

    # 확정봉이 한참 전에 멈춰 있다
    closed = e.run_tick(now_ts=int(client.live_bar["t"]) + STEP * 20)

    assert closed == 0
    assert len(e.bot_state.positions) == 1


def test_포지션이_있는_타임프레임만_받는다(tmp_path):
    """틱에서는 지표를 쓰지 않으므로 상위봉·문맥봉을 받을 이유가 없다."""

    client = TickClient()
    e = make_engine(tmp_path, client, long_position(entry_tf="5m"))
    client.set_live(1.0, 1.01, 0.99, 1.0)

    e.run_tick(now_ts=live_now(client))

    assert client.calls == 1
