"""엔진 배선 테스트 — 진입부터 청산까지.

4단계의 핵심이다. risk/engine/reporting이 처음으로 맞물려 도는 지점이라
여기서 드러나지 않는 결함은 실시간 로그에서나 발견된다.

네트워크를 타지 않는다. 전략과 캔들을 직접 주입한다.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from config.symbols import BTC, StrategySpec, SymbolSpec
from engine import state as st
from engine.loop import Engine, _closed_index_at
from reporting import logs
from reporting.notify import Notifier
from strategies.base import PositionPolicy, Prepared, Signal


BTCU = "BTC_USDT"
T0 = 1_700_000_000
HOUR = 3600
FIVE_MIN = 300


def bars(rows, start_ts=T0, warmup=20, warmup_price=60_000.0):
    """rows: (open, high, low, close) 목록.

    앞에 warmup봉을 붙인다. ATR(14)이 계산되지 않으면 트레일링이 작동하지
    않아, 실제 운용(500봉)과 다른 경로를 검증하게 된다.
    """

    pad = [(warmup_price, warmup_price + 120, warmup_price - 120, warmup_price)] * warmup
    rows = pad + list(rows)
    n = len(rows)
    ts = [start_ts - (warmup - i) * HOUR for i in range(warmup)] + [
        start_ts + i * HOUR for i in range(n - warmup)
    ]
    return pd.DataFrame(
        {
            "ts": ts,
            "dt": pd.to_datetime(ts, unit="s", utc=True),
            "open": [float(r[0]) for r in rows],
            "high": [float(r[1]) for r in rows],
            "low": [float(r[2]) for r in rows],
            "close": [float(r[3]) for r in rows],
            "volume": [100.0] * n,
            "quote_volume": [100.0] * n,
        }
    )


class FixedSignal:
    """항상 같은 신호 하나를 내는 전략. 배선만 확인한다."""

    name = "btc_trend"
    tf = SimpleNamespace(entry="1h")

    def __init__(self, side="long", entry=60_000.0, stop=58_500.0, target=63_000.0):
        self.side, self.entry, self.stop, self.target = side, entry, stop, target
        self.emit = True

    def prepare(self, frames):
        return Prepared(frames=frames, series={tf: {} for tf in frames})

    def generate(self, prepared, as_of):
        if not self.emit:
            return []
        return [
            Signal(
                strategy=self.name,
                symbol=BTCU,
                side=self.side,
                entry=self.entry,
                stop_hint=self.stop,
                target_hint=self.target,
                score=3.0,
                reasons=["test"],
                context={"atr": 200.0},
                max_tranches=3,
            )
        ]


class LimitSignal(FixedSignal):
    """지정가 신호를 내는 전략."""

    def __init__(self, price, side="long", max_tranches=3):
        stop = price * 0.97 if side == "long" else price * 1.03
        target = price * 1.05 if side == "long" else price * 0.95
        super().__init__(side=side, entry=price, stop=stop, target=target)
        self.max_tranches = max_tranches

    def generate(self, prepared, as_of):
        sigs = super().generate(prepared, as_of)
        for g in sigs:
            g.order_type = "limit"
            g.ttl_bars = 3
            g.max_tranches = self.max_tranches
        return sigs


class Fixed25Signal(FixedSignal):
    """25배 국면 전략의 비표준 포지션 정책을 내는 가짜 전략."""

    name = "btc_regime_25x"
    # 기존 엔진 정책 테스트는 15분봉으로 격리한다. 실제 전략은 5분봉이다.
    tf = SimpleNamespace(entry="15m")

    def __init__(self):
        super().__init__(entry=60_000.0, stop=59_760.0, target=60_240.0)
        self.invalidate = False

    def generate(self, prepared, as_of):
        signals = super().generate(prepared, as_of)
        for signal in signals:
            signal.position_policy = PositionPolicy(
                fixed_leverage=25,
                margin_fraction=0.10,
                min_liquidation_multiple=1.25,
                stop_buffer_atr_mult=0.0,
                partial_exit_ratio=1.0,
                trail_atr_mult=1.2,
                max_hold_bars=32,
                breakeven_offset_pct=0.0019,
                min_target_r=1.0,
                stop_pct_from_entry=0.004,
                target_pct_from_entry=0.004,
                symbol_risk_cap=0.06,
                exclusive_symbol=True,
            )
            signal.max_tranches = 1
        return signals

    def exit_reason(self, prepared, as_of, side):
        return "regime" if self.invalidate else None


def engine(tmp_path, strategy=None, balance=1_000.0):
    e = Engine(
        strategies={BTCU: [strategy]} if strategy else {},
        symbols=(BTC,),
        client=object(),
        trade_log=logs.TradeLog(tmp_path / "t.csv"),
        signal_log=logs.SignalLog(tmp_path / "s.jsonl"),
        notifier=Notifier(url=""),
        bot_state=st.BotState(balance=balance, day_start_balance=balance),
    )
    return e


def frames_at(price=60_000.0):
    return {
        "5m": bars([(price, price + 100, price - 100, price)] * 30),
        "15m": bars([(price, price + 100, price - 100, price)] * 30),
        "1h": bars([(price, price + 100, price - 100, price)] * 30),
        "1d": bars([(price, price + 500, price - 500, price)] * 30),
    }


# ------------------------------------------------------------------ 진입


def test_신호가_포지션으로_이어진다(tmp_path):
    e = engine(tmp_path, FixedSignal())
    e.evaluate_entries(BTC, frames_at())

    assert len(e.bot_state.positions) == 1
    p = e.bot_state.positions[0]
    assert p.symbol == BTCU
    assert p.side == "long"
    assert p.contracts > 0
    assert p.stop < p.entry < p.tp1


def test_진입이_CSV와_JSONL에_남는다(tmp_path):
    e = engine(tmp_path, FixedSignal())
    e.evaluate_entries(BTC, frames_at())

    trades = logs.read_trades(tmp_path / "t.csv")
    assert len(trades) == 1
    assert trades[0]["action"] == "entry"
    assert trades[0]["strategy"] == "btc_trend"

    sig = logs.read_signals(tmp_path / "s.jsonl")[0]
    assert sig["action"] == "entry"
    assert sig["tranche"] == 1
    assert sig["reasons"] == ["test"]


def test_트랜치는_엔진이_매긴다(tmp_path):
    """전략은 상태가 없어 자기가 몇 번째인지 모른다.

    봉을 넘겨 가며 부른다 — 같은 봉으로는 한 번만 평가하므로
    분할 진입을 보려면 새 봉이 필요하다.
    """

    e = engine(tmp_path, FixedSignal())
    for k in range(3):
        e.evaluate_entries(BTC, frames_at(), bar_ts=T0 + k * HOUR)

    assert len(e.bot_state.positions) == 3
    tranches = [s["tranche"] for s in logs.read_signals(tmp_path / "s.jsonl")]
    assert tranches == [1, 2, 3]


def test_최대_트랜치를_넘으면_거절(tmp_path):
    e = engine(tmp_path, FixedSignal())
    for k in range(4):
        e.evaluate_entries(BTC, frames_at(), bar_ts=T0 + k * HOUR)

    assert len(e.bot_state.positions) == 3
    last = logs.read_signals(tmp_path / "s.jsonl")[-1]
    assert last["skip_reason"] == "max_tranches_reached"


def test_분할_리스크_합계가_1회분을_넘지_않는다(tmp_path):
    """트랜치마다 1%씩 쓰면 3분할이 3배 리스크가 된다."""

    e = engine(tmp_path, FixedSignal())
    for k in range(3):
        e.evaluate_entries(BTC, frames_at(), bar_ts=T0 + k * HOUR)

    total = sum(p.risk_amount for p in e.bot_state.positions)
    assert total <= 1_000.0 * 0.01 + 1e-6


def test_반대방향_포지션이_있으면_거절(tmp_path):
    e = engine(tmp_path, FixedSignal(side="long"))
    e.evaluate_entries(BTC, frames_at(), bar_ts=T0)

    e.strategies[BTCU] = [FixedSignal(side="short", entry=60_000.0, stop=61_500.0, target=57_000.0)]
    e.evaluate_entries(BTC, frames_at(), bar_ts=T0 + HOUR)

    assert len(e.bot_state.positions) == 1
    assert logs.read_signals(tmp_path / "s.jsonl")[-1]["skip_reason"] == "opposite_position_open"


def test_손절이_너무_좁으면_진입하지_않는다(tmp_path):
    e = engine(tmp_path, FixedSignal(entry=60_000.0, stop=59_950.0))
    e.evaluate_entries(BTC, frames_at())

    assert e.bot_state.positions == []
    assert logs.read_signals(tmp_path / "s.jsonl")[0]["skip_reason"] == "stop_too_tight"


def test_전략별_일일_체결한도에_도달하면_진입하지_않는다(tmp_path):
    class DailyLimited(FixedSignal):
        def generate(self, prepared, as_of):
            signals = super().generate(prepared, as_of)
            for signal in signals:
                signal.position_policy = PositionPolicy(max_daily_entries=1)
            return signals

    e = engine(tmp_path, DailyLimited())
    e.bot_state.daily_entries["btc_trend"] = 1

    e.evaluate_entries(BTC, frames_at())

    assert e.bot_state.positions == []
    assert logs.read_signals(tmp_path / "s.jsonl")[-1]["skip_reason"] == "daily_entry_limit"


def test_25배_정책이_엔진에서도_그대로_열린다(tmp_path):
    e = engine(tmp_path, Fixed25Signal())
    e.evaluate_entries(BTC, frames_at())

    assert len(e.bot_state.positions) == 1
    position = e.bot_state.positions[0]
    assert position.leverage == 25
    assert position.risk_amount == pytest.approx(14.75, rel=0.03)
    assert position.exclusive_symbol
    assert position.stop == pytest.approx(59_760.0)


def test_25배_전략은_다른_BTC포지션과_겹치지_않는다(tmp_path):
    e = engine(tmp_path, Fixed25Signal())
    e.evaluate_entries(BTC, frames_at(), bar_ts=T0)

    other = FixedSignal()
    e.strategies[BTCU] = [other]
    e.evaluate_entries(BTC, frames_at(), bar_ts=T0 + HOUR)

    assert len(e.bot_state.positions) == 1
    assert logs.read_signals(tmp_path / "s.jsonl")[-1]["skip_reason"] == "symbol_exclusive"


def test_25배_시장가의_고정퍼센트_손절목표는_다음시가에서_다시_계산한다(tmp_path):
    e = engine(tmp_path, Fixed25Signal())
    e.defer_market_entries = True
    e.evaluate_entries(BTC, frames_at())
    assert len(e.pending_market_signals) == 1

    next_open = 61_000.0
    next_frame = {
        "5m": bars([(next_open, 61_100, 60_900, 61_050)], start_ts=T0 + HOUR),
        "15m": bars([(next_open, 61_100, 60_900, 61_050)], start_ts=T0 + HOUR),
        "1h": bars([(next_open, 61_100, 60_900, 61_050)], start_ts=T0 + HOUR),
    }
    e.process_deferred_markets(BTC, next_frame, bar_ts=T0 + HOUR)

    position = e.bot_state.positions[0]
    assert position.stop == pytest.approx(next_open * 0.996)
    assert position.tp1 == pytest.approx(next_open * 1.004)


def test_25배_정책은_설정된_목표에서_전량_익절한다(tmp_path):
    e = engine(tmp_path, Fixed25Signal())
    e.evaluate_entries(BTC, frames_at())
    position = e.bot_state.positions[0]
    e.strategies[BTCU][0].emit = False

    e.manage_positions(
        BTC,
        {
            "15m": bars(
                [(60_000, position.tp1 + 10, position.stop + 10, position.tp1)],
                start_ts=T0 + HOUR,
            )
        },
    )

    assert e.bot_state.positions == []
    exits = [t for t in logs.read_trades(tmp_path / "t.csv") if t["action"] == "exit"]
    assert exits[-1]["reason"] == "tp1"
    assert float(exits[-1]["ratio"]) == pytest.approx(1.0)


def test_익절지정가는_메이커수수료로_정산한다(tmp_path):
    class MakerTarget(Fixed25Signal):
        def generate(self, prepared, as_of):
            signals = super().generate(prepared, as_of)
            for signal in signals:
                signal.position_policy = replace(
                    signal.position_policy,
                    tp1_is_maker=True,
                )
            return signals

    e = engine(tmp_path, MakerTarget())
    e.evaluate_entries(BTC, frames_at())
    position = e.bot_state.positions[0]
    e.strategies[BTCU][0].emit = False

    e.manage_positions(
        BTC,
        {
            "15m": bars(
                [(60_000, position.tp1 + 10, position.stop + 10, position.tp1)],
                start_ts=T0 + HOUR,
            )
        },
    )

    exit_row = [
        t for t in logs.read_trades(tmp_path / "t.csv") if t["action"] == "exit"
    ][-1]
    assert float(exit_row["fee"]) < 0
    assert exit_row["note"] == "익절 지정가"


# ------------------------------------------------------------------ 청산


def open_one(tmp_path, **kw):
    e = engine(tmp_path, FixedSignal(**kw))
    e.evaluate_entries(BTC, frames_at())
    e.strategies[BTCU][0].emit = False       # 이후엔 신호를 내지 않는다
    return e


def test_손절_청산(tmp_path):
    e = open_one(tmp_path)
    p = e.bot_state.positions[0]

    f = {"1h": bars([(60_000, 60_100, p.stop - 500, p.stop - 400)])}
    e.manage_positions(BTC, f)

    assert e.bot_state.positions == []
    exits = [t for t in logs.read_trades(tmp_path / "t.csv") if t["action"] == "exit"]
    assert exits[-1]["reason"] == "stop"
    assert float(exits[-1]["pnl"]) < 0


def test_1차_익절은_절반만_청산하고_손절을_본전으로(tmp_path):
    e = open_one(tmp_path)
    p = e.bot_state.positions[0]
    tp1 = p.tp1

    f = {"1h": bars([(60_000, tp1 + 100, 59_900, tp1 + 50)])}
    e.manage_positions(BTC, f)

    assert len(e.bot_state.positions) == 1          # 절반 남음
    p = e.bot_state.positions[0]
    assert p.tp1_filled
    assert p.remaining_ratio == pytest.approx(0.5)
    assert p.stop == pytest.approx(p.entry)         # 본전 이동

    exits = [t for t in logs.read_trades(tmp_path / "t.csv") if t["action"] == "exit"]
    assert exits[-1]["reason"] == "tp1"
    assert float(exits[-1]["ratio"]) == pytest.approx(0.5)


def test_익절_뒤_트레일링으로_청산(tmp_path):
    e = open_one(tmp_path)
    p = e.bot_state.positions[0]
    tp1 = p.tp1

    # 1차 익절
    e.manage_positions(BTC, {"1h": bars([(60_000, tp1 + 100, 59_900, tp1 + 50)])})
    # 더 올라가서 트레일 갱신
    e.manage_positions(BTC, {"1h": bars([(tp1, tp1 + 3_000, tp1, tp1 + 2_900)], start_ts=T0 + HOUR)})
    trailed = e.bot_state.positions[0].stop
    assert trailed > p.entry

    # 트레일에 닿음
    e.manage_positions(
        BTC, {"1h": bars([(tp1 + 2_900, tp1 + 2_950, trailed - 100, trailed - 50)], start_ts=T0 + 2 * HOUR)}
    )

    assert e.bot_state.positions == []
    exits = [t for t in logs.read_trades(tmp_path / "t.csv") if t["action"] == "exit"]
    assert exits[-1]["reason"] == "trail"


def test_같은_봉을_두_번_처리하지_않는다(tmp_path):
    """루프는 10초마다 도는데 1시간봉은 1시간에 한 번 마감된다.

    중복 검사는 run_once가 한다 — 진입·청산·지정가 체결이 같은 봉 경계를
    공유해야 순서가 어긋나지 않기 때문이다.
    """

    e = open_one(tmp_path)
    f = frames_at()
    ts = int(f["1h"]["ts"].iloc[-1])

    # 진입한 봉은 건너뛴다 — 그 봉의 가격은 이미 지나갔다.
    # 다음 봉부터 세고, 같은 봉을 여러 번 부르면 한 번만 센다.
    e.bot_state.positions[0].last_bar_ts = ts - HOUR
    for _ in range(5):
        e.manage_positions(BTC, f, ts)

    assert e.bot_state.positions[0].bars_held == 1


def test_새_봉마다_보유_봉수가_증가(tmp_path):
    e = open_one(tmp_path)

    for i in range(3):
        e.manage_positions(BTC, {"1h": bars([(60_000, 60_100, 59_900, 60_000)], start_ts=T0 + i * HOUR)})

    assert e.bot_state.positions[0].bars_held == 3


def test_시간_손절(tmp_path):
    e = open_one(tmp_path)
    e.bot_state.positions[0].bars_held = 95        # max_hold_bars = 96

    e.manage_positions(BTC, {"1h": bars([(60_000, 60_100, 59_900, 60_000)], start_ts=T0 + 99 * HOUR)})

    assert e.bot_state.positions == []
    exits = [t for t in logs.read_trades(tmp_path / "t.csv") if t["action"] == "exit"]
    assert exits[-1]["reason"] == "time"


def test_전략_국면이_무효화되면_종가로_청산한다(tmp_path):
    strategy = Fixed25Signal()
    e = engine(tmp_path, strategy)
    e.evaluate_entries(BTC, frames_at())
    strategy.emit = False
    strategy.invalidate = True

    e.manage_positions(
        BTC,
        {"15m": bars([(60_000, 60_100, 59_900, 60_050)], start_ts=T0 + HOUR)},
    )

    assert e.bot_state.positions == []
    exits = [t for t in logs.read_trades(tmp_path / "t.csv") if t["action"] == "exit"]
    assert exits[-1]["reason"] == "regime"


def test_청산이_잔고와_연패에_반영된다(tmp_path):
    e = open_one(tmp_path)
    p = e.bot_state.positions[0]
    before = e.bot_state.balance
    changes = []
    e.notifier.balance_change = lambda *args, **kwargs: changes.append(args) or True

    e.manage_positions(BTC, {"1h": bars([(60_000, 60_100, p.stop - 500, p.stop - 400)])})

    assert e.bot_state.balance < before
    assert e.bot_state.consecutive_losses.get("btc_trend") == 1
    assert changes
    assert changes[0][2].startswith("청산(")
    assert changes[0][3] == pytest.approx(before)
    assert changes[0][6] == pytest.approx(e.bot_state.balance)


def test_상태가_저장되고_복구된다(tmp_path):
    """재시작해도 열린 포지션을 잊지 않아야 한다."""

    e = open_one(tmp_path)
    path = tmp_path / "state.json"
    st.save(e.bot_state, path)

    back = st.load(path)
    assert len(back.positions) == 1
    assert back.positions[0].stop == pytest.approx(e.bot_state.positions[0].stop)
    assert back.last_bar_ts == e.bot_state.last_bar_ts


def test_이상한_봉은_포지션을_강제청산하지_않는다(tmp_path):
    """데이터가 이상하다고 포지션을 던지면 더 위험하다."""

    e = open_one(tmp_path)
    broken = {"1h": bars([(60_000, 100, 200, 60_000)])}      # high < low

    e.manage_positions(BTC, broken)

    assert len(e.bot_state.positions) == 1


# ------------------------------------------------------------------ 시간 순서
# 아래 5개는 전부 실제로 있었던 결함의 회귀 테스트다.
# 공통 원인: "같은 확정 봉을 한 번만, 주문 이후 가격으로만" 규칙이
# 청산 쪽에만 적용되고 진입 쪽에는 빠져 있었다.


class _FakeResult:
    def __init__(self, df):
        self.df = df
        self.problems = []


def _fake_results(frames):
    return {tf: _FakeResult(df) for tf, df in frames.items()}


def cycle(e, spec, frames, bar_ts):
    """run_once가 하는 순서를 그대로 재현한다."""

    e.manage_positions(spec, frames, bar_ts)
    e.process_pending(spec, frames, bar_ts)
    e.evaluate_entries(spec, frames, bar_ts)


def test_전략이_main에_등록되어_있다():
    """등록을 빠뜨리면 엔진이 조용히 아무것도 하지 않는다."""

    from engine.loop import build_strategies

    built = build_strategies()
    assert set(built) == {"BTC_USDT", "XRP_USDT"}
    names = [s.name for group in built.values() for s in group]
    assert "btc_regression_channel" in names
    assert "btc_regime_25x" in names
    assert "btc_trend" not in names and "btc_fib" not in names
    assert any(n.startswith("xrp_patterns.") for n in names)


def test_같은_봉으로_두_번_진입하지_않는다(tmp_path, monkeypatch):
    """run_once를 여러 번 돌려도 같은 확정 봉이면 진입은 1회여야 한다.

    고치기 전에는 10초마다 같은 신호로 다시 진입해, 12시 봉 하나로
    20초 만에 3분할이 전부 채워졌다.
    """

    e = engine(tmp_path, FixedSignal())
    f = frames_at()
    monkeypatch.setattr(e, "fetch", lambda spec: _fake_results(f))
    monkeypatch.setattr(e, "_persist", lambda: None)

    for _ in range(6):
        e.run_once()

    assert len(e.bot_state.positions) == 1


def test_지정가는_주문을_낸_봉으로_체결되지_않는다(tmp_path):
    """가장 위험했던 결함. 신호는 봉 마감 후에 나오므로, 그 봉 안에서
    이미 지나간 가격에는 주문이 존재하지도 않았다.

    **주문 제출 뒤 같은 봉으로 체결을 시도한다.** run_once의 정상 순서는
    체결 검사가 먼저라 이 상황이 안 생기지만, 그 순서에만 의존하면
    호출 순서를 바꾸는 순간 미래참조가 되살아난다. 가드 자체를 검증한다.
    """

    e = engine(tmp_path, LimitSignal(price=59_000.0))
    # 이 봉은 저가가 58,000이라 지정가 59,000을 '지나간' 봉이다
    f = {"1h": bars([(60_000, 60_200, 58_000, 60_000)])}
    ts = int(f["1h"]["ts"].iloc[-1])

    e.evaluate_entries(BTC, f, ts)          # 주문 제출
    assert len(e.pending_signals) == 1
    assert e.pending_signals[0]["placed_ts"] == ts

    e.process_pending(BTC, f, ts)           # 같은 봉으로 체결 시도

    assert len(e.pending_signals) == 1, "주문을 낸 봉으로 체결되면 미래참조다"
    assert e.bot_state.positions == []
    assert e.limit_stats["filled"] == 0


def test_다음_봉에서는_체결된다(tmp_path):
    e = engine(tmp_path, LimitSignal(price=59_000.0))
    f1 = {"1h": bars([(60_000, 60_200, 59_500, 60_000)])}
    ts1 = int(f1["1h"]["ts"].iloc[-1])
    cycle(e, BTC, f1, ts1)
    assert len(e.pending_signals) == 1

    e.strategies[BTCU][0].emit = False
    f2 = {"1h": bars([(60_000, 60_100, 58_500, 59_200)], start_ts=T0 + HOUR)}
    cycle(e, BTC, f2, int(f2["1h"]["ts"].iloc[-1]))

    assert e.pending_signals == []
    assert len(e.bot_state.positions) == 1
    assert e.bot_state.positions[0].entry_tf == "1h"
    assert e.limit_stats["filled"] == 1


def test_지정가_체결봉이_손절도_지나면_같은봉에_청산한다(tmp_path):
    """체결 뒤 손절까지 내려간 봉을 다음 봉까지 열린 것으로 두면 낙관 편향이다."""

    e = engine(tmp_path, LimitSignal(price=59_000.0))
    first = {"1h": bars([(60_000, 60_200, 59_500, 60_000)])}
    cycle(e, BTC, first, int(first["1h"]["ts"].iloc[-1]))
    assert len(e.pending_signals) == 1

    e.strategies[BTCU][0].emit = False
    # 최종 stop은 stop_hint(57,230)보다 ATR 버퍼만큼 더 아래다.
    # 56,000이면 지정가 59,000과 최종 stop을 모두 지난다.
    fill_and_stop = {
        "1h": bars(
            [(60_000, 60_100, 56_000, 56_500)],
            start_ts=T0 + HOUR,
        )
    }
    cycle(e, BTC, fill_and_stop, int(fill_and_stop["1h"]["ts"].iloc[-1]))

    assert e.pending_signals == []
    assert e.bot_state.positions == []
    trades = logs.read_trades(tmp_path / "t.csv")
    assert [t["action"] for t in trades] == ["entry", "exit"]
    assert trades[-1]["reason"] == "stop"


def test_첫_주문도_TTL이_적용된다(tmp_path):
    """placed_ts가 0이면 나이가 계속 0이라 영원히 만료되지 않았다."""

    e = engine(tmp_path, LimitSignal(price=50_000.0))     # 닿지 않는 가격
    f = {"1h": bars([(60_000, 60_200, 59_500, 60_000)])}
    ts0 = int(f["1h"]["ts"].iloc[-1])
    cycle(e, BTC, f, ts0)

    assert e.pending_signals[0]["placed_ts"] == ts0, "봉 타임스탬프가 기록돼야 한다"
    assert e.pending_signals[0]["placed_ts"] != 0

    e.strategies[BTCU][0].emit = False
    for k in range(1, 5):
        fk = {"1h": bars([(60_000, 60_200, 59_500, 60_000)], start_ts=T0 + k * HOUR)}
        cycle(e, BTC, fk, int(fk["1h"]["ts"].iloc[-1]))

    assert e.pending_signals == []
    assert e.limit_stats["expired"] == 1


def test_미체결_주문도_포지션으로_센다(tmp_path):
    """빼면 최대 1포지션 전략이 대기 주문을 쌓았다가 한꺼번에 체결된다."""

    e = engine(tmp_path, LimitSignal(price=59_000.0, max_tranches=1))
    for k in range(3):
        fk = {"1h": bars([(60_000, 60_200, 59_500, 60_000)], start_ts=T0 + k * HOUR)}
        cycle(e, BTC, fk, int(fk["1h"]["ts"].iloc[-1]))

    assert len(e.pending_signals) == 1, f"대기 {len(e.pending_signals)}건 — 중복 제출"


def test_반대방향은_대기_주문도_막는다(tmp_path):
    e = engine(tmp_path, LimitSignal(price=59_000.0, side="long"))
    f = {"1h": bars([(60_000, 60_200, 59_500, 60_000)])}
    cycle(e, BTC, f, int(f["1h"]["ts"].iloc[-1]))
    assert len(e.pending_signals) == 1

    e.strategies[BTCU] = [LimitSignal(price=61_000.0, side="short")]
    f2 = {"1h": bars([(60_000, 60_200, 59_500, 60_000)], start_ts=T0 + HOUR)}
    cycle(e, BTC, f2, int(f2["1h"]["ts"].iloc[-1]))

    assert len(e.pending_signals) == 1
    assert logs.read_signals(tmp_path / "s.jsonl")[-1]["skip_reason"] == "opposite_position_open"


# ------------------------------------------------------------------ 리플레이 대비


def test_prepare_캐시가_실시간에서_얼어붙지_않는다(tmp_path):
    """실시간은 매번 같은 봉 수(500)를 받는다. 길이만 키로 쓰면
    캐시가 영원히 유지되어 지표가 기동 시점에 멈춘다."""

    calls = []

    class Counting(FixedSignal):
        def prepare(self, frames):
            calls.append(len(frames["1h"]))
            return super().prepare(frames)

    e = engine(tmp_path, Counting())

    # 길이는 같고 마지막 봉만 다른 프레임 두 개
    f1 = frames_at()
    f2 = {tf: df.assign(ts=df["ts"] + HOUR) for tf, df in f1.items()}

    e.evaluate_entries(BTC, f1, int(f1["1h"]["ts"].iloc[-1]))
    e.evaluate_entries(BTC, f2, int(f2["1h"]["ts"].iloc[-1]))

    assert len(calls) == 2, "새 봉인데 캐시를 재사용했다"


def test_prepare_캐시가_같은_프레임은_재사용한다(tmp_path):
    """리플레이는 전체 프레임을 고정으로 주고 as_of만 옮긴다."""

    calls = []

    class Counting(FixedSignal):
        def prepare(self, frames):
            calls.append(1)
            return super().prepare(frames)

    e = engine(tmp_path, Counting())
    f = frames_at()
    ts = int(f["1h"]["ts"].iloc[-1])

    for _ in range(5):
        e.evaluate_entries(BTC, f, ts)

    assert len(calls) == 1, f"같은 프레임인데 {len(calls)}회 계산했다"


def test_as_of를_지정하면_그_봉으로_판정한다(tmp_path):
    """리플레이가 과거 시점을 지정하는 경로."""

    e = engine(tmp_path, FixedSignal())
    f = frames_at()

    e.evaluate_entries(BTC, f, bar_ts=None, at=25)

    sig = logs.read_signals(tmp_path / "s.jsonl")[0]
    assert sig["action"] == "entry"


def test_일일_기준점은_봉_시각으로_옮긴다(tmp_path):
    """실제 시각을 쓰면 리플레이에서 날짜가 절대 바뀌지 않는다.

    그러면 일일 손실 한도가 한 번도 리셋되지 않아, 첫 5% 손실 이후
    모든 진입이 막힌다(실측 949건 거절).
    """

    e = engine(tmp_path)
    e.bot_state.balance = 900.0
    e.bot_state.day_key = "2020-01-01"

    e._roll_day(T0)                       # 2023-11-14 UTC

    assert e.bot_state.day_key == "2023-11-14"
    assert e.bot_state.day_start_balance == pytest.approx(900.0)

    # 같은 날 안에서는 안 옮긴다
    e.bot_state.balance = 800.0
    e._roll_day(T0 + 3600)
    assert e.bot_state.day_start_balance == pytest.approx(900.0)

    # 다음 날이면 옮긴다
    e._roll_day(T0 + 86400)
    assert e.bot_state.day_start_balance == pytest.approx(800.0)


def test_리플레이는_매번_로그를_비운다(tmp_path):
    """이어 쓰면 직전 실행 결과가 섞여 조합 비교가 오염된다.

    실제로 BTC만 돌렸는데 직전 XRP 실행의 체결이 그대로 집계됐다.
    """

    from engine.replay import replay
    from config.symbols import BTC as BTC_SPEC

    log_dir = tmp_path / "replay"
    log_dir.mkdir()
    stale = log_dir / "trades.csv"
    stale.write_text("ts,dt,strategy\n1,x,오래된실행\n", encoding="utf-8")

    try:
        replay({}, symbols=(BTC_SPEC,), max_bars=1, log_dir=log_dir)
    except SystemExit:
        pytest.skip("캐시 없음")

    assert "오래된실행" not in stale.read_text(encoding="utf-8")


def test_포지션은_자기_봉으로_진행한다(tmp_path):
    """하나로 묶으면 1시간봉 포지션이 5분마다 bars_held가 올라
    40봉 한도가 3.3시간 만에 찬다 (실측: double_top.1h 50건 중 47건 시간 손절)."""

    e = open_one(tmp_path)
    p = e.bot_state.positions[0]
    p.entry_tf = "1h"

    f = frames_at()
    base = int(f["1h"]["ts"].iloc[-1])
    p.last_bar_ts = base - HOUR          # 직전 봉에 진입했다고 둔다

    # 같은 1시간봉 안에서 5분봉이 12번 돌아도 한 번만 센다
    for k in range(12):
        e.manage_positions(BTC, f, base + k * 300)

    assert e.bot_state.positions[0].bars_held == 1, "같은 봉을 여러 번 셌다"


def test_진입한_봉으로는_청산_판정하지_않는다(tmp_path):
    """진입은 봉 마감 후에 일어난다. 그 봉으로 청산을 보면 지나간 가격이다."""

    e = open_one(tmp_path)
    f = frames_at()
    ts = int(f["1h"]["ts"].iloc[-1])
    e.bot_state.positions[0].last_bar_ts = ts

    e.manage_positions(BTC, f, ts)

    assert e.bot_state.positions[0].bars_held == 0


# ------------------------------------------------------------------ 실시간 근사 리플레이


def _minute_bars(rows, start_ts=T0):
    ts = [start_ts + i * FIVE_MIN for i in range(len(rows))]
    return pd.DataFrame(
        {
            "ts": ts,
            "dt": pd.to_datetime(ts, unit="s", utc=True),
            "open": [float(r[0]) for r in rows],
            "high": [float(r[1]) for r in rows],
            "low": [float(r[2]) for r in rows],
            "close": [float(r[3]) for r in rows],
            "volume": [100.0] * len(rows),
            "quote_volume": [100.0] * len(rows),
        }
    )


def test_상위봉은_마감된_뒤에만_선택한다():
    one_hour = bars([(60_000, 60_100, 59_900, 60_000)], warmup=0)

    assert _closed_index_at(one_hour, "1h", T0 + 5 * 60) == -1
    assert _closed_index_at(one_hour, "1h", T0 + HOUR - 1) == -1
    assert _closed_index_at(one_hour, "1h", T0 + HOUR) == 0


def test_다중시간봉_전략은_1시간봉_마감전에_평가하지_않는다(tmp_path, monkeypatch):
    calls = []

    class Hourly:
        name = "hourly"
        tf = SimpleNamespace(entry="1h")

        def prepare(self, frames):
            return Prepared(frames=frames, series={tf: {} for tf in frames})

        def generate(self, prepared, as_of):
            calls.append(as_of)
            return []

    spec = SymbolSpec(
        contract="XRP_USDT",
        strategies=(StrategySpec("driver", "5m", None, False, "test"),),
        capital_weight=1.0,
        extra_timeframes=("1h",),
    )
    frames = {
        "5m": _minute_bars([(100, 101, 99, 100)] * 12),
        "1h": bars([(100, 101, 99, 100)], warmup=0, warmup_price=100),
    }
    e = Engine(
        strategies={"XRP_USDT": [Hourly()]},
        symbols=(spec,),
        client=object(),
        trade_log=logs.TradeLog(tmp_path / "t.csv"),
        signal_log=logs.SignalLog(tmp_path / "s.jsonl"),
        notifier=Notifier(url=""),
        bot_state=st.BotState(),
    )
    monkeypatch.setattr(e, "fetch", lambda _spec: _fake_results(frames))
    monkeypatch.setattr(e, "_persist", lambda: None)

    e.run_once(decision_ts=T0 + FIVE_MIN)
    assert calls == []

    e.run_once(decision_ts=T0 + HOUR)
    assert calls == [0]


def test_시장가_신호는_다음_5분봉_시가에_체결하고_그봉을_관리한다(
    tmp_path, monkeypatch
):
    class Once:
        name = "xrp_patterns.next_open"
        tf = SimpleNamespace(entry="5m")

        def prepare(self, frames):
            return Prepared(frames=frames, series={tf: {} for tf in frames})

        def generate(self, prepared, as_of):
            if as_of != 0:
                return []
            return [
                Signal(
                    strategy=self.name,
                    symbol="XRP_USDT",
                    side="long",
                    entry=1.0,
                    stop_hint=0.98,
                    target_hint=1.05,
                    reasons=["test"],
                    context={"atr": 0.005},
                )
            ]

    spec = SymbolSpec(
        contract="XRP_USDT",
        strategies=(StrategySpec("driver", "5m", None, False, "test"),),
        capital_weight=1.0,
    )
    frames = {
        "5m": _minute_bars(
            [
                (1.0, 1.005, 0.995, 1.0),
                (1.01, 1.02, 1.0, 1.015),
            ]
        )
    }
    e = Engine(
        strategies={"XRP_USDT": [Once()]},
        symbols=(spec,),
        client=object(),
        trade_log=logs.TradeLog(tmp_path / "t.csv"),
        signal_log=logs.SignalLog(tmp_path / "s.jsonl"),
        notifier=Notifier(url=""),
        bot_state=st.BotState(),
        defer_market_entries=True,
    )
    monkeypatch.setattr(e, "fetch", lambda _spec: _fake_results(frames))
    monkeypatch.setattr(e, "_persist", lambda: None)

    e.run_once(decision_ts=T0 + FIVE_MIN)
    assert e.bot_state.positions == []
    assert len(e.pending_market_signals) == 1

    e.run_once(decision_ts=T0 + 2 * FIVE_MIN)

    assert e.pending_market_signals == []
    assert len(e.bot_state.positions) == 1
    p = e.bot_state.positions[0]
    assert p.entry > 1.01                  # 다음 시가 + 불리한 슬리피지
    assert p.bars_held == 1                # 체결된 두 번째 봉도 청산 판정에 포함


def test_날짜는_앞으로만_넘어간다(tmp_path):
    """_roll_day는 종목마다 불린다. 봉 마감 시각이 자정을 사이에 두고
    갈리면(XRP 5분봉 00:00 마감 vs BTC 1시간봉 23:00 마감) 한 회차에
    두 번 뒤집히고, 10초 폴링에서 시간당 720건의 알림이 나간다.
    """

    import calendar

    sent = []

    class Spy:
        enabled = True
        def text(self, m): sent.append(m); return True
        def warn(self, *a): return True
        def close(self): pass

    e = engine(tmp_path)
    e.notifier = Spy()
    e.bot_state.day_key = "2026-08-08"
    mid = calendar.timegm((2026, 8, 9, 0, 0, 0))

    for _ in range(6):
        e._roll_day(mid + 300)      # 08-09 (앞선 봉)
        e._roll_day(mid - 3600)     # 08-08 (뒤처진 봉)

    assert len(sent) == 1, f"{len(sent)}건 — 날짜가 뒤로도 넘어간다"
    assert e.bot_state.day_key == "2026-08-09"


def test_날짜가_바뀌면_기준점이_옮겨간다(tmp_path):
    import calendar

    e = engine(tmp_path)
    e.bot_state.day_key = "2026-08-08"
    e.bot_state.balance = 950.0
    e.bot_state.daily_entries = {"btc_regime_25x": 3}

    e._roll_day(calendar.timegm((2026, 8, 9, 0, 5, 0)))

    assert e.bot_state.day_key == "2026-08-09"
    assert e.bot_state.day_start_balance == pytest.approx(950.0)
    assert e.bot_state.daily_entries == {}
