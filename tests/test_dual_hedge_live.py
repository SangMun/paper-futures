"""EXP-015 실시간 관측 경로 테스트. 네트워크를 쓰지 않는다."""

from __future__ import annotations

import pandas as pd
import pytest

from core.live import LiveQuote
from core.exchange import FuturesContractMarket
from engine.dual_hedge import DualHedgeReplay, config_with
from strategies.xrp_dual_hedge import squeeze_breakout_ready
from tools.paper_xrp_dual_hedge_live import PaperSession, build_sessions


class FakeNotifier:
    def __init__(self):
        self.calls = []

    def embed(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return True


def market(mark_price: float = 1.0, maintenance_rate: float = 0.005) -> FuturesContractMarket:
    return FuturesContractMarket(
        contract="XRP_USDT", mark_price=mark_price,
        maintenance_rate=maintenance_rate, taker_fee_rate=0.00075,
        multiplier=10.0, risk_limit_base=50_000.0,
    )


def test_실시간_이익보장가는_종가를_기다리지_않고_구간고가에서_활성화된다():
    engine = DualHedgeReplay(
        balance=1_000,
        config=config_with(leverage=50, pair_margin_fraction=0.02),
    )
    # 새 진입 시점보다 앞선 구간은 소급 청산하지 않는다.
    assert engine.process_live_window(
        ts=0, open_=1.0, high=1.5, low=0.5, last=1.0, atr=0.001, allow_entry=True,
    ) == "entry"

    # 다음 관측 구간에서 숏만 청산되어 롱이 생존한다.
    assert engine.process_live_window(
        ts=10, open_=1.0, high=1.02, low=1.0, last=1.019, atr=0.001, allow_entry=False,
    ) == "liquidation"
    pair = engine.pair
    assert pair is not None and pair.survivor is not None
    lock = pair.profit_lock_price
    assert lock is not None

    # 종가가 아니라 관측 구간의 고가가 이익 보장가에 닿는 순간 스탑을 건다.
    # 끝값은 lock 위에 두어 같은 구간 안의 재하락 종료와 구분한다.
    assert engine.process_live_window(
        ts=20, open_=1.019, high=lock + 0.0001, low=1.018,
        last=lock + 0.0001, atr=0.001, allow_entry=False,
    ) == "profit_lock_armed"
    assert engine.pair is not None and engine.pair.stop == lock

    assert engine.process_live_window(
        ts=30, open_=lock, high=lock, low=lock, last=lock, atr=0.001, allow_entry=False,
    ) == "profit_lock_trail"
    assert engine.results[-1].pnl >= -1e-9


def test_실시간_청산은_거래가가_아닌_mark_price로만_판정한다():
    engine = DualHedgeReplay(balance=1_000, config=config_with(leverage=100))
    assert engine.process_live_window(
        ts=0, open_=1.0, high=1.0, low=1.0, last=1.0, atr=0.001,
        allow_entry=True, mark_open=1.0, mark_high=1.0, mark_low=1.0,
        mark_last=1.0, maintenance_rate=0.005,
    ) == "entry"
    pair = engine.pair
    assert pair is not None

    # 거래가는 숏 청산가를 넘었지만 Mark Price는 그 아래다. 이 구간에서
    # 청산으로 기록하면 Gate의 MMR 기준과 다르다.
    assert engine.process_live_window(
        ts=1, open_=1.0, high=pair.short.liquidation_price + 0.01,
        low=1.0, last=pair.short.liquidation_price + 0.01, atr=0.001,
        allow_entry=False, mark_open=1.0,
        mark_high=pair.short.liquidation_price - 0.0001,
        mark_low=1.0, mark_last=pair.short.liquidation_price - 0.0001,
        maintenance_rate=0.005,
    ) is None

    assert engine.process_live_window(
        ts=2, open_=1.0, high=1.0, low=1.0, last=1.0, atr=0.001,
        allow_entry=False, mark_open=1.0,
        mark_high=pair.short.liquidation_price + 0.0001,
        mark_low=1.0, mark_last=pair.short.liquidation_price + 0.0001,
        maintenance_rate=0.005,
    ) == "liquidation"


def test_첫_risk_limit을_넘는_pair는_1단계_mmr로_열지_않고_관측을_계속한다():
    engine = DualHedgeReplay(
        balance=1_000, config=config_with(leverage=100, pair_margin_fraction=0.02),
    )
    # 루프를 예외로 죽이지 않고 진입만 막는다. 진입하지 않았으면 entry 사건도 없다.
    assert engine.process_live_window(
        ts=0, open_=1.0, high=1.0, low=1.0, last=1.0, atr=0.001,
        allow_entry=True, maintenance_rate=0.005, first_tier_risk_limit=999.0,
    ) is None
    assert engine.pair is None
    assert engine.broker.balance == 1_000


def test_청산된_다리는_Gate_격리처럼_배정증거금까지만_잃는다():
    engine = DualHedgeReplay(balance=1_000, config=config_with(leverage=50))
    engine.process_live_window(
        ts=0, open_=1.0, high=1.0, low=1.0, last=1.0, atr=0.001, allow_entry=True,
    )
    pair = engine.pair
    after_entry = engine.broker.balance
    assert engine.process_live_window(
        ts=1, open_=1.0, high=1.05, low=1.0, last=1.05, atr=0.001, allow_entry=False,
    ) == "liquidation"
    # 청산 수수료는 파산가 안, 즉 증거금 안에서 나간다. 증거금 밖에서 더 빼지 않는다.
    assert after_entry - engine.broker.balance == pytest.approx(pair.short.margin)


def test_봉_전환_전_tail에서_닿은_스탑을_놓치지_않는다(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.paper_xrp_dual_hedge_live.env.get", lambda *_: None)
    session = PaperSession("continuous", 50, 5.0, 1_000, "TEST_WEBHOOK", tmp_path / "events.jsonl")
    bars = [{"ts": 0, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}]
    session.observe(LiveQuote(0, 10, 1.0, 1.0, 1.0, 1.0), market(1.0), pd.DataFrame(bars[:0]))
    session.observe(LiveQuote(0, 20, 1.0, 1.05, 1.0, 1.05), market(1.05), pd.DataFrame(bars[:0]))
    pair = session.engine.pair
    assert pair.liquidated_side == "short"
    bars[0].update(high=1.05, close=1.05)
    session.observe(LiveQuote(60, 70, 1.05, 1.05, 1.05, 1.05), market(1.05), pd.DataFrame(bars))
    session.observe(LiveQuote(60, 115, 1.05, 1.06, 1.05, 1.06), market(1.06), pd.DataFrame(bars))
    assert pair.armed
    # 마지막 관측(115초) 뒤 봉 마감까지 스탑 아래로 내려갔다가 1.06에 마감했다.
    bars.append({"ts": 60, "open": 1.05, "high": 1.06, "low": pair.stop - 0.003, "close": 1.06})
    session.observe(LiveQuote(120, 121, 1.06, 1.06, 1.06, 1.06), market(1.06), pd.DataFrame(bars))
    # 연속 전략은 같은 관측의 현재가에 다음 pair를 연다. 이전 pair는 tail 스탑으로 끝나야 한다.
    assert [r.exit_reason for r in session.engine.results] == ["profit_lock_trail"]
    assert session.engine.pair.pair_id == 2


def test_실시간_이익보장가_통과_뒤_같은_구간에서_재하락하면_즉시_종료한다():
    engine = DualHedgeReplay(balance=1_000, config=config_with(leverage=50))
    assert engine.process_live_window(
        ts=0, open_=1.0, high=1.0, low=1.0, last=1.0, atr=0.001, allow_entry=True,
    ) == "entry"
    assert engine.process_live_window(
        ts=10, open_=1.0, high=1.02, low=1.0, last=1.019, atr=0.001, allow_entry=False,
    ) == "liquidation"
    pair = engine.pair
    assert pair is not None and pair.profit_lock_price is not None
    lock = pair.profit_lock_price

    # 고가가 lock에 닿고 마지막 가격이 그 아래면 lock 통과 뒤 재하락은 확정적이다.
    assert engine.process_live_window(
        ts=20, open_=1.019, high=lock, low=1.018, last=1.019, atr=0.001, allow_entry=False,
    ) == "profit_lock_trail"
    assert engine.pair is None


def test_활성화된_스탑은_더먼_생존다리_청산가보다_우선한다():
    engine = DualHedgeReplay(balance=1_000, config=config_with(leverage=50))
    assert engine.process_live_window(
        ts=0, open_=1.0, high=1.0, low=1.0, last=1.0, atr=0.001, allow_entry=True,
    ) == "entry"
    assert engine.process_live_window(
        ts=10, open_=1.0, high=1.02, low=1.0, last=1.019, atr=0.001, allow_entry=False,
    ) == "liquidation"
    pair = engine.pair
    assert pair is not None and pair.profit_lock_price is not None and pair.survivor is not None
    lock = pair.profit_lock_price
    assert engine.process_live_window(
        ts=20, open_=1.019, high=lock + 0.0001, low=1.018,
        last=lock + 0.0001, atr=0.001, allow_entry=False,
    ) == "profit_lock_armed"

    # 롱 생존 다리의 원래 청산가까지 내려가도, 미리 활성화된 lock 스탑으로 끝난다.
    assert engine.process_live_window(
        ts=30, open_=lock + 0.001, high=lock + 0.001,
        low=pair.survivor.liquidation_price - 0.0001,
        last=pair.survivor.liquidation_price - 0.0001,
        atr=0.001, allow_entry=False,
    ) == "profit_lock_trail"
    assert engine.results[-1].exit_reason == "profit_lock_trail"


def test_스퀴즈_뒤_확정봉_돌파에서만_진입후보가_나온다():
    rows = []
    for i in range(150):
        rows.append({"open": 1.0, "high": 1.001, "low": 0.999, "close": 1.0})
    rows[-1] = {"open": 1.0, "high": 1.021, "low": 0.999, "close": 1.02}
    assert squeeze_breakout_ready(pd.DataFrame(rows), config_with())


def test_재시작_뒤_캔들로_mark_price를_소급추정하지_않는다(tmp_path, monkeypatch):
    # 테스트가 로컬 .env나 Discord에 의존하지 않게 한다.
    monkeypatch.setattr("tools.paper_xrp_dual_hedge_live.env.get", lambda *_: None)
    event_path = tmp_path / "continuous" / "events.jsonl"
    session = PaperSession("continuous", 50, 5.0, 1_000, "TEST_WEBHOOK", event_path)
    session.observe(
        LiveQuote(0, 10, 1.0, 1.0, 1.0, 1.0),
        market(),
        pd.DataFrame(columns=["ts", "open", "high", "low", "close"]),
    )
    assert session.engine.pair is not None

    # 2분 공백 뒤 거래가 캔들은 큰 상승을 보인다. 하지만 과거 캔들로 Mark
    # Price를 만들어 청산/이익보장 순서를 지어내면 안 된다. 현재 Mark Price
    # 관측에서 확인되는 청산만 기록하고, 다음 관측부터 보호선을 판정한다.
    closed = pd.DataFrame([
        {"ts": 0, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0},
        {"ts": 60, "open": 1.0, "high": 1.02, "low": 1.0, "close": 1.019},
        {"ts": 120, "open": 1.019, "high": 1.05, "low": 1.019, "close": 1.05},
    ])
    session.observe(LiveQuote(180, 181, 1.05, 1.05, 1.05, 1.05), market(1.05), closed)
    assert session.engine.pair is not None
    assert session.engine.pair.liquidated_side == "short"
    assert not session.engine.pair.armed
    assert session.state_path.exists()

    restored = PaperSession("continuous", 50, 5.0, 1_000, "TEST_WEBHOOK", event_path)
    assert restored.engine.pair is not None and not restored.engine.pair.armed
    # 같은 진행 봉의 다음 Mark 관측과 거래가 기준으로 보호선을 활성화한다.
    restored.observe(LiveQuote(180, 190, 1.05, 1.05, 1.05, 1.05), market(1.05), closed)
    assert restored.engine.pair is not None and restored.engine.pair.armed


def test_50x_100x_네세션은_레버리지별_상태와_폴링을_분리한다(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.paper_xrp_dual_hedge_live.env.get", lambda *_: None)
    sessions = build_sessions(
        balance=1_000, poll_50_seconds=5.0, poll_100_seconds=1.0, log_dir=tmp_path,
    )

    assert {(s.leverage, s.strategy) for s in sessions} == {
        (50, "continuous"), (50, "squeeze_breakout"),
        (100, "continuous"), (100, "squeeze_breakout"),
    }
    assert len({s.state_path for s in sessions}) == 4
    assert {s.poll_seconds for s in sessions if s.leverage == 50} == {5.0}
    assert {s.poll_seconds for s in sessions if s.leverage == 100} == {1.0}
    assert all(s.due(0.0) for s in sessions)
    assert not any(s.due(0.5) for s in sessions if s.leverage == 100)
    assert all(s.due(1.0) for s in sessions if s.leverage == 100)
    assert not any(s.due(1.0) for s in sessions if s.leverage == 50)


def test_다른_레버리지_상태파일은_복원하지_않는다(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.paper_xrp_dual_hedge_live.env.get", lambda *_: None)
    event_path = tmp_path / "events.jsonl"
    session = PaperSession("continuous", 50, 5.0, 1_000, "TEST_WEBHOOK", event_path)
    session._save_state()

    with pytest.raises(RuntimeError, match="다른 실험 상태 파일"):
        PaperSession("continuous", 100, 1.0, 1_000, "TEST_WEBHOOK", event_path)


def test_다른_폴링간격_상태파일은_복원하지_않는다(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.paper_xrp_dual_hedge_live.env.get", lambda *_: None)
    event_path = tmp_path / "events.jsonl"
    session = PaperSession("continuous", 50, 5.0, 1_000, "TEST_WEBHOOK", event_path)
    session._save_state()

    with pytest.raises(RuntimeError, match="다른 실험 상태 파일"):
        PaperSession("continuous", 50, 1.0, 1_000, "TEST_WEBHOOK", event_path)


def test_복구에_필요한_캔들이_없으면_설명형_오류로_중단한다(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.paper_xrp_dual_hedge_live.env.get", lambda *_: None)
    event_path = tmp_path / "events.jsonl"
    session = PaperSession("continuous", 50, 5.0, 1_000, "TEST_WEBHOOK", event_path)
    session.prev_bar_ts = 60
    session.prev_high = 1.0
    session.prev_low = 1.0
    session.prev_price = 1.0

    with pytest.raises(RuntimeError, match=r"마지막 관측\(60\) 이후의 봉 이력이 부족"):
        session.observe(
            LiveQuote(180, 181, 1.0, 1.0, 1.0, 1.0),
            market(),
            pd.DataFrame(columns=["ts", "open", "high", "low", "close"]),
        )


def test_네세션은_기존_봇_상태채널에_각각_독립된_시작상태를_보낸다(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.paper_xrp_dual_hedge_live.env.get", lambda *_: None)
    sessions = build_sessions(
        balance=1_000, poll_50_seconds=5.0, poll_100_seconds=1.0, log_dir=tmp_path,
    )
    for session in sessions:
        notifier = FakeNotifier()
        session.status_notifier = notifier
        session.announce_started(1_700_000_000)
        assert len(notifier.calls) == 1
        args, kwargs = notifier.calls[0]
        assert session.tag in kwargs["title"]
        assert "시작" in kwargs["title"]
        assert kwargs["channel"] == "system"
        assert kwargs["fields"]["상태"] == "대기"


def test_진입과_청산은_세션별로_기존_잔고채널에_따로_보낸다(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.paper_xrp_dual_hedge_live.env.get", lambda *_: None)
    session = PaperSession(
        "continuous", 50, 5.0, 1_000, "TEST_WEBHOOK", tmp_path / "events.jsonl",
    )
    session.notifier = FakeNotifier()
    session.status_notifier = FakeNotifier()
    # 실제 서비스는 첫 관측 전에 announce_started()로 당일 상태를 이미 알린다.
    session.last_status_day = session._utc_day(10)
    session.observe(
        LiveQuote(0, 10, 1.0, 1.0, 1.0, 1.0),
        market(),
        pd.DataFrame(columns=["ts", "open", "high", "low", "close"]),
    )

    assert len(session.status_notifier.calls) == 1
    _args, kwargs = session.status_notifier.calls[0]
    assert "잔고 변화 · entry" in kwargs["title"]
    assert kwargs["channel"] == "balance"
    assert kwargs["fields"]["pair"] == "1"

    session.observe(
        LiveQuote(0, 20, 1.0, 1.02, 1.0, 1.019),
        market(1.019),
        pd.DataFrame(columns=["ts", "open", "high", "low", "close"]),
    )
    assert len(session.status_notifier.calls) == 2
    _args, kwargs = session.status_notifier.calls[-1]
    assert "잔고 변화 · liquidation" in kwargs["title"]
    assert kwargs["channel"] == "balance"
    assert kwargs["fields"]["pair"] == "1"


def _stop_engine(buffer=0.001):
    engine = DualHedgeReplay(
        balance=1_000, config=config_with(leverage=100, liquidation_stop_buffer=buffer),
    )
    assert engine.process_live_window(
        ts=0, open_=1.0, high=1.0, low=1.0, last=1.0, atr=0.001, allow_entry=True,
    ) == "entry"
    return engine


def test_청산가_직전_스탑은_청산보다_먼저_체결되어_남은_증거금을_지킨다():
    engine = _stop_engine()
    pair = engine.pair
    assert pair.long.stop_price > pair.long.liquidation_price
    assert pair.short.stop_price < pair.short.liquidation_price
    before = engine.broker.balance
    # 구간이 스탑과 청산가를 모두 지나도 연속 경로상 스탑이 먼저다.
    assert engine.process_live_window(
        ts=1, open_=1.0, high=pair.short.liquidation_price + 0.001, low=1.0,
        last=pair.short.liquidation_price + 0.001, atr=0.001, allow_entry=False,
    ) == "loser_stop"
    loss = before - engine.broker.balance
    assert 0 < loss < pair.short.margin * 0.6
    assert pair.survivor is pair.long and pair.profit_lock_price is not None


def test_시작가가_이미_청산가를_넘은_갭은_스탑이_아니라_청산이다():
    engine = _stop_engine()
    pair = engine.pair
    gap = pair.short.liquidation_price + 0.001
    before = engine.broker.balance
    assert engine.process_live_window(
        ts=1, open_=1.0, high=gap, low=1.0, last=gap, atr=0.001, allow_entry=False,
        mark_open=gap, mark_high=gap, mark_low=gap, mark_last=gap,
    ) == "liquidation"
    assert before - engine.broker.balance == pytest.approx(pair.short.margin)


def test_생존_다리도_반전하면_청산가_직전_스탑으로_끝난다():
    engine = _stop_engine()
    pair = engine.pair
    up = pair.short.stop_price
    engine.process_live_window(
        ts=1, open_=1.0, high=up, low=1.0, last=up, atr=0.001, allow_entry=False,
    )
    assert pair.survivor is pair.long
    down = pair.long.liquidation_price - 0.001
    assert engine.process_live_window(
        ts=2, open_=up, high=up, low=down, last=down, atr=0.001, allow_entry=False,
    ) == "survivor_stop"
    result = engine.results[-1]
    assert result.liquidation == "both_sequential"
    # 두 다리 모두 증거금 전액보다 적게 잃는다.
    assert -2 * pair.long.margin < result.pnl < 0


def test_discord_알림은_사람이_읽는_한글_요약으로_보낸다(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.paper_xrp_dual_hedge_live.env.get", lambda *_: None)
    session = PaperSession(
        "continuous", 100, 1.0, 1_000, "TEST_WEBHOOK", tmp_path / "events.jsonl",
        liq_stop_buffer=0.001,
    )
    session.notifier = FakeNotifier()
    empty = pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
    session.observe(LiveQuote(0, 10, 1.0, 1.0, 1.0, 1.0), market(1.0), empty)
    up = session.engine.pair.short.stop_price
    session.observe(LiveQuote(0, 11, 1.0, up, 1.0, up), market(up), empty)

    entry, stop = [kwargs for _, kwargs in session.notifier.calls]
    assert entry["title"].startswith("진입 #1")
    assert set(entry["fields"]) == {"롱", "숏", "잔고"}
    assert stop["title"] == "숏 스탑 · 롱 생존 #1"
    assert "None" not in str(stop["fields"])
    print(entry, stop, sep="\n")
