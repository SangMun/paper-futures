"""엔진·기록 단위 테스트. 네트워크와 웹훅을 타지 않는다."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from config.settings import CONTRACT_SPECS, FEES
from engine import broker as bk
from engine import loop as loop_mod
from engine import state as st
from engine.loop import SKIP_STALE_DATA, Engine
from reporting import logs
from reporting.notify import Notifier


XRP = "XRP_USDT"
BTC = "BTC_USDT"


# ------------------------------------------------------------------ 체결


def test_시장가는_슬리피지가_불리하게_붙는다():
    long = bk.market_fill(XRP, "long", 1.0, 100)
    short = bk.market_fill(XRP, "short", 1.0, 100)

    assert long.price > 1.0
    assert short.price < 1.0
    assert not long.is_maker


def test_메이커는_리베이트라_수수료가_음수():
    """왕복 0.21%p 차이가 XRP 전략의 손익분기 승률을 가른다."""

    taker = bk.fee_for(1_000.0, is_maker=False)
    maker = bk.fee_for(1_000.0, is_maker=True)

    assert taker > 0
    assert maker < 0


def test_지정가는_가격이_지나가야_체결된다():
    order = bk.PendingOrder(XRP, "long", 0.99, 100, "x", placed_at=0, ttl_bars=3)

    assert bk.try_limit_fill(order, high=1.02, low=1.00) is None      # 안 닿음
    fill = bk.try_limit_fill(order, high=1.01, low=0.98)             # 통과
    assert fill is not None
    assert fill.price == 0.99                                        # 슬리피지 없음
    assert fill.is_maker


def test_숏_지정가는_위로_올라와야_체결():
    order = bk.PendingOrder(XRP, "short", 1.01, 100, "x", placed_at=0, ttl_bars=3)

    assert bk.try_limit_fill(order, high=1.00, low=0.98) is None
    assert bk.try_limit_fill(order, high=1.02, low=1.00) is not None


def test_지정가_TTL_만료():
    order = bk.PendingOrder(XRP, "long", 0.5, 100, "x", placed_at=0, ttl_bars=3)

    assert not order.expired(2)
    assert order.expired(3)


def test_브로커가_체결과_만료를_구분해_돌려준다():
    """지정가 체결률은 XRP 전략의 성립 여부를 가르므로 만료도 추적해야 한다."""

    b = bk.Broker(balance=1_000.0)
    b.place_limit(bk.PendingOrder(XRP, "long", 0.99, 100, "a", placed_at=0, ttl_bars=1))
    b.place_limit(bk.PendingOrder(XRP, "long", 0.50, 100, "b", placed_at=0, ttl_bars=1))

    filled, expired = b.process_bar(1, high=1.01, low=0.98)

    assert [o.strategy for o, _ in filled] == ["a"]
    assert [o.strategy for o in expired] == ["b"]
    assert b.pending == []


def test_손익_계산():
    """XRP 승수 10. 1계약 = 10 XRP."""

    pnl = bk.realize_pnl("long", 1.00, 1.05, contracts=10, symbol=XRP)

    assert pnl == pytest.approx(0.05 * 10 * 10)     # 5.0


def test_숏_손익은_반대():
    assert bk.realize_pnl("short", 1.00, 0.95, 10, XRP) == pytest.approx(5.0)


def test_부분_청산():
    full = bk.realize_pnl("long", 1.00, 1.05, 10, XRP)
    half = bk.realize_pnl("long", 1.00, 1.05, 10, XRP, ratio=0.5)

    assert half == pytest.approx(full / 2)


def test_호가_단위로_반올림():
    assert bk.round_to_tick(1.012345, XRP) == pytest.approx(1.0123)
    assert bk.round_to_tick(60_000.123, BTC) == pytest.approx(60_000.1)


def test_청산이_잔고에_반영된다():
    b = bk.Broker(balance=1_000.0)
    net, fee = b.close("long", XRP, entry=1.0, exit_price=1.05, contracts=10)

    assert fee > 0                              # 테이커
    assert net == pytest.approx(5.0 - fee)
    assert b.balance == pytest.approx(1_000.0 + net)


# ------------------------------------------------------------------ 상태


def test_상태_저장과_복구(tmp_path):
    p = tmp_path / "state.json"
    s = st.BotState(
        balance=1_234.5,
        consecutive_losses={"A": 2},
        daily_entries={"A": 3},
    )
    s.positions.append(
        st.OpenPosition(XRP, "long", "xrp_patterns", 1.0, 10, 2, 0.99, 1.02, 0.5, 0.7, 40, 10.0)
    )

    st.save(s, p)
    back = st.load(p)

    assert back.balance == pytest.approx(1_234.5)
    assert back.consecutive_losses == {"A": 2}
    assert back.daily_entries == {"A": 3}
    assert len(back.positions) == 1
    assert back.positions[0].symbol == XRP


def test_기존_25배_포지션도_복구하면_시간청산을_사용하지_않는다():
    position = st.OpenPosition(
        BTC,
        "long",
        "btc_regime_25x",
        60_000.0,
        400,
        25,
        59_760.0,
        60_468.0,
        1.0,
        1.2,
        12,
        12.0,
    )
    raw = st.BotState(positions=[position]).to_dict()
    del raw["positions"][0]["time_exit_enabled"]

    restored = st.BotState.from_dict(raw)

    assert not restored.positions[0].time_exit_enabled


def test_상태_파일이_없으면_초기값(tmp_path):
    s = st.load(tmp_path / "none.json")

    assert s.positions == []
    assert s.balance > 0


def test_v2_상태는_잔고와_포지션을_보존해_v3로_복구한다():
    raw = {
        "version": 2,
        "balance": 988.95,
        "day_start_balance": 993.92,
        "day_key": "2026-08-11",
        "consecutive_losses": {"legacy": 2},
        "paused_until": {"legacy": 123},
        "positions": [
            {
                "symbol": BTC,
                "side": "long",
                "strategy": "legacy",
                "entry": 60_000.0,
                "contracts": 10,
                "leverage": 2,
                "stop": 59_000.0,
                "tp1": 62_000.0,
                "partial_ratio": 0.5,
                "trail_atr_mult": 1.2,
                "max_hold_bars": 40,
                "risk_amount": 10.0,
            }
        ],
        "last_bar_ts": {"BTC_USDT_1h": 123},
    }

    restored = st.BotState.from_dict(raw)

    assert restored.balance == pytest.approx(988.95)
    assert restored.day_start_balance == pytest.approx(993.92)
    assert restored.consecutive_losses == {"legacy": 2}
    assert restored.paused_until == {"legacy": 123}
    assert restored.daily_entries == {}
    assert restored.last_bar_ts == {"BTC_USDT_1h": 123}
    assert len(restored.positions) == 1
    assert restored.positions[0].entry_tf == "5m"


def test_깨진_상태는_예외(tmp_path):
    """초기화로 대체하지 않는다. 열려 있던 포지션을 잊는 게 가장 위험하다."""

    p = tmp_path / "state.json"
    p.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="깨졌다"):
        st.load(p)


def test_버전이_다르면_예외(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"version": 99, "balance": 1, "day_start_balance": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="버전"):
        st.load(p)


def test_저장은_원자적이다(tmp_path):
    """저장 도중 죽어도 기존 파일이 남아야 한다."""

    p = tmp_path / "state.json"
    st.save(st.BotState(balance=100.0), p)
    st.save(st.BotState(balance=200.0), p)

    assert st.load(p).balance == pytest.approx(200.0)
    assert list(tmp_path.glob("*.tmp")) == []       # 임시 파일이 남지 않는다


# ------------------------------------------------------------------ 기록


def test_체결_CSV는_컬럼이_고정(tmp_path):
    log = logs.TradeLog(tmp_path / "trades.csv")
    log.write(strategy="x", symbol=XRP, side="long", action="entry", price=1.0)

    rows = logs.read_trades(tmp_path / "trades.csv")
    assert list(rows[0]) == list(logs.TRADE_COLUMNS)
    assert rows[0]["symbol"] == XRP


def test_모르는_컬럼은_예외(tmp_path):
    """CSV를 늘리지 않기 위한 장치. 늘릴 항목은 JSONL로 보낸다."""

    log = logs.TradeLog(tmp_path / "trades.csv")
    with pytest.raises(ValueError, match="JSONL"):
        log.write(strategy="x", rsi=55.0)


def test_신호_JSONL은_스키마가_유연(tmp_path):
    log = logs.SignalLog(tmp_path / "signals.jsonl")
    log.write(logs.ACTION_SKIP, "xrp_patterns", XRP, "long",
              skip_reason="stop_too_tight", rsi=42.1, zone={"score": 5})

    rec = logs.read_signals(tmp_path / "signals.jsonl")[0]
    assert rec["skip_reason"] == "stop_too_tight"
    assert rec["zone"]["score"] == 5


def test_스킵된_신호도_남는다(tmp_path):
    """README 원칙 4. 막았던 신호가 실은 좋았는지 나중에 판단해야 한다."""

    log = logs.SignalLog(tmp_path / "signals.jsonl")
    for action in (logs.ACTION_ENTRY, logs.ACTION_SKIP, logs.ACTION_PHANTOM):
        log.write(action, "x", XRP, "long")

    actions = [r["action"] for r in logs.read_signals(tmp_path / "signals.jsonl")]
    assert actions == ["entry", "skip", "phantom"]


# ------------------------------------------------------------------ 알림


def test_웹훅이_없으면_비활성():
    n = Notifier(url="")
    assert not n.enabled
    assert n.text("x") is False


def test_전송_실패가_예외를_던지지_않는다():
    """README 원칙 6. 알림 실패가 매매를 막지 않는다."""

    n = Notifier(url="http://127.0.0.1:1/nope", timeout=0.05)
    assert n.text("x") is False         # 예외 대신 False


def test_알림이_시스템_매매_잔고_채널로_분리된다():
    n = Notifier(url="legacy")
    sent = []
    n._post = lambda payload, channel="system", key="": sent.append(channel) or True
    position = SimpleNamespace(
        contracts=2,
        notional=1_000.0,
        leverage=2,
        stop_distance=0.02,
        risk_amount=10.0,
        liquidation_price=40_000.0,
        liquidation_multiple=lambda: 8.0,
    )

    n.text("시작")
    n.warn("오류", "상세")
    n.entry("btc_trend", BTC, "long", position, 60_000.0)
    n.exit("btc_trend", BTC, "tp1", 63_000.0, 5.0, side="long", partial=True)
    n.balance_change(
        "btc_trend", BTC, "부분 익절", 990.0, 5.0, 1.0, 995.0, 1_000.0
    )

    assert sent == ["system", "system", "trade_btc", "trade_btc", "balance"]
    assert n.system_url == n.trade_url == n.balance_url == "legacy"
    assert n.btc_trade_url == n.xrp_trade_url == "legacy"


def test_분리_웹훅은_각_채널_URL을_쓴다():
    n = Notifier(
        url="",
        system_url="system-url",
        trade_url="trade-url",
        btc_trade_url="btc-trade-url",
        xrp_trade_url="xrp-trade-url",
        balance_url="balance-url",
    )

    assert n._url_for("system") == "system-url"
    assert n._url_for("trade") == "trade-url"
    assert n._url_for("trade_btc") == "btc-trade-url"
    assert n._url_for("trade_xrp") == "xrp-trade-url"
    assert n._url_for("balance") == "balance-url"


def test_진입_알림에_패턴_근거와_당시_상황이_들어간다():
    n = Notifier(url="legacy")
    sent = []
    n._post = lambda payload, channel="system", key="": sent.append((channel, payload)) or True
    position = SimpleNamespace(
        contracts=3,
        notional=500.0,
        leverage=2,
        stop_distance=0.03,
        risk_amount=8.0,
        liquidation_price=0.4,
        liquidation_multiple=lambda: 7.0,
    )

    n.entry(
        "xrp_patterns.head_shoulders",
        XRP,
        "short",
        position,
        1.23,
        reasons=["head_shoulders", "volume_confirmed"],
        score=4.0,
        context={"rsi": 68.2, "neckline": 1.24, "quality": ["symmetry", "volume"]},
        stop=1.27,
        target=1.15,
        order_type="limit",
    )

    channel, payload = sent[0]
    fields = {f["name"]: f["value"] for f in payload["embeds"][0]["fields"]}
    assert channel == "trade_xrp"
    assert "head_shoulders" in fields["진입 근거"]
    assert "`rsi`: 68.2" in fields["당시 상황"]
    assert fields["손절가"] == "1.27"
    assert fields["1차 익절가"] == "1.15"


# ------------------------------------------------------------------ 루프


class DummyResult:
    def __init__(self, problems):
        import pandas as pd

        self.problems = problems
        self.df = pd.DataFrame({"close": [1.0]})


def engine(tmp_path, **kw):
    return Engine(
        client=object(),
        trade_log=logs.TradeLog(tmp_path / "t.csv"),
        signal_log=logs.SignalLog(tmp_path / "s.jsonl"),
        notifier=Notifier(url=""),
        bot_state=st.BotState(),
        **kw,
    )


def test_정합성_문제는_신규진입을_막고_기록한다(tmp_path):
    e = engine(tmp_path)
    spec = e.symbols[0]

    e._block_entries(spec, ["1h: 결측 캔들 3봉"])

    rec = logs.read_signals(tmp_path / "s.jsonl")[0]
    assert rec["skip_reason"] == SKIP_STALE_DATA
    assert rec["symbol"] == spec.contract


def test_문제_목록은_타임프레임_이름이_붙는다(tmp_path):
    e = engine(tmp_path)
    problems = e.data_problems({"1h": DummyResult(["결측 캔들 2봉"]), "1d": DummyResult([])})

    assert problems == ["1h: 결측 캔들 2봉"]


def test_날짜가_바뀌면_일일_기준점_이동(tmp_path):
    e = engine(tmp_path)
    e.bot_state.balance = 900.0
    e.bot_state.day_key = "2000-01-01"
    sent = []
    e.notifier.text = lambda message: sent.append(message) or True

    e._roll_day()

    assert e.bot_state.day_start_balance == pytest.approx(900.0)
    assert e.bot_state.day_key != "2000-01-01"
    assert sent and "생존 확인" in sent[0]


def test_전략이_0개여도_회차가_돈다(tmp_path, monkeypatch):
    """루프 안정성을 전략과 분리해 확인한다."""

    e = engine(tmp_path)
    monkeypatch.setattr(e, "fetch", lambda spec: None)      # 수집 실패 상황
    monkeypatch.setattr(st, "save", lambda s, p=None: None)

    e.run_once()        # 예외가 나지 않아야 한다


def test_대기_중_중단도_상태를_저장하고_정상_종료(tmp_path, monkeypatch):
    """systemd 재시작은 대기 구간에 도착해도 traceback을 남기지 않는다."""

    e = engine(tmp_path)
    saved = []
    monkeypatch.setattr(e, "run_once", lambda: None)
    monkeypatch.setattr(st, "save", lambda state, p=None: saved.append(state.to_dict()))

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(loop_mod.time, "sleep", interrupt)

    e.run(max_iterations=2)

    assert saved


def test_체결_방식이_CSV에_남는다(tmp_path):
    """note가 없으면 로그만으로 체결 품질을 진단할 수 없다.

    손절이 갭으로 계획보다 나쁘게 체결됐는지 알 방법이 사라진다 —
    실제로 리플레이 분석 중 이 이유로 원인 추적이 막혔다.
    """

    log = logs.TradeLog(tmp_path / "trades.csv")
    log.write(strategy="x", symbol=XRP, side="long", action="exit",
              reason="stop", price=1.0, note="갭 손절")

    row = logs.read_trades(tmp_path / "trades.csv")[0]
    assert "note" in row
    assert row["note"] == "갭 손절"


def test_체결_기록은_시장_시각을_쓴다(tmp_path):
    """실행 시각을 쓰면 리플레이에서 180일치가 몇 초 안에 찍혀
    보유 시간을 복원할 수 없다. 체결 기록은 감사 자료다."""

    log = logs.TradeLog(tmp_path / "trades.csv")
    log.write(market_ts=1_700_000_000, strategy="x", symbol=XRP,
              side="long", action="entry", price=1.0)

    row = logs.read_trades(tmp_path / "trades.csv")[0]
    assert int(row["ts"]) == 1_700_000_000
    assert row["dt"].startswith("2023-11-14")


def test_시장_시각을_안_주면_현재_시각(tmp_path):
    import time

    log = logs.TradeLog(tmp_path / "trades.csv")
    log.write(strategy="x", symbol=XRP, side="long", action="entry", price=1.0)

    row = logs.read_trades(tmp_path / "trades.csv")[0]
    assert abs(int(row["ts"]) - time.time()) < 10


# ------------------------------------------------------------------ 알림 폭주


def test_같은_경고는_한_번만_나간다():
    """루프 오류는 회차마다 발생한다. 폴링 10초면 지속 오류 하나가
    시간당 360건, 하루 8,640건을 만든다."""

    sent = []
    n = Notifier(url="http://x/y")
    n._client = type("C", (), {"post": lambda *a, **k: type("R", (), {"status_code": 204, "text": ""})()})()

    for _ in range(100):
        if n.warn("루프 오류", "ConnectionError: boom"):
            sent.append(1)

    assert len(sent) == 1, f"{len(sent)}건 전송됨 — 중복 억제가 안 된다"


def test_억제된_건수를_다음_알림에_알려준다():
    """조용히 버리면 오류가 계속되는지 알 수 없다."""

    posted = []

    class FakeResp:
        status_code = 204
        text = ""

    n = Notifier(url="http://x/y", dedupe_seconds=0.0)
    n._client = type("C", (), {"post": lambda s, u, json: posted.append(json) or FakeResp()})()

    n.warn("루프 오류", "boom")
    n.dedupe_seconds = 900.0
    for _ in range(5):
        n.warn("루프 오류", "boom")
    n.dedupe_seconds = 0.0
    n.warn("루프 오류", "boom")

    last = posted[-1]["embeds"][0].get("description", "")
    assert "생략" in last, "억제된 건수가 안 보인다"


def test_채널당_분당_상한():
    """중복이 아니어도 폭주를 막는 마지막 방어선."""

    class FakeResp:
        status_code = 204
        text = ""

    n = Notifier(url="http://x/y")
    n._client = type("C", (), {"post": lambda s, u, json: FakeResp()})()

    ok = sum(1 for i in range(50) if n.text(f"서로 다른 메시지 {i}"))
    assert ok == n.max_per_minute, f"{ok}건 전송 — 상한 {n.max_per_minute} 초과"


def test_다른_제목_경고는_따로_센다():
    class FakeResp:
        status_code = 204
        text = ""

    n = Notifier(url="http://x/y")
    n._client = type("C", (), {"post": lambda s, u, json: FakeResp()})()

    assert n.warn("수집 실패", "a")
    assert n.warn("루프 오류", "b")        # 다른 제목이면 나간다
    assert not n.warn("루프 오류", "c")    # 같은 제목이면 억제
