"""EXP-015의 실시간·무주문 페이퍼 테스트.

50x/100x × 두 진입 규칙의 네 독립 가상 계좌를 같은 공개 Gate XRP_USDT 시세로
관측한다. ``continuous``는 완료 뒤 즉시 재진입하고, ``squeeze_breakout``은 확정
1분봉의 변동성 축소 후 범위 이탈에서만 새 pair를 연다. 어떤 개인 API·주문 API도 호출하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from config import env
from config.settings import (
    CONTRACT_SPECS,
    DISCORD_XRP_DUAL_HEDGE_50_CONTINUOUS_WEBHOOK_ENV,
    DISCORD_XRP_DUAL_HEDGE_50_SQUEEZE_WEBHOOK_ENV,
    DISCORD_XRP_DUAL_HEDGE_100_CONTINUOUS_WEBHOOK_ENV,
    DISCORD_XRP_DUAL_HEDGE_100_SQUEEZE_WEBHOOK_ENV,
    DUAL_HEDGE,
    RUNTIME,
)
from core import live
from core.exchange import ExchangeError, FuturesContractMarket, GateClient
from engine.broker import Fill
from engine.dual_hedge import (
    DualHedgeReplay, HedgeLeg, HedgePair, PairResult, config_with,
)
from indicators.atr import atr
from reporting.logs import SignalLog
from reporting.notify import COLOR_ENTRY, COLOR_LOSS, COLOR_WARN, COLOR_WIN, Notifier, code_revision
from strategies.xrp_dual_hedge import squeeze_breakout_ready


class PaperSession:
    def __init__(
        self,
        strategy: str,
        leverage: int,
        poll_seconds: float,
        balance: float,
        webhook_env: str,
        event_path: Path,
        liq_stop_buffer: float | None = None,
    ):
        if strategy not in ("continuous", "squeeze_breakout"):
            raise ValueError(f"알 수 없는 진입 전략: {strategy}")
        if leverage not in (50, 100):
            raise ValueError(f"EXP-015 허용 레버리지는 50x/100x뿐입니다: {leverage}")
        if poll_seconds <= 0:
            raise ValueError("폴링 간격은 0보다 커야 합니다")

        self.strategy = strategy
        self.leverage = leverage
        self.poll_seconds = poll_seconds
        self.tag = f"{leverage}x-{strategy}"
        self._next_observation_at = 0.0
        self.engine = DualHedgeReplay(
            balance=balance,
            config=config_with(leverage=leverage, liquidation_stop_buffer=liq_stop_buffer),
        )
        self.log = SignalLog(event_path)
        self.state_path = event_path.with_name("state.json")
        # 전략별 웹훅은 진입·청산 등 원본 이벤트를 계속 받는다. 별도로 기존
        # 봇의 system/balance 채널에도 네 실험을 각각 독립된 상태·잔고 알림으로
        # 보낸다. 네 세션을 한 메시지에 묶으면 개별 실험의 중단·지연을 놓치기 쉽다.
        self.notifier = Notifier(url=env.get(webhook_env) or "", account=f"dual-{self.tag}")
        self.status_notifier = Notifier(account=f"dual-{self.tag}")
        self.prev_bar_ts = 0
        self.prev_high = 0.0
        self.prev_low = 0.0
        self.prev_price = 0.0
        self.prev_mark_price = 0.0
        self.last_squeeze_signal_ts = 0
        self.last_status_day = ""
        self._load_state()

    def due(self, monotonic_now: float) -> bool:
        """이 세션이 다음 공개 시세를 처리할 시점인지."""

        if monotonic_now < self._next_observation_at:
            return False
        self._next_observation_at = monotonic_now + self.poll_seconds
        return True

    def _load_state(self) -> None:
        """프로세스 재시작 뒤에도 가상 계좌를 이어 간다.

        다음 관측에서 중단 중 확정된 봉을 순서대로 보수 처리한다. 다만 그 사이에
        새 pair를 소급 진입시키지는 않아 재시작이 수익/손실 회피 수단이 되지 않는다.
        """

        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            session = data["session"]
            expected_session = {
                "strategy": self.strategy,
                "leverage": self.leverage,
                "poll_seconds": self.poll_seconds,
                "liq_stop_buffer": self.engine.config.liquidation_stop_buffer,
            }
            if session != expected_session:
                raise RuntimeError(
                    f"[{self.tag}] 다른 실험 상태 파일을 복원할 수 없습니다: {self.state_path}"
                )
            engine = data["engine"]
            self.engine.broker.balance = float(engine["balance"])
            self.engine.results = [PairResult(**row) for row in engine["results"]]
            self.engine._next_pair_id = int(engine["next_pair_id"])
            self.engine._eligible_live_ts = int(engine["eligible_live_ts"])
            pair_data = engine.get("pair")
            if pair_data:
                def leg(row):
                    return HedgeLeg(
                        side=row["side"], fill=Fill(**row["fill"]),
                        liquidation_price=float(row["liquidation_price"]),
                        margin=float(row["margin"]),
                        stop_price=row.get("stop_price"),
                    )
                long, short = leg(pair_data["long"]), leg(pair_data["short"])
                survivor_side = pair_data.get("survivor_side")
                self.engine.pair = HedgePair(
                    pair_id=int(pair_data["pair_id"]), opened_ts=int(pair_data["opened_ts"]),
                    opened_balance=float(pair_data["opened_balance"]), contracts=int(pair_data["contracts"]),
                    long=long, short=short,
                    survivor={"long": long, "short": short}.get(survivor_side),
                    liquidated_side=pair_data.get("liquidated_side"),
                    liquidated_ts=pair_data.get("liquidated_ts"),
                    profit_lock_price=pair_data.get("profit_lock_price"),
                    stop=pair_data.get("stop"), extreme=pair_data.get("extreme"),
                    armed=bool(pair_data.get("armed", False)),
                )
            observer = data.get("observer", {})
            self.prev_bar_ts = int(observer.get("prev_bar_ts", 0))
            self.prev_high = float(observer.get("prev_high", 0.0))
            self.prev_low = float(observer.get("prev_low", 0.0))
            self.prev_price = float(observer.get("prev_price", 0.0))
            self.prev_mark_price = float(observer.get("prev_mark_price", 0.0))
            self.last_squeeze_signal_ts = int(observer.get("last_squeeze_signal_ts", 0))
            self.last_status_day = str(observer.get("last_status_day", ""))
            print(f"[{self.tag}] 이전 페이퍼 상태를 복원했습니다.")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"[{self.tag}] 상태 파일이 손상되었습니다: {self.state_path}") from exc

    def _save_state(self) -> None:
        pair = self.engine.pair
        pair_data = asdict(pair) if pair else None
        if pair_data is not None:
            pair_data["survivor_side"] = pair.survivor.side if pair.survivor else None
        data = {
            "version": 1,
            "session": {
                "strategy": self.strategy,
                "leverage": self.leverage,
                "poll_seconds": self.poll_seconds,
                "liq_stop_buffer": self.engine.config.liquidation_stop_buffer,
            },
            "engine": {
                "balance": self.engine.broker.balance,
                "results": [asdict(row) for row in self.engine.results],
                "next_pair_id": self.engine._next_pair_id,
                "eligible_live_ts": self.engine._eligible_live_ts,
                "pair": pair_data,
            },
            "observer": {
                "prev_bar_ts": self.prev_bar_ts, "prev_high": self.prev_high,
                "prev_low": self.prev_low, "prev_price": self.prev_price,
                "prev_mark_price": self.prev_mark_price,
                "last_squeeze_signal_ts": self.last_squeeze_signal_ts,
                "last_status_day": self.last_status_day,
            },
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.state_path)

    @staticmethod
    def _utc_day(ts: int) -> str:
        return datetime.fromtimestamp(ts, UTC).date().isoformat()

    def _status_fields(self) -> dict[str, str]:
        """기존 봇 상태 채널에 보낼 이 세션 단독 상태."""

        pair = self.engine.pair
        state = "대기"
        pair_id = "없음"
        lock = "-"
        stop = "-"
        if pair is not None:
            pair_id = str(pair.pair_id)
            if pair.survivor is None:
                state = "양방향 보유"
            else:
                state = f"{pair.survivor.side} 생존 · {'트레일 중' if pair.armed else '이익보장가 대기'}"
                lock = f"{pair.profit_lock_price:g}" if pair.profit_lock_price is not None else "-"
                stop = f"{pair.stop:g}" if pair.stop is not None else "-"
        return {
            "상태": state,
            "pair": pair_id,
            "현재 잔고": f"{self.engine.broker.balance:,.2f} USDT",
            "완료 pair": str(len(self.engine.results)),
            "이익보장가": lock,
            "트레일 스탑": stop,
            "커밋": code_revision(),
        }

    def _send_status(self, label: str, ts: int) -> None:
        self.status_notifier.embed(
            title=f"XRP dual hedge / {self.tag} {label}",
            fields=self._status_fields(), channel="system",
            key=f"dual-status:{self.tag}:{label}:{self._utc_day(ts)}",
        )

    def announce_started(self, now_ts: int) -> None:
        """서비스 시작은 네 세션 각각을 기존 봇 상태 채널에 알린다."""

        self.last_status_day = self._utc_day(now_ts)
        self._save_state()
        self._send_status("시작", now_ts)

    def _announce_daily_status(self, now_ts: int) -> None:
        day = self._utc_day(now_ts)
        if day == self.last_status_day:
            return
        self.last_status_day = day
        self._save_state()
        self._send_status("생존 확인", now_ts)

    def _send_balance_change(
        self, event: str, before: float, pair_id: int | None,
    ) -> None:
        """각 세션을 묶지 않은 채 기존 잔고 채널에 비용 변화를 알린다."""

        after = self.engine.broker.balance
        change = after - before
        return_pct = (after / self.engine.initial_balance - 1) * 100
        self.status_notifier.embed(
            title=f"XRP dual hedge / {self.tag} 잔고 변화 · {event}",
            fields={
                "pair": str(pair_id) if pair_id is not None else "-",
                "이전 잔고": f"{before:,.2f} USDT",
                "잔고 증감": f"{change:+,.4f} USDT",
                "현재 잔고": f"{after:,.2f} USDT",
                "누적 수익률": f"{return_pct:+.2f}%",
            },
            channel="balance",
            key=f"dual-balance:{self.tag}:{event}:{pair_id}",
        )

    def _event_embed(
        self, event: str, market: FuturesContractMarket,
    ) -> tuple[str, dict[str, str], int]:
        """사람이 읽는 Discord 알림. 기계용 원본 값은 이벤트 로그에 남긴다."""

        def price(value: float | None) -> str:
            return "-" if value is None else f"{value:.4f}"

        def leg_line(leg) -> str:
            stop = f"스탑 {price(leg.stop_price)} · " if leg.stop_price is not None else ""
            return f"진입 {price(leg.fill.price)}\n{stop}청산 {price(leg.liquidation_price)}"

        side_ko = {"long": "롱", "short": "숏"}
        balance = self.engine.broker.balance
        pair = self.engine.pair
        fields: dict[str, str] = {}
        color = COLOR_WARN

        if event == "entry" and pair is not None:
            title = f"진입 #{pair.pair_id} · 롱+숏 {pair.contracts}계약"
            fields = {"롱": leg_line(pair.long), "숏": leg_line(pair.short)}
            color = COLOR_ENTRY
        elif event in ("liquidation", "loser_stop") and pair is not None and pair.survivor is not None:
            how = "청산" if event == "liquidation" else "스탑"
            title = f"{side_ko[pair.liquidated_side]} {how} · {side_ko[pair.survivor.side]} 생존 #{pair.pair_id}"
            fields = {
                "손익분기가": price(pair.profit_lock_price),
                f"{side_ko[pair.survivor.side]} 손절": price(
                    pair.survivor.stop_price or pair.survivor.liquidation_price
                ),
                "확정 손익 (생존 다리 제외)": f"{balance - pair.opened_balance:+,.2f} USDT",
                "Mark Price": price(market.mark_price),
            }
        elif event in ("profit_lock_armed", "trail_updated") and pair is not None and pair.survivor is not None:
            label = "손익분기 도달 · 본전 스탑 설정" if event == "profit_lock_armed" else "트레일 스탑 이동"
            title = f"{label} #{pair.pair_id}"
            fields = {
                "생존 다리": side_ko[pair.survivor.side],
                "트레일 스탑": price(pair.stop),
                "손익분기가": price(pair.profit_lock_price),
                "최고/최저가": price(pair.extreme),
            }
            color = COLOR_WIN
        else:
            labels = {
                "profit_lock_trail": "트레일 익절 종료",
                "survivor_stop": "생존 다리도 손절",
                "survivor_liquidation": "생존 다리도 청산",
                "survivor_timeout": "시간 종료",
                "double_stop": "양쪽 동시 손절",
                "double_liquidation": "양쪽 동시 청산",
            }
            result = self.engine.results[-1] if pair is None and self.engine.results else None
            title = labels.get(event, event) + (f" #{result.pair_id}" if result else "")
            if result is not None:
                fields = {"pair 손익": f"{result.pnl:+,.2f} USDT"}
                color = COLOR_WIN if result.pnl >= 0 else COLOR_LOSS

        fields["잔고"] = f"{balance:,.2f} USDT ({(balance / self.engine.initial_balance - 1) * 100:+.2f}%)"
        return title, fields, color

    def _allow_entry(self, closed) -> bool:
        if self.strategy == "continuous":
            return True

        if closed.empty:
            return False
        signal_ts = int(closed["ts"].iloc[-1])
        if signal_ts == self.last_squeeze_signal_ts:
            return False
        self.last_squeeze_signal_ts = signal_ts
        return squeeze_breakout_ready(closed, self.engine.config)

    def _process(
        self,
        window: live.TickWindow,
        mark_window: live.TickWindow,
        market: FuturesContractMarket,
        first_tier_risk_limit: float | None,
        market_ts: int,
        atr_value: float,
        allow_entry: bool,
        state_after: tuple[int, float, float, float],
    ) -> None:
        before_balance = self.engine.broker.balance
        pair_id_before = self.engine.pair.pair_id if self.engine.pair else None
        event = self.engine.process_live_window(
            ts=market_ts, open_=window.open, high=window.high, low=window.low,
            last=window.last, atr=atr_value, allow_entry=allow_entry,
            mark_open=mark_window.open, mark_high=mark_window.high,
            mark_low=mark_window.low, mark_last=mark_window.last,
            maintenance_rate=market.maintenance_rate,
            first_tier_risk_limit=first_tier_risk_limit,
        )
        if event is None:
            return
        # 상태가 바뀐 사실을 이벤트 로그·Discord보다 먼저 원자적으로 남긴다.
        # 알림 전송 중 프로세스가 끝나도 같은 pair가 이전 상태로 되돌아가
        # 재처리되지 않게 하는 순서다.
        self.prev_bar_ts, self.prev_high, self.prev_low, self.prev_price = state_after
        self._save_state()
        pair = self.engine.pair
        fields = {
            "event": event,
            "balance": round(self.engine.broker.balance, 6),
            "pair_id": pair.pair_id if pair else None,
            "profit_lock_price": pair.profit_lock_price if pair else None,
            "stop": pair.stop if pair else None,
            "last_pair_pnl": round(self.engine.results[-1].pnl, 6) if pair is None and self.engine.results else None,
            "mark_price": market.mark_price,
            "maintenance_rate": market.maintenance_rate,
        }
        self.log.write(
            action="event", strategy=f"xrp_dual_hedge.{self.tag}",
            symbol=DUAL_HEDGE.symbol, side=pair.survivor.side if pair and pair.survivor else "-",
            market_ts=market_ts, **fields,
        )
        print(f"[{self.tag}] {event} {fields}")
        title, embed_fields, color = self._event_embed(event, market)
        self.notifier.embed(
            title=title, fields=embed_fields, color=color, channel="system",
            key=f"{self.tag}:{event}:{pair.pair_id if pair else market_ts}",
        )
        # trail_updated/profit_lock_armed는 현금 잔고를 바꾸지 않는다. 1초 폴링
        # 중 balance 채널을 채우지 않도록 실제 잔고 변화가 있는 사건만 보낸다.
        if event in {
            "entry", "liquidation", "double_liquidation", "survivor_liquidation",
            "loser_stop", "double_stop", "survivor_stop",
            "profit_lock_trail", "survivor_timeout",
        }:
            self._send_balance_change(event, before_balance, pair_id_before or (pair.pair_id if pair else None))

    def observe(
        self,
        quote: live.LiveQuote,
        market: FuturesContractMarket,
        closed,
        first_tier_risk_limit: float | None = None,
    ) -> None:
        series = atr(closed, self.engine.config.atr_period)
        atr_value = float(series.iloc[-1]) if not series.empty else 0.0

        # 봉 전환 사이의 거래가 구간(직전 봉 tail, 중단 중 마감된 봉)은 스탑·트레일
        # 판정에서 빠뜨리지 않는다. 이 구간의 Mark Price 경로는 알 수 없으므로 직전
        # 관측 Mark Price로 고정해 새 청산을 지어내지 않는다. 새 pair는 소급 진입하지 않는다.
        if self.prev_price > 0 and quote.bar_ts > self.prev_bar_ts:
            prior = closed[closed["ts"] == self.prev_bar_ts]
            if prior.empty:
                raise RuntimeError(
                    f"[{self.tag}] 마지막 관측({self.prev_bar_ts}) 이후의 봉 이력이 부족합니다. "
                    "열린 pair를 추정으로 이어 가지 않고 수동 점검이 필요합니다."
                )
            held_mark = self.prev_mark_price if self.prev_mark_price > 0 else market.mark_price
            flat_mark = live.TickWindow(open=held_mark, high=held_mark, low=held_mark, last=held_mark)
            gaps = [(
                live.tail_window(prior.iloc[-1], self.prev_high, self.prev_low, self.prev_price),
                prior.iloc[-1],
            )]
            for _, row in closed[closed["ts"] > self.prev_bar_ts].iterrows():
                gaps.append((
                    live.TickWindow(
                        open=float(row["open"]), high=float(row["high"]),
                        low=float(row["low"]), last=float(row["close"]),
                    ),
                    row,
                ))
            for gap, row in gaps:
                # 그 봉을 포함해 새로 계산한 ATR을 소급 사용하지 않는다.
                gap_series = atr(closed[closed["ts"] < row["ts"]], self.engine.config.atr_period)
                gap_atr = float(gap_series.iloc[-1]) if not gap_series.empty else 0.0
                state = (int(row["ts"]), float(row["high"]), float(row["low"]), float(row["close"]))
                self._process(
                    gap, flat_mark, market, first_tier_risk_limit,
                    int(row["ts"]) + 60, gap_atr, allow_entry=False, state_after=state,
                )
                self.prev_bar_ts, self.prev_high, self.prev_low, self.prev_price = state

        window = live.window_since(
            quote, self.prev_bar_ts, self.prev_high, self.prev_low, self.prev_price,
        )
        # Gate의 REST Mark Price는 순간 스냅샷이다. 지난 1분봉의 거래가 고저로
        # Mark Price를 소급 복원하면 실제와 다른 청산을 만들 수 있어, 직전/현재
        # 관측값 사이의 확인 가능한 경로만 청산 판정에 사용한다.
        mark_open = self.prev_mark_price if self.prev_mark_price > 0 else market.mark_price
        mark_window = live.TickWindow(
            open=mark_open,
            high=max(mark_open, market.mark_price),
            low=min(mark_open, market.mark_price),
            last=market.mark_price,
        )
        self._process(
            window, mark_window, market, first_tier_risk_limit,
            quote.observed_ts, atr_value, self._allow_entry(closed),
            state_after=(quote.bar_ts, quote.bar_high, quote.bar_low, quote.last),
        )
        self.prev_bar_ts = quote.bar_ts
        self.prev_high = quote.bar_high
        self.prev_low = quote.bar_low
        self.prev_price = quote.last
        self.prev_mark_price = market.mark_price
        self._save_state()
        self._announce_daily_status(quote.observed_ts)


def build_sessions(
    *,
    balance: float,
    poll_50_seconds: float,
    poll_100_seconds: float,
    log_dir: Path,
    liq_stop_buffer: float | None = None,
) -> tuple[PaperSession, ...]:
    """50x/100x × 두 진입 전략의 상태·로그가 섞이지 않게 세션을 만든다."""

    return (
        PaperSession(
            "continuous", 50, poll_50_seconds, balance,
            DISCORD_XRP_DUAL_HEDGE_50_CONTINUOUS_WEBHOOK_ENV,
            log_dir / "50x" / "continuous" / "events.jsonl", liq_stop_buffer,
        ),
        PaperSession(
            "squeeze_breakout", 50, poll_50_seconds, balance,
            DISCORD_XRP_DUAL_HEDGE_50_SQUEEZE_WEBHOOK_ENV,
            log_dir / "50x" / "squeeze_breakout" / "events.jsonl", liq_stop_buffer,
        ),
        PaperSession(
            "continuous", 100, poll_100_seconds, balance,
            DISCORD_XRP_DUAL_HEDGE_100_CONTINUOUS_WEBHOOK_ENV,
            log_dir / "100x" / "continuous" / "events.jsonl", liq_stop_buffer,
        ),
        PaperSession(
            "squeeze_breakout", 100, poll_100_seconds, balance,
            DISCORD_XRP_DUAL_HEDGE_100_SQUEEZE_WEBHOOK_ENV,
            log_dir / "100x" / "squeeze_breakout" / "events.jsonl", liq_stop_buffer,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=60.0, help="실행 시간. 0이면 Ctrl-C까지")
    parser.add_argument("--poll-50-seconds", type=float, default=5.0)
    parser.add_argument("--poll-100-seconds", type=float, default=1.0)
    parser.add_argument("--balance", type=float, default=RUNTIME.initial_balance)
    parser.add_argument("--log-dir", type=Path, default=Path("data/xrp_dual_hedge_mark_live"))
    parser.add_argument(
        "--liq-stop-buffer", type=float, default=DUAL_HEDGE.liquidation_stop_buffer,
        help="청산가 직전 보호 스탑 비율 (예: 0.001 = 청산가의 0.1%%). 생략하면 청산까지 보유",
    )
    args = parser.parse_args()
    if (
        args.minutes < 0 or args.poll_50_seconds <= 0
        or args.poll_100_seconds <= 0 or args.balance <= 0
    ):
        raise SystemExit("minutes는 0 이상, 폴링 간격과 balance는 0보다 커야 한다")

    sessions = build_sessions(
        balance=args.balance,
        poll_50_seconds=args.poll_50_seconds,
        poll_100_seconds=args.poll_100_seconds,
        log_dir=args.log_dir,
        liq_stop_buffer=args.liq_stop_buffer,
    )
    deadline = time.monotonic() + args.minutes * 60 if args.minutes else None
    print(
        "실주문 없음 — XRP_USDT 공개 거래가·Mark Price로 "
        f"50x({args.poll_50_seconds:g}초)·100x({args.poll_100_seconds:g}초) 페어를 페이퍼 관측합니다."
    )

    now = int(time.time())
    for session in sessions:
        session.announce_started(now)

    with GateClient() as client:
        try:
            first_tier = client.fetch_first_risk_limit_tier(DUAL_HEDGE.symbol)
            while deadline is None or time.monotonic() < deadline:
                now = int(time.time())
                try:
                    result = client.fetch(DUAL_HEDGE.symbol, "1m", limit=200, drop_unclosed=False, now_ts=now)
                    # 요청 전에 잰 시각은 분 경계에서 응답의 새 봉보다 늦지 않다. 응답 뒤에 다시 잰다.
                    now = int(time.time())
                    market = client.fetch_contract_market(DUAL_HEDGE.symbol)
                    expected_multiplier = float(CONTRACT_SPECS[DUAL_HEDGE.symbol]["multiplier"])
                    if market.multiplier != expected_multiplier:
                        raise ExchangeError(
                            "Gate 계약승수가 로컬 계약 스펙과 다릅니다: "
                            f"Gate={market.multiplier:g}, local={expected_multiplier:g}"
                        )
                    if abs(market.maintenance_rate - first_tier.maintenance_rate) > 1e-12:
                        # Gate가 구간 값을 바꿨을 수 있다. 시작 시 받은 표를 고집하면
                        # 재시작 전까지 모든 세션이 멈추므로 한 번 다시 받아 비교한다.
                        first_tier = client.fetch_first_risk_limit_tier(DUAL_HEDGE.symbol)
                    if abs(market.maintenance_rate - first_tier.maintenance_rate) > 1e-12:
                        raise ExchangeError(
                            "Gate 계약 정보와 첫 risk-limit MMR이 다릅니다: "
                            f"contract={market.maintenance_rate:g}, "
                            f"tier={first_tier.maintenance_rate:g}"
                        )
                    if result.problems:
                        print(f"[feed] 캔들 문제: {result.problems}")
                    if live.is_feed_lagging(result.df, "1m", now):
                        print("[feed] 확정봉이 3분 넘게 지연되어 이번 관측은 건너뜁니다.")
                        time.sleep(min(args.poll_50_seconds, args.poll_100_seconds))
                        continue
                    quote = live.from_frame(result.df, "1m", now_ts=now)
                    if quote is not None:
                        # 진행 중인 봉 이전의 봉은 모두 확정봉이다. 시각으로 자르면 분 경계에서
                        # 막 마감된 봉이 빠져 복구 이력 부족으로 프로세스가 죽는다.
                        closed = result.df[result.df["ts"] < quote.bar_ts].reset_index(drop=True)
                        monotonic_now = time.monotonic()
                        for session in sessions:
                            if session.due(monotonic_now):
                                session.observe(
                                    quote, market, closed,
                                    first_tier_risk_limit=first_tier.risk_limit,
                                )
                except ExchangeError as exc:
                    print(f"[feed] 수집 실패: {exc}")
                # 시세는 가장 촘촘한 100x 세션 간격으로 한 번만 받는다.
                time.sleep(min(args.poll_50_seconds, args.poll_100_seconds))
        except KeyboardInterrupt:
            print("사용자 중단")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
