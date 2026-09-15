"""메인 루프.

수집 -> 검증 -> 준비 -> 신호 -> 리스크 -> 체결 -> 기록.

전략이 0개여도 돌아야 한다. 그래야 루프 자체의 안정성을 전략과 분리해
확인할 수 있다.

데이터 정합성 규칙(비대칭):
  problems가 있으면 **신규 진입을 차단**하고 **기존 포지션 청산은 허용**한다.
  데이터가 의심스럽다고 포지션을 방치하는 게 더 위험하기 때문이다.
  core.candles.validate()는 결과를 돌려줄 뿐 강제하지 않으므로 여기서 강제한다.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.settings import ENTRY, RISK, RUNTIME, RiskConfig
from config.symbols import ACTIVE, SymbolSpec
from core import candles as cd
from core import cache
from core import live
from core.exchange import (
    ExchangeError,
    FetchResult,
    GateClient,
    fetch_open_interest,
)
from engine import state as st
from engine.broker import Broker
from reporting.logs import ACTION_ENTRY, ACTION_SKIP, SignalLog, TradeLog
from reporting.notify import Notifier
from risk import exits as ex
from risk import limits as lim

SKIP_STALE_DATA = "stale_data"

# 트랜치 추가 진입을 막는 사유.
SKIP_MAX_TRANCHES = "max_tranches_reached"
SKIP_OPPOSITE_OPEN = "opposite_position_open"
SKIP_SYMBOL_EXCLUSIVE = "symbol_exclusive"
SKIP_DAILY_ENTRY_LIMIT = "daily_entry_limit"


@dataclass
class Engine:
    """가상거래 루프.

    strategies는 심볼별 전략 목록. 비어 있어도 동작한다.
    """

    strategies: dict[str, list] = field(default_factory=dict)
    symbols: tuple[SymbolSpec, ...] = ACTIVE

    # 리스크 설정. 측정할 때는 MEASUREMENT_RISK를 넣어 계좌 단위
    # 브레이크를 끈다 — 켜 두면 전략들이 서로의 성과를 오염시킨다.
    risk: RiskConfig = RISK

    client: GateClient | None = None
    broker: Broker | None = None
    bot_state: st.BotState | None = None
    trade_log: TradeLog | None = None
    signal_log: SignalLog | None = None
    notifier: Notifier | None = None

    # 대기 중인 지정가 주문. 현재는 재시작하면 사라진다. BTC 주문은 최대
    # 12시간 남을 수 있으므로 실시간 장기 운용 전 영속화해야 한다.
    pending_signals: list = field(default_factory=list)

    # 과거 리플레이에서 시장가 신호를 신호봉 종가에 체결하면, 봉이 닫힌 뒤에야
    # 알 수 있는 종가를 정확히 잡는 낙관 편향이 생긴다. 실시간 근사 모드에서는
    # 신호를 잠시 보관했다가 다음 가장 짧은 봉의 시가로 체결한다.
    # 실시간 엔진 기본값은 False다. 실제 루프는 봉 마감 직후 현재가로 체결하며,
    # 리플레이만 이 옵션을 켜 과거에 실제로 주문 가능했던 첫 가격을 사용한다.
    defer_market_entries: bool = False
    pending_market_signals: list = field(default_factory=list)

    # 지정가 체결률. XRP 전략이 메이커 체결을 전제로 설계돼 있어
    # 이 숫자가 낮으면 손익분기 승률 가정 자체가 무너진다.
    limit_stats: dict = field(default_factory=lambda: {"filled": 0, "expired": 0})

    # 전략별 prepare 결과 캐시. (프레임 길이 -> Prepared)
    #
    # 실시간에서는 매 봉 프레임이 한 칸 자라므로 항상 재계산된다(정상).
    # 리플레이에서는 전체 프레임을 고정으로 주고 as_of만 옮기므로
    # 한 번만 계산된다. 이게 없으면 51,835봉 리플레이가 254시간이다.
    _prepared: dict = field(default_factory=dict)

    # OI 프레임 캐시. CSV를 봉마다 다시 읽지 않기 위한 것으로,
    # refresh_oi가 새로 저장하면 버린다.
    _oi_cache: dict = field(default_factory=dict)

    # 이번 회차의 봉 시각. 정지 만료 판정과 로그에 쓴다.
    _now_ts: int = 0

    # 이번 판단 시점(봉 시작이 아니라 마감 시각). 다중 시간봉 리플레이에서
    # 아직 마감되지 않은 상위봉을 고르지 않기 위한 기준이다.
    _decision_ts: int = 0

    # 다음 시가 체결 중인 신호가 만들어진 봉. 새 포지션이 그 과거 봉을 다시
    # 청산 판정에 사용하지 않도록 _open()에 전달하는 임시 문맥이다.
    _deferred_signal_bar_ts: int | None = None

    def __post_init__(self):
        self.bot_state = self.bot_state or st.load()
        self.client = self.client or GateClient()
        self.broker = self.broker or Broker(balance=self.bot_state.balance)
        self.trade_log = self.trade_log or TradeLog()
        self.signal_log = self.signal_log or SignalLog()
        self.notifier = self.notifier or Notifier()

    # ---------------------------------------------------------------- 수집

    def fetch(self, spec: SymbolSpec) -> dict[str, FetchResult] | None:
        """캔들 수집. 실패하면 None을 돌려주고 이번 회차를 건너뛴다."""

        try:
            return self.client.fetch_frames(spec, limit=RUNTIME.candle_limit)
        except ExchangeError as exc:
            print(f"[{spec.contract}] 수집 실패: {exc}")
            return None

    def data_problems(self, results: dict[str, FetchResult]) -> list[str]:
        """타임프레임별 정합성 문제를 모은다."""

        problems: list[str] = []
        for tf, result in results.items():
            for p in result.problems:
                problems.append(f"{tf}: {p}")
        return problems

    # ---------------------------------------------------------------- 회차

    def run_once(
        self,
        as_of: int | None = None,
        decision_ts: int | None = None,
    ) -> None:
        """한 회차.

        as_of는 진입봉의 위치 인덱스다. 실시간은 None(= 마지막 봉)이고
        리플레이가 과거 시점을 지정할 때만 쓴다. 명시 인자로 둔 이유는
        `len(df) - 1`이라는 암묵적 규칙이 리플레이에서 깨지기 때문이다.

        decision_ts는 여러 종목·시간봉을 하나의 시장 시계로 재생할 때 쓰는
        **봉 마감 시각**이다. 이 시각까지 실제로 마감된 봉만 선택한다.
        as_of와 동시에 넘길 수 없다.
        """

        if as_of is not None and decision_ts is not None:
            raise ValueError("as_of와 decision_ts는 동시에 지정할 수 없다")

        # 이번 회차에 받은 프레임을 종목별로 모아 둔다.
        # XRP 패턴이 BTC 방향을 참조하기 때문이다 — XRP는 BTC 베타라
        # BTC가 급락 중이면 XRP 롱 패턴은 성립하지 않는다.
        cycle: dict[str, dict[str, pd.DataFrame]] = {}

        for spec in self.symbols:
            results = self.fetch(spec)
            if results is None:
                continue

            problems = self.data_problems(results)
            frames = {tf: r.df for tf, r in results.items()}
            cycle[spec.contract] = frames

            entry_tf = spec.driver_tf
            df = frames.get(entry_tf)
            if df is None or df.empty:
                continue

            # **새로 마감된 봉이 있을 때만 한 회차를 처리한다.**
            #
            # 루프는 10초마다 도는데 1시간봉은 1시간에 한 번 마감된다.
            # 봉 하나를 360번 다시 평가하면 같은 신호로 360번 진입하게 된다.
            # 청산 쪽에는 이 검사가 있었지만 진입 쪽에는 없었다.
            if decision_ts is not None:
                at = _closed_index_at(df, entry_tf, decision_ts)
            else:
                at = len(df) - 1 if as_of is None else as_of
            if at < 0 or at >= len(df):
                continue

            bar_ts = int(df["ts"].iloc[at])
            decision_close_ts = (
                int(decision_ts)
                if decision_ts is not None
                else bar_ts + cd.interval_seconds(entry_tf)
            )
            self._now_ts = bar_ts
            self._decision_ts = decision_close_ts
            self._roll_day(decision_close_ts)
            self._release_pause(decision_close_ts)
            key = f"{spec.contract}_{entry_tf}"
            if self.bot_state.last_bar_ts.get(key) == bar_ts:
                continue

            # 순서를 지킨다.
            #   다음 시가 시장가 체결 -> 청산 판정 -> 이전 봉 지정가 체결
            #   -> 이번 봉으로 신호 생성
            # 다음 시가 체결은 현재 봉이 시작할 때 이미 포지션이 생겼으므로,
            # 현재 봉의 고저가로 손절/익절 판정을 받아야 청산을 누락하지 않는다.
            # 신호가 낸 주문은 placed_ts가 이번 봉이므로 다음 봉부터 체결된다.
            if as_of is None and decision_ts is None:
                self.refresh_oi(spec, bar_ts)
            self.process_deferred_markets(spec, frames, bar_ts, at)
            self.manage_positions(
                spec, frames, bar_ts, at, decision_ts=decision_close_ts
            )
            self.process_pending(spec, frames, bar_ts, at)

            if problems:
                self._block_entries(spec, problems)
            else:
                self.evaluate_entries(
                    spec,
                    self._with_context(spec, frames, cycle),
                    bar_ts,
                    at,
                    decision_ts=decision_close_ts,
                )

            self.bot_state.last_bar_ts[key] = bar_ts

        self._persist()

    def _with_context(
        self,
        spec: SymbolSpec,
        frames: dict[str, pd.DataFrame],
        cycle: dict[str, dict[str, pd.DataFrame]],
    ) -> dict[str, pd.DataFrame]:
        """교차 종목 프레임과 미결제약정을 덧붙인다.

        키를 "BTC_USDT:1h"처럼 종목명으로 접두하는 이유는, 타임프레임 키("5m")와
        섞이면 prepare_standard가 그걸 자기 종목 캔들로 착각하기 때문이다.

        OI는 CSV에서 읽으므로 캐시한다. 실시간에서는 5분에 한 번이라
        비용이 안 보이지만, 리플레이는 봉마다 호출되어 1,500봉에 11.7초가 든다.
        `refresh_oi`가 새 데이터를 저장하면 캐시를 버린다.
        """

        if spec.contract == "BTC_USDT":
            extra = {}
        else:
            btc = cycle.get("BTC_USDT", {}).get("1h")
            extra = {"BTC_USDT:1h": btc} if btc is not None and not btc.empty else {}

        entry_tf = spec.driver_tf
        key = (spec.contract, entry_tf)
        oi = self._oi_cache.get(key)
        if oi is None:
            oi = cache.load_oi(spec.contract, entry_tf)
            self._oi_cache[key] = oi
        if not oi.empty:
            extra["OI"] = oi

        return {**frames, **extra} if extra else frames

    def refresh_oi(self, spec: SymbolSpec, bar_ts: int) -> None:
        """미결제약정을 증분으로 받아 캐시에 덧붙인다.

        캔들과 달리 별도 엔드포인트라 따로 받아야 한다. 캐시만 읽으면
        기동 시점 이후로 갱신이 멈추고, OI 판정 항목이 조용히 죽는다.

        실패해도 루프를 막지 않는다. OI가 없으면 판정표에서 그 항목만
        빠지고 나머지는 그대로 동작하도록 돼 있다.
        """

        entry_tf = spec.driver_tf
        try:
            existing = cache.load_oi(spec.contract, entry_tf)
            last = int(existing["ts"].iloc[-1]) if not existing.empty else 0
            if last >= bar_ts:
                return          # 이미 최신

            step = cd.interval_seconds(entry_tf)
            start = last + step if last else bar_ts - 200 * step

            fresh = fetch_open_interest(
                self.client, spec.contract, entry_tf, start, bar_ts
            )
            if not fresh.empty:
                cache.save_oi(spec.contract, entry_tf, fresh)
                self._oi_cache.pop((spec.contract, entry_tf), None)
        except Exception as exc:            # noqa: BLE001 - 매매를 막지 않는다
            print(f"[{spec.contract}] OI 갱신 실패: {type(exc).__name__}: {exc}")

    def _block_entries(self, spec: SymbolSpec, problems: list[str]) -> None:
        """정합성 문제로 신규 진입을 막고 기록한다.

        조용히 넘어가지 않는다. 얼마나 자주 막혔는지 알아야
        수집 문제인지 거래소 문제인지 판단할 수 있다.
        """

        detail = "; ".join(problems[:5])
        print(f"[{spec.contract}] 신규 진입 차단 — {detail}")
        self.signal_log.write(
            ACTION_SKIP,
            strategy="-",
            symbol=spec.contract,
            side="-",
            skip_reason=SKIP_STALE_DATA,
            problems=problems,
        )

    def manage_positions(
        self,
        spec: SymbolSpec,
        frames: dict[str, pd.DataFrame],
        bar_ts: int | None = None,
        at: int | None = None,
        decision_ts: int | None = None,
    ) -> None:
        """보유 포지션의 청산 판정.

        새 봉 판정은 호출부(run_once)가 한다. 진입·청산·지정가 체결이
        같은 봉 경계를 공유해야 순서가 어긋나지 않기 때문이다.
        """

        # **포지션마다 자기 봉으로 진행한다.**
        # 하나로 묶으면 1시간봉 포지션이 5분마다 bars_held가 올라
        # 40봉 한도가 3.3시간 만에 차버린다(실측: double_top.1h가
        # 50건 중 47건 시간 손절).
        for position in list(self.bot_state.positions):
            if position.symbol != spec.contract:
                continue

            df = frames.get(position.entry_tf)
            if df is None or df.empty:
                continue

            if decision_ts is not None:
                idx = _closed_index_at(df, position.entry_tf, decision_ts)
            else:
                idx = _index_at(df, bar_ts) if bar_ts else len(df) - 1
            if idx < 0:
                continue

            ts = int(df["ts"].iloc[idx])
            if position.last_bar_ts == ts:
                continue            # 이 봉은 이미 반영했다
            position.last_bar_ts = ts

            self._refresh_strategy_target(spec, position, frames, idx)
            handled = self._advance(position, df.iloc[idx], self._atr_of(df, idx))
            if not handled:
                self._exit_on_strategy_invalidation(
                    spec, position, frames, df, idx
                )

    # ------------------------------------------------------------ 실행 시계

    def run_tick(self, now_ts: float | None = None) -> int:
        """10초 시계. 보유 포지션만 실시간 시세로 감시한다.

        지표를 다시 계산하지 않고 진입 판단도 하지 않는다. 여기서 하는 일은
        "이미 정해진 손절·목표에 가격이 닿았는가"뿐이다. 그래서 진행 중인
        봉을 써도 재현성이 깨지지 않는다 — 판단의 입력이 아니라 체결의
        입력이기 때문이다.

        봉 시계(run_once)가 봉 마감까지 기다리는 동안 방치되던 구간을 메운다.
        5분봉이면 최대 5분이던 반응 지연이 폴링 간격으로 줄어든다.

        반환값은 이번에 처리한 청산 건수.
        """

        if not self.bot_state.positions and not self.pending_signals:
            return 0
        if now_ts is None:
            now_ts = time.time()
        now_ts = int(now_ts)

        exits = 0
        changed = False

        for spec in self.symbols:
            # 포지션·대기 주문이 있는 타임프레임만 받는다. 틱에서는 지표를
            # 쓰지 않으므로 상위봉이나 문맥봉을 받을 이유가 없다.
            tfs = {
                p.entry_tf
                for p in self.bot_state.positions
                if p.symbol == spec.contract
            }
            tfs |= self._pending_timeframes(spec)
            for tf in sorted(tfs, key=cd.interval_seconds):
                snap = self._live_snapshot(spec, tf, now_ts)
                if snap is None:
                    continue
                quote, closed_row, atr = snap

                for position in list(self.bot_state.positions):
                    if position.symbol != spec.contract or position.entry_tf != tf:
                        continue
                    closed_out, moved = self._tick_position(
                        position, quote, closed_row, atr
                    )
                    exits += int(closed_out)
                    changed = changed or closed_out or moved

                # 순서는 run_once와 같다 — 청산 판정 뒤 지정가 체결.
                # 위에서 만든 포지션 목록 스냅샷에 새 체결분이 들어가지
                # 않으므로, 방금 체결된 포지션이 같은 틱에서 두 번
                # 평가되지 않는다.
                if self._fill_pending_on_tick(spec, tf, quote):
                    changed = True

        if changed:
            self._persist()
        return exits

    def _pending_timeframes(self, spec: SymbolSpec) -> set[str]:
        """이 종목의 대기 지정가 주문이 걸린 타임프레임."""

        out: set[str] = set()
        for item in self.pending_signals:
            if item["spec"].contract != spec.contract:
                continue
            strategy = _strategy_named(self, spec, item["signal"].strategy)
            tf = _entry_tf_of(spec, strategy)
            if tf:
                out.add(tf)
        return out

    def _fill_pending_on_tick(self, spec: SymbolSpec, tf: str, quote) -> int:
        """대기 중인 지정가 주문을 실시간 시세로 체결한다.

        봉 마감을 기다리면 지정가가 스친 뒤 되돌아간 구간을 통째로 놓친다.
        진입이 밀리면 전략이 노린 자리가 아닌 곳에서 잡히거나 아예 못 잡는다.

        포지션 감시와 달리 **누적 고저를 그대로 쓴다.** 지정가는 가격이
        고정이라 트레일처럼 움직이지 않으므로, 봉 시작 이후 언제 닿았든
        그 순간 체결됐을 것이기 때문이다. 이미 지나간 극값으로 잘못
        체결될 여지가 없다.

        "주문을 낸 봉으로는 체결하지 않는다"는 규칙도 자동으로 지켜진다.
        진행 중인 봉은 주문을 낸 봉보다 항상 뒤다.

        TTL 만료는 여기서 보지 않는다. 봉 단위로 세는 값이라 봉 시계
        (process_pending)에 남는다.
        """

        filled = 0
        for item in list(self.pending_signals):
            if item["spec"].contract != spec.contract:
                continue

            signal = item["signal"]
            strategy = _strategy_named(self, spec, signal.strategy)
            if _entry_tf_of(spec, strategy) != tf:
                continue
            if quote.bar_ts <= int(item["placed_ts"]):
                continue

            price = signal.entry
            touched = (
                quote.bar_low <= price
                if signal.side == "long"
                else quote.bar_high >= price
            )
            if not touched:
                continue

            self.pending_signals.remove(item)
            self._now_ts = int(quote.observed_ts)
            before = len(self.bot_state.positions)
            # **체결봉 손절 검사를 넘기지 않는다(fill_bar=None).**
            #
            # 봉 경로는 봉 안의 순서를 모르니 "체결봉이 손절까지 갔으면 즉시
            # 손절"로 보수적으로 처리한다. 틱 경로는 체결 시점을 정확히 알기
            # 때문에 같은 처리를 하면 **체결 전에 찍힌 저점**으로 손절시키게
            # 된다 — window_since가 막는 것과 같은 종류의 오류다.
            #
            # 누락 위험도 없다. 아래에서 관측 기준점을 지금으로 맞추므로
            # 체결 이후 구간은 다음 틱이 곧바로 판정한다.
            self._fill_limit(item, None, bar_ts=quote.bar_ts)
            filled += 1

            # 갓 생긴 포지션의 관측 기준점을 지금으로 맞춘다.
            # 안 하면 다음 틱의 구간이 봉 시작까지 거슬러 올라가, 체결 전
            # 가격 범위로 손절·트레일을 판정하게 된다.
            if len(self.bot_state.positions) > before:
                opened = self.bot_state.positions[-1]
                opened.ticked_bar_ts = quote.bar_ts
                opened.last_tick_ts = int(quote.observed_ts)
                opened.last_tick_high = quote.bar_high
                opened.last_tick_low = quote.bar_low
                opened.last_tick_price = quote.last

        return filled

    def _live_snapshot(self, spec: SymbolSpec, tf: str, now_ts: int):
        """진행 중인 봉 + 직전 확정봉 + ATR.

        ATR은 **확정봉으로만** 계산한다. 진행 중인 봉을 넣으면 같은 시점을
        다시 계산할 때 값이 달라져 트레일 폭이 재현되지 않는다.
        """

        try:
            result = self.client.fetch(
                spec.contract,
                tf,
                limit=RUNTIME.candle_limit,
                drop_unclosed=False,
                now_ts=now_ts,
            )
        except ExchangeError as exc:
            print(f"[{spec.contract}] {tf} 실시간 수집 실패: {exc}")
            return None

        df = result.df
        quote = live.from_frame(df, tf, now_ts=now_ts)
        if quote is None:
            return None         # 진행 중인 봉이 아직 없다. 봉 시계가 처리한다.

        if live.is_feed_lagging(df, tf, now_ts):
            # 멈춘 시세로 손절을 판정하면 실제와 다른 가격에서 청산된다.
            print(f"[{spec.contract}] {tf} 시세 피드 지연. 이번 틱 건너뜀")
            return None

        closed = df.iloc[:-1]
        if len(closed) < 2:
            return None

        return quote, closed.iloc[-1], self._atr_of(closed, -1)

    def _tick_position(
        self,
        position: st.OpenPosition,
        quote: live.LiveQuote,
        closed_row,
        atr: float,
    ) -> tuple[bool, bool]:
        """포지션 하나에 실시간 시세를 반영한다. (청산됨, 상태가 바뀜)"""

        before_stop = position.stop

        # 봉이 넘어갔으면 직전 봉의 남은 구간을 먼저 처리한다.
        # 폴링 간격 때문에 봉의 마지막 몇 초는 어떤 틱도 보지 못하는데,
        # 이걸 버리면 봉 끝에서 손절이 닿아도 놓친다.
        rolled = (
            position.ticked_bar_ts
            and position.ticked_bar_ts != quote.bar_ts
            and closed_row is not None
            and int(closed_row["ts"]) == position.ticked_bar_ts
        )
        if rolled:
            tail = live.tail_window(
                closed_row,
                position.last_tick_high,
                position.last_tick_low,
                position.last_tick_price,
            )
            closed_out = self._apply_tick(position, tail, atr, quote.observed_ts)
            position.covered_bar_ts = position.ticked_bar_ts
            if closed_out:
                return True, True

        window = live.window_since(
            quote,
            position.ticked_bar_ts,
            position.last_tick_high,
            position.last_tick_low,
            position.last_tick_price,
        )
        closed_out = self._apply_tick(position, window, atr, quote.observed_ts)
        if closed_out:
            return True, True

        position.ticked_bar_ts = quote.bar_ts
        position.last_tick_ts = quote.observed_ts
        position.last_tick_high = quote.bar_high
        position.last_tick_low = quote.bar_low
        position.last_tick_price = quote.last

        return False, position.stop != before_stop

    def _apply_tick(
        self,
        position: st.OpenPosition,
        window: live.TickWindow,
        atr: float,
        observed_ts: int,
    ) -> bool:
        """구간 하나를 청산 판정에 넣는다. 전량 청산됐으면 True."""

        state = _to_exit_state(position)
        try:
            if math.isfinite(atr) and atr > 0:
                event = ex.process_tick(
                    state,
                    open_=window.open,
                    high=window.high,
                    low=window.low,
                    last=window.last,
                    atr=atr,
                )
            else:
                # ATR 워밍업 중. 청산 판정은 그대로 하고 트레일만 건너뛴다.
                event = ex.check_exit(
                    state,
                    window.open,
                    window.high,
                    window.low,
                    window.last,
                    allow_time_exit=False,
                )
        except ex.ExitError as exc:
            print(f"[{position.symbol}] 틱 청산 판정 실패: {exc}")
            return False

        _sync_from_exit_state(position, state)
        if event is None:
            return False

        self._now_ts = int(observed_ts)

        if event.reason == ex.EXIT_TP1:
            ex.apply_tp1(state, event)
            _sync_from_exit_state(position, state)
            fully_closed = state.remaining_ratio <= 1e-12
            self._record_exit(position, event, partial=not fully_closed)
            if fully_closed:
                self.bot_state.positions.remove(position)
                return True
            return False        # 부분 익절. 남은 물량은 계속 감시한다

        self._record_exit(position, event, partial=False)
        self.bot_state.positions.remove(position)
        return True

    def _refresh_strategy_target(
        self,
        spec: SymbolSpec,
        position: st.OpenPosition,
        frames: dict[str, pd.DataFrame],
        current_idx: int,
    ) -> None:
        """전략이 제공하면 직전 확정봉 기준의 동적 TP1을 반영한다.

        현재 봉의 회귀채널은 종가가 나와야 계산할 수 있다. 그 값을 현재 봉
        고저가에 대입하면 미래참조이므로 반드시 ``current_idx - 1``을 넘긴다.
        부분익절 뒤에는 목표가 필요 없고 트레일링만 관리한다.
        """

        if position.tp1_filled or current_idx < 1:
            return
        strategy = _strategy_named(self, spec, position.strategy)
        hook = getattr(strategy, "target_before_bar", None)
        if hook is None:
            return

        try:
            target = hook(frames, current_idx - 1, position.side)
        except Exception as exc:  # noqa: BLE001 - 목표 갱신 실패로 청산을 막지 않는다
            print(f"[{position.symbol}] 동적 목표 갱신 실패: {type(exc).__name__}: {exc}")
            return
        if target is None or not math.isfinite(float(target)) or float(target) <= 0:
            return
        target = float(target)
        valid = target > position.entry if position.side == "long" else target < position.entry
        if valid:
            position.tp1 = target

    def _advance(self, position: st.OpenPosition, bar, atr: float) -> bool:
        """포지션 하나에 봉 하나를 반영한다.

        청산 이벤트(부분익절 포함)를 처리했으면 True다. 같은 봉에서 TP1 뒤
        국면청산까지 연달아 적용해 봉 내부 순서를 임의로 가정하지 않기 위함이다.
        """

        position.bars_held += 1
        state = _to_exit_state(position)

        if position.covered_bar_ts == int(bar["ts"]):
            # 이 봉은 실행 시계가 마감까지 전 구간을 감시했다.
            # 같은 OHLC로 손절·익절을 다시 보면 봉 안의 순서를 잃어
            # 없던 청산이 생긴다. 여기서는 시간 손절만 담당한다.
            return self._advance_time_only(position, state, float(bar["close"]))

        ohlc = dict(
            open_=float(bar["open"]),
            high=float(bar["high"]),
            low=float(bar["low"]),
            close=float(bar["close"]),
        )

        try:
            if math.isfinite(atr) and atr > 0:
                event = ex.process_bar(state, atr=atr, **ohlc)
            else:
                # ATR 워밍업 중이거나 캔들이 모자란 구간.
                # 청산 판정은 그대로 하고 트레일 갱신만 건너뛴다.
                # process_bar에 0을 넘기면 예외가 나서 청산까지 함께 막힌다.
                event = ex.check_exit(state, **ohlc)
        except ex.ExitError as exc:
            # 봉이 이상하면 이번 봉은 건너뛴다. 포지션을 강제 청산하지 않는다.
            print(f"[{position.symbol}] 청산 판정 실패: {exc}")
            return True

        _sync_from_exit_state(position, state)

        if event is None:
            return False

        if event.reason == ex.EXIT_TP1:
            ex.apply_tp1(state, event)
            _sync_from_exit_state(position, state)
            fully_closed = state.remaining_ratio <= 1e-12
            self._record_exit(position, event, partial=not fully_closed)
            if fully_closed:
                self.bot_state.positions.remove(position)
            return True

        self._record_exit(position, event, partial=False)
        self.bot_state.positions.remove(position)
        return True

    def _advance_time_only(
        self,
        position: st.OpenPosition,
        state: ex.PositionState,
        close: float,
    ) -> bool:
        """실행 시계가 이미 본 봉. 보유 봉 수 한도만 확인한다."""

        try:
            event = ex.check_time_exit(state, close=close)
        except ex.ExitError as exc:
            print(f"[{position.symbol}] 시간 청산 판정 실패: {exc}")
            return True

        _sync_from_exit_state(position, state)
        if event is None:
            return False

        self._record_exit(position, event, partial=False)
        self.bot_state.positions.remove(position)
        return True

    def _exit_on_strategy_invalidation(
        self,
        spec: SymbolSpec,
        position: st.OpenPosition,
        frames: dict[str, pd.DataFrame],
        df: pd.DataFrame,
        current_idx: int,
    ) -> None:
        """전략 고유 무효화 조건이 생기면 현재 봉 종가로 전량 청산한다.

        봉의 고저가로 손절·익절을 먼저 처리한 뒤 호출된다. 따라서 장중에
        손절이 닿았는데 종가 국면청산으로 덮어쓰는 낙관 편향이 생기지 않는다.
        """

        strategy = _strategy_named(self, spec, position.strategy)
        hook = getattr(strategy, "exit_reason", None)
        if hook is None:
            return

        try:
            prepared = self._prepare_for(strategy, frames, df)
            reason = hook(prepared, current_idx, position.side)
        except Exception as exc:  # noqa: BLE001 - 전략 오류로 보호 주문을 방치하지 않는다
            print(
                f"[{position.symbol}] 전략 무효화 판정 실패: "
                f"{type(exc).__name__}: {exc}"
            )
            return
        if not reason:
            return

        close = float(df["close"].iloc[current_idx])
        if not math.isfinite(close) or close <= 0:
            return
        slippage = self.broker.fees.slippage_rate
        fill = close * (
            1.0 - slippage if position.side == "long" else 1.0 + slippage
        )
        event = ex.ExitEvent(
            str(reason),
            fill,
            position.remaining_ratio,
            note="확정봉에서 전략 국면 무효화",
        )
        self._record_exit(position, event, partial=False)
        self.bot_state.positions.remove(position)

    def _record_exit(self, position: st.OpenPosition, event, partial: bool) -> None:
        """청산 체결을 기록하고 잔고에 반영한다."""

        before_balance = self.bot_state.balance
        tp1_maker = event.reason == ex.EXIT_TP1 and position.tp1_is_maker
        net, fee = self.broker.close(
            position.side,
            position.symbol,
            position.entry,
            event.price,
            position.contracts,
            ratio=event.ratio,
            is_maker=tp1_maker,
        )

        account = _account_from(self.bot_state, self._now_ts)
        result_risk = self.risk
        if position.consecutive_loss_limit is not None:
            pause_hours = self.risk.pause_hours
            if position.pause_until_next_day:
                now = datetime.fromtimestamp(self._now_ts, tz=timezone.utc)
                next_day = datetime.combine(
                    now.date(), datetime.min.time(), tzinfo=timezone.utc
                ).timestamp() + 24 * 3600
                pause_hours = max((next_day - self._now_ts) / 3600, 1 / 3600)
            result_risk = replace(
                result_risk,
                consecutive_loss_limit=position.consecutive_loss_limit,
                pause_hours=pause_hours,
            )
        lim.record_result(account, net, result_risk, strategy=position.strategy)
        self.bot_state.balance = account.balance
        self.bot_state.consecutive_losses = dict(account.consecutive_losses)
        self.bot_state.paused_until = dict(account.paused_until)

        self.trade_log.write(
            market_ts=self._now_ts,
            strategy=position.strategy,
            symbol=position.symbol,
            side=position.side,
            action="exit",
            reason=event.reason,
            price=round(event.price, 8),
            contracts=position.contracts,
            leverage=position.leverage,
            stop=round(position.stop, 8),
            tp1=round(position.tp1, 8),
            ratio=event.ratio,
            fee=round(fee, 6),
            pnl=round(net, 6),
            balance=round(self.bot_state.balance, 4),
            note=event.note or ("익절 지정가" if tp1_maker else ""),
        )
        self.notifier.exit(
            position.strategy,
            position.symbol,
            event.reason,
            event.price,
            net,
            side=position.side,
            partial=partial,
        )
        self.notifier.balance_change(
            position.strategy,
            position.symbol,
            "부분 익절" if partial else f"청산({event.reason})",
            before_balance,
            net,
            fee,
            self.bot_state.balance,
            RUNTIME.initial_balance,
        )
        print(
            f"[{position.symbol}] 청산 {event.reason} "
            f"{'부분' if partial else '전량'} {net:+.2f} USDT"
        )

    def _atr_of(self, df: pd.DataFrame, at: int = -1) -> float:
        """as_of 시점의 ATR.

        프레임 길이별로 캐시한다 — 리플레이에서 봉마다 전 구간 ATR을
        다시 계산하면 O(n^2)가 된다.
        """

        from indicators.atr import atr as atr_fn

        # id(df)는 쓰지 않는다 — 가비지 수집 후 주소가 재사용되면
        # 다른 프레임의 캐시를 잘못 맞춘다. 내용으로 식별한다.
        key = ("__atr__", len(df), int(df["ts"].iloc[-1]))
        series = self._prepared.get(key)
        if series is None:
            for old in [k for k in self._prepared if k[0] == "__atr__"]:
                del self._prepared[old]
            series = atr_fn(df)
            self._prepared[key] = series

        value = series.iloc[at]
        return float(value) if pd.notna(value) else 0.0

    def evaluate_entries(
        self,
        spec: SymbolSpec,
        frames: dict[str, pd.DataFrame],
        bar_ts: int | None = None,
        at: int | None = None,
        decision_ts: int | None = None,
    ) -> None:
        """전략 신호 -> 리스크 판정 -> 진입.

        bar_ts는 이번 신호를 만든 확정 봉의 타임스탬프다. 지정가 주문에
        그대로 기록해, 그 봉으로는 체결되지 않게 한다.
        """

        for strategy in self.strategies.get(spec.contract, []):
            entry_tf = _entry_tf_of(spec, strategy)
            df = frames.get(entry_tf)
            if df is None or df.empty:
                continue

            # **전략마다 자기 봉에서 as_of를 다시 찾는다.**
            # 루프는 가장 짧은 봉(5분) 기준으로 도는데, 1시간봉 전략은
            # 그 시각 이전에 마감된 1시간봉을 봐야 한다. 인덱스를 그대로
            # 쓰면 다른 봉의 위치를 읽어 완전히 엉뚱한 판단을 한다.
            if decision_ts is not None:
                as_of = _closed_index_at(df, entry_tf, decision_ts)
            else:
                as_of = _index_at(df, bar_ts) if bar_ts else len(df) - 1
            if as_of < 0:
                continue

            # 같은 봉을 두 번 평가하지 않는다. 1시간봉 전략은 5분봉이
            # 12번 도는 동안 같은 봉을 12번 보게 된다.
            key = f"{strategy.name}@{entry_tf}"
            ts = int(df["ts"].iloc[as_of])
            if self.bot_state.last_bar_ts.get(key) == ts:
                continue
            self.bot_state.last_bar_ts[key] = ts

            prepared = self._prepare_for(strategy, frames, df)
            for signal in strategy.generate(prepared, as_of):
                self._consider(spec, signal, ts)

    def _prepare_for(self, strategy, frames: dict[str, pd.DataFrame], df: pd.DataFrame):
        """전략의 prepare 결과. 같은 프레임이면 재사용한다.

        **키에 마지막 봉 타임스탬프를 넣는 게 핵심이다.**
        길이만 쓰면 실시간에서 캐시가 영원히 유지된다 — 매번 500봉씩
        받으므로 길이가 상수이기 때문이다. 그러면 지표가 기동 시점에
        얼어붙어 새 봉을 전혀 반영하지 못한다.

        실시간: 봉이 마감될 때마다 last_ts가 바뀌어 재계산 (정상)
        리플레이: 전체 프레임 고정 -> 1회만 계산
        """

        key = (strategy.name, len(df), int(df["ts"].iloc[-1]))
        cached = self._prepared.get(key)
        if cached is not None:
            return cached

        # 같은 전략의 옛 항목은 버린다. 실시간에서 봉마다 쌓이면 메모리가 샌다.
        for old in [k for k in self._prepared if k[0] == strategy.name]:
            del self._prepared[old]

        prepared = strategy.prepare(frames)
        self._prepared[key] = prepared
        return prepared

    def _tranche_of(self, symbol: str, strategy: str, side: str) -> int | None:
        """이번 신호가 몇 번째 트랜치인지.

        전략은 상태를 갖지 않으므로 자기가 몇 번째인지 모른다. 보유 포지션을
        아는 건 엔진뿐이라 여기서 매긴다.

        반대 방향 포지션이 이미 있으면 None. 같은 전략이 롱과 숏을 동시에
        들고 있으면 손절/트레일링이 서로를 상쇄해 성과 해석이 불가능해진다.
        """

        same = [
            p
            for p in self.bot_state.positions
            if p.symbol == symbol and p.strategy == strategy
        ]
        # 미체결 지정가도 포지션으로 센다. 빼면 최대 1포지션 전략이
        # 대기 주문을 여러 개 쌓았다가 한꺼번에 체결될 수 있다.
        pending = [
            it["signal"]
            for it in (*self.pending_signals, *self.pending_market_signals)
            if it["spec"].contract == symbol and it["signal"].strategy == strategy
        ]
        if any(p.side != side for p in same) or any(g.side != side for g in pending):
            return None
        return len(same) + len(pending) + 1

    def _as_maker_entry(self, signal):
        """시장가 신호를 지정가로 돌린다. 설정으로 켤 때만.

        왕복 taker 0.19% vs maker -0.02%. XRP 패턴은 목표 수익이 0.5% 안팎이라
        이 0.21%p가 성과를 그대로 가른다 — 실거래 8건에서 수수료 전 손익이
        명목의 +0.008%인데 수수료로 0.15%를 냈다.

        대신 일부 신호는 체결되지 않는다. 그래서 기본값이 꺼짐이고,
        체결률(limit_stats)을 성과보다 먼저 봐야 한다.

        전략이 이미 지정가로 냈으면 손대지 않는다. TTL도 전략 것이 우선이다 —
        BTC 회귀채널은 12봉짜리를 쓰는데 여기서 2로 덮으면 전략이 바뀐다.
        """

        if not ENTRY.prefer_maker_entries:
            return signal
        if signal.order_type != "market":
            return signal

        return replace(
            signal,
            order_type="limit",
            ttl_bars=signal.ttl_bars or ENTRY.maker_entry_ttl_bars,
        )

    def _consider(
        self,
        spec: SymbolSpec,
        signal,
        bar_ts: int,
        allow_market_defer: bool = True,
    ) -> None:
        """신호 하나를 리스크 레이어에 통과시킨다. 거절도 기록한다."""

        from risk import sizing as sz

        policy = signal.position_policy
        if policy.max_daily_entries is not None:
            if policy.max_daily_entries < 1:
                raise ValueError("일일 진입 한도는 양의 정수여야 한다")
            daily_count = self.bot_state.daily_entries.get(signal.strategy, 0)
            if daily_count >= policy.max_daily_entries:
                self._skip(
                    signal,
                    spec,
                    SKIP_DAILY_ENTRY_LIMIT,
                    daily_entries=daily_count,
                    max_daily_entries=policy.max_daily_entries,
                )
                return
        open_same_symbol = [
            p for p in self.bot_state.positions if p.symbol == spec.contract
        ]
        pending_same_symbol = [
            it["signal"]
            for it in (*self.pending_signals, *self.pending_market_signals)
            if it["spec"].contract == spec.contract
        ]
        reserved = any(p.exclusive_symbol for p in open_same_symbol) or any(
            g.position_policy.exclusive_symbol for g in pending_same_symbol
        )
        if reserved or (
            policy.exclusive_symbol and (open_same_symbol or pending_same_symbol)
        ):
            self._skip(signal, spec, SKIP_SYMBOL_EXCLUSIVE)
            return

        tranche = self._tranche_of(spec.contract, signal.strategy, signal.side)
        if tranche is None:
            self._skip(signal, spec, SKIP_OPPOSITE_OPEN)
            return
        if tranche > signal.max_tranches:
            self._skip(signal, spec, SKIP_MAX_TRANCHES, tranche=tranche)
            return

        signal = self._as_maker_entry(signal)

        if (
            allow_market_defer
            and self.defer_market_entries
            and signal.order_type == "market"
        ):
            self.pending_market_signals.append(
                {
                    "spec": spec,
                    "signal": signal,
                    "signal_bar_ts": bar_ts,
                    # 신호봉이 마감된 바로 그 시각. 그 다음 가장 짧은 봉의
                    # 시가가 과거 데이터에서 처음 주문 가능한 가격이다.
                    "eligible_ts": self._decision_ts,
                }
            )
            self.signal_log.write(
                ACTION_SKIP,
                signal.strategy,
                spec.contract,
                signal.side,
                skip_reason="market_pending_next_open",
                entry=signal.entry,
                eligible_ts=self._decision_ts,
            )
            return

        # 청산 계획을 **먼저** 세운다.
        #
        # 전략이 주는 stop_hint는 구조적 위치일 뿐이고, 실제 손절은 거기에
        # ATR 버퍼를 더한 자리다. 순서를 뒤집어 stop_hint로 크기를 정하면
        # 좁은 손절로 계산하고 넓은 손절로 손실이 나서, 1% 리스크가
        # 실측 1.8~3.2배로 새어 나간다. 손절 거리 하한 판정도 같은 이유로
        # 실제보다 좁은 값을 보게 된다.
        plan = self._plan_for(spec, signal)
        if plan is None:
            return

        execution = (
            sz.ExecutionAssumption.maker_entry()
            if signal.order_type == "limit"
            else sz.ExecutionAssumption.taker()
        )
        position = sz.size_position(
            spec.contract,
            signal.side,
            signal.entry,
            plan.stop,
            self.bot_state.balance,
            signal.strategy,
            tranche=tranche,
            max_tranches=signal.max_tranches,
            # 열린 포지션이 묶은 증거금을 뺀 나머지만 새 포지션에 쓸 수 있다.
            # 넘기지 않으면 sizing이 매번 잔고 전액을 가용으로 본다.
            available_margin=max(
                0.0, self.bot_state.balance - st.used_margin(self.bot_state.positions)
            ),
            risk=self.risk,
            execution=execution,
            risk_multiplier=getattr(signal, "risk_multiplier", 1.0),
            fixed_leverage=policy.fixed_leverage,
            margin_fraction=policy.margin_fraction,
            min_liquidation_multiple=policy.min_liquidation_multiple,
        )

        if not position.ok:
            self.signal_log.write(
                ACTION_SKIP,
                signal.strategy,
                spec.contract,
                signal.side,
                skip_reason=position.rejected,
                entry=signal.entry,
                stop=signal.stop_hint,
                notes=position.notes,
            )
            return

        account = _account_from(self.bot_state, bar_ts)
        risk_for_signal = self.risk
        if policy.symbol_risk_cap is not None:
            risk_for_signal = replace(
                risk_for_signal,
                max_symbol_risk=max(
                    self.risk.max_symbol_risk, float(policy.symbol_risk_cap)
                ),
            )
        if policy.consecutive_loss_limit is not None:
            risk_for_signal = replace(
                risk_for_signal,
                consecutive_loss_limit=policy.consecutive_loss_limit,
            )
        verdict = lim.check_entry(
            account, spec.contract, signal.side, position.risk_amount,
            risk=risk_for_signal, strategy=signal.strategy,
        )
        if not verdict.allowed:
            self.signal_log.write(
                ACTION_SKIP,
                signal.strategy,
                spec.contract,
                signal.side,
                skip_reason=verdict.reason,
                detail=verdict.detail,
                entry=signal.entry,
            )
            return

        self._open(spec, signal, position, plan, tranche, bar_ts)

    def _plan_for(self, spec: SymbolSpec, signal):
        """전략 신호로 청산 계획을 세운다. 실패하면 None (진입 포기)."""

        policy = signal.position_policy
        try:
            return ex.build_plan(
                signal.side,
                signal.entry,
                signal.stop_hint,
                float(signal.context.get("atr", 0.0) or 0.0),
                target_hint=signal.target_hint,
                scalp=spec.contract != "BTC_USDT",
                stop_buffer_atr_mult=policy.stop_buffer_atr_mult,
                partial_exit_ratio=policy.partial_exit_ratio,
                trail_atr_mult=policy.trail_atr_mult,
                max_hold_bars=policy.max_hold_bars,
                time_exit_enabled=policy.time_exit_enabled,
                breakeven_offset_pct=policy.breakeven_offset_pct,
                min_target_r=policy.min_target_r,
            )
        except ex.ExitError as exc:
            # 계획을 세울 수 없으면 진입하지 않는다. 손절 없는 포지션이
            # 열리는 것보다 기회를 놓치는 편이 낫다.
            self._skip(signal, spec, "exit_plan_failed", detail=str(exc))
            return None

    def _open(self, spec: SymbolSpec, signal, position, plan, tranche: int, bar_ts: int) -> None:
        """진입 체결. 상태에 등록한다.

        지정가 신호는 여기서 바로 열지 않는다. 가격이 실제로 그 값을
        지나갔을 때만 체결되고, ttl_bars 안에 못 채우면 버린다.
        """

        if signal.order_type == "limit":
            self._place_limit(spec, signal, position, plan, tranche, bar_ts)
            return

        before_balance = self.bot_state.balance
        entry_fee = position.notional * self.broker.fees.taker_rate
        self.broker.balance -= entry_fee

        self.bot_state.positions.append(
            st.OpenPosition(
                symbol=spec.contract,
                side=signal.side,
                strategy=signal.strategy,
                entry=position.entry_fill_price,
                contracts=position.contracts,
                leverage=position.leverage,
                stop=plan.stop,
                tp1=plan.tp1,
                partial_ratio=plan.partial_ratio,
                trail_atr_mult=plan.trail_atr_mult,
                max_hold_bars=plan.max_hold_bars,
                risk_amount=position.risk_amount,
                time_exit_enabled=plan.time_exit_enabled,
                breakeven_offset_pct=plan.breakeven_offset_pct,
                exclusive_symbol=signal.position_policy.exclusive_symbol,
                consecutive_loss_limit=signal.position_policy.consecutive_loss_limit,
                pause_until_next_day=signal.position_policy.pause_until_next_day,
                tp1_is_maker=signal.position_policy.tp1_is_maker,
                extreme=position.entry_fill_price,
                entry_tf=_entry_tf_of(spec, _strategy_named(self, spec, signal.strategy)),
                last_bar_ts=(
                    self._deferred_signal_bar_ts
                    if self._deferred_signal_bar_ts is not None
                    else bar_ts
                ),
                opened_ts=int(time.time()),
            )
        )
        self._record_entry(signal.strategy)
        self.bot_state.balance = self.broker.balance

        self.trade_log.write(
            market_ts=bar_ts,
            strategy=signal.strategy,
            symbol=spec.contract,
            side=signal.side,
            action="entry",
            price=round(position.entry_fill_price, 8),
            contracts=position.contracts,
            notional=round(position.notional, 4),
            leverage=position.leverage,
            stop=round(plan.stop, 8),
            tp1=round(plan.tp1, 8),
            ratio=1.0,
            balance=round(self.bot_state.balance, 4),
            note="taker",
        )
        self.signal_log.write(
            ACTION_ENTRY,
            signal.strategy,
            spec.contract,
            signal.side,
            entry=position.entry_fill_price,
            stop=plan.stop,
            tp1=plan.tp1,
            tranche=tranche,
            max_tranches=signal.max_tranches,
            score=signal.score,
            reasons=signal.reasons,
            leverage=position.leverage,
            risk_amount=position.risk_amount,
            indicators=signal.context,
        )
        self.notifier.entry(
            signal.strategy,
            spec.contract,
            signal.side,
            position,
            position.entry_fill_price,
            reasons=signal.reasons,
            score=signal.score,
            context=signal.context,
            stop=plan.stop,
            target=plan.tp1,
            tranche=tranche,
            max_tranches=signal.max_tranches,
            order_type="market",
        )
        self.notifier.balance_change(
            signal.strategy,
            spec.contract,
            "시장가 진입",
            before_balance,
            self.bot_state.balance - before_balance,
            entry_fee,
            self.bot_state.balance,
            RUNTIME.initial_balance,
        )
        print(
            f"[{spec.contract}] 진입 {signal.side} {tranche}/{signal.max_tranches} "
            f"{position.contracts}계약 @ {position.entry_fill_price:g} "
            f"({', '.join(signal.reasons)})"
        )

    def process_deferred_markets(
        self,
        spec: SymbolSpec,
        frames: dict[str, pd.DataFrame],
        bar_ts: int | None = None,
        at: int | None = None,
    ) -> None:
        """실시간 근사 리플레이의 시장가 신호를 다음 가장 짧은 봉 시가에 체결.

        신호봉 종가는 봉이 닫힌 뒤에야 확정된다. 과거 데이터에서 그 종가에
        즉시 체결시키면 실제로 주문할 수 없었던 가격을 쓰게 된다. XRP 루프의
        가장 짧은 5분봉에서 다음 봉 시가를 쓰면 15분·1시간·4시간 신호도
        마감 직후 처음 관측 가능한 가격으로 통일된다.

        실제 체결가가 바뀌었으므로 손절 계획과 수량은 여기서 다시 계산한다.
        갭 때문에 목표/손절 구조가 깨졌다면 진입하지 않는다.
        """

        if not self.defer_market_entries or not self.pending_market_signals:
            return

        base_tf = spec.driver_tf
        df = frames.get(base_tf)
        if df is None or df.empty:
            return

        at = len(df) - 1 if at is None else at
        if at < 0 or at >= len(df):
            return
        bar = df.iloc[at]
        bar_ts = int(bar["ts"]) if bar_ts is None else int(bar_ts)
        open_price = float(bar["open"])

        if not math.isfinite(open_price) or open_price <= 0:
            return

        for item in list(self.pending_market_signals):
            if item["spec"].contract != spec.contract:
                continue
            if bar_ts < int(item["eligible_ts"]):
                continue

            self.pending_market_signals.remove(item)
            original = item["signal"]
            policy = original.position_policy
            changes = {"entry": open_price}
            direction = 1.0 if original.side == "long" else -1.0
            if policy.stop_pct_from_entry is not None:
                changes["stop_hint"] = open_price * (
                    1.0 - direction * float(policy.stop_pct_from_entry)
                )
            if policy.target_pct_from_entry is not None:
                changes["target_hint"] = open_price * (
                    1.0 + direction * float(policy.target_pct_from_entry)
                )
            signal = replace(original, **changes)
            self._deferred_signal_bar_ts = int(item["signal_bar_ts"])
            try:
                self._consider(
                    spec,
                    signal,
                    bar_ts,
                    allow_market_defer=False,
                )
            finally:
                self._deferred_signal_bar_ts = None

    def _place_limit(self, spec: SymbolSpec, signal, position, plan, tranche: int, bar_ts: int) -> None:
        """지정가 주문을 대기열에 넣는다.

        신호와 크기 산정 결과를 함께 들고 있어야 체결 시점에 포지션을 열 수 있다.
        엔진이 재시작되면 아직 대기 주문은 사라진다. BTC는 TTL이 12시간이라
        실시간 장기 운용 전 상태 파일에 함께 저장해야 한다.
        """

        self.pending_signals.append(
            {
                "spec": spec,
                "signal": signal,
                "position": position,
                "plan": plan,
                "tranche": tranche,
                # 신호를 만든 확정 봉의 타임스탬프. 루프 회차가 아니라
                # 캔들 단위로 나이를 세고, 이 봉으로는 체결하지 않는다.
                "placed_ts": bar_ts,
            }
        )
        self.signal_log.write(
            ACTION_SKIP,
            signal.strategy,
            spec.contract,
            signal.side,
            skip_reason="limit_pending",
            entry=signal.entry,
            ttl_bars=signal.ttl_bars,
        )

    def process_pending(
        self,
        spec: SymbolSpec,
        frames: dict[str, pd.DataFrame],
        bar_ts: int | None = None,
        at: int | None = None,
    ) -> None:
        """대기 중인 지정가 주문의 체결·만료를 처리한다.

        **주문을 낸 봉으로는 체결하지 않는다.**
        신호는 봉이 마감된 뒤에 나오므로, 그 봉 안에서 이미 지나간 가격에는
        주문이 존재하지도 않았다. 그 가격으로 체결시키면 미래참조다.

        유동성 스윕에서 특히 치명적이다 — 신호 봉이 진입 레벨을 이미 뚫고
        회복한 구조라 `low <= price`가 구조적으로 참이고, 그대로 두면
        거의 모든 주문이 체결된 것처럼 보인다.

        체결률을 반드시 기록한다. XRP 전략 전체가 메이커 체결을 전제로
        설계돼 있어서, 이 비율이 낮으면 손익분기 승률 가정이 무너진다.
        """

        entry_tf = spec.driver_tf
        df = frames.get(entry_tf)
        if (df is None or df.empty) and self.pending_signals:
            # 단위 테스트·수동 호출처럼 드라이버 봉 없이 주문 전략의 봉만
            # 주어진 경우의 호환 경로. 실시간 수집은 항상 driver_tf를 준다.
            sample = next(
                (
                    item["signal"]
                    for item in self.pending_signals
                    if item["spec"].contract == spec.contract
                ),
                None,
            )
            if sample is not None:
                strategy = _strategy_named(self, spec, sample.strategy)
                entry_tf = _entry_tf_of(spec, strategy)
                df = frames.get(entry_tf)
        if df is None or df.empty or not self.pending_signals:
            return

        at = len(df) - 1 if at is None else at
        bar = df.iloc[at]
        bar_ts = int(bar["ts"]) if bar_ts is None else bar_ts
        high, low = float(bar["high"]), float(bar["low"])

        for item in list(self.pending_signals):
            if item["spec"].contract != spec.contract:
                continue

            placed = item["placed_ts"]
            if bar_ts <= placed:
                continue        # 주문을 낸 봉. 다음 봉부터 체결 대상이다.

            signal = item["signal"]
            price = signal.entry
            touched = low <= price if signal.side == "long" else high >= price

            if touched:
                self.pending_signals.remove(item)
                self._fill_limit(item, bar)
                continue

            signal_strategy = _strategy_named(self, spec, signal.strategy)
            signal_tf = _entry_tf_of(spec, signal_strategy)
            age = (bar_ts - placed) // cd.interval_seconds(signal_tf)
            if age >= signal.ttl_bars:
                self.pending_signals.remove(item)
                self.limit_stats["expired"] += 1
                self.signal_log.write(
                    ACTION_SKIP,
                    signal.strategy,
                    spec.contract,
                    signal.side,
                    skip_reason="limit_expired",
                    entry=price,
                    age_bars=age,
                )

    def _fill_limit(self, item: dict, fill_bar=None, bar_ts: int | None = None) -> None:
        """지정가 체결. 메이커라 수수료가 리베이트(음수)다.

        fill_bar가 손절선까지 지나갔다면 같은 봉에 즉시 손절한다. 지정가는
        봉 중간에 체결될 수 있으므로 다음 봉까지 기다리면, 체결 뒤 이미 발생한
        손절을 통째로 누락해 백테스트가 낙관적으로 변한다. 반면 같은 봉 고가가
        목표에 닿았더라도 그게 체결 전인지 후인지 모르므로 익절은 인정하지 않는다.
        """

        spec, signal, position = item["spec"], item["signal"], item["position"]
        plan = item["plan"]
        self.limit_stats["filled"] += 1

        # 메이커 수수료는 음수(리베이트)라 잔고가 늘어난다.
        before_balance = self.bot_state.balance
        entry_fee = position.notional * self.broker.fees.maker_rate
        self.broker.balance -= entry_fee

        opened = st.OpenPosition(
            symbol=spec.contract,
            side=signal.side,
            strategy=signal.strategy,
            entry=signal.entry,
            contracts=position.contracts,
            leverage=position.leverage,
            stop=plan.stop,
            tp1=plan.tp1,
            partial_ratio=plan.partial_ratio,
            trail_atr_mult=plan.trail_atr_mult,
            max_hold_bars=plan.max_hold_bars,
            risk_amount=position.risk_amount,
            time_exit_enabled=plan.time_exit_enabled,
            breakeven_offset_pct=plan.breakeven_offset_pct,
            exclusive_symbol=signal.position_policy.exclusive_symbol,
            consecutive_loss_limit=signal.position_policy.consecutive_loss_limit,
            pause_until_next_day=signal.position_policy.pause_until_next_day,
            tp1_is_maker=signal.position_policy.tp1_is_maker,
            extreme=signal.entry,
            entry_tf=_entry_tf_of(spec, _strategy_named(self, spec, signal.strategy)),
            # 체결봉은 아래에서 손절만 보수적으로 검사한다. 정상적인 목표·시간
            # 진행은 다음 자기 타임프레임 봉부터 시작한다.
            #
            # 틱 체결은 _now_ts가 벽시계라 봉 경계와 맞지 않는다. 그대로 두면
            # 이 값이 어떤 봉 ts와도 같지 않아 체결봉까지 보유 봉 수에 세어지고,
            # 시간 손절이 한 봉 일찍 터진다. 진행 중인 봉 ts를 넘겨 막는다.
            last_bar_ts=self._now_ts if bar_ts is None else int(bar_ts),
            opened_ts=int(time.time()),
        )
        self.bot_state.positions.append(opened)
        self._record_entry(signal.strategy)
        self.bot_state.balance = self.broker.balance

        # 주문을 낸 봉이 아니라 **체결된 봉**의 시각이다.
        self.trade_log.write(
            market_ts=self._now_ts,
            strategy=signal.strategy,
            symbol=spec.contract,
            side=signal.side,
            action="entry",
            reason="limit",
            price=round(signal.entry, 8),
            contracts=position.contracts,
            notional=round(position.notional, 4),
            leverage=position.leverage,
            stop=round(plan.stop, 8),
            tp1=round(plan.tp1, 8),
            ratio=1.0,
            balance=round(self.bot_state.balance, 4),
            note="maker",
        )
        self.signal_log.write(
            ACTION_ENTRY,
            signal.strategy,
            spec.contract,
            signal.side,
            entry=signal.entry,
            stop=plan.stop,
            tp1=plan.tp1,
            tranche=item["tranche"],
            fill_type="maker",
            score=signal.score,
            reasons=signal.reasons,
            indicators=signal.context,
        )
        self.notifier.entry(
            signal.strategy,
            spec.contract,
            signal.side,
            position,
            signal.entry,
            reasons=signal.reasons,
            score=signal.score,
            context=signal.context,
            stop=plan.stop,
            target=plan.tp1,
            tranche=item["tranche"],
            max_tranches=signal.max_tranches,
            order_type="limit",
        )
        self.notifier.balance_change(
            signal.strategy,
            spec.contract,
            "지정가 진입",
            before_balance,
            self.bot_state.balance - before_balance,
            entry_fee,
            self.bot_state.balance,
            RUNTIME.initial_balance,
        )
        print(f"[{spec.contract}] 지정가 체결 {signal.side} @ {signal.entry:g}")

        if fill_bar is None:
            return

        stop_touched = (
            float(fill_bar["low"]) <= opened.stop
            if opened.side == "long"
            else float(fill_bar["high"]) >= opened.stop
        )
        if not stop_touched:
            return

        # stop이 닿은 경우 check_exit의 갭·슬리피지 규칙을 그대로 재사용한다.
        state = _to_exit_state(opened)
        event = ex.check_exit(
            state,
            open_=float(fill_bar["open"]),
            high=float(fill_bar["high"]),
            low=float(fill_bar["low"]),
            close=float(fill_bar["close"]),
        )
        if event is None:  # stop_touched와 모순이면 조용히 포지션을 방치하지 않는다.
            raise RuntimeError("지정가 체결봉 손절 판정 불일치")
        self._record_exit(opened, event, partial=False)
        self.bot_state.positions.remove(opened)

    def _skip(self, signal, spec: SymbolSpec, reason: str, **extra) -> None:
        self.signal_log.write(
            ACTION_SKIP,
            signal.strategy,
            spec.contract,
            signal.side,
            skip_reason=reason,
            entry=signal.entry,
            **extra,
        )

    def _record_entry(self, strategy: str) -> None:
        """실제 체결된 진입만 UTC 일일 횟수에 더한다."""

        self.bot_state.daily_entries[strategy] = (
            self.bot_state.daily_entries.get(strategy, 0) + 1
        )

    # ---------------------------------------------------------------- 상태

    def _release_pause(self, now_ts: int) -> None:
        """연속 손절 정지가 만료됐으면 푼다.

        이게 없으면 정지가 영구가 된다 — 카운터는 이익 거래로만 초기화되는데
        진입이 막혀 있으면 이익 거래가 나올 수 없다.
        """

        account = _account_from(self.bot_state, now_ts)
        released = lim.clear_expired_pause(account, self.risk)
        for tag in released:
            self.bot_state.paused_until[tag] = 0
            self.bot_state.consecutive_losses[tag] = 0
            print(f"[{self._fmt(now_ts)}] {tag} 연속 손절 정지 해제")

    @staticmethod
    def _fmt(ts: int) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    def _roll_day(self, now_ts: int = 0) -> None:
        """날짜가 바뀌면 일일 손실 기준점을 옮긴다.

        **봉 시각을 쓴다. 실제 시각이 아니다.**
        리플레이는 과거를 재생하므로 `datetime.now()`를 쓰면 날짜가
        절대 바뀌지 않는다. 그러면 일일 손실 한도가 180일 내내 한 번도
        리셋되지 않아, 첫 5% 손실 이후 모든 진입이 막힌다
        (실측 949건 거절).
        """

        ts = now_ts or self._now_ts or int(time.time())
        today = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

        # **앞으로만 넘어간다.**
        #
        # 이 함수는 종목마다 불리는데, 봉 마감 시각이 자정을 사이에 두고
        # 갈릴 수 있다 — XRP 5분봉은 00:00에 마감해 08-09를 보고,
        # BTC 1시간봉은 23:00에 마감해 08-08을 본다.
        # 뒤로도 넘어가게 두면 한 회차에 두 번 뒤집히고, 10초 폴링에서
        # 시간당 720건의 알림이 나간다(실제로 그렇게 됐다).
        #
        # 과거 날짜는 무시한다. 어차피 가장 앞선 봉이 진짜 오늘이다.
        if self.bot_state.day_key and today <= self.bot_state.day_key:
            return

        previous_day = self.bot_state.day_key
        self.bot_state.day_key = today
        self.bot_state.day_start_balance = self.bot_state.balance
        self.bot_state.daily_entries = {}
        if previous_day:
            self.notifier.text(
                "🟢 페이퍼 봇 생존 확인\n"
                f"UTC {today} · 잔고 {self.bot_state.balance:,.2f} USDT"
            )

    def _persist(self) -> None:
        self.bot_state.balance = self.broker.balance
        st.save(self.bot_state)

    # ---------------------------------------------------------------- 실행

    def run(self, max_iterations: int | None = None) -> None:
        """poll_seconds 간격으로 회차를 반복한다.

        예외로 죽지 않는다. 한 회차가 실패해도 다음 회차를 시도한다.
        가상거래는 장기간 돌면서 로그를 모으는 게 목적이라,
        일시적 오류로 중단되면 그 목적을 잃는다.
        """

        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            try:
                # 실행 시계를 먼저 돌린다.
                #
                # 봉이 막 넘어간 회차에서, 직전 봉의 마지막 몇 초(어떤 틱도
                # 보지 못한 구간)를 run_tick이 처리하고 covered_bar_ts를 올린다.
                # 그래야 뒤이은 run_once가 그 봉을 중복 판정하지 않는다.
                # 순서를 뒤집으면 봉 시계가 먼저 전체를 다시 보게 된다.
                self.run_tick()
                self.run_once()
            except KeyboardInterrupt:
                print("\n중단 요청. 상태 저장 후 종료.")
                self._persist()
                break
            except Exception as exc:                # noqa: BLE001 - 루프를 지킨다
                print(f"[loop] 회차 {iteration} 실패: {type(exc).__name__}: {exc}")
                self.notifier.warn("루프 오류", f"{type(exc).__name__}: {exc}")

            if max_iterations is None or iteration < max_iterations:
                try:
                    time.sleep(RUNTIME.poll_seconds)
                except KeyboardInterrupt:
                    print("\n중단 요청. 상태 저장 후 종료.")
                    self._persist()
                    break

    def close(self) -> None:
        self.client.close()
        self.notifier.close()


def _strategy_named(engine, spec: SymbolSpec, name: str):
    """이름으로 전략 인스턴스를 찾는다. 진입봉을 알아내는 데 쓴다."""

    for s in engine.strategies.get(spec.contract, []):
        if getattr(s, "name", None) == name:
            return s
    return None


def _index_at(df: pd.DataFrame, bar_ts: int) -> int:
    """bar_ts 시점에 **이미 마감된** 마지막 봉의 위치.

    상위봉은 아직 진행 중일 수 있다. 진행 중인 봉을 그대로 쓰면 미래참조다 —
    1시간봉이 30분 남았는데 그 종가를 보는 셈이 된다.

    봉 타임스탬프는 시작 시각이므로, `ts + step <= bar_ts + step`
    즉 `ts <= bar_ts` 인 마지막 봉을 고른다.
    """

    ts = df["ts"].to_numpy()
    return int(np.searchsorted(ts, bar_ts, side="right") - 1)


def _closed_index_at(df: pd.DataFrame, timeframe: str, decision_ts: int) -> int:
    """decision_ts까지 실제로 마감된 마지막 timeframe 봉의 위치.

    캔들 타임스탬프는 시작 시각이다. 예를 들어 10:05에 10:00 1시간봉은
    시작했을 뿐이고 11:00이 되어야 쓸 수 있다. 전체 과거 프레임을 미리
    들고 있는 리플레이에서는 이 마감 조건을 직접 강제해야 한다.
    """

    deadline = int(decision_ts) - cd.interval_seconds(timeframe)
    ts = df["ts"].to_numpy()
    return int(np.searchsorted(ts, deadline, side="right") - 1)


def _entry_tf_of(spec: SymbolSpec, strategy) -> str:
    """전략이 판단에 쓰는 진입 타임프레임.

    전략 인스턴스가 `tf.entry`를 들고 있으면 그게 우선이다. XRP 패턴은
    같은 패턴도 봉마다 성과가 달라(실측 double_top 5분 -0.21% / 1시간 +0.40%)
    인스턴스마다 다른 봉을 쓴다.

    없으면 SymbolSpec에서 이름으로 찾는다 — BTC 전략들이 이쪽이다.
    """

    tf = getattr(strategy, "tf", None)
    if tf is not None and getattr(tf, "entry", None):
        return tf.entry

    name = getattr(strategy, "name", "")
    for s in spec.strategies:
        if s.name == name:
            return s.entry_tf
    return spec.driver_tf


def _to_exit_state(position: st.OpenPosition) -> ex.PositionState:
    """저장된 포지션을 청산 판정용 상태로.

    두 타입을 나눠 둔 이유는 저장 형식(JSON 직렬화)과 계산 형식의
    수명이 다르기 때문이다. 상태 파일 버전을 올리지 않고 청산 규칙을
    바꿀 수 있어야 한다.
    """

    plan = ex.ExitPlan(
        stop=position.stop,
        tp1=position.tp1,
        partial_ratio=position.partial_ratio,
        trail_atr_mult=position.trail_atr_mult,
        max_hold_bars=position.max_hold_bars,
        time_exit_enabled=position.time_exit_enabled,
        breakeven_offset_pct=position.breakeven_offset_pct,
    )
    return ex.PositionState(
        side=position.side,
        entry=position.entry,
        plan=plan,
        bars_held=position.bars_held,
        remaining_ratio=position.remaining_ratio,
        tp1_filled=position.tp1_filled,
        extreme=position.extreme or position.entry,
        stop=position.stop,
    )


def _sync_from_exit_state(position: st.OpenPosition, state: ex.PositionState) -> None:
    """청산 판정이 바꾼 값을 저장용 포지션에 되돌린다."""

    position.stop = float(state.stop)
    position.extreme = float(state.extreme)
    position.remaining_ratio = float(state.remaining_ratio)
    position.tp1_filled = bool(state.tp1_filled)
    position.bars_held = int(state.bars_held)


def _account_from(bot_state: st.BotState, now_ts: int = 0) -> lim.AccountState:
    return lim.AccountState(
        balance=bot_state.balance,
        day_start_balance=bot_state.day_start_balance,
        consecutive_losses=dict(bot_state.consecutive_losses),
        paused_until=dict(bot_state.paused_until),
        now_ts=now_ts or int(time.time()),
        open_positions=[
            lim.OpenRisk(p.symbol, p.side, p.strategy, p.risk_amount)
            for p in bot_state.positions
        ],
    )


def build_strategies() -> dict[str, list]:
    """종목별 전략 목록.

    여기 등록하지 않으면 엔진은 신호도 진입도 0회다. 전략 클래스를 만들어도
    이 함수에 넣지 않으면 아무 일도 일어나지 않으므로, 새 전략을 추가할 때
    반드시 함께 고칠 것.

    import를 함수 안에 두는 이유는 순환 참조 때문이다 —
    strategies가 indicators를, engine이 strategies를 쓴다.
    """

    from strategies.btc_regression_channel import BtcRegressionChannel
    from strategies.btc_regime_25x import BtcRegime25x
    from strategies.xrp_patterns.cup_handle import CupHandle
    from strategies.xrp_patterns.double_top import DoubleTop
    from strategies.xrp_patterns.head_shoulders import HeadShoulders
    from strategies.xrp_patterns.kotegawa import Kotegawa
    from strategies.xrp_patterns.triangle import Triangle
    from strategies.xrp_patterns.wedge import Wedge

    # 패턴마다 잘 맞는 봉이 다르다. 949일 실측에서 같은 패턴도
    # 봉을 바꾸면 결과가 뒤집혔다 — double_top은 5분봉 -0.21%,
    # 1시간봉 +0.40%. 한 봉으로 통일하면 절반은 손해 보는 자리에 둔다.
    return {
        # 국면 필터 + 추세 일치 자리 가중.
        # 필터 없이는 103건 -6.53 (롱 +107 / 숏 -102로 상쇄).
        # 대세 역행과 횡보 숏을 빼면 41건 +168.12, 앞뒤 절반 모두 플러스.
        # 진입 이격 1%. 2024-01~2026-07 리플레이(국면 필터 켬):
        #   시장가(이격 0)  20건 -4.94%  — 목표가 채널선이라 R<1로 408건 거절
        #   1%            114건 +82.18% DD 3.2%  체결 114/210
        #   2%             58건 +80.73% DD 5.0%  체결  58/240
        # 수익은 같고 1%가 거래 2배·낙폭 절반. 실시간 2% 주문 3건은
        # 전부 0.9~2.1% 못 미쳐 만료됐고 1%였으면 1건은 체결됐다.
        "BTC_USDT": [
            BtcRegressionChannel(
                entry_offset_pct=0.01, regime_filter=True, aligned_risk_mult=2.0,
            ),
            BtcRegime25x(),
        ],
        "XRP_USDT": [
            # 컵앤핸들 — 전 봉. 어디서도 표본이 30건을 못 넘어
            # 봉별 비교가 불가능했다. 전부 켜고 실시간으로 모은다.
            CupHandle("5m"), CupHandle("15m"), CupHandle("1h"), CupHandle("4h"),

            # 이중천장 — 1시간봉만. 949일 실측에서 유일하게 유의했고
            # (+0.40%, t=2.41), 리플레이에서도 +33.82(승률 69%)로 최상위다.
            # 4시간봉은 22건뿐이라 판단이 서지 않아 뺀다.
            # 롱에서 vwap_aligned 품질 항목을 뺀다 — EXP-014.
            DoubleTop("1h", drop_long_vwap_alignment=True),

            # 쐐기 — 5분·15분봉. 949일 전체로는 유의하지 않지만
            # 지정가 전환 시 +0.12%/+0.30%로 개선됐다.
            # 5분봉은 대세 역행을 막는다 — 상승장 숏이 -94.5로 손실 전부였다.
            # 15분봉은 전 국면에서 플러스라 그대로 둔다.
            # 5분봉: 대세 역행 차단 + 추세선 지정가 + 16~24시 UTC 차단.
            # 세 조건 모두 앞뒤 절반에서 일관됐다 (지정가+16시이전 99건,
            # 앞 +0.379% / 뒤 +0.212%). 15분봉은 전 국면 플러스라 그대로 둔다.
            # 롱에서 vwap_aligned 품질 항목을 뺀다 — EXP-014.
            Wedge("5m", trend_align=True, limit_entry=True,
                  block_hours=tuple(range(16, 24)), drop_long_vwap_alignment=True),
            Wedge("15m", drop_long_vwap_alignment=True),

            # 삼각수렴 — **비활성**. 949일에서 -130.24.
            #
            # 20개 축을 훑었지만 앞뒤 절반이 모두 플러스인 조합이 없었다:
            #   진입(시장가/지정가), 익절(1~3R/트레일링), 보유(10~160봉),
            #   방향, 시간대, 변동성, 진행률, 선행추세 강도·지속 여부,
            #   MA 배열, MACD 크로스·0선, RSI, VWAP, 볼린저, RVOL, 그 조합들.
            #
            # 거의 모든 축에서 앞 구간 마이너스 / 뒤 구간 플러스로 갈렸다.
            # 두 구간의 성격이 반대라 어떤 조건으로 잘라도 한쪽이 무너진다.
            # 같은 방법으로 wedge.5m은 세 개를 찾았으므로 탐색 부실은 아니다.
            #
            # 코드와 테스트는 남긴다. 데이터가 더 쌓이면 재검증할 것 —
            # kotegawa가 180일 -60%에서 949일 +460으로 뒤집힌 전례가 있다.
            # Triangle("1h"),

            # 헤드앤숄더 — 5분봉만. 지정가 전환 시 +0.22%(164건)로
            # 유일하게 플러스였다. 1시간봉은 -0.89%로 반대.
            # 대세 역행을 막는다 — 상승장 숏 -38.0, 하락장 롱 -62.4가
            # 손실의 전부였고 추세 방향은 양쪽 다 플러스였다.
            # 숏에서 btc_aligned 품질 항목을 뺀다 — EXP-013.
            # 롱에서 vwap_aligned 품질 항목을 뺀다 — EXP-014.
            HeadShoulders(
                "5m", trend_align=True,
                drop_short_btc_alignment=True, drop_long_vwap_alignment=True,
            ),

            # 이격도 역추세 — 1시간봉이 양 종목에서 유의(+0.56%).
            # 4시간봉 국면 'none' 중 두 이평선이 붙어 있으면(진짜 방향
            # 없음) 진입하지 않는다 — EXP-011. 국면이 바뀐 지 5봉(20시간)
            # 안 됐으면 진입하지 않는다 — EXP-012.
            Kotegawa(
                "1h", require_regime=True, min_major_gap=0.015, min_regime_age=5,
            ),
        ],
    }


def build_btc_candidates() -> dict[str, list]:
    """실시간에 넣기 전 독립 측정할 BTC 후보 전략.

    build_strategies와 분리해 둔다. 후보가 손실인 상태에서 클래스를 만들었다는
    이유만으로 다음 봇 재시작부터 주문이 나가면 안 된다.
    """

    from strategies.btc_pullback_reversal import BtcPullbackReversal
    from strategies.btc_trend_continuation import BtcTrendContinuation

    return {"BTC_USDT": [BtcPullbackReversal(), BtcTrendContinuation()]}


def select_strategies(strategies: dict[str, list], spec: str) -> dict[str, list]:
    """PAPER_STRATEGIES로 이 프로세스가 돌릴 전략만 남긴다.

    쉼표 구분 접두사 목록. "btc_regime_25x" 는 그 전략 하나,
    "xrp_patterns" 는 XRP 패턴 전부. 빈 값이면 전부(단일 계좌).
    계좌를 전략별로 나누는 용도라 아무것도 안 남으면 설정 실수다.
    """

    prefixes = tuple(p.strip() for p in spec.split(",") if p.strip())
    if not prefixes:
        return strategies
    out = {
        contract: [s for s in group if s.name.startswith(prefixes)]
        for contract, group in strategies.items()
    }
    out = {k: v for k, v in out.items() if v}
    if not out:
        raise SystemExit(f"PAPER_STRATEGIES에 맞는 전략이 없다: {spec!r}")
    return out


def main() -> int:
    account = os.environ.get("PAPER_ACCOUNT", "")
    strategies = select_strategies(
        build_strategies(), os.environ.get("PAPER_STRATEGIES", "")
    )
    engine = Engine(strategies=strategies)
    tags = [s.name for group in engine.strategies.values() for s in group]
    label = f" [{account}]" if account else ""

    print(f"시작{label} — 잔고 {engine.bot_state.balance:,.2f} USDT")
    print(f"종목 {[s.contract for s in engine.symbols]}")
    print(f"전략 {tags}")
    engine.notifier.text(
        f"🟢 페이퍼 봇 시작{label}\n"
        f"잔고 {engine.bot_state.balance:,.2f} USDT · 전략 {', '.join(tags)}"
    )
    try:
        engine.run()
    finally:
        engine.notifier.text(f"🔴 페이퍼 봇 종료{label}")
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
