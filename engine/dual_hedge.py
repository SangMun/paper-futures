"""EXP-015의 시간순 XRP 양방향 격리 페이퍼 엔진.

이 모듈은 신호를 사후 선택하지 않는다. 리플레이의 각 봉은 순서대로 한 번만
처리하고, 시장가 체결·수수료·슬리피지는 ``engine.broker``의 공통 구현을 쓴다.
과거 OHLC에는 봉 내부 순서·Mark Price·부분 청산 정보가 없으므로, 같은 봉에서
롱과 숏의 청산가가 모두 닿으면 보수적으로 양쪽 청산으로 처리한다.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace

import pandas as pd

from config.settings import CONTRACT_SPECS, DUAL_HEDGE, FEES, DualHedgeConfig, FeeAssumption
from engine.broker import (
    Broker, Fill, market_fill, round_down_to_tick, round_to_tick, round_up_to_tick,
)
from indicators.atr import atr as atr_fn
from risk.sizing import isolated_liquidation_price


Side = str


@dataclass
class HedgeLeg:
    side: Side
    fill: Fill
    liquidation_price: float
    margin: float
    # 청산가 직전 보호 스탑. None이면 청산까지 보유한다.
    stop_price: float | None = None


@dataclass
class HedgePair:
    pair_id: int
    opened_ts: int
    opened_balance: float
    contracts: int
    long: HedgeLeg
    short: HedgeLeg
    survivor: HedgeLeg | None = None
    liquidated_side: Side | None = None
    liquidated_ts: int | None = None
    profit_lock_price: float | None = None
    stop: float | None = None
    extreme: float | None = None
    armed: bool = False


@dataclass(frozen=True)
class PairResult:
    pair_id: int
    opened_ts: int
    closed_ts: int
    liquidation: str
    exit_reason: str
    pnl: float
    long_entry: float
    short_entry: float
    profit_lock_price: float | None


@dataclass
class DualHedgeResult:
    initial_balance: float
    final_balance: float
    unsettled_cash_balance: float
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    bars: int
    results: list[PairResult] = field(default_factory=list)
    open_pair: HedgePair | None = None

    @property
    def completed_pairs(self) -> int:
        return len(self.results)

    @property
    def return_pct(self) -> float:
        return (self.final_balance / self.initial_balance - 1) * 100 if self.initial_balance else 0.0

    @property
    def win_rate(self) -> float:
        return 100 * sum(r.pnl > 0 for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def realized_max_drawdown_pct(self) -> float:
        peak = self.initial_balance
        worst = 0.0
        balance = self.initial_balance
        for row in self.results:
            balance += row.pnl
            peak = max(peak, balance)
            worst = max(worst, (peak - balance) / peak * 100) if peak else worst
        return worst

    def by_reason(self) -> Counter:
        return Counter(row.exit_reason for row in self.results)

    def by_liquidation(self) -> Counter:
        return Counter(row.liquidation for row in self.results)

    def by_liquidation_stats(self) -> dict[str, dict[str, float | int]]:
        """처음 청산된 방향별 완료 pair 손익. 양쪽 청산은 별도 보존한다."""

        out: dict[str, dict[str, float | int]] = {}
        for row in self.results:
            stats = out.setdefault(row.liquidation, {"pairs": 0, "wins": 0, "pnl": 0.0})
            stats["pairs"] += 1
            stats["wins"] += int(row.pnl > 0)
            stats["pnl"] += row.pnl
        for stats in out.values():
            pairs = int(stats["pairs"])
            stats["win_rate"] = 100 * int(stats["wins"]) / pairs if pairs else 0.0
        return out


class DualHedgeReplay:
    """한 번에 하나의 양방향 pair만 유지하는 EXP-015 리플레이.

    진입은 전 봉의 정보를 전혀 쓰지 않는 기준선이다. 완료 뒤 ``reentry_delay``가
    지나면 다음 봉 시가에 다시 양방향 시장가 진입한다. 따라서 성과가 있더라도
    진입 신호의 우위가 아니라 청산 후 추세 지속이라는 가설만 시험한다.
    """

    def __init__(
        self,
        balance: float = 1_000.0,
        config: DualHedgeConfig = DUAL_HEDGE,
        fees: FeeAssumption = FEES,
    ):
        self.config = config
        self.fees = fees
        self.broker = Broker(balance=balance, fees=fees)
        self.initial_balance = balance
        self.pair: HedgePair | None = None
        self.results: list[PairResult] = []
        self._next_pair_id = 1
        self._eligible_at = 0
        self._eligible_live_ts = 0

        if config.symbol not in CONTRACT_SPECS:
            raise ValueError(f"계약 스펙 없음: {config.symbol}")
        if not 0 < config.pair_margin_fraction <= 1:
            raise ValueError("pair_margin_fraction은 0 초과 1 이하여야 한다")
        if config.leverage < 2:
            raise ValueError("EXP-015 레버리지는 2 이상이어야 한다")
        if config.trail_atr_mult <= 0:
            raise ValueError("trail_atr_mult는 0보다 커야 한다")

    @property
    def _multiplier(self) -> float:
        return float(CONTRACT_SPECS[self.config.symbol]["multiplier"])

    def _open_pair(
        self,
        ts: int,
        price: float,
        maintenance_rate: float | None = None,
        first_tier_risk_limit: float | None = None,
    ) -> None:
        if first_tier_risk_limit is not None and (
            not math.isfinite(first_tier_risk_limit) or first_tier_risk_limit <= 0
        ):
            raise ValueError(f"첫 risk-limit은 0보다 큰 유한값이어야 합니다: {first_tier_risk_limit}")
        total_margin = self.broker.balance * self.config.pair_margin_fraction
        margin_per_leg = total_margin / 2
        contracts = int(
            margin_per_leg * self.config.leverage / (price * self._multiplier)
        )
        if contracts < int(CONTRACT_SPECS[self.config.symbol]["min_size"]):
            return

        before = self.broker.balance
        long_fill = market_fill(self.config.symbol, "long", price, contracts, self.fees)
        short_fill = market_fill(self.config.symbol, "short", price, contracts, self.fees)
        if first_tier_risk_limit is not None and max(long_fill.notional, short_fill.notional) > first_tier_risk_limit:
            # 상위 구간 MMR·deduction은 이 실험에서 재현하지 않는다. 루프를 죽이지
            # 않고 진입만 막아, 열린 pair가 없는 상태로 관측을 계속한다.
            return
        self.broker.balance -= long_fill.fee + short_fill.fee
        # 리플레이에는 과거 Mark Price·계약 MMR 이력이 없어 설정의 첫 구간 값을
        # 보존한다. 실시간 실행은 Gate 공개 API에서 관측한 값을 넘긴다.
        maintenance = (
            float(CONTRACT_SPECS[self.config.symbol]["maintenance_rate"])
            if maintenance_rate is None else maintenance_rate
        )
        self.pair = HedgePair(
            pair_id=self._next_pair_id,
            opened_ts=ts,
            opened_balance=before,
            contracts=contracts,
            long=HedgeLeg(
                "long", long_fill,
                isolated_liquidation_price(
                    long_fill.price, "long", margin_per_leg, contracts,
                    self._multiplier, maintenance,
                ),
                margin_per_leg,
            ),
            short=HedgeLeg(
                "short", short_fill,
                isolated_liquidation_price(
                    short_fill.price, "short", margin_per_leg, contracts,
                    self._multiplier, maintenance,
                ),
                margin_per_leg,
            ),
        )
        for leg in (self.pair.long, self.pair.short):
            leg.stop_price = self._liquidation_stop(leg)
        self._next_pair_id += 1

    def refresh_liquidation_prices(self, maintenance_rate: float) -> None:
        """현재 Gate 1단계 MMR로 열린 격리 다리의 추정 청산 Mark Price를 갱신한다.

        계약 설정이 바뀌면 Gate의 기준도 즉시 달라질 수 있다. 포지션 규모가
        첫 risk-limit보다 작은 이 실험에서는 계약 API의 1단계 MMR을 적용한다.
        """

        if not math.isfinite(maintenance_rate) or not 0 <= maintenance_rate < 1:
            raise ValueError(f"유지증거금률 범위 오류: {maintenance_rate}")
        if self.pair is None:
            return
        for leg in (self.pair.long, self.pair.short):
            leg.liquidation_price = isolated_liquidation_price(
                leg.fill.price, leg.side, leg.margin, self.pair.contracts,
                self._multiplier, maintenance_rate,
            )
            leg.stop_price = self._liquidation_stop(leg)

    def _liquidation_stop(self, leg: HedgeLeg) -> float | None:
        """청산가보다 ``liquidation_stop_buffer``만큼 진입가 쪽에 둔 보호 스탑."""

        buffer = self.config.liquidation_stop_buffer
        if buffer is None:
            return None
        if leg.side == "long":
            stop = round_up_to_tick(leg.liquidation_price * (1.0 + buffer), self.config.symbol)
            valid = stop < leg.fill.price
        else:
            stop = round_down_to_tick(leg.liquidation_price * (1.0 - buffer), self.config.symbol)
            valid = stop > leg.fill.price
        if not valid:
            raise ValueError(
                f"liquidation_stop_buffer={buffer}가 진입가와 청산가 사이를 넘습니다: "
                f"{leg.side} entry={leg.fill.price}, liq={leg.liquidation_price}"
            )
        return stop

    def _leg_trigger(
        self, leg: HedgeLeg, open_: float, high: float, low: float,
    ) -> tuple[str, float] | None:
        """이 구간에서 다리가 보호 스탑 또는 청산으로 끝났는지.

        가격 경로는 연속이므로 청산가보다 진입가 쪽에 있는 스탑에 먼저 닿는다고 본다.
        구간 시작가가 이미 청산가를 넘었을 때(갭)만 스탑을 건너뛰고 청산으로 처리한다.
        스탑을 넘어 시작했으면 시작가에서 시장가로 체결한다.
        """

        is_long = leg.side == "long"
        liq = leg.liquidation_price
        gapped = open_ <= liq if is_long else open_ >= liq
        if leg.stop_price is None or gapped:
            hit = low <= liq if is_long else high >= liq
            return ("liquidation", 0.0) if hit else None
        stop = leg.stop_price
        if not (low <= stop if is_long else high >= stop):
            return None
        return ("stop", min(open_, stop) if is_long else max(open_, stop))

    def _take_leg_loss(self, leg: HedgeLeg, kind: str, reference: float, contracts: int) -> None:
        if kind == "liquidation":
            # Gate 격리 청산은 청산 수수료를 포함한 파산가에서 배정 증거금 안에서
            # 끝나고, 남은 유지증거금은 보험기금으로 간다. 손실은 증거금까지다.
            self.broker.balance -= leg.margin
            return
        fill = self._market_close(leg, reference)
        self.broker.close(leg.side, self.config.symbol, leg.fill.price, fill, contracts)

    def _resolve_hedged(
        self, pair: HedgePair, open_: float, high: float, low: float, ts: int,
    ) -> str | None:
        """양쪽이 모두 열린 pair의 스탑·청산을 처리하고 사건 이름을 돌려준다."""

        hits = {leg.side: self._leg_trigger(leg, open_, high, low) for leg in (pair.long, pair.short)}
        if hits["long"] and hits["short"]:
            # 같은 구간에서 양쪽 모두 닿으면 순서를 알 수 없다. 한쪽 생존을 고르지 않는다.
            for leg in (pair.long, pair.short):
                self._take_leg_loss(leg, *hits[leg.side], pair.contracts)
            event = (
                "double_liquidation"
                if all(kind == "liquidation" for kind, _ in hits.values()) else "double_stop"
            )
            self._close_pair(ts, event, 0.0, "both")
            return event
        loser = pair.long if hits["long"] else pair.short if hits["short"] else None
        if loser is None:
            return None
        kind, reference = hits[loser.side]
        self._take_leg_loss(loser, kind, reference, pair.contracts)
        pair.survivor = pair.short if loser.side == "long" else pair.long
        pair.liquidated_side = loser.side
        pair.liquidated_ts = ts
        # 손익분기가는 스탑·청산 뒤 실제 잔고로 계산하므로 남은 증거금이 반영된다.
        pair.profit_lock_price = self._profit_lock_price(pair, pair.survivor)
        return "liquidation" if kind == "liquidation" else "loser_stop"

    def _resolve_survivor_exit(self, open_: float, high: float, low: float, ts: int) -> str | None:
        """생존 다리가 반대로 가서 보호 스탑 또는 청산에 닿았는지."""

        pair = self.pair
        trigger = self._leg_trigger(pair.survivor, open_, high, low)
        if trigger is None:
            return None
        kind, reference = trigger
        if kind == "liquidation":
            self._survivor_liquidation(ts)
            return "survivor_liquidation"
        self._close_pair(ts, "survivor_stop", reference, "both_sequential")
        return "survivor_stop"

    def _market_close(self, leg: HedgeLeg, reference_price: float) -> float:
        """시장가 종료의 실제 체결가를 반환한다. 방향별 불리한 슬리피지를 붙인다."""

        slipped = reference_price * (
            1.0 - self.fees.slippage_rate if leg.side == "long" else 1.0 + self.fees.slippage_rate
        )
        # 롱 종료는 매도, 숏 종료는 매수다. 최근접 호가로 반올림하면 손익 보장가
        # 아래(롱)·위(숏)에 실제 체결된 것처럼 되어 비용 보장이 깨질 수 있다.
        return (
            round_down_to_tick(slipped, self.config.symbol)
            if leg.side == "long"
            else round_up_to_tick(slipped, self.config.symbol)
        )

    def _profit_lock_price(self, pair: HedgePair, survivor: HedgeLeg) -> float:
        """전량 종료해도 pair 순익이 ``profit_buffer`` 이상인 관측 가격."""

        required = pair.opened_balance + self.config.profit_buffer - self.broker.balance
        units = pair.contracts * self._multiplier
        if survivor.side == "long":
            denominator = (1.0 - self.fees.slippage_rate) * (1.0 - self.fees.taker_rate)
            raw = (required / units + survivor.fill.price) / denominator
            # 롱 종료의 최소 관측 가격이므로 가장 가까운 호가가 아니라 올림한다.
            lock = round_up_to_tick(raw, self.config.symbol)
        else:
            denominator = (1.0 + self.fees.slippage_rate) * (1.0 + self.fees.taker_rate)
            raw = (survivor.fill.price - required / units) / denominator
            # 숏 종료의 최대 관측 가격이므로 가장 가까운 호가가 아니라 내림한다.
            lock = round_down_to_tick(raw, self.config.symbol)

        # 관측가를 호가에 맞췄더라도, 실제 시장가 종료는 슬리피지 뒤 다시
        # 불리한 호가로 맞춰진다. 그 두 번째 반올림 때문에 이론상 손익분기보다
        # 한 틱 모자랄 수 있으므로, 실제 종료 모형으로 마지막 보정을 한다.
        tick = float(CONTRACT_SPECS[self.config.symbol]["price_tick"])
        for _ in range(10):
            exit_price = self._market_close(survivor, lock)
            gross = (
                (exit_price - survivor.fill.price) * units
                if survivor.side == "long"
                else (survivor.fill.price - exit_price) * units
            )
            net = gross - exit_price * units * self.fees.taker_rate
            if net + 1e-12 >= required:
                return lock
            lock = (
                round_up_to_tick(lock + tick, self.config.symbol)
                if survivor.side == "long"
                else round_down_to_tick(lock - tick, self.config.symbol)
            )
        raise RuntimeError("이익 보장가를 호가 단위 안에서 계산하지 못했습니다")

    def _close_pair(self, ts: int, reason: str, reference_price: float, liquidation: str) -> None:
        pair = self.pair
        if pair is None:
            return
        if pair.survivor is not None:
            fill = self._market_close(pair.survivor, reference_price)
            self.broker.close(
                pair.survivor.side, self.config.symbol, pair.survivor.fill.price,
                fill, pair.contracts,
            )
        pnl = self.broker.balance - pair.opened_balance
        self.results.append(PairResult(
            pair_id=pair.pair_id,
            opened_ts=pair.opened_ts,
            closed_ts=ts,
            liquidation=liquidation,
            exit_reason=reason,
            pnl=pnl,
            long_entry=pair.long.fill.price,
            short_entry=pair.short.fill.price,
            profit_lock_price=pair.profit_lock_price,
        ))
        self.pair = None

    def _survivor_liquidation(self, ts: int) -> None:
        """청산 뒤 남은 다리도 반전으로 청산된 경우.

        격리 증거금 모형에서 생존 다리는 시간 종료 때 무한 손실을 낼 수 없다.
        원래 청산가에 닿으면 그 다리도 배정 증거금 전액만 잃고, 시장가 종료
        손익을 한 번 더 더하지 않는다.
        """

        pair = self.pair
        if pair is None or pair.survivor is None:
            return
        survivor = pair.survivor
        self.broker.balance -= survivor.margin
        pair.survivor = None
        self._close_pair(ts, "survivor_liquidation", 0.0, "both_sequential")

    def process_live_window(
        self,
        *,
        ts: int,
        open_: float,
        high: float,
        low: float,
        last: float,
        atr: float,
        allow_entry: bool,
        mark_open: float | None = None,
        mark_high: float | None = None,
        mark_low: float | None = None,
        mark_last: float | None = None,
        maintenance_rate: float | None = None,
        first_tier_risk_limit: float | None = None,
    ) -> str | None:
        """실시간 관측 구간 하나를 처리한다.

        ``high``·``low``는 직전 관측 이후 실제로 지나온 **거래가** 범위여야 한다. 호출부는
        ``core.live.window_since``를 사용해 누적 봉 고저를 그대로 넣지 않는다.
        새 pair는 현 시세 ``last``에 열고, 그 전에 지나간 이 구간에는 소급 체결·
        소급 청산을 적용하지 않는다. 청산 판정만 Gate Mark Price 구간
        (``mark_*``)을 쓴다. 인자를 생략한 단위 테스트·과거 리플레이 호환 경로는
        거래가를 대신 쓰며 실제 Mark Price 검증으로 해석하면 안 된다.
        """

        if not all(math.isfinite(v) and v > 0 for v in (open_, high, low, last)):
            return None
        mark_open = open_ if mark_open is None else mark_open
        mark_high = high if mark_high is None else mark_high
        mark_low = low if mark_low is None else mark_low
        mark_last = last if mark_last is None else mark_last
        if not all(math.isfinite(v) and v > 0 for v in (mark_open, mark_high, mark_low, mark_last)):
            return None
        if mark_low > mark_high or mark_high < max(mark_open, mark_last) or mark_low > min(mark_open, mark_last):
            raise ValueError("Mark Price 관측 구간의 고저 범위가 올바르지 않습니다")

        if self.pair is None:
            if allow_entry and ts >= self._eligible_live_ts:
                self._open_pair(
                    ts, last, maintenance_rate, first_tier_risk_limit,
                )
                # 최소 수량 미달·risk-limit 초과로 열지 않았으면 진입 사건이 아니다.
                return "entry" if self.pair is not None else None
            return None

        if maintenance_rate is not None:
            self.refresh_liquidation_prices(maintenance_rate)
        pair = self.pair
        if pair.survivor is None:
            # 보호 스탑도 Gate의 Mark Price 트리거로 둔다고 보고 청산과 같은 가격으로 판정한다.
            event = self._resolve_hedged(pair, mark_open, mark_high, mark_low, ts)
            if event in ("double_liquidation", "double_stop"):
                self._eligible_live_ts = ts + 1
            # 한쪽 종료와 이익 보장가 통과의 구간 안 순서는 알 수 없다.
            # 다음 관측부터만 이익 보장 스탑을 활성화한다.
            return event

        survivor = pair.survivor
        lock = pair.profit_lock_price
        assert lock is not None

        # 이미 거래소에 보호 스탑이 놓인 상태라면, 더 먼 원래 청산가보다 먼저
        # 그 스탑이 체결된다. 이 순서를 뒤집으면 한 구간에서 스탑과 청산가를
        # 모두 지난 경우에도 전액 청산으로 과대 손실을 기록하게 된다.
        if pair.armed:
            assert pair.stop is not None and pair.extreme is not None
            stop_hit = low <= pair.stop if survivor.side == "long" else high >= pair.stop
            if stop_hit:
                reference = min(open_, pair.stop) if survivor.side == "long" else max(open_, pair.stop)
                self._close_pair(ts, "profit_lock_trail", reference, pair.liquidated_side or "?")
                self._eligible_live_ts = ts + 1
                return "profit_lock_trail"

        event = self._resolve_survivor_exit(mark_open, mark_high, mark_low, ts)
        if event is not None:
            self._eligible_live_ts = ts + 1
            return event

        timed_out = (
            pair.liquidated_ts is not None
            and ts - pair.liquidated_ts >= self.config.max_survivor_hold_seconds
        )
        if not pair.armed:
            if timed_out:
                self._close_pair(ts, "survivor_timeout", last, pair.liquidated_side or "?")
                self._eligible_live_ts = ts + 1
                return "survivor_timeout"
            reached = high >= lock if survivor.side == "long" else low <= lock
            if reached:
                pair.armed = True
                pair.stop = lock
                pair.extreme = high if survivor.side == "long" else low
                # 구간 끝값이 lock 반대편에 있으면, lock 도달 뒤 다시 통과한
                # 순서는 확정적이다. 이 경우 다음 폴링까지 기다려 스탑을 놓치지
                # 않고 현재 관측가 기준의 보수적 시장가 종료를 기록한다.
                recrossed = last <= lock if survivor.side == "long" else last >= lock
                if recrossed:
                    reference = min(last, lock) if survivor.side == "long" else max(last, lock)
                    self._close_pair(ts, "profit_lock_trail", reference, pair.liquidated_side or "?")
                    self._eligible_live_ts = ts + 1
                    return "profit_lock_trail"
                return "profit_lock_armed"
            return None

        if timed_out:
            self._close_pair(ts, "survivor_timeout", last, pair.liquidated_side or "?")
            self._eligible_live_ts = ts + 1
            return "survivor_timeout"

        if not math.isfinite(atr) or atr <= 0:
            return None
        if survivor.side == "long":
            pair.extreme = max(pair.extreme, high)
            pair.stop = max(lock, pair.stop, pair.extreme - atr * self.config.trail_atr_mult)
        else:
            pair.extreme = min(pair.extreme, low)
            pair.stop = min(lock, pair.stop, pair.extreme + atr * self.config.trail_atr_mult)
        return "trail_updated"

    def step(self, bar_index: int, bar: pd.Series, atr: float) -> None:
        """Pandas 행 하나를 받아 처리하는 단위 테스트용 공개 진입점."""

        self._step_values(
            bar_index,
            int(bar["ts"]),
            float(bar["open"]),
            float(bar["high"]),
            float(bar["low"]),
            float(bar["close"]),
            atr,
        )

    def _step_values(
        self,
        bar_index: int,
        ts: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        atr: float,
    ) -> None:
        """시가에 진입하고, 그 뒤 해당 봉의 고저·종가를 시간순으로 한 번 처리한다."""

        if not all(math.isfinite(v) and v > 0 for v in (open_, high, low, close)):
            return

        if self.pair is None and bar_index >= self._eligible_at:
            self._open_pair(ts, open_)
        pair = self.pair
        if pair is None:
            return

        if pair.survivor is None:
            event = self._resolve_hedged(pair, open_, high, low, ts)
            if event in ("double_liquidation", "double_stop"):
                self._eligible_at = bar_index + self.config.reentry_delay_bars
            if event is not None:
                # 이 봉 내부의 어느 시점에 청산·목표·반전이 있었는지 알 수 없다.
                # 같은 봉 수익을 사후 선택하지 않고 다음 봉부터 보호 주문을 활성화한다.
                return

        survivor = pair.survivor
        if survivor is None:
            return
        lock = pair.profit_lock_price
        assert lock is not None

        # 이미 활성화된 스탑은 원래 청산가보다 먼저 체결된다. 과거 봉의
        # 고저만으로도 스탑과 청산가를 함께 지난 경우에는 이 우선순위를
        # 지켜 전액 청산으로 과대 기록하지 않는다.
        if pair.armed:
            assert pair.stop is not None and pair.extreme is not None
            stop_hit = low <= pair.stop if survivor.side == "long" else high >= pair.stop
            if stop_hit:
                reference = min(open_, pair.stop) if survivor.side == "long" else max(open_, pair.stop)
                self._close_pair(ts, "profit_lock_trail", reference, pair.liquidated_side or "?")
                self._eligible_at = bar_index + self.config.reentry_delay_bars
                return

        if self._resolve_survivor_exit(open_, high, low, ts) is not None:
            self._eligible_at = bar_index + self.config.reentry_delay_bars
            return
        timed_out = (
            pair.liquidated_ts is not None
            and ts - pair.liquidated_ts >= self.config.max_survivor_hold_seconds
        )

        if not pair.armed:
            if timed_out:
                self._close_pair(ts, "survivor_timeout", close, pair.liquidated_side or "?")
                self._eligible_at = bar_index + self.config.reentry_delay_bars
                return
            reached = close >= lock if survivor.side == "long" else close <= lock
            if reached:
                pair.armed = True
                pair.stop = lock
                pair.extreme = close
            return

        if timed_out:
            self._close_pair(ts, "survivor_timeout", close, pair.liquidated_side or "?")
            self._eligible_at = bar_index + self.config.reentry_delay_bars
            return

        if not math.isfinite(atr) or atr <= 0:
            return
        if survivor.side == "long":
            pair.extreme = max(pair.extreme, high)
            pair.stop = max(lock, pair.stop, pair.extreme - atr * self.config.trail_atr_mult)
        else:
            pair.extreme = min(pair.extreme, low)
            pair.stop = min(lock, pair.stop, pair.extreme + atr * self.config.trail_atr_mult)

    def run(self, df: pd.DataFrame, warmup: int = 20) -> DualHedgeResult:
        required = {"ts", "dt", "open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"필수 캔들 열 없음: {sorted(missing)}")
        if len(df) <= warmup:
            raise ValueError("워밍업보다 캔들이 적다")

        atr_values = atr_fn(df, self.config.atr_period).to_numpy()
        # iloc로 Series를 28만 번 만들면 장기 리플레이가 실험 도구로 쓸 수
        # 없을 만큼 느려진다. 값만 순서대로 꺼내므로 미래 봉 참조는 없다.
        for index, row in enumerate(df.itertuples(index=False)):
            if index < warmup:
                continue
            self._step_values(
                index, int(row.ts), float(row.open), float(row.high),
                float(row.low), float(row.close), float(atr_values[index]),
            )

        # 마지막 봉 종가 이후에는 더 이상 관측할 가격이 없으므로, 미청산 pair는
        # 성과에 섞지 않고 별도로 돌려준다. 강제 종료 수익으로 꾸미지 않는다.
        # broker 잔고에는 열린 pair의 진입비·청산 다리 손실이 이미 빠져 있을 수
        # 있으므로, 완료 pair만 반영한 잔고와 분리해 돌려준다.
        completed_balance = self.initial_balance + sum(row.pnl for row in self.results)
        return DualHedgeResult(
            initial_balance=self.initial_balance,
            final_balance=completed_balance,
            unsettled_cash_balance=self.broker.balance,
            start=df["dt"].iloc[warmup],
            end=df["dt"].iloc[-1],
            bars=len(df) - warmup,
            results=list(self.results),
            open_pair=self.pair,
        )


def config_with(**changes) -> DualHedgeConfig:
    """도구와 테스트가 전역 실행 설정을 바꾸지 않고 비교값을 만들게 한다."""

    return replace(DUAL_HEDGE, **changes)
