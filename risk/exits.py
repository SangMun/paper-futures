"""손절·익절·트레일링.

전략은 구조에서 손절 후보(`stop_hint`)와 1차 목표 후보(`target_hint`)만 낸다.
최종 위치는 여기서 정한다. 규칙이 전략마다 흩어지면 어떤 전략이 어떤
청산 규칙을 쓰는지 추적할 수 없게 되기 때문이다.

익절 방식은 BTC와 XRP가 같다 (명세 §7).
  VPVR HVN에서 절반 익절 -> 손절을 본전으로 이동 -> 나머지는 트레일링
HVN을 못 찾은 구간에서만 손절 거리의 배수(R)로 떨어진다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from config.settings import FEES, RISK, SCALP, RiskConfig, ScalpConfig


Side = Literal["long", "short"]

# 청산 사유. 로그 집계 단위가 된다.
EXIT_STOP = "stop"
EXIT_TP1 = "tp1"
EXIT_TRAIL = "trail"
EXIT_TIME = "time"
EXIT_REGIME = "regime"

# HVN을 못 찾았을 때 쓰는 대체 R 배수 (BTC 기준).
# XRP는 ScalpConfig.tp1_r_multiple을 쓴다.
DEFAULT_TP1_R = 1.5


class ExitError(ValueError):
    pass


@dataclass(frozen=True)
class ExitPlan:
    """진입 시점에 확정되는 청산 계획."""

    stop: float
    tp1: float
    partial_ratio: float        # 1차 목표에서 청산할 비율
    trail_atr_mult: float
    max_hold_bars: int
    time_exit_enabled: bool = True
    # TP1 뒤 남은 물량 손절을 진입가에서 비용만큼 유리하게 옮긴다.
    breakeven_offset_pct: float = 0.0

    def reward_risk(self, entry: float) -> float:
        """1차 목표까지의 손익비. 수수료 차감 전."""

        risk = abs(entry - self.stop)
        return abs(self.tp1 - entry) / risk if risk > 0 else 0.0


@dataclass
class PositionState:
    """보유 중 갱신되는 값. 엔진이 소유하고 여기 함수들이 읽고 고친다."""

    side: Side
    entry: float
    plan: ExitPlan
    bars_held: int = 0
    remaining_ratio: float = 1.0        # 남은 물량 비율
    tp1_filled: bool = False
    extreme: float | None = None        # 진입 후 최고가(롱) / 최저가(숏)
    stop: float | None = None           # 현재 손절 (트레일링으로 이동)

    def __post_init__(self):
        if self.stop is None:
            self.stop = self.plan.stop
        if self.extreme is None:
            self.extreme = self.entry


@dataclass(frozen=True)
class ExitEvent:
    reason: str
    price: float
    ratio: float                # 이번에 청산할 비율
    note: str = ""


def _positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ExitError(f"{name}은 0보다 큰 유한값이어야 한다: {value}")


def _validate_side(side: str) -> None:
    if side not in ("long", "short"):
        raise ExitError(f"side는 long/short: {side!r}")


def _validate_bar(open_: float, high: float, low: float, close: float) -> None:
    for name, value in (("open", open_), ("high", high), ("low", low), ("close", close)):
        _positive_finite(name, value)
    if low > high or low > min(open_, close) or high < max(open_, close):
        raise ExitError(
            f"OHLC 범위 오류: open={open_}, high={high}, low={low}, close={close}"
        )


def _validate_slippage(slippage_rate: float) -> None:
    if not math.isfinite(slippage_rate) or not 0 <= slippage_rate < 1:
        raise ExitError(f"슬리피지율은 0 이상 1 미만이어야 한다: {slippage_rate}")


def _buffered_stop(side: Side, stop_hint: float, atr: float, mult: float) -> float:
    """구조적 손절선에 ATR 버퍼를 더한다.

    레벨에 딱 붙여 두면 휩쏘 한 번에 털린다. 되돌림을 확인하는 자리라
    잠깐 뚫었다가 회복하는 움직임이 오히려 정상이다.
    """

    return stop_hint - atr * mult if side == "long" else stop_hint + atr * mult


def build_plan(
    side: Side,
    entry: float,
    stop_hint: float,
    atr: float,
    target_hint: float | None = None,
    scalp: bool = False,
    risk: RiskConfig = RISK,
    scalp_cfg: ScalpConfig = SCALP,
    stop_buffer_atr_mult: float | None = None,
    partial_exit_ratio: float | None = None,
    trail_atr_mult: float | None = None,
    max_hold_bars: int | None = None,
    time_exit_enabled: bool = True,
    breakeven_offset_pct: float = 0.0,
    min_target_r: float | None = None,
) -> ExitPlan:
    """진입 시점의 청산 계획.

    scalp=True면 XRP 규칙(고정 R 배수)을, 아니면 BTC 규칙(구조적 목표)을 쓴다.
    target_hint가 없으면 양쪽 다 R 배수로 떨어진다.
    """

    _validate_side(side)
    _positive_finite("진입가", entry)
    _positive_finite("손절 힌트", stop_hint)
    _positive_finite("ATR", atr)
    if target_hint is not None:
        _positive_finite("목표가 힌트", target_hint)
    partial = risk.partial_exit_ratio if partial_exit_ratio is None else partial_exit_ratio
    if not math.isfinite(partial) or not 0 < partial <= 1:
        raise ExitError(f"부분 익절 비율 범위 오류: {partial}")

    default_buffer = scalp_cfg.atr_buffer_mult if scalp else risk.atr_buffer_mult
    default_trail = scalp_cfg.trail_atr_mult if scalp else risk.trail_atr_mult
    buffer_mult = default_buffer if stop_buffer_atr_mult is None else stop_buffer_atr_mult
    trail_mult = default_trail if trail_atr_mult is None else trail_atr_mult
    max_hold = (scalp_cfg.max_hold_bars if scalp else 96) if max_hold_bars is None else max_hold_bars
    if not math.isfinite(buffer_mult) or buffer_mult < 0:
        raise ExitError(f"ATR 손절 버퍼 배수는 0 이상의 유한값이어야 한다: {buffer_mult}")
    _positive_finite("트레일 ATR 배수", trail_mult)
    if not isinstance(max_hold, int) or max_hold <= 0:
        raise ExitError(f"최대 보유 봉 수는 양의 정수여야 한다: {max_hold}")
    if not isinstance(time_exit_enabled, bool):
        raise ExitError(f"시간청산 사용 여부는 bool이어야 한다: {time_exit_enabled!r}")
    if not math.isfinite(breakeven_offset_pct) or not 0 <= breakeven_offset_pct < 1:
        raise ExitError(f"본전 비용 버퍼 범위 오류: {breakeven_offset_pct}")
    target_r_floor = risk.min_target_r if min_target_r is None else min_target_r
    if not math.isfinite(target_r_floor) or target_r_floor < 0:
        raise ExitError(f"최소 목표 R은 0 이상의 유한값이어야 한다: {target_r_floor}")
    stop = _buffered_stop(side, stop_hint, atr, buffer_mult)
    _positive_finite("최종 손절가", stop)

    if (side == "long" and stop >= entry) or (side == "short" and stop <= entry):
        raise ExitError(f"버퍼를 더한 손절이 진입가를 넘었다: entry={entry}, stop={stop}")

    stop_distance = abs(entry - stop)

    # 1차 목표는 BTC·XRP 모두 VPVR HVN(= target_hint)을 쓴다.
    #
    # 처음엔 XRP만 고정 R 배수로 뒀는데, "5분봉 VPVR은 노이즈"라는 근거 없는
    # 판단이었다. 실측하니 스윕 신호 707건에서 HVN 중앙값 3개, 목표까지
    # 중앙값 2.43R로 고정 1.5R보다 오히려 나았다.
    #
    # HVN이 없는 구간(실측 21%)에서만 R 배수로 떨어진다.
    r_mult = scalp_cfg.tp1_r_multiple if scalp else DEFAULT_TP1_R
    fallback = entry + stop_distance * r_mult * (1 if side == "long" else -1)

    if target_hint is None:
        tp1 = fallback
    else:
        tp1 = target_hint
        # 목표가 진입가보다 불리한 쪽에 있으면 계획이 성립하지 않는다.
        if (side == "long" and tp1 <= entry) or (side == "short" and tp1 >= entry):
            raise ExitError(f"{side} 목표가 진입가 반대편이다: entry={entry}, tp1={tp1}")

        # 목표가 손절보다 가까우면(실측 22%) 이겨도 잃는 것보다 적게 번다.
        # 그런 자리는 잡지 않는다. 목표를 억지로 늘리지 않는 이유는,
        # HVN에서 가격이 멈출 가능성이 높은데 그 너머를 노리는 게
        # 근거 없는 낙관이기 때문이다.
        reward = abs(tp1 - entry)
        # 고정 2% 손절 / 1% 목표처럼 경계값이 정확히 0.5R인 경우
        # 부동소수점 오차만으로 거절하지 않는다.
        if reward + entry * 1e-12 < stop_distance * target_r_floor:
            raise ExitError(
                f"목표가 손절보다 가깝다: {reward / stop_distance:.2f}R "
                f"< 하한 {target_r_floor}R"
            )

    return ExitPlan(
        stop=stop,
        tp1=tp1,
        partial_ratio=partial,
        trail_atr_mult=trail_mult,
        max_hold_bars=max_hold,
        time_exit_enabled=time_exit_enabled,
        breakeven_offset_pct=breakeven_offset_pct,
    )


def update_trailing(state: PositionState, high: float, low: float, atr: float) -> float:
    """이번 봉을 반영해 손절을 옮긴다. 되돌리지 않는다.

    트레일링은 1차 익절이 체결된 뒤에만 작동한다. 그 전에는 구조적 손절이
    유지돼야 한다. 진입 직후부터 따라붙이면 정상적인 되돌림에 털린다.
    """

    _validate_side(state.side)
    _positive_finite("고가", high)
    _positive_finite("저가", low)
    _positive_finite("ATR", atr)
    if low > high:
        raise ExitError(f"고가/저가 범위 오류: high={high}, low={low}")
    if state.stop is None:
        raise ExitError("현재 손절가가 없다")
    _positive_finite("현재 손절가", state.stop)

    extreme = state.extreme if state.extreme is not None else state.entry
    _positive_finite("극값", extreme)
    if state.side == "long":
        state.extreme = max(extreme, high)
    else:
        state.extreme = min(extreme, low)

    if not state.tp1_filled:
        return state.stop

    trail = (
        state.extreme - atr * state.plan.trail_atr_mult
        if state.side == "long"
        else state.extreme + atr * state.plan.trail_atr_mult
    )

    # 손절은 유리한 방향으로만 움직인다.
    if state.side == "long":
        state.stop = max(state.stop, trail)
    else:
        state.stop = min(state.stop, trail)
    return state.stop


def check_exit(
    state: PositionState,
    open_: float,
    high: float,
    low: float,
    close: float,
    slippage_rate: float = FEES.slippage_rate,
    allow_time_exit: bool = True,
) -> ExitEvent | None:
    """이번 봉에서 발생한 청산. 없으면 None.

    같은 봉에서 손절과 익절이 모두 닿을 수 있다. 봉 안의 순서를 알 수 없으므로
    **손절을 먼저 본다.** 유리한 쪽을 고르면 과거 검증 결과가 실제보다 좋아진다.

    allow_time_exit=False는 실행 시계(틱)에서 쓴다. 시간 손절은 보유 **봉** 수
    기준이고 bars_held는 봉 시계만 센다. 틱에서도 판정하면 같은 봉 안에서
    수십 번 평가되어 시간 손절이 봉 경계와 무관하게 터진다.
    """

    _validate_side(state.side)
    _validate_bar(open_, high, low, close)
    _validate_slippage(slippage_rate)
    if state.stop is None:
        raise ExitError("현재 손절가가 없다")
    _positive_finite("현재 손절가", state.stop)
    _positive_finite("1차 목표가", state.plan.tp1)
    _positive_finite("진입가", state.entry)
    if not math.isfinite(state.remaining_ratio) or not 0 < state.remaining_ratio <= 1:
        raise ExitError(f"잔여 비율 범위 오류: {state.remaining_ratio}")
    if not isinstance(state.bars_held, int) or state.bars_held < 0:
        raise ExitError(f"보유 봉 수 범위 오류: {state.bars_held}")

    long = state.side == "long"

    stop_hit = low <= state.stop if long else high >= state.stop
    if stop_hit:
        reason = EXIT_TRAIL if state.tp1_filled else EXIT_STOP
        gap = open_ <= state.stop if long else open_ >= state.stop
        raw_fill = open_ if gap else state.stop
        fill = (
            raw_fill * (1.0 - slippage_rate)
            if long
            else raw_fill * (1.0 + slippage_rate)
        )
        return ExitEvent(
            reason,
            fill,
            state.remaining_ratio,
            note="갭 손절" if gap else "손절 시장가",
        )

    if not state.tp1_filled:
        tp_hit = high >= state.plan.tp1 if long else low <= state.plan.tp1
        if tp_hit:
            ratio = min(state.plan.partial_ratio, state.remaining_ratio)
            return ExitEvent(EXIT_TP1, state.plan.tp1, ratio)

    if (
        allow_time_exit
        and state.plan.time_exit_enabled
        and state.bars_held >= state.plan.max_hold_bars
    ):
        fill = close * (1.0 - slippage_rate if long else 1.0 + slippage_rate)
        return ExitEvent(
            EXIT_TIME,
            fill,
            state.remaining_ratio,
            note=f"{state.bars_held}봉 경과",
        )

    return None


def process_bar(
    state: PositionState,
    open_: float,
    high: float,
    low: float,
    close: float,
    atr: float,
    slippage_rate: float = FEES.slippage_rate,
) -> ExitEvent | None:
    """완료된 봉 하나를 보수적인 순서로 처리한다.

    먼저 직전 봉까지 확정된 손절·목표를 검사한다. 아무 청산도 없을 때만
    이번 봉의 극값으로 트레일을 갱신하므로, 새 손절은 다음 봉부터 유효하다.
    이 순서를 지키면 종가에야 알 수 있는 ATR/고가를 같은 봉 체결에 쓰는
    룩어헤드가 생기지 않는다.
    """

    event = check_exit(state, open_, high, low, close, slippage_rate)
    if event is not None:
        return event
    update_trailing(state, high, low, atr)
    return None


def process_tick(
    state: PositionState,
    open_: float,
    high: float,
    low: float,
    last: float,
    atr: float,
    slippage_rate: float = FEES.slippage_rate,
) -> ExitEvent | None:
    """봉 마감을 기다리지 않고 청산을 판정한다.

    넘기는 값은 봉이 아니라 **직전 관측 이후 지나온 구간**이다
    (`core.live.window_since`가 만든다). 진행 중인 봉의 누적 고저를 그대로
    넣으면 안 된다. 누적값은 순서 정보를 잃어서, 봉 전반에 찍힌 저점과
    봉 후반에 올라간 트레일을 비교하게 되고 없던 청산이 생긴다.

    ATR은 **확정봉에서 계산한 값**을 넘긴다. 진행 중인 봉으로 ATR을 구하면
    같은 시점을 다시 계산할 때 값이 달라져 재현이 불가능해진다.

    process_bar와 같은 순서를 지킨다. 먼저 직전까지 확정된 손절·목표를 보고,
    청산이 없을 때만 트레일을 갱신한다. 새 손절은 다음 틱부터 유효하다.
    """

    event = check_exit(
        state, open_, high, low, last, slippage_rate, allow_time_exit=False
    )
    if event is not None:
        return event
    update_trailing(state, high, low, atr)
    return None


def check_time_exit(
    state: PositionState,
    close: float,
    slippage_rate: float = FEES.slippage_rate,
) -> ExitEvent | None:
    """보유 **봉** 수 한도만 판정한다.

    실행 시계가 이미 전 구간을 감시한 봉에 쓴다. 그 봉의 OHLC로 손절·익절을
    다시 보면 안 된다 — 봉 안의 순서를 잃어서, 봉 후반에 올라간 트레일과
    봉 전반의 저가를 비교하게 되고 실제로는 없었던 청산이 생긴다.
    시간 손절만 봉 시계에 남는 이유는 bars_held가 봉 단위이기 때문이다.
    """

    _validate_side(state.side)
    _positive_finite("종가", close)
    _validate_slippage(slippage_rate)
    if not isinstance(state.bars_held, int) or state.bars_held < 0:
        raise ExitError(f"보유 봉 수 범위 오류: {state.bars_held}")

    if not state.plan.time_exit_enabled:
        return None
    if state.bars_held < state.plan.max_hold_bars:
        return None

    fill = close * (
        1.0 - slippage_rate if state.side == "long" else 1.0 + slippage_rate
    )
    return ExitEvent(
        EXIT_TIME,
        fill,
        state.remaining_ratio,
        note=f"{state.bars_held}봉 경과",
    )


def apply_tp1(state: PositionState, event: ExitEvent) -> None:
    """1차 익절 체결 반영. 손절을 본전으로 옮긴다.

    본전 이동은 명세 §7-2. 절반을 확정한 뒤 나머지가 손실로 끝나는 상황을
    막는다. 트레일링은 이 시점부터 작동하기 시작한다.
    """

    if event.reason != EXIT_TP1:
        raise ExitError(f"1차 익절 이벤트가 아니다: {event.reason}")
    _positive_finite("1차 익절 체결가", event.price)
    if not math.isfinite(event.ratio) or not 0 < event.ratio <= state.remaining_ratio:
        raise ExitError(
            f"1차 익절 비율 범위 오류: event={event.ratio}, remaining={state.remaining_ratio}"
        )

    state.remaining_ratio = max(0.0, state.remaining_ratio - event.ratio)
    state.tp1_filled = True
    state.stop = (
        state.entry * (1.0 + state.plan.breakeven_offset_pct)
        if state.side == "long"
        else state.entry * (1.0 - state.plan.breakeven_offset_pct)
    )
