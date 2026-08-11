"""포지션 크기와 레버리지 역산.

README 원칙 3: 레버리지는 결과값이다.
  손절 위치 -> 손절 거리 -> 포지션 크기 -> 레버리지

레버리지를 먼저 정하면 손실액이 시장 변동성에 따라 제멋대로 움직인다.
손절 거리에서 역산하면 어떤 종목이든 1회 손실이 계좌의 1%로 고정된다.

레버리지가 바꾸는 건 손익이 아니라 두 가지뿐이다.
  - 청산가까지의 거리
  - 묶이는 증거금
포지션 크기가 이미 리스크로 고정돼 있기 때문이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from config.settings import (
    CONTRACT_SPECS,
    RISK_MULTIPLIER,
    FEES,
    LEVERAGE_CAP,
    RISK,
    TRANCHE_RISK_WEIGHTS,
    FeeAssumption,
    RiskConfig,
)


class SizingError(ValueError):
    """입력 자체가 잘못된 경우. 진입 거절은 여기에 해당하지 않는다."""


# 진입을 포기하는 이유. 로그의 skip_reason과 같은 값을 쓴다.
REJECT_STOP_TOO_TIGHT = "stop_too_tight"
REJECT_STOP_TOO_WIDE = "stop_too_wide"
REJECT_SIZE_BELOW_MIN = "size_below_min"
REJECT_LEVERAGE_INFEASIBLE = "leverage_infeasible"

# 신호 단위 리스크 배수의 상한.
# 확신이 높은 자리에 더 걸 수 있게 하되, 표본이 적은 구간이 우연이었을 때의
# 손실도 같은 배수로 커지므로 2배를 넘기지 않는다.
MAX_SIGNAL_RISK_MULT = 2.0


@dataclass(frozen=True)
class ExecutionAssumption:
    """사이징에 사용하는 진입·손절 체결 비용 가정.

    예상 손실은 주문 방식에 따라 달라진다. 기본값은 양쪽 모두 시장가로
    체결되는 보수적인 기준이다. 메이커 리베이트는 체결되지 않을 수 있으므로
    사이징에서 이익으로 잡지 않는다.
    """

    entry_fee_rate: float
    exit_fee_rate: float
    entry_slippage_rate: float
    exit_slippage_rate: float
    name: str = "custom"

    @classmethod
    def taker(cls, fees: FeeAssumption = FEES) -> "ExecutionAssumption":
        return cls(
            entry_fee_rate=fees.taker_rate,
            exit_fee_rate=fees.taker_rate,
            entry_slippage_rate=fees.slippage_rate,
            exit_slippage_rate=fees.slippage_rate,
            name="taker",
        )

    @classmethod
    def maker_entry(cls, fees: FeeAssumption = FEES) -> "ExecutionAssumption":
        return cls(
            entry_fee_rate=max(0.0, fees.maker_rate),
            exit_fee_rate=fees.taker_rate,
            entry_slippage_rate=0.0,
            exit_slippage_rate=fees.slippage_rate,
            name="maker_entry",
        )

    @classmethod
    def stress(
        cls,
        fees: FeeAssumption = FEES,
        slippage_multiple: float = 3.0,
    ) -> "ExecutionAssumption":
        if not math.isfinite(slippage_multiple) or slippage_multiple < 1.0:
            raise SizingError(
                f"스트레스 슬리피지 배수는 1 이상 유한값이어야 한다: {slippage_multiple}"
            )
        return cls(
            entry_fee_rate=fees.taker_rate,
            exit_fee_rate=fees.taker_rate,
            entry_slippage_rate=fees.slippage_rate * slippage_multiple,
            exit_slippage_rate=fees.slippage_rate * slippage_multiple,
            name=f"stress_{slippage_multiple:g}x",
        )


@dataclass(frozen=True)
class Position:
    """진입 가능하다고 판정된 포지션. 거절이면 rejected를 본다."""

    contracts: int
    notional: float             # 명목 가치 (USDT)
    margin: float               # 실제로 묶이는 증거금
    leverage: int
    stop_distance: float        # 진입가 대비 비율
    risk_amount: float          # 손절 시 예상 총손실 (가격차 + 비용)
    price_risk_amount: float    # 가격 변동으로 인한 예상 손실
    estimated_cost: float       # 진입·청산 수수료와 슬리피지 영향
    entry_fill_price: float     # 사이징에 쓴 예상 진입 체결가
    stop_fill_price: float      # 사이징에 쓴 예상 손절 체결가
    liquidation_price: float
    liquidation_distance: float  # 진입가 대비 비율

    rejected: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.rejected is None

    def liquidation_multiple(self) -> float:
        """청산 거리가 손절 거리의 몇 배인가. 클수록 안전하다."""

        return self.liquidation_distance / self.stop_distance


def _rejected(reason: str, stop_distance: float, notes: list[str]) -> Position:
    return Position(
        contracts=0,
        notional=0.0,
        margin=0.0,
        leverage=0,
        stop_distance=stop_distance,
        risk_amount=0.0,
        price_risk_amount=0.0,
        estimated_cost=0.0,
        entry_fill_price=0.0,
        stop_fill_price=0.0,
        liquidation_price=0.0,
        liquidation_distance=0.0,
        rejected=reason,
        notes=notes,
    )


def tranche_risk(risk_per_trade: float, tranche: int, max_tranches: int) -> float:
    """분할 진입에서 이번 트랜치가 쓸 리스크 비율.

    트랜치마다 risk_per_trade를 새로 쓰면 3분할이 곧 3배 리스크가 된다.
    총량을 먼저 쪼갠 뒤 그중 한 몫을 쓴다.

    max_tranches가 가중치 개수와 다르면 앞에서부터 잘라 쓰고 합이 1이 되게
    다시 정규화한다. 전략이 2분할을 원할 때 설정을 새로 만들지 않아도 된다.
    """

    if not math.isfinite(risk_per_trade) or risk_per_trade <= 0:
        raise SizingError(f"트레이드 리스크는 0보다 큰 유한값이어야 한다: {risk_per_trade}")
    if tranche < 1 or tranche > max_tranches:
        raise SizingError(f"트랜치 번호가 범위를 벗어남: {tranche}/{max_tranches}")
    if max_tranches < 1:
        raise SizingError(f"max_tranches는 1 이상이어야 한다: {max_tranches}")

    if max_tranches == 1:
        return risk_per_trade

    weights = list(TRANCHE_RISK_WEIGHTS[:max_tranches])
    if len(weights) < max_tranches:
        # 가중치가 모자라면 남은 트랜치에 균등 배분
        weights += [weights[-1]] * (max_tranches - len(weights))

    total = sum(weights)
    return risk_per_trade * weights[tranche - 1] / total


def liquidation_price(
    entry: float,
    side: str,
    leverage: float,
    maintenance_rate: float,
) -> float:
    """격리 마진 기준 청산가.

    증거금이 유지증거금 아래로 내려가는 지점이다.
      롱: entry * (1 - 1/L + mr)
      숏: entry * (1 + 1/L - mr)

    유지증거금률을 빼먹고 1/L만 쓰면 청산 거리를 실제보다 멀게 본다.
    레버리지가 높을수록 그 차이가 커진다.
    """

    if side not in ("long", "short"):
        raise SizingError(f"side는 long/short: {side!r}")
    if not math.isfinite(entry) or entry <= 0:
        raise SizingError(f"진입가는 0보다 큰 유한값이어야 한다: {entry}")
    if not math.isfinite(leverage) or leverage <= 0:
        raise SizingError(f"레버리지는 0보다 커야 한다: {leverage}")
    if not math.isfinite(maintenance_rate) or not 0 <= maintenance_rate < 1:
        raise SizingError(f"유지증거금률 범위 오류: {maintenance_rate}")

    edge = 1.0 / leverage - maintenance_rate
    return entry * (1.0 - edge) if side == "long" else entry * (1.0 + edge)


def size_position(
    symbol: str,
    side: str,
    entry: float,
    stop: float,
    balance: float,
    strategy: str,
    tranche: int = 1,
    max_tranches: int = 1,
    available_margin: float | None = None,
    risk: RiskConfig = RISK,
    fees: FeeAssumption = FEES,
    execution: ExecutionAssumption | None = None,
    risk_multiplier: float = 1.0,
    fixed_leverage: int | None = None,
    margin_fraction: float | None = None,
    min_liquidation_multiple: float | None = None,
) -> Position:
    """손절 거리에서 포지션 크기와 레버리지를 역산한다.

    거절도 정상 결과다. 예외를 던지지 않고 Position.rejected에 사유를 담는다.
    진입하지 않은 신호도 기록해야 하기 때문이다(README 원칙 4).
    """

    if symbol not in CONTRACT_SPECS:
        raise SizingError(f"계약 스펙 없음: {symbol}")
    # 세부 태그("xrp_patterns.triangle")가 등록돼 있으면 그쪽을 먼저 쓴다.
    # 가짜 돌파가 잦은 패턴에 더 낮은 상한을 걸기 위한 것이다.
    strategy_family = strategy.split(".", 1)[0]
    if strategy in LEVERAGE_CAP:
        cap_key = strategy
    elif strategy_family in LEVERAGE_CAP:
        cap_key = strategy_family
    else:
        raise SizingError(f"전략 레버리지 캡 없음: {strategy}")
    if side not in ("long", "short"):
        raise SizingError(f"side는 long/short: {side!r}")
    if not math.isfinite(entry) or not math.isfinite(stop) or entry <= 0 or stop <= 0:
        raise SizingError(f"가격은 0보다 큰 유한값이어야 한다: entry={entry}, stop={stop}")
    if not math.isfinite(balance) or balance <= 0:
        raise SizingError(f"잔고는 0보다 큰 유한값이어야 한다: {balance}")
    if available_margin is not None and (
        not math.isfinite(available_margin) or available_margin < 0
    ):
        raise SizingError(f"가용 증거금은 0 이상의 유한값이어야 한다: {available_margin}")
    if (side == "long" and stop >= entry) or (side == "short" and stop <= entry):
        raise SizingError(f"{side} 손절 위치가 반대편이다: entry={entry}, stop={stop}")

    fixed_margin_mode = fixed_leverage is not None or margin_fraction is not None
    if fixed_margin_mode and (fixed_leverage is None or margin_fraction is None):
        raise SizingError("고정 레버리지와 증거금 비율은 함께 지정해야 한다")
    if fixed_leverage is not None and (
        isinstance(fixed_leverage, bool)
        or not isinstance(fixed_leverage, int)
        or fixed_leverage <= 0
    ):
        raise SizingError(f"고정 레버리지는 양의 정수여야 한다: {fixed_leverage}")
    if margin_fraction is not None and (
        not math.isfinite(margin_fraction) or not 0 < margin_fraction <= 1
    ):
        raise SizingError(f"증거금 비율은 0 초과 1 이하여야 한다: {margin_fraction}")
    if min_liquidation_multiple is not None and (
        not math.isfinite(min_liquidation_multiple)
        or min_liquidation_multiple <= 0
    ):
        raise SizingError(
            "청산거리 배수 덮어쓰기는 0보다 큰 유한값이어야 한다: "
            f"{min_liquidation_multiple}"
        )

    spec = CONTRACT_SPECS[symbol]
    notes: list[str] = []
    execution = execution or ExecutionAssumption.taker(fees)

    risk_values = (
        risk.risk_per_trade,
        risk.max_stop_pct,
        risk.min_stop_cost_multiple,
        risk.min_liquidation_multiple,
    )
    if any(not math.isfinite(value) or value <= 0 for value in risk_values):
        raise SizingError(f"리스크 설정은 0보다 큰 유한값이어야 한다: {risk_values}")

    rates = (
        execution.entry_fee_rate,
        execution.exit_fee_rate,
        execution.entry_slippage_rate,
        execution.exit_slippage_rate,
    )
    if any(not math.isfinite(rate) or rate < 0 for rate in rates):
        raise SizingError(f"체결 비용률은 0 이상의 유한값이어야 한다: {rates}")
    if execution.entry_slippage_rate >= 1 or execution.exit_slippage_rate >= 1:
        raise SizingError(f"슬리피지율은 100% 미만이어야 한다: {rates}")

    stop_distance = abs(entry - stop) / entry

    # --- 손절 거리 게이트 ---
    min_stop = risk.min_stop_pct(fees)
    if stop_distance < min_stop:
        notes.append(
            f"손절 {stop_distance:.3%} < 하한 {min_stop:.3%} "
            f"(왕복 비용 {fees.round_trip_taker():.3%}의 {risk.min_stop_cost_multiple}배)"
        )
        return _rejected(REJECT_STOP_TOO_TIGHT, stop_distance, notes)
    if stop_distance > risk.max_stop_pct:
        notes.append(f"손절 {stop_distance:.3%} > 상한 {risk.max_stop_pct:.3%}")
        return _rejected(REJECT_STOP_TOO_WIDE, stop_distance, notes)

    # --- 계약당 총손실 ---
    risk_pct = tranche_risk(risk.risk_per_trade, tranche, max_tranches)

    # 가짜 돌파가 잦은 패턴은 리스크를 줄여 잡는다.
    # 명목이 줄면 레버리지도 함께 줄어든다 — 레버리지 캡이 발동하지 않는
    # 구간에서 "과도한 레버리지 금지"를 실제로 강제하는 경로다.
    # 전략 단위 배수(가짜 돌파가 잦은 패턴 축소)와
    # 신호 단위 배수(자리별 확신도)를 곱한다.
    multiplier = RISK_MULTIPLIER.get(strategy, 1.0)
    if not math.isfinite(multiplier) or not 0 < multiplier <= 1:
        raise SizingError(f"전략 리스크 배수는 0~1 사이여야 한다: {multiplier}")
    if not math.isfinite(risk_multiplier) or not 0 < risk_multiplier <= MAX_SIGNAL_RISK_MULT:
        raise SizingError(
            f"신호 리스크 배수는 0~{MAX_SIGNAL_RISK_MULT} 사이여야 한다: {risk_multiplier}"
        )
    risk_pct *= multiplier * risk_multiplier

    risk_budget = balance * risk_pct

    if side == "long":
        entry_fill = entry * (1.0 + execution.entry_slippage_rate)
        stop_fill = stop * (1.0 - execution.exit_slippage_rate)
        price_loss_per_unit = entry_fill - stop_fill
    else:
        entry_fill = entry * (1.0 - execution.entry_slippage_rate)
        stop_fill = stop * (1.0 + execution.exit_slippage_rate)
        price_loss_per_unit = stop_fill - entry_fill

    multiplier = float(spec["multiplier"])
    expected_price_loss_per_contract = price_loss_per_unit * multiplier
    planned_price_loss_per_contract = abs(entry - stop) * multiplier
    slippage_loss_per_contract = (
        expected_price_loss_per_contract - planned_price_loss_per_contract
    )
    fee_per_contract = multiplier * (
        entry_fill * execution.entry_fee_rate
        + stop_fill * execution.exit_fee_rate
    )
    loss_per_contract = expected_price_loss_per_contract + fee_per_contract
    if not math.isfinite(loss_per_contract) or loss_per_contract <= 0:
        raise SizingError(f"계약당 예상 손실 계산 오류: {loss_per_contract}")

    # --- 계약 수 (내림) ---
    min_size = int(spec["min_size"])
    contract_value = entry_fill * multiplier

    if fixed_margin_mode:
        # 이 실험은 "손실 1%"가 아니라 잔고 10%를 격리증거금으로 묶고
        # 정확히 25배를 쓰는 전략이다. 따라서 리스크 예산으로 계약 수를
        # 역산하면 명세가 조용히 1~3배 전략으로 바뀌므로 별도 경로를 탄다.
        target_margin = balance * float(margin_fraction)
        target_notional = target_margin * int(fixed_leverage)
        contracts = math.floor(target_notional / contract_value)
    else:
        target_margin = 0.0
        contracts = math.floor(risk_budget / loss_per_contract)

    if contracts < min_size:
        if fixed_margin_mode:
            notes.append(
                f"목표 명목 {target_notional:.4f} < 최소 주문 명목 "
                f"{min_size * contract_value:.4f} "
                f"(잔고의 {float(margin_fraction):.1%} × {fixed_leverage}배)"
            )
        else:
            notes.append(
                f"리스크 예산 {risk_budget:.4f} < 최소 주문 예상 손실 "
                f"{min_size * loss_per_contract:.4f} "
                f"(1계약 = {contract_value:.4f} USDT)"
            )
        return _rejected(REJECT_SIZE_BELOW_MIN, stop_distance, notes)

    # 반올림 후 실제 값으로 다시 계산한다.
    # 계약 단위 때문에 목표보다 작아지므로, 실제 리스크는 예산 이하가 된다.
    notional = contracts * contract_value
    price_risk_amount = contracts * planned_price_loss_per_contract
    estimated_cost = contracts * (slippage_loss_per_contract + fee_per_contract)
    risk_amount = price_risk_amount + estimated_cost
    if not fixed_margin_mode and risk_amount > risk_budget + 1e-9:
        raise SizingError(
            f"계약 내림 후 예상 손실이 예산을 넘었다: {risk_amount} > {risk_budget}"
        )

    # --- 레버리지 ---
    margin_pool = balance if available_margin is None else available_margin
    if margin_pool <= 0:
        notes.append("가용 증거금 없음")
        return _rejected(REJECT_LEVERAGE_INFEASIBLE, stop_distance, notes)

    mr = float(spec["maintenance_rate"])

    # 상한 1: 청산가가 손절보다 min_liquidation_multiple배 이상 멀 것
    #   1/L - mr >= k * stop  ->  L <= 1 / (k * stop + mr)
    liquidation_multiple = (
        risk.min_liquidation_multiple
        if min_liquidation_multiple is None
        else min_liquidation_multiple
    )
    denom = liquidation_multiple * stop_distance + mr
    lev_by_liquidation = 1.0 / denom if denom > 0 else float("inf")

    # 상한 2: 전략별 캡, 상한 3: 거래소 최대
    lev_max = min(
        lev_by_liquidation,
        LEVERAGE_CAP[cap_key],
        float(spec["max_leverage"]),
    )

    max_integer_leverage = math.floor(lev_max + 1e-9)

    if fixed_margin_mode:
        leverage = int(fixed_leverage)
        if target_margin > margin_pool + 1e-9:
            notes.append(
                f"목표 증거금 {target_margin:.4f} > 가용 {margin_pool:.4f}"
            )
            return _rejected(REJECT_LEVERAGE_INFEASIBLE, stop_distance, notes)
        if leverage > max_integer_leverage:
            notes.append(
                f"고정 레버리지 {leverage}x > 허용 {lev_max:.2f}x "
                f"(청산기준 {lev_by_liquidation:.1f}x, "
                f"전략캡 {LEVERAGE_CAP[cap_key]}x [{cap_key}])"
            )
            return _rejected(REJECT_LEVERAGE_INFEASIBLE, stop_distance, notes)
    else:
        # 하한: 이만큼은 있어야 증거금으로 이 명목을 열 수 있다
        lev_needed = notional / margin_pool
        min_integer_leverage = max(1, math.ceil(lev_needed - 1e-9))

        if min_integer_leverage > max_integer_leverage:
            notes.append(
                f"필요 레버리지 {lev_needed:.2f}x > 허용 {lev_max:.2f}x "
                f"(청산기준 {lev_by_liquidation:.1f}x, "
                f"전략캡 {LEVERAGE_CAP[cap_key]}x [{cap_key}])"
            )
            return _rejected(REJECT_LEVERAGE_INFEASIBLE, stop_distance, notes)

        # 가능한 것 중 가장 낮은 정수 레버리지를 쓴다.
        # 낮을수록 청산가가 멀어지고, 손익은 어차피 명목 크기로 정해져 있다.
        leverage = min_integer_leverage

    margin = notional / leverage
    if margin > margin_pool + 1e-9:
        notes.append(f"필요 증거금 {margin:.4f} > 가용 {margin_pool:.4f}")
        return _rejected(REJECT_LEVERAGE_INFEASIBLE, stop_distance, notes)

    liq = liquidation_price(entry_fill, side, leverage, mr)
    liq_distance = abs(entry_fill - liq) / entry_fill

    if liq_distance < liquidation_multiple * stop_distance:
        # 정수 반올림 때문에 상한을 넘긴 경우. 여기 오면 계산이 어긋난 것이다.
        notes.append(
            f"청산 거리 {liq_distance:.3%}가 손절 {stop_distance:.3%}의 "
            f"{liquidation_multiple}배 미만"
        )
        return _rejected(REJECT_LEVERAGE_INFEASIBLE, stop_distance, notes)

    if not fixed_margin_mode and risk_amount < risk_budget * 0.5:
        notes.append(
            f"계약 단위 때문에 리스크가 예산의 {risk_amount / risk_budget:.0%}로 축소"
        )

    return Position(
        contracts=contracts,
        notional=notional,
        margin=margin,
        leverage=leverage,
        stop_distance=stop_distance,
        risk_amount=risk_amount,
        price_risk_amount=price_risk_amount,
        estimated_cost=estimated_cost,
        entry_fill_price=entry_fill,
        stop_fill_price=stop_fill,
        liquidation_price=liq,
        liquidation_distance=liq_distance,
        notes=notes,
    )
