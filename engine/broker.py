"""가상 체결.

실제 주문을 내지 않고 체결을 흉내 낸다. 실전 전환 시 이 파일만 실주문
어댑터로 갈아끼운다.

가장 중요한 건 **지정가와 시장가를 구분**하는 것이다.
  테이커 왕복 = 0.19% (수수료 + 슬리피지)
  메이커 왕복 = -0.02% (리베이트)
왕복 0.21%p 차이는 XRP 단타의 손익분기 승률을 52%와 39%로 갈라놓는다.
지정가 체결을 낙관적으로 가정하면 백테스트가 통과하고 실전이 실패한다.

그래서 지정가는 **가격이 실제로 그 값을 지나갔을 때만** 체결한다.
닿기만 하면 채워진다고 보는 것도 낙관적이지만(대기열이 있으므로),
그 이상은 호가 데이터 없이 모형화할 수 없다. 가정을 문서에 남겨 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from config.settings import CONTRACT_SPECS, FEES, FeeAssumption


Side = Literal["long", "short"]
OrderType = Literal["limit", "market"]


@dataclass(frozen=True)
class Fill:
    price: float
    contracts: int
    notional: float
    fee: float                  # 음수면 리베이트
    is_maker: bool


@dataclass(frozen=True)
class PendingOrder:
    """미체결 지정가 주문."""

    symbol: str
    side: Side
    price: float
    contracts: int
    strategy: str
    placed_at: int              # 봉 인덱스
    ttl_bars: int

    def expired(self, now_bar: int) -> bool:
        return now_bar - self.placed_at >= self.ttl_bars


def round_to_tick(price: float, symbol: str) -> float:
    """호가 단위로 맞춘다. 주문 불가능한 가격으로 계산이 진행되지 않도록."""

    tick = float(CONTRACT_SPECS[symbol]["price_tick"])
    return round(round(price / tick) * tick, 10)


def fee_for(notional: float, is_maker: bool, fees: FeeAssumption = FEES) -> float:
    """체결 수수료. 메이커는 음수(리베이트)가 나온다."""

    return notional * (fees.maker_rate if is_maker else fees.taker_rate)


def market_fill(
    symbol: str,
    side: Side,
    price: float,
    contracts: int,
    fees: FeeAssumption = FEES,
) -> Fill:
    """시장가 체결. 슬리피지를 불리한 방향으로 더한다."""

    slip = price * fees.slippage_rate
    filled = price + slip if side == "long" else price - slip
    filled = round_to_tick(filled, symbol)

    notional = contracts * filled * float(CONTRACT_SPECS[symbol]["multiplier"])
    return Fill(filled, contracts, notional, fee_for(notional, False, fees), False)


def try_limit_fill(
    order: PendingOrder,
    high: float,
    low: float,
    fees: FeeAssumption = FEES,
) -> Fill | None:
    """이번 봉에서 지정가가 체결됐는지.

    롱 지정가는 가격이 그 아래로 내려와야, 숏은 위로 올라가야 체결된다.
    슬리피지는 없다 — 내가 낸 가격에 체결되는 것이 지정가의 정의다.
    """

    touched = low <= order.price if order.side == "long" else high >= order.price
    if not touched:
        return None

    notional = order.contracts * order.price * float(
        CONTRACT_SPECS[order.symbol]["multiplier"]
    )
    return Fill(order.price, order.contracts, notional, fee_for(notional, True, fees), True)


def realize_pnl(
    side: Side,
    entry: float,
    exit_price: float,
    contracts: int,
    symbol: str,
    ratio: float = 1.0,
) -> float:
    """청산 손익 (수수료 제외).

    ratio는 부분 청산 비율. 계약 수에 곱해 쓴다.
    """

    multiplier = float(CONTRACT_SPECS[symbol]["multiplier"])
    size = contracts * ratio * multiplier
    diff = exit_price - entry if side == "long" else entry - exit_price
    return diff * size


@dataclass
class Broker:
    """가상 체결기. 잔고와 미체결 주문을 들고 있다."""

    balance: float
    fees: FeeAssumption = FEES
    pending: list[PendingOrder] = None

    def __post_init__(self):
        if self.pending is None:
            self.pending = []

    def place_limit(self, order: PendingOrder) -> None:
        self.pending.append(order)

    def process_bar(self, bar_index: int, high: float, low: float) -> tuple[list[tuple[PendingOrder, Fill]], list[PendingOrder]]:
        """이번 봉의 지정가 체결과 만료를 처리한다.

        반환: (체결된 것, 만료된 것)
        만료를 따로 돌려주는 이유는 그것도 기록해야 하기 때문이다.
        지정가 체결률은 XRP 전략의 성립 여부를 가르는 값이라 추적이 필요하다.
        """

        filled: list[tuple[PendingOrder, Fill]] = []
        expired: list[PendingOrder] = []
        still: list[PendingOrder] = []

        for order in self.pending:
            fill = try_limit_fill(order, high, low, self.fees)
            if fill is not None:
                filled.append((order, fill))
                self.balance -= fill.fee
            elif order.expired(bar_index):
                expired.append(order)
            else:
                still.append(order)

        self.pending = still
        return filled, expired

    def close(
        self,
        side: Side,
        symbol: str,
        entry: float,
        exit_price: float,
        contracts: int,
        ratio: float = 1.0,
        is_maker: bool = False,
    ) -> tuple[float, float]:
        """청산. (수수료 차감 후 손익, 수수료) 반환."""

        pnl = realize_pnl(side, entry, exit_price, contracts, symbol, ratio)
        multiplier = float(CONTRACT_SPECS[symbol]["multiplier"])
        notional = contracts * ratio * exit_price * multiplier
        fee = fee_for(notional, is_maker, self.fees)

        net = pnl - fee
        self.balance += net
        return net, fee
