"""계좌 단위 안전장치.

개별 트레이드의 리스크는 sizing.py가 1%로 묶는다. 여기서는 그게 겹쳤을 때를
막는다. 1%짜리 열 개가 동시에 열려 있으면 계좌 리스크는 10%다.

특히 같은 방향 포지션은 서로 독립이 아니다. BTC 추세 롱과 BTC 피보 롱이
동시에 잡히면 사실상 하나의 베팅이고, XRP도 BTC 베타라 방향이 같으면 함께 움직인다.
합산할 때 이걸 반영하지 않으면 분산됐다고 착각하게 된다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from config.settings import RISK, RiskConfig


Side = Literal["long", "short"]

# 거절 사유. 로그의 skip_reason과 같은 값을 쓴다.
REJECT_DAILY_LOSS = "daily_loss_limit"
REJECT_CONSECUTIVE = "consecutive_loss_limit"
REJECT_TOTAL_RISK = "total_risk_cap"
REJECT_SYMBOL_RISK = "symbol_risk_cap"
REJECT_INVALID_RISK = "invalid_risk"


@dataclass(frozen=True)
class OpenRisk:
    """열려 있는 포지션 하나가 물고 있는 리스크."""

    symbol: str
    side: Side
    strategy: str
    risk_amount: float      # 손절까지 갔을 때의 손실액


@dataclass
class AccountState:
    """엔진이 들고 다니는 계좌 상태."""

    balance: float
    day_start_balance: float
    # 전략 태그별 연패 수와 정지 만료 시각.
    # 계좌 전체로 묶으면 한 전략의 연패가 다른 전략을 멈춘다.
    consecutive_losses: dict = field(default_factory=dict)
    paused_until: dict = field(default_factory=dict)
    now_ts: int = 0
    open_positions: list[OpenRisk] = field(default_factory=list)

    def daily_pnl_pct(self) -> float:
        if self.day_start_balance <= 0:
            return 0.0
        return (self.balance - self.day_start_balance) / self.day_start_balance


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str | None = None
    detail: str = ""


ALLOWED = Verdict(True)


def weighted_open_risk(
    positions: list[OpenRisk],
    balance: float,
    side: Side | None = None,
    symbol: str | None = None,
    risk: RiskConfig = RISK,
) -> float:
    """열린 리스크 합계를 계좌 대비 비율로.

    상관계수가 실측되기 전에는 포지션 방향과 무관하게 전액 합산한다.
    단순 가중치로 같은 방향 리스크를 할인하면 오히려 한도가 느슨해진다.
    side 인자는 호출부 호환성을 위해 남겨 둔다.
    """

    if not math.isfinite(balance) or balance <= 0:
        return 0.0

    total = 0.0
    for p in positions:
        if symbol is not None and p.symbol != symbol:
            continue
        if not math.isfinite(p.risk_amount) or p.risk_amount < 0:
            raise ValueError(f"열린 포지션 리스크가 잘못됐다: {p.risk_amount}")
        total += p.risk_amount
    return total / balance


def check_entry(
    state: AccountState,
    symbol: str,
    side: Side,
    risk_amount: float,
    risk: RiskConfig = RISK,
    strategy: str = "",
) -> Verdict:
    """신규 진입을 허용할지.

    거절해도 신호는 기록된다(README 원칙 4). 여기서 막은 게 실은 좋은
    신호였는지 나중에 판단할 수 있어야 하기 때문이다.
    """

    if (
        not math.isfinite(state.balance)
        or not math.isfinite(state.day_start_balance)
        or state.day_start_balance <= 0
        or not math.isfinite(risk_amount)
        or risk_amount <= 0
    ):
        return Verdict(
            False,
            REJECT_INVALID_RISK,
            f"유효하지 않은 계좌/리스크 값: balance={state.balance}, "
            f"day_start={state.day_start_balance}, risk={risk_amount}",
        )

    if state.balance <= 0:
        return Verdict(False, REJECT_DAILY_LOSS, "잔고 소진")

    # --- 계좌 단위 정지 조건 ---
    pnl = state.daily_pnl_pct()
    if pnl <= -risk.daily_loss_limit:
        return Verdict(
            False,
            REJECT_DAILY_LOSS,
            f"당일 손익 {pnl:.2%} <= 한도 -{risk.daily_loss_limit:.0%}",
        )

    # 연속 손절 정지. **전략별**이고 만료 시각이 반드시 있어야 한다.
    # 카운터는 이익 거래로만 초기화되는데 진입이 막혀 있으면 이익 거래가
    # 나올 수 없다. 만료가 없으면 첫 5연패에서 그 전략이 영구히 멈춘다.
    until = state.paused_until.get(strategy, 0)
    if until and state.now_ts < until:
        remaining = (until - state.now_ts) / 3600
        return Verdict(
            False,
            REJECT_CONSECUTIVE,
            f"{strategy} 연속 손절 {state.consecutive_losses.get(strategy, 0)}회로 "
            f"정지 중 (잔여 {remaining:.1f}시간)",
        )

    # --- 리스크 상한 ---
    new_pct = risk_amount / state.balance

    symbol_risk = weighted_open_risk(
        state.open_positions, state.balance, side=side, symbol=symbol, risk=risk
    )
    if symbol_risk + new_pct > risk.max_symbol_risk:
        return Verdict(
            False,
            REJECT_SYMBOL_RISK,
            f"{symbol} 리스크 {symbol_risk:.2%} + {new_pct:.2%} "
            f"> 상한 {risk.max_symbol_risk:.2%}",
        )

    total_risk = weighted_open_risk(
        state.open_positions, state.balance, side=side, risk=risk
    )
    if total_risk + new_pct > risk.max_total_risk:
        return Verdict(
            False,
            REJECT_TOTAL_RISK,
            f"전체 리스크 {total_risk:.2%} + {new_pct:.2%} "
            f"> 상한 {risk.max_total_risk:.2%}",
        )

    return ALLOWED


def record_result(
    state: AccountState,
    pnl: float,
    risk: RiskConfig = RISK,
    strategy: str = "",
) -> None:
    """청산 결과 반영. 연속 손절 카운터를 갱신한다.

    본전(0)은 손실로 세지 않는다. 트레일링이 본전에서 끊는 경우가 잦은데
    그걸 손절로 세면 카운터가 실제 연패보다 빨리 찬다.

    한도에 도달하면 만료 시각을 세운다. 만료가 지나면 카운터도 함께
    풀린다 — 그러지 않으면 정지가 영구가 된다.
    """

    state.balance += pnl
    if pnl < 0:
        n = state.consecutive_losses.get(strategy, 0) + 1
        state.consecutive_losses[strategy] = n
        if n >= risk.consecutive_loss_limit and state.now_ts:
            state.paused_until[strategy] = int(state.now_ts + risk.pause_hours * 3600)
    elif pnl > 0:
        state.consecutive_losses[strategy] = 0
        state.paused_until[strategy] = 0


def clear_expired_pause(state: AccountState, risk: RiskConfig = RISK) -> list[str]:
    """만료된 전략별 정지를 푼다. 풀린 전략 목록을 돌려준다.

    카운터까지 초기화하는 이유는, 남겨 두면 다음 손절 한 번에 곧바로
    다시 정지되어 사실상 영구 정지와 같아지기 때문이다.
    """

    released = []
    for tag, until in list(state.paused_until.items()):
        if until and state.now_ts >= until:
            state.paused_until[tag] = 0
            state.consecutive_losses[tag] = 0
            released.append(tag)
    return released


def start_new_day(state: AccountState) -> None:
    """일일 손실 한도의 기준점을 옮긴다. 연속 손절은 날짜와 무관하게 이어진다."""

    state.day_start_balance = state.balance
