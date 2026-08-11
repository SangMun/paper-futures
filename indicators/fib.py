"""피보나치 되돌림·확장 레벨.

명세 §3. 스윙 하나에서 지지/저항 후보 가격을 뽑는다.

작도 방향 규칙(§3.2):
  상승 임펄스(저점 -> 고점) → 되돌림은 하락, 레벨은 **지지**
  하락 임펄스(고점 -> 저점) → 되돌림은 상승, 레벨은 **저항**

피보나치는 통계적 근거가 확립된 지표가 아니다. 다수가 같은 레벨을 보기
때문에 작동하는 자기실현적 성격이 강하므로, 단독 신호로 쓰지 않고
합류 판정의 한 요소로만 취급한다(§11).
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import FIB_CORE, FIB_EXTENSION, FIB_EXTRA
from indicators.base import IndicatorError
from indicators.pivots import Swing


# 골든 포켓. 합류 점수에서 가산점을 받는다.
GOLDEN_POCKET = (0.618, 0.65)


@dataclass(frozen=True)
class FibLevel:
    ratio: float
    price: float
    swing: Swing

    @property
    def is_golden(self) -> bool:
        return self.ratio in GOLDEN_POCKET

    def label(self) -> str:
        return f"fib_{self.ratio:g}"


def retracement(swing: Swing, ratio: float) -> float:
    """되돌림 레벨 가격.

    상승: H - D*r  (고점에서 되돌아 내려온 자리)
    하락: L + D*r  (저점에서 되돌아 올라간 자리)
    """

    if not 0 <= ratio <= 1:
        raise IndicatorError(f"되돌림 비율은 0~1: {ratio}")

    d = swing.size
    return swing.high - d * ratio if swing.direction == "up" else swing.low + d * ratio


def extension(swing: Swing, ratio: float) -> float:
    """확장 레벨. 임펄스 방향으로 연장한다 (익절 목표용).

    상승: H + D*(r-1)
    하락: L - D*(r-1)
    """

    if ratio <= 1:
        raise IndicatorError(f"확장 비율은 1보다 커야 한다: {ratio}")

    d = swing.size
    return swing.high + d * (ratio - 1) if swing.direction == "up" else swing.low - d * (ratio - 1)


def core_levels(swing: Swing) -> list[FibLevel]:
    """진입 판정에 쓰는 핵심 레벨만.

    ATFX 문서의 "레벨을 너무 많이 쓰지 말 것"을 반영해 4개로 제한한다(§3.1).
    레벨을 늘릴수록 아무 가격에나 걸려서 변별력이 떨어진다.
    """

    return [FibLevel(r, retracement(swing, r), swing) for r in FIB_CORE]


def reference_levels(swing: Swing) -> list[FibLevel]:
    """참고용으로 계산만 하는 레벨. 진입 판정에는 쓰지 않는다."""

    return [FibLevel(r, retracement(swing, r), swing) for r in FIB_EXTRA]


def extension_levels(swing: Swing) -> list[FibLevel]:
    return [FibLevel(r, extension(swing, r), swing) for r in FIB_EXTENSION]


def invalidated(swing: Swing, close: float) -> bool:
    """100% 지점(임펄스 시작점)을 종가로 이탈했으면 폐기 (§3.3).

    종가 기준인 이유는 꼬리로 잠깐 뚫는 건 되돌림의 정상 범위이기 때문이다.
    """

    if swing.direction == "up":
        return close < swing.low
    return close > swing.high


def entry_side(swing: Swing) -> str:
    """이 스윙의 되돌림에서 노릴 방향.

    상승 임펄스의 되돌림은 지지에서 반등을 노리므로 롱.
    """

    return "long" if swing.direction == "up" else "short"


def next_level_below(swing: Swing, price: float, side: str) -> float:
    """진입 레벨 다음의 하위(롱)/상위(숏) 피보나치 레벨. 손절 기준이다 (§6).

    다음 레벨이 없으면 100% 지점(임펄스 시작점)을 쓴다.
    """

    ratios = sorted(set(FIB_CORE) | set(FIB_EXTRA))
    candidates = [retracement(swing, r) for r in ratios]

    if side == "long":
        below = [p for p in candidates if p < price]
        return max(below) if below else swing.low

    above = [p for p in candidates if p > price]
    return min(above) if above else swing.high
