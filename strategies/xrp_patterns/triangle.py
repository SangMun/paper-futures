"""대칭 삼각수렴.

고점은 낮아지고 저점은 높아지며 폭이 좁아지는 구조.
**방향을 미리 정하지 않는다** — 상단 돌파면 롱, 하단 이탈이면 숏.
돌파 전에 방향을 예측해 진입하는 게 이 패턴의 대표적 실수다.

진행률 70~80% 부근이 "준비 구간"이지 선진입 지점이 아니다. 꼭짓점까지
거의 다 진행한 뒤 나오는 미약한 돌파는 오히려 신뢰도가 낮다.

거래량 구조: 좁아질수록 감소, 돌파 시 증가.
저거래량 돌파는 가짜일 가능성이 커서 RVOL 확인이 특히 중요하다.

매물대 결합이 다섯 패턴 중 가장 잘 맞는다.
  삼각형 상단 + VAH 동시 돌파 → 일반 돌파보다 의미가 크다
  수렴 내부에 POC가 있으면 시장이 현재가를 균형가로 받아들이는 중
"""

from __future__ import annotations

import pandas as pd

from strategies.base import Prepared, Signal
from strategies.xrp_patterns import base as xb
from strategies.xrp_patterns import chart_base as cb


NAME = f"{xb.FAMILY}.triangle"

MIN_CONVERGENCE = 0.35

# 진행률 상한. 꼭짓점에 너무 붙으면 남은 공간이 없다.
MAX_PROGRESS = 0.9

# 최소 진행률. 이보다 이르면 아직 수렴이라 부를 수 없다.
MIN_PROGRESS = 0.5

# 패턴 전체 폭이 ATR의 이 배수보다 작으면 잡음이다.
#
# 수렴 판정만으로는 무작위 데이터에서도 우연히 성립한다 — 고점 둘이
# 내려오고 저점 둘이 올라오는 건 노이즈에서도 흔하다. 봉당 변동(ATR)보다
# 작은 "삼각형"은 구조가 아니라 잡음이므로 크기 하한을 둔다.
MIN_HEIGHT_ATR = 3.0



class Triangle:
    """대칭 수렴 + 어느 쪽이든 종가 돌파."""

    def __init__(self, timeframe: str = "5m", symbol: str = "XRP_USDT"):
        self.symbol = symbol
        self.tf = xb.LADDER[timeframe]
        # 태그에 봉을 붙인다. 같은 패턴도 봉마다 성과가 달라
        # 따로 집계해야 한다 (실측: double_top 5분 -0.21%, 1시간 +0.40%).
        self.name = f"{NAME}.{timeframe}"
        self._pivots: cb.PivotView | None = None


    def prepare(self, frames: dict[str, pd.DataFrame]) -> Prepared:
        prepared, self._pivots = cb.chart_prepare(frames, self.tf)
        return prepared

    def generate(self, prepared: Prepared, as_of: int) -> list[Signal]:
        df = prepared.frame(self.tf.entry)
        if self._pivots is None or as_of < 50 or as_of >= len(df):
            return []

        atr = xb.atr_at(prepared, as_of, self.tf)
        if atr <= 0:
            return []

        recent = self._pivots.upto(as_of, count=8)
        highs = [p for p in recent if p.kind == "high"][-2:]
        lows = [p for p in recent if p.kind == "low"][-2:]
        if len(highs) < 2 or len(lows) < 2:
            return []

        return self._match(prepared, df, as_of, highs, lows, atr) or []

    def _match(self, prepared, df, as_of, highs, lows, atr):
        h1, h2 = highs
        l1, l2 = lows

        start = min(h1.index, l1.index)
        if as_of - start > cb.MAX_PATTERN_BARS:
            return None

        # 대칭 삼각형: 고점은 내려오고 저점은 올라온다
        if h2.price >= h1.price or l2.price <= l1.price:
            return None

        upper = cb.line_at(h1, h2, as_of)
        lower = cb.line_at(l1, l2, as_of)
        if upper <= 0 or lower <= 0 or upper <= lower:
            return None

        width_start = cb.line_at(h1, h2, start) - cb.line_at(l1, l2, start)
        width_now = upper - lower
        if width_start <= 0:
            return None
        if width_start < atr * MIN_HEIGHT_ATR:
            return None
        if width_now / width_start > 1 - MIN_CONVERGENCE:
            return None

        # 진행률 — 꼭짓점까지 얼마나 왔는가
        progress = 1 - width_now / width_start
        if not (MIN_PROGRESS <= progress <= MAX_PROGRESS):
            return None

        # **방향을 미리 정하지 않는다.** 종가가 어느 쪽을 넘었는지 본다.
        if cb.close_break(df, as_of, upper, "long"):
            side, level, opposite = "long", upper, lower
        elif cb.close_break(df, as_of, lower, "short"):
            side, level, opposite = "short", lower, upper
        else:
            return None

        trend = prepared.at(self.tf.entry, "vol_trend", as_of)
        if pd.isna(trend) or trend > 0:
            return None

        if xb.blocked(prepared, as_of, side, self.tf) is not None:
            return None

        profile = cb.profile_at(df, as_of)
        # 삼각형은 중립 패턴이라 선행 추세를 요구하지 않는다.
        quality = cb.assess(df, prepared, as_of, side, start, level, profile, True, self.tf)
        quality.append(f"progress_{progress:.0%}")

        # 삼각형 경계와 가치영역 경계를 동시에 넘었는가 (이 패턴의 핵심 가점)
        if profile is not None:
            val, vah = profile.value_area
            close = float(df["close"].iloc[as_of])
            if (side == "long" and close > vah) or (side == "short" and close < val):
                quality.append("value_area_break")

        if cb.retest_ok(df, as_of - 1, as_of, level, side):
            quality.append("retested")
        if len(quality) < cb.MIN_QUALITY:
            return None

        close = float(df["close"].iloc[as_of])

        # 손절 = 무효화 = 반대편 경계. 삼각형 내부로 복귀하면 가짜 돌파다.
        stop_hint = float(opposite)
        if side == "long" and stop_hint >= close:
            return None
        if side == "short" and stop_hint <= close:
            return None

        return [
            Signal(
                strategy=self.name,
                symbol=self.symbol,
                side=side,
                entry=close,
                stop_hint=stop_hint,
                target_hint=xb.vpvr_target(df, as_of, close, side),
                score=float(len(quality)),
                reasons=["symmetric_triangle", f"break_{side}", *quality],
                context={
                    "atr": atr,
                    "upper": upper,
                    "lower": lower,
                    "progress": progress,
                    "pattern_height": width_start,
                    "measured_target": cb.pattern_target(close, width_start, side),
                    "quality": quality,
                    "rsi": float(prepared.at(self.tf.entry, "rsi", as_of)),
                },
                order_type="market",
            )
        ]
