"""이중천장 / 이중바닥.

같은 지지·저항을 두 번 시험하는 반전 구조.

**두 번째 저점·고점에서 바로 진입하지 않는다.** 가운데 고점·저점인 목선을
종가로 돌파해야 확인이다. 두 번 닿았다는 사실 자체는 확률이 낮다 —
비슷한 두 고점이 나와도 실제로 목선 아래에서 종가가 확정되는 경우는
절반 이하라는 연구가 있다.

이전 구현(`double_bottom`)은 두 번째 저점 레벨에 바로 지정가를 걸었다.
목선 확인이 빠져 있었고, 손절도 레벨 ±0.1%로 잡아 손절 하한(0.38%)에
전량 걸려 180일에 0건이었다. 여기서는 둘 다 고친다.

거래량 구조:
  첫 고점 거래량이 크고, 두 번째 고점은 약하고, 목선 이탈에서 늘어난다
매물대 결합:
  두 고점이 VAH·상단 매물대에서 형성되면 구조가 좋다
  두 번째 고점이 거래량과 함께 저항 위에 안착하면 이중천장이 아니라 돌파다
"""

from __future__ import annotations

import pandas as pd

from strategies.base import Prepared, Signal
from strategies.xrp_patterns import base as xb
from strategies.xrp_patterns import chart_base as cb


NAME = f"{xb.FAMILY}.double_top"

# 두 고점(저점)의 가격 차이 허용 오차.
PEAK_TOLERANCE = 0.006

# 두 고점 사이가 이보다 가까우면 잡음이다.
MIN_GAP_BARS = 8

# 목선이 고점에서 이 비율 이상 떨어져 있어야 한다.
# 너무 얕으면 되돌림이지 이중천장 구조가 아니다.
MIN_DEPTH = 0.006


class DoubleTop:
    """두 고점(저점) + 목선 종가 돌파."""

    def __init__(
        self,
        timeframe: str = "5m",
        symbol: str = "XRP_USDT",
        drop_long_vwap_alignment: bool = False,
    ):
        self.symbol = symbol
        # 롱(double_bottom)에서 vwap_aligned 품질 항목을 뺄지.
        # head_shoulders.py의 같은 이름 필드 docstring 참고 — EXP-014.
        self.drop_long_vwap_alignment = drop_long_vwap_alignment
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
        if self._pivots is None or as_of < 40 or as_of >= len(df):
            return []

        atr = xb.atr_at(prepared, as_of, self.tf)
        if atr <= 0:
            return []

        recent = self._pivots.upto(as_of, count=6)
        if len(recent) < 3:
            return []

        for start in range(len(recent) - 3, -1, -1):
            found = self._match(prepared, df, as_of, recent[start : start + 3], atr)
            if found is not None:
                return [found]
        return []

    def _match(self, prepared, df, as_of, three, atr) -> Signal | None:
        first, middle, second = three

        if [p.kind for p in three] == ["high", "low", "high"]:
            side, kind = "short", "double_top"
        elif [p.kind for p in three] == ["low", "high", "low"]:
            side, kind = "long", "double_bottom"
        else:
            return None

        if as_of - first.index > cb.MAX_PATTERN_BARS:
            return None
        if second.index - first.index < MIN_GAP_BARS:
            return None

        # 두 고점(저점)이 비슷한 가격
        if not cb.near(first.price, second.price, PEAK_TOLERANCE):
            return None

        # 목선까지의 깊이가 충분해야 구조가 성립한다
        depth = abs(first.price - middle.price) / first.price
        if depth < MIN_DEPTH:
            return None

        # 두 번째 고점이 첫 번째를 뚜렷하게 넘으면 이중천장이 아니라 돌파다
        if side == "short" and second.price > first.price * (1 + PEAK_TOLERANCE):
            return None
        if side == "long" and second.price < first.price * (1 - PEAK_TOLERANCE):
            return None

        neckline = float(middle.price)

        # **완성 확인**: 목선 종가 돌파
        if not cb.close_break(df, as_of, neckline, side):
            return None

        if xb.blocked(prepared, as_of, side, self.tf) is not None:
            return None

        # 반전형 — 이중천장 앞에는 상승이 있어야 뒤집을 것이 있다.
        prior_ok = cb.prior_trend_ok(
            df, first.index, "up" if side == "short" else "down", 30
        )

        profile = cb.profile_at(df, as_of)
        quality = cb.assess(df, prepared, as_of, side, first.index, neckline, profile, prior_ok, self.tf)

        # 두 번째 고점의 거래량이 첫 번째보다 약한가 (전형적 구조)
        v1 = float(df["volume"].iloc[first.index])
        v2 = float(df["volume"].iloc[second.index])
        if v1 > 0 and v2 < v1:
            quality.append("second_peak_weaker")
        if self.drop_long_vwap_alignment and side == "long":
            quality = [item for item in quality if item != "vwap_aligned"]

        if len(quality) < cb.MIN_QUALITY:
            return None

        close = float(df["close"].iloc[as_of])

        # 손절 = 무효화 지점 = 두 고점 중 바깥쪽.
        # 거기를 넘으면 반전이 아니라 추세 지속이다.
        stop_hint = (
            max(first.price, second.price) if side == "short"
            else min(first.price, second.price)
        )
        if side == "short" and stop_hint <= close:
            return None
        if side == "long" and stop_hint >= close:
            return None

        height = abs(first.price - neckline)

        return Signal(
            strategy=self.name,
            symbol=self.symbol,
            side=side,
            entry=close,
            stop_hint=float(stop_hint),
            target_hint=xb.vpvr_target(df, as_of, close, side),
            score=float(len(quality)),
            reasons=[kind, *quality],
            context={
                "atr": atr,
                "neckline": neckline,
                "first_peak": first.price,
                "second_peak": second.price,
                "pattern_height": height,
                "measured_target": cb.pattern_target(close, height, side),
                "quality": quality,
                "rsi": float(prepared.at(self.tf.entry, "rsi", as_of)),
            },
            order_type="market",
        )
