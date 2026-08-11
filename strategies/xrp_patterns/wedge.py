"""쐐기형 (상승 쐐기 / 하락 쐐기).

고점과 저점이 **같은 방향**으로 가되 폭이 좁아지는 구조.
방향보다 "가격이 진행하는 힘이 점점 줄어든다"가 핵심이다.

  상승 쐐기 → 약세 전환 후보 (숏)
  하락 쐐기 → 강세 전환 후보 (롱)

삼각수렴과의 차이는 두 추세선의 방향이다. 삼각형은 서로 반대로 좁아지고,
쐐기는 같은 방향으로 기울어진 채 좁아진다.

거래량 구조: 쐐기 내부로 갈수록 감소, 반대편 추세선 돌파 시 증가.
내부 거래량이 오히려 늘면 추세가 강화되는 중이라 쐐기로 보지 않는다.

주의: 상승 쐐기의 하방 이탈이 항상 강한 건 아니다. 일부 통계에서는
실패율이 높게 나오므로 거래량과 종가 이탈 확인이 특히 중요하다.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SCALP
from strategies.base import Prepared, Signal
from strategies.xrp_patterns import base as xb
from strategies.xrp_patterns import chart_base as cb


NAME = f"{xb.FAMILY}.wedge"

# 두 추세선이 이 비율 이상 좁아져야 수렴으로 본다.
MIN_CONVERGENCE = 0.35

# 두 추세선이 같은 방향으로 기울어야 한다 (쐐기의 정의).
MIN_SLOPE = 0.00015

# 패턴 전체 폭이 ATR의 이 배수보다 작으면 잡음이다.
#
# 수렴 판정만으로는 무작위 데이터에서도 우연히 성립한다 — 고점 둘이
# 내려오고 저점 둘이 올라오는 건 노이즈에서도 흔하다. 봉당 변동(ATR)보다
# 작은 "삼각형"은 구조가 아니라 잡음이므로 크기 하한을 둔다.
MIN_HEIGHT_ATR = 3.0



class Wedge:
    """상승/하락 쐐기 + 반대편 추세선 종가 이탈."""

    def __init__(
        self,
        timeframe: str = "5m",
        symbol: str = "XRP_USDT",
        trend_align: bool = False,
        limit_entry: bool = False,
        block_hours: tuple[int, ...] = (),
    ):
        # 돌파 봉 종가를 쫓지 않고 추세선 가격에 지정가를 건다.
        # 949일 실측에서 시장가 -0.074% / 지정가 +0.091%였고 앞뒤 절반
        # 모두 플러스였다. 진입가가 유리해지는 데다 테이커(0.19%)가
        # 메이커 리베이트(-0.02%)로 바뀌어 0.21%p가 그대로 남는다.
        self.limit_entry = limit_entry
        # 진입을 막을 UTC 시간대. 16~24시(한국 새벽)는 앞뒤 절반 모두
        # -0.574% / -0.417%로 일관되게 최악이었다. 유동성이 얇아
        # 가짜 돌파가 잦은 시간대로 보인다.
        self.block_hours = frozenset(block_hours)
        self.symbol = symbol
        # 대세(4시간봉)와 반대 방향을 막을지.
        # 949일 실측에서 상승장 숏 13건이 -94.5로 이 전략 손실의 전부였고,
        # 앞뒤 절반 모두 마이너스(-48.5 / -46.0)라 구간 우연이 아니다.
        self.trend_align = trend_align
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

        slope_high = cb.slope_of(h1, h2)
        slope_low = cb.slope_of(l1, l2)

        # 쐐기: 두 추세선이 같은 방향
        if slope_high > MIN_SLOPE and slope_low > MIN_SLOPE:
            side, kind = "short", "rising_wedge"      # 상승 쐐기 -> 약세
        elif slope_high < -MIN_SLOPE and slope_low < -MIN_SLOPE:
            side, kind = "long", "falling_wedge"      # 하락 쐐기 -> 강세
        else:
            return None

        # 수렴 확인: 시작 폭 대비 현재 폭이 줄었는가
        width_start = cb.line_at(h1, h2, start) - cb.line_at(l1, l2, start)
        width_now = cb.line_at(h1, h2, as_of) - cb.line_at(l1, l2, as_of)
        if width_start <= 0 or width_now <= 0:
            return None
        if width_start < atr * MIN_HEIGHT_ATR:
            return None
        if width_now / width_start > 1 - MIN_CONVERGENCE:
            return None

        # 이탈선: 상승 쐐기는 하단, 하락 쐐기는 상단
        if side == "short":
            level = cb.line_at(l1, l2, as_of)
        else:
            level = cb.line_at(h1, h2, as_of)
        if level <= 0:
            return None

        if not cb.close_break(df, as_of, level, side):
            return None

        # 내부 거래량이 늘고 있으면 쐐기가 아니다 — 추세가 강화되는 중이다.
        trend = prepared.at(self.tf.entry, "vol_trend", as_of)
        if pd.isna(trend) or trend > 0:
            return None

        if xb.blocked(prepared, as_of, side, self.tf) is not None:
            return None
        if self.trend_align and xb.counter_trend(prepared, as_of, side, self.tf):
            return None
        if self.block_hours:
            hour = int((int(df["ts"].iloc[as_of]) % 86400) // 3600)
            if hour in self.block_hours:
                return None

        # 선행 추세: 쐐기 방향과 같은 흐름이 앞에 있었는가
        before = max(0, start - 30)
        prior = df["close"].iloc[before : start + 1]
        prior_ok = bool(
            len(prior) > 5
            and (
                (kind == "rising_wedge" and prior.iloc[-1] > prior.iloc[0])
                or (kind == "falling_wedge" and prior.iloc[-1] < prior.iloc[0])
            )
        )

        profile = cb.profile_at(df, as_of)
        quality = cb.assess(df, prepared, as_of, side, start, level, profile, prior_ok, self.tf)
        quality.append("converging")

        if cb.retest_ok(df, as_of - 1, as_of, level, side):
            quality.append("retested")
        if len(quality) < cb.MIN_QUALITY:
            return None

        close = float(df["close"].iloc[as_of])

        # 지정가면 이탈선 가격에 건다. 돌파 봉 종가는 이미 움직임의
        # 절반 이상이 끝난 자리라 추격이 된다.
        entry_price = float(level) if self.limit_entry else close

        # 손절 = 무효화 = 반대편 추세선. 거기로 되돌아가면 쐐기 이탈이 실패한 것이다.
        stop_hint = (
            cb.line_at(h1, h2, as_of) if side == "short" else cb.line_at(l1, l2, as_of)
        )
        if side == "short" and stop_hint <= entry_price:
            return None
        if side == "long" and stop_hint >= entry_price:
            return None

        height = width_start

        return [
            Signal(
                strategy=self.name,
                symbol=self.symbol,
                side=side,
                entry=entry_price,
                stop_hint=float(stop_hint),
                target_hint=xb.vpvr_target(df, as_of, entry_price, side),
                score=float(len(quality)),
                reasons=[kind, *quality],
                context={
                    "atr": atr,
                    "break_level": level,
                    "slope_high": slope_high,
                    "slope_low": slope_low,
                    "convergence": 1 - width_now / width_start,
                    "pattern_height": height,
                    "measured_target": cb.pattern_target(entry_price, height, side),
                    "quality": quality,
                    "rsi": float(prepared.at(self.tf.entry, "rsi", as_of)),
                },
                order_type="limit" if self.limit_entry else "market",
                ttl_bars=SCALP.limit_order_ttl_bars if self.limit_entry else 0,
            )
        ]
