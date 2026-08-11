"""컵앤핸들 / 역컵앤핸들.

U자형 회복 후 짧은 조정(핸들), 핸들 상단 돌파가 확인 신호.
**반전이 아니라 상승 추세 지속 패턴**으로 쓰는 것이 정석이다.
암호화폐에서는 역컵앤핸들을 숏 패턴으로도 적용한다.

V자는 컵이 아니다. 바닥이 둥글어야 한다 — 급락 후 급반등은 매물 소화가
일어나지 않은 것이고, 그게 이 패턴이 작동하는 이유를 없앤다.

핸들 조건:
  컵 상단 절반에서 형성 (하단까지 내려오면 컵이 실패한 것)
  컵 상승분의 1/3 이하로 얕게
  거래량이 마름

주의: 정통 컵앤핸들은 비교적 장기간 형성된다. 5분봉에서 지나치게 작은
모양을 컵앤핸들로 해석하지 않도록 최소 길이를 강제한다.
"""

from __future__ import annotations

import pandas as pd

from strategies.base import Prepared, Signal
from strategies.xrp_patterns import base as xb
from strategies.xrp_patterns import chart_base as cb


NAME = f"{xb.FAMILY}.cup_handle"

# 컵의 최소 길이. 5분봉에서 작은 요철을 컵으로 보지 않기 위한 하한.
MIN_CUP_BARS = 30
MAX_CUP_BARS = 150

# 컵 깊이 범위 (림 대비).
MIN_CUP_DEPTH = 0.01
MAX_CUP_DEPTH = 0.08

# 두 림의 높이 차이 허용 오차.
RIM_TOLERANCE = 0.012

# 핸들 조건
MIN_HANDLE_BARS = 3
MAX_HANDLE_BARS = 25
MAX_HANDLE_RETRACE = 1 / 3      # 컵 상승분 대비

# 바닥이 둥근가 — 중앙부가 전체 깊이의 이 비율 이상을 차지해야 한다.
# V자는 최저점 근처에 머문 시간이 짧아 이 값이 작게 나온다.
MIN_ROUNDNESS = 0.30


class CupHandle:
    """U자 컵 + 얕은 핸들 + 핸들 상단 종가 돌파."""

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
        if self._pivots is None or as_of < MIN_CUP_BARS + 10 or as_of >= len(df):
            return []

        atr = xb.atr_at(prepared, as_of, self.tf)
        if atr <= 0:
            return []

        recent = self._pivots.upto(as_of, count=8)
        if len(recent) < 3:
            return []

        for start in range(len(recent) - 3, -1, -1):
            found = self._match(prepared, df, as_of, recent[start : start + 3], atr)
            if found is not None:
                return [found]
        return []

    def _match(self, prepared, df, as_of, three, atr) -> Signal | None:
        left, bottom, right = three

        if [p.kind for p in three] == ["high", "low", "high"]:
            side, kind, sign = "long", "cup_handle", 1.0
        elif [p.kind for p in three] == ["low", "high", "low"]:
            side, kind, sign = "short", "inverse_cup_handle", -1.0
        else:
            return None

        cup_bars = right.index - left.index
        if not (MIN_CUP_BARS <= cup_bars <= MAX_CUP_BARS):
            return None
        if as_of - left.index > cb.MAX_PATTERN_BARS:
            return None

        # 두 림의 높이가 비슷해야 한다
        if not cb.near(left.price, right.price, RIM_TOLERANCE):
            return None

        rim = (left.price + right.price) / 2.0
        depth = abs(rim - bottom.price) / rim
        if not (MIN_CUP_DEPTH <= depth <= MAX_CUP_DEPTH):
            return None

        # 둥근가 — V자 배제
        if not self._is_rounded(df, left.index, right.index, bottom.price, rim, sign):
            return None

        # 핸들: 오른쪽 림 이후의 얕은 조정
        handle = df.iloc[right.index + 1 : as_of + 1]
        handle_bars = len(handle)
        if not (MIN_HANDLE_BARS <= handle_bars <= MAX_HANDLE_BARS):
            return None

        cup_rise = abs(rim - bottom.price)
        if sign > 0:
            handle_low = float(handle["low"].min())
            retrace = (rim - handle_low) / cup_rise
            level = float(handle["high"].iloc[:-1].max()) if handle_bars > 1 else rim
            # 핸들은 컵 상단 절반에서 형성돼야 한다
            if handle_low < bottom.price + cup_rise * 0.5:
                return None
        else:
            handle_high = float(handle["high"].max())
            retrace = (handle_high - rim) / cup_rise
            level = float(handle["low"].iloc[:-1].min()) if handle_bars > 1 else rim
            if handle_high > bottom.price - cup_rise * 0.5:
                return None

        if retrace > MAX_HANDLE_RETRACE:
            return None

        # **완성 확인**: 핸들 상단(하단) 종가 돌파
        if not cb.close_break(df, as_of, level, side):
            return None

        # 핸들에서 거래량이 말라야 한다
        if not self._handle_dry(df, right.index + 1, as_of - 1):
            return None

        if xb.blocked(prepared, as_of, side, self.tf) is not None:
            return None

        # 선행 추세 — 지속 패턴이므로 컵 이전이 같은 방향이어야 한다
        before = max(0, left.index - 40)
        prior = df["close"].iloc[before : left.index + 1]
        prior_ok = bool(
            len(prior) > 5
            and (
                (side == "long" and prior.iloc[-1] > prior.iloc[0])
                or (side == "short" and prior.iloc[-1] < prior.iloc[0])
            )
        )

        profile = cb.profile_at(df, as_of)
        quality = cb.assess(df, prepared, as_of, side, left.index, level, profile, prior_ok, self.tf)
        quality.append("handle_dry")

        if cb.retest_ok(df, as_of - 1, as_of, level, side):
            quality.append("retested")
        if len(quality) < cb.MIN_QUALITY:
            return None

        close = float(df["close"].iloc[as_of])

        # 손절 = 무효화 = 핸들 극값. 핸들 저점을 깨면 돌파가 실패한 것이다.
        stop_hint = (
            float(handle["low"].min()) if side == "long" else float(handle["high"].max())
        )
        if side == "long" and stop_hint >= close:
            return None
        if side == "short" and stop_hint <= close:
            return None

        return Signal(
            strategy=self.name,
            symbol=self.symbol,
            side=side,
            entry=close,
            stop_hint=stop_hint,
            target_hint=xb.vpvr_target(df, as_of, close, side),
            score=float(len(quality)),
            reasons=[kind, *quality],
            context={
                "atr": atr,
                "rim": rim,
                "cup_bottom": bottom.price,
                "cup_bars": cup_bars,
                "handle_bars": handle_bars,
                "handle_retrace": retrace,
                "pattern_height": cup_rise,
                "measured_target": cb.pattern_target(close, cup_rise, side),
                "quality": quality,
                "rsi": float(prepared.at(self.tf.entry, "rsi", as_of)),
            },
            order_type="market",
        )

    @staticmethod
    def _is_rounded(df, left_i, right_i, bottom_price, rim, sign) -> bool:
        """바닥 근처에 머문 시간이 충분한가. V자를 배제한다."""

        seg = df.iloc[left_i : right_i + 1]
        if len(seg) < 6:
            return False

        cup_rise = abs(rim - bottom_price)
        if cup_rise <= 0:
            return False

        # 바닥에서 깊이의 30% 이내에 머문 봉의 비율
        if sign > 0:
            near_bottom = seg["low"] <= bottom_price + cup_rise * 0.3
        else:
            near_bottom = seg["high"] >= bottom_price - cup_rise * 0.3

        return bool(near_bottom.mean() >= MIN_ROUNDNESS)

    @staticmethod
    def _handle_dry(df, start, end) -> bool:
        """핸들 거래량이 컵 형성기보다 작은가."""

        if end < start:
            return False
        handle_vol = df["volume"].iloc[start : end + 1].mean()
        prior_vol = df["volume"].iloc[max(0, start - 30) : start].mean()
        return bool(prior_vol > 0 and handle_vol < prior_vol)
