"""헤드앤숄더 / 역헤드앤숄더.

상승 추세 후 왼쪽 어깨 → 머리 → 오른쪽 어깨. **목선 종가 이탈이 확인 신호.**
역패턴은 하락 추세 말기에 방향만 반대로 적용한다.

머리 모양만 보고 목선 이탈 전에 진입하지 않는다 — 목선을 종가로 넘기 전에는
완성된 패턴이 아니다. 그게 이 패턴에서 가장 흔한 실수다.

거래량 구조 (자료 공통):
  왼쪽 어깨 상승 거래량이 크고
  머리로 갈수록 줄고
  오른쪽 어깨 반등이 가장 약하고
  목선 이탈 때 늘어난다

매물대 결합:
  머리·오른쪽 어깨가 VAH(상단 가치영역)에서 거부되면 구조가 좋다
  목선 바로 아래 강한 HVN이 있으면 숏의 기대수익이 작다 → no_wall_ahead로 판정
"""

from __future__ import annotations

import pandas as pd

from config.settings import SCALP
from strategies.base import Prepared, Signal
from strategies.xrp_patterns import base as xb
from strategies.xrp_patterns import chart_base as cb


NAME = f"{xb.FAMILY}.head_shoulders"

# 머리가 어깨보다 이 비율 이상 높아야 한다. 셋이 비슷하면 삼중천장이지 H&S가 아니다.
HEAD_PROMINENCE = 0.006

# 두 어깨 높이 차이 허용 오차.
SHOULDER_TOLERANCE = 0.012


class HeadShoulders:
    """어깨-머리-어깨 + 목선 종가 이탈."""

    def __init__(
        self,
        timeframe: str = "5m",
        symbol: str = "XRP_USDT",
        trend_align: bool = False,
        drop_short_btc_alignment: bool = False,
        drop_long_vwap_alignment: bool = False,
    ):
        self.symbol = symbol
        # 대세(4시간봉)와 반대 방향을 막을지.
        # 949일 실측에서 상승장 숏 -38.0 + 하락장 롱 -62.4가 손실의 전부였고
        # (추세 방향은 상승장 롱 +35.5 / 하락장 숏 +34.5로 플러스),
        # 앞뒤 절반 모두 마이너스라 구간 우연이 아니다.
        self.trend_align = trend_align
        # 숏에서 btc_aligned 품질 항목을 뺄지 (EXP-013).
        #
        # btc_aligned는 "BTC 1시간봉이 같은 방향으로 크게 움직였다"는 확인
        # 항목이다. 실측(BTC 23,065봉)으로 -1% 트리거 뒤 24시간 수익률
        # 평균 +0.70%, 반등 확률 58% — 급락은 확인이 아니라 반등의 전조에
        # 가까웠다. 누적 하락(4~12봉)으로 재도, BTC 자체 국면(bull/bear)으로
        # 걸러도 반등은 사라지지 않았다. 그래서 재정의가 아니라 제거를
        # 택한다. 실제로 head_shoulders 숏 22건을 btc_aligned 유무로
        # 가르면 켜진 쪽 11건은 -61.1(승률 36%), 꺼진 쪽 11건은 -0.4였다.
        # 롱에서는 반대로 무해했다(+3.5)— BTC 급등 뒤엔 그런 반등이 없어서
        # (조정 확률 48%), 롱만 그대로 둔다.
        self.drop_short_btc_alignment = drop_short_btc_alignment
        # 롱에서 vwap_aligned 품질 항목을 뺄지 (EXP-014).
        #
        # vwap_aligned는 "종가가 세션 VWAP 기준으로 방향과 같은 쪽에
        # 있다"는 확인이다. XRP 5m/15m/1h 전체에서 종가가 VWAP 위인
        # 기본 비율은 51.4%로 쏠려 있지 않지만, 패턴이 완성되는 시점만
        # 보면 다르다 — 바닥형(롱)은 방금 반등한 자리라 VWAP 위로 올라선
        # 것 자체가 거의 자동으로 따라오는 반면, 천장형(숏)이 VWAP 아래로
        # 종가를 내리는 건 더 드물고 결정적인 사건에 가깝다. 실측으로
        # head_shoulders·double_top·wedge·cup_handle 네 패턴 전부에서
        # 롱의 vwap_aligned가 숏보다 일관되게 나빴다(220건 합산: 숏 +64.8
        # / 롱 -125.7).
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
        if self._pivots is None or as_of < 60 or as_of >= len(df):
            return []

        atr = xb.atr_at(prepared, as_of, self.tf)
        if atr <= 0:
            return []

        # 최근 피벗 7개 안에서 어깨-머리-어깨 다섯 점을 찾는다.
        recent = self._pivots.upto(as_of, count=7)
        if len(recent) < 5:
            return []

        for start in range(len(recent) - 5, -1, -1):
            found = self._match(prepared, df, as_of, recent[start : start + 5], atr)
            if found is not None:
                return [found]
        return []

    def _match(self, prepared, df, as_of, five, atr) -> Signal | None:
        ls, t1, head, t2, rs = five

        # 정방향(천장): 고-저-고-저-고
        if [p.kind for p in five] == ["high", "low", "high", "low", "high"]:
            side, sign = "short", 1.0
        # 역방향(바닥): 저-고-저-고-저
        elif [p.kind for p in five] == ["low", "high", "low", "high", "low"]:
            side, sign = "long", -1.0
        else:
            return None

        if as_of - ls.index > cb.MAX_PATTERN_BARS:
            return None

        # 머리가 두 어깨보다 뚜렷하게 돌출
        if sign > 0:
            if not (head.price > ls.price and head.price > rs.price):
                return None
            prominence = min(head.price / ls.price, head.price / rs.price) - 1.0
        else:
            if not (head.price < ls.price and head.price < rs.price):
                return None
            prominence = min(ls.price / head.price, rs.price / head.price) - 1.0

        if prominence < HEAD_PROMINENCE:
            return None

        # 두 어깨 높이가 비슷해야 한다
        if not cb.near(ls.price, rs.price, SHOULDER_TOLERANCE):
            return None

        # 목선 = 두 골(정방향) / 두 마루(역방향)를 잇는 선
        neckline = cb.line_at(t1, t2, as_of)
        if neckline <= 0:
            return None

        # **완성 확인**: 종가로 목선 이탈
        if not cb.close_break(df, as_of, neckline, side):
            return None

        if xb.blocked(prepared, as_of, side, self.tf) is not None:
            return None
        if self.trend_align and xb.counter_trend(prepared, as_of, side, self.tf):
            return None

        # 선행 추세: 왼쪽 어깨 이전이 패턴 방향과 반대로 진행했는가
        prior_ok = cb.prior_trend_ok(
            df, ls.index, "up" if side == "short" else "down", 30
        )

        profile = cb.profile_at(df, as_of)
        quality = cb.assess(df, prepared, as_of, side, ls.index, neckline, profile, prior_ok, self.tf)
        if self.drop_short_btc_alignment and side == "short":
            quality = [item for item in quality if item != "btc_aligned"]
        if self.drop_long_vwap_alignment and side == "long":
            quality = [item for item in quality if item != "vwap_aligned"]

        if len(quality) < cb.MIN_QUALITY:
            return None

        close = float(df["close"].iloc[as_of])

        # 손절 = 패턴 무효화 지점 = 오른쪽 어깨 바깥.
        # 목선을 되돌파해 어깨 위로 가면 패턴이 깨진 것이다.
        stop_hint = float(rs.price)
        if side == "short" and stop_hint <= close:
            return None
        if side == "long" and stop_hint >= close:
            return None

        height = abs(head.price - neckline)

        return Signal(
            strategy=self.name,
            symbol=self.symbol,
            side=side,
            entry=close,
            stop_hint=stop_hint,
            target_hint=xb.vpvr_target(df, as_of, close, side),
            score=float(len(quality)),
            reasons=["head_shoulders" if sign > 0 else "inverse_head_shoulders", *quality],
            context={
                "atr": atr,
                "neckline": neckline,
                "head": head.price,
                "left_shoulder": ls.price,
                "right_shoulder": rs.price,
                "pattern_height": height,
                # 참고치. 보장된 도달값이 아니다.
                "measured_target": cb.pattern_target(close, height, side),
                "quality": quality,
                "rsi": float(prepared.at(self.tf.entry, "rsi", as_of)),
            },
            order_type="market",        # 종가 이탈을 쫓아가는 진입이다
        )
