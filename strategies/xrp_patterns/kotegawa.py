"""이격도 역추세 (고테가와 타카시 매매법에서 착안).

고테가와(BNF)의 대표 기법은 **25일 이동평균 이격도**를 이용한 역장이다.
가격이 이평선에서 아래로 크게 벌어진 종목을 사서 평균으로 되돌아올 때 판다.
급락 자체가 아니라 "평균에서 얼마나 비정상적으로 벌어졌는가"를 본다.

**원전과 다르게 만든 부분 — 그대로 옮길 수 없어서다.**

  ① 문턱을 고정 비율(-20%)로 두지 않는다.
     일본 주식 일봉에서 나온 수치이고, XRP 5분봉에서 -20%는 사실상
     발생하지 않는다. 대신 최근 구간의 이격도 분포에서 하위 백분위를 쓴다.
     "평소보다 비정상적으로 벌어졌는가"라는 원래 의도를 유지하면서
     종목·타임프레임에 자동으로 맞춰진다.

  ② 반등 확인 캔들을 요구한다.
     떨어지는 칼을 그대로 받지 않는다. 이격도가 벌어진 것만으로는
     진입하지 않고 되돌림 거부가 나와야 한다.

  ③ 익절 목표가 이평선 회귀다.
     다른 네 패턴은 VPVR HVN을 쓰지만 이 전략은 평균 회귀가 논리 자체라
     MA를 목표로 둔다. 다만 MA가 손절보다 가까우면 진입하지 않는다.

숏에도 대칭으로 적용한다. 원전은 매수 위주지만 무기한선물은 양방향이고,
과열 뒤 되돌림은 같은 논리다.

**대세 정렬(`trend_align`)은 재보고 넣지 않았다.**

실거래에서 이 전략이 2026-08-25~26 하락 구간에 진입가를 낮춰가며
(1.4571 -> 1.3709) 5연속 손절해 -53.11 USDT를 냈다. 떨어지는 칼을 받은
모양이라 wedge.5m·head_shoulders.5m처럼 4시간봉 역행을 막아 보았는데,
275,241봉 리플레이에서 이 전략 손익이 +69.0에서 -2.3으로 무너졌다
(174건 -> 167건). 평균 회귀는 **추세와 반대인 자리가 곧 수익원**이라
돌파 패턴과 정반대로 반응한다. 위 5연패는 나쁜 구간이었지 구조적 결함이
아니었다.

같은 이유로 손절 후 쿨다운과 연속 손절 한도 축소(5 -> 3)도 되돌렸다.
전체 수익률이 +83.8%에서 각각 +65.1% / +60.6%로, 최대낙폭도 11.2%에서
14.5% / 16.6%로 함께 나빠졌다. 손실은 승리와 같은 국면에 몰려 있어
재진입을 막으면 뒤따르는 이익 거래까지 사라진다.
"""

from __future__ import annotations

import pandas as pd

from indicators import ma as ma_mod
from indicators import patterns
from strategies.base import Prepared, Signal
from strategies.xrp_patterns import base as xb
from strategies.xrp_patterns import chart_base as cb


NAME = f"{xb.FAMILY}.kotegawa"

# 이격도 기준선. 원전의 25일선을 봉 수로 옮긴 것.
MA_PERIOD = 25

# 이격도 분포를 볼 구간과 문턱(백분위).
# 하위 3%면 "최근 하루 반 중 가장 벌어진 축"에 해당한다.
DEVIATION_LOOKBACK = 400
DEVIATION_PERCENTILE = 3.0

# 이격도가 최소 이만큼은 벌어져야 한다. 백분위만 쓰면 잔잔한 구간에서
# 아주 작은 이격도도 하위 3%가 되어 신호가 남발된다.
MIN_DEVIATION = 0.006


class Kotegawa:
    """이평선 이격도 과대 → 반등 확인 → 평균 회귀 목표."""

    def __init__(self, timeframe: str = "5m", symbol: str = "XRP_USDT"):
        self.symbol = symbol
        self.tf = xb.LADDER[timeframe]
        # 태그에 봉을 붙인다. 같은 패턴도 봉마다 성과가 달라
        # 따로 집계해야 한다 (실측: double_top 5분 -0.21%, 1시간 +0.40%).
        self.name = f"{NAME}.{timeframe}"
        self._pivots: cb.PivotView | None = None


    def prepare(self, frames: dict[str, pd.DataFrame]) -> Prepared:
        prepared, _ = cb.chart_prepare(frames, self.tf)
        entry = frames[self.tf.entry]

        center = ma_mod.sma(entry, MA_PERIOD)
        deviation = (entry["close"] - center) / center.where(center > 0)

        prepared.series[self.tf.entry].update(
            {
                "kt_ma": center,
                "kt_dev": deviation,
                # 각 시점에서 "지금 이격도가 최근 분포의 몇 분위인가".
                # 과거만 보므로 미래를 참조하지 않는다.
                "kt_rank": deviation.rolling(
                    DEVIATION_LOOKBACK, min_periods=DEVIATION_LOOKBACK
                ).rank(pct=True)
                * 100.0,
            }
        )
        return prepared

    def generate(self, prepared: Prepared, as_of: int) -> list[Signal]:
        df = prepared.frame(self.tf.entry)
        if as_of < DEVIATION_LOOKBACK + 2 or as_of >= len(df):
            return []

        dev = prepared.at(self.tf.entry, "kt_dev", as_of)
        rank = prepared.at(self.tf.entry, "kt_rank", as_of)
        center = prepared.at(self.tf.entry, "kt_ma", as_of)
        atr = xb.atr_at(prepared, as_of, self.tf)

        if any(pd.isna(v) for v in (dev, rank, center)) or atr <= 0 or center <= 0:
            return []

        # 아래로 크게 벌어졌으면 롱, 위로면 숏
        if rank <= DEVIATION_PERCENTILE and dev <= -MIN_DEVIATION:
            side = "long"
        elif rank >= 100 - DEVIATION_PERCENTILE and dev >= MIN_DEVIATION:
            side = "short"
        else:
            return []

        if xb.blocked(prepared, as_of, side, self.tf) is not None:
            return []

        # 떨어지는 칼을 그대로 받지 않는다. 반등 확인 캔들이 있어야 한다.
        window = df.iloc[as_of - 1 : as_of + 1]
        if not bool(patterns.confirmation(window, float(df["low"].iloc[as_of]), side).iloc[-1]):
            return []

        close = float(df["close"].iloc[as_of])

        # 손절 = 이격이 더 벌어지는 쪽. 최근 극값 바깥.
        start = max(0, as_of - 10)
        recent = df.iloc[start : as_of + 1]
        stop_hint = float(recent["low"].min() if side == "long" else recent["high"].max())

        if side == "long" and stop_hint >= close:
            return []
        if side == "short" and stop_hint <= close:
            return []

        # 목표 = 이평선 회귀. 이 전략만 VPVR이 아니라 MA를 쓴다 —
        # 평균 회귀가 논리 자체이기 때문이다.
        target = float(center)
        if (side == "long" and target <= close) or (side == "short" and target >= close):
            return []

        oi = prepared.at(self.tf.entry, "oi_change", as_of)
        rv = prepared.at(self.tf.entry, "rvol", as_of)

        reasons = ["kotegawa", f"dev_{dev:.2%}", f"rank_{rank:.0f}"]
        score = 2.0
        if pd.notna(rv) and rv >= cb.BREAKOUT_RVOL:
            reasons.append(f"rvol_{rv:.1f}")     # 투매 소진 신호
            score += 1.0
        if pd.notna(oi) and oi <= -cb.OI_EXPANSION:
            # 이격 확대 구간의 OI 감소는 청산이 소화됐다는 뜻이라
            # 여기서는 오히려 유리하다. 돌파 패턴과 해석이 반대다.
            reasons.append(f"oi_flush_{oi:.2%}")
            score += 1.0

        return [
            Signal(
                strategy=self.name,
                symbol=self.symbol,
                side=side,
                entry=close,
                stop_hint=stop_hint,
                target_hint=target,
                score=score,
                reasons=reasons,
                context={
                    "atr": atr,
                    "ma": float(center),
                    "deviation": float(dev),
                    "deviation_rank": float(rank),
                    "oi_change": None if pd.isna(oi) else float(oi),
                    "rsi": float(prepared.at(self.tf.entry, "rsi", as_of)),
                },
                order_type="market",
            )
        ]
