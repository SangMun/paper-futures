"""피보나치 되돌림 진입.

명세 §5. ATFX 문서의 "피보나치 단독 사용 금지", "추세 무시 금지"를
구조로 강제한다 — 존 점수만으로는 진입하지 않고, 국면 필터와 확인 캔들을
모두 통과해야 신호가 나온다.

btc_trend와 국면 필터를 공유하되 진입 근거가 완전히 다르다.
  btc_trend: 이평선 눌림목 / 볼린저 되돌림
  여기:      여러 스윙의 피보나치 레벨이 겹치는 존

두 전략이 같은 봉에서 동시에 신호를 낼 수 있다. 그건 의도된 것이고,
엔진의 종목별 리스크 상한(3%)이 과도한 중복을 막는다.
"""

from __future__ import annotations

import pandas as pd

from config.settings import FIB, INDICATORS
from indicators import confluence, fib as fib_mod, patterns, pivots as pv
from indicators import vpvr
from strategies.base import Prepared, Signal, prepare_standard
from strategies.btc_trend import CONTEXT_TF, ENTRY_TF, context_index


NAME = "btc_fib"


class FibRetracement:
    """합류 존 도달 + 확인 캔들 → 진입."""

    name = NAME

    def __init__(self, symbol: str = "BTC_USDT", min_zone_score: float = FIB.min_zone_score):
        self.symbol = symbol
        self.min_zone_score = min_zone_score
        self._swings: list[pv.Swing] = []

    # ---------------------------------------------------------------- 준비

    def prepare(self, frames: dict[str, pd.DataFrame]) -> Prepared:
        """표준 지표 + 일봉 국면 정렬 + **스윙 전체 사전 계산**.

        스윙을 여기서 한 번만 만드는 게 핵심이다. 봉마다 피벗을 다시 찾으면
        전 구간 롤링을 반복해 9,600봉 리플레이가 63초에서 7초로 갈린다.
        """

        prepared = prepare_standard(frames)
        entry = frames[ENTRY_TF]
        context = frames[CONTEXT_TF]

        idx = context_index(entry["ts"].to_numpy(), context["ts"].to_numpy())
        ctx = prepared.series[CONTEXT_TF]

        aligned = {"ctx_index": pd.Series(idx, index=entry.index)}
        for name in ("regime", "ma_fast", "ma_slow"):
            src = ctx[name].to_numpy()
            fill = "none" if name == "regime" else float("nan")
            aligned[f"ctx_{name}"] = pd.Series(
                [src[i] if i >= 0 else fill for i in idx], index=entry.index
            )
        prepared.series[ENTRY_TF].update(aligned)

        self._swings = pv.all_swings(entry, n=FIB.pivot_n, min_swing_pct=FIB.min_swing_pct)
        return prepared

    # ---------------------------------------------------------------- 신호

    def generate(self, prepared: Prepared, as_of: int) -> list[Signal]:
        df = prepared.frame(ENTRY_TF)
        if as_of < 2 or as_of >= len(df):
            return []

        regime = prepared.at(ENTRY_TF, "ctx_regime", as_of)
        if regime not in ("bull", "bear"):
            return []
        side = "long" if regime == "bull" else "short"

        atr = prepared.at(ENTRY_TF, "atr", as_of)
        macd = prepared.at(ENTRY_TF, "macd", as_of)
        rsi = prepared.at(ENTRY_TF, "rsi", as_of)
        if any(pd.isna(v) for v in (atr, macd, rsi)) or atr <= 0:
            return []

        # 방향 필터. btc_trend와 같은 이유로 0선만 하드 게이트로 쓴다 —
        # 존은 되돌림 자리라, 크로스까지 요구하면 조건이 서로 모순된다.
        if side == "long" and macd <= 0:
            return []
        if side == "short" and macd >= 0:
            return []

        # 보조 확인 (§5.4). 과열이면 보류한다.
        if side == "long" and rsi >= INDICATORS.rsi_overbought:
            return []
        if side == "short" and rsi <= 100 - INDICATORS.rsi_overbought:
            return []

        zone = self._zone_at(prepared, df, as_of, side, float(atr))
        if zone is None:
            return []

        # 확인 캔들 (§5.3). 존에 닿는 것만으로는 진입하지 않는다 —
        # 레벨을 그대로 관통하는 경우를 걸러내지 못한다.
        window = df.iloc[as_of - 1 : as_of + 1]
        if not bool(patterns.confirmation(window, zone.price, side).iloc[-1]):
            return []

        entry_price = float(df["close"].iloc[as_of])
        stop_hint = self._stop_hint(zone, entry_price, side)
        if stop_hint is None:
            return []

        return [
            Signal(
                strategy=NAME,
                symbol=self.symbol,
                side=side,
                entry=entry_price,
                stop_hint=stop_hint,
                target_hint=self._target_hint(df, as_of, entry_price, side, zone),
                score=zone.score,
                reasons=[f"regime_{regime}", "zone", *zone.members[:5], "confirmed"],
                context={
                    "atr": float(atr),
                    "rsi": float(rsi),
                    "zone_price": zone.price,
                    "zone_score": zone.score,
                    "zone_members": zone.members,
                    "swings": [
                        {"low": s.low, "high": s.high, "direction": s.direction}
                        for s in zone.swings
                    ],
                },
            )
        ]

    # ---------------------------------------------------------------- 내부

    def _zone_at(self, prepared, df, as_of, side, atr):
        """이번 봉이 닿은 합류 존 중 점수가 가장 높은 것."""

        ma_fast = prepared.at(ENTRY_TF, "ctx_ma_fast", as_of)
        ma_slow = prepared.at(ENTRY_TF, "ctx_ma_slow", as_of)

        zones = confluence.find_zones(
            df,
            as_of,
            atr,
            ma_fast=None if pd.isna(ma_fast) else float(ma_fast),
            ma_slow=None if pd.isna(ma_slow) else float(ma_slow),
            precomputed_swings=self._swings,
        )

        low = float(df["low"].iloc[as_of])
        high = float(df["high"].iloc[as_of])

        for zone in zones:
            if zone.side != side or zone.score < self.min_zone_score:
                continue
            if zone.touched_by(low, high):
                return zone
        return None

    @staticmethod
    def _stop_hint(zone, entry: float, side: str) -> float | None:
        """다음 하위(롱)/상위(숏) 피보나치 레벨 (§6).

        여러 스윙이 겹친 존이면 가장 보수적인 쪽(진입가에서 먼 쪽)을 쓴다.
        ATR 버퍼는 risk 레이어가 더한다.
        """

        candidates = [
            fib_mod.next_level_below(s, zone.price, side) for s in zone.swings
        ]
        candidates = [
            c for c in candidates
            if (side == "long" and c < entry) or (side == "short" and c > entry)
        ]
        if not candidates:
            return None
        return min(candidates) if side == "long" else max(candidates)

    @staticmethod
    def _target_hint(df, as_of, entry, side, zone) -> float | None:
        """1차 목표: 임펄스 고점과 VPVR HVN 중 가까운 쪽 (§7-1)."""

        candidates: list[float] = []

        for s in zone.swings:
            impulse = s.high if side == "long" else s.low
            if (side == "long" and impulse > entry) or (side == "short" and impulse < entry):
                candidates.append(impulse)

        profile = vpvr.profile(df, as_of=as_of)
        if profile is not None:
            node = (
                profile.nearest_above(entry) if side == "long"
                else profile.nearest_below(entry)
            )
            if node is not None:
                candidates.append(float(node))

        if not candidates:
            return None
        return min(candidates) if side == "long" else max(candidates)
