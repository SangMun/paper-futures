"""BTC 추세 추종 (스윙).

일봉 MA50/200으로 국면을 정하고, 1시간봉에서 진입 시점을 잡는다.
같은 일봉 MA50/200 선은 눌림목 타점으로도 쓴다.

두 갈래로 진입한다.
  A) 눌림목  — 1시간봉이 일봉 MA50/200을 건드리고 확인 캔들이 나옴
  B) 되돌림  — RSI 35~50 + 볼린저 하단 부근 + 확인 캔들

둘을 OR로 두는 이유는 성격이 다르기 때문이다. A는 명확한 레벨이 있을 때만
발생하고 B는 레벨이 없어도 눌림을 잡는다. 로그에 어느 쪽인지 남겨서
나중에 따로 집계한다.

분할 진입이지만 트랜치 번호는 여기서 정하지 않는다. 전략은 상태를 갖지
않으므로 지금 몇 번째인지 알 수 없다. 엔진이 보유 포지션을 보고 매긴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import INDICATORS
from core.candles import interval_seconds
from indicators import patterns, vpvr
from strategies.base import Prepared, Signal, prepare_standard


NAME = "btc_trend"

ENTRY_TF = "1h"
CONTEXT_TF = "1d"

# 손절 힌트로 쓸 최근 저점 구간 (1시간봉 개수).
# 20봉 = 약 하루. 눌림의 바닥을 담되 스윙 전체로 넓어지지는 않는 길이.
SWING_LOOKBACK = 20

# 눌림목 판정 허용 오차. 이평선을 정확히 찍는 일은 드물다.
MA_TOUCH_TOLERANCE = 0.004

# 되돌림 진입에 쓰는 볼린저 위치 상한 (%B).
BB_PULLBACK_MAX = 0.35

MAX_TRANCHES = 3


def context_index(
    entry_ts: np.ndarray,
    context_ts: np.ndarray,
    entry_tf: str = ENTRY_TF,
    context_tf: str = CONTEXT_TF,
) -> np.ndarray:
    """진입 봉마다 "그때 이미 마감돼 있던" 상위 봉의 위치.

    이게 이 전략에서 미래 참조가 생기기 가장 쉬운 자리다.
    1시간봉 i를 판단하는 시점은 그 봉이 마감된 `ts_i + 3600`이고,
    일봉 d는 `ts_d + 86400`이 지나야 마감된다. 따라서 쓸 수 있는 일봉은

        ts_d + 86400 <= ts_i + 3600

인 것뿐이다. 단순히 "같은 날짜의 일봉"을 쓰면 아직 만들어지지도 않은
    종가를 보게 되고, 과거 검증 결과가 실제보다 좋아진다.

    해당하는 상위 봉이 없으면 -1.
    """

    entry_step = interval_seconds(entry_tf)
    context_step = interval_seconds(context_tf)

    deadline = entry_ts + entry_step - context_step
    return np.searchsorted(context_ts, deadline, side="right") - 1


class BtcTrend:
    """일봉 국면 + 1시간봉 진입."""

    name = NAME

    def __init__(self, symbol: str = "BTC_USDT"):
        self.symbol = symbol

    # ---------------------------------------------------------------- 준비

    def prepare(self, frames: dict[str, pd.DataFrame]) -> Prepared:
        """표준 지표 + 일봉 값을 1시간봉 축에 정렬한 것.

        정렬을 여기서 한 번만 해두면 generate()는 인덱싱만 하면 된다.
        봉마다 상위 봉을 찾아 올라가면 리플레이가 O(n^2)가 된다.
        """

        prepared = prepare_standard(frames)

        entry = frames[ENTRY_TF]
        context = frames[CONTEXT_TF]

        idx = context_index(
            entry["ts"].to_numpy(),
            context["ts"].to_numpy(),
        )

        ctx_series = prepared.series[CONTEXT_TF]
        aligned: dict[str, pd.Series] = {
            "ctx_index": pd.Series(idx, index=entry.index),
        }

        # 상위 봉 값을 진입 봉 축으로 끌어온다. idx == -1 구간은 NaN/none.
        for name in ("regime", "ma_fast", "ma_slow"):
            src = ctx_series[name].to_numpy()
            fill = "none" if name == "regime" else np.nan
            values = [src[i] if i >= 0 else fill for i in idx]
            aligned[f"ctx_{name}"] = pd.Series(values, index=entry.index)

        prepared.series[ENTRY_TF].update(aligned)
        return prepared

    # ---------------------------------------------------------------- 신호

    def generate(self, prepared: Prepared, as_of: int) -> list[Signal]:
        df = prepared.frame(ENTRY_TF)
        if as_of < 1 or as_of >= len(df):
            return []

        regime = prepared.at(ENTRY_TF, "ctx_regime", as_of)
        if regime not in ("bull", "bear"):
            return []

        side = "long" if regime == "bull" else "short"

        macd = prepared.at(ENTRY_TF, "macd", as_of)
        macd_sig = prepared.at(ENTRY_TF, "macd_signal", as_of)
        rsi = prepared.at(ENTRY_TF, "rsi", as_of)
        atr = prepared.at(ENTRY_TF, "atr", as_of)
        bb_pos = prepared.at(ENTRY_TF, "bb_pos", as_of)

        if any(pd.isna(v) for v in (macd, macd_sig, rsi, atr)):
            return []
        if atr <= 0:
            return []

        # 방향 필터. 0선 기준만 하드 게이트로 쓴다.
        #
        # 명세 §5.1은 "MACD > 시그널 **그리고** MACD > 0"을 요구하지만,
        # 그건 합류 존 진입(피보나치)을 전제로 쓰인 규칙이다. 눌림목 진입에
        # 그대로 붙이면 조건이 서로 모순된다 — RSI 35~50에 볼린저 하단이라는
        # 건 가격이 눌려 있다는 뜻이고, MACD가 시그널 위라는 건 모멘텀이
        # 이미 돌아섰다는 뜻이다. 눌림 바닥에서 둘은 동시에 성립하지 않는다.
        #
        # 실제로 BTC 1시간봉 9,600봉에서 두 조건의 교집합은 0건이었다.
        # 크로스는 하드 게이트에서 빼고 가점으로만 쓴다. "모멘텀이 돌아섰는가"는
        # 확인 캔들이 더 이른 시점에, 더 정확한 단위로 판정한다.
        if side == "long" and not macd > 0:
            return []
        if side == "short" and not macd < 0:
            return []

        # 과열 구간에서는 진입하지 않는다 (명세 §5.4).
        if side == "long" and rsi >= INDICATORS.rsi_overbought:
            return []
        if side == "short" and rsi <= 100 - INDICATORS.rsi_overbought:
            return []

        reasons: list[str] = [f"regime_{regime}", f"macd_{side}"]
        score = 1.0

        # 크로스는 가점. 눌림 바닥에서는 거의 나오지 않지만, 나왔다면
        # 모멘텀이 이미 돌아선 뒤라 신호의 질이 다르다.
        crossed = macd > macd_sig if side == "long" else macd < macd_sig
        if crossed:
            reasons.append("macd_cross")
            score += 1.0

        touched = self._ma_touch(prepared, df, as_of, side)
        reasons.extend(touched)
        score += 2.0 * len(touched)

        in_zone = INDICATORS.rsi_pullback_min <= rsi <= INDICATORS.rsi_pullback_max
        if in_zone:
            reasons.append("rsi_pullback")
            score += 1.0

        deep = self._bb_pullback(bb_pos, side)
        if deep:
            reasons.append("bb_pullback")
            score += 1.0

        # 트리거: 눌림목(A) 또는 되돌림(B) 중 하나는 있어야 한다.
        if not touched and not (in_zone and deep):
            return []

        # 확인 캔들. 레벨에 닿은 것만으로는 진입하지 않는다 (명세 §5.3).
        level = self._confirm_level(prepared, df, as_of, side, touched)
        window = df.iloc[max(0, as_of - 1) : as_of + 1]
        if not bool(patterns.confirmation(window, level, side).iloc[-1]):
            return []
        reasons.append("confirmed")

        entry = float(df["close"].iloc[as_of])
        stop_hint = self._stop_hint(df, as_of, side)
        if stop_hint is None:
            return []

        return [
            Signal(
                strategy=NAME,
                symbol=self.symbol,
                side=side,
                entry=entry,
                stop_hint=stop_hint,
                target_hint=self._target_hint(df, as_of, entry, side),
                score=score,
                reasons=reasons,
                context={
                    "atr": float(atr),
                    "rsi": float(rsi),
                    "macd_hist": float(macd - macd_sig),
                    "bb_pos": None if pd.isna(bb_pos) else float(bb_pos),
                    "regime": regime,
                },
                tranche=1,              # 엔진이 보유 포지션을 보고 다시 매긴다
                max_tranches=MAX_TRANCHES,
            )
        ]

    # ---------------------------------------------------------------- 내부

    def _ma_touch(self, prepared: Prepared, df: pd.DataFrame, as_of: int, side: str) -> list[str]:
        """이번 봉이 일봉 MA50/200을 건드렸는지.

        종가가 아니라 고저 범위로 본다. 눌림목은 꼬리로만 닿고 종가는
        위에서 마감하는 경우가 오히려 전형적이다.
        """

        low = float(df["low"].iloc[as_of])
        high = float(df["high"].iloc[as_of])
        hits: list[str] = []

        for key, label in (("ctx_ma_fast", "ma50_touch"), ("ctx_ma_slow", "ma200_touch")):
            line = prepared.at(ENTRY_TF, key, as_of)
            if pd.isna(line) or line <= 0:
                continue
            band = line * MA_TOUCH_TOLERANCE
            if low <= line + band and high >= line - band:
                hits.append(label)

        return hits

    @staticmethod
    def _bb_pullback(bb_pos: float, side: str) -> bool:
        if pd.isna(bb_pos):
            return False
        return bb_pos <= BB_PULLBACK_MAX if side == "long" else bb_pos >= 1 - BB_PULLBACK_MAX

    def _confirm_level(
        self,
        prepared: Prepared,
        df: pd.DataFrame,
        as_of: int,
        side: str,
        touched: list[str],
    ) -> float:
        """확인 캔들의 기준선.

        눌림목이면 건드린 이평선, 아니면 볼린저 중심선을 쓴다.
        기준선이 없으면 되돌림 거부 판정이 성립하지 않는다.
        """

        if touched:
            key = "ctx_ma_fast" if "ma50_touch" in touched else "ctx_ma_slow"
            return float(prepared.at(ENTRY_TF, key, as_of))

        mid = prepared.at(ENTRY_TF, "bb_mid", as_of)
        return float(mid) if not pd.isna(mid) else float(df["close"].iloc[as_of])

    @staticmethod
    def _stop_hint(df: pd.DataFrame, as_of: int, side: str) -> float | None:
        """최근 SWING_LOOKBACK봉의 극값. ATR 버퍼는 risk 레이어가 더한다."""

        start = max(0, as_of - SWING_LOOKBACK + 1)
        window = df.iloc[start : as_of + 1]
        if window.empty:
            return None

        level = float(window["low"].min()) if side == "long" else float(window["high"].max())
        entry = float(df["close"].iloc[as_of])

        # 진입가와 같은 쪽이면 손절이 성립하지 않는다.
        if (side == "long" and level >= entry) or (side == "short" and level <= entry):
            return None
        return level

    def _target_hint(self, df: pd.DataFrame, as_of: int, entry: float, side: str) -> float | None:
        """위쪽(롱) 최근접 VPVR HVN. 명세 §7-1.

        없으면 None을 돌려주고 risk 레이어가 R 배수로 떨어진다.
        """

        profile = vpvr.profile(df, as_of=as_of)
        if profile is None:
            return None

        target = (
            profile.nearest_above(entry) if side == "long" else profile.nearest_below(entry)
        )
        return float(target) if target is not None else None
