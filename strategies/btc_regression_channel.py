# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Regression-channel calculation adapted from the user-provided
# "Gaussian Channel Strategy [불스토리]" © bullstory1.

"""BTC 롤링 회귀채널 역확장 지정가 전략.

하락 채널에서는 종가가 상단선을 처음 이탈한 뒤, 상단선보다 일정 비율 위에
숏 지정가를 둔다. 상승 채널에서는 종가가 하단선을 처음 이탈한 뒤, 하단선보다
일정 비율 아래에 롱 지정가를 둔다.

1차 목표는 매 봉 직전 확정 채널선으로 갱신한다. 절반 익절 뒤 나머지는 공통
risk/exits의 트레일링 스탑이 관리한다.
"""

from __future__ import annotations

import math
from typing import Literal

import pandas as pd

from config.settings import REGRESSION_CHANNEL
from indicators import regression_channel as rc
from strategies.base import Prepared, Signal, prepare_standard


NAME = "btc_regression_channel"
ENTRY_TF = "1h"
MAJOR_TF = "4h"


def _major_regime(frames: dict[str, pd.DataFrame], entry: pd.DataFrame) -> pd.Series:
    """4시간봉 대세 국면을 1시간봉 축에 정렬한다.

    마감된 상위봉만 본다 — 진행 중인 4시간봉의 종가를 쓰면 미래참조다.
    """

    major = frames.get(MAJOR_TF)
    if major is None or major.empty:
        return pd.Series("none", index=entry.index, dtype="object")

    from indicators.ma import regime
    from strategies.btc_trend import context_index

    idx = context_index(
        entry["ts"].to_numpy(),
        major["ts"].to_numpy(),
        entry_tf=ENTRY_TF,
        context_tf=MAJOR_TF,
    )
    reg = regime(major, fast=20, slow=60).to_numpy()
    return pd.Series(
        [reg[i] if i >= 0 else "none" for i in idx], index=entry.index, dtype="object"
    )


class BtcRegressionChannel:
    """회귀채널 반대편 과확장에서 채널 복귀를 노린다."""

    name = NAME

    def __init__(
        self,
        symbol: str = "BTC_USDT",
        entry_offset_pct: float | None = None,
        direction: Literal["both", "long", "short"] = "both",
        regime_filter: bool = False,
        aligned_risk_mult: float = 1.0,
    ):
        self.symbol = symbol
        self.entry_offset_pct = (
            REGRESSION_CHANNEL.entry_offset_pct
            if entry_offset_pct is None
            else float(entry_offset_pct)
        )
        # 0이면 이격 지정가 대신 이탈봉 종가 시장가로 들어간다.
        if not math.isfinite(self.entry_offset_pct) or not 0 <= self.entry_offset_pct < 0.1:
            raise ValueError(
                f"회귀채널 진입 이격은 0~10% 사이여야 한다: {self.entry_offset_pct}"
            )
        if direction not in ("both", "long", "short"):
            raise ValueError(f"direction은 both/long/short: {direction!r}")
        self.direction = direction
        # 4시간봉 국면으로 진입을 거른다.
        #
        # 949일 실측 국면별 손익:
        #   상승 롱   4건 +124.07     횡보 롱  35건 +12.96
        #   상승 숏  10건  -49.58     횡보 숏  38건 -89.16
        #   하락 롱  14건  -35.91     하락 숏   2건 +31.09
        #
        # 대세 역행(상승숏·하락롱)과 횡보 숏을 빼면 103건 -6.53이
        # 41건 +168.12가 되고, 앞뒤 절반 모두 플러스다.
        self.regime_filter = regime_filter
        # 추세 방향과 일치하는 자리(상승 롱 / 하락 숏)의 리스크 배수.
        # 건당 기대값이 +25.5 vs 횡보장 +0.37로 약 69배 차이다.
        # 다만 그 표본이 6건뿐이라 2배를 넘기지 않는다.
        self.aligned_risk_mult = float(aligned_risk_mult)

    def prepare(self, frames: dict[str, pd.DataFrame]) -> Prepared:
        prepared = prepare_standard(frames)
        df = frames[ENTRY_TF]
        channel = rc.channel(
            df,
            length=REGRESSION_CHANNEL.length,
            width_mult=REGRESSION_CHANNEL.width_mult,
        )
        prepared.series[ENTRY_TF].update(
            {f"reg_{name}": channel[name] for name in channel.columns}
        )
        prepared.series[ENTRY_TF]["major_regime"] = _major_regime(frames, df)
        return prepared

    def target_before_bar(
        self,
        frames: dict[str, pd.DataFrame],
        as_of: int,
        side: str,
    ) -> float | None:
        """현재 청산봉보다 앞서 확정된 롤링 채널 목표."""

        if side not in ("long", "short"):
            raise ValueError(f"side는 long/short: {side!r}")
        df = frames[ENTRY_TF]
        length = REGRESSION_CHANNEL.length
        if as_of < length - 1 or as_of >= len(df):
            return None

        # 마지막 length개만 계산하면 호출당 비용이 고정이다. manage_positions가
        # 매 봉 호출하므로 전 구간을 다시 계산하면 리플레이가 O(n²)가 된다.
        window = df.iloc[as_of - length + 1 : as_of + 1]
        point = rc.channel(
            window,
            length=length,
            width_mult=REGRESSION_CHANNEL.width_mult,
        ).iloc[-1]
        target = point["lower"] if side == "long" else point["upper"]
        return None if pd.isna(target) else float(target)

    def generate(self, prepared: Prepared, as_of: int) -> list[Signal]:
        if as_of < REGRESSION_CHANNEL.length or as_of >= len(prepared.frame(ENTRY_TF)):
            return []

        df = prepared.frame(ENTRY_TF)
        slope = prepared.at(ENTRY_TF, "reg_slope", as_of)
        upper = prepared.at(ENTRY_TF, "reg_upper", as_of)
        lower = prepared.at(ENTRY_TF, "reg_lower", as_of)
        r2 = prepared.at(ENTRY_TF, "reg_r2", as_of)
        atr = prepared.at(ENTRY_TF, "atr", as_of)
        prev_upper = prepared.at(ENTRY_TF, "reg_upper", as_of - 1)
        prev_lower = prepared.at(ENTRY_TF, "reg_lower", as_of - 1)
        values = (slope, upper, lower, r2, atr, prev_upper, prev_lower)
        if any(pd.isna(value) for value in values) or float(atr) <= 0:
            return []

        close = float(df["close"].iloc[as_of])
        high = float(df["high"].iloc[as_of])
        low = float(df["low"].iloc[as_of])
        prev_high = float(df["high"].iloc[as_of - 1])
        prev_low = float(df["low"].iloc[as_of - 1])
        slope = float(slope)
        upper = float(upper)
        lower = float(lower)
        atr = float(atr)

        market = self.entry_offset_pct == 0
        if slope < 0 and high > upper and prev_high <= float(prev_upper):
            side = "short"
            channel_level = upper
            entry = close if market else upper * (1.0 + self.entry_offset_pct)
            # 현재가 아래에 sell limit을 놓으면 즉시 시장가가 된다.
            if not market and entry <= close:
                return []
            stop_hint = entry + atr * REGRESSION_CHANNEL.stop_hint_atr_mult
        elif slope > 0 and low < lower and prev_low >= float(prev_lower):
            side = "long"
            channel_level = lower
            entry = close if market else lower * (1.0 - self.entry_offset_pct)
            # 현재가 위에 buy limit을 놓으면 즉시 시장가가 된다.
            if not market and entry >= close:
                return []
            stop_hint = entry - atr * REGRESSION_CHANNEL.stop_hint_atr_mult
        else:
            return []

        if self.direction != "both" and side != self.direction:
            return []

        # 국면 시리즈가 없으면(4시간봉 미제공 등) 필터를 통과시킨다.
        # 상위봉 데이터가 없다고 매매를 멈출 이유는 없다.
        try:
            regime = prepared.at(ENTRY_TF, "major_regime", as_of)
        except KeyError:
            regime = "none"
        aligned = (regime == "bull" and side == "long") or (
            regime == "bear" and side == "short"
        )
        if self.regime_filter:
            counter = (regime == "bull" and side == "short") or (
                regime == "bear" and side == "long"
            )
            # 대세 역행은 물론, 횡보장 숏도 뺀다 (38건 -89.16, 앞뒤 모두 마이너스).
            if counter or (regime == "none" and side == "short"):
                return []

        if channel_level <= 0 or stop_hint <= 0:
            return []

        return [
            Signal(
                strategy=NAME,
                symbol=self.symbol,
                side=side,
                entry=entry,
                stop_hint=stop_hint,
                target_hint=channel_level,
                score=2.0 + float(r2),
                reasons=[
                    "regression_channel_down" if side == "short" else "regression_channel_up",
                    "channel_wick_break",
                    *(["trend_aligned"] if aligned else []),
                    f"limit_offset_{self.entry_offset_pct:.1%}",
                ],
                context={
                    "atr": atr,
                    "reg_slope": slope,
                    "reg_r2": float(r2),
                    "reg_upper": upper,
                    "reg_lower": lower,
                    "channel_target": channel_level,
                    "entry_offset_pct": self.entry_offset_pct,
                    "major_regime": regime,
                    "trend_aligned": aligned,
                },
                risk_multiplier=self.aligned_risk_mult if aligned else 1.0,
                tranche=1,
                max_tranches=1,
                order_type="market" if market else "limit",
                ttl_bars=0 if market else REGRESSION_CHANNEL.limit_ttl_bars,
            )
        ]
