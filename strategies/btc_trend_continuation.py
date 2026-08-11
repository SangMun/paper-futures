"""BTC 추세 지속 후보.

일봉 MA50/200 국면 방향으로만, 1시간봉 종가가 직전 N봉 고점/저점을
돌파하면 그 채널선에 지정가를 둔다. 다음 봉부터 제한 시간 안에 되돌아와야
체결된다. RSI 과매수·과매도는 돌파에서 정상적으로 나타날 수 있어 필터로
쓰지 않는다.

청산은 우선 기존 BTC 규칙(VPVR 절반 익절 → 나머지 트레일)을 그대로 써서
진입 방식만 비교한다. 후보가 살아남은 뒤에만 전용 청산을 탐색한다.
"""

from __future__ import annotations

import pandas as pd

from config.settings import BTC_CANDIDATES
from strategies.base import Prepared, Signal
from strategies.btc_trend import BtcTrend, ENTRY_TF


NAME = "btc_trend_continuation"


class BtcTrendContinuation:
    """일봉 방향 + 1시간 Donchian 종가 돌파."""

    name = NAME

    def __init__(self, symbol: str = "BTC_USDT"):
        self.symbol = symbol
        self._base = BtcTrend(symbol)

    def prepare(self, frames: dict[str, pd.DataFrame]) -> Prepared:
        return self._base.prepare(frames)

    def generate(self, prepared: Prepared, as_of: int) -> list[Signal]:
        lookback = BTC_CANDIDATES.continuation_lookback_bars
        if as_of < lookback:
            return []

        df = prepared.frame(ENTRY_TF)
        regime = prepared.at(ENTRY_TF, "ctx_regime", as_of)
        if regime not in ("bull", "bear"):
            return []
        side = "long" if regime == "bull" else "short"

        # 현재 봉을 채널 계산에 넣으면 자기 고가를 자기가 돌파해야 하는 모순이
        # 생긴다. 직전 lookback개 확정 봉만 사용한다.
        previous = df.iloc[as_of - lookback : as_of]
        close = float(df["close"].iloc[as_of])
        channel = (
            float(previous["high"].max())
            if side == "long"
            else float(previous["low"].min())
        )
        broken = close > channel if side == "long" else close < channel
        if not broken:
            return []

        atr = prepared.at(ENTRY_TF, "atr", as_of)
        if pd.isna(atr) or float(atr) <= 0:
            return []
        stop_hint = self._base._stop_hint(df, as_of, side)
        if stop_hint is None:
            return []

        return [
            Signal(
                strategy=NAME,
                symbol=self.symbol,
                side=side,
                entry=channel,
                stop_hint=stop_hint,
                target_hint=self._base._target_hint(df, as_of, channel, side),
                score=3.0,
                reasons=[f"regime_{regime}", f"donchian_{lookback}_close_break"],
                context={
                    "atr": float(atr),
                    "regime": regime,
                    "channel_level": channel,
                    "lookback_bars": lookback,
                    "break_close": close,
                },
                tranche=1,
                max_tranches=1,
                order_type="limit",
                ttl_bars=BTC_CANDIDATES.continuation_limit_ttl_bars,
            )
        ]
