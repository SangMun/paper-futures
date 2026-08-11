"""BTC 엄격 눌림 반전 후보.

기존 btc_trend/btc_fib은 레벨에 닿은 확인 캔들 종가에서 곧바로 진입한다.
이 후보는 신호를 세 단계의 *완료된* 1시간봉으로 나눈다.

  1. MA 또는 피보나치 합류 레벨을 실제로 거부한 setup 봉
  2. setup 봉의 고점(롱)/저점(숏)을 종가로 돌파한 break 봉
  3. 돌파 레벨에 지정가를 두고 다음 봉부터 첫 리테스트를 기다림

break 봉이 마감된 뒤에만 주문하므로 이미 지나간 가격으로 체결하지 않는다.
리테스트가 정해진 시간 안에 오지 않으면 기회를 버린다.

아직 실시간 build_strategies에는 등록하지 않는다. 동일 표본에서 성과를
확인하기 위한 후보이며, tools.replay_btc_candidates로 따로 측정한다.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from config.settings import BTC_CANDIDATES
from indicators import patterns
from strategies.base import Prepared, Signal
from strategies.btc_trend import BtcTrend, ENTRY_TF
from strategies.fib_retracement import FibRetracement


NAME = "btc_pullback_reversal"


class BtcPullbackReversal:
    """레벨 거부 → 구조 돌파 → 첫 리테스트."""

    name = NAME

    def __init__(self, symbol: str = "BTC_USDT"):
        self.symbol = symbol
        self._trend = BtcTrend(symbol)
        self._fib = FibRetracement(symbol)
        self._setup_cache: dict[int, list[tuple[Signal, str]]] = {}

    def prepare(self, frames: dict[str, pd.DataFrame]) -> Prepared:
        # Fib prepare가 표준 지표, 일봉 정렬, 확정 스윙을 모두 준비한다.
        # BtcTrend.generate도 같은 Prepared 형식을 사용하므로 중복 계산하지 않는다.
        self._setup_cache.clear()
        return self._fib.prepare(frames)

    def generate(self, prepared: Prepared, as_of: int) -> list[Signal]:
        cfg = BTC_CANDIDATES
        earliest = as_of - cfg.pullback_breakout_ttl_bars
        if earliest < 1:
            return []

        df = prepared.frame(ENTRY_TF)
        candidates: list[tuple[float, int, Signal, str]] = []

        # 이번 확정 봉이 break일 때만 주문을 낸다. 과거 break를 다시 발견해
        # 같은 레벨에 주문을 반복 제출하면 안 된다.
        for setup_at in range(earliest, as_of):
            for setup, source in self._strict_setups(prepared, setup_at):
                break_at = self._break_at(df, setup_at, as_of, setup.side)
                if break_at != as_of:
                    continue

                expected_regime = "bull" if setup.side == "long" else "bear"
                if prepared.at(ENTRY_TF, "ctx_regime", as_of) != expected_regime:
                    continue
                candidates.append((float(setup.score), setup_at, setup, source))

        if not candidates:
            return []

        # 같은 리테스트에 MA와 fib가 함께 걸리면 높은 점수, 그다음 최근 setup을 쓴다.
        _, setup_at, setup, source = max(candidates, key=lambda x: (x[0], x[1]))
        entry = self._break_level(df, setup_at, setup.side)
        atr = prepared.at(ENTRY_TF, "atr", as_of)
        if pd.isna(atr) or float(atr) <= 0:
            return []

        stop_hint = self._trend._stop_hint(df, as_of, setup.side)
        if stop_hint is None:
            return []

        regime = "bull" if setup.side == "long" else "bear"
        context = dict(setup.context)
        context.update(
            atr=float(atr),
            regime=regime,
            source_strategy=source,
            setup_ts=int(df["ts"].iloc[setup_at]),
            break_ts=int(df["ts"].iloc[as_of]),
            retest_level=entry,
        )

        return [
            replace(
                setup,
                strategy=NAME,
                symbol=self.symbol,
                entry=entry,
                stop_hint=stop_hint,
                target_hint=self._trend._target_hint(df, as_of, entry, setup.side),
                score=float(setup.score) + 3.0,
                reasons=[
                    f"source_{source}",
                    *setup.reasons,
                    "strict_level_rejection",
                    "structure_close_break",
                    "limit_at_break_level",
                ],
                context=context,
                tranche=1,
                max_tranches=1,
                order_type="limit",
                ttl_bars=cfg.pullback_retest_ttl_bars,
            )
        ]

    def _strict_setups(self, prepared: Prepared, as_of: int) -> list[tuple[Signal, str]]:
        """근거 레벨을 실제로 거부한 기존 신호만 setup으로 인정한다."""

        cached = self._setup_cache.get(as_of)
        if cached is not None:
            return cached

        df = prepared.frame(ENTRY_TF)
        out: list[tuple[Signal, str]] = []

        for signal in self._trend.generate(prepared, as_of):
            touched = [
                reason for reason in ("ma50_touch", "ma200_touch")
                if reason in signal.reasons
            ]
            level = self._trend._confirm_level(prepared, df, as_of, signal.side, touched)
            window = df.iloc[max(0, as_of - 1) : as_of + 1]
            if bool(patterns.rejection(window, level, signal.side).iloc[-1]):
                out.append((signal, "btc_trend"))

        for signal in self._fib.generate(prepared, as_of):
            level = float(signal.context["zone_price"])
            window = df.iloc[max(0, as_of - 1) : as_of + 1]
            if bool(patterns.rejection(window, level, signal.side).iloc[-1]):
                out.append((signal, "btc_fib"))

        self._setup_cache[as_of] = out
        return out

    @staticmethod
    def _break_level(df: pd.DataFrame, setup_at: int, side: str) -> float:
        return float(df["high"].iloc[setup_at] if side == "long" else df["low"].iloc[setup_at])

    def _break_at(
        self,
        df: pd.DataFrame,
        setup_at: int,
        as_of: int,
        side: str,
    ) -> int | None:
        """setup 뒤 TTL 안의 첫 종가 돌파."""

        level = self._break_level(df, setup_at, side)
        end = min(as_of, setup_at + BTC_CANDIDATES.pullback_breakout_ttl_bars)
        for i in range(setup_at + 1, end + 1):
            close = float(df["close"].iloc[i])
            if (side == "long" and close > level) or (side == "short" and close < level):
                return i
        return None
