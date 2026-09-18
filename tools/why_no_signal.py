"""지금 이 순간 각 전략이 왜 신호를 안 내는지 보여준다.

    python -m tools.why_no_signal
    PAPER_STRATEGIES=btc_regime_25x python -m tools.why_no_signal

실시간 엔진과 같은 경로(fetch -> prepare -> generate)를 마지막 확정봉에 대해
한 번 돌리고, 25x는 관문 값까지 찍는다. 상태 파일은 건드리지 않는다.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd

from config.settings import BTC_REGIME_25X
from engine.loop import Engine, _entry_tf_of, build_strategies, select_strategies
from engine import state as st
from reporting.notify import Notifier


def main() -> int:
    strategies = select_strategies(build_strategies(), os.environ.get("PAPER_STRATEGIES", ""))
    engine = Engine(
        strategies=strategies,
        notifier=Notifier(url=""),
        bot_state=st.BotState(balance=1_000.0, day_start_balance=1_000.0),
    )
    cycle: dict[str, dict[str, pd.DataFrame]] = {}
    for spec in engine.symbols:
        results = engine.fetch(spec)
        if results is None:
            print(f"[{spec.contract}] 수집 실패")
            continue
        problems = engine.data_problems(results)
        frames = {tf: r.df for tf, r in results.items()}
        cycle[spec.contract] = frames
        if problems:
            print(f"[{spec.contract}] 데이터 문제 -> 진입 차단: {problems}")
        frames = engine._with_context(spec, frames, cycle)

        for strategy in strategies.get(spec.contract, []):
            tf = _entry_tf_of(spec, strategy)
            df = frames.get(tf)
            if df is None or df.empty:
                print(f"  {strategy.name}: {tf} 프레임 없음")
                continue
            at = len(df) - 1
            bar = datetime.fromtimestamp(int(df["ts"].iloc[at]), tz=timezone.utc)
            prepared = strategy.prepare(frames)
            signals = strategy.generate(prepared, at)
            print(f"  {strategy.name:<34} {tf:>3} 봉 {bar:%m-%d %H:%M}  신호 {len(signals)}")
            if strategy.name == "btc_regime_25x":
                cfg = BTC_REGIME_25X
                v = {n: prepared.at(tf, n, at) for n in (
                    "channel_r2", "channel_move_pct", "channel_slope",
                    "channel_lower", "channel_upper", "major_regime")}
                close = float(df["close"].iloc[at])
                print(f"      R² {v['channel_r2']:.3f} (>= {cfg.channel_r2_min})  "
                      f"이동 {v['channel_move_pct']:.3f}% (>= {cfg.channel_min_move_pct})  "
                      f"기울기 {v['channel_slope']:+.2f}  국면 {v['major_regime']}")
                print(f"      종가 {close:,.1f}  채널 {v['channel_lower']:,.1f} ~ {v['channel_upper']:,.1f}")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
