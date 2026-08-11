"""수집 레이어 실통신 점검.

단위 테스트는 가짜 응답으로 돌기 때문에, 거래소가 실제로 무엇을 주는지는
따로 확인해야 한다. 계약 코드가 살아 있는지, 타임프레임별로 봉이 제대로
오는지, 결측이 얼마나 있는지를 눈으로 본다.

    python -m tools.check_candles
    python -m tools.check_candles --cache --days 30
"""

from __future__ import annotations

import argparse
import time

from config.symbols import ACTIVE
from core import cache
from core import candles as cd
from core.exchange import GateClient


def summarize(result) -> str:
    df = result.df
    if df.empty:
        return f"  {result.interval:<4} 캔들 없음"

    span = f"{df['dt'].iloc[0]:%Y-%m-%d %H:%M} ~ {df['dt'].iloc[-1]:%Y-%m-%d %H:%M} UTC"
    line = (
        f"  {result.interval:<4} {len(df):>5}봉  {span}  "
        f"종가 {df['close'].iloc[-1]:g}"
    )
    for p in result.problems:
        line += f"\n         ! {p}"
    return line


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="받은 캔들을 로컬에 저장")
    ap.add_argument("--days", type=int, default=0, help="과거 N일 구간을 받아온다")
    args = ap.parse_args()

    now = int(time.time())
    failed = False

    with GateClient() as client:
        for spec in ACTIVE:
            print(f"\n[{spec.contract}]  전략 {[s.name for s in spec.strategies]}")

            for tf in sorted(spec.required_timeframes, key=cd.interval_seconds):
                if args.days:
                    start = now - args.days * 86400
                    result = client.fetch_range(spec.contract, tf, start, now)
                else:
                    result = client.fetch(spec.contract, tf)

                print(summarize(result))
                if result.df.empty:
                    failed = True

                if args.cache and not result.df.empty:
                    p = cache.save(spec.contract, tf, result.df)
                    print(f"         -> {p.relative_to(p.parent.parent.parent)}")

    print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
