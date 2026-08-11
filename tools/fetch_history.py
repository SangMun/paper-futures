"""과거 캔들 백필.

check_candles는 실통신 점검용이라 --days가 모든 타임프레임에 똑같이 걸린다.
필요한 깊이는 타임프레임마다 다르므로(일봉 800일 vs 5분봉 180일) 여기서는
config.settings.HISTORY_DAYS를 타임프레임별로 적용한다.

이미 받아둔 구간은 다시 받지 않는다. 캐시에 없는 앞뒤 구간만 채운다.
수만 봉을 매번 다시 받으면 시간도 문제지만, 중간에 끊겼을 때
처음부터 다시 시작해야 하는 게 더 크다.

    python -m tools.fetch_history
    python -m tools.fetch_history --contract XRP_USDT --interval 5m
    python -m tools.fetch_history --force        # 캐시 무시하고 전체 재수집
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from config.settings import HISTORY_DAYS, history_days
from config.symbols import ACTIVE
from core import cache
from core import candles as cd
from core.exchange import ExchangeError, GateClient


def _fmt(n: int) -> str:
    return f"{n:,}"


def backfill(
    client: GateClient,
    contract: str,
    interval: str,
    days: int,
    force: bool = False,
    now: int | None = None,
) -> bool:
    """한 종목·타임프레임의 과거 구간을 캐시에 채운다.

    반환값은 성공 여부. 실패해도 예외를 올리지 않는 이유는
    한 타임프레임이 막혀도 나머지는 계속 받아야 하기 때문이다.
    """

    now = now or int(time.time())
    step = cd.interval_seconds(interval)
    want_start = now - days * 86400
    expected = days * 86400 // step

    print(f"  {interval:<4} 목표 {days}일 (약 {_fmt(expected)}봉)", flush=True)

    # 캐시에 이미 있는 구간은 건너뛰고 모자란 쪽만 요청한다.
    gaps: list[tuple[int, int]] = []
    have = None if force else cache.coverage(contract, interval)

    if have is None:
        gaps.append((want_start, now))
    else:
        first, last = have
        print(
            f"       캐시 보유 {pd.to_datetime(first, unit='s', utc=True):%Y-%m-%d}"
            f" ~ {pd.to_datetime(last, unit='s', utc=True):%Y-%m-%d}",
            flush=True,
        )
        if want_start < first - step:
            gaps.append((want_start, first))        # 과거쪽 확장
        if last + step < now:
            gaps.append((last, now))                # 최근분 갱신

    if not gaps:
        print("       이미 충분함, 건너뜀", flush=True)
        return True

    for start, end in gaps:
        span_days = (end - start) / 86400
        print(f"       수집 {span_days:.1f}일...", end="", flush=True)
        try:
            result = client.fetch_range(contract, interval, start, end)
        except ExchangeError as exc:
            print(f" 실패: {exc}", flush=True)
            return False

        if result.df.empty:
            print(" 응답 없음", flush=True)
            continue

        cache.save(contract, interval, result.df)
        print(f" {_fmt(len(result.df))}봉", flush=True)

    # 최종 상태를 캐시에서 다시 읽어 확인한다.
    # 저장까지 끝난 결과로 판정해야 다음 단계가 믿고 쓸 수 있다.
    df = cache.load(contract, interval)
    problems = cd.validate(df, interval)
    span = (
        f"{df['dt'].iloc[0]:%Y-%m-%d} ~ {df['dt'].iloc[-1]:%Y-%m-%d}"
        if not df.empty
        else "비어 있음"
    )
    ratio = len(df) / expected if expected else 0
    print(f"       -> {_fmt(len(df))}봉 ({ratio:.0%})  {span}", flush=True)
    for p in problems:
        print(f"       ! {p}", flush=True)

    return not df.empty


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", help="특정 종목만 (기본: ACTIVE 전체)")
    ap.add_argument("--interval", help="특정 타임프레임만")
    ap.add_argument("--force", action="store_true", help="캐시 무시하고 재수집")
    args = ap.parse_args()

    now = int(time.time())
    failed: list[str] = []

    with GateClient() as client:
        for spec in ACTIVE:
            if args.contract and spec.contract != args.contract:
                continue

            tfs = sorted(spec.required_timeframes, key=cd.interval_seconds)
            if args.interval:
                tfs = [t for t in tfs if t == args.interval]
            if not tfs:
                continue

            print(f"\n[{spec.contract}]", flush=True)
            for tf in tfs:
                if tf not in HISTORY_DAYS:
                    print(f"  {tf:<4} HISTORY_DAYS에 없음, 건너뜀", flush=True)
                    continue

                # 거래소 상한에 걸리면 잘라서 요청한다.
                # 조용히 줄이지 않고 얼마나 못 받았는지 남긴다 —
                # 표본 부족은 나중에 성과를 해석할 때 반드시 알아야 할 사실이다.
                days = history_days(tf, cd.interval_seconds(tf))
                if days < HISTORY_DAYS[tf]:
                    print(
                        f"  {tf:<4} 거래소 상한으로 {HISTORY_DAYS[tf]}일 -> {days}일 축소",
                        flush=True,
                    )

                if not backfill(client, spec.contract, tf, days, args.force, now):
                    failed.append(f"{spec.contract} {tf}")

    print()
    if failed:
        print(f"실패: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
