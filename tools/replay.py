"""리플레이 실행과 성과 집계.

    python -m tools.replay                      # 전 전략, 전 구간
    python -m tools.replay --symbol XRP_USDT
    python -m tools.replay --bars 5000          # 짧게 확인
    python -m tools.replay --strategy xrp_patterns.triangle

성과 표는 전략별·방향별로 나눈다(README 원칙 5). 크립토는 상승 편향이
있어 같은 규칙이라도 방향별 통계가 다르고, 합쳐 보면 한쪽이 다른 쪽을 가린다.
"""

from __future__ import annotations

import argparse

from config.settings import RISK
from config.symbols import ACTIVE
from engine.loop import build_strategies
from engine.replay import replay


def summarize(result) -> None:
    print()
    print(f"구간   {result.start:%Y-%m-%d} ~ {result.end:%Y-%m-%d}  ({result.bars:,}봉)")
    print(f"소요   {result.seconds:.1f}초")
    print(f"잔고   {result.initial_balance:,.2f} -> {result.final_balance:,.2f} "
          f"({result.return_pct:+.1f}%)")
    if result.open_positions:
        print(f"미청산 {result.open_positions}건")

    rows = result.positions_by_strategy()
    if not rows:
        print("\n체결 없음.")
    else:
        print()
        print(f"{'전략':<28} {'거래':>5} {'승률':>6} {'손익':>9} {'평균':>7}  {'롱/숏':>9}")
        print("-" * 74)
        for name, r in sorted(rows.items(), key=lambda x: -x[1]["pnl"]):
            print(
                f"{name:<28} {r['positions']:>5} {r['win_rate']:>5.1f}% "
                f"{r['pnl']:>+9.2f} {r['avg']:>+7.2f}  "
                f"{r['long']:>4}/{r['short']:<4}"
            )

        print(f"\n실현 잔고 최대낙폭  {result.realized_max_drawdown_pct:.1f}% "
              "(미실현 손익 제외)")

        print()
        print("청산 사유:")
        for name, r in sorted(rows.items()):
            top = ", ".join(f"{k} {v}" for k, v in r["reasons"].most_common(5))
            print(f"  {name:<28} {top}")

        print()
        print("방향별 손익:")
        for name, r in sorted(rows.items()):
            print(f"  {name:<28} 롱 {r['long_pnl']:>+8.2f} / 숏 {r['short_pnl']:>+8.2f}")

    skips = result.skip_reasons()
    if skips:
        print()
        print("진입 거절 사유:")
        for reason, count in skips.most_common(10):
            print(f"  {reason:<26} {count:,}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", help="특정 종목만")
    ap.add_argument("--strategy", help="특정 전략만 (로그 태그)")
    ap.add_argument("--bars", type=int, help="재생할 봉 수 (기본: 전 구간)")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--balance", type=float, default=1_000.0)
    ap.add_argument(
        "--with-brakes",
        action="store_true",
        help="계좌 단위 브레이크(일일 손실·연속 손절 정지)를 켠다. "
             "기본은 꺼짐 — 켜면 전략들이 서로의 성과를 오염시킨다.",
    )
    ap.add_argument(
        "--ticks",
        action="store_true",
        help="실행 시계(틱)를 재생한다. 1분봉을 틱 대용으로 써서 "
             "봉 마감 전 손절·익절·트레일을 실시간과 같은 경로로 판정한다.",
    )
    ap.add_argument(
        "--tick-tf",
        default="1m",
        help="틱 대용 타임프레임 (기본 1m)",
    )
    args = ap.parse_args()

    symbols = tuple(s for s in ACTIVE if not args.symbol or s.contract == args.symbol)
    if not symbols:
        raise SystemExit(f"알 수 없는 종목: {args.symbol}")

    strategies = build_strategies()
    selected_contracts = {s.contract for s in symbols}
    strategies = {
        contract: group
        for contract, group in strategies.items()
        if contract in selected_contracts
    }
    if args.strategy:
        strategies = {
            k: [s for s in v if s.name == args.strategy]
            for k, v in strategies.items()
        }
        strategies = {k: v for k, v in strategies.items() if v}
        if not strategies:
            raise SystemExit(f"알 수 없는 전략: {args.strategy}")

    tags = [s.name for v in strategies.values() for s in v]
    print(f"전략 {tags}")
    print(f"종목 {[s.contract for s in symbols]}")

    result = replay(
        strategies,
        symbols=symbols,
        balance=args.balance,
        warmup=args.warmup,
        max_bars=args.bars,
        progress_every=5_000,
        risk=RISK if args.with_brakes else None,
        ticks=args.ticks,
        tick_tf=args.tick_tf,
    )
    summarize(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
