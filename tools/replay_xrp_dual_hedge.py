"""EXP-015: 진입 필터 없는 XRP 양방향 격리 리플레이.

사용 예:
    ./.venv/bin/python -m tools.replay_xrp_dual_hedge --leverage 50
    ./.venv/bin/python -m tools.replay_xrp_dual_hedge --all-leverages
"""

from __future__ import annotations

import argparse

from config.settings import DUAL_HEDGE
from core import cache
from engine.dual_hedge import DualHedgeReplay, config_with


def run_once(
    leverage: int, balance: float, margin_fraction: float, timeframe: str,
    max_survivor_hours: float, liq_stop_buffer: float | None = None,
) -> None:
    df = cache.load(DUAL_HEDGE.symbol, timeframe)
    if df.empty:
        raise SystemExit(f"XRP_USDT {timeframe} 캐시가 없다. data/cache를 확인할 것.")
    config = config_with(
        leverage=leverage,
        pair_margin_fraction=margin_fraction,
        max_survivor_hold_seconds=int(max_survivor_hours * 3600),
        liquidation_stop_buffer=liq_stop_buffer,
    )
    result = DualHedgeReplay(balance=balance, config=config).run(df)
    print(
        f"{leverage:>3}x {timeframe:>2}  {result.start:%Y-%m-%d %H:%M} ~ {result.end:%Y-%m-%d %H:%M}  "
        f"{result.bars:,}봉  완료 {result.completed_pairs}  승률 {result.win_rate:.1f}%  "
        f"완료잔고 {result.initial_balance:,.2f} -> {result.final_balance:,.2f}  "
        f"({result.return_pct:+.2f}%)  MDD {result.realized_max_drawdown_pct:.2f}%  "
        f"미청산 {'있음' if result.open_pair else '없음'}"
    )
    if result.open_pair:
        print(f"      미청산을 제외한 완료잔고다. 현재 현금 잔고 {result.unsettled_cash_balance:,.2f}")
    print(f"      청산: {dict(result.by_liquidation())} / 종료: {dict(result.by_reason())}")
    direction = result.by_liquidation_stats()
    print("      방향별: " + ", ".join(
        f"{side} {row['pairs']}건 {row['pnl']:+.2f} ({row['win_rate']:.1f}%)"
        for side, row in sorted(direction.items())
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leverage", type=int, default=DUAL_HEDGE.leverage)
    parser.add_argument("--all-leverages", action="store_true", help="20, 30, 40, 50, 70, 100배 비교")
    parser.add_argument("--balance", type=float, default=1_000.0)
    parser.add_argument("--margin-fraction", type=float, default=DUAL_HEDGE.pair_margin_fraction)
    parser.add_argument(
        "--max-survivor-hours", type=float,
        default=DUAL_HEDGE.max_survivor_hold_seconds / 3600,
        help="한쪽 청산 뒤 생존 다리의 최대 보유 시간 (기본: 24)",
    )
    parser.add_argument(
        "--liq-stop-buffer", type=float, default=DUAL_HEDGE.liquidation_stop_buffer,
        help="청산가 직전 보호 스탑 비율 (예: 0.001 = 청산가의 0.1%%). 생략하면 청산까지 보유",
    )
    parser.add_argument(
        "--timeframe", choices=("1m", "5m"), default="1m",
        help="1m은 최근 약 6일의 실행 지연 확인용, 5m은 장기 표본용 (기본: 1m)",
    )
    args = parser.parse_args()
    leverages = (20, 30, 40, 50, 70, 100) if args.all_leverages else (args.leverage,)
    for leverage in leverages:
        run_once(
            leverage, args.balance, args.margin_fraction, args.timeframe,
            args.max_survivor_hours, args.liq_stop_buffer,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
