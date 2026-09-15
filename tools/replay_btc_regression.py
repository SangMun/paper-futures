"""BTC 회귀채널 진입 이격 1%·2% 비교.

    python -m tools.replay_btc_regression
    python -m tools.replay_btc_regression --offset 0.01
"""

from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
from pathlib import Path

from config.symbols import BTC
from engine.replay import replay
from strategies.btc_regression_channel import BtcRegressionChannel
from tools.replay_btc_candidates import completed_positions, max_drawdown_pct


def run_offset(
    offset: float, direction: str, bars: int | None, balance: float, regime: bool = False
) -> None:
    strategy = BtcRegressionChannel(
        entry_offset_pct=offset, direction=direction,
        regime_filter=regime, aligned_risk_mult=2.0 if regime else 1.0,
    )
    with tempfile.TemporaryDirectory(prefix="btc-regression-") as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            result = replay(
                {BTC.contract: [strategy]},
                symbols=(BTC,),
                balance=balance,
                warmup=500,
                max_bars=bars,
                log_dir=Path(tmp),
            )

    positions = completed_positions(result.trades)
    wins = [p.pnl for p in positions if p.pnl > 0]
    losses = [p.pnl for p in positions if p.pnl <= 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    payoff = avg_win / avg_loss if avg_loss else 0.0
    win_rate = 100 * len(wins) / len(positions) if positions else 0.0
    drawdown = max_drawdown_pct(balance, result.trades)
    long_pnl = sum(p.pnl for p in positions if p.side == "long")
    short_pnl = sum(p.pnl for p in positions if p.side == "short")

    placed = sum(s.get("skip_reason") == "limit_pending" for s in result.signals)
    filled = sum(
        s.get("action") == "entry" and s.get("fill_type") == "maker"
        for s in result.signals
    )
    expired = sum(s.get("skip_reason") == "limit_expired" for s in result.signals)
    fill_rate = 100 * filled / placed if placed else 0.0

    print(
        f"{offset:>4.1%} {direction:<5} {len(positions):>4}건  승률 {win_rate:>5.1f}%  "
        f"수익률 {result.return_pct:>+7.2f}%  DD {drawdown:>7.2f}%  "
        f"손익비 {payoff:>4.2f}  체결 {filled}/{placed}({fill_rate:>4.1f}%, "
        f"만료 {expired})  롱 {long_pnl:>+7.2f} / 숏 {short_pnl:>+7.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=float, action="append", help="0.01 또는 0.02")
    parser.add_argument(
        "--direction", choices=("both", "long", "short"), action="append",
        help="기본 both. 여러 번 지정하면 방향별로 함께 비교",
    )
    parser.add_argument("--bars", type=int)
    parser.add_argument("--balance", type=float, default=1_000.0)
    parser.add_argument(
        "--regime", action="store_true",
        help="실시간 설정과 같게 국면 필터 + 추세 일치 리스크 2배",
    )
    args = parser.parse_args()

    offsets = args.offset or [0.01, 0.02]
    directions = args.direction or ["both"]
    print("이격 방향   거래     승률       수익률         최대DD   손익비       지정가               방향손익")
    print("-" * 125)
    for offset in offsets:
        for direction in directions:
            run_offset(offset, direction, args.bars, args.balance, args.regime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
