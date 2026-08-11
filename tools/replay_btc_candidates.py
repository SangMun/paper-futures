"""BTC 후보 전략을 현재 실시간 전략과 분리해 리플레이한다.

    python -m tools.replay_btc_candidates
    python -m tools.replay_btc_candidates --bars 10000

각 후보를 단독으로 먼저 측정하고 마지막에 함께 돌린다. 후보는 포지션을 한
개씩만 허용하므로 TP1과 최종 청산을 하나의 포지션으로 다시 묶어 승률을 센다.
기존 ReplayResult.by_strategy는 청산 이벤트를 세므로 부분 익절이 승리 한 건으로
중복 집계될 수 있어 이 도구에서는 사용하지 않는다.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

from config.settings import FEES
from config.symbols import BTC
from engine.loop import build_btc_candidates
from engine.replay import replay


@dataclass(frozen=True)
class PositionResult:
    strategy: str
    side: str
    entry_ts: int
    pnl: float


def completed_positions(trades: list[dict]) -> list[PositionResult]:
    """TP1과 최종 청산을 포지션 한 건으로 합친다.

    후보 전략은 max_tranches=1이라 전략별 열린 포지션이 하나뿐이다. 진입
    수수료는 CSV의 pnl에 포함되지 않으므로 명목가에서 직접 더한다.
    """

    opened: dict[str, dict] = {}
    out: list[PositionResult] = []

    for trade in trades:
        strategy = trade.get("strategy", "")
        action = trade.get("action", "")
        if action == "entry":
            if strategy in opened:
                raise ValueError(f"동일 전략 포지션이 중복 진입됨: {strategy}")
            notional = float(trade.get("notional") or 0.0)
            fee_rate = FEES.maker_rate if trade.get("note") == "maker" else FEES.taker_rate
            opened[strategy] = {
                "side": trade.get("side", ""),
                "entry_ts": int(trade["ts"]),
                "pnl": -notional * fee_rate,
            }
            continue

        if action != "exit" or strategy not in opened:
            continue
        opened[strategy]["pnl"] += float(trade.get("pnl") or 0.0)
        if trade.get("reason") == "tp1":
            continue

        row = opened.pop(strategy)
        out.append(
            PositionResult(
                strategy=strategy,
                side=row["side"],
                entry_ts=row["entry_ts"],
                pnl=row["pnl"],
            )
        )

    return out


def max_drawdown_pct(initial: float, trades: list[dict]) -> float:
    balances = [initial]
    balances.extend(float(t["balance"]) for t in trades if t.get("balance"))
    peak = initial
    drawdown = 0.0
    for balance in balances:
        peak = max(peak, balance)
        drawdown = min(drawdown, (balance / peak - 1.0) * 100)
    return drawdown


def run_case(label: str, strategies: list, bars: int | None, balance: float) -> None:
    with tempfile.TemporaryDirectory(prefix="btc-candidate-") as tmp:
        # 엔진의 봉별 진입/청산 print는 장기 리플레이에서 수백 줄이다. 이 도구의
        # 목적은 후보 비교표이므로 숨기고, 체결 원문은 replay 로그로만 남긴다.
        with contextlib.redirect_stdout(io.StringIO()):
            result = replay(
                {BTC.contract: strategies},
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
        placed = sum(
            s.get("skip_reason") == "limit_pending" for s in result.signals
        )
        filled = sum(
            s.get("action") == "entry" and s.get("fill_type") == "maker"
            for s in result.signals
        )
        expired = sum(
            s.get("skip_reason") == "limit_expired" for s in result.signals
        )
        fill_rate = 100 * filled / placed if placed else 0.0

        print(
            f"{label:<18} {len(positions):>4}건  승률 {win_rate:>5.1f}%  "
            f"수익률 {result.return_pct:>+7.2f}%  DD {drawdown:>7.2f}%  "
            f"손익비 {payoff:>4.2f}  지정가 {filled}/{placed}({fill_rate:>4.1f}%, "
            f"만료 {expired})"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, help="워밍업 뒤 재생할 최대 1시간봉 수")
    parser.add_argument("--balance", type=float, default=1_000.0)
    args = parser.parse_args()

    candidates = build_btc_candidates()[BTC.contract]
    print("전략                 거래     승률       수익률         최대DD   손익비       지정가")
    print("-" * 112)
    for strategy in candidates:
        run_case(strategy.name, [strategy], args.bars, args.balance)

    # 위 단독 실행에서 전략 객체 내부 캐시가 채워졌다. 합산은 새 객체로 돌린다.
    together = build_btc_candidates()[BTC.contract]
    run_case("후보 합산", together, args.bars, args.balance)
    print("\n후보는 아직 실시간 build_strategies에 등록되지 않았습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
