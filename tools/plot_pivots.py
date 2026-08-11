"""피벗·스윙·합류 존을 차트로 출력한다.

명세 §10-3. **이 단계를 건너뛰지 않는다.**
자동 탐지된 스윙이 사람이 봤을 때 납득 가능한 위치인지 먼저 확인해야
이후 단계(합류, 진입 규칙)가 의미를 갖는다. 단위 테스트는 "코드가 규칙대로
도는가"만 검증하지, "그 규칙이 시장에서 말이 되는가"는 답하지 못한다.

    python -m tools.plot_pivots
    python -m tools.plot_pivots --contract XRP_USDT --interval 5m --bars 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # 화면 없는 환경에서도 파일로 저장
import matplotlib.pyplot as plt

from config.settings import FIB
from core import cache
from indicators import confluence, fib as fib_mod
from indicators import atr as atr_mod
from indicators import ma as ma_mod
from indicators import pivots as pv


OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "charts"


def draw(
    contract: str,
    interval: str,
    bars: int,
    pivot_n: int,
    min_swing_pct: float,
) -> Path:
    df = cache.load(contract, interval)
    if df.empty:
        raise SystemExit(f"캐시가 비어 있다: {contract} {interval}. fetch_history 먼저 실행할 것.")

    df = df.tail(bars).reset_index(drop=True)
    as_of = len(df) - 1

    pivots = pv.find_pivots(df, pivot_n, as_of)
    swings = pv.recent_swings(df, as_of, n=pivot_n, min_swing_pct=min_swing_pct)

    atr = float(atr_mod.atr(df).iloc[-1])
    ma_fast = ma_mod.sma(df, 50)
    ma_slow = ma_mod.sma(df, 200)
    zones = confluence.find_zones(
        df, as_of, atr,
        ma_fast=float(ma_fast.iloc[-1]) if ma_fast.notna().iloc[-1] else None,
        ma_slow=float(ma_slow.iloc[-1]) if ma_slow.notna().iloc[-1] else None,
        pivot_n=pivot_n, min_swing_pct=min_swing_pct,
    )

    fig, ax = plt.subplots(figsize=(16, 9))
    x = range(len(df))

    # 캔들 대신 고저 막대 + 종가선. 수백 봉에서는 이쪽이 읽기 쉽다.
    ax.vlines(x, df["low"], df["high"], color="#b0b0b0", linewidth=0.6)
    ax.plot(x, df["close"], color="#333333", linewidth=0.9, label="Close")

    if ma_fast.notna().any():
        ax.plot(x, ma_fast, color="#1f77b4", linewidth=1.0, label="MA50")
    if ma_slow.notna().any():
        ax.plot(x, ma_slow, color="#ff7f0e", linewidth=1.0, label="MA200")

    # 피벗
    highs = [p for p in pivots if p.kind == "high"]
    lows = [p for p in pivots if p.kind == "low"]
    ax.scatter([p.index for p in highs], [p.price for p in highs],
               marker="v", color="#d62728", s=28, zorder=5, label=f"Pivot High ({len(highs)})")
    ax.scatter([p.index for p in lows], [p.price for p in lows],
               marker="^", color="#2ca02c", s=28, zorder=5, label=f"Pivot Low ({len(lows)})")

    # 채택된 스윙
    for i, s in enumerate(swings):
        ax.plot(
            [s.start.index, s.end.index], [s.start.price, s.end.price],
            color="#9467bd", linewidth=2.0, alpha=0.8,
            label="Selected swing" if i == 0 else None,
        )
        mid_x = (s.start.index + s.end.index) / 2
        mid_y = (s.start.price + s.end.price) / 2
        ax.annotate(f"{s.size_pct:.1%}", (mid_x, mid_y), fontsize=8,
                    color="#9467bd", ha="center")

        # 이 스윙의 핵심 피보나치 레벨
        for level in fib_mod.core_levels(s):
            ax.hlines(level.price, s.start.index, len(df) - 1,
                      color="#9467bd", linewidth=0.5, linestyle=":", alpha=0.5)

    # 합류 존 (점수 상위만)
    for z in [z for z in zones if z.score >= FIB.min_zone_score][:6]:
        color = "#2ca02c" if z.side == "long" else "#d62728"
        ax.axhspan(z.low, z.high, color=color, alpha=0.13)
        ax.text(len(df) * 0.995, z.price, f" {z.side} {z.score:.0f}점",
                fontsize=8, color=color, va="center", ha="right")

    ax.set_title(
        f"{contract} {interval} - last {len(df)} bars "
        f"(pivot_n={pivot_n}, min_swing={min_swing_pct:.0%}, ATR={atr:.4g})"
    )
    ax.set_xlabel(f"{df['dt'].iloc[0]:%Y-%m-%d} ~ {df['dt'].iloc[-1]:%Y-%m-%d} UTC")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{contract}_{interval}_pivots.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

    print(f"피벗 {len(pivots)}개 (고점 {len(highs)} / 저점 {len(lows)})")
    print(f"채택 스윙 {len(swings)}개:")
    for s in swings:
        print(f"  {s.direction:<4} {s.start.price:.6g} -> {s.end.price:.6g} "
              f"({s.size_pct:.2%}, {s.bars}봉)")
    print(f"합류 존 {len(zones)}개 (점수 {FIB.min_zone_score} 이상: "
          f"{sum(1 for z in zones if z.score >= FIB.min_zone_score)}개)")
    print(f"-> {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="BTC_USDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--bars", type=int, default=500)
    ap.add_argument("--pivot-n", type=int, default=FIB.pivot_n)
    ap.add_argument("--min-swing", type=float, default=FIB.min_swing_pct)
    args = ap.parse_args()

    draw(args.contract, args.interval, args.bars, args.pivot_n, args.min_swing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
