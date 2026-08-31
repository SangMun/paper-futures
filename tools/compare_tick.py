"""봉 경로 vs 틱 경로 비교.

같은 구간을 두 번 돌린다. 하나는 봉 마감에만 청산을 판정하는 기존 방식,
다른 하나는 실행 시계(1분봉을 틱 대용)를 재생하는 방식이다.

**차이가 어느 방향으로 나는지가 판단 기준이다.**
틱 경로에서 손절이 더 이르게, 손실이 더 얕게 잡혀야 정상이다. 봉 마감까지
기다리는 동안 더 밀리던 걸 일찍 끊는 게 이 작업의 목적이기 때문이다.
반대로 나오면 구현이 틀린 것이다.

성과가 좋아지는지를 보는 도구가 아니다. 두 경로가 **설명 가능한 방향으로**
갈리는지 보는 도구다.

    python -m tools.compare_tick --symbol XRP_USDT --bars 1500
"""

from __future__ import annotations

import argparse

from config.settings import RISK
from config.symbols import ACTIVE
from engine.loop import build_strategies
from engine.replay import replay


def exit_reasons(result) -> dict:
    out: dict[str, int] = {}
    for row in result.trades:
        if row.get("action") == "exit":
            reason = row.get("reason", "")
            out[reason] = out.get(reason, 0) + 1
    return out


def line(label: str, bar_value, tick_value, fmt="{:>10}") -> None:
    b = fmt.format(bar_value)
    t = fmt.format(tick_value)
    print(f"  {label:<24} {b} {t}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", help="특정 종목만")
    ap.add_argument("--strategy", help="특정 전략만")
    ap.add_argument("--bars", type=int, help="재생할 봉 수 (기본: last 전체)")
    ap.add_argument(
        "--last",
        type=int,
        default=1_500,
        help="가장 최근 N봉만 재생. 1분봉 캐시가 짧아 틱 검증은 "
             "최근 구간에서만 가능하다.",
    )
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--balance", type=float, default=1_000.0)
    ap.add_argument("--tick-tf", default="1m")
    ap.add_argument("--with-brakes", action="store_true")
    ap.add_argument(
        "--maker",
        action="store_true",
        help="틱 경로에서 시장가 신호를 지정가로 돌린다(ENTRY.prefer_maker_entries). "
             "체결률이 성과보다 먼저 볼 숫자다 — 낮으면 전략이 달라진 것이다.",
    )
    args = ap.parse_args()

    symbols = tuple(s for s in ACTIVE if not args.symbol or s.contract == args.symbol)
    if not symbols:
        raise SystemExit(f"알 수 없는 종목: {args.symbol}")

    selected = {s.contract for s in symbols}
    strategies = {
        c: g for c, g in build_strategies().items() if c in selected
    }
    if args.strategy:
        strategies = {
            k: [s for s in v if s.name == args.strategy] for k, v in strategies.items()
        }
        strategies = {k: v for k, v in strategies.items() if v}
        if not strategies:
            raise SystemExit(f"알 수 없는 전략: {args.strategy}")

    common = dict(
        symbols=symbols,
        balance=args.balance,
        warmup=args.warmup,
        max_bars=args.bars,
        last=args.last,
        risk=RISK if args.with_brakes else None,
        tick_tf=args.tick_tf,
    )

    # **변수를 하나만 바꾼다.**
    #
    # 틱 시계와 메이커 전환을 동시에 켜면 진입 시점이 달라져 두 실행이
    # 아예 다른 거래를 하게 된다. 그 상태로 손익을 비교하면 어느 쪽 변경의
    # 효과인지 구분할 수 없다 — 실제로 그렇게 재서 "평균 3060초 일찍"이라는
    # 무의미한 숫자가 나왔다.
    if args.maker:
        left_label, right_label = "틱", "틱+메이커"
        left_kw, right_kw = dict(ticks=True), dict(ticks=True)
    else:
        left_label, right_label = "봉", "틱"
        left_kw, right_kw = dict(ticks=False), dict(ticks=True)

    print(f"종목 {[s.contract for s in symbols]}  최근 {args.last:,}봉")
    print(f"\n[1/2] {left_label} 경로")
    _set_maker(False)
    bar = replay(strategies, log_dir="data/compare_bar", **left_kw, **common)
    print(f"      {bar.seconds:.1f}초")

    print(f"[2/2] {right_label} 경로")
    _set_maker(args.maker)
    tick = replay(strategies, log_dir="data/compare_tick", **right_kw, **common)
    print(f"      {tick.seconds:.1f}초")
    _set_maker(False)

    print()
    print(f"구간 {bar.start:%Y-%m-%d %H:%M} ~ {bar.end:%Y-%m-%d %H:%M}")
    print()
    print(f"  {'':<24} {left_label + ' 경로':>10} {right_label + ' 경로':>10}")
    print("  " + "-" * 46)

    bar_rows = bar.completed_positions()
    tick_rows = tick.completed_positions()

    line("거래 수", len(bar_rows), len(tick_rows))
    line(
        "지정가 체결/만료",
        f"{bar.limit_filled}/{bar.limit_expired}",
        f"{tick.limit_filled}/{tick.limit_expired}",
    )
    line(
        "지정가 체결률",
        f"{bar.limit_fill_rate:.0%}",
        f"{tick.limit_fill_rate:.0%}",
    )
    line("최종 잔고", f"{bar.final_balance:,.2f}", f"{tick.final_balance:,.2f}")
    line("수익률", f"{bar.return_pct:+.2f}%", f"{tick.return_pct:+.2f}%")

    bar_reasons = exit_reasons(bar)
    tick_reasons = exit_reasons(tick)
    print()
    print("  청산 사유")
    for reason in sorted(set(bar_reasons) | set(tick_reasons)):
        line(f"  {reason}", bar_reasons.get(reason, 0), tick_reasons.get(reason, 0))

    def avg_loss(rows):
        losses = [r["pnl"] for r in rows if r["pnl"] < 0]
        return sum(losses) / len(losses) if losses else 0.0

    def avg_win(rows):
        wins = [r["pnl"] for r in rows if r["pnl"] > 0]
        return sum(wins) / len(wins) if wins else 0.0

    print()
    print("  건당 손익")
    line("  평균 손실", f"{avg_loss(bar_rows):+.3f}", f"{avg_loss(tick_rows):+.3f}")
    line("  평균 이익", f"{avg_win(bar_rows):+.3f}", f"{avg_win(tick_rows):+.3f}")

    _latency(bar, tick)

    print()
    print("판정:")
    bl, tl = avg_loss(bar_rows), avg_loss(tick_rows)
    if not _same_entries(bar, tick):
        # 진입이 다르면 손익 비교도 무효다. 지연 비교만 막고 여기서
        # 평균 손실을 비교하면, 무관한 거래들의 차이를 변경의 효과로 읽는다.
        print("  두 실행의 진입이 달라 손익 비교 불가.")
        print("  체결 방식이 바뀌면 어떤 신호가 잡히는지가 달라져,")
        print("  같은 전략의 A/B가 아니라 서로 다른 전략의 비교가 된다.")
        print("  먼저 볼 숫자는 체결률이다 — 위 표 참조.")
    elif not bar_rows or not tick_rows:
        print("  체결이 없어 비교 불가. --last를 늘릴 것.")
    elif tl > bl:
        print(f"  틱 경로의 평균 손실이 얕다 ({bl:+.3f} -> {tl:+.3f}). 기대한 방향.")
    elif tl < bl:
        print(f"  ! 틱 경로의 평균 손실이 깊다 ({bl:+.3f} -> {tl:+.3f}).")
        print("    봉 마감까지 기다리던 걸 일찍 끊었는데 손실이 커졌다면")
        print("    구현을 의심할 것 — 특히 window_since와 tail_window.")
    else:
        print("  손익 동일. 손절·목표 체결가는 양쪽 다 지정된 가격이므로")
        print("  트레일이 움직이지 않은 청산은 결과가 같은 게 정상이다.")
        print("  이 변경의 효과는 위의 '감지 지연'에서 본다.")

    return 0


def _set_maker(on: bool) -> None:
    """엔진이 import한 이름까지 함께 바꾼다. 한쪽만 바꾸면 조용히 무시된다."""

    from config import settings as _st
    from engine import loop as _loop

    cfg = _st.EntryConfig(prefer_maker_entries=on)
    _st.ENTRY = cfg
    _loop.ENTRY = cfg


def _same_entries(a, b) -> bool:
    """두 실행이 같은 거래를 했는가.

    진입이 다르면 청산을 짝지어 비교할 수 없다. 인덱스로 짝지으면
    무관한 두 거래의 시각 차이를 "감지 지연"으로 읽게 된다.
    """

    def key(result):
        return [
            (t.get("strategy"), t.get("side"), int(t["ts"]))
            for t in result.trades
            if t.get("action") == "entry"
        ]

    return key(a) == key(b)


def _latency(bar, tick) -> None:
    """청산을 **언제 알았는가**의 차이.

    두 경로가 다른 시계를 기록한다는 점이 중요하다.
      봉 경로: 봉의 시작 시각을 남긴다. 실제로 안 건 그 봉이 마감된 뒤다.
      틱 경로: 관측한 순간을 남긴다.
    그래서 라벨을 그대로 빼면 틱 경로가 늦어 보인다. 봉 경로에 봉 길이를
    더해 "알 수 있었던 가장 이른 시각"으로 맞춘 뒤 비교한다.
    """

    import pandas as pd

    from core import candles as cd

    if not _same_entries(bar, tick):
        print()
        print("  감지 지연: 두 실행의 진입이 달라 비교 불가.")
        print("    체결 시점이 바뀌면 이후 거래가 통째로 갈라진다.")
        return

    bar_exits = [t for t in bar.trades if t.get("action") == "exit"]
    tick_exits = [t for t in tick.trades if t.get("action") == "exit"]
    if not bar_exits or len(bar_exits) != len(tick_exits):
        return

    print()
    print("  감지 지연 (청산을 실제로 알게 된 시각)")

    deltas = []
    for b, t in zip(bar_exits, tick_exits):
        reason = b.get("reason", "")
        if int(b["ts"]) == int(t["ts"]):
            # 두 경로가 같은 시각을 남겼다 = 틱 시계가 감지한 게 아니라
            # 양쪽 다 봉 시계가 처리했다. 시간 손절이 여기 해당한다
            # (bars_held는 봉 단위라 틱에서 판정하지 않는다).
            print(f"    {reason:<6} 봉 시계가 처리 (양쪽 동일)")
            continue

        tf = _entry_tf_of(b.get("strategy", ""))
        step = cd.interval_seconds(tf) if tf else 0
        knew_bar = int(b["ts"]) + step         # 봉 경로가 알 수 있었던 가장 이른 시각
        knew_tick = int(t["ts"])
        gap = knew_bar - knew_tick
        deltas.append(gap)
        print(
            f"    {reason:<6} "
            f"{pd.to_datetime(knew_bar, unit='s', utc=True):%m-%d %H:%M:%S} -> "
            f"{pd.to_datetime(knew_tick, unit='s', utc=True):%m-%d %H:%M:%S}  "
            f"{gap:>+5}초"
        )

    if not deltas:
        print("    틱 시계가 감지한 청산이 없다. 구간을 늘려 볼 것.")
        return

    gained = [d for d in deltas if d > 0]
    print(
        f"    틱 감지 {len(deltas)}건 중 {len(gained)}건이 더 이르다"
        + (f" (평균 {sum(gained)/len(gained):.0f}초)" if gained else "")
    )


def _entry_tf_of(strategy: str) -> str:
    """로그 태그에서 진입봉을 추정한다. 태그 끝에 봉이 붙는 전략이 있다."""

    from config.symbols import ACTIVE

    tail = strategy.rsplit(".", 1)[-1]
    from core.candles import INTERVAL_SECONDS

    if tail in INTERVAL_SECONDS:
        return tail
    for spec in ACTIVE:
        for s in spec.strategies:
            if strategy.startswith(s.name):
                return s.entry_tf
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
