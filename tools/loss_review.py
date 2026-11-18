"""이긴 거래와 진 거래의 진입 근거 비교.

    python -m tools.loss_review data/server_20261118/xrp
    python -m tools.loss_review data/server_20261118 data/server_20261118/btc25x

디렉터리마다 trades.csv(완료 포지션)와 signals.jsonl(진입 당시 근거)을 이어 붙여,
전략별로 근거 항목 유무·지표 높낮이에 따른 승률과 평균 R을 마크다운 표로 낸다.
출력은 사람이나 AI가 읽고 "필터 후보 가설"을 세우는 용도다. 여기서 나온 차이는
같은 데이터에서 찾은 것이므로 채택 전에 반드시 리플레이·시간 분할로 다시 검증한다.

페이퍼 포워드와 리플레이 디렉터리를 한 번에 넘겨도 결과는 디렉터리별로 따로 낸다
(AGENTS.md: 두 결과 유형을 합치지 않는다).
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from engine.replay import ReplayResult
from reporting.logs import read_signals, read_trades

MIN_N = 20              # 이보다 적은 칸은 "표본부족"으로 표시한다
MATCH_WINDOW = 6 * 3600 # 체결 ts와 신호 ts 허용 차이(초). 실시간 신호는 봉 마감 뒤 벽시계 ts다


def load_positions(directory: Path) -> tuple[list[dict], int]:
    """완료 포지션에 진입 신호를 붙인다. (포지션 목록, 신호 못 찾은 수)"""

    trades = read_trades(directory / "trades.csv")
    signals = [s for s in read_signals(directory / "signals.jsonl") if s.get("action") == "entry"]
    result = ReplayResult(0, 0.0, None, None, 0.0, 0.0, trades=trades)
    positions = result.completed_positions()

    pool: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in signals:
        pool[(s.get("strategy", ""), s.get("side", ""))].append(s)

    entry_price = {
        (r["strategy"], int(r["ts"])): float(r["price"])
        for r in trades if r.get("action") == "entry"
    }

    unmatched = 0
    for p in positions:
        key = (p["strategy"], p["side"])
        cands = [s for s in pool[key] if abs(int(s["ts"]) - p["entry_ts"]) <= MATCH_WINDOW]
        if not cands:
            # 2026-08 이전 리플레이는 신호 ts가 시장 시각이 아니라 실행 벽시계다.
            # 그때는 체결가와 같은 진입가를 가진 가장 앞선 신호를 쓴다.
            price = entry_price.get((p["strategy"], p["entry_ts"]))
            cands = [s for s in pool[key] if price is not None
                     and abs(float(s.get("entry") or 0) - price) < 1e-9][:1]
        if not cands:
            p["signal"] = None
            unmatched += 1
            continue
        best = min(cands, key=lambda s: abs(int(s["ts"]) - p["entry_ts"]))
        pool[(p["strategy"], p["side"])].remove(best)  # 한 신호를 두 포지션에 쓰지 않는다
        p["signal"] = best
    return positions, unmatched


def features(p: dict) -> tuple[set[str], dict[str, float]]:
    """(근거 플래그 집합, 수치 지표). 숫자가 붙은 reason은 지표 쪽에 이미 있으므로 뺀다."""

    s = p["signal"] or {}
    flags = {r for r in s.get("reasons") or [] if not re.search(r"\d", r)}
    entry = float(s.get("entry") or 0)
    nums: dict[str, float] = {}
    for k, v in {"score": s.get("score"), **(s.get("indicators") or {})}.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            # 목선·고점·MA 같은 절대 가격은 "그때 시세가 높았나"일 뿐이라 뺀다
            if entry and 0.5 < float(v) / entry < 2:
                continue
            nums[k] = float(v)
    nums["hour_utc"] = datetime.fromtimestamp(p["entry_ts"], tz=timezone.utc).hour
    return flags, nums


def r_multiple(p: dict) -> float | None:
    risk = (p["signal"] or {}).get("risk_amount")
    return p["pnl"] / risk if risk else None


def stats(group: list[dict]) -> str:
    n = len(group)
    if not n:
        return "0 | - | - |"
    wins = sum(p["pnl"] > 0 for p in group)
    rs = [r for r in map(r_multiple, group) if r is not None]
    avg_r = f"{statistics.fmean(rs):+.2f}" if rs else "-"
    mark = " (표본부족)" if n < MIN_N else ""
    return f"{n}{mark} | {100 * wins / n:.0f}% | {avg_r} |"


def report(directory: Path) -> str:
    positions, unmatched = load_positions(directory)
    out = [f"## {directory}", ""]
    if not positions:
        return "\n".join(out + ["완료 포지션 없음.", ""])
    first = min(p["entry_ts"] for p in positions)
    last = max(p["exit_ts"] for p in positions)
    fmt = lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    out.append(f"구간 {fmt(first)} ~ {fmt(last)}, 완료 포지션 {len(positions)}건, "
               f"진입 신호 매칭 실패 {unmatched}건 (실패분은 근거 비교에서 제외)")
    out.append("")

    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for p in positions:
        by_strategy[p["strategy"]].append(p)

    for name, group in sorted(by_strategy.items(), key=lambda x: -len(x[1])):
        out += [f"### {name}", "", "| 구분 | 거래 | 승률 | 평균 R |", "|---|---|---|---|",
                f"| 전체 | {stats(group)}"]
        exits = defaultdict(list)
        for p in group:
            exits[p["exit_reason"]].append(p)
        for reason, g in sorted(exits.items()):
            out.append(f"| 청산={reason or '?'} | {stats(g)}")
        for side in ("long", "short"):
            g = [p for p in group if p["side"] == side]
            if g and len(g) < len(group):
                out.append(f"| {side} | {stats(g)}")

        matched = [p for p in group if p["signal"]]
        feats = [(p, *features(p)) for p in matched]
        all_flags = sorted({f for _, flags, _ in feats for f in flags})
        for flag in all_flags:
            with_ = [p for p, flags, _ in feats if flag in flags]
            without = [p for p, flags, _ in feats if flag not in flags]
            if with_ and without:  # 모두 갖거나 모두 없으면 비교 불가
                out.append(f"| {flag} 있음 | {stats(with_)}")
                out.append(f"| {flag} 없음 | {stats(without)}")

        all_nums = sorted({k for _, _, nums in feats for k in nums})
        for key in all_nums:
            vals = [(p, nums[key]) for p, _, nums in feats if key in nums]
            if len({v for _, v in vals}) < 2:
                continue
            med = statistics.median(v for _, v in vals)
            low = [p for p, v in vals if v <= med]
            high = [p for p, v in vals if v > med]
            if low and high:
                out.append(f"| {key} ≤ {med:.4g} | {stats(low)}")
                out.append(f"| {key} > {med:.4g} | {stats(high)}")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", type=Path, help="trades.csv·signals.jsonl이 있는 디렉터리")
    args = ap.parse_args()
    print("# 승패 근거 비교\n")
    print(f"R = 포지션 손익 / 진입 시 risk_amount. 거래 {MIN_N}건 미만 칸은 우연일 가능성이 크다.")
    print("중앙값 분할은 같은 데이터에서 정한 경계라 채택 근거가 아니라 가설 후보다.\n")
    for d in args.dirs:
        print(report(d))


if __name__ == "__main__":
    main()
