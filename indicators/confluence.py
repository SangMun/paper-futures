"""합류 구간 스코어링.

명세 §4. "어떤 스윙이 정답인가"는 스케일 선택 문제라 유일해가 없다.
하나를 고르는 대신 **여러 후보에서 나온 레벨이 겹치는 구간**을 찾는다.

절차
  1. 스윙 후보 상위 K개 (크기 기준)
  2. 각 후보에서 핵심 레벨 계산
  3. 허용 오차(기본 0.25 ATR) 이내로 모이는 레벨을 하나의 존으로 병합
  4. 존별 점수 산출

가중치는 전부 설정값(`CONFLUENCE_WEIGHTS`)이고 포워드 테스트로 조정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config.settings import CONFLUENCE_WEIGHTS, FIB, VPVR
from indicators import fib, vpvr
from indicators.base import IndicatorError, column
from indicators.pivots import Swing, recent_swings, swings_at


@dataclass
class Zone:
    """합류 구간 하나."""

    price: float                # 구성 레벨의 평균
    low: float
    high: float
    side: str                   # 이 존에서 노릴 방향
    score: float = 0.0
    members: list[str] = field(default_factory=list)
    swings: list[Swing] = field(default_factory=list)

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def touched_by(self, bar_low: float, bar_high: float) -> bool:
        """봉이 존에 닿았는지. 종가가 아니라 고저 범위로 본다."""

        return bar_low <= self.high and bar_high >= self.low


def _merge(levels: list[fib.FibLevel], tolerance: float) -> list[list[fib.FibLevel]]:
    """가격이 가까운 레벨끼리 묶는다.

    정렬 후 인접 레벨의 간격이 tolerance 이내면 같은 무리로 본다.
    """

    if not levels:
        return []

    ordered = sorted(levels, key=lambda x: x.price)
    groups: list[list[fib.FibLevel]] = [[ordered[0]]]

    for level in ordered[1:]:
        if level.price - groups[-1][-1].price <= tolerance:
            groups[-1].append(level)
        else:
            groups.append([level])

    return groups


def _independent(swings, max_overlap: float) -> int:
    """구조적으로 독립인 스윙이 몇 개인가.

    `id()`로 세면 안 된다. `build_swings`는 같은 시작점에서 새 극값이 나올
    때마다 후보를 남기므로, 상위 후보들은 대부분 **하나의 다리를 크기만 달리
    본 중첩 버전**이다. 객체는 다르지만 같은 움직임이라 독립 증거가 아니다.

    실측(BTC 1시간봉 597개 시점): 상위 후보 3개의 가격 구간 겹침 중앙값이
    1.00이었고 94.8%의 시점에서 90% 이상 겹쳤다. 그런데도 겹침 가점은
    꼬박꼬박 붙어, 문턱을 넘은 존의 51.3%가 그 가점 없이는 미달이었다.

    큰 다리부터 잡고, 이미 잡은 다리와 가격 구간이 `max_overlap` 넘게
    겹치면 같은 다리로 본다.
    """

    chosen: list = []
    for swing in sorted(swings, key=lambda x: x.size, reverse=True):
        if swing.size <= 0:
            continue
        duplicate = False
        for kept in chosen:
            share = min(swing.high, kept.high) - max(swing.low, kept.low)
            if share > 0 and share / min(swing.size, kept.size) > max_overlap:
                duplicate = True
                break
        if not duplicate:
            chosen.append(swing)
    return len(chosen)


def _score(
    group: list[fib.FibLevel],
    center: float,
    tolerance: float,
    ma_values: dict[str, float],
    hvn: tuple[float, ...],
    horizontals: tuple[float, ...],
) -> tuple[float, list[str]]:
    """존 하나의 점수와 구성 요소.

    가중치는 명세 §4의 표를 그대로 쓴다. 값 자체에 이론적 근거는 없고,
    포워드 테스트로 조정할 출발점이다.
    """

    members: list[str] = []
    score = 0.0

    # 서로 다른 스윙에서 나온 레벨이 겹칠 때 개당 가점.
    # 같은 스윙의 레벨 여러 개가 겹치는 건 합류가 아니다.
    unique = {id(level.swing): level.swing for level in group}.values()
    distinct = _independent(list(unique), FIB.swing_overlap_max)
    if distinct > 1:
        score += CONFLUENCE_WEIGHTS["fib_overlap"] * (distinct - 1)
        members.append(f"fib_overlap_{distinct}")

    if any(level.is_golden for level in group):
        score += CONFLUENCE_WEIGHTS["golden_pocket"]
        members.append("golden_pocket")

    members.extend(sorted({level.label() for level in group}))

    for name, value in ma_values.items():
        if value is not None and abs(value - center) <= tolerance:
            score += CONFLUENCE_WEIGHTS[name]
            members.append(name)

    if any(abs(node - center) <= tolerance for node in hvn):
        score += CONFLUENCE_WEIGHTS["vpvr_hvn"]
        members.append("vpvr_hvn")

    if any(abs(level - center) <= tolerance for level in horizontals):
        score += CONFLUENCE_WEIGHTS["horizontal_sr"]
        members.append("horizontal_sr")

    return score, members


def find_zones(
    df: pd.DataFrame,
    as_of: int,
    atr: float,
    ma_fast: float | None = None,
    ma_slow: float | None = None,
    pivot_n: int = FIB.pivot_n,
    min_swing_pct: float = FIB.min_swing_pct,
    tolerance_atr: float = FIB.zone_tolerance_atr,
    max_lookback: int = FIB.max_lookback,
    limit: int = FIB.max_swing_candidates,
    precomputed_swings: list[Swing] | None = None,
    profile: vpvr.VolumeProfile | None = None,
) -> list[Zone]:
    """as_of 시점의 합류 존. 점수 높은 순.

    ATR이 0 이하면 허용 오차를 정할 수 없으므로 빈 목록을 돌려준다.

    precomputed_swings를 넘기면 피벗 탐지를 건너뛴다. 리플레이는 반드시
    이쪽을 써야 한다 — 봉마다 전 구간을 다시 스캔하면 9,600봉에 63초가 걸린다.
    """

    if as_of < 0 or as_of >= len(df):
        raise IndicatorError(f"as_of가 범위를 벗어남: {as_of} (길이 {len(df)})")
    if atr <= 0 or not pd.notna(atr):
        return []

    if precomputed_swings is not None:
        swings = swings_at(precomputed_swings, as_of, max_lookback, limit)
    else:
        swings = recent_swings(
            df, as_of, n=pivot_n, min_swing_pct=min_swing_pct,
            max_lookback=max_lookback, limit=limit,
        )
    if not swings:
        return []

    close = float(column(df, "close").iloc[as_of])
    live = [s for s in swings if not fib.invalidated(s, close)]
    if not live:
        return []

    tolerance = atr * tolerance_atr

    if profile is None:
        profile = vpvr.profile(df, as_of=as_of, lookback=VPVR.lookback_bars)
    hvn = profile.hvn if profile is not None else ()

    # 수평 지지저항 = 과거 스윙의 고저점
    horizontals = tuple(p.price for s in live for p in (s.start, s.end))

    ma_values = {"ma_50": ma_fast, "ma_200": ma_slow}

    zones: list[Zone] = []
    for side in ("long", "short"):
        levels = [
            level
            for s in live
            if fib.entry_side(s) == side
            for level in fib.core_levels(s)
        ]
        for group in _merge(levels, tolerance):
            center = sum(x.price for x in group) / len(group)
            score, members = _score(group, center, tolerance, ma_values, hvn, horizontals)
            zones.append(
                Zone(
                    price=center,
                    low=min(x.price for x in group) - tolerance,
                    high=max(x.price for x in group) + tolerance,
                    side=side,
                    score=score,
                    members=members,
                    swings=sorted({id(x.swing): x.swing for x in group}.values(), key=lambda s: s.end.index),
                )
            )

    return sorted(zones, key=lambda z: z.score, reverse=True)
