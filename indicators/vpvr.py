"""가격대별 거래량 (Volume Profile, VPVR).

두 곳에서 쓴다.
  - BTC 익절 1차 목표: 현재가 위쪽 최근접 HVN (명세 §7)
  - 합류 존 스코어: HVN과 겹치면 +2 (명세 §4)

거래량은 계약 수(`volume`)가 아니라 USDT 환산 거래대금(`quote_volume`)을 쓴다.
가격대별로 "얼마나 많은 돈이 묶였는가"를 보는 지표이므로, 가격이 다른 구간을
계약 수로 비교하면 낮은 가격대가 과대평가된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import VPVR
from indicators.base import IndicatorError, check_period, column


@dataclass(frozen=True)
class VolumeProfile:
    """한 시점에서 본 거래량 분포.

    edges는 bins+1개, volumes/centers는 bins개.
    """

    edges: np.ndarray
    volumes: np.ndarray
    centers: np.ndarray
    poc: float                      # 최대 거래량 가격대
    hvn: tuple[float, ...]          # 고거래량 노드 (지지/저항, 익절 목표)
    lvn: tuple[float, ...]          # 저거래량 노드 (빠르게 통과하는 구간)
    value_area: tuple[float, float]  # 전체 거래량의 70%가 몰린 구간 (VAL, VAH)

    def nearest_above(self, price: float, levels: tuple[float, ...] | None = None) -> float | None:
        """price 위쪽 최근접 노드. 없으면 None."""

        cand = [x for x in (self.hvn if levels is None else levels) if x > price]
        return min(cand) if cand else None

    def nearest_below(self, price: float, levels: tuple[float, ...] | None = None) -> float | None:
        cand = [x for x in (self.hvn if levels is None else levels) if x < price]
        return max(cand) if cand else None


def _distribute(df: pd.DataFrame, edges: np.ndarray) -> np.ndarray:
    """각 캔들의 거래대금을 고저 범위에 걸친 빈들에 균등 배분한다.

    종가 한 점에 몰아넣지 않는 이유는, 한 봉이 여러 가격대를 지나며 체결된
    사실을 버리게 되기 때문이다. 특히 변동이 큰 봉에서 차이가 크다.
    실제 체결 분포는 알 수 없으므로 균등 배분이 가장 무난한 근사다.
    """

    n_bins = len(edges) - 1
    out = np.zeros(n_bins, dtype="float64")

    lows = column(df, "low").to_numpy()
    highs = column(df, "high").to_numpy()
    vols = column(df, "quote_volume").to_numpy()

    # quote_volume이 비어 있는 응답을 대비해 계약 수로 대체한다.
    if not np.any(vols > 0):
        vols = column(df, "volume").to_numpy()

    lo_idx = np.clip(np.searchsorted(edges, lows, side="right") - 1, 0, n_bins - 1)
    hi_idx = np.clip(np.searchsorted(edges, highs, side="right") - 1, 0, n_bins - 1)

    for lo, hi, v in zip(lo_idx, hi_idx, vols):
        if v <= 0:
            continue
        out[lo : hi + 1] += v / (hi - lo + 1)

    return out


def _cluster(centers: np.ndarray, mask: np.ndarray, volumes: np.ndarray) -> tuple[float, ...]:
    """연속한 빈들을 하나의 노드로 묶는다.

    묶지 않으면 붙어 있는 빈 여러 개가 전부 별개 레벨로 나와, 익절 목표나
    합류 판정에서 같은 가격대를 중복으로 세게 된다.
    대표 가격은 묶인 빈들의 거래량 가중 평균으로 잡는다.
    """

    nodes: list[float] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        seg_v = volumes[i : j + 1]
        seg_c = centers[i : j + 1]
        total = seg_v.sum()
        nodes.append(float((seg_c * seg_v).sum() / total) if total > 0 else float(seg_c.mean()))
        i = j + 1
    return tuple(nodes)


def _value_area(volumes: np.ndarray, centers: np.ndarray, edges: np.ndarray, ratio: float = 0.70):
    """POC에서 시작해 양옆 중 거래량이 큰 쪽을 붙여 가며 ratio에 도달할 때까지 확장."""

    total = volumes.sum()
    if total <= 0:
        return float(edges[0]), float(edges[-1])

    poc_i = int(np.argmax(volumes))
    lo = hi = poc_i
    acc = volumes[poc_i]

    while acc < total * ratio and (lo > 0 or hi < len(volumes) - 1):
        left = volumes[lo - 1] if lo > 0 else -1.0
        right = volumes[hi + 1] if hi < len(volumes) - 1 else -1.0
        if right >= left:
            hi += 1
            acc += volumes[hi]
        else:
            lo -= 1
            acc += volumes[lo]

    return float(edges[lo]), float(edges[hi + 1])


def profile(
    df: pd.DataFrame,
    as_of: int,
    lookback: int = VPVR.lookback_bars,
    num_bins: int = VPVR.num_bins,
    hvn_percentile: float = VPVR.hvn_percentile,
    lvn_percentile: float = VPVR.lvn_percentile,
) -> VolumeProfile | None:
    """as_of 시점까지의 거래량 분포. 데이터가 모자라면 None.

    as_of는 위치 인덱스이고 **해당 봉을 포함**한다. 미래 봉은 보지 않는다.
    이 함수 자체에 확정 지연은 없지만, 룩백 창이 어디서 끝나는지를
    호출부가 명시하도록 인자를 필수로 둔다.
    """

    num_bins = check_period(num_bins, "num_bins")
    if not 0.0 <= lvn_percentile < hvn_percentile <= 100.0:
        raise IndicatorError(
            f"백분위 순서가 잘못됨: lvn={lvn_percentile}, hvn={hvn_percentile}"
        )

    if as_of < 0 or as_of >= len(df):
        raise IndicatorError(f"as_of가 범위를 벗어남: {as_of} (길이 {len(df)})")

    start = max(0, as_of - lookback + 1)
    window = df.iloc[start : as_of + 1]

    # 빈 하나에 봉 하나 꼴이면 분포라고 부를 수 없다.
    if len(window) < num_bins:
        return None

    low = float(column(window, "low").min())
    high = float(column(window, "high").max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None

    edges = np.linspace(low, high, num_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    volumes = _distribute(window, edges)

    if volumes.sum() <= 0:
        return None

    # HVN 임계값은 **거래가 있었던 빈들** 안에서만 계산한다.
    # 가격이 두 구간에만 머물면 나머지 빈은 전부 0이고, 그걸 포함해
    # 백분위를 내면 임계값이 0으로 내려가 모든 빈이 HVN이 된다.
    # 그러면 존 전체가 하나로 묶여 익절 목표도 합류 판정도 무의미해진다.
    traded = volumes[volumes > 0]
    hvn_cut = float(np.percentile(traded, hvn_percentile)) if len(traded) else np.inf
    hvn_mask = (volumes >= hvn_cut) & (volumes > 0)

    # LVN은 반대로 빈 구간을 포함한다. 거래가 없던 가격대야말로
    # 가격이 빠르게 통과하는 구간이라는 뜻이기 때문이다.
    lvn_cut = float(np.percentile(volumes, lvn_percentile))

    val, vah = _value_area(volumes, centers, edges)

    return VolumeProfile(
        edges=edges,
        volumes=volumes,
        centers=centers,
        poc=float(centers[int(np.argmax(volumes))]),
        hvn=_cluster(centers, hvn_mask, volumes),
        lvn=_cluster(centers, volumes <= lvn_cut, volumes),
        value_area=(val, vah),
    )
