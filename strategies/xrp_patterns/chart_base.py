"""차트 패턴 공통 기반.

다섯 패턴(헤드앤숄더·쐐기·대칭삼각·컵앤핸들·이중천장)이 공유하는 판정.

핵심 원칙 — 전부 여기서 강제한다.
  1. **패턴 완성 전에는 진입하지 않는다.** 모양이 보인다는 것과 완성됐다는
     건 다르다. 목선/추세선을 **종가로** 이탈해야 완성이다.
  2. 꼬리로 잠깐 넘는 건 이탈이 아니다. 그래서 종가 기준만 본다.
  3. 거래량·매물대·BTC 방향을 함께 본다. 모양만으로는 진입하지 않는다.
  4. 손절은 패턴 무효화 지점. 포지션 크기는 거기서 역산된다.
  5. 패턴 높이 목표가는 참고치다. 실제 1차 목표는 VPVR HVN을 쓴다.

품질 판정표는 아홉 항목이고, 최소 `MIN_QUALITY`개를 충족해야 신호가 된다.
"모양이 맞다"와 "지금 그 자리에서 의미가 있다"를 분리하기 위해서다.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from indicators import volume as vol_mod
from indicators import vpvr
from indicators import vwap as vwap_mod
from indicators.pivots import Pivot, all_pivots, pivot_series
from strategies.xrp_patterns import base as xb


# 판정표 항목 중 최소 몇 개를 만족해야 하는가.
#
# 4에서 3으로 내린 것은 기준을 완화한 것이 아니라 **원래 자리로 되돌린
# 것이다.** 예전 판정표에는 사실상 항상 켜지는 항목이 둘 있었다 —
# `retested`(호출 인자가 잘못돼 돌파 봉 자기 자신을 검사했다)와
# `no_wall_ahead`(기준 거리가 돌파 순간 거의 0이었다). 실거래 15건 전부가
# 두 항목을 켰다. 그래서 "4개 충족"의 실질은 2개였다.
#
# 두 항목을 고친 뒤 2.5년 리플레이(275,241봉, XRP)로 잰 값:
#   기존 코드 MQ4   547건  +81.9%  낙폭 14.0%
#   수정 코드 MQ2   548건  +83.8%  낙폭 13.9%   <- 기존과 동일한 문턱
#   수정 코드 MQ3   540건  +81.1%  낙폭 13.7%
#   수정 코드 MQ4   489건  +73.6%  낙폭 12.9%
#   수정 코드 MQ5   409건  +49.4%  낙폭 17.9%
# 2와 3은 서로 노이즈 범위이고 4 이상은 수익이 확실히 준다. 둘 중 3을 쓰는
# 이유는 `no_wall_ahead`가 여전히 비교적 쉽게 켜지기 때문이다.
MIN_QUALITY = 3

# 하위 호환 기본값. 실제 값은 전략 인스턴스의 Timeframes가 정한다.
# 돌파 재확인용 상위 캔들 / 상위 지지저항 판정용.
CONFIRM_TF = "15m"
HTF_KEY = "1h"

# 엔진이 넣어 주는 미결제약정 프레임의 키.
OI_KEY = "OI"

# OI가 이 비율 이상 늘면 신규 포지션 유입으로 본다.
OI_EXPANSION = 0.002

# OI 값이 없을 때 직전 값으로 채울 최대 봉 수.
# 이 이상 낡으면 NaN으로 두어 판정 항목을 건너뛴다 — 모르는 걸
# "변화 없음"으로 읽는 것보다 빠뜨리는 편이 안전하다.
OI_STALE_LIMIT = 3

# 돌파 봉의 상대 거래량 하한. 절대 배수를 박지 않고 RVOL로 본다.
BREAKOUT_RVOL = 1.3

# 피벗 탐지 폭. 5분봉에서 너무 작으면 노이즈가 패턴이 된다.
PIVOT_N = 5

# 패턴 하나가 차지할 수 있는 최대 봉 수. 이보다 길면 "최근 구조"가 아니다.
MAX_PATTERN_BARS = 180

# 돌파 앞쪽 매물벽이 이 ATR 배수보다 멀어야 "진행 공간이 있다"로 본다.
WALL_CLEARANCE_ATR = 1.0

# 가격이 같다고 볼 허용 오차 (어깨 높이, 두 고점 등).
LEVEL_TOLERANCE = 0.004


@dataclass
class PivotView:
    """확정 피벗을 as_of로 빠르게 잘라 쓰기 위한 색인.

    `usable_pivots`를 봉마다 부르면 매번 전체를 훑어 O(n^2)가 된다.
    확정 시각으로 정렬해 두고 이분 탐색으로 자른다.
    """

    pivots: list[Pivot]
    _confirmed: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.pivots = sorted(self.pivots, key=lambda p: (p.confirmed_at, p.index))
        self._confirmed = [p.confirmed_at for p in self.pivots]

    def upto(self, as_of: int, count: int = 8) -> list[Pivot]:
        """as_of에 확정된 피벗 중 마지막 count개. 시간순."""

        end = bisect.bisect_right(self._confirmed, as_of)
        recent = self.pivots[max(0, end - count) : end]
        return sorted(recent, key=lambda p: p.index)


def close_break(df: pd.DataFrame, as_of: int, level: float, side: str) -> bool:
    """이번 봉 **종가**가 레벨을 이탈했는가.

    꼬리로 넘는 건 이탈로 보지 않는다. 직전 봉은 레벨 안쪽이어야
    "방금 이탈했다"가 성립한다 — 이미 한참 전에 넘어간 자리를
    매번 새 신호로 세면 같은 이탈을 여러 번 잡는다.
    """

    if as_of < 1:
        return False

    close = float(df["close"].iloc[as_of])
    prev = float(df["close"].iloc[as_of - 1])

    if side == "short":
        return close < level <= prev
    return close > level >= prev


def retest_ok(df: pd.DataFrame, break_at: int, as_of: int, level: float, side: str) -> bool:
    """돌파 후 리테스트에 성공했는가 (패턴 내부로 복귀하지 않음).

    **현재 어느 패턴도 이 함수를 부르지 않는다.** 진입이 돌파 봉 종가에서
    일어나므로 그 시점에는 리테스트가 아직 존재할 수 없기 때문이다.
    이전에는 다섯 패턴이 `break_at=as_of - 1`을 넘겨 호출했는데, 그러면
    검사 구간이 돌파 봉 한 줄로 좁혀져 `close < level` 조건이 돌파의 정의와
    같아지고 항상 참이 됐다 — 실거래 15건 전부가 이 항목을 켰다.

    함수 자체는 맞다. 돌파를 기억해 두고 이후 봉에서 되돌림 진입을 판정하는
    별도 모드를 만들 때 `break_at`에 실제 돌파 봉 인덱스를 넘겨 쓰면 된다.
    """

    if as_of <= break_at or break_at < 0:
        return False

    seg = df.iloc[break_at + 1 : as_of + 1]
    if seg.empty:
        return False

    if side == "short":
        # 되돌아 올라와 레벨을 건드렸지만 종가로 회복하지는 못함
        touched = bool((seg["high"] >= level).any())
        return touched and bool((seg["close"] < level).all())

    touched = bool((seg["low"] <= level).any())
    return touched and bool((seg["close"] > level).all())


def assess(
    df: pd.DataFrame,
    prepared,
    as_of: int,
    side: str,
    start: int,
    level: float,
    profile: vpvr.VolumeProfile | None,
    prior_trend_ok: bool,
    tf=None,
) -> list[str]:
    """품질 판정표. 충족한 항목 이름을 돌려준다.

    아홉 항목 중 개수로 거른다. 각 항목의 가중치를 다르게 주지 않는 이유는,
    가중치를 정할 근거가 없어서다 — 자료마다 강조점이 다르고, 합의된
    수치가 없다. 개수는 최소한 임의성이 드러난다.
    """

    tf = tf or xb.DEFAULT_TF
    entry_tf = tf.entry
    hits: list[str] = []

    # ① 선행 추세 — 패턴이 요구하는 방향의 추세가 앞에 있었는가
    if prior_trend_ok:
        hits.append("prior_trend")

    # ② 형성 중 거래량 감소
    if vol_mod.declining_between(df, start, as_of - 1):
        hits.append("volume_dry_up")

    # ③ 돌파 봉 거래량 증가
    rv = prepared.at(entry_tf, "rvol", as_of)
    if pd.notna(rv) and rv >= BREAKOUT_RVOL:
        hits.append(f"rvol_{rv:.1f}")

    # ④ 미결제약정 확대 — 돌파에 신규 자금이 들어왔는가.
    #    OI 감소는 청산성 스퀴즈다. 강해 보여도 소진되기 쉬워 가점하지 않는다.
    #    (감소 여부 자체는 각 패턴이 context["oi_change"]에 남겨 사후 분석한다)
    oi = prepared.at(entry_tf, "oi_change", as_of)
    if pd.notna(oi) and oi >= OI_EXPANSION:
        hits.append(f"oi_up_{oi:.2%}")

    # ⑤ VWAP 기준 방향 일치 — 세션 평균가 대비 어느 쪽에서 돌파했는가.
    #    상승 돌파가 VWAP 위에서 나오면 매수 우위 구간이라는 뜻이다.
    vw = prepared.at(entry_tf, "vwap", as_of)
    if pd.notna(vw):
        close_now = float(df["close"].iloc[as_of])
        if (side == "long" and close_now > vw) or (side == "short" and close_now < vw):
            hits.append("vwap_aligned")

    # ⑥ 상위 캔들 종가로도 같은 이탈이 성립하는가 (5분 -> 15분).
    #    꼬리로 잠깐 넘은 게 아니라는 확인이다.
    if prepared.at(entry_tf, "confirm_break_long" if side == "long" else "confirm_break_short", as_of):
        hits.append(f"{tf.confirm}_confirmed")

    # ⑦ BTC 방향 일치 (역방향이면 아예 blocked에서 걸린다)
    adverse = prepared.at(entry_tf, "btc_adverse", as_of)
    bias = prepared.at(entry_tf, "ctx_bias", as_of)
    if (side == "short" and (adverse == "down" or bias == "bear")) or (
        side == "long" and (adverse == "up" or bias == "bull")
    ):
        hits.append("btc_aligned")

    close = float(df["close"].iloc[as_of])

    if profile is not None:
        val, vah = profile.value_area

        # ⑨ 반전 패턴이 가치영역 경계에서 발생했는가
        #    상단 반전은 VAH 부근, 하단 반전은 VAL 부근이어야 의미가 크다
        if (side == "short" and level >= profile.poc) or (
            side == "long" and level <= profile.poc
        ):
            hits.append("at_value_edge")

        # ⑩ 돌파 방향 앞이 비어 있는가 (LVN) — 진행 공간
        ahead = [
            x for x in profile.lvn
            if (x < close if side == "short" else x > close)
        ]
        if ahead:
            nearest_lvn = max(ahead) if side == "short" else min(ahead)
            if abs(nearest_lvn - close) <= abs(close - level) * 3:
                hits.append("lvn_ahead")

        # ⑪ 돌파 방향 바로 앞에 큰 반대 매물대가 없는가.
        #
        # 기준 거리를 종가-레벨 간격으로 두면 안 된다. 돌파 봉에서는 방금
        # 레벨을 넘은 참이라 그 간격이 거의 0이고, 어떤 벽이든 "충분히 멀다"로
        # 통과한다. 실측으로 차트패턴 거래 15건 전부가 이 항목을 켰다.
        # ATR을 쓰면 "한 봉이 갈 만한 거리 안에 벽이 있는가"라는 원래 의도가
        # 종목·타임프레임과 무관하게 유지된다.
        blocking = [
            x for x in profile.hvn
            if (x < close if side == "short" else x > close)
        ]
        nearest_hvn = (
            (max(blocking) if side == "short" else min(blocking)) if blocking else None
        )
        atr = xb.atr_at(prepared, as_of, tf)
        room = atr * WALL_CLEARANCE_ATR
        # 매물대를 못 찾았으면 모르는 것이지 비어 있는 것이 아니다. 가점하지 않는다.
        if room > 0 and nearest_hvn is not None and abs(nearest_hvn - close) >= room:
            hits.append("no_wall_ahead")

    # ⑧ 상위 타임프레임(1시간봉)의 의미 있는 지지·저항에서 발생했는가.
    #    5분봉 VPVR만 보면 "상위 시간대 자리"를 확인할 수 없다.
    htf = prepared.at(entry_tf, "htf_level_near", as_of)
    if pd.notna(htf) and bool(htf):
        hits.append("htf_level")

    return hits


def pattern_target(entry: float, height: float, side: str) -> float:
    """패턴 높이 목표가. **참고치일 뿐 보장된 도달값이 아니다.**

    실제 1차 익절은 VPVR HVN을 쓴다(target_hint). 이 값은 로그에만 남겨
    나중에 "패턴 높이가 실제로 도달됐는가"를 집계하는 데 쓴다.
    """

    return entry - height if side == "short" else entry + height


def chart_prepare(frames: dict[str, pd.DataFrame], tf=None) -> tuple:
    """차트 패턴 공통 준비. (prepared, PivotView) 반환.

    피벗과 RVOL을 여기서 한 번만 계산한다. 봉마다 다시 찾으면
    5분봉 51,835봉 리플레이가 성립하지 않는다.
    """

    tf = tf or xb.DEFAULT_TF
    prepared = xb.scalp_prepare(frames, tf)
    entry = frames[tf.entry]

    series = {
        "rvol": vol_mod.rvol(entry),
        "vol_trend": vol_mod.volume_trend(entry),
        "oi_change": oi_change(frames, entry),
        "vwap": vwap_mod.vwap(entry),
        "htf_level_near": htf_levels(frames, entry, tf),
    }
    series.update(confirm_breaks(frames, entry, tf))
    prepared.series[tf.entry].update(series)
    return prepared, PivotView(all_pivots(entry, PIVOT_N))


def confirm_breaks(frames: dict[str, pd.DataFrame], entry: pd.DataFrame, tf=None) -> dict:
    """상위 캔들(15분봉) 종가 이탈 여부를 진입봉 축에 정렬한다.

    "꼬리로 잠깐 넘는 것보다 5분·15분봉 종가 돌파를 우선"이라는 요구를
    구현한 것이다. 진입 판정은 5분봉에서 하되, 같은 방향 이탈이 15분봉
    종가로도 나왔으면 가점한다.

    상위봉은 마감된 것만 본다 — 진행 중인 15분봉의 종가를 쓰면 미래참조다.
    """

    tf = tf or xb.DEFAULT_TF
    empty = pd.Series(False, index=entry.index)
    confirm = frames.get(tf.confirm)
    if confirm is None or confirm.empty:
        return {"confirm_break_long": empty, "confirm_break_short": empty}

    from strategies.btc_trend import context_index

    if tf.confirm == tf.entry:
        return {"confirm_break_long": empty, "confirm_break_short": empty}

    idx = context_index(
        entry["ts"].to_numpy(),
        confirm["ts"].to_numpy(),
        entry_tf=tf.entry,
        context_tf=tf.confirm,
    )

    c_close = confirm["close"].to_numpy()
    c_open = confirm["open"].to_numpy()

    up, down = [], []
    for i in idx:
        if i < 0:
            up.append(False)
            down.append(False)
        else:
            up.append(bool(c_close[i] > c_open[i]))
            down.append(bool(c_close[i] < c_open[i]))

    return {
        "confirm_break_long": pd.Series(up, index=entry.index),
        "confirm_break_short": pd.Series(down, index=entry.index),
    }


def htf_levels(frames: dict[str, pd.DataFrame], entry: pd.DataFrame, tf=None) -> pd.Series:
    """상위 캔들(1시간봉)의 스윙 고저점 근처인가.

    "1시간·4시간봉의 의미 있는 지지·저항에서 발생" 조건이다.
    5분봉 VPVR 가치영역만 보면 상위 시간대의 자리를 확인할 수 없다.

    상위봉 피벗도 확정된 것만 쓴다.
    """

    tf = tf or xb.DEFAULT_TF
    htf = frames.get(tf.htf)
    if htf is None or htf.empty or tf.htf == tf.entry:
        return pd.Series(False, index=entry.index)

    last_high, last_low = pivot_series(htf, n=3)

    from strategies.btc_trend import context_index

    idx = context_index(
        entry["ts"].to_numpy(),
        htf["ts"].to_numpy(),
        entry_tf=tf.entry,
        context_tf=tf.htf,
    )

    highs = last_high.to_numpy()
    lows = last_low.to_numpy()
    close = entry["close"].to_numpy()

    out = []
    for k, i in enumerate(idx):
        if i < 0:
            out.append(False)
            continue
        px = close[k]
        near_level = any(
            not np.isnan(lvl) and abs(px - lvl) / px <= LEVEL_TOLERANCE
            for lvl in (highs[i], lows[i])
        )
        out.append(near_level)

    return pd.Series(out, index=entry.index)


def oi_change(frames: dict[str, pd.DataFrame], entry: pd.DataFrame) -> pd.Series:
    """봉당 미결제약정 변화율. OI 프레임이 없으면 전부 NaN.

    돌파의 성격을 가른다.
      돌파 + 거래량 증가 + OI 증가 -> 신규 자금 유입
      돌파 + 거래량 증가 + OI 감소 -> 청산성 스퀴즈 (강해 보여도 소진되기 쉽다)
      돌파 + 거래량 감소          -> 가짜 돌파 경계

    OI는 캔들과 수집 상한이 달라(180일 vs 봉 수 기준) 커버리지가 어긋날 수
    있다. 없으면 해당 판정 항목만 빠지고 나머지는 그대로 동작한다.
    """

    oi = frames.get(OI_KEY)
    if oi is None or oi.empty:
        return pd.Series(np.nan, index=entry.index, dtype="float64")

    merged = entry[["ts"]].merge(
        oi[["ts", "open_interest"]], on="ts", how="left"
    )

    # **ffill에 한도를 둔다.**
    # 무제한으로 채우면 OI 데이터가 끝난 뒤의 봉이 전부 "직전 값"을 받아
    # 변화율이 0이 된다. 0은 "변화 없음"으로 읽혀 판정표를 조용히 통과/탈락하는데,
    # 실제로는 "값을 모른다"가 맞다. 한도를 넘으면 NaN이 되어 항목이 건너뛰어진다.
    values = merged["open_interest"].ffill(limit=OI_STALE_LIMIT)
    return (values.diff() / values.shift(1).where(lambda x: x > 0)).set_axis(entry.index)


def near(a: float, b: float, tolerance: float = LEVEL_TOLERANCE) -> bool:
    """두 가격이 같다고 볼 만큼 가까운가."""

    if a <= 0 or b <= 0:
        return False
    return abs(a - b) / max(a, b) <= tolerance


def line_at(p1: Pivot, p2: Pivot, index: int) -> float:
    """두 피벗을 잇는 직선의 index 시점 값. 목선·추세선 계산."""

    if p2.index == p1.index:
        return p1.price
    slope = (p2.price - p1.price) / (p2.index - p1.index)
    return p1.price + slope * (index - p1.index)


def slope_of(p1: Pivot, p2: Pivot) -> float:
    """봉당 가격 변화율. 수렴 판정에 쓴다."""

    if p2.index == p1.index or p1.price <= 0:
        return 0.0
    return ((p2.price - p1.price) / (p2.index - p1.index)) / p1.price


def profile_at(df: pd.DataFrame, as_of: int) -> vpvr.VolumeProfile | None:
    return vpvr.profile(df, as_of=as_of)
