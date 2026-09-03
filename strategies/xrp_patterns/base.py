"""XRP 패턴 단타 공통 기반.

패턴별로 태그를 나눠 기록하되(`xrp_patterns.liquidity_sweep` 등), 필터와
준비 과정은 공유한다. 패턴마다 필터를 따로 구현하면 어떤 신호가 어떤
이유로 걸러졌는지 비교할 수 없게 된다.

세 가지 필터가 모든 패턴에 공통으로 걸린다.
  1. 상위봉 방향 — 1시간봉 국면과 반대 방향은 잡지 않는다
  2. 변동성 서킷브레이커 — 뉴스 스파이크 구간에서는 패턴이 성립하지 않는다
  3. BTC 컨텍스트 — XRP는 BTC 베타다. BTC가 급락 중이면 XRP 롱은 무의미하다
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import SCALP
from indicators import atr as atr_mod
from strategies.base import Prepared, prepare_standard


@dataclass(frozen=True)
class Timeframes:
    """패턴 하나가 쓰는 타임프레임 묶음.

    **패턴마다 다르다.** 949일 실측에서 같은 패턴도 봉에 따라 결과가
    뒤집혔다 — double_top은 5분봉 -0.206%, 1시간봉 +0.396%.
    모듈 상수로 묶어 두면 전 패턴이 같은 봉을 강제로 공유한다.
    """

    entry: str          # 패턴 탐지·진입 판정
    context: str        # 방향 필터
    confirm: str        # 돌파 재확인 (종가)
    htf: str            # 상위 지지저항


# 진입봉마다 어떤 상위봉을 쓸지. 한 칸씩 올라간다.
LADDER: dict[str, Timeframes] = {
    "5m":  Timeframes("5m", "1h", "15m", "1h"),
    "15m": Timeframes("15m", "4h", "1h", "4h"),
    "1h":  Timeframes("1h", "4h", "4h", "4h"),
    "4h":  Timeframes("4h", "1d", "1d", "1d"),
}

DEFAULT_TF = LADDER["5m"]

# 하위 호환. 새 코드는 Timeframes를 쓴다.
ENTRY_TF = "5m"
CONTEXT_TF = "1h"

# 엔진이 넣어 주는 교차 종목 프레임의 키.
# 타임프레임 키("5m")와 섞이지 않도록 종목명을 앞에 붙인다.
BTC_CONTEXT_KEY = "BTC_USDT:1h"

FAMILY = "xrp_patterns"


def spike_flag(df: pd.DataFrame, cfg=SCALP) -> pd.Series:
    """변동성 서킷브레이커. True면 신규 진입 금지.

    한 봉의 고저폭이 평소(직전 lookback봉 중앙값)의 spike_range_mult배를
    넘으면, 그 봉부터 cooldown_bars 동안 막는다.

    중앙값을 쓰는 이유는 평균이 스파이크 자신에게 끌려가기 때문이다.
    직전 구간만 보므로 미래를 참조하지 않는다.
    """

    rng = (df["high"] - df["low"]) / df["close"]
    baseline = rng.rolling(cfg.spike_lookback, min_periods=cfg.spike_lookback).median()
    spiked = (rng > baseline * cfg.spike_range_mult).fillna(False).to_numpy()

    # 스파이크 발생 봉부터 cooldown 동안 True를 유지한다.
    out = np.zeros(len(df), dtype=bool)
    remaining = 0
    for i, hit in enumerate(spiked):
        if hit:
            remaining = cfg.spike_cooldown_bars
        if remaining > 0:
            out[i] = True
            remaining -= 1
    return pd.Series(out, index=df.index)


def context_bias(
    frames: dict[str, pd.DataFrame],
    entry_index: pd.Index,
    tf: "Timeframes" = None,
) -> pd.Series:
    """1시간봉 방향을 5분봉 축에 정렬한다. 'bull' / 'bear' / 'none'.

    btc_trend와 같은 규칙으로 마감된 상위봉만 본다. 여기서 미래를 보면
    5분봉 단타는 봉이 많아 오차가 훨씬 크게 누적된다.
    """

    from strategies.btc_trend import context_index

    tf = tf or DEFAULT_TF
    entry = frames[tf.entry]
    context = frames.get(tf.context)
    if context is None or context.empty:
        return pd.Series("none", index=entry_index, dtype="object")

    idx = context_index(
        entry["ts"].to_numpy(),
        context["ts"].to_numpy(),
        entry_tf=tf.entry,
        context_tf=tf.context,
    )

    from indicators.ma import regime

    ctx_regime = regime(context, fast=20, slow=50).to_numpy()
    values = [ctx_regime[i] if i >= 0 else "none" for i in idx]
    return pd.Series(values, index=entry_index)


def btc_adverse(
    frames: dict[str, pd.DataFrame],
    entry_index: pd.Index,
    cfg=SCALP,
    tf: "Timeframes" = None,
) -> pd.Series:
    """BTC 1시간봉이 크게 역방향으로 움직였는지. 'down' / 'up' / 'none'.

    BTC 프레임이 없으면 전부 'none' — 필터가 없는 상태로 동작한다.
    교차 종목 데이터가 없다고 XRP 매매를 멈출 이유는 없다.
    """

    btc = frames.get(BTC_CONTEXT_KEY)
    if btc is None or btc.empty:
        return pd.Series("none", index=entry_index, dtype="object")

    from strategies.btc_trend import context_index

    tf = tf or DEFAULT_TF
    entry = frames[tf.entry]
    idx = context_index(
        entry["ts"].to_numpy(),
        btc["ts"].to_numpy(),
        entry_tf=tf.entry,
        context_tf="1h",            # BTC 컨텍스트는 항상 1시간봉
    )

    change = ((btc["close"] - btc["open"]) / btc["open"]).to_numpy()
    out = []
    for i in idx:
        if i < 0:
            out.append("none")
        elif change[i] <= -cfg.btc_adverse_move_pct:
            out.append("down")
        elif change[i] >= cfg.btc_adverse_move_pct:
            out.append("up")
        else:
            out.append("none")
    return pd.Series(out, index=entry_index)


MAJOR_TF = "4h"


def major_regime(frames: dict[str, pd.DataFrame], entry_index: pd.Index, tf) -> pd.Series:
    """4시간봉 대세 국면. 'bull' / 'bear' / 'none'.

    `context_bias`(진입봉 한 칸 위)보다 훨씬 큰 흐름을 본다. 국면별 손익을
    쪼개 보니 손실이 **대세와 반대 방향**에 몰려 있었다 —
    wedge.5m은 상승장 숏만 -94.5, head_shoulders.5m은 상승장 숏 -38.0 +
    하락장 롱 -62.4로 손실 전부가 여기서 나왔다.

    한 칸 위 봉만으로는 이 구분이 안 된다. 5분봉 전략의 컨텍스트는
    1시간봉인데, 1시간봉이 반등 중이어도 4시간봉은 하락일 수 있다.
    """

    major = frames.get(MAJOR_TF)
    if major is None or major.empty:
        return pd.Series("none", index=entry_index, dtype="object")

    from indicators.ma import regime
    from strategies.btc_trend import context_index

    entry = frames[tf.entry]
    idx = context_index(
        entry["ts"].to_numpy(),
        major["ts"].to_numpy(),
        entry_tf=tf.entry,
        context_tf=MAJOR_TF,
    )
    reg = regime(major, fast=20, slow=60).to_numpy()
    return pd.Series(
        [reg[i] if i >= 0 else "none" for i in idx], index=entry_index, dtype="object"
    )


def major_gap(frames: dict[str, pd.DataFrame], entry_index: pd.Index, tf) -> pd.Series:
    """4시간봉 두 이평선 사이 간격(가격 대비). 국면 'none'을 쪼개는 데 쓴다.

    `major_regime`의 'none'은 측정된 횡보가 아니라 **정배열도 역배열도 아닌
    나머지 전부**다. 실측(XRP 4시간봉 5,707개)에서 none의 이평선 간격
    중앙값이 2.12%로 bull(2.36%)·bear(2.35%)와 거의 같았고, none의 44.9%는
    두 선이 추세 수준으로 벌어진 상태였다 — 추세는 살아 있는데 가격만
    두 선 사이에 잠깐 낀 구간이다. 되돌림·평균 회귀가 노리는 자리가 바로
    거기라, none을 통째로 막으면 노려야 할 자리까지 같이 막힌다.

    이 값이 작을 때만(두 선이 붙어 있을 때) 진짜 방향 없음으로 본다.
    """

    major = frames.get(MAJOR_TF)
    if major is None or major.empty:
        return pd.Series(float("nan"), index=entry_index, dtype="float64")

    from indicators.ma import sma
    from strategies.btc_trend import context_index

    entry = frames[tf.entry]
    idx = context_index(
        entry["ts"].to_numpy(),
        major["ts"].to_numpy(),
        entry_tf=tf.entry,
        context_tf=MAJOR_TF,
    )
    fast = sma(major, 20).to_numpy()
    slow = sma(major, 60).to_numpy()
    price = major["close"].to_numpy()
    gap = abs(fast - slow) / price
    return pd.Series(
        [gap[i] if i >= 0 else float("nan") for i in idx],
        index=entry_index, dtype="float64",
    )


def counter_trend(prepared: Prepared, as_of: int, side: str, tf) -> bool:
    """대세와 반대 방향인가.

    횡보('none')는 반대로 보지 않는다 — 실측에서 횡보 구간은 양방향 모두
    플러스였다(wedge.5m 횡보 롱 +57.8 / 숏 +54.7).
    """

    reg = prepared.at(tf.entry, "major_regime", as_of)
    return (side == "short" and reg == "bull") or (side == "long" and reg == "bear")


def scalp_prepare(frames: dict[str, pd.DataFrame], tf: "Timeframes" = None) -> Prepared:
    """표준 지표 + 단타 전용 필터 시리즈.

    필터를 전부 미리 계산해 두면 generate()는 인덱싱만 하면 된다.
    5분봉은 51,835봉이라 봉마다 다시 계산하면 리플레이가 성립하지 않는다.
    """

    tf = tf or DEFAULT_TF
    prepared = prepare_standard(frames)
    entry = frames[tf.entry]

    prepared.series[tf.entry].update(
        {
            "spike": spike_flag(entry),
            "ctx_bias": context_bias(frames, entry.index, tf),
            "btc_adverse": btc_adverse(frames, entry.index, tf=tf),
            "major_regime": major_regime(frames, entry.index, tf),
            "major_gap": major_gap(frames, entry.index, tf),
        }
    )
    return prepared


def blocked(prepared: Prepared, as_of: int, side: str, tf: "Timeframes" = None) -> str | None:
    """공통 필터에 걸리면 사유를, 아니면 None.

    사유를 문자열로 돌려주는 이유는 로그에 남겨 필터별로 몇 건이 막혔는지
    집계하기 위해서다. 막은 게 실은 좋은 신호였는지 나중에 판단해야 한다.
    """

    tf = tf or DEFAULT_TF
    if bool(prepared.at(tf.entry, "spike", as_of)):
        return "volatility_spike"

    bias = prepared.at(tf.entry, "ctx_bias", as_of)
    if side == "long" and bias == "bear":
        return "context_bearish"
    if side == "short" and bias == "bull":
        return "context_bullish"

    adverse = prepared.at(tf.entry, "btc_adverse", as_of)
    if side == "long" and adverse == "down":
        return "btc_dumping"
    if side == "short" and adverse == "up":
        return "btc_pumping"

    return None


def episode_trigger(armed: pd.Series, fires: pd.Series) -> pd.Series:
    """한 에피소드에 한 번만 True.

    볼린저 스퀴즈는 2,225봉이 조건을 만족하지만 연속 구간으로 묶으면
    245개뿐이다. 봉 단위로 세면 기회를 9.1배 과대계상한다.

    armed 구간이 시작되면 무장하고, fires가 처음 True가 될 때 한 번 발사한 뒤
    소비된다. armed가 끊겼다 다시 시작되면 재무장한다.

    뒤를 보지 않는 단방향 스캔이다. prepare()에서 한 번만 돌리므로
    generate()는 인덱싱만 하면 된다.
    """

    armed_arr = armed.fillna(False).to_numpy()
    fire_arr = fires.fillna(False).to_numpy()

    out = np.zeros(len(armed_arr), dtype=bool)
    was_armed = False
    spent = False

    for i in range(len(armed_arr)):
        if armed_arr[i]:
            if not was_armed:       # 새 에피소드 시작 -> 재무장
                spent = False
            if not spent and fire_arr[i]:
                out[i] = True
                spent = True
        was_armed = armed_arr[i]

    return pd.Series(out, index=armed.index)


def vpvr_target(df: pd.DataFrame, as_of: int, entry: float, side: str) -> float | None:
    """1차 목표: 진입가 위쪽(롱)/아래쪽(숏) 최근접 VPVR HVN.

    BTC와 같은 규칙이다. 처음엔 "5분봉 VPVR은 노이즈"라며 고정 R 배수를
    썼는데 근거 없는 판단이었다 — 실측하니 HVN 중앙값 3개, 목표까지
    중앙값 2.43R로 고정값보다 나았다.

    HVN을 못 찾으면 None. risk 레이어가 R 배수로 떨어진다.
    """

    from indicators import vpvr

    profile = vpvr.profile(df, as_of=as_of)
    if profile is None:
        return None

    node = profile.nearest_above(entry) if side == "long" else profile.nearest_below(entry)
    return float(node) if node is not None else None


def atr_at(prepared: Prepared, as_of: int, tf: "Timeframes" = None) -> float:
    value = prepared.at((tf or DEFAULT_TF).entry, "atr", as_of)
    return float(value) if pd.notna(value) else 0.0
