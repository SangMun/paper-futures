"""BTC 25배 단기 회귀채널 매매.

15분봉 48개로 약 12시간의 상승·하락 회귀채널을 만든다. 상승채널에서는
하단선에 롱 지정가, 하락채널에서는 상단선에 숏 지정가를 두고 5분봉으로
체결과 청산을 관리한다. 횡보 채널은 R²와 기간 전체 이동폭으로 제외한다.

진입과 목표는 post-only 지정가다. 목표가는 메이커 수수료를 차감하고도 전체
계좌가 2% 증가하도록 계산한다. 잔고 10%를 증거금으로 쓰므로 이는 증거금
순수익률 20%에 해당한다. 손절은 익절과 같은 가격폭의 반대편 시장가이며,
시간 경과만으로 종가 청산하지 않고 손절선 또는 목표선이 체결될 때까지 유지한다.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config.settings import BTC_REGIME_25X, FEES, BtcRegime25xConfig
from indicators import regression_channel as rc
from strategies.base import PositionPolicy, Prepared, Signal, prepare_standard
from strategies.btc_trend import context_index


NAME = "btc_regime_25x"
ENTRY_TF = "5m"
CHANNEL_TF = "15m"
MAJOR_TF = "4h"


def _major_regime(frames: dict[str, pd.DataFrame], entry: pd.DataFrame) -> pd.Series:
    """4시간봉 대세 국면을 5분봉 축에 정렬한다.

    마감된 상위봉만 본다 — 진행 중인 4시간봉 종가를 쓰면 미래참조다.
    """

    major = frames.get(MAJOR_TF)
    if major is None or major.empty:
        return pd.Series("none", index=entry.index, dtype="object")

    from indicators.ma import regime

    idx = context_index(
        entry["ts"].to_numpy(),
        major["ts"].to_numpy(),
        entry_tf=ENTRY_TF,
        context_tf=MAJOR_TF,
    )
    reg = regime(major, fast=20, slow=60).to_numpy()
    return pd.Series(
        [reg[i] if i >= 0 else "none" for i in idx], index=entry.index, dtype="object"
    )


def _maker_target_move(
    side: str,
    net_roe: float,
    leverage: int,
    maker_rate: float = FEES.maker_rate,
) -> float:
    """양쪽 메이커 수수료 뒤 목표 증거금 수익률을 만드는 가격 변동률."""

    if side not in ("long", "short"):
        raise ValueError(f"side는 long/short: {side!r}")
    if not math.isfinite(net_roe) or net_roe <= 0:
        raise ValueError(f"목표 ROE는 0보다 큰 유한값이어야 한다: {net_roe}")
    if not isinstance(leverage, int) or leverage <= 0:
        raise ValueError(f"레버리지는 양의 정수여야 한다: {leverage}")
    if not math.isfinite(maker_rate) or abs(maker_rate) >= 1:
        raise ValueError(f"메이커 수수료율 오류: {maker_rate}")

    target_per_notional = net_roe / leverage
    # 롱: d - m - m(1+d) = d(1-m)-2m
    # 숏: d - m - m(1-d) = d(1+m)-2m
    denominator = 1.0 - maker_rate if side == "long" else 1.0 + maker_rate
    move = (target_per_notional + 2.0 * maker_rate) / denominator
    if not math.isfinite(move) or move <= 0:
        raise ValueError(
            "메이커 리베이트가 목표 수익보다 커 목표 가격을 계산할 수 없다: "
            f"roe={net_roe}, leverage={leverage}, maker={maker_rate}"
        )
    return move


def _align_channel(
    entry: pd.DataFrame,
    context: pd.DataFrame,
    channel: pd.DataFrame,
) -> dict[str, pd.Series]:
    """마감된 15분 채널을 5분축으로 옮기고 현재 시각까지 직선 연장한다."""

    idx = context_index(
        entry["ts"].to_numpy(),
        context["ts"].to_numpy(),
        entry_tf=ENTRY_TF,
        context_tf=CHANNEL_TF,
    )
    entry_close_ts = entry["ts"].to_numpy(dtype="int64") + 300
    context_ts = context["ts"].to_numpy(dtype="int64")
    sources = {name: channel[name].to_numpy(dtype="float64") for name in channel}

    values = {
        name: np.full(len(entry), np.nan, dtype="float64")
        for name in ("mid", "upper", "lower", "slope", "r2", "deviation")
    }
    for at, ctx_at in enumerate(idx):
        if ctx_at < 0:
            continue
        slope = sources["slope"][ctx_at]
        if not math.isfinite(slope):
            continue
        elapsed = max(
            0.0,
            (entry_close_ts[at] - (context_ts[ctx_at] + 900)) / 900.0,
        )
        for name in ("mid", "upper", "lower"):
            values[name][at] = sources[name][ctx_at] + slope * elapsed
        for name in ("slope", "r2", "deviation"):
            values[name][at] = sources[name][ctx_at]

    return {name: pd.Series(value, index=entry.index) for name, value in values.items()}


class BtcRegime25x:
    """15분 단기 채널의 끝을 5분 지정가로 받는다."""

    name = NAME

    def __init__(
        self,
        symbol: str = "BTC_USDT",
        config: BtcRegime25xConfig = BTC_REGIME_25X,
        regime_filter: bool = True,
        allow_range: bool = False,
    ):
        self.symbol = symbol
        self.config = config
        # 15분 채널 기울기만으로는 대세를 못 본다.
        #
        # 949일 실측 국면별(4시간봉 MA20/60) 손익 — 필터 없이 1,260건 -377:
        #   상승 롱  303건 62.4% +1482.0   (앞 +665.3 / 뒤 +816.8)
        #   하락 숏  254건 57.9%  +687.8   (앞  +23.2 / 뒤 +664.6)
        #   상승 숏  101건 28.7% -1366.3   (앞 -665.1 / 뒤 -701.2)
        #   하락 롱  116건 28.4% -1595.6   (앞 -729.6 / 뒤 -866.0)
        #   횡보 롱  267건 53.2%   +46.4   (앞 +145.1 / 뒤  -98.7)
        #   횡보 숏  219건 52.5%   -42.6   (앞 +133.6 / 뒤 -176.3)
        #
        # 대세 역행 217건이 -2,962로 이익 구간을 통째로 삼킨다.
        self.regime_filter = bool(regime_filter)
        # 횡보장은 486건 합계 +3.8이고 앞뒤 절반의 부호가 갈린다.
        # 기대값 0에 낙폭만 얹으므로 기본은 제외한다.
        self.allow_range = bool(allow_range)
        # 보유 중 국면 전환 청산은 넣지 않았다.
        #
        # 처음엔 넣을 만해 보였다. 진입 필터만 건 637건 중 국면이 뒤집힌 채
        # 청산된 55건이 승률 18.2% / -1,145로 보였기 때문이다. 그런데 그
        # 라벨을 진행 중인 4시간봉으로 붙인 게 오류였다. 엔진은 마감된
        # 상위봉만 보고(context_index), 두 라벨은 27만 봉 중 12.3%에서
        # 갈리며 그 불일치가 전부 국면 전환 구간에 몰려 있다.
        #
        # 미래참조를 걷어내고 실제로 훅을 달아 돌리니 949일에 1건 발동,
        # 손익은 +45.14 -> +20.01이었다. 4시간봉 MA20/60 국면은 며칠씩
        # 유지되므로 중앙 보유 105분 안에 뒤집히지 않는다 — 구조적으로
        # 발동할 수 없는 규칙이다.

    def prepare(self, frames: dict[str, pd.DataFrame]) -> Prepared:
        prepared = prepare_standard(frames)
        entry = frames[ENTRY_TF]
        context = frames[CHANNEL_TF]
        cfg = self.config
        channel = rc.channel(
            context,
            length=cfg.channel_length,
            width_mult=cfg.channel_width_mult,
        )
        aligned = _align_channel(entry, context, channel)
        mid = aligned["mid"]
        move_pct = (
            aligned["slope"].abs() * (cfg.channel_length - 1)
            / mid.where(mid > 0)
        )
        prepared.series[ENTRY_TF].update(
            {
                **{f"channel_{name}": series for name, series in aligned.items()},
                "channel_move_pct": move_pct,
                "major_regime": _major_regime(frames, entry),
            }
        )
        return prepared

    def _policy(self, target_move: float) -> PositionPolicy:
        cfg = self.config
        return PositionPolicy(
            fixed_leverage=cfg.leverage,
            margin_fraction=cfg.margin_fraction,
            min_liquidation_multiple=cfg.min_liquidation_multiple,
            stop_buffer_atr_mult=0.0,
            partial_exit_ratio=cfg.partial_exit_ratio,
            trail_atr_mult=cfg.trail_atr_mult,
            max_hold_bars=cfg.max_hold_bars,
            time_exit_enabled=cfg.time_exit_enabled,
            breakeven_offset_pct=cfg.breakeven_offset_pct,
            min_target_r=cfg.min_target_r,
            stop_pct_from_entry=target_move,
            target_pct_from_entry=target_move,
            symbol_risk_cap=cfg.symbol_risk_cap,
            exclusive_symbol=True,
            max_daily_entries=cfg.max_daily_entries,
            consecutive_loss_limit=cfg.consecutive_loss_limit,
            pause_until_next_day=True,
            tp1_is_maker=True,
        )

    def generate(self, prepared: Prepared, as_of: int) -> list[Signal]:
        df = prepared.frame(ENTRY_TF)
        if as_of < 1 or as_of >= len(df):
            return []

        names = (
            "atr",
            "channel_mid",
            "channel_upper",
            "channel_lower",
            "channel_slope",
            "channel_r2",
            "channel_deviation",
            "channel_move_pct",
        )
        values = {name: prepared.at(ENTRY_TF, name, as_of) for name in names}
        if any(pd.isna(value) for value in values.values()):
            return []

        cfg = self.config
        close = float(df["close"].iloc[as_of])
        atr = float(values["atr"])
        mid = float(values["channel_mid"])
        upper = float(values["channel_upper"])
        lower = float(values["channel_lower"])
        slope = float(values["channel_slope"])
        r2 = float(values["channel_r2"])
        deviation = float(values["channel_deviation"])
        move_pct = float(values["channel_move_pct"])

        if (
            close <= 0
            or atr <= 0
            or not 0 < lower < mid < upper
            or deviation <= 0
            or r2 < cfg.channel_r2_min
            or move_pct < cfg.channel_min_move_pct
        ):
            return []

        if slope > 0:
            side = "long"
            entry = lower
            if not entry < close <= upper:
                return []
            entry_distance = (close - entry) / close
        elif slope < 0:
            side = "short"
            entry = upper
            if not lower <= close < entry:
                return []
            entry_distance = (entry - close) / close
        else:
            return []

        if entry_distance > cfg.max_entry_distance_pct:
            return []

        # 상위봉이 없으면(4시간봉 미제공) 필터를 통과시킨다.
        try:
            regime = prepared.at(ENTRY_TF, "major_regime", as_of)
        except KeyError:
            regime = "none"
        aligned = (regime == "bull" and side == "long") or (
            regime == "bear" and side == "short"
        )
        if self.regime_filter and not aligned:
            is_range = regime not in ("bull", "bear")
            if not (is_range and self.allow_range):
                return []

        target_margin_roe = (
            cfg.account_take_profit_pct / cfg.margin_fraction
        )
        target_move = _maker_target_move(
            side,
            target_margin_roe,
            cfg.leverage,
        )
        direction = 1.0 if side == "long" else -1.0
        stop = entry * (1.0 - direction * target_move)
        target = entry * (1.0 + direction * target_move)

        return [
            Signal(
                strategy=NAME,
                symbol=self.symbol,
                side=side,
                entry=entry,
                stop_hint=stop,
                target_hint=target,
                score=2.0 + r2,
                reasons=[
                    "rising_15m_channel" if side == "long" else "falling_15m_channel",
                    "lower_channel_limit" if side == "long" else "upper_channel_limit",
                    "account_profit_2pct",
                    *(["trend_aligned"] if aligned else []),
                ],
                context={
                    "atr": atr,
                    "channel_mid": mid,
                    "channel_upper": upper,
                    "channel_lower": lower,
                    "channel_slope": slope,
                    "channel_r2": r2,
                    "channel_move_pct": move_pct,
                    "entry_distance_pct": entry_distance,
                    "target_move_pct": target_move,
                    "target_account_profit_pct": cfg.account_take_profit_pct,
                    "target_margin_roe": target_margin_roe,
                    "leverage": cfg.leverage,
                    "margin_fraction": cfg.margin_fraction,
                    "major_regime": regime,
                    "trend_aligned": aligned,
                },
                tranche=1,
                max_tranches=1,
                order_type="limit",
                ttl_bars=cfg.limit_ttl_bars,
                position_policy=self._policy(target_move),
            )
        ]
