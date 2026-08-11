"""전략 공통 인터페이스.

전략은 신호만 만든다. 주문을 내거나 포지션 크기를 정하지 않는다.
크기 산정은 risk/, 체결은 engine/ 담당.
이렇게 나눠야 전략을 추가해도 리스크 규칙이 한 곳에 남는다.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import pandas as pd

from config.settings import INDICATORS
from indicators import atr as atr_mod
from indicators import bollinger as bb
from indicators import ma
from indicators import oscillators as osc


Side = Literal["long", "short"]


@dataclass(frozen=True)
class PositionPolicy:
    """신호가 요구하는 비표준 사이징·청산 규칙.

    기본값은 모두 ``None``이라 기존 전략은 중앙 리스크 설정을 그대로 쓴다.
    고정 레버리지처럼 의도적으로 다른 실험만 명시적으로 덮어쓴다.
    """

    fixed_leverage: int | None = None
    margin_fraction: float | None = None
    min_liquidation_multiple: float | None = None

    stop_buffer_atr_mult: float | None = None
    partial_exit_ratio: float | None = None
    trail_atr_mult: float | None = None
    max_hold_bars: int | None = None
    # False면 시간 경과만으로 종가 청산하지 않고 손절/익절 주문을 유지한다.
    time_exit_enabled: bool = True
    breakeven_offset_pct: float = 0.0
    min_target_r: float | None = None

    # 시장가 신호가 다음 봉 시가에 체결되는 리플레이에서는 최초 신호가와
    # 실제 주문 가능 가격이 다르다. 고정 % 손절·목표 전략만 새 진입가에서
    # 다시 계산하고, 구조적 가격을 쓰는 기존 전략은 None으로 둔다.
    stop_pct_from_entry: float | None = None
    target_pct_from_entry: float | None = None

    symbol_risk_cap: float | None = None
    exclusive_symbol: bool = False

    # 전략 단위 과매매 방지. None이면 중앙 리스크 설정을 그대로 쓴다.
    max_daily_entries: int | None = None
    consecutive_loss_limit: int | None = None
    pause_until_next_day: bool = False
    # TP1이 미리 둔 지정가라면 메이커 수수료로 정산한다. 손절·시간청산은
    # 항상 시장가다.
    tp1_is_maker: bool = False


@dataclass
class Signal:
    """전략이 내놓는 진입 후보.

    실제 진입 여부는 risk 레이어가 결정한다.
    거절된 신호도 로그에 남기므로 이 객체는 항상 생성된다.
    """

    strategy: str           # 로그 태그 (예: "btc_fib", "xrp_patterns.flag")
    symbol: str
    side: Side
    entry: float

    # 손절은 전략이 구조(다음 피보나치 레벨, 박스 하단 등)에서 제시한다.
    # 최종 위치는 risk 레이어가 ATR 버퍼를 더해 확정한다.
    stop_hint: float

    # 1차 익절 후보. 보통 VPVR HVN 또는 스윙 고점
    target_hint: float | None = None

    # 진입 근거. 합류 존 점수, 어떤 요소가 겹쳤는지 등
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    # 판단 시점의 지표 스냅샷 (사후 분석용)
    context: dict = field(default_factory=dict)

    # 분할 진입 시 이번 신호가 몇 번째 트랜치인지
    tranche: int = 1
    max_tranches: int = 1

    # 주문 방식. XRP 단타에서는 이게 전략의 성패를 가른다.
    #   테이커 왕복 0.19% vs 메이커 왕복 -0.02%
    #   1.5R 목표 기준 손익분기 승률이 52% vs 39%로 갈린다.
    # "limit"이면 entry가 지정가이고, 가격이 그 값을 지나야 체결된다.
    # 체결되지 않으면 ttl_bars 뒤에 신호를 버린다 — 쫓아가면 리베이트가 사라져
    # 애초에 그 신호를 잡을 이유가 없어진다.
    order_type: Literal["limit", "market"] = "market"
    ttl_bars: int = 0

    # 이 신호에만 적용할 리스크 배수. 1.0이 기본(risk_per_trade 전액).
    #
    # 같은 전략 안에서도 자리에 따라 기대값이 크게 다르다 — BTC 회귀채널은
    # 추세 일치 진입이 건당 +25.5인데 횡보장 진입은 +0.37로 약 69배 차이였다.
    # 전략 단위 배수(RISK_MULTIPLIER)로는 이 구분을 표현할 수 없다.
    #
    # 1.0을 넘길 수 있지만 상한을 둔다. 표본이 적은 구간에 크게 걸면
    # 그 구간이 우연이었을 때 손실도 그만큼 커진다.
    risk_multiplier: float = 1.0

    # 기존 중앙 규칙과 다른 실험만 사용한다. 빈 정책은 기존 동작과 동일하다.
    position_policy: PositionPolicy = field(default_factory=PositionPolicy)

    def risk_distance(self) -> float:
        """진입가 대비 손절 거리 비율. 포지션 크기 산정의 입력."""
        return abs(self.entry - self.stop_hint) / self.entry


@dataclass(frozen=True)
class Prepared:
    """전 구간을 한 번 계산해 둔 지표. 전략은 여기서 꺼내 쓴다.

    리플레이는 봉마다 `generate()`를 부른다. 그 안에서 `rsi(df)` 같은 전 구간
    계산을 하면 O(n^2)가 된다. XRP 5분봉 51,835봉 기준으로 지표 4개면
    사전 계산 1.6초 vs 매번 재계산 23시간이고, 81조합 탐색에서는
    2분 vs 77일이 된다. 탐색이 되느냐 마느냐가 갈린다.

    `generate()`에 원본 프레임 대신 이 객체를 넘기는 이유는, 규율에 기대지
    않기 위해서다. 프레임을 그대로 넘기면 전략 안에서 지표를 다시 부르는
    쪽이 더 쓰기 쉽고, 그건 아무 데서도 걸리지 않는다.

    전략별 캐싱으로 해결하지 않는 것도 의도적이다. 실시간은 봉마다 프레임이
    자라고 리플레이는 고정이라, 한쪽에 맞춘 캐시가 다른 쪽에서는 갱신되지
    않은 값을 내놓는다. 느린 게 아니라 틀린 신호가 나오고 조용히 지나간다.
    """

    # as_of가 필요한 계산(피벗, VPVR, 합류)용 원본.
    # 뒤만 보는 지표를 여기서 다시 계산하지 않는다.
    frames: dict[str, pd.DataFrame]

    # 타임프레임 -> 지표 이름 -> 전 구간 Series
    series: dict[str, dict[str, pd.Series]]

    def frame(self, tf: str) -> pd.DataFrame:
        try:
            return self.frames[tf]
        except KeyError:
            raise KeyError(f"준비되지 않은 타임프레임: {tf!r}") from None

    def get(self, tf: str, name: str) -> pd.Series:
        try:
            return self.series[tf][name]
        except KeyError:
            available = sorted(self.series.get(tf, {}))
            raise KeyError(f"준비되지 않은 지표: {tf}/{name} (보유: {available})") from None

    def at(self, tf: str, name: str, as_of: int) -> Any:
        """as_of 시점의 값. 워밍업 구간이면 NaN이 그대로 나온다.

        float으로 변환하지 않는다. 국면(`regime`)은 문자열이고
        스퀴즈(`bb_squeeze`)는 bool이라 캐스팅하면 깨진다.
        """

        return self.get(tf, name).iloc[as_of]


def standard_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    """뒤만 보는 지표를 전 구간 계산한다.

    여기 넣을 수 있는 건 롤링 윈도가 과거만 참조하는 지표뿐이다.
    피벗·VPVR·합류는 `as_of` 시점에 계산해야 하므로 제외한다.
    전부 사전 계산으로 오해하면 `as_of` 규율이 무너진다.

    전략마다 필요한 것만 골라 쓰라고 나누지 않았다. 51,835봉 전체를
    계산해도 1.6초라, 나눠서 얻는 이득보다 "어느 전략이 뭘 준비했는지"를
    추적하는 비용이 크다.
    """

    m = osc.macd(df)
    b = bb.bollinger(df)

    return {
        "ma_fast": ma.sma(df, INDICATORS.ma_fast),
        "ma_slow": ma.sma(df, INDICATORS.ma_slow),
        "regime": ma.regime(df),
        "atr": atr_mod.atr(df),
        "atr_pct": atr_mod.atr_pct(df),
        "rsi": osc.rsi(df),
        "macd": m["macd"],
        "macd_signal": m["signal"],
        "macd_hist": m["hist"],
        "bb_upper": b["upper"],
        "bb_mid": b["mid"],
        "bb_lower": b["lower"],
        "bb_bandwidth": b["bandwidth"],
        "bb_squeeze": bb.squeeze(df),
        "bb_pos": bb.position(df),
    }


# 표준 지표를 계산하려면 있어야 하는 컬럼.
_OHLC = ("open", "high", "low", "close", "volume")


def is_candle_frame(df: pd.DataFrame) -> bool:
    """캔들 프레임인가.

    엔진은 캔들 말고도 보조 프레임을 넣는다 — 교차 종목 캔들, 미결제약정 등.
    OI에는 OHLC가 없으므로 표준 지표를 돌리면 예외가 난다.
    """

    return all(c in df.columns for c in _OHLC)


def prepare_standard(frames: dict[str, pd.DataFrame]) -> Prepared:
    """캔들 프레임에 표준 지표를 적용한 기본 구현.

    캔들이 아닌 프레임(미결제약정 등)은 건너뛴다. `Prepared.frames`에는
    그대로 남으므로 전략이 직접 꺼내 쓸 수 있다.

    전략이 추가 지표를 원하면 이걸 부른 뒤 series를 덧붙인다.
    """

    return Prepared(
        frames=frames,
        series={
            tf: standard_series(df)
            for tf, df in frames.items()
            if is_candle_frame(df)
        },
    )


class Strategy(Protocol):
    """모든 전략이 따르는 형태.

    prepare()는 전 구간 지표를 한 번 계산한다.
      실시간: 새 봉이 마감될 때마다 한 번
      리플레이: 시작할 때 한 번
    양쪽이 같은 계약을 쓰므로 결과가 갈리지 않는다.

    generate()는 인덱싱만 한다. as_of는 entry_tf의 위치 인덱스이고,
    확정 지연이 있는 계산(피벗 등)은 이 값을 넘겨 미래 참조를 막는다.
    """

    name: str

    def prepare(self, frames: dict[str, pd.DataFrame]) -> Prepared:
        ...

    def generate(self, prepared: Prepared, as_of: int) -> list[Signal]:
        ...
