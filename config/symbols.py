"""종목별 구성.

전체 계획이 여기 한 곳에 드러나야 한다.
어떤 종목에 어떤 전략이 붙고, 각 전략이 어떤 타임프레임을 보는지.
"""

from dataclasses import dataclass, field
from typing import Literal


Timeframe = Literal["5m", "15m", "1h", "4h", "1d"]

_TIMEFRAME_SECONDS: dict[Timeframe, int] = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


@dataclass(frozen=True)
class StrategySpec:
    """전략 하나의 구성."""

    name: str                       # 로그 태그. 성과 집계 단위가 된다
    entry_tf: Timeframe             # 진입 판정 캔들
    context_tf: Timeframe | None    # 국면 필터용 상위 캔들 (없으면 None)
    scaled_entry: bool              # 점진적(분할) 진입 여부
    description: str
    # 돌파를 한 번 더 확인할 캔들. 진입 판정은 entry_tf에서 하되,
    # 같은 이탈이 이 봉의 종가로도 성립하면 신뢰도가 올라간다.
    confirm_tf: Timeframe | None = None


@dataclass(frozen=True)
class SymbolSpec:
    contract: str
    strategies: tuple[StrategySpec, ...]
    # 이 종목에 배분할 계좌 비율
    capital_weight: float
    # 경제지표 발표 전후 신규 진입 차단 여부
    macro_blackout: bool = True
    required_timeframes: frozenset[Timeframe] = field(default_factory=frozenset)
    # 전략이 명시하지 않아도 항상 받아 둘 타임프레임.
    # 패턴별로 다른 봉을 쓰므로 종목 단위로 모아 둔다.
    extra_timeframes: tuple[Timeframe, ...] = ()

    def __post_init__(self):
        tfs = set()
        for s in self.strategies:
            tfs.add(s.entry_tf)
            if s.context_tf:
                tfs.add(s.context_tf)
            if s.confirm_tf:
                tfs.add(s.confirm_tf)
        tfs.update(self.extra_timeframes)
        object.__setattr__(self, "required_timeframes", frozenset(tfs))

    @property
    def driver_tf(self) -> Timeframe:
        """종목 엔진을 깨우는 가장 짧은 활성 진입봉."""

        return min(
            (strategy.entry_tf for strategy in self.strategies),
            key=_TIMEFRAME_SECONDS.__getitem__,
        )


# --------------------------------------------------------------------------
# BTC — 롤링 회귀채널 과확장 복귀
# --------------------------------------------------------------------------

BTC_REGRESSION_CHANNEL = StrategySpec(
    name="btc_regression_channel",
    entry_tf="1h",
    context_tf="4h",
    scaled_entry=False,
    description=(
        "100봉 롤링 회귀채널 방향 확인 → 반대편 채널 종가 이탈 → "
        "채널 밖 2% 지점 지정가. 채널선에서 절반 익절 후 트레일링."
    ),
)

BTC_REGIME_25X = StrategySpec(
    name="btc_regime_25x",
    entry_tf="5m",
    context_tf="15m",
    # 대세 국면 필터가 4시간봉을 본다.
    confirm_tf="4h",
    scaled_entry=False,
    description=(
        "15분봉 48봉 회귀채널의 방향·R² 확인 → 상승채널 하단 롱 / "
        "하락채널 상단 숏을 5분봉 지정가로 체결. 잔고 10%·25배, "
        "메이커 비용 차감 후 전체 계좌 +2% 전량 익절, 같은 가격폭 대칭 손절."
    ),
)


# 아래 두 명세는 과거 성과 비교 도구와 후보 전략이 참조하므로 보존한다.
# 실시간 build_strategies에는 등록되지 않는다.

BTC_TREND = StrategySpec(
    name="btc_trend",
    entry_tf="1h",
    context_tf="1d",        # MA50/200 국면 판단은 일봉에서
    scaled_entry=True,      # 점진적 매매
    description=(
        "일봉 MA50/200으로 국면 확인 → 1시간봉에서 MACD/볼린저/RSI 조합 판정. "
        "MA50/200은 눌림목 타점으로도 사용. "
        "1차 진입 후 확인이 더 쌓이면 추가 진입(분할)."
    ),
)

BTC_FIB = StrategySpec(
    name="btc_fib",
    entry_tf="1h",
    context_tf="1d",
    scaled_entry=False,     # 존 단위 단발 진입
    description=(
        "확정 피벗으로 스윙 후보 추출 → 피보나치 레벨 계산 → "
        "VPVR HVN / MA50 / MA200과 겹치는 합류 존에서 확인 캔들 후 진입."
    ),
)

BTC = SymbolSpec(
    contract="BTC_USDT",
    strategies=(BTC_REGRESSION_CHANNEL, BTC_REGIME_25X),
    capital_weight=0.5,
    # 이전 BTC 후보를 같은 데이터로 재검증하는 도구가 일봉을 사용한다.
    extra_timeframes=("1d",),
)


# --------------------------------------------------------------------------
# ③ XRP — 패턴 단타
# --------------------------------------------------------------------------
# 하나의 전략처럼 운용하지만 패턴별로 태그를 나눠 기록한다.
# 합쳐서 집계하면 어떤 패턴이 기여했는지 가려낼 수 없다.

# 고전 차트 패턴 5종. 전부 "완성 확인 후 진입" 원칙을 따른다.
#
# 공통 원칙 (strategies/xrp_patterns/chart_base.py가 강제)
#   - 패턴 완성 전에는 진입 신호로 보지 않는다
#   - 꼬리로 잠깐 넘는 것보다 **종가 돌파**를 우선한다
#   - 돌파 거래량(RVOL)·매물대(VPVR)·BTC 방향을 함께 확인한다
#   - 돌파 후 리테스트 성공을 가장 높은 품질로 본다
#   - 손절은 패턴 무효화 지점에 두고 포지션 크기를 역산한다
#   - 패턴 높이 목표가는 참고치이며 보장된 도달값이 아니다
#
# 전부 돌파 확인 후 진입이라 시장가다. 지정가로 미리 걸 수 있는 자리가
# 구조적으로 없다 — 완성 시점이 곧 진입 시점이기 때문이다.
XRP_PATTERN_NAMES = (
    "head_shoulders",     # 어깨-머리-어깨 + 목선 이탈 (역패턴 포함)
    "wedge",              # 상승 쐐기(약세) / 하락 쐐기(강세)
    "triangle",           # 대칭 삼각수렴, 방향 미리 정하지 않음
    "cup_handle",         # U자 회복 + 핸들 돌파 (역패턴 포함)
    "double_top",         # 이중천장/이중바닥, 목선 확인 필수
    "kotegawa",           # 이평선 이격도 역추세 (고테가와 타카시 착안)
)

# 성과 확인 후 검토.
XRP_PATTERN_BACKLOG = (
    "session_breakout",   # 아시아 세션 레인지를 런던/뉴욕 오픈에 돌파
    "failed_breakout",    # 레인지 이탈 후 즉시 복귀 (페이크아웃)
)

XRP_PATTERNS = StrategySpec(
    name="xrp_patterns",
    entry_tf="5m",          # 가장 짧은 진입봉. 패턴별 봉은 build_strategies가 정한다
    context_tf="4h",        # 상위 봉 패턴이 4시간봉을 컨텍스트로 쓴다
    confirm_tf="1h",
    scaled_entry=False,
    description=(
        "5분봉 패턴 단타. 패턴별 하위 태그로 성과를 분리 집계한다. "
        f"대상: {', '.join(XRP_PATTERN_NAMES)}. "
        "시간 손절 필수 — 자금 회전이 목적이므로 애매한 포지션을 오래 두지 않는다."
    ),
)

XRP = SymbolSpec(
    contract="XRP_USDT",
    strategies=(XRP_PATTERNS,),
    capital_weight=0.5,
    # 패턴마다 다른 봉을 쓴다. 전부 받아 둬야 어느 패턴이든 동작한다.
    extra_timeframes=("5m", "15m", "1h", "4h", "1d"),
)


# --------------------------------------------------------------------------

ACTIVE: tuple[SymbolSpec, ...] = (BTC, XRP)


def all_strategy_tags() -> list[str]:
    """로그 집계에 사용할 전체 태그 목록."""
    tags: list[str] = []
    for sym in ACTIVE:
        for st in sym.strategies:
            if st.name == "xrp_patterns":
                tags.extend(f"{st.name}.{p}" for p in XRP_PATTERN_NAMES)
            else:
                tags.append(st.name)
    return tags
