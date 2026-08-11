"""파라미터 중앙 관리.

여기 없는 상수를 모듈에 하드코딩하지 않는다.
포워드 테스트 결과로 조정할 값과 고정할 값을 구분해 둔다.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------- 거래소

BASE_URL = "https://api.gateio.ws/api/v4"

# 종목별 계약 코드
CONTRACT_BTC = "BTC_USDT"
CONTRACT_XRP = "XRP_USDT"


@dataclass(frozen=True)
class FeeAssumption:
    """수수료·슬리피지 가정.

    2026-08-04 Gate.io API에서 확인한 기본 등급 값.
    VIP 등급에 따라 낮아지므로 실계좌 값으로 갱신할 것.

    메이커가 음수(리베이트)라는 점이 중요하다.
      테이커 왕복 = 0.075% * 2 + 슬리피지 0.02% * 2 = 약 0.19%
      메이커 왕복 = -0.01% * 2 = -0.02% (오히려 받음)
    왕복 기준 0.21%p 차이. XRP 5분봉 단타는 목표 수익이 0.5% 안팎이라
    지정가 체결 비중이 성과를 좌우한다.
    """

    taker_rate: float = 0.00075
    maker_rate: float = -0.0001     # 리베이트
    slippage_rate: float = 0.0002

    def round_trip_taker(self) -> float:
        return (self.taker_rate + self.slippage_rate) * 2

    def round_trip_maker(self) -> float:
        return self.maker_rate * 2


FEES = FeeAssumption()


# 계약 스펙. 2026-08-05 /futures/usdt/contracts 응답으로 확인.
#
# maintenance_rate는 청산가 계산에 들어간다. 이 값이 없으면 청산 거리를
# 1/레버리지로 잡게 되는데, 실제 청산은 그보다 먼저 발생한다.
#
# 주의: 유지증거금률은 포지션이 커지면 리스크 리밋 구간에 따라 올라간다.
# 아래 값은 최저 구간 기준이고, 잔고 1,000 규모에서는 이 구간을 벗어나지 않는다.
# 자금을 키우면 다시 확인할 것.
#
# price_tick은 호가 단위. 계산된 손절·익절가를 주문 가능한 값으로 맞출 때 쓴다.
CONTRACT_SPECS = {
    "BTC_USDT": {
        "multiplier": 0.0001,
        "min_size": 1,
        "max_leverage": 200,
        "maintenance_rate": 0.003,
        "price_tick": 0.1,
    },
    "XRP_USDT": {
        "multiplier": 10.0,
        "min_size": 1,          # API는 order_size_min=0이지만 1계약이 실질 최소
        "max_leverage": 100,
        "maintenance_rate": 0.005,
        "price_tick": 0.0001,
    },
}


# ---------------------------------------------------------------- 리스크

@dataclass(frozen=True)
class RiskConfig:
    """레버리지는 여기서 정하지 않는다. 손절 거리에서 역산된다."""

    # 트레이드 1회당 감수할 계좌 대비 손실 비율
    risk_per_trade: float = 0.01

    # 동시 오픈 포지션의 리스크 합계 상한
    # 테스트 단계에서는 크게 두고 데이터를 모은다. 실전 전환 시 축소.
    max_total_risk: float = 0.10

    # 한 종목에 몰아넣을 수 있는 리스크 상한.
    # BTC 추세 롱과 BTC 피보 롱이 동시에 잡히면 사실상 하나의 베팅이다.
    max_symbol_risk: float = 0.03

    # 상관 리스크 모형을 도입할 때 쓸 후보값. 현재 한도 계산에는 적용하지 않고
    # 방향과 무관하게 손절 리스크를 전액 합산한다. 근거 없는 할인은 같은 방향
    # 포지션을 더 많이 허용하는 역효과가 있기 때문이다.
    # 포워드 테스트에서 동시 보유 구간의 실제 동조율을 확인한 뒤 활성화할 것.
    correlated_risk_weight: float = 0.7

    # 청산가는 손절가보다 이 배수 이상 멀어야 한다.
    # 손절이 넓을 때만 유효한 안전장치다. XRP처럼 손절이 0.5%면
    # 이 규칙만으로 57배까지 통과하므로 아래 LEVERAGE_CAP이 실질 제약이 된다.
    min_liquidation_multiple: float = 3.0

    # 손절 거리가 이보다 크면 진입 포기 (손익비 확보 불가)
    max_stop_pct: float = 0.05

    # 손절 거리 하한 = 왕복 테이커 비용 * 이 배수.
    # 비율을 직접 박지 않고 수수료에서 유도하는 이유는, 등급이 바뀌어
    # 비용이 내려가면 하한도 같이 내려가야 하기 때문이다.
    # 기본값 2.0 -> 0.19% * 2 = 약 0.38%.
    # 이보다 좁은 손절은 이겨도 수수료가 수익의 절반을 가져간다.
    min_stop_cost_multiple: float = 2.0

    # 계좌 단위 안전장치
    daily_loss_limit: float = 0.05      # 하루 손실 5% 도달 시 당일 매매 중단
    consecutive_loss_limit: int = 5     # 연속 손절 5회 시 일시 정지
    # 정지 지속 시간. **이 값이 없으면 정지가 영구가 된다** —
    # 카운터는 이익 거래로만 초기화되는데, 진입이 막혀 있으면 이익 거래가
    # 나올 수 없기 때문이다. 실측으로 리플레이가 5,000봉에서 멈췄다.
    pause_hours: float = 12.0

    # 손절선 ATR 버퍼 (휩쏘 방지)
    atr_buffer_mult: float = 0.5

    # 트레일링 스탑 폭
    trail_atr_mult: float = 1.5

    # 1차 부분 익절 비율
    partial_exit_ratio: float = 0.5

    # 1차 목표는 손절 거리의 최소 이 배수여야 한다.
    # VPVR HVN이 진입가 바로 위에 있으면 목표가 손절보다 가까워지는데,
    # 그러면 이겨도 잃는 것보다 적게 번다. 실측으로 XRP 스윕 신호의 22%가
    # 여기 해당했다. 1.0은 "리스크만큼은 벌 수 있어야 한다"는 선이다.
    # 7단계 탐색 대상.
    min_target_r: float = 1.0

    def min_stop_pct(self, fees: FeeAssumption | None = None) -> float:
        """수수료에서 유도한 손절 거리 하한."""
        return (fees or FEES).round_trip_taker() * self.min_stop_cost_multiple


RISK = RiskConfig()


# 측정용 설정. **계좌 단위 브레이크를 끈다.**
#
# 일일 손실 한도와 연속 손절 정지는 계좌 전체에 걸리므로, 켜 둔 채로
# 전략별 성과를 재면 서로를 오염시킨다 — 실측으로 kotegawa를 빼자
# double_top 손실이 -221에서 -270으로 늘었다. 정지가 우연히 브레이크
# 역할을 하고 있었던 것이고, 그 상태의 숫자는 그 전략의 성과가 아니다.
#
# 먼저 브레이크 없이 각 전략의 맨얼굴을 보고, 손절·레버리지를 조정한 뒤,
# 마지막에 일일 손실 한도만 다시 켠다.
MEASUREMENT_RISK = RiskConfig(
    daily_loss_limit=1.0,            # 사실상 무제한
    consecutive_loss_limit=10**9,    # 사실상 무제한
)


# 분할 진입 시 트랜치별 리스크 배분.
# 합이 1.0이어야 한다. 트랜치마다 risk_per_trade를 새로 쓰면
# 3분할이 곧 3배 리스크가 되므로, 총량을 먼저 쪼개고 시작한다.
TRANCHE_RISK_WEIGHTS: tuple[float, ...] = (0.4, 0.3, 0.3)


# 전략별 레버리지 상한.
# 레버리지는 손절 거리에서 역산된 결과값이지만, 손절이 좁으면 역산값이
# 무제한으로 커진다. 계좌가 작을 때만 걸리는 안전장치다.
#   잔고 1,000 기준 필요 레버리지: BTC 손절 2.5% -> 1x, XRP 손절 0.5% -> 2x
#   캡은 그 2~3배 여유로 잡았다.
# 세부 태그가 있으면 그쪽이 우선, 없으면 전략군 값을 쓴다.
#
# 현재 설정에서는 어떤 캡도 발동하지 않는다. 필요 레버리지의 최대값이
# risk_per_trade / 실효 손절하한 = 약 1.75x라 캡(2~5x) 아래이기 때문이다.
# 리스크 비율을 올리거나 손절 하한을 낮췄을 때를 대비한 안전망이다.
LEVERAGE_CAP: dict[str, float] = {
    "btc_trend": 3.0,
    "btc_fib": 3.0,
    "btc_regression_channel": 3.0,
    "btc_regime_25x": 25.0,
    # 아래 둘은 아직 실시간 등록 전인 BTC 후보 전략이다. 백테스트 도구에서
    # 엔진의 실제 사이징 경로를 그대로 타기 위해서도 명시적인 캡이 필요하다.
    "btc_pullback_reversal": 3.0,
    "btc_trend_continuation": 3.0,
    "xrp_patterns": 5.0,

    "xrp_patterns.triangle": 2.0,
    "xrp_patterns.wedge": 2.0,
    "xrp_patterns.head_shoulders": 2.0,
}


# 전략별 리스크 배수. 1.0이 기본(risk_per_trade 전액).
#
# **"삼각형·쐐기·오른쪽 어깨에서는 과도한 레버리지 금지"를 실제로 강제하는 건
# 이쪽이다.** 레버리지 캡은 위 이유로 발동하지 않는다. 포지션 크기를 줄이면
# 명목과 레버리지가 함께 줄어 의도한 효과가 난다.
#
# 수렴 패턴은 완성 확인 후 진입해도 되돌아 들어오는 가짜 돌파가 구조적으로
# 잦다. 헤드앤숄더는 오른쪽 어깨가 머리를 다시 넘는 실패가 흔하다.
RISK_MULTIPLIER: dict[str, float] = {
    "xrp_patterns.triangle": 0.5,
    "xrp_patterns.wedge": 0.5,
    "xrp_patterns.head_shoulders": 0.5,
}


# ---------------------------------------------------------------- 지표 공통

@dataclass(frozen=True)
class IndicatorConfig:
    atr_period: int = 14

    rsi_period: int = 14
    # 추세장에서는 RSI 밴드가 이동한다.
    # 상승추세 눌림목은 보통 40 부근에서 바닥을 잡으므로
    # 교과서값 30 미만을 쓰면 신호가 거의 발생하지 않는다.
    rsi_pullback_min: float = 35.0
    rsi_pullback_max: float = 50.0
    rsi_overbought: float = 70.0

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    bb_period: int = 20
    bb_std: float = 2.0
    # 밴드폭이 최근 lookback 구간의 하위 이 백분위 이하면 스퀴즈로 본다
    bb_squeeze_percentile: float = 20.0
    bb_squeeze_lookback: int = 100

    # 국면 필터용 (일봉 기준)
    ma_fast: int = 50
    ma_slow: int = 200


INDICATORS = IndicatorConfig()


# ---------------------------------------------------------------- BTC 후보 전략

@dataclass(frozen=True)
class BtcCandidateConfig:
    """실시간 활성화 전 비교할 BTC 진입 방식.

    현재 btc_trend/btc_fib을 조용히 바꾸지 않고 독립 태그로 측정한다. 같은
    데이터에서 파라미터를 많이 돌리면 과적합되므로, 시장 구조상 의미가 분명한
    세 값만 둔다.
    """

    # 엄격 반전형: 레벨 거부 뒤 이 시간 안에 확인봉 고/저점을 돌파해야 한다.
    pullback_breakout_ttl_bars: int = 2
    # 돌파 뒤 첫 리테스트를 기다리는 최대 시간. 1시간봉 기준 3시간이다.
    pullback_retest_ttl_bars: int = 3

    # 추세 지속형: 직전 20시간 고/저점을 종가로 벗어나야 한다.
    continuation_lookback_bars: int = 20
    # 돌파선에 둔 지정가가 이 시간 안에 리테스트되지 않으면 취소한다.
    continuation_limit_ttl_bars: int = 3


BTC_CANDIDATES = BtcCandidateConfig()


@dataclass(frozen=True)
class RegressionChannelConfig:
    """BTC 롤링 회귀채널 전략의 고정 비교값.

    TradingView 원본의 기본 회귀 길이·표준편차 배수를 유지한다. 진입 이격은
    1%와 2%를 별도로 측정할 수 있게 전략 생성자에서만 덮어쓴다.
    """

    length: int = 100
    width_mult: float = 2.0
    entry_offset_pct: float = 0.02
    limit_ttl_bars: int = 12

    # stop_hint에서 0.5 ATR, 공통 exit plan에서 다시 0.5 ATR을 더해
    # 최종 손절은 진입가 바깥 약 1 ATR에 놓인다.
    stop_hint_atr_mult: float = 0.5


REGRESSION_CHANNEL = RegressionChannelConfig()


# ---------------------------------------------------------------- BTC 25x 국면 추세

@dataclass(frozen=True)
class BtcRegime25xConfig:
    """15분 회귀채널·5분 체결의 BTC 25배 단타 전략."""

    # 약 12시간의 15분봉으로 현재 단기 흐름을 잡는다.
    channel_length: int = 48
    channel_width_mult: float = 2.0
    # 채널선을 지지·저항으로 쓰려면 채널 자체가 뚜렷해야 한다. R²가 낮으면
    # 그 선은 잡음에 그은 직선이고, 지정가가 "선을 뚫고 계속 가는" 순간에
    # 체결된다(역선택). 실측: 30분 안에 끝난 104건이 승률 40.4% / -846.6이고
    # 앞뒤 절반 모두 마이너스였다.
    #
    # 949일 임계값 스윕 — 앞뒤 절반 부호가 갈리는 지점이 0.62와 0.64 사이다:
    #   0.55  675건 52.7%   -88.4   (앞 -279.2 / 뒤 +190.8)
    #   0.62  598건 53.0%   -16.3   (앞 -151.3 / 뒤 +135.0)
    #   0.64  569건 54.1%  +297.9   (앞  +28.0 / 뒤 +270.0)
    #   0.68  503건 54.9%  +475.6   (앞 +248.1 / 뒤 +227.5)
    #   0.74  414건 55.6%  +544.7   (앞 +469.7 / 뒤  +75.0)
    #   0.78  333건 53.8%   +96.1   (앞 +153.0 / 뒤  -56.8)
    #
    # 0.64~0.76이 전부 양쪽 플러스인 고원이라 단일 스파이크가 아니다.
    # 최고점(0.74)이 아니라 고원 중앙을 고른다 — 최고점 고르기는 과적합이고,
    # 0.68이 앞뒤 균형도 가장 낫다.
    channel_r2_min: float = 0.68
    # 회귀 기간 전체의 방향 이동폭이 가격의 0.4% 이상이어야 채널로 인정한다.
    channel_min_move_pct: float = 0.004

    # 현재가에서 채널 끝까지 너무 멀면 미체결 주문만 쌓이므로 기다린다.
    max_entry_distance_pct: float = 0.005
    limit_ttl_bars: int = 3

    leverage: int = 25
    margin_fraction: float = 0.10
    # 거래 한 번으로 전체 계좌가 2% 증가하는 목표다. 잔고 10%를 증거금으로
    # 쓰므로 필요한 증거금 순수익률은 20%이며, 전략이 레버리지와 실제
    # 메이커 수수료를 이용해 기초자산 목표폭으로 변환한다.
    account_take_profit_pct: float = 0.02
    # 손절 가격폭은 계산된 익절 가격폭과 정확히 대칭으로 둔다.
    min_target_r: float = 0.0
    partial_exit_ratio: float = 1.0
    trail_atr_mult: float = 1.2
    # 시간 경과만으로 종가 청산하지 않는다. max_hold_bars는 시간청산을 다시
    # 켤 때 사용할 값이며, 현재는 손절/익절 주문 중 하나가 체결될 때까지 보유한다.
    max_hold_bars: int = 12
    time_exit_enabled: bool = False
    # 25배 BTC 청산 거리는 약 3.7%, 약 0.78% 손절보다 약 4.7배 멀다.
    min_liquidation_multiple: float = 3.0
    # 대칭 가격폭 손절에 시장가 비용을 합치면 계좌 손실은 약 2.16%다.
    symbol_risk_cap: float = 0.03
    breakeven_offset_pct: float = 0.0

    max_daily_entries: int = 12
    consecutive_loss_limit: int = 3


BTC_REGIME_25X = BtcRegime25xConfig()


# ---------------------------------------------------------------- 피보나치

# 진입 판정에 사용하는 핵심 레벨.
# 레벨을 늘릴수록 아무 가격에나 걸려서 변별력이 떨어진다.
FIB_CORE = (0.382, 0.5, 0.618, 0.65)

# 참고용으로 계산만 하는 레벨
FIB_EXTRA = (0.236, 0.705, 0.786, 0.886)

# 익절 목표용 확장
FIB_EXTENSION = (1.272, 1.618)


@dataclass(frozen=True)
class FibConfig:
    """조정 대상 파라미터는 4개로 제한한다.

    조합 81개. 이 이상 늘리면 우연히 좋은 조합이 나올 확률이 올라가고,
    그걸 고르면 실전에서 재현되지 않는다.
    """

    pivot_n: int = 5                    # 후보: 3, 5, 8
    min_swing_pct: float = 0.03         # 후보: 0.02, 0.03, 0.05
    zone_tolerance_atr: float = 0.25    # 후보: 0.2, 0.25, 0.4
    min_zone_score: float = 3.0         # 후보: 2, 3, 4

    # --- 아래는 고정 ---
    max_lookback: int = 500             # 스윙 탐색 범위 (캔들 수)
    max_swing_candidates: int = 3       # 합류 계산에 쓸 스윙 후보 수
    max_hold_bars: int = 96             # 시간 손절


FIB = FibConfig()


# 합류 존 스코어 가중치
CONFLUENCE_WEIGHTS: dict[str, float] = {
    "fib_overlap": 1.0,      # 서로 다른 스윙의 레벨이 겹칠 때 개당
    "golden_pocket": 1.0,    # 0.618 / 0.65 포함
    "vpvr_hvn": 2.0,
    "ma_50": 2.0,
    "ma_200": 2.0,
    "horizontal_sr": 1.0,    # 과거 스윙 고저점
    "pivot_volume": 0.5,     # 피벗 캔들 거래량 상위
}


# ---------------------------------------------------------------- VPVR

@dataclass(frozen=True)
class VpvrConfig:
    lookback_bars: int = 500
    num_bins: int = 50
    # 거래량 상위 이 비율 안에 드는 빈을 HVN으로 본다
    hvn_percentile: float = 80.0
    lvn_percentile: float = 20.0


VPVR = VpvrConfig()


# ---------------------------------------------------------------- 단타 (XRP)

@dataclass(frozen=True)
class ScalpConfig:
    """5분봉 패턴 단타 전용.

    익절 방식은 BTC와 같다 — VPVR HVN에서 절반 익절, 나머지는 트레일링.
    처음엔 "5분봉 VPVR은 노이즈"라며 고정 R 배수를 썼는데 근거가 없었다.
    실측하니 HVN 중앙값 3개, 목표까지 중앙값 2.43R로 고정 1.5R보다 나았다.

    아래 tp1_r_multiple은 HVN을 못 찾은 구간(실측 21%)의 대체값이다.
    1:1 목표는 수수료 차감 후 손익분기 승률이 69%까지 올라가 성립하지 않는다.
    """

    tp1_r_multiple: float = 1.5         # 1차 목표 = 손절거리 * 이 배수
    trail_atr_mult: float = 0.7         # 잔여 물량 트레일링 (BTC보다 좁게)
    max_hold_bars: int = 40             # 시간 손절 (5분봉 40봉 = 3시간 20분)

    # 손절선 ATR 버퍼. BTC(0.5)보다 좁게 잡는다 — 5분봉은 봉당 변동이
    # 작아 같은 배수를 쓰면 손절이 구조에서 너무 멀어진다.
    #
    # 다만 이 값이 유동성 스윕 패턴의 성립 여부를 좌우한다. 스윕은 진입가
    # (스윕당한 레벨)와 손절(꼬리 끝)이 구조적으로 가까워서, 버퍼가 좁으면
    # 손절 거리 하한(0.38%)에 걸려 신호의 95%가 버려진다.
    # 7단계 파라미터 탐색 대상. 지금은 값을 만지지 않는다.
    atr_buffer_mult: float = 0.3

    # 지정가 진입이 이 봉 수 안에 체결되지 않으면 신호를 버린다.
    # 쫓아가서 시장가로 잡으면 메이커 리베이트가 사라져 기대값이 무너진다.
    limit_order_ttl_bars: int = 3

    # --- 상위 종목 연동 필터 ---
    # XRP는 BTC 베타다. BTC가 급락 중일 때 XRP 롱 패턴은 전부 무의미해진다.
    btc_context_tf: str = "1h"          # config.symbols.Timeframe 값 중 하나
    btc_adverse_move_pct: float = 0.01  # BTC 1시간봉 1% 이상 역방향이면 진입 차단

    # --- 변동성 서킷브레이커 ---
    # 뉴스 스파이크 구간에서는 패턴이 성립하지 않는다.
    # 5분봉 하나가 평소 고저폭의 이 배수를 넘으면 일시 정지.
    spike_range_mult: float = 5.0
    spike_lookback: int = 100           # "평소"의 기준 구간 (중앙값)
    spike_cooldown_bars: int = 6        # 정지 지속 (5분봉 6봉 = 30분)


SCALP = ScalpConfig()


# ---------------------------------------------------------------- 실행

@dataclass(frozen=True)
class RuntimeConfig:
    poll_seconds: int = 10
    candle_limit: int = 500

    # 실전 예정 규모와 맞춘다. 계좌가 작으면 거래소 최소 주문 단위에
    # 걸려 계산된 포지션 크기를 그대로 넣지 못하는 경우가 생긴다.
    initial_balance: float = 1_000.0

    # 경제지표 발표 전후 신규 진입 차단 (분)
    macro_blackout_before: int = 15
    macro_blackout_after: int = 30


RUNTIME = RuntimeConfig()


# Gate.io가 한 타임프레임에서 돌려주는 과거 캔들 총량 상한.
# 2026-08-05 확인: 10,000봉을 넘겨 요청하면
# "Candlestick too long ago. Maximum 10000 points recently are allowed" 400 응답.
# 실측으로 5분봉 34일(9,792봉)은 통과, 36일(10,368봉)은 거부.
# 경계에서 잘리지 않도록 여유를 두고 9,500으로 잡는다.
MAX_HISTORY_POINTS = 9_500

# 과거 데이터 백필 깊이 (타임프레임별 일수).
# 여기 적는 값은 "필요한 깊이"이고, 거래소 상한을 넘으면 수집 시점에 잘린다.
# 상한에 맞춰 미리 줄여두지 않는 이유는, 어디까지가 우리 요구이고
# 어디부터가 거래소 제약인지 구분이 남아야 하기 때문이다.
#
#   실제 확보 가능:  1d 800일 / 4h 800일 / 1h 395일 / 15m 98일 / 5m 32일
#
# 5분봉이 한 달치뿐이라 XRP 패턴 통계는 이것만으로 부족하다.
# 실시간 루프가 캐시에 계속 덧붙이므로 시간이 지나면 늘어난다.
HISTORY_DAYS: dict[str, int] = {
    "1d": 800,      # 800봉. MA200 + 합류 룩백 500봉 확보
    "4h": 800,      # 4,800봉
    "1h": 400,      # 9,600봉 -> 상한에 걸려 395일로 잘림
    "15m": 180,     # 17,280봉 -> 98일로 잘림
    "5m": 180,      # 51,840봉 -> 32일로 잘림
}


def history_days(interval: str, step_seconds: int) -> int:
    """거래소 상한을 반영한 실제 수집 일수."""
    want = HISTORY_DAYS.get(interval, 0)
    cap = MAX_HISTORY_POINTS * step_seconds // 86400
    return min(want, cap)


# ---------------------------------------------------------------- 로깅

TRADE_LOG_PATH = "data/trades.csv"        # 체결 기록 (컬럼 고정, 엑셀 확인용)
SIGNAL_LOG_PATH = "data/signals.jsonl"    # 신호 스냅샷 (스킵/팬텀 포함)
STATE_PATH = "data/state.json"

# 디스코드 웹훅 URL은 환경변수로 받는다. 코드에 넣지 않는다.
# 종류별 URL이 없으면 기존 단일 URL로 보낸다. 기존 배포와
# 로컬 개발 환경을 깨지 않기 위한 하위 호환 규칙이다.
DISCORD_WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"
DISCORD_SYSTEM_WEBHOOK_ENV = "DISCORD_SYSTEM_WEBHOOK_URL"
DISCORD_TRADE_WEBHOOK_ENV = "DISCORD_TRADE_WEBHOOK_URL"
DISCORD_BTC_TRADE_WEBHOOK_ENV = "DISCORD_BTC_TRADE_WEBHOOK_URL"
DISCORD_XRP_TRADE_WEBHOOK_ENV = "DISCORD_XRP_TRADE_WEBHOOK_URL"
DISCORD_BALANCE_WEBHOOK_ENV = "DISCORD_BALANCE_WEBHOOK_URL"
