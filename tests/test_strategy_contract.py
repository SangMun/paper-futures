"""전략 계약 테스트.

핵심은 `test_리플레이에서_지표를_다시_계산하지_않는다`이다.
리플레이는 봉마다 generate()를 부르므로, 그 안에서 전 구간 지표를 계산하면
O(n^2)가 되어 파라미터 탐색이 불가능해진다(2분 -> 77일).

이건 성능 문제로만 보이지만 규율에 기대면 반드시 새어 나온다.
새 전략을 추가할 때 `STRATEGIES` 목록에 넣어 자동으로 감시받게 한다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from indicators import atr as atr_mod
from indicators import bollinger as bb
from indicators import ma
from indicators import oscillators as osc
from indicators import trend
from strategies.base import Prepared, Signal, prepare_standard, standard_series


T0 = 1_700_000_000
STEP = 3600


def frame(n: int) -> pd.DataFrame:
    closes = [10.0 + (i % 7) * 0.1 + i * 0.001 for i in range(n)]
    return pd.DataFrame(
        {
            "ts": [T0 + i * STEP for i in range(n)],
            "dt": pd.to_datetime([T0 + i * STEP for i in range(n)], unit="s", utc=True),
            "open": closes,
            "high": [c * 1.004 for c in closes],
            "low": [c * 0.996 for c in closes],
            "close": closes,
            "volume": [100.0] * n,
            "quote_volume": [1000.0] * n,
        }
    )


# ------------------------------------------------------------------ 표본 전략


class GoodStrategy:
    """올바른 구현. prepare에서 계산하고 generate는 인덱싱만 한다."""

    name = "good"

    def prepare(self, frames):
        return prepare_standard(frames)

    def generate(self, prepared: Prepared, as_of: int) -> list[Signal]:
        rsi = prepared.at("1h", "rsi", as_of)
        if pd.isna(rsi) or rsi > 40:
            return []
        price = float(prepared.frame("1h")["close"].iloc[as_of])
        return [Signal(self.name, "X_USDT", "long", price, price * 0.99)]


class LeakyStrategy:
    """잘못된 구현. generate 안에서 전 구간 지표를 다시 계산한다.

    이 전략은 트립와이어가 실제로 작동하는지 확인하기 위해서만 존재한다.
    """

    name = "leaky"

    def prepare(self, frames):
        return prepare_standard(frames)

    def generate(self, prepared: Prepared, as_of: int) -> list[Signal]:
        osc.rsi(prepared.frame("1h"))       # <- 봉마다 전 구간 재계산
        return []


def _btc_trend_case():
    """실제 전략은 1시간봉 + 일봉을 함께 받는다."""

    from strategies.btc_trend import BtcTrend

    return BtcTrend(), {"1h": frame(400), "1d": frame(400)}


def _xrp_cases():
    """XRP 패턴은 인스턴스마다 다른 봉을 쓴다.

    엔진에 등록된 것과 같은 목록을 쓴다 — 새 패턴을 추가하면
    자동으로 감시 대상이 된다.
    """

    from engine.loop import build_strategies

    # 어떤 조합이 와도 되도록 전 봉을 준다.
    frames = {tf: frame(400) for tf in ("5m", "15m", "1h", "4h", "1d")}
    return [(s, frames) for s in build_strategies()["XRP_USDT"]]


# 새 전략은 여기 추가한다. 그래야 generate() 안에서 지표를 계산하는 순간 걸린다.
def _fib_case():
    from strategies.fib_retracement import FibRetracement

    return FibRetracement(), {"1h": frame(400), "1d": frame(400)}


def _btc_candidate_cases():
    from engine.loop import build_btc_candidates

    frames = {"1h": frame(400), "1d": frame(400)}
    return [(strategy, frames) for strategy in build_btc_candidates()["BTC_USDT"]]


def _regression_channel_case():
    from strategies.btc_regression_channel import BtcRegressionChannel

    return BtcRegressionChannel(), {"1h": frame(400)}


def _btc_regime_25x_case():
    from strategies.btc_regime_25x import BtcRegime25x

    return BtcRegime25x(), {
        "5m": frame(400),
        "15m": frame(400),
    }


STRATEGIES = [
    (GoodStrategy(), {"1h": frame(400)}),
    _btc_trend_case(),
    _fib_case(),
    _regression_channel_case(),
    _btc_regime_25x_case(),
    *_btc_candidate_cases(),
    *_xrp_cases(),
]


# ------------------------------------------------------------------ 호출 계수


@pytest.fixture
def indicator_calls(monkeypatch):
    """지표 함수 호출 횟수를 센다."""

    counts: dict[str, int] = {}

    def wrap(module, attr):
        original = getattr(module, attr)

        def counted(*args, **kwargs):
            counts[attr] = counts.get(attr, 0) + 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, attr, counted)

    for mod, name in [
        (osc, "rsi"),
        (osc, "macd"),
        (atr_mod, "atr"),
        (ma, "sma"),
        (bb, "bollinger"),
        (bb, "squeeze"),
        (trend, "adx"),
    ]:
        wrap(mod, name)

    return counts


def replay(strategy, frames, bars: int) -> None:
    """리플레이 한 번. 준비 1회 + 봉마다 generate.

    진입 타임프레임은 5분봉이 있으면 그쪽, 없으면 1시간봉이다.
    as_of는 진입봉의 위치 인덱스이므로 프레임을 잘못 고르면 범위가 어긋난다.
    """

    prepared = strategy.prepare(frames)
    tf = getattr(strategy, "tf", None)
    entry_tf = tf.entry if tf else ("5m" if "5m" in frames else "1h")
    n = len(frames[entry_tf])
    for i in range(n - bars, n):
        strategy.generate(prepared, i)


# ------------------------------------------------------------------ 트립와이어


@pytest.mark.parametrize("strategy,frames", STRATEGIES, ids=lambda x: getattr(x, "name", ""))
def test_리플레이에서_지표를_다시_계산하지_않는다(strategy, frames, indicator_calls):
    """지표 호출 횟수가 봉 수에 비례하면 안 된다.

    prepare()가 타임프레임당 한 번씩만 부르므로, 상한은
    타임프레임 수 * 표준 지표에서 각 함수가 쓰이는 횟수다.
    넉넉히 잡아도 봉 수(200)와는 자릿수가 다르다.
    """

    replay(strategy, frames, bars=50)
    few = dict(indicator_calls)

    indicator_calls.clear()
    replay(strategy, frames, bars=200)
    many = dict(indicator_calls)

    # 호출 횟수가 봉 수와 함께 늘면 generate 안에서 계산하고 있다는 뜻이다.
    # 고정 상한을 두지 않는 이유는 전략마다 prepare에서 쓰는 지표 수가
    # 다르기 때문이다. 검사할 것은 "총 몇 번"이 아니라 "봉 수에 비례하는가"다.
    for name, count in many.items():
        assert count == few.get(name, 0), (
            f"{name}이 봉 50개에 {few.get(name, 0)}회, 200개에 {count}회 호출됨 "
            "— generate 안에서 계산하고 있다"
        )


def test_트립와이어가_실제로_잡는지(indicator_calls):
    """감시 테스트가 통과만 하고 아무것도 안 잡는 상태를 막는다."""

    frames = {"1h": frame(400)}
    replay(LeakyStrategy(), frames, bars=200)

    assert indicator_calls["rsi"] > 200


# ------------------------------------------------------------------ Prepared


def test_prepare는_타임프레임마다_계산한다():
    frames = {"1h": frame(300), "1d": frame(300)}
    p = prepare_standard(frames)

    assert set(p.series) == {"1h", "1d"}
    assert "rsi" in p.series["1h"]


def test_at은_as_of_시점의_값():
    df = frame(300)
    p = prepare_standard({"1h": df})
    rsi = standard_series(df)["rsi"]

    assert p.at("1h", "rsi", 250) == pytest.approx(rsi.iloc[250])


def test_at은_타입을_바꾸지_않는다():
    """국면은 문자열, 스퀴즈는 bool이다. float으로 캐스팅하면 깨진다."""

    p = prepare_standard({"1h": frame(300)})

    assert isinstance(p.at("1h", "regime", 250), str)
    assert isinstance(bool(p.at("1h", "bb_squeeze", 250)), bool)


def test_준비되지_않은_지표는_보유목록과_함께_에러():
    p = prepare_standard({"1h": frame(300)})

    with pytest.raises(KeyError, match="보유"):
        p.get("1h", "없는지표")
    with pytest.raises(KeyError, match="준비되지 않은 타임프레임"):
        p.frame("4h")


def test_series는_원본과_인덱스가_같다():
    """as_of가 위치 인덱스이므로 길이가 어긋나면 다른 봉을 읽는다."""

    df = frame(300)
    p = prepare_standard({"1h": df})

    for name, s in p.series["1h"].items():
        assert len(s) == len(df), name
        assert s.index.equals(df.index), name
