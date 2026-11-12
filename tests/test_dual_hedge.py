"""EXP-015 시간순 양방향 격리 리플레이 회귀 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from engine.dual_hedge import DualHedgeReplay, config_with


def frame(rows):
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df[["ts", "dt", "open", "high", "low", "close"]]


def test_한쪽_청산_뒤_종가가_이익보장가에_도달해야_트레일이_활성화된다():
    replay = DualHedgeReplay(
        balance=1_000,
        config=config_with(leverage=20, pair_margin_fraction=0.02),
    )
    # 첫 봉에 숏만 청산. 해당 봉의 수익은 사후로 활성화하면 안 된다.
    rows = frame([
        (0, 1.0, 1.06, 0.999, 1.05),
        (60, 1.05, 1.07, 1.04, 1.06),
    ])
    replay.run(rows, warmup=0)

    assert replay.pair is not None
    assert replay.pair.liquidated_side == "short"
    assert replay.pair.armed
    assert replay.pair.stop == replay.pair.profit_lock_price


def test_같은_봉에서_양쪽_청산가를_건드리면_수익으로_해석하지_않는다():
    replay = DualHedgeReplay(
        balance=1_000,
        config=config_with(leverage=20, pair_margin_fraction=0.02),
    )
    rows = frame([(0, 1.0, 1.06, 0.94, 1.0)])
    result = replay.run(rows, warmup=0)

    assert result.completed_pairs == 1
    assert result.results[0].exit_reason == "double_liquidation"
    assert result.results[0].pnl < 0


def test_이익보장_스탑은_전량_종료해도_페어_손익분기_이상이다():
    replay = DualHedgeReplay(
        balance=1_000,
        config=config_with(leverage=20, pair_margin_fraction=0.02),
    )
    # 1: 숏 청산, 2: 롱 이익보장 활성화, 3: 기존 이익보장 스탑으로 종료.
    rows = frame([
        (0, 1.0, 1.06, 0.999, 1.05),
        (60, 1.05, 1.07, 1.04, 1.06),
        (120, 1.06, 1.061, 1.04, 1.045),
    ])
    result = replay.run(rows, warmup=0)

    assert result.completed_pairs == 1
    assert result.results[0].exit_reason == "profit_lock_trail"
    assert result.results[0].pnl >= -0.02


def test_완료_뒤_다음_봉에서만_재진입한다():
    replay = DualHedgeReplay(
        balance=1_000,
        config=config_with(leverage=20, pair_margin_fraction=0.02),
    )
    rows = frame([
        (0, 1.0, 1.06, 0.94, 1.0),
        (60, 1.0, 1.001, 0.999, 1.0),
    ])
    result = replay.run(rows, warmup=0)

    assert result.completed_pairs == 1
    assert result.open_pair is not None
    assert result.open_pair.opened_ts == 60


def test_이익보장가에_못_닿은_생존다리는_시간한도에_종료한다():
    replay = DualHedgeReplay(
        balance=1_000,
        config=config_with(
            leverage=20, pair_margin_fraction=0.02,
            max_survivor_hold_seconds=60,
        ),
    )
    rows = frame([
        (0, 1.0, 1.06, 0.999, 1.05),  # 숏 청산, 롱 생존
        (60, 1.05, 1.051, 1.049, 1.05),
    ])
    result = replay.run(rows, warmup=0)

    assert result.completed_pairs == 1
    assert result.results[0].exit_reason == "survivor_timeout"


def test_생존다리도_반전해_자기_청산가에_닿으면_격리청산된다():
    replay = DualHedgeReplay(
        balance=1_000,
        config=config_with(leverage=20, pair_margin_fraction=0.02),
    )
    rows = frame([
        (0, 1.0, 1.06, 0.999, 1.05),  # 숏 청산, 롱 생존
        (60, 1.05, 1.051, 0.94, 0.95),  # 롱의 원래 청산가도 하향 돌파
    ])
    result = replay.run(rows, warmup=0)

    assert result.completed_pairs == 1
    assert result.results[0].exit_reason == "survivor_liquidation"
    assert result.results[0].liquidation == "both_sequential"


@pytest.mark.parametrize("buffer", [None, 0.001])
@pytest.mark.parametrize("loser_side", ["long", "short"])
def test_호가단위_이익보장가는_전량종료_손익분기를_내리지_않는다(loser_side, buffer):
    """최근접 반올림은 롱 lock을 한 틱 낮춰 손실을 만들 수 있다. 청산·보호 스탑 둘 다 확인한다."""

    replay = DualHedgeReplay(
        balance=1_000,
        config=config_with(
            leverage=100, pair_margin_fraction=0.02, liquidation_stop_buffer=buffer,
        ),
    )
    replay._open_pair(0, 0.1234)
    pair = replay.pair
    assert pair is not None
    # 진입가에서 시작해 지는 다리의 청산가까지 지나는 구간. 스탑이 있으면 스탑에서 끝난다.
    entry = 0.1234
    liq = (pair.long if loser_side == "long" else pair.short).liquidation_price
    replay._resolve_hedged(pair, entry, max(entry, liq), min(entry, liq), 60)
    assert pair.liquidated_side == loser_side and pair.profit_lock_price is not None

    replay._close_pair(120, "test_profit_lock", pair.profit_lock_price, loser_side)

    assert replay.results[-1].pnl >= -1e-9
