import csv
import json

from reporting.logs import TRADE_COLUMNS
from tools.loss_review import load_positions, report


def _write(tmp_path, trades, signals):
    with (tmp_path / "trades.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_COLUMNS)
        w.writeheader()
        for row in trades:
            w.writerow(row)
    with (tmp_path / "signals.jsonl").open("w", encoding="utf-8") as f:
        for s in signals:
            f.write(json.dumps(s) + "\n")


def _trade(ts, action, price, reason="", ratio=1.0, pnl=""):
    return {"ts": ts, "strategy": "p", "symbol": "XRP_USDT", "side": "long", "action": action,
            "reason": reason, "price": price, "notional": 100 if action == "entry" else "",
            "ratio": ratio, "pnl": pnl, "note": "taker"}


def _signal(ts, entry, reasons, rsi):
    return {"ts": ts, "action": "entry", "strategy": "p", "symbol": "XRP_USDT", "side": "long",
            "entry": entry, "risk_amount": 10.0, "score": 4, "reasons": reasons,
            "indicators": {"rsi": rsi, "neckline": entry * 1.01}}


def test_partial_exits_are_one_position_and_signals_attach(tmp_path):
    _write(
        tmp_path,
        trades=[
            _trade(1000, "entry", 1.0),
            _trade(2000, "exit", 1.1, "tp1", ratio=0.5, pnl=5),
            _trade(3000, "exit", 1.05, "trail", ratio=0.5, pnl=3),
            _trade(10_000, "entry", 2.0),
            _trade(11_000, "exit", 1.9, "stop", pnl=-10),
        ],
        # 두 번째 신호 ts는 벽시계(체결 ts와 무관) — 진입가로 매칭돼야 한다
        signals=[_signal(1600, 1.0, ["retested"], 30), _signal(999_999, 2.0, [], 70)],
    )
    positions, unmatched = load_positions(tmp_path)
    assert unmatched == 0
    assert len(positions) == 2
    win, loss = sorted(positions, key=lambda p: p["entry_ts"])
    assert win["pnl"] > 0 and loss["pnl"] < 0
    assert win["signal"]["reasons"] == ["retested"]
    assert loss["signal"]["entry"] == 2.0

    text = report(tmp_path)
    assert "| retested 있음 | 1 (표본부족) | 100% |" in text
    assert "| retested 없음 | 1 (표본부족) | 0% | -1.01 |" in text  # 진입 수수료 포함
    assert "rsi" in text
    assert "neckline" not in text  # 절대 가격 지표는 제외


def test_손실_유형_힌트():
    from tools.loss_ai import loss_type_hint

    def b(reason="stop", mfe=0.1, hit=False):
        return {"trade": {"exit_reason": reason}, "price_path": {"mfe_R": mfe, "after_exit_hit_tp1": hit}}

    assert loss_type_hint(b(reason="time")) == "시간초과"
    assert loss_type_hint(b(hit=True, mfe=1.5)) == "손절사냥"
    assert loss_type_hint(b(mfe=1.2)) == "수익반납"
    assert loss_type_hint(b(mfe=0.1)) == "즉시역행"
    assert loss_type_hint(b(mfe=0.6)) == "중간역행"
