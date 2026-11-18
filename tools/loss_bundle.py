"""손실 포지션마다 AI 사후 분석용 자료 묶음(JSON + 차트 PNG)을 만든다.

    python -m tools.loss_bundle data/server_20261118 data/server_20261118/xrp \\
        --cache data/server_20261118/cache --out data/server_20261118/loss_bundles \\
        --baseline data/server_20261118/replay_xrp_live_like_raw

숫자는 전부 여기서 계산한다. AI는 이 사실을 해석만 한다(계산을 맡기면 틀린다).
캐시에 없는 구간은 Gate 공개 캔들 API로 받되 캐시 파일에는 쓰지 않는다.

양방향 격리 실험(EXP-015)은 한쪽 청산이 설계상 결과라 대상이 아니다 — 그 디렉터리를 넘기지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from core import candles as cd
from tools.loss_review import load_positions, r_multiple
from reporting.logs import read_trades

# 전략 이름에 타임프레임 접미사가 없던 초기 로그와 BTC 전략의 판단 봉
DEFAULT_TF = {
    "btc_regime_25x": "15m",
    "btc_regression_channel": "1h",
    "xrp_patterns.double_top": "1h",
    "xrp_patterns.kotegawa": "1h",
    "xrp_patterns.triangle": "1h",
}
BARS_BEFORE = 80     # 차트·패턴 형성 구간
AFTER_MIN_BARS = 24  # 청산 뒤 관찰 최소 봉 수 (보유 시간이 더 길면 그만큼)


def strategy_tf(strategy: str) -> str:
    tail = strategy.rsplit(".", 1)[-1]
    if tail in ("5m", "15m", "1h", "4h"):
        return tail
    return DEFAULT_TF.get(strategy, "1h")


def symbol_of(p: dict) -> str:
    return p.get("symbol") or ("BTC_USDT" if p["strategy"].startswith("btc") else "XRP_USDT")


class Candles:
    """캐시 CSV + 부족분은 공개 API. 같은 (종목, 봉) 요청은 한 번만 받는다."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.frames: dict[tuple[str, str], pd.DataFrame] = {}
        self.fetched: list[str] = []

    def get(self, symbol: str, tf: str, start: int, end: int) -> pd.DataFrame:
        key = (symbol, tf)
        if key not in self.frames:
            p = self.cache_dir / f"{symbol}_{tf}.csv"
            self.frames[key] = pd.read_csv(p) if p.exists() else cd.to_dataframe([])
        df = self.frames[key]
        step = cd.interval_seconds(tf)
        last = int(df["ts"].iloc[-1]) if len(df) else 0
        first = int(df["ts"].iloc[0]) if len(df) else 1 << 62
        if end > last + step or start < first:
            from core.exchange import GateClient

            with GateClient() as client:
                res = client.fetch_range(symbol, tf, min(start, last + step) if len(df) else start, end)
            new = res.df.drop(columns=["dt"], errors="ignore")
            df = pd.concat([df, new]).drop_duplicates("ts", keep="last").sort_values("ts")
            self.frames[key] = df.reset_index(drop=True)
            self.fetched.append(f"{symbol} {tf} {len(new)}봉")
        out = self.frames[key]
        return out[(out["ts"] >= start) & (out["ts"] <= end)].reset_index(drop=True)


def entry_row(trades: list[dict], strategy: str, ts: int) -> dict:
    return next(r for r in trades if r["strategy"] == strategy and r["action"] == "entry" and int(r["ts"]) == ts)


def exit_rows(trades: list[dict], strategy: str, start: int, end: int) -> list[dict]:
    return [
        {k: r[k] for k in ("dt", "reason", "price", "ratio", "pnl", "fee", "note")}
        for r in trades
        if r["strategy"] == strategy and r["action"] == "exit" and start <= int(r["ts"]) <= end
    ]


def trend(df: pd.DataFrame, n: int) -> dict:
    """마지막 확정봉 종가와 n봉 단순이평의 위치·기울기."""
    if len(df) < n + 5:
        return {"note": "봉 부족"}
    ma = df["close"].rolling(n).mean()
    close, now, prev = df["close"].iloc[-1], ma.iloc[-1], ma.iloc[-6]
    return {
        f"close_vs_sma{n}_pct": round(100 * (close / now - 1), 2),
        f"sma{n}_slope_5bars_pct": round(100 * (now / prev - 1), 2),
        "direction": "up" if close > now and now > prev else "down" if close < now and now < prev else "mixed",
    }


def atr_pct_rank(df: pd.DataFrame, n: int = 14) -> float | None:
    if len(df) < n + 50:
        return None
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean().dropna()
    return round(100 * (atr <= atr.iloc[-1]).mean(), 1)


def build(p: dict, trades: list[dict], candles: Candles) -> dict:
    strategy, side = p["strategy"], p["side"]
    symbol, tf = symbol_of(p), strategy_tf(strategy)
    step = cd.interval_seconds(tf)
    e = entry_row(trades, strategy, p["entry_ts"])
    entry, stop = float(e["price"]), float(e["stop"] or 0)
    tp1 = float(e["tp1"]) if e.get("tp1") else None
    risk = abs(entry - stop) if stop else None
    sign = 1 if side == "long" else -1
    # 시장가 진입 행의 ts는 신호봉 **시작** 시각이다(engine/loop.py가 market_ts=bar_ts로 기록).
    # 실제 체결은 신호봉 마감 직후라 실시간 신호 로그의 벽시계 ts가 더 정확하다.
    sig_ts = int((p["signal"] or {}).get("ts") or 0)
    t0 = sig_ts if 0 < sig_ts - p["entry_ts"] <= step + 60 else p["entry_ts"]
    hold = p["exit_ts"] - t0
    after = max(AFTER_MIN_BARS * step, hold)

    # 진입~청산 경로는 5분봉으로. 체결이 속한 5분봉부터 본다.
    # ponytail: 그 5분봉의 체결 전 고저도 섞여 MFE/MAE가 최대 5분만큼 과대될 수 있다. 1m 캔들로 바꾸면 해소
    path = candles.get(symbol, "5m", t0 - t0 % 300, p["exit_ts"])
    post = candles.get(symbol, "5m", p["exit_ts"] + 1, p["exit_ts"] + after)
    exits = exit_rows(trades, strategy, p["entry_ts"], p["exit_ts"])

    facts: dict = {"hold_minutes": round(hold / 60), "planned_risk_per_unit": risk}
    if risk and len(path):
        fav = (path["high"].max() - entry) if side == "long" else (entry - path["low"].min())
        adv = (entry - path["low"].min()) if side == "long" else (path["high"].max() - entry)
        facts["mfe_R"] = round(fav / risk, 2)
        facts["mae_R"] = round(adv / risk, 2)
        best_i = path["high"].idxmax() if side == "long" else path["low"].idxmin()
        facts["mfe_reached_minutes_after_entry"] = max(0, round((int(path["ts"][best_i]) - t0) / 60))
    last_exit = exits[-1] if exits else {}
    if risk and last_exit.get("reason") == "stop":
        slip = sign * (stop - float(last_exit["price"]))
        facts["stop_slippage_R"] = round(slip / risk, 3)
    if len(post):
        hi, lo = post["high"].max(), post["low"].min()
        facts["after_exit_window_minutes"] = round(after / 60)
        facts["after_exit_back_to_entry"] = bool(hi >= entry if side == "long" else lo <= entry)
        if tp1:
            facts["after_exit_hit_tp1"] = bool(hi >= tp1 if side == "long" else lo <= tp1)
        if risk:
            facts["after_exit_max_favorable_R"] = round(
                ((hi - entry) if side == "long" else (entry - lo)) / risk, 2)
    else:
        facts["after_exit"] = "청산 뒤 캔들 없음"

    # 상위 추세·변동성: 진입 시점에 확정된 봉만 쓴다
    ctx = {}
    for htf, n in (("1h", 50), ("4h", 50), ("1d", 20)):
        s = cd.interval_seconds(htf)
        df = candles.get(symbol, htf, t0 - (n + 60) * s, t0 - s)
        ctx[f"{htf}_trend"] = trend(df, n)
    hist = candles.get(symbol, tf, t0 - 500 * step, t0 - step)
    ctx[f"atr14_{tf}_percentile_500bars"] = atr_pct_rank(hist)
    if symbol != "BTC_USDT":
        btc = candles.get("BTC_USDT", "5m", t0 - 86400, p["exit_ts"])
        if len(btc):
            at_entry = btc[btc["ts"] <= t0]["close"]
            if len(at_entry):
                ctx["btc_24h_before_entry_pct"] = round(100 * (at_entry.iloc[-1] / btc["close"].iloc[0] - 1), 2)
                ctx["btc_entry_to_exit_pct"] = round(100 * (btc["close"].iloc[-1] / at_entry.iloc[-1] - 1), 2)

    sig = p["signal"] or {}
    r = r_multiple(p)
    if r is None and risk and e.get("notional"):
        # BTC 신호는 risk_amount를 남기지 않는다. 계획 손절 거리 × 수량으로 대신 계산
        r = p["pnl"] / (float(e["notional"]) / entry * risk)
    return {
        "id": f"{strategy}_{p['entry_ts']}",
        "trade": {
            "strategy": strategy, "symbol": symbol, "side": side, "decision_tf": tf,
            "entry_dt": e["dt"], "entry_price": entry, "stop": stop, "tp1": tp1,
            "leverage": e.get("leverage"), "contracts": e.get("contracts"), "entry_fill": e.get("note"),
            "filled_at": pd.Timestamp(t0, unit="s", tz="UTC").isoformat(), "exits": exits, "net_pnl": round(p["pnl"], 4),
            "R": round(r, 2) if r is not None else None,
            "exit_reason": p["exit_reason"],
        },
        "entry_rationale": {k: sig.get(k) for k in ("score", "reasons", "indicators", "tranche", "max_tranches")}
        if sig else "진입 신호 기록 없음",
        "price_path": facts,
        "market_context_at_entry": ctx,
    }


def plot(bundle: dict, candles: Candles, path: Path) -> None:
    t = bundle["trade"]
    tf = t["decision_tf"]
    step = cd.interval_seconds(tf)
    entry_ts = int(pd.Timestamp(t["filled_at"]).timestamp())
    exit_ts = int(pd.Timestamp(t["exits"][-1]["dt"]).timestamp()) if t["exits"] else entry_ts
    df = candles.get(t["symbol"], tf, entry_ts - BARS_BEFORE * step,
                     exit_ts + max(AFTER_MIN_BARS * step, exit_ts - entry_ts))
    if df.empty:
        return
    x = pd.to_datetime(df["ts"], unit="s")
    w = pd.Timedelta(seconds=step * 0.7)
    up = df["close"] >= df["open"]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.vlines(x, df["low"], df["high"], color="#777", linewidth=0.6)
    ax.bar(x[up], (df["close"] - df["open"])[up], w, df["open"][up], color="#26a69a")
    ax.bar(x[~up], (df["close"] - df["open"])[~up], w, df["open"][~up], color="#ef5350")
    span = (pd.to_datetime(entry_ts, unit="s"), pd.to_datetime(exit_ts, unit="s"))
    ax.axvspan(*span, color="#ffd54f", alpha=0.15)
    for label, y, c in (("entry", t["entry_price"], "#1565c0"), ("stop", t["stop"], "#c62828"),
                        ("tp1", t["tp1"], "#2e7d32")):
        if y:
            ax.axhline(y, color=c, linestyle="--", linewidth=1)
            ax.text(x.iloc[0], y, f" {label} {y:.6g}", color=c, va="bottom", fontsize=8)
    ind = bundle["entry_rationale"].get("indicators") if isinstance(bundle["entry_rationale"], dict) else None
    for k, v in (ind or {}).items():
        if isinstance(v, (int, float)) and 0.5 < v / t["entry_price"] < 2:  # 가격 단위 지표만 (목선·고점 등)
            ax.axhline(v, color="#8e24aa", linestyle=":", linewidth=0.8)
            ax.text(x.iloc[-1], v, f"{k} ", color="#8e24aa", ha="right", va="bottom", fontsize=7)
    for ex in t["exits"]:
        ax.scatter(pd.Timestamp(ex["dt"]).tz_localize(None), float(ex["price"]), marker="x", color="black", zorder=5)
    ax.set_title(f"{t['strategy']} {t['side']}  {t['entry_dt'][:16]}  pnl {t['net_pnl']:+.2f} ({t['R']}R)  [{tf}]")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def baseline(directory: Path) -> dict:
    """리플레이 기준 전략별 승률·평균 R·청산 사유 분포 — "평소 이 정도는 진다"의 기준."""
    positions, _ = load_positions(directory)
    out: dict = {}
    for p in positions:
        s = out.setdefault(p["strategy"], {"n": 0, "wins": 0, "R": [], "exit_reasons": {}})
        s["n"] += 1
        s["wins"] += p["pnl"] > 0
        if (r := r_multiple(p)) is not None:
            s["R"].append(r)
        s["exit_reasons"][p["exit_reason"]] = s["exit_reasons"].get(p["exit_reason"], 0) + 1
    return {
        k: {"positions": v["n"], "win_rate_pct": round(100 * v["wins"] / v["n"]),
            "avg_R": round(sum(v["R"]) / len(v["R"]), 2) if v["R"] else None, "exit_reasons": v["exit_reasons"]}
        for k, v in out.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, help="비교 기준 리플레이 디렉터리")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    candles = Candles(args.cache)
    base = baseline(args.baseline) if args.baseline else {}
    index = []
    for d in args.dirs:
        trades = read_trades(d / "trades.csv")
        positions, _ = load_positions(d)
        for p in positions:
            if p["pnl"] >= 0:
                continue
            b = build(p, trades, candles)
            b["account_dir"] = str(d)
            name = b["trade"]["strategy"].replace(".", "_")
            b["replay_baseline"] = base.get(name) or base.get(b["trade"]["strategy"]) or "기준 없음"
            stem = f"{b['trade']['entry_dt'][:16].replace(':', '')}_{name}"
            (args.out / f"{stem}.json").write_text(json.dumps(b, ensure_ascii=False, indent=2, default=str))
            plot(b, candles, args.out / f"{stem}.png")
            index.append((b["trade"]["entry_dt"][:16], b["trade"]["strategy"], b["trade"]["side"],
                          b["trade"]["net_pnl"], b["trade"]["exit_reason"], stem))
    index.sort()
    lines = ["| 진입 | 전략 | 방향 | 손익 | 청산 | 파일 |", "|---|---|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {d:+.2f} | {e} | {f} |" for a, b, c, d, e, f in index]
    (args.out / "index.md").write_text("\n".join(lines) + "\n")
    print(f"손실 {len(index)}건 -> {args.out}")
    if candles.fetched:
        print("API로 받은 구간:", ", ".join(candles.fetched))


if __name__ == "__main__":
    main()
