"""과거 데이터 리플레이.

**엔진 코드를 그대로 쓴다.** 별도 시뮬레이터를 만들지 않는다.

이 원칙이 중요한 이유는 이미 한 번 대가를 치렀기 때문이다. 4~6단계의 성과
숫자를 임시 스크립트로 쟀는데, 그 스크립트와 엔진의 지정가 체결 규칙이
어긋나 있었다. 스크립트는 "신호 다음 봉부터" 체결했고 엔진은 "신호 봉"으로
체결했다 — 엔진 쪽이 미래참조였고, 스크립트가 맞았다는 걸 한참 뒤에 알았다.

여기서는 가장 짧은 진입봉의 마감 시각을 시장 시계로 삼아
`Engine.run_once(decision_ts=...)`를 호출한다. 상위봉은 그 시각까지 마감된
것만 보며, 시장가 신호는 다음 가장 짧은 봉 시가에 체결한다. 진입 조건,
리스크 판정, 지정가 TTL, 청산 규칙은 전부 실시간과 같은 코드를 탄다.

수집만 갈아끼운다 — `ReplayClient`가 캐시에서 읽은 프레임을 그대로 돌려준다.
전체 프레임을 고정하고 판단 시각만 옮기므로 prepare가 한 번만 돈다.
"""

from __future__ import annotations

import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from config.symbols import ACTIVE, SymbolSpec
from core import cache, candles as cd
from core.exchange import FetchResult
from engine import state as st
from engine.broker import Broker
from engine.loop import Engine
from reporting.logs import SignalLog, TradeLog
from reporting.notify import Notifier


class ReplayClient:
    """캐시에서 읽은 프레임을 그대로 돌려주는 수집기.

    엔진은 `fetch_frames`만 부르므로 여기만 갈아끼우면 나머지 경로가
    실시간과 완전히 같아진다.

    정합성 검사(`problems`)는 비워 둔다. 캐시는 이미 `validate`를 통과한
    데이터이고, 여기서 다시 문제를 만들면 진입이 차단되어 리플레이가
    실시간과 다른 경로를 타게 된다.
    """

    def __init__(self, frames_by_symbol: dict[str, dict[str, pd.DataFrame]]):
        self._frames = frames_by_symbol

    def fetch_frames(self, spec: SymbolSpec, limit: int = 0, now_ts=None):
        frames = self._frames.get(spec.contract, {})
        return {
            tf: FetchResult(spec.contract, tf, df, [])
            for tf, df in frames.items()
            if tf in spec.required_timeframes
        }

    def close(self) -> None:
        pass


@dataclass
class ReplayResult:
    bars: int
    seconds: float
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    final_balance: float
    initial_balance: float
    trades: list[dict] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)
    open_positions: int = 0

    @property
    def return_pct(self) -> float:
        if self.initial_balance <= 0:
            return 0.0
        return (self.final_balance / self.initial_balance - 1) * 100

    def by_strategy(self) -> dict[str, dict]:
        """전략별 성과. 로그 태그가 집계 단위다 (README 원칙 5).

        롱/숏을 따로 센다 — 크립토는 상승 편향이 있어 같은 규칙이라도
        방향별 통계가 다르고, 합치면 한쪽이 다른 쪽을 가린다.
        """

        exits = [t for t in self.trades if t.get("action") == "exit"]
        out: dict[str, dict] = defaultdict(
            lambda: {
                "trades": 0, "wins": 0, "pnl": 0.0,
                "long": 0, "short": 0, "long_pnl": 0.0, "short_pnl": 0.0,
                "reasons": Counter(),
            }
        )

        for t in exits:
            row = out[t["strategy"]]
            pnl = float(t.get("pnl") or 0.0)
            row["trades"] += 1
            row["pnl"] += pnl
            row["wins"] += pnl > 0
            row["reasons"][t.get("reason", "?")] += 1
            side = t.get("side", "?")
            if side in ("long", "short"):
                row[side] += 1
                row[f"{side}_pnl"] += pnl

        for row in out.values():
            row["win_rate"] = 100 * row["wins"] / row["trades"] if row["trades"] else 0.0
            row["avg"] = row["pnl"] / row["trades"] if row["trades"] else 0.0

        return dict(out)

    def skip_reasons(self) -> Counter:
        return Counter(
            s["skip_reason"] for s in self.signals
            if s.get("action") == "skip" and s.get("skip_reason")
        )

    def completed_positions(self) -> list[dict]:
        """부분 익절 행을 한 포지션으로 합친 완료 거래.

        TradeLog는 TP1과 최종 청산을 각각 기록하므로 exit 행 개수를 그대로
        승률 분모로 쓰면 한 번 이긴 포지션이 두 번의 승리로 세어진다.
        진입 수수료는 exit pnl에 들어 있지 않아 여기서 함께 차감한다.
        """

        from config.settings import FEES

        active: dict[tuple[str, str], dict] = {}
        completed: list[dict] = []

        for row in self.trades:
            key = (row.get("strategy", ""), row.get("symbol", ""))
            action = row.get("action")
            if action == "entry":
                notional = float(row.get("notional") or 0.0)
                maker = row.get("note") == "maker"
                entry_fee = notional * (
                    FEES.maker_rate if maker else FEES.taker_rate
                )
                active[key] = {
                    "strategy": key[0],
                    "symbol": key[1],
                    "side": row.get("side", ""),
                    "entry_ts": int(row.get("ts") or 0),
                    "exit_ts": 0,
                    "pnl": -entry_fee,
                    "exit_reason": "",
                    "partials": 0,
                }
                continue

            if action != "exit" or key not in active:
                continue

            position = active[key]
            position["pnl"] += float(row.get("pnl") or 0.0)
            position["exit_ts"] = int(row.get("ts") or 0)
            position["exit_reason"] = row.get("reason", "")
            if row.get("reason") == "tp1" and float(row.get("ratio") or 0.0) < 1.0:
                position["partials"] += 1
                continue

            completed.append(position)
            del active[key]

        return completed

    def positions_by_strategy(self) -> dict[str, dict]:
        """완료 포지션 기준 전략별 승률·손익."""

        out: dict[str, dict] = defaultdict(
            lambda: {
                "positions": 0,
                "wins": 0,
                "pnl": 0.0,
                "long": 0,
                "short": 0,
                "long_pnl": 0.0,
                "short_pnl": 0.0,
                "reasons": Counter(),
            }
        )
        for position in self.completed_positions():
            row = out[position["strategy"]]
            pnl = float(position["pnl"])
            side = position["side"]
            row["positions"] += 1
            row["wins"] += pnl > 0
            row["pnl"] += pnl
            row["reasons"][position["exit_reason"]] += 1
            if side in ("long", "short"):
                row[side] += 1
                row[f"{side}_pnl"] += pnl

        for row in out.values():
            n = row["positions"]
            row["win_rate"] = 100 * row["wins"] / n if n else 0.0
            row["avg"] = row["pnl"] / n if n else 0.0
        return dict(out)

    @property
    def realized_max_drawdown_pct(self) -> float:
        """체결 후 기록된 실현 잔고 기준 최대 낙폭.

        보유 중 미실현 손익은 상태에 기록하지 않으므로 실제 equity DD보다
        작을 수 있다. 결과표에서 이를 명시해야 한다.
        """

        peak = self.initial_balance
        worst = 0.0
        for row in self.trades:
            raw = row.get("balance")
            if raw in (None, ""):
                continue
            balance = float(raw)
            peak = max(peak, balance)
            if peak > 0:
                worst = max(worst, (peak - balance) / peak * 100)
        return worst


def load_frames(
    spec: SymbolSpec,
    extra: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """종목이 필요로 하는 캔들 + 보조 프레임을 캐시에서 읽는다."""

    frames = {tf: cache.load(spec.contract, tf) for tf in spec.required_timeframes}
    return {tf: df for tf, df in frames.items() if not df.empty}


def replay(
    strategies: dict[str, list],
    symbols: tuple[SymbolSpec, ...] = ACTIVE,
    balance: float = 1_000.0,
    warmup: int = 500,
    max_bars: int | None = None,
    log_dir: Path | None = None,
    progress_every: int = 0,
    risk=None,
    live_like: bool = True,
) -> ReplayResult:
    """과거 전체 구간을 봉 단위로 재생한다.

    warmup은 지표가 안정되기 전 구간을 건너뛰는 봉 수다. 여기서 시작하면
    실시간 루프가 500봉을 받는 것과 같은 조건이 된다.

    risk 기본값은 MEASUREMENT_RISK — 계좌 단위 브레이크를 끈 설정이다.
    켜 두면 한 전략의 연패가 다른 전략을 멈춰 전략별 성과가 오염된다.
    실전 조건으로 재려면 RISK를 명시적으로 넘길 것.
    """

    from config.settings import MEASUREMENT_RISK

    risk = MEASUREMENT_RISK if risk is None else risk

    requested_contracts = {s.contract for s in symbols}

    # XRP 실시간 엔진은 같은 회차에 받은 BTC 1시간봉을 방향 필터로 쓴다.
    # XRP만 리플레이한다고 BTC 프레임까지 빼면 실시간과 다른 전략이 된다.
    engine_symbols = list(symbols)
    if live_like and "XRP_USDT" in requested_contracts and "BTC_USDT" not in requested_contracts:
        btc = next((s for s in ACTIVE if s.contract == "BTC_USDT"), None)
        if btc is not None:
            engine_symbols.insert(0, btc)
    engine_symbols = tuple(engine_symbols)

    frames_by_symbol = {s.contract: load_frames(s) for s in engine_symbols}

    # 교차 종목 컨텍스트와 OI는 엔진의 `_with_context`가 캐시에서 읽는다.
    # 리플레이도 같은 경로를 타므로 여기서 따로 넣지 않는다.

    # 요청한 종목 중 가장 짧은 진입봉을 시장 시계로 쓴다. BTC와 XRP를 함께
    # 돌려도 XRP 5분봉 마감 시각을 따라가고, BTC 1시간봉은 매 12번째 회차에
    # 한 번만 처리된다. 위치 인덱스 i를 종목마다 그대로 넘기면 1시간봉 i와
    # 5분봉 i가 전혀 다른 날짜를 가리킨다.
    clocks = []
    for spec in symbols:
        tf = spec.driver_tf
        df = frames_by_symbol.get(spec.contract, {}).get(tf)
        if df is not None and not df.empty:
            clocks.append((cd.interval_seconds(tf), spec, tf, df))
    if not clocks:
        raise SystemExit("캐시가 비어 있다. fetch_history를 먼저 실행할 것.")

    clock_step, primary, entry_tf, df = min(clocks, key=lambda x: x[0])

    total = len(df)
    end = total if max_bars is None else min(total, warmup + max_bars)

    # **매 실행마다 로그를 비운다.**
    # TradeLog/SignalLog는 이어 쓰기라, 안 지우면 직전 실행 결과가 섞인다.
    # 파라미터 조합을 비교할 때 이게 조용히 결과를 오염시킨다 —
    # 실제로 BTC만 돌렸는데 직전 XRP 실행의 체결이 그대로 집계됐다.
    log_dir = Path(log_dir or "data/replay")
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    bot_state = st.BotState(balance=balance, day_start_balance=balance)
    # 보조로 넣은 BTC는 컨텍스트만 제공한다. XRP만 요청했는데 BTC 거래까지
    # 섞이면 잔고와 리스크 한도가 오염된다.
    engine_strategies = {
        contract: group
        for contract, group in strategies.items()
        if contract in requested_contracts
    }

    engine = Engine(
        strategies=engine_strategies,
        symbols=engine_symbols,
        client=ReplayClient(frames_by_symbol),
        broker=Broker(balance=balance),
        bot_state=bot_state,
        trade_log=TradeLog(log_dir / "trades.csv"),
        signal_log=SignalLog(log_dir / "signals.jsonl"),
        notifier=Notifier(url=""),          # 리플레이는 알림을 보내지 않는다
        risk=risk,
        defer_market_entries=live_like,
    )
    # 상태 파일을 건드리지 않는다. 리플레이가 실계좌 상태를 덮어쓰면 안 된다.
    engine._persist = lambda: None

    started = time.perf_counter()

    for i in range(warmup, end):
        try:
            if live_like:
                close_ts = int(df["ts"].iloc[i]) + clock_step
                engine.run_once(decision_ts=close_ts)
            else:
                engine.run_once(as_of=i)
        except Exception as exc:            # noqa: BLE001 - 한 봉이 실패해도 계속
            print(f"[replay] bar {i} 실패: {type(exc).__name__}: {exc}")

        if progress_every and (i - warmup) % progress_every == 0:
            done = i - warmup
            print(f"  {done}/{end - warmup}봉  잔고 {engine.bot_state.balance:,.2f}")

    seconds = time.perf_counter() - started
    from reporting.logs import read_signals, read_trades

    return ReplayResult(
        bars=end - warmup,
        seconds=seconds,
        start=df["dt"].iloc[warmup] if warmup < len(df) else None,
        end=df["dt"].iloc[min(end, len(df)) - 1],
        final_balance=engine.bot_state.balance,
        initial_balance=balance,
        trades=read_trades(log_dir / "trades.csv"),
        signals=read_signals(log_dir / "signals.jsonl"),
        open_positions=len(engine.bot_state.positions),
    )
