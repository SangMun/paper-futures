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


# 틱 대용 타임프레임.
#
# 10초봉이 더 촘촘하지만 거래소 보관 기간이 짧아 과거 검증에 못 쓴다.
# 1분봉이면 5분봉 하나를 5조각으로 나눠 본다 — 실시간 폴링(10초)보다
# 거칠지만, 봉 마감까지 통째로 기다리던 것과는 성격이 다르다.
TICK_TF = "1m"


class ReplayClient:
    """캐시에서 읽은 프레임을 그대로 돌려주는 수집기.

    엔진은 `fetch_frames`만 부르므로 여기만 갈아끼우면 나머지 경로가
    실시간과 완전히 같아진다.

    정합성 검사(`problems`)는 비워 둔다. 캐시는 이미 `validate`를 통과한
    데이터이고, 여기서 다시 문제를 만들면 진입이 차단되어 리플레이가
    실시간과 다른 경로를 타게 된다.
    """

    def __init__(
        self,
        frames_by_symbol: dict[str, dict[str, pd.DataFrame]],
        tick_frames: dict[str, pd.DataFrame] | None = None,
        tick_tf: str = TICK_TF,
    ):
        self._frames = frames_by_symbol
        # 틱 대용 프레임(기본 1분봉). 실행 시계를 재생할 때만 쓴다.
        self._tick_frames = tick_frames or {}
        self._tick_tf = tick_tf

    def fetch_frames(self, spec: SymbolSpec, limit: int = 0, now_ts=None):
        frames = self._frames.get(spec.contract, {})
        return {
            tf: FetchResult(spec.contract, tf, df, [])
            for tf, df in frames.items()
            if tf in spec.required_timeframes
        }

    def fetch(
        self,
        contract: str,
        interval: str,
        limit: int = 0,
        drop_unclosed: bool = True,
        now_ts=None,
    ) -> FetchResult:
        """실행 시계(`Engine.run_tick`)가 부르는 경로.

        drop_unclosed=False면 now_ts 시점의 **진행 중인 봉**을 만들어 붙인다.
        그 봉은 하위 타임프레임(기본 1분봉)을 now_ts까지 누적해서 만든다.
        실시간에 거래소가 주는 것과 같은 모양이라, 엔진은 리플레이인지
        실시간인지 구분하지 못한다 — 그래야 검증에 의미가 있다.
        """

        df = self._frames.get(contract, {}).get(interval)
        if df is None or df.empty or now_ts is None:
            return FetchResult(contract, interval, df if df is not None else cd.to_dataframe([]), [])

        step = cd.interval_seconds(interval)
        now_ts = int(now_ts)
        closed = df[df["ts"] + step <= now_ts]

        if drop_unclosed:
            return FetchResult(contract, interval, closed.reset_index(drop=True), [])

        partial = self._partial_bar(contract, interval, now_ts, step, df)
        if partial is None:
            return FetchResult(contract, interval, closed.reset_index(drop=True), [])

        merged = pd.concat([closed, partial], ignore_index=True)
        return FetchResult(contract, interval, merged.reset_index(drop=True), [])

    def _partial_bar(self, contract, interval, now_ts, step, df):
        """now_ts 시점까지 누적된 진행 중인 봉 한 줄.

        하위 봉이 아직 하나도 마감되지 않았으면(봉이 막 시작한 순간)
        시가만 있는 한 점짜리 봉을 만든다. 그 시각에 실제로 알 수 있는
        값은 시가뿐이므로 미래 참조가 아니다.
        """

        bar_ts = now_ts // step * step
        row = df[df["ts"] == bar_ts]
        if row.empty:
            return None                 # 데이터에 없는 구간

        ticks = self._tick_frames.get(contract)
        tick_step = cd.interval_seconds(self._tick_tf)

        if ticks is not None and not ticks.empty:
            window = ticks[
                (ticks["ts"] >= bar_ts)
                & (ticks["ts"] + tick_step <= now_ts)
            ]
        else:
            window = None

        open_ = float(row["open"].iloc[0])
        if window is None or window.empty:
            price = open_
            return pd.DataFrame([{
                "ts": bar_ts,
                "dt": pd.to_datetime(bar_ts, unit="s", utc=True),
                "open": open_, "high": price, "low": price, "close": price,
                "volume": 0.0, "quote_volume": 0.0,
            }])

        return pd.DataFrame([{
            "ts": bar_ts,
            "dt": pd.to_datetime(bar_ts, unit="s", utc=True),
            "open": open_,
            "high": float(window["high"].max()),
            "low": float(window["low"].min()),
            "close": float(window["close"].iloc[-1]),
            "volume": float(window["volume"].sum()),
            "quote_volume": float(window["quote_volume"].sum()),
        }])

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

    # 지정가 체결률. 메이커 전환을 켜면 성과보다 먼저 봐야 할 숫자다 —
    # 체결이 안 되면 그건 같은 전략이 아니라 다른 전략이다.
    limit_filled: int = 0
    limit_expired: int = 0

    @property
    def limit_fill_rate(self) -> float:
        total = self.limit_filled + self.limit_expired
        return self.limit_filled / total if total else 0.0

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
    ticks: bool = False,
    tick_tf: str = TICK_TF,
    last: int | None = None,
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

    # 틱 대용 프레임. 없으면 실행 시계를 재생할 수 없으므로 명시적으로 막는다.
    # 조용히 봉 모드로 떨어지면 "틱으로 검증했다"는 잘못된 결론이 남는다.
    tick_frames: dict[str, pd.DataFrame] = {}
    if ticks:
        for s in engine_symbols:
            df = cache.load(s.contract, tick_tf)
            if df.empty:
                raise SystemExit(
                    f"{s.contract} {tick_tf} 캐시가 없다. "
                    f"tools/fetch_history.py로 먼저 받을 것."
                )
            tick_frames[s.contract] = df

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

    # last는 "가장 최근 N봉"이다. 1분봉 캐시가 짧아 틱 검증은 최근 구간에서만
    # 가능하므로 필요하다. warmup은 지표 안정 구간이라 앞에서 잘라내는 값이고,
    # 여기서는 시작 지점을 뒤로 옮기는 용도로 재사용한다.
    if last is not None:
        if last >= total:
            raise SystemExit(f"last={last}가 전체 {total}봉보다 크다")
        warmup = max(warmup, total - last)

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
        client=ReplayClient(frames_by_symbol, tick_frames, tick_tf),
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

    tick_step = cd.interval_seconds(tick_tf) if ticks else 0

    if ticks:
        # **틱 프레임이 리플레이 구간을 덮는지 확인한다.**
        #
        # 안 덮으면 _partial_bar가 시가만 있는 한 점짜리 봉으로 떨어져
        # 틱 경로가 봉 경로와 똑같은 결과를 낸다. 그걸 "차이 없음"으로
        # 읽으면 검증했다고 착각하게 된다. 조용히 넘어가지 않는다.
        window_start = int(df["ts"].iloc[warmup])
        window_end = int(df["ts"].iloc[end - 1]) + clock_step
        for contract, tdf in tick_frames.items():
            if tdf.empty:
                raise SystemExit(f"{contract} {tick_tf} 프레임이 비었다")
            have_start = int(tdf["ts"].iloc[0])
            # 마지막 1분봉은 ts부터 ts+tick_step까지를 덮는다.
            have_end = int(tdf["ts"].iloc[-1]) + tick_step
            if have_start > window_start or have_end < window_end:
                raise SystemExit(
                    f"{contract} {tick_tf} 캐시가 리플레이 구간을 덮지 못한다.\n"
                    f"  필요: {pd.to_datetime(window_start, unit='s', utc=True)}"
                    f" ~ {pd.to_datetime(window_end, unit='s', utc=True)}\n"
                    f"  보유: {pd.to_datetime(have_start, unit='s', utc=True)}"
                    f" ~ {pd.to_datetime(have_end, unit='s', utc=True)}\n"
                    f"  1분봉은 거래소 보관 기간이 짧다. --last로 최근 구간만 돌릴 것."
                )

    for i in range(warmup, end):
        try:
            if live_like:
                bar_start = int(df["ts"].iloc[i])
                close_ts = bar_start + clock_step

                if ticks:
                    # 봉이 진행되는 동안의 실행 시계.
                    for t in range(bar_start + tick_step, close_ts, tick_step):
                        engine.run_tick(now_ts=t)
                    # 봉 마감 직후 한 번 더. 여기서 방금 닫힌 봉의 꼬리
                    # (마지막 틱 이후 구간)를 처리하고 covered_bar_ts를
                    # 올린다. 실시간 run()의 순서와 같다.
                    engine.run_tick(now_ts=close_ts + 1)

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
        limit_filled=engine.limit_stats["filled"],
        limit_expired=engine.limit_stats["expired"],
    )
