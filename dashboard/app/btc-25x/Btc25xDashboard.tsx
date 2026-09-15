"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  buildBtc25xPreview,
  enrichCandles,
  mergeRawCandles,
} from "@/lib/market.mjs";
import type { Btc25xPoint } from "@/lib/market.mjs";
import {
  AccountStrip,
  RegressionChart,
  TradeHistory,
  accountOf,
} from "../BtcDashboard";
import type {
  Candle,
  DashboardData,
  LiveStatus,
  Timeframe,
} from "../BtcDashboard";

type DeskTimeframe = "5m" | "15m" | "4h";

type LiveResponse = {
  serverTime: number;
  candles: Partial<Record<Timeframe, Candle[]>>;
  truncated?: boolean;
  error?: string;
};

const FRAME_TABS: { value: DeskTimeframe; label: string; detail: string }[] = [
  { value: "5m", label: "5분 체결", detail: "15분 채널을 5분축으로 연장" },
  { value: "15m", label: "15분 채널", detail: "48봉 · 약 12시간" },
  { value: "4h", label: "4시간 국면", detail: "EMA 20 / 60" },
];

const DEFAULT_CONFIG = {
  channelLength: 48,
  channelWidthMult: 2,
  channelR2Min: 0.68,
  channelMinMovePct: 0.004,
  maxEntryDistancePct: 0.005,
  limitTtlBars: 3,
  leverage: 25,
  marginFraction: 0.10,
  accountTakeProfitPct: 0.02,
  targetMarginRoe: 0.20,
  makerRate: -0.0001,
  maxDailyEntries: 12,
  consecutiveLossLimit: 3,
  dailyEntries: 0,
  consecutiveLosses: 0,
  pausedUntil: 0,
};

const price = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const money = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

function fmtPrice(value: number | null | undefined) {
  return value == null || !Number.isFinite(value) ? "—" : `$${price.format(value)}`;
}

function fmtPct(value: number | null | undefined, digits = 2) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value > 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;
}

function fmtKst(timestamp: number | null | undefined) {
  if (!timestamp) return "—";
  return new Intl.DateTimeFormat("ko-KR", {
    timeZone: "Asia/Seoul",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(timestamp * 1000));
}

function regimeLabel(regime: Btc25xPoint["majorRegime"]) {
  if (regime === "bull") return "상승장 · 롱만";
  if (regime === "bear") return "하락장 · 숏만";
  return "횡보장 · 거래 금지";
}

function statusCopy(point: Btc25xPoint) {
  if (point.status === "long_ready") return { eyebrow: "LONG LIMIT READY", title: "하단 채널 롱 지정가 조건 충족", tone: "positive" };
  if (point.status === "short_ready") return { eyebrow: "SHORT LIMIT READY", title: "상단 채널 숏 지정가 조건 충족", tone: "negative" };
  if (point.status === "warming") return { eyebrow: "WARMING UP", title: "채널 계산을 준비하는 중", tone: "neutral" };
  return { eyebrow: "NO ENTRY", title: point.blockReason, tone: "neutral" };
}

function GateRow({ done, title, detail }: { done: boolean; title: string; detail: string }) {
  return (
    <li className={done ? "done" : "blocked"}>
      <i>{done ? "✓" : "×"}</i>
      <div><b>{title}</b><span>{detail}</span></div>
    </li>
  );
}

export function Btc25xDashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [frames, setFrames] = useState<Record<DeskTimeframe, Candle[]> | null>(null);
  const [timeframe, setTimeframe] = useState<DeskTimeframe>("5m");
  const [liveStatus, setLiveStatus] = useState<LiveStatus>("connecting");
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [historyLoading, setHistoryLoading] = useState<Partial<Record<DeskTimeframe, boolean>>>({});
  const [historyExhausted, setHistoryExhausted] = useState<Partial<Record<DeskTimeframe, boolean>>>({});
  const frameRef = useRef(frames);
  const historyInFlight = useRef(new Set<DeskTimeframe>());
  const framesReady = frames !== null;

  useEffect(() => { frameRef.current = frames; }, [frames]);

  useEffect(() => {
    if (!expanded) return;
    const previousOverflow = document.body.style.overflow;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setExpanded(false);
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [expanded]);

  const enrichFrame = useCallback((raw: Candle[], target: DeskTimeframe) => (
    enrichCandles(raw, target === "15m" ? 48 : 100, 2, target) as Candle[]
  ), []);

  const loadOlder = useCallback(async (target: Timeframe) => {
    if (!["5m", "15m", "4h"].includes(target)) return;
    const tf = target as DeskTimeframe;
    if (historyInFlight.current.has(tf)) return;
    const current = frameRef.current?.[tf];
    if (!current?.length || current.length >= 5_000) {
      setHistoryExhausted((previous) => ({ ...previous, [tf]: true }));
      return;
    }
    historyInFlight.current.add(tf);
    setHistoryLoading((previous) => ({ ...previous, [tf]: true }));
    try {
      const before = current[0].time;
      const response = await fetch(`/api/btc-live?tf=${tf}&before=${before}&limit=800`, { cache: "no-store" });
      const payload = await response.json() as LiveResponse;
      if (!response.ok) throw new Error(payload.error ?? "과거 캔들 요청 실패");
      const older = (payload.candles[tf] ?? []).filter((candle) => candle.time < before);
      if (!older.length) {
        setHistoryExhausted((previous) => ({ ...previous, [tf]: true }));
        return;
      }
      setFrames((previous) => previous && ({
        ...previous,
        [tf]: enrichFrame(mergeRawCandles(previous[tf], older, 5_000) as Candle[], tf),
      }));
    } catch {
      // 사용자가 다시 이동하거나 버튼을 누르면 재시도한다.
    } finally {
      historyInFlight.current.delete(tf);
      setHistoryLoading((previous) => ({ ...previous, [tf]: false }));
    }
  }, [enrichFrame]);

  useEffect(() => {
    let disposed = false;
    fetch("/btc-dashboard.json")
      .then((response) => {
        if (!response.ok) throw new Error("대시보드 데이터를 읽지 못했습니다.");
        return response.json() as Promise<DashboardData>;
      })
      .then((snapshot) => {
        if (disposed) return;
        const initial = {} as Record<DeskTimeframe, Candle[]>;
        (["5m", "15m", "4h"] as const).forEach((tf) => {
          initial[tf] = enrichFrame(snapshot.candles[tf].map((candle) => ({ ...candle, closed: true })), tf);
        });
        setData(snapshot);
        setFrames(initial);

        void Promise.allSettled((["5m", "15m", "4h"] as const).map(async (tf) => {
          const after = initial[tf].at(-1)?.time;
          if (after == null) return;
          const response = await fetch(`/api/btc-live?tf=${tf}&after=${after}`, { cache: "no-store" });
          const payload = await response.json() as LiveResponse;
          if (!response.ok) throw new Error(payload.error ?? "누락 캔들 요청 실패");
          const updates = payload.candles[tf] ?? [];
          if (disposed || !updates.length) return;
          setFrames((previous) => {
            if (!previous) return previous;
            const raw = payload.truncated ? updates : mergeRawCandles(previous[tf], updates, 5_000);
            return { ...previous, [tf]: enrichFrame(raw as Candle[], tf) };
          });
        }));
      })
      .catch((reason: Error) => setError(reason.message));
    return () => { disposed = true; };
  }, [enrichFrame]);

  useEffect(() => {
    if (!framesReady) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const refresh = async () => {
      try {
        const response = await fetch("/api/btc-live?tf=5m", { cache: "no-store" });
        const payload = await response.json() as LiveResponse;
        if (!response.ok) throw new Error(payload.error ?? "실시간 시세 요청 실패");
        if (disposed) return;
        setFrames((previous) => {
          if (!previous) return previous;
          const next = { ...previous };
          (["5m", "15m", "4h"] as const).forEach((tf) => {
            const updates = payload.candles[tf] ?? [];
            if (updates.length) next[tf] = enrichFrame(mergeRawCandles(next[tf], updates) as Candle[], tf);
          });
          return next;
        });
        setLastUpdate(payload.serverTime);
        setLiveStatus("online");
      } catch {
        if (!disposed) setLiveStatus("stale");
      } finally {
        if (!disposed) timer = setTimeout(refresh, 3_000);
      }
    };
    void refresh();
    return () => {
      disposed = true;
      if (timer) clearTimeout(timer);
    };
  }, [framesReady, enrichFrame]);

  const config = data?.strategy25x ?? DEFAULT_CONFIG;
  const preview = useMemo(() => frames
    ? buildBtc25xPreview(frames["5m"], frames["15m"], frames["4h"], config)
    : null, [frames, config]);

  if (error) return <main className="center-state"><p>{error}</p></main>;
  if (!data || !frames || !preview) {
    return <main className="center-state"><div className="loading-mark" /><p>BTC 25× 전략 데스크를 준비하는 중</p></main>;
  }

  const point = preview.live;
  const confirmed = preview.confirmed;
  const copy = statusCopy(point);
  const confirmedCopy = statusCopy(confirmed);
  const selectedCandles = timeframe === "5m" ? preview.chart as Candle[] : frames[timeframe];
  const marketPrice = frames["5m"].at(-1)?.close ?? data.market.price;
  const previousPrice = frames["5m"].at(-2)?.close ?? marketPrice;
  const change = previousPrice ? marketPrice / previousPrice - 1 : 0;
  const direction = point.side === "long" ? "상승 채널" : point.side === "short" ? "하락 채널" : "판정 대기";
  const account = accountOf(data, "btc25x");
  const margin = account.balance * config.marginFraction;
  const notional = margin * config.leverage;
  const targetProfit = account.balance * config.accountTakeProfitPct;
  const estimatedStopLoss = account.balance * 0.0216;
  const positions = data.positions25x ?? [];
  const trades = (data.trades ?? []).filter((trade) => trade.strategy === "btc_regime_25x");
  const paused = config.pausedUntil > (lastUpdate ?? data.generatedAt);
  const gates = point.gates ?? {
    direction: false, channel: false, r2: false, move: false, priceInside: false, distance: false, regime: false,
  };

  return (
    <main className="app-shell desk-25x">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">₿</div>
          <div><span className="brand-name">PAPER / DESK</span><span className="brand-sub">REGIME CHANNEL · 25×</span></div>
        </div>
        <nav className="strategy-nav" aria-label="전략 메뉴">
          <Link className="nav-item" href="/">BTC 회귀채널</Link>
          <Link className="nav-item active" href="/btc-25x">BTC 25×</Link>
          <Link className="nav-item" href="/xrp-patterns">XRP 패턴</Link>
        </nav>
        <div className="system-status">
          <span className={`status-dot ${liveStatus}`} /><span>{liveStatus === "online" ? "실시간 시세" : liveStatus === "stale" ? "스냅샷 대체" : "시세 연결 중"}</span><small>PAPER · READ ONLY</small>
        </div>
      </header>

      <AccountStrip data={data} active="btc25x" />

      <section className="market-heading leverage-heading">
        <div>
          <div className="breadcrumb">BITCOIN <span>/</span> REGIME CHANNEL <em>25× ISOLATED</em></div>
          <h1>단기 국면 채널</h1>
          <p>4시간 대세와 15분 채널이 같은 방향일 때, 5분봉 경계 지정가만 기다립니다.</p>
        </div>
        <div className="quote-block">
          <span className="quote-symbol">BTC / USDT <em>PERP</em></span>
          <strong>{fmtPrice(marketPrice)}</strong>
          <span className={change >= 0 ? "quote-change up" : "quote-change down"}>{fmtPct(change)} · 현재 5분봉</span>
        </div>
      </section>

      <section className="workspace-grid">
        <article className={`panel chart-panel ${expanded ? "expanded" : ""}`}>
          <div className="panel-header chart-toolbar">
            <div className="timeframe-tabs desk-tabs" role="tablist" aria-label="BTC 25배 시간축 선택">
              {FRAME_TABS.map((item) => (
                <button key={item.value} type="button" role="tab" aria-selected={timeframe === item.value} className={timeframe === item.value ? "active" : ""} onClick={() => setTimeframe(item.value)}>
                  {item.label}<small>{item.detail}</small>
                </button>
              ))}
            </div>
            <div className="indicator-toggles">
              <span className="strategy-formula">48 REG · R² ≥ 0.68 · MOVE ≥ 0.40%</span>
              <button type="button" className="screen-toggle" aria-pressed={expanded} onClick={() => setExpanded((value) => !value)}>{expanded ? "화면 축소" : "화면 확대"}</button>
            </div>
          </div>
          <RegressionChart
            key={timeframe}
            candles={selectedCandles}
            timeframe={timeframe}
            showChannel={timeframe !== "4h"}
            showAverages={timeframe === "4h"}
            expanded={expanded}
            historyLoading={historyLoading[timeframe] === true}
            historyExhausted={historyExhausted[timeframe] === true}
            onLoadOlder={loadOlder}
            trades={trades}
          />
          <footer className="chart-footer">
            <span><i className="legend-candle up" /> 상승</span><span><i className="legend-candle down" /> 하락</span>
            {timeframe === "4h" ? <span><i className="legend-line regime" /> EMA 20 / 60</span> : <span><i className="legend-line channel" /> 15분 회귀 상·하단</span>}
            <span className="chart-note"><b>{FRAME_TABS.find((item) => item.value === timeframe)?.detail}</b> · 결측 자동 보정 · 3초 갱신</span>
          </footer>
        </article>

        <aside className="panel decision-panel leverage-decision">
          <div className="decision-topline"><span>25× STRATEGY PULSE</span><span className={`snapshot-badge ${liveStatus}`}>LIVE PREVIEW</span></div>
          <div className={`signal-state ${copy.tone}`}>
            <span>{copy.eyebrow}</span><h2>{copy.title}</h2>
            <p>진행 중 5분봉 예상 · 실제 확정봉: {confirmedCopy.eyebrow}</p>
          </div>
          <div className="leverage-stamp"><strong>25×</strong><span>ISOLATED</span><small>잔고의 10% 증거금</small></div>

          <div className="decision-metrics leverage-metrics">
            <div><span>4H 대세</span><strong>{regimeLabel(point.majorRegime)}</strong><small>확정 EMA20 / EMA60</small></div>
            <div><span>15M 방향</span><strong>{direction}</strong><small>{fmtPct((point.slope ?? 0) / Math.max(point.mid ?? 1, 1), 3)} / 15분봉</small></div>
            <div><span>채널 R²</span><strong>{fmtPct(point.r2 ?? 0, 1)}</strong><small>기준 {fmtPct(config.channelR2Min, 0)}</small></div>
            <div><span>12H 이동폭</span><strong>{fmtPct(point.movePct ?? 0, 2)}</strong><small>기준 +0.40%</small></div>
          </div>

          <div className="order-map">
            <div className="order-map-head"><span>ORDER MAP</span><b>{point.side?.toUpperCase() ?? "WAIT"}</b></div>
            <div><span>지정가 진입</span><b>{fmtPrice(point.entry)}</b><em>채널 {point.side === "long" ? "하단" : "상단"}</em></div>
            <div><span>익절 주문</span><b className="up">{fmtPrice(point.target)}</b><em>{fmtPct(point.targetMovePct)} 가격폭</em></div>
            <div><span>손절 주문</span><b className="down">{fmtPrice(point.stop)}</b><em>익절과 대칭</em></div>
            <div><span>예상 청산가</span><b>{fmtPrice(point.liquidation)}</b><em>유지증거금 0.3%</em></div>
          </div>

          <div className="gate-list">
            <span className="section-kicker">ENTRY GATES</span>
            <ol>
              <GateRow done={gates.channel} title="채널 계산 유효성" detail={`ATR 14 ${fmtPrice(selectedCandles.at(-1)?.atr)}`} />
              <GateRow done={gates.r2} title="채널 적합도" detail={`${fmtPct(point.r2 ?? 0, 1)} / 최소 68.0%`} />
              <GateRow done={gates.move} title="방향 이동폭" detail={`${fmtPct(point.movePct ?? 0)} / 최소 0.40%`} />
              <GateRow done={gates.distance && gates.priceInside} title="경계 지정가 거리" detail={`${fmtPct(point.entryDistancePct ?? 0)} / 최대 0.50%`} />
              <GateRow done={gates.regime} title="4H 대세 정렬" detail={regimeLabel(point.majorRegime)} />
            </ol>
          </div>
        </aside>
      </section>

      <section className="account-section leverage-account">
        <div className="section-heading">
          <div><span>RISK / POSITION</span><h2>25× 주문 계획</h2></div>
          <p>계좌 {fmtKst(data.generatedAt)} 스냅샷 · 시세 {fmtKst(lastUpdate)}</p>
        </div>
        <div className="metric-grid">
          <article className="metric-card"><span className="metric-label">격리 증거금</span><strong>{money.format(margin)} USDT</strong><span className="metric-detail">잔고의 10%</span></article>
          <article className="metric-card"><span className="metric-label">예상 명목가치</span><strong>{money.format(notional)} USDT</strong><span className="metric-detail">증거금 × 25</span></article>
          <article className="metric-card positive"><span className="metric-label">목표 계좌 이익</span><strong>+{money.format(targetProfit)} USDT</strong><span className="metric-detail">계좌 +2.00%</span></article>
          <article className="metric-card negative"><span className="metric-label">예상 손절 손실</span><strong>−{money.format(estimatedStopLoss)} USDT</strong><span className="metric-detail">시장가 비용 포함 약 −2.16%</span></article>
        </div>

        <article className="panel safety-panel">
          <div><span>오늘 진입</span><strong>{config.dailyEntries} / {config.maxDailyEntries}</strong><small>UTC 기준</small></div>
          <div><span>연속 손실</span><strong>{config.consecutiveLosses} / {config.consecutiveLossLimit}</strong><small>3회면 다음 UTC일까지 정지</small></div>
          <div><span>전략 상태</span><strong className={paused ? "down" : "up"}>{paused ? "PAUSED" : "ACTIVE"}</strong><small>{paused ? `${fmtKst(config.pausedUntil)}까지` : "진입 가능"}</small></div>
          <div><span>BTC 25× 포지션</span><strong>{positions.length}건</strong><small>종목 독점 적용</small></div>
        </article>

        <article className="panel positions-panel">
          <div className="positions-header"><div><span className="section-kicker">OPEN POSITION</span><h3>BTC 25× 격리 포지션</h3></div><span className="count-badge">{positions.length}</span></div>
          {positions.length === 0 ? (
            <div className="empty-position"><div className="empty-radar"><span /></div><div><strong>현재 25× 포지션이 없습니다</strong><p>{point.blockReason}. 조건이 모두 맞으면 5분봉 채널 경계에 지정가를 둡니다.</p></div><span className="waiting-chip">{point.status === "waiting" ? "GATE WAITING" : "LIMIT READY"}</span></div>
          ) : (
            <div className="position-table-wrap"><table><thead><tr><th>방향</th><th>평단가</th><th>현재가</th><th>레버리지 ROE</th><th>손절</th><th>익절</th><th>계약</th></tr></thead><tbody>{positions.map((position, index) => {
              const directionValue = position.side === "long" ? 1 : -1;
              const roe = ((marketPrice - position.entry) / position.entry) * directionValue * position.leverage * 100;
              return <tr key={`${position.symbol}-${index}`}><td><span className={`side-chip ${position.side}`}>{position.side.toUpperCase()}</span></td><td>{fmtPrice(position.entry)}</td><td>{fmtPrice(marketPrice)}</td><td className={roe >= 0 ? "up" : "down"}>{roe > 0 ? "+" : ""}{roe.toFixed(2)}%</td><td>{fmtPrice(position.stop)}</td><td>{fmtPrice(position.tp1)}</td><td>{position.contracts.toLocaleString()}</td></tr>;
            })}</tbody></table></div>
          )}
        </article>
      </section>

      <TradeHistory trades={trades} generatedAt={data.generatedAt} initialSymbol="BTC" />

      <footer className="page-footer"><span>DATA · Gate.io live public candles + AWS paper state</span><span>읽기 전용 화면 · 주문 기능 없음</span></footer>
    </main>
  );
}
