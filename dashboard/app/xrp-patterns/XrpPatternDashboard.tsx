"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { enrichXrpCandles, mergeRawCandles } from "@/lib/market.mjs";
import { AccountStrip, RegressionChart, TradeHistory, accountOf } from "../BtcDashboard";
import type {
  Candle,
  DashboardData,
  LiveStatus,
  Timeframe,
  XrpPatternMonitor,
  XrpSignalEvent,
} from "../BtcDashboard";

type XrpFrame = "5m" | "15m" | "1h" | "4h" | "1d";
type DeskFrame = Exclude<XrpFrame, "1d">;

type LiveResponse = {
  serverTime: number;
  candles: Partial<Record<XrpFrame, Candle[]>>;
  btcContext?: Candle[];
  truncated?: boolean;
  error?: string;
};

const FRAME_TABS: { value: DeskFrame; label: string; detail: string }[] = [
  { value: "5m", label: "5분 패턴", detail: "쐐기 · 헤드앤숄더 · 컵핸들" },
  { value: "15m", label: "15분 패턴", detail: "쐐기 · 컵핸들" },
  { value: "1h", label: "1시간 패턴", detail: "이중천장 · 코테가와 · 컵핸들" },
  { value: "4h", label: "4시간 패턴", detail: "컵핸들 장기 구조" },
];

const CONTEXT_FRAME: Record<DeskFrame, XrpFrame> = {
  "5m": "1h",
  "15m": "4h",
  "1h": "4h",
  "4h": "1d",
};

const price = new Intl.NumberFormat("en-US", { minimumFractionDigits: 4, maximumFractionDigits: 4 });
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

function familyLabel(family: string) {
  const labels: Record<string, string> = {
    cup_handle: "컵앤핸들",
    double_top: "이중천장·바닥",
    wedge: "수렴 쐐기",
    head_shoulders: "헤드앤숄더",
    kotegawa: "코테가와 이격도",
    triangle: "대칭 삼각수렴",
  };
  return labels[family] ?? family.replaceAll("_", " ");
}

function eventFamily(strategy: string) {
  return strategy.replace("xrp_patterns.", "").replace(/\.(5m|15m|1h|4h)$/, "");
}

function lastEventFor(monitor: XrpPatternMonitor, events: XrpSignalEvent[]) {
  const base = monitor.name.replace(/\.(5m|15m|1h|4h)$/, "");
  return events.find((event) => event.strategy === monitor.name || event.strategy === base);
}

function eventIsActive(event: XrpSignalEvent, monitors: XrpPatternMonitor[]) {
  return monitors.some((monitor) => (
    monitor.name === event.strategy
    || monitor.name.replace(/\.(5m|15m|1h|4h)$/, "") === event.strategy
  ));
}

function reasonLabel(reason: string) {
  const labels: Record<string, string> = {
    double_top: "쌍봉 천장",
    double_bottom: "쌍바닥",
    rising_wedge: "상승 쐐기 하단 이탈",
    falling_wedge: "하락 쐐기 상단 이탈",
    head_shoulders: "헤드앤숄더",
    inverse_head_shoulders: "역헤드앤숄더",
    cup_handle: "컵앤핸들",
    inverse_cup_handle: "역컵앤핸들",
    kotegawa: "이평선 이격도",
    prior_trend: "선행 추세",
    volume_dry_up: "거래량 수축",
    vwap_aligned: "VWAP 정렬",
    btc_aligned: "BTC 방향 일치",
    no_wall_ahead: "전방 매물벽 없음",
    htf_level: "상위봉 레벨",
    retested: "리테스트",
    max_tranches_reached: "이미 포지션 보유",
    stop_too_tight: "손절폭 부족",
    exit_plan_failed: "손익비 미달",
  };
  if (labels[reason]) return labels[reason];
  if (reason.startsWith("rvol_")) return `RVOL ${reason.slice(5)}×`;
  if (reason.startsWith("dev_")) return `이격 ${reason.slice(4)}`;
  return reason.replaceAll("_", " ");
}

function regimeOf(candle: Candle | undefined) {
  if (!candle || candle.sma20 == null || candle.sma50 == null) return "none" as const;
  if (candle.close > candle.sma20 && candle.sma20 > candle.sma50) return "bull" as const;
  if (candle.close < candle.sma20 && candle.sma20 < candle.sma50) return "bear" as const;
  return "none" as const;
}

function regimeCopy(regime: "bull" | "bear" | "none") {
  if (regime === "bull") return "상승 정렬";
  if (regime === "bear") return "하락 정렬";
  return "중립·혼조";
}

function eventStatus(event: XrpSignalEvent | undefined) {
  if (!event) return { label: "SCANNING", tone: "neutral" };
  if (event.action === "entry") return { label: "LAST ENTRY", tone: event.side === "long" ? "positive" : "negative" };
  return { label: "LAST BLOCK", tone: "blocked" };
}

export function XrpPatternDashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [frames, setFrames] = useState<Record<XrpFrame, Candle[]> | null>(null);
  const [btcContext, setBtcContext] = useState<Candle[]>([]);
  const [timeframe, setTimeframe] = useState<DeskFrame>("5m");
  const [liveStatus, setLiveStatus] = useState<LiveStatus>("connecting");
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [historyLoading, setHistoryLoading] = useState<Partial<Record<XrpFrame, boolean>>>({});
  const [historyExhausted, setHistoryExhausted] = useState<Partial<Record<XrpFrame, boolean>>>({});
  const frameRef = useRef(frames);
  const historyInFlight = useRef(new Set<XrpFrame>());
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

  const enrichFrame = useCallback((raw: Candle[], target: XrpFrame) => (
    enrichXrpCandles(raw, target) as Candle[]
  ), []);

  const loadOlder = useCallback(async (target: Timeframe) => {
    if (!["5m", "15m", "1h", "4h", "1d"].includes(target)) return;
    const tf = target as XrpFrame;
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
      const response = await fetch(`/api/xrp-live?tf=${tf}&before=${before}&limit=800`, { cache: "no-store" });
      const payload = await response.json() as LiveResponse;
      if (!response.ok) throw new Error(payload.error ?? "XRP 과거 캔들 요청 실패");
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
      // 차트가 과거 끝에 다시 닿으면 재시도한다.
    } finally {
      historyInFlight.current.delete(tf);
      setHistoryLoading((previous) => ({ ...previous, [tf]: false }));
    }
  }, [enrichFrame]);

  useEffect(() => {
    let disposed = false;
    fetch("/btc-dashboard.json")
      .then((response) => {
        if (!response.ok) throw new Error("XRP 대시보드 데이터를 읽지 못했습니다.");
        return response.json() as Promise<DashboardData>;
      })
      .then((snapshot) => {
        if (disposed) return;
        if (!snapshot.xrpCandles || !snapshot.xrpMarket) throw new Error("XRP 스냅샷이 아직 생성되지 않았습니다.");
        const initial = {} as Record<XrpFrame, Candle[]>;
        (["5m", "15m", "1h", "4h", "1d"] as const).forEach((tf) => {
          initial[tf] = enrichFrame(snapshot.xrpCandles![tf].map((candle) => ({ ...candle, closed: true })), tf);
        });
        setData(snapshot);
        setFrames(initial);

        void Promise.allSettled(([
          "5m", "15m", "1h", "4h", "1d",
        ] as const).map(async (tf) => {
          const after = initial[tf].at(-1)?.time;
          if (after == null) return;
          const response = await fetch(`/api/xrp-live?tf=${tf}&after=${after}`, { cache: "no-store" });
          const payload = await response.json() as LiveResponse;
          if (!response.ok) throw new Error(payload.error ?? "XRP 누락 캔들 요청 실패");
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
        const response = await fetch("/api/xrp-live?tf=5m", { cache: "no-store" });
        const payload = await response.json() as LiveResponse;
        if (!response.ok) throw new Error(payload.error ?? "XRP 실시간 시세 요청 실패");
        if (disposed) return;
        setFrames((previous) => {
          if (!previous) return previous;
          const next = { ...previous };
          (["5m", "15m", "1h", "4h", "1d"] as const).forEach((tf) => {
            const updates = payload.candles[tf] ?? [];
            if (updates.length) next[tf] = enrichFrame(mergeRawCandles(next[tf], updates) as Candle[], tf);
          });
          return next;
        });
        if (payload.btcContext?.length) setBtcContext(payload.btcContext);
        setLastUpdate(payload.serverTime);
        setLiveStatus("online");
      } catch {
        if (!disposed) setLiveStatus("stale");
      } finally {
        if (!disposed) timer = setTimeout(refresh, 5_000);
      }
    };
    void refresh();
    return () => {
      disposed = true;
      if (timer) clearTimeout(timer);
    };
  }, [framesReady, enrichFrame]);

  if (error) return <main className="center-state"><p>{error}</p></main>;
  if (!data || !frames || !data.xrpMarket) {
    return <main className="center-state"><div className="loading-mark" /><p>XRP 패턴 스캐너를 준비하는 중</p></main>;
  }

  const current = frames[timeframe].at(-1);
  const marketPrice = frames["5m"].at(-1)?.close ?? data.xrpMarket.price;
  const previousPrice = frames["5m"].at(-2)?.close ?? marketPrice;
  const change = previousPrice ? marketPrice / previousPrice - 1 : 0;
  const contextTf = CONTEXT_FRAME[timeframe];
  const contextCandle = [...frames[contextTf]].reverse().find((candle) => candle.closed !== false);
  const regime = regimeOf(contextCandle);
  const btcCandle = [...btcContext].reverse().find((candle) => candle.closed !== false) ?? btcContext.at(-1);
  const btcMove = btcCandle?.open ? btcCandle.close / btcCandle.open - 1 : null;
  const btcAdverse = btcMove != null && Math.abs(btcMove) >= 0.01;
  const monitors = data.xrpPatterns ?? [];
  const events = data.xrpSignals ?? [];
  const selectedMonitors = monitors.filter((monitor) => monitor.timeframe === timeframe);
  const trades = (data.trades ?? []).filter((trade) => trade.symbol === "XRP_USDT");
  const positions = data.xrpPositions ?? [];
  const account = accountOf(data, "xrp");
  const wins = trades.filter((trade) => trade.pnl > 0).length;
  const pnl = trades.reduce((sum, trade) => sum + trade.pnl, 0);
  const latestEntry = events.find((event) => event.action === "entry" && eventIsActive(event, monitors));

  return (
    <main className="app-shell xrp-desk">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark xrp-mark">X</div>
          <div><span className="brand-name">PAPER / DESK</span><span className="brand-sub">XRP PATTERN SCANNER</span></div>
        </div>
        <nav className="strategy-nav" aria-label="전략 메뉴">
          <Link className="nav-item" href="/">BTC 회귀채널</Link>
          <Link className="nav-item" href="/btc-25x">BTC 25×</Link>
          <Link className="nav-item active" href="/xrp-patterns">XRP 패턴</Link>
        </nav>
        <div className="system-status">
          <span className={`status-dot ${liveStatus}`} /><span>{liveStatus === "online" ? "실시간 시세" : liveStatus === "stale" ? "스냅샷 대체" : "시세 연결 중"}</span><small>PAPER · READ ONLY</small>
        </div>
      </header>

      <AccountStrip data={data} active="xrp" />

      <section className="market-heading xrp-heading">
        <div>
          <div className="breadcrumb">RIPPLE <span>/</span> MULTI PATTERN <em>{monitors.length} ACTIVE SCANNERS</em></div>
          <h1>패턴 스캐너</h1>
          <p>완성된 모양이 아니라 종가 돌파와 품질 점수를 통과한 패턴만 주문 후보가 됩니다.</p>
        </div>
        <div className="quote-block">
          <span className="quote-symbol">XRP / USDT <em>PERP</em></span>
          <strong>{fmtPrice(marketPrice)}</strong>
          <span className={change >= 0 ? "quote-change up" : "quote-change down"}>{fmtPct(change)} · 현재 5분봉</span>
        </div>
      </section>

      <section className="workspace-grid xrp-workspace">
        <article className={`panel chart-panel ${expanded ? "expanded" : ""}`}>
          <div className="panel-header chart-toolbar">
            <div className="timeframe-tabs desk-tabs xrp-tabs" role="tablist" aria-label="XRP 패턴 시간축 선택">
              {FRAME_TABS.map((item) => (
                <button key={item.value} type="button" role="tab" aria-selected={timeframe === item.value} className={timeframe === item.value ? "active" : ""} onClick={() => setTimeframe(item.value)}>
                  {item.label}<small>{item.detail}</small>
                </button>
              ))}
            </div>
            <div className="indicator-toggles">
              <span className="strategy-formula">SMA 20/50 · VWAP · 확정 PIVOT 5</span>
              <button type="button" className="screen-toggle" aria-pressed={expanded} onClick={() => setExpanded((value) => !value)}>{expanded ? "화면 축소" : "화면 확대"}</button>
            </div>
          </div>
          <RegressionChart
            key={timeframe}
            candles={frames[timeframe]}
            timeframe={timeframe}
            showChannel={false}
            showAverages
            averageMode="sma20/50"
            showVwap
            showPivots
            instrumentLabel="리플"
            priceDigits={4}
            expanded={expanded}
            historyLoading={historyLoading[timeframe] === true}
            historyExhausted={historyExhausted[timeframe] === true}
            onLoadOlder={loadOlder}
            trades={trades}
          />
          <footer className="chart-footer">
            <span><i className="legend-candle up" /> 상승</span><span><i className="legend-candle down" /> 하락</span>
            <span><i className="legend-line fast-ma" /> SMA 20</span><span><i className="legend-line slow-ma" /> SMA 50</span><span><i className="legend-line channel" /> VWAP</span>
            <span className="chart-note"><b>엔진과 동일한 5봉 확정 피벗</b> · 임의 연결선 없음</span>
          </footer>
          <div className="xrp-indicator-strip">
            <div><span>RSI 14</span><strong>{current?.rsi14 == null ? "—" : current.rsi14.toFixed(1)}</strong><small>{(current?.rsi14 ?? 50) <= 30 ? "과매도" : (current?.rsi14 ?? 50) >= 70 ? "과매수" : "중립"}</small></div>
            <div><span>RVOL 20</span><strong>{current?.rvol == null ? "—" : `${current.rvol.toFixed(2)}×`}</strong><small>돌파 가점 {current?.rvol != null && current.rvol >= 1.3 ? "충족" : "1.30× 필요"}</small></div>
            <div><span>VWAP 거리</span><strong>{current?.vwap == null ? "—" : fmtPct(current.close / current.vwap - 1)}</strong><small>{current?.vwap != null && current.close >= current.vwap ? "VWAP 위" : "VWAP 아래"}</small></div>
            <div><span>ATR 14</span><strong>{current?.atr == null ? "—" : fmtPct(current.atr / current.close)}</strong><small>손절 버퍼 계산</small></div>
            <div><span>25MA 이격</span><strong>{fmtPct(current?.deviation)}</strong><small>코테가와 기준 ±0.60%</small></div>
            <div><span>이격 순위</span><strong>{current?.deviationRank == null ? "—" : `${current.deviationRank.toFixed(1)}%`}</strong><small>최근 400봉 분포</small></div>
          </div>
        </article>

        <aside className="panel decision-panel pattern-radar">
          <div className="decision-topline"><span>PATTERN RADAR</span><span className={`snapshot-badge ${liveStatus}`}>ENGINE MAP</span></div>
          <div className="signal-state neutral">
            <span>SCANNING {timeframe.toUpperCase()}</span><h2>{selectedMonitors.length}개 패턴 인스턴스 감시 중</h2>
            <p>실제 주문은 AWS 엔진의 확정봉 판정에서만 생성됩니다.</p>
          </div>
          <div className="decision-metrics pattern-context-grid">
            <div><span>{contextTf.toUpperCase()} 문맥</span><strong>{regimeCopy(regime)}</strong><small>SMA20 / SMA50</small></div>
            <div><span>BTC 1H</span><strong className={btcAdverse ? "down" : "up"}>{fmtPct(btcMove)}</strong><small>±1% 역방향 차단</small></div>
            <div><span>변동성 회로</span><strong className={current?.spike ? "down" : "up"}>{current?.spike ? "COOLDOWN" : "NORMAL"}</strong><small>평소 범위 5배 감지</small></div>
            <div><span>공통 품질</span><strong>4 / 12</strong><small>최소 통과 항목</small></div>
          </div>
          <div className="scanner-list">
            <div className="scanner-list-head"><span>ACTIVE ON {timeframe.toUpperCase()}</span><small>최근 AWS 이벤트</small></div>
            {selectedMonitors.map((monitor) => {
              const event = lastEventFor(monitor, events);
              const status = eventStatus(event);
              return (
                <div className="scanner-row" key={monitor.name}>
                  <i>{monitor.orderType === "limit" ? "LMT" : "MKT"}</i>
                  <div><b>{familyLabel(monitor.family)}</b><span>{monitor.contextTimeframe} 문맥 → {monitor.confirmTimeframe} 확인</span></div>
                  <em className={status.tone}>{status.label}</em>
                  <small>{event ? fmtKst(event.time) : "기록 없음"}</small>
                </div>
              );
            })}
          </div>
          <div className="last-engine-event">
            <span className="section-kicker">LAST CONFIRMED ENTRY</span>
            {latestEntry ? <>
              <div><b className={latestEntry.side === "long" ? "up" : "down"}>{latestEntry.side?.toUpperCase()}</b><strong>{familyLabel(eventFamily(latestEntry.strategy))}</strong><time>{fmtKst(latestEntry.time)}</time></div>
              <p>{latestEntry.reasons.slice(0, 4).map(reasonLabel).join(" · ")}</p>
            </> : <p>아직 기록된 XRP 진입 신호가 없습니다.</p>}
          </div>
        </aside>
      </section>

      <section className="account-section pattern-fleet-section">
        <div className="section-heading"><div><span>ENGINE / FLEET</span><h2>활성 패턴 구성</h2></div><p>삼각수렴은 현재 손실 검증 결과로 비활성 · 코드만 보존</p></div>
        <article className="panel pattern-fleet">
          <div className="fleet-header"><span>패턴</span><span>진입봉</span><span>문맥 → 확인</span><span>주문</span><span>추가 필터</span><span>최근 이벤트</span></div>
          {monitors.map((monitor) => {
            const event = lastEventFor(monitor, events);
            return <div className="fleet-row" key={monitor.name}>
              <div><strong>{familyLabel(monitor.family)}</strong><small>{monitor.name}</small></div>
              <b>{monitor.timeframe}</b><span>{monitor.contextTimeframe} → {monitor.confirmTimeframe}</span>
              <em className={monitor.orderType === "limit" ? "limit" : "market"}>{monitor.orderType.toUpperCase()}</em>
              <span>{[monitor.trendAlign ? "4H 추세정렬" : "공통필터", monitor.blockedUtcHours.length ? "UTC 16–24 차단" : null].filter(Boolean).join(" · ")}</span>
              <span>{event ? `${event.action === "entry" ? "진입" : reasonLabel(event.skipReason ?? "skip")} · ${fmtKst(event.time)}` : "기록 없음"}</span>
            </div>;
          })}
          <div className="fleet-row disabled-row"><div><strong>대칭 삼각수렴</strong><small>xrp_patterns.triangle</small></div><b>OFF</b><span>재검증 대기</span><em>DISABLED</em><span>949일 손실 −130.24</span><span>신규 주문 없음</span></div>
        </article>
      </section>

      <section className="account-section xrp-account">
        <div className="section-heading"><div><span>ACCOUNT / SIGNALS</span><h2>XRP 페이퍼 상태</h2></div><p>계좌 {fmtKst(data.generatedAt)} · 시세 {fmtKst(lastUpdate)}</p></div>
        <div className="metric-grid">
          <article className="metric-card"><span className="metric-label">계좌 잔고</span><strong>{money.format(account.balance)} USDT</strong><span className="metric-detail">XRP 패턴 전용 페이퍼 계좌</span></article>
          <article className="metric-card"><span className="metric-label">XRP 오픈 포지션</span><strong>{positions.length}건</strong><span className="metric-detail">패턴별 1회 진입</span></article>
          <article className="metric-card"><span className="metric-label">완료 거래 / 승률</span><strong>{trades.length} / {trades.length ? `${(wins / trades.length * 100).toFixed(1)}%` : "—"}</strong><span className="metric-detail">AWS 실제 페이퍼 로그</span></article>
          <article className={`metric-card ${pnl >= 0 ? "positive" : "negative"}`}><span className="metric-label">XRP 누적 손익</span><strong>{pnl >= 0 ? "+" : ""}{money.format(pnl)} USDT</strong><span className="metric-detail">진입 비용 포함</span></article>
        </div>

        <article className="panel positions-panel">
          <div className="positions-header"><div><span className="section-kicker">OPEN POSITION</span><h3>XRP 패턴 포지션</h3></div><span className="count-badge">{positions.length}</span></div>
          {positions.length === 0 ? <div className="empty-position"><div className="empty-radar"><span /></div><div><strong>현재 XRP 포지션이 없습니다</strong><p>패턴 완성·공통 필터·리스크 계산을 모두 통과한 주문을 기다립니다.</p></div><span className="waiting-chip">PATTERN SCANNING</span></div> :
            <div className="position-table-wrap"><table><thead><tr><th>패턴</th><th>방향</th><th>평단가</th><th>현재가</th><th>손절</th><th>1차 목표</th><th>레버리지</th></tr></thead><tbody>{positions.map((position, index) => <tr key={`${position.strategy}-${index}`}><td>{familyLabel(eventFamily(position.strategy))}</td><td><span className={`side-chip ${position.side}`}>{position.side.toUpperCase()}</span></td><td>{fmtPrice(position.entry)}</td><td>{fmtPrice(marketPrice)}</td><td>{fmtPrice(position.stop)}</td><td>{fmtPrice(position.tp1)}</td><td>{position.leverage}×</td></tr>)}</tbody></table></div>}
        </article>

        <article className="panel signal-log-panel">
          <header><div><span className="section-kicker">SIGNAL / BLOCK LOG</span><h3>최근 판단 로그</h3></div><small>진입뿐 아니라 차단된 이유까지 표시</small></header>
          <div className="signal-log-list">{events.slice(0, 12).map((event, index) => {
            const active = eventIsActive(event, monitors);
            return <div className="signal-log-row" key={`${event.time}-${event.strategy}-${index}`}>
              <time>{fmtKst(event.time)}</time><span>{familyLabel(eventFamily(event.strategy))}{!active ? <i>LEGACY / OFF</i> : null}</span>
              <b className={event.side === "long" ? "up" : "down"}>{event.side?.toUpperCase() ?? "—"}</b>
              <em className={event.action === "entry" ? "entry" : "skip"}>{event.action === "entry" ? "진입" : "차단"}</em>
              <p>{event.action === "entry" ? event.reasons.slice(0, 4).map(reasonLabel).join(" · ") : reasonLabel(event.skipReason ?? event.detail ?? "skip")}</p>
            </div>;
          })}</div>
        </article>
      </section>

      <TradeHistory trades={trades} generatedAt={data.generatedAt} initialSymbol="XRP" />
      <footer className="page-footer"><span>DATA · Gate.io live XRP/BTC candles + AWS paper state</span><span>패턴선은 관찰용 · 주문 판단은 AWS 확정봉 엔진</span></footer>
    </main>
  );
}
