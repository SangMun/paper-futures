"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  buildLiveStrategy,
  enrichCandles,
  mergeRawCandles,
} from "@/lib/market.mjs";
import type { StrategyPoint } from "@/lib/market.mjs";

type Timeframe = "1m" | "5m" | "15m" | "1h" | "4h" | "1d";

type Candle = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  mid: number | null;
  upper: number | null;
  lower: number | null;
  slope: number | null;
  r2: number | null;
  ema20: number | null;
  ema60: number | null;
  atr: number | null;
  closed?: boolean;
};

type LiveStatus = "connecting" | "online" | "stale";

type LiveResponse = {
  serverTime: number;
  source: string;
  candles: Partial<Record<Timeframe, Candle[]>>;
  mode?: "live" | "history";
  error?: string;
};

type Position = {
  symbol: string;
  side: "long" | "short";
  strategy: string;
  entry: number;
  contracts: number;
  leverage: number;
  stop: number;
  tp1: number;
  remaining_ratio: number;
  tp1_filled: boolean;
};

type DashboardData = {
  generatedAt: number;
  source: string;
  market: {
    symbol: string;
    displaySymbol: string;
    price: number;
    changePct: number;
  };
  strategy: {
    signal: "waiting" | "long_watch" | "short_watch";
    side: "long" | "short" | null;
    entry: number | null;
    target: number | null;
    slope: number | null;
    r2: number | null;
    atr: number | null;
    upper: number | null;
    lower: number | null;
    majorRegime: "bull" | "bear" | "none";
    entryOffsetPct: number;
    channelLength: number;
    channelWidthMult: number;
    executionTimeframe: string;
    contextTimeframe: string;
  };
  account: {
    balance: number;
    dayStartBalance: number;
    dayKey: string;
  };
  positions: Position[];
  candles: Record<Timeframe, Candle[]>;
};

const TIMEFRAMES: { value: Timeframe; label: string }[] = [
  { value: "1m", label: "1분" },
  { value: "5m", label: "5분" },
  { value: "15m", label: "15분" },
  { value: "1h", label: "1시간" },
  { value: "4h", label: "4시간" },
  { value: "1d", label: "1일" },
];

const VISIBLE_BARS: Record<Timeframe, number> = {
  "1m": 180,
  "5m": 160,
  "15m": 145,
  "1h": 130,
  "4h": 120,
  "1d": 110,
};

const priceFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const balanceFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function formatPrice(value: number | null | undefined) {
  return value == null ? "—" : `$${priceFormatter.format(value)}`;
}

function formatPercent(value: number | null | undefined, digits = 2) {
  if (value == null || !Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}%`;
}

function formatKst(timestamp: number) {
  return new Intl.DateTimeFormat("ko-KR", {
    timeZone: "Asia/Seoul",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(timestamp * 1000));
}

function chartTime(timestamp: number, timeframe: Timeframe) {
  return new Intl.DateTimeFormat("ko-KR", {
    timeZone: "Asia/Seoul",
    month: "numeric",
    day: "numeric",
    ...(timeframe !== "1d" ? { hour: "2-digit", minute: "2-digit", hour12: false } : {}),
  }).format(new Date(timestamp * 1000));
}

function regimeLabel(regime: DashboardData["strategy"]["majorRegime"]) {
  if (regime === "bull") return "상승 국면";
  if (regime === "bear") return "하락 국면";
  return "중립 국면";
}

function signalCopy(signal: DashboardData["strategy"]["signal"]) {
  if (signal === "long_watch") {
    return { eyebrow: "LONG SETUP", title: "하단 이탈 감지 · 롱 지정가 대기", tone: "positive" };
  }
  if (signal === "short_watch") {
    return { eyebrow: "SHORT SETUP", title: "상단 이탈 감지 · 숏 지정가 대기", tone: "negative" };
  }
  return { eyebrow: "NO ENTRY", title: "채널 과확장 신호를 기다리는 중", tone: "neutral" };
}

function connectionCopy(status: LiveStatus) {
  if (status === "online") return "실시간 시세";
  if (status === "stale") return "스냅샷 대체";
  return "시세 연결 중";
}

function RegressionChart({
  candles,
  timeframe,
  showChannel,
  showAverages,
  expanded,
  historyLoading,
  historyExhausted,
  onLoadOlder,
}: {
  candles: Candle[];
  timeframe: Timeframe;
  showChannel: boolean;
  showAverages: boolean;
  expanded: boolean;
  historyLoading: boolean;
  historyExhausted: boolean;
  onLoadOlder: (timeframe: Timeframe) => void;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ startX: number; endIndex: number } | null>(null);
  const [dimensions, setDimensions] = useState({ width: 900, height: 510 });
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [barsVisible, setBarsVisible] = useState(VISIBLE_BARS[timeframe]);
  const [endTime, setEndTime] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);

  const endIndex = useMemo(() => {
    if (endTime == null) return candles.length;
    for (let index = candles.length - 1; index >= 0; index -= 1) {
      if (candles[index].time <= endTime) return index + 1;
    }
    return Math.min(candles.length, barsVisible);
  }, [candles, endTime, barsVisible]);
  const startIndex = Math.max(0, endIndex - Math.min(barsVisible, endIndex));
  const visible = useMemo(
    () => candles.slice(startIndex, endIndex),
    [candles, startIndex, endIndex],
  );
  const focused = hoverIndex == null ? visible.at(-1) : visible[hoverIndex];
  const followingLive = endTime == null;

  useEffect(() => {
    const element = wrapRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      const width = Math.max(320, Math.floor(entry.contentRect.width));
      const height = expanded
        ? Math.max(520, window.innerHeight - 190)
        : width < 640 ? 420 : 520;
      setDimensions({ width, height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [expanded]);

  useEffect(() => {
    if (startIndex < 24 && !historyLoading && !historyExhausted) {
      onLoadOlder(timeframe);
    }
  }, [startIndex, historyLoading, historyExhausted, onLoadOlder, timeframe]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || visible.length < 2) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = dimensions.width * dpr;
    canvas.height = dimensions.height * dpr;
    canvas.style.width = `${dimensions.width}px`;
    canvas.style.height = `${dimensions.height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const width = dimensions.width;
    const height = dimensions.height;
    const left = 16;
    const right = width < 540 ? 54 : 72;
    const top = 18;
    const priceBottom = height - 102;
    const volumeTop = height - 78;
    const volumeBottom = height - 30;
    const chartWidth = width - left - right;
    const priceHeight = priceBottom - top;
    const stepX = chartWidth / visible.length;
    const candleWidth = Math.max(1.2, Math.min(7, stepX * 0.62));

    const values = visible.flatMap((candle) => {
      const extras: number[] = [];
      if (showChannel) {
        if (candle.upper != null) extras.push(candle.upper);
        if (candle.lower != null) extras.push(candle.lower);
      }
      if (showAverages) {
        if (candle.ema20 != null) extras.push(candle.ema20);
        if (candle.ema60 != null) extras.push(candle.ema60);
      }
      return [candle.low, candle.high, ...extras];
    });
    let minPrice = Math.min(...values);
    let maxPrice = Math.max(...values);
    const pricePadding = Math.max((maxPrice - minPrice) * 0.08, maxPrice * 0.001);
    minPrice -= pricePadding;
    maxPrice += pricePadding;
    const range = Math.max(1, maxPrice - minPrice);
    const maxVolume = Math.max(...visible.map((candle) => candle.volume), 1);

    const x = (index: number) => left + stepX * index + stepX / 2;
    const y = (price: number) => top + ((maxPrice - price) / range) * priceHeight;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0c1114";
    ctx.fillRect(0, 0, width, height);

    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 5; i += 1) {
      const lineY = top + (priceHeight / 5) * i;
      const price = maxPrice - (range / 5) * i;
      ctx.strokeStyle = "rgba(148, 169, 177, 0.11)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, lineY + 0.5);
      ctx.lineTo(width - right, lineY + 0.5);
      ctx.stroke();
      ctx.fillStyle = "#6e7d83";
      ctx.fillText(priceFormatter.format(price), width - right + 8, lineY);
    }

    if (showChannel) {
      const valid = visible
        .map((candle, index) => ({ candle, index }))
        .filter(({ candle }) => candle.upper != null && candle.lower != null);
      if (valid.length > 1) {
        ctx.beginPath();
        valid.forEach(({ candle, index }, pointIndex) => {
          const command = pointIndex === 0 ? "moveTo" : "lineTo";
          ctx[command](x(index), y(candle.upper as number));
        });
        [...valid].reverse().forEach(({ candle, index }) => {
          ctx.lineTo(x(index), y(candle.lower as number));
        });
        ctx.closePath();
        ctx.fillStyle = "rgba(91, 229, 217, 0.075)";
        ctx.fill();
      }
    }

    const drawLine = (
      key: "upper" | "lower" | "mid" | "ema20" | "ema60",
      color: string,
      lineWidth = 1,
      dash: number[] = [],
    ) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.setLineDash(dash);
      let started = false;
      visible.forEach((candle, index) => {
        const value = candle[key];
        if (value == null) return;
        if (!started) {
          ctx.moveTo(x(index), y(value));
          started = true;
        } else {
          ctx.lineTo(x(index), y(value));
        }
      });
      if (started) ctx.stroke();
      ctx.setLineDash([]);
    };

    if (showChannel) {
      drawLine("upper", "rgba(100, 232, 218, 0.9)", 1.2);
      drawLine("lower", "rgba(100, 232, 218, 0.9)", 1.2);
      drawLine("mid", "rgba(100, 232, 218, 0.46)", 1, [5, 5]);

      const liveIndex = visible.length - 1;
      const live = visible[liveIndex];
      const previous = visible[liveIndex - 1];
      if (live?.closed === false && previous) {
        (["upper", "lower", "mid"] as const).forEach((key) => {
          const from = previous[key];
          const to = live[key];
          if (from == null || to == null) return;
          ctx.strokeStyle = key === "mid" ? "rgba(213,255,99,.72)" : "#d5ff63";
          ctx.lineWidth = key === "mid" ? 1.2 : 1.8;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.moveTo(x(liveIndex - 1), y(from));
          ctx.lineTo(x(liveIndex), y(to));
          ctx.stroke();
          ctx.setLineDash([]);
        });
      }
    }
    if (showAverages) {
      drawLine("ema20", "rgba(255, 199, 105, 0.92)", 1.1);
      drawLine("ema60", "rgba(180, 139, 255, 0.86)", 1.1);
    }

    const lastChannel = visible.at(-1);
    if (showChannel && timeframe === "1h" && lastChannel?.upper && lastChannel.lower) {
      const levels = [
        { value: lastChannel.upper * 1.02, color: "rgba(255,111,107,.55)", label: "SHORT +2%" },
        { value: lastChannel.lower * 0.98, color: "rgba(213,255,99,.55)", label: "LONG -2%" },
      ];
      levels.forEach((level) => {
        if (level.value < minPrice || level.value > maxPrice) return;
        const levelY = y(level.value);
        ctx.strokeStyle = level.color;
        ctx.setLineDash([3, 5]);
        ctx.beginPath();
        ctx.moveTo(left, levelY);
        ctx.lineTo(width - right, levelY);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = level.color;
        ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillText(level.label, left + 5, levelY - 8);
      });
    }

    visible.forEach((candle, index) => {
      const center = x(index);
      const rising = candle.close >= candle.open;
      const color = rising ? "#d5ff63" : "#ff6f6b";
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(center, y(candle.high));
      ctx.lineTo(center, y(candle.low));
      ctx.stroke();

      const bodyTop = y(Math.max(candle.open, candle.close));
      const bodyBottom = y(Math.min(candle.open, candle.close));
      const bodyHeight = Math.max(1, bodyBottom - bodyTop);
      if (candle.closed === false) {
        ctx.globalAlpha = 0.38;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(center - candleWidth / 2, bodyTop, candleWidth, bodyHeight);
      } else if (rising) {
        ctx.globalAlpha = 0.78;
        ctx.fillRect(center - candleWidth / 2, bodyTop, candleWidth, bodyHeight);
      } else {
        ctx.globalAlpha = 0.88;
        ctx.fillRect(center - candleWidth / 2, bodyTop, candleWidth, bodyHeight);
      }
      ctx.globalAlpha = 1;

      const volumeHeight = (candle.volume / maxVolume) * (volumeBottom - volumeTop);
      ctx.globalAlpha = 0.24;
      ctx.fillRect(center - candleWidth / 2, volumeBottom - volumeHeight, candleWidth, volumeHeight);
      ctx.globalAlpha = 1;
    });

    const liveCandle = visible.at(-1);
    if (liveCandle?.closed === false) {
      const liveX = x(visible.length - 1);
      ctx.strokeStyle = "rgba(213,255,99,.24)";
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(liveX, top);
      ctx.lineTo(liveX, volumeBottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#d5ff63";
      ctx.font = "700 9px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "right";
      ctx.fillText("LIVE", Math.min(width - right - 2, liveX), top + 8);
      ctx.textAlign = "left";
    }

    ctx.strokeStyle = "rgba(148, 169, 177, 0.16)";
    ctx.beginPath();
    ctx.moveTo(left, volumeTop - 8);
    ctx.lineTo(width - right, volumeTop - 8);
    ctx.stroke();
    ctx.fillStyle = "#59686e";
    ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText("QUOTE VOL", left, volumeTop - 16);

    const timePoints = [0, 0.25, 0.5, 0.75, 1];
    timePoints.forEach((ratio, pointIndex) => {
      const index = Math.min(visible.length - 1, Math.floor((visible.length - 1) * ratio));
      const label = chartTime(visible[index].time, timeframe);
      ctx.fillStyle = "#65747a";
      ctx.textAlign = pointIndex === 0 ? "left" : pointIndex === timePoints.length - 1 ? "right" : "center";
      ctx.fillText(label, x(index), height - 12);
    });
    ctx.textAlign = "left";

    if (hoverIndex != null && visible[hoverIndex]) {
      const candle = visible[hoverIndex];
      const hoverX = x(hoverIndex);
      const hoverY = y(candle.close);
      ctx.strokeStyle = "rgba(236, 245, 246, 0.35)";
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(hoverX, top);
      ctx.lineTo(hoverX, volumeBottom);
      ctx.moveTo(left, hoverY);
      ctx.lineTo(width - right, hoverY);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [visible, timeframe, showChannel, showAverages, dimensions, hoverIndex]);

  const handlePointer = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const left = 16;
    const right = rect.width < 540 ? 54 : 72;
    const chartWidth = rect.width - left - right;
    const relativeX = Math.max(0, Math.min(chartWidth, event.clientX - rect.left - left));
    const index = Math.min(visible.length - 1, Math.floor((relativeX / chartWidth) * visible.length));
    setHoverIndex(index);
  };

  return (
    <div className="chart-wrap" ref={wrapRef}>
      <div className="ohlc-strip" aria-live="polite">
        <span>{focused ? formatKst(focused.time) : "—"}{focused?.closed === false ? " · LIVE" : ""}</span>
        <span>O <b>{formatPrice(focused?.open)}</b></span>
        <span>H <b>{formatPrice(focused?.high)}</b></span>
        <span>L <b>{formatPrice(focused?.low)}</b></span>
        <span>C <b className={focused && focused.close >= focused.open ? "up" : "down"}>{formatPrice(focused?.close)}</b></span>
      </div>
      <canvas
        ref={canvasRef}
        onPointerMove={handlePointer}
        onPointerLeave={() => setHoverIndex(null)}
        aria-label={`${timeframe} 비트코인 캔들과 회귀채널 차트`}
      />
    </div>
  );
}

function MetricCard({
  label,
  value,
  detail,
  tone = "default",
}: {
  label: string;
  value: string;
  detail: string;
  tone?: "default" | "positive" | "negative";
}) {
  return (
    <article className={`metric-card ${tone}`}>
      <span className="metric-label">{label}</span>
      <strong>{value}</strong>
      <span className="metric-detail">{detail}</span>
    </article>
  );
}

export function BtcDashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [liveCandles, setLiveCandles] = useState<Record<Timeframe, Candle[]> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [timeframe, setTimeframe] = useState<Timeframe>("1h");
  const [showChannel, setShowChannel] = useState(true);
  const [showAverages, setShowAverages] = useState(false);
  const [liveStatus, setLiveStatus] = useState<LiveStatus>("connecting");
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);
  const [frozenSignal, setFrozenSignal] = useState<(StrategyPoint & { detectedAt: number }) | null>(null);
  const liveReady = liveCandles !== null;

  useEffect(() => {
    fetch("/btc-dashboard.json")
      .then((response) => {
        if (!response.ok) throw new Error("대시보드 데이터를 읽지 못했습니다.");
        return response.json() as Promise<DashboardData>;
      })
      .then((snapshot) => {
        const initial = {} as Record<Timeframe, Candle[]>;
        for (const item of TIMEFRAMES) {
          initial[item.value] = enrichCandles(
            snapshot.candles[item.value].map((candle) => ({ ...candle, closed: true })),
          );
        }
        setData(snapshot);
        setLiveCandles(initial);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (!liveReady) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const refresh = async () => {
      try {
        const response = await fetch(`/api/btc-live?tf=${timeframe}`, { cache: "no-store" });
        const payload = await response.json() as LiveResponse;
        if (!response.ok) throw new Error(payload.error ?? "실시간 시세 요청 실패");
        if (disposed) return;
        setLiveCandles((previous) => {
          if (!previous) return previous;
          const next = { ...previous };
          for (const [key, updates] of Object.entries(payload.candles)) {
            const tf = key as Timeframe;
            if (!updates?.length || !next[tf]) continue;
            next[tf] = enrichCandles(mergeRawCandles(next[tf], updates));
          }
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
  }, [timeframe, liveReady]);

  const strategyView = useMemo(() => {
    if (!liveCandles) return null;
    return buildLiveStrategy(liveCandles["1h"], liveCandles["4h"], 0.02);
  }, [liveCandles]);

  useEffect(() => {
    const live = strategyView?.live;
    if (!live?.candleTime) return;
    const timer = setTimeout(() => {
      setFrozenSignal((previous) => {
        if (previous?.candleTime === live.candleTime) return previous;
        if (live.signal === "waiting") return null;
        return { ...live, detectedAt: Math.floor(Date.now() / 1000) };
      });
    }, 0);
    return () => clearTimeout(timer);
  }, [strategyView]);

  if (error) {
    return <main className="center-state"><p>{error}</p></main>;
  }

  if (!data || !liveCandles || !strategyView) {
    return <main className="center-state"><div className="loading-mark" /><p>BTC 전략 데스크를 준비하는 중</p></main>;
  }

  const liveStrategy = strategyView.live;
  const confirmedStrategy = strategyView.confirmed;
  const activeStrategy = frozenSignal?.candleTime === liveStrategy.candleTime
    ? frozenSignal
    : liveStrategy;
  const signal = signalCopy(activeStrategy.signal);
  const minute = liveCandles["1m"];
  const marketPrice = minute.at(-1)?.close ?? data.market.price;
  const previousMinute = minute.at(-2)?.close ?? marketPrice;
  const marketChangePct = previousMinute ? (marketPrice / previousMinute - 1) * 100 : 0;
  const dailyPnl = data.account.balance - data.account.dayStartBalance;
  const dailyPct = data.account.dayStartBalance
    ? (dailyPnl / data.account.dayStartBalance) * 100
    : 0;
  const slopePct = liveStrategy.slope && marketPrice
    ? (liveStrategy.slope / marketPrice) * 100
    : 0;
  const channelDirection = (liveStrategy.slope ?? 0) < 0 ? "하방" : "상방";
  const channelQuality = (liveStrategy.r2 ?? 0) >= 0.7 ? "강함" : (liveStrategy.r2 ?? 0) >= 0.4 ? "보통" : "약함";
  const signalIsFrozen = frozenSignal?.candleTime === liveStrategy.candleTime;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">₿</div>
          <div>
            <span className="brand-name">PAPER / DESK</span>
            <span className="brand-sub">REGRESSION RETURN</span>
          </div>
        </div>
        <nav className="strategy-nav" aria-label="전략 메뉴">
          <button className="nav-item active" type="button">BTC 회귀채널</button>
          <button className="nav-item disabled" type="button" disabled>BTC 25×</button>
          <button className="nav-item disabled" type="button" disabled>XRP 패턴</button>
        </nav>
        <div className="system-status">
          <span className={`status-dot ${liveStatus}`} />
          <span>{connectionCopy(liveStatus)}</span>
          <small>PAPER · READ ONLY</small>
        </div>
      </header>

      <section className="market-heading">
        <div>
          <div className="breadcrumb">BITCOIN <span>/</span> REGRESSION CHANNEL</div>
          <h1>회귀채널 복귀</h1>
          <p>과확장된 가격이 채널 경계로 돌아오는 순간만 기다립니다.</p>
        </div>
        <div className="quote-block">
          <span className="quote-symbol">BTC / USDT <em>PERP</em></span>
          <strong>{formatPrice(marketPrice)}</strong>
          <span className={marketChangePct >= 0 ? "quote-change up" : "quote-change down"}>
            {formatPercent(marketChangePct)} · 현재 1분봉
          </span>
        </div>
      </section>

      <section className="workspace-grid">
        <article className="panel chart-panel">
          <div className="panel-header chart-toolbar">
            <div className="timeframe-tabs" role="tablist" aria-label="캔들 시간봉 선택">
              {TIMEFRAMES.map((item) => (
                <button
                  key={item.value}
                  type="button"
                  role="tab"
                  aria-selected={timeframe === item.value}
                  className={timeframe === item.value ? "active" : ""}
                  onClick={() => setTimeframe(item.value)}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <div className="indicator-toggles">
              <button
                type="button"
                className={showChannel ? "active channel" : ""}
                aria-pressed={showChannel}
                onClick={() => setShowChannel((value) => !value)}
              >
                <i /> 회귀채널 100 · 2σ
              </button>
              <button
                type="button"
                className={showAverages ? "active averages" : ""}
                aria-pressed={showAverages}
                onClick={() => setShowAverages((value) => !value)}
              >
                <i /> EMA 20 / 60
              </button>
            </div>
          </div>
          <RegressionChart
            candles={liveCandles[timeframe]}
            timeframe={timeframe}
            showChannel={showChannel}
            showAverages={showAverages}
          />
          <footer className="chart-footer">
            <span><i className="legend-candle up" /> 상승</span>
            <span><i className="legend-candle down" /> 하락</span>
            <span><i className="legend-line channel" /> 회귀 상·하단</span>
            <span><i className="legend-line live" /> 진행 중 계산</span>
            <span className="chart-note">최근 <b>99개 확정봉 + 현재 1시간봉</b>으로 LIVE 채널 계산</span>
          </footer>
        </article>

        <aside className="panel decision-panel">
          <div className="decision-topline">
            <span>STRATEGY PULSE</span>
            <span className={`snapshot-badge ${liveStatus}`}>{liveStatus === "online" ? "LIVE PREVIEW" : "SNAPSHOT"}</span>
          </div>
          <div className={`signal-state ${signal.tone}`}>
            <span>{signal.eyebrow}</span>
            <h2>{signal.title}</h2>
            <p>{signalIsFrozen ? `신호 포착 ${formatKst(frozenSignal.detectedAt)} · 가격 고정` : "현재 진행 중 1시간봉 기준 · 3초 갱신"}</p>
            {activeStrategy.signal !== "waiting" && (
              <div className="signal-order-preview">
                <span>지정가 <b>{formatPrice(activeStrategy.entry)}</b></span>
                <span>절반 익절 <b>{formatPrice(activeStrategy.target)}</b></span>
              </div>
            )}
          </div>

          <div className="preview-warning">
            <span>READ ONLY</span>
            <p>실시간 예상 판단입니다. AWS 주문 엔진은 아직 확정봉 방식이며 이 화면에서 주문하지 않습니다.</p>
          </div>

          <div className="decision-metrics">
            <div><span>채널 방향</span><strong>{channelDirection}</strong><small>{formatPercent(slopePct, 3)} / 봉</small></div>
            <div><span>추세 적합도 R²</span><strong>{formatPercent((liveStrategy.r2 ?? 0) * 100, 1)}</strong><small>{channelQuality}</small></div>
            <div><span>4H 국면</span><strong>{regimeLabel(liveStrategy.majorRegime)}</strong><small>진행 중 MA20 / MA60</small></div>
            <div><span>ATR 14</span><strong>{formatPrice(liveStrategy.atr)}</strong><small>진행 중 봉 포함</small></div>
          </div>

          <div className="channel-levels comparison">
            <div className="level-head"><span>CHANNEL</span><b>LIVE</b><b>CONFIRMED</b></div>
            <div><span>상단</span><b>{formatPrice(liveStrategy.upper)}</b><em>{formatPrice(confirmedStrategy.upper)}</em></div>
            <div><span>중앙</span><b>{formatPrice(liveStrategy.mid)}</b><em>{formatPrice(confirmedStrategy.mid)}</em></div>
            <div><span>하단</span><b>{formatPrice(liveStrategy.lower)}</b><em>{formatPrice(confirmedStrategy.lower)}</em></div>
          </div>

          <div className="logic-flow">
            <span className="section-kicker">ENTRY LOGIC</span>
            <ol>
              <li className={(liveStrategy.slope ?? 0) !== 0 ? "done" : ""}>
                <i>01</i><div><b>채널 방향 확정</b><span>100봉 회귀 기울기 확인</span></div>
              </li>
              <li className={activeStrategy.signal !== "waiting" ? "done" : "current"}>
                <i>02</i><div><b>반대편 경계 이탈</b><span>직전 봉 대비 신규 꼬리 돌파</span></div>
              </li>
              <li>
                <i>03</i><div><b>2% 밖 지정가</b><span>12시간 내 미체결 시 취소</span></div>
              </li>
              <li>
                <i>04</i><div><b>채널선 절반 익절</b><span>나머지 물량 ATR 트레일링</span></div>
              </li>
            </ol>
          </div>
        </aside>
      </section>

      <section className="account-section">
        <div className="section-heading">
          <div><span>ACCOUNT / POSITION</span><h2>계좌 상태</h2></div>
          <p>계좌 {formatKst(data.generatedAt)} 스냅샷 · 시세 {lastUpdate ? formatKst(lastUpdate) : "연결 중"}</p>
        </div>
        <div className="metric-grid">
          <MetricCard
            label="총 잔고"
            value={`${balanceFormatter.format(data.account.balance)} USDT`}
            detail="실현 잔고"
          />
          <MetricCard
            label="오늘 손익"
            value={`${dailyPnl >= 0 ? "+" : ""}${balanceFormatter.format(dailyPnl)} USDT`}
            detail={formatPercent(dailyPct)}
            tone={dailyPnl >= 0 ? "positive" : "negative"}
          />
          <MetricCard
            label="BTC 오픈 포지션"
            value={`${data.positions.length}건`}
            detail={data.positions.length ? "회귀채널 전략 보유 중" : "현재 노출 없음"}
          />
          <MetricCard
            label="사용 중 증거금"
            value={data.positions.length ? "계산 중" : "0.00 USDT"}
            detail="읽기 전용 페이퍼 계좌"
          />
        </div>

        <article className="panel positions-panel">
          <div className="positions-header">
            <div><span className="section-kicker">OPEN POSITIONS</span><h3>BTC 회귀채널 포지션</h3></div>
            <span className="count-badge">{data.positions.length}</span>
          </div>
          {data.positions.length === 0 ? (
            <div className="empty-position">
              <div className="empty-radar"><span /></div>
              <div>
                <strong>현재 보유 중인 포지션이 없습니다</strong>
                <p>진입이 발생하면 방향, 평단가, 현재가, 손절·익절선과 수익률이 이곳에 표시됩니다.</p>
              </div>
              <span className="waiting-chip">SIGNAL WAITING</span>
            </div>
          ) : (
            <div className="position-table-wrap">
              <table>
                <thead><tr><th>방향</th><th>평단가</th><th>현재가</th><th>가격 수익률</th><th>레버리지 ROE</th><th>손절</th><th>1차 목표</th><th>잔여 수량</th></tr></thead>
                <tbody>
                  {data.positions.map((position, index) => {
                    const direction = position.side === "long" ? 1 : -1;
                    const priceMove = ((marketPrice - position.entry) / position.entry) * 100 * direction;
                    const roe = priceMove * position.leverage;
                    return (
                      <tr key={`${position.symbol}-${index}`}>
                        <td><span className={`side-chip ${position.side}`}>{position.side.toUpperCase()}</span></td>
                        <td>{formatPrice(position.entry)}</td>
                        <td>{formatPrice(marketPrice)}</td>
                        <td className={priceMove >= 0 ? "up" : "down"}>{formatPercent(priceMove)}</td>
                        <td className={roe >= 0 ? "up" : "down"}>{formatPercent(roe)}</td>
                        <td>{formatPrice(position.stop)}</td>
                        <td>{formatPrice(position.tp1)}</td>
                        <td>{(position.remaining_ratio * 100).toFixed(0)}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </article>
      </section>

      <footer className="page-footer">
        <span>DATA · {liveStatus === "online" ? "Gate.io live public candles" : data.source}</span>
        <span>읽기 전용 화면 · 주문 기능 없음</span>
      </footer>
    </main>
  );
}
