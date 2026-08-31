"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  buildLiveStrategy,
  confirmedPivots,
  enrichCandles,
  mergeRawCandles,
  TIMEFRAME_SECONDS,
} from "@/lib/market.mjs";
import type { StrategyPoint } from "@/lib/market.mjs";

export type Timeframe = "1m" | "5m" | "15m" | "1h" | "4h" | "1d";

export type Candle = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  quoteVolume?: number;
  mid: number | null;
  upper: number | null;
  lower: number | null;
  slope: number | null;
  r2: number | null;
  ema20: number | null;
  ema60: number | null;
  atr: number | null;
  rsi14: number | null;
  macd: number | null;
  macdSignal: number | null;
  macdHist: number | null;
  sma20?: number | null;
  sma25?: number | null;
  sma50?: number | null;
  vwap?: number | null;
  rvol?: number | null;
  deviation?: number | null;
  deviationRank?: number | null;
  spike?: boolean;
  closed?: boolean;
};

export type LiveStatus = "connecting" | "online" | "stale";

type LiveResponse = {
  serverTime: number;
  source: string;
  candles: Partial<Record<Timeframe, Candle[]>>;
  mode?: "live" | "history" | "catchup";
  truncated?: boolean;
  error?: string;
};

export type Position = {
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

type TradeExitEvent = {
  time: number;
  reason: string;
  price: number;
  ratio: number;
  fee: number;
  pnl: number;
  note: string;
};

export type TradeRecord = {
  id: string;
  entryTime: number;
  exitTime: number;
  strategy: string;
  symbol: string;
  side: "long" | "short";
  entryPrice: number;
  exitPrice: number;
  contracts: number;
  notional: number;
  leverage: number;
  stop: number;
  tp1: number;
  entryOrder: "market" | "limit";
  score: number;
  reasons: string[];
  context: Record<string, unknown>;
  exitEvents: TradeExitEvent[];
  exitMode: string;
  entryFee: number;
  fees: number;
  pnl: number;
  balanceAfter: number;
  holdingSeconds: number;
};

export type DashboardData = {
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
  positions25x?: Position[];
  strategy25x?: {
    channelLength: number;
    channelWidthMult: number;
    channelR2Min: number;
    channelMinMovePct: number;
    maxEntryDistancePct: number;
    limitTtlBars: number;
    leverage: number;
    marginFraction: number;
    accountTakeProfitPct: number;
    targetMarginRoe: number;
    makerRate: number;
    maxDailyEntries: number;
    consecutiveLossLimit: number;
    dailyEntries: number;
    consecutiveLosses: number;
    pausedUntil: number;
  };
  trades?: TradeRecord[];
  candles: Record<Timeframe, Candle[]>;
  xrpMarket?: {
    symbol: string;
    displaySymbol: string;
    price: number;
    changePct: number;
  };
  xrpPositions?: Position[];
  xrpCandles?: Record<Exclude<Timeframe, "1m">, Candle[]>;
  xrpPatterns?: XrpPatternMonitor[];
  xrpSignals?: XrpSignalEvent[];
};

export type XrpPatternMonitor = {
  name: string;
  family: string;
  timeframe: "5m" | "15m" | "1h" | "4h";
  contextTimeframe: "1h" | "4h" | "1d";
  confirmTimeframe: "15m" | "1h" | "4h" | "1d";
  orderType: "market" | "limit";
  trendAlign: boolean;
  blockedUtcHours: number[];
};

export type XrpSignalEvent = {
  time: number;
  action: "entry" | "skip";
  strategy: string;
  side: "long" | "short" | null;
  skipReason: string | null;
  entry: number | null;
  stop: number | null;
  target: number | null;
  score: number | null;
  reasons: string[];
  indicators: Record<string, unknown>;
  detail: string | null;
  notes: string[];
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

function tradePrice(value: number, symbol: string) {
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: symbol.startsWith("XRP") ? 4 : 1,
    maximumFractionDigits: symbol.startsWith("XRP") ? 4 : 1,
  }).format(value);
}

function holdingTime(seconds: number) {
  if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))}분`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return minutes ? `${hours}시간 ${minutes}분` : `${hours}시간`;
}

function reasonLabel(reason: string) {
  const fixed: Record<string, string> = {
    double_top: "쌍봉 천장",
    double_bottom: "쌍바닥",
    symmetric_triangle: "대칭 삼각수렴",
    kotegawa: "코테가와 역추세",
    break_short: "하단 돌파",
    break_long: "상단 돌파",
    prior_trend: "선행 추세 확인",
    volume_dry_up: "거래량 수축",
    vwap_aligned: "VWAP 정렬",
    "15m_confirmed": "15분봉 확인",
    "4h_confirmed": "4시간봉 확인",
    btc_aligned: "BTC 방향 일치",
    no_wall_ahead: "전방 매물벽 없음",
    lvn_ahead: "LVN 공간 확인",
    htf_level: "상위봉 레벨",
    at_value_edge: "가치영역 경계",
    second_peak_weaker: "두 번째 고점 약화",
    retested: "리테스트 확인",
    value_area_break: "가치영역 돌파",
  };
  if (fixed[reason]) return fixed[reason];
  if (reason.startsWith("rvol_")) return `상대 거래량 ${reason.slice(5)}×`;
  if (reason.startsWith("progress_")) return `패턴 진행도 ${reason.slice(9)}`;
  if (reason.startsWith("dev_")) return `이평 이격 ${reason.slice(4)}`;
  if (reason.startsWith("rank_")) return `이격 순위 ${reason.slice(5)}`;
  return reason.replaceAll("_", " ");
}

function patternLabel(trade: TradeRecord) {
  const primary = trade.reasons.find((reason) => [
    "double_top", "double_bottom", "symmetric_triangle", "kotegawa",
  ].includes(reason));
  if (primary) return reasonLabel(primary);
  if (trade.strategy === "btc_regression_channel") return "BTC 회귀채널 복귀";
  if (trade.strategy === "btc_regime_25x") return "BTC 25× 단기 국면";
  return trade.strategy.replace("xrp_patterns.", "").replaceAll("_", " ");
}

function exitCopy(mode: string) {
  const copy: Record<string, { label: string; tone: "positive" | "negative" | "neutral" }> = {
    tp1_trail: { label: "1차 익절 → 트레일링", tone: "positive" },
    take_profit: { label: "목표가 익절", tone: "positive" },
    trailing: { label: "트레일링 청산", tone: "positive" },
    stop_loss: { label: "손절", tone: "negative" },
    time_exit: { label: "시간 청산", tone: "neutral" },
  };
  return copy[mode] ?? { label: mode.replaceAll("_", " "), tone: "neutral" as const };
}

function eventLabel(reason: string) {
  if (reason === "tp1") return "1차 익절";
  if (reason === "trail") return "트레일링 청산";
  if (reason === "stop") return "손절";
  if (reason === "time") return "시간 청산";
  return reason;
}

// Accumulated pinch deltaY needed for one zoom step; a pinch fires many tiny
// wheel events and stepping on each one would blow past the whole range.
const PINCH_ZOOM_STEP = 12;

export function RegressionChart({
  candles,
  timeframe,
  showChannel,
  showAverages,
  expanded,
  historyLoading,
  historyExhausted,
  onLoadOlder,
  averageMode = "ema20/60",
  showVwap = false,
  showPivots = false,
  instrumentLabel = "비트코인",
  priceDigits = 1,
  trades = [],
}: {
  candles: Candle[];
  timeframe: Timeframe;
  showChannel: boolean;
  showAverages: boolean;
  expanded: boolean;
  historyLoading: boolean;
  historyExhausted: boolean;
  onLoadOlder: (timeframe: Timeframe) => void;
  averageMode?: "ema20/60" | "sma20/50";
  showVwap?: boolean;
  showPivots?: boolean;
  instrumentLabel?: string;
  priceDigits?: number;
  trades?: TradeRecord[];
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ startX: number; endIndex: number } | null>(null);
  const wheelPanRef = useRef(0);
  const pinchZoomRef = useRef(0);
  const [dimensions, setDimensions] = useState({ width: 900, height: 510 });
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [barsVisible, setBarsVisible] = useState(VISIBLE_BARS[timeframe]);
  const [endTime, setEndTime] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const chartPriceFormatter = useMemo(() => new Intl.NumberFormat("en-US", {
    minimumFractionDigits: priceDigits,
    maximumFractionDigits: priceDigits,
  }), [priceDigits]);
  const formatChartPrice = (value: number | null | undefined) => (
    value == null ? "—" : `$${chartPriceFormatter.format(value)}`
  );

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
  const visiblePivots = useMemo(() => {
    if (!showPivots) return [];
    return confirmedPivots(candles, 5)
      .filter((pivot) => pivot.index >= startIndex && pivot.index < endIndex)
      .slice(-24)
      .map((pivot) => ({ ...pivot, visibleIndex: pivot.index - startIndex }));
  }, [candles, startIndex, endIndex, showPivots]);

  // Entry ("S") and exit ("F") markers for completed trades that fall inside
  // the visible window. Long trades are green, short trades red.
  const visibleMarks = useMemo(() => {
    if (!trades.length || !visible.length) return [];
    const firstTime = visible[0].time;
    const lastTime = visible.at(-1)!.time + TIMEFRAME_SECONDS[timeframe];
    const locate = (time: number) => {
      if (time < firstTime || time >= lastTime) return null;
      for (let index = visible.length - 1; index >= 0; index -= 1) {
        if (visible[index].time <= time) return index;
      }
      return null;
    };
    return trades.flatMap((trade) => ([
      { index: locate(trade.entryTime), price: trade.entryPrice, label: "S", side: trade.side },
      { index: locate(trade.exitTime), price: trade.exitPrice, label: "F", side: trade.side },
    ])).filter((mark) => mark.index != null) as {
      index: number; price: number; label: string; side: "long" | "short";
    }[];
  }, [trades, visible, timeframe]);

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
        const fast = averageMode === "sma20/50" ? candle.sma20 : candle.ema20;
        const slow = averageMode === "sma20/50" ? candle.sma50 : candle.ema60;
        if (fast != null) extras.push(fast);
        if (slow != null) extras.push(slow);
      }
      if (showVwap && candle.vwap != null) extras.push(candle.vwap);
      return [candle.low, candle.high, ...extras];
    });
    let minPrice = Math.min(...values);
    let maxPrice = Math.max(...values);
    const pricePadding = Math.max((maxPrice - minPrice) * 0.08, maxPrice * 0.001);
    minPrice -= pricePadding;
    maxPrice += pricePadding;
    // Never impose a one-dollar minimum here: it is harmless for BTC but
    // compresses a ~$1 XRP market into a single line at the top of the chart.
    const range = Math.max(Number.EPSILON, maxPrice - minPrice);
    const displayVolume = (candle: Candle) => candle.quoteVolume ?? candle.volume;
    const maxVolume = Math.max(...visible.map(displayVolume), 1);

    const x = (index: number) => left + stepX * index + stepX / 2;
    const y = (price: number) => top + ((maxPrice - price) / range) * priceHeight;
    const expectedStep = TIMEFRAME_SECONDS[timeframe];
    const crossesGap = (previousIndex: number, index: number) => (
      index !== previousIndex + 1
      || visible[index].time - visible[previousIndex].time > expectedStep * 1.5
    );

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
      ctx.fillText(chartPriceFormatter.format(price), width - right + 8, lineY);
    }

    if (showChannel) {
      const valid = visible
        .map((candle, index) => ({ candle, index }))
        .filter(({ candle }) => candle.upper != null && candle.lower != null);
      const groups: typeof valid[] = [];
      valid.forEach((point) => {
        const group = groups.at(-1);
        const previousPoint = group?.at(-1);
        if (!previousPoint || crossesGap(previousPoint.index, point.index)) groups.push([point]);
        else group.push(point);
      });
      groups.filter((group) => group.length > 1).forEach((group) => {
        ctx.beginPath();
        group.forEach(({ candle, index }, pointIndex) => {
          const command = pointIndex === 0 ? "moveTo" : "lineTo";
          ctx[command](x(index), y(candle.upper as number));
        });
        [...group].reverse().forEach(({ candle, index }) => {
          ctx.lineTo(x(index), y(candle.lower as number));
        });
        ctx.closePath();
        ctx.fillStyle = "rgba(91, 229, 217, 0.075)";
        ctx.fill();
      });
    }

    const drawLine = (
      key: "upper" | "lower" | "mid" | "ema20" | "ema60" | "sma20" | "sma50" | "vwap",
      color: string,
      lineWidth = 1,
      dash: number[] = [],
    ) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.setLineDash(dash);
      let started = false;
      let previousIndex = -1;
      visible.forEach((candle, index) => {
        const value = candle[key];
        if (value == null) {
          started = false;
          previousIndex = -1;
          return;
        }
        if (!started || (previousIndex >= 0 && crossesGap(previousIndex, index))) {
          ctx.moveTo(x(index), y(value));
          started = true;
        } else {
          ctx.lineTo(x(index), y(value));
        }
        previousIndex = index;
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
      drawLine(averageMode === "sma20/50" ? "sma20" : "ema20", "rgba(255, 199, 105, 0.92)", 1.1);
      drawLine(averageMode === "sma20/50" ? "sma50" : "ema60", "rgba(180, 139, 255, 0.86)", 1.1);
    }
    if (showVwap) {
      drawLine("vwap", "rgba(91, 229, 217, 0.82)", 1, [5, 4]);
    }
    if (showPivots) {
      visiblePivots.forEach((pivot) => {
        ctx.fillStyle = pivot.kind === "high" ? "#ff6f6b" : "#d5ff63";
        ctx.beginPath();
        ctx.arc(x(pivot.visibleIndex), y(pivot.price), 2.4, 0, Math.PI * 2);
        ctx.fill();
      });
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

      const volumeHeight = (displayVolume(candle) / maxVolume) * (volumeBottom - volumeTop);
      ctx.globalAlpha = 0.24;
      ctx.fillRect(center - candleWidth / 2, volumeBottom - volumeHeight, candleWidth, volumeHeight);
      ctx.globalAlpha = 1;
    });

    visible.forEach((candle, index) => {
      if (index === 0 || candle.time - visible[index - 1].time <= expectedStep * 1.5) return;
      const gapX = x(index) - stepX / 2;
      ctx.strokeStyle = "rgba(255, 199, 105, 0.48)";
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(gapX, top);
      ctx.lineTo(gapX, volumeBottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255, 199, 105, 0.78)";
      ctx.font = "700 8px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillText("DATA GAP", gapX + 5, top + 8);
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

    visibleMarks.forEach((mark) => {
      if (mark.price < minPrice || mark.price > maxPrice) return;
      const color = mark.side === "long" ? "#16a34a" : "#ff6f6b";
      const tipX = x(mark.index);
      const tipY = y(mark.price);
      // Upside-down teardrop: round head with the point resting on the price.
      const radius = 7.5;
      const tipDistance = 12;
      const headY = tipY - tipDistance;
      const spread = Math.acos(radius / tipDistance);
      ctx.beginPath();
      ctx.arc(tipX, headY, radius, Math.PI / 2 + spread, Math.PI / 2 - spread + Math.PI * 2);
      ctx.lineTo(tipX, tipY);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#0c1114";
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.fillStyle = "#0c1114";
      ctx.font = "700 10px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.fillText(mark.label, tipX, headY);
      ctx.textAlign = "left";
    });

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
  }, [visible, visiblePivots, visibleMarks, timeframe, showChannel, showAverages, averageMode, showVwap, showPivots, dimensions, hoverIndex, chartPriceFormatter]);

  const resetToLive = () => {
    setBarsVisible(Math.min(VISIBLE_BARS[timeframe], candles.length));
    setEndTime(null);
    setHoverIndex(null);
  };

  const viewEndAt = (nextEndIndex: number) => {
    const minimumEnd = Math.min(barsVisible, candles.length);
    const clamped = Math.max(minimumEnd, Math.min(candles.length, nextEndIndex));
    setEndTime(clamped >= candles.length ? null : candles[clamped - 1]?.time ?? null);
  };

  const zoomBy = (direction: "in" | "out", anchorRatio = 0.5) => {
    const minimum = Math.min(30, candles.length);
    const maximum = Math.min(1_500, candles.length);
    const factor = direction === "in" ? 0.8 : 1.25;
    const nextBars = Math.max(minimum, Math.min(maximum, Math.round(barsVisible * factor)));
    if (nextBars === barsVisible || !visible.length) return;

    const anchorIndex = startIndex + Math.floor((visible.length - 1) * anchorRatio);
    let nextStart = Math.round(anchorIndex - (nextBars - 1) * anchorRatio);
    nextStart = Math.max(0, Math.min(candles.length - nextBars, nextStart));
    const nextEnd = nextStart + nextBars;
    setBarsVisible(nextBars);
    setEndTime(nextEnd >= candles.length ? null : candles[nextEnd - 1]?.time ?? null);
    setHoverIndex(null);
  };

  const panByBars = (olderBy: number) => {
    viewEndAt(endIndex - olderBy);
    setHoverIndex(null);
  };

  const handlePointer = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const left = 16;
    const right = rect.width < 540 ? 54 : 72;
    const chartWidth = rect.width - left - right;
    if (dragRef.current) {
      const pixelsPerBar = chartWidth / Math.max(visible.length, 1);
      const movedBars = Math.round((event.clientX - dragRef.current.startX) / pixelsPerBar);
      const minimumEnd = Math.min(barsVisible, candles.length);
      const nextEnd = Math.max(
        minimumEnd,
        Math.min(candles.length, dragRef.current.endIndex - movedBars),
      );
      setEndTime(nextEnd >= candles.length ? null : candles[nextEnd - 1]?.time ?? null);
      setHoverIndex(null);
      return;
    }
    const relativeX = Math.max(0, Math.min(chartWidth, event.clientX - rect.left - left));
    const index = Math.min(visible.length - 1, Math.floor((relativeX / chartWidth) * visible.length));
    setHoverIndex(index);
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { startX: event.clientX, endIndex };
    setDragging(true);
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
    setDragging(false);
  };

  // Native listener: React registers onWheel passively, so preventDefault there
  // cannot stop the page scrolling or the browser's horizontal back-swipe.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (event: WheelEvent) => {
      const rect = canvas.getBoundingClientRect();
      const left = 16;
      const right = rect.width < 540 ? 54 : 72;
      const chartWidth = rect.width - left - right;
      // A trackpad pinch reaches the page as a wheel event with ctrlKey set.
      if (event.ctrlKey) {
        event.preventDefault();
        pinchZoomRef.current += event.deltaY;
        if (Math.abs(pinchZoomRef.current) < PINCH_ZOOM_STEP) return;
        const direction = pinchZoomRef.current < 0 ? "in" : "out";
        pinchZoomRef.current = 0;
        wheelPanRef.current = 0;
        const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left - left) / chartWidth));
        zoomBy(direction, ratio);
        return;
      }
      // Claim every event that carries any horizontal component, before deciding
      // whether it pans. Chrome commits to its back-swipe on the first
      // unprevented event of a gesture, and a trackpad swipe opens with a few
      // tiny deltaX frames where deltaY still dominates — bailing out on those
      // hands the whole gesture to browser navigation, and preventDefault on the
      // later frames is then too late.
      if (event.deltaX !== 0) event.preventDefault();
      // Pan only while horizontal genuinely dominates, so a swipe's stray deltaY
      // cannot zoom in and out mid-pan.
      if (Math.abs(event.deltaX) <= Math.abs(event.deltaY)) return;
      const pixelsPerBar = chartWidth / Math.max(visible.length, 1);
      wheelPanRef.current += event.deltaX / pixelsPerBar;
      const bars = Math.trunc(wheelPanRef.current);
      if (!bars) return;
      wheelPanRef.current -= bars;
      panByBars(-bars);
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  });

  const handleKeyDown = (event: React.KeyboardEvent<HTMLCanvasElement>) => {
    if (event.key === "+" || event.key === "=") zoomBy("in", followingLive ? 1 : 0.5);
    else if (event.key === "-") zoomBy("out", followingLive ? 1 : 0.5);
    else if (event.key === "ArrowLeft") panByBars(10);
    else if (event.key === "ArrowRight") panByBars(-10);
    else if (event.key === "0") resetToLive();
    else return;
    event.preventDefault();
  };

  return (
    <div className="chart-wrap" ref={wrapRef}>
      <div className="ohlc-strip" aria-live="polite">
        <span>{focused ? formatKst(focused.time) : "—"}{focused?.closed === false ? " · LIVE" : ""}</span>
        <span>O <b>{formatChartPrice(focused?.open)}</b></span>
        <span>H <b>{formatChartPrice(focused?.high)}</b></span>
        <span>L <b>{formatChartPrice(focused?.low)}</b></span>
        <span>C <b className={focused && focused.close >= focused.open ? "up" : "down"}>{formatChartPrice(focused?.close)}</b></span>
      </div>
      <div className="chart-navigation" aria-label="차트 확대와 이동">
        <button type="button" onClick={() => zoomBy("out", followingLive ? 1 : 0.5)} aria-label="더 많은 캔들 보기">−</button>
        <span>{visible.length}봉 <small>/ {candles.length} 로드</small></span>
        <button type="button" onClick={() => zoomBy("in", followingLive ? 1 : 0.5)} aria-label="캔들 확대">＋</button>
        <button type="button" className={followingLive ? "active" : ""} onClick={resetToLive}>현재 · 기본</button>
        <button
          type="button"
          onClick={() => onLoadOlder(timeframe)}
          disabled={historyLoading || historyExhausted}
        >
          {historyLoading ? "불러오는 중" : historyExhausted ? "최대 구간" : "과거 +800"}
        </button>
      </div>
      <canvas
        ref={canvasRef}
        className={dragging ? "dragging" : ""}
        tabIndex={0}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointer}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onPointerLeave={() => { if (!dragRef.current) setHoverIndex(null); }}
        onKeyDown={handleKeyDown}
        aria-label={`${timeframe} ${instrumentLabel} 캔들 차트. 드래그나 두 손가락 좌우 스와이프로 과거를 이동하고 두 손가락 확대 제스처로 확대합니다.`}
      />
      <div className="chart-range-caption">
        <span>{visible[0] ? formatKst(visible[0].time) : "—"}</span>
        <span>{followingLive ? "LIVE FOLLOW" : "HISTORY VIEW"}</span>
        <span>{visible.at(-1) ? formatKst(visible.at(-1)!.time) : "—"}</span>
      </div>
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

function MomentumStrip({ candle, timeframe }: { candle: Candle | undefined; timeframe: Timeframe }) {
  const rsi = candle?.rsi14;
  const rsiLabel = rsi == null ? "워밍업" : rsi >= 70 ? "과매수" : rsi <= 30 ? "과매도" : "중립";
  const macdLabel = candle?.macd == null || candle.macdSignal == null
    ? "워밍업"
    : candle.macd > candle.macdSignal ? "상승 모멘텀" : "하락 모멘텀";
  const atrPct = candle?.atr != null && candle.close
    ? candle.atr / candle.close * 100
    : null;

  return (
    <div className="momentum-strip" aria-label={`${timeframe} 보조 지표`}>
      <div className="momentum-heading">
        <span>MOMENTUM / {timeframe.toUpperCase()}</span>
        <small>보조 확인용 · 진입 조건에는 미반영</small>
      </div>
      <div className="rsi-meter">
        <div><span>RSI 14</span><strong>{rsi == null ? "—" : rsi.toFixed(1)}</strong><em>{rsiLabel}</em></div>
        <div className="rsi-track" aria-hidden="true">
          <i className="rsi-low" /><i className="rsi-mid" /><i className="rsi-high" />
          {rsi != null && <b style={{ left: `${Math.max(0, Math.min(100, rsi))}%` }} />}
        </div>
      </div>
      <div className="momentum-value">
        <span>MACD 12·26·9</span>
        <strong>{candle?.macd == null ? "—" : candle.macd.toFixed(1)}</strong>
        <small>{macdLabel}</small>
      </div>
      <div className="momentum-value">
        <span>SIGNAL</span>
        <strong>{candle?.macdSignal == null ? "—" : candle.macdSignal.toFixed(1)}</strong>
        <small>시그널 EMA 9</small>
      </div>
      <div className={`momentum-value ${(candle?.macdHist ?? 0) >= 0 ? "positive" : "negative"}`}>
        <span>HISTOGRAM</span>
        <strong>{candle?.macdHist == null ? "—" : candle.macdHist.toFixed(1)}</strong>
        <small>MACD − SIGNAL</small>
      </div>
      <div className="momentum-value">
        <span>ATR 14 / PRICE</span>
        <strong>{atrPct == null ? "—" : `${atrPct.toFixed(2)}%`}</strong>
        <small>현재 변동성</small>
      </div>
    </div>
  );
}

export function TradeHistory({
  trades,
  generatedAt,
  initialSymbol = "all",
}: {
  trades: TradeRecord[];
  generatedAt: number;
  initialSymbol?: "all" | "BTC" | "XRP";
}) {
  const [symbolFilter, setSymbolFilter] = useState<"all" | "BTC" | "XRP">(initialSymbol);
  const [exitFilter, setExitFilter] = useState<"all" | "profit" | "stop" | "time">("all");
  const [expandedId, setExpandedId] = useState<string | null>(trades[0]?.id ?? null);
  const visible = useMemo(() => trades.filter((trade) => {
    if (symbolFilter !== "all" && !trade.symbol.startsWith(symbolFilter)) return false;
    if (exitFilter === "profit" && !["tp1_trail", "take_profit", "trailing"].includes(trade.exitMode)) return false;
    if (exitFilter === "stop" && trade.exitMode !== "stop_loss") return false;
    if (exitFilter === "time" && trade.exitMode !== "time_exit") return false;
    return true;
  }), [trades, symbolFilter, exitFilter]);
  const wins = visible.filter((trade) => trade.pnl > 0).length;
  const totalPnl = visible.reduce((sum, trade) => sum + trade.pnl, 0);
  const totalFees = visible.reduce((sum, trade) => sum + trade.fees, 0);

  return (
    <section className="trade-history-section">
      <div className="section-heading trade-history-heading">
        <div><span>EXECUTION / JOURNAL</span><h2>거래내역</h2></div>
        <p>AWS 페이퍼 실행 로그 · {formatKst(generatedAt)} 동기화</p>
      </div>

      <div className="trade-summary-grid">
        <div><span>완료 거래</span><strong>{visible.length}건</strong><small>부분익절은 한 거래로 합산</small></div>
        <div><span>승률</span><strong>{visible.length ? `${(wins / visible.length * 100).toFixed(1)}%` : "—"}</strong><small>{wins}승 · {visible.length - wins}패</small></div>
        <div className={totalPnl >= 0 ? "positive" : "negative"}><span>수수료 포함 손익</span><strong>{totalPnl >= 0 ? "+" : ""}{totalPnl.toFixed(2)} USDT</strong><small>진입 비용까지 포함</small></div>
        <div><span>체결 비용</span><strong>{totalFees >= 0 ? "+" : ""}{totalFees.toFixed(2)} USDT</strong><small>음수는 메이커 리베이트</small></div>
      </div>

      <article className="panel trade-journal">
        <header className="trade-filterbar">
          <div className="trade-filter-group" aria-label="종목 필터">
            {(["all", "BTC", "XRP"] as const).map((value) => (
              <button key={value} type="button" className={symbolFilter === value ? "active" : ""} onClick={() => setSymbolFilter(value)}>
                {value === "all" ? "전체 종목" : value}
              </button>
            ))}
          </div>
          <div className="trade-filter-group result" aria-label="청산 방식 필터">
            {([
              ["all", "전체 결과"], ["profit", "익절"], ["stop", "손절"], ["time", "시간청산"],
            ] as const).map(([value, label]) => (
              <button key={value} type="button" className={exitFilter === value ? "active" : ""} onClick={() => setExitFilter(value)}>{label}</button>
            ))}
          </div>
        </header>

        {visible.length === 0 ? (
          <div className="trade-empty">
            <strong>이 조건에 해당하는 완료 거래가 없습니다</strong>
            <p>새 거래가 최종 청산되면 진입 근거와 체결 과정까지 이곳에 기록됩니다.</p>
          </div>
        ) : (
          <div className="trade-list">
            {visible.map((trade) => {
              const expanded = expandedId === trade.id;
              const exit = exitCopy(trade.exitMode);
              const margin = trade.notional / Math.max(1, trade.leverage);
              const roe = margin ? trade.pnl / margin * 100 : 0;
              const rsi = typeof trade.context.rsi === "number" ? trade.context.rsi : null;
              return (
                <article key={trade.id} className={`trade-record ${expanded ? "expanded" : ""}`}>
                  <button
                    type="button"
                    className="trade-record-summary"
                    aria-expanded={expanded}
                    onClick={() => setExpandedId(expanded ? null : trade.id)}
                  >
                    <span className="trade-record-time">{formatKst(trade.exitTime)}<small>{holdingTime(trade.holdingSeconds)} 보유</small></span>
                    <span className="trade-record-symbol"><b>{trade.symbol.replace("_", " / ")}</b><small>{trade.leverage}× · {trade.entryOrder === "limit" ? "지정가" : "시장가"}</small></span>
                    <span className="trade-record-setup">
                      <span className={`side-chip ${trade.side}`}>{trade.side.toUpperCase()}</span>
                      <span className="trade-record-pattern"><b>{patternLabel(trade)}</b><small>{trade.strategy}</small></span>
                    </span>
                    <span className={`exit-badge ${exit.tone}`}>{exit.label}</span>
                    <span className={`trade-record-pnl ${trade.pnl >= 0 ? "up" : "down"}`}>
                      <b>{trade.pnl >= 0 ? "+" : ""}{trade.pnl.toFixed(2)} USDT</b>
                      <small>{formatPercent(roe)}</small>
                    </span>
                    <span className="trade-expand-mark">{expanded ? "−" : "+"}</span>
                  </button>

                  {expanded && (
                    <div className="trade-record-detail">
                      <div className="trade-decision">
                        <div className="trade-detail-title">
                          <div><span>ENTRY DECISION</span><strong>{patternLabel(trade)} 신호로 {trade.side === "long" ? "롱" : "숏"} 진입</strong></div>
                          <div><span>판정 점수</span><b>{trade.score ? `${trade.score.toFixed(1)}점` : "—"}</b></div>
                        </div>
                        <div className="reason-chips">
                          {trade.reasons.length
                            ? trade.reasons.map((reason) => <span key={reason}>{reasonLabel(reason)}</span>)
                            : <span>기록된 세부 근거 없음</span>}
                        </div>
                        {rsi != null && <p>진입 당시 RSI <b>{rsi.toFixed(1)}</b> · 손절 {tradePrice(trade.stop, trade.symbol)} · 1차 목표 {tradePrice(trade.tp1, trade.symbol)}</p>}
                      </div>

                      <div className="execution-path">
                        <div className="execution-step entry">
                          <i>IN</i><span>진입</span><strong>{tradePrice(trade.entryPrice, trade.symbol)}</strong>
                          <small>{formatKst(trade.entryTime)} · {trade.entryOrder === "limit" ? "메이커 지정가" : "테이커 시장가"}</small>
                        </div>
                        {trade.exitEvents.map((event, index) => (
                          <div key={`${event.time}-${index}`} className={`execution-step ${event.reason}`}>
                            <i>{event.reason === "tp1" ? "½" : "OUT"}</i>
                            <span>{eventLabel(event.reason)}</span>
                            <strong>{tradePrice(event.price, trade.symbol)}</strong>
                            <small>{Math.round(event.ratio * 100)}% · {event.pnl >= 0 ? "+" : ""}{event.pnl.toFixed(2)} USDT{event.note ? ` · ${event.note}` : ""}</small>
                          </div>
                        ))}
                      </div>

                      <div className="trade-number-grid">
                        <div><span>평균 청산가</span><b>{tradePrice(trade.exitPrice, trade.symbol)}</b></div>
                        <div><span>계약 / 명목</span><b>{trade.contracts.toLocaleString()} / {balanceFormatter.format(trade.notional)} USDT</b></div>
                        <div><span>수수료·리베이트</span><b>{trade.fees >= 0 ? "+" : ""}{trade.fees.toFixed(4)} USDT</b></div>
                        <div><span>청산 후 잔고</span><b>{balanceFormatter.format(trade.balanceAfter)} USDT</b></div>
                      </div>
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        )}
      </article>
    </section>
  );
}

export function BtcDashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [liveCandles, setLiveCandles] = useState<Record<Timeframe, Candle[]> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [timeframe, setTimeframe] = useState<Timeframe>("1h");
  const [showChannel, setShowChannel] = useState(true);
  const [showAverages, setShowAverages] = useState(false);
  const [showMomentum, setShowMomentum] = useState(true);
  const [liveStatus, setLiveStatus] = useState<LiveStatus>("connecting");
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);
  const [frozenSignal, setFrozenSignal] = useState<(StrategyPoint & { detectedAt: number }) | null>(null);
  const [chartExpanded, setChartExpanded] = useState(false);
  const [historyLoading, setHistoryLoading] = useState<Partial<Record<Timeframe, boolean>>>({});
  const [historyExhausted, setHistoryExhausted] = useState<Partial<Record<Timeframe, boolean>>>({});
  const liveCandlesRef = useRef(liveCandles);
  const historyInFlight = useRef(new Set<Timeframe>());
  const liveReady = liveCandles !== null;

  useEffect(() => {
    liveCandlesRef.current = liveCandles;
  }, [liveCandles]);

  const loadOlder = useCallback(async (target: Timeframe) => {
    if (historyInFlight.current.has(target)) return;
    const current = liveCandlesRef.current?.[target];
    if (!current?.length || current.length >= 5_000) {
      setHistoryExhausted((previous) => ({ ...previous, [target]: true }));
      return;
    }

    historyInFlight.current.add(target);
    setHistoryLoading((previous) => ({ ...previous, [target]: true }));
    try {
      const before = current[0].time;
      const response = await fetch(
        `/api/btc-live?tf=${target}&before=${before}&limit=800`,
        { cache: "no-store" },
      );
      const payload = await response.json() as LiveResponse;
      if (!response.ok) throw new Error(payload.error ?? "과거 캔들 요청 실패");
      const updates = payload.candles[target] ?? [];
      const older = updates.filter((candle) => candle.time < before);
      if (!older.length) {
        setHistoryExhausted((previous) => ({ ...previous, [target]: true }));
        return;
      }
      const existingTimes = new Set(current.map((candle) => candle.time));
      const addedCount = older.filter((candle) => !existingTimes.has(candle.time)).length;
      if (current.length + addedCount >= 5_000) {
        setHistoryExhausted((previous) => ({ ...previous, [target]: true }));
      }
      setLiveCandles((previous) => {
        if (!previous) return previous;
        const merged = enrichCandles(mergeRawCandles(previous[target], older, 5_000), 100, 2, target);
        return { ...previous, [target]: merged };
      });
    } catch {
      // 일시적인 네트워크 실패는 다음 이동/버튼 입력에서 다시 시도한다.
    } finally {
      historyInFlight.current.delete(target);
      setHistoryLoading((previous) => ({ ...previous, [target]: false }));
    }
  }, []);

  useEffect(() => {
    if (!chartExpanded) return;
    const previousOverflow = document.body.style.overflow;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setChartExpanded(false);
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [chartExpanded]);

  useEffect(() => {
    let disposed = false;
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
            100,
            2,
            item.value,
          );
        }
        if (disposed) return;
        setData(snapshot);
        setLiveCandles(initial);

        void Promise.allSettled(TIMEFRAMES.map(async ({ value: target }) => {
          const after = initial[target].at(-1)?.time;
          if (after == null) return;
          const response = await fetch(`/api/btc-live?tf=${target}&after=${after}`, { cache: "no-store" });
          const payload = await response.json() as LiveResponse;
          if (!response.ok) throw new Error(payload.error ?? "누락 캔들 요청 실패");
          const updates = payload.candles[target] ?? [];
          if (disposed || !updates.length) return;
          setLiveCandles((previous) => {
            if (!previous) return previous;
            const raw = payload.truncated
              ? updates
              : mergeRawCandles(previous[target], updates, 5_000);
            return {
              ...previous,
              [target]: enrichCandles(raw, 100, 2, target),
            };
          });
        }));
      })
      .catch((reason: Error) => setError(reason.message));
    return () => { disposed = true; };
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
            next[tf] = enrichCandles(mergeRawCandles(next[tf], updates), 100, 2, tf);
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

  const chartTrades = useMemo(
    () => (data?.trades ?? []).filter((trade) => trade.symbol.startsWith("BTC")),
    [data?.trades],
  );

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
  const selectedMomentum = liveCandles[timeframe].at(-1);

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
          <Link className="nav-item active" href="/">BTC 회귀채널</Link>
          <Link className="nav-item" href="/btc-25x">BTC 25×</Link>
          <Link className="nav-item" href="/xrp-patterns">XRP 패턴</Link>
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
        <article className={`panel chart-panel ${chartExpanded ? "expanded" : ""}`}>
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
              <button
                type="button"
                className={showMomentum ? "active momentum" : ""}
                aria-pressed={showMomentum}
                onClick={() => setShowMomentum((value) => !value)}
              >
                <i /> RSI / MACD
              </button>
              <button
                type="button"
                className="screen-toggle"
                aria-pressed={chartExpanded}
                onClick={() => setChartExpanded((value) => !value)}
              >
                {chartExpanded ? "화면 축소" : "화면 확대"}
              </button>
            </div>
          </div>
          <RegressionChart
            key={timeframe}
            candles={liveCandles[timeframe]}
            timeframe={timeframe}
            showChannel={showChannel}
            showAverages={showAverages}
            expanded={chartExpanded}
            historyLoading={historyLoading[timeframe] === true}
            historyExhausted={historyExhausted[timeframe] === true}
            onLoadOlder={loadOlder}
            trades={chartTrades}
          />
          {showMomentum && <MomentumStrip candle={selectedMomentum} timeframe={timeframe} />}
          <footer className="chart-footer">
            <span><i className="legend-candle up" /> 상승</span>
            <span><i className="legend-candle down" /> 하락</span>
            <span><i className="legend-line channel" /> 회귀 상·하단</span>
            <span><i className="legend-line live" /> 진행 중 계산</span>
            <span className="chart-note"><b>연속 100봉 기준 · 결측 자동 보정</b> · 드래그·두 손가락 스와이프 이동 · 핀치 확대 · 최대 5,000봉</span>
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

      <TradeHistory trades={data.trades ?? []} generatedAt={data.generatedAt} />

      <footer className="page-footer">
        <span>DATA · {liveStatus === "online" ? "Gate.io live public candles" : data.source}</span>
        <span>읽기 전용 화면 · 주문 기능 없음</span>
      </footer>
    </main>
  );
}
