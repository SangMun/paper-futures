export type Timeframe = "1m" | "5m" | "15m" | "1h" | "4h" | "1d";

export type RawCandle = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  quoteVolume?: number;
  closed?: boolean;
};

export type ConfirmedPivot = {
  index: number;
  time: number;
  price: number;
  kind: "high" | "low";
  confirmedAt: number;
};

export type EnrichedCandle = RawCandle & {
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
  channelMovePct?: number | null;
  sma20?: number | null;
  sma25?: number | null;
  sma50?: number | null;
  vwap?: number | null;
  rvol?: number | null;
  deviation?: number | null;
  deviationRank?: number | null;
  spike?: boolean;
};

export type StrategyPoint = {
  signal: "waiting" | "long_watch" | "short_watch";
  side: "long" | "short" | null;
  entry: number | null;
  target: number | null;
  candleTime: number | null;
  slope: number | null;
  r2: number | null;
  atr: number | null;
  mid: number | null;
  upper: number | null;
  lower: number | null;
  majorRegime: "bull" | "bear" | "none";
  partial?: boolean;
};

export type Btc25xPoint = {
  status: "warming" | "waiting" | "long_ready" | "short_ready";
  side: "long" | "short" | null;
  blockReason: string;
  partial?: boolean;
  candleTime?: number;
  close?: number;
  entry?: number | null;
  stop?: number | null;
  target?: number | null;
  liquidation?: number | null;
  targetMovePct?: number | null;
  targetMarginRoe?: number;
  entryDistancePct?: number | null;
  mid?: number | null;
  upper?: number | null;
  lower?: number | null;
  slope?: number | null;
  r2?: number | null;
  movePct?: number | null;
  majorRegime?: "bull" | "bear" | "none";
  trendAligned?: boolean;
  gates?: Record<"direction" | "channel" | "r2" | "move" | "priceInside" | "distance" | "regime", boolean>;
};

export const TIMEFRAME_SECONDS: Readonly<Record<Timeframe, number>>;
export function normalizeGateCandles(raw: unknown[], timeframe: Timeframe, serverTime: number): RawCandle[];
export function mergeRawCandles(existing: RawCandle[], updates: RawCandle[], maxLength?: number): RawCandle[];
export function confirmedPivots(candles: RawCandle[], n?: number): ConfirmedPivot[];
export function enrichCandles(rawCandles: RawCandle[], length?: number, widthMult?: number, timeframe?: Timeframe | null): EnrichedCandle[];
export function enrichXrpCandles(rawCandles: RawCandle[], timeframe?: Timeframe | null): EnrichedCandle[];
export function buildLiveStrategy(hourly: EnrichedCandle[], major: EnrichedCandle[], entryOffsetPct?: number): {
  live: StrategyPoint;
  confirmed: StrategyPoint;
};
export function alignBtc25xChannel(entryCandles: EnrichedCandle[], channelCandles: EnrichedCandle[], channelLength?: number): EnrichedCandle[];
export function buildBtc25xPreview(entryCandles: EnrichedCandle[], channelCandles: EnrichedCandle[], majorCandles: EnrichedCandle[], settings?: Record<string, number>): {
  live: Btc25xPoint;
  confirmed: Btc25xPoint;
  chart: EnrichedCandle[];
};
