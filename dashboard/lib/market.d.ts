export type Timeframe = "1m" | "5m" | "15m" | "1h" | "4h" | "1d";

export type RawCandle = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  closed?: boolean;
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

export const TIMEFRAME_SECONDS: Readonly<Record<Timeframe, number>>;
export function normalizeGateCandles(raw: unknown[], timeframe: Timeframe, serverTime: number): RawCandle[];
export function mergeRawCandles(existing: RawCandle[], updates: RawCandle[], maxLength?: number): RawCandle[];
export function enrichCandles(rawCandles: RawCandle[], length?: number, widthMult?: number): EnrichedCandle[];
export function buildLiveStrategy(hourly: EnrichedCandle[], major: EnrichedCandle[], entryOffsetPct?: number): {
  live: StrategyPoint;
  confirmed: StrategyPoint;
};
