export const TIMEFRAME_SECONDS = Object.freeze({
  "1m": 60,
  "5m": 300,
  "15m": 900,
  "1h": 3600,
  "4h": 14400,
  "1d": 86400,
});

const finite = (value) => {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
};

export function normalizeGateCandles(raw, timeframe, serverTime) {
  const step = TIMEFRAME_SECONDS[timeframe];
  if (!step || !Array.isArray(raw)) return [];

  const byTime = new Map();
  for (const item of raw) {
    const time = Number.parseInt(item?.t, 10);
    const open = finite(item?.o);
    const high = finite(item?.h);
    const low = finite(item?.l);
    const close = finite(item?.c);
    const volume = finite(item?.sum ?? item?.v ?? 0);
    if (![time, open, high, low, close, volume].every(Number.isFinite)) continue;
    byTime.set(time, {
      time,
      open,
      high,
      low,
      close,
      volume,
      closed: time + step <= serverTime,
    });
  }
  return [...byTime.values()].sort((a, b) => a.time - b.time);
}

export function mergeRawCandles(existing, updates, maxLength = 5_000) {
  const byTime = new Map();
  for (const candle of [...existing, ...updates]) {
    if (!Number.isFinite(candle?.time)) continue;
    byTime.set(candle.time, {
      time: candle.time,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
      volume: candle.volume,
      closed: candle.closed !== false,
    });
  }
  return [...byTime.values()]
    .sort((a, b) => a.time - b.time)
    .slice(-maxLength);
}

function ema(values, period) {
  const result = Array(values.length).fill(null);
  if (values.length < period) return result;
  const seed = values.slice(0, period);
  if (!seed.every(Number.isFinite)) return result;
  let previous = seed.reduce((sum, value) => sum + value, 0) / period;
  result[period - 1] = previous;
  const alpha = 2 / (period + 1);
  for (let index = period; index < values.length; index += 1) {
    const value = values[index];
    if (Number.isFinite(value)) previous += alpha * (value - previous);
    result[index] = previous;
  }
  return result;
}

function wilder(values, period) {
  const result = Array(values.length).fill(null);
  const start = values.findIndex(Number.isFinite);
  if (start < 0 || start + period > values.length) return result;
  const seed = values.slice(start, start + period);
  if (!seed.every(Number.isFinite)) return result;
  let previous = seed.reduce((sum, value) => sum + value, 0) / period;
  result[start + period - 1] = previous;
  const alpha = 1 / period;
  for (let index = start + period; index < values.length; index += 1) {
    const value = values[index];
    if (Number.isFinite(value)) previous += alpha * (value - previous);
    result[index] = previous;
  }
  return result;
}

function atr(candles, period = 14) {
  const trueRanges = candles.map((candle, index) => {
    if (index === 0) return null;
    const previousClose = candles[index - 1].close;
    return Math.max(
      candle.high - candle.low,
      Math.abs(candle.high - previousClose),
      Math.abs(candle.low - previousClose),
    );
  });
  return wilder(trueRanges, period);
}

function rollingRegression(values, length, widthMult) {
  const result = values.map(() => ({
    mid: null,
    upper: null,
    lower: null,
    slope: null,
    r2: null,
  }));
  if (values.length < length) return result;

  const xMean = (length - 1) / 2;
  let ssX = 0;
  for (let x = 0; x < length; x += 1) ssX += (x - xMean) ** 2;

  for (let end = length - 1; end < values.length; end += 1) {
    const window = values.slice(end - length + 1, end + 1);
    if (!window.every(Number.isFinite)) continue;
    const yMean = window.reduce((sum, value) => sum + value, 0) / length;
    let covariance = 0;
    let ssY = 0;
    for (let x = 0; x < length; x += 1) {
      const centeredY = window[x] - yMean;
      covariance += (x - xMean) * centeredY;
      ssY += centeredY ** 2;
    }
    const slope = covariance / ssX;
    const mid = yMean + slope * (length - 1 - xMean);
    const r2 = ssY > 0
      ? Math.max(0, Math.min(1, (covariance ** 2) / (ssX * ssY)))
      : 0;
    const variance = ssY / length;
    const deviation = Math.sqrt(variance) * Math.sqrt(Math.max(1 - r2, 0)) * widthMult;
    result[end] = {
      mid,
      upper: mid + deviation,
      lower: mid - deviation,
      slope,
      r2,
    };
  }
  return result;
}

export function enrichCandles(rawCandles, length = 100, widthMult = 2) {
  const closes = rawCandles.map((candle) => candle.close);
  const channels = rollingRegression(closes, length, widthMult);
  const ema20 = ema(closes, 20);
  const ema60 = ema(closes, 60);
  const atr14 = atr(rawCandles, 14);

  return rawCandles.map((candle, index) => ({
    ...candle,
    ...channels[index],
    ema20: ema20[index],
    ema60: ema60[index],
    atr: atr14[index],
  }));
}

function signalFrom(candles, entryOffsetPct) {
  const latest = candles.at(-1);
  const previous = candles.at(-2);
  if (!latest || !previous || latest.upper == null || latest.lower == null) {
    return { signal: "waiting", side: null, entry: null, target: null };
  }

  if (
    latest.slope < 0
    && previous.upper != null
    && latest.high > latest.upper
    && previous.high <= previous.upper
  ) {
    return {
      signal: "short_watch",
      side: "short",
      entry: latest.upper * (1 + entryOffsetPct),
      target: latest.upper,
    };
  }
  if (
    latest.slope > 0
    && previous.lower != null
    && latest.low < latest.lower
    && previous.low >= previous.lower
  ) {
    return {
      signal: "long_watch",
      side: "long",
      entry: latest.lower * (1 - entryOffsetPct),
      target: latest.lower,
    };
  }
  return { signal: "waiting", side: null, entry: null, target: null };
}

function regimeFrom(candles) {
  const latest = candles.at(-1);
  if (!latest || latest.ema20 == null || latest.ema60 == null) return "none";
  if (latest.close > latest.ema20 && latest.ema20 > latest.ema60) return "bull";
  if (latest.close < latest.ema20 && latest.ema20 < latest.ema60) return "bear";
  return "none";
}

export function buildLiveStrategy(hourly, major, entryOffsetPct = 0.02) {
  const confirmedHourly = hourly.filter((candle) => candle.closed !== false);
  const confirmedMajor = major.filter((candle) => candle.closed !== false);
  const live = hourly.at(-1);
  const confirmed = confirmedHourly.at(-1);
  return {
    live: {
      ...signalFrom(hourly, entryOffsetPct),
      candleTime: live?.time ?? null,
      partial: live?.closed === false,
      slope: live?.slope ?? null,
      r2: live?.r2 ?? null,
      atr: live?.atr ?? null,
      mid: live?.mid ?? null,
      upper: live?.upper ?? null,
      lower: live?.lower ?? null,
      majorRegime: regimeFrom(major),
    },
    confirmed: {
      ...signalFrom(confirmedHourly, entryOffsetPct),
      candleTime: confirmed?.time ?? null,
      slope: confirmed?.slope ?? null,
      r2: confirmed?.r2 ?? null,
      atr: confirmed?.atr ?? null,
      mid: confirmed?.mid ?? null,
      upper: confirmed?.upper ?? null,
      lower: confirmed?.lower ?? null,
      majorRegime: regimeFrom(confirmedMajor),
    },
  };
}
