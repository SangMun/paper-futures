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
    // The paper engine calculates VWAP/RVOL from Gate's contract volume (`v`).
    // `sum` is quote turnover and is kept separately for the volume histogram.
    const volume = finite(item?.v ?? item?.sum ?? 0);
    const quoteVolume = finite(item?.sum);
    if (![time, open, high, low, close, volume].every(Number.isFinite)) continue;
    byTime.set(time, {
      time,
      open,
      high,
      low,
      close,
      volume,
      ...(Number.isFinite(quoteVolume) ? { quoteVolume } : {}),
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
      ...(Number.isFinite(candle.quoteVolume) ? { quoteVolume: candle.quoteVolume } : {}),
      closed: candle.closed !== false,
    });
  }
  return [...byTime.values()]
    .sort((a, b) => a.time - b.time)
    .slice(-maxLength);
}

export function confirmedPivots(candles, n = 5) {
  if (!Number.isInteger(n) || n < 1 || candles.length < n * 2 + 1) return [];
  const pivots = [];
  const previousPrice = new Map();

  for (let index = n; index < candles.length - n; index += 1) {
    const candle = candles[index];
    const window = candles.slice(index - n, index + n + 1);
    let isHigh = candle.high === Math.max(...window.map((item) => item.high));
    let isLow = candle.low === Math.min(...window.map((item) => item.low));
    if (isHigh && isLow) {
      isHigh = false;
      isLow = false;
    }
    if (!isHigh && !isLow) continue;

    const kind = isHigh ? "high" : "low";
    const price = isHigh ? candle.high : candle.low;
    const previous = pivots.at(-1);
    if (previous?.kind === kind && previousPrice.get(kind) === price) continue;
    previousPrice.set(kind, price);
    pivots.push({
      index,
      time: candle.time,
      price,
      kind,
      confirmedAt: index + n,
    });
  }
  return pivots;
}

function ema(values, period) {
  const result = Array(values.length).fill(null);
  const start = values.findIndex(Number.isFinite);
  if (start < 0 || start + period > values.length) return result;
  const seed = values.slice(start, start + period);
  if (!seed.every(Number.isFinite)) return result;
  let previous = seed.reduce((sum, value) => sum + value, 0) / period;
  result[start + period - 1] = previous;
  const alpha = 2 / (period + 1);
  for (let index = start + period; index < values.length; index += 1) {
    const value = values[index];
    if (Number.isFinite(value)) previous += alpha * (value - previous);
    result[index] = previous;
  }
  return result;
}

function sma(values, period) {
  const result = Array(values.length).fill(null);
  let sum = 0;
  let valid = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (Number.isFinite(value)) {
      sum += value;
      valid += 1;
    }
    if (index >= period) {
      const leaving = values[index - period];
      if (Number.isFinite(leaving)) {
        sum -= leaving;
        valid -= 1;
      }
    }
    if (index >= period - 1 && valid === period) result[index] = sum / period;
  }
  return result;
}

function sessionVwap(candles) {
  const result = Array(candles.length).fill(null);
  let session = null;
  let priceVolume = 0;
  let volume = 0;
  candles.forEach((candle, index) => {
    const nextSession = Math.floor(candle.time / 86_400);
    if (nextSession !== session) {
      session = nextSession;
      priceVolume = 0;
      volume = 0;
    }
    const typical = (candle.high + candle.low + candle.close) / 3;
    priceVolume += typical * candle.volume;
    volume += candle.volume;
    result[index] = volume > 0 ? priceVolume / volume : null;
  });
  return result;
}

function relativeVolume(candles, lookback = 20) {
  return candles.map((candle, index) => {
    if (index < lookback) return null;
    const prior = candles.slice(index - lookback, index);
    const average = prior.reduce((sum, item) => sum + item.volume, 0) / lookback;
    return average > 0 ? candle.volume / average : null;
  });
}

function rollingDeviationRank(deviations, lookback = 400) {
  return deviations.map((value, index) => {
    if (!Number.isFinite(value) || index < lookback - 1) return null;
    const window = deviations.slice(index - lookback + 1, index + 1);
    if (!window.every(Number.isFinite)) return null;
    let less = 0;
    let equal = 0;
    window.forEach((candidate) => {
      if (candidate < value) less += 1;
      else if (candidate === value) equal += 1;
    });
    return (less + (equal + 1) / 2) / lookback * 100;
  });
}

function spikeFlags(candles, lookback = 100, multiple = 5, cooldown = 6) {
  const ranges = candles.map((candle) => (
    candle.close > 0 ? (candle.high - candle.low) / candle.close : null
  ));
  const hits = ranges.map((value, index) => {
    if (!Number.isFinite(value) || index < lookback - 1) return false;
    const window = ranges.slice(index - lookback + 1, index + 1).filter(Number.isFinite).sort((a, b) => a - b);
    if (window.length !== lookback) return false;
    const middle = lookback / 2;
    const median = (window[middle - 1] + window[middle]) / 2;
    return value > median * multiple;
  });
  let remaining = 0;
  return hits.map((hit) => {
    if (hit) remaining = cooldown;
    const blocked = remaining > 0;
    if (remaining > 0) remaining -= 1;
    return blocked;
  });
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

function rsi(values, period = 14) {
  const gains = values.map((value, index) => (
    index === 0 ? null : Math.max(0, value - values[index - 1])
  ));
  const losses = values.map((value, index) => (
    index === 0 ? null : Math.max(0, values[index - 1] - value)
  ));
  const averageGain = wilder(gains, period);
  const averageLoss = wilder(losses, period);
  return values.map((_, index) => {
    const gain = averageGain[index];
    const loss = averageLoss[index];
    if (!Number.isFinite(gain) || !Number.isFinite(loss)) return null;
    if (loss === 0) return 100;
    const relativeStrength = gain / loss;
    return 100 - (100 / (1 + relativeStrength));
  });
}

function macd(values, fast = 12, slow = 26, signalPeriod = 9) {
  const fastLine = ema(values, fast);
  const slowLine = ema(values, slow);
  const line = values.map((_, index) => (
    Number.isFinite(fastLine[index]) && Number.isFinite(slowLine[index])
      ? fastLine[index] - slowLine[index]
      : null
  ));
  const signal = ema(line, signalPeriod);
  const histogram = line.map((value, index) => (
    Number.isFinite(value) && Number.isFinite(signal[index])
      ? value - signal[index]
      : null
  ));
  return { line, signal, histogram };
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

function enrichContinuous(rawCandles, length, widthMult) {
  const closes = rawCandles.map((candle) => candle.close);
  const channels = rollingRegression(closes, length, widthMult);
  const ema20 = ema(closes, 20);
  const ema60 = ema(closes, 60);
  const atr14 = atr(rawCandles, 14);
  const rsi14 = rsi(closes, 14);
  const momentum = macd(closes, 12, 26, 9);

  return rawCandles.map((candle, index) => ({
    ...candle,
    ...channels[index],
    ema20: ema20[index],
    ema60: ema60[index],
    atr: atr14[index],
    rsi14: rsi14[index],
    macd: momentum.line[index],
    macdSignal: momentum.signal[index],
    macdHist: momentum.histogram[index],
  }));
}

export function enrichCandles(rawCandles, length = 100, widthMult = 2, timeframe = null) {
  const step = timeframe == null ? null : TIMEFRAME_SECONDS[timeframe];
  if (!step || rawCandles.length < 2) {
    return enrichContinuous(rawCandles, length, widthMult);
  }

  const enriched = [];
  let segmentStart = 0;
  for (let index = 1; index <= rawCandles.length; index += 1) {
    const atEnd = index === rawCandles.length;
    const hasGap = !atEnd && rawCandles[index].time - rawCandles[index - 1].time > step * 1.5;
    if (!atEnd && !hasGap) continue;
    enriched.push(...enrichContinuous(rawCandles.slice(segmentStart, index), length, widthMult));
    segmentStart = index;
  }
  return enriched;
}

export function enrichXrpCandles(rawCandles, timeframe = null) {
  const enriched = enrichCandles(rawCandles, 100, 2, timeframe);
  const closes = enriched.map((candle) => candle.close);
  const sma20 = sma(closes, 20);
  const sma25 = sma(closes, 25);
  const sma50 = sma(closes, 50);
  const vwap = sessionVwap(enriched);
  const rvol = relativeVolume(enriched, 20);
  const deviations = closes.map((close, index) => (
    Number.isFinite(sma25[index]) && sma25[index] > 0 ? (close - sma25[index]) / sma25[index] : null
  ));
  const deviationRank = rollingDeviationRank(deviations, 400);
  const spike = spikeFlags(enriched, 100, 5, 6);
  return enriched.map((candle, index) => ({
    ...candle,
    sma20: sma20[index],
    sma25: sma25[index],
    sma50: sma50[index],
    vwap: vwap[index],
    rvol: rvol[index],
    deviation: deviations[index],
    deviationRank: deviationRank[index],
    spike: spike[index],
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

function contextAt(entryTime, entryStep, contextCandles, contextStep) {
  const deadline = entryTime + entryStep - contextStep;
  for (let index = contextCandles.length - 1; index >= 0; index -= 1) {
    if (contextCandles[index].time <= deadline) return contextCandles[index];
  }
  return null;
}

export function alignBtc25xChannel(entryCandles, channelCandles, channelLength = 48) {
  return entryCandles.map((candle) => {
    const context = contextAt(candle.time, 300, channelCandles, 900);
    if (!context || context.mid == null || context.slope == null) {
      return {
        ...candle,
        mid: null,
        upper: null,
        lower: null,
        slope: null,
        r2: null,
        channelMovePct: null,
      };
    }
    const elapsed = Math.max(0, (candle.time + 300 - (context.time + 900)) / 900);
    const mid = context.mid + context.slope * elapsed;
    const upper = context.upper == null ? null : context.upper + context.slope * elapsed;
    const lower = context.lower == null ? null : context.lower + context.slope * elapsed;
    return {
      ...candle,
      mid,
      upper,
      lower,
      slope: context.slope,
      r2: context.r2,
      channelMovePct: mid > 0 ? Math.abs(context.slope) * (channelLength - 1) / mid : null,
    };
  });
}

function btc25xPoint(entryCandles, channelCandles, majorCandles, settings) {
  const cfg = {
    channelLength: 48,
    channelR2Min: 0.68,
    channelMinMovePct: 0.004,
    maxEntryDistancePct: 0.005,
    leverage: 25,
    marginFraction: 0.10,
    accountTakeProfitPct: 0.02,
    makerRate: -0.0001,
    ...settings,
  };
  const aligned = alignBtc25xChannel(entryCandles, channelCandles, cfg.channelLength);
  const latest = aligned.at(-1);
  if (!latest || latest.mid == null || latest.upper == null || latest.lower == null || latest.slope == null) {
    return { status: "warming", side: null, blockReason: "채널 계산에 필요한 봉을 모으는 중" };
  }

  const major = contextAt(latest.time, 300, majorCandles, 14_400);
  const majorRegime = !major || major.ema20 == null || major.ema60 == null
    ? "none"
    : major.close > major.ema20 && major.ema20 > major.ema60
      ? "bull"
      : major.close < major.ema20 && major.ema20 < major.ema60 ? "bear" : "none";
  const side = latest.slope > 0 ? "long" : latest.slope < 0 ? "short" : null;
  const entry = side === "long" ? latest.lower : side === "short" ? latest.upper : null;
  const priceInside = side === "long"
    ? entry < latest.close && latest.close <= latest.upper
    : side === "short" ? latest.lower <= latest.close && latest.close < entry : false;
  const entryDistancePct = entry && latest.close > 0 ? Math.abs(latest.close - entry) / latest.close : null;
  const movePct = latest.channelMovePct;
  const trendAligned = (majorRegime === "bull" && side === "long")
    || (majorRegime === "bear" && side === "short");
  const channelValid = (latest.atr ?? 0) > 0
    && 0 < latest.lower && latest.lower < latest.mid && latest.mid < latest.upper;
  const targetMarginRoe = cfg.accountTakeProfitPct / cfg.marginFraction;
  const denominator = side === "long" ? 1 - cfg.makerRate : 1 + cfg.makerRate;
  const targetMovePct = side
    ? (targetMarginRoe / cfg.leverage + 2 * cfg.makerRate) / denominator
    : null;
  const direction = side === "long" ? 1 : -1;
  const stop = entry && targetMovePct ? entry * (1 - direction * targetMovePct) : null;
  const target = entry && targetMovePct ? entry * (1 + direction * targetMovePct) : null;
  const liquidation = entry && side
    ? entry * (side === "long" ? 1 - (1 / cfg.leverage - 0.003) : 1 + (1 / cfg.leverage - 0.003))
    : null;
  const gates = {
    direction: side != null,
    channel: channelValid,
    r2: (latest.r2 ?? 0) >= cfg.channelR2Min,
    move: (movePct ?? 0) >= cfg.channelMinMovePct,
    priceInside,
    distance: entryDistancePct != null && entryDistancePct <= cfg.maxEntryDistancePct,
    regime: trendAligned,
  };
  const ready = Object.values(gates).every(Boolean);
  const blockReason = !gates.channel
    ? "ATR 또는 회귀채널 값이 유효하지 않음"
    : !gates.r2
      ? `채널 R²가 ${cfg.channelR2Min.toFixed(2)} 미만`
    : !gates.move
      ? "12시간 채널 이동폭이 0.4% 미만"
        : !gates.priceInside
          ? "현재가가 유효 채널 범위를 벗어남"
          : !gates.distance
            ? "채널 경계까지 거리가 0.5% 초과"
            : !gates.regime ? "4시간 대세 방향과 15분 채널이 불일치" : "모든 진입 조건 충족";

  return {
    status: ready ? `${side}_ready` : "waiting",
    side,
    blockReason,
    partial: latest.closed === false,
    candleTime: latest.time,
    close: latest.close,
    entry,
    stop,
    target,
    liquidation,
    targetMovePct,
    targetMarginRoe,
    entryDistancePct,
    mid: latest.mid,
    upper: latest.upper,
    lower: latest.lower,
    slope: latest.slope,
    r2: latest.r2,
    movePct,
    majorRegime,
    trendAligned,
    gates,
  };
}

export function buildBtc25xPreview(entryCandles, channelCandles, majorCandles, settings = {}) {
  const live = btc25xPoint(entryCandles, channelCandles, majorCandles, settings);
  const confirmedEntry = entryCandles.filter((candle) => candle.closed !== false);
  const confirmed = btc25xPoint(confirmedEntry, channelCandles, majorCandles, settings);
  return {
    live,
    confirmed,
    chart: alignBtc25xChannel(entryCandles, channelCandles, settings.channelLength ?? 48),
  };
}
