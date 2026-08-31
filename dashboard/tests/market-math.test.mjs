import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  buildLiveStrategy,
  confirmedPivots,
  enrichCandles,
  mergeRawCandles,
  normalizeGateCandles,
} from "../lib/market.mjs";

const closeEnough = (actual, expected, tolerance = 1e-8) => {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${actual} != ${expected} (tolerance ${tolerance})`,
  );
};

test("browser indicator math matches the Python-generated snapshot", async () => {
  const snapshot = JSON.parse(
    await readFile(new URL("../public/btc-dashboard.json", import.meta.url), "utf8"),
  );
  for (const timeframe of ["1m", "5m", "15m", "1h", "4h", "1d"]) {
    const source = snapshot.candles[timeframe];
    const enriched = enrichCandles(source.map((candle) => ({ ...candle, closed: true })));
    const actual = enriched.at(-1);
    const expected = source.at(-1);
    for (const field of [
      "mid", "upper", "lower", "slope", "r2", "ema20", "ema60", "atr",
      "rsi14", "macd", "macdSignal", "macdHist",
    ]) {
      closeEnough(actual[field], expected[field]);
    }
  }
});

test("Gate contract volume is used for strategy math and quote turnover is preserved", () => {
  const [candle] = normalizeGateCandles([{
    t: "1000", o: "1.0", h: "1.1", l: "0.9", c: "1.05", v: "45", sum: "472.5",
  }], "5m", 1400);
  assert.equal(candle.volume, 45);
  assert.equal(candle.quoteVolume, 472.5);
});

test("browser pivots match the engine's confirmation delay and flat-candle rule", () => {
  const candles = Array.from({ length: 13 }, (_, index) => ({
    time: index * 300,
    open: 10,
    high: index === 6 ? 15 : 10 + index * 0.01,
    low: index === 3 ? 5 : 9 - index * 0.01,
    close: 10,
    volume: 100,
    closed: true,
  }));
  const pivots = confirmedPivots(candles, 2);
  assert.deepEqual(
    pivots.map(({ index, kind, confirmedAt }) => ({ index, kind, confirmedAt })),
    [
      { index: 3, kind: "low", confirmedAt: 5 },
      { index: 6, kind: "high", confirmedAt: 8 },
    ],
  );
});

test("partial hourly candle moves LIVE channel without changing CONFIRMED channel", async () => {
  const snapshot = JSON.parse(
    await readFile(new URL("../public/btc-dashboard.json", import.meta.url), "utf8"),
  );
  const originalHour = snapshot.candles["1h"].map((candle) => ({ ...candle, closed: true }));
  const originalMajor = snapshot.candles["4h"].map((candle) => ({ ...candle, closed: true }));
  const previous = originalHour.at(-1);
  const raw = [{
    t: String(previous.time + 3600),
    o: String(previous.close),
    h: String(previous.close * 1.015),
    l: String(previous.close * 0.998),
    c: String(previous.close * 1.01),
    sum: "1200000",
  }];
  const update = normalizeGateCandles(raw, "1h", previous.time + 3600 + 120);
  assert.equal(update[0].closed, false);

  const hourly = enrichCandles(mergeRawCandles(originalHour, update));
  const major = enrichCandles(originalMajor);
  const view = buildLiveStrategy(hourly, major);

  assert.equal(view.live.partial, true);
  assert.equal(view.live.candleTime, previous.time + 3600);
  assert.equal(view.confirmed.candleTime, previous.time);
  closeEnough(view.confirmed.upper, previous.upper);
  assert.notEqual(view.live.upper, view.confirmed.upper);
});

test("historical pages prepend without disturbing the newest candle", async () => {
  const snapshot = JSON.parse(
    await readFile(new URL("../public/btc-dashboard.json", import.meta.url), "utf8"),
  );
  const current = snapshot.candles["1h"].slice(-120).map((candle) => ({ ...candle, closed: true }));
  const first = current[0];
  const newest = current.at(-1);
  const older = [2, 1].map((distance) => ({
    time: first.time - distance * 3600,
    open: first.open,
    high: first.high,
    low: first.low,
    close: first.close,
    volume: first.volume,
    closed: true,
  }));

  const merged = enrichCandles(mergeRawCandles(current, older));
  assert.equal(merged.length, current.length + older.length);
  assert.equal(merged[0].time, first.time - 2 * 3600);
  assert.equal(merged.at(-1).time, newest.time);
  assert.ok(merged.every((candle, index) => index === 0 || candle.time > merged[index - 1].time));
});

test("indicator windows never bridge a missing candle range", () => {
  const makeCandle = (time, close) => ({
    time,
    open: close - 1,
    high: close + 2,
    low: close - 2,
    close,
    volume: 100,
    closed: true,
  });
  const oldSegment = Array.from({ length: 120 }, (_, index) => makeCandle(index * 60, 60_000 + index));
  const newSegment = Array.from(
    { length: 12 },
    (_, index) => makeCandle(20_000 + index * 60, 63_000 + index),
  );
  const enriched = enrichCandles([...oldSegment, ...newSegment], 100, 2, "1m");
  const latest = enriched.at(-1);

  assert.equal(enriched[119].mid == null, false);
  assert.equal(latest.mid, null);
  assert.equal(latest.rsi14, null);
  assert.equal(latest.macd, null);
});
