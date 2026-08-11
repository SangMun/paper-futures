import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  buildLiveStrategy,
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
    for (const field of ["mid", "upper", "lower", "slope", "r2", "ema20", "ema60", "atr"]) {
      closeEnough(actual[field], expected[field]);
    }
  }
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
