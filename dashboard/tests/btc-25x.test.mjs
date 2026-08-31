import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  alignBtc25xChannel,
  buildBtc25xPreview,
  enrichCandles,
} from "../lib/market.mjs";

const closeEnough = (actual, expected, tolerance = 1e-10) => {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${actual} != ${expected} (tolerance ${tolerance})`,
  );
};

test("BTC 25x preview uses the exported live strategy parameters", async () => {
  const snapshot = JSON.parse(
    await readFile(new URL("../public/btc-dashboard.json", import.meta.url), "utf8"),
  );
  const config = snapshot.strategy25x;
  assert.deepEqual(
    {
      channelLength: config.channelLength,
      channelR2Min: config.channelR2Min,
      leverage: config.leverage,
      marginFraction: config.marginFraction,
      accountTakeProfitPct: config.accountTakeProfitPct,
      maxDailyEntries: config.maxDailyEntries,
      consecutiveLossLimit: config.consecutiveLossLimit,
    },
    {
      channelLength: 48,
      channelR2Min: 0.68,
      leverage: 25,
      marginFraction: 0.1,
      accountTakeProfitPct: 0.02,
      maxDailyEntries: 12,
      consecutiveLossLimit: 3,
    },
  );

  const entry = enrichCandles(
    snapshot.candles["5m"].map((candle) => ({ ...candle, closed: true })),
    100,
    2,
    "5m",
  );
  const channel = enrichCandles(
    snapshot.candles["15m"].map((candle) => ({ ...candle, closed: true })),
    config.channelLength,
    config.channelWidthMult,
    "15m",
  );
  const major = enrichCandles(
    snapshot.candles["4h"].map((candle) => ({ ...candle, closed: true })),
    100,
    2,
    "4h",
  );
  const preview = buildBtc25xPreview(entry, channel, major, config);

  assert.equal(preview.chart.length, entry.length);
  assert.equal(preview.chart.at(-1).time, entry.at(-1).time);
  assert.ok(["long", "short"].includes(preview.live.side));
  assert.equal(typeof preview.live.blockReason, "string");
  assert.deepEqual(Object.keys(preview.live.gates), [
    "direction", "channel", "r2", "move", "priceInside", "distance", "regime",
  ]);

  const targetMarginRoe = config.accountTakeProfitPct / config.marginFraction;
  const denominator = preview.live.side === "long"
    ? 1 - config.makerRate
    : 1 + config.makerRate;
  const expectedMove = (targetMarginRoe / config.leverage + 2 * config.makerRate) / denominator;
  closeEnough(preview.live.targetMovePct, expectedMove);
  closeEnough(
    Math.abs(preview.live.target - preview.live.entry) / preview.live.entry,
    expectedMove,
  );
  closeEnough(
    Math.abs(preview.live.stop - preview.live.entry) / preview.live.entry,
    expectedMove,
  );
});

test("5m chart never reads a 15m channel before that candle has closed", () => {
  const entry = [0, 300, 600, 900, 1200, 1500].map((time) => ({
    time,
    open: 100,
    high: 101,
    low: 99,
    close: 100,
    volume: 1,
    atr: 1,
    closed: true,
  }));
  const channel = [
    { time: 0, mid: 100, upper: 105, lower: 95, slope: 3, r2: 0.9 },
    { time: 900, mid: 200, upper: 205, lower: 195, slope: 6, r2: 0.95 },
  ];

  const aligned = alignBtc25xChannel(entry, channel, 48);
  assert.equal(aligned[0].mid, null);
  assert.equal(aligned[1].mid, null);
  closeEnough(aligned[2].mid, 100);
  closeEnough(aligned[3].mid, 101);
  closeEnough(aligned[4].mid, 102);
  closeEnough(aligned[5].mid, 200);
});

test("BTC 25x desk exposes each strategy timeframe, order map and risk guard", async () => {
  const source = await readFile(
    new URL("../app/btc-25x/Btc25xDashboard.tsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /5분 체결/);
  assert.match(source, /15분 채널/);
  assert.match(source, /4시간 국면/);
  assert.match(source, /ORDER MAP/);
  assert.match(source, /ENTRY GATES/);
  assert.match(source, /25× 격리 포지션/);
  assert.match(source, /실제 확정봉/);
  assert.match(source, /btc_regime_25x/);
});
