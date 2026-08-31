import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { enrichXrpCandles } from "../lib/market.mjs";

test("dashboard snapshot mirrors the nine XRP pattern instances registered in the engine", async () => {
  const snapshot = JSON.parse(
    await readFile(new URL("../public/btc-dashboard.json", import.meta.url), "utf8"),
  );
  assert.deepEqual(Object.keys(snapshot.xrpCandles), ["5m", "15m", "1h", "4h", "1d"]);
  assert.ok(Object.values(snapshot.xrpCandles).every((candles) => candles.length >= 500));
  assert.equal(snapshot.xrpPatterns.length, 9);
  assert.deepEqual(snapshot.xrpPatterns.map((monitor) => monitor.name), [
    "xrp_patterns.cup_handle.5m",
    "xrp_patterns.cup_handle.15m",
    "xrp_patterns.cup_handle.1h",
    "xrp_patterns.cup_handle.4h",
    "xrp_patterns.double_top.1h",
    "xrp_patterns.wedge.5m",
    "xrp_patterns.wedge.15m",
    "xrp_patterns.head_shoulders.5m",
    "xrp_patterns.kotegawa.1h",
  ]);
  assert.ok(!snapshot.xrpPatterns.some((monitor) => monitor.family === "triangle"));
  assert.equal(snapshot.xrpPatterns.find((monitor) => monitor.name === "xrp_patterns.wedge.5m").orderType, "limit");
  assert.deepEqual(
    snapshot.xrpPatterns.find((monitor) => monitor.name === "xrp_patterns.wedge.5m").blockedUtcHours,
    [16, 17, 18, 19, 20, 21, 22, 23],
  );
});

test("XRP browser indicators expose the same common gates used by pattern scanning", () => {
  const candles = Array.from({ length: 430 }, (_, index) => {
    const close = 1 + index * 0.0001;
    return {
      time: index * 300,
      open: close - 0.00005,
      high: close + 0.0002,
      low: close - 0.0002,
      close,
      volume: index === 429 ? 260 : 100,
      closed: true,
    };
  });
  const enriched = enrichXrpCandles(candles, "5m");
  const latest = enriched.at(-1);
  assert.equal(enriched.length, candles.length);
  assert.ok(Number.isFinite(latest.sma20));
  assert.ok(Number.isFinite(latest.sma25));
  assert.ok(Number.isFinite(latest.sma50));
  assert.ok(Number.isFinite(latest.vwap));
  assert.equal(latest.rvol, 2.6);
  assert.ok(Number.isFinite(latest.deviation));
  assert.ok(Number.isFinite(latest.deviationRank));
  assert.equal(typeof latest.spike, "boolean");
});

test("XRP VWAP and RVOL match the Python strategy snapshot", async () => {
  const snapshot = JSON.parse(
    await readFile(new URL("../public/btc-dashboard.json", import.meta.url), "utf8"),
  );
  for (const timeframe of ["5m", "15m", "1h", "4h"]) {
    const source = snapshot.xrpCandles[timeframe];
    const actual = enrichXrpCandles(source.map((candle) => ({ ...candle, closed: true })), timeframe).at(-1);
    const expected = source.at(-1);
    assert.ok(Math.abs(actual.vwap - expected.vwap) <= 1e-10);
    assert.ok(Math.abs(actual.rvol - expected.rvol) <= 1e-10);
  }
});

test("XRP page shows live filters, active fleet, blocked events and read-only execution history", async () => {
  const [source, route] = await Promise.all([
    readFile(new URL("../app/xrp-patterns/XrpPatternDashboard.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/xrp-live/route.ts", import.meta.url), "utf8"),
  ]);
  assert.match(source, /5분 패턴/);
  assert.match(source, /15분 패턴/);
  assert.match(source, /1시간 패턴/);
  assert.match(source, /4시간 패턴/);
  assert.match(source, /SMA 20\/50 · VWAP · 확정 PIVOT 5/);
  assert.match(source, /ACTIVE ON/);
  assert.match(source, /삼각수렴은 현재 손실 검증 결과로 비활성/);
  assert.match(source, /최근 판단 로그/);
  assert.match(source, /TradeHistory/);
  assert.match(route, /contract: "XRP_USDT"/);
  assert.match(route, /fetchCandles\("BTC_USDT", "1h"\)/);
});
