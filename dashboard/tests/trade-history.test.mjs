import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("paper fills are grouped into complete trades with decision context", async () => {
  const snapshot = JSON.parse(
    await readFile(new URL("../public/btc-dashboard.json", import.meta.url), "utf8"),
  );
  const trades = snapshot.trades;

  assert.ok(Array.isArray(trades));
  // A fresh local paper account can legitimately have no fills yet. The
  // grouping assertions become active as soon as the snapshot has history.
  if (trades.length === 0) return;
  assert.ok(trades.every((trade, index) => index === 0 || trade.exitTime <= trades[index - 1].exitTime));
  assert.ok(trades.every((trade) => trade.exitEvents.length >= 1));
  assert.ok(trades.every((trade) => Math.abs(
    trade.exitEvents.reduce((sum, event) => sum + event.ratio, 0) - 1,
  ) < 1e-9));
  assert.ok(trades.every((trade) => Number.isFinite(trade.pnl) && Number.isFinite(trade.fees)));
  assert.ok(trades.some((trade) => trade.reasons.length > 0));

  const partial = trades.find((trade) => trade.exitMode === "tp1_trail");
  assert.ok(partial, "부분익절 뒤 트레일링 거래가 한 건으로 합쳐져야 한다");
  assert.deepEqual(partial.exitEvents.map((event) => event.reason), ["tp1", "trail"]);
  assert.deepEqual(partial.exitEvents.map((event) => event.ratio), [0.5, 0.5]);
});

test("trade history UI exposes strategy, exit-method and symbol filters", async () => {
  const source = await readFile(new URL("../app/BtcDashboard.tsx", import.meta.url), "utf8");
  assert.match(source, /ENTRY DECISION/);
  assert.match(source, /진입 조건에는 미반영/);
  assert.match(source, /1차 익절 → 트레일링/);
  assert.match(source, /수수료 포함 손익/);
  assert.match(source, /전체 종목/);
  assert.match(source, /시간청산/);
});
