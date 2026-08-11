import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the BTC strategy dashboard shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<html lang="ko">/i);
  assert.match(html, /<title>BTC 회귀채널 복귀 · Paper Desk<\/title>/i);
  assert.match(html, /BTC 전략 데스크를 준비하는 중/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/i);
});

test("ships every requested timeframe and strategy-specific controls", async () => {
  const [source, rawData, packageJson] = await Promise.all([
    readFile(new URL("../app/BtcDashboard.tsx", import.meta.url), "utf8"),
    readFile(new URL("../public/btc-dashboard.json", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);
  const snapshot = JSON.parse(rawData);

  assert.deepEqual(Object.keys(snapshot.candles), ["1m", "5m", "15m", "1h", "4h", "1d"]);
  for (const candles of Object.values(snapshot.candles)) {
    assert.ok(candles.length >= 100);
  }
  assert.equal(snapshot.strategy.channelLength, 100);
  assert.equal(snapshot.strategy.channelWidthMult, 2);
  assert.equal(snapshot.strategy.entryOffsetPct, 0.02);
  assert.equal(snapshot.strategy.executionTimeframe, "1h");
  assert.equal(snapshot.strategy.contextTimeframe, "4h");

  assert.match(source, /회귀채널 100 · 2σ/);
  assert.match(source, /LIVE PREVIEW/);
  assert.match(source, /99개 확정봉 \+ 현재 1시간봉/);
  assert.match(source, /AWS 주문 엔진은 아직 확정봉 방식/);
  assert.match(source, /EMA 20 \/ 60/);
  assert.match(source, /BTC 회귀채널 포지션/);
  assert.match(source, /가격 수익률/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
