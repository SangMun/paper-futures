import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${pathname}`, {
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

test("server-renders the BTC 25x strategy dashboard shell", async () => {
  const response = await render("/btc-25x");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /<title>BTC 25× 국면 채널 · Paper Desk<\/title>/i);
  assert.match(html, /BTC 25× 전략 데스크를 준비하는 중/);
});

test("server-renders the XRP pattern dashboard shell", async () => {
  const response = await render("/xrp-patterns");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /<title>XRP 패턴 스캐너 · Paper Desk<\/title>/i);
  assert.match(html, /XRP 패턴 스캐너를 준비하는 중/);
});

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
  const [source, routeSource, rawData, packageJson] = await Promise.all([
    readFile(new URL("../app/BtcDashboard.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/btc-live/route.ts", import.meta.url), "utf8"),
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
  // 실시간 봇 설정(engine/loop.py)의 1% 이격을 스냅샷이 그대로 싣는다.
  assert.equal(snapshot.strategy.entryOffsetPct, 0.01);
  assert.equal(snapshot.strategy.executionTimeframe, "1h");
  assert.equal(snapshot.strategy.contextTimeframe, "4h");

  assert.match(source, /회귀채널 100 · 2σ/);
  assert.match(source, /LIVE PREVIEW/);
  assert.match(source, /연속 100봉 기준 · 결측 자동 보정/);
  assert.match(source, /AWS 주문 엔진은 아직 확정봉 방식/);
  assert.match(source, /EMA 20 \/ 60/);
  assert.match(source, /RSI \/ MACD/);
  assert.match(source, /진입 조건에는 미반영/);
  assert.match(source, /BTC 회귀채널 포지션/);
  assert.match(source, /가격 수익률/);
  // React onWheel은 passive라 preventDefault가 안 돼 네이티브 리스너로 바꿨다.
  assert.match(source, /addEventListener\("wheel", onWheel, \{ passive: false \}\)/);
  assert.match(source, /onPointerDown=\{handlePointerDown\}/);
  assert.match(source, /화면 확대/);
  assert.match(source, /과거 \+800/);
  assert.match(routeSource, /query\.set\("to"/);
  assert.match(routeSource, /historyMode/);
  assert.match(routeSource, /fetchAfter/);
  assert.match(routeSource, /catchupMode/);
  assert.match(routeSource, /maximumBars = 5_000/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
