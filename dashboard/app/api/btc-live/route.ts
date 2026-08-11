import { TIMEFRAME_SECONDS, normalizeGateCandles } from "@/lib/market.mjs";
import type { Timeframe } from "@/lib/market.mjs";

const GATE_CANDLES = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks";
const ALLOWED = new Set(Object.keys(TIMEFRAME_SECONDS));

async function fetchCandles(
  timeframe: Timeframe,
  { limit = 12, before }: { limit?: number; before?: number } = {},
) {
  const query = new URLSearchParams({
    contract: "BTC_USDT",
    interval: timeframe,
    limit: String(limit),
  });
  if (before != null) query.set("to", String(before - 1));
  const response = await fetch(`${GATE_CANDLES}?${query}`, {
    headers: { "User-Agent": "xrp-paper-bot-dashboard/0.2" },
    cache: "no-store",
    signal: AbortSignal.timeout(8_000),
  });
  if (!response.ok) throw new Error(`Gate HTTP ${response.status}`);
  return response.json() as Promise<unknown[]>;
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const requested = url.searchParams.get("tf") ?? "1h";
  if (!ALLOWED.has(requested)) {
    return Response.json({ error: "지원하지 않는 시간봉입니다." }, { status: 400 });
  }

  const beforeRaw = url.searchParams.get("before");
  const before = beforeRaw == null ? null : Number.parseInt(beforeRaw, 10);
  if (beforeRaw != null && (!Number.isFinite(before) || before <= 0)) {
    return Response.json({ error: "과거 조회 시각이 올바르지 않습니다." }, { status: 400 });
  }
  const requestedLimit = Number.parseInt(url.searchParams.get("limit") ?? "800", 10);
  const historyLimit = Math.max(100, Math.min(1_000, Number.isFinite(requestedLimit) ? requestedLimit : 800));
  const historyMode = before != null;
  const timeframes = historyMode
    ? [requested as Timeframe]
    : [...new Set([requested, "1m", "1h", "4h"])] as Timeframe[];
  const serverTime = Math.floor(Date.now() / 1000);
  try {
    const responses = await Promise.all(
      timeframes.map((timeframe) => fetchCandles(
        timeframe,
        historyMode ? { limit: historyLimit, before: before as number } : undefined,
      )),
    );
    const candles = Object.fromEntries(
      timeframes.map((timeframe, index) => [
        timeframe,
        normalizeGateCandles(responses[index], timeframe, serverTime),
      ]),
    );
    return Response.json(
      {
        serverTime,
        source: historyMode ? "Gate.io historical public candles" : "Gate.io live public candles",
        mode: historyMode ? "history" : "live",
        candles,
      },
      { headers: { "Cache-Control": "no-store, max-age=0" } },
    );
  } catch (error) {
    const detail = error instanceof Error ? error.message : "unknown error";
    return Response.json({ error: "실시간 시세를 읽지 못했습니다.", detail }, { status: 502 });
  }
}
