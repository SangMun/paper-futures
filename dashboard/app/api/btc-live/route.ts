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
    headers: { "User-Agent": "paper-futures-dashboard/0.2" },
    cache: "no-store",
    signal: AbortSignal.timeout(8_000),
  });
  if (!response.ok) throw new Error(`Gate HTTP ${response.status}`);
  return response.json() as Promise<unknown[]>;
}

async function fetchRange(timeframe: Timeframe, from: number, to: number) {
  const query = new URLSearchParams({
    contract: "BTC_USDT",
    interval: timeframe,
    from: String(from),
    to: String(to),
  });
  const response = await fetch(`${GATE_CANDLES}?${query}`, {
    headers: { "User-Agent": "paper-futures-dashboard/0.2" },
    cache: "no-store",
    signal: AbortSignal.timeout(8_000),
  });
  if (!response.ok) throw new Error(`Gate HTTP ${response.status}`);
  return response.json() as Promise<unknown[]>;
}

async function fetchAfter(timeframe: Timeframe, after: number, serverTime: number) {
  const step = TIMEFRAME_SECONDS[timeframe];
  const maximumBars = 5_000;
  const firstMissing = after + step;
  const requestedBars = Math.max(0, Math.floor((serverTime - firstMissing) / step) + 1);
  const truncated = requestedBars > maximumBars;
  const rangeStart = truncated
    ? Math.max(firstMissing, serverTime - (maximumBars - 1) * step)
    : firstMissing;
  const chunkBars = 900;
  const combined: unknown[] = [];

  for (let from = rangeStart; from <= serverTime; from += chunkBars * step) {
    const to = Math.min(serverTime, from + (chunkBars - 1) * step);
    combined.push(...await fetchRange(timeframe, from, to));
  }
  return { raw: combined, truncated };
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
  const afterRaw = url.searchParams.get("after");
  const after = afterRaw == null ? null : Number.parseInt(afterRaw, 10);
  if (afterRaw != null && (!Number.isFinite(after) || after <= 0)) {
    return Response.json({ error: "이어받을 시각이 올바르지 않습니다." }, { status: 400 });
  }
  if (before != null && after != null) {
    return Response.json({ error: "과거 조회와 이어받기는 동시에 사용할 수 없습니다." }, { status: 400 });
  }
  const requestedLimit = Number.parseInt(url.searchParams.get("limit") ?? "800", 10);
  const historyLimit = Math.max(100, Math.min(1_000, Number.isFinite(requestedLimit) ? requestedLimit : 800));
  const historyMode = before != null;
  const catchupMode = after != null;
  const timeframes = historyMode || catchupMode
    ? [requested as Timeframe]
    : [...new Set([
      requested,
      ...(requested === "5m" ? ["15m"] : []),
      ...(requested === "15m" ? ["5m"] : []),
      "1m",
      "1h",
      "4h",
    ])] as Timeframe[];
  const serverTime = Math.floor(Date.now() / 1000);
  try {
    const catchups = catchupMode
      ? await Promise.all(timeframes.map((timeframe) => fetchAfter(timeframe, after as number, serverTime)))
      : null;
    const responses = catchups?.map((item) => item.raw) ?? await Promise.all(
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
        source: historyMode
          ? "Gate.io historical public candles"
          : catchupMode ? "Gate.io catch-up public candles" : "Gate.io live public candles",
        mode: historyMode ? "history" : catchupMode ? "catchup" : "live",
        truncated: catchups?.some((item) => item.truncated) ?? false,
        candles,
      },
      { headers: { "Cache-Control": "no-store, max-age=0" } },
    );
  } catch (error) {
    const detail = error instanceof Error ? error.message : "unknown error";
    return Response.json(
      {
        error: historyMode
          ? "과거 캔들을 읽지 못했습니다."
          : catchupMode ? "누락 캔들을 이어받지 못했습니다." : "실시간 시세를 읽지 못했습니다.",
        detail,
      },
      { status: 502 },
    );
  }
}
