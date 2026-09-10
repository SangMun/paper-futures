"""Gate.io 선물 캔들 수집.

여기만 네트워크를 만진다. 정제 규칙은 candles.py에 있다.

캔들 조회(`/futures/usdt/candlesticks`)는 공개 엔드포인트라 API 키가 필요 없다.
주문 관련 인증 호출은 실전 전환 시점에 별도 모듈로 추가한다. 페이퍼 단계에서는
이 파일이 키를 다루지 않는다는 점이 중요하다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
import pandas as pd

from config.settings import BASE_URL, RUNTIME
from config.symbols import SymbolSpec, Timeframe
from core import candles as cd


# 거래소가 한 번에 돌려주는 최대 캔들 수. 초과 요청은 잘려서 온다.
MAX_LIMIT = 2000

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class ExchangeError(RuntimeError):
    """재시도 후에도 응답을 얻지 못했을 때."""


@dataclass(frozen=True)
class FetchResult:
    """캔들과 함께 정합성 점검 결과를 같이 들고 다닌다.

    호출부가 problems를 확인하지 않고 df만 꺼내 쓰는 일을 줄이기 위해
    한 덩어리로 묶었다.
    """

    contract: str
    interval: Timeframe
    df: pd.DataFrame
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems

    def __len__(self) -> int:
        return len(self.df)


class GateClient:
    """공개 캔들 API 클라이언트.

    with 문으로 쓰면 커넥션을 재사용한다. 실시간 루프는 10초마다 도는데
    매번 새 연결을 맺으면 지연이 붙고 레이트리밋에도 불리하다.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._max_retries = max_retries
        self._backoff = backoff
        # transport는 테스트에서 가짜 응답을 끼워 넣기 위한 자리다.
        # 재시도·페이지네이션은 네트워크 없이 검증할 수 있어야 한다.
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={
                "User-Agent": "paper-futures-v2/0.1",
                "Accept": "application/json",
            },
        )

    # ---------------------------------------------------------------- 내부

    def _get(self, endpoint: str, params: dict) -> list:
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            if attempt:
                time.sleep(self._backoff * (2 ** (attempt - 1)))
            try:
                resp = self._client.get(endpoint, params=params)
            except httpx.HTTPError as exc:
                last_error = exc
                continue

            if resp.status_code in _RETRY_STATUS:
                last_error = ExchangeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                continue
            if resp.status_code != 200:
                # 400대 오류는 재시도해도 같은 결과다. 파라미터 문제로 보고 즉시 중단.
                raise ExchangeError(
                    f"HTTP {resp.status_code} {endpoint} {params}: {resp.text[:200]}"
                )
            return resp.json()

        raise ExchangeError(
            f"{endpoint} 요청 실패 ({self._max_retries}회 시도): {last_error}"
        )

    # ---------------------------------------------------------------- 조회

    def fetch(
        self,
        contract: str,
        interval: Timeframe,
        limit: int = RUNTIME.candle_limit,
        drop_unclosed: bool = True,
        now_ts: float | None = None,
    ) -> FetchResult:
        """최근 캔들 limit개. 기본적으로 미마감 캔들은 잘라낸다.

        미마감 캔들이 잘리므로 반환 개수는 limit보다 하나 적을 수 있다.
        필요한 봉 수가 정해져 있으면 여유를 두고 요청할 것.
        """

        if limit > MAX_LIMIT:
            raise ValueError(f"limit 상한은 {MAX_LIMIT}, 요청값 {limit}")

        raw = self._get(
            "/futures/usdt/candlesticks",
            {"contract": contract, "interval": interval, "limit": limit},
        )
        df = cd.to_dataframe(raw)
        if drop_unclosed:
            df = cd.drop_unclosed(df, interval, now_ts=now_ts)

        return FetchResult(contract, interval, df, cd.validate(df, interval))

    def fetch_range(
        self,
        contract: str,
        interval: Timeframe,
        start_ts: int,
        end_ts: int,
        drop_unclosed: bool = True,
        now_ts: float | None = None,
    ) -> FetchResult:
        """[start_ts, end_ts] 구간 전체. 2000봉 제한을 넘으면 나눠 받는다.

        시각 검증과 파라미터 탐색용으로 긴 구간이 필요하다.
        """

        step = cd.interval_seconds(interval)
        if end_ts <= start_ts:
            raise ValueError(f"구간이 뒤집힘: {start_ts} -> {end_ts}")

        acc = cd.to_dataframe([])
        cursor = start_ts

        while cursor <= end_ts:
            chunk_end = min(cursor + (MAX_LIMIT - 1) * step, end_ts)
            raw = self._get(
                "/futures/usdt/candlesticks",
                {
                    "contract": contract,
                    "interval": interval,
                    "from": int(cursor),
                    "to": int(chunk_end),
                },
            )
            page = cd.to_dataframe(raw)
            if page.empty:
                # 거래가 없어 캔들이 아예 없는 구간. 다음 페이지로 건너뛴다.
                if chunk_end >= end_ts:
                    break
                cursor = chunk_end + step
                continue

            acc = cd.merge(acc, page)
            last_ts = int(page["ts"].iloc[-1])
            if last_ts < cursor:
                break               # 진행이 없으면 무한루프 방지
            cursor = last_ts + step

        if drop_unclosed:
            acc = cd.drop_unclosed(acc, interval, now_ts=now_ts)

        return FetchResult(contract, interval, acc, cd.validate(acc, interval))

    def fetch_frames(
        self,
        spec: SymbolSpec,
        limit: int = RUNTIME.candle_limit,
        now_ts: float | None = None,
    ) -> dict[Timeframe, FetchResult]:
        """종목이 필요로 하는 타임프레임을 한 번에 채운다.

        어떤 타임프레임이 필요한지는 SymbolSpec이 전략 목록에서 이미 계산해 둔다.
        전략을 추가해도 수집 코드를 고칠 일이 없도록 여기서 그 값을 그대로 쓴다.
        """

        return {
            tf: self.fetch(spec.contract, tf, limit=limit, now_ts=now_ts)
            for tf in sorted(spec.required_timeframes, key=cd.interval_seconds)
        }

    # ---------------------------------------------------------------- 수명

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GateClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# ---------------------------------------------------------------- 미결제약정

# contract_stats는 한 번에 돌려주는 개수가 캔들과 다르다.
OI_MAX_LIMIT = 1000

# 미결제약정은 180일 이전을 요청하면 400
# ("from time exceeds 180-day limit"). 캔들과 상한이 다르다 —
# 캔들은 봉 수(10,000) 기준이고 이쪽은 날짜 기준이다.
OI_MAX_DAYS = 180

OI_COLUMNS = ("ts", "dt", "open_interest", "open_interest_usd", "lsr_taker")


def _oi_frame(raw: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "ts": int(x["time"]),
            "open_interest": float(x.get("open_interest", 0.0) or 0.0),
            "open_interest_usd": float(x.get("open_interest_usd", 0.0) or 0.0),
            "lsr_taker": float(x.get("lsr_taker", 0.0) or 0.0),
        }
        for x in raw
        if isinstance(x, dict) and "time" in x
    ]
    df = pd.DataFrame(rows, columns=[c for c in OI_COLUMNS if c != "dt"])
    if df.empty:
        return df.assign(dt=pd.Series(dtype="datetime64[ns, UTC]"))[list(OI_COLUMNS)]

    df = df.sort_values("ts", kind="stable").drop_duplicates(subset="ts", keep="last")
    df.insert(1, "dt", pd.to_datetime(df["ts"], unit="s", utc=True))
    return df.reset_index(drop=True)


def fetch_open_interest(
    client: "GateClient",
    contract: str,
    interval: str,
    start_ts: int,
    end_ts: int,
) -> pd.DataFrame:
    """미결제약정 시계열. 캔들과 같은 타임프레임으로 받는다.

    OI는 캔들 엔드포인트에 없어 별도로 받아야 한다. 돌파 판정에 쓴다 —
    돌파 + 거래량 증가 + OI 증가면 신규 자금 유입, OI 감소면 청산성
    스퀴즈다. 후자는 강해 보여도 소진되기 쉽다.
    """

    step = cd.interval_seconds(interval)
    # 180일 상한을 넘기면 거래소가 거부한다. 조용히 잘라내되
    # 호출부가 얼마나 못 받았는지는 반환 구간으로 알 수 있다.
    floor = int(end_ts - OI_MAX_DAYS * 86400) + step
    start_ts = max(start_ts, floor)

    frames: list[pd.DataFrame] = []
    cursor = start_ts

    while cursor < end_ts:
        raw = client._get(
            "/futures/usdt/contract_stats",
            {
                "contract": contract,
                "interval": interval,
                "limit": OI_MAX_LIMIT,
                "from": int(cursor),
            },
        )
        page = _oi_frame(raw if isinstance(raw, list) else [])
        if page.empty:
            break

        frames.append(page)
        last = int(page["ts"].iloc[-1])
        if last <= cursor:
            break
        cursor = last + step

    if not frames:
        return _oi_frame([])

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("ts").drop_duplicates(subset="ts", keep="last")
    return out.loc[(out["ts"] >= start_ts) & (out["ts"] <= end_ts)].reset_index(drop=True)
