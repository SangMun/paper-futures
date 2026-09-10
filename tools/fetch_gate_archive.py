"""Gate 공식 월별 아카이브에서 과거 선물 캔들을 보강한다.

REST 캔들 API는 최근 구간만 제공하므로 장기간 백테스트 데이터는 Gate의
Historical Quotation Data를 사용한다. 월별 아카이브는 완결된 달만 내려받고,
현재 달과 중복 구간은 REST로 모은 기존 캐시를 우선한다.

    python -m tools.fetch_gate_archive --contract XRP_USDT --interval 5m --days 180

아카이브 선물 캔들은 quote_volume을 제공하지 않는다. 계약 수(volume)에 계약
승수와 HLC 평균가격을 곱해 근사한다. 기존 REST 구간에는 거래소가 반환한 정확한
quote_volume이 그대로 유지된다.
"""

from __future__ import annotations

import argparse
import gzip
import io
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from config.settings import CONTRACT_SPECS
from core import cache
from core import candles as cd


ARCHIVE_ROOT = "https://download.gatedata.org/futures_usdt"
ARCHIVE_COLUMNS = ("ts", "volume", "close", "high", "low", "open")


class ArchiveError(RuntimeError):
    """아카이브가 없거나 형식·정합성이 잘못됐을 때."""


def archive_url(contract: str, interval: str, month: datetime) -> str:
    ym = month.strftime("%Y%m")
    return f"{ARCHIVE_ROOT}/candlesticks_{interval}/{ym}/{contract}-{ym}.csv.gz"


def month_starts(start_ts: int, end_ts: int) -> list[datetime]:
    """[start_ts, end_ts)와 겹치는 달의 UTC 1일 목록."""

    start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    end_month = datetime(end.year, end.month, 1, tzinfo=timezone.utc)

    out: list[datetime] = []
    while cursor < end_month:
        out.append(cursor)
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1, tzinfo=timezone.utc)
    return out


def parse_archive(payload: bytes, contract: str) -> pd.DataFrame:
    """gzip CSV 하나를 표준 캔들 DataFrame으로 변환한다."""

    if contract not in CONTRACT_SPECS:
        raise ArchiveError(f"계약 승수 정보 없음: {contract}")

    try:
        raw = gzip.decompress(payload)
        df = pd.read_csv(
            io.BytesIO(raw),
            header=None,
            names=list(ARCHIVE_COLUMNS),
            dtype="float64",
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ArchiveError(f"아카이브 파싱 실패: {exc}") from exc

    if df.empty:
        raise ArchiveError("아카이브가 비어 있음")
    if df[list(ARCHIVE_COLUMNS)].isna().any().any():
        raise ArchiveError("아카이브에 NaN 포함")

    df["ts"] = df["ts"].astype("int64")
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    multiplier = float(CONTRACT_SPECS[contract]["multiplier"])
    df["quote_volume"] = df["volume"] * multiplier * typical_price
    df.insert(1, "dt", pd.to_datetime(df["ts"], unit="s", utc=True))
    return df[list(cd.COLUMNS)].sort_values("ts").reset_index(drop=True)


def fetch_month(contract: str, interval: str, month: datetime) -> pd.DataFrame:
    url = archive_url(contract, interval, month)
    request = urllib.request.Request(url, headers={"User-Agent": "paper-futures-v2/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise ArchiveError(f"HTTP {exc.code}: {url}") from exc
    except urllib.error.URLError as exc:
        raise ArchiveError(f"다운로드 실패: {url}: {exc}") from exc
    return parse_archive(payload, contract)


def critical_problems(problems: list[str]) -> list[str]:
    """과거 캐시에 저장하면 안 되는 정합성 문제만 고른다.

    Gate는 거래가 없거나 시장 데이터가 중단된 구간의 캔들을 생략한다. 그런
    타임스탬프 간격은 보존하고 백테스트에서 신호를 건너뛰어야 하므로 경고로
    남긴다. 가격 NaN·역순·중복·OHLC 모순은 저장을 중단한다.
    """

    return [p for p in problems if not p.startswith("결측 캔들 ")]


def download_history(
    contract: str,
    interval: str,
    days: int,
    now: int | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """완결 월 아카이브와 기존 REST 캐시를 병합하고 저장한다."""

    if days < 1:
        raise ValueError(f"days는 1 이상이어야 한다: {days}")
    cd.interval_seconds(interval)

    now = int(time.time()) if now is None else int(now)
    target_start = now - days * 86400
    months = month_starts(target_start, now)
    if not months:
        raise ArchiveError("완결된 월이 없어 내려받을 아카이브가 없음")

    frames: list[pd.DataFrame] = []
    for month in months:
        print(f"  {month:%Y-%m} 다운로드...", end="", flush=True)
        frame = fetch_month(contract, interval, month)
        print(f" {len(frame):,}봉", flush=True)
        frames.append(frame)

    archive = pd.concat(frames, ignore_index=True)
    archive = archive.loc[
        (archive["ts"] >= target_start) & (archive["ts"] <= now)
    ].reset_index(drop=True)

    # 두 번째 인자가 중복 타임스탬프에서 이긴다. REST의 정확한 quote_volume과
    # 최신 정정값을 보존하기 위해 기존 캐시를 뒤에 둔다.
    existing = cache.load(contract, interval)
    merged = cd.merge(archive, existing)
    merged = merged.loc[
        (merged["ts"] >= target_start) & (merged["ts"] <= now)
    ].reset_index(drop=True)

    problems = cd.validate(merged, interval)
    critical = critical_problems(problems)
    if critical:
        raise ArchiveError("병합 데이터 정합성 실패: " + "; ".join(critical))
    for problem in problems:
        print(f"  경고: {problem} — 채우지 않고 원본 간격을 보존", flush=True)

    path = cache.path_for(contract, interval)
    backup: str | None = None
    if path.exists():
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = path.with_name(f"{path.name}.{stamp}.bak")
        shutil.copy2(path, backup_path)
        backup = str(backup_path)

    # merged 안에서 이미 REST 우선순위를 해결했다. cache.save의 표준 컬럼
    # 저장 규약을 재사용한다.
    cache.save(contract, interval, merged)
    return cache.load(contract, interval), backup


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="XRP_USDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument(
        "--all",
        action="store_true",
        help="아카이브에 있는 모든 타임프레임(1m/5m/1h/4h/1d)을 받는다. "
             "15m/30m/8h는 아카이브에 없다.",
    )
    args = ap.parse_args()

    targets = ["5m", "1h", "4h", "1d"] if args.all else [args.interval]
    failed = []
    for interval in targets:
        print(f"[{args.contract} {interval}] 최근 {args.days}일 보강")
        try:
            df, backup = download_history(args.contract, interval, args.days)
        except (ArchiveError, ValueError) as exc:
            print(f"  실패: {exc}")
            failed.append(interval)
            continue
        span = (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 86400
        print(f"  완료: {len(df):,}봉, {span:.0f}일")
    if args.all:
        return 1 if failed else 0

    try:
        df, backup = download_history(args.contract, args.interval, args.days)
    except (ArchiveError, ValueError) as exc:
        print(f"실패: {exc}")
        return 1

    span_days = (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 86400
    print(
        f"완료: {len(df):,}봉, {df['dt'].iloc[0]} ~ {df['dt'].iloc[-1]} "
        f"({span_days:.1f}일)"
    )
    if backup:
        print(f"기존 캐시 백업: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
