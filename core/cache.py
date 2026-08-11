"""과거 캔들 로컬 캐시.

시각 검증과 파라미터 탐색은 같은 구간을 여러 번 읽는다. 매번 거래소에서
다시 받으면 느리고, 무엇보다 실행 시점마다 데이터가 조금씩 달라져
"방금 본 차트"와 "지금 계산한 결과"가 어긋난다.

CSV로 저장한다. parquet이 빠르지만 의존성이 하나 더 늘고,
이 용도에서는 수백~수천 봉이라 차이가 체감되지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core import candles as cd


CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def path_for(contract: str, interval: str) -> Path:
    return CACHE_DIR / f"{contract}_{interval}.csv"


def load(contract: str, interval: str) -> pd.DataFrame:
    """캐시가 없으면 빈 DataFrame."""

    p = path_for(contract, interval)
    if not p.exists():
        return cd.to_dataframe([])

    df = pd.read_csv(p)
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    # 거래소에서 갓 받은 것과 컬럼 순서까지 같게 맞춘다.
    # 캐시로 그린 차트와 실시간 데이터로 그린 차트가 달라 보이면 안 된다.
    return df[list(cd.COLUMNS)].reset_index(drop=True)


def save(contract: str, interval: str, df: pd.DataFrame) -> Path:
    """기존 캐시와 병합해서 저장. 겹치는 구간은 새 데이터 우선."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    merged = cd.merge(load(contract, interval), df)

    p = path_for(contract, interval)
    merged.drop(columns=["dt"]).to_csv(p, index=False)
    return p


def coverage(contract: str, interval: str) -> tuple[int, int] | None:
    """캐시가 담고 있는 (시작, 끝) 타임스탬프. 비어 있으면 None."""

    df = load(contract, interval)
    if df.empty:
        return None
    return int(df["ts"].iloc[0]), int(df["ts"].iloc[-1])


# ---------------------------------------------------------------- 미결제약정

def oi_path(contract: str, interval: str) -> Path:
    return CACHE_DIR / f"{contract}_{interval}_oi.csv"


def save_oi(contract: str, interval: str, df: pd.DataFrame) -> Path:
    """OI 시계열 캐시. 캔들과 별도 파일로 둔다.

    합쳐 두지 않는 이유는 수집 상한이 달라서다 — 캔들은 봉 수 기준
    10,000봉, OI는 날짜 기준 180일이라 커버리지가 어긋난다.
    """

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = oi_path(contract, interval)

    if p.exists():
        old = pd.read_csv(p)
        df = pd.concat([old, df.drop(columns=["dt"], errors="ignore")], ignore_index=True)
        df = df.sort_values("ts").drop_duplicates(subset="ts", keep="last")

    df.drop(columns=["dt"], errors="ignore").to_csv(p, index=False)
    return p


def load_oi(contract: str, interval: str) -> pd.DataFrame:
    p = oi_path(contract, interval)
    if not p.exists():
        return pd.DataFrame(columns=["ts", "dt", "open_interest", "open_interest_usd", "lsr_taker"])

    df = pd.read_csv(p)
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df.sort_values("ts").reset_index(drop=True)
