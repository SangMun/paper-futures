"""캔들 정제.

거래소 응답을 DataFrame으로 바꾸는 순수 함수만 둔다. HTTP 호출은 exchange.py.
분리하는 이유는 미마감 캔들 제거와 정합성 검사가 실시간 루프와 과거 데이터 검증
양쪽에서 같은 규칙으로 돌아야 하기 때문이다. 네트워크 없이 테스트할 수 있어야
그 동일성을 보장할 수 있다.

인덱스 규약: 항상 0부터 시작하는 정수 인덱스(RangeIndex)를 유지한다.
설계 문서의 `as_of: int`는 위치 인덱스이므로, 타임스탬프를 인덱스로 올리면
전략 쪽 계약이 깨진다.
"""

from __future__ import annotations

import pandas as pd


# 지원 타임프레임과 초 단위 길이. config.symbols.Timeframe과 짝을 맞춘다.
INTERVAL_SECONDS: dict[str, int] = {
    # 1분봉은 전략의 진입봉이 아니다. 리플레이에서 실행 시계(틱)를
    # 재생하는 대용으로만 쓴다 — 5분봉 하나를 5조각으로 나눠 본다.
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

# 정제 후 항상 존재하는 컬럼.
# volume       : 계약 수 (거래소 원본 "v")
# quote_volume : USDT 환산 거래대금 (거래소 원본 "sum")
#                VPVR처럼 가격대별 가치를 볼 때는 이쪽을 쓴다.
COLUMNS = ("ts", "dt", "open", "high", "low", "close", "volume", "quote_volume")

_PRICE_COLUMNS = ("open", "high", "low", "close")


class CandleError(ValueError):
    """응답 형식이 예상과 다를 때."""


def interval_seconds(interval: str) -> int:
    try:
        return INTERVAL_SECONDS[interval]
    except KeyError:
        raise CandleError(f"지원하지 않는 타임프레임: {interval!r}") from None


def to_dataframe(raw: list[dict]) -> pd.DataFrame:
    """Gate 선물 캔들 응답을 정제된 DataFrame으로.

    시간 오름차순 정렬, 중복 타임스탬프 제거(뒤엣것 채택)까지 여기서 끝낸다.
    거래소가 순서를 보장하더라도 의존하지 않는다.
    """

    if not isinstance(raw, list):
        raise CandleError(f"리스트가 아닌 응답: {type(raw).__name__}")

    rows = []
    for item in raw:
        if not isinstance(item, dict):
            raise CandleError(f"캔들 항목 형식이 다름: {item!r}")
        try:
            rows.append(
                {
                    "ts": int(item["t"]),
                    "open": float(item["o"]),
                    "high": float(item["h"]),
                    "low": float(item["l"]),
                    "close": float(item["c"]),
                    "volume": float(item["v"]),
                    # 일부 응답에 sum이 빠지는 경우가 보고되어 방어한다.
                    "quote_volume": float(item.get("sum", 0.0) or 0.0),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CandleError(f"캔들 파싱 실패: {item!r}") from exc

    df = pd.DataFrame(rows, columns=[c for c in COLUMNS if c != "dt"])
    if df.empty:
        df = df.astype(
            {
                "ts": "int64",
                **{c: "float64" for c in _PRICE_COLUMNS},
                "volume": "float64",
                "quote_volume": "float64",
            }
        )

    df = df.sort_values("ts", kind="stable")
    df = df.drop_duplicates(subset="ts", keep="last")
    df.insert(1, "dt", pd.to_datetime(df["ts"], unit="s", utc=True))

    return df.reset_index(drop=True)


def drop_unclosed(
    df: pd.DataFrame,
    interval: str,
    now_ts: float | None = None,
) -> pd.DataFrame:
    """마감되지 않은 마지막 캔들을 제거한다.

    미마감 캔들은 고가·저가·종가가 계속 변하므로, 이걸 그대로 지표에 넣으면
    같은 시점을 다시 계산할 때 값이 달라진다. 과거 검증과 실시간의 결과가
    어긋나는 원인이 되므로 수집 단계에서 잘라낸다.

    now_ts를 넘기지 않으면 현재 시각을 쓴다. 테스트에서는 고정값을 넘긴다.
    """

    if df.empty:
        return df

    step = interval_seconds(interval)
    if now_ts is None:
        now_ts = pd.Timestamp.now(tz="UTC").timestamp()

    # 캔들 t는 시작 시각. t + step 이 지나야 마감된 것으로 본다.
    closed = df["ts"] + step <= now_ts
    return df.loc[closed].reset_index(drop=True)


def validate(df: pd.DataFrame, interval: str) -> list[str]:
    """정합성 문제를 문자열 목록으로 돌려준다. 예외를 던지지 않는다.

    수집 단계에서 매매를 멈출지는 호출부가 판단한다. 결측 캔들 하나 때문에
    루프 전체가 죽으면 안 되지만, 조용히 넘어가서도 안 된다.
    """

    problems: list[str] = []
    if df.empty:
        return ["캔들이 비어 있음"]

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"컬럼 누락: {missing}")
        return problems

    if df[list(_PRICE_COLUMNS)].isna().any().any():
        problems.append("가격에 NaN 포함")

    if (df[list(_PRICE_COLUMNS)] <= 0).any().any():
        problems.append("0 이하 가격 포함")

    ts = df["ts"]
    if not ts.is_monotonic_increasing:
        problems.append("타임스탬프 역순 구간 존재")
    if ts.duplicated().any():
        problems.append(f"중복 타임스탬프 {int(ts.duplicated().sum())}건")

    body_max = df[["open", "close"]].max(axis=1)
    body_min = df[["open", "close"]].min(axis=1)
    bad_high = int((df["high"] < body_max - 1e-12).sum())
    bad_low = int((df["low"] > body_min + 1e-12).sum())
    if bad_high:
        problems.append(f"고가가 시가/종가보다 낮은 캔들 {bad_high}건")
    if bad_low:
        problems.append(f"저가가 시가/종가보다 높은 캔들 {bad_low}건")
    if int((df["low"] > df["high"]).sum()):
        problems.append("저가 > 고가 캔들 존재")

    # 거래가 없는 구간은 거래소가 캔들을 건너뛴다. 채워 넣지 않고 보고만 한다.
    # 없는 캔들을 만들어내면 ATR·거래량 통계가 조용히 왜곡되기 때문이다.
    step = interval_seconds(interval)
    if len(df) > 1:
        gaps = ts.diff().iloc[1:]
        missing_bars = int(((gaps / step) - 1).round().clip(lower=0).sum())
        if missing_bars:
            first_gap = df.loc[gaps[gaps > step].index[0], "dt"]
            problems.append(
                f"결측 캔들 {missing_bars}봉 (첫 구간 {first_gap:%Y-%m-%d %H:%M} UTC)"
            )

    return problems


def merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """기존 캔들에 신규 응답을 덮어쓴다. 겹치는 타임스탬프는 신규 우선.

    실시간 루프에서 매번 500봉을 새로 받는 대신 뒤쪽만 갱신할 때,
    그리고 과거 데이터를 페이지 단위로 이어붙일 때 쓴다.
    """

    if old.empty:
        return new.reset_index(drop=True)
    if new.empty:
        return old.reset_index(drop=True)

    merged = pd.concat([old, new], ignore_index=True)
    merged = merged.sort_values("ts", kind="stable")
    merged = merged.drop_duplicates(subset="ts", keep="last")
    return merged.reset_index(drop=True)


def resample(df: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    """상위 타임프레임으로 합성한다.

    Gate 아카이브에 15분·30분봉이 없어서 5분봉에서 만든다.
    REST로 받으면 10,000봉 상한에 걸려 98일치뿐인데, 합성하면
    5분봉이 커버하는 949일 전체를 쓸 수 있다.

    **경계에 걸친 미완성 구간은 버린다.** 15분봉 하나는 5분봉 3개로
    이뤄지는데, 2개만 있는 마지막 구간을 남기면 고저가 실제보다 좁아진다.
    """

    src, tgt = interval_seconds(source), interval_seconds(target)
    if tgt <= src or tgt % src:
        raise CandleError(f"{source} -> {target} 합성 불가 (배수가 아님)")

    per = tgt // src
    bucket = df["ts"] // tgt * tgt

    grouped = df.groupby(bucket)
    out = grouped.agg(
        ts=("ts", "min"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
    )
    complete = grouped.size() == per
    out = out.loc[complete].reset_index(drop=True)

    out["ts"] = (out["ts"] // tgt * tgt).astype("int64")
    out.insert(1, "dt", pd.to_datetime(out["ts"], unit="s", utc=True))
    return out[list(COLUMNS)]
