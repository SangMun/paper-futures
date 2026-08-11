"""체결·신호 기록.

두 갈래로 나눈다.
  CSV   체결 기록. 컬럼 고정. 엑셀로 열어 보는 용도라 항목이 늘어나면 안 된다.
  JSONL 신호 스냅샷. 지표 항목이 계속 늘어나므로 스키마가 유연해야 한다.

진입하지 않은 신호도 남긴다(README 원칙 4). 리스크 상한에 걸려 스킵한 신호가
실은 더 좋았는지를 나중에 판단하려면, 그때의 판단 근거가 남아 있어야 한다.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import SIGNAL_LOG_PATH, TRADE_LOG_PATH


# 체결 CSV 컬럼. 순서와 개수를 바꾸지 않는다.
# 늘려야 할 항목이 생기면 JSONL 쪽에 넣는다.
TRADE_COLUMNS = (
    "ts",
    "dt",
    "strategy",
    "symbol",
    "side",
    "action",           # entry / exit
    "reason",           # 청산 사유 (진입이면 빈 값)
    "price",
    "contracts",
    "notional",
    "leverage",
    "stop",
    "tp1",
    "ratio",            # 이번에 체결한 비율 (부분 익절 대응)
    "fee",
    "pnl",
    "balance",
    # 체결 방식 ("갭 손절" / "손절 시장가" / "40봉 경과" 등).
    # 이게 없으면 로그만으로 체결 품질을 진단할 수 없다 — 손절이
    # 계획보다 나쁘게 체결됐는지 알 방법이 사라진다.
    "note",
)

# 신호 action 값
ACTION_ENTRY = "entry"
ACTION_SKIP = "skip"
ACTION_PHANTOM = "phantom"      # 진입하지 않았지만 가상 추적하는 신호


def _now() -> tuple[int, str]:
    now = datetime.now(tz=timezone.utc)
    return int(now.timestamp()), now.isoformat()


@dataclass
class TradeLog:
    """체결 CSV. 헤더는 파일 생성 시 한 번만 쓴다."""

    path: Path = Path(TRADE_LOG_PATH)

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(TRADE_COLUMNS)

    def write(self, market_ts: int | None = None, **fields: Any) -> None:
        """알려진 컬럼만 기록한다. 모르는 키는 조용히 버리지 않고 예외.

        market_ts는 체결이 일어난 **봉의 시각**이다. 넘기지 않으면 현재 시각을
        쓴다. 실시간에서는 둘이 거의 같지만 리플레이에서는 완전히 다르다 —
        실행 시각을 쓰면 180일치 체결이 전부 11초 안에 찍혀 보유 시간을
        복원할 수 없다. 체결 기록은 감사 자료이므로 시장 시각이어야 한다.
        """

        unknown = set(fields) - set(TRADE_COLUMNS)
        if unknown:
            raise ValueError(f"CSV에 없는 컬럼: {sorted(unknown)} — JSONL로 보낼 것")

        if market_ts is None:
            ts, dt = _now()
        else:
            ts = int(market_ts)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        row = {"ts": ts, "dt": dt, **fields}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(c, "") for c in TRADE_COLUMNS])


@dataclass
class SignalLog:
    """신호 JSONL. 한 줄에 하나의 판단 스냅샷."""

    path: Path = Path(SIGNAL_LOG_PATH)

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        action: str,
        strategy: str,
        symbol: str,
        side: str,
        skip_reason: str | None = None,
        market_ts: int | None = None,
        **extra: Any,
    ) -> None:
        if market_ts is None:
            ts, dt = _now()
        else:
            ts = int(market_ts)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        record = {
            "ts": ts,
            "dt": dt,
            "action": action,
            "strategy": strategy,
            "symbol": symbol,
            "side": side,
            "skip_reason": skip_reason,
            **extra,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_safe) + "\n")


def _json_safe(obj: Any) -> Any:
    """numpy 스칼라와 dataclass를 JSON에 넣을 수 있게 바꾼다."""

    if hasattr(obj, "item"):        # numpy 스칼라
        return obj.item()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def read_trades(path: Path | str = TRADE_LOG_PATH) -> list[dict]:
    """집계용 읽기. 파일이 없으면 빈 목록."""

    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_signals(path: Path | str = SIGNAL_LOG_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
