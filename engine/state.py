"""상태 저장과 복구.

봇이 재시작돼도 열려 있는 포지션과 계좌 상태를 잃지 않아야 한다.
가상거래에서도 마찬가지다 — 재시작 때마다 초기화되면 장기 로그가 끊긴다.

원자적으로 쓴다. 저장 도중 죽으면 파일이 반쯤 쓰인 상태로 남고,
다음 기동에서 JSON 파싱이 깨져 상태를 통째로 잃는다.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config.settings import RUNTIME, STATE_PATH


STATE_VERSION = 3


@dataclass
class OpenPosition:
    """열려 있는 포지션. 재시작 후 청산 판정을 이어가려면 이만큼이 필요하다."""

    symbol: str
    side: str
    strategy: str
    entry: float
    contracts: int
    leverage: int
    stop: float
    tp1: float
    partial_ratio: float
    trail_atr_mult: float
    max_hold_bars: int
    risk_amount: float

    # 선택적 전략 정책. 기본값이 있어 기존 version=3 상태도 그대로 읽힌다.
    time_exit_enabled: bool = True
    breakeven_offset_pct: float = 0.0
    exclusive_symbol: bool = False
    consecutive_loss_limit: int | None = None
    pause_until_next_day: bool = False
    tp1_is_maker: bool = False

    # 이 포지션이 어느 봉으로 진행되는가.
    # 전략마다 봉이 달라(5분~4시간) 하나로 묶으면 상위 봉 포지션이
    # 12~48배 빨리 시간 손절된다.
    entry_tf: str = "5m"
    # 마지막으로 반영한 봉의 타임스탬프. 같은 봉을 두 번 세지 않는다.
    last_bar_ts: int = 0

    bars_held: int = 0
    remaining_ratio: float = 1.0
    tp1_filled: bool = False
    extreme: float = 0.0
    opened_ts: int = 0

    def __post_init__(self):
        if not self.extreme:
            self.extreme = self.entry


@dataclass
class BotState:
    balance: float = RUNTIME.initial_balance
    day_start_balance: float = RUNTIME.initial_balance
    day_key: str = ""                   # "2026-08-05" 형태. 바뀌면 일일 기준 갱신
    # **전략별**로 센다. 계좌 전체로 묶으면 한 전략의 연패가 다른 전략을
    # 멈춰, 전략별 성과 측정이 오염된다 — 실측으로 kotegawa를 빼자
    # double_top 손실이 -221에서 -270으로 늘었다. 정지가 우연히
    # 브레이크 역할을 하고 있었던 것이다.
    consecutive_losses: dict[str, int] = field(default_factory=dict)
    paused_until: dict[str, int] = field(default_factory=dict)
    # UTC 날짜별 전략 체결 횟수. 날짜가 바뀌면 엔진이 비운다.
    daily_entries: dict[str, int] = field(default_factory=dict)
    positions: list[OpenPosition] = field(default_factory=list)
    last_bar_ts: dict[str, int] = field(default_factory=dict)   # 심볼_TF -> 마지막 처리 봉

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "balance": self.balance,
            "day_start_balance": self.day_start_balance,
            "day_key": self.day_key,
            "consecutive_losses": dict(self.consecutive_losses),
            "paused_until": dict(self.paused_until),
            "daily_entries": dict(self.daily_entries),
            "positions": [asdict(p) for p in self.positions],
            "last_bar_ts": self.last_bar_ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BotState":
        version = data.get("version")
        if version != STATE_VERSION:
            # 형식이 바뀌면 조용히 넘어가지 않는다. 잘못 해석한 포지션으로
            # 매매를 이어가는 것보다 멈추는 편이 낫다.
            raise ValueError(
                f"상태 파일 버전이 다르다: {version} (기대 {STATE_VERSION}). "
                "수동으로 확인하거나 파일을 지울 것."
            )
        positions = []
        for saved in data.get("positions", []):
            raw = dict(saved)
            # 시간청산 제거 전에 저장된 25배 포지션도 재시작 뒤 갑자기 종가로
            # 닫히지 않게 새 정책을 이어받는다. 다른 전략의 기존 기본값은 유지한다.
            if (
                "time_exit_enabled" not in raw
                and raw.get("strategy") == "btc_regime_25x"
            ):
                raw["time_exit_enabled"] = False
            positions.append(OpenPosition(**raw))

        return cls(
            balance=data["balance"],
            day_start_balance=data["day_start_balance"],
            day_key=data.get("day_key", ""),
            consecutive_losses=dict(data.get("consecutive_losses", {})),
            paused_until=dict(data.get("paused_until", {})),
            daily_entries=dict(data.get("daily_entries", {})),
            positions=positions,
            last_bar_ts=data.get("last_bar_ts", {}),
        )


def save(state: BotState, path: Path | str = STATE_PATH) -> Path:
    """임시 파일에 쓰고 교체한다. 도중에 죽어도 기존 파일이 남는다."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load(path: Path | str = STATE_PATH) -> BotState:
    """없으면 초기 상태. 깨졌으면 예외를 올린다.

    깨진 파일을 초기 상태로 대체하지 않는 이유는, 열려 있던 포지션을
    잊은 채 매매를 재개하는 게 가장 위험한 실패 모드이기 때문이다.
    """

    path = Path(path)
    if not path.exists():
        return BotState()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"상태 파일이 깨졌다: {path} ({exc})") from exc

    return BotState.from_dict(data)
