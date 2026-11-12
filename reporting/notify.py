"""디스코드 알림.

README 원칙 6: 알림 실패가 매매를 막지 않는다.
모든 전송은 예외를 삼키고 False를 돌려준다. 웹훅이 죽어도 루프는 계속 돈다.

전송 실패를 조용히 넘기지는 않는다. 실패 횟수를 세어 두고, 연속으로 실패하면
표준출력에 한 번만 알린다. 매 봉마다 같은 오류를 찍으면 로그가 묻힌다.
"""

from __future__ import annotations

import json
from numbers import Real
import json
import subprocess
import time
from dataclasses import dataclass, field

import httpx
from pathlib import Path

from config import env
from config.settings import (
    DISCORD_BALANCE_WEBHOOK_ENV,
    DISCORD_BTC_TRADE_WEBHOOK_ENV,
    DISCORD_SYSTEM_WEBHOOK_ENV,
    DISCORD_TRADE_WEBHOOK_ENV,
    DISCORD_WEBHOOK_ENV,
    DISCORD_XRP_TRADE_WEBHOOK_ENV,
)


def code_revision() -> str:
    """실행 중인 코드의 커밋. 성과가 어느 코드에서 나왔는지 알림 로그만으로 추적한다.

    추적 대상 파일에 커밋하지 않은 수정이 있으면 ``+dirty``를 붙인다.
    git이 없거나 저장소가 아니면 매매를 막지 않고 ``unknown``을 돌려준다.
    """

    root = Path(__file__).resolve().parent.parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()

    try:
        sha = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{sha}+dirty" if dirty else sha


# 디스코드 임베드 색상
COLOR_ENTRY = 0x3498DB      # 파랑
COLOR_WIN = 0x2ECC71        # 초록
COLOR_LOSS = 0xE74C3C       # 빨강
COLOR_WARN = 0xF39C12       # 주황


@dataclass
class Notifier:
    """웹훅 전송기.

    URL이 없으면 비활성 상태로 동작한다. 알림 설정 없이도 봇이 돌아야 하므로
    URL 부재는 오류가 아니다.
    """

    url: str | None = None
    system_url: str | None = None
    trade_url: str | None = None
    btc_trade_url: str | None = None
    xrp_trade_url: str | None = None
    balance_url: str | None = None
    # 계좌를 전략별로 나눠 돌릴 때 상태·잔고 채널을 공유하므로, 어느 계좌의
    # 메시지인지 제목 앞에 붙인다. 단일 계좌면 빈 값이라 제목이 그대로다.
    account: str = field(default_factory=lambda: env.get("PAPER_ACCOUNT") or "")
    timeout: float = 5.0
    quiet_after: int = 3            # 연속 실패 이 횟수부터는 출력하지 않는다

    # 같은 내용을 이 시간 안에 다시 보내지 않는다.
    #
    # 루프 오류는 회차마다 발생한다. 폴링 10초면 지속되는 오류 하나가
    # 시간당 360건, 하루 8,640건을 만든다. 실제로 채널이 같은 메시지로
    # 뒤덮였다. 첫 알림은 그대로 보내고, 반복은 묶어서 한 번만 알린다.
    dedupe_seconds: float = 900.0

    # 채널당 분당 전송 상한. 중복이 아니어도 폭주를 막는 마지막 방어선이다.
    max_per_minute: int = 10

    _failures: dict[str, int] = field(default_factory=dict, init=False)
    # (채널, 내용) -> 마지막 전송 시각 / 그 뒤 눌린 횟수
    _last_sent: dict[tuple, float] = field(default_factory=dict, init=False)
    _suppressed: dict[tuple, int] = field(default_factory=dict, init=False)
    # 채널 -> 최근 전송 시각 목록
    _recent: dict[str, list] = field(default_factory=dict, init=False)
    _client: httpx.Client | None = None

    def __post_init__(self):
        # url을 명시하면(빈 문자열 포함) 테스트·리플레이의
        # 강제 오버라이드로 취급한다. Notifier(url="")가 실제 .env의
        # 분리 웹훅을 읽어 알림을 보내면 안 된다.
        explicit_legacy = self.url is not None
        legacy = self.url if explicit_legacy else env.get(DISCORD_WEBHOOK_ENV)

        if self.system_url is None:
            self.system_url = legacy if explicit_legacy else (
                env.get(DISCORD_SYSTEM_WEBHOOK_ENV) or legacy
            )
        if self.trade_url is None:
            self.trade_url = legacy if explicit_legacy else (
                env.get(DISCORD_TRADE_WEBHOOK_ENV) or legacy
            )
        if self.btc_trade_url is None:
            self.btc_trade_url = self.trade_url if explicit_legacy else (
                env.get(DISCORD_BTC_TRADE_WEBHOOK_ENV) or self.trade_url
            )
        if self.xrp_trade_url is None:
            self.xrp_trade_url = self.trade_url if explicit_legacy else (
                env.get(DISCORD_XRP_TRADE_WEBHOOK_ENV) or self.trade_url
            )
        if self.balance_url is None:
            self.balance_url = legacy if explicit_legacy else (
                env.get(DISCORD_BALANCE_WEBHOOK_ENV) or legacy
            )
        self.url = legacy

    @property
    def enabled(self) -> bool:
        return any(
            (
                self.system_url,
                self.trade_url,
                self.btc_trade_url,
                self.xrp_trade_url,
                self.balance_url,
            )
        )

    def _url_for(self, channel: str) -> str | None:
        urls = {
            "system": self.system_url,
            "trade": self.trade_url,
            "trade_btc": self.btc_trade_url,
            "trade_xrp": self.xrp_trade_url,
            "balance": self.balance_url,
        }
        if channel not in urls:
            raise ValueError(f"알 수 없는 알림 채널: {channel}")
        return urls[channel]

    def _throttled(self, key: tuple, channel: str) -> str | None:
        """보내지 말아야 할 이유. 보내도 되면 None.

        같은 내용의 반복과 채널 폭주를 둘 다 막는다.
        """

        now = time.monotonic()

        last = self._last_sent.get(key)
        if last is not None and now - last < self.dedupe_seconds:
            self._suppressed[key] = self._suppressed.get(key, 0) + 1
            return "중복"

        recent = [t for t in self._recent.get(channel, []) if now - t < 60.0]
        self._recent[channel] = recent
        if len(recent) >= self.max_per_minute:
            return "속도제한"

        return None

    def _mark_sent(self, key: tuple, channel: str) -> int:
        """전송 기록. 그동안 눌려 있던 중복 건수를 돌려준다."""

        now = time.monotonic()
        self._last_sent[key] = now
        self._recent.setdefault(channel, []).append(now)
        return self._suppressed.pop(key, 0)

    def _post(self, payload: dict, channel: str = "system", key: str = "") -> bool:
        url = self._url_for(channel)
        if not url:
            return False

        dedupe_key = (channel, key or json.dumps(payload, sort_keys=True)[:400])
        reason = self._throttled(dedupe_key, channel)
        if reason:
            return False

        repeated = self._mark_sent(dedupe_key, channel)
        if repeated:
            # 눌려 있던 동안 몇 번 더 발생했는지 알려 준다.
            # 조용히 버리면 오류가 계속되는지 알 수 없다.
            payload = dict(payload)
            note = f"\n(직전 알림 이후 같은 내용 {repeated}건 생략)"
            if "content" in payload:
                payload["content"] = (payload["content"] + note)[:2000]
            elif payload.get("embeds"):
                embeds = [dict(e) for e in payload["embeds"]]
                embeds[0]["description"] = (embeds[0].get("description", "") + note)[:4096]
                payload["embeds"] = embeds

        try:
            if self._client is None:
                self._client = httpx.Client(timeout=self.timeout)
            resp = self._client.post(url, json=payload)
            # 디스코드는 성공 시 204를 준다
            if resp.status_code >= 400:
                raise httpx.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:                    # noqa: BLE001 - 매매를 막지 않는다
            failures = self._failures.get(channel, 0) + 1
            self._failures[channel] = failures
            if failures <= self.quiet_after:
                print(f"[notify:{channel}] 전송 실패 ({failures}회): {exc}")
            return False

        failures = self._failures.pop(channel, 0)
        if failures:
            print(f"[notify:{channel}] 복구됨 (실패 {failures}회 후)")
        return True

    # ---------------------------------------------------------------- 전송

    def text(self, message: str) -> bool:
        # 계좌명은 첫 줄 끝에 붙인다. 생존 확인처럼 계좌마다 같은 문장이 오면
        # 어느 계좌 잔고인지 알 길이 없다.
        if self.account:
            head, sep, tail = message.partition("\n")
            message = f"{head} [{self.account}]{sep}{tail}"
        return self._post({"content": message[:2000]}, "system")

    def _reset_throttle(self) -> None:
        """테스트용. 눌림 상태를 비운다."""

        self._last_sent.clear()
        self._suppressed.clear()
        self._recent.clear()

    def embed(
        self,
        title: str,
        fields: dict[str, str],
        color: int = COLOR_ENTRY,
        description: str = "",
        channel: str = "system",
        key: str = "",
    ) -> bool:
        if self.account:
            title = f"[{self.account}] {title}"
        return self._post(
            {
                "embeds": [
                    {
                        "title": title[:256],
                        "description": description[:4096],
                        "color": color,
                        "fields": [
                            {"name": k[:256], "value": str(v)[:1024], "inline": True}
                            for k, v in list(fields.items())[:25]
                        ],
                    }
                ]
            },
            channel,
            key,
        )

    @staticmethod
    def _tag(strategy: str, symbol: str, side: str = "") -> str:
        asset = symbol.removesuffix("_USDT")
        parts = [asset, strategy]
        if side:
            parts.append(side.upper())
        return "".join(f"[{part}]" for part in parts)

    @staticmethod
    def _trade_channel(symbol: str) -> str:
        if symbol == "BTC_USDT":
            return "trade_btc"
        if symbol == "XRP_USDT":
            return "trade_xrp"
        return "trade"

    @staticmethod
    def _value(value) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, Real):
            return f"{float(value):.6g}"
        if isinstance(value, (list, tuple, dict)):
            rendered = json.dumps(value, ensure_ascii=False, default=str)
            return rendered[:300] + ("…" if len(rendered) > 300 else "")
        return str(value)[:300]

    @classmethod
    def _context_text(cls, context: dict | None) -> str:
        if not context:
            return "기록된 상황 지표 없음"
        lines = [f"`{key}`: {cls._value(value)}" for key, value in context.items()]
        rendered = "\n".join(lines)
        return rendered[:1021] + ("…" if len(rendered) > 1021 else "")

    def entry(
        self,
        strategy: str,
        symbol: str,
        side: str,
        position,
        entry: float,
        *,
        reasons: list[str] | None = None,
        score: float = 0.0,
        context: dict | None = None,
        stop: float | None = None,
        target: float | None = None,
        tranche: int = 1,
        max_tranches: int = 1,
        order_type: str = "market",
    ) -> bool:
        reason_text = "\n".join(f"• {reason}" for reason in (reasons or []))
        return self.embed(
            f"{self._tag(strategy, symbol, side)} 진입",
            {
                "패턴/전략": strategy,
                "진입 근거": reason_text or "기록된 근거 없음",
                "판정 점수": f"{score:g}",
                "당시 상황": self._context_text(context),
                "진입가": f"{entry:g}",
                "손절가": "N/A" if stop is None else f"{stop:g}",
                "1차 익절가": "N/A" if target is None else f"{target:g}",
                "계약": f"{position.contracts:,}",
                "명목": f"{position.notional:,.2f} USDT",
                "레버리지": f"{position.leverage}x",
                "주문/분할": f"{order_type} · {tranche}/{max_tranches}",
                "손절": f"{position.stop_distance:.2%}",
                "리스크": f"{position.risk_amount:,.2f} USDT",
                "청산가": f"{position.liquidation_price:g} ({position.liquidation_multiple():.1f}x)",
            },
            COLOR_ENTRY,
            channel=self._trade_channel(symbol),
        )

    def exit(
        self,
        strategy: str,
        symbol: str,
        reason: str,
        price: float,
        pnl: float,
        *,
        side: str = "",
        partial: bool = False,
    ) -> bool:
        event = "부분 익절" if partial else f"청산 · {reason}"
        return self.embed(
            f"{self._tag(strategy, symbol, side)} {event}",
            {
                "전략": strategy,
                "사유": reason,
                "청산가": f"{price:g}",
                "손익": f"{pnl:+,.2f} USDT",
            },
            COLOR_WIN if pnl >= 0 else COLOR_LOSS,
            channel=self._trade_channel(symbol),
        )

    def balance_change(
        self,
        strategy: str,
        symbol: str,
        action: str,
        before: float,
        change: float,
        fee: float,
        after: float,
        initial_balance: float,
    ) -> bool:
        return_pct = (
            (after / initial_balance - 1.0) * 100.0 if initial_balance > 0 else 0.0
        )
        return self.embed(
            f"{self._tag(strategy, symbol)} 잔고 변화 · {action}",
            {
                "이전 잔고": f"{before:,.2f} USDT",
                "잔고 증감": f"{change:+,.4f} USDT",
                "수수료 (+비용/-리베이트)": f"{fee:+,.4f} USDT",
                "현재 잔고": f"{after:,.2f} USDT",
                "누적 수익률": f"{return_pct:+.2f}%",
            },
            COLOR_WIN if change >= 0 else COLOR_LOSS,
            channel="balance",
        )

    def warn(self, title: str, detail: str) -> bool:
        """경고. 같은 제목은 dedupe_seconds 안에 한 번만 나간다.

        내용이 아니라 **제목**으로 묶는다. 오류 메시지에 매번 다른 값이
        섞이면(타임스탬프, 잔고 등) 내용 기준으로는 중복이 잡히지 않는다.
        """

        return self.embed(
            title, {}, COLOR_WARN, description=detail,
            channel="system", key=f"warn:{title}",
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
