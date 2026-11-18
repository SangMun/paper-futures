"""요청할 때만 도는 손실 사후 분석 → 디스코드.

    python -m tools.loss_ai              # 서버 동기화 → 새 손실만 분석 → 디스코드 전송
    python -m tools.loss_ai --dry-run    # 분석까지만. 디스코드 전송·분석 완료 표시 없음
    python -m tools.loss_ai --no-sync    # 이미 받아 둔 데이터로

정해진 시간에 돌지 않는다. 서버 봇이 거래 로그를 계속 쌓고, 이 명령을 실행할 때
마지막 분석 이후의 손실만 모아 분석한다.

- 숫자는 코드가 계산한다(tools/loss_bundle). AI는 해석만 한다.
- AI는 이 Mac의 Claude Code(`claude -p`, Pro 구독)로 부른다. API 키를 쓰지 않는다.
- 서버 파일은 읽기만 한다(rsync로 내려받기).

.env (공개 저장소라 값은 코드에 두지 않는다):
    LOSS_AI_SSH_HOST=ubuntu@<서버>
    LOSS_AI_SSH_KEY=~/.ssh/<키>.pem
    DISCORD_ANALYSIS_WEBHOOK_URL=https://discord.com/api/webhooks/...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from config import env
from reporting.logs import read_trades
from tools import loss_bundle as lb
from tools.loss_review import load_positions

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "loss_ai"
SERVER = WORK / "server"
STATE = WORK / "analyzed.json"
HISTORY = WORK / "analyses.jsonl"
REMOTE = "paper-futures/v2/data/"
# 계좌별 데이터 디렉터리. "" 는 계좌 분리(2026-09-15) 전 공용 로그다.
# EXP-015 양방향 실험은 한쪽 청산이 설계상 결과라 대상이 아니다.
ACCOUNTS = ("", "xrp", "btc25x", "btcchan")
BASELINE = "replay_xrp_live_like_raw"
LOSS_TYPES = ("즉시역행", "중간역행", "수익반납", "손절사냥", "시간초과", "체결문제", "기타")

PROMPT = """너는 암호화폐 선물 페이퍼 트레이딩 봇의 손실 사후 분석가다.
현재 디렉터리의 `bundles/*.json`(손실 1건당 1개)과 같은 이름의 `.png` 차트를 Read로 모두 읽고,
`context.md`(전략 설명·지난 실험 결론·최근 분석 이력)도 읽어라.

규칙:
- 숫자는 JSON에 있는 값만 인용한다. 직접 계산하거나 지어내지 않는다.
- `price_path.loss_type_hint`는 코드의 기계적 분류다. 차트와 근거를 보고 맞으면 따르고, 다르면 이유를 쓴다.
- "정상 범위의 손실"(전략이 감수하는 확률적 손실)과 "피할 수 있었던 손실"(근거상 들어가지 말았어야 함)을 구분한다.
- 규칙 변경을 확정적으로 권하지 않는다. 가설로만 쓰고, context.md에 이미 기각된 실험과 같은 제안은 하지 않는다.
- 같은 전략에서 같은 유형이 반복되면 repeated에 적는다(이번 묶음 + 최근 이력 기준).

출력은 아래 JSON 하나만. 코드 블록·설명 문장 없이.
{"trades": [{"id": "<json의 id>", "type": "<%s 중 하나>", "cause": "<한두 문장 원인, 근거 숫자 포함>",
  "avoidable": "예|아니오|불확실", "confidence": "높음|중간|낮음"}],
 "summary": "<이번 묶음 전체 3~5문장>",
 "repeated": ["<반복 패턴>", ...],
 "hypotheses": ["<검증할 가설, 리플레이로 확인 필요>", ...]}
""" % "|".join(LOSS_TYPES)


def sync() -> None:
    host = env.get("LOSS_AI_SSH_HOST")
    key = os.path.expanduser(env.get("LOSS_AI_SSH_KEY") or "")
    if not host or not key:
        sys.exit(".env에 LOSS_AI_SSH_HOST, LOSS_AI_SSH_KEY가 필요하다")
    rules = []
    for a in ACCOUNTS:
        prefix = f"/{a}" if a else ""
        rules += [f"--include={prefix}/trades.csv", f"--include={prefix}/signals.jsonl"]
        if a:
            rules.append(f"--include=/{a}/")
    rules += [f"--include=/{BASELINE}/", f"--include=/{BASELINE}/trades.csv",
              f"--include=/{BASELINE}/signals.jsonl", "--include=/cache/"]
    for sym in ("BTC_USDT", "XRP_USDT"):
        rules += [f"--include=/cache/{sym}_{tf}.csv" for tf in ("5m", "15m", "1h", "4h", "1d")]
    rules.append("--exclude=*")
    SERVER.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", "-e", f"ssh -i {key} -o BatchMode=yes", *rules,
         f"{host}:{REMOTE}", f"{SERVER}/"],
        check=True,
    )


def loss_type_hint(b: dict) -> str:
    """코드의 기계적 분류. AI가 확인·수정한다."""
    t, pp = b["trade"], b["price_path"]
    mfe = pp.get("mfe_R") or 0
    if t["exit_reason"] == "time":
        return "시간초과"
    if pp.get("after_exit_hit_tp1"):
        return "손절사냥"
    if mfe >= 1:
        return "수익반납"
    return "즉시역행" if mfe < 0.3 else "중간역행"


def new_losses(analyzed: dict) -> list[tuple[str, Path, dict]]:
    out = []
    for a in ACCOUNTS:
        d = SERVER / a if a else SERVER
        if not (d / "trades.csv").exists():
            continue
        positions, _ = load_positions(d)
        for p in positions:
            pid = f"{a or 'legacy'}:{p['strategy']}:{p['entry_ts']}"
            if p["pnl"] < 0 and pid not in analyzed:
                out.append((pid, d, p))
    return sorted(out, key=lambda x: x[2]["entry_ts"])


def context_md(strategies: set[str]) -> str:
    lines = ["# 참고 맥락", ""]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"## 매매 전략\n(.*?)\n---", readme, re.S)
    if m:
        lines += ["## 전략 설명 (README)", m.group(1).strip(), ""]
    exp = (ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8")
    m = re.search(r"## 실험 색인\n(.*?)\n## ", exp, re.S)
    if m:
        lines += ["## 지난 실험 결론 (EXPERIMENTS.md 색인) — 기각된 제안은 반복하지 말 것", m.group(1).strip(), ""]
    if HISTORY.exists():
        past = [json.loads(x) for x in HISTORY.read_text(encoding="utf-8").splitlines() if x.strip()]
        past = [x for x in past if x.get("strategy") in strategies][-30:]
        if past:
            lines += ["## 같은 전략의 최근 손실 분석 이력", ""]
            lines += [f"- {x['entry']} {x['strategy']} {x['side']} {x['type']}: {x['cause']}" for x in past]
    return "\n".join(lines) + "\n"


def ask_claude(run_dir: Path) -> dict:
    res = subprocess.run(
        ["claude", "-p", PROMPT, "--output-format", "json",
         "--allowedTools", "Read", "--add-dir", str(run_dir)],
        cwd=run_dir, capture_output=True, text=True, timeout=1800,
    )
    if res.returncode != 0:
        sys.exit(f"claude 실행 실패 ({res.returncode}): {res.stderr[-500:]}")
    (run_dir / "claude_raw.json").write_text(res.stdout, encoding="utf-8")
    text = json.loads(res.stdout).get("result", "")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        sys.exit(f"분석 결과에서 JSON을 찾지 못했다. {run_dir}/claude_raw.json 확인")
    return json.loads(m.group(0))


def post_discord(url: str, bundles: dict[str, dict], result: dict) -> None:
    by_id = {t["id"]: t for t in result.get("trades", [])}
    head = {
        "title": f"🔍 손실 분석 {len(bundles)}건",
        "description": (result.get("summary") or "")[:4000],
        "color": 0x5865F2,
        "fields": [
            {"name": "반복 패턴", "value": ("\n".join(f"• {x}" for x in result.get("repeated", [])) or "없음")[:1024]},
            {"name": "검증할 가설 (리플레이 전 적용 금지)",
             "value": ("\n".join(f"• {x}" for x in result.get("hypotheses", [])) or "없음")[:1024]},
        ],
    }
    embeds = [head]
    for bid, b in bundles.items():
        t = b["trade"]
        a = by_id.get(bid, {})
        mark = {"예": "⚠️", "아니오": "✅", "불확실": "❔"}.get(a.get("avoidable", ""), "❔")
        embeds.append({
            "title": f"🔻 {t['strategy']} {t['side']}  {t['net_pnl']:+.2f} USDT ({t['R']}R)"[:256],
            "description": (a.get("cause") or "분석 누락")[:4000],
            "color": 0xED4245,
            "fields": [
                {"name": "진입", "value": t["filled_at"][:16].replace("T", " "), "inline": True},
                {"name": "유형", "value": a.get("type", "-"), "inline": True},
                {"name": "피할 수 있었나", "value": f"{mark} {a.get('avoidable', '-')} ({a.get('confidence', '-')})",
                 "inline": True},
            ],
        })
    with httpx.Client(timeout=15) as c:
        for i in range(0, len(embeds), 10):    # 디스코드 메시지당 embed 10개 제한
            r = c.post(url, json={"embeds": embeds[i:i + 10]})
            r.raise_for_status()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=20, help="한 번에 분석할 최대 건수(오래된 것부터)")
    args = ap.parse_args()

    webhook = env.get("DISCORD_ANALYSIS_WEBHOOK_URL")
    if not args.dry_run and not webhook:
        sys.exit(".env에 DISCORD_ANALYSIS_WEBHOOK_URL이 필요하다 (또는 --dry-run)")
    if not args.no_sync:
        sync()

    analyzed = json.loads(STATE.read_text()) if STATE.exists() else {}
    todo = new_losses(analyzed)
    if not todo:
        print("새 손실 없음")
        return
    todo = todo[: args.limit]

    run_dir = WORK / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (run_dir / "bundles").mkdir(parents=True)
    candles = lb.Candles(SERVER / "cache")
    base = lb.baseline(SERVER / BASELINE) if (SERVER / BASELINE / "trades.csv").exists() else {}
    bundles: dict[str, dict] = {}
    trades_cache: dict[Path, list] = {}
    for pid, d, p in todo:
        trades = trades_cache.setdefault(d, read_trades(d / "trades.csv"))
        b = lb.build(p, trades, candles)
        b["id"] = pid
        b["price_path"]["loss_type_hint"] = loss_type_hint(b)
        b["replay_baseline"] = base.get(b["trade"]["strategy"], "기준 없음")
        stem = f"{b['trade']['filled_at'][:16].replace(':', '')}_{b['trade']['strategy'].replace('.', '_')}"
        (run_dir / "bundles" / f"{stem}.json").write_text(json.dumps(b, ensure_ascii=False, indent=2, default=str))
        lb.plot(b, candles, run_dir / "bundles" / f"{stem}.png")
        bundles[pid] = b
    (run_dir / "context.md").write_text(context_md({b["trade"]["strategy"] for b in bundles.values()}),
                                        encoding="utf-8")
    print(f"손실 {len(bundles)}건 분석 중... ({run_dir})")

    result = ask_claude(run_dir)
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    post_discord(webhook, bundles, result)
    by_id = {t["id"]: t for t in result.get("trades", [])}
    now = datetime.now(timezone.utc).isoformat()
    with HISTORY.open("a", encoding="utf-8") as f:
        for pid, b in bundles.items():
            a = by_id.get(pid, {})
            f.write(json.dumps({"id": pid, "analyzed_at": now, "strategy": b["trade"]["strategy"],
                                "side": b["trade"]["side"], "entry": b["trade"]["filled_at"][:16],
                                "type": a.get("type"), "cause": a.get("cause"),
                                "avoidable": a.get("avoidable")}, ensure_ascii=False) + "\n")
            analyzed[pid] = now
    STATE.write_text(json.dumps(analyzed, indent=1))
    print(f"디스코드 전송 완료, {len(bundles)}건 분석 완료로 표시")


if __name__ == "__main__":
    main()
