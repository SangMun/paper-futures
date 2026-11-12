#!/usr/bin/env bash
# Lightsail 서버에서 계좌별 런타임 파일을 직접 받아 대시보드 스냅샷을 다시 만든다.
#
# refresh_from_s3.sh 와 같은 일을 하지만 S3(하루 1회 백업)가 아니라 서버의
# 현재 파일을 받는다. AWS 자격증명 없이 ssh 키만 있으면 된다.
#
#   BOT_HOST=ubuntu@<서버> SSH_KEY=~/.ssh/key.pem dashboard/scripts/refresh_from_server.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

HOST="${BOT_HOST:?BOT_HOST=ubuntu@<서버 주소> 지정}"
SSH_KEY="${SSH_KEY:?SSH_KEY=<키 경로> 지정}"
REMOTE="${BOT_ROOT:-/home/ubuntu/paper-futures/v2}/data/"

mkdir -p data
# 루트의 state/trades/signals 는 2026-09-15 분리 전 합산 계좌, 하위 폴더는 계좌별.
rsync -az -e "ssh -i $SSH_KEY" \
  --include='*/' --include='state.json' --include='trades.csv' --include='signals.jsonl' \
  --exclude='*' --exclude='cache/' --exclude='prev/' \
  "$HOST:$REMOTE" data/

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
"$PY" dashboard/scripts/export_dashboard_data.py

"$PY" - <<'PYEOF'
import json, pathlib
d = json.loads(pathlib.Path("dashboard/public/btc-dashboard.json").read_text())
for name, acct in d["accounts"].items():
    print(f"{name:<8} {acct['balance']:>9.2f} USDT  거래 {acct['closedTrades']}건 (분리 전 {acct['legacyTrades']})")
PYEOF
