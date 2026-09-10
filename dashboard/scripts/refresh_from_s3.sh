#!/usr/bin/env bash
# S3 백업에서 최신 봇 상태를 받아 대시보드 스냅샷을 다시 만든다.
#
# 봇은 Lightsail에서 돌고 대시보드는 로컬에서 뜬다. 그 사이를 잇는 경로가
# 없어서 스냅샷이 12일 동안 멈춰 있었다(2026-08-28 -> 2026-09-09).
# 서버가 이미 매일 S3로 백업하므로 그걸 되받는 쪽이 새 인프라 없이 해결된다.
#
# 사용법:
#   dashboard/scripts/refresh_from_s3.sh            # 최신 백업
#   dashboard/scripts/refresh_from_s3.sh 2026-09-05 # 특정 날짜

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export AWS_PROFILE="${AWS_PROFILE:-paper-bot-owner}"
export AWS_REGION="${AWS_REGION:-ap-northeast-2}"
BUCKET="${BACKUP_BUCKET:-<backup-bucket>}"

if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "AWS 자격증명이 없거나 만료됐다 (프로파일: $AWS_PROFILE)" >&2
  exit 1
fi

day="${1:-}"
if [ -z "$day" ]; then
  day="$(aws s3 ls "s3://$BUCKET/runtime/daily/" | tail -1 | awk '{print $2}' | tr -d /)"
  [ -n "$day" ] || { echo "백업을 찾지 못했다: s3://$BUCKET/runtime/daily/" >&2; exit 1; }
fi

echo "백업 날짜: $day"
mkdir -p data
aws s3 cp "s3://$BUCKET/runtime/daily/$day/" data/ --recursive --only-show-errors

# state.json 만 받으면 체결 로그와 시점이 어긋난다. 세 파일을 같이 확인한다.
for f in state.json trades.csv; do
  [ -s "data/$f" ] || { echo "data/$f 이 비었다 — 백업이 불완전하다" >&2; exit 1; }
done

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
"$PY" dashboard/scripts/export_dashboard_data.py

"$PY" - <<'PYEOF'
import datetime, json, pathlib
d = json.loads(pathlib.Path("dashboard/public/btc-dashboard.json").read_text())
ts = d["generatedAt"]
when = datetime.datetime.fromtimestamp(ts, datetime.UTC)
age = (datetime.datetime.now(datetime.UTC) - when).total_seconds() / 3600
print(f"스냅샷 {when:%Y-%m-%d %H:%M UTC} ({age:.1f}시간 전)")
print(f"잔고   {(d.get('account') or {}).get('balance', 0):.2f} USDT")
PYEOF
