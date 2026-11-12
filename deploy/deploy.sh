#!/usr/bin/env bash
# 커밋 기반 pull 배포. systemd timer(paper-futures-deploy.timer)가 주기적으로 실행한다.
#
# Lightsail에는 IAM 역할을 붙일 수 없고, GitHub 러너에 SSH를 열면 넓은 IP 대역을
# 허용해야 한다. 그래서 서버가 GitHub에서 직접 당겨온다. 인바운드 포트가 필요 없다.
#
#   1. origin/$BRANCH를 fetch. 새 커밋이 없으면 종료
#   2. 매매 계좌(data/*/state.json)에 열린 포지션이 있으면 다음 주기로 미룸
#   3. 새 커밋으로 checkout, requirements가 바뀌었으면 pip install
#   4. pytest 실패 → 이전 커밋으로 복귀하고 경고 (서비스는 재시작하지 않음)
#   5. 서비스 재시작 → 하나라도 active가 아니면 이전 커밋으로 롤백
#   6. 결과를 디스코드 시스템 채널로 알림
#
# 수동 실행: sudo /home/ubuntu/paper-futures/v2/deploy/deploy.sh
# 배포 일시 중지: touch $REPO_DIR/DEPLOY_HOLD
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/paper-futures/v2}"
BRANCH="${BRANCH:-main}"
RUN_AS="${RUN_AS:-ubuntu}"
# 공백 구분. 실행 중이 아닌 유닛은 건너뛴다.
SERVICES="${SERVICES:-paper-futures@btc25x paper-futures@btcchan paper-futures@xrp paper-xrp-dual-hedge}"
SYSTEMCTL="${SYSTEMCTL:-systemctl}"
VENV_PY="${VENV_PY:-$REPO_DIR/.venv/bin/python}"
LOCK_FILE="${LOCK_FILE:-/tmp/paper-futures-deploy.lock}"
SETTLE_SECONDS="${SETTLE_SECONDS:-15}"
# 실패한 커밋을 기록해 같은 커밋으로 5분마다 재시도·재시작하지 않는다. 새 커밋이 오면 다시 시도한다.
FAILED_FILE="$REPO_DIR/data/.deploy-failed"

exec 9>"$LOCK_FILE"
if command -v flock >/dev/null; then
    flock -n 9 || { echo "다른 배포가 진행 중"; exit 0; }
fi

as_user() {
    if [[ "$(id -un)" == "$RUN_AS" ]]; then "$@"; else runuser -u "$RUN_AS" -- "$@"; fi
}
g() { as_user git -C "$REPO_DIR" "$@"; }

notify() {
    local url
    url="$(grep -E '^DISCORD_SYSTEM_WEBHOOK_URL=' "$REPO_DIR/.env" 2>/dev/null | cut -d= -f2- || true)"
    echo "$1"
    [[ -n "$url" ]] || return 0
    # 알림 실패가 배포 결과를 바꾸지 않는다.
    curl -fsS -m 10 -H 'Content-Type: application/json' \
        -d "$("$VENV_PY" -c 'import json,sys; print(json.dumps({"content": sys.argv[1]}))' "$1")" \
        "$url" >/dev/null || echo "디스코드 전송 실패" >&2
}

[[ -e "$REPO_DIR/DEPLOY_HOLD" ]] && { echo "DEPLOY_HOLD 있음, 건너뜀"; exit 0; }

g fetch -q origin "$BRANCH"
old="$(g rev-parse HEAD)"
new="$(g rev-parse "origin/$BRANCH")"
dirty="$(g status --porcelain --untracked-files=no)"
if [[ "$old" == "$new" && -z "$dirty" ]]; then
    exit 0
fi
if [[ "$(cat "$FAILED_FILE" 2>/dev/null)" == "$new" ]]; then
    exit 0
fi

# ponytail: 매매 계좌만 게이트한다. dual hedge 연속 진입 계좌는 항상 pair를 들고 있어
# 기다리면 영원히 배포가 안 된다. 그쪽은 SIGINT 종료 시 상태를 저장하고 재시작 때 복원한다.
open_positions="$(as_user "$VENV_PY" - "$REPO_DIR" <<'PY'
import json, pathlib, sys
n = 0
for p in pathlib.Path(sys.argv[1], "data").glob("*/state.json"):
    try:
        n += len(json.loads(p.read_text()).get("positions", []))
    except (OSError, ValueError):
        pass
print(n)
PY
)"
if [[ "$open_positions" != "0" ]]; then
    echo "열린 포지션 ${open_positions}개, 배포 보류: ${new:0:7}"
    exit 0
fi

short_old="${old:0:7}"; short_new="${new:0:7}"
req_changed="$(g diff --name-only "$old" "$new" -- requirements.txt)"

g reset -q --hard "$new"
if [[ -n "$req_changed" ]]; then
    as_user "$VENV_PY" -m pip install -q -r "$REPO_DIR/requirements.txt"
fi

if ! (cd "$REPO_DIR" && as_user env PYTHONDONTWRITEBYTECODE=1 "$VENV_PY" -m pytest -q -x -p no:cacheprovider >/tmp/paper-futures-deploy-test.log 2>&1); then
    g reset -q --hard "$old"
    [[ -n "$req_changed" ]] && as_user "$VENV_PY" -m pip install -q -r "$REPO_DIR/requirements.txt"
    as_user sh -c "echo $new > '$FAILED_FILE'"
    notify "🔴 배포 중단 ${short_new}: 테스트 실패, ${short_old} 유지 ($(tail -1 /tmp/paper-futures-deploy-test.log))"
    exit 1
fi

restart_all() {
    local s restarted=()
    for s in $SERVICES; do
        if "$SYSTEMCTL" is-enabled --quiet "$s" 2>/dev/null; then
            "$SYSTEMCTL" restart "$s"
            restarted+=("$s")
        fi
    done
    sleep "$SETTLE_SECONDS"
    for s in "${restarted[@]}"; do
        "$SYSTEMCTL" is-active --quiet "$s" || { echo "$s"; return 1; }
    done
}

if failed="$(restart_all)"; then
    notify "🚀 배포 완료 ${short_old} → ${short_new} ($(g log -1 --format=%s "$new"))"
else
    g reset -q --hard "$old"
    [[ -n "$req_changed" ]] && as_user "$VENV_PY" -m pip install -q -r "$REPO_DIR/requirements.txt"
    restart_all >/dev/null || true
    as_user sh -c "echo $new > '$FAILED_FILE'"
    notify "🔴 배포 롤백 ${short_new} → ${short_old}: ${failed} 기동 실패"
    exit 1
fi
