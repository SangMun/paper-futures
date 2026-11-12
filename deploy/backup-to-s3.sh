#!/usr/bin/env bash

# Paper-bot data backup. S3 is an archive only; the trading engine continues to
# read and write the local data directory even when a backup fails.

set -Eeuo pipefail

mode="${1:-}"
bot_root="${BOT_ROOT:-/home/ubuntu/paper-futures/v2}"
bucket="${BACKUP_BUCKET:-}"
region="${AWS_REGION:-ap-northeast-2}"
profile="${AWS_PROFILE:-xrp-paper-bot-backup}"
aws_bin="${AWS_BIN:-$(command -v aws || true)}"

if [[ "$mode" != "runtime" && "$mode" != "market" ]]; then
    echo "usage: $0 {runtime|market}" >&2
    exit 2
fi

if [[ -z "$bucket" ]]; then
    echo "BACKUP_BUCKET is required" >&2
    exit 2
fi

if [[ -z "$aws_bin" || ! -x "$aws_bin" ]]; then
    echo "AWS CLI was not found" >&2
    exit 2
fi

if [[ ! -d "$bot_root/data" ]]; then
    echo "bot data directory does not exist: $bot_root/data" >&2
    exit 2
fi

aws_args=(--profile "$profile" --region "$region" --no-cli-pager)

backup_runtime() {
    local day file found dir name experiment_dir
    day="$(date -u +%F)"
    found=0

    # data/ 는 단일 계좌, data/<account>/ 는 전략별 계좌(paper-futures@.service).
    # 계좌 폴더는 runtime/daily/<day>/<account>/ 아래로 올린다.
    for dir in "$bot_root/data" "$bot_root"/data/*/; do
        dir="${dir%/}"
        name="${dir#"$bot_root/data"}"
        name="${name#/}"
        for file in state.json trades.csv signals.jsonl; do
            if [[ -f "$dir/$file" ]]; then
                "$aws_bin" "${aws_args[@]}" s3 cp \
                    "$dir/$file" \
                    "s3://$bucket/runtime/daily/$day/${name:+$name/}$file" \
                    --sse AES256 \
                    --only-show-errors
                found=1
            fi
        done
    done

    # EXP-015는 전략별 CSV가 아니라 계좌별 JSONL/상태 파일을 쓴다.
    # 백업이 기존 runtime 파일만 보면 포워드 실험 증거가 서버에만 남는다.
    for name in xrp_dual_hedge_live xrp_dual_hedge_mark_live; do
        experiment_dir="$bot_root/data/$name"
        # 빈 디렉터리는 세션 구성만으로도 만들어질 수 있다. 실제 포워드 증거가
        # 기록된 경우에만 성공으로 간주해 기존 "백업 대상 없음" 감시를 보존한다.
        if [[ -d "$experiment_dir" ]] && find "$experiment_dir" -type f \
            \( -name state.json -o -name events.jsonl \) -print -quit | grep -q .; then
            "$aws_bin" "${aws_args[@]}" s3 sync \
                "$experiment_dir/" \
                "s3://$bucket/runtime/daily/$day/$name/" \
                --exclude "*.tmp" \
                --sse AES256 \
                --only-show-errors
            found=1
        fi
    done

    if [[ "$found" -eq 0 ]]; then
        echo "no runtime files found under $bot_root/data" >&2
        exit 1
    fi

    echo "runtime backup complete: s3://$bucket/runtime/daily/$day/"
}

backup_market() {
    local refresh_failed
    refresh_failed=0

    # Preserve the long local history, then append only the missing recent
    # range. A Gate API failure must not prevent the existing cache backup.
    if [[ "${BACKUP_REFRESH_MARKET_DATA:-1}" == "1" ]]; then
        if ! "$bot_root/.venv/bin/python" -m tools.fetch_history; then
            echo "market-data refresh failed; backing up the existing cache" >&2
            refresh_failed=1
        fi
    fi

    if [[ ! -d "$bot_root/data/cache" ]]; then
        echo "market-data cache does not exist: $bot_root/data/cache" >&2
        exit 1
    fi

    "$aws_bin" "${aws_args[@]}" s3 sync \
        "$bot_root/data/cache/" \
        "s3://$bucket/market-data/current/" \
        --exclude "*_oi.csv" \
        --exclude "*.tmp" \
        --exclude "*.bak" \
        --sse AES256 \
        --only-show-errors

    echo "market-data backup complete: s3://$bucket/market-data/current/"
    return "$refresh_failed"
}

case "$mode" in
    runtime) backup_runtime ;;
    market) backup_market ;;
esac
