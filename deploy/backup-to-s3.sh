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
    local day file found
    day="$(date -u +%F)"
    found=0

    for file in state.json trades.csv signals.jsonl; do
        if [[ -f "$bot_root/data/$file" ]]; then
            "$aws_bin" "${aws_args[@]}" s3 cp \
                "$bot_root/data/$file" \
                "s3://$bucket/runtime/daily/$day/$file" \
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
