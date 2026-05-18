#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/config/deepseek.env"

if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  source "$CONFIG_FILE"
  set +a
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "请先设置 DEEPSEEK_API_KEY，例如："
  echo 'export DEEPSEEK_API_KEY="你的 DeepSeek API Key"'
  echo "或者在 $CONFIG_FILE 中配置。"
  exit 1
fi

export LOCAL_LLM_ENABLED=1
export LOCAL_LLM_PROVIDER=deepseek
export LOCAL_LLM_BASE_URL="https://api.deepseek.com"
export LOCAL_LLM_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"
export LOCAL_LLM_API_KEY="$DEEPSEEK_API_KEY"
REQUEST_TIMEOUT="${DEEPSEEK_TIMEOUT:-15}"
if [[ "${REQUEST_TIMEOUT}" -lt 15 ]]; then
  REQUEST_TIMEOUT=15
fi
export LOCAL_LLM_TIMEOUT="$REQUEST_TIMEOUT"

cd "$PROJECT_DIR"
exec python3 main.py
