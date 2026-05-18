#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HOST="${1:-root@47.238.152.124}"
REMOTE_STAGE="/opt/xiaorui-deploy"
TMP_DIR="/tmp/xiaorui-site-bundle"
TMP_TAR="/tmp/xiaorui-site-bundle.tar.gz"
SCP_OPTS=(-O -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o TCPKeepAlive=yes -o IPQoS=throughput)
SSH_OPTS=(-o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o TCPKeepAlive=yes -o IPQoS=throughput)

echo "1) 生成下载包"
/bin/zsh "$PROJECT_DIR/scripts/build_release_downloads.sh"

echo "2) 准备上传内容"
rm -rf "$TMP_DIR" "$TMP_TAR"
mkdir -p "$TMP_DIR"
cp -R "$PROJECT_DIR/website" "$TMP_DIR/site"
cp "$PROJECT_DIR/deploy/nginx/xiaorui-site.conf" "$TMP_DIR/xiaorui-site.conf"
cp "$PROJECT_DIR/scripts/deploy_download_site_server.sh" "$TMP_DIR/deploy_download_site_server.sh"
tar -czf "$TMP_TAR" -C "$TMP_DIR" .

echo "3) 上传并在服务器部署"
scp "${SCP_OPTS[@]}" "$TMP_TAR" "$HOST:$REMOTE_STAGE-bundle.tar.gz"
ssh "${SSH_OPTS[@]}" "$HOST" "\
mkdir -p '$REMOTE_STAGE' && \
tar -xzf '$REMOTE_STAGE-bundle.tar.gz' -C '$REMOTE_STAGE' && \
chmod +x '$REMOTE_STAGE/deploy_download_site_server.sh' && \
/bin/bash '$REMOTE_STAGE/deploy_download_site_server.sh' '$REMOTE_STAGE'"

echo "4) 完成"
echo "访问 http://47.238.152.124/ 查看下载官网"
