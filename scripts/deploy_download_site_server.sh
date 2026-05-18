#!/bin/bash

set -euo pipefail

STAGE_DIR="${1:-/opt/xiaorui-deploy}"
SITE_DIR="$STAGE_DIR/site"
SITE_ROOT="/var/www/xiaorui-site"
NGINX_CONF_SRC="$STAGE_DIR/xiaorui-site.conf"
NGINX_CONF_DST="/etc/nginx/conf.d/xiaorui-site.conf"

SERVICE_FILE="/etc/systemd/system/focus-server.service"
if [ ! -f "$SERVICE_FILE" ]; then
  SERVICE_FILE="/opt/focus-server/focus-server.service"
fi

if [ ! -d "$SITE_DIR" ]; then
  echo "未找到站点目录：$SITE_DIR"
  exit 1
fi

if [ ! -f "$NGINX_CONF_SRC" ]; then
  echo "未找到 Nginx 配置：$NGINX_CONF_SRC"
  exit 1
fi

if [ ! -f "$SERVICE_FILE" ]; then
  echo "未找到 focus-server.service，请先确认云端后端已部署。"
  exit 1
fi

echo "1) 安装 Nginx（如未安装）"
if ! command -v nginx >/dev/null 2>&1; then
  apt-get update
  apt-get install -y nginx
fi

echo "2) 准备网站目录"
mkdir -p "$SITE_ROOT"
cp -R "$SITE_DIR"/. "$SITE_ROOT"/

echo "3) 安装 Nginx 配置"
mkdir -p /etc/nginx/conf.d
cp "$NGINX_CONF_SRC" "$NGINX_CONF_DST"
rm -f /etc/nginx/sites-enabled/default

echo "4) 调整后端到 127.0.0.1:8000"
python3 - <<'PY' "$SERVICE_FILE"
from pathlib import Path
import re
import sys

service_path = Path(sys.argv[1])
text = service_path.read_text()
updated = re.sub(r"--host\s+\S+\s+--port\s+\d+", "--host 127.0.0.1 --port 8000", text)
service_path.write_text(updated)
PY

echo "5) 重载并重启服务"
systemctl daemon-reload
systemctl restart focus-server
systemctl enable nginx
nginx -t
systemctl restart nginx

echo "6) 完成"
echo "站点目录: $SITE_ROOT"
echo "Nginx 配置: $NGINX_CONF_DST"
