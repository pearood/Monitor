#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$PROJECT_DIR/dist"
DOWNLOAD_DIR="$PROJECT_DIR/website/downloads"
APP_BUNDLE_NAME="小睿伴学.app"
APP_ZIP_NAME="小睿伴学-macos-arm64.zip"

mkdir -p "$DOWNLOAD_DIR"

if [ -d "$DIST_DIR/$APP_BUNDLE_NAME" ]; then
  echo "发现打包产物：$APP_BUNDLE_NAME"
  ditto -c -k --sequesterRsrc --keepParent "$DIST_DIR/$APP_BUNDLE_NAME" "$DOWNLOAD_DIR/$APP_ZIP_NAME"
  echo "已生成：$DOWNLOAD_DIR/$APP_ZIP_NAME"
  exit 0
fi

if [ -d "$DIST_DIR/FocusDesktop" ]; then
  echo "发现单目录产物：FocusDesktop"
  ditto -c -k --sequesterRsrc --keepParent "$DIST_DIR/FocusDesktop" "$DOWNLOAD_DIR/$APP_ZIP_NAME"
  echo "已生成：$DOWNLOAD_DIR/$APP_ZIP_NAME"
  exit 0
fi

echo "未找到可打包的桌面产物。"
echo "请先完成 PyInstaller 打包，再重新运行本脚本。"
exit 1
