#!/bin/zsh
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP="$PROJECT_DIR/AI Search.app"
BUILD_DIR="$(mktemp -d /private/tmp/ai-search-app.XXXXXX)"
BUILD_APP="$BUILD_DIR/AI Search.app"
CONTENTS="$BUILD_APP/Contents"
trap '/bin/rm -rf "$BUILD_DIR"' EXIT
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"
/usr/bin/clang -fobjc-arc -Wall -Wextra -framework Cocoa -framework WebKit "$PROJECT_DIR/macos/AI_Search_Launcher.m" -o "$CONTENTS/MacOS/AI Search"
/usr/bin/ditto "$PROJECT_DIR/macos/Info.plist" "$CONTENTS/Info.plist"
/usr/bin/ditto "$PROJECT_DIR/macos/project-path.txt" "$CONTENTS/Resources/project-path.txt"
/usr/bin/ditto "$PROJECT_DIR/AI Search.icns" "$CONTENTS/Resources/AI Search.icns"
/usr/bin/xattr -cr "$BUILD_APP"
/usr/bin/codesign --force --deep --sign - "$BUILD_APP"
/bin/rm -rf "$APP"
/usr/bin/ditto "$BUILD_APP" "$APP"
/usr/bin/xattr -d com.apple.FinderInfo "$APP" 2>/dev/null || true
/usr/bin/xattr -d 'com.apple.fileprovider.fpfs#P' "$APP" 2>/dev/null || true
/usr/bin/codesign --verify --deep --strict "$APP"
echo "$APP"
