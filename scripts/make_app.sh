#!/usr/bin/env bash
# Build Jarvis.app from the SwiftPM package.
#
# SwiftPM produces a bare executable; macOS permissions (microphone,
# AppleScript automation) are granted per app bundle, so this wraps the
# binary in a minimal .app with the required usage descriptions and an
# ad-hoc signature.
#
# Usage: scripts/make_app.sh [output-dir]   (default: frontend/dist)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_DIR="$REPO_ROOT/frontend/JarvisApp"
OUT_DIR="${1:-$REPO_ROOT/frontend/dist}"
APP="$OUT_DIR/Jarvis.app"

echo "==> swift build -c release"
(cd "$PKG_DIR" && swift build -c release --product JarvisApp)
BIN="$(cd "$PKG_DIR" && swift build -c release --show-bin-path)/JarvisApp"

echo "==> Assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/Jarvis"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>               <string>Jarvis</string>
    <key>CFBundleDisplayName</key>        <string>Jarvis</string>
    <key>CFBundleIdentifier</key>         <string>dev.jarvis.assistant</string>
    <key>CFBundleVersion</key>            <string>0.3.0</string>
    <key>CFBundleShortVersionString</key> <string>0.3.0</string>
    <key>CFBundleExecutable</key>         <string>Jarvis</string>
    <key>CFBundlePackageType</key>        <string>APPL</string>
    <key>LSMinimumSystemVersion</key>     <string>14.0</string>
    <!-- Menu-bar app: no Dock icon -->
    <key>LSUIElement</key>                <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Jarvis listens for the wake word and voice commands.</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>Jarvis controls applications (open, close, windows, media) on your behalf.</string>
</dict>
</plist>
PLIST

# Ad-hoc signature so TCC permission grants stick across rebuilds.
codesign --force --sign - "$APP"

echo "Done: $APP"
echo "Launch with: open \"$APP\""
