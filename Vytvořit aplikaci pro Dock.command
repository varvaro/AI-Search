#!/bin/zsh
cd "/Users/miroslavvarvarovsky/Documents/AI Search" || exit 1
./macos/build_macos_app.sh
/usr/bin/open -R "AI Search.app"
