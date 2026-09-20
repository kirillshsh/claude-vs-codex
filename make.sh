#!/usr/bin/env bash
# Полная сборка: ассеты -> рендер -> звук.   LANG_CARTOON=en ./make.sh  — английская версия.
set -euo pipefail
cd "$(dirname "$0")"
L="${LANG_CARTOON:-ru}"
[ -s sprites/clawd_meta.json ] || bash tools/fetch_assets.sh
mkdir -p out
CARTOON_LANG="$L" python3 render.py video --out "out/video_$L.mp4"
ffmpeg -y -v error -i "out/video_$L.mp4" -i audio/soundtrack.m4a -map 0:v -map 1:a -c copy -movflags +faststart "out/claude_vs_codex_$L.mp4"
echo "готово: out/claude_vs_codex_$L.mp4"
