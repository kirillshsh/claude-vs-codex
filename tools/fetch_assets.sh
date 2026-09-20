#!/usr/bin/env bash
# Собирает исходные ассеты персонажей из ВАШИХ установленных приложений и с claude.ai.
# Спрайты Clawd (Anthropic) и Codex-пета (OpenAI) в репозиторий не входят — это их IP.
# Нужны: macOS, Claude.app, ChatGPT.app, node (npx), ffmpeg, python3 + pillow/numpy/scipy.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/assets_src"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/128 Safari/537.36"
CLAUDE_APP="${CLAUDE_APP:-/Applications/Claude.app}"
CHATGPT_APP="${CHATGPT_APP:-/Applications/ChatGPT.app}"
mkdir -p "$SRC/clawd" "$ROOT/sprites" "$ROOT/fonts" "$ROOT/laptop"

echo "== Clawd GIF (claude.ai)"
for p in images/clawd.svg \
         images/clawd/core/Clawd-CrabWalking.gif images/clawd/core/Clawd-Lurking.gif images/clawd/core/Clawd-Waving.gif \
         images/clawd/core/Clawd-Jumping.gif images/clawd/core/Clawd-Pointing.gif \
         images/clawd/persona/Clawd-Cloud-once.gif images/clawd/persona/Clawd-Cloud-still.png \
         images/clawd/persona/Clawd-RacingCar.gif images/clawd/persona/Clawd-Book.gif \
         images/clawd/persona/Clawd-Magnifier.gif images/clawd/persona/Clawd-Cape.gif \
         images/home-page-assets/Clawd-JumpingHappy.gif \
         images/spotlights/claude-code-celebration/Clawd-Dancing.gif; do
  f="$SRC/clawd/$(basename "$p")"
  [ -s "$f" ] || curl -sfL --retry 3 -A "$UA" -o "$f" "https://claude.ai/$p" || echo "  !! не скачался: $p"
done

echo "== Codex-пет (спрайтшит из ChatGPT.app)"
if [ ! -s "$SRC/codex-spritesheet.png" ]; then
  npx --yes @electron/asar extract "$CHATGPT_APP/Contents/Resources/app.asar" "$SRC/_chatgpt" >/dev/null 2>&1
  sheet="$(ls "$SRC"/_chatgpt/webview/assets/codex-spritesheet-*.webp | head -1)"
  python3 -c "import sys; from PIL import Image; Image.open(sys.argv[1]).save(sys.argv[2])" "$sheet" "$SRC/codex-spritesheet.png"
  rm -rf "$SRC/_chatgpt"
fi

echo "== Clawd с ноутбуком (Claude.app)"
if [ -z "$(ls "$ROOT/laptop" 2>/dev/null)" ]; then
  ffmpeg -v error -c:v libvpx-vp9 -i "$CLAUDE_APP/Contents/Resources/ion-dist/images/install-hub/clawd-laptop.webm" \
         -pix_fmt rgba "$ROOT/laptop/f_%03d.png"
fi

echo "== Шрифты (OFL)"
[ -s "$ROOT/fonts/PressStart2P.ttf" ] || curl -sfL -o "$ROOT/fonts/PressStart2P.ttf" "https://github.com/google/fonts/raw/main/ofl/pressstart2p/PressStart2P-Regular.ttf"
[ -s "$ROOT/fonts/Tiny5.ttf" ] || curl -sfL -o "$ROOT/fonts/Tiny5.ttf" "https://github.com/google/fonts/raw/main/ofl/tiny5/Tiny5-Regular.ttf"

echo "== Нарезка в пиксельную сетку"
cd "$ROOT"
python3 extract_clawd.py "$SRC/clawd" sprites | tail -3
python3 pixelize_codex.py "$SRC/codex-spritesheet.png" sprites/codex_f4.png 4 14
python3 pixelize_codex.py "$SRC/codex-spritesheet.png" sprites/codex_f3.png 3 16
python3 pixelize_codex.py "$SRC/codex-spritesheet.png" sprites/codex_f2.png 2 20
python3 extract_laptop.py
echo "готово: sprites/, fonts/, laptop/"
