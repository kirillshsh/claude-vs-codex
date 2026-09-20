import json, sys
import numpy as np
from PIL import Image, ImageDraw

d = sys.argv[1]
names = sys.argv[2].split(',')
out = sys.argv[3]
maxf = int(sys.argv[4]) if len(sys.argv) > 4 else 16
S = int(sys.argv[5]) if len(sys.argv) > 5 else 5
meta = json.load(open(f'{d}/clawd_meta.json'))
rows = []
for n in names:
    m = meta[n]
    strip = Image.open(f'{d}/clawd_{n}.png')
    idx = np.linspace(0, m['n'] - 1, min(maxf, m['n'])).astype(int)
    fr = [strip.crop((i * m['w'], 0, (i + 1) * m['w'], m['h'])) for i in idx]
    rows.append((n, fr, idx))
cw = max(m['w'] for m in (meta[n] for n in names)) * S + 8
W = cw * maxf
H = sum(meta[n]['h'] * S + 22 for n in names)
sheet = Image.new('RGBA', (W, H), (48, 52, 60, 255))
dr = ImageDraw.Draw(sheet)
y = 0
for n, fr, idx in rows:
    dr.text((4, y + 4), n, fill=(255, 255, 255, 255))
    y += 18
    for k, (f, i) in enumerate(zip(fr, idx)):
        big = f.resize((f.width * S, f.height * S), Image.NEAREST)
        sheet.alpha_composite(big, (k * cw + 4, y))
        dr.text((k * cw + 4, y + big.height - 10), str(i), fill=(200, 200, 90, 255))
    y += meta[n]['h'] * S + 4
sheet.save(out)
print(sheet.size)
