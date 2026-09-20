import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw
from px import *
from chars import load

cl, cx = load()
S = 5
names = ['CrabWalking', 'Pointing', 'Jumping', 'JumpingHappy']
rows = []
for n in names:
    fr = cl.frames[n]
    rows.append((n, fr))
cw = 30 * S + 6
maxf = 22
Wd = cw * maxf
Hd = sum(fr[0].shape[0] * S + 22 for n, fr in rows) * 2 + 40
sheet = Image.new('RGBA', (Wd, Hd), (48, 52, 60, 255))
dr = ImageDraw.Draw(sheet)
y = 0
for n, fr in rows:
    dr.text((4, y + 4), n + ' anchor=%s' % (cl.anchor[n],), fill=(255, 255, 255, 255))
    y += 18
    for k, f in enumerate(fr):
        kk = k % maxf
        if k > 0 and kk == 0:
            y += f.shape[0] * S + 16
        im = Image.fromarray(f).resize((f.shape[1] * S, f.shape[0] * S), Image.NEAREST)
        sheet.alpha_composite(im, (kk * cw + 4, y))
        dr.text((kk * cw + 4, y + im.height - 10), str(k), fill=(200, 200, 90, 255))
    y += fr[0].shape[0] * S + 4
sheet.save(sys.argv[1])
print(sheet.size)

# codex rows
S2 = 2
rows2 = [0, 1, 2, 4, 5, 8, 9, 10]
cwc = 96 * S2 // 2 + 4
sh2 = Image.new('RGBA', (cwc * 16 + 10, len(rows2) * (104 * S2 // 2 + 14) + 10), (48, 52, 60, 255))
dr2 = ImageDraw.Draw(sh2)
y = 0
for r in rows2:
    dr2.text((2, y), 'row %d' % r, fill=(255, 255, 255, 255))
    for c in range(cx.ROW_N[r]):
        f = cx.frame(r, c, 2)
        im = Image.fromarray(f)
        sh2.alpha_composite(im, (c * cwc + 4, y + 12))
        dr2.text((c * cwc + 6, y + 12), str(c), fill=(255, 255, 90, 255))
    y += 104 + 14
sh2.save(sys.argv[2])
print(sh2.size)
