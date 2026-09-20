import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw
from px import *
from chars import load

cl, cx = load()
S = 6
names = sys.argv[2:] if len(sys.argv) > 2 else ['Cloud-once', 'Laptop']
maxf = 16
blocks = []
for n in names:
    fr = cl.frames[n]
    blocks.append((n, fr))
fw = max(fr[0].shape[1] for n, fr in blocks) * S + 8
Wd = fw * maxf
Hd = 0
for n, fr in blocks:
    rows = (len(fr) + maxf - 1) // maxf
    Hd += 18 + rows * (fr[0].shape[0] * S + 16)
sheet = Image.new('RGBA', (Wd, Hd + 10), (48, 52, 60, 255))
dr = ImageDraw.Draw(sheet)
y = 0
for n, fr in blocks:
    dr.text((4, y + 4), n + ' n=%d anchor=%s size=%s' % (len(fr), cl.anchor[n], fr[0].shape[:2]), fill=(255, 255, 255, 255))
    y += 18
    for k, f in enumerate(fr):
        kk = k % maxf
        if k > 0 and kk == 0:
            y += f.shape[0] * S + 16
        bgim = Image.new('RGBA', (f.shape[1] * S, f.shape[0] * S), (70, 76, 90, 255))
        im = Image.fromarray(f).resize((f.shape[1] * S, f.shape[0] * S), Image.NEAREST)
        bgim.alpha_composite(im)
        sheet.alpha_composite(bgim, (kk * fw + 4, y))
        dr.text((kk * fw + 4, y + im.height + 2), str(k), fill=(200, 200, 90, 255))
    y += fr[0].shape[0] * S + 16
sheet.save(sys.argv[1])
print(sheet.size)
