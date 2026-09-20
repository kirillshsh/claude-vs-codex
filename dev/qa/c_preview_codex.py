import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw
from px import *
from chars import load

cl, cx = load()
rows = [int(r) for r in sys.argv[2].split(',')]
S = 3
cw = 96 * S // 2 + 6
sh = Image.new('RGBA', (cw * 8 + 10, len(rows) * (104 * S // 2 + 16) + 10), (70, 76, 90, 255))
dr = ImageDraw.Draw(sh)
y = 0
for r in rows:
    dr.text((2, y), 'row %d' % r, fill=(255, 255, 255, 255))
    for c in range(cx.ROW_N[r]):
        f = cx.frame(r, c, 2)
        im = Image.fromarray(f).resize((96 * S // 2, 104 * S // 2), Image.NEAREST)
        sh.alpha_composite(im, (c * cw + 4, y + 12))
        dr.text((c * cw + 6, y + 12), str(c), fill=(255, 255, 90, 255))
    y += 104 * S // 2 + 16
sh.save(sys.argv[1])
print(sh.size)
