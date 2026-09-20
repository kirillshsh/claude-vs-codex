import sys
sys.path.insert(0, '.')
import numpy as np
from PIL import Image, ImageDraw
from px import *
from chars import load
cl, cx = load()
cells = [(4, c) for c in range(5)] + [(5, c) for c in range(4)] + [(7, c) for c in range(6)] + [(10, 3), (10, 4), (9, 3), (9, 4), (8, 2), (1, 0), (1, 3)]
k = 4
w, h = 48, 52
img = Image.new('RGB', (len(cells) * (w + 2) * k, (h + 6) * k), (60, 60, 70))
d = ImageDraw.Draw(img)
for j, (r, c) in enumerate(cells):
    f = cx.frame(r, c, 1)
    a = f[:, :, 3:4] / 255.0
    bgc = np.zeros((h, w, 3)); bgc[:] = (90, 90, 100)
    rgb = (bgc * (1 - a) + f[:, :, :3] * a).astype(np.uint8)
    img.paste(Image.fromarray(rgb).resize((w * k, h * k), Image.NEAREST), (j * (w + 2) * k, 6 * k))
    d.text((j * (w + 2) * k + 2, 2), f'r{r}c{c}', fill=(255, 255, 0))
img.save('qa/b_prev_cx.png')
print(img.size, cx.anchor)
