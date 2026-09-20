import sys
sys.path.insert(0, '.')
import numpy as np
from PIL import Image
from px import *
from chars import load
cl, cx = load()
out = np.zeros((40, 4 * 40, 3), np.uint8) + 90
for i, rot in enumerate([2, 1, 3, 0]):
    spr = cl.face('dizzy')
    if rot:
        spr = np.rot90(spr, rot)
    h, w = spr.shape[:2]
    a = spr[:, :, 3:4] / 255.0
    y0, x0 = 40 - h - 2, i * 40 + (40 - w) // 2
    out[y0:y0 + h, x0:x0 + w] = (out[y0:y0 + h, x0:x0 + w] * (1 - a) + spr[:, :, :3] * a).astype(np.uint8)
Image.fromarray(out).resize((out.shape[1] * 6, out.shape[0] * 6), Image.NEAREST).save('qa/b_var.png')
