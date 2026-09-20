import sys
sys.path.insert(0, '.')
import numpy as np
from PIL import Image, ImageDraw
from px import *
from chars import load
cl, cx = load()
def strip(name, idxs, k=8):
    fr = cl.frames[name]
    h, w = fr[0].shape[:2]
    img = Image.new('RGB', (len(idxs) * (w + 2) * k, (h + 4) * k), (60, 60, 70))
    d = ImageDraw.Draw(img)
    for j, i in enumerate(idxs):
        f = fr[i]
        a = f[:, :, 3:4] / 255.0
        bgc = np.zeros((h, w, 3)); bgc[:] = (90, 90, 100)
        # checker to see alpha
        bgc[(np.add.outer(np.arange(h), np.arange(w)) % 2) == 0] = (80, 80, 90)
        rgb = (bgc * (1 - a) + f[:, :, :3] * a).astype(np.uint8)
        im = Image.fromarray(rgb).resize((w * k, h * k), Image.NEAREST)
        img.paste(im, (j * (w + 2) * k, 4 * k))
        d.text((j * (w + 2) * k + 2, 2), str(i), fill=(255, 255, 0))
    return img
strip('RacingCar', list(range(3, 13))).save('qa/b_prev_rc1.png')
strip('RacingCar', list(range(33, 43))).save('qa/b_prev_rc2.png')
strip('Laptop', list(range(14, 22))).save('qa/b_prev_lp.png')
for i in [4,5,6,10,11,12]:
    f = cl.frames['RacingCar'][i]
    ys, xs = np.where(f[:, :, 3] > 0)
    print(i, 'bbox x', xs.min(), xs.max(), 'y', ys.min(), ys.max())
print('anchor RC', cl.anchor['RacingCar'], 'Laptop', cl.anchor['Laptop'])
f = cl.frames['Laptop'][20]
ys, xs = np.where(f[:, :, 3] > 0); print('laptop20 bbox', xs.min(), xs.max(), ys.min(), ys.max())
f = cl.frames['Laptop'][0]
ys, xs = np.where(f[:, :, 3] > 0); print('laptop0 bbox', xs.min(), xs.max(), ys.min(), ys.max())
