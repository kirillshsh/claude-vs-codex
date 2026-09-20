"""clawd-laptop.webm (из Claude.app, VP9 с альфой) -> сетка 50px, цвета прищёлкнуты к официальной палитре."""
import glob, json
import numpy as np
from PIL import Image

fs = sorted(glob.glob('laptop/f_*.png'))
unit, ox, oy = 50, 36, 1
fr = [np.array(Image.open(f)) for f in fs]
H, W = fr[0].shape[:2]
nx, ny = int(np.ceil((W - ox) / unit)), int(np.ceil((H - oy) / unit))
xs = np.clip((ox + unit * (np.arange(nx) + 0.5)).astype(int), 0, W - 1)
ys = np.clip((oy + unit * (np.arange(ny) + 0.5)).astype(int), 0, H - 1)
PAL = np.array([[217, 119, 87], [191, 105, 77], [0, 0, 0]], np.float32)
sm = []
for a in fr:
    s = a[ys][:, xs].copy()
    op = s[:, :, 3] >= 128
    rgb = s[:, :, :3].astype(np.float32)
    idx = ((rgb[:, :, None, :] - PAL[None, None]) ** 2).sum(-1).argmin(-1)
    s[:, :, :3] = PAL[idx].astype(np.uint8)
    s[:, :, 3] = op * 255
    s[~op] = 0
    sm.append(s)
op = np.any(np.stack([s[:, :, 3] > 0 for s in sm]), 0)
yy, xx = np.where(op)
x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
sm = [s[y0:y1, x0:x1] for s in sm]
Image.fromarray(np.concatenate(sm, 1)).save('sprites/clawd_Laptop.png')
m = json.load(open('sprites/clawd_meta.json'))
m['Laptop'] = {'w': int(x1 - x0), 'h': int(y1 - y0), 'n': len(sm), 'durations': [83] * len(sm), 'crop': [int(x0), int(y0)]}
json.dump(m, open('sprites/clawd_meta.json', 'w'), indent=1)
print('laptop', x1 - x0, y1 - y0, len(sm))
