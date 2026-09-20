"""Нарезка официальных GIF Clawd (claude.ai/images/clawd/...) в точную пиксельную сетку.

В больших GIF (2750x1850) шаг сетки 50px (полпикселя арта, тело Clawd = 12x8 арт-пикселей = 24x16 юнитов).
Маленькие (1189x800) — тот же арт в масштабе 1189/2750, шаг 21.618px.
Каждый кадр семплируется в центрах ячеек -> спрайт 1px = 1 юнит.
"""
import json, sys, glob, os
import numpy as np
from PIL import Image, ImageSequence

SRC = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)


def frames_of(path):
    im = Image.open(path)
    out = []
    for fr in ImageSequence.Iterator(im):
        out.append((np.array(fr.convert('RGBA')), fr.info.get('duration', 80)))
    return out


def best_offset(frames, unit, axis):
    # позиции переходов цвета вдоль оси -> фаза сетки
    hist = np.zeros(1000)
    for a, _ in frames:
        op = a[:, :, 3] > 0
        rgb = a[:, :, :3].astype(int) * op[:, :, None]
        if axis == 0:
            d = np.any(rgb[:, 1:] != rgb[:, :-1], axis=(0, 2)) | np.any(op[:, 1:] != op[:, :-1], axis=0)
        else:
            d = np.any(rgb[1:] != rgb[:-1], axis=(1, 2)) | np.any(op[1:] != op[:-1], axis=1)
        pos = np.where(d)[0] + 1
        ph = (pos % unit) / unit
        for p in ph:
            hist[int(p * 999)] += 1
    k = int(np.argmax(hist))
    return k / 999 * unit


meta = {}
for path in sorted(glob.glob(os.path.join(SRC, '*.gif'))):
    name = os.path.basename(path)[len('Clawd-'):-4]
    frames = frames_of(path)
    W, H = frames[0][0].shape[1], frames[0][0].shape[0]
    unit = 50.0 * W / 2750.0
    ox = best_offset(frames, unit, 0)
    oy = best_offset(frames, unit, 1)
    nx = int(np.ceil((W - ox) / unit))
    ny = int(np.ceil((H - oy) / unit))
    xs = np.clip((ox + unit * (np.arange(nx) + 0.5)).astype(int), 0, W - 1)
    ys = np.clip((oy + unit * (np.arange(ny) + 0.5)).astype(int), 0, H - 1)
    small = []
    for a, d in frames:
        s = a[ys][:, xs].copy()
        s[s[:, :, 3] < 128] = 0
        s[s[:, :, 3] >= 128, 3] = 255
        small.append(s)
    # общий bbox по всем кадрам
    op = np.any(np.stack([s[:, :, 3] > 0 for s in small]), axis=0)
    yy, xx = np.where(op)
    x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
    small = [s[y0:y1, x0:x1] for s in small]
    h, w = small[0].shape[:2]
    strip = np.concatenate(small, axis=1)
    Image.fromarray(strip).save(os.path.join(OUT, f'clawd_{name}.png'))
    meta[name] = {'w': int(w), 'h': int(h), 'n': len(small), 'durations': [int(d or 80) for _, d in frames],
                  'unit': unit, 'offset': [float(ox), float(oy)], 'crop': [int(x0), int(y0)]}
    print(name, 'unit %.3f' % unit, 'off %.1f %.1f' % (ox, oy), 'size', w, h, 'frames', len(small))

json.dump(meta, open(os.path.join(OUT, 'clawd_meta.json'), 'w'), indent=1)
