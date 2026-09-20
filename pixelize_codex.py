"""Пикселизация спрайтшита Codex-пета (ChatGPT.app / Codex, codex-spritesheet-v6) под пиксельную сетку мира.

Кадр 192x208 -> уменьшение в F раз (area) -> порог альфы -> квантование в палитру спрайтшита
-> чистый 1px контур по границе маски.
"""
import sys, json
import numpy as np
from PIL import Image

SRC = sys.argv[1]
OUT = sys.argv[2]
F = float(sys.argv[3])
NCOL = int(sys.argv[4]) if len(sys.argv) > 4 else 14

sheet = np.array(Image.open(SRC).convert('RGBA')).astype(np.float32) / 255.0
CW, CH = 192, 208
COLS, ROWS = 8, 11


def srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


def downscale(cell, f):
    # премультиплицированное усреднение в линейном свете
    a = cell[:, :, 3:4]
    rgb = srgb_to_lin(cell[:, :, :3]) * a
    h, w = cell.shape[:2]
    nh, nw = int(round(h / f)), int(round(w / f))
    im_rgb = Image.fromarray(np.uint8(0) + np.zeros((1, 1), np.uint8))  # dummy
    out = np.zeros((nh, nw, 4), np.float32)
    for ch in range(3):
        out[:, :, ch] = np.array(Image.fromarray(rgb[:, :, ch]).resize((nw, nh), Image.BOX))
    out[:, :, 3] = np.array(Image.fromarray(a[:, :, 0]).resize((nw, nh), Image.BOX))
    al = out[:, :, 3:4]
    col = np.where(al > 1e-4, out[:, :, :3] / np.maximum(al, 1e-4), 0)
    return lin_to_srgb(col), out[:, :, 3]


def to_lab(rgb):
    lin = srgb_to_lin(rgb)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    A = 500 * (f[..., 0] - f[..., 1])
    B = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, A, B], -1)


# палитра: k-means по всем непрозрачным пикселям уменьшенных кадров
cells = []
for r in range(ROWS):
    for c in range(COLS):
        cell = sheet[r * CH:(r + 1) * CH, c * CW:(c + 1) * CW]
        if cell[:, :, 3].max() < 0.1:
            cells.append(None)
            continue
        cells.append(downscale(cell, F))

pix = np.concatenate([rgb[al > 0.5] for x in cells if x is not None for rgb, al in [x]], 0)
lab = to_lab(pix)
rng = np.random.default_rng(1)
sample = lab[rng.choice(len(lab), min(40000, len(lab)), replace=False)]
cent = sample[rng.choice(len(sample), NCOL, replace=False)]
for it in range(40):
    d = ((sample[:, None, :] - cent[None]) ** 2).sum(-1)
    lbl = d.argmin(1)
    for k in range(NCOL):
        if np.any(lbl == k):
            cent[k] = sample[lbl == k].mean(0)
# цвет палитры = средний sRGB кластера
samp_rgb = pix[rng.choice(len(pix), min(40000, len(pix)), replace=False)]
lab_s = to_lab(samp_rgb)
lbl = ((lab_s[:, None, :] - cent[None]) ** 2).sum(-1).argmin(1)
pal = np.array([samp_rgb[lbl == k].mean(0) if np.any(lbl == k) else [0, 0, 0] for k in range(NCOL)])
pal_lab = to_lab(pal)
order = np.argsort(pal_lab[:, 0])
pal, pal_lab = pal[order], pal_lab[order]
outline = pal[0]

frames = []
for x in cells:
    if x is None:
        frames.append(None)
        continue
    rgb, al = x
    mask = al > 0.45
    L = to_lab(rgb)
    idx = ((L[:, :, None, :] - pal_lab[None, None]) ** 2).sum(-1).argmin(-1)
    q = pal[idx]
    # контур: пиксели маски на границе -> самый тёмный цвет палитры
    pad = np.pad(mask, 1)
    edge = mask & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    q[edge] = outline
    o = np.zeros(mask.shape + (4,), np.uint8)
    o[:, :, :3] = np.clip(q * 255 + 0.5, 0, 255).astype(np.uint8)
    o[:, :, 3] = mask * 255
    frames.append(o)

fh, fw = next(f for f in frames if f is not None).shape[:2]
atlas = np.zeros((ROWS * fh, COLS * fw, 4), np.uint8)
for i, f in enumerate(frames):
    if f is None:
        continue
    r, c = divmod(i, COLS)
    atlas[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw] = f
Image.fromarray(atlas).save(OUT)
json.dump({'fw': fw, 'fh': fh, 'palette': (pal * 255).round().astype(int).tolist()}, open(OUT + '.json', 'w'))
print('frame', fw, fh, 'palette', (pal * 255).round().astype(int).tolist())
