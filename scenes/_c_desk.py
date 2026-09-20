"""Ассеты уютной ночной комнаты для s6 (агент C): стол-плоскость (Mode 7), стена с окном, ночной город, реквизит."""
import math

import numpy as np

from px import *
from cam3d import Panorama
from worlds import noise2d

DESK_X0, DESK_X1, DESK_Z0, DESK_Z1 = -160.0, 160.0, -140.0, 44.0
WALL_Z = 44.0
WALL_X0, WALL_W, WALL_H = -160.0, 320, 170
WIN = (-38.0, 42.0, 40.0, 104.0)   # окно на стене: X0, X1, Y0, Y1 (мировые юниты)


def _c(h):
    return hexc(h)


def make_desk_texture(tpu=2):
    """деревянная столешница: доски вдоль X, волокна, стыки, сучки. RGB uint8."""
    w = int((DESK_X1 - DESK_X0) * tpu)
    h = int((DESK_Z1 - DESK_Z0) * tpu)
    rng = np.random.default_rng(12)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tones = [_c('#8c5a36'), _c('#7c4e2f'), _c('#976340'), _c('#83532f')]
    dark, seam, light = _c('#5e3920'), _c('#3c2414'), _c('#a8744a')
    ph = 22  # ширина доски в текселях
    plank = (yy // ph).astype(int)
    tex = np.zeros((h, w, 3), np.float32)
    n_pl = int(plank.max()) + 1
    tone_idx = rng.integers(0, len(tones), n_pl)
    offs = rng.uniform(0, 500, n_pl)
    for i in range(n_pl):
        tex[plank == i] = tones[tone_idx[i]]
    # волокна: волнистые линии
    wave = np.sin((xx + offs[plank]) / 41.0) * 2.2 + np.sin((xx + offs[plank] * 1.7) / 13.0) * 0.8
    g = (yy % ph + wave) % 5.0
    tex[g < 0.9] = tex[g < 0.9] * 0.86 + dark * 0.14
    bay = BAYER4[yy.astype(int) % 4, xx.astype(int) % 4]
    n = noise2d(max(w, h) if max(w, h) <= 2048 else 1024, 16, 3, 2)[:h, :w]
    tex[(n > 0.62) & (bay < 0.5)] = tex[(n > 0.62) & (bay < 0.5)] * 0.9 + light * 0.1
    # стыки досок и торцы
    tex[(yy % ph) < 1] = seam
    tex[(yy % ph) == 1] = tex[(yy % ph) == 1] * 0.8 + dark * 0.2
    for i in range(n_pl):
        for k in range(3):
            x = int(rng.uniform(0, w))
            tex[i * ph:(i + 1) * ph, x:x + 1] = seam
    # сучки
    for k in range(40):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        m = ((xx - cx) / 5.0) ** 2 + ((yy - cy) / 2.2) ** 2 < 1
        tex[m] = dark
        m2 = ((xx - cx) / 2.5) ** 2 + ((yy - cy) / 1.0) ** 2 < 1
        tex[m2] = seam
    return np.clip(tex, 0, 255).astype(np.uint8)


def make_wall():
    """стена (RGBA, 1 px = 1 юнит): обои, окно с рамой (стекло прозрачное), шторы, полка, гирлянда-крючки."""
    img = np.zeros((WALL_H, WALL_W, 4), np.uint8)
    yy, xx = np.mgrid[0:WALL_H, 0:WALL_W]
    base, stripe, dado, dado_d = _c('#2c2446'), _c('#262040'), _c('#3a2a3a'), _c('#2a1f2c')
    col = np.zeros((WALL_H, WALL_W, 3), np.float32)
    col[:] = base
    col[(xx % 12) < 2] = stripe
    # мелкий дизер-узор обоев
    m = ((xx % 12) == 6) & ((yy % 8) < 2)
    col[m] = _c('#342b52')
    # нижняя панель (у стола)
    wy = lambda Y: WALL_H - 1 - int(Y)
    col[wy(14):, :] = dado
    col[wy(14):wy(13) + 1, :] = _c('#4a3848')
    col[wy(2):, :] = dado_d
    img[:, :, :3] = col.astype(np.uint8)
    img[:, :, 3] = 255
    # окно
    X0, X1, Y0, Y1 = WIN
    ix0, ix1 = int(X0 - WALL_X0), int(X1 - WALL_X0)
    iy0, iy1 = wy(Y1), wy(Y0)
    frame, frame_l, frame_d = _c('#6b4a34'), _c('#8a6446'), _c('#40291c')
    img[iy0 - 3:iy1 + 5, ix0 - 3:ix1 + 3, :3] = frame
    img[iy0 - 3:iy1 + 5, ix0 - 3:ix0 - 2, :3] = frame_l
    img[iy0 - 3, ix0 - 3:ix1 + 3, :3] = frame_l
    # подоконник
    img[iy1 + 2:iy1 + 6, ix0 - 7:ix1 + 7, :3] = _c('#8a6446')
    img[iy1 + 2, ix0 - 7:ix1 + 7, :3] = _c('#a88062')
    img[iy1 + 5, ix0 - 7:ix1 + 7, :3] = frame_d
    # стекло — прозрачное, с переплётом-крестом
    img[iy0:iy1, ix0:ix1, 3] = 0
    mx, my = (ix0 + ix1) // 2, (iy0 + iy1) // 2
    img[iy0:iy1, mx - 1:mx + 1, :3] = frame
    img[iy0:iy1, mx - 1:mx + 1, 3] = 255
    img[my - 1:my + 1, ix0:ix1, :3] = frame
    img[my - 1:my + 1, ix0:ix1, 3] = 255
    # блики на стекле (диагональные штрихи)
    for (sx, sy, L) in ((ix0 + 5, iy0 + 4, 7), (ix0 + 8, iy0 + 4, 4), (mx + 5, my + 4, 6)):
        for k in range(L):
            if 0 <= sy + k < WALL_H and 0 <= sx + L - k < WALL_W:
                img[sy + k, sx + L - k, :3] = _c('#8fa0d8')
                img[sy + k, sx + L - k, 3] = 255
    # шторы
    cur, cur_d, cur_l = _c('#7a3a5e'), _c('#5a2846'), _c('#9a4e76')
    for side in (0, 1):
        for k in range(16):
            x = (ix0 - 12 + k) if side == 0 else (ix1 - 4 + k)
            for y in range(iy0 - 8, iy1 + 16):
                if not (0 <= x < WALL_W and 0 <= y < WALL_H):
                    continue
                # складки
                ph = (k + (y // 5) % 2) % 5
                c = cur_l if ph == 1 else (cur_d if ph >= 3 else cur)
                # подхват снизу: штора сужается
                pull = max(0, (y - (iy1 - 10))) * 0.35
                if side == 0 and k > 16 - pull:
                    continue
                if side == 1 and k < pull:
                    continue
                img[y, x, :3] = c
                img[y, x, 3] = 255
    # карниз
    img[iy0 - 11:iy0 - 9, ix0 - 16:ix1 + 16, :3] = _c('#3a2a24')
    img[iy0 - 11, ix0 - 16:ix1 + 16, :3] = _c('#6b4a34')
    # полка слева с книгами
    sx0, sx1, sy_ = int(-120 - WALL_X0), int(-66 - WALL_X0), wy(66)
    img[sy_:sy_ + 3, sx0:sx1, :3] = _c('#6b4a34')
    img[sy_, sx0:sx1, :3] = _c('#8a6446')
    rng = np.random.default_rng(4)
    x = sx0 + 3
    bcols = [_c('#c85a5a'), _c('#5a8ac8'), _c('#e0b050'), _c('#6ab07a'), _c('#9a6ac0'), _c('#d08040')]
    while x < sx1 - 8:
        bw = int(rng.integers(3, 6))
        bh = int(rng.integers(11, 17))
        c = bcols[int(rng.integers(0, len(bcols)))]
        img[sy_ - bh:sy_, x:x + bw, :3] = c
        img[sy_ - bh:sy_, x, :3] = np.clip(c * 1.2, 0, 255)
        img[sy_ - bh + 2, x:x + bw, :3] = np.clip(c * 0.7, 0, 255)
        x += bw + (1 if rng.random() < 0.7 else 4)
    # горшок с растением на полке
    px_ = sx1 - 8
    img[sy_ - 6:sy_, px_:px_ + 6, :3] = _c('#b0603a')
    img[sy_ - 6, px_ - 1:px_ + 7, :3] = _c('#c8744a')
    for (dx, dy) in ((0, -8), (1, -10), (2, -12), (3, -9), (4, -11), (5, -8), (-1, -9), (6, -10), (2, -7), (3, -7)):
        img[sy_ - 6 + dy + 3:sy_ - 6, px_ + dx, :3] = _c('#4a9a4a') if dx % 2 else _c('#6ac060')
    # рамка-картинка справа: сердечко
    fx0, fy0 = int(70 - WALL_X0), wy(84)
    img[fy0:fy0 + 22, fx0:fx0 + 26, :3] = _c('#6b4a34')
    img[fy0 + 2:fy0 + 20, fx0 + 2:fx0 + 24, :3] = _c('#e8dcc8')
    hrt = ['.RR.RR.', 'RRRRRRR', 'RRRRRRR', '.RRRRR.', '..RRR..', '...R...']
    for j, r in enumerate(hrt):
        for i, ch in enumerate(r):
            if ch == 'R':
                img[fy0 + 7 + j, fx0 + 9 + i, :3] = _c('#e05070')
    return img


def fairy_lights():
    """гирлянда над окном: мировые точки (X, Y, Z) и цвета."""
    X0, X1, Y0, Y1 = WIN
    pts, cols = [], []
    palette = [_c('#ffd27a'), _c('#ff9a8a'), _c('#9ad8ff'), _c('#b8f08a')]
    n = 15
    for i in range(n):
        u = i / (n - 1)
        X = X0 - 14 + (X1 - X0 + 28) * u
        Y = Y1 + 9 - 7 * math.sin(math.pi * u)
        pts.append((X, Y, WALL_Z - 0.5))
        cols.append(palette[i % len(palette)])
    return pts, cols


def make_skyline(f_ref=260.0, height=46, seed=9):
    """силуэты ночного города с огоньками окон (панорама)."""
    pw = int(2 * math.pi * f_ref)
    img = np.zeros((height, pw, 4), np.uint8)
    rng = np.random.default_rng(seed)
    far, near = _c('#232a52'), _c('#171b3a')
    lit = [_c('#ffd47a'), _c('#ffe6a8'), _c('#f0b060')]
    for layer, (col, hmin, hmax) in enumerate(((far, 8, 26), (near, 6, 40))):
        x = int(rng.integers(0, 10))
        while x < pw - 30:
            bw = int(rng.integers(8, 26))
            bh = int(rng.integers(hmin, hmax))
            y0 = height - bh
            img[y0:, x:x + bw, :3] = col
            img[y0:, x:x + bw, 3] = 255
            if layer == 1 and rng.random() < 0.25:
                ax = x + bw // 2
                img[max(0, y0 - 6):y0, ax, :3] = col
                img[max(0, y0 - 6):y0, ax, 3] = 255
            # окна
            for wy_ in range(y0 + 3, height - 2, 4):
                for wx in range(x + 2, x + bw - 2, 3):
                    if rng.random() < (0.22 if layer == 1 else 0.12):
                        c = lit[int(rng.integers(0, len(lit)))] * (1.0 if layer == 1 else 0.7)
                        img[wy_:wy_ + 2, wx, :3] = c
            x += bw + int(rng.integers(0, 6))
    return Panorama(img, f_ref, base_row=height - 1)


def make_moon(r=11):
    s = 2 * r + 3
    img = np.zeros((s, s, 4), np.uint8)
    yy, xx = np.mgrid[0:s, 0:s] - (s - 1) / 2
    d = np.hypot(xx, yy)
    m = d <= r
    img[m, :3] = _c('#f4ecd0')
    img[m, 3] = 255
    shade = m & (np.hypot(xx - 3.5, yy + 2.5) > r)  # серп тени справа-снизу
    img[shade & (d > r - 3), :3] = _c('#d8ccaa')
    for (cx, cy, cr) in ((-4, -3, 2.6), (3, 4, 2.0), (-2, 5, 1.4), (5, -4, 1.3)):
        mm = np.hypot(xx - cx, yy - cy) <= cr
        img[mm & m, :3] = _c('#ddd2b0')
    return img


LAMP = from_ascii([
    '......oooooooo......',
    '....ooCCCCCCCCoo....',
    '...oCCCCCCCCCCCCo...',
    '..oCCCWCCCCCCCCCCo..',
    '.oCCCWCCCCCCCCCCCdo.',
    '.oCCWCCCCCCCCCCCCdo.',
    'oCCCCCCCCCCCCCCCCCdo',
    'oddddddddddddddddddo',
    '.oYYYYYYYYYYYYYYYYo.',
    '...oooooYYYYooooo...',
    '........oSSo........',
    '........oSSo........',
    '........oSSo........',
    '........oSSo........',
    '........oSSo........',
    '........oSSo........',
    '........oSSo........',
    '.......oSSSSo.......',
    '.....ooSSSSSSoo.....',
    '....oSSSSSSSSSSo....',
    '....oooooooooooo....',
], {'o': (40, 26, 30), 'C': (240, 176, 96), 'W': (255, 226, 160), 'd': (196, 124, 64), 'Y': (255, 244, 190),
    'S': (120, 92, 100)})

MUG = from_ascii([
    '.ooooooooo...',
    'oMMMMMMMMMo..',
    'oMWMMMMMMMoooo',
    'oMWMRRMRRMo..o',
    'oMWMRRRRRMo..o',
    'oMWMMRRRMMo..o',
    'oMWMMMRMMMoooo',
    'oMMMMMMMMMo..',
    '.ooooooooo...',
], {'o': (46, 30, 40), 'M': (236, 232, 244), 'W': (255, 255, 255), 'R': (232, 84, 110)})

BOOKS = from_ascii([
    '..oooooooooooooooo..',
    '..oBBBBBBBBBBBBBBo..',
    '..oBWWWWWWWWWWWWBo..',
    '..oooooooooooooooo..',
    'oooooooooooooooooooo',
    'oGGGGGGGGGGGGGGGGGGo',
    'oGYYYYYYYYYYYYYYYYGo',
    'oooooooooooooooooooo',
    '.oooooooooooooooooo.',
    '.oRRRRRRRRRRRRRRRRo.',
    '.oRWWWWWWWWWWWWWWRo.',
    '.oRRRRRRRRRRRRRRRRo.',
    '.oooooooooooooooooo.',
], {'o': (40, 28, 36), 'B': (90, 138, 210), 'G': (90, 170, 110), 'R': (206, 90, 90), 'W': (240, 232, 214),
    'Y': (240, 210, 120)})

CACTUS = from_ascii([
    '.....oo.....',
    '....oGGo....',
    '.oo.oGgo....',
    'oGGooGgo.oo.',
    'oGgooGgooGGo',
    '.oGGGGgooGgo',
    '..ooGGgGGGo.',
    '....oGgooo..',
    '....oGgo....',
    '..oooooooo..',
    '..oTTTTTTo..',
    '..oTtTTTTo..',
    '...oTTTTo...',
    '...oooooo...',
], {'o': (30, 40, 30), 'G': (106, 180, 96), 'g': (70, 136, 70), 'T': (200, 110, 70), 't': (230, 150, 100)})
