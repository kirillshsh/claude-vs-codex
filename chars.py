"""Персонажи: Clawd (официальные GIF с claude.ai, сетка 12x8 арт-пикселей = 24x16 юнитов)
и Codex (официальный спрайтшит пета из Codex / ChatGPT.app, пикселизован).

Точка привязки у обоих: (x, y) = центр по X и линия земли (под лапами).
"""
import json
import math
import numpy as np
from scipy import ndimage
from px import *

BODY = np.array([217, 119, 87], np.uint8)
SHADE = np.array([191, 105, 77], np.uint8)
EYE = np.array([0, 0, 0], np.uint8)
PINK = np.array([245, 150, 150], np.uint8)
RED = np.array([232, 56, 86], np.uint8)
TEAR = np.array([120, 200, 255], np.uint8)


def anim_frame(t, durations_ms, loop=True, start=0, end=None):
    """индекс кадра по времени t (сек) для последовательности [start, end)."""
    end = len(durations_ms) if end is None else end
    ds = durations_ms[start:end]
    total = sum(ds) / 1000.0
    if total <= 0:
        return start
    if loop:
        t = t % total
    elif t >= total:
        return end - 1
    acc = 0.0
    for i, d in enumerate(ds):
        acc += d / 1000.0
        if t < acc:
            return start + i
    return end - 1


class Clawd:
    def __init__(self):
        self.meta = json.load(open(ROOT + '/sprites/clawd_meta.json'))
        self.frames = {}
        self.anchor = {}
        for name, m in self.meta.items():
            s = load_rgba(f'{ROOT}/sprites/clawd_{name}.png')
            fr = [s[:, i * m['w']:(i + 1) * m['w']] for i in range(m['n'])]
            self.frames[name] = fr
            f0 = fr[0]
            if not (f0[:, :, 3] > 0).any():
                f0 = fr[len(fr) // 2]
            ys, xs = np.where(f0[:, :, 3] > 0)
            self.anchor[name] = ((xs.min() + xs.max() + 1) / 2.0, ys.max() + 1)
        # базовый idle 24x16
        f0 = self.frames['Waving'][0]
        ys, xs = np.where(f0[:, :, 3] > 0)
        self.idle = f0[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        assert self.idle.shape[:2] == (16, 24), self.idle.shape
        self._faces = {}
        self._cache = {}

    # ------------------------------------------------ лица на базе idle
    def face(self, kind='idle', blush=False, legs=True, tears=0):
        key = (kind, blush, legs, tears)
        if key in self._faces:
            return self._faces[key]
        s = self.idle.copy()

        def put(r, c, col):
            s[r, c, :3] = col
            s[r, c, 3] = 255

        # стереть глаза
        for r in (2, 3):
            for c in (6, 7, 16, 17):
                put(r, c, BODY)
        L, R = 6, 16
        if kind in ('idle', 'blush'):
            for r in (2, 3):
                for c in (L, L + 1, R, R + 1):
                    put(r, c, EYE)
        elif kind == 'blink':
            for c in (L, L + 1, L + 2, R - 1, R, R + 1):
                put(3, c, EYE)
        elif kind == 'happy':
            for c in (L, L + 1, R, R + 1):
                put(2, c, EYE)
            for c in (L - 1, L + 2, R - 1, R + 2):
                put(3, c, EYE)
        elif kind == 'angry':  # как в официальном Pointing: внешний верхний пиксель убран
            put(2, L + 1, EYE)
            put(2, R, EYE)
            for c in (L, L + 1, R, R + 1):
                put(3, c, EYE)
        elif kind == 'furious':
            put(1, L + 2, EYE)
            put(1, R - 1, EYE)
            put(2, L + 1, EYE)
            put(2, R, EYE)
            for c in (L, L + 1, R, R + 1):
                put(3, c, EYE)
        elif kind == 'sad':
            put(2, L, EYE)
            put(2, R + 1, EYE)
            for c in (L, L + 1, R, R + 1):
                put(3, c, EYE)
        elif kind == 'surprised':
            for r in (1, 2, 3):
                for c in (L, L + 1, R, R + 1):
                    put(r, c, EYE)
        elif kind == 'dizzy':
            for (r, c) in [(1, 0), (1, 2), (2, 1), (3, 0), (3, 2)]:
                put(r, L - 1 + c, EYE)
                put(r, R - 1 + c, EYE)
        elif kind == 'love':
            heart = ['.X.X.', 'XXXXX', '.XXX.', '..X..']
            for r, row in enumerate(heart):
                for c, ch in enumerate(row):
                    if ch == 'X':
                        put(r, L - 1 + c, RED)
                        put(r, R - 2 + c, RED)
            put(1, L, np.array([255, 190, 200], np.uint8))
            put(1, R - 1, np.array([255, 190, 200], np.uint8))
        elif kind == 'back':
            pass
        if blush:
            for c in (L - 2, L - 1, R + 2, R + 3):
                put(4, c, PINK)
        if tears:
            put(4, L, TEAR)
            put(4, R + 1, TEAR)
        if not legs:
            s = s[:12]
        self._faces[key] = s
        return s

    # ------------------------------------------------ отрисовка
    def sprite(self, name, idx, k):
        key = (name, idx, k)
        if key not in self._cache:
            self._cache[key] = scale_nn(self.frames[name][idx], k)
        return self._cache[key]

    def draw_frame(self, dst, name, idx, x, y, k=2, flipx=False, alpha=1.0, tint=None, add=None):
        spr = self.sprite(name, idx, k)
        ax, ay = self.anchor[name]
        w = spr.shape[1]
        ox = ax * k
        if flipx:
            ox = w - ax * k
        blit(dst, spr, x - ox, y - ay * k, alpha=alpha, flipx=flipx, tint=tint, add=add)

    def draw_anim(self, dst, name, t, x, y, k=2, flipx=False, loop=True, start=0, end=None, speed=1.0, **kw):
        idx = anim_frame(t * speed, self.meta[name]['durations'], loop=loop, start=start, end=end)
        self.draw_frame(dst, name, idx, x, y, k, flipx, **kw)
        return idx

    def draw_face(self, dst, kind, x, y, k=2, flipx=False, blush=False, legs=True, tears=0, sx=1.0, sy=1.0,
                  lean_px=0, alpha=1.0, tint=None, add=None):
        s = self.face(kind, blush, legs, tears)
        s = scale_nn(s, k)
        if sx != 1.0 or sy != 1.0:
            s = squash(s, sx, sy)
        pad = 0
        if lean_px:
            s, pad = lean(s, lean_px)
        h, w = s.shape[:2]
        blit(dst, s, x - w / 2.0, y - h, alpha=alpha, flipx=flipx, tint=tint, add=add)
        return w, h


class Codex:
    ROW_N = [7, 8, 8, 4, 5, 8, 6, 6, 6, 8, 8]
    IDLE, RUN_R, RUN_L, WAVE, JUMP, FAIL, THINK, WORK, HAPPY, LOOK_A, LOOK_B = range(11)

    def __init__(self):
        self.atlas = {1: load_rgba(ROOT + '/sprites/codex_f4.png'), 2: load_rgba(ROOT + '/sprites/codex_f2.png')}
        self.fw = {1: 48, 2: 96}
        self.fh = {1: 52, 2: 104}
        self.anchor = {}
        for z in (1, 2):
            f = self.frame(0, 0, z)
            ys, xs = np.where(f[:, :, 3] > 0)
            self.anchor[z] = ((xs.min() + xs.max() + 1) / 2.0, ys.max() + 1)
        self._custom = {}

    def frame(self, row, col, zoom=1):
        fw, fh = self.fw[zoom], self.fh[zoom]
        return self.atlas[zoom][row * fh:(row + 1) * fh, col * fw:(col + 1) * fw]

    # ------------------------------------------------ экран-лицо
    def screen_info(self, row, col, zoom):
        key = ('scr', row, col, zoom)
        if key in self._custom:
            return self._custom[key]
        f = self.frame(row, col, zoom).astype(np.int32)
        op = f[:, :, 3] > 0
        lum = f[:, :, 0] * 0.3 + f[:, :, 1] * 0.59 + f[:, :, 2] * 0.11
        er = ndimage.binary_erosion(op, iterations=2)
        dark = (lum < 62) & er
        lab, n = ndimage.label(dark)
        if n == 0:
            self._custom[key] = None
            return None
        sizes = ndimage.sum(dark, lab, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        comp = lab == k
        comp = ndimage.binary_fill_holes(comp)
        ys, xs = np.where(comp)
        # цвета
        inner = ndimage.binary_erosion(comp, iterations=1)
        cols = f[inner][:, :3]
        l2 = cols[:, 0] * 0.3 + cols[:, 1] * 0.59 + cols[:, 2] * 0.11
        base = np.median(cols[l2 < 70], axis=0) if (l2 < 70).any() else np.array([25, 35, 79])
        glyph = cols[np.argmax(l2)]
        info = dict(mask=comp, inner=inner, bbox=(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), base=base, glyph=glyph)
        self._custom[key] = info
        return info

    def custom_face(self, row, col, zoom, kind):
        key = ('face', row, col, zoom, kind)
        if key in self._custom:
            return self._custom[key]
        f = self.frame(row, col, zoom).copy()
        info = self.screen_info(row, col, zoom)
        if info is None:
            return f
        x0, y0, x1, y1 = info['bbox']
        if kind == 'back':
            # заливка экрана: по строкам интерполяция между цветами головы слева и справа от экрана
            comp = ndimage.binary_dilation(info['mask'], iterations=1) & (f[:, :, 3] > 0)
            op = f[:, :, 3] > 0
            head_cols = f[op & ~comp][:, :3].astype(np.float32)
            hl = head_cols @ np.array([0.3, 0.59, 0.11])
            pal = np.unique(head_cols[hl > 70].astype(np.uint8), axis=0).astype(np.float32)
            lum = f[:, :, :3].astype(np.float32) @ np.array([0.3, 0.59, 0.11], np.float32)
            for y in range(f.shape[0]):
                xs_ = np.where(comp[y])[0]
                if len(xs_) == 0:
                    continue
                xl, xr = xs_.min(), xs_.max()
                a = xl - 1
                while a > 0 and (not op[y, a] or lum[y, a] < 70):
                    a -= 1
                b = xr + 1
                while b < f.shape[1] - 1 and (not op[y, b] or lum[y, b] < 70):
                    b += 1
                ca = f[y, a, :3].astype(np.float32) if op[y, a] else f[y, b, :3].astype(np.float32)
                cb = f[y, b, :3].astype(np.float32) if op[y, b] else ca
                for x in xs_:
                    tt = (x - xl + 0.5) / max(1, (xr - xl + 1))
                    col = ca * (1 - tt) + cb * tt
                    dd = ((pal - col) ** 2).sum(1)
                    j = np.argsort(dd)[:2]
                    # упорядоченный дизер между двумя ближайшими цветами палитры
                    d0, d1 = dd[j[0]] ** 0.5, dd[j[1]] ** 0.5
                    fr = d0 / max(1e-3, d0 + d1)
                    pick = j[1] if BAYER4[y % 4, x % 4] < fr else j[0]
                    f[y, x, :3] = pal[pick].astype(np.uint8)
            # убираем «>-» на груди: яркие пиксели в нижней половине -> цвет тела слева
            op = f[:, :, 3] > 0
            lum = f[:, :, :3].astype(np.int32) @ np.array([30, 59, 11]) / 100
            hy = int(f.shape[0] * 0.62)
            for y in range(hy, f.shape[0]):
                for x in range(f.shape[1]):
                    if op[y, x] and lum[y, x] > 175:
                        xx = x - 1
                        while xx > 0 and lum[y, xx] > 175:
                            xx -= 1
                        f[y, x, :3] = f[y, xx, :3]
            self._custom[key] = f
            return f
        # перерисовать лицо
        inner = info['inner']
        f[inner, :3] = info['base'].astype(np.uint8)
        gw = x1 - x0
        gh = y1 - y0
        glyph_col = info['glyph'].astype(np.uint8)
        g = GLYPHS[zoom].get(kind)
        if g is None:
            return f
        gl, gr = g
        cy = y0 + gh // 2
        for (gm, cx) in ((gl, x0 + int(round(gw * 0.30))), (gr, x0 + int(round(gw * 0.70)))):
            if gm is None:
                continue
            hh, ww = gm.shape[:2]
            yy0, xx0 = cy - hh // 2, cx - ww // 2
            for yy in range(hh):
                for xx in range(ww):
                    if gm[yy, xx, 3] > 0:
                        col = gm[yy, xx, :3] if gm[yy, xx, 0] != 1 else glyph_col
                        f[yy0 + yy, xx0 + xx, :3] = col
        self._custom[key] = f
        return f

    def draw(self, dst, row, col, x, y, zoom=1, flipx=False, face=None, alpha=1.0, tint=None, add=None,
             sx=1.0, sy=1.0, lean_px=0):
        f = self.frame(row, col, zoom) if face is None else self.custom_face(row, col, zoom, face)
        ax, ay = self.anchor[zoom]
        if sx != 1.0 or sy != 1.0:
            h, w = f.shape[:2]
            f = squash(f, sx, sy)
            ax, ay = ax * f.shape[1] / w, ay * f.shape[0] / h
        pad = 0
        if lean_px:
            f, pad = lean(f, lean_px, 0.8)
        w = f.shape[1]
        ox = ax + pad
        if flipx:
            ox = w - ax - pad
        blit(dst, f, x - ox, y - ay, alpha=alpha, flipx=flipx, tint=tint, add=add)

    def draw_anim(self, dst, row, t, x, y, zoom=1, fps=10, loop=True, frames=None, flipx=False, **kw):
        seq = frames if frames is not None else list(range(self.ROW_N[row]))
        i = int(t * fps)
        i = i % len(seq) if loop else min(i, len(seq) - 1)
        self.draw(dst, row, seq[i], x, y, zoom, flipx, **kw)
        return seq[i]


def _g(rows, color=None):
    cm = {'X': (1, 1, 1), 'R': (240, 70, 110), 'P': (255, 190, 210), 'W': (255, 255, 255)}
    return from_ascii(rows, cm)


GLYPHS = {
    1: {
        'heart': (_g(['RR.RR', 'RRRRR', '.RRR.', '..R..']), _g(['RR.RR', 'RRRRR', '.RRR.', '..R..'])),
        'happy': (_g(['.X.', 'X.X']), _g(['.X.', 'X.X'])),
        'lt3': (_g(['..X', '.X.', 'X..', '.X.', '..X']), _g(['XX.', '..X', '.X.', '..X', 'XX.'])),
        'sad': (_g(['X..', '.XX']), _g(['..X', 'XX.'])),
        'dot': (_g(['X', 'X', 'X']), _g(['X', 'X', 'X'])),
    },
    2: {
        'heart': (_g(['.RR.RR.', 'RRPRRRR', 'RRRRRRR', '.RRRRR.', '..RRR..', '...R...']),
                  _g(['.RR.RR.', 'RRPRRRR', 'RRRRRRR', '.RRRRR.', '..RRR..', '...R...'])),
        'happy': (_g(['..XX..', '.X..X.', 'X....X']), _g(['..XX..', '.X..X.', 'X....X'])),
        'lt3': (_g(['...XX', '..XX.', '.XX..', 'XX...', '.XX..', '..XX.', '...XX']),
                _g(['XXXX.', '....X', '....X', '.XXX.', '....X', '....X', 'XXXX.'])),
        'sad': (_g(['XX...', '..XX.', '....X']), _g(['...XX', '.XX..', 'X....'])),
        'dot': (_g(['XX', 'XX', 'XX', 'XX']), _g(['XX', 'XX', 'XX', 'XX'])),
    },
}

CLAWD = None
CODEX = None


def load():
    global CLAWD, CODEX
    if CLAWD is None:
        CLAWD = Clawd()
        CODEX = Codex()
    return CLAWD, CODEX
