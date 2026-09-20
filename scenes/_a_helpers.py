"""Помощники сцен s0/s1/s2 (агент A). Только чтение общего движка, ничего не меняет в нём."""
import math
import numpy as np
from px import *
import bg
from cam3d import *
from chars import load, anim_frame

CL, CX = load()
_PIX = None
_WORLDS = {}


def get_pix():
    global _PIX
    if _PIX is None:
        _PIX = CodexPixelizer()
    return _PIX


def get_world(mode='day', **kw):
    from worlds import MeadowWorld
    key = (mode, tuple(sorted(kw.items())))
    if key not in _WORLDS:
        _WORLDS[key] = MeadowWorld(mode, **kw)
    return _WORLDS[key]


# ------------------------------------------------------------------ геометрия
def proj(cam, X, Y, Z):
    """экранные (x, y, px_per_unit) или None"""
    p = cam.project(X, Y, Z)
    if p is None:
        return None
    return p[0], p[1], cam.f / p[2]


def cam_look(eye, target, f, roll_shake=None):
    return look_at(eye, target, f)


def lerp3(a, b, t):
    return tuple(lerp(x, y, t) for x, y in zip(a, b))


def shake_offset(t, amp, seed=0, freq=31.0):
    """детерминированная тряска (целые пиксели)."""
    if amp <= 0:
        return 0, 0
    k = int(t * freq)
    r = np.random.default_rng(k * 7919 + seed * 104729 + 17)
    dx = int(round(r.uniform(-1, 1) * amp))
    dy = int(round(r.uniform(-1, 1) * amp))
    return dx, dy


def shift_canvas(c, dx, dy):
    """сдвиг кадра на целые пиксели с повтором края (тряска экрана)."""
    if dx == 0 and dy == 0:
        return c
    o = np.empty_like(c)
    ys = np.clip(np.arange(H) - dy, 0, H - 1)
    xs = np.clip(np.arange(W) - dx, 0, W - 1)
    o[:] = c[ys][:, xs]
    return o


def decay_shake(t, t0, dur, amp):
    if t < t0 or t > t0 + dur:
        return 0.0
    return amp * (1 - (t - t0) / dur) ** 1.5


# ------------------------------------------------------------------ титульный шрифт
def title_sprite(s, fill, hi, lo, ext, ol, scale=3, depth=1, gloss=(255, 255, 255), font='big', size=None):
    """«аркадный» логотип: заливка с полосами света/тени, блик, экструзия вниз-вправо, контур.
    Всё строится в разрешении глифа и масштабируется целым scale (ровная пиксельная сетка)."""
    m = text_mask(s, font, size)
    h, w = m.shape
    pad = 1
    Hh, Ww = h + 2 * pad + depth, w + 2 * pad + depth
    base = np.zeros((Hh, Ww), bool)
    base[pad:pad + h, pad:pad + w] = m
    ext_m = np.zeros((Hh, Ww), bool)
    for d in range(1, depth + 1):
        ext_m[pad + d:pad + d + h, pad + d:pad + d + w] |= m
    full = base | ext_m
    olm = outline_mask(full, 1) & ~full
    out = np.zeros((Hh, Ww, 4), np.uint8)
    out[olm, :3] = ol
    out[olm, 3] = 255
    e = ext_m & ~base
    out[e, :3] = ext
    out[e, 3] = 255
    rows = np.arange(Hh)[:, None] - pad + np.zeros((1, Ww), int)
    col = np.where(rows <= 1, 1, np.where(rows <= h - 3, 2, 3))
    cmap = {1: hi, 2: fill, 3: lo}
    for k, cc in cmap.items():
        mm = base & (col == k)
        out[mm, :3] = cc
        out[mm, 3] = 255
    # блик: верхний-левый пиксель штрихов в верхней части
    top = base & ~np.roll(base, 1, 0) & ~np.roll(base, 1, 1)
    if gloss is not None:
        out[top & (rows <= 2), :3] = gloss
    out = scale_nn(out, scale)
    return out


def shine_sweep(spr, phase, width=3, color=(255, 255, 255), slope=0.6):
    """диагональная полоса блеска по непрозрачным «светлым» пикселям спрайта. phase 0..1 — проход слева направо."""
    h, w = spr.shape[:2]
    o = spr.copy()
    x0 = -h * slope - width + phase * (w + h * slope + 2 * width)
    yy, xx = np.mgrid[0:h, 0:w]
    band = (xx + yy * slope >= x0) & (xx + yy * slope < x0 + width)
    lum = spr[:, :, :3].astype(np.int32) @ np.array([30, 59, 11]) / 100
    m = band & (spr[:, :, 3] > 0) & (lum > 60)
    o[m, :3] = color
    return o


# ------------------------------------------------------------------ молния между двумя точками
def _thick_line(dst, a, b, color, r):
    """линия толщиной (2r+1) — штампом квадратов, без сглаживания."""
    x0, y0, x1, y1 = a[0], a[1], b[0], b[1]
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for i in range(n + 1):
        u = i / max(1, n)
        x, y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
        rect(dst, round(x) - r, round(y) - r, 2 * r + 1, 2 * r + 1, color)


def bolt(dst, x0, y0, x1, y1, seed, core=(255, 255, 240), glow=(255, 210, 80), jag=10.0, step=15.0, branches=True,
         thick=1, glow_r=None):
    """ломаная молния между двумя точками: светящийся ореол + белая сердцевина + отростки."""
    rng = np.random.default_rng(seed)
    L = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(L / step))
    nx, ny = -(y1 - y0) / max(L, 1e-6), (x1 - x0) / max(L, 1e-6)
    pts = []
    for i in range(n + 1):
        u = i / n
        off = 0.0 if i in (0, n) else rng.uniform(-jag, jag) * (0.35 + 0.65 * math.sin(math.pi * u))
        along = 0.0 if i in (0, n) else rng.uniform(-0.3, 0.3) * step
        pts.append((x0 + (x1 - x0) * u + nx * off + (x1 - x0) / max(L, 1) * along,
                    y0 + (y1 - y0) * u + ny * off + (y1 - y0) / max(L, 1) * along))
    gr = thick + 1 if glow_r is None else glow_r
    br = []
    if branches:
        for (a, b) in zip(pts[1:-2], pts[2:-1]):
            if rng.random() < 0.45:
                ang = math.atan2(y1 - y0, x1 - x0) + rng.choice([-1, 1]) * rng.uniform(0.5, 1.2)
                ln = rng.uniform(7, 18)
                ex, ey = b[0] + math.cos(ang) * ln, b[1] + math.sin(ang) * ln
                br.append((b, (ex, ey)))
                if rng.random() < 0.5:
                    ang2 = ang + rng.uniform(-0.9, 0.9)
                    br.append(((ex, ey), (ex + math.cos(ang2) * ln * 0.6, ey + math.sin(ang2) * ln * 0.6)))
    for (a, b) in zip(pts[:-1], pts[1:]):
        _thick_line(dst, a, b, glow, gr)
    for (a, b) in br:
        _thick_line(dst, a, b, glow, 0)
    for (a, b) in zip(pts[:-1], pts[1:]):
        _thick_line(dst, a, b, core, thick - 1 if thick > 1 else 0)
        if thick >= 1:
            line(dst, a[0], a[1], b[0], b[1], core)
            line(dst, a[0], a[1] + 1, b[0], b[1] + 1, core)
    return pts


# ------------------------------------------------------------------ «!» и значки
EXCL = excl_sprite(2)
EXCL3 = excl_sprite(3)
EXCL4 = excl_sprite(4)


def pop_icon(dst, spr, cx, bottom_y, age, dur=0.22, bob=True):
    """иконка с поп-анимацией, стоит нижним краем на bottom_y, по центру cx."""
    if age < 0:
        return
    k = pop_scale(age, dur)
    s = spr if k >= 0.999 else squash(spr, k, k)
    h, w = s.shape[:2]
    by = bottom_y
    if bob and age > dur:
        by += round(math.sin((age - dur) * 9) * 1)
    blit(dst, s, cx - w / 2, by - h)


# ------------------------------------------------------------------ спрайты травы переднего плана
def make_tuft(seed, pal, w=9, h=7):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 4), np.uint8)
    g = pal['grass']
    cols = [g[3], g[2], g[0], g[1], g[4]]
    nb = int(rng.integers(3, 6))
    for i in range(nb):
        x = int(rng.integers(1, w - 1))
        hh = int(rng.integers(h // 2, h + 1))
        lean_ = rng.choice([-1, 0, 1])
        for k in range(hh):
            yy = h - 1 - k
            xx = x + (lean_ if k > hh * 0.6 else 0)
            if 0 <= xx < w:
                c = cols[min(4, k * 5 // max(1, hh))]
                img[yy, xx, :3] = np.clip(c, 0, 255)
                img[yy, xx, 3] = 255
    return img


def make_flower(seed, pal):
    rng = np.random.default_rng(seed)
    fl = pal['flowers'][int(rng.integers(0, len(pal['flowers'])))]
    g = pal['grass']
    img = np.zeros((6, 5, 4), np.uint8)
    for (y, x) in [(0, 2), (1, 1), (1, 3), (2, 2)]:
        img[y, x, :3] = fl
        img[y, x, 3] = 255
    img[1, 2, :3] = (255, 240, 150)
    img[1, 2, 3] = 255
    for y in range(3, 6):
        img[y, 2, :3] = g[2]
        img[y, 2, 3] = 255
    img[4, 1, :3] = g[0]
    img[4, 1, 3] = 255
    return img


class Foreground:
    """россыпь пучков травы/цветов в мировых координатах (билборды)."""

    def __init__(self, pal, region, n=60, seed=5, ppu=2.0, flowers=0.25):
        x0, x1, z0, z1 = region
        rng = np.random.default_rng(seed)
        self.items = []
        for i in range(n):
            X, Z = rng.uniform(x0, x1), rng.uniform(z0, z1)
            if rng.random() < flowers:
                spr = make_flower(int(rng.integers(1e6)), pal)
            else:
                spr = make_tuft(int(rng.integers(1e6)), pal, int(rng.integers(7, 12)), int(rng.integers(5, 9)))
            self.items.append((X, Z, spr, rng.uniform(0, 6.28)))
        self.ppu = ppu
        self._pos = None

    @property
    def pos(self):
        if self._pos is None or len(self._pos) != len(self.items):
            self._pos = np.array([(it[0], it[1]) for it in self.items], np.float32)
        return self._pos

    def world_items(self, t=0.0, sway=True, fog=None, max_px=None, near=None, limit=30):
        """near=(x, z, dmax) — только пучки рядом с камерой (ближний параллакс); limit — сколько ближайших."""
        idx = range(len(self.items))
        if near is not None:
            p = self.pos
            d2 = (p[:, 0] - near[0]) ** 2 + (p[:, 1] - near[1]) ** 2
            sel = np.where(d2 < near[2] ** 2)[0]
            if len(sel) > limit:
                sel = sel[np.argsort(d2[sel])[:limit]]
            idx = sel
        out = []
        for i in idx:
            X, Z, spr, ph = self.items[i]

            def fn(d, cm, X=X, Z=Z, spr=spr, ph=ph):
                p = cm.project(X, 0, Z)
                if p is None:
                    return
                s = cm.f / p[2] / self.ppu
                if max_px is not None and spr.shape[1] * s > max_px:
                    return
                sp = spr
                if sway:
                    lp = int(round(math.sin(t * 2.0 + ph) * 1.0))
                    if lp:
                        sp, pad = lean(spr, lp, 1.0)
                draw_sprite_3d(d, cm, sp, X, 0.0, Z, self.ppu, anchor=(0.5, 1.0), fog=fog)
            out.append((X, 0.0, Z, fn))
        return out


# ------------------------------------------------------------------ тени облаков по земле
class CloudShadows:
    def __init__(self, seed=3, size=256, scale=1 / 260.0, cover=0.42, dark=0.78):
        from worlds import noise2d
        self.n = noise2d(size, 4, seed, octaves=3, persistence=0.5)
        self.size = size
        self.scale = scale
        self.cover = cover
        self.dark = dark

    def shade_fn(self, t, vel=(9.0, 4.0)):
        def shade(out, X, Z, tt):
            Xn = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            Zn = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
            u = ((Xn + t * vel[0]) * self.scale * self.size).astype(np.int64) % self.size
            v = ((Zn + t * vel[1]) * self.scale * self.size).astype(np.int64) % self.size
            val = self.n[v, u]
            thr = 1 - self.cover
            k = np.clip((val - thr) / 0.06, 0, 1)
            k = np.round(k * 4) / 4
            m = DITHER < k
            out[m] *= self.dark
            return out
        return shade


# ------------------------------------------------------------------ векторные линии (numpy, без попиксельного Python)
def lines_np(dst, segs, color, alpha=1.0):
    """segs: (N, 4) x0, y0, x1, y1. Рисует 1-px линии выборкой точек с шагом 1 px (как Брезенхем)."""
    segs = np.asarray(segs, np.float32)
    if len(segs) == 0:
        return
    L = np.maximum(np.abs(segs[:, 2] - segs[:, 0]), np.abs(segs[:, 3] - segs[:, 1])).astype(np.int32) + 1
    idx = np.repeat(np.arange(len(segs)), L)
    starts = np.concatenate([[0], np.cumsum(L)[:-1]])
    k = np.arange(L.sum()) - np.repeat(starts, L)
    u = k / np.maximum(np.repeat(L, L) - 1, 1)
    xs = np.round(segs[idx, 0] + (segs[idx, 2] - segs[idx, 0]) * u).astype(np.int32)
    ys = np.round(segs[idx, 1] + (segs[idx, 3] - segs[idx, 1]) * u).astype(np.int32)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys = xs[ok], ys[ok]
    col = np.asarray(color, np.float32)
    if alpha >= 1:
        dst[ys, xs] = col
    else:
        dst[ys, xs] = dst[ys, xs] * (1 - alpha) + col * alpha


# ------------------------------------------------------------------ фокус-линии (аниме)
def focus_lines(dst, cx, cy, t, n=48, seed=7, color=(255, 255, 255), inner=70, alpha=1.0, speed=1.0, thick_every=5):
    rng = np.random.default_rng(seed + int(t * 15 * speed))
    R = math.hypot(W, H)
    segs = []
    for i in range(n):
        a = (i + rng.uniform(-0.4, 0.4)) / n * 2 * math.pi
        r0 = inner * rng.uniform(0.8, 1.4)
        ca, sa = math.cos(a), math.sin(a)
        segs.append((cx + ca * r0, cy + sa * r0, cx + ca * R, cy + sa * R))
        if i % thick_every == 0:
            segs.append((cx + ca * r0 + sa, cy + sa * r0 - ca, cx + ca * R + sa, cy + sa * R - ca))
    # обрезаем лучи по экрану, чтобы не выбирать лишние точки
    segs = np.array(segs, np.float32)
    for j in (2, 3):
        pass
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    tmax = np.ones(len(segs), np.float32)
    for (d, p0, lo, hi) in ((dx, segs[:, 0], 0, W - 1), (dy, segs[:, 1], 0, H - 1)):
        with np.errstate(divide='ignore', invalid='ignore'):
            t_hi = np.where(d > 0, (hi - p0) / d, np.where(d < 0, (lo - p0) / d, 1.0))
        tmax = np.minimum(tmax, np.clip(t_hi, 0, 1))
    segs[:, 2] = segs[:, 0] + dx * tmax
    segs[:, 3] = segs[:, 1] + dy * tmax
    lines_np(dst, segs, color, alpha)


# ------------------------------------------------------------------ Clawd: дополнительные позы
def clawd_frame(name, idx):
    return CL.frames[name][idx]


def clawd_draw_frame_3d(dst, cam, spr, X, Z, Y=0.0, anchor_xy=None, flipx=False, sx=1.0, sy=1.0, lean_px=0, fog=None,
                        tint=None, add=None):
    """произвольный спрайт Clawd (24-юнитная сетка) в мире, якорь по низу-центру непрозрачных пикселей."""
    if anchor_xy is None:
        ys, xs = np.where(spr[:, :, 3] > 0)
        ax, ay = (xs.min() + xs.max() + 1) / 2.0, ys.max() + 1
    else:
        ax, ay = anchor_xy
    if sx != 1.0 or sy != 1.0:
        h0, w0 = spr.shape[:2]
        spr = squash(spr, sx, sy)
        ax, ay = ax * spr.shape[1] / w0, ay * spr.shape[0] / h0
    if lean_px:
        spr, pad = lean(spr, lean_px)
        ax += pad
    anchor = (ax / spr.shape[1], ay / spr.shape[0])
    if flipx:
        anchor = (1 - anchor[0], anchor[1])
    return draw_sprite_3d(dst, cam, spr, X, Y, Z, 1.0, anchor=anchor, flipx=flipx, fog=fog, tint=tint, add=add)


def codex_head_top(cam, X, Z, Y=0.0):
    """экранная точка над головой Codex (для реплик/значков)."""
    return proj(cam, X, Y + 22.5, Z)


def clawd_head_top(cam, X, Z, Y=0.0):
    return proj(cam, X, Y + 16.5, Z)


# ------------------------------------------------------------------ лучи света (аниме-«блеск»)
def light_rays(dst, cx, cy, t, n=12, color=(255, 246, 200), alpha=0.45, r_in=10, r_out=260, spin=0.35, seed=1):
    x0, x1 = int(max(0, cx - r_out)), int(min(W, cx + r_out))
    y0, y1 = int(max(0, cy - r_out)), int(min(H, cy + r_out))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = YY[y0:y1, x0:x1], XX[y0:y1, x0:x1]
    ang = np.arctan2(yy - cy, xx - cx) + t * spin
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    k = (ang / (2 * math.pi) * n) % 1.0
    wedge = k < 0.42
    fall = np.clip((d - r_in) / 20, 0, 1) * np.clip(1 - (d - r_in) / (r_out - r_in), 0, 1)
    a = np.round(fall * alpha * 6) / 6
    m = wedge & (DITHER[y0:y1, x0:x1] < a)
    sub = dst[y0:y1, x0:x1]
    sub[m] = sub[m] * 0.4 + np.asarray(color, np.float32) * 0.6


# ------------------------------------------------------------------ пыль в мире
def dust_world(dst, cam, X, Z, age, seed=0, units=1.0, life=0.55, n=5, Y=0.6):
    if age < 0 or age > life:
        return
    p = proj(cam, X, Y, Z)
    if p is None:
        return
    sx, sy, s = p
    bg.dust_puffs(dst, age, sx, sy, n=n, seed=seed, life=life, spread=max(3.0, s * 7 * units),
                  size=max(1.2, s * 1.9 * units), color=(236, 232, 214), shade=(196, 188, 168))


# ------------------------------------------------------------------ движение с остановкой
def travel(u, stop_frac=0.22):
    """0..1 -> 0..1: постоянная скорость, в конце плавное торможение (последние stop_frac)."""
    u = clamp(u)
    a = 1 - stop_frac
    tot = a + stop_frac / 2
    if u < a:
        return u / tot
    v = u - a
    return (a + v - v * v / (2 * stop_frac)) / tot


def hop_y(t, t0, dur, height):
    if t < t0 or t > t0 + dur:
        return 0.0
    return height * math.sin(math.pi * (t - t0) / dur)


# ------------------------------------------------------------------ пиксельные птицы и листик
BIRD = [from_ascii(r, {'k': (40, 44, 60)}) for r in (
    ['k...k', '.k.k.', '..k..'],
    ['.....', 'kk.kk', '..k..'],
    ['.kkk.', 'k.k.k', '.....'],
)]

LEAF_S = [from_ascii(r, {'g': (70, 150, 60), 'd': (40, 100, 44), 'y': (150, 200, 80)}) for r in (
    ['.gg.', 'gyyg', '.gd.'],
    ['..g.', '.gyg', 'gyd.', '.d..'],
    ['.g..', 'gyg.', '.dyg', '..d.'],
    ['gg..', 'gyyg', '..dg'],
)]


# ------------------------------------------------------------------ именная плашка и звезда-бейдж
def name_plate(text_spr, accent, dark=(22, 12, 30), edge=(255, 255, 255), slant=10, padx=16, pady=5, lean_right=True):
    """тёмная скошенная плашка (как в файтингах) под титульной надписью: гарантированный контраст."""
    th, tw = text_spr.shape[:2]
    h = th + 2 * pady
    w = tw + 2 * padx + slant
    o = np.zeros((h + 3, w, 4), np.uint8)
    for y in range(h):
        off = int(round(slant * (1 - y / max(1, h - 1)))) if lean_right else int(round(slant * y / max(1, h - 1)))
        o[y, off:off + w - slant, :3] = dark
        o[y, off:off + w - slant, 3] = 255
        o[y, off, :3] = accent
        o[y, off + w - slant - 1, :3] = accent
    # акцентная полоса снизу и светлая кромка сверху
    for y in range(h, h + 3):
        off = 0 if lean_right else slant
        o[y, off:off + w - slant, :3] = accent
        o[y, off:off + w - slant, 3] = 255
    o[0, :, :3] = np.where(o[0, :, 3:4] > 0, np.asarray(edge, np.uint8)[None], o[0, :, :3])
    ox = (w - tw) // 2
    oy = pady - 1
    m = text_spr[:, :, 3] > 0
    sub = o[oy:oy + th, ox:ox + tw]
    sub[m] = text_spr[m]
    return o


def starburst(r_out, r_in, n=12, fill=(150, 24, 24), inner=(206, 52, 30), ol=(34, 6, 6), seed=3, jitter=0.18):
    """зубчатая «взрывная» звезда-бейдж (комикс)."""
    rng = np.random.default_rng(seed)
    R = int(math.ceil(r_out)) + 2
    size = 2 * R + 1
    yy, xx = np.mgrid[0:size, 0:size]
    dx, dy = xx - R, yy - R
    ang = np.arctan2(dy, dx)
    d = np.sqrt(dx * dx + dy * dy)
    radii = [(r_out if i % 2 == 0 else r_in) * (1 + rng.uniform(-jitter, jitter)) for i in range(2 * n)]
    k = (ang / (2 * math.pi) * 2 * n) % (2 * n)
    i0 = np.floor(k).astype(int)
    fr = k - i0
    r0 = np.array(radii)[i0]
    r1 = np.array(radii)[(i0 + 1) % (2 * n)]
    rr = r0 * (1 - fr) + r1 * fr
    body = d <= rr
    core = d <= rr * 0.72
    o = np.zeros((size, size, 4), np.uint8)
    olm = outline_mask(body, 1) & ~body
    o[olm, :3] = ol
    o[olm, 3] = 255
    o[body, :3] = fill
    o[body, 3] = 255
    o[core, :3] = inner
    return o


# ------------------------------------------------------------------ комиксовое облако драки (пиксельное)
def brawl_cloud2(dst, t, cx, cy, R, seed=11, ry_k=0.6, n_ring=15, n_in=6,
                 ol=(62, 54, 78), shade=(178, 172, 194), body=(236, 234, 244), hi=(255, 255, 255)):
    """клубящееся облако драки: объединение пульсирующих клубов, контур 2 px, тень снизу-справа, блики сверху-слева.
    Возвращает y верхнего края (для голов, выглядывающих из-за облака)."""
    rng = np.random.default_rng(seed)
    puffs = []
    for i in range(n_ring):
        a = i / n_ring * 2 * math.pi + math.sin(t * 5.0 + i) * 0.12 + t * 1.3
        rr = R * rng.uniform(0.30, 0.42) * (1 + 0.16 * math.sin(t * 17 + i * 2.3))
        puffs.append((cx + math.cos(a) * R * 0.78, cy + math.sin(a) * R * ry_k * 0.78, rr))
    for i in range(n_in):
        a = rng.uniform(0, 2 * math.pi) + t * 2.0
        d = rng.uniform(0.1, 0.45)
        rr = R * rng.uniform(0.35, 0.5) * (1 + 0.12 * math.sin(t * 13 + i))
        puffs.append((cx + math.cos(a) * R * d, cy + math.sin(a) * R * d * ry_k, rr))
    x0 = int(max(0, cx - R * 1.35 - 3))
    x1 = int(min(W, cx + R * 1.35 + 3))
    y0 = int(max(0, cy - R * ry_k * 1.4 - R * 0.5 - 3))
    y1 = int(min(H, cy + R * ry_k * 1.4 + R * 0.5 + 3))
    if x0 >= x1 or y0 >= y1:
        return cy
    yy, xx = YY[y0:y1, x0:x1], XX[y0:y1, x0:x1]
    m = np.zeros(yy.shape, bool)
    light = np.zeros(yy.shape, bool)
    dark = np.zeros(yy.shape, bool)
    for (px_, py_, rr) in puffs:
        d2 = (xx - px_) ** 2 + (yy - py_) ** 2
        inside = d2 <= rr * rr
        m |= inside
        light |= ((xx - px_ + rr * 0.28) ** 2 + (yy - py_ + rr * 0.30) ** 2 <= (rr * 0.62) ** 2)
        dark |= inside & ((xx - px_ - rr * 0.22) ** 2 + (yy - py_ - rr * 0.30) ** 2 > (rr * 0.92) ** 2)
    # нижняя часть облака — в тени
    dark |= m & (yy > cy + R * ry_k * 0.35)
    olm = outline_mask(m, 2) & ~m
    sub = dst[y0:y1, x0:x1]
    sub[olm] = ol
    sub[m] = body
    sub[m & dark & ~light] = shade
    sub[m & light] = hi
    ys = np.where(m.any(1))[0]
    return y0 + (ys.min() if len(ys) else 0)


# ------------------------------------------------------------------ Clawd 3/4 со спины (официальный кадр 3/4 без глаз)
_BACK34 = None


def clawd_back34():
    """CrabWalking[4] (стоит, 3/4 вправо, теневая сторона слева) с закрашенными глазами — как 'back' в chars.py."""
    global _BACK34
    if _BACK34 is None:
        f = CL.frames['CrabWalking'][4].copy()
        eye = (f[:, :, 3] > 0) & (f[:, :, :3].sum(2) == 0)
        f[eye, :3] = (217, 119, 87)
        _BACK34 = f
    return _BACK34


# ------------------------------------------------------------------ «дыхание» и отдача
def breathe(t, freq=1.5, amp=0.018, phase=0.0):
    """лёгкое покачивание по вертикали (множитель sy)."""
    return 1.0 + amp * math.sin(t * freq * 2 * math.pi + phase)


def beats(t, times, dur=0.26):
    """0..1: сила отдачи от ближайшего «удара» (крика, топота)."""
    k = 0.0
    for tb in times:
        a = t - tb
        if 0 <= a < dur:
            k = max(k, math.sin(math.pi * a / dur))
    return k
