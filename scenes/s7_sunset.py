"""s7_sunset — финал: закат на утёсе над морем. Сидят, прижимаются, обнимаются; кран в небо, звёзды, титр.

Шоты:
  0.0–5.6   общий со спины: сидят на краю утёса лицом к солнцу, Codex подсаживается, кладёт голову, обнимает
  5.6–11.6  встречный крупный (камера над морем): тёплый свет на лицах, ^^ -> сердечки, объятие, сердца
  11.6–18   снова со спины, выше; солнце садится, кран вверх в звёздное небо, «CLAWD ♥ CODEX», «конец»
Мир: край утёса по Z≈0 (земля при Z<0), море — плоскость Y=-60, солнце по азимуту +Z.
"""
import math
import numpy as np
from scipy import ndimage
from px import *
from bg import (P, hearts, stars_field, blit_dither, make_round_tree, make_pine, make_bush, smooth_noise,
                sparkles)
from cam3d import (Cam, look_at, lerp_cam, Ground, Panorama, draw_sprite_3d, CodexPixelizer, make_cloud_pano,
                   make_hill_pano, ground_shadow)
from worlds import make_grass_texture, noise2d
from chars import load
from scene_base import Scene

SEA_Y = -60.0
SUN_AZ = 0.0


def mix(a, b, t):
    return np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t


SUN_SIDE = [hexc(c) for c in ['#2a1848', '#4a2266', '#7e2c78', '#c24478', '#ee6a5e', '#fb9a52', '#ffcf78']]
ANTI_SIDE = [hexc(c) for c in ['#181a44', '#23265a', '#35306c', '#5a3a7a', '#8a4a82', '#b8607e', '#d48a86']]
DUSK_SUN = [hexc(c) for c in ['#0c0a22', '#16123a', '#2a1a52', '#4a2466', '#7a3070', '#b04a6a', '#de7a62']]
DUSK_ANTI = [hexc(c) for c in ['#07081c', '#0c0f2c', '#141a3c', '#20244c', '#342c5a', '#4c3462', '#664064']]


def edge_z(X):
    """линия края утёса (Z) в зависимости от X."""
    return -0.7 + np.sin(X * 0.13) * 0.6 + np.sin(X * 0.37 + 1.3) * 0.3


# ------------------------------------------------------------------ Codex: лица на исходнике 192x208
def _screen_comp(cell):
    a = cell[:, :, 3] > 0.5
    lum = cell[:, :, :3] @ np.array([0.3, 0.59, 0.11], np.float32)
    er = ndimage.binary_erosion(a, iterations=6)
    dark = (lum < 0.24) & er
    dark[int(cell.shape[0] * 0.62):] = False
    lab, n = ndimage.label(dark)
    if n == 0:
        return None
    sizes = ndimage.sum(dark, lab, range(1, n + 1))
    comp = ndimage.binary_fill_holes(lab == (int(np.argmax(sizes)) + 1))
    return comp


def _heart_mask(w, h):
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx + 0.5) / w * 2.6 - 1.3
    y = 1.25 - (yy + 0.5) / h * 2.5
    return (x * x + y * y - 1) ** 3 - x * x * y ** 3 <= 0


def _paint_face(cell, kind):
    cell = cell.copy()
    comp = _screen_comp(cell)
    if comp is None:
        return cell
    ys, xs = np.where(comp)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    inner = ndimage.binary_erosion(comp, iterations=5)
    cols = cell[inner][:, :3]
    lum = cols @ np.array([0.3, 0.59, 0.11], np.float32)
    base = np.median(cols[lum < 0.22], axis=0)
    glyph = cols[np.argmax(lum)]
    cell[inner, :3] = base
    gw, gh = x1 - x0, y1 - y0
    cy = y0 + gh * 0.5
    for fx in (0.3, 0.7):
        cx = x0 + gw * fx
        if kind == 'heart':
            hw, hh = int(gw * 0.24), int(gw * 0.21)
            m = _heart_mask(hw, hh)
            yy, xx = np.where(m)
            Y, X = (cy - hh / 2 + yy).astype(int), (cx - hw / 2 + xx).astype(int)
            cell[Y, X, :3] = (1.0, 0.33, 0.52)
            # блик
            hl = (yy < hh * 0.35) & (xx < hw * 0.35) & (xx > hw * 0.12)
            cell[Y[hl], X[hl], :3] = (1.0, 0.78, 0.86)
        elif kind == 'happy':
            r = gw * 0.1
            th = max(2.0, gw * 0.035)
            yy, xx = np.mgrid[int(cy - r - th):int(cy + th + 2), int(cx - r - th):int(cx + r + th + 1)]
            d = np.sqrt((xx - cx) ** 2 + (yy - (cy + r * 0.35)) ** 2)
            m = (np.abs(d - r) < th / 2 + 0.5) & (yy <= cy + r * 0.35)
            cell[yy[m], xx[m], :3] = glyph
    return cell


def _paint_back(cell):
    cell = cell.copy()
    comp = _screen_comp(cell)
    if comp is None:
        return cell
    comp = ndimage.binary_dilation(comp, iterations=3) & (cell[:, :, 3] > 0.5)
    op = cell[:, :, 3] > 0.5
    lum = cell[:, :, :3] @ np.array([0.3, 0.59, 0.11], np.float32)
    for y in range(cell.shape[0]):
        xs = np.where(comp[y])[0]
        if len(xs) == 0:
            continue
        xl, xr = xs.min(), xs.max()
        a = xl - 1
        while a > 0 and (not op[y, a] or lum[y, a] < 0.3):
            a -= 1
        b = xr + 1
        while b < cell.shape[1] - 1 and (not op[y, b] or lum[y, b] < 0.3):
            b += 1
        ca = cell[y, a, :3] if op[y, a] else cell[y, b, :3]
        cb = cell[y, b, :3] if op[y, b] else ca
        tt = ((xs - xl + 0.5) / max(1, xr - xl + 1))[:, None]
        cell[y, xs, :3] = ca * (1 - tt) + cb * tt
    # «>-» на груди
    hy = int(cell.shape[0] * 0.66)
    sub = cell[hy:]
    l2 = sub[:, :, :3] @ np.array([0.3, 0.59, 0.11], np.float32)
    bright = (l2 > 0.72) & (sub[:, :, 3] > 0.5)
    bright = ndimage.binary_dilation(bright, iterations=2) & (sub[:, :, 3] > 0.5)
    for y in range(sub.shape[0]):
        xs = np.where(bright[y])[0]
        if len(xs) == 0:
            continue
        xl = xs.min() - 1
        ref = sub[y, max(0, xl), :3]
        sub[y, xs, :3] = ref
    return cell


def face_heart(cell):
    return _paint_face(cell, 'heart')


def face_happy(cell):
    return _paint_face(cell, 'happy')


def face_back(cell):
    return _paint_back(cell)


_VY = (YY - H / 2) / (H / 2)
_VX = (XX - W / 2) / (W / 2)
_VIG = np.clip((np.sqrt(_VX * _VX * 0.7 + _VY * _VY) - 0.8) / 0.5, 0, 1)


def vignette(dst, strength=0.3):
    m = DITHER < _VIG * 0.9
    dst[m] *= (1 - strength)


class Pix(CodexPixelizer):
    """пикселизатор с палитрой Codex + розовые для сердечек."""

    def __init__(self):
        super().__init__()
        extra = np.array([[255, 84, 132], [255, 200, 220], [210, 50, 96]], np.float32)
        self.pal = np.concatenate([self.pal, extra], 0)
        g = (np.arange(32) * 8 + 4).astype(np.float32)
        R, G, B = np.meshgrid(g, g, g, indexing='ij')
        rgb = np.stack([R, G, B], -1).reshape(-1, 3)
        lab = self._lab(rgb / 255.0)
        plab = self._lab(self.pal / 255.0)
        d = ((lab[:, None, :] - plab[None]) ** 2).sum(-1)
        self.lut = self.pal[d.argmin(1)].reshape(32, 32, 32, 3).astype(np.uint8)


# ------------------------------------------------------------------ освещение спрайтов
def backlit(spr, rim=(255, 186, 120), mul=(0.70, 0.64, 0.70), add=(12, 2, 24), rim_amt=1.0, thick=None):
    o = spr.copy()
    a = o[:, :, 3] > 0
    h, w = a.shape
    rgb = o[:, :, :3].astype(np.float32) * np.asarray(mul, np.float32) + np.asarray(add, np.float32)
    # лёгкий вертикальный градиент: верх теплее
    yy = np.arange(h, dtype=np.float32)[:, None] / max(1, h - 1)
    warm = np.clip(0.35 - yy, 0, 0.35) / 0.35
    step = (BAYER4[np.arange(h)[:, None] % 4, np.arange(w)[None, :] % 4] < warm * 0.6)
    rgb[step & a] = rgb[step & a] * 0.75 + np.asarray(rim, np.float32) * 0.25
    th = thick if thick is not None else max(1, int(round(min(h, w) / 22)))
    r = np.asarray(rim, np.float32)
    top = np.zeros_like(a)
    side = np.zeros_like(a)
    for k in range(1, th + 1):
        up = np.zeros_like(a)
        up[k:] = a[:-k]
        top |= a & ~up
    lf = np.zeros_like(a)
    lf[:, 1:] = a[:, :-1]
    rt = np.zeros_like(a)
    rt[:, :-1] = a[:, 1:]
    side = a & (~lf | ~rt) & ~top
    rgb[top] = rgb[top] * (1 - rim_amt) + r * rim_amt
    rgb[side] = rgb[side] * (1 - 0.5 * rim_amt) + r * 0.5 * rim_amt
    o[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return o


def grass_clump(seed, h, w, dark=(34, 20, 46), tip=(255, 170, 110)):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 4), np.uint8)
    n = max(5, w // 2)
    for i in range(n):
        x0 = rng.uniform(w * 0.25, w * 0.75)
        bh = rng.uniform(h * 0.45, h)
        lean_ = rng.uniform(-0.45, 0.45) * bh
        steps = int(bh)
        for k in range(steps):
            f = k / max(1, steps - 1)
            x = x0 + lean_ * f * f
            y = h - 1 - k
            thick = max(1, int(round((1 - f) * 2.2)))
            for dx in range(thick):
                xi = int(round(x)) + dx
                if 0 <= xi < w and 0 <= y < h:
                    img[y, xi, :3] = tip if f > 0.82 else dark
                    img[y, xi, 3] = 255
    return img


def warmlit(spr, amt=1.0):
    o = spr.copy()
    rgb = o[:, :, :3].astype(np.float32)
    lit = rgb * np.array([1.03, 0.93, 0.86], np.float32) + np.array([10, 4, 0], np.float32)
    rgb = rgb * (1 - amt) + lit * amt
    o[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return o


def proj_sprite(dst, cam, spr, X, Y, Z, ppu_src, anchor, proc=None, crop_bottom=0.0):
    """как draw_sprite_3d, но с пост-обработкой уже отмасштабированного спрайта (свет в экранных пикселях)."""
    p = cam.project(X, Y, Z)
    if p is None:
        return None
    sx, sy, zc = p
    s = cam.f / zc / ppu_src
    h, w = spr.shape[:2]
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img = squash(spr, nw / w, nh / h)
    if crop_bottom > 0:
        cut = int(round(nh * crop_bottom))
        img = img[:nh - cut]
    if proc is not None:
        img = proc(img)
    x0 = sx - anchor[0] * nw
    y0 = sy - anchor[1] * nh
    blit(dst, img, x0, y0)
    return x0, y0, nw, nh, zc, s


def draw_arm(dst, p0, p1, r, body=(79, 122, 245), shade=(54, 86, 199), ol=(13, 23, 43), proc=None):
    """пухлая ручка: отрезок p0->p1 толщиной 2r, с контуром и тенью снизу; proc — пост-обработка спрайта."""
    x0, y0 = p0
    x1, y1 = p1
    pad = int(r) + 3
    bx0, by0 = int(min(x0, x1)) - pad, int(min(y0, y1)) - pad
    w, h = int(abs(x1 - x0)) + 2 * pad + 1, int(abs(y1 - y0)) + 2 * pad + 1
    yy, xx = np.mgrid[0:h, 0:w]
    ax, ay, bx, by = x0 - bx0, y0 - by0, x1 - bx0, y1 - by0
    vx, vy = bx - ax, by - ay
    L2 = max(1e-6, vx * vx + vy * vy)
    tt = np.clip(((xx - ax) * vx + (yy - ay) * vy) / L2, 0, 1)
    d = np.hypot(xx - (ax + tt * vx), yy - (ay + tt * vy))
    m = d <= r
    img = np.zeros((h, w, 4), np.uint8)
    img[m, :3] = body
    # тень — нижняя часть относительно оси
    below = (yy - (ay + tt * vy)) > r * 0.25
    img[m & below, :3] = shade
    img[m, 3] = 255
    edge = m & ~ndimage.binary_erosion(m)
    img[edge, :3] = ol
    if proc is not None:
        img = proc(img)
    blit(dst, img, bx0, by0)


def arm_sprite(length, thick, body=(79, 122, 245), shade=(54, 86, 199), ol=(13, 23, 43)):
    """пухлая ручка Codex (капсула) для объятия."""
    L, T = max(3, int(length)), max(3, int(thick))
    img = np.zeros((T + 2, L + 2, 4), np.uint8)
    yy, xx = np.mgrid[0:T + 2, 0:L + 2]
    r = T / 2.0
    cy = (T + 1) / 2.0
    cx0, cx1 = r + 0.5, L + 1.5 - r
    dx = np.clip(xx, cx0, cx1) - xx
    d = np.sqrt(dx ** 2 + (yy - cy) ** 2)
    m = d <= r
    img[m, :3] = body
    img[m & (yy > cy + r * 0.25), :3] = shade
    img[m, 3] = 255
    edge = m & ~ndimage.binary_erosion(m)
    img[edge, :3] = ol
    return img


# мини-титры после «КОНЕЦ»
CRED1 = text_sprite('ВСЁ СДЕЛАНО КОДОМ', (198, 176, 158), scale=1, outline=(22, 14, 26), shadow=(22, 14, 26))
CRED2 = text_sprite('CLAUDE OPUS 5.2', (255, 236, 210), scale=2, outline=(30, 12, 20), shadow=(30, 12, 20))
CRED3 = text_sprite('АНИМАЦИЯ · МУЗЫКА · ЗВУК', (146, 138, 172), scale=1, outline=(18, 12, 24))
CRED4 = text_sprite('ПРОМПТ: KIRILL SH', (176, 166, 196), scale=1, outline=(18, 12, 24))
CRED5 = text_sprite('TELEGRAM @SHNEURAL', (132, 188, 240), scale=1, outline=(14, 18, 34))


class Sunset(Scene):
    name = 's7_sunset'
    dur = 21.0

    def __init__(self):
        self.cl, self.cx = load()
        self.pix = Pix()
        pg = P('golden')
        ps = P('sunset')
        # земля: контровая (со спины) и тёплая (встречный план)
        back_pal = dict(pg)
        back_pal['grass'] = [hexc(c) for c in ['#3a2a4a', '#6a4058', '#2c203e', '#20182e', '#8a5060']]
        back_pal['flowers'] = [hexc(c) for c in ['#8a4a6a', '#a8604e', '#9a6a5a', '#6a4a7a', '#904a5a']]
        self.clumps = [grass_clump(100 + i, 34 + i * 5, 26 + i * 3) for i in range(4)]
        self.g_back = Ground(make_grass_texture(back_pal, 512, seed=21, flowers=0.2, tufts=0.6), tpu=2.0, origin=(-128, -128))
        front_pal = dict(pg)
        front_pal['grass'] = [hexc(c) for c in ['#7a9a3e', '#c8c060', '#5a7a36', '#44602e', '#e8d878']]
        self.g_front = Ground(make_grass_texture(front_pal, 512, seed=22, flowers=0.8), tpu=2.0, origin=(-128, -128))
        # панорамы
        self.clouds = make_cloud_pano(ps, seed=31, height=150, n=12)
        self.islands = self._islands()
        land_cols = [hexc('#4a5a36'), hexc('#7a8a44')]
        self.land_hills = make_hill_pano(land_cols, seed=41, height=36, amp=16,
                                         tree_pal=[hexc(c) for c in ['#24301e', '#3a4a26', '#56662e', '#8a8a40']],
                                         trunk_pal=[hexc('#3a2a1e'), hexc('#5a3e28')])
        self.land_far = make_hill_pano([hexc('#6a6a7a'), hexc('#8a7a86')], seed=42, height=44, amp=24)
        # одинокое дерево на утёсе (силуэт со спины / в тепле спереди)
        tp = [hexc(c) for c in ['#1a1224', '#261a34', '#342446', '#4a3058']]
        self.tree_back = make_round_tree(5, 16, 14, tp, [hexc('#140c1c'), hexc('#1e1428')])
        tp2 = [hexc(c) for c in ['#2e3a1e', '#4a5a26', '#6e7a30', '#a8a040']]
        self.tree_front = make_round_tree(5, 16, 14, tp2, [hexc('#4a3220'), hexc('#6a4a2c')])
        self.bush_front = make_bush(9, 8, tp2)
        rng = np.random.default_rng(3)
        self.front_trees = [(float(rng.uniform(-260, 260)), float(rng.uniform(-420, -140)),
                             make_round_tree(int(rng.integers(1e6)), int(rng.integers(8, 14)), 8, tp2,
                                             [hexc('#4a3220'), hexc('#6a4a2c')]) if rng.random() < 0.6 else
                             make_pine(int(rng.integers(1e6)), int(rng.integers(26, 40)), tp2, [hexc('#4a3220'), hexc('#6a4a2c')]))
                            for _ in range(26)]
        # спрайты Clawd
        self.clawd_back = self.cl.face('back', legs=False)
        # отражение/блики на воде — случайные фазы
        self.rng = np.random.default_rng(11)

    # -------------------------------------------------------------- фон
    def _islands(self):
        f_ref = 260.0
        pw = int(2 * math.pi * f_ref)
        h = 26
        img = np.zeros((h, pw, 4), np.uint8)
        rng = np.random.default_rng(12)
        # острова только в секторе заката (азимут ~ ±0.9 рад)
        for k in range(7):
            az = rng.uniform(-0.9, 0.9)
            cx = int((az / (2 * math.pi)) * pw) % pw
            wdt = int(rng.uniform(20, 60))
            hh = rng.uniform(5, 18)
            for dx in range(-wdt, wdt + 1):
                x = (cx + dx) % pw
                prof = hh * (1 - (abs(dx) / wdt) ** 1.6) + math.sin(dx * 0.4) * 1.2
                top = int(h - 1 - max(0, prof))
                for y in range(top, h):
                    col = hexc('#5a2a5e') if y < h - 3 else hexc('#7a3a66')
                    if y == top:
                        col = hexc('#c05a70')
                    img[y, x, :3] = col
                    img[y, x, 3] = 255
        return Panorama(img, f_ref, base_row=h - 1)

    def sky(self, dst, cam, dusk, rays):
        dx, dy, dz = rays
        hor = np.sqrt(dx * dx + dz * dz)
        elev = np.arctan2(dy, hor)
        az = np.arctan2(dx, dz)
        w = (1 + np.cos(az - SUN_AZ)) / 2  # 1 — сторона солнца
        n = len(SUN_SIDE) - 1
        v = np.clip(1 - elev / 0.8, 0, 1) * n
        k = np.clip(np.floor(v).astype(int), 0, n - 1)
        fr = np.round((v - k) * 4) / 4
        pick = (DITHER < fr)
        kk = np.where(pick, k + 1, k)
        sun_pal = np.stack([mix(a, b, dusk) for a, b in zip(SUN_SIDE, DUSK_SUN)])
        anti_pal = np.stack([mix(a, b, dusk) for a, b in zip(ANTI_SIDE, DUSK_ANTI)])
        cs = sun_pal[kk]
        ca = anti_pal[kk]
        wq = np.round(w * 5) / 5
        wd = np.where(DITHER < (w * 5 - np.floor(w * 5)), np.minimum(1, np.floor(w * 5) / 5 + 0.2), np.floor(w * 5) / 5)
        dst[:] = cs * wd[:, :, None] + ca * (1 - wd[:, :, None])
        return elev, az

    def sun(self, dst, cam, elev_sun, glow=1.0):
        # проекция направления солнца
        d = 10000.0
        X = cam.x + math.sin(SUN_AZ) * d * math.cos(elev_sun)
        Z = cam.z + math.cos(SUN_AZ) * d * math.cos(elev_sun)
        Y = cam.y + d * math.sin(elev_sun)
        p = cam.project(X, Y, Z)
        if p is None:
            return None
        sx, sy, _ = p
        R = cam.f * 0.075
        # ореол — дизерные кольца
        for i, (rr, col, a) in enumerate([(R * 3.2, (255, 170, 110), 0.25), (R * 2.2, (255, 190, 120), 0.45),
                                          (R * 1.5, (255, 214, 140), 0.7)]):
            m = disc_mask(rr)
            r0 = (m.shape[0] - 1) // 2
            x0, y0 = int(round(sx - r0)), int(round(sy - r0))
            hh, ww = m.shape
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + hh)
            if xa < xb and ya < yb:
                mm = m[ya - y0:yb - y0, xa - x0:xb - x0] & (DITHER[ya:yb, xa:xb] < a * glow)
                dst[ya:yb, xa:xb][mm] = dst[ya:yb, xa:xb][mm] * 0.45 + np.asarray(col, np.float32) * 0.55
        m = disc_mask(R)
        r0 = (m.shape[0] - 1) // 2
        yy = np.arange(m.shape[0])[:, None] / m.shape[0]
        top, bot = hexc('#fff3b8'), hexc('#ffa24a')
        spr = np.zeros(m.shape + (4,), np.uint8)
        col = top * (1 - yy[..., None]) + bot * yy[..., None]
        spr[..., :3] = np.broadcast_to(col, m.shape + (3,)).astype(np.uint8)
        spr[..., 3] = m * 255
        # полосы у нижнего края (ретро-закат)
        for k, frac in enumerate((0.62, 0.74, 0.84, 0.92)):
            row = int(frac * m.shape[0])
            spr[row:row + 1 + k // 2, :, 3] = 0
        blit(dst, spr, sx - r0, sy - r0)
        return sx, sy, R

    def sea(self, dst, cam, t, rays, covered, sun_screen, dusk):
        dx, dy, dz = rays
        with np.errstate(divide='ignore', invalid='ignore'):
            ts = (SEA_Y - cam.y) / dy
        m = (dy < -1e-4) & (ts > 0) & ~covered
        if not m.any():
            return m
        X = cam.x + ts * dx
        Z = cam.z + ts * dz
        dist = ts
        near, far = hexc('#2a1840'), hexc('#a84a72')
        k = np.clip(dist / 900.0, 0, 1) ** 0.55
        kq = np.round(k * 6) / 6
        base = near[None, None] * (1 - kq[..., None]) + far[None, None] * kq[..., None]
        base = base * (1 - dusk * 0.55)
        # волны: горизонтальные штрихи
        wave = np.sin(Z * 0.21 + np.sin(X * 0.035 + t * 0.6) * 2.2 + t * 1.3) + 0.6 * np.sin(Z * 0.53 - X * 0.02 + t * 2.1)
        hi = wave > 1.25
        base[hi] = base[hi] * 0.6 + hexc('#e07a7a') * 0.4 * (1 - dusk * 0.6)
        lo = wave < -1.3
        base[lo] = base[lo] * 0.8
        # солнечная дорожка
        if sun_screen is not None:
            sx, sy, R = sun_screen
            az = np.arctan2(X - cam.x, Z - cam.z)
            width = 0.02 + 0.06 * np.clip(1 - dist / 1400, 0, 1)
            path = np.abs(az - SUN_AZ) < width
            glit = path & (wave > 0.55 + 0.5 * np.sin(X * 0.7 + t * 5.0) * 0.5)
            g1 = hexc('#ffe08a') * (1 - dusk * 0.5)
            g2 = hexc('#ff9a5a') * (1 - dusk * 0.5)
            base[glit] = g1
            base[path & ~glit & (wave > 0.1)] = g2
        dst[m] = base[m]
        return m

    def ground_hit(self, cam, rays):
        dx, dy, dz = rays
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -cam.y / dy
        ok = (dy < -1e-5) & (t > 0) & (t < 3000)
        X = cam.x + t * dx
        Z = cam.z + t * dz
        beyond = ~ok | (Z > edge_z(X))
        return beyond, X, Z, t

    def cliff_face(self, dst, cam, rays, covered, lit):
        dx, dy, dz = rays
        with np.errstate(divide='ignore', invalid='ignore'):
            tc = (0.0 - cam.z) / dz
        Y = cam.y + tc * dy
        X = cam.x + tc * dx
        m = (tc > 0) & (Y < 0) & (Y > SEA_Y) & ~covered
        if not m.any():
            return m
        if lit:
            c_hi, c_mid, c_dk, c_ln = hexc('#d08a5a'), hexc('#a8664a'), hexc('#7a4440'), hexc('#4a2630')
            g0, g1 = hexc('#6a8a30'), hexc('#a8b848')
        else:
            c_hi, c_mid, c_dk, c_ln = hexc('#4a3040'), hexc('#3a2436'), hexc('#2a1a2c'), hexc('#160c1a')
            g0, g1 = hexc('#2a2038'), hexc('#4a3050')
        row = np.floor(-Y / 6.0)
        off = (row % 2) * 7.0
        blk = np.floor((X + off) / 14.0)
        hsh = np.sin(blk * 12.9898 + row * 78.233) * 43758.5453
        hsh = hsh - np.floor(hsh)
        col = np.where((hsh > 0.66)[..., None], c_hi, np.where((hsh > 0.25)[..., None], c_mid, c_dk))
        fy = (-Y / 6.0) - row
        fx = ((X + off) / 14.0) - blk
        # освещённый верх каждого блока, трещины
        col = np.where((fy < 0.16)[..., None], c_hi * 0.5 + col * 0.5, col)
        crack = (fy > 0.9) | (fx < 0.05)
        col = np.where(crack[..., None], c_ln, col)
        deep = np.clip((-Y - 25) / 35, 0, 1)
        col = np.where((DITHER < deep)[..., None], col * 0.72, col)
        # травяной козырёк с «висящими» пучками
        hang = 2.2 + (np.sin(X * 1.7) > 0.55) * 1.6 + (np.sin(X * 0.61 + 2) > 0.8) * 1.4
        grass = Y > -hang
        col = np.where(grass[..., None], np.where((Y > -1.0)[..., None], g1, g0), col)
        dst[m] = col[m]
        return m

    def birds(self, dst, t, x0, y0, n=5, speed=22, seed=1):
        rng = np.random.default_rng(seed)
        col = hexc('#2a1636')
        for i in range(n):
            ox, oy = rng.uniform(0, 26), rng.uniform(-6, 6)
            x = x0 + ox + t * speed
            y = y0 + oy + math.sin(t * 1.3 + i) * 2
            flap = int((t * 6 + i * 0.7) % 2)
            if flap == 0:
                pset(dst, x - 1, y - 1, col)
                pset(dst, x, y, col)
                pset(dst, x + 1, y - 1, col)
                pset(dst, x - 2, y - 2, col)
                pset(dst, x + 2, y - 2, col)
            else:
                pset(dst, x - 2, y, col)
                pset(dst, x - 1, y, col)
                pset(dst, x, y, col)
                pset(dst, x + 1, y, col)
                pset(dst, x + 2, y, col)

    def fireflies(self, dst, t, n=18, seed=4, alpha=1.0):
        rng = np.random.default_rng(seed)
        for i in range(n):
            bx, by = rng.uniform(0, W), rng.uniform(H * 0.45, H * 0.95)
            ph = rng.uniform(0, 6.28)
            x = bx + math.sin(t * 0.7 + ph) * 10
            y = by + math.sin(t * 0.9 + ph * 1.3) * 6 - t * 3
            y = (y - H * 0.4) % (H * 0.6) + H * 0.4
            b = 0.5 + 0.5 * math.sin(t * 3 + ph)
            if b * alpha > 0.35:
                pset(dst, x, y, (255, 230, 150))
                if b > 0.8:
                    for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        pset(dst, x + ddx, y + ddy, (255, 200, 120), 0.5)

    # -------------------------------------------------------------- персонажи
    def codex_sprite(self, width_px, kind, row=0, col=0):
        fn = {'back': face_back, 'heart': face_heart, 'happy': face_happy}.get(kind)
        return self.pix.get(row, col, width_px, face_fn=fn)

    def _codex_seated(self, cam, cox, Zs, kind, row, col, lean_codex, proc, crop=True, Y=0.0):
        p = cam.project(cox, Y, Zs)
        s = cam.f / p[2]
        spr = self.codex_sprite(24 * s, kind, row, col)
        ax = self.cx.anchor[1][0] / 48.0 * spr.shape[1]
        feet = int(round(self.cx.anchor[1][1] / 52.0 * spr.shape[0]))
        spr = spr[:int(spr.shape[0] * 0.86)] if crop else spr[:feet]
        if lean_codex:
            spr, pad = lean(spr, -lean_codex * s / 2.6, 0.9)
            ax += pad
        return proc(spr), ax, s

    def draw_pair_back(self, dst, cam, t, clx, cox, lean_codex=0.0, lean_clawd=0.0, hug=0.0, rim=1.0,
                       codex_col=0, codex_row=0, codex_on_top=False, codex_y=0.0):
        """оба сидят спиной к камере на краю (Z≈-5)."""
        Zs = -5.0
        for X, rx in ((clx, 12), (cox, 9)):
            p = cam.project(X, 0, Zs - 4)
            if p:
                sx, sy, zc = p
                sc = cam.f / zc
                m = ellipse_mask(rx * sc, max(1.0, 2.2 * sc * 0.5))
                blit_mask(dst, m, sx - (m.shape[1] - 1) / 2, sy - (m.shape[0] - 1) / 2 + 1, (30, 14, 40), 0.35)
        spr, ax, s = self._codex_seated(cam, cox, Zs, 'back', codex_row, codex_col, lean_codex,
                                        lambda im: backlit(im, rim_amt=rim))
        spr_c = self.clawd_back
        if lean_clawd:
            spr_c, _ = lean(spr_c, lean_clawd, 0.95)

        def d_codex():
            return proj_sprite(dst, cam, spr, cox, codex_y, Zs, s, (ax / spr.shape[1], 1.0))  # уже в экранном масштабе

        def d_clawd():
            return proj_sprite(dst, cam, spr_c, clx, 0, Zs, 1.0, (0.5, 1.0), proc=lambda im: backlit(im, rim_amt=rim))
        if codex_on_top:
            box = d_clawd()
            codex_box = d_codex()
        else:
            codex_box = d_codex()
            box = d_clawd()
        return box, codex_box

    def clawd_swing(self, kind, blush, t):
        """Clawd сидит на краю: лапки свисают и по очереди «болтаются» (укорачиваются на юнит — ракурс)."""
        spr = self.cl.face(kind, blush=blush, legs=True).copy()
        ph = int(t / 0.32) % 2
        legs = [(4, 5), (8, 9), (14, 15), (18, 19)]
        for i, (a_, b_) in enumerate(legs):
            if (i % 2) == ph:
                spr[15, a_:b_ + 1] = 0
        return spr

    def draw_pair_front(self, dst, cam, t, clx, cox, clawd_face, codex_face, lean_codex=0.0, lean_clawd=0.0,
                        hug=0.0, blush=True, codex_col=0, bob=0.0, codex_row=0, codex_on_top=False):
        """встречный план: сидят на краю лицом к камере, ноги свешиваются перед скалой.
        X зеркален (камера смотрит на -Z): Clawd слева при clx > cox."""
        Zs = 0.8
        feet_codex = -3.4
        spr, ax, s = self._codex_seated(cam, cox, Zs, codex_face, codex_row, codex_col, lean_codex, warmlit,
                                        crop=False, Y=feet_codex)
        spr_c = self.clawd_swing(clawd_face, blush, t)
        if lean_clawd:
            spr_c, _ = lean(spr_c, lean_clawd, 0.7)

        def d_codex():
            return proj_sprite(dst, cam, spr, cox, feet_codex + bob, Zs, s, (ax / spr.shape[1], 1.0))

        def d_clawd():
            return proj_sprite(dst, cam, spr_c, clx, -4.0, Zs, 1.0, (0.5, 1.0), proc=warmlit)
        if codex_on_top:
            box = d_clawd()
            codex_box = d_codex()
        else:
            codex_box = d_codex()
            box = d_clawd()
        return box, codex_box

    # -------------------------------------------------------------- шоты
    def shot_back(self, t, cam, dusk, sun_elev, clx, cox, lean_codex, lean_clawd, hug, hearts_t=None, stars=0.0,
                  birds_t=None, rim=1.0, codex_col=0, title=None, tree=True, codex_row=0, codex_on_top=False,
                  codex_y=0.0):
        c = canvas()
        rays = cam.rays()
        self.sky(c, cam, dusk, rays)
        if stars > 0:
            stars_field(c, t, n=140, seed=7, y_max=int(max(0, cam.horizon_y() - 10)), alpha=stars)
        self.clouds.draw(c, cam, drift=t * 3.0, y_offset=-26, tint=(1.0 - 0.45 * dusk, 0.9 - 0.4 * dusk, 1.0 - 0.3 * dusk))
        sun = self.sun(c, cam, sun_elev, glow=1 - dusk * 0.7)
        self.islands.draw(c, cam, y_offset=1, tint=(1 - dusk * 0.5, 1 - dusk * 0.55, 1 - dusk * 0.4))
        beyond, GX, GZ, gt = self.ground_hit(cam, rays)
        self.sea(c, cam, t, rays, ~beyond, sun, dusk)
        valid, _ = self.g_back.render(c, cam, fog=hexc('#5a2a5a'), fog_near=160, fog_far=900, mask_out=beyond)
        # светлая кромка у обрыва (контровой свет)
        rim_m = valid & (GZ > edge_z(GX) - 1.6)
        c[rim_m] = c[rim_m] * 0.4 + hexc('#ff9a6a') * 0.6 * (1 - dusk * 0.5)
        if dusk > 0:
            c *= (1 - dusk * 0.35)
        # дерево-силуэт слева
        if tree:
            proj_sprite(c, cam, self.tree_back, -96, 0, -24, 0.75, (0.5, 0.97), proc=lambda im: backlit(im, rim_amt=rim * 0.8))
        if birds_t is not None:
            self.birds(c, birds_t, 60, cam.horizon_y() - 46)
        self.draw_pair_back(c, cam, t, clx, cox, lean_codex, lean_clawd, hug, rim=rim, codex_col=codex_col,
                            codex_row=codex_row, codex_on_top=codex_on_top, codex_y=codex_y)
        if hearts_t is not None:
            p = cam.project((clx + cox) / 2, 26, -5)
            if p:
                hearts(c, hearts_t, p[0], p[1], n=7, seed=5, interval=0.45, life=2.6, rise=18, spread=8)
        vignette(c, 0.25)
        return c

    def render(self, t):
        # ---------------- шот 1: со спины, наезд; Codex подсаживается и обнимает
        if t < 5.6:
            k = ease_in_out(seg(t, 0, 5.6))
            cam = lerp_cam(look_at((13, 10.5, -76), (0, 24, 60), 250), look_at((-3, 9.5, -54), (-1, 23, 60), 258), k)
            cox = lerp(15.0, 5.5, ease_in_out(seg(t, 0.9, 2.1)))
            clx = -13.0
            lean_c = 3.0 * ease_in_out(seg(t, 2.2, 3.2))
            lean_cl = 1.4 * ease_in_out(seg(t, 3.3, 4.0))
            hug = seg(t, 3.5, 4.4)
            ht = t - 4.1 if t > 4.1 else None
            hopy = 1.3 * abs(math.sin((t - 0.9) * math.pi * 3.2)) if 0.9 < t < 2.1 else 0.0
            return self.shot_back(t, cam, 0.0, 0.055, clx, cox, lean_c, lean_cl, hug, hearts_t=ht, birds_t=t, codex_y=hopy)
        # ---------------- шот 2: встречный крупный
        if t < 11.6:
            u = t - 5.6
            k = ease_in_out(seg(u, 0, 6.0))
            cam = lerp_cam(look_at((-5, 12.5, 42), (2.5, 13.5, -5), 312), look_at((9, 13.2, 38), (2.5, 13.8, -5), 336), k)
            c = canvas()
            rays = cam.rays()
            self.sky(c, cam, 0.15, rays)
            stars_field(c, t, n=60, seed=9, y_max=int(max(0, cam.horizon_y() - 30)), alpha=0.55)
            # луна
            pm = cam.project(cam.x + 700, cam.y + 330, cam.z - 3000)
            if pm:
                disc(c, pm[0], pm[1], 7, (250, 236, 210))
                disc(c, pm[0] + 3, pm[1] - 2, 6, (212, 196, 206))
            self.land_far.draw(c, cam, y_offset=1, tint=(0.85, 0.8, 0.95))
            self.land_hills.draw(c, cam, y_offset=2, tint=(1.05, 0.95, 0.85))
            beyond, GX, GZ, gt = self.ground_hit(cam, rays)
            self.g_front.render(c, cam, fog=hexc('#b08a8a'), fog_near=120, fog_far=700, mask_out=beyond)
            self.cliff_face(c, cam, rays, ~beyond, lit=True)
            for (X, Z, spr) in sorted(self.front_trees, key=lambda a: a[1]):
                proj_sprite(c, cam, spr, X, 0, Z, 0.5, (0.5, 0.97), proc=warmlit)
            proj_sprite(c, cam, self.bush_front, 40, 0, -18, 0.6, (0.5, 0.97), proc=warmlit)
            # позы: смотрят друг на друга -> объятие -> сердечки
            clawd_face = 'idle'
            codex_face = None
            if 0.5 < u < 0.62:
                clawd_face = 'blink'
            if u > 0.7:
                codex_face = 'happy'
            if u > 0.95:
                clawd_face = 'happy'
            if u > 3.1:
                clawd_face = 'love'
                codex_face = 'heart'
            lean_c = 3.0 * ease_in_out(seg(u, 1.5, 2.3))
            lean_cl = 1.4 * ease_in_out(seg(u, 2.1, 2.7))
            hug = seg(u, 1.9, 2.7)
            cox = lerp(-12.5, -5.5, ease_in_out(seg(u, 1.0, 1.9)))
            clx = 11.0
            bob = 0.5 * abs(math.sin(u * 3.0)) if 2.7 < u < 3.6 else 0.0
            self.draw_pair_front(c, cam, t, clx, cox, clawd_face, codex_face, lean_c, lean_cl, hug, bob=bob)
            # сердечки
            p = cam.project(2.0, 21, -5)
            if p and u > 2.4:
                hearts(c, u - 2.4, p[0], p[1], n=9, seed=8, interval=0.33, life=2.4, rise=22, spread=16)
            if p and 3.1 < u < 3.6:
                big = scale_nn(HEART, 3)
                k2 = pop_scale(u - 3.1, 0.3)
                if k2 > 0.05:
                    bh = squash(big, k2, k2)
                    blit(c, bh, p[0] - bh.shape[1] / 2, p[1] - 18 - bh.shape[0] / 2)
            elif p and u >= 3.6:
                bh = scale_nn(HEART, 3)
                yy = p[1] - 18 - (u - 3.6) * 8
                blit_dither(c, bh, p[0] - bh.shape[1] / 2, yy - bh.shape[0] / 2, clamp(1 - (u - 3.6) / 1.2))
            self.fireflies(c, t, alpha=1.0)
            vignette(c, 0.25)
            return c
        # ---------------- шот 3: со спины, солнце садится, кран в небо, титр
        u = t - 11.6
        k = ease_in_out(seg(u, 0.3, 3.9))
        cam0 = look_at((-3, 10.5, -70), (0, 24, 60), 250)
        cam1 = look_at((-3, 26, -64), (0, 170, 60), 250)
        cam = lerp_cam(cam0, cam1, k)
        dusk = ease_in_out(seg(u, 0.0, 5.0)) * 0.85
        sun_elev = lerp(0.03, -0.09, ease_in_out(seg(u, 0.0, 4.2)))
        c = self.shot_back(t, cam, dusk, sun_elev, -13.0, 5.5, 3.0, 1.4, 1.0, hearts_t=(u + 0.6) if u < 3.0 else None,
                           stars=seg(u, 1.0, 4.5), rim=1 - dusk * 0.7, tree=False)
        # падающая звезда
        if 2.2 < u < 2.9:
            q = (u - 2.2) / 0.7
            x0, y0 = 340 - q * 120, 30 + q * 40
            for i in range(10):
                pset(c, x0 + i * 1.7, y0 - i * 0.6, (255, 250, 220), 1 - i / 10)
        # титр (гаснет к 5.6, освобождая небо под мини-титры)
        ta = 1 - clamp(seg(u, 5.0, 5.6))
        if u > 2.9 and ta > 0.02:
            a = u - 2.9
            s1 = text_sprite('CLAUDE', (217, 119, 87), scale=3, outline=(30, 12, 20), shadow=(30, 12, 20))
            s2 = text_sprite('♥', (255, 84, 132), scale=3, outline=(30, 12, 20), shadow=(30, 12, 20))
            s3 = text_sprite('CODEX', (96, 136, 250), scale=3, outline=(12, 16, 40), shadow=(12, 16, 40))
            gap = 10
            total = s1.shape[1] + s2.shape[1] + s3.shape[1] + gap * 2
            x = W / 2 - total / 2
            y = 104
            for i, sp in enumerate((s1, s2, s3)):
                pa = a - i * 0.25
                if pa > 0:
                    kk = pop_scale(pa, 0.3)
                    ss = squash(sp, kk, kk) if kk < 0.999 else sp
                    hb = 0
                    if i == 1 and pa > 0.3:
                        hb = -abs(math.sin(pa * 4.0)) * 3
                    px_, py_ = x + (sp.shape[1] - ss.shape[1]) / 2, y + (sp.shape[0] - ss.shape[0]) / 2 + hb
                    if ta > 0.999:
                        blit(c, ss, px_, py_)
                    else:
                        blit_dither(c, ss, px_, py_, ta)
                x += sp.shape[1] + gap
            if a > 1.0:
                end = text_sprite('КОНЕЦ', (255, 236, 210), font='big', scale=2, outline=(30, 12, 30))
                blit_dither(c, end, W / 2 - end.shape[1] / 2, 146, clamp((a - 1.0) / 0.35) * ta)
        # мини-титры: кто всё это сделал
        if u > 5.75:
            for sp, yy, d in ((CRED1, 98, 0.0), (CRED2, 114, 0.20), (CRED3, 142, 0.40),
                             (CRED4, 158, 0.54), (CRED5, 176, 0.68)):
                aa = clamp((u - 5.8 - d) / 0.55)
                if aa > 0.02:
                    blit_dither(c, sp, W / 2 - sp.shape[1] / 2, yy, aa)
        return c


SCENE = Sunset()
