"""Хелперы сцен s5_rain / s6_montage (агент C). Общие файлы движка не трогаем — только используем.

Здесь: кэш миров на диске, дождь с глубиной и всплесками, круги на лужах, Codex в hi-res с кастомными
правками спрайтшита (спина без лица, взгляд через плечо), лучи солнца, радуга «в небе» (привязана к камере),
лист-зонтик, терминал, мелкие пиксельные иконки.
"""
import hashlib
import math
import os
import pickle

import numpy as np
from scipy import ndimage

from px import *
import bg
from cam3d import *

CACHE_DIR = ROOT + '/scenes/_c_cache'


# ======================================================================= кэш
def _engine_hash():
    """хэш файлов движка И наших модулей-строителей: правка любого из них пересобирает кэш."""
    h = hashlib.md5()
    for fn in ('px.py', 'bg.py', 'cam3d.py', 'worlds.py', 'chars.py', 'scenes/_c_helpers.py', 'scenes/_c_desk.py',
               'scenes/_c_flight.py', 'scenes/s5_rain.py', 'scenes/s6_montage.py'):
        try:
            with open(f'{ROOT}/{fn}', 'rb') as f:
                h.update(f.read())
        except OSError:
            pass
    return h.hexdigest()[:10]


_EH = None


def cached(name, builder, version=1):
    """объект из дискового кэша (pickle); ключ учитывает хэш файлов движка, чтобы не залипнуть на старом."""
    global _EH
    if _EH is None:
        _EH = _engine_hash()
    path = f'{CACHE_DIR}/{name}_v{version}_{_EH}.pkl'
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        pass
    obj = builder()
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        # убрать устаревшие версии этого же объекта
        for fn in os.listdir(CACHE_DIR):
            if fn.startswith(f'{name}_v') and fn.endswith('.pkl') and not path.endswith(fn):
                try:
                    os.remove(os.path.join(CACHE_DIR, fn))
                except OSError:
                    pass
        tmp = f'{path}.{os.getpid()}.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        pass
    return obj


# ======================================================================= проекция
def project_pts(cam, X, Y, Z):
    """векторная проекция массивов точек -> (sx, sy, zc)"""
    cy_, sy_ = math.cos(cam.yaw), math.sin(cam.yaw)
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    dx, dy, dz = np.asarray(X, np.float64) - cam.x, np.asarray(Y, np.float64) - cam.y, np.asarray(Z, np.float64) - cam.z
    xr = dx * cy_ - dz * sy_
    zr = dx * sy_ + dz * cy_
    yc = dy * cp + zr * sp
    zc = -dy * sp + zr * cp
    zs = np.where(zc > 0.5, zc, np.nan)
    return cam.cx + cam.f * xr / zs, cam.cy - cam.f * yc / zs, zc


def cam_path(t, keys):
    """keys: [(time, eye(x,y,z), target(x,y,z), f), ...] — плавная камера по ключам (ease_in_out на отрезке)."""
    if t <= keys[0][0]:
        k = keys[0]
        return look_at(k[1], k[2], k[3])
    for a, b in zip(keys[:-1], keys[1:]):
        if t <= b[0]:
            u = ease_in_out(seg(t, a[0], b[0]))
            e = tuple(lerp(a[1][i], b[1][i], u) for i in range(3))
            g = tuple(lerp(a[2][i], b[2][i], u) for i in range(3))
            return look_at(e, g, lerp(a[3], b[3], u))
    k = keys[-1]
    return look_at(k[1], k[2], k[3])


def lerp3(a, b, u):
    return tuple(lerp(a[i], b[i], u) for i in range(3))


# ======================================================================= дождь
def rain(dst, t, n=240, speed=300.0, wind=0.18, color=(150, 165, 205), tail=None, length=6, seed=1,
         stop_lo=0, stop_hi=H + 8, stopmap=None, hit_frac=1.0, splash=True, splash_color=None, intensity=1.0,
         x_range=(-50, W + 50)):
    """Экранный дождь без состояния. Каждая капля падает до своего stop (случайная строка «земли» между stop_lo и
    stop_hi); stopmap (W,) — поверхность (лист, голова): доля hit_frac капель разбивается о неё.
    Рисуется сплошными цветами (без полупрозрачности)."""
    N = int(n * clamp(intensity, 0, 1))
    if N <= 0:
        return
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(x_range[0], x_range[1], n)[:N]
    ph = rng.uniform(0, H + 80, n)[:N]
    sp = rng.uniform(0.85, 1.15, n)[:N] * speed
    stop_lo = min(stop_lo, stop_hi - 12)
    stop = stop_lo + rng.random(n)[:N] * (stop_hi - stop_lo)
    hit = rng.random(n)[:N] < hit_frac
    span = H + 80
    y = (ph + t * sp) % span - 40
    if stopmap is not None:
        xs_at = np.clip((x0 + wind * stop).astype(int), 0, W - 1)
        sm = stopmap[xs_at]
        # уточняем колонку на высоте поверхности
        xs_at2 = np.clip((x0 + wind * np.minimum(sm, stop)).astype(int), 0, W - 1)
        sm = np.minimum(sm, stopmap[xs_at2])
        stop = np.where(hit & (sm < stop), sm, stop)
    col = np.asarray(color, np.float32)
    tcol = np.asarray(tail if tail is not None else color, np.float32)
    L = np.arange(length)
    ys = y[:, None] - L[None, :]
    xs = x0[:, None] + wind * ys
    ok = (ys < stop[:, None]) & (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    yi, xi = ys.astype(int), xs.astype(int)
    kk = np.broadcast_to(L[None, :], ys.shape)
    head = ok & (kk < max(1, length // 2))
    rest = ok & ~head
    dst[yi[rest], xi[rest]] = tcol
    dst[yi[head], xi[head]] = col
    if splash:
        sc = np.asarray(splash_color if splash_color is not None else color, np.float32)
        age = (y - stop) / sp
        m = (y >= stop) & (age < 0.11) & (stop < H + 2)
        if m.any():
            hx = (x0 + wind * stop)[m]
            hy = stop[m]
            a = age[m]
            pts = []
            e1 = a < 0.035
            e2 = (a >= 0.035) & (a < 0.075)
            e3 = a >= 0.075
            # кадр 1: корона из 2 точек; кадр 2: шире и выше; кадр 3: две капельки в стороны
            for (dx, dy, mm) in ((-1, -1, e1), (1, -1, e1), (0, 0, e1),
                                 (-2, -1, e2), (2, -1, e2), (-1, -2, e2), (1, -2, e2),
                                 (-3, -2, e3), (3, -2, e3)):
                if mm.any():
                    pts.append((hx[mm] + dx, hy[mm] + dy))
            for (px_, py_) in pts:
                xi2, yi2 = px_.astype(int), py_.astype(int)
                ok2 = (xi2 >= 0) & (xi2 < W) & (yi2 >= 0) & (yi2 < H)
                dst[yi2[ok2], xi2[ok2]] = sc


def ripples(dst, cam, t, puddles, color, color2=None, rate=5.0, life=0.55, rmax=2.6, seed=0, amount=1.0):
    """расходящиеся круги от капель на лужах. puddles: [(X, Z, R)] (мировые юниты)."""
    if amount <= 0:
        return
    col = np.asarray(color, np.float32)
    col2 = np.asarray(color2 if color2 is not None else color, np.float32)
    allx, ally, allc = [], [], []
    for pi, (PX, PZ, PR) in enumerate(puddles):
        slots = max(2, int(PR * PR / 10 * amount))
        period = slots / rate
        for k in range(slots):
            phase = ((pi * 7919 + k * 104729 + seed * 13) % 1000) / 1000.0 * period
            m = math.floor((t + phase) / period)
            age = (t + phase) - m * period
            if age > life:
                continue
            h1 = ((pi * 73856093) ^ (k * 19349663) ^ (m * 83492791) ^ seed) & 0xffffffff
            r1 = ((h1 * 2654435761) & 0xffffffff) / 4294967296.0
            r2 = ((h1 * 2246822519 + 3266489917) & 0xffffffff) / 4294967296.0
            rr = (PR - rmax * 0.8) * math.sqrt(r1)
            ang = r2 * 2 * math.pi
            cxw, czw = PX + math.cos(ang) * rr, PZ + math.sin(ang) * rr
            rad = 0.4 + rmax * ease_out(age / life)
            p = cam.project(cxw, 0.0, czw)
            if p is None:
                continue
            npts = int(clamp(rad * cam.f / p[2] * 5, 6, 48))
            th = np.linspace(0, 2 * math.pi, npts, endpoint=False)
            allx.append(cxw + np.cos(th) * rad)
            ally.append(czw + np.sin(th) * rad)
            allc.append(np.full(npts, age / life < 0.55))
    if not allx:
        return
    X = np.concatenate(allx)
    Z = np.concatenate(ally)
    C = np.concatenate(allc)
    sx, sy, zc = project_pts(cam, X, np.zeros_like(X), Z)
    ok = np.isfinite(sx) & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    xi, yi = sx[ok].astype(int), sy[ok].astype(int)
    c = C[ok]
    dst[yi[c], xi[c]] = col
    dst[yi[~c], xi[~c]] = col2


def stopmap_from_sprite(stopmap, spr, info, inset=0):
    """обновить карту поверхностей по верхнему краю нарисованного спрайта. info — результат draw_sprite_3d."""
    if info is None:
        return
    x0, y0, nw, nh, zc, s = info
    h, w = spr.shape[:2]
    a = spr[:, :, 3] > 0
    top = np.where(a.any(0), a.argmax(0), 10 ** 6).astype(np.float64)
    xs = np.arange(int(math.floor(x0)), int(math.ceil(x0 + nw)))
    xs = xs[(xs >= 0) & (xs < W)]
    if len(xs) == 0:
        return
    src = np.clip(((xs - x0) * w / nw).astype(int), 0, w - 1)
    ty = y0 + top[src] * nh / h + inset
    stopmap[xs] = np.minimum(stopmap[xs], ty)


# ======================================================================= Codex: hi-res правки кадра
def _lum(rgb):
    return rgb[..., 0] * 0.3 + rgb[..., 1] * 0.59 + rgb[..., 2] * 0.11


def _hr_screen_mask(cell):
    a = cell[:, :, 3] > 0.5
    lum = _lum(cell[:, :, :3])
    hh = int(cell.shape[0] * 0.64)
    er = ndimage.binary_erosion(a, iterations=5)
    dark = (lum < 0.30) & er
    dark[hh:] = False
    lab, n = ndimage.label(dark)
    if n == 0:
        return None
    sizes = ndimage.sum(dark, lab, range(1, n + 1))
    comp = lab == (int(np.argmax(sizes)) + 1)
    comp = ndimage.binary_fill_holes(comp)
    comp = ndimage.binary_dilation(comp, iterations=3) & a
    return comp


def _hr_remove_chest(cell):
    """убрать «>-» на груди (для вида со спины)."""
    f = cell.copy()
    a = f[:, :, 3] > 0.5
    lum = _lum(f[:, :, :3])
    y0 = int(f.shape[0] * 0.60)
    for y in range(y0, f.shape[0]):
        row_b = a[y] & (lum[y] > 0.62)
        if not row_b.any():
            continue
        for x in np.where(row_b)[0]:
            xx = x - 1
            while xx > 0 and a[y, xx] and lum[y, xx] > 0.62:
                xx -= 1
            f[y, x, :3] = f[y, xx, :3] if a[y, xx] else f[y, x, :3] * 0.6
    return f


def codex_nochest(cell):
    return _hr_remove_chest(cell)


def codex_back_hr(cell):
    """вид со спины: экран-лицо закрашен цветами головы (построчная интерполяция), «>-» на груди убран."""
    f = cell.copy()
    comp = _hr_screen_mask(f)
    if comp is not None:
        a = f[:, :, 3] > 0.5
        lum = _lum(f[:, :, :3])
        for y in np.where(comp.any(1))[0]:
            xs_ = np.where(comp[y])[0]
            xl, xr = xs_.min(), xs_.max()
            aa = xl - 1
            while aa > 0 and (not a[y, aa] or lum[y, aa] < 0.30 or comp[y, aa]):
                aa -= 1
            bb = xr + 1
            while bb < f.shape[1] - 1 and (not a[y, bb] or lum[y, bb] < 0.30 or comp[y, bb]):
                bb += 1
            ca = f[y, aa, :3] if a[y, aa] else f[y, bb, :3]
            cb = f[y, bb, :3] if a[y, bb] else ca
            tt = ((xs_ - xl + 0.5) / max(1, xr - xl + 1))[:, None]
            f[y, xs_, :3] = ca * (1 - tt) + cb * tt
        # сгладить построчные полосы внутри закрашенной области (вертикальный бокс-фильтр)
        sm = np.stack([ndimage.uniform_filter1d(f[:, :, ch], 13, axis=0) for ch in range(3)], -1)
        f[comp, :3] = sm[comp]
    return _hr_remove_chest(f)


def draw_codex(dst, cam, pix, codex, X, Z, row=0, col=0, Y=0.0, face=None, face_fn=None, flipx=False, fog=None,
               fog_near=200, fog_far=1400, tint=None, add=None, lean_px=0, sx=1.0, sy=1.0, ret_spr=False):
    """как cam3d.draw_codex_3d, но с face_fn (правка hi-res кадра до пикселизации). Возвращает info (+спрайт)."""
    p = cam.project(X, Y, Z)
    if p is None:
        return (None, None) if ret_spr else None
    zc = p[2]
    scale = cam.f / zc
    width_px = CODEX_CELL_W * scale
    if width_px < 6:
        return (None, None) if ret_spr else None
    if face is not None:
        z = 2 if width_px > 60 else 1
        spr = codex.custom_face(row, col, z, face)
        ax, ay = codex.anchor[z]
    else:
        spr = pix.get(row, col, width_px, face_fn)
        ax, ay = codex.anchor[1][0] * spr.shape[1] / 48.0, codex.anchor[1][1] * spr.shape[0] / 52.0
    src_ppu = spr.shape[1] / CODEX_CELL_W
    if sx != 1.0 or sy != 1.0:
        h0, w0 = spr.shape[:2]
        spr = squash(spr, sx, sy)
        ax, ay = ax * spr.shape[1] / w0, ay * spr.shape[0] / h0
        src_ppu = spr.shape[1] / (CODEX_CELL_W * sx)
    if lean_px:
        spr, pad = lean(spr, lean_px, 0.8)
        ax += pad
    anchor = (ax / spr.shape[1], ay / spr.shape[0])
    if flipx:
        anchor = (1 - anchor[0], anchor[1])
    info = draw_sprite_3d(dst, cam, spr, X, Y, Z, src_ppu, anchor=anchor, flipx=flipx, fog=fog, fog_near=fog_near,
                          fog_far=fog_far, tint=tint, add=add)
    if ret_spr:
        return info, (spr[:, ::-1] if flipx else spr)
    return info


def clawd_sprite(clawd, face='idle', anim=None, t=0.0, frame=None, blush=False, legs=True, tears=0, start=0,
                 end=None, loop=True):
    if anim is None:
        spr = clawd.face(face, blush, legs, tears)
        return spr, (spr.shape[1] / 2.0, spr.shape[0])
    from chars import anim_frame
    idx = frame if frame is not None else anim_frame(t, clawd.meta[anim]['durations'], loop=loop, start=start, end=end)
    return clawd.frames[anim][idx], clawd.anchor[anim]


def draw_spr(dst, cam, spr, anc, X, Y, Z, ppu=1.0, flipx=False, sx=1.0, sy=1.0, lean_src=0, tint=None, add=None,
             fog=None, fog_near=200, fog_far=1400, dx_px=0.0, dy_px=0.0, pivot=0.85, alpha=1.0, dither=None):
    """произвольный спрайт в мире с якорем anc (px исходника). squash sx/sy — одним ресэмплом прямо в экранный
    размер; lean_src — наклон в пикселях исходника (сдвиг целыми арт-пикселями); dx_px/dy_px — экранный сдвиг.
    Возвращает (info, img) где info = (x0, y0, nw, nh, zc, s)."""
    p = cam.project(X, Y, Z)
    if p is None:
        return None, None
    ax, ay = anc
    if lean_src:
        spr, pad = lean(spr, lean_src, pivot)
        ax += pad
    if flipx:
        spr = spr[:, ::-1]
        ax = spr.shape[1] - ax
    s = cam.f / p[2] / ppu
    h, w = spr.shape[:2]
    nw, nh = max(1, int(round(w * s * sx))), max(1, int(round(h * s * sy)))
    if nw > W * 3 or nh > H * 3:
        return None, None
    img = squash(spr, nw / w, nh / h) if (nw != w or nh != h) else spr
    x0 = p[0] - ax * nw / w + dx_px
    y0 = p[1] - ay * nh / h + dy_px
    if fog is not None and p[2] > fog_near:
        fr = clamp((p[2] - fog_near) / max(1, fog_far - fog_near))
        fr = round(fr * 5) / 5
        img = img.copy()
        img[:, :, :3] = (img[:, :, :3] * (1 - fr) + np.asarray(fog, np.float32) * fr).astype(np.uint8)
    if dither is not None and dither < 0.999:
        if tint is not None or add is not None:
            img = img.copy()
            rgb = img[:, :, :3].astype(np.float32)
            if tint is not None:
                rgb *= np.asarray(tint, np.float32)
            if add is not None:
                rgb += np.asarray(add, np.float32)
            img[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        bg.blit_dither(dst, img, x0, y0, dither)
    else:
        blit(dst, img, x0, y0, alpha=alpha, tint=tint, add=add)
    return (x0, y0, nw, nh, p[2], s), img


# ======================================================================= свет, радуга, лучи
_HY, _HX = [g.astype(np.float32) for g in np.mgrid[0:H:2, 0:W:2]]


def sunbeams(dst, sx, sy, t, amount=1.0, n=7, color=(255, 236, 170), spread=1.2, base_ang=math.pi / 2, seed=3,
             y_max=H, strength=0.18):
    """лучи от точки (sx, sy) веером; считаются в половинном разрешении (клетки 2x2 — «пиксельные» лучи),
    осветление фиксированной долей через дизер-маску."""
    if amount <= 0:
        return
    ang = np.arctan2(_HY - sy, _HX - sx)
    dist = np.hypot(_HX - sx, _HY - sy)
    rng = np.random.default_rng(seed)
    centers = base_ang + rng.uniform(-spread / 2, spread / 2, n)
    widths = rng.uniform(0.03, 0.08, n)
    val = np.zeros(_HX.shape, np.float32)
    for c, wdt, k in zip(centers, widths, range(n)):
        c2 = c + math.sin(t * 0.6 + k) * 0.02
        d = np.abs((ang - c2 + math.pi) % (2 * math.pi) - math.pi)
        np.maximum(val, np.clip(1 - d / wdt, 0, 1) * (0.6 + 0.4 * math.sin(t * 1.3 + k * 1.7)), out=val)
    val *= np.clip(1 - dist / (H * 1.3), 0, 1) * amount
    val = np.repeat(np.repeat(val, 2, 0), 2, 1)[:H, :W]
    if y_max < H:
        val[int(max(0, y_max)):] = 0
    # ядро луча — сплошное осветление (как color math на SNES), края — дизер
    core = val > 0.5
    edge = (val > 0.15) & ~core & (DITHER < (val - 0.15) / 0.35)
    m = core | edge
    col = np.asarray(color, np.float32)
    dst[m] = dst[m] * (1 + 0.5 * strength) + col * (0.12 * strength)


def sky_xy(cam, az, elev):
    """экранные координаты направления (азимут от +Z к +X, угол возвышения) — для солнца/радуги/молнии."""
    X = cam.x + math.sin(az) * 1000 * math.cos(elev)
    Z = cam.z + math.cos(az) * 1000 * math.cos(elev)
    Y = cam.y + math.sin(elev) * 1000
    p = cam.project(X, Y, Z)
    if p is None:
        return None
    return p[0], p[1]


def rainbow_sky(dst, cam, az, amount=1.0, radius_ang=0.62, band=None, clip_y=None, reveal=1.0):
    """радуга, привязанная к небу: центр на азимуте az чуть ниже горизонта; reveal — рост дуги слева направо."""
    c = sky_xy(cam, az, -0.10)
    if c is None or amount <= 0:
        return
    cx, cy = c
    r = cam.f * math.tan(radius_ang)
    b = band if band is not None else max(1, int(round(cam.f / 120)))
    tmp = dst.copy()
    bg.rainbow(tmp, cx, cy, r, alpha=1.0, band=b, clip_y=clip_y)
    ang = np.arctan2(-(YY - cy), (XX - cx))  # pi слева .. 0 справа
    grow = (math.pi - ang) / math.pi  # 0 слева -> 1 справа
    m = (grow <= reveal) & (DITHER < amount) & (np.abs(tmp - dst).sum(2) > 0)
    dst[m] = tmp[m]


# ======================================================================= иконки и мелочи
CHECK = from_ascii([
    '.......GG',
    '......GGG',
    '.....GGG.',
    'GG..GGG..',
    'GGGGGG...',
    '.GGGG....',
    '..GG.....',
], {'G': (110, 240, 120)})

DROP = from_ascii([
    '.W.',
    'WLB',
    'LBB',
    '.B.',
], {'W': (235, 250, 255), 'L': (150, 210, 255), 'B': (90, 150, 230)})

def make_leaf_canopy(w=40):
    """купол листа из bg.LEAF (без черешка) — для зонтика с собственным черешком."""
    return bg.LEAF[:bg.LEAF_CANOPY_H].copy()


LEAF_CANOPY = make_leaf_canopy()
STEM_D, STEM_M, STEM_L = hexc('#4a3020'), hexc('#5f4128'), hexc('#825c38')


def draw_stem(dst, x0, y0, x1, y1, thick=2, bow=0.085, grip=0.5):
    """черешок листа: слегка изогнутая дуга от хвата (x0,y0) к куполу (x1,y1).
    Книзу (у хвата) толще, слева светлая грань, справа — тень."""
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1.5:
        return
    nx, ny = -dy / L, dx / L               # перпендикуляр к хорде
    if nx * dx < 0:                        # выгибаем в сторону наклона — дуга, а не палка
        nx, ny = -nx, -ny
    mx, my = (x0 + x1) / 2 + nx * bow * L, (y0 + y1) / 2 + ny * bow * L
    n = max(6, int(L / 2.5))
    pts = []
    for i in range(n + 1):
        t = i / n
        a, b, c = (1 - t) ** 2, 2 * (1 - t) * t, t * t
        w = max(1, int(round(thick * (1.0 + grip * (1.0 - t) ** 2))))
        pts.append((a * x0 + b * mx + c * x1, a * y0 + b * my + c * y1, w))
    for i in range(n):
        xa, ya, wa = pts[i]
        xb, yb, _ = pts[i + 1]
        off = -((wa - 1) // 2)
        for j in range(wa):
            col = STEM_L if j == 0 else (STEM_D if (j == wa - 1 and wa > 2) else STEM_M)
            line(dst, xa + off + j, ya, xb + off + j, yb, col)


def leaf_drips(dst, spr, info, t, seed=0, n=4, speed=0.75,
               color=(200, 214, 244), tail=(136, 152, 196)):
    """капли, скатывающиеся с фестончатого края купола и падающие вниз."""
    if info is None:
        return
    x0, y0, nw, nh = info[:4]
    h, w = spr.shape[:2]
    a = spr[:, :, 3] > 0
    for i in range(n):
        col = int(((i + 0.5) / n * 0.74 + 0.13) * w)
        col = max(0, min(w - 1, col))
        ys = np.where(a[:, col])[0]
        if not len(ys):
            continue
        ex = x0 + (col + 0.5) * nw / w
        ey = y0 + (ys.max() + 1) * nh / h
        ph = (t * speed + i * 0.41 + seed * 0.17) % 1.0
        if ph > 0.82:
            continue
        if ph < 0.26:                       # набухает и висит на кончике ребра
            pset(dst, ex, ey, color)
            if ph > 0.16:
                pset(dst, ex, ey + 1, tail)
            continue
        d = (ph - 0.26) / 0.56
        fy = ey + 1 + d * d * 58.0
        ln = 2 + int(d * 5)
        line(dst, ex, fy - ln, ex, fy - 1, tail, alpha=0.8)
        pset(dst, ex, fy, color)


def terminal_sprite(lines, width_chars=18, cursor=True, title_col=(70, 74, 96), bg_col=(18, 20, 32),
                    border=(120, 128, 170), colors=None, font='big'):
    """окно терминала: заголовок с тремя точками, строки моноширинным 8px шрифтом."""
    cw, ch = 8, 10
    w = width_chars * cw + 10
    h = 12 + len(lines) * ch + 6
    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, :3] = bg_col
    img[:, :, 3] = 255
    img[:9, :, :3] = title_col
    for i, c in enumerate(((255, 96, 96), (255, 204, 80), (110, 220, 110))):
        img[3:6, 4 + i * 5:7 + i * 5, :3] = c
    # рамка
    img[0, :, :3] = border
    img[-1, :, :3] = border
    img[:, 0, :3] = border
    img[:, -1, :3] = border
    # скруглённые углы
    for (yy, xx) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        img[yy, xx, 3] = 0
    colors = colors or [(220, 226, 240)] * len(lines)
    for i, s in enumerate(lines):
        x = 5
        y = 12 + i * ch
        if not s:
            continue
        if s.startswith('\x01'):  # галочка
            blit_rgba(img, CHECK, x, y)
            x += CHECK.shape[1] + 3
            s = s[1:]
        if s:
            m = text_mask(s, font)
            hh, ww = m.shape
            sub = img[y:y + hh, x:x + ww]
            sub[m[:sub.shape[0], :sub.shape[1]], :3] = colors[i]
    return img


def blit_rgba(dst_img, spr, x, y):
    h, w = spr.shape[:2]
    H2, W2 = dst_img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W2), min(y + h, H2)
    if x0 >= x1 or y0 >= y1:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    m = s[:, :, 3] > 0
    dst_img[y0:y1, x0:x1][m] = s[m]


def grade_to(dst, mul, add=(0, 0, 0)):
    dst *= np.asarray(mul, np.float32)
    dst += np.asarray(add, np.float32)


# ======================================================================= виньетка с дизером (без колец)
_VIG = {}


def vignette_d(dst, strength=0.35, inner=0.62, soft=0.75):
    """затемнение краёв: уровни 1/8 с упорядоченным дизером между ними — без заметных колец."""
    key = (round(strength, 3), inner, soft)
    if key not in _VIG:
        yy = (YY - H / 2) / (H / 2)
        xx = (XX - W / 2) / (W / 2)
        d = np.sqrt(xx * xx * 0.75 + yy * yy)
        v = np.clip((d - inner) / soft, 0, 1) ** 1.3 * strength
        lv = np.floor(v * 8 + DITHER8) / 8
        _VIG[key] = (1 - np.clip(lv, 0, 1)).astype(np.float32)[:, :, None]
        if len(_VIG) > 64:
            _VIG.pop(next(iter(_VIG)))
    dst *= _VIG[key]


def wipe_mask(progress, direction=(1.0, 0.25), edge=0.22, block=2):
    """маска «фронта света»: progress 0..1, фронт идёт по направлению direction с дизер-кромкой."""
    dx, dy = direction
    n = math.hypot(dx, dy)
    dx, dy = dx / n, dy / n
    proj_ = (XX * dx + YY * dy)
    lo = min(0 * dx, W * dx) + min(0 * dy, H * dy)
    hi = max(0 * dx, W * dx) + max(0 * dy, H * dy)
    w = (proj_ - lo) / (hi - lo)
    thr = (progress * (1 + edge) - w) / edge
    d = BAYER8[(YY // block) % 8, (XX // block) % 8]
    return d < thr


def draw_leaf(dst, cam, canopy, base, top, ppu=1.0, tint=None, lean_src=0, stem_thick=None, sy=1.0):
    """лист: черешок от base (X,Y,Z) к top (X,Y,Z) + купол с низом в top.
    sy < 1 — купол приплюснут: растущий лопух смотрится листом, а не раскрытым зонтом."""
    pb = cam.project(*base)
    pt = cam.project(*top)
    if pb is not None and pt is not None:
        s_px = cam.f / pt[2]
        th = stem_thick if stem_thick is not None else max(1, int(round(s_px * 0.9)))
        draw_stem(dst, pb[0], pb[1], pt[0], pt[1], thick=th)
    return draw_spr(dst, cam, canopy, (canopy.shape[1] / 2, canopy.shape[0] - 1), top[0], top[1], top[2], ppu,
                    tint=tint, lean_src=lean_src, pivot=1.0, sy=sy)
