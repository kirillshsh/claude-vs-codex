"""Хелперы агента B (сцены s3_code, s4_race): быстрые рендеры неба/земли Mode 7, текстурированные 3D-квады,
3D-линии, частицы, HUD, штампы. Всё без состояния — только функции от аргументов (и времени t).
Общий движок (px, cam3d, bg, chars, worlds) НЕ меняем — только используем."""
import math
import numpy as np
from PIL import Image, ImageDraw
from px import *
from cam3d import Cam, look_at, lerp_cam, draw_sprite_3d, ground_shadow, mip_chain, CODEX_CELL_W


# ======================================================================= камера
def cam_basis(cam):
    """мировые векторы осей камеры: right, up, fwd (fwd — камерная глубина 1)."""
    cy_, sy_ = math.cos(cam.yaw), math.sin(cam.yaw)
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    right = np.array([cy_, 0.0, -sy_])
    up = np.array([sy_ * sp, cp, cy_ * sp])
    fwd = np.array([sy_ * cp, -sp, cy_ * cp])
    return right, up, fwd


def cam_pos(cam):
    return np.array([cam.x, cam.y, cam.z])


def shake_cam(cam, t, amount, seed=0):
    """тряска камеры (углы + чуть позиция). amount ~ 0..3"""
    if amount <= 0:
        return cam
    c = cam.copy()
    c.yaw += (math.sin(t * 57 + seed) * 0.6 + math.sin(t * 91 + seed * 2.1) * 0.4) * 0.006 * amount
    c.pitch += (math.sin(t * 63 + seed * 3.3) * 0.6 + math.sin(t * 101 + seed) * 0.4) * 0.006 * amount
    return c


def cam_shot(eye, target, f):
    return look_at(tuple(eye), tuple(target), f)


def lerp3(a, b, k):
    return tuple(a[i] + (b[i] - a[i]) * k for i in range(3))


def facing_flip(cam, X, Y, Z, dirx=1.0, dirz=0.0):
    """True, если «вперёд» (dirx, dirz) в мире смотрит на экране влево -> спрайт с видом «вправо» надо отразить."""
    p0 = cam.project(X, Y, Z)
    p1 = cam.project(X + dirx * 4.0, Y, Z + dirz * 4.0)
    if p0 is None or p1 is None:
        return False
    return p1[0] < p0[0] - 0.05


# ======================================================================= небо
def sky_fast(dst, cam, stops, span=0.75, below=None):
    """векторизованный аналог cam3d.sky_dome (тот же вид, ~15x быстрее)."""
    hy = cam.horizon_y()
    rows = np.arange(H)
    elev = np.arctan((hy - rows) / cam.f)
    S = np.asarray([hexc(s) if isinstance(s, str) else np.asarray(s, np.float32) for s in stops])
    n = len(S) - 1
    v = np.clip(1 - elev / span, 0, 1) * n
    k = np.minimum(n - 1, np.floor(v)).astype(int)
    fr = np.round((v - k) * 4) / 4
    m = DITHER < fr[:, None]
    dst[:] = np.where(m[:, :, None], S[k + 1][:, None, :], S[k][:, None, :])
    if below is not None:
        dst[int(max(0, math.ceil(hy))):] = below


def lerp_stops(a, b, k):
    return [np.asarray(x, np.float32) * (1 - k) + np.asarray(y, np.float32) * k for x, y in zip(a, b)]


def pano_draw(pano, dst, cam, drift=0.0, y_offset=0.0, tint=None, dither_alpha=1.0, add=None):
    """как Panorama.draw, но прозрачность — дизером (пиксельная), + tint/add."""
    k = cam.f / pano.f_ref
    hy = cam.horizon_y()
    img = pano.img
    pw = img.shape[1]
    az = cam.yaw * pano.parallax + np.arctan((np.arange(W) - cam.cx) / cam.f)
    col = (np.floor((az / (2 * math.pi)) * pw + drift) % pw).astype(int)
    h = img.shape[0]
    sy0 = int(math.floor(hy - pano.base_row * k + y_offset))
    sy1 = int(math.ceil(hy + (h - pano.base_row) * k + y_offset))
    ys = np.arange(max(0, sy0), min(H, sy1))
    if len(ys) == 0:
        return
    rows = np.floor((ys - (hy + y_offset)) / k + pano.base_row).astype(int)
    okr = (rows >= 0) & (rows < h)
    ys, rows = ys[okr], rows[okr]
    if len(ys) == 0:
        return
    patch = img[rows][:, col]
    m = patch[:, :, 3] > 0
    if dither_alpha < 0.999:
        m &= DITHER[ys] < dither_alpha
    if not m.any():
        return
    rgb = patch[:, :, :3].astype(np.float32)
    if tint is not None:
        rgb *= np.asarray(tint, np.float32)
    if add is not None:
        rgb += np.asarray(add, np.float32)
    sub = dst[ys]
    sub[m] = rgb[m]
    dst[ys] = sub


# ======================================================================= земля Mode 7 (быстрая)
def ground_rows(cam, y0=0.0, max_dist=4000.0):
    """геометрия пола для строк ниже горизонта: (r0, X, Z, t, dy) либо None. t — камерная глубина (по строке)."""
    hdist = cam.y - y0
    if hdist <= 0.05:
        return None
    hy = cam.horizon_y()
    r0 = max(0, int(math.floor(hy)) + 1)
    if r0 >= H:
        return None
    ys = np.arange(r0, H)
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    v = -(ys - cam.cy) / cam.f
    dy = v * cp - sp
    ok = dy < -1e-6
    t = np.full(len(ys), np.inf)
    t[ok] = -hdist / dy[ok]
    good = np.where(ok & (t < max_dist))[0]
    if len(good) == 0:
        return None
    a = good[0]
    r0 += a
    v, dy, t = v[a:], dy[a:], t[a:]
    zr = v * sp + cp
    cy_, sy_ = math.cos(cam.yaw), math.sin(cam.yaw)
    u = (np.arange(W) - cam.cx) / cam.f
    X = cam.x + t[:, None] * (u[None, :] * cy_ + (zr * sy_)[:, None])
    Z = cam.z + t[:, None] * (-u[None, :] * sy_ + (zr * cy_)[:, None])
    return r0, X.astype(np.float32), Z.astype(np.float32), t, dy


def apply_fog_rows(out, r0, t, fog, fog_near, fog_far, haze=0.35):
    fr = np.clip((t - fog_near) / max(1.0, fog_far - fog_near), 0, 1)
    fr = np.round(fr * 6) / 6
    if not (fr > 0).any():
        return out
    fm = DITHER[r0:r0 + len(t)] < fr[:, None]
    fogc = np.asarray(fog, np.float32)
    out[fm] = fogc
    out *= (1 - fr[:, None, None] * haze)
    out += fogc * (fr[:, None, None] * haze)
    return out


class FastGround:
    """Mode-7 земля: LOD по строкам (строки монотонны по глубине), без булевых масок — быстро.
    mips: список float32 (h, w, 3) (напр. world.ground.mips) или tex uint8."""

    def __init__(self, mips=None, tex=None, tpu=2.0, origin=(0.0, 0.0), levels=5):
        if mips is None:
            mips = mip_chain(tex[:, :, :3], levels)
        self.mips = [m.astype(np.float32) for m in mips]
        self.tpu, self.origin = tpu, origin

    def render(self, dst, cam, fog=None, fog_near=260.0, fog_far=1500.0, shade=None, lod_bias=1.0, y0=0.0,
               max_dist=4000.0, tint=None):
        g = ground_rows(cam, y0, max_dist)
        if g is None:
            return None
        r0, X, Z, t, dy = g
        R = len(t)
        foot = t * self.tpu / cam.f * lod_bias / np.maximum(np.abs(dy), 0.25) * 0.5
        lvl = np.clip(np.floor(np.log2(np.maximum(foot, 1e-6))), 0, len(self.mips) - 1).astype(int)
        out = np.empty((R, W, 3), np.float32)
        U = (X - self.origin[0]) * self.tpu
        V = (Z - self.origin[1]) * self.tpu
        i = 0
        while i < R:
            L = lvl[i]
            j = i + 1
            while j < R and lvl[j] == L:
                j += 1
            m = self.mips[L]
            hh, ww = m.shape[:2]
            sc = 1.0 / (2 ** L)
            uu = np.floor(U[i:j] * sc).astype(np.int64) % ww
            vv = np.floor(V[i:j] * sc).astype(np.int64) % hh
            out[i:j] = m[vv, uu]
            i = j
        if shade is not None:
            out = shade(out, X, Z, t, r0)
        if tint is not None:
            out *= np.asarray(tint, np.float32)
        if fog is not None:
            apply_fog_rows(out, r0, t, fog, fog_near, fog_far)
        dst[r0:r0 + R] = out
        return r0, X, Z, t


# ======================================================================= текстурированный квад в 3D
def quad_screen_bbox(cam, corners, pad=2):
    pts = []
    behind = False
    for P in corners:
        xc, yc, zc = cam.to_cam(*P)
        if zc <= 0.5:
            behind = True
            continue
        pts.append((cam.cx + cam.f * xc / zc, cam.cy - cam.f * yc / zc))
    if behind:
        if not pts:
            return None
        return 0, 0, W, H
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = int(max(0, math.floor(min(xs)) - pad)), int(min(W, math.ceil(max(xs)) + pad))
    y0, y1 = int(max(0, math.floor(min(ys)) - pad)), int(min(H, math.ceil(max(ys)) + pad))
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def draw_quad(dst, cam, O, U, V, color=None, tex=None, two_sided=False, dither_alpha=1.0, tint=None, add=None,
              alpha_dither_tex=True, return_mask=False):
    """Квад O + s*U + r*V (s,r ∈ [0,1)). tex: (h,w,3|4) uint8, s -> x текстуры, r -> y текстуры.
    Альфа текстуры <255 -> дизерная прозрачность. Возвращает True если что-то нарисовано."""
    O = np.asarray(O, np.float64)
    U = np.asarray(U, np.float64)
    V = np.asarray(V, np.float64)
    N = np.cross(U, V)
    C = cam_pos(cam)
    side = np.dot(C - O, N)
    if side <= 0 and not two_sided:
        return False
    bb = quad_screen_bbox(cam, [O, O + U, O + U + V, O + V])
    if bb is None:
        return False
    x0, y0, x1, y1 = bb
    right, up, fwd = cam_basis(cam)
    u = ((np.arange(x0, x1) - cam.cx) / cam.f)[None, :]
    v = (-(np.arange(y0, y1) - cam.cy) / cam.f)[:, None]
    Dx = right[0] * u + up[0] * v + fwd[0]
    Dy = right[1] * u + up[1] * v + fwd[1]
    Dz = right[2] * u + up[2] * v + fwd[2]
    den = Dx * N[0] + Dy * N[1] + Dz * N[2]
    num = np.dot(O - C, N)
    with np.errstate(divide='ignore', invalid='ignore'):
        tt = num / den
    Px = C[0] + tt * Dx - O[0]
    Py = C[1] + tt * Dy - O[1]
    Pz = C[2] + tt * Dz - O[2]
    uu = np.dot(U, U)
    vv_ = np.dot(V, V)
    s = (Px * U[0] + Py * U[1] + Pz * U[2]) / uu
    r = (Px * V[0] + Py * V[1] + Pz * V[2]) / vv_
    m = (tt > 0.3) & (s >= 0) & (s < 1) & (r >= 0) & (r < 1)
    if not m.any():
        return False
    sub = dst[y0:y1, x0:x1]
    if tex is None:
        if dither_alpha < 0.999:
            m &= DITHER[y0:y1, x0:x1] < dither_alpha
        col = np.asarray(color, np.float32)
        if tint is not None:
            col = col * np.asarray(tint, np.float32)
        sub[m] = col
        return m if return_mask else True
    th, tw = tex.shape[:2]
    ix = np.clip((s[m] * tw).astype(np.int64), 0, tw - 1)
    iy = np.clip((r[m] * th).astype(np.int64), 0, th - 1)
    texel = tex[iy, ix]
    rgb = texel[:, :3].astype(np.float32)
    if tint is not None:
        rgb *= np.asarray(tint, np.float32)
    if add is not None:
        rgb += np.asarray(add, np.float32)
    if tex.shape[2] == 4:
        a = texel[:, 3].astype(np.float32) / 255.0 * dither_alpha
        keep = a > DITHER[y0:y1, x0:x1][m] if alpha_dither_tex else a > 0.5
    else:
        keep = np.ones(len(rgb), bool)
        if dither_alpha < 0.999:
            keep = DITHER[y0:y1, x0:x1][m] < dither_alpha
    ys_, xs_ = np.where(m)
    sub[ys_[keep], xs_[keep]] = rgb[keep]
    if return_mask:
        mm = np.zeros_like(m)
        mm[ys_[keep], xs_[keep]] = True
        return mm
    return True


def draw_box(dst, cam, x0, x1, y0, y1, z0, z1, cols, edge=None, edge2=None, tex_front=None, tex_side=None):
    """Коробка в мире (оси выровнены). cols: dict top, front(-Z), back(+Z), left(-X), right(+X).
    edge — цвет неоновой кромки по верхнему периметру; edge2 — по вертикальным рёбрам."""
    # (O, U, V) с внешней нормалью (U x V наружу)
    faces = {
        'top': ((x0, y1, z1), (x1 - x0, 0, 0), (0, 0, z0 - z1)),
        'front': ((x0, y1, z0), (x1 - x0, 0, 0), (0, y0 - y1, 0)),
        'back': ((x1, y1, z1), (x0 - x1, 0, 0), (0, y0 - y1, 0)),
        'left': ((x0, y1, z1), (0, 0, z0 - z1), (0, y0 - y1, 0)),
        'right': ((x1, y1, z0), (0, 0, z1 - z0), (0, y0 - y1, 0)),
    }
    C = cam_pos(cam)
    for name in ('back', 'left', 'right', 'front', 'top'):
        O, U, V = faces[name]
        tex = None
        if name == 'front' and tex_front is not None:
            tex = tex_front
        if name in ('left', 'right') and tex_side is not None:
            tex = tex_side
        draw_quad(dst, cam, O, U, V, color=cols.get(name, (60, 60, 80)), tex=tex)
    if edge is not None:
        top = [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
        for i in range(4):
            a, b = top[i], top[(i + 1) % 4]
            # ребро видно, если видна хотя бы одна из прилегающих граней: для простоты — верх виден всегда (камера выше)
            line3d(dst, cam, a, b, edge)
    if edge2 is not None:
        for (x, z) in ((x0, z0), (x1, z0), (x0, z1), (x1, z1)):
            # вертикальное ребро рисуем, только если оно на «видимом силуэте» — упрощённо: ближние 3
            pass


def clip_seg_near(c0, c1, near=0.8):
    if c0[2] < near and c1[2] < near:
        return None
    if c0[2] < near:
        k = (near - c0[2]) / (c1[2] - c0[2])
        c0 = tuple(c0[i] + (c1[i] - c0[i]) * k for i in range(3))
    elif c1[2] < near:
        k = (near - c1[2]) / (c0[2] - c1[2])
        c1 = tuple(c1[i] + (c0[i] - c1[i]) * k for i in range(3))
    return c0, c1


def to_screen(cam, c):
    return cam.cx + cam.f * c[0] / c[2], cam.cy - cam.f * c[1] / c[2]


def line_fast(dst, x0, y0, x1, y1, color, alpha=1.0, dither=None):
    """отрезок (векторизованный, без Python-цикла по пикселям)."""
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    if n > 4000:
        return
    xs = np.round(np.linspace(x0, x1, n)).astype(int)
    ys = np.round(np.linspace(y0, y1, n)).astype(int)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys = xs[ok], ys[ok]
    if dither is not None:
        keep = DITHER[ys, xs] < dither
        xs, ys = xs[keep], ys[keep]
    if alpha >= 0.999:
        dst[ys, xs] = color
    else:
        dst[ys, xs] = dst[ys, xs] * (1 - alpha) + np.asarray(color, np.float32) * alpha


def line3d(dst, cam, P0, P1, color, alpha=1.0, thick=1, dither=None):
    c0 = cam.to_cam(*P0)
    c1 = cam.to_cam(*P1)
    cl = clip_seg_near(c0, c1)
    if cl is None:
        return
    a, b = to_screen(cam, cl[0]), to_screen(cam, cl[1])
    # грубая отсечка по экрану
    if max(a[0], b[0]) < -50 or min(a[0], b[0]) > W + 50 or max(a[1], b[1]) < -50 or min(a[1], b[1]) > H + 50:
        return
    a = (min(max(a[0], -2000), 2000), min(max(a[1], -2000), 2000))
    b = (min(max(b[0], -2000), 2000), min(max(b[1], -2000), 2000))
    line_fast(dst, a[0], a[1], b[0], b[1], color, alpha, dither)
    if thick > 1:
        line_fast(dst, a[0], a[1] + 1, b[0], b[1] + 1, color, alpha, dither)


# ======================================================================= полигоны в экране
def fill_poly(dst, pts, color, dither=None, alpha=1.0):
    """заливка многоугольника (экранные координаты) без сглаживания; dither — доля пикселей (0..1)."""
    if len(pts) < 3:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = int(max(0, math.floor(min(xs)))), int(min(W, math.ceil(max(xs)) + 1))
    y0, y1 = int(max(0, math.floor(min(ys)))), int(min(H, math.ceil(max(ys)) + 1))
    if x0 >= x1 or y0 >= y1:
        return
    im = Image.new('L', (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(im).polygon([(p[0] - x0, p[1] - y0) for p in pts], fill=255)
    m = np.array(im) > 0
    if dither is not None:
        m &= DITHER[y0:y1, x0:x1] < dither
    if not m.any():
        return
    sub = dst[y0:y1, x0:x1]
    if alpha >= 0.999:
        sub[m] = color
    else:
        sub[m] = sub[m] * (1 - alpha) + np.asarray(color, np.float32) * alpha


def convex_hull(pts):
    pts = sorted(set((round(p[0], 3), round(p[1], 3)) for p in pts))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def light_cone(dst, cam, apex, base_c, base_r, color, dither=0.2, n=18):
    """световой конус (прожектор) от apex к кругу на полу, дизером."""
    pts = []
    ca = cam.to_cam(*apex)
    for i in range(n):
        a = i / n * 2 * math.pi
        P = (base_c[0] + math.cos(a) * base_r, base_c[1], base_c[2] + math.sin(a) * base_r)
        c = cam.to_cam(*P)
        if c[2] > 0.8:
            pts.append(to_screen(cam, c))
    if ca[2] > 0.8:
        pts.append(to_screen(cam, ca))
    else:
        # вершина позади камеры: продлеваем образующие к краю экрана — берём точки на полпути
        for i in range(0, n, 3):
            a = i / n * 2 * math.pi
            P = (base_c[0] + math.cos(a) * base_r, base_c[1], base_c[2] + math.sin(a) * base_r)
            M = tuple(P[j] + (apex[j] - P[j]) * 0.9 for j in range(3))
            c = cam.to_cam(*M)
            if c[2] > 0.8:
                pts.append(to_screen(cam, c))
    if len(pts) < 3:
        return
    pts = [(min(max(x, -3000), 3000), min(max(y, -3000), 3000)) for x, y in pts]
    hull = convex_hull(pts)
    fill_poly(dst, hull, color, dither=dither)


# ======================================================================= спрайты в мире
def bb_sprite(dst, cam, spr, X, Y, Z, ppu, anchor=(0.5, 1.0), flipx=False, fog=None, fog_near=260, fog_far=1500,
              tint=None, add=None, alpha=1.0, dither_alpha=None, max_px=None):
    """draw_sprite_3d с ранней отсечкой по экрану и дизерной прозрачностью."""
    p = cam.project(X, Y, Z)
    if p is None:
        return None
    sx, sy, zc = p
    s = cam.f / zc / ppu
    h, w = spr.shape[:2]
    nw, nh = w * s, h * s
    if nw < 1 or nh < 1:
        return None
    if nw > W * 3 or nh > H * 3:
        return None
    x0 = sx - anchor[0] * nw
    y0 = sy - anchor[1] * nh
    if x0 > W or x0 + nw < 0 or y0 > H or y0 + nh < 0:
        return None
    if dither_alpha is not None and dither_alpha < 0.999:
        nwi, nhi = max(1, int(round(nw))), max(1, int(round(nh)))
        img = squash(spr, nwi / w, nhi / h) if (nwi != w or nhi != h) else spr
        if flipx:
            img = img[:, ::-1]
        if tint is not None or add is not None:
            img = img.copy()
            c = img[:, :, :3].astype(np.float32)
            if tint is not None:
                c *= np.asarray(tint, np.float32)
            if add is not None:
                c += np.asarray(add, np.float32)
            img[:, :, :3] = np.clip(c, 0, 255).astype(np.uint8)
        from bg import blit_dither
        blit_dither(dst, img, x0, y0, dither_alpha)
        return (x0, y0, nwi, nhi, zc, s)
    return draw_sprite_3d(dst, cam, spr, X, Y, Z, ppu, anchor=anchor, flipx=flipx, fog=fog, fog_near=fog_near,
                          fog_far=fog_far, tint=tint, add=add, alpha=alpha)


def codex_img(pix, codex, row, col, width_px, face=None, cap=96):
    """кадр Codex под экранную ширину width_px (кап — дальше увеличиваем ближайшим соседом, чтобы
    крупные планы оставались пиксельными). Возвращает (spr, ax, ay) — якорь в пикселях спрайта."""
    if face is not None:
        z = 2 if width_px > 60 else 1
        spr = codex.custom_face(row, col, z, face)
        ax, ay = codex.anchor[z]
        return spr, ax, ay
    wp = int(max(6, round(min(width_px, cap))))
    spr = pix.get(row, col, wp)
    ax = codex.anchor[1][0] * spr.shape[1] / 48.0
    ay = codex.anchor[1][1] * spr.shape[0] / 52.0
    return spr, ax, ay


def draw_codex_b(dst, cam, pix, codex, X, Y, Z, row=0, col=0, face=None, flipx=False, lean_px=0, sx=1.0, sy=1.0,
                 rot=0, cap=96, fog=None, fog_near=260, fog_far=1500, tint=None, add=None, dither_alpha=None,
                 y_lift_px=0.0):
    """Codex в мире с капом пикселизации. rot — поворот на 90°*rot (лежит). y_lift_px — сдвиг якоря вверх
    в пикселях атласа f4 (48x52), напр. для прыжковых кадров."""
    p = cam.project(X, Y, Z)
    if p is None:
        return None
    zc = p[2]
    scale = cam.f / zc
    width_px = CODEX_CELL_W * scale * sx
    if width_px < 5:
        return None
    spr, ax, ay = codex_img(pix, codex, row, col, width_px, face, cap)
    k = spr.shape[1] / 48.0
    ay = ay - y_lift_px * k
    if sx != 1.0 or sy != 1.0:
        h0, w0 = spr.shape[:2]
        spr = squash(spr, sx, sy)
        ax, ay = ax * spr.shape[1] / w0, ay * spr.shape[0] / h0
    if lean_px:
        spr, pad = lean(spr, lean_px * spr.shape[1] / 48.0, 0.8)
        ax += pad
    if rot % 4:
        # лежачие позы: крутим и ставим якорь в низ-центр bbox
        spr = np.ascontiguousarray(np.rot90(spr, rot % 4))
        ys, xs = np.where(spr[:, :, 3] > 0)
        ax = (xs.min() + xs.max() + 1) / 2.0
        ay = ys.max() + 1
    src_ppu = spr.shape[1] / (CODEX_CELL_W * (sx if not rot % 2 else 1.0)) if not rot % 2 else spr.shape[0] / CODEX_CELL_W
    anchor = (ax / spr.shape[1], ay / spr.shape[0])
    if flipx:
        anchor = (1 - anchor[0], anchor[1])
    return bb_sprite(dst, cam, spr, X, Y, Z, src_ppu, anchor=anchor, flipx=flipx, fog=fog, fog_near=fog_near,
                     fog_far=fog_far, tint=tint, add=add, dither_alpha=dither_alpha)


def clawd_img(clawd, anim=None, frame=0, face='idle', blush=False, legs=True, tears=0):
    if anim is None:
        spr = clawd.face(face, blush, legs, tears)
        return spr, spr.shape[1] / 2.0, spr.shape[0]
    spr = clawd.frames[anim][frame]
    ax, ay = clawd.anchor[anim]
    return spr, ax, ay


def draw_clawd_b(dst, cam, clawd, X, Y, Z, anim=None, frame=0, face='idle', flipx=False, lean_px=0, sx=1.0, sy=1.0,
                 rot=0, blush=False, tears=0, fog=None, fog_near=260, fog_far=1500, tint=None, add=None,
                 dither_alpha=None):
    spr, ax, ay = clawd_img(clawd, anim, frame, face, blush, True, tears)
    if sx != 1.0 or sy != 1.0:
        h0, w0 = spr.shape[:2]
        spr = squash(spr, sx, sy)
        ax, ay = ax * spr.shape[1] / w0, ay * spr.shape[0] / h0
    if lean_px:
        spr, pad = lean(spr, lean_px)
        ax += pad
    if rot % 4:
        spr = np.ascontiguousarray(np.rot90(spr, rot % 4))
        ys, xs = np.where(spr[:, :, 3] > 0)
        ax = (xs.min() + xs.max() + 1) / 2.0
        ay = ys.max() + 1
    anchor = (ax / spr.shape[1], ay / spr.shape[0])
    if flipx:
        anchor = (1 - anchor[0], anchor[1])
    return bb_sprite(dst, cam, spr, X, Y, Z, 1.0, anchor=anchor, flipx=flipx, fog=fog, fog_near=fog_near,
                     fog_far=fog_far, tint=tint, add=add, dither_alpha=dither_alpha)


def sort_draw(dst, cam, items):
    """items: (X, Y, Z, fn[, bias]) — рисуем от дальних к ближним по камерной глубине (+bias)."""
    keyed = []
    for it in items:
        X, Y, Z, fn = it[:4]
        bias = it[4] if len(it) > 4 else 0.0
        zc = cam.to_cam(X, Y, Z)[2] + bias
        keyed.append((zc, id(fn), fn))
    keyed.sort(key=lambda k: (-k[0], k[1]))
    for zc, _, fn in keyed:
        if zc > 0.3:
            fn(dst, cam)


# ======================================================================= частицы (без состояния)
def hash01(*keys):
    h = 2166136261
    for k in keys:
        h ^= (int(k) * 2654435761) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return (h & 0xFFFFFF) / float(0x1000000)


def world_to_screen(cam, P):
    c = cam.to_cam(*P)
    if c[2] <= 0.5:
        return None
    return cam.cx + cam.f * c[0] / c[2], cam.cy - cam.f * c[1] / c[2], c[2]


def dither_disc(dst, cx, cy, r, color, alpha=1.0):
    if r < 0.5:
        if 0 <= int(cx) < W and 0 <= int(cy) < H and DITHER[int(cy), int(cx)] < alpha:
            dst[int(cy), int(cx)] = color
        return
    m = disc_mask(r)
    rr = (m.shape[0] - 1) // 2
    x, y = int(round(cx - rr)), int(round(cy - rr))
    h, w = m.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    mm = m[y0 - y:y1 - y, x0 - x:x1 - x]
    if alpha < 0.999:
        mm = mm & (DITHER[y0:y1, x0:x1] < alpha)
    dst[y0:y1, x0:x1][mm] = color


def puff(dst, cx, cy, r, col, shade, alpha=1.0):
    """клуб пыли/дыма из двух тонов (тень снизу-справа), дизерная прозрачность"""
    dither_disc(dst, cx + r * 0.18, cy + r * 0.2, r, shade, alpha)
    dither_disc(dst, cx - r * 0.12, cy - r * 0.15, max(0.5, r * 0.78), col, alpha)


# ======================================================================= текст/HUD
def text_img(s, color, font='big', size=None, scale=1, outline=None, shadow=None):
    return text_sprite(s, color, font=font, size=size, scale=scale, outline=outline, shadow=shadow)


def blit_center(dst, spr, cx, cy, alpha=1.0):
    h, w = spr.shape[:2]
    blit(dst, spr, cx - w / 2, cy - h / 2, alpha=alpha)


def pop_text(dst, s, cx, cy, age, color, scale=3, outline=(20, 10, 30), shadow=(20, 10, 30), dur=0.22, over=2.2,
             font='big', slam=False):
    """текст с «поп»: slam=True — влетает из крупного масштаба (удар), иначе растёт из 0 с перелётом."""
    if age < 0:
        return None
    spr = text_sprite(s, color, font=font, scale=scale, outline=outline, shadow=shadow)
    if slam:
        if age < dur:
            k = 1.0 + (over - 1.0) * (1 - ease_in(age / dur))
        else:
            k = 1.0 + 0.08 * math.sin((age - dur) * 30) * math.exp(-(age - dur) * 12)
    else:
        k = pop_scale(age, dur)
    if abs(k - 1) > 0.01:
        spr = squash(spr, k, k)
    blit_center(dst, spr, cx, cy)
    return spr.shape


def darken(dst, k):
    dst *= k


def ring_mask(r, thick=2.0):
    rr = int(math.ceil(r)) + 1
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    d = np.sqrt(xx * xx + yy * yy)
    return (d <= r + 0.5) & (d > r - thick + 0.5)


def draw_ring(dst, cx, cy, r, color, thick=2.0, alpha=1.0, squash_y=1.0):
    m = ring_mask(r, thick)
    if squash_y != 1.0:
        n = max(1, int(round(m.shape[0] * squash_y)))
        ys = np.clip((np.arange(n) + 0.5) / squash_y, 0, m.shape[0] - 1).astype(int)
        m = m[ys]
    h, w = m.shape
    if alpha < 0.999:
        x, y = int(round(cx - w / 2)), int(round(cy - h / 2))
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x0 >= x1 or y0 >= y1:
            return
        mm = m[y0 - y:y1 - y, x0 - x:x1 - x] & (DITHER[y0:y1, x0:x1] < alpha)
        dst[y0:y1, x0:x1][mm] = color
    else:
        blit_mask(dst, m, cx - w / 2, cy - h / 2, color)


_VIG = {}


def vig_map(strength=0.4):
    """предрасчитанная ступенчатая виньетка (множитель H x W x 1)."""
    if strength not in _VIG:
        yy = (YY - H / 2) / (H / 2)
        xx = (XX - W / 2) / (W / 2)
        d = np.sqrt(xx * xx * 0.8 + yy * yy)
        v = np.clip((d - 0.75) / 0.6, 0, 1) * strength
        q = np.round(v * 6) / 6
        _VIG[strength] = (1 - q)[:, :, None].astype(np.float32)
    return _VIG[strength]


def steam_b(dst, t, x, y, seed=1, n=3, k=1.0, color=(245, 240, 250)):
    """клубы пара (гнев), масштаб k — экранных px на юнит*0.35"""
    for i in range(n):
        a = (t * 2.2 + i / n + seed * 0.37) % 1.0
        px_ = x + (i - (n - 1) / 2) * 6 * k + math.sin(a * 6 + i) * 2 * k
        py_ = y - a * 16 * k
        r = (1.2 + a * 2.6) * k
        dither_disc(dst, px_, py_, r, color, 1 - a)


# ======================================================================= карточка раунда (общая для s3/s4)
ORANGE_C = np.array([255, 146, 64], np.float32)
BLUE_C = np.array([80, 168, 255], np.float32)

BELL = scale_nn(from_ascii([
    '.....oo.....',
    '...ooYYoo...',
    '..oYYYYWYo..',
    '.oYYYYYYWYo.',
    '.oYYYYYYYYo.',
    'oSYYYYYYYYYo',
    'oSYYYYYYYYYo',
    'oSSYYYYYYYSo',
    'oooooooooooo',
    '.....oo.....',
    '....oBBo....',
    '.....oo.....',
], {'o': (70, 40, 10), 'Y': (250, 200, 70), 'W': (255, 250, 210), 'S': (200, 130, 30), 'B': (120, 80, 40)}), 4)


def round_card(c, t, title, word, word_col, word_ol, left_spr=None, right_spr=None, dings=(0.22, 0.56),
               band=(92, 188)):
    """полоса-«табло» с колоколом, «РАУНД N» (поп) и словом раунда (удар), портреты слева/справа."""
    by0, by1 = band
    rect(c, 0, by0, W, by1 - by0, (6, 4, 12), 0.85)
    split = W // 2
    rect(c, 0, by0, split, 2, ORANGE_C)
    rect(c, split, by0, W - split, 2, BLUE_C)
    rect(c, 0, by1 - 2, split, 2, ORANGE_C)
    rect(c, split, by1 - 2, W - split, 2, BLUE_C)
    shake = 0.0
    for d in dings:
        a = t - d
        if 0 <= a < 0.5:
            for q in range(3):
                r = (a - q * 0.07) * 260
                if r > 4:
                    draw_ring(c, W // 2, 50, r, (255, 230, 150), thick=2, alpha=max(0.0, 0.9 - a * 1.8))
        if 0 <= a < 0.25:
            shake = max(shake, 1 - a / 0.25)
    bx = W // 2 - BELL.shape[1] // 2 + int(round(math.sin(t * 90) * 3 * shake))
    blit(c, BELL, bx, 26)
    for i, d in enumerate(dings):
        a = t - d
        if 0 <= a < 0.45:
            side = -1 if i == 0 else 1
            pop_text(c, 'ДЗЫНЬ!', W // 2 + side * 110, 50 - a * 24, a, (255, 236, 150), scale=2, dur=0.12)
    # портреты (VS-экран)
    a = t - (dings[1] - 0.12)
    if a >= 0:
        k = ease_out_back(min(1.0, a / 0.3), 1.6)
        if left_spr is not None:
            h, w = left_spr.shape[:2]
            x = lerp(-w - 10, 18, k)
            blit(c, left_spr, x, (by0 + by1) / 2 - h / 2 + 2)
        if right_spr is not None:
            h, w = right_spr.shape[:2]
            x = lerp(W + 10, W - 18 - w, k)
            blit(c, right_spr, x, (by0 + by1) / 2 - h / 2 + 2)
    a1 = t - dings[0]
    if a1 >= 0:
        pop_text(c, title, W // 2, by0 + 26, a1, (255, 244, 220), scale=3, dur=0.2)
    a2 = t - dings[1]
    if a2 >= 0:
        ox = int(round(math.sin(t * 80) * 4 * max(0, 1 - a2 / 0.3))) if a2 > 0.1 else 0
        pop_text(c, word, W // 2 + ox, by0 + 66, a2, word_col, scale=5, dur=0.1, over=3.0, slam=True,
                 outline=word_ol, shadow=word_ol)
        if 0.1 <= a2 < 0.18:
            from bg import flash
            flash(c, 0.6 * (1 - (a2 - 0.1) / 0.08))


_VIGD = {}


def vig_dither(strength=0.25):
    """виньетка с дизерными переходами (без видимых колец на светлом небе)."""
    if strength not in _VIGD:
        yy = (YY - H / 2) / (H / 2)
        xx = (XX - W / 2) / (W / 2)
        d = np.sqrt(xx * xx * 0.8 + yy * yy)
        v = np.clip((d - 0.8) / 0.55, 0, 1) * strength
        q = np.floor(v * 10 + DITHER8) / 10
        _VIGD[strength] = (1 - q)[:, :, None].astype(np.float32)
    return _VIGD[strength]


def focus_lines(dst, t, n=40, seed=4, col=(255, 255, 255), inner=0.62):
    """манга-линии фокуса по краям кадра (меняются каждые 2 кадра)."""
    fr = int(t * 15)
    cx, cy = W / 2, H / 2
    for i in range(n):
        h1, h2, h3 = hash01(i, seed, fr), hash01(i, seed + 1, fr), hash01(i, seed + 2)
        ang = (i + h1 * 0.8) / n * 2 * math.pi
        r0 = (inner + h2 * 0.18) * math.hypot(cx, cy)
        r1 = math.hypot(cx, cy) * 1.05
        x0, y0 = cx + math.cos(ang) * r0, cy + math.sin(ang) * r0 * 0.62
        x1, y1 = cx + math.cos(ang) * r1, cy + math.sin(ang) * r1 * 0.62
        line_fast(dst, x0, y0, x1, y1, col, dither=0.35 + 0.3 * h3)


def shadow_b(dst, cam, X, Z, rx_units, strength=0.35, Y=0.0):
    """тень-эллипс на земле: пиксели затемняются (умножение), без полупрозрачного смешивания."""
    p = cam.project(X, Y, Z)
    if p is None:
        return
    sx, sy, zc = p
    s = cam.f / zc
    rx = rx_units * s
    if rx < 1:
        return
    q = cam.project(X, Y, Z + rx_units)
    ry = abs(q[1] - sy) if q is not None else rx * 0.3
    ry = max(1.0, min(rx * 0.5, ry))
    m = ellipse_mask(rx, ry)
    h, w = m.shape
    x0, y0 = int(round(sx - (w - 1) / 2)), int(round(sy - (h - 1) / 2))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return
    mm = m[ya - y0:yb - y0, xa - x0:xb - x0]
    dst[ya:yb, xa:xb][mm] *= (1.0 - strength)
