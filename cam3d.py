"""2.5D-движок в духе SNES Mode 7 / HD-2D: перспективная камера, текстурированная земля, панорама неба и гор,
спрайты-«билборды» в 3D с сортировкой по глубине, туман, тени.

Мир: X вправо, Y вверх, Z вглубь. Единица мира = 1 «юнит» Clawd (Clawd = 24 x 16 юнитов, Codex ≈ 18 x 21.5).
Земля — плоскость Y=0. Текстура земли: 2 тексела на юнит.

Камера: Cam(x, y, z, yaw, pitch, f). yaw=0 смотрит вдоль +Z, yaw>0 — поворот вправо (к +X);
pitch>0 — наклон вниз. f — фокус в пикселях (больше f = уже угол, «теле»).
Удобно ставить камеру через look_at(eye, target, f).
"""
import math
import numpy as np
from scipy import ndimage
from px import *

# ======================================================================= камера


class Cam:
    def __init__(self, x=0.0, y=20.0, z=-120.0, yaw=0.0, pitch=0.1, f=260.0, cx=W / 2, cy=H / 2):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.yaw, self.pitch, self.f = float(yaw), float(pitch), float(f)
        self.cx, self.cy = cx, cy

    def copy(self):
        return Cam(self.x, self.y, self.z, self.yaw, self.pitch, self.f, self.cx, self.cy)

    def to_cam(self, X, Y, Z):
        cy_, sy_ = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        dx, dy, dz = X - self.x, Y - self.y, Z - self.z
        xr = dx * cy_ - dz * sy_
        zr = dx * sy_ + dz * cy_
        yc = dy * cp + zr * sp
        zc = -dy * sp + zr * cp
        return xr, yc, zc

    def project(self, X, Y, Z):
        xc, yc, zc = self.to_cam(X, Y, Z)
        if zc <= 0.5:
            return None
        return self.cx + self.f * xc / zc, self.cy - self.f * yc / zc, zc

    def horizon_y(self):
        return self.cy - self.f * math.tan(self.pitch)

    def rays(self):
        """направления лучей для всех пикселей (мировые, не нормированы: камерная глубина = 1)."""
        u = (XX - self.cx) / self.f
        v = -(YY - self.cy) / self.f
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        dy = v * cp - sp
        zr = v * sp + cp
        cy_, sy_ = math.cos(self.yaw), math.sin(self.yaw)
        dx = u * cy_ + zr * sy_
        dz = -u * sy_ + zr * cy_
        return dx, dy, dz


def look_at(eye, target, f=260.0):
    ex, ey, ez = eye
    tx, ty, tz = target
    dx, dy, dz = tx - ex, ty - ey, tz - ez
    yaw = math.atan2(dx, dz)
    pitch = -math.atan2(dy, math.hypot(dx, dz))
    return Cam(ex, ey, ez, yaw, pitch, f)


def lerp_cam(a, b, t):
    """интерполяция двух камер (углы — по кратчайшей)."""
    def la(x, y):
        d = (y - x + math.pi) % (2 * math.pi) - math.pi
        return x + d * t
    return Cam(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t), la(a.yaw, b.yaw), la(a.pitch, b.pitch),
               lerp(a.f, b.f, t))


def orbit(center, radius, angle, height, f=260.0, look_height=None):
    """камера на орбите вокруг точки center=(X,Z); angle=0 — камера на -Z стороне (смотрит к +Z)."""
    cxw, czw = center
    ex = cxw - math.sin(angle) * radius
    ez = czw - math.cos(angle) * radius
    lh = height * 0.25 if look_height is None else look_height
    return look_at((ex, height, ez), (cxw, lh, czw), f)


def shake(cam, t, amount=1.0, seed=0):
    c = cam.copy()
    c.yaw += math.sin(t * 53 + seed) * 0.004 * amount
    c.pitch += math.sin(t * 61 + seed * 3) * 0.004 * amount
    return c


# ======================================================================= текстуры земли
def mip_chain(tex, levels=4):
    out = [tex.astype(np.float32)]
    for i in range(levels - 1):
        t = out[-1]
        h, w = t.shape[0] // 2 * 2, t.shape[1] // 2 * 2
        t = t[:h, :w]
        t = (t[0::2, 0::2] + t[1::2, 0::2] + t[0::2, 1::2] + t[1::2, 1::2]) / 4
        out.append(t)
    return out


class Ground:
    """плоскость Y=y0 с текстурой. tex: (h, w, 3|4) uint8; texel = 1/tpu юнита; origin — мировые (X,Z) тексела (0,0).
    wrap=True — тайлинг; иначе вне текстуры — border (или прозрачно при alpha)."""

    def __init__(self, tex, tpu=2.0, origin=(0.0, 0.0), wrap=True, border=(80, 140, 70), y0=0.0, levels=4):
        self.alpha = tex.shape[2] == 4
        self.rgb = tex[:, :, :3]
        self.a = tex[:, :, 3] > 0 if self.alpha else None
        self.mips = mip_chain(self.rgb, levels)
        self.tpu, self.origin, self.wrap, self.border, self.y0 = tpu, origin, wrap, np.asarray(border, np.float32), y0
        self.h, self.w = self.rgb.shape[:2]

    def render(self, dst, cam, fog=None, fog_near=200.0, fog_far=1400.0, max_dist=4000.0, mask_out=None,
               lod_bias=1.0, shade=None):
        dx, dy, dz = cam.rays()
        hdist = cam.y - self.y0
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -hdist / dy
        valid = (dy < -1e-5) & (t > 0) & (t < max_dist) if hdist > 0 else (dy > 1e-5) & (t > 0) & (t < max_dist)
        if mask_out is not None:
            valid &= ~mask_out
        if not valid.any():
            return valid, t
        X = cam.x + t * dx
        Z = cam.z + t * dz
        u = (X - self.origin[0]) * self.tpu
        v = (Z - self.origin[1]) * self.tpu
        # LOD: сколько текселей на пиксель
        foot = t * self.tpu / cam.f * lod_bias / np.maximum(np.abs(dy) * 1.0, 0.25) * 0.5
        lvl = np.clip(np.floor(np.log2(np.maximum(foot, 1e-6))), 0, len(self.mips) - 1).astype(int)
        lvl[~valid] = 0
        out = np.zeros((H, W, 3), np.float32)
        inside = np.ones((H, W), bool)
        for L, m in enumerate(self.mips):
            sel = valid & (lvl == L)
            if not sel.any():
                continue
            uu = np.floor(u[sel] / (2 ** L)).astype(np.int64)
            vv = np.floor(v[sel] / (2 ** L)).astype(np.int64)
            hh, ww = m.shape[:2]
            if self.wrap:
                uu %= ww
                vv %= hh
                out[sel] = m[vv, uu]
            else:
                ins = (uu >= 0) & (uu < ww) & (vv >= 0) & (vv < hh)
                vals = np.empty((len(uu), 3), np.float32)
                vals[:] = self.border
                vals[ins] = m[vv[ins], uu[ins]]
                out[sel] = vals
                if self.alpha:
                    uu0 = np.clip(uu * (2 ** L), 0, self.w - 1)
                    vv0 = np.clip(vv * (2 ** L), 0, self.h - 1)
                    a = np.zeros(len(uu), bool)
                    a[ins] = self.a[vv0[ins], uu0[ins]]
                    tmp = inside[sel]
                    tmp[:] = a
                    inside[sel] = tmp
        if self.alpha:
            valid &= inside
        if shade is not None:
            out = shade(out, X, Z, t)
        if fog is not None:
            fr = np.clip((t - fog_near) / max(1.0, fog_far - fog_near), 0, 1)
            fr = np.round(fr * 6) / 6
            fm = DITHER < fr
            out[fm] = fog
            # мягкая дымка поверх
            out = out * (1 - fr[:, :, None] * 0.35) + np.asarray(fog, np.float32) * fr[:, :, None] * 0.35
        dst[valid] = out[valid]
        return valid, t


# ======================================================================= небо и панорама
def sky_dome(dst, cam, stops, span=0.9, below=None):
    """градиент по углу возвышения (не зависит от yaw) с дизером."""
    hy = cam.horizon_y()
    rows = np.arange(H)
    elev = np.arctan((hy - rows) / cam.f)  # приблизительный угол над горизонтом
    v = np.clip(1 - elev / span, 0, 1) * (len(stops) - 1)  # 0 = зенит
    n = len(stops) - 1
    stops = [np.asarray(s, np.float32) for s in stops]
    for y in range(H):
        fv = v[y]
        k = int(min(n - 1, math.floor(fv)))
        fr = round((fv - k) * 4) / 4
        row = DITHER[y] < fr
        dst[y] = np.where(row[:, None], stops[k + 1], stops[k])
    if below is not None:
        dst[int(max(0, math.ceil(hy))):] = below


class Panorama:
    """цилиндрическая лента (горы/облака/город): img (h, w, 4), base_row — строка, лежащая на горизонте.
    Ширина ленты = 2π * f_ref пикселей (полный круг), высоты — в пикселях при f_ref."""

    def __init__(self, img, f_ref=260.0, base_row=None, parallax=1.0):
        self.img = img
        self.f_ref = f_ref
        self.base_row = img.shape[0] - 1 if base_row is None else base_row
        self.parallax = parallax

    def draw(self, dst, cam, drift=0.0, y_offset=0.0, tint=None, alpha=1.0):
        k = cam.f / self.f_ref
        hy = cam.horizon_y()
        pw = self.img.shape[1]
        # колонка ленты для каждого экранного x
        az = cam.yaw * self.parallax + np.arctan((np.arange(W) - cam.cx) / cam.f)
        col = (np.floor((az / (2 * math.pi)) * pw + drift) % pw).astype(int)
        h = self.img.shape[0]
        # строки ленты для экранных y
        sy0 = int(math.floor(hy - self.base_row * k + y_offset))
        sy1 = int(math.ceil(hy + (h - self.base_row) * k + y_offset))
        ys = np.arange(max(0, sy0), min(H, sy1))
        if len(ys) == 0:
            return
        rows = np.floor((ys - (hy + y_offset)) / k + self.base_row).astype(int)
        okr = (rows >= 0) & (rows < h)
        ys, rows = ys[okr], rows[okr]
        if len(ys) == 0:
            return
        patch = self.img[rows][:, col]  # (len(ys), W, 4)
        a = patch[:, :, 3:4].astype(np.float32) / 255 * alpha
        rgb = patch[:, :, :3].astype(np.float32)
        if tint is not None:
            rgb *= np.asarray(tint, np.float32)
        dst[ys] = dst[ys] * (1 - a) + rgb * a


def make_mountain_pano(pal, f_ref=260.0, seed=3, height=80, snow=True, layers=2, haze=None):
    pw = int(2 * math.pi * f_ref)
    img = np.zeros((height, pw, 4), np.uint8)
    rr = np.random.default_rng(seed)
    xs = np.arange(pw)
    haze = pal['mount'][2] if haze is None else haze
    for li in range(layers):
        far = li == 0
        base = height - (6 if far else 0)
        hmin, hmax = (height * 0.35, height * 0.85) if far else (height * 0.2, height * 0.55)
        step = 90 if far else 120
        peaks = []
        x = 0.0
        while x < pw:
            peaks.append((x, base - rr.uniform(hmin, hmax), rr.uniform(0.5, 0.95)))
            x += rr.uniform(step * 0.6, step * 1.25)
        # периодичность: дублируем пики по краям
        ext = [(px - pw, py, sl) for (px, py, sl) in peaks] + peaks + [(px + pw, py, sl) for (px, py, sl) in peaks]
        tops = np.stack([py + np.abs(xs - px) * sl for (px, py, sl) in ext])
        idx = tops.argmin(0)
        top = tops.min(0).astype(int)
        lit_c = pal['mount'][1] * (0.55 if far else 1) + (pal['sky'][-2] * 0.45 if far else 0)
        sh_c = pal['mount'][0] * (0.55 if far else 1) + (pal['sky'][-2] * 0.45 if far else 0)
        sn = pal['snow'] * (0.7 if far else 1) + (pal['sky'][-2] * 0.3 if far else 0)
        for x in range(pw):
            px_, py_, sl = ext[idx[x]]
            zig = ((x * 7) % 5 - 2) // 2
            for y in range(max(0, top[x]), height):
                ridge_x = px_ + (y - py_) * 0.18 + zig
                c = lit_c if x < ridge_x else sh_c
                if snow and py_ < height * (0.45 if far else 0.62):
                    sd = 6 + ((x * 3 + 1) % 4)
                    if y < py_ + sd + abs(x - px_) * 0.25:
                        c = sn if x < ridge_x else sn * 0.86
                fr = (y - (height - 18)) / 20
                if fr > 0 and DITHER[y % H, x % W] < fr:
                    c = haze
                img[y, x, :3] = np.clip(c, 0, 255)
                img[y, x, 3] = 255
    return Panorama(img, f_ref, base_row=height - 1)


def make_hill_pano(colors, f_ref=260.0, seed=5, height=40, amp=18, trees=None, tree_pal=None, trunk_pal=None,
                   density=1.0):
    """низкие холмы с деревьями на гребне (дальний план)."""
    from bg import smooth_noise, make_round_tree, make_pine
    pw = int(2 * math.pi * f_ref)
    img = np.zeros((height, pw, 4), np.uint8)
    prof = smooth_noise(pw, 70, seed, octaves=3, persistence=0.45, period=pw)
    top = (height - 4 - prof * amp).astype(int)
    for x in range(pw):
        img[top[x]:, x, :3] = colors[0]
        img[top[x]:, x, 3] = 255
        img[top[x], x, :3] = colors[1]
    if tree_pal is not None:
        rng = np.random.default_rng(seed + 1)
        x = 0
        while x < pw - 20:
            if rng.random() < 0.6:
                tr = make_round_tree(int(rng.integers(1e6)), int(rng.integers(3, 6)), 3, tree_pal, trunk_pal)
            else:
                tr = make_pine(int(rng.integers(1e6)), int(rng.integers(9, 15)), tree_pal, trunk_pal)
            th, tw = tr.shape[:2]
            ty = top[min(pw - 1, x + tw // 2)] - th + 2
            m = tr[:, :, 3] > 0
            ys, xs_ = np.where(m)
            Y, X = ty + ys, x + xs_
            ok = (Y >= 0) & (Y < height) & (X < pw)
            img[Y[ok], X[ok]] = tr[ys[ok], xs_[ok]]
            x += int(tw * rng.uniform(0.7, 3.0) / density) + int(rng.integers(0, 24))
    return Panorama(img, f_ref, base_row=height - 1)


def make_cloud_pano(pal, f_ref=260.0, seed=8, height=120, n=14):
    from bg import make_cloud
    pw = int(2 * math.pi * f_ref)
    img = np.zeros((height, pw, 4), np.uint8)
    rng = np.random.default_rng(seed)
    for i in range(n):
        w_ = int(rng.integers(40, 90))
        h_ = int(rng.integers(14, 26))
        c = make_cloud(int(rng.integers(1e6)), w_, h_, pal['cloud'])
        x = int(rng.integers(0, pw - w_))
        y = int(rng.integers(0, height - h_ - 30))
        m = c[:, :, 3] > 0
        sub = img[y:y + h_, x:x + w_]
        sub[m] = c[m]
    return Panorama(img, f_ref, base_row=height - 1)


# ======================================================================= билборды
class Billboard:
    """элемент сцены для сортировки по глубине."""

    def __init__(self, X, Z, Y=0.0, kind='sprite', **kw):
        self.X, self.Y, self.Z = X, Y, Z
        self.kind = kind
        self.kw = kw


def draw_sprite_3d(dst, cam, spr, X, Y, Z, px_per_unit_src, anchor=(0.5, 1.0), flipx=False, fog=None,
                   fog_near=200.0, fog_far=1400.0, tint=None, add=None, alpha=1.0, min_scale=0.02):
    """спрайт стоит в точке (X,Y,Z) мира; px_per_unit_src — пикселей исходника на юнит мира.
    масштабирование — ближайшим соседом."""
    p = cam.project(X, Y, Z)
    if p is None:
        return None
    sx, sy, zc = p
    s = cam.f / zc / px_per_unit_src  # экранных px на пиксель исходника
    if s < min_scale:
        return None
    h, w = spr.shape[:2]
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if nw > W * 3 or nh > H * 3:
        return None
    img = squash(spr, nw / w, nh / h) if (nw != w or nh != h) else spr
    x0 = sx - anchor[0] * nw
    y0 = sy - anchor[1] * nh
    if flipx:
        img = img[:, ::-1]
    if fog is not None and zc > fog_near:
        fr = clamp((zc - fog_near) / max(1, fog_far - fog_near))
        fr = round(fr * 5) / 5
        img = img.copy()
        img[:, :, :3] = (img[:, :, :3] * (1 - fr) + np.asarray(fog, np.float32) * fr).astype(np.uint8)
    blit(dst, img, x0, y0, alpha=alpha, tint=tint, add=add)
    return (x0, y0, nw, nh, zc, s)


def ground_shadow(dst, cam, X, Z, rx_units, color=(0, 0, 0), alpha=0.28, Y=0.0):
    """тень-эллипс на земле (проецируется как сплюснутый эллипс)."""
    p = cam.project(X, Y, Z)
    if p is None:
        return
    sx, sy, zc = p
    s = cam.f / zc
    rx = rx_units * s
    # сжатие по вертикали — по углу взгляда
    q = cam.project(X, Y, Z + rx_units)
    ry = abs(q[1] - sy) if q is not None else rx * 0.3
    ry = max(1.0, min(rx * 0.6, ry))
    if rx < 1:
        return
    m = ellipse_mask(rx, ry)
    blit_mask(dst, m, sx - (m.shape[1] - 1) / 2, sy - (m.shape[0] - 1) / 2, color, alpha)


# ======================================================================= пикселизация Codex под нужный размер
class CodexPixelizer:
    """Пикселизует кадр исходного спрайтшита Codex (192x208) под нужную высоту на экране.
    Кэширует по (row, col, width_px). Палитра — общая для всех размеров (из атласа ÷2)."""

    def __init__(self, sheet_path=None):
        from PIL import Image
        sheet_path = sheet_path or (ROOT + '/assets_src/codex-spritesheet.png')
        self.sheet = np.array(Image.open(sheet_path).convert('RGBA')).astype(np.float32) / 255.0
        import json
        pal = json.load(open(ROOT + '/sprites/codex_f2.png.json'))['palette']
        self.pal = np.array(pal, np.float32)
        # LUT 32^3 -> цвет палитры (ближайший в линейном приближении Lab-ish)
        g = (np.arange(32) * 8 + 4).astype(np.float32)
        R, G, B = np.meshgrid(g, g, g, indexing='ij')
        rgb = np.stack([R, G, B], -1).reshape(-1, 3)
        lab = self._lab(rgb / 255.0)
        plab = self._lab(self.pal / 255.0)
        d = ((lab[:, None, :] - plab[None]) ** 2).sum(-1)
        self.lut = self.pal[d.argmin(1)].reshape(32, 32, 32, 3).astype(np.uint8)
        self.outline = self.pal[np.argmin(self.pal.sum(1))].astype(np.uint8)
        self.cache = {}

    @staticmethod
    def _lab(rgb):
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
        xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
        f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
        return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)

    def cell(self, row, col):
        return self.sheet[row * 208:(row + 1) * 208, col * 192:(col + 1) * 192]

    def get(self, row, col, width_px, face_fn=None):
        width_px = int(max(6, round(width_px)))
        key = (row, col, width_px, face_fn.__name__ if face_fn else None)
        if key in self.cache:
            return self.cache[key]
        from PIL import Image
        cell = self.cell(row, col)
        if face_fn is not None:
            cell = face_fn(cell)
        f = 192.0 / width_px
        nw, nh = width_px, int(round(208 / f))
        a = cell[:, :, 3]
        lin = np.where(cell[:, :, :3] <= 0.04045, cell[:, :, :3] / 12.92, ((cell[:, :, :3] + 0.055) / 1.055) ** 2.4)
        pre = lin * a[:, :, None]
        out = np.zeros((nh, nw, 4), np.float32)
        for ch in range(3):
            out[:, :, ch] = np.array(Image.fromarray(pre[:, :, ch]).resize((nw, nh), Image.BOX))
        out[:, :, 3] = np.array(Image.fromarray(a).resize((nw, nh), Image.BOX))
        al = out[:, :, 3]
        col_ = np.where(al[:, :, None] > 1e-4, out[:, :, :3] / np.maximum(al[:, :, None], 1e-4), 0)
        srgb = np.where(col_ <= 0.0031308, col_ * 12.92, 1.055 * np.power(np.clip(col_, 0, None), 1 / 2.4) - 0.055)
        q = np.clip((srgb * 255).astype(int) // 8, 0, 31)
        rgb = self.lut[q[:, :, 0], q[:, :, 1], q[:, :, 2]]
        mask = al > 0.45
        pad = np.pad(mask, 1)
        edge = mask & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
        rgb[edge] = self.outline
        img = np.zeros((nh, nw, 4), np.uint8)
        img[:, :, :3] = rgb
        img[:, :, 3] = mask * 255
        self.cache[key] = img
        return img


# мировые размеры персонажей (юниты)
CLAWD_W, CLAWD_H = 24.0, 16.0
CODEX_CELL_W = 24.0  # ширина ячейки 192px исходника в юнитах (контент ≈ 18 x 21.5)


def draw_clawd_3d(dst, cam, clawd, X, Z, Y=0.0, face='idle', anim=None, t=0.0, flipx=False, blush=False,
                  legs=True, tears=0, fog=None, fog_near=200, fog_far=1400, tint=None, add=None, start=0, end=None,
                  loop=True, frame=None, lean_px=0, sx=1.0, sy=1.0):
    """Clawd в мире. anim=None — статичное лицо face; иначе имя официальной анимации ('CrabWalking' и т.п.).
    Масштаб: 1 юнит исходника = 1 юнит мира."""
    if anim is None:
        spr = clawd.face(face, blush, legs, tears)
        ax, ay = spr.shape[1] / 2.0, spr.shape[0]
    else:
        from chars import anim_frame
        idx = frame if frame is not None else anim_frame(t, clawd.meta[anim]['durations'], loop=loop, start=start, end=end)
        spr = clawd.frames[anim][idx]
        ax, ay = clawd.anchor[anim]
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
    return draw_sprite_3d(dst, cam, spr, X, Y, Z, 1.0, anchor=anchor, flipx=flipx, fog=fog, fog_near=fog_near,
                          fog_far=fog_far, tint=tint, add=add)


def draw_codex_3d(dst, cam, pix, codex, X, Z, row=0, col=0, Y=0.0, face=None, flipx=False, fog=None,
                  fog_near=200, fog_far=1400, tint=None, add=None, lean_px=0, sx=1.0, sy=1.0):
    """Codex в мире: пикселизуется прямо под экранный размер (плотность пикселя = пиксель мира).
    face: None | 'heart' | 'happy' | 'back' | 'lt3' | 'sad' | 'dot' (через атлас ÷2 для крупных, иначе из исходника)."""
    p = cam.project(X, Y, Z)
    if p is None:
        return None
    sx_, sy_, zc = p
    scale = cam.f / zc  # экранных px на юнит
    width_px = CODEX_CELL_W * scale
    if width_px < 6:
        return None
    if face is not None:
        # кастомные лица: берём атлас (÷2 или ÷4) с лицом и масштабируем ближайшим соседом
        z = 2 if width_px > 60 else 1
        spr = codex.custom_face(row, col, z, face)
        ax, ay = codex.anchor[z]
        src_ppu = spr.shape[1] / CODEX_CELL_W
    else:
        spr = pix.get(row, col, width_px)
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
    return draw_sprite_3d(dst, cam, spr, X, Y, Z, src_ppu, anchor=anchor, flipx=flipx, fog=fog, fog_near=fog_near,
                          fog_far=fog_far, tint=tint, add=add)


def draw_world(dst, cam, items):
    """items: список (Z_depth_key, callable(dst, cam)). Рисует от дальних к ближним по камерной глубине."""
    keyed = []
    for (X, Y, Z, fn) in items:
        xc, yc, zc = cam.to_cam(X, Y, Z)
        keyed.append((zc, fn))
    keyed.sort(key=lambda k: -k[0])
    for zc, fn in keyed:
        if zc > 0.5:
            fn(dst, cam)
