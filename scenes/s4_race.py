"""Сцена 4 — «РАУНД 2: ГОНКА» (14.0 c).
0.0–1.2   карточка раунда.
1.2–3.0   старт: Claude запрыгивает в карт (офиц. RacingCar 4–11, шлем!), Codex вскакивает на ховерборд;
          отсчёт 3-2-1 с огнями на стартовой арке, «СТАРТ!».
3.0–10.5  гонка 2.5D: трекинг сбоку у земли, «вертолёт», встречный низкий проезд с панорамой, толкотня
          с искрами, погоня сзади к финишной арке, реверс с кубком на переднем плане.
10.5–14.0 столкновение у кубка -> облако драки, кубок в рапиде взлетает (камера — за ним вверх), падает и
          раскалывается; оба лежат оглушённые, небо темнеет, первая молния.
"""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator
from px import *
from bg import (P as PAL, SPARK, SPARK_S, STAR5, ANGER, ANGER2, blit_dither, flash, brawl_cloud, lightning_bolt,
                make_hoverboard, make_cloud, TROPHY, TROPHY_L, TROPHY_R, PEDESTAL)
from cam3d import Cam, look_at, orbit, Panorama, CodexPixelizer, make_cloud_pano
from worlds import MeadowWorld, noise2d
from chars import load, anim_frame
from scene_base import Scene
from scenes._b_helpers import *

ROAD_HW = 26.0
LANE_C, LANE_X = -10.0, 10.0      # Claude (карт) — ближняя полоса (-Z), Codex (выше ростом) — дальняя (+Z)
X_START = 40.0                    # стартовая линия/арка
V = 175.0                         # крейсерская скорость
T_GO = 3.0
T_HIT = 10.5
CLAWD_START_X, CODEX_START_X = -14.0, 22.0


def pack_x(t):
    """центр «пачки» гонщиков по X (аналитически от времени)."""
    if t <= T_GO:
        return 0.0
    s = (t - T_GO) / 0.9
    if s < 1:
        return 0.9 * V * (s ** 3 - 0.5 * s ** 4)
    return 0.45 * V + V * (t - T_GO - 0.9)


X_FIN = pack_x(T_HIT) + 12.0      # центр пьедестала = финишная линия = арка

# относительные смещения по X (лидерство меняется) и полосы по Z
OFF_T = [0.0, 3.0, 3.3, 3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 6.4, 6.7, 7.0, 7.4, 7.8, 8.2, 8.6, 9.0, 9.5, 10.0, 10.5, 14]
OFF_C = [CLAWD_START_X, CLAWD_START_X, -4, 8, 9, 3, -2, -6, -7, -5, -1, 5, 5, 4, 5, 3, 2, 2, 0, 1, 10, 10]
OFF_X = [CODEX_START_X, CODEX_START_X, 16, 4, 3, 3, 6, 8, 9, 6, 3, -1, -3, -4, -2, -4, 0, -1, 1, 0, 11, 11]
Z_T = [0.0, 7.15, 7.42, 7.52, 7.75, 7.95, 8.08, 8.18, 8.4, 8.7, 9.9, 10.5, 14]
Z_C = [LANE_C, LANE_C, -9.0, -12.0, -14.0, -12.5, -5.5, -8.0, -11.0, -10.0, -10.0, -2.5, -2.5]
Z_X = [LANE_X, LANE_X, 2.5, 7.0, 12.0, 9.0, 4.0, 8.5, 11.0, 10.0, 10.0, 2.5, 2.5]
BUMP1, BUMP2 = 7.47, 8.12

SKY_DAY = PAL('day')['sky']
SKY_STORM = PAL('storm')['sky']


def paint_road(tex, pal):
    """грунтовая дорога вдоль X (строки текстуры), колеи, травяной гребень посередине, камушки."""
    size = tex.shape[0]
    rng = np.random.default_rng(21)
    zc_row = 256 * 2  # Z=0
    n1 = noise2d(size, 16, 5, 3)
    n2 = noise2d(size, 64, 6, 2)
    dirt = [np.asarray(c, np.float32) for c in pal['dirt']]
    cols = np.arange(size)
    edge = (noise2d(size, 24, 9, 2)[0] - 0.5) * 6 + (np.sin(cols * 0.21) * 1.2)
    for r in range(zc_row - 64, zc_row + 65):
        z = (r - zc_row) / 2.0
        az = abs(z)
        lim = ROAD_HW + edge * 0.5
        inside = az < lim
        if not inside.any():
            continue
        row = tex[r].astype(np.float32)
        base = np.where((n1[r] > 0.55)[:, None], dirt[2] * 0.6 + dirt[0] * 0.4, dirt[0])
        base = np.where((n1[r] < 0.38)[:, None], dirt[1] * 0.5 + dirt[0] * 0.5, base)
        # колеи
        for lane in (LANE_C, LANE_X):
            d = abs(z - lane)
            if d < 4.5:
                k = 0.72 if d < 2.5 else 0.86
                base = base * k + dirt[1] * (1 - k) * 0.4
        # мелкая фактура
        fine = (n2[r] > 0.6) & (rng.random(size) < 0.35)
        base[fine] = dirt[1]
        lite = (n2[r] < 0.35) & (rng.random(size) < 0.25)
        base[lite] = dirt[2]
        # край дороги
        near_edge = inside & (az > lim - 1.2)
        base[near_edge] = dirt[1] * 0.85
        # травяной гребень по центру
        if az < 2.2:
            g = np.asarray(pal['grass'][0], np.float32)
            m = rng.random(size) < (0.85 - az * 0.3)
            base[m] = np.where((rng.random(size) < 0.3)[m][:, None], np.asarray(pal['grass'][2], np.float32), g)
        row[inside] = base[inside]
        tex[r] = np.clip(row, 0, 255).astype(np.uint8)
    # камушки
    for i in range(1400):
        x = int(rng.integers(0, size))
        z = rng.uniform(-ROAD_HW + 2, ROAD_HW - 2)
        r = int(zc_row + z * 2)
        c = np.array([150, 140, 125]) if rng.random() < 0.6 else np.array([120, 108, 96])
        tex[r, x] = c
        if rng.random() < 0.5:
            tex[r, (x + 1) % size] = c * 0.8
    # пучки травы по краям
    for i in range(2600):
        x = int(rng.integers(0, size))
        side = 1 if rng.random() < 0.5 else -1
        z = side * (ROAD_HW + rng.uniform(-2.5, 1.0))
        r = int(zc_row + z * 2)
        g = pal['grass'][3] if rng.random() < 0.5 else pal['grass'][1]
        tex[r, x] = g
        tex[r - 1, (x - 1) % size] = g
        tex[r - 1, (x + 1) % size] = g
    return tex


def checker_line(out, X, Z, x0, x1, sq=3.0):
    """шахматная полоса поперёк дороги в [x0, x1]."""
    m = (X >= x0) & (X < x1) & (np.abs(Z) < ROAD_HW - 1)
    if not m.any():
        return
    ch = ((np.floor(X[m] / sq) + np.floor(Z[m] / sq)) % 2) == 0
    vals = np.where(ch[:, None], np.array([242, 240, 232], np.float32), np.array([28, 26, 34], np.float32))
    out[m] = vals


def make_banner(text, w=104, h=18, fg=(30, 24, 40), bg_=(255, 250, 238), checker=True):
    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, :3] = bg_
    img[:, :, 3] = 255
    if checker:
        for y in range(h):
            for x in range(w):
                if y < 3 or y >= h - 3:
                    if ((x // 3) + (y // 3)) % 2 == 0:
                        img[y, x, :3] = (26, 24, 30)
    m = text_mask(text, 'big')
    th, tw = m.shape
    x0, y0 = (w - tw) // 2, (h - th) // 2
    img[y0:y0 + th, x0:x0 + tw][m] = list(fg) + [255]
    img[0, :, :3] = img[-1, :, :3] = (20, 18, 24)
    img[:, 0, :3] = img[:, -1, :3] = (20, 18, 24)
    return img


def make_lights(state):
    """коробка светофора: 3 огня. state: 0..3 — сколько красных; 4 — все зелёные."""
    w, h = 23, 9
    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, :3] = (26, 24, 32)
    img[:, :, 3] = 255
    img[0, :, :3] = img[-1, :, :3] = (10, 10, 14)
    img[:, 0, :3] = img[:, -1, :3] = (10, 10, 14)
    for i in range(3):
        cx = 4 + i * 7
        on = (state == 4) or (i < state)
        col = (70, 230, 90) if state == 4 else ((255, 60, 50) if on else (70, 24, 26))
        hl = (200, 255, 200) if state == 4 else ((255, 190, 170) if on else (90, 40, 40))
        m = disc_mask(2.2)
        ys, xs = np.where(m)
        for y, x in zip(ys, xs):
            yy, xx = 2 + y, cx - 2 + x
            if 0 <= yy < h and 0 <= xx < w:
                img[yy, xx, :3] = col
        img[3, cx - 1, :3] = hl
    return img


def flame_sprite(t, big=False):
    """язык пламени выхлопа (направлен влево)."""
    ph = int(t * 30) % 3
    L = (9 if big else 6) + ph
    img = np.zeros((5, L + 1, 4), np.uint8)
    for x in range(L + 1):
        k = x / L
        hh = 2 if k < 0.35 else (1 if k < 0.75 else 0)
        col = (255, 250, 210) if k < 0.3 else ((255, 200, 70) if k < 0.6 else (255, 110, 40))
        for y in range(2 - hh, 3 + hh):
            img[y, L - x, :3] = col
            img[y, L - x, 3] = 255
    return img


class RaceScene(Scene):
    name = 's4_race'
    dur = 14.0

    def __init__(self):
        self.cl, self.cx = load()
        self.pix = CodexPixelizer()
        self.world = MeadowWorld('day', extra_paint=paint_road, n_trees=40)
        wd = self.world
        self.ground = FastGround(mips=wd.ground.mips, tpu=2.0, origin=wd.ground.origin)
        self.fog = wd.fog
        self.p = wd.p
        pal_s = PAL('storm')
        self.storm_clouds = make_cloud_pano(pal_s, seed=41, height=150, n=26)
        self.offc = PchipInterpolator(OFF_T, OFF_C)
        self.offx = PchipInterpolator(OFF_T, OFF_X)
        self.zc = PchipInterpolator(Z_T, Z_C)
        self.zx = PchipInterpolator(Z_T, Z_X)
        # деревья вдоль дороги (детерминированно): масштаб ~1 (плотность пикселя как у Clawd), дальше от дороги
        sprs = [spr for (_, _, spr, sc) in wd.trees]
        rng = np.random.default_rng(77)
        trees = []
        x = -300.0
        while x < X_FIN + 900:
            spr = sprs[int(rng.integers(0, len(sprs)))]
            trees.append((x + rng.uniform(-6, 6), rng.uniform(64, 86), spr, rng.uniform(0.95, 1.3)))
            if rng.random() < 0.6:
                spr = sprs[int(rng.integers(0, len(sprs)))]
                trees.append((x + rng.uniform(0, 20), rng.uniform(100, 230), spr, rng.uniform(1.0, 1.4)))
            if rng.random() < 0.45:
                spr = sprs[int(rng.integers(0, len(sprs)))]
                trees.append((x + rng.uniform(-8, 8), -rng.uniform(110, 230), spr, rng.uniform(1.0, 1.4)))
            x += rng.uniform(22, 36)
        self.trees = trees
        self.tree_x = np.array([t_[0] for t_ in trees])
        self.tree_z = np.array([t_[1] for t_ in trees])
        # стойка забора
        post = np.zeros((10, 3, 4), np.uint8)
        post[:, :, :3] = (122, 86, 52)
        post[:, 0, :3] = (150, 110, 70)
        post[:, 2, :3] = (86, 58, 36)
        post[0, :, :3] = (170, 130, 86)
        post[:, :, 3] = 255
        self.post = post
        self.banner_start = make_banner('СТАРТ')
        self.banner_fin = make_banner('ФИНИШ')
        self.lights = [make_lights(i) for i in range(5)]
        self.checker_tex = self._checker_tex()
        self.helmet = self._helmet()
        # облака-билборды для кадра «кубок в небе»
        self.sky_clouds = []
        e = np.array([X_FIN + 56, 8.0, -8.0])
        apex = np.array([X_FIN - 8, 92.0, -4.0])
        dv = (apex - e) / np.linalg.norm(apex - e)
        for i in range(10):
            c = make_cloud(int(rng.integers(1e6)), int(rng.integers(50, 90)), int(rng.integers(18, 28)), self.p['cloud'])
            dist = rng.uniform(170, 330)
            lat = np.array([0.0, 0.0, 1.0]) * rng.uniform(-150, 150) + np.array([1.0, 0.3, 0.0]) * rng.uniform(-90, 90)
            P = e + dv * dist + lat
            self.sky_clouds.append((float(P[0]), float(max(40.0, P[1] + rng.uniform(-60, 80))), float(P[2]), c))
        self.port_l = scale_nn(self.cl.face('angry'), 3)
        self.port_r = self.pix.get(0, 0, 66)
        self.card_bg = self._card_bg()

        def crop(spr):
            ys, xs = np.where(spr[:, :, 3] > 0)
            return spr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        self.half_l = crop(TROPHY_L)
        self.half_r = crop(TROPHY_R)
        self.half_l_lie = np.ascontiguousarray(np.rot90(self.half_l, 1))
        self.half_r_lie = np.ascontiguousarray(np.rot90(self.half_r, 3))

    # ============================================================ ассеты
    def _checker_tex(self):
        img = np.zeros((6, 6, 4), np.uint8)
        for y in range(6):
            for x in range(6):
                img[y, x, :3] = (240, 238, 230) if ((x // 3) + (y // 3)) % 2 == 0 else (28, 26, 34)
        img[:, :, 3] = 255
        return img

    def _helmet(self):
        f = self.cl.frames['RacingCar'][7]
        ys, xs = np.where(f[:, :, 3] > 0)
        top = ys.min()
        # шлем — верхние строки (тёмно-серые/белые пиксели над телом)
        crop = f[top:top + 8].copy()
        body = (crop[:, :, 0] > 150) & (crop[:, :, 1] < 140) & (crop[:, :, 3] > 0)
        crop[body, 3] = 0
        ys, xs = np.where(crop[:, :, 3] > 0)
        return crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    def _card_bg(self):
        # диагональные шахматные клетки (фон карточки гонки)
        sq = 16
        yy, xx = np.mgrid[0:H + 64, 0:W + 64]
        ch = ((xx // sq + yy // sq) % 2) == 0
        img = np.where(ch[:, :, None], np.array([22, 20, 30], np.float32), np.array([10, 9, 16], np.float32))
        return img

    # ============================================================ кинематика
    def pos_cl(self, t):
        return pack_x(t) + float(self.offc(min(t, 14))), float(self.zc(min(t, 14)))

    def pos_cx(self, t):
        return pack_x(t) + float(self.offx(min(t, 14))), float(self.zx(min(t, 14)))

    def speed(self, t):
        if t <= T_GO:
            return 0.0
        s = (t - T_GO) / 0.9
        return V * (3 * s * s - 2 * s ** 3) if s < 1 else V

    # ============================================================ камера
    T_TILT, T_AFTER = 10.85, 12.15

    def camera(self, t):
        if t < 2.1:
            # старт сбоку: медленный наезд
            k = ease_in_out(seg(t, 1.2, 2.1))
            eye = lerp3((0, 16, -88), (3, 12.5, -70), k)
            tgt = lerp3((6, 13, 0), (6, 13, 0), k)
            return cam_shot(eye, tgt, 265), 'start'
        if t < 2.4:
            k = seg(t, 2.1, 2.4)
            eye = lerp3((132, 6, -36), (125, 6, -34), k)
            return cam_shot(eye, (8, 17, 0), 245), 'cd3'
        if t < 2.7:
            k = seg(t, 2.4, 2.7)
            xc, zc = self.pos_cl(t)
            eye = lerp3((xc + 34, 8.0, zc - 34), (xc + 30, 8.0, zc - 30), k)
            return cam_shot(eye, (xc + 4, 4.0, zc), 290), 'cd2'
        if t < T_GO:
            k = seg(t, 2.7, T_GO)
            xx, zx = self.pos_cx(t)
            eye = lerp3((xx - 20, 15, zx - 42), (xx - 17, 14.5, zx - 38), k)
            return cam_shot(eye, (xx + 2, 9.5, zx), 290), 'cd1'
        if t < 4.4:
            # рывок со старта: камера стоит, затем разгоняется и ведёт гонщиков сбоку
            px = pack_x(t)
            k = ease_in_out(seg(t, T_GO + 0.1, 3.9))
            eye = (lerp(16.0, px + 6.0, k), 13.0, -76.0)
            tgt = (lerp(16.0, px + 14.0, ease_in_out(seg(t, T_GO, 3.8))), 9.0, 0.0)
            return cam_shot(eye, tgt, 260), 'launch'
        if t < 5.6:
            # «вертолёт»: высоко, облетает сверху
            k = seg(t, 4.4, 5.6)
            px = pack_x(t)
            ang = lerp(-0.6, 0.22, ease_in_out(k))
            ex = px + 28 - math.sin(ang) * 108
            ez = -math.cos(ang) * 108
            return cam_shot((ex, lerp(96, 74, k), ez), (px + 28, 0, 0), 340), 'heli'
        if t < 7.0:
            # встречный низкий проезд: камера у обочины впереди, панорамирует за пролетающими
            ex = pack_x(6.5) + 22.0
            px = pack_x(t)
            tgt = (max(px + 6.0, ex + 8.0) if ex - px < 8 else px + 6.0, 9.0, 0.0)
            return cam_shot((ex, 4.5, -36.0), tgt, 285), 'headon'
        if t < 8.6:
            # толкотня: плотный трекинг сбоку
            px = pack_x(t)
            k = seg(t, 7.0, 8.6)
            eye = (px + lerp(-4, 8, k), 27.0, -66.0)
            return cam_shot(eye, (px + lerp(2, 6, k), 6.0, 0.0), 265), 'jostle'
        if t < 9.6:
            # погоня: 3/4 сзади, впереди растёт финишная арка
            px = pack_x(t)
            k = seg(t, 8.6, 9.6)
            eye = (px - lerp(60, 52, k), lerp(20, 16, k), -lerp(58, 50, k))
            return cam_shot(eye, (px + 70, 7, 4), 250), 'chase'
        return self.cam_finish(t)

    def trophy_state(self, t):
        """(X, Y, Z, spin, phase) кубка. phase: 'ped' | 'fly' | 'broken'"""
        t0, tap, tland = 10.55, 11.75, 12.32
        if t < t0:
            return X_FIN, 8.0, 0.0, 0.0, 'ped'
        if t < tap:
            k = ease_out(seg(t, t0, tap))
            return X_FIN - 8 * k, 8.0 + 84 * k, -4 * k, (t - t0) * 4.0, 'fly'
        if t < tland:
            k = seg(t, tap, tland)
            return X_FIN - 8 - 26 * k, 92.0 * (1 - k * k), -4 + 4 * k, (tap - t0) * 4.0 + (t - tap) * 14.0, 'fly'
        return X_FIN - 34, 0.0, 0.0, 0.0, 'broken'

    def cam_finish(self, t):
        e0, e1 = (X_FIN + 64, 5.0, -8.0), (X_FIN + 45, 6.0, -7.0)
        if t < self.T_TILT:
            k = ease_out(seg(t, 9.6, self.T_TILT))
            return cam_shot(lerp3(e0, e1, k), (X_FIN - 40, 13, 0.0), 235), 'finish'
        if t < self.T_AFTER:
            # камера задирается вверх за кубком (и чуть «зумит»), затем опускается за падением
            tx, ty, tz, _, ph = self.trophy_state(t)
            k = ease_in_out(seg(t, self.T_TILT, 11.35))
            eye = lerp3(e1, (X_FIN + 48, 9.0, -10.0), seg(t, self.T_TILT, self.T_AFTER))
            tgt = (lerp(X_FIN - 40, tx, k), lerp(13.0, ty + 3.0, k), lerp(0.0, tz, k))
            f = lerp(235, 520, ease_in_out(seg(t, self.T_TILT, 11.55))) if t < 11.75 else lerp(520, 300, seg(t, 11.75, self.T_AFTER))
            return cam_shot(eye, tgt, f), 'tilt'
        # после: вид из-за места аварии на арку, кран назад/вверх, небо темнеет
        k = ease_in_out(seg(t, self.T_AFTER, 14.0))
        eye = lerp3((X_FIN - 64, 7.5, -5.0), (X_FIN - 98, 23.0, -10.0), k)
        tgt = lerp3((X_FIN - 22, 8.0, 0.0), (X_FIN - 10, 17.0, 0.0), k)
        return cam_shot(eye, tgt, 240), 'after'

    # ============================================================ фон/мир
    def storm_k(self, t):
        return ease_in_out(seg(t, 12.45, 13.6))

    def backdrop(self, c, cam, t, sk):
        wd = self.world
        stops = lerp_stops(SKY_DAY, SKY_STORM, sk) if sk > 0 else SKY_DAY
        fog = self.fog * (1 - sk) + hexc('#4a5270') * sk
        sky_fast(c, cam, stops, span=0.75, below=fog)
        tint = None
        if sk < 0.95:
            pano_draw(wd.clouds, c, cam, drift=t * 4.0, y_offset=-8, tint=tint)
        pano_draw(wd.mount, c, cam, y_offset=1, tint=tint)
        pano_draw(wd.hills_far, c, cam, y_offset=2, tint=tint)
        pano_draw(wd.hills, c, cam, y_offset=3, tint=tint)
        if sk > 0:
            pano_draw(self.storm_clouds, c, cam, drift=t * 14.0, y_offset=lerp(-150, -6, sk),
                      dither_alpha=min(1.0, sk * 1.4))
        return fog, tint

    def ground_shade(self, t):
        def shade(out, X, Z, tt, r0):
            checker_line(out, X, Z, X_START - 3.0, X_START + 3.0)
            checker_line(out, X, Z, X_FIN - 3.0, X_FIN + 3.0)
            return out
        return shade

    def scenery_items(self, cam, fog, tint, t):
        items = []
        right, up, fwd = cam_basis(cam)
        dx = self.tree_x - cam.x
        dz = self.tree_z - cam.z
        zc = dx * fwd[0] + dz * fwd[2] + (0 - cam.y) * fwd[1]
        xc = dx * right[0] + dz * right[2]
        vis = (zc > 2) & (zc < 1400) & (np.abs(xc) < zc * (W / cam.f) * 0.75 + 60)
        for i in np.where(vis)[0]:
            X, Z, spr, sc = self.trees[i]

            def fn(d, cm, X=X, Z=Z, spr=spr, sc=sc):
                bb_sprite(d, cm, spr, X, 0.0, Z, 1.0 / sc, anchor=(0.5, 0.97), fog=fog, fog_near=260, fog_far=1500,
                          tint=tint)
            items.append((X, 0.0, Z, fn))
        # забор по дальней обочине: каждый пролёт — свой элемент (правильная глубина)
        FZ = 36.0
        x0 = math.floor((cam.x - 300) / 9.0) * 9.0
        colr = np.array([150, 112, 72], np.float32) * (np.asarray(tint) if tint is not None else 1.0)
        for i in range(120):
            X = x0 + i * 9.0
            zc_ = (X - cam.x) * fwd[0] + (FZ - cam.z) * fwd[2] - cam.y * fwd[1]
            if zc_ < 1 or zc_ > 650:
                continue

            def seg_fn(d, cm, X=X):
                line3d(d, cm, (X, 6.5, FZ), (X + 9.0, 6.5, FZ), colr)
                line3d(d, cm, (X, 3.5, FZ), (X + 9.0, 3.5, FZ), colr * 0.8)
                bb_sprite(d, cm, self.post, X, 0.0, FZ, 1.25, tint=tint)
            items.append((X + 4.5, 0.0, FZ, seg_fn))
        return items

    def gantry_items(self, X0, banner_img, lights_state=None):
        """арка как набор элементов (у каждой стойки своя глубина)."""
        colp = dict(top=(236, 236, 240), front=(206, 54, 54), back=(206, 54, 54), left=(170, 40, 40),
                    right=(170, 40, 40))
        colw = dict(top=(250, 250, 250), front=(232, 228, 222), back=(232, 228, 222), left=(200, 196, 190),
                    right=(200, 196, 190))
        items = []
        for zs in (-34.0, 34.0):
            items.append((X0, 20.0, zs, lambda d, cm, zs=zs: draw_box(d, cm, X0 - 1.2, X0 + 1.2, 0.0, 44.0, zs - 1.2,
                                                                      zs + 1.2, colp)))

        def bar(d, cm):
            draw_box(d, cm, X0 - 1.5, X0 + 1.5, 42.0, 46.0, -35.2, 35.2, colw)
            bw, bh = 56.0, 10.0
            draw_quad(d, cm, (X0 + 1.6, 41.8, -bw / 2), (0, 0, bw), (0, -bh, 0), tex=banner_img)
            draw_quad(d, cm, (X0 - 1.6, 41.8, bw / 2), (0, 0, -bw), (0, -bh, 0), tex=banner_img)
            if lights_state is not None:
                spr = self.lights[lights_state]
                bb_sprite(d, cm, spr, X0, 46.0, 0.0, 0.75, anchor=(0.5, 1.0))
                if lights_state >= 1:
                    p = cm.project(X0, 52.0, 0.0)
                    if p:
                        col = (70, 230, 90) if lights_state == 4 else (255, 70, 60)
                        dither_disc(d, p[0], p[1], cm.f / p[2] * 20, col, 0.12)
        items.append((X0, 44.0, 0.0, bar))
        return items

    # ============================================================ гонщики
    def kart_frame(self, t):
        d = self.cl.meta['RacingCar']['durations']
        if t < 1.3:
            return anim_frame(t, d, loop=True, start=0, end=4)
        if t < 2.05:
            return anim_frame(t - 1.3, d, loop=False, start=4, end=12)
        sp = 1.0 if t < T_GO else 1.6
        return anim_frame((t - 2.05) * sp, d, loop=True, start=12, end=35)

    def draw_kart(self, d, cm, t, X, Z, tint=None, rev=0.0):
        fr = self.kart_frame(t)
        fl = facing_flip(cm, X, 5.0, Z)
        v = self.speed(t)
        bob = 0.0
        if t >= 2.05:
            bob = abs(math.sin(t * (22 if v > 1 else 30))) * (0.5 if v > 1 else 0.35 + rev * 0.5)
        lean_px = 0
        if T_GO <= t < T_GO + 0.45:
            lean_px = -2  # «задирает нос» на старте
        for tb, sg in ((BUMP1, 1), (BUMP2, -1)):
            if tb <= t < tb + 0.2:
                kk = 1 - (t - tb) / 0.2
                lean_px = int(round(2 * sg * kk))
                bob += 2.2 * kk * abs(math.sin((t - tb) * 30))
        # пыль из-под колёс + взрыв пыли на старте
        if v > 5:
            self.dust_trail(d, cm, t, 'cl', tint)
        if T_GO <= t < T_GO + 0.7:
            a = t - T_GO
            for i in range(9):
                ang = math.pi * (0.55 + 0.9 * hash01(i, 41))
                dist = (4 + a * 38) * (0.6 + 0.4 * hash01(i, 42))
                P = (CLAWD_START_X - 22 + math.cos(ang) * dist, 2 + math.sin(ang) * dist * 0.25 + a * 6,
                     LANE_C + (hash01(i, 43) - 0.5) * 8)
                sp = world_to_screen(cm, P)
                if sp:
                    k = cm.f / sp[2]
                    puff(d, sp[0], sp[1], (2.5 + a * 7) * k, (226, 206, 170), (176, 150, 116), alpha=1 - a / 0.7)
        shadow_b(d, cm, X, Z, 21.0 if fr >= 4 else 10.0, 0.3)
        draw_clawd_b(d, cm, self.cl, X, bob, Z, anim='RacingCar', frame=fr, flipx=fl, lean_px=lean_px, tint=tint)
        # выхлоп
        if t >= 2.05:
            big = (T_GO <= t < T_GO + 0.6) or rev > 0.5
            spr = flame_sprite(t, big)
            rx = X - 25.5 if not fl else X + 25.5
            s = world_to_screen(cm, (rx, 4.2 + bob, Z - 0.3))
            if s:
                k = max(1, int(round(cm.f / s[2] * 0.8)))
                img = scale_nn(spr, k)
                if fl:
                    img = img[:, ::-1]
                    blit(d, img, s[0], s[1] - img.shape[0] / 2)
                else:
                    blit(d, img, s[0] - img.shape[1], s[1] - img.shape[0] / 2)

    def board_y(self, t):
        return 3.0 + math.sin(t * 7.0) * 0.7

    def draw_rider(self, d, cm, t, X, Z, tint=None, pose=None):
        """Codex на ховерборде. pose: None -> по времени."""
        hb = self.board_y(t)
        # тяга: дизерные голубые лучи от сопел к земле
        for jx in (-8.0, 8.0):
            p0 = world_to_screen(cm, (X + jx, hb, Z - 0.6))
            p1 = world_to_screen(cm, (X + jx, 0.0, Z - 0.6))
            if p0 and p1:
                k = cm.f / p0[2]
                wdt = max(1, int(round(k * 1.4)))
                fl = 0.08 * math.sin(t * 40 + jx)
                for q in range(wdt):
                    xq0 = p0[0] - wdt / 2 + q
                    xq1 = p1[0] - wdt / 2 + q
                    for (ka, kb, dd) in ((0.0, 0.35, 0.95), (0.35, 0.7, 0.6), (0.7, 1.0, 0.3)):
                        line_fast(d, lerp(xq0, xq1, ka), lerp(p0[1], p1[1], ka), lerp(xq0, xq1, kb),
                                  lerp(p0[1], p1[1], kb), (160, 238, 255) if dd > 0.5 else (120, 210, 255),
                                  dither=dd + fl)
        v = self.speed(t)
        if v > 5:
            self.jet_trail(d, cm, t, X, Z)
        shadow_b(d, cm, X, Z, 10.5 - hb * 0.4, 0.28)
        board = make_hoverboard(t)
        fl = facing_flip(cm, X, 5.0, Z)
        bb_sprite(d, cm, board, X, hb, Z, 1.25, anchor=(0.5, 1.0), flipx=fl, tint=tint)
        deck = hb + 8 / 1.25 - 0.6
        row, col, lift = 4, 2, 5.0
        if pose is not None:
            row, col, lift = pose
        lp = 0
        for tb, sg in ((BUMP1, -1), (BUMP2, 1)):
            if tb <= t < tb + 0.22:
                kk = 1 - (t - tb) / 0.22
                lp = int(round(6 * sg * kk))
                deck += 3.0 * kk
        draw_codex_b(d, cm, self.pix, self.cx, X, deck, Z, row=row, col=col, y_lift_px=lift, tint=tint, lean_px=lp)

    def dust_trail(self, d, cm, t, who, tint=None, rate=24.0, life=0.7):
        n = int(life * rate) + 1
        base = math.floor(t * rate)
        for j in range(n):
            e = base - j
            te = e / rate
            age = t - te
            if age < 0 or age > life or te < T_GO:
                continue
            X, Z = self.pos_cl(te)
            v = self.speed(te)
            if v < 5:
                continue
            h1, h2 = hash01(e, 3), hash01(e, 7)
            P = (X - 18 + (h1 - 0.5) * 4 - age * 22, 1.2 + age * (6 + 5 * h2), Z + (h2 - 0.5) * 3)
            s = world_to_screen(cm, P)
            if s is None:
                continue
            k = cm.f / s[2]
            r = (1.6 + age * 7.5) * k
            a = 1 - age / life
            col = np.array([226, 206, 170])
            sh = np.array([176, 150, 116])
            if tint is not None:
                col = col * np.asarray(tint)
                sh = sh * np.asarray(tint)
            puff(d, s[0], s[1], r, col, sh, alpha=a * 0.9)

    def jet_trail(self, d, cm, t, X, Z, rate=30.0, life=0.35):
        n = int(life * rate) + 1
        base = math.floor(t * rate)
        for j in range(n):
            e = base - j
            te = e / rate
            age = t - te
            if age < 0 or age > life or te < T_GO:
                continue
            Xe, Ze = self.pos_cx(te)
            h1 = hash01(e, 11)
            P = (Xe - 13 - age * 10, self.board_y(te) + 0.8 + (h1 - 0.5) * 1.5, Ze - 0.5)
            s = world_to_screen(cm, P)
            if s is None:
                continue
            k = cm.f / s[2]
            col = (255, 255, 255) if age < 0.08 else ((142, 227, 240) if age < 0.2 else (63, 160, 255))
            sz = max(1, int(round(k * (1.2 - age * 2))))
            rect(d, s[0], s[1], sz, sz, col)

    # ============================================================ эффекты гонки
    def speed_streaks(self, c, t, strength, direction=-1, seed=3, n=26):
        if strength <= 0:
            return
        for i in range(n):
            h1, h2, h3 = hash01(i, seed), hash01(i, seed + 1), hash01(i, seed + 2)
            if h3 > strength:
                continue
            y = 8 + h1 * (H - 16)
            L = 30 + h2 * 70
            ph = (h3 * 997 + t * (1100 + h2 * 600)) % (W + 200) - 100
            x = W - ph if direction < 0 else ph
            line_fast(c, x, y, x + L, y, (255, 255, 255), dither=0.8)
            line_fast(c, x + L * 0.3, y + 1, x + L, y + 1, (200, 225, 255), dither=0.5)

    def zoom_lines(self, c, cx, cy, t, strength=1.0, n=34, seed=5):
        for i in range(n):
            h1, h2, h3 = hash01(i, seed), hash01(i, seed + 1), hash01(i, seed + 2)
            if h3 > strength:
                continue
            ang = h1 * 2 * math.pi
            ph = (t * (3.2 + h2 * 2) + h2) % 1.0
            r0 = 90 + ph * 260
            r1 = r0 + 30 + h2 * 50
            x0, y0 = cx + math.cos(ang) * r0, cy + math.sin(ang) * r0 * 0.7
            x1, y1 = cx + math.cos(ang) * r1, cy + math.sin(ang) * r1 * 0.7
            line_fast(c, x0, y0, x1, y1, (255, 255, 255), dither=0.5)

    def sparks_burst(self, c, cam, t, t0, P, n=18, seed=1):
        a = t - t0
        if a < 0 or a > 0.5:
            return
        s0 = world_to_screen(cam, P)
        if s0 is None:
            return
        k = cam.f / s0[2]
        for i in range(n):
            ang = hash01(i, seed) * 2 * math.pi
            sp = (30 + hash01(i, seed + 3) * 50) * k
            x = s0[0] + math.cos(ang) * sp * a
            y = s0[1] + math.sin(ang) * sp * a * 0.8 + 90 * k * a * a
            if a < 0.12 and i % 3 == 0:
                spr = SPARK
                blit(c, spr, x - 2, y - 2)
            else:
                col = (255, 255, 220) if a < 0.15 else (255, 200, 80)
                rect(c, x, y, max(1, k * 0.5), max(1, k * 0.5), col)
                if a < 0.25:
                    x2 = x - math.cos(ang) * sp * 0.03
                    y2 = y - math.sin(ang) * sp * 0.03 * 0.8
                    line_fast(c, x2, y2, x, y, (255, 230, 140))
        if a < 0.08:
            spr = scale_nn(SPARK, max(1, int(k * 0.6)))
            blit(c, spr, s0[0] - spr.shape[1] / 2, s0[1] - spr.shape[0] / 2)

    # ============================================================ карточка
    def card(self, t):
        off = int(t * 60) % 32
        c = self.card_bg[off:off + H, off:off + W].copy()
        # спидлайны на фоне
        for i in range(22):
            h1, h2 = hash01(i, 71), hash01(i, 72)
            y = h1 * H
            L = 30 + h2 * 90
            x = W - ((h2 * 800 + t * 700) % (W + 200)) + 60
            line_fast(c, x, y, x + L, y, (60, 56, 80))
        round_card(c, t, 'РАУНД 2', 'ГОНКА', (255, 214, 90), (60, 30, 8), self.port_l, self.port_r)
        if t > 1.08:
            flash(c, seg(t, 1.08, 1.2) * 0.9)
        return c

    # ============================================================ кадр
    def render(self, t):
        if t < 1.2:
            return self.card(t)
        cam, shot = self.camera(t)
        amp = 0.0
        for tb, a0 in ((BUMP1, 1.6), (BUMP2, 1.6), (T_HIT, 4.0), (12.32, 2.2)):
            if tb <= t < tb + 0.35:
                amp = max(amp, a0 * (1 - (t - tb) / 0.35))
        if shot == 'launch' and T_GO <= t < T_GO + 0.5:
            amp = max(amp, 1.2 * (1 - (t - T_GO) / 0.5))
        if shot == 'cd2':
            amp = max(amp, 0.5)
        if shot in ('launch', 'jostle', 'chase') and self.speed(t) > 60:
            amp = max(amp, 0.28)      # вибрация «операторской машины» на скорости
        cam = shake_cam(cam, t, amp, seed=5)
        sk = self.storm_k(t)
        c = canvas()
        fog, tint = self.backdrop(c, cam, t, sk)
        if shot == 'tilt':
            self.sky_extras(c, cam, t)
        self.ground.render(c, cam, fog=fog, fog_near=260, fog_far=1500, shade=self.ground_shade(t), tint=tint)
        items = self.scenery_items(cam, fog, tint, t)
        light_state = 0 if t < 2.1 else (1 if t < 2.4 else (2 if t < 2.7 else (3 if t < T_GO else 4)))
        items += self.gantry_items(X_START, self.banner_start, light_state)
        items += self.gantry_items(X_FIN, self.banner_fin, None)
        tx, ty, tz, spin, ph = self.trophy_state(t)

        def ped(d, cm):
            if t < T_HIT:
                bb_sprite(d, cm, PEDESTAL, X_FIN, 0.0, 0.0, 2.0, tint=tint)
            else:
                spr = np.ascontiguousarray(np.rot90(PEDESTAL, 1))
                bb_sprite(d, cm, spr, X_FIN + 5, 0.0, -2.0, 2.0, tint=tint)
        items.append((X_FIN, 0.0, 0.0, ped))
        if ph == 'ped':
            items.append((X_FIN, 8.0, -0.2, lambda d, cm: bb_sprite(d, cm, TROPHY, X_FIN, 8.0, -0.2, 2.0, tint=tint)))
        elif ph == 'fly':
            items.append((tx, ty, tz, lambda d, cm: self.draw_trophy_spin(d, cm, t, tx, ty, tz, spin, tint)))
        if t < T_HIT:
            xc, zc = self.pos_cl(t)
            xx, zx = self.pos_cx(t)
            rev = 1.0 if shot == 'cd2' else 0.0
            # «киношный чит»: в крупных планах отсчёта соперник вне кадра
            if shot != 'cd1':
                items.append((xc, 0.0, zc, lambda d, cm: self.draw_kart(d, cm, t, xc, zc, None, rev)))
            if shot != 'cd2':
                items.append((xx, 0.0, zx, lambda d, cm: self.rider_or_jump(d, cm, t, xx, zx)))
        else:
            self.aftermath_items(items, t, tint)
        sort_draw(c, cam, items)
        # ---- эффекты поверх
        v = self.speed(t)
        if shot in ('launch', 'jostle') and v > 40:
            self.speed_streaks(c, t, min(1.0, v / V) * (0.8 if shot == 'launch' else 1.0))
        if shot == 'chase':
            pc = cam.project(X_FIN, 10, 0)
            if pc:
                self.zoom_lines(c, pc[0], pc[1], t, 0.8)
        if shot == 'headon':
            px = pack_x(t)
            ex = pack_x(6.5) + 22.0
            if abs(px - ex) < 60:
                self.speed_streaks(c, t, 0.9 * (1 - abs(px - ex) / 60), seed=9)
        for tb in (BUMP1, BUMP2):
            if tb <= t < tb + 0.5:
                xc, zc = self.pos_cl(tb)
                xx, zx = self.pos_cx(tb)
                self.sparks_burst(c, cam, t, tb, ((xc + xx) / 2 - 4, 8.0, (zc + zx) / 2), seed=int(tb * 10))
        if shot == 'jostle':
            for tb in (BUMP1, BUMP2):
                if tb + 0.05 <= t < tb + 0.55:
                    xc, zc = self.pos_cl(t)
                    xx, zx = self.pos_cx(t)
                    for (X, Y, Z) in ((xc + 4, 22.0, zc), (xx + 5, 31.0, zx)):
                        sp = world_to_screen(cam, (X, Y, Z))
                        if sp:
                            kk = max(1, int(round(cam.f / sp[2] * 0.4)))
                            spr = ANGER2 if int(t * 8) % 2 else ANGER
                            blit(c, scale_nn(spr, kk), sp[0] - 3 * kk, sp[1] - 3 * kk)
        if T_HIT <= t < 12.45:
            self.brawl(c, cam, t)
        if T_HIT + 0.04 <= t < T_HIT + 0.5:
            a = t - (T_HIT + 0.04)
            sp = world_to_screen(cam, (X_FIN - 4, 30.0, 0.0))
            if sp:
                pop_text(c, 'БАМ!', sp[0] + 50, sp[1] - 6, a, (255, 90, 70), scale=4, dur=0.1, over=2.0, slam=True,
                         outline=(255, 244, 220), shadow=(60, 10, 10))
        if 12.32 <= t < 12.9:
            self.shatter_fx(c, cam, t)
        if t >= 12.3:
            self.daze_fx(c, cam, t)
        if sk > 0:
            self.storm_fx(c, cam, t, sk)
        c *= vig_dither(0.22)
        if T_HIT <= t < T_HIT + 0.06:
            flash(c, 0.85)
        self.overlay_text(c, t, shot)
        if t < 1.35:
            flash(c, 0.9 * (1 - seg(t, 1.2, 1.35)))
        return c

    def sky_extras(self, c, cam, t):
        """солнце + облака-билборды для кадра «кубок в небе»."""
        tc = 11.55
        ce = self.camera(tc)[0]
        tx, ty, tz, _, _ = self.trophy_state(tc)
        dvec = np.array([tx, ty + 1.0, tz]) - cam_pos(ce)
        sd = dvec / np.linalg.norm(dvec) + np.array([0.012, 0.018, 0.0])
        sd = sd / np.linalg.norm(sd)
        P = cam_pos(cam) + sd * 3000
        sp = world_to_screen(cam, tuple(P))
        if sp:
            x, y = sp[0], sp[1]
            # блики объектива вдоль линии солнце -> центр кадра
            for kf, rr, colf in ((0.55, 5, (255, 210, 160)), (1.15, 9, (170, 220, 255)), (1.55, 4, (255, 170, 220)),
                                 (1.9, 12, (200, 255, 200))):
                fx = x + (W / 2 - x) * kf
                fy = y + (H / 2 - y) * kf
                dither_disc(c, fx, fy, rr, colf, 0.3)
            for i in range(12):
                ang = i * math.pi / 6 + t * 0.4
                r0, r1 = 20, 30 + 6 * (i % 2)
                line_fast(c, x + math.cos(ang) * r0, y + math.sin(ang) * r0, x + math.cos(ang) * r1,
                          y + math.sin(ang) * r1, (255, 246, 190), dither=0.6)
            dither_disc(c, x, y, 18, (255, 240, 170), 0.35)
            dither_disc(c, x, y, 13, (255, 244, 200), 1.0)
            dither_disc(c, x - 3, y - 3, 6, (255, 255, 240), 1.0)
        for (X, Y, Z, spr) in self.sky_clouds:
            bb_sprite(c, cam, spr, X, Y, Z, 0.4, anchor=(0.5, 0.5))

    def rider_or_jump(self, d, cm, t, X, Z):
        """Codex: стоит -> приседает -> прыжок на подлетевшую доску -> едет."""
        if t < 2.05:
            shadow_b(d, cm, X, Z, 8.0, 0.28)
        if t < 1.45:
            fr = int(t * 7) % 7
            draw_codex_b(d, cm, self.pix, self.cx, X, 0.0, Z, row=0, col=fr)
            if t > 1.3:
                self.board_fly_in(d, cm, t, X, Z)
            return
        if t < 1.62:
            draw_codex_b(d, cm, self.pix, self.cx, X, 0.0, Z, row=4, col=0)
            self.board_fly_in(d, cm, t, X, Z)
            return
        if t < 1.95:
            k = seg(t, 1.62, 1.95)
            y = math.sin(k * math.pi) * 14 + k * (self.board_y(t) + 5.8)
            col = 1 if k < 0.45 else 2
            self.board_fly_in(d, cm, t, X, Z)
            draw_codex_b(d, cm, self.pix, self.cx, X, y, Z, row=4, col=col, y_lift_px=3.0 if col == 1 else 5.0)
            return
        if t < 2.05:
            # приземление: чуть присел
            self.draw_rider(d, cm, t, X, Z, pose=(4, 3, 4.0))
            return
        pose = None
        ph = (t * 1.7) % 1.0
        if 0.62 < ph < 0.8:
            pose = (4, 1, 3.0)     # балансирует: руки в стороны
        if BUMP1 + 0.02 <= t < BUMP1 + 0.4:
            pose = (5, 1, 0.0)
        if BUMP2 + 0.02 <= t < BUMP2 + 0.35:
            pose = (5, 3, 0.0)
        self.draw_rider(d, cm, t, X, Z, pose=pose)

    def board_fly_in(self, d, cm, t, X, Z):
        """доска влетает справа и зависает под Codex"""
        k = ease_out(seg(t, 1.3, 1.62))
        bx = lerp(X + 120, X, k)
        hb = lerp(10, self.board_y(t), k)
        board = make_hoverboard(t)
        bb_sprite(d, cm, board, bx, hb, Z - 0.5, 1.25, anchor=(0.5, 1.0))
        if k < 1:
            # след
            for i in range(5):
                s = world_to_screen(cm, (bx + 8 + i * 6, hb + 2.5, Z - 0.5))
                if s:
                    line_fast(d, s[0], s[1], s[0] + 4, s[1], (142, 227, 240), dither=0.7 - i * 0.12)

    def draw_trophy_spin(self, d, cm, t, X, Y, Z, spin, tint=None):
        c = math.cos(spin)
        sx = max(0.12, abs(c))
        spr = squash(TROPHY, sx, 1.0)
        if c < 0:
            spr = spr[:, ::-1]
        bb_sprite(d, cm, spr, X, Y, Z, 2.0, anchor=(0.5, 0.5), tint=tint)
        a = (t * 2.5) % 1.0
        if a < 0.2:
            s_ = world_to_screen(cm, (X - 2, Y + 3, Z - 0.5))
            if s_:
                k = max(1, int(round(cm.f / s_[2] * 0.25)))
                spr2 = scale_nn(SPARK if 0.05 < a < 0.15 else SPARK_S, k)
                blit(d, spr2, s_[0] - spr2.shape[1] // 2, s_[1] - spr2.shape[0] // 2)

    # ============================================================ удар и последствия
    CL_LIE = (X_FIN - 14.0, -22.0)
    CX_LIE = (X_FIN - 10.0, 20.0)
    TR_LAND = (X_FIN - 34.0, 0.0)

    def brawl(self, c, cam, t):
        a = t - T_HIT
        tl = a if a < 0.45 else 0.45 + (a - 0.45) * 0.3     # после удара — рапид
        s_ = world_to_screen(cam, (X_FIN - 4, 9.0, 0.0))
        if s_ is None:
            return
        k = cam.f / s_[2]
        grow = pop_scale(a, 0.18)
        fade = 1.0 - seg(t, 12.15, 12.45)
        if fade <= 0:
            return
        r = 24 * k * grow * (0.55 + 0.45 * fade)
        if r < 3:
            return
        if fade < 1:
            tmp = c.copy()
            brawl_cloud(tmp, tl, s_[0], s_[1], r, seed=11)
            m = (np.abs(tmp - c).sum(-1) > 1) & (DITHER < fade)
            c[m] = tmp[m]
        else:
            brawl_cloud(c, tl, s_[0], s_[1], r, seed=11)

    def shatter_fx(self, c, cam, t):
        a = t - 12.32
        tx, tz = self.TR_LAND
        s_ = world_to_screen(cam, (tx, 1.0, tz))
        if s_ is None:
            return
        k = cam.f / s_[2]
        for i in range(12):
            ang = math.pi * (1.05 + hash01(i, 31) * 0.9)
            sp = (22 + hash01(i, 32) * 30) * k
            x = s_[0] + math.cos(ang) * sp * a * 1.6
            y = s_[1] + math.sin(ang) * sp * a + 70 * k * a * a
            col = (255, 214, 64) if i % 2 else (255, 246, 190)
            rect(c, x, y, max(1, k * 0.45), max(1, k * 0.45), col)
        if a < 0.45:
            for i in range(7):
                ang = math.pi * (1.0 + i / 6.0)
                dist = (5 + a * 24) * k
                puff(c, s_[0] + math.cos(ang) * dist * 1.4, s_[1] + math.sin(ang) * dist * 0.3, (2.4 - a * 3) * k,
                     (236, 220, 190), (190, 170, 140), alpha=1 - a / 0.45)
        if a < 0.34:
            pop_text(c, 'ХРЯСЬ!', s_[0], s_[1] - 22 * k * 0.5 - 16, a, (255, 236, 150), scale=2, dur=0.1,
                     outline=(60, 30, 8), shadow=(60, 30, 8))

    def aftermath_items(self, items, t, tint):
        """после удара: облако скрывает гонщиков; к ~12.4 проявляются оглушённые."""
        if t < 12.1:
            return
        dither_a = seg(t, 12.15, 12.42)
        cxp, czp = self.CL_LIE
        xxp, zxp = self.CX_LIE

        def cl(d, cm):
            twitch = 1 if int(t * 5) % 3 == 0 else 0
            draw_clawd_b(d, cm, self.cl, cxp, 0.0, czp, face='dizzy', rot=2, sy=1.0 - 0.06 * twitch,
                         dither_alpha=dither_a if dither_a < 1 else None)
        items.append((cxp, 0.0, czp, cl))
        items.append((cxp - 10, 0.0, czp - 9, lambda d, cm: bb_sprite(d, cm, self.helmet, cxp - 10, 0.0, czp - 9, 1.0)))

        def cx(d, cm):
            draw_codex_b(d, cm, self.pix, self.cx, xxp, 0.0, zxp, row=5, col=2, rot=1,
                         dither_alpha=dither_a if dither_a < 1 else None)
        items.append((xxp, 0.0, zxp, cx))
        board = np.ascontiguousarray(np.rot90(make_hoverboard(0.0)[:5], 1))
        items.append((xxp + 6, 0.0, zxp + 12,
                      lambda d, cm: bb_sprite(d, cm, board, xxp + 6, 0.0, zxp + 12, 1.25)))
        a = t - 12.32
        if a >= 0:
            tx, tz = self.TR_LAND
            for (spr, lie, dz) in ((self.half_l, self.half_l_lie, 7.0), (self.half_r, self.half_r_lie, -7.0)):
                k = min(1.0, a / 0.3)
                Z = tz + dz * ease_out(k)
                Y = max(0.0, math.sin(k * math.pi) * 6.0)
                img = spr if (a < 0.3 and int(a * 24) % 2 == 0) else lie
                if a >= 0.3:
                    img = lie
                items.append((tx, 0.0, Z, lambda d, cm, Z=Z, Y=Y, img=img: bb_sprite(d, cm, img, tx, Y, Z, 2.0)))

    def daze_fx(self, c, cam, t):
        for (X, Y, Z, ph) in ((self.CL_LIE[0], 12.0, self.CL_LIE[1], 0.0), (self.CX_LIE[0], 14.0, self.CX_LIE[1], 1.3)):
            s_ = world_to_screen(cam, (X, Y, Z))
            if s_ is None:
                continue
            k = cam.f / s_[2]
            for i in range(3):
                ang = t * 5.0 + ph + i * 2 * math.pi / 3
                x = s_[0] + math.cos(ang) * 9 * k
                y = s_[1] + math.sin(ang) * 2.5 * k
                spr = STAR5 if math.sin(ang) > 0 else SPARK_S
                blit(c, spr, x - spr.shape[1] // 2, y - spr.shape[0] // 2)

    def storm_fx(self, c, cam, t, sk):
        c *= np.asarray(1 - np.array([0.36, 0.34, 0.2]) * sk, np.float32)
        tb = 13.2
        if tb <= t < tb + 0.3:
            a = t - tb
            if a < 0.12 or 0.17 < a < 0.22:
                flash(c, 0.55 if a < 0.06 else 0.3, color=(215, 225, 255))
                lightning_bolt(c, 1234, 330, int(min(H - 40, max(40, cam.horizon_y() + 6))))

    # ============================================================ тексты
    def overlay_text(self, c, t, shot):
        for (t0, s_) in ((2.1, '3'), (2.4, '2'), (2.7, '1')):
            a = t - t0
            if 0 <= a < 0.28:
                pop_text(c, s_, W // 2, 214, a, (255, 236, 120), scale=7, dur=0.14, outline=(60, 20, 10),
                         shadow=(60, 20, 10))
        a = t - T_GO
        if 0 <= a < 0.75:
            if a < 0.55 or int(a * 20) % 2 == 0:
                pop_text(c, 'СТАРТ!', W // 2, 70, a, (120, 255, 140), scale=6, dur=0.12, over=1.8, slam=True,
                         outline=(10, 50, 20), shadow=(10, 50, 20))


SCENE = RaceScene()
