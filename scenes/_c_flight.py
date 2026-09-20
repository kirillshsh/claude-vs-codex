"""s6, шот 2: эпичный Mode-7 полёт на облачке в закат (агент C).

Мир: плоскость Y=0 с большой текстурой (поля-лоскуты, река, лес, деревушка, пляж, море), закатное небо с огромным
солнцем, дальние облака-панорама и 3D-облака-билборды, которые проносятся мимо камеры. Clawd — официальные кадры
Cloud-once (езда на облачке), Codex сидит сзади на том же облаке (экранно-согласованный сдвиг по оси «вправо» камеры).
"""
import math

import numpy as np

import bg
from px import *
from cam3d import *
from chars import anim_frame
from worlds import noise2d
from scenes._c_helpers import *

TEX_W, TEX_H = 1024, 2048            # X x Z, 1 тексел = 1 юнит
ORIGIN = (-512.0, -400.0)
COAST_Z = 330.0
Y_F = 70.0                            # высота полёта
SPEED = 138.0                         # юнит/с
SKY = [hexc(h) for h in ('#3a2a6e', '#5e3486', '#a24a84', '#e8706a', '#ff9e66', '#ffd08a')]
FOG = hexc('#f0a07c')
SUN_AZ, SUN_EL = 0.0, 0.05


def coast_z(X):
    return COAST_Z + 18 * np.sin(X / 70.0) + 9 * np.sin(X / 23.0 + 1.3)


def make_land_texture(seed=31):
    rng = np.random.default_rng(seed)
    h, w = TEX_H, TEX_W
    Zg, Xg = np.mgrid[0:h, 0:w].astype(np.float32)
    X = Xg + ORIGIN[0]
    Z = Zg + ORIGIN[1]
    tex = np.zeros((h, w, 3), np.float32)
    bay = BAYER4[Zg.astype(int) % 4, Xg.astype(int) % 4]
    # ---------------------------------------------------------------- поля-лоскуты
    kinds = [
        (hexc('#e6b45e'), hexc('#c8943e'), 'wheat'),
        (hexc('#a6c45e'), hexc('#86a44a'), 'green'),
        (hexc('#78a84c'), hexc('#5a8a3c'), 'green'),
        (hexc('#a07ab8'), hexc('#7e5a9a'), 'rows'),
        (hexc('#a8704a'), hexc('#84543a'), 'rows'),
        (hexc('#d89a5a'), hexc('#b8783e'), 'rows'),
        (hexc('#8cb850'), hexc('#6c983e'), 'dots'),
    ]
    z = 0
    while z < h:
        rh = int(rng.integers(26, 70))
        x = 0
        while x < w:
            fw = int(rng.integers(30, 110))
            base, alt, kind = kinds[int(rng.integers(0, len(kinds)))]
            sub = tex[z:z + rh, x:x + fw]
            sub[:] = base
            zz, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
            if kind == 'rows':
                m = (zz % 4) < 1 if rng.random() < 0.5 else (xx % 4) < 1
                sub[m] = alt
            elif kind == 'wheat':
                m = BAYER4[zz % 4, xx % 4] < 0.25
                sub[m] = alt
            elif kind == 'dots':
                m = ((zz % 5) == 2) & ((xx % 5) == 2)
                sub[m] = alt
            # живая изгородь по краю
            hedge = hexc('#3e6a3a')
            sub[:1, :] = hedge
            sub[:, :1] = hedge
            x += fw
        z += rh
    # ---------------------------------------------------------------- лес (шумовые пятна)
    n = noise2d(1024, 8, seed, 3)
    n = np.concatenate([n, n[::-1]], 0)  # 2048 по Z
    forest = n > 0.6
    tex[forest] = hexc('#2e5a34')
    tops = forest & (((Zg.astype(int) * 7 + Xg.astype(int) * 3) % 11) < 3)
    tex[tops] = hexc('#4a8a44')
    shad = forest & (((Zg.astype(int) * 7 + Xg.astype(int) * 3) % 11) == 4)
    tex[shad] = hexc('#1e4028')
    # ---------------------------------------------------------------- дороги
    road = hexc('#caa272')
    for (a, b, ph) in ((0.5, 90.0, 0.3), (-0.3, 140.0, 1.1)):
        xr = 120 * a + 40 * np.sin(Z / b + ph)
        m = np.abs(X - xr - Z * a * 0.2) < 2.0
        tex[m] = road
    # ---------------------------------------------------------------- река
    xr = 70 * np.sin(Z / 150.0) + 28 * np.sin(Z / 57.0 + 0.7) - 60
    wdt = 8 + 3 * np.sin(Z / 40.0)
    d = np.abs(X - xr)
    tex[(d < wdt + 2)] = hexc('#4a6a3a')                   # берега
    river = d < wdt
    tex[river] = hexc('#6a82c8')
    rip = river & (((Zg.astype(int) // 3 + (X - xr).astype(int)) % 9) == 0)
    tex[rip] = hexc('#f2b890')
    # ---------------------------------------------------------------- деревушки у реки
    for k in range(9):
        cz = rng.uniform(-300, 280)
        cxr = 70 * math.sin(cz / 150.0) + 28 * math.sin(cz / 57.0 + 0.7) - 60 + rng.choice([-1, 1]) * rng.uniform(22, 40)
        for j in range(int(rng.integers(5, 10))):
            hx = int(cxr + rng.uniform(-18, 18) - ORIGIN[0])
            hz = int(cz + rng.uniform(-14, 14) - ORIGIN[1])
            roof = hexc('#c8503a') if rng.random() < 0.6 else hexc('#e0845a')
            if 0 <= hz < h - 5 and 0 <= hx < w - 5:
                tex[hz:hz + 4, hx:hx + 5] = hexc('#f0e0c0')
                tex[hz:hz + 2, hx:hx + 5] = roof
                tex[hz + 4, hx:hx + 5] = hexc('#5a4030')
    # ---------------------------------------------------------------- пляж и море
    cz = coast_z(X)
    beach = (Z > cz - 16) & (Z <= cz)
    tex[beach] = hexc('#f2d49a')
    tex[beach & (bay < 0.2)] = hexc('#e0bc80')
    wet = (Z > cz - 5) & (Z <= cz)
    tex[wet] = hexc('#c8a47c')
    sea = Z > cz
    dz = np.clip((Z - cz) / 260.0, 0, 1)
    shallow, deep = hexc('#6e74b8'), hexc('#3e3676')
    nn = noise2d(1024, 24, seed + 5, 2)
    nn = np.concatenate([nn, nn[::-1]], 0)
    lv = np.clip(np.floor(dz * 5 + (nn - 0.5) * 1.2 + 0.5), 0, 5) / 5   # полосы глубины с рваными краями
    tex[sea] = (shallow[None, :] * (1 - lv[sea][:, None]) + deep[None, :] * lv[sea][:, None])
    surf = sea & (Z < cz + 3)
    tex[surf] = hexc('#fff2de')
    foam2 = sea & (Z > cz + 7) & (Z < cz + 9) & (np.sin(X / 6.0) > 0.2)
    tex[foam2] = hexc('#cfd0f0')
    # волны: случайные короткие горизонтальные штрихи (без регулярной сетки)
    r = rng.random((h, w))
    open_sea = sea & (Z > cz + 12)
    s1 = open_sea & (r < 0.010)
    s1 = s1 | np.roll(s1, 1, 1) | np.roll(s1, 2, 1) | np.roll(s1, 3, 1)
    tex[s1 & open_sea] = hexc('#8c7cc0')
    s2 = open_sea & (r > 0.9955)
    s2 = s2 | np.roll(s2, 1, 1) | np.roll(s2, 2, 1)
    tex[s2 & open_sea] = hexc('#f0a488')
    # тёплый закатный свет
    tex = tex * np.array([1.06, 0.95, 0.88], np.float32) + np.array([10, 2, 0], np.float32)
    return np.clip(tex, 0, 255).astype(np.uint8)


def _build():
    tex = make_land_texture()
    ground = Ground(tex, tpu=1.0, origin=ORIGIN, wrap=True)
    pal = bg.P('sunset')
    clouds_pano = make_cloud_pano(pal, seed=41, height=130, n=14)
    rng = np.random.default_rng(8)
    cloud_sprites = []
    cpal = [hexc('#ffd4a0'), hexc('#f8a88a'), hexc('#d07890'), hexc('#8a4a82')]
    for i in range(10):
        cloud_sprites.append(bg.make_cloud(int(rng.integers(1e6)), int(rng.integers(44, 92)), int(rng.integers(16, 28)),
                                           cpal))
    return dict(ground=ground, pano=clouds_pano, clouds=cloud_sprites)


class Flight:
    def __init__(self, cl, cx, pix):
        self.cl, self.cx, self.pix = cl, cx, pix
        d = cached('s6_flight', _build)
        self.ground, self.pano, self.cloud_sprites = d['ground'], d['pano'], d['clouds']
        rng = np.random.default_rng(3)
        # 3D-облака вдоль маршрута: (X, Y, Z, idx, ppu)
        self.clouds3d = []
        for i in range(34):
            side = rng.choice([-1, 1])
            X = side * rng.uniform(90, 360) + (60 if side > 0 else 0)
            Y = rng.uniform(20, 150)
            Z = rng.uniform(-60, 1300)
            self.clouds3d.append((X, Y, Z, int(rng.integers(0, len(self.cloud_sprites))), rng.uniform(0.4, 0.62)))
        # облака, которые проносятся ближе к камере (дают скорость), но не вплотную
        self.clouds3d.append((150.0, 50.0, 170.0, 2, 0.55))
        self.clouds3d.append((-40.0, 40.0, 300.0, 5, 0.5))
        self.clouds3d.append((170.0, 104.0, 360.0, 7, 0.5))
        self.ride = list(range(16, 50))
        self.ride_dur = self.cl.meta['Cloud-once']['durations'][16:50]
        self.anchor = self.cl.anchor['Cloud-once']
        g = np.random.default_rng(5).random((256, 256)).astype(np.float32)
        self.glit = g

    def pos(self, u):
        return (0.0, Y_F + 2.2 * math.sin(u * 2.6), 40.0 + SPEED * u)

    def camera(self, u, dur):
        P = self.pos(u)
        k = ease_in_out(u / dur)
        off_e = lerp3((74.0, 9.0, -20.0), (42.0, 22.0, -96.0), k)
        off_t = lerp3((0.0, 8.0, 20.0), (-6.0, 1.0, 92.0), k)
        eye = (P[0] + off_e[0], Y_F + off_e[1], P[2] + off_e[2])
        tgt = (P[0] + off_t[0], Y_F + off_t[1], P[2] + off_t[2])
        return look_at(eye, tgt, lerp(250.0, 262.0, k))

    def render(self, u, dur):
        cam = self.camera(u, dur)
        c = canvas()
        sky_dome(c, cam, SKY, span=0.62)
        self.sun(c, cam, u)
        self.pano.draw(c, cam, drift=u * 3.0, y_offset=-6, tint=(1.0, 0.95, 0.95))
        valid, tpar = self.ground.render(c, cam, fog=FOG, fog_near=240, fog_far=1500, max_dist=4000)
        self.glitter(c, cam, u, valid)
        items = []
        for (X, Y, Z, idx, ppu) in self.clouds3d:
            spr = self.cloud_sprites[idx]

            def fn(d, cm, spr=spr, X=X, Y=Y, Z=Z, ppu=ppu):
                p = cm.project(X, Y, Z)
                if p is None:
                    return
                # вплотную к камере облако растворяется (иначе слишком крупные «ступеньки»)
                a = clamp((p[2] - 45.0) / 45.0)
                if a <= 0:
                    return
                draw_spr(d, cm, spr, (spr.shape[1] / 2, spr.shape[0] / 2), X, Y, Z, ppu, fog=FOG, fog_near=300,
                         fog_far=1400, dither=a if a < 1 else None)
            items.append((X, Y, Z, fn))
        P = self.pos(u)

        def riders(d, cm):
            self.draw_riders(d, cm, u, P)
        items.append((P[0], P[1], P[2], riders))
        draw_world(c, cam, items)
        # лёгкие штрихи скорости у краёв кадра
        bg.speed_lines(c, u, 20, H - 20, n=6, seed=4, speed=900, color=(255, 236, 214), alpha=0.35, length=(10, 26))
        vignette_d(c, 0.35, inner=0.62)
        return c

    def draw_riders(self, d, cm, u, P):
        right = (math.cos(cm.yaw), 0.0, -math.sin(cm.yaw))
        fwd = (math.sin(cm.yaw), 0.0, math.cos(cm.yaw))
        # Codex сзади (слева на экране), чуть дальше по глубине, ниже — «сидит» в облаке
        off = -14.0
        bob = 0.6 * math.sin(u * 5.2 + 1.0)
        Xc = P[0] + right[0] * off + fwd[0] * 4.0
        Zc = P[2] + right[2] * off + fwd[2] * 4.0
        col = 2 if (u % 1.6) < 0.5 else (1 if int(u * 5) % 2 == 0 else 5)
        draw_codex(d, cm, self.pix, self.cx, Xc, Zc, 8, col, Y=P[1] + 5.5 + bob, tint=(1.04, 0.95, 0.92))
        i = anim_frame(u, self.ride_dur)
        spr = self.cl.frames['Cloud-once'][self.ride[i]]
        draw_spr(d, cm, spr, self.anchor, P[0], P[1], P[2], 1.0, tint=(1.04, 0.97, 0.94))

    def sun(self, c, cam, u):
        p = sky_xy(cam, SUN_AZ, SUN_EL)
        if p is None:
            return
        sx, sy = p
        r = cam.f * 0.085
        for (rr, col, a) in ((r * 2.6, (255, 176, 110), 0.22), (r * 1.9, (255, 192, 120), 0.38),
                             (r * 1.35, (255, 214, 140), 0.62)):
            bg.blit_dither_disc(c, sx, sy, rr, col, a)
        disc(c, sx, sy, r, (255, 208, 120))
        disc(c, sx, sy - r * 0.12, r * 0.84, (255, 232, 160))
        disc(c, sx, sy - r * 0.25, r * 0.6, (255, 246, 206))
        # горизонтальные полоски облаков поперёк солнца
        for (dy, L, col) in ((r * 0.25, 2.6, (206, 108, 128)), (r * 0.6, 1.8, (168, 82, 120)),
                             (-r * 0.35, 1.4, (232, 140, 130))):
            y = int(sy + dy)
            x0 = int(sx - r * L + math.sin(u * 0.7 + dy) * 4)
            x1 = int(sx + r * L * 0.8 + math.sin(u * 0.7 + dy) * 4)
            if 0 <= y < H - 1:
                c[y:y + 2, max(0, x0):min(W, x1)] = col

    def glitter(self, c, cam, u, valid):
        """солнечная дорожка на море: искры в полосе под солнцем."""
        p = sky_xy(cam, SUN_AZ, SUN_EL)
        if p is None:
            return
        sx = p[0]
        hy = cam.horizon_y()
        dx, dy, dz = cam.rays()
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -cam.y / dy
        X = cam.x + t * dx
        Z = cam.z + t * dz
        sea = valid & (Z > coast_z(X) + 6)
        depth = np.clip((YY - hy) / max(1.0, H - hy), 0, 1)
        band = np.abs(XX - sx) < (5 + depth * 46) * (0.8 + 0.2 * np.sin(YY * 0.9 + u * 3))
        gi = self.glit[(np.nan_to_num(Z) * 0.5).astype(int) % 256, (np.nan_to_num(X) * 0.35).astype(int) % 256]
        tw = np.sin(gi * 40 + u * 9.0) > 0.9 - 0.12 * (1 - depth)
        m = sea & band & tw & (YY > hy + 1)
        c[m] = (255, 236, 190)
        m2 = sea & band & (np.sin(gi * 40 + u * 9.0 + 1.5) > 0.9) & (YY > hy + 1)
        c[m2] = (255, 206, 150)
