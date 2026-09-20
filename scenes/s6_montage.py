"""s6_montage — монтаж дружбы (7.0 c).

Шоты:
  1  0.0–3.2  уютный ночной стол: оба за ОДНИМ ноутбуком (Clawd печатает, Codex row 7), парящий терминал
              «$ git merge codex» → «✓ MERGED», сердечки; медленный долли-проезд
  2  3.2–7.0  эпичный Mode-7 полёт: Clawd на облачке (официальные кадры Cloud-once), Codex сзади на том же облаке;
              поля, река, берег, море, облака проносятся мимо; летят в огромное закатное солнце
"""
import math

import numpy as np

import bg
from px import *
from cam3d import *
from chars import load, anim_frame
from scene_base import Scene
from scenes._c_helpers import *
from scenes import _c_desk as desk
from scenes import _c_flight as flight

T_CUT, T_END = 3.2, 7.0

CODEX_D = (-9.0, 2.0)       # Codex за ноутбуком
CLAWD_D = (13.5, 4.0)       # Clawd справа, печатает (кадры отражены)
LAMP_P = (-52.0, 24.0)
MUG_P = (46.0, -8.0)
BOOKS_P = (70.0, 20.0)
CACTUS_P = (-80.0, -2.0)
LIGHT_C = (-50.0, 12.0)     # центр светового пятна лампы на столе
NIGHT = [hexc(h) for h in ('#0a0c22', '#10163a', '#1a2350', '#28346c', '#35427e')]
WARM = (1.0, 0.93, 0.86)


def _strip_laptop(f):
    """убрать из кадра печатающего Clawd его собственный ноутбук (чёрные пиксели справа от тела)."""
    g = f.copy()
    blk = (g[:, :, 3] > 0) & (g[:, :, :3].astype(int).sum(2) < 40)
    xs = np.arange(g.shape[1])[None, :]
    g[blk & (xs >= 22), 3] = 0
    return g


def _build_desk():
    tex = desk.make_desk_texture()
    ground = Ground(tex, tpu=2.0, origin=(desk.DESK_X0, desk.DESK_Z0), wrap=False, border=(40, 26, 20))
    return dict(ground=ground, wall=desk.make_wall(), skyline=desk.make_skyline(), moon=desk.make_moon())


class S6Montage(Scene):
    name = 's6_montage'
    dur = T_END

    def __init__(self):
        self.cl, self.cx = load()
        self.pix = CodexPixelizer()
        d = cached('s6_desk', _build_desk)
        self.ground, self.wall, self.skyline, self.moon = d['ground'], d['wall'], d['skyline'], d['moon']
        self.lights = desk.fairy_lights()
        rng = np.random.default_rng(21)
        n = 160
        self.star_az = rng.uniform(-1.0, 1.1, n)
        self.star_el = rng.uniform(0.04, 0.95, n) ** 0.8
        self.star_ph = rng.uniform(0, 6.28, n)
        self.star_big = rng.random(n) < 0.15
        lap = self.cl.frames['Laptop']
        self.typing = [_strip_laptop(f) for f in lap[16:35]]
        self.typing_dur = self.cl.meta['Laptop']['durations'][16:35]
        self.clawd_anchor = self.cl.anchor['Laptop']
        self.fl = None

    def render(self, t):
        if t < T_CUT:
            return self.shot_desk(t)
        if self.fl is None:
            self.fl = flight.Flight(self.cl, self.cx, self.pix)
        return self.fl.render(t - T_CUT, T_END - T_CUT)

    # ------------------------------------------------------------------ шот 1: ночной стол
    def lamp_shade(self, cam):
        def shade(out, X, Z, tdist):
            d = np.hypot(X - LIGHT_C[0], (Z - LIGHT_C[1]) * 1.15)
            v = np.clip(1.18 - d / 105.0, 0.34, 1.18)
            v = np.floor(v * 10 + DITHER * 1.0) / 10
            warm = np.stack([v * 1.0, v * 0.9, v * 0.78], -1).astype(np.float32)
            # экран ноутбука подсвечивает стол перед ним голубым
            d2 = np.hypot(X - (CODEX_D[0] + 10), (Z - CODEX_D[1] + 6) * 1.3)
            b = np.clip(1 - d2 / 22.0, 0, 1)
            b = np.floor(b * 3 + DITHER) / 3 * 0.18
            out = out * warm
            out += np.stack([b * 40, b * 70, b * 110], -1)
            return out
        return shade

    def shot_desk(self, u):
        k = ease_in_out(u / T_CUT)
        cam = look_at(lerp3((-26.0, 15.0, -68.0), (7.0, 12.5, -55.0), k), lerp3((-3.0, 15.0, 0.0), (1.0, 16.5, 0.0), k),
                      lerp(262.0, 290.0, k))
        c = canvas()
        sky_dome(c, cam, NIGHT, span=0.8)
        # звёзды
        sx, sy = self.sky_points(cam, self.star_az, self.star_el)
        tw = 0.5 + 0.5 * np.sin(u * 3.0 + self.star_ph)
        ok = np.isfinite(sx) & (sx >= 1) & (sx < W - 1) & (sy >= 1) & (sy < H - 1) & (tw > 0.2)
        cols = np.where((tw[ok] > 0.7)[:, None], np.array([[255, 246, 214]], np.float32),
                        np.array([[150, 150, 190]], np.float32))
        xi, yi = sx[ok].astype(int), sy[ok].astype(int)
        c[yi, xi] = cols
        big = self.star_big[ok] & (tw[ok] > 0.75)
        for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c[yi[big] + dy, xi[big] + dx] = (170, 170, 210)
        mp = sky_xy(cam, 0.2, 0.36)
        if mp is not None:
            blit(c, self.moon, mp[0] - self.moon.shape[1] / 2, mp[1] - self.moon.shape[0] / 2)
        # город — подтягиваем к низу окна
        wb = cam.project(0.0, desk.WIN[2], desk.WALL_Z)
        if wb is not None:
            self.skyline.draw(c, cam, y_offset=wb[1] - cam.horizon_y() + 3)
        self.ground.render(c, cam, shade=self.lamp_shade(cam), max_dist=600)
        # стена (окно прозрачное)
        draw_spr(c, cam, self.wall, (-desk.WALL_X0, self.wall.shape[0]), 0.0, 0.0, desk.WALL_Z, 1.0,
                 tint=(1.0, 0.97, 1.06))
        # тёплый свет лампы на стене: дизер-пятно
        lw = cam.project(LAMP_P[0], 30.0, desk.WALL_Z - 0.2)
        if lw is not None:
            rr = cam.f / lw[2] * 70
            d = np.hypot(XX - lw[0], (YY - lw[1]) * 1.2) / rr
            wall_m = YY < (cam.project(0.0, 0.0, desk.WALL_Z) or (0, H))[1]
            q = np.clip(1 - d, 0, 1)
            q = np.floor(q * 4 + DITHER) / 4
            m = wall_m & (q > 0)
            c[m] = c[m] * (1 + 0.28 * q[m, None]) + np.array([26, 14, 2], np.float32) * q[m, None]
        # гирлянда
        pts, lcols = self.lights
        for i, ((X, Y, Z), lc) in enumerate(zip(pts, lcols)):
            p = cam.project(X, Y, Z)
            if p is None:
                continue
            on = 0.55 + 0.45 * math.sin(u * 4.0 + i * 1.3)
            rect(c, p[0] - 1, p[1] - 1, 2, 2, lc * (0.6 + 0.4 * on))
            if on > 0.6:
                bg.blit_dither_disc(c, p[0], p[1], 2.4, lc * 0.8, 0.35 * on)
                pset(c, p[0], p[1], (255, 255, 240))
        # провод гирлянды
        for a_, b_ in zip(pts[:-1], pts[1:]):
            pa, pb = cam.project(*a_), cam.project(*b_)
            if pa is not None and pb is not None:
                line(c, pa[0], pa[1] - 2, pb[0], pb[1] - 2, (30, 24, 40))
        # реквизит и персонажи
        items = []

        def prop(spr, X, Z, ppu, tint=None):
            def fn(d, cm):
                ground_shadow(d, cm, X, Z, spr.shape[1] / ppu * 0.45, alpha=0.3)
                draw_spr(d, cm, spr, (spr.shape[1] / 2, spr.shape[0]), X, 0.0, Z, ppu, tint=tint)
            items.append((X, 0.0, Z, fn))
        prop(desk.LAMP, LAMP_P[0], LAMP_P[1], 0.8)
        prop(desk.MUG, MUG_P[0], MUG_P[1], 1.0, tint=WARM)
        prop(desk.BOOKS, BOOKS_P[0], BOOKS_P[1], 1.0, tint=(0.9, 0.86, 0.9))
        prop(desk.CACTUS, CACTUS_P[0], CACTUS_P[1], 0.9, tint=WARM)
        TYPE0, TYPE1, MERGE = 0.62, 1.72, 1.9
        ui = {}

        def codex(d, cm):
            ground_shadow(d, cm, CODEX_D[0] + 3, CODEX_D[1], 12, alpha=0.35)
            if u < MERGE:
                col = [0, 4][int(u * 7) % 2] if TYPE0 <= u < TYPE1 else (3 if (u % 2.2) < 0.15 else 0)
            else:
                col = [1, 5][int(u * 4) % 2]
            ui['cx'] = draw_codex(d, cm, self.pix, self.cx, CODEX_D[0], CODEX_D[1], 7, col, tint=WARM)
        items.append((CODEX_D[0], 0.0, CODEX_D[1], codex))

        def clawd(d, cm):
            ground_shadow(d, cm, CLAWD_D[0], CLAWD_D[1], 11, alpha=0.35)
            if u < MERGE + 0.15:
                i = anim_frame(u, self.typing_dur)
                spr = self.typing[i]
                info, _ = draw_spr(d, cm, spr, self.clawd_anchor, CLAWD_D[0], 0.0, CLAWD_D[1], 1.0, flipx=True,
                                   tint=WARM)
            else:
                q = u - (MERGE + 0.15)
                idx = anim_frame(q, self.cl.meta['JumpingHappy']['durations'], loop=False, start=4, end=16)
                spr = self.cl.frames['JumpingHappy'][idx]
                info, _ = draw_spr(d, cm, spr, self.cl.anchor['JumpingHappy'], CLAWD_D[0] + 2, 0.0, CLAWD_D[1], 1.0,
                                   tint=WARM)
            ui['cl'] = info
        items.append((CLAWD_D[0], 0.0, CLAWD_D[1], clawd))
        draw_world(c, cam, items)
        # пар от кружки
        mp_ = cam.project(MUG_P[0], 10.0, MUG_P[1])
        if mp_ is not None:
            bg.steam(c, u, mp_[0] - 1, mp_[1], seed=2, n=2, color=(200, 196, 214))
        # конус света лампы (дизер)
        self.lamp_cone(c, cam)
        # терминал
        self.terminal(c, cam, u, TYPE0, TYPE1, MERGE)
        # сердечки после MERGED
        if u >= MERGE + 0.2:
            p = cam.project(3.0, 17.0, 2.0)
            if p is not None:
                bg.hearts(c, u - (MERGE + 0.2), p[0], p[1], n=8, seed=5, interval=0.13, life=1.2, rise=40,
                          spread=26, sway=6, small_every=3)
        vignette_d(c, 0.42, inner=0.58)
        return c

    def sky_points(self, cam, az, el):
        X = cam.x + np.sin(az) * 1000 * np.cos(el)
        Z = cam.z + np.cos(az) * 1000 * np.cos(el)
        Y = cam.y + np.sin(el) * 1000
        sx, sy, zc = project_pts(cam, X, Y, Z)
        return sx, sy

    def lamp_cone(self, c, cam):
        top = cam.project(LAMP_P[0], 18.5, LAMP_P[1])
        if top is None:
            return
        s = cam.f / top[2]
        w0 = 7.0 * s
        ys = np.arange(int(top[1]), min(H, int(top[1] + 17.5 * s)))
        if len(ys) == 0:
            return
        for y in ys:
            q = (y - top[1]) / (17.5 * s)
            hw = w0 + q * 16 * s
            x0, x1 = int(max(0, top[0] - hw)), int(min(W, top[0] + hw))
            if x0 >= x1:
                continue
            a = 0.22 * (1 - q * 0.6)
            m = DITHER[y, x0:x1] < a
            row = c[y, x0:x1]
            row[m] = row[m] * 0.75 + np.array([255, 230, 170], np.float32) * 0.25

    def terminal(self, c, cam, u, T0, T1, MERGE):
        POP = 0.45
        if u < POP:
            return
        cmd = '$ git merge codex'
        n = int(clamp((u - T0) / (T1 - T0)) * len(cmd)) if u >= T0 else 0
        line1 = cmd[:max(0, n)]
        if u < MERGE:
            cur = '_' if int(u * 4) % 2 == 0 else ' '
            lines = [line1 + cur, '']
            colors = [(220, 230, 250), (120, 240, 140)]
        else:
            lines = [cmd, '\x01MERGED']
            colors = [(220, 230, 250), (120, 240, 140)]
        flash_b = MERGE <= u < MERGE + 0.3 and int((u - MERGE) * 20) % 2 == 0
        spr = terminal_sprite(lines, width_chars=18, colors=colors, bg_col=(16, 20, 38), title_col=(52, 58, 92),
                              border=(150, 255, 170) if flash_b else (126, 150, 220))
        p = cam.project(1.0, 31.5, 2.0)
        if p is None:
            return
        s = pop_scale(u - POP, 0.3)
        if s < 0.999:
            spr = squash(spr, s, s)
        x = p[0] - spr.shape[1] / 2
        y = p[1] - spr.shape[0] + math.sin(u * 2.4) * 1.5
        # тень-подложка (дизер) и свечение рамки после MERGED
        sh = np.zeros_like(spr)
        sh[:, :, 3] = spr[:, :, 3]
        bg.blit_dither(c, np.dstack([np.zeros(spr.shape[:2] + (3,), np.uint8), spr[:, :, 3:]]), x + 3, y + 3, 0.5)
        blit(c, spr, x, y)
        if MERGE <= u < MERGE + 0.3:
            # вспышка-рамка вокруг окна терминала (текст не трогаем — читается)
            k = 1 - (u - MERGE) / 0.3
            h_, w_ = spr.shape[:2]
            g = np.zeros((h_ + 4, w_ + 4, 4), np.uint8)
            g[:, :, :3] = (120, 255, 150)
            g[:, :, 3] = 255
            g[2:-2, 2:-2, 3] = 0
            bg.blit_dither(c, g, x - 2, y - 2, 0.9 * k)


SCENE = S6Montage()
