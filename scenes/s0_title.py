"""s0 — титульная карточка (4.0 с).
Закатный луг в 2.5D, камера делает «долли-зум» (отъезд + наезд трансфокатором: герои стоят на месте,
а солнце, горы и земля «наваливаются» сзади) и опускается. CLAWD влетает слева, CODEX — справа,
VS врезается в экран со вспышкой и молнией, маскоты под именами сверлят друг друга взглядом.
render.py: fade-in 0.5 с, iris-out 0.6 с в конце."""
import math
import numpy as np
from px import *
import bg
from cam3d import *
from scene_base import Scene
from scenes._a_helpers import *

T_CLAWD, T_CLAWD_HIT = 0.50, 0.70
T_CODEX, T_CODEX_HIT = 0.84, 1.04
T_VS, T_VS_HIT = 1.26, 1.38
T_SUB = 1.90
CLX, CDX = -28.0, 28.0
CLZ, CDZ = -12.0, 8.0
NAME_Y = 40
VS_Y = 124


class Title(Scene):
    name = 's0_title'
    dur = 4.0

    def __init__(self):
        self.w = get_world('sunset')
        self.pix = get_pix()
        # на экране персонаж — «CLAUDE» (внутренние имена кода — Clawd)
        t1 = title_sprite('CLAUDE', hexc('#E8835E'), hexc('#FFB08E'), hexc('#CC6644'), hexc('#7C3420'),
                          hexc('#1E0A06'), scale=3, depth=1, gloss=hexc('#FFE8DA'))
        t2 = title_sprite('CODEX', hexc('#5A7CFF'), hexc('#9AB2FF'), hexc('#4466E6'), hexc('#1A2A86'),
                          hexc('#070A26'), scale=3, depth=1, gloss=hexc('#E8EEFF'))
        self.C1 = name_plate(t1, hexc('#D97757'), dark=hexc('#1A0E22'), edge=hexc('#F2A07E'), lean_right=True)
        self.C2 = name_plate(t2, hexc('#4A6CF0'), dark=hexc('#0E1024'), edge=hexc('#86A2FF'), lean_right=False)
        self.VS = title_sprite('VS', hexc('#FFD84A'), hexc('#FFF6B0'), hexc('#FF9A2E'), hexc('#B8341A'),
                               hexc('#1A0404'), scale=5, depth=1, gloss=hexc('#FFFFFF'))
        self.BURST = starburst(44, 30, n=11, fill=hexc('#9E1B22'), inner=hexc('#D2361F'), ol=hexc('#2A0608'))
        self.sub_full = tr('КТО ЖЕ ЛУЧШЕ?')
        rng = np.random.default_rng(12)
        self.stars = [(rng.uniform(-0.75, 0.75), rng.uniform(0.2, 0.62), rng.uniform(0, 6.28), rng.uniform(2, 5),
                       rng.random() < 0.18) for _ in range(110)]
        self.mount_drift = 1003
        self.sun_r = 0.082
        self.sun_y = 110.0
        # деревья: убираем всё, что попадает в центральный конус взгляда (там солнце и VS)
        self.trees = []
        for it in self.w.tree_items():
            X, _, Z, fn = it
            if Z > 0 and abs(X) < Z * 0.62 + 30:
                continue
            self.trees.append(it)

    # ------------------------------------------------------------------ камера: долли-зум + кран вниз
    def cam(self, t):
        u = ease_in_out(t / self.dur)
        d = lerp(70.0, 132.0, u)
        f = 300.0 * d / 70.0
        h = lerp(7.5, 3.8, u)
        return look_at((0.0, h, -d), (0.0, 10.0, 0.0), f)

    def far_point(self, cam, az, el):
        D = 1e5
        return cam.project(cam.x + math.sin(az) * math.cos(el) * D, cam.y + math.sin(el) * D,
                           cam.z + math.cos(az) * math.cos(el) * D)

    # ------------------------------------------------------------------ небо
    def draw_sky(self, c, cam, t):
        w = self.w
        sky_dome(c, cam, w.sky, span=0.62, below=w.fog)
        # звёзды в тёмной части
        for (az, el, ph, sp, big) in self.stars:
            p = self.far_point(cam, az, el)
            if p is None:
                continue
            x, y = p[0], p[1]
            if not (0 <= x < W and 0 <= y < H):
                continue
            fade_el = clamp((el - 0.2) / 0.2)
            b = (0.55 + 0.45 * math.sin(t * sp + ph)) * fade_el
            if b < 0.25:
                continue
            col = (255, 240, 220)
            if b > 0.5 or DITHER[int(y), int(x)] < b:
                pset(c, x, y, col)
            if big and b > 0.75:
                for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    pset(c, x + dx, y + dy, (255, 214, 170))
        # солнце с ореолом (за горами): экранная высота закреплена за VS, размер — угловой (растёт при зуме)
        el = math.atan((cam.horizon_y() - self.sun_y) / cam.f) - cam.pitch * 0.0
        p = self.far_point(cam, 0.0, el)
        if p is not None:
            sx, sy = p[0], p[1]
            r = cam.f * self.sun_r
            self.draw_sun(c, sx, sy, r, t)
        w.clouds.draw(c, cam, drift=t * 5.0 + 300, y_offset=-10)
        w.mount.draw(c, cam, drift=self.mount_drift, y_offset=1)
        w.hills_far.draw(c, cam, drift=40, y_offset=2)
        w.hills.draw(c, cam, drift=80, y_offset=3)

    def draw_sun(self, c, sx, sy, r, t):
        R = r * 2.6
        x0, x1 = int(max(0, sx - R)), int(min(W, sx + R + 1))
        y0, y1 = int(max(0, sy - R)), int(min(H, sy + R + 1))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2) / r
        sub = c[y0:y1, x0:x1]
        dth = DITHER[y0:y1, x0:x1]
        # ореол: три кольца с дизером
        for (rad, colr, a) in ((2.6, (255, 150, 110), 0.35), (1.9, (255, 190, 120), 0.55), (1.35, (255, 226, 150), 0.8)):
            m = (d < rad) & (dth < a * np.clip((rad - d) / 0.5, 0, 1))
            sub[m] = sub[m] * 0.35 + np.asarray(colr, np.float32) * 0.65
        body = d <= 1.0
        sub[body] = (255, 236, 170)
        sub[(d <= 0.78)] = (255, 248, 214)
        # «закатные» полосы внизу диска
        rel = (yy - sy) / r
        for k, yb in enumerate((0.25, 0.5, 0.72)):
            m = body & (rel > yb) & (rel < yb + 0.06 + 0.04 * k)
            sub[m] = (255, 170, 110)

    # ------------------------------------------------------------------ персонажи
    def clawd_state(self, t):
        face = 'idle'
        if 0.30 < t < 0.40:
            face = 'blink'
        Y, sx, sy = 0.0, 1.0, 1.0
        if t >= T_CLAWD_HIT:
            face = 'angry'
            a = t - T_CLAWD_HIT
            if a < 0.30:
                Y = 6.0 * math.sin(math.pi * a / 0.30)
                sx, sy = (0.9, 1.12) if a < 0.12 else (1.0, 1.0)
            elif a < 0.40:
                sx, sy = 1.14, 0.84
        if t >= T_VS_HIT:
            a = t - T_VS_HIT
            if a < 0.22:
                Y = 4.0 * math.sin(math.pi * a / 0.22)
            elif a < 0.30:
                sx, sy = 1.12, 0.88
            # сердитое «пыхтение»
            k = (a * 3.2) % 1.0
            if a > 0.5 and k < 0.12:
                sx, sy = 1.06, 0.94
            # два злых подскока, чтобы пауза не была статичной
            for tb in (2.12, 2.88):
                Y = max(Y, hop_y(t, tb, 0.24, 3.4))
                if tb + 0.24 <= t < tb + 0.32:
                    sx, sy = 1.12, 0.86
            sy *= breathe(t, 1.3, 0.02)
        return face, Y, sx, sy

    def codex_state(self, t):
        row, col = 0, 0
        if 0.18 < t < 0.28:
            col = 1
        Y, sx, sy = 0.0, 1.0, 1.0
        if t >= T_CODEX_HIT:
            row = 5
            a = t - T_CODEX_HIT
            col = 1 if int(a / 0.18) % 2 == 0 else 5
            if a < 0.30:
                Y = 6.0 * math.sin(math.pi * a / 0.30)
            elif a < 0.40:
                sx, sy = 1.1, 0.88
        if t >= T_VS_HIT:
            a = t - T_VS_HIT
            if a < 0.22:
                Y = 4.0 * math.sin(math.pi * a / 0.22)
            elif a < 0.30:
                sx, sy = 1.1, 0.9
            for tb in (2.30, 3.02):
                Y = max(Y, hop_y(t, tb, 0.24, 3.4))
                if tb + 0.24 <= t < tb + 0.32:
                    sx, sy = 1.12, 0.88
            sy *= breathe(t, 1.15, 0.02, 1.7)
        return row, col, Y, sx, sy

    # ------------------------------------------------------------------ титры
    def slide_x(self, t, t0, t1, x_from, x_to):
        if t < t0:
            return None
        p = seg(t, t0, t1)
        return lerp(x_from, x_to, ease_out_back(p, 1.2) if p < 1 else 1.0)

    def draw_name(self, c, spr, t, t0, t1, x_from, x_to, trail_col, direction):
        x = self.slide_x(t, t0, t1, x_from, x_to)
        if x is None:
            return
        s = spr
        a = t - t1
        if 0 <= a < 0.10:
            s = squash(spr, 0.86, 1.16)
        elif 0.10 <= a < 0.18:
            s = squash(spr, 1.06, 0.95)
        if 2.30 < t < 2.75:
            s = shine_sweep(s, seg(t, 2.30, 2.75), width=9, color=(255, 255, 240))
        h, w = s.shape[:2]
        moving = t < t1
        if moving:
            p = seg(t, t0, t1)
            v = (x_to - x_from) * 0.12 * (1 - p)
            for i, al in enumerate((0.5, 0.28, 0.12)):
                bg.blit_dither(c, spr, x - w / 2 - v * (i + 1) * 1.4, NAME_Y - h / 2, al)
        # штрихи скорости
        if t < t1 + 0.16:
            k = 1 - clamp((t - t1) / 0.16) if t >= t1 else 1.0
            rng = np.random.default_rng(int(t0 * 100))
            for i in range(7):
                yy = NAME_Y - h / 2 + 3 + rng.uniform(0, h - 6)
                L = rng.uniform(30, 110) * k
                xe = x - direction * (w / 2 + 2)
                line(c, xe, yy, xe - direction * L, yy, trail_col, 0.9)
        blit(c, s, x - w / 2, NAME_Y - h / 2)
        # искры от удара на переднем крае надписи
        if 0 <= a < 0.28:
            rng = np.random.default_rng(int(t1 * 1000))
            ex = x + direction * (w / 2)
            for i in range(6):
                ang = rng.uniform(-1.2, 1.2)
                sp = rng.uniform(40, 110)
                px_ = ex + direction * math.cos(ang) * sp * a
                py_ = NAME_Y + math.sin(ang) * sp * a * 0.8
                spr_ = SPARK if a < 0.12 else SPARK_S
                blit(c, spr_, px_ - spr_.shape[1] // 2, py_ - spr_.shape[0] // 2)

    def draw_vs(self, c, t):
        if t < T_VS:
            return
        if t < T_VS_HIT:
            p = seg(t, T_VS, T_VS_HIT)
            k = lerp(4.2, 1.0, ease_in(p))
        else:
            a = t - T_VS_HIT
            k = 1.0 + 0.16 * math.exp(-a * 14) * math.cos(a * 40)
        # удар: молния сверху в VS и лучи-вспышка
        if T_VS_HIT <= t < T_VS_HIT + 0.20 or (T_VS_HIT + 0.62 <= t < T_VS_HIT + 0.72):
            bolt(c, 236 + (8 if t > T_VS_HIT + 0.5 else -14), -4, 240, VS_Y - 22, seed=int(t * 20) + 5, jag=12.0,
                 step=18.0, thick=2, core=(255, 255, 245), glow=(255, 200, 90))
        if T_VS_HIT <= t < T_VS_HIT + 0.28:
            a = (t - T_VS_HIT) / 0.28
            rng = np.random.default_rng(77)
            for i in range(22):
                ang = i / 22 * 2 * math.pi + rng.uniform(-0.08, 0.08)
                r0 = 46 + a * 70 + rng.uniform(0, 14)
                r1 = r0 + (1 - a) * rng.uniform(30, 70)
                ca, sa = math.cos(ang), math.sin(ang) * 0.62
                line(c, 240 + ca * r0, VS_Y + sa * r0, 240 + ca * r1, VS_Y + sa * r1,
                     (255, 250, 210) if i % 2 else (255, 200, 90))
        b = self.BURST
        spin_k = 1.0 + 0.04 * math.sin(t * 5.0)
        kb = k * spin_k
        if abs(kb - 1) > 0.01:
            b = squash(b, kb, kb)
        bh, bw = b.shape[:2]
        blit(c, b, 240 - bw / 2, VS_Y - bh / 2 + 1)
        s = self.VS
        if 2.55 < t < 2.95:
            s = shine_sweep(s, seg(t, 2.55, 2.95), width=12, color=(255, 255, 255))
        if abs(k - 1) > 0.01:
            s = squash(s, k, k)
        h, w = s.shape[:2]
        blit(c, s, 240 - w / 2, VS_Y - h / 2)
        # искорки вокруг VS
        if t > T_VS_HIT:
            for i, (ox, oy, per) in enumerate(((-52, -22, 0.9), (50, -26, 1.1), (46, 24, 0.8), (-48, 20, 1.3))):
                bg.glint(c, 240 + ox, VS_Y + oy, t, period=per, offset=i * 0.37)

    def draw_sub(self, c, t):
        if t < T_SUB:
            return
        n = min(len(self.sub_full), int((t - T_SUB) / 0.045) + 1)
        s = self.sub_full[:n]
        spr = text_sprite(self.sub_full, (255, 236, 214), font='big', outline=(34, 12, 40), shadow=(34, 12, 40))
        full_w = spr.shape[1]
        part = text_sprite(s, (255, 236, 214), font='big', outline=(34, 12, 40), shadow=(34, 12, 40))
        blit(c, part, 240 - full_w / 2, 226)
        if n < len(self.sub_full) and int(t * 12) % 2 == 0:
            rect(c, 240 - full_w / 2 + part.shape[1] + 1, 228, 6, 8, (255, 236, 214))

    # ------------------------------------------------------------------ кадр
    def render(self, t):
        cam = self.cam(t)
        c = canvas()
        self.draw_sky(c, cam, t)
        self.w.draw_ground(c, cam, fog_near=300, fog_far=1800)
        items = list(self.trees)
        face, cY, csx, csy = self.clawd_state(t)
        row, col, xY, xsx, xsy = self.codex_state(t)

        def dclawd(d, cm):
            ground_shadow(d, cm, CLX, CLZ, 11 * (1 - min(0.5, cY / 20)), alpha=0.35)
            draw_clawd_3d(d, cm, CL, CLX, CLZ, Y=cY, face=face, sx=csx, sy=csy)

        def dcodex(d, cm):
            ground_shadow(d, cm, CDX, CDZ, 8 * (1 - min(0.5, xY / 20)), alpha=0.35)
            draw_codex_3d(d, cm, self.pix, CX, CDX, CDZ, row, col, Y=xY, sx=xsx, sy=xsy)

        items.append((CLX, 0.0, CLZ, dclawd))
        items.append((CDX, 0.0, CDZ, dcodex))
        draw_world(c, cam, items)

        # затемнение ближней земли (фокус на героях) и виньетка
        for y in range(196, H):
            k = (y - 196) / (H - 196) * 0.5
            q = round(k * 8) / 8
            m = DITHER[y] < q
            c[y, m] *= 0.62
        bg.vignette(c, 0.35)
        # пыль от злых подскоков
        for (tb, X, Z, sd) in ((2.36, CLX, CLZ, 71), (3.12, CLX, CLZ, 72), (2.54, CDX, CDZ, 73), (3.26, CDX, CDZ, 74)):
            dust_world(c, cam, X, Z, t - tb, seed=sd, units=0.7, life=0.4, n=5)
        # вспышка удара VS (гасит мир, но не молнии и титры)
        if T_VS_HIT <= t < T_VS_HIT + 0.14:
            bg.flash(c, 0.85 * (1 - (t - T_VS_HIT) / 0.14), (255, 250, 230))
        # молния между соперниками после VS
        if t >= T_VS_HIT:
            a = t - T_VS_HIT
            crack = a < 0.35 or ((a - 0.35) % 0.62) < 0.12
            if crack:
                pa = proj(cam, CLX + 12, 11.0, CLZ)
                pb = proj(cam, CDX - 9, 13.0, CDZ)
                if pa and pb:
                    bolt(c, pa[0] + 3, pa[1], pb[0] - 3, pb[1], seed=int(t * 15), jag=11.0, step=16.0, thick=1)
        # значки гнева
        if t >= T_CLAWD_HIT + 0.1:
            p = proj(cam, CLX + 13.5, 15.0 + cY, CLZ)
            if p:
                bg.anger_mark(c, p[0], p[1], t, 2)
        if t >= T_CODEX_HIT + 0.1:
            p = proj(cam, CDX + 11.5, 19.0 + xY, CDZ)
            if p:
                bg.anger_mark(c, p[0], p[1], t + 0.3, 2)

        # титры
        self.draw_name(c, self.C1, t, T_CLAWD, T_CLAWD_HIT, -170, 120, (255, 190, 150), +1)
        self.draw_name(c, self.C2, t, T_CODEX, T_CODEX_HIT, W + 170, 360, (170, 190, 255), -1)
        self.draw_vs(c, t)
        self.draw_sub(c, t)
        amp = max(decay_shake(t, T_CLAWD_HIT, 0.16, 2.0), decay_shake(t, T_CODEX_HIT, 0.16, 2.0),
                  decay_shake(t, T_VS_HIT, 0.40, 4.0))
        dx, dy = shake_offset(t, amp, seed=3)
        return shift_canvas(c, dx, dy)


SCENE = Title()
