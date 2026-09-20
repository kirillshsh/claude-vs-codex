"""s1 — солнечный луг и кубок (9.0 с). render.py: iris-in 0.6 с, в конце жёсткая склейка.

Кадры:
  A 0.00–3.20  общий план сверху: камера облетает поляну по дуге и опускается к кубку; тени облаков, птицы
  B 3.20–6.35  низкий ракурс у самой травы: Clawd семенит слева, Codex бежит справа, пыль; встают по бокам
               от кубка, «!» над обоими, прыжок, радость
  D 6.35–7.00  быстрый наезд на сияющий кубок (лучи, блик)
  E 7.00–8.20  «дуэль»: медленный наезд, оба поворачиваются друг к другу, ветер несёт листик
  F 8.20–8.60  крупно Clawd (с точки Codex): глаза сужаются
  G 8.60–9.00  крупно Codex (с точки Clawd): '>_' -> '><'
"""
import math
import numpy as np
from px import *
import bg
from bg import PEDESTAL, TROPHY
from cam3d import *
from scene_base import Scene
from scenes._a_helpers import *

SHOTS = [(0.0, 'A'), (3.20, 'B'), (6.35, 'D'), (7.00, 'E'), (8.20, 'F'), (8.60, 'G')]

# финальные места у кубка
CL_X, CD_X = -22.0, 21.0
CL_X0, CD_X0 = -112.0, 118.0
CL_T0, CL_T1 = 2.50, 5.10
CD_T0, CD_T1 = 2.72, 5.26
T_EXCL_CL, T_EXCL_CD = 5.55, 5.63
T_HAPPY = 5.86
T_TURN_CL, T_TURN_CD = 7.42, 7.62


def u_lh(v):
    return ease_in_out(v)


class Meadow1(Scene):
    name = 's1_meadow'
    dur = 9.0

    def __init__(self):
        self.w = get_world('day')
        self.pix = get_pix()
        self.pal = self.w.p
        self.shadows = CloudShadows(seed=5, scale=1 / 340.0, cover=0.34, dark=0.87)
        # трава переднего плана — только по краям кадра, чтобы не закрывать героев
        fg = Foreground(self.pal, (-50, 50, -48, -30), n=60, seed=11, ppu=2.0, flowers=0.12)
        fg.items = [it for it in fg.items if abs(it[0]) > 14 + (it[1] + 48) * 0.9][:26]
        self.fg = fg
        # широкий слой травы по всей поляне — для ближнего параллакса на облёте
        fw = Foreground(self.pal, (-210, 210, -210, 210), n=620, seed=23, ppu=2.0, flowers=0.18)
        fw.items = [it for it in fw.items if 30 < math.hypot(it[0], it[1]) < 215]
        self.fg_wide = fw
        self.trees_all = self.w.tree_items()
        self.trees_front = [it for it in self.trees_all
                            if not (it[2] > 0 and abs(it[0]) < 0.17 * (it[2] + 60) + 8)]
        rng = np.random.default_rng(3)
        self.birds = [(rng.uniform(0, 6.28), rng.uniform(120, 190), rng.uniform(70, 95), rng.uniform(0.18, 0.28),
                       rng.uniform(0, 1)) for _ in range(4)]

    # ================================================================== общие части
    def base(self, cam, t, shade=None, fog_near=260, fog_far=1500):
        c = canvas()
        self.w.draw_backdrop(c, cam, t)
        self.w.ground.render(c, cam, fog=self.w.fog, fog_near=fog_near, fog_far=fog_far, shade=shade)
        return c

    def trophy_items(self, t, glint=True, shine=None):
        def fn(d, cm):
            ground_shadow(d, cm, 0.0, 0.0, 8.5, alpha=0.3)
            draw_sprite_3d(d, cm, PEDESTAL, 0, 0, 0, 2.0, fog=self.w.fog)
            spr = TROPHY if shine is None else shine_sweep(TROPHY, shine, width=3, color=(255, 255, 250))
            draw_sprite_3d(d, cm, spr, 0, 8.0, 0, 2.0, fog=self.w.fog)
        return [(0.0, 0.0, 0.0, fn)]

    def trophy_fx(self, c, cam, t, big=False):
        """блики/искры кубка в экранных координатах."""
        p_top = proj(cam, -1.6, 15.2, 0.0)
        p_mid = proj(cam, 0.0, 12.5, 0.0)
        if p_top is None or p_mid is None:
            return
        s = p_mid[2]
        bg.glint(c, p_top[0], p_top[1], t, period=1.4, offset=0.2)
        p2 = proj(cam, 2.6, 10.5, 0.0)
        bg.glint(c, p2[0], p2[1], t, period=1.9, offset=0.9)
        w_ = max(10, 16 * s)
        bg.sparkles(c, t, p_mid[0] - w_ / 2, p_mid[1] - w_ * 0.6, w_, w_ * 0.9, n=4 if not big else 7, seed=4,
                    period=1.1)

    # ================================================================== хореография
    def clawd_pose(self, t):
        """(X, Y, kind, data, flip, sx, sy): kind 'anim' -> (name, frame) | 'face' -> (face, blush)"""
        if t < CL_T0:
            return None
        u = seg(t, CL_T0, CL_T1)
        X = lerp(CL_X0, CL_X, travel(u, 0.22))
        Y, sx, sy = 0.0, 1.0, 1.0
        if t < CL_T1:
            dist = X - CL_X0
            return (X, 0.0, 'anim', ('CrabWalking', 5 + int(dist / 1.25) % 15), False, 1.0, 1.0)
        # стоит, смотрит на кубок (3/4 вправо), дышит после пробежки
        if t < T_EXCL_CL:
            if 5.30 < t < 5.40:
                return (X, 0.0, 'anim', ('CrabWalking', 4), False, 1.04, 0.96)
            b = breathe(t, 1.9, 0.035)
            return (X, 0.0, 'anim', ('CrabWalking', 4), False, 2 - b, b)
        if t < T_HAPPY:
            Y = hop_y(t, T_EXCL_CL, 0.28, 3.2)
            a = t - T_EXCL_CL
            if a < 0.05:
                sx, sy = 1.1, 0.9
            elif a < 0.18:
                sx, sy = 0.94, 1.08
            return (X, Y, 'face', ('surprised', False), False, sx, sy)
        if t < T_TURN_CL - 0.08:
            a = t - T_HAPPY
            if a < 0.07:
                sx, sy = 1.12, 0.88
            # радостные подпрыгивания
            k = (a % 0.5) / 0.5
            Y = 1.4 * math.sin(math.pi * min(1, k / 0.45)) if k < 0.45 else 0.0
            return (X, Y, 'face', ('happy', True), False, sx, sy)
        if t < T_TURN_CL:
            return (X, 0.0, 'face', ('idle', False), False, 1.0, 1.0)
        # «дуэль»: злобно сопит и подаётся вперёд
        b = breathe(t, 1.55, 0.03)
        Yb = 0.35 + 0.35 * math.sin(t * 3.1)
        return (X, Yb, 'anim', ('CrabWalking', 4), False, 2 - b, b)

    def codex_pose(self, t):
        """(X, Y, row, col, sx, sy)"""
        if t < CD_T0:
            return None
        u = seg(t, CD_T0, CD_T1)
        X = lerp(CD_X0, CD_X, travel(u, 0.22))
        if t < CD_T1:
            dist = CD_X0 - X
            return (X, 0.0, 2, int(dist / 2.3) % 8, 1.0, 1.0)
        if t < T_EXCL_CD:
            if CD_T1 < t < CD_T1 + 0.08:
                return (X, 0.0, 10, 7, 1.06, 0.95)
            b = breathe(t, 1.75, 0.03, 0.8)
            col = 6 if t > 5.34 else 7   # переводит взгляд вверх-влево, на кубок
            return (X, 0.0, 10, col, 2 - b, b)
        if t < T_HAPPY + 0.05:
            Y = hop_y(t, T_EXCL_CD, 0.26, 3.0)
            return (X, Y, 4, 1, 1.0, 1.0)
        if t < T_TURN_CD - 0.1:
            a = t - (T_HAPPY + 0.05)
            k = int(a / 0.14)
            col = [1, 1, 2, 1, 5, 5, 1, 2][k % 8]
            Y = 1.6 if col == 2 else 0.0
            return (X, Y, 8, col, 1.0, 1.0)
        if t < T_TURN_CD:
            return (X, 0.0, 0, 0, 1.0, 1.0)
        b = breathe(t, 1.4, 0.028, 2.2)
        col = 7 if t < 7.98 else 3       # чуть опускает голову — «сверлит» взглядом
        return (X, 0.0, 10, col, 2 - b, b)

    def char_items(self, t, cam_for_fx=None):
        items = []
        cp = self.clawd_pose(t)
        if cp is not None:
            X, Y, kind, data, flip, sx, sy = cp

            def fcl(d, cm, X=X, Y=Y, kind=kind, data=data, flip=flip, sx=sx, sy=sy):
                ground_shadow(d, cm, X, 0.0, 11 * (1 - min(0.5, Y / 16)), alpha=0.3)
                if kind == 'anim':
                    draw_clawd_3d(d, cm, CL, X, 0.0, Y=Y, anim=data[0], frame=data[1], flipx=flip, sx=sx, sy=sy)
                else:
                    draw_clawd_3d(d, cm, CL, X, 0.0, Y=Y, face=data[0], blush=data[1], flipx=flip, sx=sx, sy=sy)
            items.append((X, 0.0, 0.0, fcl))
        xp = self.codex_pose(t)
        if xp is not None:
            X, Y, row, col, sx, sy = xp

            def fcd(d, cm, X=X, Y=Y, row=row, col=col, sx=sx, sy=sy):
                ground_shadow(d, cm, X, 0.0, 8 * (1 - min(0.5, Y / 16)), alpha=0.3)
                draw_codex_3d(d, cm, self.pix, CX, X, 0.0, row, col, Y=Y, sx=sx, sy=sy)
            items.append((X, 0.0, 0.0, fcd))
        return items

    def dust_fx(self, c, cam, t):
        # пыль из-под ног при ходьбе/беге (в мировых точках испускания)
        for (t0, t1, x0, x1, back, seed0) in ((CL_T0, CL_T1, CL_X0, CL_X, -9.0, 100), (CD_T0, CD_T1, CD_X0, CD_X, 7.0, 300)):
            k = 0
            te = t0 + 0.05
            while te < t1 - 0.25:
                age = t - te
                if 0 <= age < 0.55:
                    X = lerp(x0, x1, travel(seg(te, t0, t1), 0.22)) + back
                    dust_world(c, cam, X, 1.5 * ((k % 2) * 2 - 1), age, seed=seed0 + k, units=0.8, life=0.55, n=4)
                k += 1
                te += 0.17
            # торможение: облачко побольше
            age = t - (t1 - 0.12)
            if 0 <= age < 0.7:
                dust_world(c, cam, x1 - back * 0.3, 1.0, age, seed=seed0 + 99, units=1.4, life=0.7, n=7)
        # приземление после «!»
        for (tt, X, sd) in ((T_EXCL_CL + 0.28, CL_X, 501), (T_EXCL_CD + 0.26, CD_X, 502)):
            age = t - tt
            if 0 <= age < 0.45:
                dust_world(c, cam, X, 0.5, age, seed=sd, units=0.9, life=0.45, n=6)

    def excl_fx(self, c, cam, t):
        cp = self.clawd_pose(t)
        xp = self.codex_pose(t)
        if cp is not None and t >= T_EXCL_CL and t < T_HAPPY + 0.35:
            p = proj(cam, cp[0], 17.5 + cp[1], 0.0)
            if p:
                pop_icon(c, EXCL4 if p[2] > 3.6 else EXCL3, p[0], p[1] - 2, t - T_EXCL_CL)
        if xp is not None and t >= T_EXCL_CD and t < T_HAPPY + 0.40:
            p = proj(cam, xp[0], 23.0 + xp[1], 0.0)
            if p:
                pop_icon(c, EXCL4 if p[2] > 3.6 else EXCL3, p[0], p[1] - 2, t - T_EXCL_CD)
        # радостные искорки
        if T_HAPPY + 0.05 < t < T_TURN_CL:
            for i, (X, Yh, off) in enumerate(((CL_X - 13, 14, 0.0), (CL_X + 13, 12, 0.35), (CD_X - 11, 20, 0.2),
                                              (CD_X + 11, 17, 0.55))):
                p = proj(cam, X, Yh, 0.0)
                if p:
                    bg.glint(c, p[0], p[1], t, period=0.7, offset=off)

    # ================================================================== кадры
    def cam_A(self, lt):
        v = seg(lt, 0.0, 3.2)
        # угол — почти равномерно (≈37°/с), чтобы деревья и тени всё время проезжали мимо
        # почти 200° облёта вокруг кубка у самых крон: деревья, горы и тени облаков проносятся мимо
        ang = lerp(-3.72, -0.76, 0.82 * v + 0.18 * ease_in_out(v))
        h = lerp(34.0, 20.0, ease_in_out(1 - (1 - v) ** 1.5))
        r = lerp(78.0, 56.0, 0.55 * v + 0.45 * ease_out(v))
        f = lerp(225.0, 300.0, ease_in_out(v))
        return orbit((0.0, 0.0), r, ang, h, f=f, look_height=lerp(6.0, 7.5, u_lh(v)))

    def shot_A(self, t, lt):
        cam = self.cam_A(lt)
        c = self.base(cam, t, shade=self.shadows.shade_fn(t, vel=(24.0, 11.0)))
        # мягкие лучи от кубка — маяк для глаза
        p = proj(cam, 0.0, 12.0, 0.0)
        if p:
            light_rays(c, p[0], p[1], t, n=9, color=(255, 250, 215), alpha=0.16, r_in=6, r_out=26 + p[2] * 14,
                       spin=0.6)
        items = self.w.tree_items() + self.trophy_items(t) + self.char_items(t)
        items += self.fg_wide.world_items(t, fog=self.w.fog, max_px=120, near=(cam.x, cam.z, 120), limit=16)
        draw_world(c, cam, items)
        self.dust_fx(c, cam, t)
        self.trophy_fx(c, cam, t, big=True)
        # стайка птиц пересекает поляну прямо над кубком — живой передний план
        for i, (ox, oz, oy, ph) in enumerate(((0, 0, 0, 0.0), (-11, -8, 3, 0.4), (-9, 9, -2, 0.8), (-21, 2, 1, 0.2))):
            q = lt / 3.2
            X = lerp(-150.0, 150.0, q) + ox
            Z = lerp(62.0, -58.0, q) + oz
            Y = 46.0 + oy + 1.8 * math.sin(lt * 3 + ph * 5)
            p = proj(cam, X, Y, Z)
            if p and 0.25 < p[2]:
                fr = BIRD[int(lt * 9 + ph * 3) % 3]
                k = 3 if p[2] > 3.0 else (2 if p[2] > 1.5 else 1)
                spr = scale_nn(fr, k)
                blit(c, spr, p[0] - spr.shape[1] / 2, p[1] - spr.shape[0] / 2)
        return c

    def shot_B(self, t, lt):
        u = seg(lt, 0.0, 3.15)
        eye = (lerp(-6.0, 2.0, ease_in_out(u)), lerp(3.4, 4.2, u), lerp(-66.0, -56.0, ease_in_out(u)))
        tgt = (lerp(-4.0, 0.0, ease_in_out(u)), 9.0, 0.0)
        cam = look_at(eye, tgt, 250.0)
        c = self.base(cam, t)
        items = self.trees_front + self.trophy_items(t) + self.char_items(t)
        items += self.fg.world_items(t, fog=None, max_px=56)
        items += self.fg_wide.world_items(t, fog=self.w.fog, max_px=64, near=(cam.x, cam.z, 78), limit=16)
        draw_world(c, cam, items)
        self.dust_fx(c, cam, t)
        self.trophy_fx(c, cam, t)
        self.excl_fx(c, cam, t)
        return c

    def shot_D(self, t, lt):
        u = ease_in_out(seg(lt, 0.0, 0.58))
        eye = (0.0, lerp(6.0, 10.5, u), lerp(-40.0, -15.5, u))
        tgt = (0.0, lerp(11.0, 12.6, u), 0.0)
        cam = look_at(eye, tgt, lerp(270.0, 320.0, u))
        c = self.base(cam, t)
        hy = proj(cam, 0.0, 12.5, 0.0)
        if hy:
            light_rays(c, hy[0], hy[1], t, n=14, alpha=0.55 * seg(lt, 0.0, 0.3), r_in=12, r_out=300, spin=0.5)
        shine = seg(lt, 0.18, 0.50) if 0.18 <= lt <= 0.50 else None
        items = self.trees_front + self.trophy_items(t, shine=shine) + self.char_items(t)
        draw_world(c, cam, items)
        # большая звезда-блик на ободе
        p = proj(cam, -2.2, 16.0, 0.0)
        if p and lt > 0.30:
            a = lt - 0.30
            k = pop_scale(a, 0.12)
            if k > 0:
                st = scale_nn(STAR5, 3)
                st = squash(st, k, k) if k < 0.999 else st
                blit(c, st, p[0] - st.shape[1] / 2, p[1] - st.shape[0] / 2)
        self.trophy_fx(c, cam, t, big=True)
        return c

    def shot_E(self, t, lt):
        # «дуэль»: камера у самой земли, снизу вверх, медленный наезд
        u = seg(lt, 0.0, 1.2)
        eye = (0.0, lerp(2.3, 2.0, u), lerp(-45.0, -37.0, ease_in_out(u)))
        cam = look_at(eye, (0.0, 10.5, 0.0), lerp(228.0, 240.0, u))
        c = self.base(cam, t)
        items = self.trees_front + self.trophy_items(t) + self.char_items(t)
        items += self.fg.world_items(t, fog=None, max_px=56)
        items += self.fg_wide.world_items(t, fog=self.w.fog, max_px=64, near=(cam.x, cam.z, 70), limit=16)
        draw_world(c, cam, items)
        self.trophy_fx(c, cam, t)
        self.excl_fx(c, cam, t)
        # искра между взглядами (намёк на молнию в s2)
        if 8.02 <= t < 8.16:
            pa = proj(cam, CL_X + 8.0, 12.8, 0.0)
            pb = proj(cam, CD_X - 6.0, 15.5, 0.0)
            if pa and pb:
                bolt(c, pa[0], pa[1], pb[0], pb[1], seed=int(t * 30), jag=6.0, step=14.0, thick=1,
                     core=(255, 255, 235), glow=(255, 214, 90), branches=False)
        # ветер: два листика пролетают между ними (как перед дуэлью)
        for (t0, z, yb, sd) in ((7.64, -10.0, 21.0, 0), (7.78, -16.0, 25.0, 1), (7.95, -8.0, 19.0, 2)):
            a = t - t0
            if 0 <= a < 0.8:
                X = lerp(-48.0, 48.0, a / 0.8)
                Y = yb + 2.5 * math.sin(a * 10.0 + sd * 2) - a * 3
                p = proj(cam, X, Y, z)
                if p:
                    fr = LEAF_S[(int(a * 16) + sd) % 4]
                    spr = scale_nn(fr, 3)
                    blit(c, spr, p[0] - 6, p[1] - 5)
        return c

    def closeup(self, t, lt, eye, target, f0, f1, dur, draw_char):
        # резкий наезд трансфокатором в первые 0.12 с, затем медленное «подползание»
        f = lerp(f0 * 0.78, f0, ease_out(seg(lt, 0.0, 0.12))) + (f1 - f0) * seg(lt, 0.12, dur)
        cam = look_at(eye, target, f)
        c = self.base(cam, t)
        items = self.w.tree_items()
        items.append((target[0], 0.0, target[2], draw_char))
        draw_world(c, cam, items)
        return c, cam

    def shot_F(self, t, lt):
        face = 'idle' if lt < 0.13 else 'angry'

        def dc(d, cm):
            ground_shadow(d, cm, CL_X, 0.0, 11, alpha=0.3)
            draw_clawd_3d(d, cm, CL, CL_X, 0.0, face=face)
        c, cam = self.closeup(t, lt, (CL_X + 36.0, 7.0, -20.0), (CL_X, 10.8, 0.0), 560.0, 620.0, 0.4, dc)
        if lt >= 0.14:
            p = proj(cam, CL_X + 9.0, 14.0, 0.0)
            if p:
                k = pop_scale(lt - 0.14, 0.12)
                sp = scale_nn(ANGER if int(lt * 8) % 2 == 0 else ANGER2, 4)
                if k < 0.999:
                    sp = squash(sp, max(0.1, k), max(0.1, k))
                blit(c, sp, p[0] - sp.shape[1] / 2, p[1] - sp.shape[0] / 2)
        dx, dy = shake_offset(t, 1.2 * clamp((lt - 0.13) / 0.1), seed=31, freq=22)
        return shift_canvas(c, dx, dy)

    def shot_G(self, t, lt):
        col, row = (0, 0) if lt < 0.12 else (1, 5)

        def dc(d, cm):
            ground_shadow(d, cm, CD_X, 0.0, 8, alpha=0.3)
            draw_codex_3d(d, cm, self.pix, CX, CD_X, 0.0, row, col)
        c, cam = self.closeup(t, lt, (CD_X - 34.0, 8.0, -19.0), (CD_X, 14.2, 0.0), 540.0, 600.0, 0.4, dc)
        if lt >= 0.12:
            p = proj(cam, CD_X + 9.5, 21.0, 0.0)
            if p:
                k = pop_scale(lt - 0.12, 0.12)
                sp = scale_nn(ANGER if int(lt * 8) % 2 == 0 else ANGER2, 4)
                if k < 0.999:
                    sp = squash(sp, max(0.1, k), max(0.1, k))
                blit(c, sp, p[0] - sp.shape[1] / 2, p[1] - sp.shape[0] / 2)
        dx, dy = shake_offset(t, 1.2 * clamp((lt - 0.12) / 0.1), seed=37, freq=22)
        return shift_canvas(c, dx, dy)

    def render(self, t):
        name, start = 'A', 0.0
        for (s, n) in SHOTS:
            if t >= s:
                name, start = n, s
        return getattr(self, 'shot_' + name)(t, t - start)


SCENE = Meadow1()
