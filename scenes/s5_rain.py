"""s5_rain — грусть под дождём и примирение (17.0 c).

Шоты:
  1  0.0– 4.0  высокий общий план, медленный кран вниз: сидят далеко друг от друга, между ними половинки кубка, молния
  2  4.0– 6.9  крупный план Clawd сверху: дрожит, капля стекает, «апчхи!»
  3  6.9–10.2  Codex оглядывается через плечо, сомневается, срывает большой лопух, идёт к Clawd
  4 10.2–13.3  нижний ракурс: лопух-зонтик над Clawd, капли бьют по листу, «МИР?» → «МИР!» + сердечко
  5 13.3–17.0  дождь кончается, фронт света (storm → golden), лучи, радуга, объятия, половинки кубка срастаются
"""
import math

import numpy as np

import bg
from px import *
from cam3d import *
from chars import load
from scene_base import Scene
from worlds import MeadowWorld, paint_disc
from scenes._c_helpers import *

T1, T2, T3, T4, T_END = 4.0, 6.9, 10.2, 13.3, 17.0

CL = (-30.0, 0.0)          # Clawd сидит
CX0 = (32.0, 4.0)          # Codex дуется спиной
CX_HUG = (-14.0, 2.0)      # Codex рядом с Clawd (шот 4–5)
TL_POS = (-1.0, -6.0)      # половинки кубка
TR_POS = (11.0, -1.0)
# лопух возле Codex: (основание X,Z), (верх купола X,Y,Z), px/юнит, наклон; последний сорвёт Codex
# ppu считается от ширины спрайта bg.LEAF (56 px исходника) — экранный размер купола от неё не зависит
PLANT = [((22.0, 18.0), (15.0, 14.0, 19.0), 2.10, -6),
         ((25.0, 19.0), (31.0, 17.0, 20.0), 2.17, 6),
         ((24.0, 15.0), (25.0, 11.0, 13.0), 1.96, 4)]
PLUCK = PLANT[2]
PUDDLES = [(5.0, 14.0, 10.0), (-20.0, -19.0, 5.5), (26.0, -16.0, 6.0), (-60.0, 12.0, 8.0), (62.0, 30.0, 7.0),
           (-14.0, 46.0, 9.0), (38.0, 56.0, 6.5), (-48.0, -44.0, 6.0), (12.0, -48.0, 5.5), (-88.0, -20.0, 7.5),
           (90.0, -6.0, 8.0), (-34.0, 80.0, 9.0), (70.0, -44.0, 6.0)]

TINT_S = (0.9, 0.87, 0.97)    # свет грозы
TINT_G = (1.03, 0.99, 0.92)   # золотой свет
RAIN_FAR = dict(color=(92, 104, 140), tail=(74, 84, 118), length=4, speed=250.0, wind=0.16)
RAIN_NEAR = dict(color=(172, 186, 224), tail=(118, 132, 172), length=8, speed=430.0, wind=0.2)
RIPPLE_C = ((128, 142, 186), (96, 108, 150))


def _puddle_painter(mode):
    water = {'storm': ('#4b5579', '#6a76a2', '#27302a'), 'golden': ('#9cb8e4', '#e8eef8', '#6a5238')}[mode]
    wc, hc, rim = hexc(water[0]), hexc(water[1]), hexc(water[2])

    def fn(tex, p):
        size = tex.shape[0]
        for i, (X, Z, R) in enumerate(PUDDLES):
            u, v = int((X + size / 4) * 2), int((Z + size / 4) * 2)
            paint_disc(tex, u, v, R * 2 + 2.0, rim, jag=1.6, seed=i)
            paint_disc(tex, u, v, R * 2, wc, jag=1.6, seed=i)
            # отражение неба: две короткие светлые полоски
            for k in range(2):
                yy = v - int(R * 0.5) + k * 4
                x0 = u - int(R * 0.7) + k * 5
                tex[yy % size, (x0 + np.arange(int(R * 0.8))) % size] = hc
    return fn


def _build_world(mode):
    return MeadowWorld(mode, extra_paint=_puddle_painter(mode))


def _crop(spr):
    a = spr[:, :, 3] > 0
    ys, xs = np.where(a)
    return spr[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()


class S5Rain(Scene):
    name = 's5_rain'
    dur = T_END

    def __init__(self):
        self.cl, self.cx = load()
        self.pix = CodexPixelizer()
        self.storm = cached('s5_world_storm', lambda: _build_world('storm'))
        self._golden = None
        self.tl_lying = _crop(np.rot90(bg.TROPHY_L, 1))
        self.tr_lying = _crop(np.rot90(bg.TROPHY_R, -1))
        self.tl_up = bg.TROPHY_L   # не обрезаем: у обеих половинок общий якорь — сходятся в целый кубок
        self.tr_up = bg.TROPHY_R
        self.t_heart = bg.trophy_heart()
        self.canopy = LEAF_CANOPY
        # где рука Codex (доля кадра атласа ÷2) — к ней крепим черешок
        self.hand = {}
        for (r, c_) in [(3, 1), (3, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (0, 0)]:
            f = self.cx.frame(r, c_, 2)
            a = f[:, :, 3] > 0
            ys = np.where(a.any(1))[0]
            top, bot = ys.min(), ys.max()
            y_a, y_b = int(top + (bot - top) * 0.3), int(top + (bot - top) * 0.8)
            cols = np.where(a[y_a:y_b].any(0))[0]
            xl = cols.min()
            yl = y_a + np.where(a[y_a:y_b, xl])[0].mean()
            self.hand[(r, c_)] = ((xl + 1) / f.shape[1], yl / f.shape[0])

    @property
    def golden(self):
        if self._golden is None:
            self._golden = cached('s5_world_golden', lambda: _build_world('golden'))
        return self._golden

    # ------------------------------------------------------------------ общие куски
    def env(self, c, cam, t, world, lightning=None, rainbow=None):
        sky_dome(c, cam, world.sky, span=0.75, below=world.fog)
        world.clouds.draw(c, cam, drift=t * 5.0, y_offset=-8)
        if lightning is not None:
            lightning(c)
        if rainbow is not None:
            rainbow(c)
        world.mount.draw(c, cam, y_offset=1)
        world.hills_far.draw(c, cam, y_offset=2)
        world.hills.draw(c, cam, y_offset=3)
        world.draw_ground(c, cam)

    def ground_rain(self, c, cam, t, amount=1.0):
        """круги на лужах + дальний слой дождя (рисуется ДО персонажей — они его перекрывают)."""
        if amount <= 0:
            return
        ripples(c, cam, t, PUDDLES, RIPPLE_C[0], RIPPLE_C[1], amount=amount, rmax=1.7, rate=4.0, life=0.5)
        hy = max(0.0, cam.horizon_y())
        rain(c, t, n=230, seed=11, stop_lo=hy + 2, stop_hi=H + 6, intensity=amount, **RAIN_FAR)

    def near_rain(self, c, cam, t, amount=1.0, stopmap=None, hit_frac=1.0, n=110):
        if amount <= 0:
            return
        hy = max(0.0, cam.horizon_y())
        rain(c, t, n=n, seed=23, stop_lo=hy + (H - hy) * 0.45, stop_hi=H + 10, stopmap=stopmap, hit_frac=hit_frac,
             splash_color=(196, 208, 238), intensity=amount, **RAIN_NEAR)

    def trophy_items(self, tint, fuse=None):
        """половинки кубка; fuse — прогресс сращивания 0..1 (None — лежат)."""
        items = []
        if fuse is None or fuse <= 0:
            for (spr, (X, Z)) in ((self.tl_lying, TL_POS), (self.tr_lying, TR_POS)):
                def fn(d, cm, spr=spr, X=X, Z=Z):
                    ground_shadow(d, cm, X, Z, 5, alpha=0.25)
                    draw_spr(d, cm, spr, (spr.shape[1] / 2, spr.shape[0]), X, 0, Z, 1.5, tint=tint)
                items.append((X, 0.0, Z, fn))
            return items
        mid = ((TL_POS[0] + TR_POS[0]) / 2, (TL_POS[1] + TR_POS[1]) / 2 - 1)
        if fuse < 0.8:
            for k, (spr_l, spr_u, (X, Z)) in enumerate(((self.tl_lying, self.tl_up, TL_POS),
                                                          (self.tr_lying, self.tr_up, TR_POS))):
                p1 = seg(fuse, 0.0, 0.32)          # прыжок 1: встают
                p2 = seg(fuse, 0.38, 0.8)          # прыжок 2: навстречу друг другу
                Xc = lerp(X, mid[0], ease_in_out(p2))
                Zc = lerp(Z, mid[1], ease_in_out(p2))
                Y = 9 * math.sin(math.pi * p1) + 7 * math.sin(math.pi * p2)
                spr = spr_l if p1 < 0.5 else spr_u
                anc = (spr.shape[1] / 2, spr.shape[0])
                sq = 1.0
                if 0.32 <= fuse < 0.38:
                    sq = 0.85  # приземлились — присели

                def fn(d, cm, spr=spr, Xc=Xc, Zc=Zc, Y=Y, anc=anc, sq=sq):
                    ground_shadow(d, cm, Xc, Zc, 5, alpha=0.25)
                    draw_spr(d, cm, spr, anc, Xc, Y, Zc, 1.5, tint=tint, sy=sq, sx=2 - sq)
                items.append((Xc, 0.0, Zc - 0.01 * k, fn))
            return items

        def fn(d, cm):
            ground_shadow(d, cm, mid[0], mid[1], 6, alpha=0.25)
            k = seg(fuse, 0.8, 0.9)
            sq = 1.0 + 0.22 * math.sin(math.pi * k)
            draw_spr(d, cm, self.t_heart, (self.t_heart.shape[1] / 2, self.t_heart.shape[0]), mid[0], 0, mid[1], 1.5,
                     sx=sq, sy=2 - sq, tint=tint)
        items.append((mid[0], 0.0, mid[1], fn))
        return items

    def plant_items(self, t, tint, plucked=False):
        items = []
        for i, (base, top, ppu, ln) in enumerate(PLANT):
            if plucked and i == len(PLANT) - 1:
                continue
            sway = int(round(math.sin(t * 2.3 + i * 1.9) * 2.0))

            def fn(d, cm, base=base, top=top, ppu=ppu, ln=ln, sway=sway):
                # растущий лопух: купол приплюснут и наклонён — лист на черешке, а не раскрытый зонт
                draw_leaf(d, cm, self.canopy, (base[0], 0.0, base[1]), top, ppu, tint=tint, lean_src=ln + sway,
                          sy=0.68)
            items.append((top[0], 0.0, top[2], fn))
        return items

    def codex_leaf(self, d, cm, X, Z, row, col, face_fn=None, face=None, tint=None, canopy=(0.0, 27.0), sy=1.0,
                   lean_leaf=0, Y=0.0):
        """Codex держит лопух: Codex, черешок от руки к куполу, купол. canopy = (dX от Codex, высота)."""
        info, spr = draw_codex(d, cm, self.pix, self.cx, X, Z, row=row, col=col, face_fn=face_fn, face=face,
                               tint=tint, sy=sy, ret_spr=True, Y=Y)
        cxw, cyw, czw = X + canopy[0], canopy[1], Z - 0.6
        pc = cm.project(cxw, cyw, czw)
        if info is not None and pc is not None:
            x0, y0, nw, nh, zc, s = info
            hx, hy = self.hand.get((row, col), (0.2, 0.55))
            draw_stem(d, x0 + hx * nw, y0 + hy * nh, pc[0], pc[1], thick=max(1, int(round(cm.f / pc[2] * 0.9))))
        cinfo, _ = draw_spr(d, cm, self.canopy, (self.canopy.shape[1] / 2, self.canopy.shape[0] - 1), cxw, cyw, czw,
                            PLUCK[2], tint=tint, lean_src=lean_leaf, pivot=1.0)
        return info, cinfo

    def clawd(self, d, cm, face='sad', legs=True, tears=1, blush=False, X=None, Y=0.0, Z=None, tint=TINT_S, **kw):
        X = CL[0] if X is None else X
        Z = CL[1] if Z is None else Z
        ground_shadow(d, cm, X, Z, 11, alpha=0.24)
        spr, anc = clawd_sprite(self.cl, face, legs=legs, tears=tears, blush=blush)
        return draw_spr(d, cm, spr, anc, X, Y, Z, 1.0, tint=tint, **kw)

    # ------------------------------------------------------------------ рендер
    def render(self, t):
        if t < T1:
            return self.shot1(t)
        if t < T2:
            return self.shot2(t - T1)
        if t < T3:
            return self.shot3(t - T2)
        if t < T4:
            return self.shot4(t - T3)
        return self.shot5(t - T4)

    # ------------------------------------------------------------------ шот 1: кран вниз
    def shot1(self, u):
        world = self.storm
        p = u / T1
        k = 1 - (1 - p) ** 2.0
        k = ease_in_out(p) * 0.4 + k * 0.6
        eye = lerp3((8.0, 82.0, -170.0), (2.0, 19.0, -88.0), k)
        tgt = lerp3((0.0, 0.0, 12.0), (1.0, 9.5, 0.0), k)
        cam = look_at(eye, tgt, lerp(250.0, 305.0, k))
        c = canvas()
        LT = 3.15
        lt = u - LT
        fl = 0.0
        if 0 <= lt < 0.07:
            fl = 1.0
        elif 0.12 <= lt < 0.17:
            fl = 0.65
        elif 0.17 <= lt < 0.7:
            fl = 0.3 * (1 - (lt - 0.17) / 0.53)
        bolt_on = (0 <= lt < 0.07) or (0.12 <= lt < 0.22)

        def bolt(d):
            if bolt_on:
                pos = sky_xy(cam, -0.36, 0.05)
                if pos is not None:
                    bg.lightning_bolt(d, 5, int(pos[0]), int(cam.horizon_y()) + 4)
        self.env(c, cam, u, world, lightning=bolt)
        if fl > 0:
            # небо и земля вспыхивают сильнее персонажей (их догоняем add'ом ниже)
            grade_to(c, (1 + 0.35 * fl, 1 + 0.35 * fl, 1 + 0.45 * fl), (26 * fl, 28 * fl, 44 * fl))
        self.ground_rain(c, cam, u)
        tint = TINT_S
        add = (22 * fl, 24 * fl, 34 * fl) if fl > 0 else None
        items = world.tree_items() + self.trophy_items(tint) + self.plant_items(u, tint)
        startle = 0 <= lt < 0.45

        def clawd(d, cm):
            face = 'surprised' if startle else ('blink' if (u % 3.1) < 0.14 else 'sad')
            sh = 1 if (int(u * 22) % 2 == 0 and (u % 1.7) < 0.9) else 0
            jump = 2.5 * math.sin(math.pi * seg(lt, 0.0, 0.25)) if startle else 0.0
            self.clawd(d, cm, face, Y=jump, add=add, dx_px=sh, sy=1.0 + 0.03 * math.sin(u * 2.2))
        items.append((CL[0], 0.0, CL[1], clawd))

        def codex(d, cm):
            ground_shadow(d, cm, CX0[0], CX0[1], 9, alpha=0.3)
            draw_codex(d, cm, self.pix, self.cx, CX0[0], CX0[1], 0, 0, face_fn=codex_back_hr, tint=tint, add=add,
                       sy=1.0 - 0.02 * (0.5 + 0.5 * math.sin(u * 1.6)))
        items.append((CX0[0], 0.0, CX0[1], codex))
        draw_world(c, cam, items)
        self.near_rain(c, cam, u)
        vignette_d(c, 0.4)
        return c

    # ------------------------------------------------------------------ шот 2: крупно Clawd сверху, «апчхи!»
    def shot2(self, u):
        world = self.storm
        k = ease_in_out(u / (T2 - T1))
        cam = look_at(lerp3((-62.0, 22.0, -40.0), (-57.0, 19.0, -34.0), k),
                      lerp3((-24.0, 5.0, 2.0), (-24.5, 5.5, 2.0), k), lerp(318.0, 330.0, k))
        c = canvas()
        tt = u + T1
        self.env(c, cam, tt, world)
        self.ground_rain(c, cam, tt)
        tint = TINT_S
        items = world.tree_items() + self.trophy_items(tint) + self.plant_items(tt, tint)
        A0, SN = 1.15, 1.6   # вдох «а-а-а...» и сам чих
        sx, sy, lean_s, face, dy = 1.0, 1.0, 0, 'sad', 0.0
        if A0 <= u < SN:
            q = ease_in_out(seg(u, A0, SN - 0.05))
            sy, sx, lean_s, face = 1.0 + 0.13 * q, 1.0 - 0.05 * q, -int(round(2 * q)), 'blink'
        elif SN <= u < SN + 0.45:
            q = seg(u, SN, SN + 0.45)
            sq = math.sin(math.pi * min(1.0, q * 2.2))
            sx, sy, lean_s, face = 1.0 + 0.16 * sq, 1.0 - 0.2 * sq, int(round(2 * sq)), 'blink'
            dy = 1.5 * math.sin(math.pi * min(1.0, q * 1.6))
        elif (u % 2.3) < 0.12:
            face = 'blink'
        shiver_on = (u < A0 - 0.1) or (u > SN + 0.55)
        sh = (1 if int(u * 26) % 2 == 0 else -1) if shiver_on and (u % 1.1) < 0.75 else 0
        ui = {}

        def clawd(d, cm):
            ui['cl'] = self.clawd(d, cm, face, Y=dy, dx_px=sh, sx=sx, sy=sy, lean_src=lean_s)
        items.append((CL[0], 0.0, CL[1], clawd))

        def codex(d, cm):
            ground_shadow(d, cm, CX0[0], CX0[1], 9, alpha=0.3)
            draw_codex(d, cm, self.pix, self.cx, CX0[0], CX0[1], 0, 0, face_fn=codex_back_hr, tint=tint)
        items.append((CX0[0], 0.0, CX0[1], codex))
        draw_world(c, cam, items)
        info, img = ui.get('cl', (None, None))
        stop = np.full(W, 1e9)
        if info is not None:
            stopmap_from_sprite(stop, img, info)
        self.near_rain(c, cam, tt, stopmap=stop, hit_frac=0.45, n=130)
        if info is not None:
            x0, y0, nw, nh = info[:4]
            s = nw / 24.0
            # капля стекает по боку и срывается с «клешни»
            q = seg(u, 0.2, 1.1)
            if 0 < q < 1:
                drop = scale_nn(DROP, 2)
                if q < 0.65:
                    yy = y0 + (0.6 + 3.4 * ease_in(q / 0.65)) * s
                    xx = x0 + 21.5 * s
                else:
                    qq = (q - 0.65) / 0.35
                    xx = x0 + (21.8 + 2.2 * min(1.0, qq * 2)) * s
                    yy = y0 + (4.0 + 9.0 * qq * qq) * s
                blit(c, drop, xx - drop.shape[1] / 2, yy - drop.shape[0])
            # слёзы катятся
            if face != 'blink':
                for j, ex in enumerate((6.5, 17.5)):
                    tq = ((u + j * 0.41) % 0.95) / 0.95
                    if tq < 0.8:
                        ty = y0 + (4.6 + 5.0 * ease_in(tq / 0.8)) * s
                        tx = x0 + ex * s
                        rect(c, tx - 1, ty, 3, 3, (120, 200, 255))
                        pset(c, tx, ty, (225, 246, 255))
            # чих: облачко и брызги
            if SN <= u < SN + 0.55:
                a = u - SN
                fx, fy = x0 + nw * 0.5, y0 + nh * 0.42
                if a < 0.22:
                    # манга-«взрыв» чиха: короткие лучи вокруг мордочки
                    r0 = nw * 0.56 + a * 90
                    for i in range(10):
                        ang = i * math.pi / 5 + 0.3
                        ca, sa = math.cos(ang), math.sin(ang) * 0.62
                        line(c, fx + ca * r0, fy + sa * r0, fx + ca * (r0 + 9), fy + sa * (r0 + 9), (232, 240, 252))
                rng = np.random.default_rng(3)
                for i in range(6):
                    vx = rng.uniform(70, 160) * (1 if i % 2 else -1)
                    vy = rng.uniform(-90, -40)
                    px_, py_ = fx + vx * a * 1.2, fy + vy * a + 320 * a * a
                    rect(c, px_ - 1, py_ - 1, 2, 3, (130, 196, 250))
                    pset(c, px_ - 1, py_ - 1, (232, 246, 255))
            if u >= SN - 0.03:
                draw_bubble(c, 'апчхи!', x0 + nw * 0.7, y0 - 5, u - SN + 0.03, tail='left', font='small', seed=2)
        vignette_d(c, 0.45)
        return c

    # ------------------------------------------------------------------ шот 3: Codex решается
    def shot3(self, u):
        world = self.storm
        T = T3 - T2
        k = ease_in_out(seg(u, 2.2, T + 0.3))
        eye = lerp3((64.0, 13.0, -32.0), (54.0, 12.5, -40.0), k)
        tgt = lerp3((12.0, 10.0, 6.0), (-6.0, 9.0, 6.0), k)
        cam = look_at(eye, tgt, 240.0)
        c = canvas()
        tt = u + T2
        self.env(c, cam, tt, world)
        self.ground_rain(c, cam, tt)
        tint = TINT_S
        GL0, TURN, THINK, EXCL, PICK, WALK = 0.5, 0.9, 1.25, 1.95, 2.1, 2.5
        plucked = u >= PICK
        items = world.tree_items() + self.trophy_items(tint) + self.plant_items(tt, tint, plucked=plucked)
        X, Z = CX0
        walk = seg(u, WALK, T + 0.4)
        Xw = X - 22.0 * walk
        ui = {}

        def codex(d, cm):
            ground_shadow(d, cm, Xw, Z, 9, alpha=0.3)
            if u < GL0:
                info = draw_codex(d, cm, self.pix, self.cx, X, Z, 0, 0, face_fn=codex_back_hr, tint=tint,
                                  sy=1.0 - 0.025 * math.sin(math.pi * seg(u, 0.05, 0.45)))
            elif u < TURN:
                info = draw_codex(d, cm, self.pix, self.cx, X, Z, 10, 4, face_fn=codex_nochest, tint=tint)
            elif u < THINK:
                info = draw_codex(d, cm, self.pix, self.cx, X, Z, 10, 3, tint=tint)
            elif u < EXCL:
                seq = [0, 1, 1, 2, 2, 1, 2]
                col = seq[min(len(seq) - 1, int((u - THINK) / 0.1))]
                info = draw_codex(d, cm, self.pix, self.cx, X, Z, 6, col, tint=tint)
            elif u < PICK:
                hopy = 2.5 * math.sin(math.pi * seg(u, EXCL, PICK))
                info = draw_codex(d, cm, self.pix, self.cx, X, Z, 8, 1, Y=hopy, tint=tint)
            elif u < WALK:
                q = ease_out(seg(u, PICK, PICK + 0.3))
                top = PLUCK[1]
                cxo = lerp(top[0] - X, -2.0, q)
                cyo = lerp(top[1], 27.0, q)
                info, _ = self.codex_leaf(d, cm, X, Z, 3, 2, tint=tint, canopy=(cxo, cyo))
            else:
                col = int((u - WALK) * 7) % 8
                bob = 0.5 * abs(math.sin((u - WALK) * 7 * math.pi / 2))
                info, _ = self.codex_leaf(d, cm, Xw, Z, 2, col, tint=tint, canopy=(-3.0, 27.0 + bob))
            ui['cx'] = info
        items.append((Xw, 0.0, Z, codex))

        def clawd(d, cm):
            sh = 1 if (int(tt * 22) % 2 == 0 and (tt % 1.3) < 0.7) else 0
            self.clawd(d, cm, 'blink' if (tt % 2.7) < 0.12 else 'sad', dx_px=sh)
        items.append((CL[0], 0.0, CL[1], clawd))
        draw_world(c, cam, items)
        self.near_rain(c, cam, tt)
        info = ui.get('cx')
        if info is not None:
            x0, y0, nw, nh = info[:4]
            ix = x0 + nw * 0.72
            if THINK + 0.1 <= u < EXCL:
                q = quest_sprite(2)
                s = pop_scale(u - THINK - 0.1)
                if s > 0.05:
                    qq = squash(q, s, s)
                    blit(c, qq, ix - qq.shape[1] / 2, y0 - qq.shape[0] - 3 + math.sin(u * 9) * 1.2)
            if EXCL <= u < WALK + 0.15:
                e = excl_sprite(2)
                s = pop_scale(u - EXCL)
                if s > 0.05:
                    ee = squash(e, s, s)
                    blit(c, ee, ix - ee.shape[1] / 2, y0 - ee.shape[0] - 3)
        vignette_d(c, 0.42)
        return c

    # ------------------------------------------------------------------ шот 4: «МИР?» / «МИР!»
    def shot4(self, u):
        world = self.storm
        T = T4 - T3
        k = ease_in_out(u / T)
        cam = look_at(lerp3((-18.0, 3.4, -42.0), (-19.0, 3.6, -38.0), k),
                      lerp3((-21.0, 17.0, 0.0), (-21.0, 17.5, 0.0), k), lerp(246.0, 256.0, k))
        c = canvas()
        tt = u + T3
        self.env(c, cam, tt, world)
        self.ground_rain(c, cam, tt)
        tint = TINT_S
        items = world.tree_items() + self.trophy_items(tint) + self.plant_items(tt, tint, plucked=True)
        ARR = 0.5
        arrive = ease_out(seg(u, 0.0, ARR))
        Xc = lerp(CX_HUG[0] + 16.0, CX_HUG[0], arrive)
        Zc = CX_HUG[1]
        SAY1, LOOK, SAY2 = 0.6, 1.5, 1.85
        ui = {}

        def codex(d, cm):
            ground_shadow(d, cm, Xc, Zc, 9, alpha=0.3)
            if u < ARR:
                col = int(u * 7) % 8
                info, cinfo = self.codex_leaf(d, cm, Xc, Zc, 2, col, tint=tint, canopy=(lerp(-3.0, -9.0, arrive), 27.0))
            else:
                col = 2 if int((u - ARR) * 2.5) % 2 == 0 else 1
                face = 'happy' if u > SAY2 + 0.35 else None
                info, cinfo = self.codex_leaf(d, cm, Xc, Zc, 3, col, face=face, tint=tint,
                                              canopy=(-9.0, 27.0 + 0.4 * math.sin(tt * 3)))
            ui['cx'] = info
            ui['canopy'] = cinfo
        items.append((Xc, 0.0, Zc, codex))

        def clawd(d, cm):
            if u < LOOK:
                face = 'blink' if 1.0 < u < 1.1 else 'sad'
                sh = 1 if (int(tt * 22) % 2 == 0 and u < ARR) else 0
                ui['cl'] = self.clawd(d, cm, face, dx_px=sh)
            elif u < SAY2:
                q = seg(u, LOOK, LOOK + 0.22)
                ui['cl'] = self.clawd(d, cm, 'surprised', tears=0, Y=2.0 * math.sin(math.pi * q),
                                      sy=1.0 + 0.08 * math.sin(math.pi * q))
            else:
                q = seg(u, SAY2, SAY2 + 0.25)
                ui['cl'] = self.clawd(d, cm, 'happy', tears=0, blush=True, Y=3.0 * math.sin(math.pi * q))
        items.append((CL[0], 0.0, CL[1], clawd))
        draw_world(c, cam, items)
        stop = np.full(W, 1e9)
        ci = ui.get('canopy')
        if ci is not None:
            stopmap_from_sprite(stop, self.canopy, ci)
        self.near_rain(c, cam, tt, stopmap=stop, hit_frac=1.0, n=150)
        leaf_drips(c, self.canopy, ci, tt, seed=2, n=4)   # капли скатываются с края купола
        cd = ui.get('cx')
        cl_ = ui.get('cl', (None, None))[0]
        if cd is not None and SAY1 <= u < SAY2 + 0.45:
            x0, y0, nw, nh = cd[:4]
            draw_bubble(c, 'МИР?', x0 + nw * 0.62, y0 + nh * 0.1, u - SAY1, tail='right', seed=4, text_scale=2)
        if cl_ is not None and u >= SAY2:
            x0, y0, nw, nh = cl_[:4]
            draw_bubble(c, 'МИР!', x0 + nw * 0.3, y0 - 3, u - SAY2, tail='left', seed=5, text_scale=2)
        if cl_ is not None and cd is not None and u >= SAY2 + 0.25:
            a = u - (SAY2 + 0.25)
            hx = (cl_[0] + cl_[2] + cd[0]) / 2 - 4
            hy = min(cl_[1], cd[1]) + 6 - a * 12
            s = pop_scale(a, 0.3)
            hs = scale_nn(HEART, 3)
            if s > 0.05:
                hh = squash(hs, s, s)
                blit(c, hh, hx - hh.shape[1] / 2, hy - hh.shape[0] / 2)
        vignette_d(c, 0.42)
        return c

    # ------------------------------------------------------------------ шот 5: радуга и объятия
    CAM5 = [(0.0, (-12.0, 8.5, -60.0), (-17.0, 12.0, 0.0), 270.0),
            (1.6, (-13.0, 9.5, -54.0), (-18.0, 12.0, 0.0), 276.0),
            (3.7, (-6.0, 27.0, -106.0), (-8.0, 15.0, 0.0), 292.0)]

    def shot5(self, u):
        tt = u + T4
        cam = cam_path(u, self.CAM5)
        rain_amt = 1.0 - seg(u, 0.0, 0.8)
        clear = seg(u, 0.25, 1.35)   # фронт света идёт слева направо
        a = self.frame5(cam, u, tt, self.storm, TINT_S, rain_amt) if clear < 1 else None
        b = self.frame5(cam, u, tt, self.golden, TINT_G, 0.0) if clear > 0 else None
        if a is None:
            c = b
        elif b is None:
            c = a
        else:
            m = wipe_mask(clear, direction=(1.0, 0.45), edge=0.13, block=1)[:, :, None]
            c = np.where(m, b, a)
        vignette_d(c, 0.42 * (1 - clear) + 0.22)
        return c

    def frame5(self, cam, u, tt, world, tint, rain_amt):
        c = canvas()
        golden = world is not self.storm
        rb = None
        if golden:
            def rb(d):
                rainbow_sky(d, cam, -0.06, amount=0.95, radius_ang=0.36, reveal=ease_in_out(seg(u, 0.9, 2.3)))
        self.env(c, cam, tt, world, rainbow=rb)
        if golden:
            sun = sky_xy(cam, -0.62, 0.36)
            if sun is not None:
                amt = seg(u, 0.35, 1.3) * (1 - 0.35 * seg(u, 2.6, 3.7))
                # солнце выглядывает в разрыве туч: лучи веером + диск с дизер-ореолом
                sunbeams(c, sun[0], sun[1], tt, amount=amt, n=9, spread=1.2, base_ang=1.0, strength=0.28,
                         color=(255, 238, 176))
                if amt > 0.05:
                    for (rr, a_, col_) in ((22, 0.35, (255, 236, 190)), (14, 0.7, (255, 242, 206))):
                        bg.blit_dither_disc(c, sun[0], sun[1], rr, col_, a_ * amt)
                    r0 = 8 * min(1.0, amt * 1.5)
                    disc(c, sun[0], sun[1], r0, (255, 244, 200))
                    disc(c, sun[0], sun[1] - 1, r0 * 0.7, (255, 252, 232))
        self.ground_rain(c, cam, tt, amount=rain_amt)
        fuse = seg(u, 1.95, 2.95)
        items = world.tree_items() + self.trophy_items(tint, fuse=fuse if fuse > 0 else None) + \
            self.plant_items(tt, tint, plucked=True)
        STAND, JUMP0, HUG = 0.8, 1.0, 1.4
        ui = {}

        def clawd(d, cm):
            if u < STAND:
                ui['cl'] = self.clawd(d, cm, 'happy', tears=0, blush=True, tint=tint)
            elif u < HUG:
                q = seg(u, STAND, STAND + 0.3)
                ui['cl'] = self.clawd(d, cm, 'happy', legs=True, tears=0, blush=True, tint=tint,
                                      Y=4 * math.sin(math.pi * q))
            else:
                q = seg(u, HUG, HUG + 0.25)
                face = 'love' if ((u - HUG) % 1.5) < 1.15 else 'happy'
                sq = 1.0 - 0.1 * math.sin(math.pi * q)
                ui['cl'] = self.clawd(d, cm, face, legs=True, tears=0, blush=True, tint=tint, sy=sq, sx=2 - sq,
                                      lean_src=2)
        items.append((CL[0], 0.0, CL[1], clawd))
        HX, HZ = CL[0] + 17.0, CL[1] - 2.5

        def codex(d, cm):
            if u < 0.45:
                ground_shadow(d, cm, CX_HUG[0], CX_HUG[1], 9, alpha=0.3)
                info, _ = self.codex_leaf(d, cm, CX_HUG[0], CX_HUG[1], 3, 2, face='happy', tint=tint,
                                          canopy=(-9.0, 27.0))
            elif u < JUMP0:
                ground_shadow(d, cm, CX_HUG[0], CX_HUG[1], 9, alpha=0.3)
                info = draw_codex(d, cm, self.pix, self.cx, CX_HUG[0], CX_HUG[1], 8, 1 if int(u * 6) % 2 else 5,
                                  tint=tint)
            elif u < HUG:
                q = seg(u, JUMP0, HUG)
                X = lerp(CX_HUG[0], HX, ease_in_out(q))
                Zq = lerp(CX_HUG[1], HZ, q)
                ground_shadow(d, cm, X, Zq, 9, alpha=0.3)
                info = draw_codex(d, cm, self.pix, self.cx, X, Zq, 8, 2, Y=9 * math.sin(math.pi * q), tint=tint)
            else:
                q = seg(u, HUG, HUG + 0.25)
                ground_shadow(d, cm, HX, HZ, 9, alpha=0.3)
                sq = 1.0 - 0.1 * math.sin(math.pi * q)
                info = draw_codex(d, cm, self.pix, self.cx, HX, HZ, 4, 1, face='heart', tint=tint, sy=sq, lean_px=-2)
            ui['cx'] = info
        cz = HZ if u >= JUMP0 else CX_HUG[1]
        items.append((CX_HUG[0], 0.0, cz, codex))
        if 0.45 <= u < 1.4:
            # лопух отброшен — улетает, кувыркаясь
            q = seg(u, 0.45, 1.4)
            lx = CX_HUG[0] - 9 + 70 * q
            ly = 27 + 26 * math.sin(math.pi * q * 0.75)

            def leaf(d, cm):
                draw_spr(d, cm, self.canopy, (self.canopy.shape[1] / 2, self.canopy.shape[0] - 1), lx, ly,
                         CX_HUG[1] - 1, PLUCK[2], tint=tint, lean_src=int(round(6 * math.sin(tt * 9))),
                         sy=0.75 + 0.25 * math.cos(tt * 8))
            items.append((lx, ly, CX_HUG[1] - 1, leaf))
        draw_world(c, cam, items)
        self.near_rain(c, cam, tt, amount=rain_amt)
        cl_ = ui.get('cl', (None, None))[0]
        if cl_ is not None and u >= HUG:
            x0, y0, nw, nh = cl_[:4]
            bg.hearts(c, u - HUG, x0 + nw * 0.85, y0 - 2, n=8, seed=3, interval=0.28, life=1.9, rise=30, spread=12)
        if fuse >= 0.8:
            mid = ((TL_POS[0] + TR_POS[0]) / 2, (TL_POS[1] + TR_POS[1]) / 2 - 1)
            p = cam.project(mid[0], 5.0, mid[1])
            if p is not None:
                a = u - (1.95 + 0.8)
                if a < 0.45:
                    r = 3 + a * 50
                    for i in range(8):
                        ang = i * math.pi / 4 + 0.3
                        blit(c, SPARK, p[0] + math.cos(ang) * r - 2, p[1] + math.sin(ang) * r * 0.75 - 2)
                bg.glint(c, p[0] + 3, p[1] - 4, u, period=1.1)
                bg.sparkles(c, u, p[0] - 14, p[1] - 18, 28, 22, n=5, seed=7, period=1.0)
        if golden:
            grade_to(c, (1.02, 1.0, 0.97), (4, 2, 0))
        return c


SCENE = S5Rain()
