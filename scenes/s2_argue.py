"""s2 — спор (12.0 с). render.py: вход жёсткой склейкой, в конце белая вспышка 0.25 с.

Кадры:
  1  0.0–2.5   «через плечо» Codex (его затылок крупно справа), Clawd — официальная анимация Pointing,
               реплика «Я ЛУЧШЕ!»
  2  2.5–5.0   обратная восьмёрка: затылок Clawd слева, Codex в отчаянии (ряд 5) — «НЕТ, Я ЛУЧШЕ!»
  3  5.0–8.5   эскалация: низкая камера облетает их на ~150° вокруг кубка; спрайты меняются по углу
               (фронт / профиль / спина), реплики всё чаще и крупнее «Я!» «Я!!» «Я!!!» «Я!!!!», пар, венки
  4  8.5–10.5  аниме-сплит: огромные злые глаза Clawd слева, экран Codex '><' справа, молния через стык
  5  10.5–12.0 прыжок друг на друга -> облако драки со звёздами и «#@!%», тряска
"""
import math
import numpy as np
from px import *
import bg
from bg import PEDESTAL, TROPHY
from cam3d import *
from scene_base import Scene
from scenes._a_helpers import *

SHOTS = [(0.0, 'ots1'), (2.5, 'ots2'), (5.0, 'orbit'), (8.5, 'split'), (10.5, 'brawl')]
CL_X, CD_X = -22.0, 21.0
DEG = math.pi / 180


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ots_cam(near, far, phi_deg, rho, side, far_off_deg, h, ty, f):
    """камера «через плечо»: на расстоянии rho от ближнего героя под углом phi к его взгляду (side: +1 — ближний
    смотрит в -X, -1 — в +X); дальний герой — в far_off_deg от оси кадра."""
    ux = math.cos(math.radians(phi_deg)) * side
    uz = -math.sin(math.radians(phi_deg))
    E = (near[0] + rho * ux, near[1] + rho * uz)
    ang = math.atan2(far[0] - E[0], far[1] - E[1])
    yaw = ang - math.radians(far_off_deg)
    T = (E[0] + 50 * math.sin(yaw), ty, E[1] + 50 * math.cos(yaw))
    return look_at((E[0], h, E[1]), T, f)


class Argue(Scene):
    name = 's2_argue'
    dur = 12.0

    def __init__(self):
        self.w = get_world('day')
        self.pix = get_pix()
        self.trees_all = self.w.tree_items()
        # для фронтальных/низких планов — без деревьев прямо за кубком
        self.trees_front = [it for it in self.trees_all if not (it[2] > 0 and abs(it[0]) < 0.17 * (it[2] + 60) + 8)]
        self.grawlix = []
        cols = [(255, 70, 70), (255, 214, 64), (190, 110, 255), (255, 140, 40), (90, 200, 255)]
        for i, ch in enumerate('#@!%$&*'):
            self.grawlix.append(text_sprite(ch, cols[i % len(cols)], scale=3, outline=(30, 12, 20)))
        # сплит-экран: заготовки (фоны панелей считаются один раз)
        self.face_cl = CL.face('furious')
        self.cd_scr = CX.screen_info(5, 1, 2)
        self.split_bgL = canvas()
        self.split_bgR = canvas()
        vgradient(self.split_bgL, 0, H, [hexc('#4a100c'), hexc('#a02c1a'), hexc('#e0643a')])
        vgradient(self.split_bgR, 0, H, [hexc('#0a0f30'), hexc('#1a2e8e'), hexc('#3a64e0')])

    # ================================================================== общее
    def base(self, cam, t):
        c = canvas()
        self.w.draw_backdrop(c, cam, t)
        self.w.draw_ground(c, cam)
        return c

    def trophy_items(self, t, wobble=0.0):
        def fn(d, cm):
            ground_shadow(d, cm, 0.0, 0.0, 8.5, alpha=0.3)
            draw_sprite_3d(d, cm, PEDESTAL, 0, 0, 0, 2.0, fog=self.w.fog)
            spr = TROPHY
            if wobble:
                spr, pad = lean(TROPHY, int(round(wobble)), 1.0)
            draw_sprite_3d(d, cm, spr, 0, 8.0, 0, 2.0, fog=self.w.fog)
        return [(0.0, 0.0, 0.0, fn)]

    def glints(self, c, cam, t):
        p = proj(cam, -1.6, 15.2, 0.0)
        if p:
            bg.glint(c, p[0], p[1], t, period=1.6, offset=0.3)

    def clawd_item(self, X, Z=0.0, Y=0.0, face='furious', anim=None, frame=None, flip=False, sx=1.0, sy=1.0,
                   shadow=True, lean_px=0, tint=None):
        def fn(d, cm):
            if shadow:
                ground_shadow(d, cm, X, Z, 11 * (1 - min(0.5, Y / 16)), alpha=0.3)
            if anim is None:
                draw_clawd_3d(d, cm, CL, X, Z, Y=Y, face=face, flipx=flip, sx=sx, sy=sy, lean_px=lean_px, tint=tint)
            else:
                draw_clawd_3d(d, cm, CL, X, Z, Y=Y, anim=anim, frame=frame, flipx=flip, sx=sx, sy=sy, lean_px=lean_px,
                              tint=tint)
        return (X, Y, Z, fn)

    def codex_item(self, X, Z=0.0, Y=0.0, row=0, col=0, face=None, sx=1.0, sy=1.0, shadow=True, flip=False, tint=None):
        def fn(d, cm):
            if shadow:
                ground_shadow(d, cm, X, Z, 8 * (1 - min(0.5, Y / 16)), alpha=0.3)
            draw_codex_3d(d, cm, self.pix, CX, X, Z, row, col, Y=Y, face=face, sx=sx, sy=sy, flipx=flip, tint=tint)
        return (X, Y, Z, fn)

    def bubble(self, c, cam, text, X, Yh, Z, age, side=None, scale=1, shake=0.0, seed=0, dy=-3):
        p = proj(cam, X, Yh, Z)
        if p is None or age < 0:
            return
        tail = side or ('left' if p[0] < W / 2 else 'right')
        draw_bubble(c, text, p[0], p[1] + dy, age, tail=tail, text_scale=scale, shake=shake, seed=seed)

    # ================================================================== 1: через плечо Codex
    def pointing_frame(self, lt):
        if lt < 0.10:
            return 5
        if lt < 0.18:
            return 6
        if lt < 0.80:
            return 7 + int((lt - 0.18) / 0.07) % 9
        if lt < 0.88:
            return 16
        if lt < 0.96:
            return 17
        k = int((lt - 0.96) / 0.085)
        return 18 + k if k < 25 else 22 + (k - 25) % 21

    def shot_ots1(self, t, lt):
        u = ease_in_out(seg(lt, 0.0, 2.5))
        # медленный «подъезд» за плечом: rho уменьшается, фокус растёт — Clawd крупнее
        cam = ots_cam((CD_X, 0.0), (CL_X, 0.0), 55.0, lerp(31.0, 29.5, u), +1, lerp(-13.0, -12.0, u), 13.5, 10.5,
                      lerp(440.0, 468.0, u))
        c = self.base(cam, t)
        fr = self.pointing_frame(lt)
        hop = hop_y(lt, 0.22, 0.20, 2.0) + hop_y(lt, 0.50, 0.20, 2.0)
        items = self.trees_all + self.trophy_items(t)
        items.append(self.clawd_item(CL_X, Y=hop, anim='Pointing', frame=fr))
        # Codex со спины (в тени переднего плана): сначала держится за голову (как в конце s1),
        # от крика Clawd вздрагивает и опускает руки
        if lt < 1.02:
            crow, ccol = 5, (1 if int(lt / 0.16) % 2 == 0 else 5)
        elif lt < 1.92:
            crow, ccol = 0, 0
        else:
            # к концу кадра снова закипает и поднимает кулак — задел на ответную реплику
            crow, ccol = 5, 0
        rec = beats(lt, (1.02, 1.62), 0.3)
        sy = breathe(lt, 1.25, 0.022) + 0.05 * rec
        leanc = 2.6 * rec + 0.9 * math.sin(lt * 2.1 + 1.0)
        Yc = 0.7 * rec + (0.9 * math.sin(math.pi * (lt - 1.92) / 0.2) if 1.92 <= lt < 2.12 else 0.0)

        def dcodex_back(d, cm):
            ground_shadow(d, cm, CD_X, 0.0, 8, alpha=0.3)
            draw_codex_3d(d, cm, self.pix, CX, CD_X, 0.0, crow, ccol, Y=Yc, face='back', sy=sy,
                          lean_px=leanc, tint=(0.84, 0.86, 0.94))
        items.append((CD_X, 0.0, 0.0, dcodex_back))
        draw_world(c, cam, items)
        self.glints(c, cam, t)
        if lt > 0.3:
            p = proj(cam, CL_X - 9.0, 21.0 + hop, 0.0)
            if p:
                bg.anger_mark(c, p[0], p[1], t, 2)
        self.bubble(c, cam, 'Я ЛУЧШЕ!', CL_X - 4.0, 19.0, 0.0, lt - 1.02, side='right', scale=2)
        return c

    # ================================================================== 2: через плечо Clawd
    def shot_ots2(self, t, lt):
        u = ease_in_out(seg(lt, 0.0, 2.5))
        # камера уходит вбок и медленно «обкатывает» Clawd: спина занимает ~треть кадра и уползает за левый край
        cam = ots_cam((CL_X, 0.0), (CD_X, 0.0), lerp(73.0, 66.0, u), lerp(41.0, 39.0, u), -1,
                      lerp(8.5, 10.0, u), lerp(13.5, 12.8, u), 12.5, lerp(410.0, 436.0, u))
        c = self.base(cam, t)
        # Codex: '>_' (кулак) -> '><' (руки на голове), топает
        if lt < 0.30:
            row, col = 5, 0
        else:
            row, col = 5, [1, 1, 5, 5][int((lt - 0.30) / 0.11) % 4]
        stomps = (0.34, 0.70, 1.06, 1.60, 1.96)
        Y = 0.0
        for h0 in stomps:
            Y = max(Y, hop_y(lt, h0, 0.18, 2.2))
        items = self.trees_all + self.trophy_items(t)
        items.append(self.codex_item(CD_X, Y=Y, row=row, col=col))
        # Clawd со спины: дышит от злости и получает отдачу на каждый крик соперника
        rec = beats(lt, (0.42,) + stomps, 0.26)
        tint_b = (0.88, 0.86, 0.92)
        syb = breathe(lt, 1.4, 0.022) + 0.05 * rec
        leanb = -3.2 * rec - 0.8 * math.sin(lt * 2.4)
        Yb = 0.9 * rec

        def dback(d, cm):
            ground_shadow(d, cm, CL_X, 0.0, 11, alpha=0.3)
            clawd_draw_frame_3d(d, cm, clawd_back34(), CL_X, 0.0, Y=Yb, anchor_xy=CL.anchor['CrabWalking'],
                                tint=tint_b, sy=syb, lean_px=leanb)
        items.append((CL_X, 0.0, 0.0, dback))
        draw_world(c, cam, items)
        self.glints(c, cam, t)
        if lt > 0.3:
            p = proj(cam, CD_X + 10.5, 21.0 + Y, 0.0)
            if p:
                bg.anger_mark(c, p[0], p[1], t + 0.2, 2)
            p = proj(cam, CD_X, 23.0 + Y, 0.0)
            if p:
                bg.steam(c, t, p[0], p[1] - 2, seed=2, n=3)
        # Clawd тоже кипит: пар над его затылком
        p = proj(cam, CL_X - 2.0, 17.0 + Yb, 0.0)
        if p and lt > 0.5:
            bg.steam(c, t + 0.4, p[0], p[1] - 2, seed=6, n=2, color=(246, 238, 232))
        self.bubble(c, cam, 'НЕТ, Я ЛУЧШЕ!', CD_X + 2.0, 24.5, 0.0, lt - 0.42, side='right', scale=2)
        return c

    # ================================================================== 3: облёт
    def orbit_angle(self, lt):
        return lerp(-92.0, 90.0, ease_in_out(seg(lt, 0.0, 3.5))) * DEG

    def orbit_cam(self, lt):
        u = seg(lt, 0.0, 3.5)
        a = self.orbit_angle(lt)
        r = lerp(47.0, 37.0, ease_in_out(u))
        h = lerp(3.4, 2.6, u)
        f = lerp(240.0, 252.0, u)
        return orbit((0.0, 0.0), r, a, h, f=f, look_height=11.0)

    def pair_pos(self, lt):
        """«поворотный стол»: пара кружит вокруг кубка вслед за камерой (чуть отставая) — оба всегда видны."""
        a = self.orbit_angle(lt) * 0.92
        ax, az = math.cos(a), -math.sin(a)  # экранное «вправо» при угле камеры a
        return (CL_X * ax, CL_X * az), (CD_X * ax, CD_X * az), a

    def shot_orbit(self, t, lt):
        cam = self.orbit_cam(lt)
        (clx, clz), (cdx, cdz), pa = self.pair_pos(lt)
        amp = 0.0
        # реплики: (кто, текст, время, масштаб, тряска)
        lines = [('cl', 'Я!', 0.10, 2, 0.0), ('cd', 'Я!!', 0.78, 2, 0.0), ('cl', 'Я!!!', 1.46, 2, 1.0),
                 ('cd', 'Я!!!!', 2.14, 3, 2.0)]
        # ---- Clawd: кричит (указывает / злая морда + прыжок) и слушает (профиль)
        Ycl, sx, sy = 0.0, 1.0, 1.0
        if lt < 0.10:
            cl_kw = dict(anim='CrabWalking', frame=4)
        elif lt < 0.72:
            k = int((lt - 0.10) / 0.085)
            cl_kw = dict(anim='Pointing', frame=min(42, 18 + k))
            Ycl = hop_y(lt, 0.10, 0.2, 2.2)
        elif lt < 1.46:
            cl_kw = dict(face='furious')
            k = (lt * 5.0) % 1.0
            if k < 0.14:
                sx, sy = 1.08, 0.92
        elif lt < 2.14:
            k = int((lt - 1.46) / 0.085)
            cl_kw = dict(anim='Pointing', frame=min(42, 18 + k))
            Ycl = hop_y(lt, 1.46, 0.22, 3.0)
        else:
            if lt < 2.6:
                fr = anim_frame((lt - 2.14) * 2.4, CL.meta['Jumping']['durations'], loop=False, start=4, end=15)
                cl_kw = dict(anim='Jumping', frame=fr)
            else:
                cl_kw = dict(face='furious')
                k = (lt * 6.0) % 1.0
                if k < 0.16:
                    sx, sy = 1.1, 0.9
        # ---- Codex
        Ycd = 0.0
        if lt < 0.78:
            cd_kw = dict(row=10, col=7)
            Ycd = 0.0
        elif lt < 1.46:
            col = [1, 2][int((lt - 0.78) / 0.12) % 2]
            cd_kw = dict(row=5, col=col)
            Ycd = max(hop_y(lt, 0.78, 0.2, 2.6), hop_y(lt, 1.08, 0.2, 2.0))
        elif lt < 2.14:
            cd_kw = dict(row=10, col=7)
            Ycd = hop_y(lt, 1.52, 0.14, 0.8)
        else:
            col = [1, 2, 1, 2][int((lt - 2.14) / 0.1) % 4]
            cd_kw = dict(row=5, col=col)
            for h0 in (2.14, 2.46, 2.78, 3.10):
                Ycd = max(Ycd, hop_y(lt, h0, 0.17, 3.0))
        c = self.base(cam, t)
        items = self.trees_all + self.trophy_items(t)
        items.append(self.clawd_item(clx, clz, Y=Ycl, sx=sx, sy=sy, **cl_kw))
        items.append(self.codex_item(cdx, cdz, Y=Ycd, **cd_kw))
        draw_world(c, cam, items)
        self.glints(c, cam, t)
        # пар и венки
        for (X, Z, Yh, Y, ph) in ((clx, clz, 17.0, Ycl, 0.0), (cdx, cdz, 22.5, Ycd, 0.5)):
            p = proj(cam, X, Yh + Y, Z)
            if p:
                bg.steam(c, t + ph, p[0], p[1] - 2, seed=3, n=3)
        ax, az = math.cos(pa), -math.sin(pa)
        pa_ = proj(cam, clx + 11.0 * ax, 15.5 + Ycl, clz + 11.0 * az)
        if pa_:
            bg.anger_mark(c, pa_[0], pa_[1], t, 2)
        pb_ = proj(cam, cdx + 10.0 * ax, 20.5 + Ycd, cdz + 10.0 * az)
        if pb_:
            bg.anger_mark(c, pb_[0], pb_[1], t + 0.25, 2)
        # реплики: каждая живёт до следующей своей (>= 1.2 с)
        for i, (who, txt, t0, sc, shk) in enumerate(lines):
            nxt = [l[2] for l in lines[i + 1:] if l[0] == who]
            t_end = nxt[0] if nxt else 9.9
            if not (t0 <= lt < t_end):
                continue
            if who == 'cl':
                self.bubble(c, cam, txt, clx - 3 * ax, 19.0 + Ycl, clz - 3 * az, lt - t0, side='left', scale=sc,
                            shake=shk, seed=i)
            else:
                self.bubble(c, cam, txt, cdx + 3 * ax, 25.0 + Ycd, cdz + 3 * az, lt - t0, side='right', scale=sc,
                            shake=shk, seed=i)
            if shk and lt - t0 < 0.5:
                amp = max(amp, shk * 1.5 * (1 - (lt - t0) / 0.5))
        dx, dy = shake_offset(t, amp, seed=11)
        return shift_canvas(c, dx, dy)

    # ================================================================== 4: сплит-экран
    def shot_split(self, t, lt):
        c = canvas()
        # диагональный стык: x = sx0 + (sx1 - sx0) * y/H
        sx0, sx1 = 272.0, 208.0
        edge = sx0 + (sx1 - sx0) * (YY[:, 0] / (H - 1))
        left = XX < edge[:, None]
        # фоны панелей: тёплый и холодный + фокус-линии
        cL = self.split_bgL.copy()
        cR = self.split_bgR.copy()
        focus_lines(cL, 118, 120, t, n=44, seed=3, color=(255, 196, 150), inner=80, alpha=0.8)
        focus_lines(cR, 364, 120, t, n=44, seed=9, color=(170, 200, 255), inner=80, alpha=0.8)
        # въезд панелей
        inL = ease_out(seg(lt, 0.0, 0.14))
        inR = ease_out(seg(lt, 0.05, 0.19))
        offL = int(round((1 - inL) * -280))
        offR = int(round((1 - inR) * 280))
        # два рывка-наезда (аниме «зум-панч»)
        stage = 0 if lt < 0.9 else (1 if lt < 1.5 else 2)
        punch = (0.9 <= lt < 0.98) or (1.5 <= lt < 1.58)
        jit = shake_offset(t, 1.0 + stage * 0.8 + (2.0 if punch else 0.0), seed=5)
        # --- Clawd: яростные глаза
        k = (14, 17, 20)[stage]
        f = scale_nn(self.face_cl, k)
        ex, ey = 12.0 * k, 3.0 * k  # точка между глаз
        fx0 = 118 - ex + offL + jit[0]
        fy0 = 120 - ey + jit[1]
        blit(cL, f, fx0, fy0)
        # тень над глазами (как в аниме) — дизером
        y_a, y_b = int(fy0), int(fy0 + 2.0 * k)
        for y in range(max(0, y_a), min(H, y_b)):
            q = 0.5 * (1 - (y - y_a) / max(1, y_b - y_a))
            m = (DITHER[y] < q) & (XX[y] >= fx0) & (XX[y] < fx0 + f.shape[1])
            cL[y, m] *= 0.55
        # «дрожь ярости» у краёв головы
        for (sgn, xb) in ((-1, fx0 - 4), (1, fx0 + f.shape[1] + 3)):
            for j in range(3):
                yy_ = fy0 + k * (1.2 + j * 1.3)
                line(cL, xb, yy_, xb + sgn * 8, yy_ - 3, (255, 230, 200))
        # --- Codex: экран '><'
        cell = CX.frame(5, 1, 2)
        x0, y0, x1, y1 = self.cd_scr['bbox']
        kk = (5, 6, 7)[stage]
        g = scale_nn(cell, kk)
        cx_, cy_ = (x0 + x1) / 2 * kk, (y0 + y1) / 2 * kk
        blit(cR, g, 364 - cx_ + offR - jit[0], 120 - cy_ - jit[1])
        c[left] = cL[left]
        c[~left] = cR[~left]
        # толстый стык-молния между панелями
        for y in range(H):
            xe = int(edge[y])
            c[y, max(0, xe - 4):xe + 4] = (16, 10, 18)
            c[y, max(0, xe - 2):xe + 2] = (255, 250, 232)
        # молния через стык: от глаза к глазу
        if lt > 0.28:
            a = lt - 0.28
            on = a < 0.5 or (a % 0.3) < 0.22 or punch
            if on:
                xl = fx0 + 16.8 * k
                yl = fy0 + 3.0 * k
                xr = 364 - cx_ + offR - jit[0] + (x0 + (x1 - x0) * 0.30) * kk
                yr = 120 - 2
                th = 2 + stage
                bolt(c, xl, yl, xr, yr, seed=int(t * 24), jag=14.0 + 4 * stage, step=20.0, thick=th,
                     core=(255, 255, 250), glow=(255, 226, 90))
                bolt(c, xl, yl + 4, xr, yr + 6, seed=int(t * 24) + 7, jag=10.0, step=26.0, thick=1,
                     core=(255, 255, 255), glow=(150, 220, 255), branches=False)
                # вспышка-искра в точке встречи на стыке
                mx = 240
                st = scale_nn(STAR5 if int(t * 20) % 2 else SPARK, 2 + stage)
                blit(c, st, mx - st.shape[1] / 2, (yl + yr) / 2 - st.shape[0] / 2)
        # вспышки: первый разряд и рывки
        if 0.28 <= lt < 0.36:
            bg.flash(c, 0.6 * (1 - (lt - 0.28) / 0.08))
        if punch:
            bg.flash(c, 0.35)
        # горизонтальные штрихи скорости поверх
        bg.speed_lines(c, t, 4, 36, n=5, seed=2, speed=900, color=(255, 240, 220), alpha=0.9, length=(20, 60))
        bg.speed_lines(c, t, 226, 266, n=5, seed=4, speed=900, color=(220, 235, 255), alpha=0.9, length=(20, 60),
                       direction=1)
        return c

    # ================================================================== 5: драка
    def shot_brawl(self, t, lt):
        T_HIT = 0.34
        u = seg(lt, 0.0, 1.5)
        k = ease_in_out(seg(lt, 0.25, 1.5))
        cam = look_at((lerp(-3.0, 0.0, u), lerp(5.0, 6.5, k), lerp(-60.0, -47.0, k)), (0.0, lerp(9.5, 10.5, k), -6.0),
                      lerp(250.0, 262.0, u))
        amp = 0.0
        if lt >= T_HIT:
            amp = 1.4 + decay_shake(lt, T_HIT, 0.35, 3.0)
        c = self.base(cam, t)
        wob = 0.0
        if lt >= T_HIT:
            wob = math.sin((lt - T_HIT) * 38) * 1.4 * math.exp(-(lt - T_HIT) * 2.0)
        items = self.trees_front + self.trophy_items(t, wobble=wob)
        cxw, cyw, czw = 0.0, 8.5, -8.0
        if lt < T_HIT:
            # присед и прыжок навстречу
            if lt < 0.10:
                items.append(self.clawd_item(CL_X, face='furious', sx=1.12, sy=0.86))
                items.append(self.codex_item(CD_X, row=4, col=0))
            else:
                q = seg(lt, 0.10, T_HIT)
                e = ease_in(q)
                Xc = lerp(CL_X, cxw - 6.0, e)
                Zc = lerp(0.0, czw, e)
                Yc = lerp(0.0, cyw - 6.0, e) + 7.0 * math.sin(math.pi * q)
                items.append(self.clawd_item(Xc, Zc, Y=Yc, face='furious', sx=0.92, sy=1.12, shadow=True,
                                             lean_px=3))
                Xd = lerp(CD_X, cxw + 6.0, e)
                Yd = lerp(0.0, cyw - 8.0, e) + 7.0 * math.sin(math.pi * q)
                items.append(self.codex_item(Xd, Zc, Y=Yd, row=4, col=2))
        draw_world(c, cam, items)
        pc = proj(cam, cxw, cyw, czw)
        if lt >= T_HIT and pc:
            a = lt - T_HIT
            s = pc[2]
            R = 12.5 * s * (0.55 + 0.45 * ease_out_back(seg(a, 0.0, 0.18), 2.0))
            # пыль вокруг облака
            for i in range(4):
                dust_world(c, cam, cxw + (i - 1.5) * 10.0, czw, (a + i * 0.17) % 0.6, seed=40 + i, units=1.5,
                           life=0.6, n=6)
            # головы и конечности высовываются из-за края облака (рисуем до облака)
            self.brawl_heads(c, a, pc, s, R)
            self.brawl_limbs(c, a, pc, s, R)
            brawl_cloud2(c, t, pc[0], pc[1], R, seed=11)
            # «линии действия» вокруг облака
            rng = np.random.default_rng(int(t * 20))
            for i in range(5):
                ang = rng.uniform(0, 2 * math.pi)
                r0 = R * rng.uniform(1.42, 1.6)
                x0_, y0_ = pc[0] + math.cos(ang) * r0, pc[1] + math.sin(ang) * r0 * 0.62
                x1_, y1_ = pc[0] + math.cos(ang) * (r0 + 10), pc[1] + math.sin(ang) * (r0 + 10) * 0.62
                line(c, x0_, y0_, x1_, y1_, (60, 50, 70))
            # символы ругани
            self.grawlix_fx(c, a, pc, s, R)
            # удар: вспышка и звезда
            if a < 0.08:
                bg.flash(c, 0.7 * (1 - a / 0.08))
            if a < 0.22:
                st = scale_nn(STAR5, 5 if a < 0.1 else 4)
                blit(c, st, pc[0] - st.shape[1] / 2, pc[1] - st.shape[0] / 2 - R * 0.2)
        dx, dy = shake_offset(t, amp, seed=21)
        return shift_canvas(c, dx, dy)

    def brawl_limbs(self, c, a, pc, s, R):
        """ножки Clawd и «варежки» Codex то тут, то там торчат из-за края облака."""
        u = max(2.0, s)
        slots = [(-1.02, 0.34, 'cl_leg'), (1.04, 0.12, 'cd_hand'), (-0.50, 0.64, 'cd_hand'), (0.60, 0.60, 'cl_leg'),
                 (-1.10, -0.04, 'cd_hand'), (1.08, -0.22, 'cl_claw'), (0.18, 0.72, 'cl_leg')]
        k = int(a / 0.12)
        for j in range(3):
            ox, oy, kind = slots[(k + j * 3) % len(slots)]
            x = pc[0] + ox * R * 0.95
            y = pc[1] + oy * R * 0.62
            if kind in ('cl_leg', 'cl_claw'):
                col = hexc('#D97757')
                w_, h_ = (2 * u, 5 * u) if kind == 'cl_leg' else (5 * u, 2 * u)
                rect(c, x - w_ / 2 - 1, y - h_ / 2 - 1, w_ + 2, h_ + 2, (60, 26, 18))
                rect(c, x - w_ / 2, y - h_ / 2, w_, h_, col)
            else:
                r_ = 1.7 * u
                disc(c, x, y, r_ + 1, (16, 22, 60))
                disc(c, x, y, r_, hexc('#4A6CF0'))
                disc(c, x - r_ * 0.3, y - r_ * 0.3, r_ * 0.45, hexc('#7F9BFF'))

    def brawl_heads(self, c, a, pc, s, R):
        """по очереди из-за верхнего края облака высовываются головы (рисуются ДО облака — низ закрыт им)."""
        k = int(a / 0.2)
        ph = (a % 0.2) / 0.2
        pop = math.sin(math.pi * min(1.0, ph / 0.85))
        edge = pc[1] - R * 0.6 * 0.78 - R * 0.2  # примерная верхняя кромка облака
        if k % 2 == 0:
            spr = CL.face('furious' if k % 4 == 0 else 'dizzy')[:8]
            sc = max(2, int(round(s)))
            spr = scale_nn(spr, sc)
            h = spr.shape[0]
            x = pc[0] - R * 0.38 - spr.shape[1] / 2
            blit(c, spr, x, edge - h * 0.8 * pop + h * 0.15)
        else:
            full = self.pix.get(5, 2 if k % 4 == 1 else 1, 24.0 * s)
            ys, xs = np.where(full[:, :, 3] > 0)
            hh = int((ys.max() - ys.min()) * 0.62)
            head = full[ys.min():ys.min() + hh, xs.min():xs.max() + 1]
            h = head.shape[0]
            x = pc[0] + R * 0.36 - head.shape[1] / 2
            blit(c, head, x, edge - h * 0.75 * pop + h * 0.2)

    def grawlix_fx(self, c, a, pc, s, R):
        rng = np.random.default_rng(5)
        for i in range(9):
            t0 = 0.05 + i * 0.1
            age = a - t0
            if age < 0 or age > 0.6:
                continue
            ang = rng.uniform(-math.pi * 0.92, -math.pi * 0.08)
            sp = rng.uniform(0.9, 1.3)
            x = pc[0] + math.cos(ang) * R * (0.6 + sp * age * 2.2)
            y = pc[1] + math.sin(ang) * R * (0.45 + sp * age * 1.6) + 90 * age * age
            spr = self.grawlix[i % len(self.grawlix)]
            k = pop_scale(age, 0.12)
            if k < 0.999:
                spr = squash(spr, max(0.1, k), max(0.1, k))
            blit(c, spr, x - spr.shape[1] / 2, y - spr.shape[0] / 2)

    # ================================================================== кадр
    def render(self, t):
        name, start = 'ots1', 0.0
        for (s_, n) in SHOTS:
            if t >= s_:
                name, start = n, s_
        return getattr(self, 'shot_' + name)(t, t - start)


SCENE = Argue()
