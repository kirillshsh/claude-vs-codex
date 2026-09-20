"""Сцена 3 — «РАУНД 1: КОД» (9.0 c).
0.0–1.3  карточка раунда: код бежит по фону, гонг, «РАУНД 1» + «КОД».
1.3–7.8  ночная хакер-арена 2.5D: неоновый Mode-7 пол, две станции друг против друга, голо-терминалы,
         HUD-прогресс «шея в шею». Камера: медленная орбита -> быстрые наезды на каждого -> крупняки ->
         драматичный нижний ракурс.
7.8–9.0  оба 100% в один кадр -> штамп «НИЧЬЯ!» с ударом, оба в бешенстве разворачиваются друг к другу.
"""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator
from px import *
from bg import SPARK, SPARK_S, STAR5, SWEAT, ANGER, ANGER2, blit_dither, vignette, flash
from cam3d import Cam, look_at, orbit, Panorama, CodexPixelizer
from chars import load
from scene_base import Scene
from scenes._b_helpers import *

# ---------------------------------------------------------------- мир
SXC, SXX = -46.0, 46.0            # центры станций Clawd / Codex
ST_HW, ST_HD, ST_H = 16.0, 12.0, 7.0
CL_AX = SXC - 6.0                 # якорь Clawd (лапы), ноутбук у него справа (+X)
CX_AX = SXX - 2.0                 # якорь Codex (не зеркалим — глиф '>_' должен смотреть вправо)
HOLO_W, HOLO_H, HOLO_Y, HOLO_Z = 48.0, 32.0, 60.0, 6.0   # HOLO_Y — верхний край

ORANGE = np.array([255, 146, 64], np.float32)
ORANGE_L = np.array([255, 214, 150], np.float32)
ORANGE_D = np.array([120, 46, 20], np.float32)
BLUE = np.array([80, 168, 255], np.float32)
BLUE_L = np.array([180, 230, 255], np.float32)
BLUE_D = np.array([24, 52, 130], np.float32)
MAGENTA = np.array([255, 120, 230], np.float32)
FOG = np.array([38, 16, 58], np.float32)
SKY = ['#06050e', '#0c0a20', '#18113c', '#2e1654', '#4e1f68']
INK = (24, 12, 36)

T_CARD, T_FIN = 1.3, 7.8

# прогресс (ключи — моменты смены лидера совпадают с монтажом)
PK_T = [1.3, 1.8, 2.3, 2.8, 3.3, 3.95, 4.6, 5.2, 5.8, 6.5, 7.1, 7.5, 7.8]
PK_CL = [0, 6, 14, 22, 31, 47, 55, 64, 73, 84, 93, 98, 100]
PK_CX = [0, 5, 12, 24, 33, 44, 57, 63, 71, 85, 92, 99, 100]

CODE_SNIPS = [
    'def win():', '  return True', 'if claude > codex:', 'while True:', '  fight()', 'import speed',
    'git push --force', 'npm run battle', 'for i in range(100):', '  code += 1', tr('// TODO: победить'),
    tr('print("Я лучше!")'), 'return 42;', 'sudo make win', 'async function go() {', '  await race();', '}',
    'let x = 0x7F;', 'class Hero:', '  self.best = True', 'assert me > you', 'fn main() {', 'mov eax, 1',
    '#include <win.h>', 'SELECT * FROM wins', 'try: crush()', 'except: retry()', 'const v = "max";',
    '>>> 1 + 1', 'git commit -m "win"', 'yield victory', 'if (!lose) {', 'deploy --prod', 'λx.x',
]


def _text_rgba(s, color, font='small', size=None):
    m = text_mask(s, font, size)
    o = np.zeros(m.shape + (4,), np.uint8)
    o[m] = list(np.asarray(color, np.uint8)) + [255]
    return o


def _paste(dst, spr, x, y):
    """вставка RGBA в RGBA-картинку (без смешивания, по маске)."""
    h, w = spr.shape[:2]
    H_, W_ = dst.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W_, x + w), min(H_, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    m = s[:, :, 3] > 0
    dst[y0:y1, x0:x1][m] = s[m]


# ---------------------------------------------------------------- фон арены
def make_arena_pano(clawd, codex, pix, f_ref=260.0, seed=12):
    pw = int(2 * math.pi * f_ref)
    h = 150
    img = np.zeros((h, pw, 4), np.uint8)
    rng = np.random.default_rng(seed)

    def az_of(x):
        a = x / pw * 2 * math.pi
        return (a + math.pi) % (2 * math.pi) - math.pi

    def rect(x0, y0, w, hh, col):
        xs = np.arange(x0, x0 + w) % pw
        y0c, y1c = max(0, y0), min(h, y0 + hh)
        if y0c >= y1c:
            return
        img[y0c:y1c][:, xs, :3] = col
        img[y0c:y1c][:, xs, 3] = 255

    # звёзды
    for i in range(300):
        x, y = int(rng.integers(0, pw)), int(rng.integers(0, 90))
        b = rng.uniform(0.35, 1.0)
        c = np.array([230, 220, 255]) * b
        img[y, x, :3] = c
        img[y, x, 3] = 255
    # дальний город
    x = 0
    while x < pw:
        w = int(rng.integers(12, 36))
        top = int(rng.integers(38, 104))
        col = (22, 14, 44) if rng.random() < 0.5 else (28, 18, 52)
        rect(x, top, w, h - top, col)
        rect(x, top, w, 1, (48, 30, 80))
        if top < 60 and rng.random() < 0.6:
            ax_ = x + w // 2
            rect(ax_, top - 8, 1, 8, (40, 26, 70))
            img[top - 9, ax_ % pw, :3] = (255, 60, 80)
            img[top - 9, ax_ % pw, 3] = 255
        for yy in range(top + 3, h - 36, 4):
            for xx in range(x + 2, x + w - 2, 3):
                if rng.random() < 0.2:
                    wc = [(255, 200, 120), (120, 220, 255), (255, 120, 210), (200, 170, 255)][int(rng.integers(0, 4))]
                    img[yy, xx % pw, :3] = np.array(wc) * rng.uniform(0.45, 0.8)
                    img[yy, xx % pw, 3] = 255
        x += w + int(rng.integers(0, 5))
    # трибуны
    rect(0, 116, pw, h - 116, (12, 9, 26))
    for ry, rc in ((124, (30, 22, 52)), (132, (26, 19, 46)), (140, (22, 16, 40))):
        rect(0, ry, pw, 1, rc)
    # толпа: головы + светящиеся палочки фанатов (оранжевые со стороны Clawd, синие — Codex)
    for xx in range(0, pw, 3):
        hy = 113 + int(rng.integers(0, 3))
        rect(xx, hy, 2, 118 - hy, (8, 6, 18))
        rect(xx, hy + 1, 3, 1, (8, 6, 18))
        if rng.random() < 0.3:
            az = az_of(xx)
            side = az < 0
            if abs(az) < 0.12:
                side = rng.random() < 0.5
            c = (255, 150, 70) if side else (90, 180, 255)
            ly = hy - int(rng.integers(1, 4))
            img[ly, xx % pw, :3] = c
            img[ly, xx % pw, 3] = 255
            img[ly + 1, xx % pw, :3] = np.array(c) * 0.6
            img[ly + 1, xx % pw, 3] = 255
    for xx in range(0, pw, 2):
        for ry in (127, 135, 143):
            if rng.random() < 0.18:
                az = az_of(xx)
                c = (255, 150, 70) if az < 0 else (90, 180, 255)
                img[ry, xx % pw, :3] = np.array(c) * rng.uniform(0.5, 1.0)
                img[ry, xx % pw, 3] = 255

    # экраны-«джамботроны»
    def screen(az, label, col, face_spr):
        cxp = int(round(az / (2 * math.pi) * pw)) % pw
        sw, sh = 70, 38
        x0, y0 = cxp - sw // 2, 74
        rect(x0 - 2, y0 - 2, sw + 4, sh + 4, (70, 66, 96))
        rect(x0 - 1, y0 - 1, sw + 2, sh + 2, (24, 22, 36))
        rect(x0, y0, sw, sh, (10, 12, 26))
        for yy in range(y0, y0 + sh, 2):
            rect(x0, yy, sw, 1, (14, 16, 34))
        rect(cxp - 1, y0 + sh + 2, 3, 116 - (y0 + sh + 2), (40, 36, 60))
        t = _text_rgba(label, col, 'big')
        fh, fw = face_spr.shape[:2]
        gx = cxp - fw // 2
        for yy in range(fh):
            for xx in range(fw):
                if face_spr[yy, xx, 3] > 0:
                    img[y0 + 4 + yy, (gx + xx) % pw, :3] = face_spr[yy, xx, :3]
                    img[y0 + 4 + yy, (gx + xx) % pw, 3] = 255
        th, tw = t.shape[:2]
        tx = cxp - tw // 2
        for yy in range(th):
            for xx in range(tw):
                if t[yy, xx, 3] > 0:
                    img[y0 + sh - th - 4 + yy, (tx + xx) % pw, :3] = col
                    img[y0 + sh - th - 4 + yy, (tx + xx) % pw, 3] = 255
        rect(x0, y0 + sh - 1, sw, 1, np.array(col) * 0.5)

    screen(-0.98, 'CLAUDE', (255, 150, 70), clawd.face('angry'))
    screen(0.98, 'CODEX', (90, 180, 255), pix.get(0, 0, 24))
    return Panorama(img, f_ref, base_row=h - 1)


# ---------------------------------------------------------------- фильтрованная сетка
def filt_lines(x, fp, G, lw):
    """доля пикселя, покрытая линиями ширины lw с шагом G (центры в k*G). Бокс-фильтр — без муара."""
    fp = np.maximum(fp, 1e-3)
    half = lw * 0.5

    def F(s):
        s = s + half
        return np.floor(s / G) * lw + np.minimum(np.mod(s, G), lw)
    return (F(x + fp * 0.5) - F(x - fp * 0.5)) / fp


def single_line(d, fp, lw):
    """покрытие одиночной линии |d| < lw/2 (бокс-фильтр)."""
    fp = np.maximum(fp, 1e-3)
    a = np.clip(d - fp * 0.5, -lw * 0.5, lw * 0.5)
    b = np.clip(d + fp * 0.5, -lw * 0.5, lw * 0.5)
    return (b - a) / fp


# ---------------------------------------------------------------- сцена
class CodeScene(Scene):
    name = 's3_code'
    dur = 9.0

    def __init__(self):
        self.cl, self.cx = load()
        self.pix = CodexPixelizer()
        self.pano = make_arena_pano(self.cl, self.cx, self.pix)
        self.p_cl = PchipInterpolator(PK_T, PK_CL)
        self.p_cx = PchipInterpolator(PK_T, PK_CX)
        rng = np.random.default_rng(3)
        # «код» для голограмм: строки токенов (ширина в текселях, цвет)
        self.code = {}
        for side in ('cl', 'cx'):
            lines = []
            for i in range(44):
                ind = int(rng.choice([0, 0, 1, 1, 2, 3])) * 4
                toks = []
                n = int(rng.integers(2, 6))
                for k in range(n):
                    toks.append((int(rng.integers(3, 13)), int(rng.integers(0, 6))))
                lines.append((ind, toks))
            self.code[side] = lines
        # суммарная длина кода (для синхронизации с прогрессом)
        self.code_len = {s: sum(ind + sum(w + 2 for w, _ in toks) for ind, toks in L) for s, L in self.code.items()}
        # фон карточки: стена кода
        self.card_wall = self._make_code_wall(seed=4)
        self.card_wall2 = self._make_code_wall(seed=9, big=True)
        # таблички станций
        self.plate = {'cl': self._make_plate('CLAUDE', ORANGE), 'cx': self._make_plate('CODEX', BLUE)}
        self.side_tex = {'cl': self._make_side(ORANGE), 'cx': self._make_side(BLUE)}
        # штамп
        self.stamp = self._make_stamp('НИЧЬЯ!')
        # портреты для карточки
        self.port_l = scale_nn(self.cl.face('angry'), 3)
        self.port_r = self.pix.get(0, 0, 66)
        self.bell = scale_nn(from_ascii([
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
        ], {'o': (70, 40, 10), 'Y': (250, 200, 70), 'W': (255, 250, 210), 'S': (200, 130, 30),
            'B': (120, 80, 40)}), 4)
        # кэш текста HUD
        self.hud_lbl = {'cl': text_sprite('CLAUDE', tuple(ORANGE_L.astype(int)), outline=INK),
                        'cx': text_sprite('CODEX', tuple(BLUE_L.astype(int)), outline=INK)}

    # ============================================================ таймлайн прогресса
    def prog(self, t):
        if t <= PK_T[0]:
            return 0.0, 0.0
        if t >= PK_T[-1]:
            return 100.0, 100.0
        return float(np.clip(self.p_cl(t), 0, 100)), float(np.clip(self.p_cx(t), 0, 100))

    # ============================================================ ассеты
    def _make_code_wall(self, seed, big=False):
        rng = np.random.default_rng(seed)
        hh = 360
        img = np.zeros((hh, W, 3), np.float32)
        cols = [(255, 121, 198), (80, 250, 123), (241, 250, 140), (230, 230, 240), (189, 147, 249), (98, 114, 164),
                (255, 170, 90), (110, 200, 255)]
        step = 18 if big else 11
        for y in range(0, hh - 10, step):
            x = int(rng.integers(-30, 20))
            while x < W:
                s = CODE_SNIPS[int(rng.integers(0, len(CODE_SNIPS)))]
                m = text_mask(s, 'big' if big else 'small')
                c = np.array(cols[int(rng.integers(0, len(cols)))], np.float32)
                h_, w_ = m.shape
                x0, x1 = max(0, x), min(W, x + w_)
                if x1 > x0:
                    sub = img[y:y + h_, x0:x1]
                    mm = m[:, x0 - x:x1 - x]
                    sub[mm] = c
                x += w_ + int(rng.integers(14, 60 if not big else 120))
        return img

    def _make_plate(self, label, col):
        # лицевая панель станции (текстура 64x14 -> 32x7 юнитов)
        w, h = 64, 14
        img = np.zeros((h, w, 4), np.uint8)
        img[:, :, :3] = (22, 22, 36)
        img[:, :, 3] = 255
        img[0, :, :3] = col
        img[1, :, :3] = np.asarray(col) * 0.45
        img[-1, :, :3] = (10, 10, 18)
        img[:, 0, :3] = (14, 14, 24)
        img[:, -1, :3] = (14, 14, 24)
        t = _text_rgba(label, col, 'big')
        th, tw = t.shape[:2]
        _paste(img, t, (w - tw) // 2, 4)
        # огоньки по краям
        for x in (4, 6, w - 7, w - 5):
            img[6:8, x, :3] = np.asarray(col)
        return img

    def _make_side(self, col):
        w, h = 48, 14
        img = np.zeros((h, w, 4), np.uint8)
        img[:, :, :3] = (30, 30, 48)
        img[:, :, 3] = 255
        img[0, :, :3] = col
        img[1, :, :3] = np.asarray(col) * 0.4
        for x in range(4, w - 4, 6):
            img[5:10, x:x + 3, :3] = (20, 20, 32)
            img[5, x:x + 3, :3] = np.asarray(col) * 0.5
        return img

    def _make_stamp(self, s):
        red = np.array([226, 44, 52], np.float32)
        red2 = np.array([186, 28, 40], np.float32)
        m = text_mask(s, 'big')
        k = 4
        m = np.repeat(np.repeat(m, k, 0), k, 1)
        th, tw = m.shape
        pad = 11
        w, h = tw + pad * 2, th + pad * 2
        full = np.zeros((h, w), bool)
        full[pad:pad + th, pad:pad + tw] = m
        # двойная рамка
        fr = np.zeros((h, w), bool)
        fr[0:4, :] = True
        fr[-4:, :] = True
        fr[:, 0:4] = True
        fr[:, -4:] = True
        fr[6:7, 6:-6] = True
        fr[-7:-6, 6:-6] = True
        fr[6:-6, 6:7] = True
        fr[6:-6, -7:-6] = True
        ink = full | fr
        rng = np.random.default_rng(5)
        noise = rng.random((h // 2 + 1, w // 2 + 1))
        noise = np.repeat(np.repeat(noise, 2, 0), 2, 1)[:h, :w]
        holes = noise > 0.93
        ink2 = ink & ~holes
        ol = outline_mask(ink, 1) & ~ink
        out = np.zeros((h + 2, w + 2, 4), np.uint8)
        big = np.zeros((h + 2, w + 2), bool)
        big[1:-1, 1:-1] = ink2
        olb = np.zeros((h + 2, w + 2), bool)
        olb[1:-1, 1:-1] = ol
        olb = outline_mask(big, 1) & ~big
        out[olb] = [255, 244, 228, 255]
        tone = np.where((noise > 0.55)[..., None], red2, red)
        rgb = np.zeros((h + 2, w + 2, 3), np.float32)
        rgb[1:-1, 1:-1] = tone
        out[big, :3] = rgb[big].astype(np.uint8)
        out[big, 3] = 255
        return out

    # ============================================================ голограмма (текстура на кадр)
    def holo_tex(self, side, t, prog, big=True):
        col = ORANGE if side == 'cl' else BLUE
        colL = ORANGE_L if side == 'cl' else BLUE_L
        w, h = (96, 64) if big else (48, 32)
        k = 1 if big else 0.5
        img = np.zeros((h, w, 4), np.uint8)
        img[:, :, :3] = (8, 12, 28) if side == 'cx' else (20, 10, 16)
        img[:, :, 3] = 212
        # рамка + заголовок
        img[0, :, :3] = col
        img[-1, :, :3] = col
        img[:, 0, :3] = col
        img[:, -1, :3] = col
        img[0, :, 3] = img[-1, :, 3] = img[:, 0, 3] = img[:, -1, 3] = 255
        tb = int(8 * k)
        img[1:tb, 1:-1, :3] = np.asarray(col) * 0.35
        img[1:tb, 1:-1, 3] = 220
        img[tb, 1:-1, :3] = np.asarray(col) * 0.7
        img[tb, 1:-1, 3] = 255
        if big:
            for i, dc in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
                img[3:5, 3 + i * 4:5 + i * 4, :3] = dc
                img[3:5, 3 + i * 4:5 + i * 4, 3] = 255
            name = 'claude.py' if side == 'cl' else 'codex.js'
            _paste(img, _text_rgba(name, colL, 'small', 8), 17, 1)
        # код
        syn = [np.asarray(col), np.array([80, 250, 123]), np.array([241, 250, 140]), np.array([230, 230, 240]),
               np.array([189, 147, 249]), np.array([140, 150, 190])]
        lines = self.code[side]
        typed = prog / 100.0 * self.code_len[side]
        # раскладываем набранное по строкам
        acc = 0
        cur = 0
        partial = 0.0
        for i, (ind, toks) in enumerate(lines):
            L = ind + sum(wd + 2 for wd, _ in toks)
            if acc + L > typed:
                cur = i
                partial = typed - acc
                break
            acc += L
        else:
            cur = len(lines) - 1
            partial = 1e9
        lh = 5 if big else 3
        y_top = tb + (3 if big else 2)
        y_bot = h - (18 if big else 9)
        nvis = max(1, (y_bot - y_top) // lh)
        first = max(0, cur - nvis + 1)
        for li in range(first, cur + 1):
            ind, toks = lines[li]
            y = y_top + (li - first) * lh
            x = 3 + int(ind * k)
            budget = partial if li == cur else 1e9
            used = ind
            for (wd, ci) in toks:
                if used >= budget:
                    break
                ww = int(min(wd, budget - used))
                ww = int(ww * k) if not big else ww
                if x + ww > w - 3:
                    ww = w - 3 - x
                if ww > 0:
                    hh_ = 2 if big else 1
                    img[y:y + hh_, x:x + ww, :3] = syn[ci]
                    img[y:y + hh_, x:x + ww, 3] = 255
                x += int((wd + 2) * k) if not big else wd + 2
                used += wd + 2
            if li == cur and prog < 100:
                # курсор
                if int(t * 6) % 2 == 0 and x < w - 4:
                    img[y - 1:y + (3 if big else 2), x:x + (2 if big else 1), :3] = (255, 255, 255)
                    img[y - 1:y + (3 if big else 2), x:x + (2 if big else 1), 3] = 255
        # прогресс
        pb_y = h - (8 if big else 5)
        pb_h = 4 if big else 2
        pw_ = w - 8
        img[pb_y - 1:pb_y + pb_h + 1, 3:3 + pw_ + 2, :3] = (4, 4, 10)
        img[pb_y - 1:pb_y + pb_h + 1, 3:3 + pw_ + 2, 3] = 255
        fillw = int(round(pw_ * prog / 100.0))
        done = prog >= 100
        fc = (255, 255, 255) if (done and int(t * 12) % 2 == 0) else col
        if fillw > 0:
            img[pb_y:pb_y + pb_h, 4:4 + fillw, :3] = fc
            img[pb_y:pb_y + pb_h, 4:4 + fillw, 3] = 255
            img[pb_y, 4:4 + fillw, :3] = np.minimum(np.asarray(fc, np.float32) + 70, 255)
        if big:
            lab = ('CLAUDE ' if side == 'cl' else 'CODEX ') + f'{int(prog)}%'
            ts = _text_rgba(lab, colL if not done else (255, 255, 255), 'big')
            _paste(img, ts, 4, pb_y - 10)
        # сканлайны
        img[1::2, :, :3] = (img[1::2, :, :3].astype(np.float32) * 0.8).astype(np.uint8)
        return img

    # ============================================================ сетка пола
    def floor(self, dst, cam, t, prog_cl, prog_cx):
        g = ground_rows(cam, 0.0, 3000.0)
        if g is None:
            return
        r0, X, Z, tt, dy = g
        R = len(tt)
        f = cam.f
        cy_, sy_ = math.cos(cam.yaw), math.sin(cam.yaw)
        dXr = np.empty_like(X)
        dZr = np.empty_like(Z)
        if R > 1:
            dXr[:-1] = np.abs(np.diff(X, axis=0))
            dXr[-1] = dXr[-2]
            dZr[:-1] = np.abs(np.diff(Z, axis=0))
            dZr[-1] = dZr[-2]
        else:
            dXr[:] = 0.1
            dZr[:] = 0.1
        fpX = np.abs(tt * cy_ / f)[:, None] + dXr
        fpZ = np.abs(tt * sy_ / f)[:, None] + dZr
        G, lw = 10.0, 0.7
        cxl = filt_lines(X, fpX, G, lw)
        czl = filt_lines(Z, fpZ, G, lw)
        I = np.maximum(cxl, czl)
        bl = np.maximum(filt_lines(X, fpX, G, lw * 4), filt_lines(Z, fpZ, G, lw * 4)) * 0.3
        I = np.maximum(I, bl)
        left = X < 0
        # волны «энергии» от станций (каждый удар по клавишам)
        wave = np.zeros_like(X)
        pool = np.zeros_like(X)
        for sxp, spd in ((SXC, 1.0), (SXX, 1.07)):
            d = np.sqrt((X - sxp) ** 2 + Z ** 2)
            ph = np.mod(d - t * 70.0 * spd, 36.0)
            ring = np.clip(1 - np.abs(ph - 18.0) / 5.0, 0, 1) * np.clip(1 - d / 150.0, 0, 1)
            wave = np.maximum(wave, ring)
            pool = np.maximum(pool, np.clip(1 - d / 42.0, 0, 1))
        I = np.clip(I * (1 + 1.6 * wave), 0, 1.2)
        # кольцо арены и центральная линия
        rr = np.sqrt(X ** 2 + Z ** 2)
        fpm = np.maximum(fpX, fpZ)
        ring = np.maximum(single_line(rr - 104.0, fpm, 1.2), single_line(rr - 110.0, fpm, 0.6) * 0.7)
        cen = single_line(X, fpX, 1.4)
        D = DITHER[r0:r0 + R]
        Iq = np.clip(np.floor(I * 4 + D) / 4, 0, 1)
        base = np.where(left[:, :, None], np.array([20, 10, 22], np.float32), np.array([8, 12, 30], np.float32))
        lc = np.where(left[:, :, None], ORANGE, BLUE)
        out = base + (lc - base) * Iq[:, :, None]
        core = (I > 0.85)
        out[core] = out[core] * 0.6 + 255 * 0.4 * np.where(left[core][:, None], np.array([1.0, 0.85, 0.7]), np.array([0.75, 0.9, 1.0]))
        # световые пятна под станциями
        pq = np.floor(pool * pool * 3 + D) / 3
        glow = np.where(left[:, :, None], ORANGE * 0.22, BLUE * 0.22)
        out += glow * pq[:, :, None]
        rq = np.clip(np.floor(ring * 3 + D) / 3, 0, 1)
        out = out * (1 - rq[:, :, None]) + MAGENTA * rq[:, :, None]
        cq = np.clip(np.floor(cen * 3 + D) / 3, 0, 1)
        out = out * (1 - cq[:, :, None]) + np.array([255, 235, 255], np.float32) * cq[:, :, None]
        apply_fog_rows(out, r0, tt, FOG, 90.0, 700.0, haze=0.45)
        dst[r0:r0 + R] = out

    # ============================================================ объекты
    def station(self, dst, cam, side, t):
        sx = SXC if side == 'cl' else SXX
        col = ORANGE if side == 'cl' else BLUE
        cols = dict(top=(62, 64, 92), front=(26, 26, 42), back=(18, 18, 30), left=(34, 34, 54), right=(34, 34, 54))
        draw_box(dst, cam, sx - ST_HW, sx + ST_HW, 0.0, ST_H, -ST_HD, ST_HD, cols, edge=None,
                 tex_front=self.plate[side], tex_side=self.side_tex[side])
        # неоновая кромка по верху (пульс)
        pulse = 0.75 + 0.25 * math.sin(t * 9 + (0 if side == 'cl' else 1.7))
        c = np.minimum(col * pulse + 40, 255)
        y = ST_H + 0.05
        pts = [(sx - ST_HW, y, -ST_HD), (sx + ST_HW, y, -ST_HD), (sx + ST_HW, y, ST_HD), (sx - ST_HW, y, ST_HD)]
        for i in range(4):
            line3d(dst, cam, pts[i], pts[(i + 1) % 4], c)
        # подсветка пола у основания
        for (a, b) in (((sx - ST_HW, 0.3, -ST_HD), (sx + ST_HW, 0.3, -ST_HD)),):
            line3d(dst, cam, a, b, col)
        # проектор голограммы сзади
        draw_box(dst, cam, sx - 3, sx + 3, ST_H, ST_H + 2.0, ST_HD - 5, ST_HD - 1,
                 dict(top=(90, 92, 120), front=(40, 40, 60), left=(30, 30, 48), right=(30, 30, 48), back=(20, 20, 30)))

    def holo_frame(self, side):
        sx = SXC if side == 'cl' else SXX
        th = 0.22 if side == 'cl' else -0.22
        U = np.array([math.cos(th), 0.0, math.sin(th)]) * HOLO_W
        V = np.array([0.0, -HOLO_H, 0.0])
        O = np.array([sx, HOLO_Y, HOLO_Z]) - U * 0.5
        return O, U, V

    def holo(self, dst, cam, side, t, prog, glitch=0.0, fade=1.0):
        if fade <= 0.02:
            return
        O, U, V = self.holo_frame(side)
        # размер на экране -> выбор LOD
        pc = cam.project(*(O + U * 0.5 + V * 0.5))
        big = True
        if pc is not None:
            wpx = cam.f / pc[2] * HOLO_W
            big = wpx > 70
        tex = self.holo_tex(side, t, prog, big)
        if glitch > 0:
            # горизонтальный сбой полосой
            r = int(hash01(int(t * 30), 7) * tex.shape[0] * 0.8)
            tex[r:r + 3] = np.roll(tex[r:r + 3], int(3 + glitch * 4), axis=1)
        flick = (0.92 + 0.08 * math.sin(t * 50)) * fade
        draw_quad(dst, cam, O, U, V, tex=tex, two_sided=True, dither_alpha=flick)

    def beam(self, dst, cam, side, fade=1.0):
        """проекционный луч (дизерный клин от проектора к низу голограммы)"""
        if fade <= 0.02:
            return
        O, U, V = self.holo_frame(side)
        sx = SXC if side == 'cl' else SXX
        col = ORANGE if side == 'cl' else BLUE
        src = cam.project(sx, ST_H + 2.0, ST_HD - 3)
        pa, pb = cam.project(*(O + V)), cam.project(*(O + V + U))
        if src and pa and pb:
            fill_poly(dst, [(src[0], src[1]), (pa[0], pa[1]), (pb[0], pb[1])], col * 0.8 + 50, dither=0.14 * fade)

    def clawd_typing(self, t, speed=1.25):
        # цикл набора — кадры 19..34
        d = self.cl.meta['Laptop']['durations']
        from chars import anim_frame
        return anim_frame(t * speed, d, loop=True, start=19, end=35)

    def codex_frame(self, t, lead, intense):
        # row 7: 0 '>_', 1 '^^', 2 '--', 3 '||', 4 '>_', 5 '^^'
        if lead >= 2.5:
            return 1 if int(t * 5) % 2 else 5
        if intense:
            return 3 if int(t * 7) % 3 else 4
        blink = (t % 2.3) < 0.12
        if blink:
            return 2
        return 0 if int(t * 4) % 2 else 4

    def sparks(self, dst, cam, side, t, rate=16.0, life=0.4, power=1.0):
        """искры от клавиш: частицы в мире (баллистика), без состояния."""
        if side == 'cl':
            kx, ky, kz = CL_AX + 16.0, ST_H + 3.0, -2.0
            col = ORANGE_L
        else:
            kx, ky, kz = CX_AX + 8.5, ST_H + 4.0, -2.0
            col = BLUE_L
        n = int(life * rate) + 1
        base = math.floor(t * rate)
        for j in range(n + 1):
            e = base - j
            te = e / rate
            age = t - te
            if age < 0 or age > life:
                continue
            for q in range(2):
                h1 = hash01(e, q, 1 if side == 'cl' else 2)
                h2 = hash01(e, q, 11)
                h3 = hash01(e, q, 23)
                vx = (h1 - 0.5) * 30 * power
                vy = (18 + h2 * 22) * power
                vz = (h3 - 0.7) * 16
                P = (kx + (h2 - 0.5) * 4 + vx * age, ky + vy * age - 60 * age * age, kz + vz * age)
                s = world_to_screen(cam, P)
                if s is None:
                    continue
                k = age / life
                if k < 0.35 and (e + q) % 3 == 0:
                    spr = SPARK if k < 0.18 else SPARK_S
                    blit(dst, spr, s[0] - spr.shape[1] // 2, s[1] - spr.shape[0] // 2)
                else:
                    c = (255, 255, 230) if k < 0.3 else col
                    pset(dst, s[0], s[1], c)
                    if cam.f / s[2] > 4:
                        pset(dst, s[0] + 1, s[1], c)
                        pset(dst, s[0], s[1] + 1, c)
                        pset(dst, s[0] + 1, s[1] + 1, c)

    # ============================================================ HUD
    def hud(self, dst, t, pcl, pcx, slide=1.0, flash_done=False):
        oy = int(round(-40 * (1 - ease_out(slide))))
        if slide <= 0:
            return
        bw, bh = 170, 7
        yb = 18 + oy
        # плашки-подложки
        rect(dst, 6, 3 + oy, 206, 26, INK, 0.72)
        rect(dst, W - 212, 3 + oy, 206, 26, INK, 0.72)
        blit(dst, self.hud_lbl['cl'], 8, 4 + oy)
        blit(dst, self.hud_lbl['cx'], W - 8 - self.hud_lbl['cx'].shape[1], 4 + oy)
        for side, p in (('cl', pcl), ('cx', pcx)):
            col = ORANGE if side == 'cl' else BLUE
            colL = ORANGE_L if side == 'cl' else BLUE_L
            x0 = 10 if side == 'cl' else W - 10 - bw
            rect(dst, x0 - 1, yb - 1, bw + 2, bh + 2, (250, 244, 230))
            rect(dst, x0, yb, bw, bh, (16, 12, 24))
            fw = int(round(bw * p / 100.0))
            fc = col
            if flash_done and int(t * 14) % 2 == 0:
                fc = np.array([255, 255, 255], np.float32)
            if fw > 0:
                if side == 'cl':
                    rect(dst, x0, yb, fw, bh, fc)
                    rect(dst, x0, yb, fw, 2, np.minimum(fc + 60, 255))
                    rect(dst, x0, yb + bh - 1, fw, 1, fc * 0.6)
                else:
                    rect(dst, x0 + bw - fw, yb, fw, bh, fc)
                    rect(dst, x0 + bw - fw, yb, fw, 2, np.minimum(fc + 60, 255))
                    rect(dst, x0 + bw - fw, yb + bh - 1, fw, 1, fc * 0.6)
            # деления
            for q in range(1, 10):
                xx = x0 + int(bw * q / 10)
                rect(dst, xx, yb + bh - 2, 1, 2, (16, 12, 24))
            # проценты
            ps = text_sprite(f'{int(p)}%', tuple(colL.astype(int)) if not flash_done else (255, 255, 255), outline=INK)
            if side == 'cl':
                blit(dst, ps, x0 + bw + 4 - ps.shape[1] + 30, 4 + oy)
            else:
                blit(dst, ps, x0 - 30, 4 + oy)
        # VS
        vs = text_sprite('VS', (255, 224, 90), scale=2, outline=INK, shadow=INK)
        blit(dst, vs, W // 2 - vs.shape[1] // 2, 5 + oy)

    # ============================================================ карточка
    def card(self, t):
        c = canvas((8, 6, 16))
        # стена кода (два слоя параллакса), тёплая слева, холодная справа
        off = int(t * 55) % self.card_wall.shape[0]
        wall = np.roll(self.card_wall, -off, axis=0)[:H]
        off2 = int(t * 120) % self.card_wall2.shape[0]
        wall2 = np.roll(self.card_wall2, -off2, axis=0)[:H]
        tint = np.where((XX + (YY - H / 2) * 0.35 < W / 2)[:, :, None], np.array([1.0, 0.62, 0.42]), np.array([0.5, 0.72, 1.0]))
        c += wall * 0.36 * tint
        m2 = wall2.sum(-1) > 0
        c[m2] = c[m2] * 0.4 + (wall2 * 0.3 * tint)[m2]
        round_card(c, t, 'РАУНД 1', 'КОД', (120, 255, 150), (10, 40, 20), self.port_l, self.port_r)
        # уход: вспышка в конце карточки
        if t > 1.18:
            flash(c, seg(t, 1.18, 1.3) * 0.9)
        return c

    # ============================================================ камера
    def camera(self, t):
        """(cam, shot) для арены"""
        if t < 3.3:
            # медленная орбита вокруг арены (фронтальная полусфера — спрайты не надо зеркалить)
            k = ease_in_out(seg(t, 1.3, 3.3))
            ang = lerp(-0.80, 0.42, k)
            rad = lerp(132, 116, k)
            hgt = lerp(36, 27, k)
            cam = orbit((0.0, 0.0), rad, ang, hgt, f=290, look_height=17)
            return cam, 'orbit'
        if t < 3.95:
            # быстрый наезд на Claude (средний план + низ голограммы)
            k = ease_out(seg(t, 3.3, 3.95) * 1.2)
            eye = lerp3((SXC + 78, 34, -118), (SXC + 30, 19, -54), k)
            tgt = lerp3((SXC + 4, 22, 0), (SXC + 1, 23, 0), k)
            return cam_shot(eye, tgt, lerp(250, 285, k)), 'push_cl'
        if t < 4.6:
            k = ease_out(seg(t, 3.95, 4.6) * 1.2)
            eye = lerp3((SXX - 78, 34, -118), (SXX - 30, 19, -54), k)
            tgt = lerp3((SXX - 4, 22, 0), (SXX - 1, 23, 0), k)
            return cam_shot(eye, tgt, lerp(250, 285, k)), 'push_cx'
        if t < 5.2:
            k = seg(t, 4.6, 5.2)
            eye = lerp3((CL_AX + 16, 15, -23), (CL_AX + 13, 14, -19), k)
            tgt = (CL_AX + 3.5, 15.5, 0)
            return cam_shot(eye, tgt, 300), 'ecu_cl'
        if t < 5.8:
            k = seg(t, 5.2, 5.8)
            eye = lerp3((CX_AX + 6, 20, -23), (CX_AX + 4.5, 19.5, -19), k)
            tgt = (CX_AX - 0.5, 20, 0)
            return cam_shot(eye, tgt, 300), 'ecu_cx'
        if t < T_FIN:
            # драматичный нижний ракурс: камера у самого пола облетает арену дугой снизу вверх на голограммы
            k = ease_in_out(seg(t, 5.8, T_FIN))
            ang = lerp(-0.95, 0.95, k)
            rad = lerp(80, 72, math.sin(k * math.pi))
            cam = orbit((0.0, 0.0), rad, ang, lerp(2.4, 3.2, k), f=200, look_height=lerp(26, 30, k))
            return cam, 'low'
        k = ease_out(seg(t, T_FIN, 9.0))
        eye = lerp3((0, 16, -80), (0, 15, -73), k)
        return cam_shot(eye, (0, 17, 0), 250), 'fin'

    # ============================================================ кадр арены
    def arena(self, t):
        cam, shot = self.camera(t)
        pcl, pcx = self.prog(t)
        fin = t >= T_FIN
        # удар штампа -> тряска
        amp = 0.0
        st_imp = T_FIN + 0.15
        if fin and t > st_imp:
            amp = 3.0 * max(0.0, 1 - (t - st_imp) / 0.35)
        if shot in ('push_cl', 'push_cx'):
            # короткий «удар» в начале наезда
            k0 = 3.3 if shot == 'push_cl' else 3.95
            amp = max(amp, 0.8 * max(0.0, 1 - (t - k0) / 0.15))
        cam = shake_cam(cam, t, amp, seed=3)
        c = canvas()
        sky_fast(c, cam, SKY, span=0.8, below=FOG)
        pano_draw(self.pano, c, cam, y_offset=2)
        self.floor(c, cam, t, pcl, pcx)
        # прожекторы сверху
        for sx, col in ((SXC, ORANGE), (SXX, BLUE)):
            light_cone(c, cam, (sx, 190.0, 30.0), (sx, 0.0, 0.0), 24.0, col * 0.55 + 90, dither=0.13)
        items = []
        lead = pcl - pcx
        intense = shot in ('low',)
        # --- станция + луч проектора + персонаж: один элемент (персонаж всегда поверх своей станции)
        glitch = 0.0
        hfade = 1.0 - seg(t, T_FIN + 0.08, T_FIN + 0.24)
        mad = fin and t >= T_FIN + 0.2
        if not mad:
            fr = self.clawd_typing(t, 1.7 if intense else 1.3)
            cf = self.codex_frame(t, -lead, intense)
            sq = 1.0 - 0.025 * (int(t * 9) % 2)

        def st_cl(d, cm):
            self.station(d, cm, 'cl', t)
            self.beam(d, cm, 'cl', hfade)
            if mad:
                self.clawd_mad(d, cm, t)
            else:
                draw_clawd_b(d, cm, self.cl, CL_AX, ST_H, 0.0, anim='Laptop', frame=fr)

        def st_cx(d, cm):
            self.station(d, cm, 'cx', t)
            self.beam(d, cm, 'cx', hfade)
            if mad:
                self.codex_mad(d, cm, t)
            else:
                draw_codex_b(d, cm, self.pix, self.cx, CX_AX, ST_H, 0.0, row=7, col=cf, sy=sq)
        items.append((SXC, ST_H * 0.5, 0.0, st_cl, 0.0))
        items.append((SXX, ST_H * 0.5, 0.0, st_cx, 0.0))
        # голограммы
        if T_FIN + 0.04 < t < T_FIN + 0.24:
            glitch = 1.0
        if shot == 'low' and t > 7.3:
            glitch = 1.0 if hash01(int(t * 30), 3) > 0.7 else 0.0
        for side in ('cl', 'cx'):
            O, U, V = self.holo_frame(side)
            ctr = O + U * 0.5 + V * 0.5
            p = pcl if side == 'cl' else pcx

            def hf(d, cm, side=side, p=p):
                self.holo(d, cm, side, t, p, glitch, fade=hfade)
            items.append((ctr[0], ctr[1], ctr[2], hf, 0.0))
        sort_draw(c, cam, items)
        # искры
        if not fin:
            pw_ = 1.3 if intense else 1.0
            self.sparks(c, cam, 'cl', t, rate=18 if intense else 14, power=pw_)
            self.sparks(c, cam, 'cx', t + 0.37, rate=18 if intense else 14, power=pw_)
        # пот на крупных/нижнем
        if shot in ('ecu_cl', 'low') and not fin:
            self.sweat(c, cam, CL_AX + 11, ST_H + 14, -1.0, t)
        if shot in ('ecu_cx', 'low') and not fin:
            self.sweat(c, cam, CX_AX + 7, ST_H + 20, -1.0, t + 0.4)
        c *= vig_map(0.45)
        # линии фокуса (манга) на крупных планах
        if shot in ('ecu_cl', 'ecu_cx'):
            focus_lines(c, t, seed=4 if shot == 'ecu_cl' else 8,
                        col=(255, 214, 150) if shot == 'ecu_cl' else (170, 220, 255))
        # финал (экранные штамп/молнии — поверх виньетки)
        if fin:
            self.finale(c, cam, t)
        # HUD
        slide = seg(t, 1.38, 1.62)
        done = t >= T_FIN
        self.hud(c, t, pcl, pcx, slide, flash_done=done and t < T_FIN + 0.5)
        # вход в арену — вспышка «удара гонга»
        if t < 1.45:
            flash(c, 0.9 * (1 - seg(t, 1.3, 1.45)))
        # 100% — вспышка
        if T_FIN <= t < T_FIN + 0.05:
            flash(c, 0.4)
        return c

    def sweat(self, dst, cam, X, Y, Z, t):
        a = (t * 1.6) % 1.0
        s = world_to_screen(cam, (X + a * 2, Y - a * 6, Z - 1))
        if s is None:
            return
        k = min(3, max(1, int(round(cam.f / s[2] * 0.4))))
        spr = scale_nn(SWEAT, k)
        if a < 0.8:
            blit(dst, spr, s[0], s[1])

    # ============================================================ финал: штамп + ярость
    def clawd_mad(self, d, cm, t):
        a = t - (T_FIN + 0.2)
        # прыжок-разворот
        hopY = 0.0
        sx_, sy_ = 1.0, 1.0
        if a < 0.18:
            k = a / 0.18
            hopY = math.sin(k * math.pi) * 7
            sx_, sy_ = (0.85, 1.2) if k < 0.5 else (1.0, 1.0)
        elif a < 0.26:
            sx_, sy_ = 1.2, 0.82
        lean_px = 2 if a > 0.26 else 0
        tr = (int(t * 20) % 2) if a > 0.26 else 0
        draw_clawd_b(d, cm, self.cl, CL_AX + 4 + tr * 0.4, ST_H + hopY, 0.0, face='furious', sx=sx_, sy=sy_,
                     lean_px=lean_px)

    def codex_mad(self, d, cm, t):
        a = t - (T_FIN + 0.2)
        hopY = 0.0
        sx_, sy_ = 1.0, 1.0
        if a < 0.18:
            k = a / 0.18
            hopY = math.sin(k * math.pi) * 7
            sx_, sy_ = (0.88, 1.15) if k < 0.5 else (1.0, 1.0)
            col = 3 if k < 0.5 else 1
            row = 10 if k < 0.5 else 5
        elif a < 0.26:
            sx_, sy_ = 1.15, 0.86
            row, col = 5, 1
        else:
            row, col = 5, 1
        lean_px = -3 if a > 0.26 else 0
        tr = (int(t * 20 + 1) % 2) if a > 0.26 else 0
        if row == 10:
            draw_codex_b(d, cm, self.pix, self.cx, CX_AX + 1 - tr * 0.4, ST_H + hopY, 0.0, row=10, col=3, sx=sx_,
                         sy=sy_)
        else:
            draw_codex_b(d, cm, self.pix, self.cx, CX_AX + 1 - tr * 0.4, ST_H + hopY, 0.0, row=row, col=col,
                         sx=sx_, sy=sy_, lean_px=lean_px)

    def finale(self, c, cam, t):
        a = t - T_FIN
        # взгляды-молнии между глазами (после разворота)
        am = t - (T_FIN + 0.2)
        if am > 0.24:
            e1 = world_to_screen(cam, (CL_AX + 4 + 5.5, ST_H + 14, -1))
            e2 = world_to_screen(cam, (CX_AX + 1 - 3.5, ST_H + 15, -1))
            if e1 and e2:
                self.glare(c, e1[:2], e2[:2], t)
            # пар из ушей/макушки
            for (X, Y, sd) in ((CL_AX + 4, ST_H + 17, 1), (CX_AX + 1, ST_H + 23, 2)):
                s = world_to_screen(cam, (X, Y, -1))
                if s:
                    steam_b(c, t, s[0], s[1], seed=sd, k=cam.f / s[2] * 0.35)
            # венки гнева
            for (X, Y) in ((CL_AX + 13, ST_H + 19), (CX_AX + 8, ST_H + 25)):
                s = world_to_screen(cam, (X, Y, -1))
                if s:
                    k = max(1, int(round(cam.f / s[2] * 0.45)))
                    spr = ANGER2 if int(t * 8) % 2 else ANGER
                    blit(c, scale_nn(spr, k), s[0] - 3 * k, s[1] - 3 * k)
        if 0.3 < am < 0.46:
            # «вжух» — дуги разворота
            for (X, Y, sg) in ((CL_AX, ST_H + 9, 1), (CX_AX, ST_H + 12, -1)):
                s = world_to_screen(cam, (X, Y, -1))
                if s:
                    for q in range(3):
                        r = 14 + q * 5
                        for i in range(8):
                            ang = math.pi * (0.2 + i * 0.08) + (0 if sg > 0 else 0.0)
                            x = s[0] - sg * math.cos(ang) * r
                            y = s[1] - math.sin(ang) * r * 0.8
                            pset(c, x, y, (255, 255, 255))
        # штамп
        sa = t - (T_FIN + 0.03)
        if sa >= 0:
            spr = self.stamp
            dur = 0.12
            if sa < dur:
                k = 1.0 + 0.9 * (1 - ease_in(sa / dur))
            else:
                k = 1.0 + 0.06 * math.sin((sa - dur) * 34) * math.exp(-(sa - dur) * 10)
            img = squash(spr, k, k) if abs(k - 1) > 0.01 else spr
            cx_, cy_ = W // 2, 74
            if sa >= dur:
                j = max(0.0, 1 - (sa - dur) / 0.3)
                cx_ += int(round(math.sin(t * 97) * 4 * j))
                cy_ += int(round(math.sin(t * 131) * 3 * j))
            # пыль/осколки от удара
            if dur <= sa < dur + 0.45:
                q = sa - dur
                for i in range(14):
                    ang = hash01(i, 5) * math.pi * 2
                    dist = 60 + q * 260 * (0.6 + 0.4 * hash01(i, 9))
                    px_ = W // 2 + math.cos(ang) * dist * 1.3
                    py_ = 74 + math.sin(ang) * dist * 0.45
                    r = 5 * (1 - q / 0.45) + 1
                    puff(c, px_, py_, r, (240, 236, 244), (170, 160, 190), alpha=1 - q / 0.45)
            blit(c, img, cx_ - img.shape[1] / 2, cy_ - img.shape[0] / 2)
            if dur <= sa < dur + 0.06:
                flash(c, 0.5)

    def glare(self, c, p1, p2, t):
        n = 9
        seed = int(t * 15)
        pts = []
        for i in range(n + 1):
            k = i / n
            x = p1[0] + (p2[0] - p1[0]) * k
            y = p1[1] + (p2[1] - p1[1]) * k
            if 0 < i < n:
                y += (hash01(seed, i) - 0.5) * 14
                x += (hash01(seed, i, 3) - 0.5) * 6
            pts.append((x, y))
        mid = n // 2
        for i in range(n):
            a, b = pts[i], pts[i + 1]
            col = ORANGE if i < mid else BLUE
            for dy_ in (-2, 2):
                line_fast(c, a[0], a[1] + dy_, b[0], b[1] + dy_, col, dither=0.5)
            line_fast(c, a[0], a[1] - 1, b[0], b[1] - 1, col)
            line_fast(c, a[0], a[1] + 1, b[0], b[1] + 1, col)
            line_fast(c, a[0], a[1], b[0], b[1], (255, 255, 255))
        mx, my = pts[mid]
        spr = scale_nn(SPARK, 2)
        blit(c, spr, mx - spr.shape[1] // 2, my - spr.shape[0] // 2)

    # ============================================================
    def render(self, t):
        if t < T_CARD:
            return self.card(t)
        return self.arena(t)


SCENE = CodeScene()
