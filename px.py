"""Мини-движок пиксельной графики: холст 480x270 (float32 RGB), блит спрайтов, дизеринг, текст, частицы."""
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 480, 270
FPS = 30
ROOT = os.path.dirname(os.path.abspath(__file__))

BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16.0
BAYER8 = None


def _bayer8():
    b = np.zeros((8, 8))
    b4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
    for y in range(8):
        for x in range(8):
            b[y, x] = 4 * b4[y % 4, x % 4] + [[0, 2], [3, 1]][y // 4][x // 4]
    return (b + 0.5) / 64.0


BAYER8 = _bayer8()
YY, XX = np.mgrid[0:H, 0:W]
DITHER = BAYER4[YY % 4, XX % 4]
DITHER8 = BAYER8[YY % 8, XX % 8]


def hexc(h):
    h = h.lstrip('#')
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], np.float32)


def canvas(color=(0, 0, 0)):
    c = np.empty((H, W, 3), np.float32)
    c[:] = color
    return c


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def ease_in_out(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in(t):
    t = clamp(t)
    return t ** 3


def ease_out_back(t, s=1.70158):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def seg(t, a, b):
    """0..1 прогресс t внутри отрезка [a,b]."""
    if b <= a:
        return 1.0 if t >= b else 0.0
    return clamp((t - a) / (b - a))


def hop(t, period, height):
    """прыжки-подскоки: |sin|"""
    return -abs(math.sin(math.pi * t / period)) * height


# ---------------------------------------------------------------- sprites
def load_rgba(path):
    return np.array(Image.open(path).convert('RGBA'))


def scale_nn(spr, k):
    if k == 1:
        return spr
    return np.repeat(np.repeat(spr, k, 0), k, 1)


def flip(spr):
    return spr[:, ::-1]


def blit(dst, spr, x, y, alpha=1.0, flipx=False, tint=None, add=None):
    """spr HxWx4 uint8; (x,y) — левый верхний угол. tint: множитель RGB (0..1), add: прибавка RGB."""
    if spr is None:
        return
    x, y = int(round(x)), int(round(y))
    if flipx:
        spr = spr[:, ::-1]
    h, w = spr.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1 or alpha <= 0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[:, :, 3:4].astype(np.float32) * (alpha / 255.0)
    rgb = s[:, :, :3].astype(np.float32)
    if tint is not None:
        rgb = rgb * np.asarray(tint, np.float32)
    if add is not None:
        rgb = rgb + np.asarray(add, np.float32)
    d = dst[y0:y1, x0:x1]
    d *= (1 - a)
    d += rgb * a


def blit_mask(dst, mask, x, y, color, alpha=1.0):
    """mask HxW bool/float — рисует одним цветом."""
    x, y = int(round(x)), int(round(y))
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)[:, :, None] * alpha
    d = dst[y0:y1, x0:x1]
    d *= (1 - m)
    d += np.asarray(color, np.float32) * m


def silhouette(spr, color):
    o = spr.copy()
    o[:, :, :3] = np.asarray(color, np.uint8)
    return o


def outline_mask(mask, thick=1):
    m = mask.copy()
    p = np.pad(mask, thick)
    for dy in range(-thick, thick + 1):
        for dx in range(-thick, thick + 1):
            if abs(dx) + abs(dy) > thick:
                continue
            m |= p[thick + dy:thick + dy + mask.shape[0], thick + dx:thick + dx + mask.shape[1]]
    return m


def lean(spr, amount, pivot_frac=0.85):
    """«наклон» пиксельного спрайта: сдвиг строк выше пивота (shear), amount в px у верхней строки."""
    h, w = spr.shape[:2]
    pad = int(math.ceil(abs(amount))) + 1
    out = np.zeros((h, w + 2 * pad, 4), np.uint8)
    piv = h * pivot_frac
    for y in range(h):
        k = max(0.0, (piv - y) / piv)
        dx = int(round(amount * k))
        out[y, pad + dx:pad + dx + w] = spr[y]
    return out, pad


def squash(spr, sx, sy):
    """масштаб ближайшим соседом с сохранением низа-центра (для squash&stretch)."""
    h, w = spr.shape[:2]
    nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    ys = np.clip((np.arange(nh) + 0.5) * h / nh, 0, h - 1).astype(int)
    xs = np.clip((np.arange(nw) + 0.5) * w / nw, 0, w - 1).astype(int)
    return spr[ys][:, xs]


# ---------------------------------------------------------------- drawing prims
def rect(dst, x, y, w, h, color, alpha=1.0):
    x0, y0 = int(max(0, round(x))), int(max(0, round(y)))
    x1, y1 = int(min(W, round(x + w))), int(min(H, round(y + h)))
    if x0 >= x1 or y0 >= y1:
        return
    if alpha >= 1:
        dst[y0:y1, x0:x1] = color
    else:
        dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - alpha) + np.asarray(color, np.float32) * alpha


def pset(dst, x, y, color, alpha=1.0):
    x, y = int(round(x)), int(round(y))
    if 0 <= x < W and 0 <= y < H:
        if alpha >= 1:
            dst[y, x] = color
        else:
            dst[y, x] = dst[y, x] * (1 - alpha) + np.asarray(color, np.float32) * alpha


def line(dst, x0, y0, x1, y1, color, alpha=1.0):
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        pset(dst, x0, y0, color, alpha)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def disc_mask(r):
    rr = int(math.ceil(r))
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    return (xx + 0.0) ** 2 + (yy + 0.0) ** 2 <= r * r + r * 0.8


def disc(dst, cx, cy, r, color, alpha=1.0):
    m = disc_mask(r)
    rr = (m.shape[0] - 1) // 2
    blit_mask(dst, m, cx - rr, cy - rr, color, alpha)


def ellipse_mask(rx, ry):
    ax, ay = int(math.ceil(rx)), int(math.ceil(ry))
    yy, xx = np.mgrid[-ay:ay + 1, -ax:ax + 1]
    return (xx / max(rx, 0.5)) ** 2 + (yy / max(ry, 0.5)) ** 2 <= 1.0 + 0.6 / max(1.0, min(rx, ry))


def vgradient(dst, y0, y1, stops, dither=DITHER, x0=0, x1=W):
    """вертикальный градиент между цветами stops (список RGB) с упорядоченным дизерингом (полосы)."""
    y0, y1 = int(y0), int(y1)
    n = len(stops) - 1
    ys = np.arange(y0, y1)
    v = (ys - y0) / max(1, (y1 - y0 - 1)) * n
    for i, y in enumerate(ys):
        if y < 0 or y >= H:
            continue
        fv = v[i]
        k = int(min(n - 1, math.floor(fv)))
        fr = fv - k
        # полосы: квантуем fr на 5 уровней, чтобы дизер был «ступеньками»
        fr = round(fr * 4) / 4
        row = dither[y, x0:x1] < fr
        c0 = np.asarray(stops[k], np.float32)
        c1 = np.asarray(stops[k + 1], np.float32)
        dst[y, x0:x1] = np.where(row[:, None], c1, c0)


def dither_fill_mask(val, dither=DITHER):
    """val HxW 0..1 -> бинарная маска с упорядоченным дизерингом"""
    return val > dither


def dissolve(a, b, p, block=2):
    """пиксельное растворение a->b по матрице Байера (крупные блоки)."""
    d = BAYER8[(YY // block) % 8, (XX // block) % 8]
    m = (d < p)[:, :, None]
    return np.where(m, b, a)


def iris(dst, cx, cy, r, color=(0, 0, 0)):
    if r <= 0:
        dst[:] = color
        return
    # чуть «пиксельный» круг: считаем по блокам 2x2
    yy, xx = YY // 2 * 2, XX // 2 * 2
    m = (xx - cx) ** 2 + (yy - cy) ** 2 > r * r
    dst[m] = color


def fade(dst, color, p, dithered=True):
    if p <= 0:
        return
    if dithered:
        q = round(clamp(p) * 8) / 8
        m = DITHER < q
        dst[m] = color
    else:
        dst[:] = dst * (1 - p) + np.asarray(color, np.float32) * p


# ---------------------------------------------------------------- text
FONT_PATH = ROOT + '/fonts/PressStart2P.ttf'
FONT_SMALL_PATH = ROOT + '/fonts/Tiny5.ttf'
_font_cache = {}
_text_cache = {}


# LANG: 'ru' (по умолчанию) либо 'en' — переменная окружения CARTOON_LANG.
LANG = os.environ.get('CARTOON_LANG', 'ru')
TR = {
    'КТО ЖЕ ЛУЧШЕ?': 'WHO IS BETTER?',
    'Я ЛУЧШЕ!': "I'M BETTER!",
    'НЕТ, Я ЛУЧШЕ!': "NO, I'M BETTER!",
    'Я!': 'ME!', 'Я!!': 'ME!!', 'Я!!!': 'ME!!!', 'Я!!!!': 'ME!!!!',
    'РАУНД 1': 'ROUND 1', 'РАУНД 2': 'ROUND 2', 'КОД': 'CODE', 'ГОНКА': 'RACE',
    'НИЧЬЯ!': 'DRAW!', 'ДЗЫНЬ!': 'DING!',
    'СТАРТ': 'START', 'СТАРТ!': 'GO!', 'ФИНИШ': 'FINISH',
    'БАМ!': 'BAM!', 'ХРЯСЬ!': 'CRACK!',
    'МИР?': 'TRUCE?', 'МИР!': 'TRUCE!', 'апчхи!': 'achoo!',
    'КОНЕЦ': 'THE END',
    'ВСЁ СДЕЛАНО КОДОМ': 'MADE ENTIRELY IN CODE',
    'АНИМАЦИЯ · МУЗЫКА · ЗВУК': 'ANIMATION · MUSIC · SOUND',
    'ПРОМПТ: KIRILL SH': 'PROMPT BY KIRILL SH',
    '// TODO: победить': '// TODO: win',
    'print("Я лучше!")': 'print("I am better!")',
}


def tr(s):
    """Перевод экранного текста для английской версии."""
    return TR.get(s, s) if LANG == 'en' else s


def text_mask(s, font='big', size=None):
    s = tr(s)
    key = (s, font, size)
    if key in _text_cache:
        return _text_cache[key]
    path = FONT_PATH if font == 'big' else FONT_SMALL_PATH
    size = size or (8 if font == 'big' else 10)
    fk = (path, size)
    if fk not in _font_cache:
        _font_cache[fk] = ImageFont.truetype(path, size)
    f = _font_cache[fk]
    bbox = f.getbbox(s)
    w, h = bbox[2] - min(0, bbox[0]) + 2, bbox[3] + 2
    im = Image.new('L', (max(1, w), max(1, h)), 0)
    d = ImageDraw.Draw(im)
    d.fontmode = '1'
    d.text((0, 0), s, font=f, fill=255)
    m = np.array(im) > 127
    ys, xs = np.where(m)
    if len(ys):
        m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    _text_cache[key] = m
    return m


def text_sprite(s, color, font='big', size=None, scale=1, outline=None, shadow=None, spacing=0):
    m = text_mask(s, font, size)
    if scale > 1:
        m = np.repeat(np.repeat(m, scale, 0), scale, 1)
    pad = 2 if (outline is not None or shadow is not None) else 0
    h, w = m.shape
    out = np.zeros((h + 2 * pad + (scale if shadow is not None else 0), w + 2 * pad + (scale if shadow is not None else 0), 4), np.uint8)
    if shadow is not None:
        k = scale
        om = outline_mask(m, 1) if outline is not None else m
        sub = out[pad + k:pad + k + h, pad + k:pad + k + w]
        sub[om] = list(shadow) + [255]
    if outline is not None:
        om = outline_mask(m, 1)
        big = np.zeros((h + 2 * pad, w + 2 * pad), bool)
        big[pad:pad + h, pad:pad + w] = om
        out[:h + 2 * pad, :w + 2 * pad][big] = list(outline) + [255]
    sub = out[pad:pad + h, pad:pad + w]
    sub[m] = list(np.asarray(color, np.uint8)) + [255]
    return out


# ---------------------------------------------------------------- tiny pixel icons (ASCII art)
def from_ascii(rows, cmap):
    h, w = len(rows), max(len(r) for r in rows)
    o = np.zeros((h, w, 4), np.uint8)
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            if ch in cmap:
                o[y, x, :3] = cmap[ch]
                o[y, x, 3] = 255
    return o


HEART = from_ascii([
    '.oo.oo.',
    'oRRoRPo',
    'oRRRRRo',
    'oRRRRRo',
    '.oRRRo.',
    '..oRo..',
    '...o...',
], {'o': (120, 20, 50), 'R': (240, 70, 110), 'P': (255, 200, 215)})

HEART_S = from_ascii([
    'RR.RR',
    'RRRPR',
    'RRRRR',
    '.RRR.',
    '..R..',
], {'R': (240, 70, 110), 'P': (255, 200, 215)})

ANGER = from_ascii([
    '.R...R.',
    'RR...RR',
    '.......',
    '.......',
    '.......',
    'RR...RR',
    '.R...R.',
], {'R': (230, 40, 50)})

ANGER2 = from_ascii([
    '..R.R..',
    '.RR.RR.',
    'RR...RR',
    '.......',
    'RR...RR',
    '.RR.RR.',
    '..R.R..',
], {'R': (230, 40, 50)})

SPARK = from_ascii([
    '..Y..',
    '..Y..',
    'YYWYY',
    '..Y..',
    '..Y..',
], {'Y': (255, 220, 90), 'W': (255, 255, 240)})

SPARK_S = from_ascii([
    '.Y.',
    'YWY',
    '.Y.',
], {'Y': (255, 220, 90), 'W': (255, 255, 240)})

STAR5 = from_ascii([
    '...Y...',
    '...Y...',
    '..YYY..',
    'YYYWYYY',
    '.YYYYY.',
    '.YY.YY.',
    'Y.....Y',
], {'Y': (255, 214, 64), 'W': (255, 250, 210)})

SWEAT = from_ascii([
    '.B.',
    'BBB',
    'BWB',
    '.B.',
], {'B': (120, 190, 255), 'W': (230, 245, 255)})

EXCL = from_ascii([
    'oo',
    'YY',
    'YY',
    'YY',
    'YY',
    '..',
    'YY',
], {'Y': (255, 220, 60), 'o': (255, 245, 180)})

NOTE = from_ascii([
    '..KKK',
    '..K.K',
    '..K..',
    'KKK..',
    'KKK..',
], {'K': (40, 30, 50)})


def excl_sprite(scale=2):
    m = from_ascii([
        '.oo.',
        'oYYo',
        'oYYo',
        'oYYo',
        'oYYo',
        '.oo.',
        '.oo.',
        'oYYo',
        '.oo.',
    ], {'Y': (255, 224, 70), 'o': (70, 40, 20)})
    return scale_nn(m, scale)


def quest_sprite(scale=2):
    m = from_ascii([
        '.oooo.',
        'oYYYYo',
        'oYooYo',
        '.o.oYo',
        '..oYo.',
        '..oYo.',
        '..oo..',
        '..oo..',
        '.oYYo.',
        '..oo..',
    ], {'Y': (255, 224, 70), 'o': (70, 40, 20)})
    return scale_nn(m, scale)


# ---------------------------------------------------------------- particles
class Particles:
    def __init__(self, seed=0):
        self.p = []
        self.rng = np.random.default_rng(seed)

    def emit(self, **kw):
        self.p.append(dict(kw))


def rng_for(*keys):
    h = 0
    for k in keys:
        h = (h * 1000003 + hash(k)) & 0xFFFFFFFF
    return np.random.default_rng(h)


# ---------------------------------------------------------------- speech bubble
def bubble_sprite(text, font='big', size=None, fg=(30, 24, 40), bg=(255, 252, 240), border=(30, 24, 40),
                  tail='left', pad=(5, 4), text_scale=1, shake=0):
    tm = text_mask(text, font, size)
    if text_scale > 1:
        tm = np.repeat(np.repeat(tm, text_scale, 0), text_scale, 1)
    th, tw = tm.shape
    bw, bh = tw + pad[0] * 2 + 2, th + pad[1] * 2 + 2
    tail_h = 6
    o = np.zeros((bh + tail_h, bw, 4), np.uint8)
    # тело с «скруглёнными» пиксельными углами
    body = np.ones((bh, bw), bool)
    for (yy, xx) in [(0, 0), (0, 1), (1, 0), (0, bw - 1), (0, bw - 2), (1, bw - 1), (bh - 1, 0), (bh - 1, 1), (bh - 2, 0),
                     (bh - 1, bw - 1), (bh - 1, bw - 2), (bh - 2, bw - 1)]:
        body[yy, xx] = False
    full = np.zeros((bh + tail_h, bw), bool)
    full[:bh] = body
    # хвостик
    if tail in ('left', 'right'):
        tx = 7 if tail == 'left' else bw - 8
        for i in range(tail_h):
            wdt = max(1, 5 - i)
            if tail == 'left':
                full[bh - 1 + i, tx - i // 2: tx - i // 2 + wdt] = True
            else:
                full[bh - 1 + i, tx + i // 2 - wdt + 1: tx + i // 2 + 1] = True
    ol = outline_mask(full, 1)
    big = np.zeros((bh + tail_h + 2, bw + 2), bool)
    big[1:-1, 1:-1] = full
    olb = outline_mask(big, 1)
    o2 = np.zeros((bh + tail_h + 2, bw + 2, 4), np.uint8)
    o2[olb] = list(border) + [255]
    o2[big] = list(bg) + [255]
    ys, xs = np.where(tm)
    o2[1 + 1 + pad[1] + ys, 1 + 1 + pad[0] + xs] = list(fg) + [255]
    return o2


def pop_scale(t, dur=0.25):
    """масштаб «поп»-появления"""
    if t <= 0:
        return 0.0
    if t >= dur:
        return 1.0
    return max(0.05, ease_out_back(t / dur, 2.2))


def draw_bubble(dst, text, tip_x, tip_y, age, tail='left', dur_pop=0.22, font='big', size=None, text_scale=1,
                shake=0.0, fg=(30, 24, 40), bg=(255, 252, 240), border=(30, 24, 40), seed=0):
    """Реплика в «пузыре» с поп-анимацией. (tip_x, tip_y) — кончик хвостика (у рта персонажа).
    age — секунды с момента появления (<0 — не рисуется)."""
    if age < 0:
        return
    spr = bubble_sprite(text, font=font, size=size, fg=fg, bg=bg, border=border, tail=tail, text_scale=text_scale)
    s = pop_scale(age, dur_pop)
    h0, w0 = spr.shape[:2]
    if s < 0.999:
        spr = squash(spr, s, s)
    h, w = spr.shape[:2]
    tx = (7 if tail == 'left' else w0 - 8) * (w / w0)
    x = tip_x - tx
    y = tip_y - h
    if shake:
        r = np.random.default_rng(int(age * 30) + seed * 7919)
        x += int(r.integers(-1, 2) * shake)
        y += int(r.integers(-1, 2) * shake)
    x = min(max(x, 1), W - w - 1)
    blit(dst, spr, x, y)


def big_text(dst, s, cx, cy, color=(255, 224, 90), scale=3, outline=(40, 20, 10), shadow=(40, 20, 10), font='big',
             size=None, pop_age=None, alpha=1.0):
    """крупный пиксельный текст по центру (cx, cy); pop_age — секунды с появления для поп-анимации."""
    spr = text_sprite(s, color, font=font, size=size, scale=scale, outline=outline, shadow=shadow)
    if pop_age is not None:
        if pop_age < 0:
            return
        k = pop_scale(pop_age, 0.25)
        if k < 0.999:
            spr = squash(spr, k, k)
    h, w = spr.shape[:2]
    blit(dst, spr, cx - w / 2, cy - h / 2, alpha=alpha)
