"""Общая библиотека мира: палитры, слои луга (с параллаксом), деревья, облака, реквизит, погода, эффекты.

Все функции без состояния: картинка зависит только от аргументов (в т.ч. времени t), поэтому кадры можно
рендерить параллельно и в любом порядке.
"""
import math
import numpy as np
from px import *

# ======================================================================= палитры
PALETTES = {
    'day': dict(
        sky=['#2f6fd0', '#4a8be0', '#6faaf0', '#9cccf8', '#c9e8ff'],
        cloud=['#ffffff', '#eef5fd', '#cfe0f3', '#a9c3e6'],
        mount=['#7f9fd0', '#94b2dc', '#b0c8e8'], snow='#eef5ff',
        hill_far=['#79b58b', '#8cc79a'], hill_mid=['#4f9d55', '#6dbb5e', '#3e8446'],
        hill_near=['#3f8f45', '#5aae4f', '#2f7438'],
        tree=['#1d4f2a', '#2e7d3a', '#45a046', '#74c85e'], trunk=['#5a3d26', '#7a5536'],
        grass=['#58b24f', '#86d765', '#3f9443', '#2f7a3a', '#a8ec7e'],
        dirt=['#8a6440', '#6e4d31', '#a57c52'],
        flowers=['#ff8fb1', '#ffe066', '#ffffff', '#b98cff', '#ff6b6b'],
    ),
    'storm': dict(
        sky=['#1f2336', '#2a3048', '#383f5c', '#475073', '#56608a'],
        cloud=['#5b6488', '#4b5375', '#3c4362', '#2e344e'],
        mount=['#39405c', '#424a68', '#4c5574'], snow='#6b7596',
        hill_far=['#2f4a4a', '#365653'], hill_mid=['#2a4a39', '#335a44', '#223d30'],
        hill_near=['#244233', '#2e5340', '#1b3427'],
        tree=['#0f261a', '#173524', '#20472f', '#2d5c3d'], trunk=['#2c2019', '#3b2b20'],
        grass=['#2f5a40', '#3d7050', '#244a34', '#1b3a28', '#4a8060'],
        dirt=['#4a3a2e', '#3a2d24', '#5a4838'],
        flowers=['#9a6a86', '#a09a70', '#b0b8c8', '#7a6aa0', '#a06070'],
    ),
    'golden': dict(  # после дождя / вечернее золото
        sky=['#4a78c8', '#7aa0dc', '#b8c4e0', '#f2d6b0', '#ffe4b0'],
        cloud=['#fff4e0', '#ffe2c0', '#f0bfa0', '#c998a0'],
        mount=['#8c96c0', '#a4a8c8', '#c2bccc'], snow='#fff0e0',
        hill_far=['#8fb58a', '#a4c496'], hill_mid=['#5fa352', '#80c05e', '#4a8844'],
        hill_near=['#4a9244', '#6cb452', '#377a3a'],
        tree=['#23502a', '#347e38', '#52a444', '#8ccc5c'], trunk=['#5a3d26', '#7a5536'],
        grass=['#62b44a', '#98da62', '#48964a', '#357c3c', '#b8ee80'],
        dirt=['#8a6440', '#6e4d31', '#a57c52'],
        flowers=['#ff8fb1', '#ffe066', '#ffffff', '#b98cff', '#ff6b6b'],
    ),
    'sunset': dict(
        sky=['#241640', '#3d1f5c', '#6e2a73', '#b23f78', '#e8645f', '#f99a54', '#ffd27a'],
        cloud=['#ffc58a', '#f59a7a', '#c86a86', '#7a3f78'],
        mount=['#5a2f6e', '#6e3a78', '#86487e'], snow='#f0a0a0',
        hill_far=['#4a2860', '#5a3070'], hill_mid=['#35204e', '#46295e', '#2a1840'],
        hill_near=['#2a1a40', '#3a2352', '#1d1230'],
        tree=['#140c22', '#1e1230', '#2a1a40', '#3a2452'], trunk=['#140c22', '#1e1230'],
        grass=['#2f1f48', '#4a2e5e', '#241838', '#1a1028', '#6a3e6e'],
        dirt=['#2a1a38', '#20142c', '#3a2448'],
        flowers=['#ff8fb1', '#ffb070', '#ffd0a0', '#c070c0', '#ff7070'],
    ),
}


def P(mode):
    p = PALETTES[mode]
    out = {}
    for k, v in p.items():
        out[k] = [hexc(c) for c in v] if isinstance(v, list) else hexc(v)
    return out


# ======================================================================= шум
def smooth_noise(n, scale, seed, octaves=3, persistence=0.5, period=None):
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    amp, total = 1.0, 0.0
    freq = 1.0 / scale
    for o in range(octaves):
        if period:
            m = max(2, int(round(period * freq)))
            vals = rng.random(m)
            x = np.arange(n) * m / period
            i = np.floor(x).astype(int)
            f = x - i
            a, b = vals[i % m], vals[(i + 1) % m]
        else:
            m = int(n * freq) + 3
            vals = rng.random(m)
            x = np.arange(n) * freq
            i = x.astype(int)
            f = x - i
            a, b = vals[i], vals[i + 1]
        f = (1 - np.cos(f * np.pi)) / 2
        out += amp * (a * (1 - f) + b * f)
        total += amp
        amp *= persistence
        freq *= 2
    return out / total


# ======================================================================= спрайты-генераторы
def _rgba(h, w):
    return np.zeros((h, w, 4), np.uint8)


def _put(img, y, x, col):
    if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
        img[y, x, :3] = np.clip(col, 0, 255)
        img[y, x, 3] = 255


def make_cloud(seed, w, h, pal):
    """пиксельное облако: объединение кругов, плоское дно, 3 тона + подсветка."""
    rng = np.random.default_rng(seed)
    img = _rgba(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    base = h - 2
    mask = np.zeros((h, w), bool)
    centers = []
    n = max(3, w // 9)
    for i in range(n):
        cx = w * (0.12 + 0.76 * i / (n - 1)) + rng.uniform(-2, 2)
        mid = 1 - abs(i / (n - 1) - 0.5) * 2
        r = h * (0.28 + 0.42 * mid) * rng.uniform(0.85, 1.1)
        cy = base - r * 0.55
        centers.append((cx, cy, r))
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    mask &= yy <= base
    # тонировка
    light = np.zeros((h, w))
    for (cx, cy, r) in centers:
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
        nx, ny = (xx - cx) / r, (yy - cy) / r
        l = (-0.55 * nx - 0.85 * ny)
        light = np.maximum(light, np.where(d <= 1, l, -9))
    rel = (yy - (base - h * 0.35)) / (h * 0.35)
    img[mask, 3] = 255
    col = np.where(light[:, :, None] > 0.55, pal[0], pal[1])
    col = np.where((rel[:, :, None] > 0.35) | (light[:, :, None] < -0.25), pal[2], col)
    col = np.where((rel[:, :, None] > 0.78), pal[3], col)
    img[:, :, :3] = np.where(mask[:, :, None], col, 0).astype(np.uint8)
    return img


def make_round_tree(seed, r, trunk_h, pal, tpal):
    rng = np.random.default_rng(seed)
    w = int(2 * r + 6)
    h = int(2 * r + trunk_h + 4)
    img = _rgba(h, w)
    cx, cy = w / 2, r + 2
    yy, xx = np.mgrid[0:h, 0:w]
    blobs = [(cx, cy, r)]
    for i in range(4):
        a = rng.uniform(0, 2 * math.pi)
        rr = r * rng.uniform(0.45, 0.65)
        blobs.append((cx + math.cos(a) * r * 0.5, cy + math.sin(a) * r * 0.35 + r * 0.1, rr))
    mask = np.zeros((h, w), bool)
    light = np.full((h, w), -9.0)
    for (bx, by, br) in blobs:
        m = (xx - bx) ** 2 + (yy - by) ** 2 <= br * br
        mask |= m
        l = (-(xx - bx) * 0.6 - (yy - by) * 0.8) / br
        light = np.where(m, np.maximum(light, l), light)
    # ствол
    tw = max(2, int(r / 3))
    tx0 = int(cx - tw / 2)
    ty0 = int(cy + r * 0.5)
    for y in range(ty0, h):
        for x in range(tx0, tx0 + tw):
            _put(img, y, x, tpal[1] if x == tx0 else tpal[0])
    noise = rng.random((h, w))
    col = np.where(light[:, :, None] > 0.5, pal[3], pal[2])
    col = np.where(light[:, :, None] < 0.1, pal[1], col)
    col = np.where((light[:, :, None] < -0.45), pal[0], col)
    # фактура листвы
    col = np.where(((noise < 0.08) & (light > 0.1))[:, :, None], pal[1], col)
    col = np.where(((noise > 0.94) & (light > -0.2))[:, :, None], pal[3], col)
    img[mask, :3] = col[mask].astype(np.uint8)
    img[mask, 3] = 255
    # контур
    ol = outline_mask(mask, 1) & ~mask
    ol[ty0:, :] = False
    img[ol, :3] = (pal[0] * 0.8).astype(np.uint8)
    img[ol, 3] = 255
    return img


def make_pine(seed, h_, pal, tpal):
    rng = np.random.default_rng(seed)
    w = int(h_ * 0.62) + 4
    h = h_ + 4
    img = _rgba(h, w)
    cx = w / 2
    tiers = 4
    for i in range(tiers):
        top = 1 + i * h_ * 0.2
        bot = top + h_ * 0.34
        for y in range(int(top), int(bot)):
            frac = (y - top) / max(1, bot - top)
            hw = (0.5 + frac * (0.35 + i * 0.12)) * w * 0.5
            hw = min(hw, w / 2 - 1)
            for x in range(int(cx - hw), int(cx + hw) + 1):
                rel = (x - (cx - hw)) / max(1, 2 * hw)
                c = pal[2] if rel < 0.35 else pal[1]
                if rel < 0.12 or (y == int(bot) - 1):
                    c = pal[3] if rel < 0.12 else pal[0]
                if rel > 0.85:
                    c = pal[0]
                _put(img, y, x, c)
    tb = int(1 + (tiers - 1) * h_ * 0.2 + h_ * 0.34)
    for y in range(tb - 1, h):
        _put(img, y, int(cx) - 1, tpal[1])
        _put(img, y, int(cx), tpal[0])
    return img


def make_bush(seed, r, pal):
    rng = np.random.default_rng(seed)
    w, h = int(2 * r + 4), int(r * 1.3 + 3)
    img = _rgba(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), bool)
    light = np.full((h, w), -9.0)
    for i in range(3):
        bx = w * (0.3 + 0.2 * i) + rng.uniform(-1, 1)
        br = r * (0.55 + 0.25 * (i == 1))
        by = h - 1 - br * 0.6
        m = (xx - bx) ** 2 + (yy - by) ** 2 <= br * br
        mask |= m & (yy < h)
        light = np.where(m, np.maximum(light, (-(xx - bx) * 0.5 - (yy - by) * 0.85) / br), light)
    col = np.where(light[:, :, None] > 0.45, pal[3], pal[2])
    col = np.where(light[:, :, None] < 0.0, pal[1], col)
    img[mask, :3] = col[mask].astype(np.uint8)
    img[mask, 3] = 255
    ol = outline_mask(mask, 1) & ~mask
    img[ol, :3] = (pal[0] * 0.85).astype(np.uint8)
    img[ol, 3] = 255
    return img


# ======================================================================= реквизит
TROPHY_ASCII = [
    '.....ooooooo.....',
    '...ooWWYYYYYoo...',
    '..oWWYYYYYYYYSo..',
    'oooWYYYYnYYYYSooo',
    'o.oWYYYnnYYYYSo.o',
    'o.oWYYYYnYYYYSo.o',
    'o..oWYYYnYYYSo..o',
    '.o.oWYYYnYYYSo.o.',
    '..ooWYYnnnYYSoo..',
    '....oWYYYYYSo....',
    '.....ooWYSoo.....',
    '.......oYo.......',
    '.......oYo.......',
    '.....ooWYSoo.....',
    '....oWWYYYSSo....',
    '...obbbbbbbbbo...',
    '...oBBBBBBBBBo...',
    '...ooooooooooo...',
]
TROPHY_CMAP = {'o': (90, 56, 16), 'W': (255, 246, 190), 'Y': (246, 198, 66), 'S': (200, 140, 40),
               'n': (176, 116, 26), 'b': (120, 84, 56), 'B': (92, 62, 40)}
TROPHY = from_ascii(TROPHY_ASCII, TROPHY_CMAP)

# две половинки разбитого кубка (разлом зигзагом)
_TL = []
_TR = []
_cut = [8, 9, 8, 9, 8, 7, 8, 9, 8, 9, 8, 8, 8, 9, 8, 9, 8, 9]
for y, row in enumerate(TROPHY_ASCII):
    c = _cut[y]
    _TL.append(row[:c] + '.' * (len(row) - c))
    _TR.append('.' * c + row[c:])
TROPHY_L = from_ascii(_TL, TROPHY_CMAP)
TROPHY_R = from_ascii(_TR, TROPHY_CMAP)

HEART_TROPHY = None


def trophy_heart():
    """кубок с сердцем вместо «1»"""
    rows = [r.replace('n', 'Y') for r in TROPHY_ASCII]
    img = from_ascii(rows, TROPHY_CMAP)
    heart = ['R.R', 'RRR', '.R.']
    for y, r in enumerate(heart):
        for x, ch in enumerate(r):
            if ch == 'R':
                img[4 + y, 7 + x, :3] = (232, 56, 86)
    return img


def make_pedestal(w=26, h=16):
    img = _rgba(h, w)
    top, body, dark, ol, hl = hexc('#c3c7d6'), hexc('#9296ab'), hexc('#6c7088'), hexc('#3e4156'), hexc('#dde0ea')
    for y in range(h):
        for x in range(w):
            inset = 0 if y < 4 else 2
            if x < inset or x >= w - inset:
                continue
            if y in (0,) or x in (inset, w - inset - 1) or y == h - 1 or y == 3 or y == 4 and (x < 2 or x > w - 3):
                c = ol
            elif y < 3:
                c = hl if y == 1 else top
            else:
                c = dark if x > w - inset - 5 else body
                if x == inset + 1:
                    c = top
            _put(img, y, x, c)
    # трещинки
    for (y, x) in [(7, 8), (8, 9), (9, 9), (10, 10), (11, 15), (12, 16)]:
        _put(img, y, x, ol)
    return img


PEDESTAL = make_pedestal()


# лист-зонтик: 5 тонов от тёмной изнанки к подсвеченной макушке + почти чёрный контур
LEAF_TONES = ('#173f1d', '#20552a', '#2f8a3c', '#4fb14a', '#8ee06a')
LEAF_OUTLINE = '#0d2d13'
# кончики рёбер по краю купола (доля полуширины); между ними край подтянут вверх — фестоны
LEAF_TIPS = (-1.0, -0.77, -0.42, 0.0, 0.42, 0.77, 1.0)
LEAF_CANOPY_H = 0   # высота купола в спрайте LEAF (ниже — черешок); ставится в make_leaf_umbrella


def make_leaf_umbrella(w=56):
    """лист-зонтик из лопуха: симметричный свод с фестончатым краем и загнутой вниз тёмной
    изнанкой, 5 радиальных прожилок по поверхности свода, узелок-навершие на макушке,
    слегка изогнутый черешок с утолщением в месте хвата. Свет — сверху-слева."""
    global LEAF_CANOPY_H
    k = w / 56.0
    cx = w / 2.0
    rx = (w - 2) / 2.0
    apex = max(2, int(round(4 * k)))          # верх свода (над ним — узелок)
    ry_dome = 13.0 * k                        # высота свода
    ry_rim = 5.0 * k                          # насколько провисает ближняя половина обода
    rim_y = apex + ry_dome                    # линия обода на силуэте (боковые точки)
    bay, tip = 2.2 * k, 1.2 * k               # подъём фестона и вынос кончика ребра
    tones = [hexc(c) for c in LEAF_TONES]

    def top_y(dx):
        return apex + ry_dome * (1.0 - math.sqrt(max(0.0, 1.0 - dx * dx)))

    def bot_y(dx):
        off = 0.0
        for i in range(len(LEAF_TIPS) - 1):
            a, b = LEAF_TIPS[i], LEAF_TIPS[i + 1]
            if a <= dx <= b:
                off = -bay * math.sin(math.pi * (dx - a) / (b - a))
                break
        d = min(abs(dx - tp) for tp in LEAF_TIPS)
        if d < 0.07:
            off += tip * (1.0 - d / 0.07)
        return rim_y + ry_rim * math.sqrt(max(0.0, 1.0 - dx * dx)) + off

    canopy_h = int(math.ceil(rim_y + ry_rim + tip)) + 2
    h = canopy_h + int(round(24 * k))
    img = _rgba(h, w)
    tone = np.full((h, w), -1, np.int8)
    # ---- свод: над линией обода — наружная поверхность, под ней — изнанка у загнутого края
    for x in range(w):
        dx = (x + 0.5 - cx) / rx
        if abs(dx) > 1.0:
            continue
        yt, yb = top_y(dx), bot_y(dx)
        for y in range(int(round(yt)), int(round(yb)) + 1):
            yy = y + 0.5
            if yy >= rim_y:
                dep = (yy - rim_y) / max(1.0, yb - rim_y)
                t = 1 if (dep < 0.55 and dx < 0.45) else 0
            else:
                q = min(1.0, (yy - yt) / max(1.0, rim_y - yt))
                lum = 0.56 - 0.60 * dx - 0.70 * q
                t = 4 if lum > 0.72 else (3 if lum > 0.26 else 2)
                if yy - yt < 1.6 and dx < 0.5:
                    t = 4               # блик по верхней кромке слева
            tone[y, x] = t
    # ---- прожилки: идут по поверхности свода от макушки к кончикам рёбер
    for dxt in LEAF_TIPS[1:-1]:
        st = max(-1.0, min(1.0, dxt))
        ct = math.sqrt(max(0.0, 1.0 - st * st))
        n = max(8, int(rx * 2.4))
        for i in range(n + 1):
            u = 0.14 + (0.99 - 0.14) * i / n
            x = cx + rx * u * st
            y = apex + ry_dome * (1.0 - math.sqrt(max(0.0, 1.0 - u * u))) + ry_rim * ct * u * u
            xi, yi = int(x), int(y)
            if 0 <= yi < h and 0 <= xi < w and tone[yi, xi] >= 0:
                tone[yi, xi] = min(4, tone[yi, xi] + 1)
    # ---- узелок-навершие на макушке
    nh = max(2, int(round(3 * k)))
    nw = max(2, int(round(3 * k)))
    for i in range(nh):
        yy = apex - nh + i
        ww = max(1, nw - (nh - 1 - i))
        x0n = int(round(cx - ww / 2.0))
        for xx in range(x0n, x0n + ww):
            if 0 <= yy < h:
                tone[yy, xx] = 3 if i == 0 else 1
    m = tone >= 0
    for t in range(5):
        img[(tone == t), :3] = tones[t]
    img[m, 3] = 255
    # ---- контур (черешок обводить не надо — он рисуется после)
    ol = outline_mask(m, 1) & ~m
    img[ol, :3] = hexc(LEAF_OUTLINE)
    img[ol, 3] = 255
    rows = np.where(img[:, :, 3].any(1))[0]
    LEAF_CANOPY_H = int(rows.max()) + 1
    # ---- черешок: лёгкая дуга, книзу толще (хват), слева светлая грань, справа тень
    sd, sm, sl = hexc('#4a3020'), hexc('#5f4128'), hexc('#825c38')
    y_top = LEAF_CANOPY_H - 2
    for y in range(y_top, h):
        t = (y - y_top) / max(1.0, h - 1 - y_top)
        x = cx - 0.5 + 2.6 * k * t * t
        ww = max(2, int(round((2.0 + 1.6 * t * t * t) * k)))
        xb = int(round(x - (ww - 1) / 2.0))
        for j in range(ww):
            _put(img, y, xb + j, sl if j == 0 else (sd if j == ww - 1 else sm))
    return img


LEAF = make_leaf_umbrella(56)


def make_hoverboard(t):
    w = 30
    img = _rgba(8, w)
    ol, b0, b1, st = hexc('#0f1640'), hexc('#2b3a9c'), hexc('#4a62d8'), hexc('#8ee3f0')
    for x in range(1, w - 1):
        _put(img, 0, x, ol)
        _put(img, 1, x, b1)
        _put(img, 2, x, st if 4 < x < w - 5 else b1)
        _put(img, 3, x, b0)
        _put(img, 4, x, ol)
    for y in (1, 2, 3):
        _put(img, y, 0, ol)
        _put(img, y, w - 1, ol)
    # сопла-струи
    fl = [hexc('#ffffff'), hexc('#8ee3f0'), hexc('#3fa0ff')]
    ph = int(t * 20) % 3
    for jx in (5, w - 7):
        for x in (jx, jx + 1):
            _put(img, 5, x, fl[0])
            _put(img, 6, x, fl[(1 + ph) % 3])
            if ph != 2:
                _put(img, 7, x, fl[2])
    return img


def make_laptop_front(scale=1):
    """маленький открытый ноутбук вид спереди-сбоку для сцен (если нужен)."""
    rows = [
        '.oooooooooooo.',
        '.oSSSSSSSSSSo.',
        '.oSggggggggSo.',
        '.oSgGGgGGggSo.',
        '.oSggggggggSo.',
        '.oSgGgGGGggSo.',
        '.oSSSSSSSSSSo.',
        'oooooooooooooo',
        'oKKKKKKKKKKKKo',
        '.oooooooooooo.',
    ]
    return scale_nn(from_ascii(rows, {'o': (30, 30, 40), 'S': (60, 64, 80), 'g': (24, 28, 40), 'G': (120, 230, 140),
                                      'K': (170, 176, 190)}), scale)


# ======================================================================= слои луга
class Meadow:
    """Пейзаж луга с параллаксом. width — ширина мира (для скролла), ground_y — линия земли."""

    def __init__(self, mode='day', width=W, ground_y=214, seed=7, trees=True, horizon=172, hills=True,
                 tree_scale=1.0):
        self.mode, self.width, self.gy, self.seed = mode, width, ground_y, seed
        self.p = P(mode)
        self.horizon = horizon
        p = self.p
        rng = np.random.default_rng(seed)
        # небо
        self.sky = canvas()
        vgradient(self.sky, 0, horizon + 10, p['sky'])
        self.sky[horizon + 10:] = p['sky'][-1]
        # горы (параллакс 0.15)
        mw = int(width * 0.15 + W) + 8
        self.mount = self._mountains(mw, horizon, rng)
        # дальние холмы (0.3)
        fw = int(width * 0.3 + W) + 8
        self.hill_far = self._hills(fw, horizon - 8, 16, p['hill_far'], seed + 1, scale=70, trees=False)
        # средние холмы с деревьями (0.55)
        mw2 = int(width * 0.55 + W) + 8
        self.hill_mid = self._hills(mw2, horizon + 6, 20, p['hill_mid'], seed + 2, scale=90, trees=trees,
                                    tree_scale=tree_scale)
        # передний план — земля (1.0)
        self.ground = self._ground(width + 8, ground_y)
        self.clouds = [make_cloud(seed * 10 + i, int(rng.integers(40, 80)), int(rng.integers(16, 26)), p['cloud'])
                       for i in range(6)]
        self.cloud_pos = [(float(rng.uniform(0, W + 200)), float(rng.uniform(12, horizon - 70)), float(rng.uniform(2, 6)))
                          for _ in range(6)]

    def _mountains(self, w, horizon, rng):
        p = self.p
        img = _rgba(H, w)
        rr = np.random.default_rng(self.seed + 21)
        xs = np.arange(w)

        def range_layer(base_y, hmin, hmax, step, col_lit, col_sh, haze, snow, snow_min):
            peaks = []
            x = -30.0
            while x < w + 60:
                peaks.append((x, base_y - rr.uniform(hmin, hmax), rr.uniform(0.55, 0.95)))
                x += rr.uniform(step * 0.6, step * 1.2)
            tops = np.stack([py + np.abs(xs - px) * sl for (px, py, sl) in peaks])
            idx = tops.argmin(0)
            top = tops.min(0) + np.round(smooth_noise(w, 5, int(rr.integers(1e6)), octaves=2) * 2)
            top = top.astype(int)
            for x in range(w):
                px, py, sl = peaks[idx[x]]
                lit = x < px + (smooth_noise(1, 1, x)[0] * 0 if False else 0)
                zig = int(((x * 7) % 5) - 2) // 2
                for y in range(max(0, top[x]), horizon + 14):
                    # граница свет/тень — от вершины вниз с наклоном
                    ridge_x = px + (y - py) * 0.18 + zig
                    c = col_lit if x < ridge_x else col_sh
                    if snow is not None and py < snow_min:
                        sd = 7 + ((x * 3 + 1) % 4)
                        if y < py + sd + abs(x - px) * 0.25:
                            c = snow if x < ridge_x else snow * 0.86
                    fr = (y - (horizon - 16)) / 26
                    if fr > 0 and DITHER[y % H, x % W] < fr:
                        c = haze
                    img[y, x, :3] = c
                    img[y, x, 3] = 255
            return top

        haze = p['mount'][2]
        far_lit = p['mount'][1] * 0.55 + p['sky'][-2] * 0.45
        far_sh = p['mount'][0] * 0.55 + p['sky'][-2] * 0.45
        range_layer(horizon - 10, 26, 56, 70, far_lit, far_sh, haze, p['snow'] * 0.7 + p['sky'][-2] * 0.3, horizon - 52)
        range_layer(horizon + 2, 18, 44, 90, p['mount'][1], p['mount'][0], haze, p['snow'], horizon - 30)
        return img

    def _hills(self, w, base, amp, pal, seed, scale=80, trees=False, tree_scale=1.0):
        img = _rgba(H, w)
        prof = smooth_noise(w, scale, seed, octaves=3, persistence=0.45)
        top = (base - prof * amp).astype(int)
        rng = np.random.default_rng(seed)
        for x in range(w):
            y0 = max(0, top[x])
            img[y0:, x, :3] = pal[0]
            img[y0:, x, 3] = 255
            if len(pal) > 1:
                img[y0, x, :3] = pal[1]
                if y0 + 1 < H and (x % 3 == 0):
                    img[y0 + 1, x, :3] = pal[1]
            if len(pal) > 2:
                # тень снизу с дизером
                for y in range(y0 + 6, H):
                    if DITHER[y, x % W] < min(1.0, (y - y0 - 6) / 30):
                        img[y, x, :3] = pal[2]
        if trees:
            p = self.p
            x = int(rng.integers(0, 20))
            while x < w - 10:
                kind = rng.random()
                if kind < 0.55:
                    r = int(rng.integers(5, 9) * tree_scale)
                    tr = make_round_tree(int(rng.integers(0, 1e6)), r, int(rng.integers(3, 6) * tree_scale), p['tree'], p['trunk'])
                elif kind < 0.8:
                    tr = make_pine(int(rng.integers(0, 1e6)), int(rng.integers(14, 22) * tree_scale), p['tree'], p['trunk'])
                else:
                    tr = make_bush(int(rng.integers(0, 1e6)), int(rng.integers(3, 6) * tree_scale), p['tree'])
                th, tw = tr.shape[:2]
                ty = top[min(w - 1, x + tw // 2)] - th + 3
                m = tr[:, :, 3] > 0
                ys, xs = np.where(m)
                for yy, xx in zip(ys, xs):
                    Y, X = ty + yy, x + xx
                    if 0 <= Y < H and 0 <= X < w:
                        img[Y, X] = tr[yy, xx]
                x += int(tw * rng.uniform(0.6, 2.6)) + int(rng.integers(0, 30))
        return img

    def _ground(self, w, gy):
        p = self.p
        img = _rgba(H, w)
        g = p['grass']
        img[gy:, :, :3] = g[0]
        img[gy:, :, 3] = 255
        rng = np.random.default_rng(self.seed + 5)
        # нижняя тень
        for y in range(gy + 3, H):
            fr = min(1.0, (y - gy - 3) / 40)
            m = DITHER[y, np.arange(w) % W] < fr * 0.8
            img[y, m, :3] = g[2]
        for y in range(gy + 30, H):
            fr = min(1.0, (y - gy - 30) / 30)
            m = DITHER[y, np.arange(w) % W] < fr * 0.6
            img[y, m, :3] = g[3]
        # кромка
        img[gy, :, :3] = g[1]
        # пучки травы и цветы (статичные)
        for x in range(w):
            if rng.random() < 0.35:
                hh = int(rng.integers(1, 4))
                for k in range(hh):
                    if gy - 1 - k >= 0:
                        img[gy - 1 - k, x, :3] = g[1] if k == hh - 1 else g[0]
                        img[gy - 1 - k, x, 3] = 255
        for i in range(int(w / 7)):
            x = int(rng.integers(0, w))
            y = int(rng.integers(gy + 2, H - 2))
            c = p['flowers'][int(rng.integers(0, len(p['flowers'])))]
            img[y, x, :3] = c
            if rng.random() < 0.5 and x + 1 < w:
                img[y - 1, x, :3] = c * 0.9 + 25
            if rng.random() < 0.4:
                img[y + 1, x, :3] = g[3]
        # тёмные «кочки»
        for i in range(int(w / 5)):
            x = int(rng.integers(0, w - 3))
            y = int(rng.integers(gy + 4, H - 1))
            img[y, x:x + int(rng.integers(2, 5)), :3] = g[2]
            img[y - 1, x + 1, :3] = g[1]
        return img

    # ------------------------------------------------------------------
    def draw_sky(self, dst, t=0.0, cam_x=0.0, clouds=True, cloud_speed=1.0):
        dst[:] = self.sky
        if clouds:
            self.draw_clouds(dst, t, cam_x, cloud_speed)

    def draw_clouds(self, dst, t, cam_x=0.0, speed=1.0, alpha=1.0, tint=None):
        span = W + 240
        for spr, (x0, y, v) in zip(self.clouds, self.cloud_pos):
            x = (x0 + t * v * speed - cam_x * 0.08) % span - 120
            blit(dst, spr, x, y, alpha=alpha, tint=tint)

    def draw_layers(self, dst, cam_x=0.0, t=0.0, far=True, mid=True, ground=True, mount=True, tint=None):
        if mount:
            blit(dst, self.mount, -int(cam_x * 0.15), 0, tint=tint)
        if far:
            blit(dst, self.hill_far, -int(cam_x * 0.3), 0, tint=tint)
        if mid:
            blit(dst, self.hill_mid, -int(cam_x * 0.55), 0, tint=tint)
        if ground:
            blit(dst, self.ground, -int(cam_x), 0, tint=tint)

    def draw(self, dst, t=0.0, cam_x=0.0, clouds=True, tint=None, cloud_speed=1.0):
        self.draw_sky(dst, t, cam_x, clouds=clouds, cloud_speed=cloud_speed)
        self.draw_layers(dst, cam_x, t, tint=tint)

    def grass_front(self, dst, t, cam_x=0.0, y=None, density=0.5, seed=3, color=None, tip=None):
        """качающиеся травинки переднего плана поверх персонажей (у самой земли)."""
        y = self.gy + 1 if y is None else y
        g = self.p['grass']
        c0 = g[3] if color is None else color
        c1 = g[0] if tip is None else tip
        rng = np.random.default_rng(seed)
        xs = rng.integers(0, W + 40, int((W + 40) * density))
        hs = rng.integers(2, 6, len(xs))
        for x, hh in zip(xs, hs):
            X = int((x - cam_x) % (W + 40)) - 20
            sway = math.sin(t * 2.2 + x * 0.15) * 0.8
            for k in range(hh):
                dx = int(round(sway * k / hh))
                pset(dst, X + dx, y - k, c1 if k >= hh - 2 else c0)


# ======================================================================= погода и эффекты
def rain(dst, t, n=260, speed=260.0, wind=0.25, ground_y=H, color=(170, 190, 230), alpha=0.7, seed=5,
         length=5, surfaces=(), splash=True, intensity=1.0):
    """дождь без состояния. surfaces: список (x0, x1, y) — поверхности, о которые капли разбиваются."""
    rng = np.random.default_rng(seed)
    N = int(n * intensity)
    x0 = rng.uniform(-60, W + 60, n)[:N]
    ph = rng.uniform(0, H + 60, n)[:N]
    sp = rng.uniform(0.85, 1.15, n)[:N] * speed
    col = np.asarray(color, np.float32)
    span = H + 60
    for i in range(N):
        pos = (ph[i] + t * sp[i]) % span - 30
        x = x0[i] + wind * pos
        # до какой высоты летит капля
        stop = ground_y + rng_stop(i, seed)
        for (sx0, sx1, sy) in surfaces:
            if sx0 <= x <= sx1:
                stop = min(stop, sy)
        if pos < stop:
            for k in range(length):
                yy = pos - k
                if yy < stop:
                    pset(dst, x - wind * k, yy, col, alpha * (1 - k / length * 0.6))
        # всплеск: сколько времени прошло с момента удара
        if splash:
            dt_hit = ((pos - stop) % span) / sp[i]
            if pos >= stop and dt_hit < 0.09:
                hx = x0[i] + wind * stop
                a = alpha * (1 - dt_hit / 0.09)
                pset(dst, hx - 1, stop - 1, col, a)
                pset(dst, hx + 1, stop - 1, col, a)
                pset(dst, hx - 2, stop - 2, col, a * 0.7)
                pset(dst, hx + 2, stop - 2, col, a * 0.7)


def rng_stop(i, seed):
    return ((i * 7919 + seed * 104729) % 23) - 4


def rainbow(dst, cx, cy, r, alpha=1.0, band=2, clip_y=None):
    cols = ['#ff5a5a', '#ff9a4a', '#ffe066', '#6ee07a', '#4ab8ff', '#6a7aff', '#b07aff']
    n = len(cols)
    R_out = r + n * band
    y0, y1 = int(max(0, cy - R_out)), int(min(H, cy if clip_y is None else clip_y))
    x0, x1 = int(max(0, cx - R_out)), int(min(W, cx + R_out))
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    k = ((R_out - d) / band).astype(int)
    for i, c in enumerate(cols):
        m = (k == i) & (d <= R_out) & (d > r)
        m &= DITHER[y0:y1, x0:x1] < alpha
        dst[y0:y1, x0:x1][m] = dst[y0:y1, x0:x1][m] * 0.35 + hexc(c) * 0.65


def lightning_bolt(dst, seed, x_top, y_bot, color=(255, 255, 220), glow=(180, 190, 255)):
    rng = np.random.default_rng(seed)
    x, y = x_top, 0
    pts = [(x, y)]
    while y < y_bot:
        y += rng.integers(6, 14)
        x += rng.integers(-9, 10)
        pts.append((x, min(y, y_bot)))
    for (a, b) in zip(pts[:-1], pts[1:]):
        line(dst, a[0] + 1, a[1], b[0] + 1, b[1], glow, 0.6)
        line(dst, a[0] - 1, a[1], b[0] - 1, b[1], glow, 0.6)
        line(dst, a[0], a[1], b[0], b[1], color)
        # ответвление
        if rng.random() < 0.3:
            line(dst, b[0], b[1], b[0] + rng.integers(-14, 14), b[1] + rng.integers(6, 14), glow, 0.8)


def flash(dst, amount, color=(255, 255, 255)):
    if amount > 0:
        dst[:] = dst * (1 - amount) + np.asarray(color, np.float32) * amount


def hearts(dst, t, x, y, n=6, seed=1, start=0.0, interval=0.35, life=2.2, rise=26.0, sway=6.0, small_every=2,
           spread=10):
    rng = np.random.default_rng(seed)
    offs = rng.uniform(-spread, spread, n)
    for i in range(n):
        a = t - (start + i * interval)
        if a < 0 or a > life:
            continue
        hx = x + offs[i] + math.sin(a * 3.0 + i) * sway * min(1, a)
        hy = y - a * rise
        spr = HEART_S if (i % small_every == 1) else HEART
        fade_a = 1.0 if a < life - 0.5 else (life - a) / 0.5
        # «пиксельное» появление: первые 0.12 c — маленькое сердце
        if a < 0.12:
            spr = HEART_S
        blit_dither(dst, spr, hx - spr.shape[1] / 2, hy - spr.shape[0] / 2, fade_a)


def blit_dither(dst, spr, x, y, alpha=1.0):
    """блит с «пиксельной» прозрачностью через матрицу Байера (без полупрозрачности)."""
    if alpha >= 0.999:
        blit(dst, spr, x, y)
        return
    if alpha <= 0:
        return
    x, y = int(round(x)), int(round(y))
    h, w = spr.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    m = (s[:, :, 3] > 0) & (DITHER[y0:y1, x0:x1] < alpha)
    dst[y0:y1, x0:x1][m] = s[:, :, :3][m]


def sparkles(dst, t, x0, y0, w, h, n=6, seed=2, period=1.2):
    rng = np.random.default_rng(seed)
    for i in range(n):
        ph = rng.uniform(0, period)
        px_, py_ = x0 + rng.uniform(0, w), y0 + rng.uniform(0, h)
        a = ((t + ph) % period) / period
        if a < 0.5:
            spr = SPARK_S if a < 0.15 or a > 0.4 else SPARK
            blit(dst, spr, px_ - spr.shape[1] // 2, py_ - spr.shape[0] // 2)


def glint(dst, x, y, t, period=2.0, offset=0.0):
    a = ((t + offset) % period) / period
    if a < 0.18:
        spr = SPARK if 0.05 < a < 0.13 else SPARK_S
        blit(dst, spr, x - spr.shape[1] // 2, y - spr.shape[0] // 2)


def stars_field(dst, t, n=90, seed=4, y_max=H, alpha=1.0, color=(255, 246, 214)):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, W, n)
    ys = rng.uniform(0, y_max, n)
    ph = rng.uniform(0, 6.28, n)
    sp = rng.uniform(1.0, 3.0, n)
    big = rng.random(n) < 0.12
    col = np.asarray(color, np.float32)
    for i in range(n):
        b = 0.55 + 0.45 * math.sin(t * sp[i] + ph[i])
        a = alpha * b
        if a <= 0.05:
            continue
        pset(dst, xs[i], ys[i], col, a)
        if big[i] and b > 0.7:
            for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                pset(dst, xs[i] + dx, ys[i] + dy, col, a * 0.5)


def dust_puffs(dst, t, x, y, n=6, seed=9, life=0.6, spread=18, size=4, color=(235, 230, 220), shade=(190, 180, 170)):
    """облачка пыли (от шагов/приземления), t — время с момента удара."""
    if t < 0 or t > life:
        return
    rng = np.random.default_rng(seed)
    k = t / life
    for i in range(n):
        ang = rng.uniform(math.pi * 0.9, math.pi * 2.1)
        dist = spread * ease_out(k) * rng.uniform(0.6, 1.0)
        px_ = x + math.cos(ang) * dist
        py_ = y + math.sin(ang) * dist * 0.35 - k * 4
        r = size * (1 - k * 0.7) * rng.uniform(0.7, 1.2)
        if r < 0.6:
            continue
        m = disc_mask(r)
        rr = (m.shape[0] - 1) // 2
        blit_mask(dst, m, px_ - rr, py_ - rr, shade)
        m2 = disc_mask(max(0.5, r - 1))
        r2 = (m2.shape[0] - 1) // 2
        blit_mask(dst, m2, px_ - r2 - 0.5, py_ - r2 - 0.8, color)


def brawl_cloud(dst, t, cx, cy, r=34, seed=11):
    """мультяшное облако драки: пульсирующие клубы + вылетающие звёздочки."""
    rng = np.random.default_rng(seed)
    n = 11
    for i in range(n):
        a = i / n * 2 * math.pi + math.sin(t * 7 + i) * 0.3
        rr = r * (0.55 + 0.25 * math.sin(t * 11 + i * 1.7))
        px_ = cx + math.cos(a) * r * 0.55
        py_ = cy + math.sin(a) * r * 0.32
        disc(dst, px_, py_ + 2, rr * 0.62, (150, 146, 160))
    for i in range(n):
        a = i / n * 2 * math.pi + math.sin(t * 7 + i) * 0.3
        rr = r * (0.5 + 0.22 * math.sin(t * 11 + i * 1.7))
        px_ = cx + math.cos(a) * r * 0.5
        py_ = cy + math.sin(a) * r * 0.28
        disc(dst, px_, py_, rr * 0.6, (236, 232, 240))
    disc(dst, cx, cy, r * 0.45, (250, 248, 252))
    # звёздочки/символы
    for i in range(5):
        ph = (t * 2.3 + i * 0.37) % 1.0
        ang = rng.uniform(0, 2 * math.pi) + i
        sx = cx + math.cos(ang) * (r * 0.5 + ph * r * 0.9)
        sy = cy + math.sin(ang) * (r * 0.3 + ph * r * 0.6) - ph * 6
        blit(dst, STAR5 if i % 2 else SPARK, sx - 3, sy - 3)


def speed_lines(dst, t, y0, y1, n=18, seed=3, speed=600.0, color=(255, 255, 255), alpha=0.8, length=(14, 40), direction=-1):
    rng = np.random.default_rng(seed)
    for i in range(n):
        y = rng.uniform(y0, y1)
        L = rng.uniform(*length)
        ph = rng.uniform(0, W + 80)
        x = (ph + t * speed * rng.uniform(0.8, 1.2)) % (W + 80) - 40
        if direction < 0:
            x = W - x
        line(dst, x, y, x + L * (1 if direction > 0 else 1), y, color, alpha)


def anger_mark(dst, x, y, t, scale=1):
    """пульсирующая «венка» гнева"""
    s = ANGER2 if int(t * 6) % 2 else ANGER
    blit(dst, scale_nn(s, scale), x - 3 * scale, y - 3 * scale)


def steam(dst, t, x, y, seed=1, n=3, color=(240, 240, 245)):
    for i in range(n):
        a = (t * 1.6 + i / n) % 1.0
        px_ = x + (i - (n - 1) / 2) * 7 + math.sin(a * 6 + i) * 2
        py_ = y - a * 16
        r = 1.5 + a * 2.5
        blit_dither_disc(dst, px_, py_, r, color, 1 - a)


def blit_dither_disc(dst, cx, cy, r, color, alpha):
    m = disc_mask(r)
    rr = (m.shape[0] - 1) // 2
    x, y = int(round(cx - rr)), int(round(cy - rr))
    h, w = m.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    mm = m[y0 - y:y1 - y, x0 - x:x1 - x] & (DITHER[y0:y1, x0:x1] < alpha)
    dst[y0:y1, x0:x1][mm] = color


def shadow_ellipse(dst, cx, cy, rx, ry=2, color=(0, 0, 0), alpha=0.25):
    m = ellipse_mask(rx, ry)
    blit_mask(dst, m, cx - (m.shape[1] - 1) / 2, cy - (m.shape[0] - 1) / 2, color, alpha)


def vignette(dst, strength=0.35):
    yy = (YY - H / 2) / (H / 2)
    xx = (XX - W / 2) / (W / 2)
    d = np.sqrt(xx * xx * 0.8 + yy * yy)
    v = np.clip((d - 0.75) / 0.6, 0, 1) * strength
    q = np.round(v * 6) / 6
    dst *= (1 - q)[:, :, None]


def grade(dst, mul=(1, 1, 1), add=(0, 0, 0)):
    dst *= np.asarray(mul, np.float32)
    dst += np.asarray(add, np.float32)
