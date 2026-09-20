"""Готовые 2.5D-миры: текстуры земли (вид сверху), панорамы гор/холмов/облаков, деревья-билборды.

    world = MeadowWorld('day')          # 'day' | 'storm' | 'golden' | 'sunset'
    c = canvas()
    world.draw_backdrop(c, cam, t)      # небо + облака + горы + холмы
    world.draw_ground(c, cam)           # земля Mode 7
    items = world.tree_items() + [ (X, 0, Z, lambda d, cm: ...), ... ]
    draw_world(c, cam, items)           # деревья и персонажи с сортировкой по глубине
"""
import math
import numpy as np
from px import *
from bg import P, make_round_tree, make_pine, make_bush, smooth_noise
from cam3d import *


def noise2d(size, cells, seed, octaves=3, persistence=0.5):
    """периодический 2D value-noise size x size, 0..1."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size), np.float32)
    amp, tot = 1.0, 0.0
    c = cells
    for o in range(octaves):
        grid = rng.random((c, c)).astype(np.float32)
        x = np.arange(size) * c / size
        i = np.floor(x).astype(int)
        f = x - i
        f = (1 - np.cos(f * np.pi)) / 2
        i0, i1 = i % c, (i + 1) % c
        a = grid[i0][:, i0] * (1 - f)[None, :] + grid[i0][:, i1] * f[None, :]
        b = grid[i1][:, i0] * (1 - f)[None, :] + grid[i1][:, i1] * f[None, :]
        out += amp * (a * (1 - f)[:, None] + b * f[:, None])
        tot += amp
        amp *= persistence
        c *= 2
    return out / tot


def bayer_tile(size):
    return BAYER4[np.arange(size)[:, None] % 4, np.arange(size)[None, :] % 4]


def make_grass_texture(pal, size=1024, seed=1, flowers=1.0, tufts=1.0):
    g = pal['grass']
    rng = np.random.default_rng(seed)
    tex = np.zeros((size, size, 3), np.float32)
    tex[:] = g[0]
    n1 = noise2d(size, 6, seed, 3)
    n2 = noise2d(size, 24, seed + 1, 2)
    d = bayer_tile(size)
    # крупные пятна: светлее/темнее
    light = (n1 - 0.55) * 4
    dark = (0.42 - n1) * 4
    tex[(light > d)] = g[1] * 0.35 + g[0] * 0.65
    tex[(light > d + 0.6)] = g[1] * 0.7 + g[0] * 0.3
    tex[(dark > d)] = g[2]
    tex[(dark > d + 0.7)] = g[3] * 0.6 + g[2] * 0.4
    # мелкая фактура
    m = (n2 > 0.62) & (rng.random((size, size)) < 0.25)
    tex[m] = g[2]
    m = (n2 < 0.38) & (rng.random((size, size)) < 0.2)
    tex[m] = g[1] * 0.6 + g[0] * 0.4
    # пучки травы «v»
    for i in range(int(size * size / 90 * tufts)):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        c = g[3] if rng.random() < 0.5 else g[1]
        tex[y, x] = c
        tex[(y - 1) % size, (x - 1) % size] = c
        tex[(y - 1) % size, (x + 1) % size] = c
    # цветы
    fl = pal['flowers']
    for i in range(int(size * size / 700 * flowers)):
        cx, cy = int(rng.integers(0, size)), int(rng.integers(0, size))
        col = fl[int(rng.integers(0, len(fl)))]
        for k in range(int(rng.integers(1, 5))):
            x = (cx + int(rng.integers(-4, 5))) % size
            y = (cy + int(rng.integers(-4, 5))) % size
            tex[y, x] = col
            tex[y, (x + 1) % size] = col
            tex[(y + 1) % size, x] = col * 0.85
            tex[(y + 1) % size, (x + 1) % size] = col * 0.85
            tex[y, x] = np.minimum(col + 60, 255)
    # камушки
    for i in range(int(size * size / 5000)):
        x, y = int(rng.integers(0, size - 3)), int(rng.integers(0, size - 3))
        tex[y:y + 2, x:x + 3] = (130, 128, 120)
        tex[y, x:x + 2] = (170, 168, 160)
        tex[y + 2, x:x + 3] = g[3]
    return np.clip(tex, 0, 255).astype(np.uint8)


def paint_disc(tex, cx, cy, r, color, jag=1.5, seed=0):
    size = tex.shape[0]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-int(r) - 3:int(r) + 4, -int(r) - 3:int(r) + 4]
    ang = np.arctan2(yy, xx)
    rr = r + np.sin(ang * 5 + seed) * jag + np.sin(ang * 11 + seed * 2) * jag * 0.5
    m = xx ** 2 + yy ** 2 <= rr ** 2
    ys = (cy + yy[m]) % size
    xs = (cx + xx[m]) % size
    tex[ys, xs] = color
    return tex


class MeadowWorld:
    FOG = {'day': '#b8d8f4', 'storm': '#4a5270', 'golden': '#f0d8b8', 'sunset': '#c86a78'}

    def __init__(self, mode='day', seed=7, tree_ring=(160, 900), n_trees=70, tex_size=1024, clear_radius=110,
                 center=(0.0, 0.0), extra_paint=None):
        self.mode = mode
        self.p = P(mode)
        p = self.p
        self.fog = hexc(self.FOG[mode])
        tex = make_grass_texture(p, tex_size, seed)
        if extra_paint is not None:
            extra_paint(tex, p)
        self.ground = Ground(tex, tpu=2.0, origin=(-tex_size / 4, -tex_size / 4), wrap=True)
        self.mount = make_mountain_pano(p, seed=seed + 1, height=90)
        self.hills = make_hill_pano([p['hill_mid'][0], p['hill_mid'][1]], seed=seed + 2, height=34, amp=14,
                                    tree_pal=p['tree'], trunk_pal=p['trunk'])
        self.hills_far = make_hill_pano([p['hill_far'][0], p['hill_far'][1]], seed=seed + 3, height=40, amp=20)
        self.clouds = make_cloud_pano(p, seed=seed + 4, height=150, n=16)
        self.sky = p['sky']
        # деревья-билборды вокруг поляны
        rng = np.random.default_rng(seed + 9)
        self.trees = []
        for i in range(n_trees):
            ang = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(*tree_ring)
            X = center[0] + math.cos(ang) * dist
            Z = center[1] + math.sin(ang) * dist
            if math.hypot(X - center[0], Z - center[1]) < clear_radius:
                continue
            k = rng.random()
            if k < 0.5:
                spr = make_round_tree(int(rng.integers(1e6)), int(rng.integers(9, 15)), int(rng.integers(6, 10)), p['tree'], p['trunk'])
            elif k < 0.8:
                spr = make_pine(int(rng.integers(1e6)), int(rng.integers(26, 42)), p['tree'], p['trunk'])
            else:
                spr = make_bush(int(rng.integers(1e6)), int(rng.integers(5, 9)), p['tree'])
            self.trees.append((X, Z, spr, float(rng.uniform(1.6, 2.4))))

    def draw_backdrop(self, dst, cam, t=0.0, clouds=True, cloud_drift=4.0):
        sky_dome(dst, cam, self.sky[::1], span=0.75, below=self.fog)
        if clouds:
            self.clouds.draw(dst, cam, drift=t * cloud_drift, y_offset=-8)
        self.mount.draw(dst, cam, y_offset=1)
        self.hills_far.draw(dst, cam, y_offset=2)
        self.hills.draw(dst, cam, y_offset=3)

    def draw_ground(self, dst, cam, fog_near=260, fog_far=1500):
        return self.ground.render(dst, cam, fog=self.fog, fog_near=fog_near, fog_far=fog_far)

    def tree_items(self, fog_near=260, fog_far=1500):
        items = []
        for (X, Z, spr, sc) in self.trees:
            def fn(d, cm, X=X, Z=Z, spr=spr, sc=sc):
                draw_sprite_3d(d, cm, spr, X, 0.0, Z, 1.0 / sc, anchor=(0.5, 0.97), fog=self.fog,
                               fog_near=fog_near, fog_far=fog_far)
            items.append((X, 0.0, Z, fn))
        return items

    def render_base(self, cam, t=0.0, clouds=True):
        c = canvas()
        self.draw_backdrop(c, cam, t, clouds)
        self.draw_ground(c, cam)
        return c
