import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from px import *
from bg import PEDESTAL, TROPHY, glint
from cam3d import *
from worlds import MeadowWorld
from chars import load

cl, cx = load()
pix = CodexPixelizer()
t0 = time.time()
for mode in ['day', 'sunset']:
    t0 = time.time()
    world = MeadowWorld(mode)
    print(mode, 'world init', round(time.time() - t0, 3))
cam = orbit((0, 0), 170, 0.35, 70, f=300, look_height=0)
c = canvas()
N = 10
t0 = time.time()
for i in range(N):
    world.draw_backdrop(c, cam, i * 0.1)
print('backdrop', round((time.time() - t0) / N * 1000, 1), 'ms')
t0 = time.time()
for i in range(N):
    sky_dome(c, cam, world.sky, span=0.75, below=world.fog)
print('sky_dome', round((time.time() - t0) / N * 1000, 1), 'ms')
t0 = time.time()
for i in range(N):
    world.clouds.draw(c, cam, drift=i)
print('clouds', round((time.time() - t0) / N * 1000, 1), 'ms')
t0 = time.time()
for i in range(N):
    world.draw_ground(c, cam)
print('ground', round((time.time() - t0) / N * 1000, 1), 'ms')
t0 = time.time()
for i in range(N):
    draw_world(c, cam, world.tree_items())
print('trees', round((time.time() - t0) / N * 1000, 1), 'ms')
cam2 = look_at((-20, 4, -70), (0, 14, 0), f=260)
t0 = time.time()
for i in range(N):
    world.draw_backdrop(c, cam2, i * 0.1)
    world.draw_ground(c, cam2)
    draw_world(c, cam2, world.tree_items())
print('low shot total', round((time.time() - t0) / N * 1000, 1), 'ms')
t0 = time.time()
for i in range(N):
    draw_codex_3d(c, cam2, pix, cx, 10, 0, 5, i % 8)
print('codex', round((time.time() - t0) / N * 1000, 1), 'ms')
t0 = time.time()
for i in range(N):
    vgradient(c, 0, H, [hexc('#241640'), hexc('#6e2a73'), hexc('#e8645f')])
print('vgradient', round((time.time() - t0) / N * 1000, 1), 'ms')
