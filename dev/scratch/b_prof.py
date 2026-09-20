import time, sys
sys.path.insert(0, '.')
from px import *
from cam3d import *
from worlds import MeadowWorld
t0 = time.time()
world = MeadowWorld('day')
print('world init', time.time() - t0)
cam = look_at((-20, 6, -70), (0, 10, 0), f=260)
c = canvas()
def tm(name, fn, n=10):
    t0 = time.time()
    for i in range(n):
        fn()
    print(f'{name}: {(time.time() - t0) / n * 1000:.1f} ms')
tm('sky_dome', lambda: sky_dome(c, cam, world.sky, span=0.75, below=world.fog))
tm('clouds', lambda: world.clouds.draw(c, cam, drift=3, y_offset=-8))
tm('mount', lambda: world.mount.draw(c, cam, y_offset=1))
tm('hills_far', lambda: world.hills_far.draw(c, cam, y_offset=2))
tm('hills', lambda: world.hills.draw(c, cam, y_offset=3))
tm('ground', lambda: world.draw_ground(c, cam))
items = world.tree_items()
tm('trees', lambda: draw_world(c, cam, items))
cam2 = orbit((0, 0), 170, 0.35, 70, f=300, look_height=0)
tm('ground high', lambda: world.draw_ground(c, cam2))
tm('rays', lambda: cam.rays())
