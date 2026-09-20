import time, sys
sys.path.insert(0, '.')
import numpy as np
from PIL import Image
from px import *
from cam3d import *
from worlds import MeadowWorld
from scenes._b_helpers import *
world = MeadowWorld('day')
fg = FastGround(mips=world.ground.mips, tpu=2.0, origin=world.ground.origin)
cam = look_at((-20, 6, -70), (0, 10, 0), f=260)
a = canvas(); b = canvas()
sky_dome(a, cam, world.sky, span=0.75, below=world.fog)
sky_fast(b, cam, world.sky, span=0.75, below=world.fog)
print('sky diff', np.abs(a - b).max())
world.draw_ground(a, cam)
fg.render(b, cam, fog=world.fog)
d = np.abs(a - b).max(-1)
print('ground diff px>1:', (d > 1).mean())
def tm(name, fn, n=20):
    t0 = time.time()
    for i in range(n): fn()
    print(f'{name}: {(time.time() - t0) / n * 1000:.2f} ms')
tm('sky_fast', lambda: sky_fast(b, cam, world.sky, 0.75, world.fog))
tm('fastground', lambda: fg.render(b, cam, fog=world.fog))
cam2 = orbit((0, 0), 170, 0.35, 70, f=300, look_height=0)
tm('fastground high', lambda: fg.render(b, cam2, fog=world.fog))
tm('pano clouds', lambda: pano_draw(world.clouds, b, cam, drift=3, y_offset=-8))
tm('quad', lambda: draw_quad(b, cam, (-10, 20, 0), (20, 0, 0), (0, -20, 0), color=(255, 0, 0)))
draw_quad(b, cam, (-10, 20, 0), (20, 0, 0), (0, -20, 0), color=(255, 0, 0))
draw_box(b, cam, 20, 40, 0, 8, -10, 10, dict(top=(200,200,220), front=(120,120,150), left=(90,90,120), right=(90,90,120), back=(60,60,80)), edge=(255,140,60))
Image.fromarray(np.clip(np.concatenate([a, b], 1), 0, 255).astype(np.uint8)).resize((W * 4, H * 2), Image.NEAREST).save('qa/b_t1.png')
