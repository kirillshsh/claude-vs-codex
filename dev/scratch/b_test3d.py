import time
from px import *
from bg import PEDESTAL, TROPHY, glint
from cam3d import *
from worlds import MeadowWorld
from chars import load
from PIL import Image
cl, cx = load()
pix = CodexPixelizer()
t0 = time.time()
world = MeadowWorld('day')
print('world', time.time() - t0)

def shot(cam, t=0.0, clawd_face='idle', codex_face=None, clawd_pos=(-30, 0), codex_pos=(30, 0)):
    c = canvas()
    world.draw_backdrop(c, cam, t)
    world.draw_ground(c, cam)
    items = world.tree_items()
    def ped(d, cm):
        draw_sprite_3d(d, cm, PEDESTAL, 0, 0, 0, 2.0, fog=world.fog)
        draw_sprite_3d(d, cm, TROPHY, 0, 8.0, 0, 2.0, fog=world.fog)
    items.append((0, 0, 0, ped))
    def clawd(d, cm):
        ground_shadow(d, cm, clawd_pos[0], clawd_pos[1], 11)
        draw_clawd_3d(d, cm, cl, clawd_pos[0], clawd_pos[1], face=clawd_face)
    def codex(d, cm):
        ground_shadow(d, cm, codex_pos[0], codex_pos[1], 8)
        draw_codex_3d(d, cm, pix, cx, codex_pos[0], codex_pos[1], face=codex_face)
    items.append((clawd_pos[0], 0, clawd_pos[1], clawd))
    items.append((codex_pos[0], 0, codex_pos[1], codex))
    draw_world(c, cam, items)
    return c

shots = []
t1 = time.time()
shots.append(shot(orbit((0, 0), 170, 0.35, 70, f=300, look_height=0)))            # высокий общий
shots.append(shot(look_at((-20, 4, -70), (0, 14, 0), f=260)))                     # низкий ракурс на кубок
shots.append(shot(look_at((42, 13, 34), (-30, 9, 0), f=340), clawd_face='angry', codex_face='back'))  # через плечо Codex
shots.append(shot(look_at((-44, 12, 30), (30, 12, 0), f=340), clawd_face='back', codex_pos=(30, 0)))  # через плечо Clawd
print('4 shots', time.time() - t1)
big = np.concatenate([np.concatenate(shots[:2], 1), np.concatenate(shots[2:], 1)], 0)
Image.fromarray(np.clip(big, 0, 255).astype(np.uint8)).resize((W * 4, H * 4), Image.NEAREST).save('/path/to/scratchpad/b_test3d.png')
