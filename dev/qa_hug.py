# варианты объятия без самодельной руки: обычный сидячий Codex / официальные кадры с раскинутыми руками поверх Clawd
import sys, numpy as np
from PIL import Image
sys.path.insert(0, '.')
import scenes.s7_sunset as m
from px import *
S = m.SCENE
rows = []
for (row, col, top) in [(0, 0, False), (4, 2, True), (8, 2, True)]:
    cam = m.look_at((2, 10.5, -54), (0, 10, 60), 256)
    a = S.shot_back(4.8, cam, 0.0, 0.055, -13.0, 5.5, 3.0 if row == 0 else 0.0, 1.4, 1.0, codex_row=row, codex_col=col, codex_on_top=top)
    camf = m.look_at((-1, 8.5, 35), (-1, 11.5, -5), 330)
    c = m.canvas()
    rays = camf.rays()
    S.sky(c, camf, 0.15, rays)
    beyond, GX, GZ, gt = S.ground_hit(camf, rays)
    S.g_front.render(c, camf, fog=hexc('#b08a8a'), fog_near=120, fog_far=700, mask_out=beyond)
    S.cliff_face(c, camf, rays, ~beyond, lit=True)
    S.draw_pair_front(c, camf, 9.8, 11.0, -5.5, 'love', 'heart' if row == 0 else 'heart', 3.0 if row == 0 else 0.0, 1.4, 1.0,
                      codex_row=row, codex_col=col, codex_on_top=top)
    rows.append(np.concatenate([a, c], 1))
big = np.concatenate(rows, 0)
Image.fromarray(np.clip(big, 0, 255).astype(np.uint8)).resize((big.shape[1] * 2, big.shape[0] * 2), Image.NEAREST).save('qa/d_hug_variants.png')
print('ok')
