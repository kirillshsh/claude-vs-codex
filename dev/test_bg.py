import time
from px import *
from bg import *
from chars import load
from PIL import Image
cl, cx = load()
t0 = time.time()
out = []
for mode in ['day', 'storm', 'golden']:
    m = Meadow(mode, width=W, ground_y=214, seed=7)
    c = canvas()
    m.draw(c, t=3.0)
    blit(c, PEDESTAL, 240 - 13, 214 - 16)
    blit(c, TROPHY, 240 - 8, 214 - 16 - 18)
    glint(c, 236, 200, 0.05)
    cl.draw_face(c, 'idle' if mode == 'day' else ('sad' if mode == 'storm' else 'happy'), 170, 214, k=2)
    cx.draw(c, 0, 0, 310, 214, 1)
    if mode == 'storm':
        rain(c, 3.0, ground_y=214)
        blit(c, LEAF, 290, 150)
    if mode == 'golden':
        rainbow(c, 240, 190, 120, alpha=0.8, clip_y=172)
        hearts(c, 1.0, 240, 170, n=6)
        blit(c, make_hoverboard(0.1), 360, 230)
    m.grass_front(c, 3.0)
    out.append(c)
print('time', time.time() - t0)
big = np.concatenate(out, 0)
Image.fromarray(np.clip(big, 0, 255).astype(np.uint8)).resize((W * 3, H * 9), Image.NEAREST).save('test_bg.png')
