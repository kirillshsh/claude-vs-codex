import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from px import *
from scenes._a_helpers import *

w = get_world('sunset')
# профиль гор: верхняя непрозрачная строка по колонкам
a = w.mount.img[:, :, 3] > 0
top = np.where(a.any(0), a.argmax(0), a.shape[0])
pw = a.shape[1]
print('mount pano', w.mount.img.shape, 'min top', top.min(), 'max', top.max())
# найти окно шириной 60 колонок с максимальным средним top (т.е. низкие горы)
k = 70
cs = np.convolve(np.concatenate([top, top[:k]]), np.ones(k) / k, 'valid')[:pw]
best = np.argsort(-cs)[:5]
print('valleys (col start, mean top):', [(int(b), round(float(cs[b]), 1)) for b in best])

c = canvas((40, 30, 60))
C1 = title_sprite('CLAWD', hexc('#D97757'), hexc('#F4A98A'), hexc('#B45A3C'), hexc('#6E2E1C'), hexc('#1E0C08'), scale=3, depth=1)
C2 = title_sprite('CODEX', hexc('#4A6CF0'), hexc('#8DA6FF'), hexc('#3050C8'), hexc('#1C2A78'), hexc('#080C2A'), scale=3, depth=1)
VS = title_sprite('VS', hexc('#FFD23C'), hexc('#FFF4A0'), hexc('#FF8A24'), hexc('#A02818'), hexc('#200808'), scale=5, depth=1)
print(C1.shape, C2.shape, VS.shape)
blit(c, C1, 20, 20)
blit(c, C2, 250, 20)
blit(c, VS, 200, 80)
for i, sz in enumerate([8, 10, 16]):
    s = text_sprite('КТО ЖЕ ЛУЧШЕ?', (255, 240, 220), font='small', size=sz, outline=(30, 12, 40))
    blit(c, s, 20, 150 + i * 30)
    print('small', sz, s.shape)
s = text_sprite('КТО ЖЕ ЛУЧШЕ?', (255, 240, 220), font='small', size=8, scale=2, outline=(30, 12, 40))
blit(c, s, 200, 150)
s = text_sprite('КТО ЖЕ ЛУЧШЕ?', (255, 240, 220), font='big', outline=(30, 12, 40), shadow=(30, 12, 40))
blit(c, s, 200, 200)
blit(c, shine_sweep(C1, 0.5, width=6), 20, 60)
Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)).resize((W * 3, H * 3), Image.NEAREST).save(sys.argv[1])
