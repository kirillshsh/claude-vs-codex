import sys, numpy as np
from PIL import Image
sys.path.insert(0, '/path/to/claude-vs-codex')
import render
from px import scale_nn
times = [float(x) for x in sys.argv[1].split(',')]
x0, y0, x1, y1 = [int(v) for v in sys.argv[2].split(',')]
k = int(sys.argv[3])
out = sys.argv[4]
tiles = []
for t in times:
    f = render.scene_frame('s5_rain', t)
    tiles.append(scale_nn(f[y0:y1, x0:x1], k))
img = np.concatenate(tiles, 1)
Image.fromarray(img).save(out)
print(out, img.shape)
