import sys, numpy as np
sys.path.insert(0, '.')
import render
from PIL import Image, ImageDraw
step = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
out = sys.argv[2] if len(sys.argv) > 2 else 'qa/global_sheet.png'
ts = np.arange(0, 90, step)
frames = [render.frame_at(int(round(t * 30))) for t in ts]
cols = 10
rows = (len(frames) + cols - 1) // cols
W, H = 480, 270
img = Image.new('RGB', (cols * (W // 2 + 4), rows * (H // 2 + 12)), (8, 8, 10))
d = ImageDraw.Draw(img)
for k, (f, t) in enumerate(zip(frames, ts)):
    r, c = divmod(k, cols)
    x, y = c * (W // 2 + 4), r * (H // 2 + 12)
    img.paste(Image.fromarray(f).resize((W // 2, H // 2), Image.NEAREST), (x, y + 10))
    d.text((x + 2, y), f'{t:.0f}s', fill=(255, 220, 120))
img.save(out)
print(out, img.size)
