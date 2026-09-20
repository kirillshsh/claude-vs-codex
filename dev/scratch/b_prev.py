import sys
sys.path.insert(0, '.')
import numpy as np
from PIL import Image
from px import *
from chars import load
cl, cx = load()
# RacingCar frames and Laptop frames strip
def strip(name, per_row=12, k=4):
    fr = cl.frames[name]
    h, w = fr[0].shape[:2]
    n = len(fr)
    rows = (n + per_row - 1) // per_row
    img = np.zeros((rows * (h + 2), per_row * (w + 2), 4), np.uint8)
    img[:, :, :3] = (60, 60, 70)
    img[:, :, 3] = 255
    for i, f in enumerate(fr):
        r, c = divmod(i, per_row)
        y, x = r * (h + 2), c * (w + 2)
        a = f[:, :, 3:4] / 255.0
        img[y:y + h, x:x + w, :3] = (img[y:y + h, x:x + w, :3] * (1 - a) + f[:, :, :3] * a).astype(np.uint8)
    return Image.fromarray(img).resize((img.shape[1] * k, img.shape[0] * k), Image.NEAREST)
strip('RacingCar').save('qa/b_prev_racing.png')
strip('Laptop').save('qa/b_prev_laptop.png')
print(cl.anchor['RacingCar'], cl.anchor['Laptop'])
