import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw
import render
from px import W, H, FPS
names = ['s4_race', 's5_rain', 's6_montage', 's7_sunset']
entries, total = render.build([e for e in render.TIMELINE if e[0] in names])
starts = {e['name']: e['start'] for e in entries}
req = [('s5_rain', x) for x in (0.0, 0.2, 0.4, 0.6)] + [('s6_montage', x) for x in (0.0, 0.17, 0.33, 0.5)] + \
      [('s7_sunset', x) for x in (0.0, 0.2, 0.4, 0.6)]
frames, labels = [], []
for nm, off in req:
    i = int(round((starts[nm] + off) * FPS))
    frames.append(render.frame_at(i, names))
    labels.append(f'{nm}+{off:.2f}')
render.sheet(frames, labels, 4, 1, sys.argv[1])
