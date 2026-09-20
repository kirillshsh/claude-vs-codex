"""Замер скорости сцен относительно эталонного кадра движка (устойчив к общей загрузке машины)."""
import sys, os, time, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from px import *
from cam3d import *
from scenes._a_helpers import get_world

REF_MS = 32.3  # эталон: низкий кадр test_3d без нагрузки (backdrop + ground + trees)
w = get_world('day')
cam = look_at((-20, 4, -70), (0, 14, 0), f=260)


def ref():
    c = canvas()
    w.draw_backdrop(c, cam, 0.5)
    w.draw_ground(c, cam)
    draw_world(c, cam, w.tree_items())


ref()
for name in ['s0_title', 's1_meadow', 's2_argue']:
    S = importlib.import_module('scenes.' + name).SCENE
    ts = [i / 30 for i in range(0, int(S.dur * 30), 2)]
    rs, ss = 0.0, 0.0
    ratios = []
    for t in ts:
        a = time.perf_counter(); ref(); b = time.perf_counter(); S.render(t); e = time.perf_counter()
        rs += b - a
        ss += e - b
        ratios.append((e - b) / (b - a))
    ratios.sort()
    avg = ss / rs * REF_MS
    p95 = ratios[int(len(ratios) * 0.95)] * REF_MS
    print(f'{name}: ~{avg:.0f} ms/кадр в среднем, p95 ~{p95:.0f} ms (по {len(ts)} кадрам, нормировано на эталон)')
