"""Подбор стартового угла облёта в s1: ищем путь камеры, который проходит вплотную к деревьям (параллакс)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from px import *
from scenes._a_helpers import get_world

w = get_world('day')
trees = np.array([(X, Z, sc) for (X, Z, spr, sc) in w.trees])
best = []
for a0_deg in range(-190, -40, 5):
    a0 = math.radians(a0_deg)
    a1 = math.radians(-6)
    near = []
    for i in range(65):
        v = i / 64
        ang = a0 + (a1 - a0) * (0.82 * v + 0.18 * (v * v * (3 - 2 * v)))
        r = lerp(235.0, 56.0, 0.55 * v + 0.45 * (1 - (1 - v) ** 3))
        ex, ez = -math.sin(ang) * r, -math.cos(ang) * r
        d = np.hypot(trees[:, 0] - ex, trees[:, 1] - ez)
        near.append((d.min(), v))
    near.sort()
    # сколько деревьев прошло ближе 40 юнитов и когда
    passes = sorted(set(round(v, 2) for (dd, v) in near if dd < 42))
    best.append((min(d for d, _ in near), len(passes), a0_deg, passes[:6]))
best.sort(key=lambda x: (x[0]))
for b in best[:10]:
    print('мин. дистанция %.0f, близких моментов %d, старт %d°, моменты %s' % b)
