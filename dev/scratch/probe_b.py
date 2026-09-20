import os, sys, math
ROOT = '/path/to/claude-vs-codex'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
from scenes.s4_race import SCENE as S4, pack_x, T_GO, T_HIT, X_FIN, X_START, BUMP1, BUMP2
from scenes._b_helpers import world_to_screen
from px import W, H

print('X_FIN', X_FIN, 'X_START', X_START)
print('pack_x(6.5)+22 =', pack_x(6.5) + 22.0)

# момент пролёта мимо камеры в shot headon
ex = pack_x(6.5) + 22.0
for who, pos in (('cl', S4.pos_cl), ('cx', S4.pos_cx)):
    lo, hi = 5.6, 7.0
    for _ in range(60):
        m = (lo + hi) / 2
        if pos(m)[0] < ex:
            lo = m
        else:
            hi = m
    print('pass camera', who, round((lo + hi) / 2, 3))

print('\nscreen-x гонщиков по шотам:')
for t in [1.25,1.4,1.6,1.9,2.05,2.2,2.5,2.8,3.0,3.2,3.6,4.0,4.4,4.8,5.2,5.6,5.9,6.2,6.5,6.63,6.8,7.0,7.3,7.47,7.8,8.12,8.4,8.6,9.0,9.4,9.6,10.0,10.3,10.5,10.9,11.3,11.75,12.3,12.6,13.2,13.8]:
    cam, shot = S4.camera(t)
    xc, zc = S4.pos_cl(t)
    xx, zx = S4.pos_cx(t)
    a = world_to_screen(cam, (xc, 6.0, zc))
    b = world_to_screen(cam, (xx, 8.0, zx))
    tx, ty, tz, spin, ph = S4.trophy_state(t)
    c = world_to_screen(cam, (tx, ty, tz))
    f = lambda s: 'None' if s is None else f'({s[0]:6.0f},{s[1]:5.0f})'
    print(f'{t:5.2f} {shot:7s} cl{f(a)} cx{f(b)} v={S4.speed(t):6.1f} trophy {ph:6s} {f(c)}')

print('\ns3:')
from scenes.s3_code import SCENE as S3, T_FIN, PK_T
for t in [1.3,1.6,2.0,2.4,2.8,3.2,3.3,3.6,3.95,4.2,4.6,4.9,5.2,5.5,5.8,6.2,6.6,7.0,7.4,7.8,7.95,8.1,8.4,8.8]:
    cam, shot = S3.camera(t)
    pcl, pcx = S3.prog(t)
    a = world_to_screen(cam, (-52.0, 10.0, 0.0))
    b = world_to_screen(cam, (44.0, 11.0, 0.0))
    f = lambda s: 'None' if s is None else f'({s[0]:6.0f},{s[1]:5.0f})'
    print(f'{t:5.2f} {shot:8s} cl{f(a)} cx{f(b)} prog {pcl:5.1f}/{pcx:5.1f}')
