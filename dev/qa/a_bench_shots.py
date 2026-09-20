import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from px import *
from cam3d import *
from scenes._a_helpers import get_world
import cProfile, pstats

REF_MS = 32.3
w = get_world('day')
cam = look_at((-20, 4, -70), (0, 14, 0), f=260)


def ref():
    c = canvas()
    w.draw_backdrop(c, cam, 0.5)
    w.draw_ground(c, cam)
    draw_world(c, cam, w.tree_items())


ref()
from scenes.s2_argue import SCENE as S, SHOTS
ends = [s for s, _ in SHOTS[1:]] + [12.0]
for (st, nm), en in zip(SHOTS, ends):
    ts = [st + i / 15 for i in range(int((en - st) * 15))]
    rs = ss = 0
    for t in ts:
        a = time.perf_counter(); ref(); b = time.perf_counter(); S.render(t); e = time.perf_counter()
        rs += b - a
        ss += e - b
    print(nm, round(ss / rs * REF_MS), 'ms')
which = sys.argv[1] if len(sys.argv) > 1 else 'brawl'
t0 = dict((n, s) for s, n in SHOTS)[which]
pr = cProfile.Profile()
pr.enable()
for i in range(10):
    S.render(t0 + 0.4 + i * 0.1)
pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(10)
