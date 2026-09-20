import sys, time
sys.path.insert(0, '.')
import importlib
name = sys.argv[1]
ts = [float(x) for x in sys.argv[2].split(',')]
mod = importlib.import_module('scenes.' + name)
S = mod.SCENE
for t in ts:
    S.render(t)  # warm caches
tot = 0
for t in ts:
    c0 = time.process_time()
    for i in range(4):
        S.render(t + i * 0.011)
    dt = (time.process_time() - c0) / 4 * 1000
    tot += dt
    print(f't={t:5.2f}  cpu {dt:6.1f} ms')
print(f'avg cpu {tot / len(ts):.1f} ms')
