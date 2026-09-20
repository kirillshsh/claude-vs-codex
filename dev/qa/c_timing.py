import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
name = sys.argv[1]
t0 = time.process_time(); w0 = time.time()
mod = importlib.import_module('scenes.' + name)
sc = mod.SCENE
print(f'init cpu {time.process_time()-t0:.2f}s wall {time.time()-w0:.2f}s')
import numpy as np
ts = np.arange(0, sc.dur, 1/30.0)[::3]
# прогрев (ленивые миры/кэши)
for t in ts[::10]:
    sc.render(t)
res = {}
for t in ts:
    a = time.process_time()
    sc.render(t)
    res[t] = (time.process_time() - a) * 1000
v = np.array(list(res.values()))
print(f'{name}: mean {v.mean():.0f} ms  p90 {np.percentile(v,90):.0f}  max {v.max():.0f} ms  (cpu)')
bins = {}
for t, ms in res.items():
    b = int(t)
    bins.setdefault(b, []).append(ms)
print('  per-second mean:', ' '.join(f'{b}:{np.mean(x):.0f}' for b, x in sorted(bins.items())))
