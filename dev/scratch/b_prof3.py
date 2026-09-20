import sys, time, cProfile, pstats
sys.path.insert(0, '.')
from scenes.s3_code import SCENE
import numpy as np
for t in (0.5, 2.0, 3.6, 4.8, 6.5, 8.4):
    t0 = time.time()
    for i in range(5):
        SCENE.render(t + i * 0.01)
    print(t, f'{(time.time() - t0) / 5 * 1000:.1f} ms')
pr = cProfile.Profile()
pr.enable()
for i in range(10):
    SCENE.render(2.0 + i * 0.03)
    SCENE.render(6.5 + i * 0.03)
pr.disable()
pstats.Stats(pr).sort_stats('cumtime').print_stats(25)
