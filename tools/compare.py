"""Покадровое сравнение двух роликов: python3 tools/compare.py a.mp4 b.mp4 [шаг_сек]
Кадры берутся в 480x270 (nearest), печатается средняя абсолютная разница по сценам и худшие моменты."""
import subprocess, sys
import numpy as np

A, B = sys.argv[1], sys.argv[2]
STEP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
W, H = 480, 270


def frames(path):
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-vf', f'fps={1 / STEP},scale={W}:{H}:flags=neighbor',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.int16)


a, b = frames(A), frames(B)
n = min(len(a), len(b))
d = np.abs(a[:n] - b[:n]).mean(axis=(1, 2, 3))
print(f'кадров {len(a)} / {len(b)}, сравнено {n}; средняя разница {d.mean():.2f} из 255, медиана {np.median(d):.2f}')
SC = [('s0_title', 0, 4), ('s1_meadow', 4, 13), ('s2_argue', 13, 25), ('s3_code', 25, 34), ('s4_race', 34, 48),
      ('s5_rain', 48, 65), ('s6_montage', 65, 72), ('s7_sunset', 72, 93)]
for name, t0, t1 in SC:
    seg = d[int(t0 / STEP):int(t1 / STEP)]
    if len(seg):
        print(f'  {name:11s} среднее {seg.mean():6.2f}  макс {seg.max():6.2f}')
worst = np.argsort(d)[-5:][::-1]
print('худшие моменты:', ', '.join(f'{i * STEP:.1f}s={d[i]:.1f}' for i in worst))
