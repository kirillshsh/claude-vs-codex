"""Численная проверка стемов (слушать некому — смотрим на цифры).

    python3 audio/qa.py audio/cues/music.json [--start 0 --end 90] [--step 0.5]

Печатает: длительность, пик, клиппинг, огибающую RMS по времени (ASCII),
дыры тишины, спектральный баланс (низ/середина/верх).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import SR, load_wav, n_samples  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def assemble(manifests, dur=90.0):
    buf = np.zeros((n_samples(dur), 2), np.float32)
    rows = []
    for mf in manifests:
        with open(mf) as f:
            man = json.load(f)
        for s in man.get('stems', []):
            p = s['file'] if os.path.isabs(s['file']) else os.path.join(ROOT, s['file'])
            if not os.path.exists(p):
                print(f'!! нет файла {p}')
                continue
            x = load_wav(p) * float(s.get('gain', 1.0))
            i = int(round(float(s['t']) * SR))
            x = x[:max(0, len(buf) - i)]
            if len(x):
                buf[i:i + len(x)] += x
            rows.append((s['t'], s['t'] + len(x) / SR, s.get('bus', 'sfx'), os.path.basename(p),
                         float(np.max(np.abs(x))) if len(x) else 0.0))
    return buf, rows


def db(v):
    return -99.0 if v <= 1e-6 else 20 * np.log10(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifests', nargs='+')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, default=90.0)
    ap.add_argument('--step', type=float, default=0.5)
    ap.add_argument('--dur', type=float, default=90.0)
    a = ap.parse_args()
    buf, rows = assemble(a.manifests, a.dur)
    mono = buf.mean(1)
    print(f'пик {np.max(np.abs(buf)):.3f} ({db(np.max(np.abs(buf))):.1f} dBFS), '
          f'клиппинг сэмплов: {int(np.sum(np.abs(buf) > 0.999))}')
    print(f'стемов: {len(rows)}')
    for t0, t1, bus, name, pk in sorted(rows):
        print(f'  {t0:6.2f}–{t1:6.2f}  {bus:5s} {name:34s} пик {pk:.2f} ({db(pk):5.1f} dB)')
    n = n_samples(a.step)
    print('\nRMS по времени:')
    quiet = []
    for k in range(int(a.start / a.step), int(a.end / a.step)):
        seg = mono[k * n:(k + 1) * n]
        if not len(seg):
            break
        r = float(np.sqrt(np.mean(seg ** 2)))
        d = db(r)
        bar = '#' * int(max(0, (d + 60) / 2))
        t = k * a.step
        print(f'{t:6.2f} {d:6.1f} |{bar}')
        if d < -50:
            quiet.append(t)
    if quiet:
        runs, s = [], quiet[0]
        for i in range(1, len(quiet)):
            if quiet[i] - quiet[i - 1] > a.step * 1.5:
                runs.append((s, quiet[i - 1] + a.step))
                s = quiet[i]
        runs.append((s, quiet[-1] + a.step))
        long = [r for r in runs if r[1] - r[0] >= 1.0]
        if long:
            print('\nтихо (>1 c):', ', '.join(f'{x:.1f}–{y:.1f}' for x, y in long))
    X = np.abs(np.fft.rfft(mono[int(a.start * SR):int(a.end * SR)]))
    f = np.fft.rfftfreq(len(mono[int(a.start * SR):int(a.end * SR)]), 1 / SR)
    tot = X.sum() + 1e-9
    for lo, hi, nm in [(20, 200, 'низ'), (200, 2000, 'середина'), (2000, 20000, 'верх')]:
        print(f'{nm:9s} {100 * X[(f >= lo) & (f < hi)].sum() / tot:5.1f}%')


if __name__ == '__main__':
    main()
