"""Доля энергии по полосам для каждого стема музыки."""
import json
import os
import sys

import numpy as np

ROOT = '/path/to/claude-vs-codex'
sys.path.insert(0, os.path.join(ROOT, 'audio'))
from synth import SR, load_wav  # noqa: E402

BANDS = [(6000, 24000, '>6k'), (2500, 6000, '2.5-6k'), (1500, 4000, '1.5-4k')]


def shares(x):
    m = x.mean(1) if x.ndim == 2 else x
    X = np.abs(np.fft.rfft(m * np.hanning(len(m))))
    f = np.fft.rfftfreq(len(m), 1 / SR)
    P = X ** 2
    out = {}
    for lo, hi, nm in BANDS:
        sel = (f >= lo) & (f < hi)
        out[nm] = (100 * P[sel].sum() / (P.sum() + 1e-30),
                   100 * X[sel].sum() / (X.sum() + 1e-30))
    return out


def main():
    man = json.load(open(os.path.join(ROOT, 'audio/cues/music.json')))['stems']
    tag = sys.argv[1] if len(sys.argv) > 1 else ''
    rows = {}
    print(f'{"стем":14s} ' + ' '.join(f'{nm:>16s}' for _, _, nm in BANDS) + '   (энергия% / магнитуда%)')
    for s in man:
        x = load_wav(os.path.join(ROOT, s['file']))
        sh = shares(x)
        nm = os.path.basename(s['file'])[:-4]
        rows[nm] = {k: v[0] for k, v in sh.items()}
        line = f'{nm:14s} ' + ' '.join(f'{sh[n][0]:6.1f} /{sh[n][1]:6.1f}  ' for _, _, n in BANDS)
        bad = []
        if sh['>6k'][0] > 12:
            bad.append('>6k!')
        if sh['2.5-6k'][0] > 18:
            bad.append('2.5-6k!')
        print(line + ('  ' + ' '.join(bad) if bad else '  ok'))
    if tag:
        json.dump(rows, open(f'/path/to/scratchpad/bands_{tag}.json', 'w'))
    return rows


if __name__ == '__main__':
    main()
