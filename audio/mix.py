"""Сведение всех стемов в мастер и мукс с видео.

    python3 audio/mix.py                      # собрать audio/master.wav из audio/cues/*.json
    python3 audio/mix.py --mux out/claude_v2.mp4 --out out/claude_vs_codex_sound.mp4

Манифест audio/cues/<кто>.json:
  {"stems": [{"file": "audio/stems/music/s2.wav", "t": 13.0, "gain": 1.0,
              "pan": 0.0, "fin": 0.05, "fout": 0.3, "bus": "music"}]}
Пути к файлам — от корня проекта (cartoon/) либо абсолютные.

Мастер-шина: шины сводятся раздельно, музыка и атмосфера приседают под громкие sfx
(сайдчейн), затем мягкое снятие «стеклянного» верха и лимитер без искажений.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import SR, load_wav, save_wav, n_samples, _fft_filter, peak_eq  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUR = 93.0
BUS_GAIN = {'music': 0.82, 'sfx': 0.80, 'amb': 0.50, 'ui': 0.62}
DUCK_DB = {'music': -4.5, 'amb': -6.0}   # насколько шина приседает под громкий sfx


def build(dur=DUR, cue_dir=None, verbose=True):
    """Возвращает {шина: стерео-буфер}."""
    cue_dir = cue_dir or os.path.join(ROOT, 'audio', 'cues')
    n = n_samples(dur)
    buses = {}
    total = 0
    for cf in sorted(glob.glob(os.path.join(cue_dir, '*.json'))):
        with open(cf) as f:
            man = json.load(f)
        for s in man.get('stems', []):
            p = s['file'] if os.path.isabs(s['file']) else os.path.join(ROOT, s['file'])
            if not os.path.exists(p):
                print(f'  !! нет файла {p}')
                continue
            bus = s.get('bus', 'sfx')
            x = load_wav(p) * float(s.get('gain', 1.0)) * BUS_GAIN.get(bus, 1.0)
            pan = float(s.get('pan', 0.0))
            if pan:
                x = x * np.array([np.sqrt((1 - pan) / 2) * np.sqrt(2),
                                  np.sqrt((1 + pan) / 2) * np.sqrt(2)], np.float32)
            fin, fout = float(s.get('fin', 0)), float(s.get('fout', 0))
            if fin:
                k = min(n_samples(fin), len(x))
                x[:k] *= np.linspace(0, 1, k, dtype=np.float32)[:, None]
            if fout:
                k = min(n_samples(fout), len(x))
                x[-k:] *= np.linspace(1, 0, k, dtype=np.float32)[:, None]
            i = int(round(float(s['t']) * SR))
            if i < 0:
                x, i = x[-i:], 0
            x = x[:max(0, n - i)]
            if len(x) == 0:
                continue
            buses.setdefault(bus, np.zeros((n, 2), np.float32))[i:i + len(x)] += x
            total += 1
        if verbose:
            print(f'  {os.path.basename(cf)}: {len(man.get("stems", []))} стемов')
    if verbose:
        for b, v in buses.items():
            print(f'  шина {b:6s} пик {np.max(np.abs(v)):.2f}')
        print(f'  всего {total} стемов')
    return buses


def _env(x, atk=0.008, rel=0.32):
    """Огибающая по пику с разной скоростью атаки/восстановления (блоками)."""
    B = 256
    m = np.abs(x).max(1)
    k = int(np.ceil(len(m) / B))
    blocks = np.zeros(k, np.float32)
    for i in range(k):
        blocks[i] = m[i * B:(i + 1) * B].max(initial=0.0)
    out = np.zeros(k, np.float32)
    ca, cr = np.exp(-B / (atk * SR)), np.exp(-B / (rel * SR))
    v = 0.0
    for i, b in enumerate(blocks):
        c = ca if b > v else cr
        v = b + (v - b) * c
        out[i] = v
    return np.repeat(out, B)[:len(m)]


def duck(target, trigger, depth_db=-4.5, thresh=0.05):
    """Приседание target под trigger (сайдчейн)."""
    e = _env(trigger)
    k = np.clip((e - thresh) / 0.28, 0, 1)
    g = 10 ** (depth_db / 20 * k)
    return (target * g[:, None]).astype(np.float32)


def master(buses, deharsh_db=-1.5, air_cut=12000):
    sfx = sum((buses[b] for b in ('sfx', 'ui') if b in buses),
              np.zeros((len(next(iter(buses.values()))), 2), np.float32))
    out = np.zeros_like(sfx)
    for b, x in buses.items():
        if b in DUCK_DB:
            x = duck(x, sfx, DUCK_DB[b])
        out += x
    # снять «стекло»: провал в 2.5–6 кГц и мягкий срез самого верха
    for ch in range(2):
        y = peak_eq(out[:, ch], 3600.0, deharsh_db, q=0.55)
        out[:, ch] = _fft_filter(y, lambda f: 1 / np.sqrt(1 + (f / air_cut) ** 4))
    return out


def limit(x, ceil=0.94, knee=0.70):
    """Пик к потолку без искажений, мягкое насыщение только над коленом."""
    m = float(np.max(np.abs(x))) or 1.0
    x = (x * (ceil / m)).astype(np.float32) if m > ceil else x.copy()
    over = np.abs(x) > knee
    if over.any():
        s = np.sign(x[over])
        e = (np.abs(x[over]) - knee) / (1.0 - knee)
        x[over] = s * (knee + (1.0 - knee) * np.tanh(e * 1.3) / np.tanh(1.3))
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dur', type=float, default=DUR)
    ap.add_argument('--cues')
    ap.add_argument('--wav', default=os.path.join(ROOT, 'audio', 'master.wav'))
    ap.add_argument('--mux')
    ap.add_argument('--out')
    ap.add_argument('--lufs', type=float, default=-16.0)
    ap.add_argument('--raw', action='store_true', help='без мастер-обработки')
    a = ap.parse_args()
    buses = build(a.dur, a.cues)
    buf = sum(buses.values()) if a.raw else master(buses)
    buf = limit(buf)
    save_wav(a.wav, buf)
    print('wav', a.wav, f'{len(buf) / SR:.2f}s  пик {np.max(np.abs(buf)):.2f}')
    if a.mux:
        out = a.out or a.mux.replace('.mp4', '_sound.mp4')
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', a.mux, '-i', a.wav,
               '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
               '-af', f'loudnorm=I={a.lufs}:TP=-1.5:LRA=11,aresample=48000', '-ar', '48000',
               '-c:a', 'aac', '-b:a', '256k', '-movflags', '+faststart', '-shortest', out]
        subprocess.run(cmd, check=True)
        print('mux', out)


if __name__ == '__main__':
    main()
