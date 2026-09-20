"""Сборка мультика: таймлайн сцен + переходы, QA-листы кадров, рендер видео (параллельно) -> ffmpeg.

  python render.py sheet  --scene s1_meadow --start 0 --end 9 --step 0.5 [--cols 6] [--scale 2] --out qa/s1.png
  python render.py stills --scene s1_meadow --times 0,1.2,3.4 [--scale 4] --out qa/s1_stills.png
  python render.py video  [--scenes s1_meadow,s2_argue] [--workers 8] --out out/cartoon.mp4
  python render.py timing
"""
import argparse
import importlib
import math
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from px import W, H, FPS, canvas, iris, fade, dissolve, text_sprite, blit, clamp  # noqa: E402

# (модуль сцены, плановая длительность, переход-вход, переход-выход)
TIMELINE = [
    ('s0_title', 4.0, ('fade', 0.4), ('iris', 0.35)),
    ('s1_meadow', 9.0, ('iris', 0.4), ('cut', 0.0)),
    ('s2_argue', 12.0, ('cut', 0.0), ('flash', 0.18)),
    ('s3_code', 9.0, ('flash', 0.18), ('flash', 0.18)),
    ('s4_race', 14.0, ('flash', 0.18), ('fade', 0.3)),
    ('s5_rain', 17.0, ('fade', 0.35), ('dissolve', 0.0)),
    ('s6_montage', 7.0, ('dissolve', 0.5), ('dissolve', 0.0)),
    ('s7_sunset', 18.0, ('dissolve', 0.6), ('fade', 1.0)),
]

_scenes = {}


class Placeholder:
    def __init__(self, name, dur):
        self.name, self.dur = name, dur

    def render(self, t):
        c = canvas((20, 22, 30))
        s = text_sprite(f'{self.name}  {t:4.1f}s', (200, 200, 220))
        blit(c, s, W // 2 - s.shape[1] // 2, H // 2 - 4)
        return c


def get_scene(name, dur=5.0):
    if name not in _scenes:
        try:
            mod = importlib.import_module('scenes.' + name)
            _scenes[name] = mod.SCENE
        except ModuleNotFoundError as e:
            if e.name and e.name.endswith(name):
                _scenes[name] = Placeholder(name, dur)
            else:
                raise
    return _scenes[name]


def build(timeline):
    out, t0 = [], 0.0
    for (name, dur, tin, tout) in timeline:
        sc = get_scene(name, dur)
        d = float(getattr(sc, 'dur', dur))
        out.append(dict(name=name, start=t0, dur=d, tin=tin, tout=tout))
        t0 += d
    return out, t0


def apply_transitions(c, entry, t, prev_entry):
    kind, d = entry['tin']
    if d > 0 and t < d:
        p = clamp(t / d)
        if kind == 'fade':
            fade(c, (0, 0, 0), 1 - p)
        elif kind == 'iris':
            iris(c, W / 2, H / 2, p * math.hypot(W, H) / 2)
        elif kind == 'flash':
            c[:] = c * p + 255 * (1 - p)
        elif kind == 'dissolve' and prev_entry is not None:
            prev = get_scene(prev_entry['name']).render(prev_entry['dur'] - 1.0 / FPS)
            c[:] = dissolve(prev, c, p)
    kind, d = entry['tout']
    if d > 0 and t > entry['dur'] - d:
        p = clamp((t - (entry['dur'] - d)) / d)
        if kind == 'fade':
            fade(c, (0, 0, 0), p)
        elif kind == 'iris':
            iris(c, W / 2, H / 2, (1 - p) * math.hypot(W, H) / 2)
        elif kind == 'flash':
            c[:] = c * (1 - p) + 255 * p
    return c


_built = None


def frame_at(i, names=None):
    global _built
    if _built is None:
        tl = [e for e in TIMELINE if names is None or e[0] in names]
        _built = build(tl)
    entries, total = _built
    T = i / FPS
    for k, e in enumerate(entries):
        if T < e['start'] + e['dur'] or k == len(entries) - 1:
            t = min(T - e['start'], e['dur'] - 1e-6)
            c = get_scene(e['name']).render(t)
            c = apply_transitions(c, e, t, entries[k - 1] if k > 0 else None)
            return np.clip(c, 0, 255).astype(np.uint8)


def _worker_init(names):
    global _NAMES
    _NAMES = names


def _render_idx(i):
    return frame_at(i, _NAMES).tobytes()


def cmd_video(args):
    names = args.scenes.split(',') if args.scenes else None
    tl = [e for e in TIMELINE if names is None or e[0] in names]
    entries, total = build(tl)
    n = int(round(total * FPS))
    if args.frames:
        a, b = map(int, args.frames.split(':'))
        rng = range(a, min(b, n))
    else:
        rng = range(n)
    print(f'frames {len(rng)}  duration {total:.2f}s', flush=True)
    for e in entries:
        print(f"  {e['name']:12s} {e['start']:6.2f} +{e['dur']:.2f}")
    scale = args.scale
    vf = f'scale={W * scale}:{H * scale}:flags=neighbor,format=yuv420p'
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
           '-i', '-', '-vf', vf, '-c:v', 'libx264', '-preset', args.preset, '-crf', str(args.crf), '-tune', 'animation',
           '-movflags', '+faststart', args.out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers, initializer=_worker_init, initargs=(names,)) as pool:
            for k, buf in enumerate(pool.imap(_render_idx, rng, chunksize=6)):
                ff.stdin.write(buf)
                if k % 150 == 0:
                    print(f'  {k}/{len(rng)}  {time.time() - t0:.0f}s', flush=True)
    else:
        _worker_init(names)
        for k, i in enumerate(rng):
            ff.stdin.write(_render_idx(i))
            if k % 150 == 0:
                print(f'  {k}/{len(rng)}  {time.time() - t0:.0f}s', flush=True)
    ff.stdin.close()
    ff.wait()
    print(f'done {args.out} in {time.time() - t0:.0f}s')


def sheet(frames, labels, cols, scale, out):
    fw, fh = W * scale, H * scale
    rows = math.ceil(len(frames) / cols)
    img = Image.new('RGB', (cols * (fw + 4) + 4, rows * (fh + 16) + 4), (10, 10, 12))
    d = ImageDraw.Draw(img)
    for k, (f, lab) in enumerate(zip(frames, labels)):
        r, c = divmod(k, cols)
        x, y = 4 + c * (fw + 4), 4 + r * (fh + 16)
        img.paste(Image.fromarray(f).resize((fw, fh), Image.NEAREST), (x, y + 12))
        d.text((x, y), lab, fill=(255, 230, 120))
    img.save(out)
    print(out, img.size)


def scene_frame(name, t):
    sc = get_scene(name)
    c = sc.render(min(max(t, 0), sc.dur - 1e-6))
    return np.clip(c, 0, 255).astype(np.uint8)


def cmd_sheet(args):
    sc = get_scene(args.scene)
    end = min(args.end, sc.dur) if args.end is not None else sc.dur
    ts = list(np.arange(args.start, end + 1e-9, args.step))
    t0 = time.time()
    frames = [scene_frame(args.scene, t) for t in ts]
    print(f'{len(ts)} frames in {time.time() - t0:.2f}s ({(time.time() - t0) / max(1, len(ts)) * 1000:.0f} ms/frame)')
    sheet(frames, [f'{args.scene} t={t:.2f}' for t in ts], args.cols, args.scale, args.out)


def cmd_stills(args):
    ts = [float(x) for x in args.times.split(',')]
    frames = [scene_frame(args.scene, t) for t in ts]
    sheet(frames, [f'{args.scene} t={t:.2f}' for t in ts], args.cols or min(len(ts), 2), args.scale, args.out)


def cmd_timing(args):
    entries, total = build(TIMELINE)
    for e in entries:
        print(f"{e['name']:12s} start {e['start']:6.2f} dur {e['dur']:5.2f}  in {e['tin']} out {e['tout']}")
    print('total', total)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')
    v = sub.add_parser('video')
    v.add_argument('--out', default='out/cartoon.mp4')
    v.add_argument('--scenes')
    v.add_argument('--frames')
    v.add_argument('--workers', type=int, default=8)
    v.add_argument('--scale', type=int, default=4)
    v.add_argument('--crf', type=int, default=16)
    v.add_argument('--preset', default='slow')
    s = sub.add_parser('sheet')
    s.add_argument('--scene', required=True)
    s.add_argument('--start', type=float, default=0)
    s.add_argument('--end', type=float)
    s.add_argument('--step', type=float, default=0.5)
    s.add_argument('--cols', type=int, default=6)
    s.add_argument('--scale', type=int, default=1)
    s.add_argument('--out', required=True)
    st = sub.add_parser('stills')
    st.add_argument('--scene', required=True)
    st.add_argument('--times', required=True)
    st.add_argument('--cols', type=int)
    st.add_argument('--scale', type=int, default=2)
    st.add_argument('--out', required=True)
    sub.add_parser('timing')
    a = ap.parse_args()
    {'video': cmd_video, 'sheet': cmd_sheet, 'stills': cmd_stills, 'timing': cmd_timing}[a.cmd](a)
