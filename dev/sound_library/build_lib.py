#!/usr/bin/env python3
"""
Build the sample library: decode -> trim -> analyse -> reject harsh -> normalise -> write.

Reads a job list (JSON) of {src_path, cat, tags, src_url, license, name}
Writes audio/samples/<cat>/<name>.wav and returns per-file analysis records.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SR = 48000
ROOT = "/path/to/claude-vs-codex"
OUT = os.path.join(ROOT, "audio/samples")

# max duration per category (seconds)
MAXDUR = {"crowd": 30.0, "ambience": 30.0}
MAXDUR_DEFAULT = 8.0
# max share of energy above 5 kHz
MAXHI = {"bell": 0.50, "ui": 0.50, "retro8bit": 0.45}
MAXHI_DEFAULT = 0.35


def decode(path):
    """Decode any audio file to float32 stereo 48k via ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ac", "2", "-ar", str(SR),
           "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True, timeout=120)
    if p.returncode != 0 or len(p.stdout) < 8:
        return None
    a = np.frombuffer(p.stdout, dtype="<f4")
    a = a[: len(a) // 2 * 2].reshape(-1, 2).astype(np.float64)
    return a


def trim(a, floor_db=-55.0, pad_ms=4.0):
    """Trim leading/trailing silence relative to the file's own peak."""
    m = np.abs(a).max(axis=1)
    pk = m.max()
    if pk <= 0:
        return None
    thr = pk * (10 ** (floor_db / 20.0))
    idx = np.flatnonzero(m > thr)
    if idx.size == 0:
        return None
    pad = int(SR * pad_ms / 1000.0)
    lo = max(0, idx[0] - pad)
    hi = min(len(a), idx[-1] + 1 + pad)
    return a[lo:hi]


def spectrum(mono):
    """Average power spectrum over 2048-sample Hann frames."""
    n = 2048
    if len(mono) < n:
        mono = np.pad(mono, (0, n - len(mono)))
    hop = n // 2
    win = np.hanning(n)
    frames = []
    for i in range(0, len(mono) - n + 1, hop):
        frames.append(np.abs(np.fft.rfft(mono[i:i + n] * win)) ** 2)
        if len(frames) >= 400:
            break
    if not frames:
        return None, None
    P = np.mean(frames, axis=0)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    return f, P


def analyse(a):
    mono = a.mean(axis=1)
    f, P = spectrum(mono)
    if P is None or P.sum() <= 0:
        return None
    centroid = float((f * P).sum() / P.sum())
    hi = float(P[f > 5000].sum() / P.sum())
    return centroid, hi


def fade(a, ms=3.0):
    """Tiny fades so trimmed edges don't click."""
    n = min(int(SR * ms / 1000.0), len(a) // 4)
    if n > 1:
        r = np.linspace(0, 1, n)[:, None]
        a[:n] *= r
        a[-n:] *= r[::-1]
    return a


def process(job):
    src = job["src_path"]
    cat = job["cat"]
    try:
        a = decode(src)
        if a is None:
            return {"skip": "decode", "src": src}

        # clipping check on the ORIGINAL signal, before any gain change
        pk_raw = float(np.abs(a).max())
        if pk_raw <= 1e-6:
            return {"skip": "silent", "src": src}
        clipped = int((np.abs(a) >= 0.9995).sum())
        if pk_raw >= 0.999 and clipped > len(a) * 0.0008:
            return {"skip": "clipping", "src": src}

        a = trim(a)
        if a is None:
            return {"skip": "silent", "src": src}

        dur = len(a) / SR
        if dur < 0.02:
            return {"skip": "tooshort", "src": src}
        if dur > MAXDUR.get(cat, MAXDUR_DEFAULT):
            return {"skip": f"toolong:{dur:.1f}s", "src": src}

        res = analyse(a)
        if res is None:
            return {"skip": "nospec", "src": src}
        centroid, hi = res
        if hi > MAXHI.get(cat, MAXHI_DEFAULT):
            return {"skip": f"harsh:hi={hi:.2f}", "src": src, "cat": cat}

        a = fade(a)
        a *= (10 ** (-1.0 / 20.0)) / max(np.abs(a).max(), 1e-9)   # peak -1 dBFS

        # content hash for dedupe (post-normalisation, coarse)
        q = np.round(a[:: max(1, len(a) // 4000)] * 512).astype(np.int16)
        h = hashlib.md5(q.tobytes() + str(round(dur, 2)).encode()).hexdigest()[:16]

        rel = f"audio/samples/{cat}/{job['name']}.wav"
        dst = os.path.join(ROOT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        pcm = np.clip(a, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SR),
             "-ac", "2", "-i", "-", "-c:a", "pcm_s16le", dst],
            input=pcm.tobytes(), capture_output=True, timeout=120)
        if p.returncode != 0:
            return {"skip": "write", "src": src}

        return {"ok": True, "file": rel, "cat": cat, "tags": job["tags"],
                "dur": round(dur, 3), "peak_db": -1.0,
                "centroid_hz": int(centroid), "hi_ratio": round(hi, 3),
                "src": job["src_url"], "license": job["license"], "hash": h}
    except Exception as e:
        return {"skip": f"err:{type(e).__name__}", "src": src}


if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1]))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        results = list(ex.map(process, jobs, chunksize=8))
    json.dump(results, open(sys.argv[2], "w"))
    ok = [r for r in results if r.get("ok")]
    print(f"processed={len(results)} kept={len(ok)}")
    from collections import Counter
    print("skips:", Counter(r["skip"].split(":")[0] for r in results if "skip" in r).most_common())
