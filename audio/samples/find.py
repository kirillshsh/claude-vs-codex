#!/usr/bin/env python3
"""
Lookup helper for the real-sample library in audio/samples/.

Every sample is 48 kHz / stereo / 16-bit WAV, silence-trimmed and peak-normalised
to -1 dBFS, so you can load one and drop it straight into the mix.

    from audio.samples.find import find, info, cats, tags_in

    find("impact", tags=["wood", "heavy"])       # heavy wooden hits
    find("impact", max_hi=0.15)                  # only the SOFT impacts
    find("percussion", tags=["kick"])            # real kick drums
    find("whoosh", max_dur=0.5)                  # short camera-cut swishes

`max_hi` is the important one: hi_ratio is the share of spectral energy above
5 kHz. Lower = duller / softer. Anything harsh was already rejected at build
time, but for a gentle mix prefer max_hi=0.10-0.20.

Paths come back absolute, so they work regardless of the caller's cwd.
"""
from __future__ import annotations

import json
import os
import random
from typing import Iterable, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # .../cartoon
_CATALOGS = ("catalog.json", "catalog_ambience.json")

_cache: list[dict] | None = None


def _load() -> list[dict]:
    """Load (and merge) every catalog next to this file. Cached."""
    global _cache
    if _cache is not None:
        return _cache
    items: list[dict] = []
    for name in _CATALOGS:
        p = os.path.join(_HERE, name)
        if not os.path.exists(p):
            continue
        try:
            with open(p) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        items.extend(data.get("items", []) if isinstance(data, dict) else data)
    _cache = items
    return items


def reload() -> int:
    """Forget the cache (use after a catalog is rewritten)."""
    global _cache
    _cache = None
    return len(_load())


def _abs(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(_ROOT, rel)


def find(cat: str | None = None,
         tags: Iterable[str] = (),
         max_hi: float = 1.0,
         max_dur: float | None = None,
         min_dur: float | None = None,
         max_centroid: float | None = None,
         any_tags: Iterable[str] = (),
         license: str | None = None,
         limit: int | None = None,
         shuffle: bool = False,
         seed: int | None = None,
         existing_only: bool = True) -> list[str]:
    """Return absolute paths of samples matching every constraint given.

    cat           category name, e.g. "impact" (None = any)
    tags          ALL of these tags must be present
    any_tags      AT LEAST ONE of these tags must be present
    max_hi        keep only hi_ratio <= this  (softness filter)
    max_dur/min_dur   duration bounds in seconds
    max_centroid  keep only spectral centroid <= this (Hz)
    license       exact licence string, e.g. "CC0"
    limit         cap the number of results
    shuffle       randomise order (otherwise sorted by hi_ratio, softest first)
    """
    want = {t.lower() for t in tags}
    want_any = {t.lower() for t in any_tags}
    out = []
    for it in _load():
        if cat and it.get("cat") != cat:
            continue
        have = {str(t).lower() for t in it.get("tags", ())}
        if want and not want <= have:
            continue
        if want_any and not (want_any & have):
            continue
        if it.get("hi_ratio", 0.0) > max_hi:
            continue
        d = it.get("dur", 0.0)
        if max_dur is not None and d > max_dur:
            continue
        if min_dur is not None and d < min_dur:
            continue
        if max_centroid is not None and it.get("centroid_hz", 0) > max_centroid:
            continue
        if license and it.get("license") != license:
            continue
        p = _abs(it["file"])
        if existing_only and not os.path.exists(p):
            continue
        out.append((it, p))

    if shuffle:
        random.Random(seed).shuffle(out)
    else:
        out.sort(key=lambda x: (x[0].get("hi_ratio", 0.0), x[0]["file"]))
    paths = [p for _, p in out]
    return paths[:limit] if limit else paths


def info(path_or_name: str) -> dict | None:
    """Full catalog record for a path or bare filename."""
    key = os.path.basename(path_or_name)
    for it in _load():
        if os.path.basename(it["file"]) == key or it["file"] == path_or_name:
            d = dict(it)
            d["abspath"] = _abs(it["file"])
            return d
    return None


def cats() -> dict[str, int]:
    """Category -> number of samples."""
    out: dict[str, int] = {}
    for it in _load():
        out[it.get("cat", "?")] = out.get(it.get("cat", "?"), 0) + 1
    return dict(sorted(out.items()))


def tags_in(cat: str | None = None) -> dict[str, int]:
    """Tag -> count, optionally within one category. Handy for discovery."""
    out: dict[str, int] = {}
    for it in _load():
        if cat and it.get("cat") != cat:
            continue
        for t in it.get("tags", ()):
            out[t] = out.get(t, 0) + 1
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def pick(cat: str, tags: Sequence[str] = (), seed: int | None = None,
         **kw) -> str | None:
    """One sample, chosen deterministically from `seed`. None if nothing fits."""
    hits = find(cat, tags, shuffle=True, seed=seed, **kw)
    return hits[0] if hits else None


if __name__ == "__main__":
    print(f"{len(_load())} samples\n")
    for c, n in cats().items():
        soft = len(find(c, max_hi=0.15))
        print(f"  {c:<12} {n:>4}   ({soft} soft, hi_ratio<=0.15)")
    print("\nimpact/wood example:", (find('impact', ['wood'])[:1] or ['-'])[0])
