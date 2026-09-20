#!/usr/bin/env python3
"""Drop duplicate renders, then emit catalog.json."""
import json
import os
from collections import Counter

ROOT = "/path/to/claude-vs-codex"
res = json.load(open("/path/to/scratchpad/results.json"))
ok = [r for r in res if r.get("ok")]

seen, items, dropped = set(), [], 0
for r in sorted(ok, key=lambda x: x["file"]):
    if r["hash"] in seen:
        try:
            os.remove(os.path.join(ROOT, r["file"]))
        except OSError:
            pass
        dropped += 1
        continue
    seen.add(r["hash"])
    items.append({k: r[k] for k in
                  ("file", "cat", "tags", "dur", "peak_db",
                   "centroid_hz", "hi_ratio", "src", "license")})

items.sort(key=lambda x: (x["cat"], x["file"]))
cat = {
    "generated_by": "sample-library agent",
    "format": "48000 Hz / stereo / 16-bit PCM WAV, trimmed, peak-normalised to -1 dBFS",
    "filters": {
        "max_hi_ratio": {"default": 0.35, "bell": 0.50, "ui": 0.50, "retro8bit": 0.45},
        "max_dur_s": {"default": 8.0, "crowd": 30.0},
        "note": "hi_ratio = share of spectral energy above 5 kHz; "
                "lower = softer. Clipped sources were rejected.",
    },
    "count": len(items),
    "by_cat": dict(sorted(Counter(i["cat"] for i in items).items())),
    "items": items,
}
out = os.path.join(ROOT, "audio/samples/catalog.json")
json.dump(cat, open(out, "w"), indent=1)
print(f"dropped dupes: {dropped}")
print(f"catalog items: {len(items)}")
for k, v in cat["by_cat"].items():
    print(f"  {k:<12} {v}")
