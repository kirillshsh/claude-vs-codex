# Real sample library

1020 real recorded/game-audio samples to replace the pure-synth SFX that "cut the
ears". Everything here is a **real sample from a free pack**, not synthesis.

All files: **48 kHz · stereo · 16-bit PCM WAV**, silence trimmed off both ends,
3 ms edge fades (no clicks), peak normalised to **−1.0 dBFS**.

## Categories

| category | files | what's in it |
|---|--:|---|
| `impact/` | 233 | hits, punches, wood/metal/glass/plate impacts, hammers, crashes, explosions, splatter — for the fight, the race crash, the cup splitting |
| `retro8bit/` | 250 | chiptune coins, jumps, power-ups, lasers, zaps, game-over stings, NES/steel jingles |
| `ui/` | 138 | clicks, selects, toggles, scrolls, confirms, errors, notifications, popups |
| `mech/` | 113 | car/space engines, motors, gears, electric discharge, hover hum, skids, rattles |
| `footstep/` | 104 | grass, dirt, gravel, snow, wood, concrete, wet/puddle steps, cloth/armour movement |
| `bell/` | 68 | bells, gongs, church bell, chimes, dings, shimmer/sparkle/magic tinkles |
| `percussion/` | 60 | **real drum kits** — kick, snare, hihat, toms from 17 kits (acoustic, CR78, LINN, R8, Techno, breakbeat, bongos…) |
| `whoosh/` | 42 | swishes, swooshes, bamboo/weapon swings, air whooshes — for camera cuts |
| `crowd/` | 12 | applause, cheering, small crowds (up to 30 s) |

`ambience/` and `catalog_ambience.json` belong to a **different agent** (nature
atmospheres) — not covered here, but `find.py` reads both catalogs transparently.

## Sources

Roughly 80 % CC0. Full per-pack breakdown with links and licences:
[`LICENSES/SOURCES.md`](LICENSES/SOURCES.md). Original licence texts shipped in
the packs are in `LICENSES/`.

- **[kenney.nl](https://kenney.nl/assets)** (CC0) — Impact Sounds, Interface
  Sounds, UI Audio, Digital Audio, Sci-Fi Sounds, RPG Audio, Music Jingles,
  Casino Audio. 8 packs, the backbone of `impact` / `ui` / `retro8bit`.
- **[opengameart.org](https://opengameart.org)** (mostly CC0, some CC-BY /
  CC-BY-SA) — 47 packs, incl. *512 Sound Effects (8-bit style)*, *100 CC0 SFX
  1 & 2*, *100 CC0 metal and wood SFX*, *37 hits/punches*, *RPG Sound Pack*,
  *swishes*, *crowd cheering*, footstep and engine packs.
- **[Tone.js/audio](https://github.com/Tonejs/audio/tree/master/drum-samples)** —
  17 real drum kits (from `cwilso/web-audio-samples`). This is the `percussion/`
  category, meant for the music agent to replace synthesised drums.

Licence totals: CC0 805 · CC-BY 3.0 58 · CC-BY 4.0 51 · CC-BY-SA 3.0 45 ·
CC-BY-SA 4.0 1 · Tone.js drums 60. Nothing here needs attribution for a personal
project, but the per-pack table has the credits if it ever ships.

## Harshness filtering — why this library shouldn't cut the ears

For every file we measured the average spectrum and recorded two numbers in the
catalog:

- **`hi_ratio`** — share of spectral energy above 5 kHz. **This is the knob to
  use.** Lower = duller and softer.
- **`centroid_hz`** — spectral centroid ("brightness" in Hz).

Files were **rejected at build time** when `hi_ratio` exceeded 0.35 (0.50 for
`bell`/`ui`, 0.45 for `retro8bit`), when the source was **clipped**, or when they
were too long (>8 s; >30 s for `crowd`). 250 of 1270 candidates were thrown out —
135 for harshness, 42 for clipping, 36 for length.

Surviving files are already tame (median `hi_ratio` is 0.00–0.09 in every
category), but **for a gentle mix still ask for `max_hi=0.10…0.20`** — there are
plenty: 201 soft impacts, 209 soft retro, 109 soft UI, all 42 whooshes.

## How to use `find.py`

```python
import sys; sys.path.insert(0, "audio/samples")
from find import find, pick, info, cats, tags_in

find("impact", tags=["wood", "heavy"])   # heavy wooden hits (absolute paths)
find("impact", max_hi=0.10)              # only the SOFT impacts
find("percussion", tags=["kick"])        # real kick drums
find("whoosh", max_dur=0.5)              # short camera-cut swishes
find("bell", max_centroid=2000)          # mellow bells only
find(license="CC0")                      # CC0 subset

pick("impact", ["metal"], seed=7)        # one deterministic choice
info("kenney_impactwood_heavy_000.wav")  # full record for one file
cats(); tags_in("impact")                # discovery
```

`find(...)` signature:

```
find(cat=None, tags=(), max_hi=1.0, max_dur=None, min_dur=None,
     max_centroid=None, any_tags=(), license=None,
     limit=None, shuffle=False, seed=None)
```

`tags` = **all** must match, `any_tags` = **at least one**. Results are sorted
softest-first (by `hi_ratio`) unless `shuffle=True`. Paths come back **absolute**,
so cwd doesn't matter. Run `python3 audio/samples/find.py` for a live summary.

## `catalog.json`

```json
{"items": [{"file": "audio/samples/impact/kenney_impactwood_heavy_000.wav",
            "cat": "impact", "tags": ["impact", "wood", "heavy"], "dur": 0.31,
            "peak_db": -1.0, "centroid_hz": 112, "hi_ratio": 0.0,
            "src": "https://kenney.nl/assets/impact-sounds", "license": "CC0"}]}
```

Top level also carries `count`, `by_cat`, `format` and the `filters` actually
applied. Filenames are prefixed by origin (`kenney_`, `oga_`, `drum_<kit>_`).
