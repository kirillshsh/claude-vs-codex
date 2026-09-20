#!/usr/bin/env python3
"""Walk the downloaded packs, categorise every audio file, emit a job list."""
import json
import os
import re
from collections import defaultdict

ROOT = "/path/to/claude-vs-codex"
DL = os.path.join(ROOT, "audio/_dl")
AUDIO_EXT = (".wav", ".ogg", ".mp3", ".flac", ".aiff", ".aif")

# ---------------------------------------------------------------- categories
# (regex, category, extra tags) -- FIRST match wins, so order is priority.
RULES = [
    (r"applause|clap|crowd|cheer|audience", "crowd", ["crowd"]),
    (r"footstep|foot_step|walking|gravel|\bsteps?\b|snow_walk", "footstep", ["step"]),
    (r"bell|gong|chime|\bding|tink|sparkle|shimmer|glitter|glock|triangle_", "bell", ["bell"]),
    (r"whoosh|swoosh|swish|swosh|swing|swipe|woosh|slice|slash|\bair_\d|wind_short|megaswosh", "whoosh", ["whoosh"]),
    (r"engine|motor|gear\d|machine|electric|spark|thrust|hover|servo|robot|"
     r"handsaw|handbrake|drill|steam|turbine|tire|skid|rattle|fan_", "mech", ["mech"]),
    (r"impact|hit|punch|hammer|crash|smash|break|crack|splat|thud|bang|slam|"
     r"explos|glass|shatter|knock|bonk|whack|kick_door|collide|collision", "impact", ["impact"]),
    (r"coin|powerup|power_up|pickup|jump|laser|zap|blip|bleep|chiptune|8bit|"
     r"8-bit|sfx_|bfxr|jingle|fanfare|game_?over|level_?up|shoot|alarm", "retro8bit", ["retro"]),
    (r"click|select|switch|toggle|scroll|confirm|error|rollover|mouse|button|"
     r"\bui_|interface|menu|beep|tick|pluck|bong|question|glitch|notif|"
     r"maximize|minimize|\bback\b|\bdrop\b|\bopen\b|\bclose\b", "ui", ["ui"]),
]

# descriptive tags harvested from the filename
TAGWORDS = {
    "wood": "wood", "metal": "metal", "glass": "glass", "stone": "stone",
    "water": "water", "wet": "wet", "grass": "grass", "snow": "snow",
    "dirt": "dirt", "concrete": "concrete", "carpet": "carpet",
    "heavy": "heavy", "light": "light", "medium": "medium", "soft": "soft",
    "big": "heavy", "small": "light", "low": "low", "high": "high",
    "punch": "punch", "explos": "explosion", "door": "door", "coin": "coin",
    "laser": "laser", "jump": "jump", "powerup": "powerup", "power_up": "powerup",
    "engine": "engine", "motor": "motor", "gear": "gear", "electric": "electric",
    "spark": "spark", "bell": "bell", "gong": "gong", "chime": "chime",
    "click": "click", "error": "error", "confirm": "confirm", "select": "select",
    "switch": "switch", "beep": "beep", "alarm": "alarm", "crack": "crack",
    "break": "break", "hammer": "hammer", "kick": "kick", "snare": "snare",
    "hihat": "hihat", "hat": "hihat", "tom": "tom", "crash": "crash",
    "cymbal": "cymbal", "applause": "applause", "cheer": "cheer",
    "jingle": "jingle", "fanfare": "fanfare", "whoosh": "whoosh",
    "card": "card", "dice": "dice", "chip": "chip", "shatter": "shatter",
    "thud": "thud", "slam": "slam", "creak": "creak", "cloth": "cloth",
    "book": "book", "key": "key", "lock": "lock", "hover": "hover",
}

# ------------------------------------------------------- source descriptions
KENNEY_PACKS = {
    "impact-sounds": "Impact Sounds", "interface-sounds": "Interface Sounds",
    "ui-audio": "UI Audio", "digital-audio": "Digital Audio",
    "sci-fi-sounds": "Sci-Fi Sounds", "rpg-audio": "RPG Audio",
    "music-jingles": "Music Jingles", "casino-audio": "Casino Audio",
}
# packs whose sounds should bypass the keyword rules and land in a fixed cat
FORCE_CAT = {
    "kenney/music-jingles": "retro8bit",
    "kenney/digital-audio": "retro8bit",
    "kenney/ui-audio": "ui",
    "kenney/interface-sounds": "ui",
    "oga/512-sound-effects-8-bit-style": "retro8bit",
    "oga/8-bit-sound-effects-library": "retro8bit",
    "oga/8-bit-platformer-sfx": "retro8bit",
    "oga/8bit-sfx": "retro8bit",
    "oga/63-digital-sound-effects-lasers-phasers-space-etc": "retro8bit",
    "oga/42-snow-and-gravel-footsteps": "footstep",
    "oga/fantozzis-footsteps-grasssand-stone": "footstep",
    "oga/footsteps-0": "footstep",
    "oga/footsteps-leather-cloth-armor": "footstep",
    "oga/metal-footsteps-on-concrete": "footstep",
    "oga/37-hitspunches": "impact",
    "oga/2-wooden-squish-splatter-sequences": "impact",
    "oga/muffled-distant-explosion": "impact",
    "oga/car-engine-start-01": "mech",
    "oga/car-engine-start-up-02": "mech",
    "oga/steamboat-engine-sound": "mech",
    "oga/electric-sound-effects-library": "mech",
    "oga/some-sounds-0": "mech",
    "oga/bell-dingschimes": "bell",
    "oga/applause-in-a-large-hall-or-church": "crowd",
    "oga/interface-sounds-starter-pack": "ui",
    "oga/menu-selection-click": "ui",
    "oga/completion-sound": "ui",
    "oga/coins-sound-effects-library": "retro8bit",
    "oga/swishes-sound-pack": "whoosh",
    "oga/swish-bamboo-stick-weapon-swhoshes": "whoosh",
    "oga/air-whoosh": "whoosh",
    "oga/sword-swing": "whoosh",
    "oga/wind-hit-time-morph": "whoosh",
    "oga/punches-hits-swords-and-squishes": "impact",
    "oga/free-crowd-cheering-sounds": "crowd",
    "oga/applause": "crowd",
    "oga/crowd-shoutingspeaking-ambience": "crowd",
    "oga/fireworks-with-applause-happy-people": "crowd",
    "oga/female-warrior-cheer": "crowd",
    "oga/stunt-rally-sounds": "mech",
    "oga/rolling-0": "mech",
    "oga/church-bell": "bell",
    "oga/point-bell": "bell",
    "oga/bell-arpeggio-24": "bell",
    "oga/gonk-gong": "bell",
    "oga/shimmer-glitter-magic": "bell",
    "oga/music-box-game-over-iii": "bell",
    "oga/ui-sound-effects-button-clicks-user-feedback-notifications": "ui",
    "oga/sci-fi-ui-sfx": "ui",
    "oga/well-done": "ui",
    "oga/video-game-cheerful-ending": "retro8bit",
    "oga/18-random-video-game-sound-effects": "retro8bit",
    "oga/magic-spell-sfx": "bell",
    "oga/various-sound-effects": "ui",
    "oga/80-cc0-rpg-sfx": "impact",
    "oga/jc-sounds-fantasy-sfx-pack-vol-1": "bell",
    "oga/spell-sounds-starter-pack": "bell",
}

oga_meta = {r["slug"]: r for r in json.load(open(
    "/path/to/scratchpad/oga_dl.json"))}


def norm_lic(lics):
    if not lics:
        return "CC0"
    if "CC0" in lics:
        return "CC0"
    for pref in ("CC-BY 3.0", "CC-BY 4.0", "CC-BY-SA 3.0", "OGA-BY 3.0"):
        if pref in lics:
            return pref
    return lics[0]


def slugify(s):
    s = re.sub(r"\.[A-Za-z0-9]+$", "", s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return re.sub(r"_+", "_", s)[:48] or "snd"


def categorise(name, packkey):
    low = name.lower()
    forced = FORCE_CAT.get(packkey)
    cat = None
    for rx, c, _ in RULES:
        if re.search(rx, low):
            cat = c
            break
    # a forced pack wins unless the filename clearly says footstep/impact/crowd
    if forced:
        if cat in ("footstep", "crowd") and forced != cat:
            pass
        elif cat == "impact" and forced in ("retro8bit", "mech"):
            pass
        else:
            cat = forced
    return cat


def tags_for(name, cat):
    low = name.lower()
    t = [cat]
    for k, v in TAGWORDS.items():
        if k in low and v not in t:
            t.append(v)
    return t[:8]


jobs = []
seen_names = set()
stem_count = defaultdict(int)
MAX_PER_STEM = 5          # coin1..coin454 -> keep a handful
# scarce categories need more variants from a single numbered series
STEM_CAP = {"whoosh": 16, "crowd": 12, "bell": 10, "footstep": 12, "percussion": 8}
MAX_PER_CAT = {"retro8bit": 260, "ui": 200, "impact": 300, "mech": 150,
               "footstep": 120, "bell": 90, "whoosh": 90, "crowd": 40,
               "percussion": 120}
cat_count = defaultdict(int)


def add(path, packkey, src_url, lic, cat=None, tags=None, prefix=""):
    base = os.path.basename(path)
    if base.startswith("._") or base.startswith("."):
        return
    if not base.lower().endswith(AUDIO_EXT):
        return
    if re.search(r"preview|sampler|demo|readme", base, re.I):
        return
    cat = cat or categorise(base, packkey)
    if not cat:
        return
    if cat_count[cat] >= MAX_PER_CAT.get(cat, 150):
        return
    stem = re.sub(r"[\d_\- ]+$", "", slugify(prefix + base))
    key = (cat, stem)
    if stem_count[key] >= STEM_CAP.get(cat, MAX_PER_STEM):
        return
    name = f"{prefix}{slugify(base)}"
    n, i = name, 2
    while n in seen_names:
        n = f"{name}_{i}"
        i += 1
    seen_names.add(n)
    stem_count[key] += 1
    cat_count[cat] += 1
    jobs.append({"src_path": path, "cat": cat, "tags": tags or tags_for(base, cat),
                 "src_url": src_url, "license": lic, "name": n})


def walk(root):
    for dp, _, fns in os.walk(root):
        if "__MACOSX" in dp:
            continue
        for f in sorted(fns):
            yield os.path.join(dp, f)


# ---- Kenney ---------------------------------------------------------------
for pack, title in KENNEY_PACKS.items():
    for p in walk(os.path.join(DL, "kenney", pack)):
        add(p, f"kenney/{pack}", f"https://kenney.nl/assets/{pack}", "CC0",
            prefix="kenney_")

# ---- Tone.js drum kits (percussion) ---------------------------------------
tj = os.path.join(DL, "tonejs")
if os.path.isdir(tj):
    for kit in sorted(os.listdir(tj)):
        kdir = os.path.join(tj, kit)
        if not os.path.isdir(kdir):
            continue
        for p in walk(kdir):
            base = slugify(os.path.basename(p))
            add(p, "tonejs", "https://github.com/Tonejs/audio/tree/master/drum-samples",
                "see Tone.js audio repo (from cwilso/web-audio-samples)",
                cat="percussion",
                tags=["percussion", "drum", slugify(kit), base],
                prefix=f"drum_{slugify(kit)}_")

# ---- OpenGameArt ----------------------------------------------------------
ogadir = os.path.join(DL, "oga")
for slug in sorted(os.listdir(ogadir)):
    d = os.path.join(ogadir, slug)
    if not os.path.isdir(d):
        continue
    meta = oga_meta.get(slug, {})
    lic = norm_lic(meta.get("lic", []))
    url = f"https://opengameart.org/content/{slug}"
    for p in walk(d):
        add(p, f"oga/{slug}", url, lic, prefix="oga_")

out = "/path/to/scratchpad/jobs.json"
json.dump(jobs, open(out, "w"))
print(f"jobs: {len(jobs)}")
for c, n in sorted(cat_count.items(), key=lambda x: -x[1]):
    print(f"  {c:<12} {n}")
