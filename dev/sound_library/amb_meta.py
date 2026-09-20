#!/usr/bin/env python3
"""Shortlist candidates by title heuristics + fetch archive.org metadata in parallel."""
import json, re, sys, urllib.request, time
from concurrent.futures import ThreadPoolExecutor

SP = "/path/to/scratchpad"
CAND = SP + "/amb_candidates.json"
OUT = SP + "/amb_shortlist.json"

# words that usually mean urban / human / musical contamination
BAD = ["traffic", "car ", "cars", "street", "city", "town", "road", "highway", "motorway",
       "train", "railway", "tram", "metro", "subway", "bus ", "plane", "airport", "aircraft",
       "helicopter", "construction", "market", "crowd", "people", "talk", "voice", "speech",
       "interview", "song of the", "music", "concert", "band", "church", "bell", "choir",
       "protest", "demonstration", "festival", "machine", "engine", "motor", "factory",
       "harbour", "port ", "ferry", "boat", "ship", "siren", "alarm", "walla", "shopping",
       "restaurant", "cafe", "bar ", "pub ", "school", "children", "kids", "dog", "barking",
       "radio", "tv ", "television", "announce", "prayer", "mosque", "azan", "fireworks",
       "drone", "generator", "pump", "turbine", "chainsaw", "lawnmower", "mower", "tractor"]

GOOD = {
    "rain": ["rain", "downpour", "drizzle", "pluie", "regen", "lluvia", "puddle", "storm"],
    "thunder": ["thunder", "lightning", "gewitter", "orage", "tormenta", "tuono", "storm"],
    "wind": ["wind", "breeze", "gust", "vent", "viento", "wiatr", "windy"],
    "sea": ["sea", "wave", "waves", "surf", "ocean", "beach", "shore", "coast", "olas", "mer "],
    "birds": ["bird", "birds", "chorus", "song", "chirp", "skylark", "lark", "sparrow",
              "blackbird", "warbler", "nightingale", "meadow", "vogel", "oiseaux"],
    "insects": ["cricket", "crickets", "cicada", "cicadas", "grasshopper", "insect", "insects",
                "grillen", "cigarra", "grillo", "night"],
    "nature_misc": ["stream", "creek", "brook", "river", "water", "leaves", "grass", "rustl",
                    "footstep", "steps", "walking", "splash", "forest", "spring", "waterfall"],
    "room_tone": ["room", "indoor", "inside", "office", "apartment", "library", "empty",
                  "hall", "tone", "interior"],
}


def get(url, tries=3, timeout=45):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 amb"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            time.sleep(1 + i)
    return None


def score(rec, cat):
    t = rec["title"].lower()
    s = 0.0
    s -= rec["lic_rank"] * 1.5
    for w in GOOD.get(cat, []):
        if w in t:
            s += 3
            break
    # bonus for repeated / strong match
    s += sum(1.0 for w in GOOD.get(cat, []) if w in t) * 0.4
    for w in BAD:
        if w in t:
            s -= 4
    if len(t) > 120:
        s -= 0.5
    return s


def main():
    recs = json.load(open(CAND))
    per_cat = {}
    for r in recs:
        for c in r["cats"]:
            per_cat.setdefault(c, []).append(r)

    picked = {}
    LIMITS = {"rain": 55, "thunder": 55, "wind": 45, "sea": 45, "birds": 45,
              "insects": 40, "nature_misc": 50, "room_tone": 26}
    for cat, lst in per_cat.items():
        scored = sorted(((score(r, cat), r) for r in lst), key=lambda x: -x[0])
        n = LIMITS.get(cat, 35)
        for sc, r in scored[:n]:
            key = r["identifier"]
            e = picked.setdefault(key, dict(r))
            e.setdefault("pick_cats", [])
            if cat not in e["pick_cats"]:
                e["pick_cats"].append(cat)
            e["score_%s" % cat] = round(sc, 2)
    idents = list(picked)
    sys.stderr.write("shortlist items: %d\n" % len(idents))

    def fetch(ident):
        b = get("https://archive.org/metadata/" + ident)
        if not b:
            return ident, None
        try:
            return ident, json.loads(b)
        except Exception:
            return ident, None

    out = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        for i, (ident, md) in enumerate(ex.map(fetch, idents)):
            if not md or "files" not in md:
                continue
            rec = picked[ident]
            # choose best derivative: mp3 preferred, ogg fallback; skip huge originals
            best = None
            for f in md["files"]:
                fmt = (f.get("format") or "").lower()
                name = f.get("name", "")
                if not re.search(r"\.(mp3|ogg|m4a)$", name, re.I):
                    continue
                try:
                    size = int(f.get("size") or 0)
                    length = float(f.get("length") or 0)
                except Exception:
                    size, length = 0, 0
                if length and length < 45:
                    continue
                cand = {"name": name, "format": fmt, "size": size, "length": length}
                if best is None:
                    best = cand
                elif "mp3" in fmt and "mp3" not in best["format"]:
                    best = cand
                elif cand["size"] > best["size"] and "mp3" in fmt and "mp3" in best["format"]:
                    pass  # keep first mp3
            if not best:
                continue
            rec["file"] = best["name"]
            rec["fmt"] = best["format"]
            rec["fsize"] = best["size"]
            rec["dur"] = best["length"]
            rec["url"] = "https://archive.org/download/%s/%s" % (
                ident, urllib.parse.quote(best["name"]))
            rec["bitrate"] = (best["size"] * 8 / best["length"]) if best["length"] else 0
            out.append(rec)
            if (i + 1) % 50 == 0:
                sys.stderr.write("meta %d/%d\n" % (i + 1, len(idents)))

    json.dump(out, open(OUT, "w"), indent=1)
    from collections import Counter
    c = Counter()
    for r in out:
        for x in r["pick_cats"]:
            c[x] += 1
    print("with audio:", len(out))
    print(dict(c))
    print(Counter(r["license"] for r in out))
    tot = sum(min(r["fsize"], 6_000_000) for r in out)
    print("est download MB (capped 6MB each):", round(tot / 1e6))


if __name__ == "__main__":
    import urllib.parse  # noqa
    main()
