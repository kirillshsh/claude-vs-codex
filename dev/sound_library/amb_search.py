#!/usr/bin/env python3
"""Search archive.org radio-aporee-maps (+ general PD audio) for ambience candidates."""
import json, urllib.parse, urllib.request, sys, time, os, re

OUT = "/path/to/scratchpad/amb_candidates.json"

QUERIES = {
    "rain": ["rain", "heavy rain", "rainfall", "rain drops", "raining", "downpour",
             "rain on roof", "rain leaves", "puddle", "light rain", "rain window"],
    "thunder": ["thunder", "thunderstorm", "lightning", "storm thunder", "tormenta",
                "gewitter", "orage", "thunder rain"],
    "wind": ["wind", "breeze", "windy", "wind trees", "gust", "wind grass",
             "wind field", "strong wind", "viento", "vent"],
    "sea": ["sea waves", "ocean waves", "surf", "beach waves", "seashore", "sea shore",
            "waves rocks", "coast waves", "olas", "meer wellen", "sea"],
    "birds": ["birds", "birdsong", "dawn chorus", "birds singing", "skylark", "meadow birds",
              "birds meadow", "songbirds", "birds morning", "bird song"],
    "insects": ["crickets", "cicadas", "grasshoppers", "insects night", "insects evening",
                "cricket", "cigarras", "grillen", "night insects"],
    "nature_misc": ["stream", "creek", "brook", "rustling leaves", "grass rustle",
                    "footsteps gravel", "water splash", "small stream", "forest leaves",
                    "river", "walking forest", "steps mud"],
    "room_tone": ["room tone", "roomtone", "room ambience", "empty room", "indoor ambience",
                  "office ambience", "quiet room", "inside room", "apartment ambience",
                  "library ambience"],
}

# licence ranking: lower = better. ND excluded entirely (we create derivatives).
def lic_rank(url):
    if not url:
        return None
    u = url.lower()
    if "-nd" in u or "/nd" in u:
        return None
    if "publicdomain/zero" in u:
        return 0
    if "publicdomain/mark" in u or "publicdomain" in u:
        return 1
    if re.search(r"licenses/by/", u):
        return 2
    if re.search(r"licenses/by-sa/", u):
        return 3
    if re.search(r"licenses/by-nc/", u):
        return 5
    if re.search(r"licenses/by-nc-sa/", u):
        return 6
    return 8


def lic_name(url):
    u = (url or "").lower()
    if "publicdomain/zero" in u: return "CC0-1.0"
    if "publicdomain/mark" in u: return "PublicDomainMark-1.0"
    if "licenses/by/" in u: return "CC-BY-3.0"
    if "licenses/by-sa/" in u: return "CC-BY-SA-3.0"
    if "licenses/by-nc-sa/" in u: return "CC-BY-NC-SA-3.0"
    if "licenses/by-nc/" in u: return "CC-BY-NC-3.0"
    return url or "unknown"


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 ambience-collector"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            sys.stderr.write("retry %d %s: %s\n" % (i + 1, url[:90], e))
            time.sleep(2 + i * 2)
    return None


def search(q, rows=40):
    base = "https://archive.org/advancedsearch.php?q=%s&output=json&rows=%d&fl[]=identifier&fl[]=title&fl[]=licenseurl&fl[]=creator"
    url = base % (urllib.parse.quote(q), rows)
    b = get(url)
    if not b:
        return []
    try:
        return json.loads(b)["response"]["docs"]
    except Exception:
        return []


def main():
    found = {}          # identifier -> record
    for cat, terms in QUERIES.items():
        for t in terms:
            q = 'collection:"radio-aporee-maps" AND title:(%s)' % t
            docs = search(q, 40)
            for d in docs:
                ident = d["identifier"]
                lr = lic_rank(d.get("licenseurl"))
                if lr is None or lr > 6:
                    continue
                rec = found.setdefault(ident, {
                    "identifier": ident,
                    "title": d.get("title", ""),
                    "creator": d.get("creator", ""),
                    "licenseurl": d.get("licenseurl", ""),
                    "license": lic_name(d.get("licenseurl")),
                    "lic_rank": lr,
                    "cats": [],
                    "terms": [],
                    "src": "https://archive.org/details/" + ident,
                })
                if cat not in rec["cats"]:
                    rec["cats"].append(cat)
                if t not in rec["terms"]:
                    rec["terms"].append(t)
            sys.stderr.write("%-12s %-22s -> %d docs (total %d)\n" % (cat, t, len(docs), len(found)))
    recs = list(found.values())
    json.dump(recs, open(OUT, "w"), indent=1)
    print("TOTAL", len(recs))
    from collections import Counter
    c = Counter()
    for r in recs:
        for x in r["cats"]:
            c[x] += 1
    print(dict(c))
    print(Counter(r["license"] for r in recs))


if __name__ == "__main__":
    main()
