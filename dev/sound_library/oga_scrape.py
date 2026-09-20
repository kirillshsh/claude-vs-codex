#!/usr/bin/env python3
"""Scrape OpenGameArt content pages: extract file URLs + license. Parallel."""
import json, re, sys, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor

SLUGS = [
    # retro / 8-bit
    "512-sound-effects-8-bit-style", "8-bit-sound-effects-library",
    "8-bit-platformer-sfx", "coins-sound-effects-library",
    "63-digital-sound-effects-lasers-phasers-space-etc",
    "ignisforge-free-sfx-sampler-43-synthesized-retro-sound-effects",
    "8-bit-jump-sound", "8-bit-sfx", "bfxr-made-sounds", "8bit-sfx",
    # impact / combat
    "100-cc0-metal-and-wood-sfx", "37-hitspunches", "100-cc0-sfx", "100-cc0-sfx-2",
    "2-wooden-squish-splatter-sequences", "muffled-distant-explosion",
    "50-rpg-sound-effects", "rpg-sound-pack", "sound-effects-pack",
    "battle-sound-effects", "sword-clash", "punches-hits",
    # footsteps
    "42-snow-and-gravel-footsteps", "fantozzis-footsteps-grasssand-stone",
    "footsteps-0", "footsteps-leather-cloth-armor", "metal-footsteps-on-concrete",
    "platformer-sounds-terminal-interaction-door-shots-bang-and-footsteps",
    # mech / engine / electric
    "electric-sound-effects-library", "car-engine-start-01",
    "car-engine-start-up-02", "steamboat-engine-sound",
    # bell / ui
    "bell-dingschimes", "completion-sound", "fantasy-sound-effects-library",
    "interface-sounds-starter-pack", "menu-selection-click",
    # crowd
    "applause-in-a-large-hall-or-church",
    # misc grab-bags
    "random-sfx", "some-sounds-0", "random-sounds-samples",
    "haydenwoffle-sfx-dump", "atmospheric-interaction-sound-pack",
    "library-of-game-sounds",
]

AUD = re.compile(r'https://opengameart\.org/sites/default/files/[^"\'<>\s]+?\.(?:zip|wav|ogg|mp3|flac|7z|tar\.gz)',
                 re.I)
LIC = re.compile(r'(CC0|CC-BY-SA 4\.0|CC-BY-SA 3\.0|CC-BY 4\.0|CC-BY 3\.0|GPL 3\.0|GPL 2\.0|OGA-BY 3\.0)')


def fetch(slug):
    url = f"https://opengameart.org/content/{slug}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 sample-collector"})
        html = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "replace")
    except Exception as e:
        return {"slug": slug, "error": str(e)[:80]}
    files = sorted(set(urllib.parse.unquote(u) for u in AUD.findall(html)))
    lic = sorted(set(LIC.findall(html)))
    title = ""
    m = re.search(r'<h2[^>]*>(.*?)</h2>', html, re.S)
    if m:
        title = re.sub(r'<[^>]+>', '', m.group(1)).strip()[:80]
    return {"slug": slug, "url": url, "title": title, "licenses": lic, "files": files}


with ThreadPoolExecutor(max_workers=10) as ex:
    out = list(ex.map(fetch, SLUGS))

json.dump(out, open(sys.argv[1], "w"), indent=1)
for r in out:
    if "error" in r:
        print(f"ERR  {r['slug']}: {r['error']}")
    else:
        print(f"OK   {r['slug']:<55} lic={','.join(r['licenses']) or '?':<22} files={len(r['files'])}")
