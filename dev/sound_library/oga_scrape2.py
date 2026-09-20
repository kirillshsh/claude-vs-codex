#!/usr/bin/env python3
"""Scrape OpenGameArt content pages: extract file URLs + license. Parallel."""
import json, re, sys, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor

SLUGS = ['swishes-sound-pack', 'punches-hits-swords-and-squishes', 'sword-swing', 'swish-bamboo-stick-weapon-swhoshes', 'wind-hit-time-morph', 'free-crowd-cheering-sounds', 'applause', 'stunt-rally-sounds', 'fireworks-with-applause-happy-people', 'video-game-cheerful-ending', 'crowd-shoutingspeaking-ambience', 'well-done', 'female-warrior-cheer', 'church-bell', 'point-bell', 'shimmer-glitter-magic', 'bell-arpeggio-24', 'various-sound-effects', 'music-box-game-over-iii', 'gonk-gong', 'ui-sound-effects-button-clicks-user-feedback-notifications', 'sci-fi-ui-sfx', 'jc-sounds-fantasy-sfx-pack-vol-1', '80-cc0-rpg-sfx', 'magic-spell-sfx', 'spell-sounds-starter-pack', '18-random-video-game-sound-effects', 'air-whoosh', '80-cc0-creature-sfx', '15-monster-gruntpaindeath-sounds', 'rolling-0']

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
