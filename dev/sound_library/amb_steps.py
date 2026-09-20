#!/usr/bin/env python3
"""Footsteps and water splashes are transient, so the steady-window picker rejects them.
Search for them specifically and extract with the steadiness tests turned off."""
import json, os, re, sys, urllib.parse, urllib.request, time, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amb_dsp import (SR, HOP, decode, spectral_stats, speech_mod, butter_lp, butter_hp,
                     normalize_peak, fade, write_wav, measure)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import amb_search as S

SP = "/path/to/scratchpad"
PROJ = SP + "/cartoon"
DL = PROJ + "/audio/_dl_amb/steps"
OUT = PROJ + "/audio/samples/ambience/nature_misc"
FPS = SR / HOP

TERMS = ["footsteps", "footsteps gravel", "walking gravel", "steps snow", "walking path",
         "walking mud", "puddle", "splash", "splashing", "water drops", "dripping",
         "rain puddle", "wading", "boots", "walking leaves", "water dripping"]


def collect():
    found = {}
    for t in TERMS:
        for d in S.search('collection:"radio-aporee-maps" AND title:(%s)' % t, 25):
            lr = S.lic_rank(d.get("licenseurl"))
            if lr is None or lr > 3:
                continue
            tt = d.get("title", "").lower()
            if not any(k in tt for k in ("step", "walk", "splash", "puddle", "drip",
                                         "drop", "wading", "boots", "gravel")):
                continue
            if any(b in tt for b in ("traffic", "city", "street", "crowd", "market", "car")):
                continue
            found[d["identifier"]] = dict(identifier=d["identifier"], title=d.get("title", ""),
                                          creator=d.get("creator", ""),
                                          licenseurl=d.get("licenseurl", ""),
                                          license=S.lic_name(d.get("licenseurl")), lic_rank=lr,
                                          src="https://archive.org/details/" + d["identifier"])
    out = []
    for ident, rec in found.items():
        b = S.get("https://archive.org/metadata/" + ident)
        if not b:
            continue
        try:
            md = json.loads(b)
        except Exception:
            continue
        for f in md.get("files", []):
            if re.search(r"\.mp3$", f.get("name", ""), re.I):
                try:
                    ln = float(f.get("length") or 0)
                    sz = int(f.get("size") or 0)
                except Exception:
                    continue
                if ln < 40:
                    continue
                rec.update(file=f["name"], fsize=sz, dur=ln,
                           url="https://archive.org/download/%s/%s" % (
                               ident, urllib.parse.quote(f["name"])))
                out.append(rec)
                break
    return out


def dl(recs, cap=3_000_000):
    os.makedirs(DL, exist_ok=True)
    jobs = []
    for r in recs:
        dst = os.path.join(DL, r["identifier"] + ".mp3")
        r["local"] = dst
        off = int(r["fsize"] * 0.10) // 1024 * 1024 if r["fsize"] > cap * 2.2 else 0
        jobs.append((r["url"], dst, "%d-%d" % (off, off + cap)))

    def one(j):
        url, dst, rng = j
        if os.path.exists(dst) and os.path.getsize(dst) > 200_000:
            return
        subprocess.run(["curl", "-sL", "--retry", "4", "--retry-delay", "2",
                        "--connect-timeout", "10", "--max-time", "150", "-r", rng,
                        "-o", dst, url], capture_output=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, jobs))
    return [r for r in recs if os.path.exists(r["local"]) and os.path.getsize(r["local"]) > 100_000]


def pick(recs):
    """Transient density is the signal here: many sharp, short events over a quiet bed."""
    cands = []
    for rec in recs:
        x = decode(rec["local"], stereo=True)
        if x is None or len(x) < SR * 10:
            continue
        x = x[int(SR * 2.0): len(x) - int(SR * 1.0)]
        m = x.mean(axis=1)
        st = spectral_stats(m)
        if st is None:
            continue
        rms = st["rms"]
        rdb = 20 * np.log10(rms + 1e-9)
        n = len(rms)
        for d in (12, 10, 8):
            wf = int(d * FPS)
            if wf + 4 > n:
                continue
            best = None
            for a in range(0, n - wf, int(1.0 * FPS)):
                seg = rdb[a:a + wf]
                hi = float(np.median(st["hi"][a:a + wf]))
                cen = float(np.median(st["cen"][a:a + wf]))
                if hi > 0.34 or cen < 300 or cen > 5000:
                    continue
                if float(np.median(seg)) < -55:
                    continue
                if speech_mod(st["mid"][a:a + wf]) > 0.40:
                    continue
                # count distinct onsets: frame jumps of >5 dB above the local floor
                floor = float(np.percentile(seg, 25))
                onsets = int(np.sum((seg[1:] - seg[:-1] > 4.0) & (seg[1:] > floor + 6)))
                if onsets < 3:
                    continue
                sc = min(onsets, 14) * 1.0 + (0.34 - hi) * 6 - abs(np.median(seg) + 28) * 0.05
                if best is None or sc > best["score"]:
                    best = dict(score=float(sc), a=a, dur=d, hi=hi, cen=cen,
                                onsets=onsets, rms_db=float(np.median(seg)))
            if best:
                best["rec"] = rec
                best["x"] = x
                cands.append(best)
                break
    cands.sort(key=lambda c: -(c["score"] - 1.5 * c["rec"]["lic_rank"]))
    return cands


def kind(title):
    t = title.lower()
    if any(k in t for k in ("splash", "puddle", "drip", "drop", "wading")):
        return "splash"
    return "footsteps"


def main():
    recs = collect()
    print("candidate sources:", len(recs))
    recs = dl(recs)
    print("downloaded:", len(recs))
    cands = pick(recs)
    print("usable windows:", len(cands))
    items, names, seen = [], {}, set()
    for c in cands:
        rec = c["rec"]
        if rec["identifier"] in seen:
            continue
        seen.add(rec["identifier"])
        a = int(c["a"] * HOP)
        seg = butter_hp(c["x"][a: a + int(c["dur"] * SR)].copy(), 30.0)
        for fcv in (10000, 9000, 8000, 7000):
            s2 = butter_lp(seg, fcv)
            if measure(s2)["hi_ratio"] <= 0.30:
                seg, fc = s2, fcv
                break
        else:
            seg, fc = butter_lp(seg, 7000), 7000
        seg = normalize_peak(fade(seg, ms=40), -3.0)
        k = kind(rec["title"])
        nn = names.get(k, 0) + 1
        names[k] = nn
        name = "nature_misc_%s_%02d" % (k, nn)
        fp = os.path.join(OUT, name + ".wav")
        write_wav(fp, seg)
        mm = measure(seg)
        items.append(dict(file=os.path.relpath(fp, PROJ), cat="nature_misc",
                          tags=sorted({k, "transient", "events"}), dur=mm["dur"],
                          peak_db=mm["peak_db"], rms_db=mm["rms_db"],
                          centroid_hz=mm["centroid_hz"], hi_ratio=mm["hi_ratio"],
                          lo_ratio=mm["lo_ratio"], lowpass_hz=fc, loop=False,
                          onsets=c["onsets"], src=rec["src"], title=rec["title"],
                          creator=rec.get("creator", ""), license=rec["license"],
                          license_url=rec.get("licenseurl", "")))
        print("  %-34s %5.1fs onsets=%2d cen%5d hi%.3f" % (
            name, mm["dur"], c["onsets"], mm["centroid_hz"], mm["hi_ratio"]))
        if names.get("footsteps", 0) >= 5 and names.get("splash", 0) >= 4:
            break
        if len(items) >= 9:
            break
    json.dump(items, open(SP + "/items_steps.json", "w"), indent=1)


if __name__ == "__main__":
    main()
