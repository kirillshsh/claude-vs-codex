#!/usr/bin/env python3
"""Select clean windows from downloaded fragments, render 48k/stereo/16-bit WAVs + loops."""
import json, os, re, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amb_dsp import (SR, HOP, decode, spectral_stats, speech_mod, db, butter_lp, butter_hp,
                     normalize_peak, fade, make_loop, write_wav, measure)

SP = "/path/to/scratchpad"
PROJ = SP + "/cartoon"
DL = PROJ + "/audio/_dl_amb"
AMB = PROJ + "/audio/samples/ambience"
LIC = PROJ + "/audio/samples/LICENSES_AMB"
FPS = SR / HOP  # analysis frames per second

# per-category tuning. cen=(min, ideal, max) Hz, lo=(min, ideal, max) low-band energy share.
CFG = {
    "rain":        dict(durs=[22, 18, 14], hi_max=0.30, std_max=5.0, rms_min=-52, keep=10, loops=5,
                        cen=(550, 1400, 3200), lo=(0.08, 0.40, 0.80)),
    "wind":        dict(durs=[18, 15, 12], hi_max=0.30, std_max=6.5, rms_min=-55, keep=9, loops=4,
                        cen=(120, 520, 1800), lo=(0.30, 0.70, 0.97)),
    "sea":         dict(durs=[22, 18, 15], hi_max=0.30, std_max=7.5, rms_min=-50, keep=9, loops=4,
                        cen=(350, 1100, 2800), lo=(0.12, 0.45, 0.88)),
    "birds":       dict(durs=[18, 15, 12], hi_max=0.34, std_max=7.0, rms_min=-55, keep=9, loops=4,
                        cen=(1200, 2800, 5500), lo=(0.01, 0.15, 0.65)),
    "insects":     dict(durs=[18, 15, 12], hi_max=0.45, std_max=5.5, rms_min=-55, keep=9, loops=4,
                        cen=(1500, 3800, 7500), lo=(0.02, 0.14, 0.60)),
    "nature_misc": dict(durs=[16, 13, 11], hi_max=0.34, std_max=7.0, rms_min=-55, keep=11, loops=4,
                        cen=(350, 1500, 4200), lo=(0.04, 0.35, 0.90)),
    "room_tone":   dict(durs=[18, 15, 12], hi_max=0.28, std_max=4.5, rms_min=-72, keep=8, loops=3,
                        cen=(80, 420, 1600), lo=(0.40, 0.80, 0.995)),
}


def band_score(v, lo, ideal, hi):
    """1.0 at the ideal value, falling to 0 at the band edges; None outside the band."""
    if v < lo or v > hi:
        return None
    if v <= ideal:
        return (v - lo) / max(ideal - lo, 1e-9)
    return (hi - v) / max(hi - ideal, 1e-9)

DESC = {
    "rain": [("leaves", ["leaf", "leaves", "tree", "forest", "wald", "bosque"]),
             ("roof", ["roof", "window", "awning", "tent", "shelter", "balcony", "umbrella"]),
             ("puddle", ["puddle", "pond", "gutter", "drip", "drop"])],
    "wind": [("trees", ["tree", "forest", "leaves", "spruce", "pine"]),
             ("field", ["field", "meadow", "grass", "steppe", "prairie", "hill", "moor"]),
             ("coast", ["sea", "coast", "beach", "cliff", "shore"])],
    "sea": [("rocks", ["rock", "cliff", "stone", "boulder"]),
            ("beach", ["beach", "sand", "shore", "playa", "strand"])],
    "birds": [("dawn", ["dawn", "morning", "sunrise", "chorus", "early"]),
              ("meadow", ["meadow", "field", "grass", "prairie", "marsh", "wetland"]),
              ("forest", ["forest", "wood", "wald", "tree", "bosque"])],
    "insects": [("cicadas", ["cicada", "cigarra", "zikade"]),
                ("crickets", ["cricket", "grillo", "grillen", "grasshopper", "locust"]),
                ("night", ["night", "evening", "dusk", "nacht", "noche"])],
    "nature_misc": [("stream", ["stream", "creek", "brook", "river", "waterfall", "spring", "rapids"]),
                    ("splash", ["splash", "puddle", "drip", "drop", "water"]),
                    ("footsteps", ["footstep", "step", "walk", "walking", "gravel", "mud", "snow"]),
                    ("leaves", ["leaves", "leaf", "rustl", "grass", "branch", "bush"])],
    "room_tone": [("office", ["office", "studio", "work", "desk", "computer"]),
                  ("hall", ["hall", "library", "museum", "church", "corridor", "stair"]),
                  ("room", ["room", "apartment", "flat", "house", "indoor", "inside", "interior"])],
}


def descriptor(cat, title, meas):
    t = (title or "").lower()
    for name, keys in DESC.get(cat, []):
        if any(k in t for k in keys):
            base = name
            break
    else:
        base = {"birds": "day", "insects": "night", "nature_misc": "nature"}.get(cat, cat)
    r = meas["rms_db"]
    if cat == "rain":
        lvl = "heavy" if r > -24 else ("medium" if r > -32 else "light")
        return "%s_%s" % (lvl, base) if base != "rain" else lvl
    if cat == "wind":
        # peak-normalised, so rms_db is a crest proxy: dense/steady sits high, sparse sits low
        lvl = "gusty" if meas["_std"] > 3.0 else ("steady" if r > -26 else "breeze")
        return "%s_%s" % (lvl, base) if base != "wind" else lvl
    if cat == "sea":
        lvl = "close" if r > -24 else ("swell" if r > -30 else "distant")
        return "%s_%s" % (lvl, base) if base != "sea" else lvl
    if cat == "room_tone":
        lvl = "quiet" if r < -48 else "soft"
        return "%s_%s" % (base, lvl)
    return base


def analyse(path):
    x = decode(path, stereo=True)
    if x is None or len(x) < SR * 8:
        return None, None, None
    # drop decode edges of the truncated fragment
    cut = int(SR * 2.0)
    x = x[cut: len(x) - int(SR * 1.0)]
    if len(x) < SR * 8:
        return None, None, None
    if float(np.abs(x).max()) < 1e-4:
        return None, None, None
    m = x.mean(axis=1)
    st = spectral_stats(m)
    if st is None:
        return None, None, None
    return x, m, st


def score_window(st, a, b, cfg, cat):
    rms = st["rms"][a:b]
    hi = st["hi"][a:b]
    cen = st["cen"][a:b]
    lo = st["lo"][a:b]
    mid = st["mid"][a:b]
    rdb = 20 * np.log10(rms + 1e-9)
    med_db = float(np.median(rdb))
    std_db = float(np.std(rdb))
    hi_med = float(np.median(hi))
    lo_med = float(np.median(lo))
    cen_med = float(np.median(cen))
    cen_std = float(np.std(cen))
    trans = float(np.max(rms) / (np.median(rms) + 1e-9))
    sm = speech_mod(mid)
    bad = []
    if med_db < cfg["rms_min"]:
        bad.append("quiet")
    if med_db > -8:
        bad.append("hot")
    if hi_med > cfg["hi_max"]:
        bad.append("hiss")
    if hi_med < 0.004 and cat != "room_tone":
        bad.append("muffled")          # no HF at all -> rumble or a dead/band-limited source
    if std_db > cfg["std_max"]:
        bad.append("unsteady")
    if trans > (12 if cat in ("nature_misc", "sea") else 7):
        bad.append("transient")
    if sm > 0.38:
        bad.append("voice")
    if cen_std > cen_med * 0.55 and cat != "nature_misc":
        bad.append("spectral-drift")
    cen_sc = band_score(cen_med, *cfg["cen"])
    lo_sc = band_score(lo_med, *cfg["lo"])
    if cen_sc is None:
        bad.append("centroid-out-of-band")
    if lo_sc is None:
        bad.append("tonal-balance")
    if bad:
        return None
    cl = lambda v: max(0.0, min(1.0, v))
    s = 0.0
    s += 2.2 * cen_sc                              # right spectral character for the category
    s += 1.4 * lo_sc                               # body vs thinness
    s += 1.8 * cl((cfg["hi_max"] - hi_med) / cfg["hi_max"])
    s += 1.6 * cl(1 - std_db / cfg["std_max"])
    s += 1.2 * cl(1 - sm / 0.38)
    s += 1.0 * cl(1 - trans / 7.0)
    s += 1.0 * cl(1 - abs(med_db + 27) / 22.0)
    s += 0.7 * cl(1 - cen_std / (cen_med + 1e-6))
    return dict(score=s, rms_db=round(med_db, 2), std=round(std_db, 2),
                hi=round(hi_med, 3), lo=round(lo_med, 3), cen=int(cen_med),
                trans=round(trans, 2), sm=round(sm, 3))


def pick_windows(st, cfg, cat, nwin=2):
    n = len(st["rms"])
    out = []
    for d in cfg["durs"]:
        wf = int(d * FPS)
        if wf + 4 > n:
            continue
        hop = max(1, int(1.0 * FPS))
        cands = []
        for a in range(0, n - wf, hop):
            r = score_window(st, a, a + wf, cfg, cat)
            if r:
                r["a"] = a
                r["dur"] = d
                cands.append(r)
        if not cands:
            continue
        cands.sort(key=lambda r: -r["score"])
        chosen = []
        for c in cands:
            if all(abs(c["a"] - o["a"]) > wf * 0.8 for o in chosen):
                chosen.append(c)
            if len(chosen) >= nwin:
                break
        out.extend(chosen)
        break                       # longest workable duration wins
    return out


def render(x, a_frames, dur_s, hi_med, hi_target=0.30):
    """Gentle top-end trim; step the cutoff down until the >5 kHz share is under target."""
    a = int(a_frames * HOP)
    base = butter_hp(x[a: a + int(dur_s * SR)].copy(), 28.0)
    start = 11000 if hi_med < 0.16 else (10000 if hi_med < 0.24 else 9000)
    for fc in [f for f in (11000, 10000, 9000, 8000, 7000, 6200) if f <= start]:
        seg = butter_lp(base, fc)
        if measure(seg)["hi_ratio"] <= hi_target:
            return seg, fc
    return butter_lp(base, 6200), 6200


def process_cat(cat):
    cfg = CFG[cat]
    plan = json.load(open(os.path.join(DL, cat, "_plan.json")))
    outdir = os.path.join(AMB, cat)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(LIC, exist_ok=True)
    results = []
    stats = {"decoded": 0, "no_window": 0, "failed": 0}
    for rec in plan:
        p = rec["local"]
        if not os.path.exists(p):
            continue
        try:
            x, m, st = analyse(p)
        except Exception as e:
            stats["failed"] += 1
            continue
        if x is None:
            stats["failed"] += 1
            continue
        stats["decoded"] += 1
        wins = pick_windows(st, cfg, cat, nwin=2)
        if not wins:
            stats["no_window"] += 1
            continue
        for wi, w in enumerate(wins):
            results.append(dict(rec=rec, win=w, wi=wi, x=x))
    # rank across all candidates, prefer license quality and score, spread across sources
    results.sort(key=lambda r: -(r["win"]["score"] - 0.25 * r["rec"]["lic_rank"] - 0.5 * r["wi"]))
    keep, seen = [], {}
    for r in results:
        ident = r["rec"]["identifier"]
        if seen.get(ident, 0) >= (2 if len(results) < cfg["keep"] * 2 else 1):
            continue
        seen[ident] = seen.get(ident, 0) + 1
        keep.append(r)
        if len(keep) >= cfg["keep"]:
            break

    # Diversity pass: pure score converges on one flavour (dense, steady). `trans` (peak/median
    # RMS) separates sparse-and-peaky material from dense-and-flat, so reserve slots at both
    # ends of that axis, swapping out the weakest picks from the over-represented middle.
    if cat in ("rain", "wind", "sea", "nature_misc", "birds") and len(results) > cfg["keep"] + 4:
        tr = sorted(r["win"]["trans"] for r in results)
        lo_t, hi_t = tr[int(len(tr) * 0.25)], tr[int(len(tr) * 0.75)]
        for want_hi in (True, False):
            sel = lambda r: (r["win"]["trans"] >= hi_t) if want_hi else (r["win"]["trans"] <= lo_t)
            have = [r for r in keep if sel(r)]
            pool = [r for r in results if sel(r) and r not in keep
                    and r["rec"]["identifier"] not in {k["rec"]["identifier"] for k in keep}]
            while len(have) < 2 and pool:
                mid = [r for r in keep if not sel(r)]
                if len(mid) < 3:
                    break
                keep.remove(mid[-1])
                add = pool.pop(0)
                keep.append(add)
                have.append(add)

    items, used_names = [], {}
    for ki, r in enumerate(keep):
        rec, w = r["rec"], r["win"]
        seg, fc = render(r["x"], w["a"], w["dur"], w["hi"])
        seg_n = normalize_peak(fade(seg), -3.0)
        meas = measure(seg_n)
        meas["_std"] = w["std"]
        d = descriptor(cat, rec["title"], meas)
        base = re.sub(r"[^a-z0-9_]+", "_", ("%s_%s" % (cat, d)).lower()).strip("_")
        k = used_names.get(base, 0) + 1
        used_names[base] = k
        name = "%s_%02d" % (base, k)
        fp = os.path.join(outdir, name + ".wav")
        write_wav(fp, seg_n)
        mm = measure(seg_n)
        item = dict(file=os.path.relpath(fp, PROJ), cat=cat,
                    tags=sorted(set(d.split("_") + [t for t in rec.get("terms", [])[:2]])),
                    dur=mm["dur"], peak_db=mm["peak_db"], rms_db=mm["rms_db"],
                    centroid_hz=mm["centroid_hz"], hi_ratio=mm["hi_ratio"],
                    lo_ratio=mm["lo_ratio"], lowpass_hz=fc, loop=False,
                    src=rec["src"], title=rec["title"], creator=rec.get("creator", ""),
                    license=rec["license"], license_url=rec.get("licenseurl", ""))
        items.append(item)
        # loop variant
        if mm["dur"] >= 12 and ki < cfg.get("loops", 4):
            best = None
            for xf in (1.5, 2.0, 1.0, 2.5):
                lp, rep = make_loop(seg, xf=xf)
                if lp is None:
                    continue
                if best is None or rep["joint_rms_db"] < best[2]["joint_rms_db"]:
                    best = (lp, xf, rep)
                if rep["joint_rms_db"] < 0.8:
                    break
            if best and best[2]["joint_rms_db"] < 1.5:
                lp = normalize_peak(best[0], -3.0)
                lf = os.path.join(outdir, name + "_loop.wav")
                write_wav(lf, lp)
                lm = measure(lp)
                li = dict(item)
                li.update(file=os.path.relpath(lf, PROJ), dur=lm["dur"], loop=True,
                          peak_db=lm["peak_db"], rms_db=lm["rms_db"],
                          centroid_hz=lm["centroid_hz"], hi_ratio=lm["hi_ratio"],
                          lo_ratio=lm["lo_ratio"], crossfade_s=best[1],
                          loop_joint_db=best[2]["joint_rms_db"],
                          tags=sorted(set(item["tags"] + ["loopable"])))
                items.append(li)
    print("%-12s decoded=%d no_window=%d failed=%d -> %d files (%d loops)" % (
        cat, stats["decoded"], stats["no_window"], stats["failed"],
        len(items), sum(1 for i in items if i["loop"])))
    json.dump(items, open(os.path.join(SP, "items_%s.json" % cat), "w"), indent=1)
    return items


if __name__ == "__main__":
    for c in sys.argv[1:]:
        process_cat(c)
