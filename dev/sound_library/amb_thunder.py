#!/usr/bin/env python3
"""Thunder is event-shaped, not steady: detect transients numerically and cut around them."""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amb_dsp import (SR, HOP, decode, spectral_stats, speech_mod, db, butter_lp, butter_hp,
                     normalize_peak, fade, write_wav, measure)

SP = "/path/to/scratchpad"
PROJ = SP + "/cartoon"
DL = PROJ + "/audio/_dl_amb"
AMB = PROJ + "/audio/samples/ambience"
FPS = SR / HOP
KEEP = 16


def rolling_median(v, w):
    n = len(v)
    out = np.empty(n)
    h = w // 2
    for i in range(n):
        out[i] = np.median(v[max(0, i - h): min(n, i + h + 1)])
    return out


def find_events(st):
    rms_db = 20 * np.log10(st["rms"] + 1e-9)
    # a long roll must not be swallowed by its own baseline -> wide median window
    base = rolling_median(rms_db, int(15 * FPS))
    prom = rms_db - base
    lo = st["lo"]
    cen = st["cen"]
    n = len(prom)
    cand = np.where((prom > 6.0) & (lo > 0.10))[0]
    if len(cand) == 0:
        return []
    groups, cur = [], [cand[0]]
    for i in cand[1:]:
        if i - cur[-1] <= int(2.0 * FPS):
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)

    events = []
    for g in groups:
        pk = g[int(np.argmax(prom[g]))]
        p = float(prom[pk])
        if p < 7.0:
            continue
        # onset: walk back to where the level leaves the baseline
        a = pk
        while a > 0 and prom[a] > 2.0 and pk - a < int(3.0 * FPS):
            a -= 1
        # tail: walk forward until it settles back (thunder rolls for a long time)
        b = pk
        while b < n - 1 and prom[b] > 1.2 and b - pk < int(13.0 * FPS):
            b += 1
        dur = (b - a) / FPS
        if dur < 2.0 or dur > 15.0:
            continue
        rise = (pk - a) / FPS
        decay = (b - pk) / FPS
        # impulsive events decay far more slowly than they rise; vehicles are near-symmetric
        if decay < rise * 1.3:
            continue
        seg_hi = float(np.median(st["hi"][a:b]))
        seg_cen = float(np.median(cen[a:b]))
        seg_lo = float(np.median(lo[a:b]))
        if seg_hi > 0.35:
            continue
        # a body with literally no energy above 5 kHz is engine/handling rumble, not thunder
        if seg_cen < 130 or seg_hi < 0.0015:
            continue
        if speech_mod(st["mid"][a:b]) > 0.45:
            continue
        # sharpness of the attack: biggest frame-to-frame jump inside the rise
        lead = rms_db[max(0, pk - int(1.2 * FPS)): pk + 1]
        attack = float(np.max(np.diff(lead))) if len(lead) > 2 else 0.0
        bed_db = float(np.median(rms_db[max(0, a - int(4 * FPS)): a])) if a > int(2 * FPS) else -99
        events.append(dict(a=a, pk=pk, b=b, prom=round(p, 2), rise=round(rise, 2),
                           decay=round(decay, 2), attack=round(attack, 2),
                           dur=round(dur, 2), hi=round(seg_hi, 3), cen=int(seg_cen),
                           lo=round(seg_lo, 3), bed_db=round(bed_db, 1),
                           score=p + 3.0 * seg_lo + min(dur, 9.0) * 0.8
                                 + min(attack, 12.0) * 0.4 - seg_hi * 8))
    return events


def classify(ev):
    """close = sharp bright crack; distant = slow low roll."""
    if ev["attack"] > 5.0 and ev["rise"] < 0.7 and ev["cen"] > 550 and ev["prom"] > 11:
        return "close"
    if ev["cen"] < 420 or ev["rise"] > 1.0:
        return "distant_rumble"
    if ev["cen"] < 900:
        return "distant"
    return "mid"


def main():
    plan = json.load(open(os.path.join(DL, "thunder", "_plan.json")))
    outdir = os.path.join(AMB, "thunder")
    os.makedirs(outdir, exist_ok=True)
    cands = []
    dec = 0
    for rec in plan:
        p = rec["local"]
        if not os.path.exists(p):
            continue
        x = decode(p, stereo=True)
        if x is None or len(x) < SR * 10:
            continue
        x = x[int(SR * 2.0): len(x) - int(SR * 1.0)]
        m = x.mean(axis=1)
        st = spectral_stats(m)
        if st is None:
            continue
        dec += 1
        evs = find_events(st)
        evs.sort(key=lambda e: -e["score"])
        for k, e in enumerate(evs[:5]):
            cands.append(dict(rec=rec, ev=e, x=x, rank=k))
    cands.sort(key=lambda c: -(c["ev"]["score"] - 1.5 * c["rec"]["lic_rank"] - 2.0 * c["rank"]))

    # keep a spread of kinds, max 2 per source recording
    want = {"close": 6, "mid": 4, "distant": 4, "distant_rumble": 4}
    got, seen, keep = {}, {}, []
    for c in cands:
        k = classify(c["ev"])
        ident = c["rec"]["identifier"]
        if seen.get(ident, 0) >= 2 or got.get(k, 0) >= want.get(k, 3):
            continue
        seen[ident] = seen.get(ident, 0) + 1
        got[k] = got.get(k, 0) + 1
        keep.append((k, c))
        if len(keep) >= KEEP:
            break
    # top up with whatever is left if a kind was short
    if len(keep) < 10:
        for c in cands:
            if any(c is kc for _, kc in keep):
                continue
            ident = c["rec"]["identifier"]
            if seen.get(ident, 0) >= 2:
                continue
            seen[ident] = seen.get(ident, 0) + 1
            keep.append((classify(c["ev"]), c))
            if len(keep) >= 12:
                break

    items, names = [], {}
    for kind, c in keep:
        ev, rec, x = c["ev"], c["rec"], c["x"]
        a = max(0, int((ev["a"] / FPS - 0.8) * SR))
        b = min(len(x), int((ev["b"] / FPS + 0.8) * SR))
        seg = butter_hp(x[a:b].copy(), 22.0)
        fc = 10000 if ev["hi"] < 0.2 else 9000
        for f in (fc, 8000, 7000):
            s2 = butter_lp(seg, f)
            if measure(s2)["hi_ratio"] <= 0.30:
                seg, fc = s2, f
                break
        else:
            seg, fc = butter_lp(seg, 7000), 7000
        seg = normalize_peak(fade(seg, ms=40), -3.0)
        base = "thunder_%s" % kind
        n = names.get(base, 0) + 1
        names[base] = n
        name = "%s_%02d" % (base, n)
        fp = os.path.join(outdir, name + ".wav")
        write_wav(fp, seg)
        mm = measure(seg)
        tags = [kind] + ([t for t in ("crack", "rumble") if
                          (t == "crack" and ev["rise"] < 0.45) or (t == "rumble" and ev["dur"] > 6)])
        if ev["bed_db"] > -45:
            tags.append("rain_bed")
        items.append(dict(file=os.path.relpath(fp, PROJ), cat="thunder",
                          tags=sorted(set(tags)), dur=mm["dur"], peak_db=mm["peak_db"],
                          rms_db=mm["rms_db"], centroid_hz=mm["centroid_hz"],
                          hi_ratio=mm["hi_ratio"], lo_ratio=mm["lo_ratio"], lowpass_hz=fc,
                          loop=False, prominence_db=ev["prom"], rise_s=ev["rise"],
                          decay_s=ev["decay"], attack_db=ev["attack"],
                          src=rec["src"], title=rec["title"], creator=rec.get("creator", ""),
                          license=rec["license"], license_url=rec.get("licenseurl", "")))
    print("thunder decoded=%d events=%d -> %d files" % (dec, len(cands), len(items)))
    for i in items:
        print("  %-30s %5.1fs prom%5.1f rise%.2f atk%5.1f cen%5d hi%.3f" % (
            os.path.basename(i["file"]), i["dur"], i["prominence_db"], i["rise_s"],
            i.get("attack_db", 0), i["centroid_hz"], i["hi_ratio"]))
    json.dump(items, open(os.path.join(SP, "items_thunder.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
