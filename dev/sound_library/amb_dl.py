#!/usr/bin/env python3
"""Download capped mp3 fragments for one category with 8 parallel curl workers."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

SP = "/path/to/scratchpad"
PROJ = SP + "/cartoon"
DL = PROJ + "/audio/_dl_amb"
SHORT = SP + "/amb_shortlist.json"


def fetch(job):
    url, dst, rng = job
    if os.path.exists(dst) and os.path.getsize(dst) > 200_000:
        return dst, True, "cached"
    cmd = ["curl", "-sL", "--retry", "4", "--retry-delay", "2", "--retry-connrefused",
           "--connect-timeout", "10", "--max-time", "150", "-r", rng, "-o", dst, url]
    try:
        subprocess.run(cmd, timeout=170, capture_output=True)
    except Exception as e:
        return dst, False, str(e)[:60]
    ok = os.path.exists(dst) and os.path.getsize(dst) > 100_000
    if not ok and os.path.exists(dst):
        os.remove(dst)
    return dst, ok, "ok" if ok else "empty"


def main():
    cat = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 3_200_000
    recs = [r for r in json.load(open(SHORT)) if cat in r.get("pick_cats", [])]
    outdir = os.path.join(DL, cat)
    os.makedirs(outdir, exist_ok=True)
    jobs, plan = [], []
    for r in recs:
        ident = r["identifier"]
        ext = ".mp3" if "mp3" in r["fmt"] else (os.path.splitext(r["file"])[1] or ".audio")
        dst = os.path.join(outdir, ident + ext)
        size = r.get("fsize") or 0
        if size > cap * 2.2:
            off = int(size * 0.12) // 1024 * 1024
            rng = "%d-%d" % (off, off + cap)
        else:
            off, rng = 0, "0-%d" % cap
        r["byte_off"] = off
        r["local"] = dst
        plan.append(r)
        jobs.append((r["url"], dst, rng))

    fails = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (dst, ok, why) in enumerate(ex.map(fetch, jobs)):
            if not ok:
                fails.append((os.path.basename(dst), why))
            if (i + 1) % 15 == 0:
                sys.stderr.write("  %s %d/%d\n" % (cat, i + 1, len(jobs)))

    plan = [r for r in plan if os.path.exists(r["local"])]
    json.dump(plan, open(os.path.join(outdir, "_plan.json"), "w"), indent=1)
    tot = sum(os.path.getsize(r["local"]) for r in plan)
    print("done %s: %d/%d files, %.0f MB, %d fails" % (cat, len(plan), len(jobs), tot / 1e6, len(fails)))
    for f in fails[:10]:
        print("   FAIL", f[0], f[1])


if __name__ == "__main__":
    main()
