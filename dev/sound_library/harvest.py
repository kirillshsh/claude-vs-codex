import os, sys, json, math
AUD = '/path/to/claude-vs-codex/audio'
sys.path.insert(0, AUD)
import synth
synth.save_wav = lambda *a, **k: None
synth.write_manifest = lambda *a, **k: None

import importlib
out = {}

A = importlib.import_module('make_sfx_a')
A.save_wav = lambda *a, **k: None
for fn in ('build_s0','build_s0_type','build_s0_amb','build_s1','build_s1_amb','build_s2_amb','build_s2'):
    getattr(A, fn)()
out['a'] = [list(e) for e in sorted(A.EVENTS)]

B = importlib.import_module('make_sfx_b')
B.build_s3(); B.build_s4()
out['b'] = [list(e) for e in sorted(B.EVENTS)]

print(json.dumps(out, ensure_ascii=False, indent=0))
