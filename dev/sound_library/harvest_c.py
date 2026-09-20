import os, sys, json
import numpy as np
AUD = '/path/to/claude-vs-codex/audio'
sys.path.insert(0, AUD)
import synth
synth.save_wav = lambda *a, **k: None
synth.write_manifest = lambda *a, **k: None

_add = synth.Track.add
def add(self, x, t, gain=1.0, pan=0.0):
    if not hasattr(self, '_ev'): self._ev = []
    pk = float(np.max(np.abs(np.asarray(x)))) if len(np.asarray(x)) else 0.0
    self._ev.append((float(t), pk * gain, float(pan)))
    return _add(self, x, t, gain=gain, pan=pan)
synth.Track.add = add

LAST = {}
_st = synth.Track.stereo
def stereo(self, *a, **k):
    LAST['tr'] = self
    return _st(self, *a, **k)
synth.Track.stereo = stereo

import importlib
C = importlib.import_module('make_sfx_c')
C.save_wav = lambda *a, **k: None
C.write_manifest = lambda *a, **k: None
OUT = []
_emit = C.emit
def emit(name, buf, t, bus='sfx', tail=0.0, head=0.0):
    tr = LAST.get('tr')
    evs = getattr(tr, '_ev', []) if tr is not None else []
    OUT.append({'name': name, 't0': t, 'bus': bus,
                'ev': sorted([(round(t + e[0], 3), round(e[1], 3), round(e[2], 2)) for e in evs])})
    LAST['tr'] = None
    return _emit(name, buf, t, bus=bus, tail=tail, head=head)
C.emit = emit
for fn in ('build_rain','build_rain_leaf','build_thunder','build_s5_voices','build_s5_foley',
           'build_s5_magic','build_s5_after','build_s6_amb','build_s6_ui','build_s6_swoosh',
           'build_s6_flight','build_s7_amb','build_s7_cicadas','build_s7_legs','build_s7_stars',
           'build_s7_events'):
    getattr(C, fn)()
for s in OUT:
    print(f"--- {s['name']} ({s['bus']}) t0={s['t0']}  событий={len(s['ev'])}")
    for t, pk, pan in s['ev']:
        print(f'   {t:7.2f}  пик {pk:.2f}  pan {pan:+.2f}')
