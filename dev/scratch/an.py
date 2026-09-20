import os, sys, json, glob
import numpy as np
AUD = '/path/to/claude-vs-codex/audio'
ROOT = os.path.dirname(AUD)
sys.path.insert(0, AUD)
from synth import SR, load_wav, n_samples

DUR = 93.0
n = n_samples(DUR)
buses = {}
rows = []
for cf in sorted(glob.glob(os.path.join(AUD, 'cues', '*.json'))):
    man = json.load(open(cf))
    for s in man['stems']:
        p = os.path.join(ROOT, s['file'])
        x = load_wav(p) * float(s.get('gain', 1.0))
        bus = s.get('bus', 'sfx')
        i = int(round(float(s['t']) * SR))
        x = x[:max(0, n - i)]
        pk = float(np.max(np.abs(x))) if len(x) else 0
        r = float(np.sqrt(np.mean(x.mean(1)**2))) if len(x) else 0
        rows.append((float(s['t']), float(s['t'])+len(x)/SR, bus, os.path.basename(p), pk, r))
        buses.setdefault(bus, np.zeros((n,2), np.float32))[i:i+len(x)] += x

def db(v): return -99.0 if v <= 1e-7 else 20*np.log10(v)

print('=== СТЕМЫ (сырые, до BUS_GAIN) ===')
print(f'{"нач":>6} {"кон":>6} {"шина":5s} {"файл":22s} {"пик dB":>7s} {"RMS dB":>7s}')
for a,b,bus,nm,pk,r in sorted(rows):
    print(f'{a:6.2f} {b:6.2f} {bus:5s} {nm:22s} {db(pk):7.1f} {db(r):7.1f}')

BUS_GAIN = {'music':0.82,'sfx':0.80,'amb':0.50,'ui':0.62}
print('\n=== ШИНЫ после BUS_GAIN: RMS по 0.5 с ===')
W = n_samples(0.5)
names = ['music','sfx','ui','amb']
print('  t   ' + ''.join(f'{x:>8s}' for x in names) + '   sfx+ui  запас_sfx_над_муз')
for k in range(int(DUR/0.5)):
    line = f'{k*0.5:6.2f}'
    vals = {}
    for bname in names:
        v = buses.get(bname)
        if v is None: vals[bname]=0; line += f'{-99:8.1f}'; continue
        seg = v[k*W:(k+1)*W].mean(1) * BUS_GAIN[bname]
        r = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0
        vals[bname]=r
        line += f'{db(r):8.1f}'
    su = np.sqrt(vals['sfx']**2 + vals['ui']**2)
    line += f'{db(su):9.1f}{db(su)-db(vals["music"]):9.1f}'
    print(line)
