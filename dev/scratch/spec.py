import os, sys, json, glob
import numpy as np
AUD='/path/to/claude-vs-codex/audio'
ROOT=os.path.dirname(AUD); sys.path.insert(0,AUD)
from synth import SR, load_wav
print(f'{"файл":22s}{"шина":6s}{"<200":>7s}{"200-2k":>8s}{"2.5-6k":>8s}{">6k":>7s}{"центроид":>10s}')
for cf in sorted(glob.glob(os.path.join(AUD,'cues','*.json'))):
    for s in json.load(open(cf))['stems']:
        x = load_wav(os.path.join(ROOT, s['file'])).mean(1)
        X = np.abs(np.fft.rfft(x)); f = np.fft.rfftfreq(len(x), 1/SR)
        tot = X.sum()+1e-9
        b = lambda lo,hi: 100*X[(f>=lo)&(f<hi)].sum()/tot
        cen = float((f*X).sum()/tot)
        print(f'{os.path.basename(s["file"]):22s}{s.get("bus","sfx"):6s}{b(20,200):7.1f}{b(200,2000):8.1f}{b(2500,6000):8.1f}{b(6000,20000):7.1f}{cen:10.0f}')
