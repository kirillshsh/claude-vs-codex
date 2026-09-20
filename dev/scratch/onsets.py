import os, sys, json, glob
import numpy as np
AUD = '/path/to/claude-vs-codex/audio'
ROOT = os.path.dirname(AUD)
sys.path.insert(0, AUD)
from synth import SR, load_wav, n_samples
DUR = 93.0; n = n_samples(DUR)
BUS_GAIN = {'music':0.82,'sfx':0.80,'amb':0.50,'ui':0.62}
buses = {}
for cf in sorted(glob.glob(os.path.join(AUD,'cues','*.json'))):
    for s in json.load(open(cf))['stems']:
        x = load_wav(os.path.join(ROOT, s['file'])) * float(s.get('gain',1.0))
        bus = s.get('bus','sfx'); i = int(round(float(s['t'])*SR)); x = x[:max(0,n-i)]
        buses.setdefault(bus, np.zeros((n,2),np.float32))[i:i+len(x)] += x

sfx = buses['sfx']*BUS_GAIN['sfx'] + buses['ui']*BUS_GAIN['ui']
mono = np.abs(sfx).max(1)
# спектральный поток по кадрам 512
H = 512
nf = len(mono)//H
X = np.abs(np.fft.rfft(sfx.mean(1)[:nf*H].reshape(nf,H)*np.hanning(H), axis=1))
flux = np.maximum(0, np.diff(X, axis=0)).sum(1)
flux = np.concatenate([[0], flux])
# порог: медиана по скользящему окну
thr = np.zeros_like(flux)
W = 40
for i in range(len(flux)):
    a,b = max(0,i-W), min(len(flux), i+W)
    thr[i] = np.median(flux[a:b])*2.2 + np.percentile(flux,60)*0.45
on = []
last = -1e9
for i in range(1,len(flux)-1):
    if flux[i] > thr[i] and flux[i] >= flux[i-1] and flux[i] > flux[i+1]:
        t = i*H/SR
        if t - last > 0.045:
            on.append(t); last = t
on = np.array(on)
print(f'всего атак (spectral flux, sfx+ui): {len(on)}')
print('\nатак в каждую секунду (0 не печатаю):')
cnt = np.zeros(94, int)
for t in on: cnt[int(t)] += 1
line=[]
for s in range(93):
    if cnt[s]: line.append(f'{s}:{cnt[s]}')
print(' '.join(line))
print('\nсекунды с >=4 атаками:')
print(' '.join(f'{s}s={cnt[s]}' for s in range(93) if cnt[s]>=4))
print('\nмакс. атак в скользящем окне 1.0 с:')
worst=[]
for k in range(0,int(93/0.25)):
    t0=k*0.25; c=int(((on>=t0)&(on<t0+1.0)).sum())
    worst.append((c,t0))
worst.sort(reverse=True)
for c,t0 in worst[:25]: print(f'  {t0:6.2f}–{t0+1:6.2f}: {c}')
# межатаковые интервалы
d = np.diff(on)
print(f'\nинтервалы между атаками: медиана {np.median(d):.3f}s, 10% {np.percentile(d,10):.3f}, 25% {np.percentile(d,25):.3f}')
print(f'доля интервалов < 0.15s: {100*np.mean(d<0.15):.0f}%, < 0.35s: {100*np.mean(d<0.35):.0f}%')
