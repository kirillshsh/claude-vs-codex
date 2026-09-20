import json, os, sys, numpy as np
ROOT='/path/to/claude-vs-codex'
sys.path.insert(0, os.path.join(ROOT,'audio'))
from synth import SR, load_wav, n_samples
man=json.load(open(os.path.join(ROOT,'audio/cues/music.json')))['stems']
buf=np.zeros((n_samples(93.0),2),np.float32)
print('стем                t0      t1     длит   пик')
prev=None
for s in man:
    x=load_wav(os.path.join(ROOT,s['file']))
    fin,fout=s['fin'],s['fout']
    if fin:
        k=min(n_samples(fin),len(x)); x[:k]*=np.linspace(0,1,k,dtype=np.float32)[:,None]
    if fout:
        k=min(n_samples(fout),len(x)); x[-k:]*=np.linspace(1,0,k,dtype=np.float32)[:,None]
    i=int(round(s['t']*SR)); x=x[:max(0,len(buf)-i)]; buf[i:i+len(x)]+=x
    t1=s['t']+len(x)/SR
    print(f"{os.path.basename(s['file']):16s} {s['t']:6.2f} {t1:6.3f} {len(x)/SR:6.3f} {np.max(np.abs(x)):.3f}"
          + ('' if prev is None or abs(prev-s['t'])<1e-6 else f'  <- ЗАЗОР {s["t"]-prev:+.3f}'))
    prev=t1
mono=buf.mean(1)
# щелчки: максимальный скачок между соседними сэмплами
d=np.abs(np.diff(mono)); worst=np.argsort(d)[-6:][::-1]
print('\nмакс. скачки (возможные щелчки):', ', '.join(f'{w/SR:.3f}c={d[w]:.3f}' for w in worst))
# тишина: окна 0.25 c ниже -45 dB
n=n_samples(0.25); gaps=[]
for k in range(len(mono)//n):
    seg=mono[k*n:(k+1)*n]; r=np.sqrt(np.mean(seg**2))
    if r<10**(-45/20): gaps.append(k*0.25)
runs=[]
for g in gaps:
    if runs and abs(g-runs[-1][1])<0.26: runs[-1][1]=g+0.25
    else: runs.append([g,g+0.25])
print('окна тише -45 dB:', [f'{a:.2f}-{b:.2f}' for a,b in runs] or 'нет')
print(f'хвост: последние 0.5 c RMS {20*np.log10(np.sqrt(np.mean(mono[-n_samples(0.5):]**2))+1e-12):.1f} dB, '
      f'последние 100 сэмплов макс {np.max(np.abs(mono[-100:])):.5f}')
print(f'общая длительность музыки: {prev:.3f} c')
