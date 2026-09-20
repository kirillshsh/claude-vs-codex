import json, re, sys
import numpy as np
SCR='/path/to/scratchpad/'
d=json.load(open(SCR+'harvest_out.json'))
ev=[]   # (t, src, label, peak)
BED=('ветер','гул','дробь','шорох','встречный ветер','рёв толпы','гроза','пульсирующ')
for t,what,snd,peak,pan in d['a']:
    ev.append((t,'A',what,peak))
for t,screen,sound in d['b']:
    ev.append((t,'B',screen,None))
cur=None
for line in open(SCR+'harvest_c.txt'):
    m=re.match(r'--- (\S+) \((\w+)\)', line)
    if m: cur=(m.group(1), m.group(2)); continue
    m=re.match(r'\s+(\d+\.\d+)\s+пик (\d+\.\d+)', line)
    if m and cur and cur[1] in ('sfx','ui'):
        ev.append((float(m.group(1)),'C',cur[0],float(m.group(2))))
ev.sort()
print('всего строк событий (sfx/ui, без амбиентов):', len(ev))

# кластеры «одновременности»: события в окне 0.12 с
cl=[]; cure=[ev[0]]
for e in ev[1:]:
    if e[0]-cure[-1][0] <= 0.12: cure.append(e)
    else: cl.append(cure); cure=[e]
cl.append(cure)
print('кластеров (окно 0.12с):', len(cl))
big=[c for c in cl if len(c)>=3]
print(f'\nкластеров с 3+ одновременными событиями: {len(big)}')
for c in big:
    print(f'  {c[0][0]:6.2f}–{c[-1][0]:6.2f} ({len(c)}): ' + ' | '.join(f'{x[1]}:{x[2][:34]}' for x in c))
print()
# плотность моментов в секунду (моменты = кластеры)
cnt=np.zeros(94,int)
for c in cl: cnt[int(c[0][0])]+=1
print('моментов в секунду:', ' '.join(f'{s}:{cnt[s]}' for s in range(93) if cnt[s]))
print('\nсекунд с >=5 моментами:', ' '.join(f'{s}s={cnt[s]}' for s in range(93) if cnt[s]>=5))
# интервалы между кластерами
ts=np.array([c[0][0] for c in cl]); dd=np.diff(ts)
print(f'\nинтервал между моментами: медиана {np.median(dd):.3f}, p10 {np.percentile(dd,10):.3f}, p25 {np.percentile(dd,25):.3f}')
print(f'доля < 0.20с: {100*np.mean(dd<0.20):.0f}%  < 0.35с: {100*np.mean(dd<0.35):.0f}%')
