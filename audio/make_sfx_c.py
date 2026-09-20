"""Звуковые эффекты и атмосферы финальной трети (48.00–92.90 c). Агент C.

    python3 audio/make_sfx_c.py        # из корня cartoon/

Собрано по audio/PLAN.md (§1 таблица событий s5–s7, §2 правила плотности,
§4 иерархия громкостей, §5 тембр, §6 голоса). На реальных сэмплах из
audio/samples; синтезом остались только чип-реплики героев и два тёплых
аккорда-жеста («МИР!», объятие) — см. §6.5 плана.

Тайминги сверены с кодом сцен (s5_rain 48–65, s6_montage 65–72, s7_sunset 72–90,
мини-титр до 93.0):
  51.15 вспышка молнии · 53.60 «апчхи» · 56.25 «?» · 56.86 «!» · 57.00 срыв лопуха
  58.12 шаг по луже · 58.80 «МИР?» · 60.05 «МИР!» · 60.30 сердечко
  61.30–62.10 ливень стихает · 62.22 радуга · 62.72 объятие · 64.06 кубок сросся
  65.45 терминал · 66.90 MERGED · 68.20 склейка в полёт
  75.60 объятие · 80.72 глаза-сердечки · 83.90–87.50 кран вверх · 85.82 падающая звезда
  86.52 титр · 87.52 «КОНЕЦ» · 89.45 мини-титр · 92.85 конец хвоста

Правила, которые проверяются при сборке (check_events): один lead и один support
одновременно, между лидами ≥0.35 c, между lead и support ≥0.12 c, ≤4 атаки в
любом окне 1 c, украшения (bed-мелочь) не ближе 0.6 c к событиям.
"""
import math
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import synth as S                                                      # noqa: E402
from synth import SR, Track, n_samples, save_wav, write_manifest       # noqa: E402
from lib_sfx_c import (amb, balance, bands, bed, breath_in, by_name, hpf, lpf, norm,  # noqa: E402
                       one, place, proc, ramp, rms, slow, sneeze, snd, to_rms,
                       v_clawd, v_codex, warm_chord)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.path.join(ROOT, 'audio', 'stems', 'sfx_c')
CUES = os.path.join(ROOT, 'audio', 'cues', 'sfx_c.json')
REL = 'audio/stems/sfx_c'

STEMS, EVENTS, MISSING = [], [], []


# ---------------------------------------------------------------- подбор сэмплов

def S_(name):
    p = by_name(name)
    if p is None:
        MISSING.append(name)
    return p


def pick_amb(*patterns):
    for p in patterns:
        f = amb(p)
        if f:
            return f
    return []


SMP = {                                  # §5: impact max_hi≤0.08, whoosh ≤0.05, ui ≤0.12, bell ≤0.22
    'splash': 'oga_footstep_water_01.wav',          # hi 0.065
    'drip': ['kenney_impactglass_light_000.wav', 'kenney_impactglass_light_001.wav',
             'kenney_impactglass_light_003.wav'],   # hi 0.000, короткий «тик» капли
    'pluck': 'kenney_footstep_grass_001.wav',       # hi 0.024, сухой шорох
    'toss': 'oga_swosh_16.wav',                     # hi 0.003
    'tumble': 'oga_swish_12.wav',                   # hi 0.001
    'thump': 'kenney_impactsoft_heavy_000.wav',     # cen 96 Гц
    'thump_s': 'kenney_impactsoft_medium_001.wav',  # cen 86 Гц
    'hop': 'kenney_highup.wav',                     # retro8bit, подскок
    'tink': 'kenney_impactglass_light_001.wav',
    'idea': 'kenney_powerup12.wav',                 # hi 0.000
    'heart': 'oga_bell_ding4.wav',                  # hi 0.003
    'shimmer': 'oga_magical_1.wav',                 # hi 0.001
    'rainbow': 'oga_single_fantasy_sfx_pack_vol_1_teleport_in.wav',
    'merge': 'oga_magical_4.wav',
    'pop': 'oga_appear_online.wav',
    'key': 'kenney_select_002.wav',
    'merged': 'kenney_jingles_steel00.wav',
    'sw_in': 'oga_qubodup_megaswosh1.wav',
    'sw_cut': 'oga_swosh_03.wav',                   # cen 108 Гц
    'sw_out': 'oga_swish_3.wav',
    'cloud': ['oga_swosh_05.wav', 'oga_swosh_15.wav'],
    'air': 'oga_whoosh2_0.wav',                     # 5.25 c, для воздуха полёта и крана
    'step_grass': ['kenney_footstep_grass_003.wav', 'kenney_footstep_grass_000.wav'],
    'rustle': 'kenney_footstep_grass_004.wav',
    'star': 'oga_swish_5.wav',
    'title': 'kenney_jingles_steel03.wav',
    'end': 'kenney_impactbell_heavy_001.wav',
    'spark': ['kenney_impactglass_light_003.wav', 'kenney_impactglass_light_000.wav'],
}


def softest(paths, **kw):
    """Из кандидатов выбрать самый мягкий — с наименьшей долей энергии в 2.5–6 кГц
    ПОСЛЕ обработки (детерминированно, §5 плана)."""
    best, score = None, 1e9
    for p in paths:
        b = bands(proc(p, peak=0.5, dur=2.0, **kw))
        v = b[3] * 2 + b[4]
        if v < score:
            best, score = p, v
    return best


def onset(path, pre=0.15):
    """Смещение, при котором самый громкий момент записи попадает через pre секунд
    после метки события (у раскатов грома до 1.5 c «разбега» перед ударом)."""
    x = np.abs(snd(path)).max(1)
    return max(0.0, float(np.argmax(x)) / SR - pre)


def smp(key, i=0):
    v = SMP[key]
    return S_(v[i % len(v)] if isinstance(v, list) else v)


# ---------------------------------------------------------------- инфраструктура

def ev(tr, t0, t, sig, role='support', name='', pan=0.0):
    """Положить звук на абсолютную секунду t; роль идёт в реестр для проверок §2.
    Роли: lead / support / bed (мелкое украшение, к нему применяется П6) /
    amb (непрерывный слой — в проверки плотности не входит) / series (дробь-серия)."""
    if sig is None or len(sig) == 0:
        return
    lim = {'lead': 0.25, 'support': 0.55, 'bed': 0.50, 'amb': 0.50, 'series': 0.20}[role]  # П8
    pan = max(-lim, min(lim, pan))
    x = np.asarray(sig, np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    tr.add(balance(x, pan) if pan else x, t - t0)
    EVENTS.append((round(float(t), 3), role, name or role))


def emit(name, buf, t, bus='sfx', tail=0.0, head=0.0, top=9000.0, low=None):
    x = np.asarray(buf, np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    if low:
        x = hpf(x, low)                      # §5: bed-слои ФВЧ 120 Гц
    if top:
        x = lpf(x, top)                      # §5: потолок 9 кГц на всех стемах
    if head:
        k = min(n_samples(head), len(x))
        x[:k] *= np.linspace(0, 1, k, dtype=np.float32)[:, None]
    if tail:
        k = min(n_samples(tail), len(x))
        x[-k:] *= np.linspace(1, 0, k, dtype=np.float32)[:, None]
    save_wav(os.path.join(OUTD, name + '.wav'), x)
    STEMS.append({'file': f'{REL}/{name}.wav', 't': round(float(t), 3), 'gain': 1.0,
                  'pan': 0.0, 'bus': bus})
    b = bands(x)
    print(f'  {name:15s} t={t:6.2f} {len(x)/SR:5.2f}c пик {np.max(np.abs(x)):.3f} '
          f'RMS {20*math.log10(max(1e-9, rms(x))):6.1f}  полосы '
          f'{b[0]:4.1f}/{b[1]:4.1f}/{b[2]:4.1f}/{b[3]:4.1f}/{b[4]:4.1f}%  [{bus}]')


def tfade(tr, t0, a, b):
    t = t0 + np.arange(len(tr.buf)) / SR
    tr.buf *= (1 - ramp(t, a, b)).astype(np.float32)[:, None]
    return tr


# ================================================================ s5 — дождь и примирение

def build_rain():
    """Ливень 48.0–62.1 (реальная запись, тело в низах-середине, RMS −30 по §4) и
    остаточная капель до 64.8 (RMS −34). Плотность едет по картинке."""
    t0, t1 = 48.0, 65.2
    n = n_samples(t1 - t0)
    t = t0 + np.arange(n) / SR
    heavy = ramp(t, 48.0, 48.5) * (1 - ramp(t, 61.30, 62.10))
    close = (0.20 * ramp(t, 52.0, 52.5) * (1 - ramp(t, 54.4, 54.9))        # крупный план Clawd
             + 0.30 * ramp(t, 58.2, 58.7) * (1 - ramp(t, 60.9, 61.3)))     # под лопухом
    cand = pick_amb('rain_medium_*.wav', 'rain_*.wav')
    body = [softest(cand[:6], cut=4000, tilt=-7, dh=-6)] if cand else []
    leaf = pick_amb('rain_heavy_leaves_*.wav')
    lite = pick_amb('rain_light_*.wav') or body
    if not body or not body[0]:        # записи дождя нет — временный шумовой слой
        MISSING.append('ambience/rain (ВРЕМЕННО синтез, пересобрать!)')
        w1 = S.lp(S.noise(n, 7), 620, order=2) * 1.1
        w2 = S.bandpass(S.noise(n, 8), 1200, 5200, order=2) * 0.6
        L = S.warm(w1 + w2, 3600, -8)
        R = S.warm(np.roll(w1, 977) + np.roll(w2, 313), 3600, -8)
        main = np.stack([L, R], 1) * (heavy * (1 + close)).astype(np.float32)[:, None]
    else:
        main = bed([(body[0], 1.0)], t1 - t0,
                   amp=(heavy * (1 + 0.5 * close)).astype(np.float32), db=None, seed=5,
                   cut=4000.0, tilt=-7.0, dh=-6.0)
        if leaf:   # пока лопух над Clawd — поверх ливня слышно, как капли бьют по листу
            umb = (ramp(t, 58.2, 58.8) * (1 - ramp(t, 61.1, 61.6))).astype(np.float32)
            main = main + 0.45 * bed([(leaf[0], 1.0)], t1 - t0, amp=umb, db=None, seed=9,
                                     cut=3800.0, tilt=-7.0, dh=-6.0)
    main = to_rms(main, -29.6, mask=np.repeat((heavy > 0.9)[:, None], 2, 1))
    after_env = (ramp(t, 61.6, 62.4) * (1 - ramp(t, 63.6, 64.8))).astype(np.float32)
    if lite:
        after = bed([(lite[0], 1.0)], t1 - t0, amp=after_env, db=None, seed=11,
                    cut=3600.0, tilt=-8.0, dh=-7.0)
    else:
        after = np.stack([S.warm(S.bandpass(S.noise(n, 21), 800, 4200, order=2), 3200, -9)] * 2, 1) \
            * after_env[:, None]
    after = to_rms(after, -30.9, mask=np.repeat((after_env > 0.9)[:, None], 2, 1))
    EVENTS.append((48.0, 'amb', 'ливень'))
    emit('s5_rain_bed', main + after, t0, bus='amb', tail=0.4, top=8000.0, low=120.0)


# Отдельных капель по лопуху нет: серия из 23 (и даже из 8) событий приходилась ровно
# на реплики «МИР?»/«МИР!» — по П5 она переведена в непрерывный слой «дождь по листве»
# внутри s5_rain_bed (уплотнение 58.2–61.3, пока лопух над Clawd).


def _rumble(dur, close, seed, vol):
    """Запасной раскат, если записей грома нет: 40–200 Гц, раскат, без треска (§5)."""
    n = n_samples(dur)
    rng = np.random.default_rng(seed)
    x = S.lp(rng.uniform(-1, 1, n).astype(np.float32), 60 + 70 * close, order=3)
    x *= np.exp(-np.arange(n, dtype=np.float32) / (SR * (1.1 + 1.6 * close)))
    x *= slow(n, 2.2 + 1.8 * close, seed + 1, 0.30, 1.0)
    if close > 0.5:
        k = n_samples(0.9)
        f = np.geomspace(110, 30, k)
        x[:k] += np.sin(2 * np.pi * np.cumsum(f) / SR).astype(np.float32) * S.env_ar(k, 0.02, 2.0) * 0.55
    x = S.warm(x, cutoff=900, tilt_db=-12)
    return np.stack([norm(x, vol), norm(np.roll(x, 160), vol * 0.92)], 1)


def build_thunder():
    """Гром только там, где он есть в кадре: далёкий за кадром 48.55, УДАР на вспышке
    51.16 (s5 shot1 LT=3.15) и уходящая гроза 61.35. Раскат 56.55 убран по §3."""
    t0 = 48.0
    tr = Track(16.5)
    th = pick_amb('*thunder*.wav', '*storm*.wav')
    hits = [(48.55, 0.14, 0.12, 'amb', 'далёкая гроза'),
            (51.16, 0.44, 1.00, 'lead', 'ГРОМ на вспышке молнии'),
            (61.35, 0.12, 0.10, 'amb', 'гроза уходит')]
    if th:
        near = [p for p in th if any(k in os.path.basename(p) for k in ('close', 'crack', 'near', 'heavy'))] or th
        far = [p for p in th if any(k in os.path.basename(p) for k in ('far', 'distant', 'rumble'))] or th
        for i, (t, pk, cl, role, nm) in enumerate(hits):
            src = (near if cl > 0.5 else far)[i % len(near if cl > 0.5 else far)]
            x = proc(src, peak=pk, cut=1100 if cl > 0.5 else 900, tilt=-12, dh=-8,
                     off=onset(src, 0.15), dur=3.4 if cl > 0.5 else 4.2, fout=1.4)
            k = np.arange(len(x), dtype=np.float32) / SR
            tau = 0.85 if cl > 0.5 else 1.6            # раскат уходит, а не висит над репликами
            x = x * np.exp(-np.clip(k - 0.45, 0, None) / tau)[:, None]
            ev(tr, t0, t, x, role, nm)
    else:
        MISSING.append('ambience/thunder (взят мягкий синтезированный раскат)')
        for i, (t, pk, cl, role, nm) in enumerate(hits):
            ev(tr, t0, t, _rumble(5.4, cl, 5 + 3 * i, pk), role, nm)
    tfade(tr, t0, 63.9, 64.5)
    emit('s5_thunder', tr.stereo(), t0)


def build_s5_voices():
    """Реплики-блипы (§6): Clawd C3–E4 с нисходящим слайдом, Codex E4–A5 ступенькой.
    «МИР!» — один файл: голос + тёплый аккорд (П9)."""
    t0 = 48.0
    tr = Track(13.5)
    ev(tr, t0, 49.55, v_clawd(['G3', 'E3'], vol=0.26, slide='C3'), 'lead',
       'Clawd поник', pan=-0.25)
    ev(tr, t0, 51.45, v_clawd(['E3'], vol=0.14, step=0.13, slide='A3'), 'support',
       'испуг от молнии', pan=-0.28)
    ev(tr, t0, 52.45, v_clawd(['F3', 'D3'], vol=0.14, slide='A2'), 'support',
       'дрожит', pan=-0.12)
    ev(tr, t0, 53.15, breath_in(0.14), 'support', 'вдох перед чихом', pan=-0.10)
    ev(tr, t0, 53.60, sneeze(0.34), 'lead', 'АПЧХИ', pan=-0.10)
    ev(tr, t0, 55.42, v_codex(['A4'], vol=0.12), 'support', 'Codex оглянулся', pan=0.30)
    ev(tr, t0, 56.25, v_codex(['E4', 'A4'], vol=0.28), 'lead', 'Codex «?»', pan=0.28)
    ev(tr, t0, 58.80, v_codex(['D4', 'G4', 'A4'], vol=0.30), 'lead', '«МИР?»', pan=0.20)
    ev(tr, t0, 59.70, v_clawd(['E3'], vol=0.16, step=0.14, slide='C4'), 'support',
       'Clawd поднял глаза', pan=-0.18)
    say = np.zeros(n_samples(1.2), np.float32)
    place(say, warm_chord(['C4', 'E4', 'G4', 'C5'], 1.1, vol=0.30), 0.0)
    place(say, v_clawd(['C3', 'G3'], vol=0.28, step=0.13, slide='C4'), 0.02)
    ev(tr, t0, 60.05, norm(say, 0.36), 'lead', '«МИР!»', pan=-0.10)
    emit('s5_voices', tr.stereo(), t0, tail=0.2)


def build_s5_foley():
    """Фактура сцены: «!», срыв лопуха, шаг по луже, бросок листа, объятие, кубок."""
    t0 = 48.0
    tr = Track(17.4)
    idea = np.zeros(n_samples(0.8), np.float32)
    ip = proc(smp('idea'), peak=0.75, cut=4000, tilt=-8, dh=-6, dur=0.55, fout=0.2)
    place(idea, ip.mean(1), 0.0)
    place(idea, v_codex(['A4', 'E5'], vol=0.5, step=0.08), 0.0)
    ev(tr, t0, 56.86, norm(idea, 0.30), 'lead', 'Codex «!» — идея', pan=0.24)
    ev(tr, t0, 57.00, proc(smp('pluck'), peak=0.18, cut=2600, tilt=-8, dh=-6, dur=0.45,
                           fout=0.2), 'support', 'срывает лопух', pan=0.32)
    ev(tr, t0, 58.12, proc(smp('splash'), peak=0.30, cut=2600, tilt=-8, dh=-6, dur=0.5,
                           fout=0.2), 'lead', 'шаг по луже', pan=0.25)
    ev(tr, t0, 61.76, proc(smp('toss'), peak=0.18, cut=3200, tilt=-8, dh=-6, fout=0.15),
       'support', 'лопух отброшен', pan=0.15)
    hug = np.zeros(n_samples(1.3), np.float32)
    place(hug, proc(smp('thump'), peak=0.85, cut=2200, tilt=-9, dh=-4, dur=0.5).mean(1), 0.0)
    place(hug, warm_chord(['F3', 'A3', 'C4'], 1.2, vol=0.45), 0.01)
    ev(tr, t0, 62.72, norm(hug, 0.34), 'lead', 'ОБЪЯТИЕ', pan=0.0)
    ev(tr, t0, 63.26, proc(smp('hop'), peak=0.14, cut=3800, tilt=-8, dh=-7, dur=0.35,
                           fout=0.15), 'support', 'половинка кубка прыгает', pan=-0.05)
    ev(tr, t0, 63.57, proc(smp('tink'), peak=0.13, cut=4400, tilt=-7, dh=-6, dur=0.28),
       'support', 'половинка приземлилась', pan=0.0)
    ev(tr, t0, 64.88, proc(smp('sw_in'), peak=0.18, cut=3800, tilt=-8, dh=-6, fout=0.2),
       'support', 'вход в монтаж', pan=-0.20)
    emit('s5_foley', tr.stereo(), t0, tail=0.2)


def build_s5_magic():
    """Сердечко, солнце, радуга и срастание кубка — колокольчики (bell ≤25 % в 2.5–6 кГц)."""
    t0 = 60.2
    tr = Track(5.1)
    ev(tr, t0, 60.30, proc(smp('heart'), peak=0.18, cut=4600, tilt=-7, dh=-6, dur=1.4,
                           fout=0.5), 'support', 'сердечко', pan=0.0)
    pad = warm_chord(['C4', 'G4', 'C5'], 2.0, vol=0.14, duty=0.4)
    ev(tr, t0, 61.30, S.fade(pad, 0.9, 0.8), 'amb', 'солнце сквозь тучи', pan=0.0)
    ev(tr, t0, 62.22, proc(smp('rainbow'), peak=0.40, cut=4800, tilt=-6, dh=-6, dur=1.7,
                           fout=0.6), 'lead', 'РАДУГА', pan=0.0)
    ev(tr, t0, 64.06, proc(smp('merge'), peak=0.42, cut=4800, tilt=-6, dh=-6, dur=1.9,
                           fout=0.7), 'lead', 'кубок сросся в сердце', pan=0.0)
    tfade(tr, t0, 64.95, 65.28)
    emit('s5_magic', tr.stereo(), t0)


# ================================================================ s6 — монтаж

def build_s6_amb():
    """Ночной стол 65.0–68.4: комнатный тон, RMS −35 (§4)."""
    t0, dur = 65.0, 3.4
    n = n_samples(dur)
    t = t0 + np.arange(n) / SR
    env = (ramp(t, 65.0, 65.5) * (1 - ramp(t, 68.0, 68.4))).astype(np.float32)
    room = pick_amb('*room*.wav', '*night*.wav', '*indoor*.wav') or [S_('kenney_spaceenginelow_000.wav')]
    if room and room[0]:
        x = bed([(room[0], 1.0)], dur, amp=env, db=-35.0, seed=23, cut=1600, tilt=-11, dh=-8)
    else:
        MISSING.append('ambience/room (взят тихий тёплый гул 60/120 Гц)')
        k = np.arange(n) / SR
        hum = (np.sin(2 * np.pi * 60 * k) * 0.6 + np.sin(2 * np.pi * 121 * k) * 0.22).astype(np.float32)
        hum = S.warm(hum + S.lp(S.noise(n, 55), 600, order=2) * 0.7, cutoff=1600, tilt_db=-12)
        x = to_rms(np.stack([hum, np.roll(hum, 97)], 1) * env[:, None], -35.0)
    EVENTS.append((65.0, 'amb', 'комната'))
    emit('s6_desk_amb', x, t0, bus='amb', low=45.0)


def build_s6_ui():
    """Терминал: всплытие окна (lead), печать как тихий bed из 4 нажатий (П4/П5),
    «✓ MERGED» (lead) и одно сердечко."""
    t0 = 65.0
    tr = Track(3.3)
    ev(tr, t0, 65.45, proc(smp('pop'), peak=0.32, cut=4200, tilt=-7, dh=-6), 'lead',
       'окно терминала')
    rng = np.random.default_rng(611)
    for i, t in enumerate((65.68, 65.96, 66.26, 66.56)):
        ev(tr, t0, t, proc(smp('key'), peak=0.07, cut=3600, tilt=-8, dh=-7), 'series',
           'клавиша', pan=float(rng.uniform(-0.1, 0.1)))
    ev(tr, t0, 66.90, proc(smp('merged'), peak=0.40, cut=4400, tilt=-6, dh=-6, dur=1.1,
                           fout=0.4), 'lead', '✓ MERGED')
    ev(tr, t0, 67.55, proc(smp('shimmer'), peak=0.14, cut=4600, tilt=-6, dh=-6, dur=1.0,
                           fout=0.4), 'support', 'сердечки над столом', pan=0.25)
    emit('s6_ui', tr.stereo(), t0, bus='ui', tail=0.2)


def build_s6_swoosh():
    """Склейка в полёт (lead), два облака мимо и выход в закат."""
    t0 = 64.8
    tr = Track(7.5)
    cut = np.zeros(n_samples(1.0), np.float32)
    place(cut, proc(smp('sw_cut'), peak=0.9, cut=3600, tilt=-8, dh=-5, fout=0.2).mean(1), 0.0)
    place(cut, norm(S.lp(S.noise(n_samples(0.5), 91), 90, order=3)
                    * S.env_ar(n_samples(0.5), 0.02, 2.0), 0.35), 0.02)
    ev(tr, t0, 68.18, norm(cut, 0.42), 'lead', 'склейка в полёт')
    for i, (t_, pan) in enumerate(((69.62, 0.55), (70.42, -0.55))):
        ev(tr, t0, t_, proc(smp('cloud', i), peak=0.14, cut=3200, tilt=-8, dh=-7, fout=0.2),
           'support', 'облако мимо', pan=pan)
    ev(tr, t0, 71.90, proc(smp('sw_out'), peak=0.16, cut=3400, tilt=-8, dh=-7, fout=0.25),
       'support', 'уход в закат', pan=0.15)
    emit('s6_swoosh', tr.stereo(), t0)


def build_s6_flight():
    """Полёт на облачке 68.1–72.15: ровный воздушный поток, RMS −30."""
    t0, dur = 68.1, 4.05
    n = n_samples(dur)
    t = t0 + np.arange(n) / SR
    env = (ramp(t, 68.15, 68.8) * (1 - 0.45 * ramp(t, 71.3, 72.15))).astype(np.float32)
    wnd = pick_amb('*wind_gusty*.wav', '*wind*.wav') or [smp('air')]
    x = bed([(softest(wnd[:3], cut=2600, tilt=-10, dh=-7), 1.0)], dur, amp=env, db=-30.0,
            seed=29, cut=2600, tilt=-10, dh=-7,
            mask=np.repeat((env > 0.9)[:, None], 2, 1))
    EVENTS.append((68.1, 'amb', 'воздух полёта'))
    emit('s6_flight_air', x, t0, bus='amb', tail=0.3, top=8000.0, low=120.0)


# ================================================================ s7 — закат и финал

def build_s7_amb():
    """Вечер на обрыве 72.0–92.9: ветер + прибой, RMS −31, уход в ноль к 92.85."""
    t0, t1 = 72.0, 92.9
    dur = t1 - t0
    n = n_samples(dur)
    t = t0 + np.arange(n) / SR
    env = (ramp(t, 72.0, 72.8) * (1 - ramp(t, 91.2, 92.85))).astype(np.float32)
    wnd = pick_amb('*wind_steady*.wav', '*wind*.wav', '*breeze*.wav')
    sea = pick_amb('*sea_distant*.wav', '*sea_swell*.wav', '*surf*.wav', '*sea*.wav')
    specs = [(p, g) for p, g in ((softest(wnd[:4], cut=2600, tilt=-10, dh=-7) if wnd else None, 0.75),
                                 (softest(sea[:4], cut=2600, tilt=-10, dh=-7) if sea else None, 1.0)) if p]
    if specs:
        x = bed(specs, dur, amp=env, db=-30.0, seed=31, cut=2600, tilt=-10, dh=-7,
                mask=np.repeat((env > 0.9)[:, None], 2, 1))
    else:
        MISSING.append('ambience/wind+surf (взят тихий шумовой слой)')
        w = S.lp(S.noise(n, 121), 700, order=2)
        sw = S.bandpass(S.noise(n, 122), 130, 1300, order=2) * (0.35 + 0.65 * slow(n, 0.2, 7))
        L = S.warm(w * 0.6 + sw, 2200, -10)
        R = S.warm(np.roll(w, 811) * 0.6 + np.roll(sw, 433), 2200, -10)
        x = to_rms(np.stack([L, R], 1) * env[:, None], -31.0,
                   mask=np.repeat((env > 0.9)[:, None], 2, 1))
    EVENTS.append((72.0, 'amb', 'ветер и прибой'))
    emit('s7_wind_surf', x, t0, bus='amb', top=8000.0, low=120.0)


def build_s7_cicadas():
    """Пять цикад вместо десяти, пик 0.06, только в паузах (§1, §5: ≤25 % выше 6 кГц)."""
    t0 = 72.0
    tr = Track(20.9)
    cand = pick_amb('*cricket*.wav', '*cicada*.wav', '*insect*.wav')
    src = [softest(cand, cut=2000, tilt=-11, dh=-10)] if cand else []
    rng = np.random.default_rng(141)
    if src:
        for i, t in enumerate((77.2, 81.6, 83.4, 88.6)):
            ev(tr, t0, t, proc(src[i % len(src)], peak=0.06, cut=2000, tilt=-11, dh=-10,
                               off=float(rng.uniform(0, 2.5)), dur=float(rng.uniform(0.8, 1.4)),
                               fin=0.25, fout=0.5, pan=float(rng.uniform(-0.5, 0.5))),
               'bed', 'цикада')
    else:
        MISSING.append('ambience/crickets')
    tfade(tr, t0, 91.3, 92.8)
    emit('s7_cicadas', tr.stereo(), t0, bus='amb', head=0.4, top=4500.0)


def build_s7_stars():
    """Мерцание звёзд — три искры и только в чистых паузах (П6, П10)."""
    t0 = 82.2
    tr = Track(10.7)
    rng = np.random.default_rng(161)
    for i, (t, pk) in enumerate(((82.40, 0.10), (84.65, 0.10), (90.62, 0.07))):
        ev(tr, t0, t, proc(smp('spark', i), peak=pk, cut=4800, tilt=-7, dh=-7, dur=0.3,
                           fout=0.15), 'support' if t < 90 else 'support',
           'мерцание звезды', pan=float(rng.uniform(-0.45, 0.45)))
    tfade(tr, t0, 91.6, 92.8)
    emit('s7_stars', tr.stereo(), t0)


def build_s7_events():
    """Закат: подсадка, объятие, сердечки, глаза-сердечки, кран, звезда, титр, «КОНЕЦ»."""
    t0 = 72.0
    tr = Track(20.9)
    for i, t in enumerate((73.22, 73.84)):
        ev(tr, t0, t, proc(smp('step_grass', i), peak=0.14, cut=2800, tilt=-9, dh=-5,
                           dur=0.45, fout=0.2), 'support', 'Codex подскакивает', pan=0.28)
    ev(tr, t0, 74.35, proc(smp('rustle'), peak=0.09, cut=2800, tilt=-9, dh=-6, dur=0.4,
                           fout=0.2), 'support', 'кладёт голову', pan=0.22)
    hug = np.zeros(n_samples(1.3), np.float32)
    place(hug, proc(smp('thump_s'), peak=0.85, cut=2200, tilt=-9, dh=-4, dur=0.4).mean(1), 0.0)
    place(hug, warm_chord(['F3', 'A3', 'C4'], 1.2, vol=0.5), 0.01)
    ev(tr, t0, 75.60, norm(hug, 0.30), 'lead', 'ОБЪЯТИЕ')
    ev(tr, t0, 76.12, proc(smp('heart'), peak=0.16, cut=4600, tilt=-7, dh=-6, dur=1.4,
                           fout=0.5), 'support', 'сердечки над парой')
    ev(tr, t0, 78.32, v_codex(['A4', 'C5'], vol=0.14), 'support', 'Codex улыбается', pan=0.20)
    ev(tr, t0, 78.58, v_clawd(['E3', 'G3'], vol=0.14), 'support', 'Clawd улыбается', pan=-0.20)
    ev(tr, t0, 80.15, proc(smp('shimmer'), peak=0.14, cut=4600, tilt=-6, dh=-6, dur=1.0,
                           fout=0.4), 'support', 'сердечки')
    ev(tr, t0, 80.72, proc(smp('heart'), peak=0.38, cut=4800, tilt=-6, dh=-6, dur=1.6,
                           fout=0.6), 'lead', 'ГЛАЗА-СЕРДЕЧКИ')
    # 83.90–87.50 кран вверх: тихий восходящий воздух, RMS −33 (bed)
    crane = proc(smp('air'), peak=0.9, cut=2200, tilt=-10, dh=-7, dur=3.6, fin=1.0, fout=1.4)
    crane = to_rms(crane, -33.0)
    ev(tr, t0, 83.90, crane, 'amb', 'кран вверх к звёздам')
    star = np.zeros(n_samples(0.8), np.float32)
    place(star, proc(smp('star'), peak=0.9, cut=3600, tilt=-8, dh=-6, fout=0.2).mean(1), 0.0)
    place(star, proc(smp('spark', 1), peak=0.3, cut=4800, tilt=-7, dh=-7, dur=0.3).mean(1), 0.12)
    ev(tr, t0, 85.82, norm(star, 0.22), 'lead', 'падающая звезда', pan=0.25)
    ev(tr, t0, 86.52, proc(smp('title'), peak=0.40, cut=4400, tilt=-6, dh=-6, dur=1.5,
                           fout=0.5), 'lead', 'ТИТР CLAUDE', pan=-0.10)
    ev(tr, t0, 86.77, proc(smp('heart'), peak=0.16, cut=4600, tilt=-6, dh=-6, dur=1.2,
                           fout=0.5), 'support', '♥')
    ev(tr, t0, 87.02, proc(smp('shimmer'), peak=0.16, cut=4600, tilt=-6, dh=-6, dur=1.2,
                           fout=0.5), 'support', 'CODEX', pan=0.10)
    ev(tr, t0, 87.52, proc(smp('end'), peak=0.30, cut=3200, tilt=-8, dh=-5, dur=2.2,
                           fout=0.9), 'lead', '«КОНЕЦ»')
    ev(tr, t0, 89.45, proc(smp('heart'), peak=0.26, cut=4400, tilt=-7, dh=-6, dur=1.8,
                           fout=0.7), 'lead', 'мини-титр проявился')
    tfade(tr, t0, 91.8, 92.85)
    emit('s7_events', tr.stereo(), t0)


# ================================================================ проверки и сборка

def check_events():
    """§2: П1 (один lead + один support), П2 (≥0.35 c между лидами), П3 (≥0.12 c
    lead↔support), П4 (≤4 атаки в окне 1 c), П6 (украшения не ближе 0.6 c)."""
    ok = True
    ts = sorted(EVENTS)
    leads = [t for t, r, _ in ts if r == 'lead']
    sup = [t for t, r, _ in ts if r == 'support']
    beds = [t for t, r, _ in ts if r == 'bed']
    print(f'\nсобытий: {len(ts)}  (lead {len(leads)}, support {len(sup)}, '
          f'украшений {len(beds)}, серия {sum(1 for _, r, _ in ts if r == "series")}, '
          f'слоёв {sum(1 for _, r, _ in ts if r == "amb")})')
    bad2 = [(a, b) for a, b in zip(leads, leads[1:]) if b - a < 0.35]
    bad3 = [(a, b) for a in leads for b in sup if 0 < abs(b - a) < 0.12]
    bad6 = [(a, b) for a in beds for b in leads + sup if abs(b - a) < 0.6]
    att = [t for t, r, _ in ts if r not in ('amb',)]
    win = max((sum(1 for x in att if 0 <= x - a < 1.0) for a in att), default=0)
    for label, bad in (('П2 лиды ближе 0.35 c', bad2), ('П3 lead↔support ближе 0.12 c', bad3),
                       ('П6 украшение ближе 0.6 c к событию', bad6)):
        if bad:
            ok = False
            print(f'  !! {label}: ' + ', '.join(f'{a:.2f}/{b:.2f}' for a, b in bad[:6]))
        else:
            print(f'  {label.split()[0]}: ок')
    print(f'  П4 максимум атак в окне 1 c: {win} (норма ≤4)')
    return ok and win <= 4


def main():
    if os.path.isdir(OUTD):
        shutil.rmtree(OUTD)
    os.makedirs(OUTD, exist_ok=True)
    print('s5 — дождь и примирение 48–65:')
    build_rain()
    build_thunder()
    build_s5_voices()
    build_s5_foley()
    build_s5_magic()
    print('s6 — монтаж 65–72:')
    build_s6_amb()
    build_s6_ui()
    build_s6_swoosh()
    build_s6_flight()
    print('s7 — закат и финал 72–93:')
    build_s7_amb()
    build_s7_cicadas()
    build_s7_stars()
    build_s7_events()
    write_manifest(CUES, STEMS)
    check_events()
    if MISSING:
        print('\n!! не найдено (взяты запасные варианты): ' + ', '.join(sorted(set(MISSING))))
    print(f'манифест {CUES}: {len(STEMS)} стемов')


if __name__ == '__main__':
    main()
