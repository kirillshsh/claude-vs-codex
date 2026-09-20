#!/usr/bin/env python3
"""Чиптюн-саундтрек к «CLAUDE vs CODEX» (93.000 с, 30 fps).

    python3 audio/make_music.py            # из корня cartoon/

Детерминированно пересоздаёт audio/stems/music/*.wav и манифест audio/cues/music.json.

Каркас (абсолютные секунды фильма / тональность / темп):
    0.00– 4.00  m0_title    C-dur      140  фанфара, стабы на удары логотипа, каданс 3.60
    4.00–13.00  m1_meadow   C-dur      132  весёлый пульс-лид, ходячий бас
   13.00–25.00  m2_argue    a-moll     150  перекличка мотивов L/R, нагнетание к 24.8
   25.00–34.00  m3_code     D-дорийский 160 арпеджио-«клавиши» в два канала, «НИЧЬЯ!» 32.8
   34.00–44.50  m4_race     e-moll→fis-moll 168  старт 37.0, модуляция 42.71, разгон к 44.5
   44.50–48.00  m4b_crash   fis-moll→a-moll  удар 44.5, рапид, раскол 46.15, молния 47.2
   48.00–60.00  m5_rain     a-moll      92  разрежённо, одинокая мелодия, «МИР?» 58.8
   60.00–65.00  m5b_peace   C-dur       92  «МИР!» 60.05, радуга 62.2, объятия 62.7
   65.00–72.00  m6_montage  F-dur      120  оба мотива в каноне, полёт 68.2
   72.00–83.60  m7_sunset   C-dur       84  лирика, мотивы переплетаются
   83.60–92.90  m7b_end     C-dur       84  колокольчики, титр 86.5/86.75/87.0, «КОНЕЦ» 87.5,
                                            аккорд догорает к 89.95, кода под мини-титры 89.4–92.47,
                                            последний ненулевой сэмпл < 92.900

Диатоника гарантирована конструктивно: все высоты берутся через Key.m() из lib_music,
каждая нота логируется, check_keys() в конце печатает сверку «гамма / использованные ноты».
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import synth as S                                                      # noqa: E402
from synth import Track, write_manifest                                # noqa: E402
from lib_music import (Key, begin, check_keys, N, CH, SQ, ARP, SWEEP,   # noqa: E402
                       motif, pad, stab, bass_bar, hits, groove, soften,
                       tail_fade, finish, report, dr)

OUT_DIR = os.path.join(ROOT, 'audio', 'stems', 'music')
CUE_PATH = os.path.join(ROOT, 'audio', 'cues', 'music.json')
STEMS = []


# ---------------------------------------------------------------- общие мелочи

def trem(x, hz=7.0, depth=0.3):
    t = np.arange(len(x), dtype=np.float32) / S.SR
    return (x * (1 - depth + depth * np.sin(2 * np.pi * hz * t))).astype(np.float32)


def roll(tr, t0, t1, r0, r1, gen, g0=0.3, g1=1.0, pan=0.0):
    """Ускоряющаяся дробь: частота ударов едет от r0 до r1 в секунду."""
    t, i = t0, 0
    while t < t1 and i < 500:
        u = (t - t0) / max(1e-6, t1 - t0)
        tr.add(gen(), t, gain=g0 + (g1 - g0) * u, pan=pan)
        t += 1.0 / (r0 + (r1 - r0) * u)
        i += 1
    return tr


def offbeats(tr, key, deg, t0, n, step, octave=4, vol=0.16, pan=0.0, duty=0.25, gain=1.0):
    """Короткие аккордовые «чиканья» на слабых долях."""
    for i in range(n):
        tr.add(stab(key, deg, step * 0.55, octave=octave, n=3, vol=vol, duty=duty),
               t0 + i * step, gain=gain, pan=pan)
    return tr


def emit(name, tr, t, rms_db, fin=0.02, fout=0.06, peak=0.80, pan=0.0, tail=0.05, win=None):
    # неслышимый подвал ниже 30 Гц только съедает запас по пику
    for ch in (0, 1):
        tr.buf[:, ch] = S.hp(tr.buf[:, ch], 30)
    tail_fade(tr, tail)
    finish(tr, rms_db, peak, win=win)
    path = os.path.join(OUT_DIR, name + '.wav')
    tr.save(path, limit=False)
    report(name, tr)
    STEMS.append({'file': f'audio/stems/music/{name}.wav', 't': round(float(t), 3),
                  'gain': 1.0, 'pan': pan, 'fin': fin, 'fout': fout, 'bus': 'music'})
    return path


# ================================================================== s0 — заставка (0.00–4.00)

def m0_title():
    K = Key('C', 'major')
    begin('m0_title', K)
    bpm, b = 140.0, 60.0 / 140.0            # доля 0.4286 c
    tr = Track(4.0)

    # --- ритм-секция: маршевая четвертная бочка + рабочий на слабых
    for i in range(9):
        if i * b < 3.95:
            tr.add(dr.kick(0.19, 175, 44, 0.95), i * b, gain=0.85)
    for i in range(4):
        tr.add(dr.snare(0.14, 0.62), (2 * i + 1) * b, gain=0.85)
    for i in range(16):
        if i * b / 2 < 3.55:
            tr.add(dr.hat(0.04, 0.20), i * b / 2, gain=0.75, pan=0.16)
    tr.add(dr.crash(0.85, 0.5), 0.0, gain=0.9)

    # --- стабы ровно на удары логотипа
    tr.add(N(K.m(0, 2), 0.42, wave='tri', vol=0.42), 0.0)                       # C2
    tr.add(stab(K, 0, 0.36, octave=3, n=4, vol=0.24, drv=2.0), 0.0)             # C-dur
    tr.add(stab(K, 0, 0.30, octave=4, n=3, vol=0.24, duty=0.45), 0.70, pan=-0.32)   # CLAUDE
    tr.add(N(K.m(0, 3), 0.30, wave='tri', vol=0.34), 0.70, pan=-0.20)
    tr.add(stab(K, 5, 0.30, octave=3, n=3, vol=0.24, duty=0.5), 1.04, pan=0.32)     # CODEX (a-moll)
    tr.add(N(K.m(5, 2), 0.30, wave='tri', vol=0.34), 1.04, pan=0.20)
    tr.add(stab(K, 4, 0.52, octave=3, n=4, vol=0.26, drv=2.4), 1.38)                # VS -> G-dur
    tr.add(N(K.m(4, 2), 0.50, wave='tri', vol=0.44), 1.38)
    tr.add(dr.crash(0.7, 0.45), 1.38, gain=1.0)
    tr.add(dr.kick(0.22, 190, 40, 1.0), 1.38, gain=1.0)

    # --- взлёт к подзаголовку
    tr.add(SQ([(K.m(d, 4), 0.175) for d in (0, 2, 4, 5, 6, 7)], bpm,
              wave='pulse', duty=0.45, vol=0.34, legato=0.95,
              tone=dict(cutoff=6800)), 1.45, pan=-0.10)

    # --- подзаголовок: аккорд + два лейтмотива вопрос/ответ
    tr.add(pad(K, 0, 1.70, octave=3, n=4, vol=0.13, cut=3000), 1.90)
    tr.add(motif(K, 'claude', bpm, octave=5, scale=0.5, wave='pulse', duty=0.45, vol=0.30,
                 tone=dict(cutoff=6600, harsh=-5)), 1.90, pan=-0.30)
    tr.add(motif(K, 'codex', bpm, octave=5, scale=0.5, wave='pulse', duty=0.5, vol=0.22,
                 tone=dict(cutoff=6200, harsh=-5)), 2.75, pan=0.30)

    # --- кадансовый удар 3.60 и звон до конца
    tr.add(CH(K.notes([0, 7, 9, 11, 14, 16], 2), 0.40, wave='tri', vol=0.15,
              atk=0.002, dec=0.06, sus=0.6, rel=0.3), 3.60)
    tr.add(CH(K.notes([7, 9, 11, 14], 2), 0.38, wave='pulse', duty=0.25, vol=0.11,
              atk=0.002, dec=0.05, sus=0.5, rel=0.28), 3.60)
    tr.add(dr.crash(0.40, 0.5), 3.60, gain=1.0)
    tr.add(dr.kick(0.24, 200, 38, 1.0), 3.60, gain=1.0)
    tr.add(N(K.m(0, 5), 0.38, wave='pulse', duty=0.45, vol=0.22, rel=0.3,
             tone=dict(cutoff=6600, harsh=-5)), 3.60, pan=-0.15)
    return tr


# ================================================================== s1 — поляна (4.00–13.00)

def m1_meadow():
    K = Key('C', 'major')
    begin('m1_meadow', K)
    bpm, b = 132.0, 60.0 / 132.0
    bar = 4 * b                                     # 1.8182 c
    tr = Track(9.0)
    prog = [0, 3, 0, 4, 5]                          # C F C G a
    for i, deg in enumerate(prog):
        t0 = i * bar
        # три голоса и ни одного лишнего: бас (окт.2) + слабые доли (окт.4) + лид (окт.5)
        tr.add(bass_bar(K, deg, bar, bpm, octave=2, vol=0.44), t0)
        offbeats(tr, K, deg, t0 + b / 2, 4, b, octave=4, vol=0.11, pan=0.30,
                 gain=0.9 if i < 4 else 0.55)

    # барабаны: лёгкие, на последнем такте (дуэль) — разрежаются
    groove(tr, 0.0, 4, bar, {'k': 'x.......x...x...', 's': '....x.......x...',
                             'h': 'x.x.x.x.x.x.x.x.'}, gain=0.75)
    groove(tr, 4 * bar, 1, bar, {'k': 'x.......x.......', 's': '....x...........',
                                 'h': 'x...x...x...x...'}, gain=0.55)

    # пульс-лид (duty .25), слегка влево
    tr.add(motif(K, 'claude', bpm, octave=5, scale=1.0, wave='pulse', duty=0.45, vol=0.30,
                 tone=dict(cutoff=6600, harsh=-5)), 0.0, pan=-0.26)
    tr.add(SQ([(K.m(4, 5), .5), (K.m(2, 5), .5), (K.m(3, 5), 1), (K.m(2, 5), 1), (K.m(0, 5), 1)],
              bpm, wave='pulse', duty=0.45, vol=0.28, tone=dict(cutoff=6600, harsh=-5)), bar, pan=-0.26)
    # бег героев (3.20): быстрая гаммка
    tr.add(SQ([(K.m(d, 5), .5) for d in (0, 1, 2, 3, 4, 3, 2, 1)],
              bpm, wave='pulse', duty=0.45, vol=0.28, tone=dict(cutoff=6600, harsh=-5)), 2 * bar, pan=-0.26)
    # прыжок и радость (5.55/5.86)
    tr.add(SQ([(K.m(4, 5), .5), (K.m(4, 5), .5), (K.m(6, 5), .5), (K.m(7, 5), .5),
               (K.m(6, 5), 1), (K.m(4, 5), 1)], bpm, wave='pulse', duty=0.45, vol=0.30,
              tone=dict(cutoff=6600, harsh=-5)), 3 * bar, pan=-0.26)
    # кубок сияет (6.35)
    tr.add(ARP(K.notes([4, 7, 9, 11], 5), 0.62, rate=16, bpm=bpm, wave='sine', vol=0.10,
               tone=dict(cutoff=11000, harsh=-3)), 6.35, pan=0.22)
    # дуэль (8.20): тёмный мотив Codex внизу, затем спуск в склейку
    tr.add(motif(K, 'codex', bpm, octave=4, scale=0.4, wave='pulse', duty=0.45, vol=0.26,
                 tone=dict(cutoff=5800, harsh=-5)), 8.20, pan=0.28)
    tr.add(SQ([(K.m(d, 4), 0.22) for d in (4, 3, 2, 1)], bpm,
              wave='pulse', duty=0.5, vol=0.24, legato=0.95,
              tone=dict(cutoff=5800, harsh=-5)), 8.60)
    tr.add(N(K.m(5, 1), 0.55, wave='tri', vol=0.40, rel=0.3), 8.50)
    return tr


# ================================================================== s2 — ссора (13.00–25.00)

def m2_argue():
    K = Key('A', 'minor')
    begin('m2_argue', K)
    bpm, b = 150.0, 0.4
    bar = 1.6
    tr = Track(12.0)
    prog = [0, 0, 3, 3, 5, 6, 0, 0]                 # a a d d F G a a
    for i, deg in enumerate(prog):
        t0 = i * bar
        if t0 >= 12.0:
            break
        tr.add(bass_bar(K, deg, bar, bpm, octave=2, vol=0.50, pattern=[0, 0, 7, 0, 4, 0, 7, 2]), t0)
        # злые стабы на 1 и 3 доли
        for k in (0, 2):
            tr.add(stab(K, deg, 0.22, octave=3, n=3, vol=0.28, duty=0.5, drv=2.6),
                   t0 + k * b, gain=0.8 + 0.05 * i)

    groove(tr, 0.0, 5, bar, {'k': 'x.....x.x.......', 's': '....x.......x...',
                             'h': 'x...o...x...o...'}, gain=0.8)
    groove(tr, 5 * bar, 2, bar, {'k': 'x..x..x.x...x...', 's': '....x.......x..x',
                                 'h': 'x.o.x.o.x.o.x.o.', 't': '..............x.'}, gain=0.9)

    # --- 1-2 шоты: «Я ЛУЧШЕ!» / «НЕТ, Я ЛУЧШЕ!»
    # Claude — треугольник наверху, Codex — импульс октавой ниже: разные тембр и регистр
    for t0, g in ((0.25, 0.9), (1.30, 1.0)):
        tr.add(motif(K, 'claude', bpm, octave=5, scale=0.5, wave='tri', vol=0.38,
                     tone=dict(cutoff=6600, harsh=-4)), t0, gain=g, pan=-0.45)
    for t0, g in ((2.60, 0.9), (3.65, 1.0)):
        tr.add(motif(K, 'codex', bpm, octave=4, scale=0.5, wave='pulse', duty=0.33, vol=0.32,
                     tone=dict(cutoff=5800, harsh=-5)), t0, gain=g, pan=0.45)

    # --- 3 шот: перекличка учащается (5.0–8.5)
    ex = [(5.00, 'claude', 0.34, 5), (5.55, 'codex', 0.34, 4),
          (6.10, 'claude', 0.26, 5), (6.52, 'codex', 0.26, 4),
          (6.94, 'claude', 0.20, 5), (7.26, 'codex', 0.20, 4),
          (7.58, 'claude', 0.15, 5), (7.82, 'codex', 0.15, 4),
          (8.06, 'claude', 0.12, 5), (8.25, 'codex', 0.12, 5)]
    for i, (t0, who, sc, oc) in enumerate(ex):
        cl = who == 'claude'
        tr.add(motif(K, who, bpm, octave=oc, scale=sc,
                     wave='pulse', duty=0.45 if cl else 0.33, vol=0.34,
                     tone=dict(cutoff=6600 if cl else 4400, harsh=-4 if cl else -7)),
               t0, gain=0.85 + 0.03 * i, pan=-0.45 if cl else 0.45)
        tr.add(dr.tom('E2' if cl else 'B2', 0.18, 0.5), t0, gain=0.5 + 0.04 * i)

    # --- 4 шот: сплит-экран, мотивы одновременно, тяжёлые стабы на каждую долю
    for t0 in (8.50, 9.50):
        tr.add(motif(K, 'claude', bpm, octave=5, scale=0.6, wave='tri', vol=0.34,
                     tone=dict(cutoff=6600, harsh=-4)), t0, pan=-0.45)
        tr.add(motif(K, 'codex', bpm, octave=4, scale=0.6, wave='pulse', duty=0.33, vol=0.30,
                     tone=dict(cutoff=5800, harsh=-5)), t0, pan=0.45)
    for i in range(8):
        tr.add(stab(K, 0 if i % 2 == 0 else 6, 0.20, octave=3, n=4, vol=0.25, drv=2.8),
               8.50 + i * 0.25, gain=0.9)
    groove(tr, 8.5, 1, 2.0, {'k': 'x.x.x.x.x.x.x.x.', 's': '....x.......x...',
                             'h': 'x.o.x.o.x.o.x.o.'}, gain=0.9)

    # --- 5 шот: драка. Восходящий вал 16-ми + дробь томов, кульминация под вспышку 11.75
    # один вал вместо двух — раньше две гаммы в соседних октавах давали кашу
    tr.add(SQ([(K.m(d, 4), 0.2) for d in range(15)], bpm,
              wave='pulse', duty=0.45, vol=0.32, legato=0.95,
              tone=dict(cutoff=6000, harsh=-5)), 10.55, pan=-0.12)
    roll(tr, 10.50, 11.72, 8, 26, lambda: dr.tom('A2', 0.14, 0.62), 0.45, 1.0)
    roll(tr, 10.50, 11.72, 6, 20, lambda: dr.snare(0.11, 0.5), 0.3, 0.95, pan=0.1)
    tr.add(CH(K.notes([0, 7, 12, 14], 1), 0.45, wave='square', vol=0.16,
              atk=0.002, dec=0.08, sus=0.5, rel=0.3), 11.75)
    tr.add(N(K.m(0, 1), 0.55, wave='tri', vol=0.46, rel=0.3), 11.75)
    tr.add(dr.crash(0.55, 0.32), 11.75, gain=1.0)
    tr.add(dr.kick(0.26, 210, 36, 0.8), 11.75, gain=1.0)
    return tr


# ================================================================== s3 — код (25.00–34.00)

def m3_code():
    K = Key('D', 'dorian')
    begin('m3_code', K)
    bpm = 160.0
    b = 60.0 / bpm                                  # 0.375
    bar = 4 * b                                     # 1.5
    tr = Track(9.0)

    # --- карточка раунда: гонг
    # bitcrush убран: на громком месте он давал ту самую «стеклянную» грязь
    gong = CH(K.notes([0, 7, 11, 14], 2), 1.30, wave='tri', vol=0.18,
              atk=0.012, dec=0.25, sus=0.55, rel=0.8, tone=dict(cutoff=4800, harsh=-4))
    tr.add(S.reverb(gong, room=0.6, mix=0.32), 0.0)
    tr.add(N(K.m(0, 1), 1.10, wave='tri', vol=0.40, rel=0.6), 0.0)
    tr.add(dr.crash(1.0, 0.34), 0.0, gain=1.0)
    tr.add(dr.kick(0.26, 200, 38, 1.0), 0.0, gain=1.0)
    tr.add(SQ([(K.m(d, 4), 0.25) for d in (0, 1, 2, 3, 4, 5)], bpm,
              wave='pulse', duty=0.45, vol=0.28, legato=0.95,
              tone=dict(cutoff=6600, harsh=-5)), 0.94, pan=-0.15)

    # --- арена: два «набора текста» в разные каналы
    segs = [(1.30, 1.50, 0), (1.50, 3.00, 0), (3.00, 4.50, 3),
            (4.50, 6.00, 4), (6.00, 7.50, 6), (7.50, 7.80, 0)]
    # «клавиши»: верхний набор — треугольник 16-ми, нижний — вдвое реже и шире по duty.
    # Пэд убран: гармонию и так держат два арпеджио, он только мутил середину.
    for t0, t1, deg in segs:
        d = t1 - t0
        tr.add(ARP(K.triad(deg, 4, 4), d, rate=16, bpm=bpm, wave='pulse', duty=0.45, vol=0.17,
                   tone=dict(cutoff=6400, harsh=-5)), t0, pan=-0.34)
        tr.add(ARP(list(reversed(K.triad(deg, 3, 4))), d, rate=8, bpm=bpm,
                   wave='pulse', duty=0.33, vol=0.14,
                   tone=dict(cutoff=5400, harsh=-5)), t0 + b / 8, pan=0.34)

    for t0, t1, deg in segs[1:5]:
        tr.add(bass_bar(K, deg, t1 - t0, bpm, octave=2, vol=0.38,
                        pattern=[0, 0, 4, 0, 7, 0, 4, 2]), t0)
    tr.add(bass_bar(K, 0, 0.75, bpm, octave=2, vol=0.38, pattern=[0, 4, 7, 4]), 7.05)

    # тикающий грув
    groove(tr, 1.50, 4, bar, {'k': 'x.....x...x.....', 's': '....x.......x...',
                              'h': 'x.o.x.o.x.o.x.o.', 'i': '......x.......x.'}, gain=0.72)

    # лейтмотивы поверх «клавиш»
    for t0, who, sc in ((2.00, 'claude', 0.45), (3.00, 'codex', 0.45),
                        (4.50, 'claude', 0.45), (5.50, 'codex', 0.45),
                        (6.60, 'claude', 0.28), (7.05, 'codex', 0.28)):
        cl = who == 'claude'
        tr.add(motif(K, who, bpm, octave=5 if cl else 4, scale=sc,
                     wave='pulse', duty=0.45 if cl else 0.33, vol=0.30,
                     tone=dict(cutoff=7000 if cl else 4200, harsh=-4 if cl else -7)),
               t0, pan=-0.36 if cl else 0.36)

    # --- «НИЧЬЯ!» 7.80 и напряжённый аккорд в склейку
    tr.add(CH(K.notes([0, 7, 14, 21], 1), 0.34, wave='square', vol=0.16,
              atk=0.002, dec=0.06, sus=0.45, rel=0.22), 7.80)
    tr.add(N(K.m(0, 1), 0.60, wave='tri', vol=0.46, rel=0.35), 7.80)
    tr.add(dr.crash(0.8, 0.34), 7.80, gain=1.0)
    tr.add(dr.kick(0.26, 205, 36, 1.0), 7.80, gain=1.0)
    tr.add(dr.snare(0.2, 0.7), 7.80, gain=0.9)
    hold = CH(K.notes([0, 2, 4, 8], 3), 1.05, wave='pulse', duty=0.4, vol=0.15,
              atk=0.02, dec=0.1, sus=0.85, rel=0.2, tone=dict(cutoff=5400, harsh=-5))
    tr.add(trem(hold, 9.0, 0.35), 7.95)
    roll(tr, 8.25, 8.98, 7, 22, lambda: dr.snare(0.1, 0.45), 0.25, 0.95)
    return tr


# ================================================================== s4 — гонка (34.00–44.50)

def m4_race():
    K = Key('E', 'minor')
    K2 = Key('F#', 'minor')
    bpm = 168.0
    b = 60.0 / bpm                                  # 0.35714
    bar = 4 * b                                     # 1.42857
    tr = Track(10.5)

    begin('m4_race:e', K)
    # --- карточка раунда 0.0–1.2
    tr.add(stab(K, 0, 0.42, octave=3, n=4, vol=0.24, drv=2.4), 0.0)
    tr.add(N(K.m(0, 1), 1.10, wave='tri', vol=0.42, rel=0.5), 0.0)
    tr.add(dr.crash(0.9, 0.34), 0.0, gain=1.0)
    tr.add(dr.kick(0.26, 205, 38, 0.82), 0.0, gain=1.0)
    roll(tr, 0.28, 1.18, 8, 26, lambda: dr.snare(0.1, 0.5), 0.25, 0.95)

    # --- старт: рёв моторов и отсчёт
    tr.add(pad(K, 0, 1.75, octave=2, n=3, vol=0.14, cut=1200), 1.20)
    for t0 in (1.28, 1.78):
        tr.add(SWEEP(K.m(0, 2), K.m(0, 4), 0.38, wave='saw', vol=0.16,
                     atk=0.02, dec=0.08, sus=0.8, rel=0.1), t0, pan=-0.2 if t0 < 1.5 else 0.2)
    for t0, deg in ((2.10, 0), (2.40, 2), (2.70, 4)):
        tr.add(N(K.m(deg, 5), 0.17, wave='pulse', duty=0.5, vol=0.30, rel=0.05,
                 tone=dict(cutoff=5600, harsh=-5)), t0)
        tr.add(dr.click(0.02, 3000, 0.35), t0, gain=0.8)

    # --- гонка: e-moll, такты от 3.0
    B16 = [0, 0, 7, 0, 0, 0, 7, 0, 0, 4, 7, 0, 7, 0, 4, 2]
    prog = [(3.0, 0), (3.0 + bar, 5), (3.0 + 2 * bar, 2), (3.0 + 3 * bar, 6)]   # e C G D
    tr.add(dr.crash(1.0, 0.36), 3.0, gain=1.0)
    # три голоса: бас 16-ми (окт.2), слабые доли (окт.4), лид (окт.5). Пэд убран.
    for t0, deg in prog:
        tr.add(bass_bar(K, deg, bar, bpm, octave=2, vol=0.54, pattern=B16), t0)
        offbeats(tr, K, deg, t0 + b / 2, 4, b, octave=4, vol=0.12, pan=0.3)
    groove(tr, 3.0, 4, bar, {'k': 'x..x..x...x.x...', 's': '....x.......x...',
                             'h': 'x.o.x.o.x.o.x.o.'}, gain=0.85)

    seq_m = [(3.00, 'claude'), (3.72, 'codex'), (4.43, 'claude'), (5.14, 'codex'),
             (5.86, 'claude'), (6.57, 'codex'), (7.29, 'claude'), (8.00, 'codex')]
    for t0, who in seq_m:
        cl = who == 'claude'
        tr.add(motif(K, who, bpm, octave=5 if cl else 4, scale=0.45,
                     wave='pulse', duty=0.45 if cl else 0.33, vol=0.36,
                     tone=dict(cutoff=6800 if cl else 4200, harsh=-4 if cl else -7)),
               t0, pan=-0.32 if cl else 0.32)
    # толчки корпусами
    for t0 in (7.47, 8.12):
        tr.add(dr.crash(0.45, 0.26), t0, gain=0.9)
        tr.add(dr.tom('E2', 0.22, 0.65), t0, gain=1.0)
        tr.add(stab(K, 0, 0.18, octave=3, n=3, vol=0.20, drv=3.0), t0)

    # --- модуляция на тон вверх (fis-moll) и разгон к финишу
    begin('m4_race:fis', K2)
    t_mod = 3.0 + 4 * bar                            # 8.7143
    tr.add(dr.crash(0.8, 0.32), t_mod, gain=1.0)
    tr.add(bass_bar(K2, 0, bar, bpm, octave=2, vol=0.56, pattern=B16), t_mod)
    offbeats(tr, K2, 0, t_mod + b / 2, 8, b, octave=4, vol=0.13, pan=0.3)
    groove(tr, t_mod, 1, bar, {'k': 'x..x..x...x.x.x.', 's': '....x.......x...',
                               'h': 'x.o.x.o.x.o.x.o.'}, gain=0.95)
    tr.add(motif(K2, 'claude', bpm, octave=5, scale=0.45, wave='tri', vol=0.38,
                 tone=dict(cutoff=6800, harsh=-4)), t_mod, pan=-0.32)
    tr.add(motif(K2, 'codex', bpm, octave=4, scale=0.45, wave='pulse', duty=0.45, vol=0.36,
                 tone=dict(cutoff=5600, harsh=-5)), t_mod + 0.72, pan=0.32)
    # восходящий вал ровно в склейку 10.50
    tr.add(SQ([(K2.m(d, 4), 0.14) for d in range(12)], bpm,
              wave='pulse', duty=0.45, vol=0.34, legato=0.96,
              tone=dict(cutoff=6600, harsh=-4)), 10.00, pan=-0.10)
    tr.add(N(K2.m(0, 1), 0.55, wave='tri', vol=0.46, rel=0.2), 9.95)
    roll(tr, 9.90, 10.48, 10, 30, lambda: dr.snare(0.09, 0.5), 0.35, 1.0)
    return tr


# ================================================================== s4 финал — удар (44.50–48.00)

def m4b_crash():
    K2 = Key('F#', 'minor')
    KA = Key('A', 'minor')
    tr = Track(3.5)

    begin('m4b_crash:fis', K2)
    # --- столкновение
    tr.add(dr.crash(1.35, 0.40), 0.0, gain=1.0)
    tr.add(dr.kick(0.34, 220, 32, 1.0), 0.0, gain=1.0)
    tr.add(dr.tom('F#1', 0.45, 0.8), 0.0, gain=1.0)
    tr.add(N(K2.m(0, 1), 0.70, wave='tri', vol=0.46, rel=0.4), 0.0)
    tr.add(stab(K2, 0, 0.40, octave=2, n=4, vol=0.22, drv=3.2), 0.0)
    tr.add(SWEEP(K2.m(0, 5), K2.m(0, 1), 0.30, wave='saw', vol=0.22,
                 atk=0.002, dec=0.05, sus=0.8, rel=0.08), 0.0)

    # --- рапид: кубок летит вверх
    up = ARP(K2.notes([0, 2, 4, 7, 9, 11], 4), 1.28, rate=6, bpm=168.0,
             wave='sine', vol=0.14)
    tr.add(S.reverb(up, room=0.6, mix=0.4), 0.35, pan=-0.10)
    tr.add(S.reverb(pad(K2, 0, 1.30, octave=3, n=3, vol=0.10, cut=2000), room=0.6, mix=0.3), 0.35)
    tr.add(trem(N(K2.m(4, 5), 0.70, wave='sine', vol=0.16, atk=0.06, rel=0.4), 6.0, 0.35),
           1.00, pan=0.12)
    # --- кубок раскалывается (46.15)
    # bitcrush снят: это громкое место, крошево давало резь. Осколки — треугольник + мягкий шум.
    shard = SQ([(K2.m(d, 4), 0.12) for d in (11, 9, 7, 5, 4, 2, 0, -2)], 168.0,
               wave='pulse', duty=0.45, vol=0.24, legato=0.9, tone=dict(cutoff=6800, harsh=-5))
    tr.add(shard, 1.65, pan=0.18)
    nz = S.n_samples(0.25)
    tr.add(soften(S.bandpass(S.noise(nz, 91) * S.env_ar(nz, 0.004, 3.0), 700, 3400),
                  cutoff=5400, harsh=-7, atk=3.0) * 0.30, 1.65)

    # --- обессиленно: переход в a-moll, небо темнеет
    begin('m4b_crash:a', KA)
    tr.add(S.lp(N(KA.m(0, 1), 1.85, wave='tri', vol=0.34, atk=0.06, dec=0.4, sus=0.7, rel=0.9), 260),
           1.70)
    for i, (t0, deg) in enumerate(((1.90, 0), (2.34, -1), (2.78, -2), (3.10, -3))):
        x = N(KA.m(deg, 3), 0.55, wave='tri', vol=0.16, atk=0.04, dec=0.15, sus=0.7, rel=0.3)
        tr.add(S.reverb(S.lp(x, 1800), room=0.5, mix=0.3), t0, pan=-0.16)
    # молния 47.20 (локально 2.70)
    tr.add(dr.tom('A1', 0.60, 0.72), 2.70, gain=1.0)
    n = S.n_samples(0.45)
    tr.add(soften(S.bandpass(S.noise(n, 57) * S.env_ar(n, 0.006, 2.2), 400, 2600),
                  cutoff=4800, harsh=-7, atk=4.0) * 0.20, 2.70)
    tr.add(S.lp(S.noise(n, 58) * S.env_ar(n, 0.01, 1.4), 300) * 0.30, 2.72)
    return tr


# ================================================================== s5 — гроза (48.00–60.00)

def m5_rain():
    K = Key('A', 'minor')
    begin('m5_rain', K)
    bpm = 92.0
    b = 60.0 / bpm                                   # 0.65217
    bar = 4 * b                                      # 2.60870
    tr = Track(12.0)

    # непрерывный низкий дрон — сцена тихая, но без дыр тишины
    tr.add(S.lp(N(K.m(0, 1), 12.0, wave='tri', vol=0.24, atk=0.5, dec=0.6, sus=0.8, rel=2.2), 300),
           0.0)
    for i, deg in enumerate((0, 3, 0, 5, 4)):        # a d a F e
        t0 = i * bar
        tr.add(pad(K, deg, bar * 0.99, octave=3, n=3, vol=0.15, atk=0.3, cut=1500, rev=0.32), t0)

    tr.add(S.lp(dr.kick(0.55, 95, 28, 0.7), 190), 0.20, gain=0.8)      # далёкий раскат

    # одинокая мелодия
    tr.add(S.reverb(motif(K, 'codex', bpm, octave=4, scale=1.0, wave='tri', vol=0.20,
                          atk=0.04, dec=0.2, sus=0.7, rel=0.4), room=0.5, mix=0.28),
           0.60, pan=-0.22)
    tr.add(S.reverb(SQ([(K.m(2, 4), 1), (K.m(4, 4), 1), (None, 0.5), (K.m(0, 4), 2)], bpm,
                       wave='tri', vol=0.17, atk=0.05, dec=0.2, sus=0.7, rel=0.5),
                    room=0.5, mix=0.28), 3.40, pan=-0.22)
    # дрожит (4.0–5.2) и «апчхи» (5.4)
    tr.add(trem(N(K.m(0, 4), 1.00, wave='tri', vol=0.15, atk=0.08, rel=0.4), 8.0, 0.45),
           4.20, pan=-0.15)
    tr.add(SQ([(K.m(4, 4), 0.35), (K.m(0, 4), 0.6)], bpm, wave='tri', vol=0.17), 5.40, pan=-0.18)
    # Codex срывает лопух и идёт (6.9–10.2): осторожные шаги вверх
    for i, deg in enumerate((0, 1, 2, 3, 4)):
        tr.add(S.reverb(N(K.m(deg, 3), 0.52, wave='tri', vol=0.15, atk=0.03, dec=0.15,
                          sus=0.6, rel=0.3), room=0.45, mix=0.25), 7.17 + i * b, pan=0.26)
    # «МИР?» 58.80 — вопрос, повисший на II ступени
    tr.add(S.reverb(SQ([(K.m(4, 3), 0.4), (K.m(6, 3), 0.4), (K.m(8, 3), 0.7)], bpm,
                       wave='tri', vol=0.20, atk=0.03, dec=0.2, sus=0.75, rel=0.45),
                    room=0.5, mix=0.3), 10.80, pan=0.28)
    return tr


# ================================================================== s5 — «МИР!» и радуга (60.00–65.00)

def m5b_peace():
    K = Key('C', 'major')
    begin('m5b_peace', K)
    bpm = 92.0
    b = 60.0 / bpm
    tr = Track(5.0)

    def glow(deg, t0, d, vol=1.0):
        # три голоса: тёплое тело (окт.3), колокольчики (окт.5), бас (окт.2).
        # Импульсный слой в окт.4 убран — он и дублировал тело, и резал середину.
        tr.add(CH(K.triad(deg, 3, 5), d, wave='tri', vol=0.15 * vol,
                  atk=0.03, dec=0.2, sus=0.8, rel=d * 0.4, tone=dict(cutoff=5000)), t0)
        tr.add(S.reverb(CH(K.triad(deg, 5, 3), d * 0.9, wave='sine', vol=0.11 * vol,
                           atk=0.012, dec=0.3, sus=0.55, rel=d * 0.4,
                           tone=dict(cutoff=11000, harsh=-3)), room=0.5, mix=0.3),
               t0, pan=0.16)
        tr.add(N(K.m(deg, 2), d * 0.8, wave='tri', vol=0.36 * vol, rel=d * 0.3), t0)

    glow(0, 0.05, 1.25)          # «МИР!» — большой C-dur
    tr.add(dr.kick(0.24, 170, 40, 0.8), 0.05, gain=0.8)
    tr.add(dr.crash(0.7, 0.28), 0.05, gain=0.8)
    tr.add(ARP(K.notes([0, 2, 4], 6), 0.42, rate=16, bpm=bpm, wave='sine', vol=0.09),
           0.35, pan=0.22)       # сердечко
    glow(3, 1.30, 1.30, 0.95)    # F
    glow(4, 2.60, 0.70, 0.95)    # G под радугу

    # радуга 62.20: восходящий подъём через две октавы
    # одна линия вместо двух октав сразу
    tr.add(S.reverb(SQ([(K.m(d, 4), 0.138) for d in range(12)], bpm,
                       wave='pulse', duty=0.45, vol=0.30, legato=0.96,
                       tone=dict(cutoff=6600, harsh=-5)),
                    room=0.45, mix=0.22), 2.20, pan=-0.12)

    glow(0, 3.30, 1.70)          # объятия — светлый устой
    tr.add(dr.crash(0.8, 0.3), 3.30, gain=0.8)
    tr.add(S.reverb(CH(K.triad(0, 5, 4), 1.6, wave='sine', vol=0.09,
                       atk=0.01, dec=0.4, sus=0.5, rel=0.8), room=0.55, mix=0.35), 3.30, pan=0.18)
    for i in range(6):
        tr.add(dr.kick(0.2, 165, 42, 0.7), 3.30 + i * 2 * b, gain=0.6)
        tr.add(dr.hat(0.04, 0.16), 3.30 + i * 2 * b + b, gain=0.6, pan=0.2)
    return tr


# ================================================================== s6 — монтаж (65.00–72.00)

def m6_montage():
    K = Key('F', 'major')
    begin('m6_montage', K)
    bpm = 120.0
    b = 0.5
    tr = Track(7.0)
    segs = [(0.0, 2.0, 0), (2.0, 3.2, 4), (3.2, 4.2, 0),
            (4.2, 5.2, 5), (5.2, 6.2, 3), (6.2, 7.0, 4)]
    # бас (окт.2) + подложка + два мотива в разных октавах; слабые доли убраны
    for t0, t1, deg in segs:
        d = t1 - t0
        tr.add(bass_bar(K, deg, d, bpm, octave=2, vol=0.44), t0)
        tr.add(pad(K, deg, d * 0.97, octave=3, n=3, vol=0.13, cut=2200), t0)

    groove(tr, 0.0, 2, 2.0, {'k': 'x.......x.......', 's': '....x.......x...',
                             'h': 'x.x.x.x.x.x.x.x.'}, gain=0.7)
    groove(tr, 3.2, 2, 2.0, {'k': 'x.....x.x.......', 's': '....x.......x...',
                             'h': 'x.o.x.o.x.o.x.o.'}, gain=0.85)
    tr.add(dr.crash(0.9, 0.35), 3.20, gain=0.9)

    # оба мотива в каноне — впервые вместе
    for t0, sc, g in ((0.0, 0.5, 0.9), (2.0, 0.5, 0.95), (3.2, 1.0, 1.0), (5.6, 0.4, 1.0)):
        tr.add(motif(K, 'claude', bpm, octave=5, scale=sc, wave='pulse', duty=0.45, vol=0.27,
                     tone=dict(cutoff=6400, harsh=-5)), t0, gain=g, pan=-0.35)
        tr.add(motif(K, 'codex', bpm, octave=4, scale=sc, wave='pulse', duty=0.33, vol=0.26,
                     tone=dict(cutoff=5600, harsh=-5)), t0 + 0.5 * sc, gain=g, pan=0.35)
    # полёт: тихое мерцание сверху, только пока мотивы дышат
    tr.add(ARP(K.triad(0, 5, 4), 2.00, rate=16, bpm=bpm, wave='sine', vol=0.055,
               tone=dict(cutoff=11000, harsh=-3)), 3.20, pan=0.25)
    return tr


# ================================================================== s7 — закат (72.00–83.60)

def m7_sunset():
    K = Key('C', 'major')
    begin('m7_sunset', K)
    bpm = 84.0
    b = 60.0 / bpm                                   # 0.714286
    bar = 4 * b                                      # 2.857143
    tr = Track(11.6)
    prog = [(0.0, 0), (bar, 5), (2 * bar, 3), (3 * bar, 4)]     # C a F G
    for t0, deg in prog:
        d = min(bar, 11.6 - t0)
        tr.add(pad(K, deg, d * 0.99, octave=3, n=4, vol=0.15, atk=0.12,
                   cut=2600, rev=0.3, spread=0.04), t0)
        tr.add(S.lp(N(K.m(deg, 2), d * 0.92, wave='tri', vol=0.34,
                      atk=0.04, dec=0.25, sus=0.75, rel=d * 0.3), 700), t0)

    tr.add(pad(K, 0, 0.6, octave=3, n=4, vol=0.13, atk=0.1, cut=2600, rev=0.3), 4 * bar)

    # мягкий пульс, без жёстких барабанов
    for i in range(0, 16, 2):
        t0 = i * b
        if t0 < 11.55:
            tr.add(dr.kick(0.18, 150, 40, 0.55), t0, gain=0.55 if i % 4 == 0 else 0.3)
    for i in range(8):
        tr.add(dr.openhat(0.16, 0.07), b * 2 + i * 2 * b, gain=0.55, pan=0.24)

    # мотивы переплетаются
    tr.add(motif(K, 'claude', bpm, octave=5, scale=1.0, wave='pulse', duty=0.45, vol=0.22,
                 atk=0.03, dec=0.15, sus=0.8, rel=0.35,
                 tone=dict(cutoff=5400, harsh=-5)), 0.60, pan=-0.30)
    tr.add(motif(K, 'codex', bpm, octave=4, scale=1.0, wave='pulse', duty=0.45, vol=0.20,
                 tone=dict(cutoff=5800, harsh=-5)), 1.30, pan=0.30)
    tr.add(motif(K, 'claude', bpm, octave=5, scale=0.8, wave='pulse', duty=0.45, vol=0.22,
                 atk=0.03, sus=0.8, rel=0.3, tone=dict(cutoff=5400, harsh=-5)), 3.50, pan=-0.30)
    tr.add(motif(K, 'codex', bpm, octave=4, scale=0.8, wave='pulse', duty=0.45, vol=0.21,
                 tone=dict(cutoff=5800, harsh=-5)), 4.30, pan=0.30)
    # встречный крупный план (77.60): оба вместе, шире
    tr.add(motif(K, 'claude', bpm, octave=5, scale=1.0, wave='pulse', duty=0.45, vol=0.24,
                 atk=0.03, sus=0.85, rel=0.4, tone=dict(cutoff=5400, harsh=-5)), 5.80, pan=-0.32)
    tr.add(motif(K, 'codex', bpm, octave=4, scale=1.0, wave='pulse', duty=0.45, vol=0.22,
                 tone=dict(cutoff=5800, harsh=-5)), 6.20, pan=0.32)
    # подъём над G в финальную часть
    tr.add(SQ([(K.m(4, 4), .6), (K.m(5, 4), .6), (K.m(6, 4), .6), (K.m(7, 4), .6),
               (K.m(8, 4), 1.2)], bpm, wave='tri', vol=0.24, atk=0.03, sus=0.8, rel=0.3),
           8.70, pan=-0.16)
    tr.add(S.reverb(N(K.m(7, 4), 0.55, wave='sine', vol=0.13, atk=0.03, rel=0.3),
                    room=0.5, mix=0.3), 11.05, pan=0.2)
    return tr


# ================================================================== s7 — титр, кода, конец (83.60–92.90)

def m7b_end():
    K = Key('C', 'major')
    begin('m7b_end', K)
    bpm = 84.0
    b = 60.0 / bpm
    tr = Track(9.3)

    for t0, d, deg in ((0.0, 1.50, 0), (1.43, 1.50, 5), (2.86, 0.80, 3)):
        tr.add(pad(K, deg, d, octave=3, n=4, vol=0.14, atk=0.15, cut=2400, rev=0.32), t0)
        tr.add(S.lp(N(K.m(deg, 2), d * 0.9, wave='tri', vol=0.30,
                      atk=0.05, sus=0.75, rel=d * 0.35), 700), t0)

    # эхо мотива Claude колокольчиками
    tr.add(S.delay(S.reverb(motif(K, 'claude', bpm, octave=6, scale=0.7, wave='sine', vol=0.12,
                                  atk=0.004, dec=0.3, sus=0.4, rel=0.5), room=0.55, mix=0.35),
                   time=0.357, fb=0.3, mix=0.25), 0.10, pan=-0.26)
    # звёздные колокольчики
    for t0, deg in ((1.00, 7), (1.50, 9), (1.95, 11), (2.30, 8)):
        tr.add(S.reverb(N(K.m(deg, 5), 0.85, wave='sine', vol=0.11,
                          atk=0.004, dec=0.35, sus=0.35, rel=0.5), room=0.6, mix=0.4),
               t0, pan=0.24 if deg % 2 else -0.24)
    # падающая звезда 86.20–86.90
    tr.add(S.delay(S.reverb(SQ([(K.m(d, 5), 0.1225) for d in (7, 6, 5, 4, 3, 2, 1, 0)], bpm,
                               wave='sine', vol=0.11, legato=0.9), room=0.5, mix=0.3),
                   time=0.2, fb=0.25, mix=0.2), 2.60, pan=0.18)

    # титр: CLAUDE (IV) – ♥ (V) – CODEX (vi) – КОНЕЦ (I)
    def hit(deg, t0, d, vol=1.0):
        tr.add(CH(K.triad(deg, 3, 4), d, wave='tri', vol=0.14 * vol,
                  atk=0.006, dec=0.12, sus=0.65, rel=d * 0.5), t0)
        tr.add(CH(K.triad(deg, 4, 3), d * 0.8, wave='pulse', duty=0.25, vol=0.09 * vol,
                  atk=0.004, dec=0.1, sus=0.55, rel=d * 0.4), t0, pan=-0.2)
        tr.add(N(K.m(deg, 2), d * 0.8, wave='tri', vol=0.30 * vol, rel=d * 0.35), t0)
        tr.add(dr.kick(0.2, 165, 40, 0.65), t0, gain=0.7)

    hit(3, 2.90, 0.40)       # CLAUDE
    hit(4, 3.15, 0.40)       # ♥
    hit(5, 3.40, 0.55)       # CODEX

    # финальный аккорд: бьём в 87.50, догорает к 89.95
    t_end, d_end = 3.90, 2.45
    tr.add(CH(K.notes([0, 7, 11, 14, 16, 18], 2), d_end, wave='tri', vol=0.11,
              atk=0.008, dec=0.35, sus=0.62, rel=1.75), t_end)
    tr.add(CH(K.notes([14, 16, 18, 21], 2), d_end * 0.9, wave='pulse', duty=0.25, vol=0.058,
              atk=0.006, dec=0.4, sus=0.5, rel=1.6), t_end, pan=-0.18)
    tr.add(S.reverb(CH(K.notes([21, 23, 25, 28], 2), d_end * 0.95, wave='sine', vol=0.085,
                       atk=0.004, dec=0.5, sus=0.45, rel=1.8), room=0.7, mix=0.4),
           t_end, pan=0.16)
    tr.add(S.lp(N(K.m(0, 2), d_end, wave='tri', vol=0.34,
                  atk=0.008, dec=0.4, sus=0.6, rel=1.7), 700), t_end)
    tr.add(dr.crash(1.1, 0.17), t_end, gain=0.8)
    tr.add(dr.kick(0.26, 175, 36, 0.8), t_end, gain=0.8)

    # --- кода под мини-титры (89.40–92.47): оба мотива по разу, на 7 dB тише темы.
    # Пока догорает C-dur, из него проступает тихий устой, дальше Claude и Codex
    # прощаются колокольчиками и всё уходит в ноль к 92.90.
    tr.add(S.reverb(pad(K, 0, 3.10, octave=3, n=4, vol=0.068, atk=0.45, rel=1.5, cut=1500),
                    room=0.75, mix=0.42), 5.70)
    tr.add(S.lp(N(K.m(0, 2), 3.20, wave='tri', vol=0.14, atk=0.5, dec=0.5, sus=0.7, rel=1.6), 500),
           5.70)
    for t0, who, pan in ((5.80, 'claude', -0.28), (7.30, 'codex', 0.28)):
        # один тембр на мотив: синусовый колокольчик, без треугольного дубля
        bell = motif(K, who, bpm, octave=5, scale=0.55, wave='sine', vol=0.098,
                     atk=0.008, dec=0.28, sus=0.4, rel=0.55,
                     tone=dict(cutoff=12000, harsh=-3))
        tr.add(S.delay(S.reverb(bell, room=0.75, mix=0.45), time=0.357, fb=0.26, mix=0.2),
               t0, pan=pan)
    for t0, deg in ((6.62, 7), (8.30, 4)):
        tr.add(S.reverb(N(K.m(deg, 5), 0.8, wave='sine', vol=0.059,
                          atk=0.004, dec=0.35, sus=0.3, rel=0.45), room=0.75, mix=0.45),
               t0, pan=-0.22 if deg % 2 else 0.22)
    # последний «дзинь» — тоника, гаснет к 92.9
    tr.add(S.reverb(CH(K.notes([0, 4, 7], 5), 0.70, wave='sine', vol=0.054,
                       atk=0.005, dec=0.3, sus=0.28, rel=0.42), room=0.8, mix=0.45), 8.58, pan=0.14)
    return tr


# ================================================================== сборка

FILM_DUR = 93.0

PLAN = [
    # имя,        функция,      t,     RMS,   fin,  fout, хвост, окно нормировки
    ('m0_title',   m0_title,     0.00, -14.0, 0.02, 0.10, 0.06),
    ('m1_meadow',  m1_meadow,    4.00, -17.0, 0.06, 0.03, 0.03),   # iris-in / жёсткая склейка
    ('m2_argue',   m2_argue,    13.00, -15.5, 0.02, 0.03, 0.03),   # склейка 25.0 — вспышка
    ('m3_code',    m3_code,     25.00, -16.0, 0.02, 0.03, 0.03),   # склейка 34.0 — вспышка
    ('m4_race',    m4_race,     34.00, -13.5, 0.02, 0.03, 0.02),
    ('m4b_crash',  m4b_crash,   44.50, -17.5, 0.01, 0.30, 0.10),   # уход в затемнение 48.0
    ('m5_rain',    m5_rain,     48.00, -24.0, 0.30, 0.15, 0.10),
    ('m5b_peace',  m5b_peace,   60.00, -17.0, 0.02, 0.25, 0.10),   # dissolve в s6
    ('m6_montage', m6_montage,  65.00, -17.0, 0.25, 0.25, 0.10),   # dissolve в s7
    ('m7_sunset',  m7_sunset,   72.00, -16.0, 0.25, 0.05, 0.04),
    # RMS считается по окну 0–3.9 (83.60–87.50): кода дописана после сведения,
    # нормировка по всей длине подняла бы уже готовую часть
    ('m7b_end',    m7b_end,     83.60, -15.46, 0.03, 0.05, 0.40, (0.0, 3.9)),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('стемы:')
    end = 0.0
    for row in PLAN:
        name, fn, t, rdb, fin, fout, tail = row[:7]
        win = row[7] if len(row) > 7 else None
        tr = fn()
        emit(name, tr, t, rdb, fin=fin, fout=fout, tail=tail, win=win)
        end = max(end, t + tr.dur)
    ok = check_keys()
    write_manifest(CUE_PATH, STEMS)
    print(f'\nманифест {CUE_PATH}: {len(STEMS)} стемов, музыка кончается на {end:.3f} c')
    if not ok:
        print('!! есть ноты вне заявленной гаммы')
        return 1
    if end > FILM_DUR + 1e-6:
        print(f'!! музыка выходит за {FILM_DUR:.3f} c')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
