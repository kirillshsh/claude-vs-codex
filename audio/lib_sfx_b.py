"""Помощники для звуковых эффектов блока B (25–48 с): дуэль кодинга + гонка.

Версия 2 — на РЕАЛЬНЫХ сэмплах из audio/samples/. Синтез не используется вообще:
всё (клавиши, удары, лязг, свуши, моторы, колокола, толпа, авария) берётся из библиотеки,
обрезается, транспонируется и смягчается (warm/deharsh/soft_attack).

Роли и громкости — по audio/PLAN.md §4:
    lead     0.44–0.50 (s3), 0.44–0.56 (s4)   один в любой момент, не чаще 0.35 с
    support  0.14–0.20                        = lead −7…−9 дБ, не ближе 0.12 с к лиду
    bed      RMS по таблице §4 (−27…−31)      непрерывные слои
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, 'audio', 'samples'))

from synth import (SR, Track, bandpass, deharsh, fade, hp, lp, load_wav, n_samples,  # noqa: E402
                   peak_eq, save_wav, soft_attack, warm, write_manifest)
from find import find, info  # noqa: E402

LEAD_LO, LEAD_HI = 0.26, 0.56      # §4 PLAN.md: лид 0.26–0.56 по участкам
SUP_LO, SUP_HI = 0.07, 0.22        # support = lead −7…−9 дБ

# ---------------------------------------------------------------- загрузка сэмплов

_CACHE = {}


def smp(name):
    """Сэмпл по имени файла из каталога -> моно float32 (кэшируется)."""
    if name in _CACHE:
        return _CACHE[name]
    rec = info(name)
    path = rec['abspath'] if rec else os.path.join(_ROOT, 'audio', 'samples', name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'нет сэмпла {name}')
    x = load_wav(path).mean(1).astype(np.float32)
    _CACHE[name] = x
    return x


def meta(name):
    return info(name) or {}


def pick_many(cat, tags=(), n=5, max_hi=0.12, **kw):
    """Несколько разных сэмплов одной категории — для рандомизации (клавиши)."""
    hits = find(cat, tags, max_hi=max_hi, **kw)
    return [os.path.basename(p) for p in hits[:n]]


# ---------------------------------------------------------------- уровни

def rms(x):
    x = np.asarray(x, np.float32)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def set_rms(x, db):
    r = rms(x)
    if r <= 1e-9:
        return np.asarray(x, np.float32)
    return (np.asarray(x, np.float32) * (10 ** (db / 20) / r)).astype(np.float32)


def set_peak(x, p):
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    if m <= 1e-9:
        return np.asarray(x, np.float32)
    return (np.asarray(x, np.float32) * (p / m)).astype(np.float32)


def env_curve(n, pts):
    xs = np.array([p[0] for p in pts], np.float64)
    ys = np.array([p[1] for p in pts], np.float64)
    return np.interp(np.linspace(0, 1, n), xs, ys).astype(np.float32)


def soft_cap(x, thr=0.45, ceil=0.72):
    """Мягкий потолок: ниже thr — без изменений, выше — плавно поджимается к ceil."""
    x = np.asarray(x, np.float32)
    a = np.abs(x)
    over = a > thr
    if not over.any():
        return x
    y = x.copy()
    k = ceil - thr
    y[over] = np.sign(x[over]) * (thr + k * np.tanh((a[over] - thr) / k))
    return y.astype(np.float32)


# ---------------------------------------------------------------- обработка

def pitch(x, st):
    """Транспонировать на st полутонов (меняется и длительность — как у магнитофона)."""
    if abs(st) < 1e-3:
        return np.asarray(x, np.float32)
    r = 2.0 ** (st / 12.0)
    n = max(2, int(len(x) / r))
    return np.interp(np.arange(n) * r, np.arange(len(x)), x).astype(np.float32)


def take(x, dur=None, fin=0.004, fout=0.04, start=0.0):
    """Кусок сэмпла нужной длины с мягкими краями."""
    p = n_samples(start)
    y = np.asarray(x, np.float32)[p:]
    if dur is not None:
        y = y[:n_samples(dur)]
    return fade(y, fin, fout)


def soften(x, cutoff=6500, dh=-5.0, atk_ms=4.0):
    """Общий «смягчитель»: срез верха + провал в 2.5–6 кГц + мягкая атака."""
    y = warm(np.asarray(x, np.float32), cutoff)
    y = deharsh(y, dh, 3600.0)
    return soft_attack(y, atk_ms)


def lead(x, peak=0.62, **kw):
    return set_peak(soften(x, **kw), float(np.clip(peak, LEAD_LO, LEAD_HI)))


def support(x, peak=0.32, **kw):
    return set_peak(soften(x, **kw), float(np.clip(peak, SUP_LO, SUP_HI)))


def loopify(x, xf=0.25):
    """Склеить сэмпл в бесшовное кольцо (кроссфейд хвоста в голову)."""
    x = np.asarray(x, np.float32)
    m = n_samples(xf)
    if len(x) < 3 * m:
        return x
    y = x[:len(x) - m].copy()
    y[:m] = y[:m] * np.linspace(0, 1, m, dtype=np.float32) + x[len(x) - m:] * np.linspace(1, 0, m, dtype=np.float32)
    return y


def varispeed(x, dur, tpts, rpts, loop=True):
    """Проигрывание с переменной скоростью: rate=1 — исходная высота.

    tpts/rpts — опорные точки (сек от начала, скорость). Для моторов это и есть
    «высота едет от скорости».
    """
    x = np.asarray(x, np.float32)
    n = n_samples(dur)
    tt = np.linspace(0, dur, n)
    r = np.interp(tt, tpts, rpts)
    idx = np.cumsum(r)
    if loop:
        idx = np.mod(idx, len(x) - 1)
    else:
        idx = np.clip(idx, 0, len(x) - 1)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


# ---------------------------------------------------------------- панорама и укладка

def pan2(x, p0, p1=None):
    """Моно -> стерео (n, 2); p1 задаёт проезд панорамы (доплер)."""
    x = np.asarray(x, np.float32)
    n = len(x)
    p = np.full(n, float(p0), np.float64) if p1 is None else np.linspace(p0, p1, n)
    l = np.sqrt((1 - p) / 2) * math.sqrt(2)
    r = np.sqrt((1 + p) / 2) * math.sqrt(2)
    return np.stack([x * l, x * r], 1).astype(np.float32)


def add(tr, sig, t, pan=0.0, pan_to=None, gain=1.0, fin=0.0, fout=0.0):
    sig = fade(np.asarray(sig, np.float32), fin, fout)
    tr.add(pan2(sig, pan, pan_to), t, gain=gain)
    return tr


def tail_fade(tr, sec=0.12):
    m = n_samples(sec)
    if m < len(tr.buf):
        tr.buf[-m:] *= np.linspace(1, 0, m, dtype=np.float32)[:, None]
    return tr


# ---------------------------------------------------------------- готовые слои

def key_bed(names_cl, names_cx, dur, t_from, t_to, rate_cl, rate_cx, pan_cl, pan_cx,
            peak=0.28, seed=777, off_gain=0.56, fast_from=None, fast_k=1.2):
    """Клавиатурная дробь из реальных щелчков: 4–6 вариантов, разброс высоты и громкости."""
    out = Track(dur)
    rng = np.random.default_rng(seed)
    for names, rate, pan_fn, lo_st in ((names_cl, rate_cl, pan_cl, -2.5), (names_cx, rate_cx, pan_cx, 2.0)):
        pool = [take(smp(nm), 0.14, 0.002, 0.05) for nm in names]
        t = t_from
        i = 0
        while t < t_to:
            x = pool[i % len(pool)]
            st = lo_st + float(rng.uniform(-1.5, 1.5))
            y = soften(pitch(x, st), 6000, -6.0, 3.0)
            v = peak * float(rng.uniform(0.62, 1.0))
            pp = pan_fn(t)
            g = off_gain if abs(pp) >= 0.8 else 1.0
            add(out, set_peak(y, v), t, pan=pp + float(rng.uniform(-0.05, 0.05)), gain=g)
            k = fast_k if (fast_from is not None and t >= fast_from) else 1.0
            t += (1.0 / (rate * k)) * float(rng.uniform(0.62, 1.42))
            i += 1
    return out


def bed(name, dur, rms_db=-29.0, tpts=None, rpts=None, amp_pts=None, lp_hz=2000.0,
        hp_hz=40.0, xf=0.3, start=0.0):
    """Непрерывный слой из сэмпла: кольцо + переменная скорость + огибающая."""
    x = loopify(take(smp(name), None, 0.0, 0.0, start=start), xf)
    if tpts is None:
        y = varispeed(x, dur, [0.0, dur], [1.0, 1.0])
    else:
        y = varispeed(x, dur, tpts, rpts)
    if amp_pts is not None:
        tt = np.linspace(0, dur, len(y))
        y = y * np.interp(tt, [p[0] for p in amp_pts], [p[1] for p in amp_pts]).astype(np.float32)
    y = lp(hp(y, hp_hz), lp_hz)
    return set_rms(y, rms_db)


def smear(x, room=0.13, cutoff=2400.0, mix=0.75, seed=17):
    """Размазать серию щелчков в непрерывный слой (П5 PLAN.md).

    Свёртка с плотным затухающим шумом: у каждого щелчка вырастает диффузный хвост,
    промежутки заполняются, и детектор атак перестаёт видеть отдельные события.
    Дискретные тапы здесь не годятся — они сами читаются как новые атаки.
    """
    x = np.asarray(x, np.float32)
    n_ir = n_samples(room)
    rng = np.random.default_rng(seed)
    ir = rng.uniform(-1, 1, n_ir).astype(np.float32) * np.exp(-np.linspace(0, 5.0, n_ir, dtype=np.float32))
    ir = lp(ir, cutoff)
    ir /= (np.sqrt(np.sum(ir ** 2)) + 1e-9)
    n = len(x) + n_ir - 1
    N = 1 << (n - 1).bit_length()
    wet = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[:len(x)].astype(np.float32)
    return lp(x * (1 - mix) + wet * mix, cutoff)


def attacks(x, thr=0.22, gap=0.05, win=1024, hop=256):
    """Детектор атак (spectral flux) — метрика §0 PLAN.md. Возвращает секунды атак."""
    mono = x.mean(1) if x.ndim == 2 else np.asarray(x, np.float32)
    n = (len(mono) - win) // hop
    if n < 2:
        return []
    w = np.hanning(win).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(mono, win)[::hop][:n] * w
    mag = np.abs(np.fft.rfft(frames, axis=1))
    flux = np.maximum(0.0, np.diff(mag, axis=0)).sum(1)
    if not len(flux):
        return []
    flux = flux / (flux.max() + 1e-9)
    out, last = [], -1e9
    for i in range(1, len(flux) - 1):
        if flux[i] >= thr and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
            t = (i * hop + win / 2) / SR
            if t - last >= gap:
                out.append(t)
                last = t
    return out
