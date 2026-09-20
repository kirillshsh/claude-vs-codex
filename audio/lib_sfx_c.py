"""Помощники SFX для финальной трети мультика (48–93 c) — на РЕАЛЬНЫХ сэмплах.

Синтез остался только для абстрактных чип-реплик героев (blip/crab/digi); всё
остальное — записи из audio/samples (катало­ги catalog.json + catalog_ambience.json).

Основное:
    snd(path)                    загрузка сэмпла с кэшем (стерео float32)
    one(cat, tags, seed=...)     детерминированный выбор одного сэмпла
    amb(pattern)                 файлы атмосфер по маске имени
    proc(x, peak=..., ...)       обрезка/фейды/смягчение/панорама одного звука
    stitch(path, dur, seed)      непрерывный кусок нужной длины (без слышимой петли)
    bed(specs, dur, amp, db)     длинная атмосфера из нескольких записей

Смягчение тембра (synth.warm / synth.deharsh / synth.soft_attack) вшито в proc()
и bed(): цель — не больше 12 % энергии выше 6 кГц и 20 % в полосе 2.5–6 кГц.
"""
import glob
import math
import os
import sys

import numpy as np

import synth as S
from synth import (SR, bitcrush, deharsh, drive, env_ar, load_wav, lp, n_samples,  # noqa: F401
                   note, soft_attack, warm)

_HERE = os.path.dirname(os.path.abspath(__file__))
SAMP = os.path.join(_HERE, 'samples')
if SAMP not in sys.path:
    sys.path.insert(0, SAMP)

try:
    from find import find as _find, info as _info          # noqa: E402
except Exception:                                          # каталога ещё нет
    _find = _info = None


# ---------------------------------------------------------------- утилиты

def norm(x, peak=0.5):
    m = float(np.max(np.abs(x))) or 1.0
    return (np.asarray(x, np.float32) * (peak / m)).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))


def to_rms(x, db, mask=None):
    seg = x if mask is None else x[mask]
    r = rms(seg) or 1e-9
    return (np.asarray(x, np.float32) * (10 ** (db / 20) / r)).astype(np.float32)


def ramp(t, a, b):
    """0 до a, 1 после b, линейно между (векторно)."""
    return np.clip((np.asarray(t, np.float64) - a) / max(1e-9, b - a), 0.0, 1.0)


def slow(n, hz, seed, lo=0.0, hi=1.0):
    """Медленный случайный управляющий сигнал (узлы с частотой hz, линейная интерполяция)."""
    m = max(2, int(n * hz / SR) + 2)
    v = np.random.default_rng(seed).random(m).astype(np.float32)
    x = np.interp(np.linspace(0, m - 1, n), np.arange(m), v).astype(np.float32)
    return (lo + (hi - lo) * x).astype(np.float32)


def place(dst, sig, t, gain=1.0):
    """Подмешать моно-сигнал в моно-буфер на секунду t."""
    p = int(round(t * SR))
    if p < 0:
        sig, p = sig[-p:], 0
    if p >= len(dst) or len(sig) == 0:
        return dst
    s = sig[:len(dst) - p]
    dst[p:p + len(s)] += s * gain
    return dst


def bands(x, lo_hz=(20, 250, 1200, 2500, 6000), hi_hz=(250, 1200, 2500, 6000, 20000)):
    """Доли энергии по полосам 20–250 / 250–1200 / 1200–2500 / 2500–6000 / 6000–20000 Гц."""
    m = np.asarray(x, np.float64)
    m = m.mean(1) if m.ndim == 2 else m
    if len(m) < 1024 or not np.any(m):
        return [0.0] * 5
    X = np.abs(np.fft.rfft(m))
    f = np.fft.rfftfreq(len(m), 1 / SR)
    tot = X.sum() + 1e-9
    return [round(100 * X[(f >= a) & (f < b)].sum() / tot, 1) for a, b in zip(lo_hz, hi_hz)]


# ---------------------------------------------------------------- доступ к сэмплам

_CACHE = {}


def snd(path):
    """Сэмпл как стерео float32 (n, 2), с кэшем."""
    if path not in _CACHE:
        _CACHE[path] = load_wav(path)
    return _CACHE[path]


def one(cat, tags=(), seed=0, **kw):
    """Детерминированный выбор одного сэмпла категории (по хэшу seed, не по random)."""
    if _find is None:
        return None
    hits = _find(cat, tags, **kw)
    return hits[seed % len(hits)] if hits else None


def by_name(name):
    """Точный файл по имени (ищем и в основном дереве сэмплов, и в атмосферах)."""
    hits = glob.glob(os.path.join(SAMP, '**', name), recursive=True)
    return sorted(hits)[0] if hits else None


def amb(pattern, loops=False):
    """Файлы атмосфер по маске имени; loops=False отбрасывает *_loop-дубли."""
    hits = sorted(glob.glob(os.path.join(SAMP, 'ambience', '**', pattern), recursive=True))
    if not loops:
        hits = [h for h in hits if '_loop' not in os.path.basename(h)]
    return hits


def have_amb(pattern):
    return bool(amb(pattern, loops=True))


# ---------------------------------------------------------------- обработка

def _soften(ch, cut, tilt, dh, dhf):
    y = warm(ch, cutoff=cut, tilt_db=tilt) if cut else ch
    return deharsh(y, db=dh, freq=dhf) if dh else y


def _stereo(x):
    x = np.asarray(x, np.float32)
    return np.stack([x, x], 1) if x.ndim == 1 else x


def balance(x, pan):
    """Панорама стерео-сигнала (сохраняет образ, меняет баланс)."""
    if not pan:
        return x
    l = math.sqrt((1 - pan) / 2) * math.sqrt(2)
    r = math.sqrt((1 + pan) / 2) * math.sqrt(2)
    return (np.asarray(x, np.float32) * np.array([l, r], np.float32)).astype(np.float32)


def proc(x, peak=None, fin=0.0, fout=0.0, cut=6000.0, tilt=-6.0, dh=-4.0, dhf=3600.0,
         soft=3.0, dur=None, off=0.0, pan=0.0, gain=1.0):
    """Один звук: вырезка (off/dur) -> смягчение -> фейды -> пик -> панорама."""
    y = _stereo(snd(x) if isinstance(x, str) else x).copy()
    if off:
        y = y[n_samples(off):]
    if dur:
        y = y[:n_samples(dur)]
    if len(y) < 4:
        return y
    y = np.stack([_soften(y[:, 0], cut, tilt, dh, dhf),
                  _soften(y[:, 1], cut, tilt, dh, dhf)], 1).astype(np.float32)
    if soft:
        k = min(n_samples(soft / 1000.0), len(y))
        if k > 1:
            y[:k] *= (np.linspace(0, 1, k, dtype=np.float32) ** 0.7)[:, None]
    if fin:
        k = min(n_samples(fin), len(y))
        y[:k] *= np.linspace(0, 1, k, dtype=np.float32)[:, None]
    if fout:
        k = min(n_samples(fout), len(y))
        y[-k:] *= np.linspace(1, 0, k, dtype=np.float32)[:, None]
    if peak is not None:
        y = norm(y, peak)
    return balance(y * gain, pan)


# ---------------------------------------------------------------- длинные слои

def stitch(path, dur, seed=1, xfade=1.5):
    """Непрерывный кусок нужной длины из записи: один длинный отрезок, а если запись
    короче — несколько отрезков со случайных мест, склеенных равномощным кроссфейдом
    (каждый второй проигрывается задом наперёд, чтобы не читалась петля)."""
    src = _stereo(snd(path))
    n, nx = n_samples(dur), n_samples(xfade)
    if len(src) >= n:
        st = int(np.random.default_rng(seed).integers(0, len(src) - n + 1))
        return np.array(src[st:st + n], np.float32)
    out = np.zeros((n, 2), np.float32)
    rng = np.random.default_rng(seed)
    pos, k = 0, 0
    while pos < n:
        x = src[::-1] if k % 2 else src
        L = min(len(x), n - pos + nx)
        st = int(rng.integers(0, max(1, len(x) - L + 1)))
        seg = np.array(x[st:st + L], np.float32)
        w = np.ones(len(seg), np.float32)
        if k:
            m = min(nx, len(seg))
            w[:m] = np.sqrt(np.linspace(0, 1, m, dtype=np.float32))
        if pos + L < n:
            m = min(nx, len(seg))
            w[-m:] = np.sqrt(np.linspace(1, 0, m, dtype=np.float32))
        e = min(n, pos + L)
        out[pos:e] += seg[:e - pos] * w[:e - pos, None]
        if e >= n:
            break
        pos = e - nx
        k += 1
    return out


def bed(specs, dur, amp=None, db=-26.0, seed=3, cut=5200.0, tilt=-6.0, dh=-5.0, mask=None,
        width=1.0):
    """Длинная атмосфера: specs = [(путь, вес), ...]; amp — огибающая плотности (len = n)."""
    n = n_samples(dur)
    out = np.zeros((n, 2), np.float32)
    for i, (p, g) in enumerate(specs):
        if not p:
            continue
        out += stitch(p, dur, seed=seed + 7 * i) * g
    out = np.stack([_soften(out[:, 0], cut, tilt, dh, 3600.0),
                    _soften(out[:, 1], cut, tilt, dh, 3600.0)], 1).astype(np.float32)
    if width != 1.0:
        mid = out.mean(1, keepdims=True)
        out = mid + (out - mid) * width
    if amp is not None:
        out = out * np.asarray(amp, np.float32)[:, None]
    if db is not None:
        m = np.asarray(mask) if mask is not None else (np.abs(out).max(1) > 1e-5)
        if m.ndim == 1:
            m = np.repeat(m[:, None], 2, 1)
        out = to_rms(out, db, mask=m)
    return out.astype(np.float32)


# ---------------------------------------------------------------- голоса (единственный синтез)
# Параметры — по §6 audio/PLAN.md: Clawd C3–E4, duty 0.30–0.35, ФНЧ 3.5 кГц, вибрато;
# Codex E4–A5 (на октаву ниже прежнего), duty 0.22–0.28, ФНЧ 5 кГц, ступенчатый слайд.

def _oct_down(p):
    return S.nt(p) / 2.0


def v_clawd(pitches, vol=0.26, step=0.12, gap=0.06, slide=None):
    """Реплика Clawd: «крабовый» импульс + треугольник октавой ниже + тело 140 Гц.
    1–3 слога, нисходящий слайд с вибрато — узнаётся контуром, а не яркостью."""
    parts = []
    for i, p in enumerate(pitches):
        sl = slide if (slide is not None and i == len(pitches) - 1) else None
        a = note(p, step, wave='pulse', duty=0.32, vol=0.5, atk=0.008, dec=0.05, sus=0.8,
                 rel=0.05, slide=sl, vib=(5.5, 0.22))
        b = note(_oct_down(p), step, wave='tri', vol=0.5, atk=0.008, dec=0.05, sus=0.8,
                 rel=0.05, slide=(_oct_down(sl) if sl else None)) * 0.25
        parts.append(a + b)
        parts.append(np.zeros(n_samples(gap), np.float32))
    x = np.concatenate(parts[:-1]) if parts else np.zeros(1, np.float32)
    n = len(x)
    body = np.sin(2 * np.pi * 140 * np.arange(n) / SR).astype(np.float32) * 0.316
    env = np.abs(x)
    env = np.convolve(env, np.ones(n_samples(0.01), np.float32) / n_samples(0.01), 'same')
    x = S.lp(x, 3500, order=2) + body * np.clip(env * 3, 0, 1)
    return norm(soft_attack(deharsh(x, db=-5.0, freq=3400), ms=5.0), vol)


def v_codex(pitches, vol=0.26, step=0.095, gap=0.05, tail=True):
    """Реплика Codex: импульс duty 0.25 + квадрат октавой ниже + тело 200 Гц,
    ступенчатое восхождение (микро-арпеджио) вместо вибрато, короткий дилей."""
    parts = []
    for p in pitches:
        a = note(p, step, wave='pulse', duty=0.25, vol=0.5, atk=0.004, dec=0.04, sus=0.85,
                 rel=0.03)
        b = note(_oct_down(p), step, wave='square', vol=0.5, atk=0.004, dec=0.04, sus=0.85,
                 rel=0.03) * 0.20
        parts.append(a + b)
        parts.append(np.zeros(n_samples(gap), np.float32))
    x = np.concatenate(parts[:-1]) if parts else np.zeros(1, np.float32)
    n = len(x)
    body = np.sin(2 * np.pi * 200 * np.arange(n) / SR).astype(np.float32) * 0.25
    env = np.convolve(np.abs(x), np.ones(n_samples(0.01), np.float32) / n_samples(0.01), 'same')
    x = S.lp(x, 5000, order=2) + body * np.clip(env * 3, 0, 1)
    if tail:
        x = S.delay(x, time=0.08, fb=0.2, mix=0.15, taps=3)
    return norm(soft_attack(deharsh(x, db=-6.0, freq=3600), ms=4.0), vol)


def warm_chord(pitches, dur=0.9, vol=0.3, duty=0.34):
    """Тёплый аккорд-жест («МИР!», объятие). Синтез, но приглушённый: ФНЧ 3.8 кГц."""
    n = n_samples(dur)
    x = np.zeros(n, np.float32)
    for i, p in enumerate(pitches):
        a = note(p, dur * 0.95, wave='pulse', duty=duty, vol=0.4, atk=0.02, dec=0.2,
                 sus=0.55, rel=0.4, detune=0.03 * (i - 1))
        b = note(p, dur * 0.95, wave='tri', vol=0.35, atk=0.03, dec=0.22, sus=0.5, rel=0.4)
        x[:len(a)] += (a + b)[:n]
    return norm(soft_attack(deharsh(S.lp(x, 3800, order=2), db=-5.0), ms=8.0), vol)


def sneeze(vol=0.34, seed=13):
    """«Апчхи» Clawd: мягкий шумовой выдох + падающий слог (без слов, без верха)."""
    n = n_samples(0.26)
    w = np.random.default_rng(seed).uniform(-1, 1, n).astype(np.float32)
    puff = S.bandpass(w, 420, 2600, order=2) * env_ar(n, 0.006, 3.0)
    x = np.zeros(n_samples(0.55), np.float32)
    place(x, norm(puff, 0.55), 0.0)
    place(x, v_clawd(['E4'], vol=0.75, step=0.24, slide='A2'), 0.02)
    return norm(S.lp(x, 3500, order=2), vol)


def breath_in(vol=0.14, seed=17):
    """Вдох перед чихом: мягкий восходящий шум без свиста."""
    n = n_samples(0.42)
    w = np.random.default_rng(seed).uniform(-1, 1, n).astype(np.float32)
    f = np.linspace(0, 1, n, dtype=np.float32)
    x = S.bandpass(w, 350, 1800, order=2) * (f ** 1.6) * np.linspace(1, 0.25, n, dtype=np.float32)
    return norm(S.warm(x, cutoff=3200, tilt_db=-9), vol)


def hpf(x, f=120.0, order=2):
    """ФВЧ для bed-слоёв (чтобы не мутить низ музыки)."""
    y = np.asarray(x, np.float32)
    if y.ndim == 1:
        return S.hp(y, f, order)
    return np.stack([S.hp(y[:, 0], f, order), S.hp(y[:, 1], f, order)], 1).astype(np.float32)


def lpf(x, f=9000.0, order=4):
    """ФНЧ-потолок на стем (§5 PLAN: все sfx/ui — 9 кГц)."""
    y = np.asarray(x, np.float32)
    if y.ndim == 1:
        return S.lp(y, f, order)
    return np.stack([S.lp(y[:, 0], f, order), S.lp(y[:, 1], f, order)], 1).astype(np.float32)
