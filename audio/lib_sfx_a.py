"""SFX-примитивы для первых 25 секунд (агент A) — на РЕАЛЬНЫХ сэмплах.

Правила этой версии:
  * всё ударное/шумовое/звенящее берётся из audio/samples/ (каталог + find.py);
  * синтез оставлен ровно для двух вещей — голоса-блипы Claude и Codex
    (`cl_voice`, `cl_growl`, `cd_voice`) — и для тихих подкладов под сэмпл;
  * тембр смягчается синтезовыми `warm` / `deharsh` / `soft_attack`;
  * всё моно (чтобы работал pan у Track.add), кроме фонов — они стерео.

Помощники:
    p = SM('impact', 'oga_slam_02.wav')      # абсолютный путь, с проверкой
    x = take(p, dur=0.35, warm_cut=6000)     # моно-кусок, смягчённый
    x = lay(a, b)                            # сложить слои разной длины
    x = rev(take(p))                         # реверс (получается «разгон»)
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SAMPLES = os.path.join(_ROOT, 'audio', 'samples')
sys.path.insert(0, _HERE)
sys.path.insert(0, _SAMPLES)

from synth import (SR, bitcrush, deharsh, env_adsr, load_wav, lp, n_samples,  # noqa: E402
                   noise, nt, osc, soft_attack, warm)

try:
    from find import find, info                                  # noqa: E402,F401
except ImportError:                                              # каталога нет — не беда
    find = info = None


# ---------------------------------------------------------------- загрузка сэмплов

_CACHE = {}


def SM(cat, name):
    """Абсолютный путь к сэмплу audio/samples/<cat>/<name>; падает, если файла нет."""
    p = os.path.join(_SAMPLES, cat, name)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def raw(path):
    """Сэмпл как стерео float32 48 кГц (кэшируется)."""
    if path not in _CACHE:
        _CACHE[path] = load_wav(path)
    return _CACHE[path]


def mono(path):
    x = raw(path)
    return x.mean(1).astype(np.float32)


def take(path, dur=None, t0=0.0, speed=1.0, atk=4.0, warm_cut=None, deh=None,
         fout=0.03, reverse=False, hp_cut=None):
    """Моно-кусок сэмпла, готовый к укладке.

    dur      — обрезать до стольких секунд (с плавным хвостом fout)
    t0       — начать не с нуля
    speed    — ресэмплинг: >1 выше и короче, <1 ниже и длиннее
    atk      — смягчение атаки, мс (убирает щелчок «в лоб»)
    warm_cut — прогнать через warm(cutoff) — главный инструмент против «режет уши»
    deh      — глубина провала 2.5–6 кГц в dB (отрицательная)
    """
    x = mono(path)
    if t0:
        x = x[n_samples(t0):]
    if speed != 1.0 and len(x) > 4:
        m = max(2, int(round(len(x) / speed)))
        x = np.interp(np.linspace(0, len(x) - 1, m), np.arange(len(x)), x).astype(np.float32)
    if reverse:
        x = x[::-1].copy()
    if dur is not None:
        n = n_samples(dur)
        if len(x) > n:
            x = x[:n].copy()
            k = min(n_samples(fout), len(x))
            if k > 1:
                x[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
    if hp_cut:
        from synth import hp
        x = hp(x, hp_cut)
    if warm_cut:
        x = warm(x, warm_cut)
    if deh:
        x = deharsh(x, deh)
    if atk:
        x = soft_attack(x, atk)
    return np.asarray(x, np.float32)


def stereo_take(path, dur=None, t0=0.0, loop_to=None, warm_cut=None, fin=0.0, fout=0.0):
    """Стерео-кусок (для фонов). loop_to — зациклить до стольких секунд."""
    x = raw(path)
    if t0:
        x = x[n_samples(t0):]
    if dur is not None:
        x = x[:n_samples(dur)]
    if loop_to is not None:
        n = n_samples(loop_to)
        if len(x) < n:
            xf = n_samples(0.25)
            reps = int(math.ceil(n / max(1, len(x) - xf))) + 1
            parts = [x]
            for _ in range(reps):
                head = parts[-1]
                tail = head[-xf:] * np.linspace(1, 0, xf, dtype=np.float32)[:, None]
                nxt = x.copy()
                nxt[:xf] = nxt[:xf] * np.linspace(0, 1, xf, dtype=np.float32)[:, None] + tail
                parts.append(nxt)
            x = np.concatenate([parts[0][:-xf]] + [q for q in parts[1:]])
        x = x[:n]
    if warm_cut:
        x = np.stack([warm(x[:, 0], warm_cut), warm(x[:, 1], warm_cut)], 1)
    x = np.asarray(x, np.float32).copy()
    if fin:
        k = min(n_samples(fin), len(x))
        x[:k] *= np.linspace(0, 1, k, dtype=np.float32)[:, None]
    if fout:
        k = min(n_samples(fout), len(x))
        x[-k:] *= np.linspace(1, 0, k, dtype=np.float32)[:, None]
    return x


def rev(x):
    return np.asarray(x, np.float32)[::-1].copy()


# ---------------------------------------------------------------- утилиты

def lay(*sigs):
    """Сложить слои разной длины (выравнивание по началу)."""
    sigs = [np.asarray(s, np.float32) for s in sigs if s is not None and len(s)]
    if not sigs:
        return np.zeros(1, np.float32)
    m = max(len(s) for s in sigs)
    out = np.zeros(m, np.float32)
    for s in sigs:
        out[:len(s)] += s
    return out


def at(x, delay, total=None):
    """Сдвинуть сигнал на delay секунд внутри своего же массива (для слоёв удара)."""
    x = np.asarray(x, np.float32)
    p = n_samples(delay)
    n = n_samples(total) if total else p + len(x)
    out = np.zeros(max(n, p + len(x)), np.float32)
    out[p:p + len(x)] += x
    return out


def pk(x, peak):
    x = np.asarray(x, np.float32)
    m = float(np.max(np.abs(x)))
    return x if m < 1e-9 else (x * (peak / m)).astype(np.float32)


def rms_to(x, db=-30.0):
    x = np.asarray(x, np.float32)
    r = float(np.sqrt(np.mean(x ** 2)))
    return x if r < 1e-9 else (x * (10 ** (db / 20.0) / r)).astype(np.float32)


def soft_clip(x, knee=0.60, ceil=0.86):
    """Мягкий потолок только для редких наложений; одиночное событие почти не трогает."""
    x = np.asarray(x, np.float32)
    a = np.abs(x)
    over = a > knee
    if not np.any(over):
        return x
    y = x.copy()
    y[over] = np.sign(x[over]) * (knee + (ceil - knee) * np.tanh((a[over] - knee) / (ceil - knee)))
    return y.astype(np.float32)


def bands(x, edges=(20, 250, 1200, 2500, 6000, 20000)):
    """Доли энергии по полосам (для QA): список процентов между edges."""
    x = np.asarray(x, np.float32)
    if x.ndim == 2:
        x = x.mean(1)
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    tot = X[(f >= edges[0]) & (f < edges[-1])].sum() + 1e-12
    return [100.0 * X[(f >= a) & (f < b)].sum() / tot for a, b in zip(edges[:-1], edges[1:])]


def _ctr(n, pts):
    xs = np.array([a for a, _ in pts], float)
    ys = np.array([b for _, b in pts], float)
    return np.interp(np.linspace(0.0, 1.0, n), xs, ys)


# ---------------------------------------------------------------- голоса (единственный синтез)
#
# Параметры — по §6 audio/PLAN.md:
#   Clawd: C3–E4 (130–330 Гц), duty 0.30–0.35, pulse + tri октавой ниже 25 %,
#          слог 0.10–0.14 с, зазор 0.05–0.07, 1–3 слога, слайд ±3–5 полутонов плавный,
#          вибрато 5–6 Гц глубиной 0.2–0.25, ФНЧ 3.5 кГц + подмешать 120–160 Гц на −10 дБ.
#   Codex: E4–A5 (330–880 Гц) — на октаву ниже прежнего, duty 0.22–0.28,
#          pulse + square октавой ниже 20 %, слог 0.08–0.11 с, зазор 0.04–0.06,
#          1–3 слога, слайд ±5–7 полутонов ступеньками, без вибрато, микро-арпеджио,
#          ФНЧ 5 кГц + подмешать 180–220 Гц на −12 дБ, короткий дилей 0.08 с mix 0.15.

CL_RANGE = (130.0, 330.0)
CD_RANGE = (330.0, 880.0)


def _clip_hz(f, lo, hi):
    return np.clip(f, lo, hi)


def cl_syll(dur=0.12, base=175.0, shape=(0.0, 4.0, -4.0), vol=1.0, duty=0.33, seed=0):
    """Слог Clawd: плавный слайд с вибрато, тело ниже 3.5 кГц."""
    n = n_samples(dur)
    c = _ctr(n, [(0.0, shape[0]), (0.45, shape[1]), (1.0, shape[2])])
    vib = 0.22 * np.sin(2 * np.pi * 5.5 * np.arange(n) / SR)
    f = _clip_hz(base * 2.0 ** ((c + vib) / 12.0), *CL_RANGE)
    e = env_adsr(n, 0.008, 0.045, 0.80, 0.055)
    x = osc(f, n, 'pulse', duty) * 0.75 + osc(f * 0.5, n, 'tri') * 0.25
    x = (x * e).astype(np.float32)
    body = (osc(np.full(n, 140.0), n, 'sine') * e * 10 ** (-10 / 20.0)).astype(np.float32)
    return pk(lp(bitcrush(x + body, 6), 3500, order=3), vol)


def cd_syll(dur=0.10, base=420.0, shape=(0.0, 6.0, 2.0), steps=4, vol=1.0, duty=0.25, seed=0):
    """Слог Codex: ступенчатый слайд, октавой ниже прежней версии, тело ниже 5 кГц."""
    n = n_samples(dur)
    c = np.round(_ctr(steps, [(0.0, shape[0]), (0.5, shape[1]), (1.0, shape[2])]))
    c = np.repeat(c, int(math.ceil(n / steps)))[:n]
    f = _clip_hz(base * 2.0 ** (c / 12.0), *CD_RANGE)
    e = env_adsr(n, 0.006, 0.03, 0.88, 0.030)
    x = osc(f, n, 'pulse', duty) * 0.80 + osc(f * 0.5, n, 'square') * 0.20
    x = (x * e).astype(np.float32)
    body = (osc(np.full(n, 200.0), n, 'sine') * e * 10 ** (-12 / 20.0)).astype(np.float32)
    return pk(lp(bitcrush(x + body, 6), 5000, order=3), vol)


def _tail_delay(x, time=0.08, mix=0.15):
    d = n_samples(time)
    out = np.zeros(len(x) + d, np.float32)
    out[:len(x)] += x
    out[d:d + len(x)] += x * mix
    return out


def cl_voice(sylls, base=175.0, gap=0.06, vol=0.65, seed=0):
    """Реплика Clawd: 1–3 слога, [(длительность, контур ±3–5 полутонов, сдвиг базы)]."""
    out = []
    for i, (d, shape, off) in enumerate(sylls[:3]):
        out.append(cl_syll(min(max(d, 0.10), 0.14), base * 2 ** (off / 12.0), shape, seed=seed + i * 3))
        out.append(np.zeros(n_samples(gap), np.float32))
    return pk(soft_attack(np.concatenate(out), 6.0), vol)


def cd_voice(sylls, base=420.0, gap=0.05, vol=0.62, seed=0, steps=4, duty=0.25):
    """Реплика Codex: 1–3 слога, ступенчатый контур, короткий дилей в хвосте."""
    out = []
    for i, (d, shape, off) in enumerate(sylls[:3]):
        out.append(cd_syll(min(max(d, 0.08), 0.11), base * 2 ** (off / 12.0), shape,
                           steps=steps, duty=duty, seed=seed + i * 3))
        out.append(np.zeros(n_samples(gap), np.float32))
    x = _tail_delay(np.concatenate(out), 0.08, 0.15)
    return pk(soft_attack(x, 5.0), vol)


def cl_growl(dur=0.20, base=132.0, vol=0.5, seed=0):
    """Низкое сердитое «грр» краба — тоже голос, поэтому синтез."""
    n = n_samples(dur)
    f = base * 2.0 ** (_ctr(n, [(0.0, 2.0), (0.5, -1.0), (1.0, -4.0)]) / 12.0)
    f = f * (1 + 0.06 * np.sin(2 * np.pi * 5.5 * np.arange(n) / SR))
    e = env_adsr(n, 0.012, 0.07, 0.72, 0.08)
    x = osc(f, n, 'pulse', 0.33) * 0.72 + osc(f * 0.5, n, 'tri') * 0.28
    x = lay((x * e).astype(np.float32), lp(noise(n, 61 + seed), 380) * e * 0.22)
    return pk(lp(x, 3500, order=3), vol)


def sub(dur=0.22, f0=90.0, f1=45.0, vol=0.3):
    """Тихий низ-подклад под сэмпл удара (когда сэмплу не хватает тела 60–200 Гц)."""
    n = n_samples(dur)
    f = np.geomspace(f0, f1, n)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR).astype(np.float32) * env_adsr(n, 0.004, 0.06, 0.4, 0.12)
    return pk(x, vol)


def stereo_pair(left, right, offset=0.0, peak=0.5):
    """Один cue из двух слоёв в разных каналах (например, две панели сплита).

    По П9 плана два одновременных звука — это одно событие, а не два cue."""
    nr = n_samples(offset)
    n = max(len(left), nr + len(right))
    out = np.zeros((n, 2), np.float32)
    out[:len(left), 0] += left
    out[nr:nr + len(right), 1] += right
    out[:len(left), 1] += left * 0.28
    out[nr:nr + len(right), 0] += right * 0.28
    m = float(np.max(np.abs(out))) or 1.0
    return (out * (peak / m)).astype(np.float32)


__all__ = [n for n in dir() if not n.startswith('_')]
