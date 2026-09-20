"""Чиптюн-синтез для саундтрека мультика. Только numpy, 48 кГц, float32.

Базовое:
    x = note('A4', 0.25, wave='pulse', duty=0.25, vol=0.6)      # моно-сигнал
    x = seq([('C4',1),('E4',1),(None,1),('G4',2)], bpm=132)     # мелодия
    t = Track(8.0); t.add(x, 0.0, gain=0.8, pan=-0.3); t.save('stems/foo.wav')

Барабаны: kick(), snare(), hat(), openhat(), crash(), tom('G2'), click()
Эффекты:  lp/hp/bandpass, delay, reverb, bitcrush, drive, chorus, fade, gain_db,
          pitch_sweep, sweep_noise, arp, vibrato_note, glide
Всё возвращает float32 mono (кроме Track.stereo/save).
"""
import json
import math
import os
import struct
import wave

import numpy as np

SR = 48000
_RNG = np.random.default_rng(1234)

# ---------------------------------------------------------------- ноты

_STEPS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def nt(name):
    """'A4' -> 440.0 ; 'F#3', 'Bb5' тоже; число возвращается как есть."""
    if isinstance(name, (int, float)):
        return float(name)
    s = name.strip()
    semi = _STEPS[s[0].upper()]
    i = 1
    while i < len(s) and s[i] in '#b':
        semi += 1 if s[i] == '#' else -1
        i += 1
    octv = int(s[i:])
    return 440.0 * 2 ** ((semi - 9) / 12 + (octv - 4))


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic': [0, 2, 3, 5, 7, 8, 11],
    'penta': [0, 3, 5, 7, 10],
    'pentamaj': [0, 2, 4, 7, 9],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
}


def scale(root, kind='major', n=16, start_oct=0):
    """Список частот по ступеням гаммы вверх от root ('C4')."""
    f0 = nt(root)
    st = SCALES[kind]
    out = []
    for i in range(n):
        o, d = divmod(i + start_oct * len(st), len(st))
        out.append(f0 * 2 ** (o + st[d] / 12))
    return out


# ---------------------------------------------------------------- огибающие

def n_samples(dur):
    return max(1, int(round(dur * SR)))


def env_adsr(n, atk=0.005, dec=0.05, sus=0.7, rel=0.08, curve=1.0):
    a, d, r = n_samples(atk), n_samples(dec), n_samples(rel)
    a = min(a, n)
    d = min(d, max(0, n - a))
    r = min(r, max(0, n - a - d))
    s = max(0, n - a - d - r)
    parts = [np.linspace(0, 1, a, endpoint=False, dtype=np.float32) if a else np.zeros(0, np.float32),
             np.linspace(1, sus, d, endpoint=False, dtype=np.float32) if d else np.zeros(0, np.float32),
             np.full(s, sus, np.float32),
             np.linspace(sus, 0, r, dtype=np.float32) if r else np.zeros(0, np.float32)]
    e = np.concatenate(parts)[:n]
    if len(e) < n:
        e = np.concatenate([e, np.zeros(n - len(e), np.float32)])
    return e ** curve if curve != 1.0 else e


def env_ar(n, atk=0.002, curve=2.0):
    """Ударная огибающая: резкая атака + экспоненциальный спад."""
    a = min(n_samples(atk), n)
    e = np.empty(n, np.float32)
    e[:a] = np.linspace(0, 1, a, dtype=np.float32) if a else 0
    k = np.linspace(0, 1, max(1, n - a), dtype=np.float32)
    e[a:] = (1 - k) ** curve
    return e


def env_exp(n, tau=0.15):
    return np.exp(-np.arange(n, dtype=np.float32) / (tau * SR))


# ---------------------------------------------------------------- осцилляторы

def _phase(freq, n):
    if np.isscalar(freq):
        return (np.arange(n, dtype=np.float64) * (freq / SR)) % 1.0
    f = np.asarray(freq, dtype=np.float64)
    if len(f) != n:
        f = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(f)), f)
    return (np.cumsum(f) / SR) % 1.0


def osc(freq, n, wave='square', duty=0.5, seed=None):
    """wave: square|pulse|tri|saw|sine|noise|metal."""
    if wave == 'noise':
        return noise(n, seed)
    if wave == 'metal':
        p = _phase(freq, n)
        q = _phase(np.asarray(freq) * 1.4983 if np.isscalar(freq) else np.asarray(freq) * 1.4983, n)
        return ((p < 0.5).astype(np.float32) * 2 - 1) * 0.6 + ((q < 0.5).astype(np.float32) * 2 - 1) * 0.5
    p = _phase(freq, n)
    if wave in ('square', 'pulse'):
        return ((p < duty).astype(np.float32) * 2 - 1)
    if wave == 'tri':
        return (2 * np.abs(2 * (p - np.floor(p + 0.5))) - 1).astype(np.float32)
    if wave == 'saw':
        return (2 * p - 1).astype(np.float32)
    if wave == 'sine':
        return np.sin(2 * np.pi * p).astype(np.float32)
    raise ValueError(wave)


def noise(n, seed=None, rate=None):
    """Белый шум; rate (Гц) — sample&hold, даёт грубый 8-битный тембр."""
    rng = _RNG if seed is None else np.random.default_rng(seed)
    if rate:
        m = max(1, int(round(n * rate / SR)))
        return np.repeat(rng.uniform(-1, 1, m).astype(np.float32), int(math.ceil(n / m)))[:n]
    return rng.uniform(-1, 1, n).astype(np.float32)


def note(pitch, dur, wave='square', duty=0.5, vol=0.5, atk=0.004, dec=0.04, sus=0.75, rel=0.06,
         vib=None, slide=None, detune=0.0, seed=None, curve=1.0):
    """Одна нота. vib=(гц, глубина_в_полутонах); slide='E5' — глиссандо к ноте."""
    n = n_samples(dur)
    f0 = nt(pitch)
    f = np.full(n, f0, np.float64)
    if slide is not None:
        f = np.geomspace(f0, nt(slide), n)
    if vib:
        hz, depth = vib
        f = f * 2 ** (depth * np.sin(2 * np.pi * hz * np.arange(n) / SR) / 12)
    if detune:
        f = f * 2 ** (detune / 12)
    x = osc(f, n, wave, duty, seed) * env_adsr(n, atk, dec, sus, rel, curve)
    return (x * vol).astype(np.float32)


def seq(events, bpm=120, wave='square', duty=0.5, vol=0.5, legato=0.92, **kw):
    """events: [(нота|None, длительность_в_долях), ...] -> один массив."""
    spb = 60.0 / bpm
    out = []
    for item in events:
        p, b = item[0], item[1]
        extra = item[2] if len(item) > 2 else {}
        d = b * spb
        n = n_samples(d)
        if p is None:
            out.append(np.zeros(n, np.float32))
            continue
        kk = dict(wave=wave, duty=duty, vol=vol, **kw)
        kk.update(extra)
        x = note(p, d * legato, **kk)
        out.append(np.concatenate([x, np.zeros(max(0, n - len(x)), np.float32)])[:n])
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def chord(pitches, dur, wave='square', duty=0.5, vol=0.35, spread=0.0, **kw):
    n = n_samples(dur)
    out = np.zeros(n, np.float32)
    for i, p in enumerate(pitches):
        x = note(p, dur, wave=wave, duty=duty, vol=vol, detune=spread * (i - len(pitches) / 2), **kw)
        out[:len(x)] += x[:n]
    return out


def arp(pitches, dur, rate=16, bpm=120, wave='square', duty=0.25, vol=0.4, **kw):
    """Арпеджио: перебор pitches шагами 1/rate доли на протяжении dur секунд."""
    step = 60.0 / bpm * 4 / rate
    n = max(1, int(round(dur / step)))
    return seq([(pitches[i % len(pitches)], step * bpm / 60) for i in range(n)],
               bpm=bpm, wave=wave, duty=duty, vol=vol, **kw)


# ---------------------------------------------------------------- барабаны

def kick(dur=0.20, f0=170, f1=42, vol=0.9, click=0.35):
    n = n_samples(dur)
    f = np.geomspace(f0, f1, n)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR).astype(np.float32) * env_ar(n, 0.001, 2.2)
    if click:
        m = n_samples(0.006)
        x[:m] += noise(m, 7) * click
    return (x * vol).astype(np.float32)


def snare(dur=0.16, vol=0.7, tone=190, bright=1.0):
    n = n_samples(dur)
    x = noise(n, 11, rate=int(9000 * bright)) * env_ar(n, 0.001, 2.6)
    x += np.sin(2 * np.pi * np.cumsum(np.geomspace(tone * 1.6, tone, n)) / SR).astype(np.float32) * env_ar(n, 0.001, 4) * 0.5
    return (hp(x, 220) * vol).astype(np.float32)


def hat(dur=0.045, vol=0.35, seed=21):
    n = n_samples(dur)
    return (hp(noise(n, seed, rate=24000) * env_ar(n, 0.0005, 3.5), 6500) * vol).astype(np.float32)


def openhat(dur=0.22, vol=0.3, seed=23):
    n = n_samples(dur)
    return (hp(noise(n, seed, rate=22000) * env_ar(n, 0.001, 1.6), 5500) * vol).astype(np.float32)


def crash(dur=1.1, vol=0.5, seed=31):
    n = n_samples(dur)
    x = noise(n, seed, rate=30000) * env_ar(n, 0.002, 1.1)
    return (hp(x, 3000) * vol).astype(np.float32)


def tom(pitch='G2', dur=0.22, vol=0.6):
    n = n_samples(dur)
    f0 = nt(pitch)
    f = np.geomspace(f0 * 1.7, f0, n)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR).astype(np.float32) * env_ar(n, 0.001, 2.0)
    return (x * vol).astype(np.float32)


def click(dur=0.02, f=2200, vol=0.5):
    n = n_samples(dur)
    return (osc(f, n, 'square') * env_ar(n, 0.0005, 4) * vol).astype(np.float32)


def drums(pattern, bpm=120, step=16, kit=None, vol=1.0):
    """pattern: {'k': 'x...x...x...x...', 's': '....x.......x...', 'h': 'x.x.x.x.'} .
    Символы: x громко, o тихо, . пауза. Длина строки — число шагов 1/step доли."""
    kit = kit or {'k': kick, 's': snare, 'h': hat, 'H': openhat, 'c': crash, 't': tom}
    sps = 60.0 / bpm * 4 / step
    L = max(len(v) for v in pattern.values())
    out = np.zeros(n_samples(L * sps) + SR, np.float32)
    for key, row in pattern.items():
        gen = kit[key]
        for i, ch in enumerate(row):
            if ch in '.- ':
                continue
            g = 1.0 if ch == 'x' else (0.55 if ch == 'o' else 0.8)
            s = gen()
            p = n_samples(i * sps)
            out[p:p + len(s)] += s * g * vol
    return out


# ---------------------------------------------------------------- эффекты (FFT, быстро)

def _fft_filter(x, resp_fn):
    n = len(x)
    if n == 0:
        return x
    N = 1 << (n - 1).bit_length()
    X = np.fft.rfft(x, N)
    f = np.fft.rfftfreq(N, 1 / SR)
    return np.fft.irfft(X * resp_fn(f), N)[:n].astype(np.float32)


def lp(x, cutoff, order=2):
    return _fft_filter(x, lambda f: 1 / np.sqrt(1 + (f / max(1e-3, cutoff)) ** (2 * order)))


def hp(x, cutoff, order=2):
    return _fft_filter(x, lambda f: 1 / np.sqrt(1 + (max(1e-3, cutoff) / np.maximum(f, 1e-3)) ** (2 * order)))


def bandpass(x, lo, hi, order=2):
    return lp(hp(x, lo, order), hi, order)


def peak_eq(x, freq, gain_db, q=1.0):
    g = 10 ** (gain_db / 20) - 1
    return _fft_filter(x, lambda f: 1 + g * np.exp(-((np.log2(np.maximum(f, 1e-3) / freq) * q * 2) ** 2)))


def delay(x, time=0.18, fb=0.35, mix=0.35, taps=14, pingpong=False):
    d = n_samples(time)
    out = np.concatenate([x, np.zeros(d * taps, np.float32)]).astype(np.float32)
    wet = np.zeros_like(out)
    g = 1.0
    for k in range(1, taps + 1):
        g *= fb
        if g < 0.002:
            break
        wet[k * d:k * d + len(x)] += x * g
    return (out + wet * mix).astype(np.float32)


def reverb(x, room=0.45, mix=0.3, damp=4500, pre=0.012, seed=5):
    """Свёртка с затухающим шумом — дешёвый «зал»."""
    n_ir = n_samples(room)
    rng = np.random.default_rng(seed)
    ir = rng.uniform(-1, 1, n_ir).astype(np.float32) * np.exp(-np.linspace(0, 6, n_ir, dtype=np.float32))
    ir = lp(ir, damp)
    ir = np.concatenate([np.zeros(n_samples(pre), np.float32), ir])
    ir /= (np.sqrt(np.sum(ir ** 2)) + 1e-9)
    n = len(x) + len(ir) - 1
    N = 1 << (n - 1).bit_length()
    wet = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[:n].astype(np.float32)
    dry = np.concatenate([x, np.zeros(n - len(x), np.float32)])
    return (dry * (1 - mix * 0.5) + wet * mix * 1.6).astype(np.float32)


def bitcrush(x, bits=4, rate=None):
    y = x
    if rate:
        step = max(1, int(SR / rate))
        y = np.repeat(y[::step], step)[:len(x)]
    lv = 2 ** bits
    return (np.round(y * lv) / lv).astype(np.float32)


def drive(x, amount=3.0):
    return np.tanh(x * amount).astype(np.float32) / np.tanh(amount)


def chorus(x, depth=0.004, rate=1.3, mix=0.4):
    n = len(x)
    t = np.arange(n) / SR
    d = (depth * (1 + np.sin(2 * np.pi * rate * t)) / 2 * SR).astype(np.int32)
    idx = np.clip(np.arange(n) - d, 0, n - 1)
    return (x * (1 - mix) + x[idx] * mix).astype(np.float32)


def pitch_sweep(f0, f1, dur, wave='square', vol=0.5, duty=0.5, curve='exp', **kw):
    n = n_samples(dur)
    a, b = nt(f0), nt(f1)
    f = np.geomspace(a, b, n) if curve == 'exp' else np.linspace(a, b, n)
    e = kw.pop('env', None)
    x = osc(f, n, wave, duty)
    x = x * (env_adsr(n, **kw) if e is None else e)
    return (x * vol).astype(np.float32)


def sweep_noise(dur, f0=8000, f1=400, vol=0.5, seed=3, q=2):
    """Вжух: полосовой шум с едущей частотой (блоками по 512)."""
    n = n_samples(dur)
    x = noise(n, seed)
    out = np.zeros(n, np.float32)
    B = 1024
    fs = np.geomspace(f0, f1, max(1, n // B) + 1)
    for i, s in enumerate(range(0, n, B)):
        seg = x[s:s + B]
        f = fs[min(i, len(fs) - 1)]
        out[s:s + len(seg)] = bandpass(seg, f / q, f * q)
    return (out * env_adsr(n, 0.02, 0.05, 0.9, 0.15) * vol).astype(np.float32)


def warm(x, cutoff=7000, tilt_db=-5.0, pivot=800.0):
    """Смягчение тембра: ФНЧ как у чип-ЦАПа + плавный наклон спектра вниз выше pivot.
    Снимает «стеклянный» верх у квадратов и шумов, не трогая тело звука."""
    y = lp(x, cutoff, order=2)
    return _fft_filter(y, lambda f: 10 ** (tilt_db / 20 * np.clip(np.log2(np.maximum(f, 1.0) / pivot) / 4, 0, 1)))


def deharsh(x, db=-6.0, freq=3600.0, q=0.55):
    """Широкий провал в полосе 2.5–6 кГц — там живёт боль в ушах."""
    return peak_eq(x, freq, db, q)


def soft_attack(x, ms=4.0, curve=0.7):
    """Короткое смягчение атаки: убирает щелчок в первом сэмпле."""
    n = min(n_samples(ms / 1000.0), len(x))
    if n < 2:
        return x
    y = x.copy()
    y[:n] *= np.linspace(0, 1, n, dtype=np.float32) ** curve
    return y


def fade(x, fin=0.0, fout=0.0):
    x = x.copy()
    a, b = n_samples(fin) if fin else 0, n_samples(fout) if fout else 0
    if a:
        x[:a] *= np.linspace(0, 1, min(a, len(x)), dtype=np.float32)[:len(x)]
    if b and b < len(x):
        x[-b:] *= np.linspace(1, 0, b, dtype=np.float32)
    return x


def gain_db(x, db):
    return (x * 10 ** (db / 20)).astype(np.float32)


def normalize(x, peak=0.9):
    m = float(np.max(np.abs(x))) or 1.0
    return (x * (peak / m)).astype(np.float32)


def loop_to(x, dur, xfade=0.0):
    """Зациклить сигнал до нужной длины."""
    n = n_samples(dur)
    if len(x) == 0:
        return np.zeros(n, np.float32)
    reps = int(math.ceil(n / len(x))) + 1
    return np.tile(x, reps)[:n].astype(np.float32)


def silence(dur):
    return np.zeros(n_samples(dur), np.float32)


# ---------------------------------------------------------------- дорожка

class Track:
    """Стерео-буфер фиксированной длины; add() кладёт сигнал на абсолютную секунду."""

    def __init__(self, dur, sr=SR):
        self.sr = sr
        self.buf = np.zeros((n_samples(dur), 2), np.float32)

    @property
    def dur(self):
        return len(self.buf) / self.sr

    def add(self, sig, t, gain=1.0, pan=0.0, fin=0.0, fout=0.0):
        if sig is None or len(sig) == 0:
            return self
        x = fade(np.asarray(sig, np.float32), fin, fout) * gain
        p = int(round(t * self.sr))
        if p < 0:
            x, p = x[-p:], 0
        if p >= len(self.buf) or len(x) == 0:
            return self
        x = x[:len(self.buf) - p]
        l = math.sqrt((1 - pan) / 2) * math.sqrt(2)
        r = math.sqrt((1 + pan) / 2) * math.sqrt(2)
        if x.ndim == 2:
            self.buf[p:p + len(x)] += x
        else:
            self.buf[p:p + len(x), 0] += x * l
            self.buf[p:p + len(x), 1] += x * r
        return self

    def add_stereo(self, left, right, t, gain=1.0):
        n = min(len(left), len(right))
        self.add(np.stack([left[:n], right[:n]], 1) * gain, t)
        return self

    def stereo(self):
        return self.buf

    def peak(self):
        return float(np.max(np.abs(self.buf))) if len(self.buf) else 0.0

    def save(self, path, limit=True):
        x = self.buf
        if limit:
            x = np.tanh(x * 1.05).astype(np.float32)
        save_wav(path, x, self.sr)
        return path


def save_wav(path, x, sr=SR):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    d = np.clip(x, -1.0, 1.0)
    pcm = (d * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def load_wav(path, target_sr=SR):
    with wave.open(path, 'rb') as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, '<i2').astype(np.float32) / 32768
    elif sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    else:
        x = np.frombuffer(raw, '<i4').astype(np.float32) / 2147483648
    x = x.reshape(-1, ch)
    if ch == 1:
        x = np.repeat(x, 2, 1)
    elif ch > 2:
        x = x[:, :2]
    if sr != target_sr:
        m = int(round(len(x) * target_sr / sr))
        idx = np.linspace(0, len(x) - 1, m)
        x = np.stack([np.interp(idx, np.arange(len(x)), x[:, 0]),
                      np.interp(idx, np.arange(len(x)), x[:, 1])], 1)
    return x.astype(np.float32)


def write_manifest(path, stems):
    """stems: [{'file': 'stems/x.wav', 't': 13.0, 'gain': 1.0, 'pan': 0.0, 'bus': 'sfx'}]"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'stems': stems}, f, ensure_ascii=False, indent=1)
    return path
