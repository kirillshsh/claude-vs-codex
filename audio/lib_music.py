"""Помощники для чиптюн-саундтрека (audio/make_music.py).

Две главные идеи.

1. ДИАТОНИКА ПО ПОСТРОЕНИЮ. Все высоты берутся только через Key.m(ступень, октава),
   каждая нота логируется, check_keys() численно доказывает, что стем не вышел из гаммы.

2. МЯГКИЙ ТЕМБР ПО УМОЛЧАНИЮ. Каждый голос (а не только итоговый стем) проходит через
   soften() = soft_attack + warm + deharsh из synth.py, и через _tame(), который выше C5
   не даёт играть узкими импульсами (duty<.4 -> .45), а выше C6 переводит на треугольник.
   Явный bright=True отключает укрощение там, где нужен резкий акцент.

       k = Key('C', 'major')
       begin('m1_meadow', k)
       x = SQ([(k.m(0, 5), 1)], bpm=132, wave='tri', tone=dict(cutoff=6000))
       tr.add(dr.kick(), 0.0)                 # живой сэмпл из audio/samples/percussion
       check_keys()

Барабаны — реальные сэмплы (модуль dr ниже), сигнатуры совместимы с synth.kick/snare/...,
так что вызовы не отличаются от синтезированных.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth as S  # noqa: E402
from synth import SR, Track  # noqa: E402,F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PC_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_ROOT_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def pc(name):
    """'F#' -> 6, 'Bb' -> 10."""
    p = _ROOT_PC[name[0].upper()]
    for ch in name[1:]:
        p += 1 if ch == '#' else (-1 if ch == 'b' else 0)
    return p % 12


class Key:
    """Тональность: корень + лад. m(ступень, октава) -> MIDI-номер.

    Ступени 0-based и могут быть отрицательными / больше октавы — Key сам
    переносит по октавам, поэтому выйти из гаммы невозможно.
    """

    def __init__(self, root, kind='major'):
        self.root, self.kind = root, kind
        self.root_pc = pc(root)
        self.steps = S.SCALES[kind]

    @property
    def pcs(self):
        return sorted({(self.root_pc + s) % 12 for s in self.steps})

    @property
    def label(self):
        return f'{self.root} {self.kind}'

    def m(self, degree, octave=4):
        o, d = divmod(int(degree), len(self.steps))
        return 12 * (octave + 1) + self.root_pc + self.steps[d] + 12 * o

    def f(self, degree, octave=4):
        return S.midi(self.m(degree, octave))

    def triad(self, degree, octave=3, n=3):
        """Трезвучие/септаккорд от ступени: [d, d+2, d+4, ...]."""
        return [self.m(degree + 2 * i, octave) for i in range(n)]

    def notes(self, degrees, octave=4):
        return [self.m(d, octave) for d in degrees]


# ---------------------------------------------------------------- лог нот

_LOG = {}
_KEYS = {}
_CUR = [None]


def begin(label, key):
    _CUR[0] = label
    _LOG.setdefault(label, set())
    _KEYS[label] = key


def log(ms):
    lbl = _CUR[0]
    if lbl is None:
        return
    it = ms if isinstance(ms, (list, tuple, set, np.ndarray)) else [ms]
    for m in it:
        if m is not None:
            _LOG[lbl].add(int(m) % 12)


def check_keys(verbose=True):
    """Печатает и проверяет: все ноты каждого стема лежат в заявленной гамме."""
    ok = True
    if verbose:
        print('\n--- проверка тональности ---')
    for lbl, used in _LOG.items():
        key = _KEYS[lbl]
        allowed = set(key.pcs)
        bad = sorted(used - allowed)
        if bad:
            ok = False
        if verbose:
            print(f'  {lbl:16s} {key.label:12s} гамма [{" ".join(PC_NAMES[p] for p in key.pcs)}]'
                  f'  ноты [{" ".join(PC_NAMES[p] for p in sorted(used))}]'
                  f'  {"OK" if not bad else "ЧУЖИЕ: " + " ".join(PC_NAMES[p] for p in bad)}')
    return ok


# ---------------------------------------------------------------- тембр

# базовая «чип-ЦАП» обработка каждого голоса
TONE = dict(cutoff=8500.0, tilt=-3.5, harsh=-4.0, pivot=1000.0)
ATK_MIN = 0.006          # 6 мс — ни одна нота не начинается щелчком
SOFT_C5, SOFT_C6 = 72, 84


def soften(x, cutoff=None, tilt=None, harsh=None, pivot=None, atk=0.0):
    """ФНЧ чип-ЦАПа + наклон вниз + провал в болевой полосе 2.5–6 кГц."""
    c = TONE['cutoff'] if cutoff is None else cutoff
    t = TONE['tilt'] if tilt is None else tilt
    h = TONE['harsh'] if harsh is None else harsh
    p = TONE['pivot'] if pivot is None else pivot
    if atk:
        x = S.soft_attack(x, atk)
    if c:
        x = S.warm(x, c, t, p)
    if h:
        x = S.deharsh(x, h)
    return np.asarray(x, np.float32)


def _tame(m, wave, duty):
    """Регистровое правило: чем выше нота, тем мягче обязан быть тембр."""
    if m is None or wave not in ('square', 'pulse', 'saw'):
        return wave, duty
    if wave == 'saw':
        return ('tri', duty) if m >= SOFT_C5 else (wave, duty)
    if m >= SOFT_C6:
        return 'tri', 0.5
    if m >= SOFT_C5 and duty < 0.4:
        return wave, 0.45
    return wave, duty


# ---------------------------------------------------------------- генераторы (лог + тембр)

def N(m, dur, wave='tri', duty=0.5, bright=False, tone=None, **kw):
    """Одна нота по MIDI-номеру."""
    log(m)
    kw.setdefault('atk', ATK_MIN)
    w, d = (wave, duty) if bright else _tame(m, wave, duty)
    x = S.note(S.midi(m), dur, wave=w, duty=d, **kw)
    return x if tone is False else soften(x, **(tone or {}))


def CH(ms, dur, wave='tri', duty=0.5, vol=0.3, spread=0.0, bright=False, tone=None, **kw):
    """Аккорд: каждый голос укрощается отдельно по своему регистру."""
    log(ms)
    kw.setdefault('atk', ATK_MIN)
    n = S.n_samples(dur)
    out = np.zeros(n, np.float32)
    for i, m in enumerate(ms):
        w, d = (wave, duty) if bright else _tame(m, wave, duty)
        x = S.note(S.midi(m), dur, wave=w, duty=d, vol=vol,
                   detune=spread * (i - len(ms) / 2), **kw)
        out[:len(x)] += x[:n]
    return out if tone is False else soften(out, **(tone or {}))


def SQ(events, bpm, wave='tri', duty=0.5, bright=False, tone=None, **kw):
    """events: [(midi|None, доли), ...] или [(midi, доли, {доп. kwargs}), ...]."""
    kw.setdefault('atk', ATK_MIN)
    ev = []
    for it in events:
        m, b = it[0], it[1]
        log(m)
        extra = dict(it[2]) if len(it) > 2 else {}
        if not bright:
            w, d = _tame(m, extra.get('wave', wave), extra.get('duty', duty))
            extra['wave'], extra['duty'] = w, d
        e = (None if m is None else S.midi(m), b)
        ev.append(e + ((extra,) if extra else ()))
    x = S.seq(ev, bpm=bpm, wave=wave, duty=duty, **kw)
    return x if tone is False else soften(x, **(tone or {}))


def ARP(ms, dur, rate=16, bpm=120, wave='tri', duty=0.5, vol=0.4, **kw):
    """Арпеджио: перебор ms шагами 1/rate доли на протяжении dur секунд."""
    step = 60.0 / bpm * 4 / rate
    n = max(1, int(round(dur / step)))
    return SQ([(ms[i % len(ms)], step * bpm / 60) for i in range(n)],
              bpm, wave=wave, duty=duty, vol=vol, **kw)


def SWEEP(m0, m1, dur, tone=None, **kw):
    """Глиссандо между двумя нотами гаммы (логируются обе)."""
    log([m0, m1])
    x = S.pitch_sweep(S.midi(m0), S.midi(m1), dur, **kw)
    return x if tone is False else soften(x, **(tone or dict(cutoff=4200, harsh=-6)))


# ---------------------------------------------------------------- лейтмотивы

RHY = [1.0, 0.5, 0.5, 2.0]          # общий ритм обоих мотивов (в долях)
CLAUDE_DEG = [0, 2, 4, 5]           # светлый подъём 1–3–5–6
CODEX_DEG = [0, -2, -4, -5]         # зеркальный спуск (минорная окраска)

_ROOT_CLAUDE = {'major': 0, 'lydian': 0, 'pentamaj': 0, 'minor': 2, 'harmonic': 2, 'dorian': 2}
_ROOT_CODEX = {'major': 5, 'lydian': 5, 'pentamaj': 5, 'minor': 0, 'harmonic': 0, 'dorian': 4}


def motif_degs(key, who, root=None):
    base = (_ROOT_CLAUDE if who == 'claude' else _ROOT_CODEX)[key.kind] if root is None else root
    degs = CLAUDE_DEG if who == 'claude' else CODEX_DEG
    return [base + d for d in degs]


def motif(key, who, bpm, octave=5, scale=1.0, root=None, rhy=None, **kw):
    """Четырёхнотный лейтмотив. scale сжимает/растягивает общий ритм."""
    degs = motif_degs(key, who, root)
    r = rhy or RHY
    return SQ([(key.m(d, octave), b * scale) for d, b in zip(degs, r)], bpm, **kw)


def motif_notes(key, who, octave=5, root=None):
    return [key.m(d, octave) for d in motif_degs(key, who, root)]


# ---------------------------------------------------------------- фактуры

def pad(key, deg, dur, octave=3, n=3, vol=0.22, wave='tri', atk=0.08, rel=None,
        cut=2600, rev=0.0, spread=0.0):
    """Мягкая подложка: не голос, а «воздух» — режется низко, чтобы не мутить середину."""
    rel = dur * 0.45 if rel is None else rel
    x = CH(key.triad(deg, octave, n), dur, wave=wave, vol=vol, atk=atk, dec=0.12,
           sus=0.85, rel=rel, spread=spread, tone=dict(cutoff=cut, tilt=-3, harsh=-3))
    if rev:
        x = S.reverb(x, room=0.55, mix=rev)
    return x


def stab(key, deg, dur, octave=3, n=4, vol=0.3, wave='pulse', duty=0.5, drv=0.0,
         cut=6000, harsh=-5.0):
    """Короткий аккорд-акцент. Драйв добавляет гармоник — поэтому режем после него."""
    x = CH(key.triad(deg, octave, n), dur, wave=wave, duty=duty, vol=vol,
           atk=max(0.004, ATK_MIN * 0.7), dec=0.05, sus=0.5, rel=max(0.02, dur * 0.35),
           tone=False)
    if drv:
        x = S.drive(x, drv)
    return soften(x, cutoff=cut, harsh=harsh, atk=3.0)


BASS_8TH = [0, 4, 2, 4, 7, 4, 2, 4]      # «ходячий» рисунок по аккордовым тонам


def bass_bar(key, deg, bar_len, bpm, octave=2, pattern=None, vol=0.42, wave='tri', **kw):
    """Бас на один такт. Низ не режет уши — ему достаточно мягкого ФНЧ."""
    p = BASS_8TH if pattern is None else pattern
    beats = bar_len / (60.0 / bpm) / len(p)
    kw.setdefault('tone', dict(cutoff=4500, tilt=-3, harsh=-2))
    return SQ([(key.m(deg + d, octave), beats) for d in p], bpm, wave=wave, vol=vol,
              legato=0.86, **kw)


# ---------------------------------------------------------------- живые барабаны

class _Drums:
    """Ударные из audio/samples/percussion. Сигнатуры совместимы с synth.kick/snare/...

    Выбраны самые тёплые по hi_ratio (доля энергии выше 5 кГц) экземпляры из каталога,
    каждый дополнительно обрезается по длине и прогоняется через warm/deharsh.
    """

    KICK = 'audio/samples/percussion/drum_r8_kick.wav'              # hi 0.000, centroid 62 Гц
    SNARE = 'audio/samples/percussion/drum_kpr77_snare.wav'         # hi 0.016, centroid 431 Гц
    SNARE_B = 'audio/samples/percussion/drum_breakbeat13_snare.wav'  # hi 0.018, centroid 220 Гц
    HAT = 'audio/samples/percussion/drum_thecheebacabra2_hihat.wav'  # hi 0.036, centroid 1576 Гц
    OPENHAT = 'audio/samples/percussion/drum_breakbeat9_hihat.wav'   # hi 0.088, 1.2 c — годится в «тарелку»
    TOMS = [('A2', 'audio/samples/percussion/drum_r8_tom1.wav'),
            ('F2', 'audio/samples/percussion/drum_r8_tom2.wav'),
            ('C2', 'audio/samples/percussion/drum_r8_tom3.wav')]

    def __init__(self):
        self._c = {}
        self.available = all(os.path.exists(os.path.join(ROOT, p))
                             for p in (self.KICK, self.SNARE, self.HAT, self.OPENHAT))

    # ---- загрузка / обработка
    def raw(self, rel, dur=None, rel_fade=0.03, cutoff=None, harsh=None, hpf=None,
            ratio=1.0, atk=1.5):
        key = (rel, dur, rel_fade, cutoff, harsh, hpf, round(ratio, 4), atk)
        if key in self._c:
            return self._c[key]
        x = S.load_wav(os.path.join(ROOT, rel)).mean(1).astype(np.float32)
        if abs(ratio - 1.0) > 1e-3:
            idx = np.arange(0, len(x), ratio)
            x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
        if hpf:
            x = S.hp(x, hpf)
        if dur:
            x = x[:S.n_samples(dur)].copy()
            k = min(S.n_samples(rel_fade), len(x) - 1)
            if k > 0:
                x[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
        if cutoff:
            x = S.warm(x, cutoff, -6.0, 500.0)
        if harsh:
            x = S.deharsh(x, harsh)
        x = S.soft_attack(x, atk)
        x = np.asarray(x, np.float32)
        self._c[key] = x
        return x

    # ---- сигнатуры как у synth
    def kick(self, dur=0.22, f0=None, f1=None, vol=1.0, click=None):
        return self.raw(self.KICK, dur=min(dur * 1.5, 0.34), cutoff=6000, hpf=52) * (0.95 * vol)

    def snare(self, dur=0.16, vol=0.7, tone=None, bright=None):
        a = self.raw(self.SNARE, dur=min(dur * 1.3, 0.22), cutoff=7500, harsh=-4, hpf=110)
        b = self.raw(self.SNARE_B, dur=min(dur * 1.3, 0.22), cutoff=6000, harsh=-4, hpf=110)
        n = max(len(a), len(b))
        out = np.zeros(n, np.float32)
        out[:len(a)] += a * 0.75
        out[:len(b)] += b * 0.45
        return out * (0.85 * vol / 0.7)

    def hat(self, dur=0.045, vol=0.35, seed=None):
        return self.raw(self.HAT, dur=min(max(dur, 0.05), 0.1), cutoff=12000, harsh=-4,
                        hpf=380) * (0.85 * vol / 0.35)

    def openhat(self, dur=0.22, vol=0.3, seed=None):
        return self.raw(self.OPENHAT, dur=min(dur, 0.32), cutoff=9000, harsh=-5,
                        hpf=380) * (0.75 * vol / 0.3)

    def tom(self, pitch='G2', dur=0.22, vol=0.6):
        f = S.nt(pitch)
        i = 0 if f >= 100 else (1 if f >= 75 else 2)
        ref, path = self.TOMS[i]
        ratio = float(np.clip(f / S.nt(ref), 0.72, 1.42))
        return self.raw(path, dur=min(dur * 1.6, 0.45), cutoff=5000, hpf=45,
                        ratio=ratio) * (0.90 * vol / 0.6)

    def crash(self, dur=1.1, vol=0.5, seed=None):
        """В каталоге нет тарелки: берём длинный живой хэт как «шипение» и добавляем
        тёплое тело из тома — получается акцент без стеклянного верха."""
        wash = self.raw(self.OPENHAT, dur=min(dur, 1.1), rel_fade=min(dur, 1.1) * 0.55,
                        cutoff=7000, harsh=-6, hpf=300)
        body = self.tom('C2', dur=min(dur * 0.5, 0.4), vol=0.5)
        n = max(len(wash), len(body))
        out = np.zeros(n, np.float32)
        out[:len(wash)] += wash * 0.85
        out[:len(body)] += body * 0.5
        return out * (0.80 * vol / 0.5)

    def click(self, dur=0.02, f=2200, vol=0.5):
        n = S.n_samples(max(dur, 0.03))
        x = S.osc(f * 0.55, n, 'sine') * S.env_ar(n, 0.003, 3.5)
        return soften(x * vol * 0.8, cutoff=6000, harsh=-4, atk=2.5)


dr = _Drums()


def _kit():
    return {'k': dr.kick, 's': dr.snare, 'h': dr.hat, 'H': dr.openhat, 'c': dr.crash,
            't': dr.tom, 'T': lambda: dr.tom('C2', 0.26, 0.7), 'i': dr.click}


def hits(tr, t0, spec, step_dur, gain=1.0, pan=0.0, kit=None):
    """spec: {'k': 'x...x...x...x...', 's': '....x.......x...'} по сетке step_dur.

    'x' — громко, 'o' — тихо, '.'/'-'/' ' — пауза.
    """
    kit = dict(_kit(), **(kit or {}))
    for key, row in spec.items():
        gen = kit[key]
        for i, ch in enumerate(row):
            if ch in '.- ':
                continue
            g = 1.0 if ch == 'x' else (0.5 if ch == 'o' else 0.75)
            tr.add(gen(), t0 + i * step_dur, gain=gain * g, pan=pan)
    return tr


def groove(tr, t0, bars, bar_len, spec, gain=1.0, pan=0.0, kit=None, per_bar=None):
    """Повторить рисунок spec bars раз. per_bar(b) -> spec заменяет рисунок такта."""
    for b in range(bars):
        sp = spec if per_bar is None else (per_bar(b) or spec)
        hits(tr, t0 + b * bar_len, sp, bar_len / 16.0, gain=gain, pan=pan, kit=kit)
    return tr


# ---------------------------------------------------------------- уровни / запись

def rms(x):
    return float(np.sqrt(np.mean(np.square(np.asarray(x, np.float64))))) if len(x) else 0.0


def db(v):
    return -99.0 if v <= 1e-9 else float(20 * np.log10(v))


def tail_fade(tr, sec=0.05, head=0.004):
    n = S.n_samples(sec)
    h = S.n_samples(head)
    if h and h < len(tr.buf):
        tr.buf[:h] *= np.linspace(0, 1, h, dtype=np.float32)[:, None]
    if n and n < len(tr.buf):
        tr.buf[-n:] *= np.linspace(1, 0, n, dtype=np.float32)[:, None]
    return tr


def limit_peaks(x, ceil=0.80, hold=0.012, smooth=0.005):
    """Прозрачный пик-лимитер: гладкая огибающая усиления, а не искажение формы волны.

    hold > smooth гарантирует, что сглаженное усиление нигде не выше требуемого,
    то есть перелёта через ceil не будет.
    """
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    m = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    if float(m.max() if len(m) else 0.0) <= ceil:
        return np.asarray(x, np.float32)
    g = np.minimum(1.0, ceil / np.maximum(m, 1e-9)).astype(np.float32)
    h, s = max(1, int(hold * SR)), max(1, int(smooth * SR))
    g = minimum_filter1d(g, size=2 * h + 1, mode='nearest')
    g = uniform_filter1d(g, size=2 * s + 1, mode='nearest')
    return (x * (g[:, None] if x.ndim == 2 else g)).astype(np.float32)


def finish(tr, rms_db=-16.0, peak_max=0.80, iters=8, win=None):
    """Выровнять стем по RMS; транзиенты выше peak_max прижать лимитером.

    win=(t0, t1) — считать RMS только по этому окну. Нужно, когда к готовому
    стему дописан тихий хвост: иначе нормировка по всей длине подняла бы
    уже сведённую часть.
    """
    x = tr.buf
    tgt = 10 ** (rms_db / 20)
    raw = 0.0
    sl = slice(None) if win is None else slice(int(win[0] * SR), int(win[1] * SR))
    for i in range(iters):
        r = rms(x[sl])
        if r > 1e-9:
            x *= np.float32(tgt / r)
        p = float(np.max(np.abs(x)))
        if i == 0:
            raw = p
        if p <= peak_max:
            break
        x[:] = limit_peaks(x, peak_max)
    p = float(np.max(np.abs(x)))
    if p > peak_max:
        x *= np.float32(peak_max / p)
    tr.headroom = raw
    return tr


def bands(x, edges=((6000, 24000), (2500, 6000))):
    """Доли энергии по полосам (в процентах) — численный контроль резкости."""
    m = x.mean(1) if x.ndim == 2 else x
    X = np.abs(np.fft.rfft(m * np.hanning(len(m)))) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    tot = X.sum() + 1e-30
    return [100 * X[(f >= lo) & (f < hi)].sum() / tot for lo, hi in edges]


def report(name, tr):
    x = tr.buf
    p = float(np.max(np.abs(x)))
    r = rms(x)
    hi, mid = bands(x)
    raw = getattr(tr, 'headroom', p)
    extra = f' лимит {db(p) - db(raw):+4.1f}' if raw > p * 1.02 else '            '
    print(f'  {name:14s} {tr.dur:5.2f}s  пик {p:.3f}  RMS {db(r):6.1f} dB{extra}'
          f'   >6к {hi:4.1f}%  2.5-6к {mid:4.1f}%')
    return p, r
