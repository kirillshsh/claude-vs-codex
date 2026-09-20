#!/usr/bin/env python3
"""Analysis + rendering helpers for the ambience library."""
import numpy as np, subprocess, os, math

SR = 48000
NFFT = 4096
HOP = 2048


def decode(path, sr=SR, stereo=True):
    """Decode (possibly truncated/headerless) audio fragment via ffmpeg -> float32 array."""
    ac = 2 if stereo else 1
    cmd = ["ffmpeg", "-v", "error", "-err_detect", "ignore_err", "-i", path,
           "-ac", str(ac), "-ar", str(sr), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    raw = p.stdout
    if len(raw) < sr * 4 * ac:
        return None
    a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if ac == 2:
        a = a[: len(a) // 2 * 2].reshape(-1, 2)
    return a


def frames(mono, nfft=NFFT, hop=HOP):
    n = (len(mono) - nfft) // hop + 1
    if n < 4:
        return None
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    w = np.hanning(nfft).astype(np.float32)
    return mono[idx] * w


def spectral_stats(mono, sr=SR):
    """Per-frame rms, centroid, hi_ratio(>5k), lo_ratio(<500), mid band energy."""
    fr = frames(mono)
    if fr is None:
        return None
    S = np.abs(np.fft.rfft(fr, axis=1)).astype(np.float32)
    freqs = np.fft.rfftfreq(NFFT, 1.0 / sr)
    P = S * S
    band = freqs >= 40
    tot = P[:, band].sum(axis=1) + 1e-12
    hi = P[:, freqs > 5000].sum(axis=1) / tot
    lo = P[:, (freqs >= 40) & (freqs < 500)].sum(axis=1) / tot
    cen = (P[:, band] * freqs[band]).sum(axis=1) / tot
    rms = np.sqrt((fr ** 2).mean(axis=1)) + 1e-9
    # speech/voice proxy: 200-4000 Hz band envelope modulation at 2-10 Hz
    mid = np.sqrt(P[:, (freqs > 200) & (freqs < 4000)].sum(axis=1)) + 1e-9
    return dict(rms=rms, hi=hi, lo=lo, cen=cen, mid=mid, freqs=freqs, P=P)


def speech_mod(mid_env, hop=HOP, sr=SR):
    """Fraction of envelope energy in the 2-10 Hz (syllabic) band -> voice indicator."""
    if len(mid_env) < 32:
        return 0.0
    e = np.log(mid_env + 1e-9)
    e = e - e.mean()
    fr = sr / hop                       # envelope sample rate (~23.4 Hz)
    sp = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
    f = np.fft.rfftfreq(len(e), 1.0 / fr)
    tot = sp[f > 0.05].sum() + 1e-12
    return float(sp[(f >= 2.0) & (f <= 10.0)].sum() / tot)


def db(x):
    return 20 * math.log10(max(float(x), 1e-9))


def butter_lp(x, fc, sr=SR, order=4):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, min(fc / (sr / 2), 0.99), btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=0).astype(np.float32)


def butter_hp(x, fc, sr=SR, order=2):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, max(fc / (sr / 2), 1e-4), btype="high", output="sos")
    return sosfiltfilt(sos, x, axis=0).astype(np.float32)


def normalize_peak(x, target_db=-3.0):
    pk = float(np.abs(x).max()) + 1e-12
    g = (10 ** (target_db / 20.0)) / pk
    return (x * g).astype(np.float32)


def fade(x, sr=SR, ms=60):
    n = min(int(sr * ms / 1000), len(x) // 4)
    if n < 8:
        return x
    r = np.linspace(0, 1, n, dtype=np.float32) ** 0.5
    y = x.copy()
    if y.ndim == 2:
        y[:n] *= r[:, None]
        y[-n:] *= r[::-1][:, None]
    else:
        y[:n] *= r
        y[-n:] *= r[::-1]
    return y


def make_loop(x, sr=SR, xf=1.5):
    """Crossfade tail into head -> seamless loop. Returns (loop, joint_report)."""
    C = int(sr * xf)
    N = len(x)
    if N <= C * 2 + sr:
        return None, None
    out = x[: N - C].copy()
    a = x[N - C:]                 # tail
    b = x[:C]                     # head
    t = np.linspace(0, 1, C, dtype=np.float32)
    fo = np.cos(t * np.pi / 2)    # equal power
    fi = np.sin(t * np.pi / 2)
    if x.ndim == 2:
        out[:C] = a * fo[:, None] + b * fi[:, None]
    else:
        out[:C] = a * fo + b * fi
    # joint check: RMS of the wrap region vs interior
    m = out.mean(axis=1) if out.ndim == 2 else out
    w = int(sr * 0.25)
    wrap = np.concatenate([m[-w:], m[:w]])
    rms_wrap = float(np.sqrt((wrap ** 2).mean()))
    rms_body = float(np.sqrt((m[w: len(m) - w] ** 2).mean()))
    jump_db = abs(db(rms_wrap) - db(rms_body))
    # also the instantaneous step across the seam
    step = abs(float(m[0]) - float(m[-1]))
    return out, dict(joint_rms_db=round(jump_db, 2), seam_step=round(step, 5),
                     rms_wrap_db=round(db(rms_wrap), 2), rms_body_db=round(db(rms_body), 2))


def write_wav(path, x, sr=SR):
    from scipy.io import wavfile
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    y = np.clip(x, -1.0, 1.0)
    wavfile.write(path, sr, (y * 32767.0).astype(np.int16))


def measure(x, sr=SR):
    """Final measurement of a rendered buffer."""
    m = x.mean(axis=1) if x.ndim == 2 else x
    st = spectral_stats(m, sr)
    if st is None:
        return {}
    return dict(
        dur=round(len(m) / sr, 2),
        peak_db=round(db(np.abs(x).max()), 2),
        rms_db=round(db(np.sqrt((m ** 2).mean())), 2),
        centroid_hz=int(np.median(st["cen"])),
        hi_ratio=round(float(np.median(st["hi"])), 3),
        lo_ratio=round(float(np.median(st["lo"])), 3),
        speech_mod=round(speech_mod(st["mid"]), 3),
    )
