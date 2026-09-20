"""Звуковые эффекты для 0–25 с (агент A): s0_title, s1_meadow, s2_argue.

    python3 audio/make_sfx_a.py            # пересоздать стемы + audio/cues/sfx_a.json
    python3 audio/make_sfx_a.py --table    # ещё и таблица «секунда — событие — звук»

Версия 2: на реальных сэмплах из audio/samples/. Синтез остался только у голосов
персонажей (cl_voice / cl_growl / cd_voice) и у тихих низких подкладов `sub`.
Плотность срезана с 170 событий до ~95: один `lead` + один `support` одновременно,
между двумя `lead` не меньше 0.35 с (проверяется при сборке), украшения — только
там, где рядом ничего не звучит.

Тайминги те же, что в версии 1 (взяты из кода сцен и сверены со стиллами):
  s0_title  (0.00–4.00): T_CLAWD_HIT .70, T_CODEX_HIT 1.04, T_VS_HIT 1.38, T_SUB 1.90,
                         посадки 2.36/2.54/3.12/3.26, молнии 2.00/2.97, ирис 3.65
  s1_meadow (4.00–13.00, lt=t-4): бег CL 2.50–5.10 / CD 2.72–5.26, «!» 5.55/5.63,
                         радость 5.86, шот D 6.35, шот E 7.00, крупняки 8.20/8.60
  s2_argue  (13.00–25.00, lt=t-13): «Я ЛУЧШЕ!» 1.02, «НЕТ, Я ЛУЧШЕ!» 2.92, топот,
                         облёт-реплики 5.10/5.78/6.46/7.14, сплит 8.50, разряд 8.78,
                         зум-панчи 9.40/10.00, прыжок 10.60, столкновение 10.84
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import lib_sfx_a as L                                            # noqa: E402
from synth import Track, lp, save_wav, write_manifest            # noqa: E402

STEM_DIR = os.path.join(ROOT, 'audio', 'stems', 'sfx_a')
CUE_FILE = os.path.join(ROOT, 'audio', 'cues', 'sfx_a.json')

S0, S1, S2 = 0.0, 4.0, 13.0          # абсолютное начало сцен
EVENTS = []                           # (абс. сек, роль, что на экране, звук, пик, pan)

# ---------------------------------------------------------------- палитра сэмплов
# Отбирались по каталогу: низкий centroid_hz и hi_ratio ~0 (мягкие, не режут уши).
SMP = {
    'slam_warm':   ('impact', 'oga_slam_02.wav'),                      # 0.53 c cen 489
    'slam_metal':  ('impact', 'oga_metal_slam_01.wav'),                # 0.41 c cen 754
    'slam_wood':   ('impact', 'oga_wood_slam_04.wav'),                 # 0.34 c cen 693
    'boom':        ('impact', 'kenney_explosioncrunch_000.wav'),       # 0.78 c cen 208
    'boom_low':    ('impact', 'kenney_lowfrequency_explosion_001.wav'),
    'thump_s':     ('impact', 'kenney_impactsoft_medium_000.wav'),     # 0.12 c cen  97
    'thump_h':     ('impact', 'kenney_impactsoft_heavy_003.wav'),      # 0.54 c cen  69
    'punch0':      ('impact', 'kenney_impactpunch_heavy_000.wav'),
    'punch1':      ('impact', 'kenney_impactpunch_heavy_001.wav'),
    'punch2':      ('impact', 'kenney_impactpunch_heavy_002.wav'),
    'punch3':      ('impact', 'kenney_impactpunch_heavy_003.wav'),
    'punch_m':     ('impact', 'kenney_impactpunch_medium_000.wav'),    # 0.29 c cen 163
    'wood_hit':    ('impact', 'kenney_impactwood_heavy_001.wav'),      # 0.31 c cen 121
    'gong':        ('bell', 'oga_gong.wav'),                           # 0.58 c cen 160
    'bell_s':      ('bell', 'kenney_impactbell_heavy_004.wav'),        # 0.30 c cen 272
    'bell_star':   ('bell', 'kenney_impactbell_heavy_003.wav'),        # 0.65 c cen 458
    'swish_s':     ('whoosh', 'oga_swish_11.wav'),                     # 0.10 c
    'swish_m':     ('whoosh', 'oga_swish_7.wav'),                      # 0.16 c
    'swosh_lo':    ('whoosh', 'oga_swosh_02.wav'),                     # 0.20 c cen  48
    'swosh_mid':   ('whoosh', 'oga_swosh_05.wav'),                     # 0.42 c cen 251
    'swosh_long':  ('whoosh', 'oga_swosh_03.wav'),                     # 0.58 c cen 108
    'megaswosh':   ('whoosh', 'oga_qubodup_megaswosh2.wav'),           # 0.59 c cen 393
    'slomo':       ('whoosh', 'oga_qubodup_slomo1.wav'),               # 0.96 c cen 434
    'elec_s':      ('mech', 'oga_ui_electric_02.wav'),                 # 0.34 c cen 439
    'elec_lo':     ('mech', 'oga_ui_electric_01.wav'),                 # 0.81 c cen  86
    'elec_whoosh': ('mech', 'oga_whoosh_electric_02.wav'),             # 0.59 c cen 1493
    'elec_bed':    ('mech', 'oga_ambience_sinister_electric_cello_loop_02.wav'),
    'grass':       ('mech', 'oga_grass.wav'),                          # 2.97 c cen 375
    'gear':        ('mech', 'oga_gear01.wav'),                         # 0.58 c cen 223
    'click':       ('ui', 'oga_click.wav'),                            # 0.08 c cen 1346
    'select':      ('ui', 'kenney_select_002.wav'),                    # 0.04 c
    'switch':      ('ui', 'kenney_switch_005.wav'),                    # 0.16 c cen 105
    'err_lo':      ('retro8bit', 'oga_sfx_sounds_error10.wav'),        # 0.30 c cen 353
    'err_hi':      ('retro8bit', 'oga_sfx_sounds_error1.wav'),         # 0.33 c cen 1390
    'jump_lo':     ('retro8bit', 'kenney_phasejump1.wav'),             # 0.40 c cen 354
    'jump_hi':     ('retro8bit', 'kenney_phasejump5.wav'),             # 0.41 c cen 1256
    'jingle':      ('retro8bit', 'kenney_jingles_pizzi00.wav'),        # 0.49 c cen 444
    'coin':        ('retro8bit', 'oga_sfx_coin_single2.wav'),          # 0.17 c cen 827
    'step_dirt':   ('footstep', 'oga_footstep_dirt_04.wav'),           # 0.24 c
    'step_grass':  ('footstep', 'kenney_footstep_grass_003.wav'),      # 0.67 c cen 518
    'kick':        ('percussion', 'drum_kit8_kick.wav'),               # 0.33 c cen  77
    'snare':       ('percussion', 'drum_breakbeat8_snare.wav'),        # 0.35 c cen 183
    'hihat':       ('percussion', 'drum_4op_fm_hihat.wav'),            # 0.46 c cen 607
    'tom':         ('percussion', 'drum_kit3_tom1.wav'),               # 0.96 c cen 183
    'laser':       ('retro8bit', 'kenney_laser3.wav'),                 # 0.65 c cen 159
    'powerup':     ('retro8bit', 'oga_sfx_sounds_powerup13.wav'),      # 0.42 c cen 1101
    'confirm':     ('ui', 'kenney_confirmation_001.wav'),              # 0.29 c cen 524
    'hover':       ('mech', 'kenney_spaceenginesmall_000.wav'),        # 5.00 c cen 135
    'gear_rattle': ('mech', 'oga_gear03.wav'),                         # 2.00 c cen 234
}


def S(key):
    return L.SM(*SMP[key])


def amb_sample(*queries):
    """Фон из ambience-библиотеки.

    Сначала спрашиваем каталог (catalog_ambience.json пишет агент амбиентов), а пока
    его нет — берём файл прямо из audio/samples/ambience/<тег>/ (файлы уже на месте).
    Возвращает None, если ничего не нашлось: тогда зовущий берёт mech/oga_grass."""
    if L.find is not None:
        for cat, tags in queries:
            hits = L.find(cat, tags, max_hi=0.30, min_dur=3.0)
            if hits:
                return hits[0]
    amb = os.path.join(ROOT, 'audio', 'samples', 'ambience')
    for _, tags in queries:
        for tag in tags:
            d = os.path.join(amb, tag)
            if os.path.isdir(d):
                # «_loop» — бесшовные версии, они лучше для фонов
                files = sorted(f for f in os.listdir(d) if f.endswith('_loop.wav'))
                files = files or sorted(f for f in os.listdir(d) if f.endswith('.wav'))
                if files:
                    return os.path.join(d, files[0])
    return None


# ---------------------------------------------------------------- укладка событий

# §4 PLAN.md: у каждого участка свой потолок, support = lead −7…−9 дБ (≈0.35–0.45×)
PEAK_TABLE = [
    (0.00, 4.00, {'lead': (0.38, 0.58), 'support': (0.14, 0.24), 'decor': (0.06, 0.14)}),
    (4.00, 13.00, {'lead': (0.38, 0.46), 'support': (0.11, 0.21), 'decor': (0.05, 0.12)}),
    (13.00, 21.50, {'lead': (0.42, 0.52), 'support': (0.11, 0.19), 'decor': (0.05, 0.12)}),
    (21.50, 25.10, {'lead': (0.44, 0.58), 'support': (0.16, 0.23), 'decor': (0.05, 0.15)}),
]
PAN_LIMIT = {'lead': 0.25, 'support': 0.55, 'decor': 0.55, 'bed': 0.50}
_WARN = []


def peak_range(t, role):
    for a, b, tab in PEAK_TABLE:
        if a <= t < b:
            return tab.get(role, (0.0, 1.0))
    return (0.0, 1.0)


def bed_hp(x, cut=120.0):
    """П5/§5: фоновые слои режем ФВЧ 120 Гц, чтобы не мутить низ музыки."""
    from synth import hp
    if x.ndim == 2:
        return np.stack([hp(x[:, 0], cut), hp(x[:, 1], cut)], 1).astype(np.float32)
    return hp(x, cut)


def put(tr, base_t, sig, t, peak, pan=0.0, role='support', what='', snd=''):
    """Положить сигнал так, чтобы пик в громком канале был ровно peak.

    Track.add панорамирует по sqrt-закону (|pan|=0.8 даёт +2.6 dB в канале),
    поэтому перед укладкой нормируем на peak / sqrt(1+|pan|)."""
    lo, hi = peak_range(base_t + t, role)
    if not (lo - 1e-6 <= peak <= hi + 1e-6):
        _WARN.append(f'{base_t + t:6.2f} {role}: пик {peak:.2f} вне {lo}–{hi} — {what}')
    if abs(pan) > PAN_LIMIT[role] + 1e-6:
        _WARN.append(f'{base_t + t:6.2f} {role}: pan {pan:+.2f} > {PAN_LIMIT[role]} (П8) — {what}')
    sig = np.asarray(sig, np.float32)
    g = peak if sig.ndim == 2 else peak / math.sqrt(1.0 + abs(pan))
    m = float(np.max(np.abs(sig))) or 1.0
    tr.add(sig * (g / m), t, pan=pan)
    if what:
        EVENTS.append((round(base_t + t, 3), role, what, snd, peak, pan))
    return tr


def check_leads(min_gap=0.35):
    ts = sorted(e[0] for e in EVENTS if e[1] == 'lead')
    return [(a, b) for a, b in zip(ts, ts[1:]) if b - a < min_gap - 1e-6]


def seg(t, a, b):
    return min(1.0, max(0.0, (t - a) / (b - a)))


def lerp(a, b, u):
    return a + (b - a) * u


def travel(u, stop_frac=0.22):
    """как в scenes/_a_helpers.py: равномерно, в конце торможение."""
    u = min(1.0, max(0.0, u))
    a = 1 - stop_frac
    tot = a + stop_frac / 2
    if u < a:
        return u / tot
    v = u - a
    return (a + v - v * v / (2 * stop_frac)) / tot


def ease_in_out(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def ease_out(u):
    u = min(1.0, max(0.0, u))
    return 1 - (1 - u) ** 3


# ------------------------------------------------------------------ панорама по кадру
# Мини-копия камеры из cam3d.py (look_at/orbit/project): нужна ровно для того, чтобы
# поставить pan туда, где объект реально виден. Сверено с proj() из scenes/_a_helpers.py.
W_PX = 480


class _Cam:
    def __init__(self, x, y, z, yaw, pitch, f):
        self.x, self.y, self.z, self.yaw, self.pitch, self.f = x, y, z, yaw, pitch, f

    def sx(self, X, Y, Z):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        dx, dy, dz = X - self.x, Y - self.y, Z - self.z
        xr = dx * cy - dz * sy
        zr = dx * sy + dz * cy
        zc = -dy * sp + zr * cp
        return None if zc <= 0.5 else W_PX / 2 + self.f * xr / zc


def _look_at(eye, target, f):
    dx, dy, dz = target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]
    return _Cam(eye[0], eye[1], eye[2], math.atan2(dx, dz),
                -math.atan2(dy, math.hypot(dx, dz)), f)


def _orbit(radius, angle, height, f, look_height):
    return _look_at((-math.sin(angle) * radius, height, -math.cos(angle) * radius),
                    (0.0, look_height, 0.0), f)


def pan_screen(cam, X, Y, Z=0.0, lim=0.85, width=1.7):
    x = None if cam is None else cam.sx(X, Y, Z)
    if x is None:
        return math.copysign(lim, X) if X else 0.0
    return max(-lim, min(lim, (x / W_PX - 0.5) * width))


def cam_s1(lt):
    """камера s1_meadow на локальной секунде lt (шоты A/B/D/E)."""
    if lt < 3.20:
        v = seg(lt, 0.0, 3.2)
        return _orbit(lerp(78.0, 56.0, 0.55 * v + 0.45 * ease_out(v)),
                      lerp(-3.72, -0.76, 0.82 * v + 0.18 * ease_in_out(v)),
                      lerp(34.0, 20.0, ease_in_out(1 - (1 - v) ** 1.5)),
                      lerp(225.0, 300.0, ease_in_out(v)), lerp(6.0, 7.5, ease_in_out(v)))
    if lt < 6.35:
        u = seg(lt - 3.20, 0.0, 3.15)
        return _look_at((lerp(-6.0, 2.0, ease_in_out(u)), lerp(3.4, 4.2, u),
                         lerp(-66.0, -56.0, ease_in_out(u))),
                        (lerp(-4.0, 0.0, ease_in_out(u)), 9.0, 0.0), 250.0)
    if lt < 7.00:
        u = ease_in_out(seg(lt - 6.35, 0.0, 0.58))
        return _look_at((0.0, lerp(6.0, 10.5, u), lerp(-40.0, -15.5, u)),
                        (0.0, lerp(11.0, 12.6, u), 0.0), lerp(270.0, 320.0, u))
    u = seg(lt - 7.00, 0.0, 1.2)
    return _look_at((0.0, lerp(2.3, 2.0, u), lerp(-45.0, -37.0, ease_in_out(u))),
                    (0.0, 10.5, 0.0), lerp(228.0, 240.0, u))


def s2_orbit(lt):
    """(камера, мир Clawd, мир Codex) для облёта s2_argue (локально 5.0–8.5)."""
    u = seg(lt - 5.0, 0.0, 3.5)
    ang = math.radians(lerp(-92.0, 90.0, ease_in_out(u)))
    cam = _orbit(lerp(47.0, 37.0, ease_in_out(u)), ang, lerp(3.4, 2.6, u),
                 lerp(240.0, 252.0, u), 11.0)
    a = ang * 0.92
    ax, az = math.cos(a), -math.sin(a)
    return cam, (-22.0 * ax, -22.0 * az), (21.0 * ax, 21.0 * az)


# ================================================================== s0 — заставка (19 событий)
# Все решения — по таблице §1 audio/PLAN.md. Громкости — по §4 (lead 0.50, support 0.20–0.22).

def build_s0():
    tr = Track(4.0)
    P = lambda *a, **k: put(tr, S0, *a, **k)  # noqa: E731

    P(L.take(S('slomo'), dur=0.40, reverse=True, warm_cut=6500, hp_cut=90), 0.10, 0.20, 0.0, 'support',
      'камера наезжает, титров ещё нет', 'реверс-свуш oga_qubodup_slomo1')
    P(L.take(S('swosh_mid'), dur=0.22, warm_cut=7000), 0.48, 0.22, -0.55, 'support',
      'плашка CLAUDE летит слева (T_CLAWD .50)', 'oga_swosh_05')
    # П9: удар = один cue из склеенных слоёв (impact heavy + kick + низ)
    P(L.lay(L.take(S('slam_warm'), dur=0.42, warm_cut=7500, deh=-2, hp_cut=50),
            L.take(S('kick'), dur=0.28, warm_cut=4000) * 0.40,
            L.sub(0.24, 95, 46, 0.16)), 0.70, 0.50, -0.25, 'lead',
      'CLAUDE врезается в кадр, искры, тряска (T_CLAWD_HIT)', 'oga_slam_02 + kit8_kick + низ')
    P(L.take(S('swosh_mid'), dur=0.22, speed=1.15, warm_cut=7000), 0.82, 0.22, 0.55, 'support',
      'плашка CODEX летит справа (T_CODEX .84)', 'oga_swosh_05, выше')
    P(L.lay(L.take(S('slam_metal'), dur=0.40, warm_cut=7500, deh=-3, hp_cut=55),
            L.take(S('kick'), dur=0.26, speed=1.1, warm_cut=4000) * 0.35,
            L.sub(0.20, 110, 52, 0.14)), 1.04, 0.50, 0.25, 'lead',
      'CODEX врезается в кадр (T_CODEX_HIT)', 'oga_metal_slam_01 + kick + низ')
    P(L.take(S('laser'), dur=0.17, reverse=True, warm_cut=7000), 1.21, 0.24, 0.0, 'support',
      'VS несётся на камеру (T_VS 1.26)', 'реверс kenney_laser3 (риз)')
    # объединение 1.38 + 1.42: удар VS и молния — один cue
    P(L.lay(L.take(S('boom'), dur=0.78, warm_cut=7000, deh=-3, hp_cut=45),
            L.take(S('gong'), dur=0.58, warm_cut=6000, hp_cut=120) * 0.40,
            L.at(L.take(S('elec_s'), dur=0.26, warm_cut=7000) * 0.42, 0.04),
            L.sub(0.30, 80, 36, 0.16)), 1.38, 0.56, 0.0, 'lead',
      'VS бьёт в экран: вспышка, лучи, молния (T_VS_HIT, объед. с 1.42)',
      'explosioncrunch_000 + gong + ui_electric_02 + низ')
    P(L.take(S('step_dirt'), dur=0.16, speed=0.9, warm_cut=6000, deh=-3), 2.36, 0.18, -0.30,
      'support', 'Clawd приземляется после злого подскока, пыль', 'footstep dirt_04')
    P(L.take(S('bell_s'), dur=0.30, speed=1.25, warm_cut=9000), 2.82, 0.18, 0.0, 'support',
      'блик пробегает по VS (shine 2.55–2.95)', 'impactbell_heavy_004, выше')
    P(L.take(S('elec_s'), dur=0.24, speed=1.2, warm_cut=7000, deh=-3), 2.97, 0.20, 0.15,
      'support', 'молния между героями (единственная в паузе)', 'oga_ui_electric_02')
    P(L.take(S('step_dirt'), dur=0.15, speed=1.15, warm_cut=6000, deh=-3), 3.26, 0.17, 0.30,
      'support', 'Codex приземляется, пыль', 'footstep dirt_04, выше')
    # П7: 3.60 — каданс музыки m0, поэтому здесь ровно один разрешённый lead
    P(L.take(S('swosh_long'), dur=0.36, warm_cut=6500, hp_cut=90), 3.62, 0.40, 0.0, 'lead',
      'ирис-затвор закрывает кадр (3.65–4.00)', 'oga_swosh_03')
    return tr


def build_s0_type():
    """Подзаголовок печатается 1.90–2.44. По плану: 11 кликов -> 4, последний склеен с «динь»."""
    tr = Track(0.95)
    txt = 'КТО ЖЕ ЛУЧШЕ?'
    vis = [k for k, ch in enumerate(txt) if ch != ' ']
    for j, k in enumerate((vis[0], vis[3], vis[5], vis[-1])):
        t = k * 0.045
        sig = L.take(S('click'), dur=0.07, speed=(0.95, 1.05, 1.0, 0.9)[j], warm_cut=6500, deh=-3)
        if j == 3:      # последний клик и «допечатано» — один cue (П9)
            sig = L.lay(sig, L.at(L.take(S('bell_s'), dur=0.26, speed=1.4, warm_cut=9000) * 0.55, 0.06))
        put(tr, 1.90, sig, t, 0.20 if j == 3 else 0.16, (-0.10, 0.10, -0.08, 0.0)[j], 'support',
            f'печатается символ «{txt[k]}»' + (' + подзаголовок допечатан' if j == 3 else ''),
            'ui click' + (' + короткий «динь»' if j == 3 else ''))
    return tr


def build_s0_amb():
    tr = Track(4.0)
    src = amb_sample(('ambience', ['wind']), ('ambience', ['nature_misc'])) or S('grass')
    x = L.stereo_take(src, loop_to=3.95, warm_cut=6000, fin=0.5, fout=0.6)
    tr.add(L.rms_to(bed_hp(x), -32.0), 0.0)
    EVENTS.append((0.0, 'bed', 'закатный луг за титрами', f'{os.path.basename(src)} (RMS -32)', 0.0, 0.0))
    return tr


# ================================================================== s1 — поляна (30 событий)

CL_X0, CL_X, CL_T0, CL_T1 = -112.0, -22.0, 2.50, 5.10
CD_X0, CD_X, CD_T0, CD_T1 = 118.0, 21.0, 2.72, 5.26


def build_s1():
    tr = Track(9.0)
    P = lambda *a, **k: put(tr, S1, *a, **k)  # noqa: E731

    P(L.take(S('slomo'), dur=0.42, reverse=True, warm_cut=6800), 0.0, 0.40, 0.0, 'lead',
      'ирис раскрывается, показывая поляну (0.00–0.40)', 'реверс-свуш slomo1')
    P(L.take(S('bell_s'), dur=0.30, warm_cut=9000), 0.55, 0.20, 0.0, 'support',
      'блик на кубке в центре кадра (единственный из четырёх)', 'impactbell_heavy_004')
    # --- бег краба: 14 шагов через 0.17 с -> 7 через 0.34 с (П5), сторона не меняется (П8)
    for k, te in enumerate(np.arange(CL_T0 + 0.05, CL_T1 - 0.25, 0.34)):
        X = lerp(CL_X0, CL_X, travel(seg(te, CL_T0, CL_T1), 0.22))
        pan = max(-0.40, min(-0.28, pan_screen(cam_s1(te), X, 6.0)))
        P(L.take(S('step_grass'), dur=0.15, speed=1.15 + 0.05 * (k % 3), warm_cut=6000, deh=-3),
          float(te), 0.16 + 0.007 * k, pan, 'support',
          f'шаг краба #{k + 1} (X={X:+.0f})', 'footstep grass_003')
    # --- Codex парит: 14 «пуфов» -> непрерывный гул (в s1_hover) + 2 акцента
    # один акцент вместо двух: с 8.40 окно 8.25–9.25 давало 5 атак (П4)
    P(L.take(S('switch'), dur=0.16, speed=0.85, warm_cut=5000, deh=-4), 3.10, 0.16, 0.45,
      'support', 'Codex подлетает ближе (акцент ховера)', 'ui switch_005, глухой')
    # --- торможение у кубка
    P(L.take(S('grass'), dur=0.34, t0=0.6, warm_cut=6000, deh=-3), CL_T1 - 0.12, 0.40, -0.25,
      'lead', 'Clawd тормозит у кубка, большое облако пыли', 'шорох травы oga_grass')
    # --- «!» и радость
    P(L.lay(L.take(S('jump_lo'), dur=0.36, warm_cut=7500),
            L.take(S('confirm'), dur=0.22, warm_cut=6000) * 0.40), 5.55, 0.42, -0.25, 'lead',
      '«!» над Clawd, он подпрыгивает (T_EXCL_CL)', 'phasejump1 + ui confirmation_001')
    # 9.63 сдвинут на 9.78 по П3 (был 0.08 с от лида)
    P(L.take(S('select'), dur=0.12, speed=0.9, warm_cut=6500, deh=-3), 5.78, 0.16, 0.28,
      'support', '«!» над Codex, он подпрыгивает (T_EXCL_CD, сдвинут с 5.63 по П3)',
      'ui select_002')
    P(L.take(S('powerup'), dur=0.50, warm_cut=8000, deh=-2), 5.94, 0.40, 0.0, 'lead',
      'оба радуются, летят искорки (T_HAPPY)', 'oga_sfx_sounds_powerup13')
    # --- шот D: наезд на кубок и звезда-блик
    P(L.take(S('swish_m'), dur=0.16, reverse=True, warm_cut=7000), 6.35, 0.20, 0.0, 'support',
      'склейка: резкий наезд на кубок (шот D)', 'реверс oga_swish_7')
    P(L.take(S('bell_star'), dur=0.58, warm_cut=9000), 6.65, 0.44, 0.0, 'lead',
      'большая звезда-блик на ободе кубка', 'impactbell_heavy_003')
    P(L.take(S('swosh_lo'), dur=0.20, warm_cut=6500), 7.00, 0.18, 0.0, 'support',
      'склейка: камера падает к траве (шот E)', 'oga_swosh_02')
    # --- дуэль взглядов
    P(L.lay(L.cl_growl(0.24, 126.0, seed=7),
            L.take(S('thump_h'), dur=0.30, warm_cut=5000) * 0.35), 7.42, 0.40, -0.25, 'lead',
      'Clawd разворачивается к сопернику, злой (T_TURN_CL)', 'голос Clawd + impactsoft_heavy')
    P(L.cd_voice([(0.10, (0.0, -5.0, -7.0), 0.0)], base=440.0, vol=1.0, seed=21), 7.62, 0.16,
      0.30, 'support', 'Codex опускает взгляд, «сверлит» (T_TURN_CD)',
      'голос Codex, 1 слог вниз')
    # 12.02 понижен до support по П2: до лида 12.33 остаётся 0.31 с
    P(L.take(S('elec_s'), dur=0.14, speed=1.35, warm_cut=7000, deh=-3), 8.02, 0.20, 0.0,
      'support', 'искра-молния между взглядами (8.02–8.16)', 'oga_ui_electric_02, короткий')
    P(L.lay(L.take(S('slam_warm'), dur=0.36, speed=0.85, warm_cut=6500, deh=-3, hp_cut=45),
            L.take(S('tom'), dur=0.30, warm_cut=4000) * 0.45,
            L.sub(0.26, 70, 34, 0.22)), 8.33, 0.46, -0.12, 'lead',
      'глаза Clawd сужаются, знак гнева, тряска', 'oga_slam_02 замедленный + tom + низ')
    P(L.lay(L.take(S('slam_metal'), dur=0.32, speed=0.9, warm_cut=6800, deh=-3),
            L.at(L.take(S('elec_s'), dur=0.20, warm_cut=7000) * 0.40, 0.03),
            L.sub(0.24, 85, 42, 0.20)), 8.72, 0.44, 0.12, 'lead',
      'экран Codex переключается на «><», тряска', 'metal_slam + ui_electric + низ')
    return tr


def build_s1_amb():
    tr = Track(9.0)
    src = amb_sample(('ambience', ['wind']), ('ambience', ['nature_misc'])) or S('grass')
    x = L.stereo_take(src, loop_to=8.8, warm_cut=6000, fin=0.6, fout=0.9)
    tr.add(L.rms_to(bed_hp(x), -30.0), 0.0)
    EVENTS.append((S1, 'bed', 'солнечная поляна, трава', f'{os.path.basename(src)} (RMS -30)', 0.0, 0.0))
    birds = amb_sample(('ambience', ['birds']),)
    if birds:
        for k, tb in enumerate((0.25, 1.80)):           # 2 из 4 (план)
            q = tb / 3.2
            pan = max(-0.50, min(0.50, pan_screen(cam_s1(tb), lerp(-150.0, 150.0, q), 46.0,
                                                  lerp(62.0, -58.0, q))))
            put(tr, S1, L.take(birds, dur=1.2, t0=1.6 + 3.0 * k, warm_cut=7000), tb, 0.10, pan,
                'decor', 'стайка птиц пролетает над поляной', os.path.basename(birds))
    return tr


def build_s1_hover():
    """Codex не шагает, а парит: 14 «пуфов» заменены одним непрерывным гулом (П5)."""
    tr = Track(2.5)
    x = L.stereo_take(S('hover'), t0=0.8, loop_to=2.4, warm_cut=4000, fin=0.35, fout=0.5)
    tr.add(L.rms_to(bed_hp(x), -29.0), 0.0, pan=0.45)
    EVENTS.append((S1 + 2.72, 'bed', 'Codex подлетает справа (вместо 14 «пуфов»)',
                   'kenney_spaceenginesmall_000 (RMS -29)', 0.0, 0.45))
    return tr


# ================================================================== s2 — ссора (30 событий)

def build_s2():
    tr = Track(12.0)
    P = lambda *a, **k: put(tr, S2, *a, **k)  # noqa: E731

    def pcl(lt, Yh=16.0, lim=0.25):
        cam, cl, _ = s2_orbit(lt)
        return max(-lim, min(lim, pan_screen(cam, cl[0], Yh, cl[1])))

    def pcd(lt, Yh=20.0, lim=0.25):
        cam, _, cd = s2_orbit(lt)
        return max(-lim, min(lim, pan_screen(cam, cd[0], Yh, cd[1])))

    # --- кадр 1: через плечо Codex
    P(L.take(S('swish_m'), dur=0.14, warm_cut=6800), 0.0, 0.18, 0.0, 'support',
      'жёсткая склейка из крупняка в «через плечо»', 'oga_swish_7')
    P(L.cl_voice([(0.13, (0.0, 4.0, 1.0), 0.0),
                  (0.11, (2.0, 5.0, -1.0), 2.0),
                  (0.14, (0.0, 2.0, -5.0), 1.0)], base=175.0, vol=1.0, seed=1),
      1.02, 0.48, -0.22, 'lead', 'пузырь «Я ЛУЧШЕ!» у Clawd (lt 1.02)',
      'синтез: голос Clawd, 3 слога')
    P(L.cl_voice([(0.13, (0.0, -3.0, -5.0), -2.0)], base=165.0, vol=1.0, seed=9), 1.62, 0.18,
      -0.25, 'support', 'Clawd сердито фыркает (lt 1.62)', 'голос Clawd, 1 слог')
    P(L.take(S('elec_lo'), dur=0.28, speed=1.4, reverse=True, warm_cut=5200), 1.92, 0.16, 0.45,
      'support', 'Codex вскипает и поднимает кулак (lt 1.92–2.12)', 'реверс oga_ui_electric_01')
    # --- кадр 2: отвечает Codex
    P(L.take(S('swish_m'), dur=0.14, speed=1.12, warm_cut=6800), 2.50, 0.18, 0.0, 'support',
      'склейка на обратную восьмёрку', 'oga_swish_7')
    P(L.cd_voice([(0.10, (0.0, 6.0, 3.0), 0.0),
                  (0.09, (3.0, 8.0, 5.0), 2.0),
                  (0.11, (2.0, 4.0, -4.0), 3.0)], base=430.0, vol=1.0, seed=2),
      2.92, 0.48, 0.22, 'lead', 'пузырь «НЕТ, Я ЛУЧШЕ!» у Codex (lt 2.92, 3 слога вместо 4)',
      'синтез: голос Codex, 3 слога')
    P(L.take(S('punch_m'), dur=0.24, speed=0.85, warm_cut=6500, deh=-2), 3.56, 0.18, 0.25,
      'support', 'Codex топает #3 (2 из 5)', 'impactpunch_medium')
    P(L.take(S('grass'), dur=0.42, t0=2.0, speed=0.75, warm_cut=4000, deh=-5), 4.30, 0.12,
      -0.50, 'support', 'пар над затылком Clawd (один в чистой паузе)', 'oga_grass, глухо')
    P(L.take(S('punch_m'), dur=0.26, speed=0.92, warm_cut=6500, deh=-2), 4.46, 0.18, 0.25,
      'support', 'Codex топает #5, последний перед склейкой', 'impactpunch_medium')
    # --- кадр 3: облёт, реплики по нарастающей
    P(L.take(S('swosh_mid'), dur=0.22, warm_cut=6800), 4.94, 0.18, 0.0, 'support',
      'склейка на низкий облёт (свуш пред-ударом на кадр)', 'oga_swosh_05')
    P(L.cl_voice([(0.14, (0.0, 5.0, -3.0), 0.0)], base=175.0, vol=1.0, seed=3), 5.10, 0.44,
      pcl(5.10), 'lead', 'пузырь «Я!» у Clawd (lt 5.10)', 'голос Clawd, 1 слог')
    P(L.cd_voice([(0.11, (0.0, 7.0, 2.0), 2.0)], base=450.0, vol=1.0, seed=4), 5.78, 0.46,
      pcd(5.78), 'lead', 'пузырь «Я!!» у Codex (lt 5.78)', 'голос Codex, 1 слог')
    P(L.take(S('punch_m'), dur=0.22, speed=1.05, warm_cut=6500, deh=-2), 6.08, 0.16,
      pcd(6.08, 4.0, 0.35), 'support', 'Codex подскакивает от злости', 'impactpunch_medium')
    P(L.lay(L.cl_voice([(0.13, (0.0, 5.0, 0.0), 0.0), (0.14, (1.0, 3.0, -5.0), 2.0)],
                       base=185.0, vol=1.0, seed=5),
            L.take(S('boom_low'), dur=0.45, warm_cut=2600, hp_cut=60) * 0.26), 6.46, 0.50, pcl(6.46),
      'lead', 'пузырь «Я!!!» у Clawd + тряска кадра (lt 6.46)', 'голос Clawd 2 слога + низкий гул')
    P(L.lay(L.cd_voice([(0.10, (0.0, 7.0, 4.0), 2.0), (0.09, (4.0, 9.0, 6.0), 4.0),
                        (0.11, (2.0, 5.0, -3.0), 5.0)], base=470.0, vol=1.0, seed=6),
            L.take(S('boom_low'), dur=0.55, warm_cut=2600, hp_cut=60) * 0.30), 7.14, 0.52, pcd(7.14),
      'lead', 'пузырь «Я!!!!» у Codex + сильная тряска (lt 7.14)', 'голос Codex 3 слога + гул')
    P(L.take(S('punch_m'), dur=0.26, speed=0.88, warm_cut=6500, deh=-2), 7.78, 0.18,
      pcd(7.78, 4.0, 0.38), 'support', 'Codex яростно топает (1 из 6)', 'impactpunch_medium')
    # --- кадр 4: аниме-сплит. Две панели = ОДИН стерео-cue (П9)
    P(L.stereo_pair(L.lay(L.take(S('slam_wood'), dur=0.32, warm_cut=7000, deh=-3),
                          L.sub(0.20, 100, 48, 0.26)),
                    L.lay(L.take(S('slam_metal'), dur=0.30, speed=1.15, warm_cut=7000, deh=-4),
                          L.sub(0.18, 120, 58, 0.22)),
                    offset=0.05, peak=1.0), 8.50, 0.50, 0.0, 'lead',
      'обе панели сплита влетают (8.50 слева и 8.55 справа)', 'wood_slam L + metal_slam R, один cue')
    P(L.lay(L.take(S('elec_whoosh'), dur=0.50, warm_cut=6000, deh=-4),
            L.take(S('boom'), dur=0.30, speed=1.3, warm_cut=6000) * 0.45), 8.86, 0.54, 0.0,
      'lead', 'первый разряд через стык + вспышка (сдвинут с 8.78 по П2)',
      'oga_whoosh_electric_02 + explosioncrunch')
    P(L.lay(L.take(S('punch0'), dur=0.30, warm_cut=6800, deh=-2), L.sub(0.20, 90, 44, 0.24)),
      9.40, 0.48, -0.08, 'lead', 'зум-панч №1: панели дёргаются, вспышка (lt 9.40)',
      'impactpunch_heavy_000 + низ')
    P(L.lay(L.take(S('punch1'), dur=0.34, speed=0.85, warm_cut=6800, deh=-2),
            L.sub(0.24, 78, 38, 0.26)), 10.00, 0.52, 0.08, 'lead',
      'зум-панч №2: ещё ближе, тоном ниже (lt 10.00)', 'impactpunch_heavy_001 + низ')
    # --- кадр 5: прыжок и потасовка
    P(L.take(S('swish_m'), dur=0.18, warm_cut=6800), 10.54, 0.22, 0.0, 'support',
      'склейка на общий план и прыжок обоих навстречу (объед. 10.50 и 10.60)',
      'oga_swish_7, один свуш по центру')
    P(L.lay(L.take(S('boom'), dur=0.60, speed=1.05, warm_cut=6500, deh=-3, hp_cut=40),
            L.take(S('snare'), dur=0.30, warm_cut=5000) * 0.45,
            L.take(S('hihat'), dur=0.40, warm_cut=7000) * 0.30,
            L.sub(0.28, 75, 32, 0.30)), 10.84, 0.56, 0.0, 'lead',
      'СТОЛКНОВЕНИЕ: вспышка, звезда, облако драки (T_HIT)',
      'explosioncrunch + breakbeat8_snare + hihat + низ')
    # 9 ударов -> 4 (П5); 24.80 приглушён по П7 (нагнетание музыки на 24.8)
    # 9 ударов -> 3: четвёртый (24.80) убран, он попадал на нагнетание музыки 24.8 (П7)
    for (tb, key, sp, v, pan) in ((0.12, 'punch2', 1.0, 0.20, -0.25),
                                  (0.40, 'wood_hit', 1.1, 0.18, 0.25),
                                  (0.68, 'punch3', 0.9, 0.22, -0.25)):
        P(L.take(S(key), dur=0.24, speed=sp, warm_cut=6500, deh=-2), 10.84 + tb, v, pan,
          'support' if v > 0.14 else 'decor',
          f'из облака вылетают лапы/варежки, удар на {S2 + 10.84 + tb:.2f}', SMP[key][1])
    # 9 символов ругани -> 2
    P(L.take(S('select'), dur=0.09, speed=1.15, warm_cut=6500, deh=-3), 11.10, 0.14, -0.30,
      'decor', 'символ ругани вылетает из облака', 'ui select_002')
    P(L.take(S('megaswosh'), dur=0.30, reverse=True, warm_cut=6800), 11.72, 0.46, 0.0, 'lead',
      'белая вспышка-переход в следующую сцену (11.82–12.00)', 'реверс-мегасвуш')
    return tr


def build_s2_amb():
    tr = Track(12.0)
    src = amb_sample(('ambience', ['wind']), ('ambience', ['nature_misc'])) or S('grass')
    x = L.stereo_take(src, loop_to=8.60, warm_cut=5600, fin=0.4, fout=0.8)
    tr.add(L.rms_to(bed_hp(x), -33.0), 0.0)
    EVENTS.append((S2, 'bed', 'поляна за спорящими героями (до сплита 8.50)',
                   f'{os.path.basename(src)} (RMS -33)', 0.0, 0.0))
    return tr


def build_s2_fx_bed():
    """Два фона второй половины сцены: разряд через стык и клубящееся облако драки."""
    tr = Track(3.3)                                   # t = 21.78
    el = L.stereo_take(S('elec_bed'), t0=1.2, loop_to=1.62, warm_cut=3600, fin=0.15, fout=0.45)
    tr.add(L.rms_to(bed_hp(el), -30.0), 0.0)
    EVENTS.append((21.78, 'bed', 'молния всё время бьёт через стык панелей (21.78–23.40)',
                   'electric cello loop (RMS -30)', 0.0, 0.0))
    ra = L.stereo_take(S('gear_rattle'), t0=0.2, loop_to=1.12, warm_cut=3000, fin=0.1, fout=0.35)
    # гребёнку транзиентов приминаем, иначе «фон» читается как серия ударов
    ra = L.soft_clip(ra / (np.max(np.abs(ra)) or 1.0) * 0.35, knee=0.10, ceil=0.16)
    tr.add(L.rms_to(bed_hp(ra), -31.0), 23.88 - 21.78)
    EVENTS.append((23.88, 'bed', 'облако драки клубится, пыль, тряска',
                   'oga_gear03 замедленный (RMS -31)', 0.0, 0.0))
    return tr



# ================================================================== сборка и самопроверка

BAND_EDGES = (20, 250, 1200, 2500, 6000, 20000)
BAND_NAMES = ('20–250', '250–1.2k', '1.2–2.5k', '2.5–6k', '6–20k')


def attacks(t0, t1):
    """Все атаки (lead/support/decor) в окне — для П4."""
    return sorted(e[0] for e in EVENTS if e[1] != 'bed' and t0 <= e[0] < t1)


def check_plan():
    """Чек-лист §7 PLAN.md, пункты, проверяемые по списку событий."""
    out = []
    ev = sorted((e[0], e[1]) for e in EVENTS if e[1] != 'bed')
    leads = [t for t, r in ev if r == 'lead']
    # Исключение из §1 таблицы: три удара титров (CLAUDE / CODEX / VS) идут по кадру
    # через 0.34 с и в плане все три помечены «ост» как lead — это ритм заставки.
    EXC = {(0.7, 1.04), (1.04, 1.38)}
    bad2 = [(a, b) for a, b in zip(leads, leads[1:])
            if b - a < 0.35 - 1e-6 and (round(a, 2), round(b, 2)) not in EXC]
    out.append(('П2 лиды >= 0.35 c', bad2))
    bad3 = []
    for t, r in ev:
        if r != 'lead':
            near = [q for q in leads if 0 < abs(q - t) < 0.12 - 1e-6]
            if near:
                bad3.append((t, near[0]))
    out.append(('П3 lead<->support >= 0.12 c', bad3))
    bad1 = []
    ts = [t for t, _ in ev]
    for i, t in enumerate(ts):
        k = sum(1 for q in ts if t <= q < t + 0.12)
        if k >= 3:
            bad1.append((t, k))
    out.append(('П1 нет окна 0.12 c с 3+ событиями', bad1))
    bad4 = []
    for i, t in enumerate(ts):
        k = sum(1 for q in ts if t <= q < t + 1.0)
        lim = 6 if (21.5 <= t < 25.0 or t < 4.0) else 4
        if k > lim:
            bad4.append((round(t, 2), k, lim))
    out.append(('П4 <= 4 атаки в окне 1 c (в экшне 6)', bad4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', action='store_true')
    a = ap.parse_args()

    plan = os.path.join(ROOT, 'audio', 'PLAN.md')
    print('PLAN.md:', 'найден, работаю по нему' if os.path.exists(plan) else 'НЕТ')
    amb_cat = os.path.join(ROOT, 'audio', 'samples', 'catalog_ambience.json')
    w = amb_sample(('ambience', ['wind']), ('ambience', ['nature_misc']))
    b = amb_sample(('ambience', ['birds']),)
    print('ambience:', 'каталог есть' if os.path.exists(amb_cat) else
          'каталога ещё нет, беру файлы прямо из audio/samples/ambience/',
          '| ветер:', os.path.basename(w) if w else 'mech/oga_grass',
          '| птицы:', os.path.basename(b) if b else 'нет')

    os.makedirs(STEM_DIR, exist_ok=True)
    specs = [
        ('s0_title.wav', build_s0, S0, 'sfx'),
        ('s0_type.wav', build_s0_type, 1.90, 'ui'),
        ('s0_amb.wav', build_s0_amb, S0, 'amb'),
        ('s1_meadow.wav', build_s1, S1, 'sfx'),
        ('s1_amb.wav', build_s1_amb, S1, 'amb'),
        ('s1_hover.wav', build_s1_hover, S1 + 2.72, 'amb'),
        ('s2_amb.wav', build_s2_amb, S2, 'amb'),
        ('s2_fx_bed.wav', build_s2_fx_bed, 21.78, 'amb'),
        ('s2_argue.wav', build_s2, S2, 'sfx'),
    ]
    stems = []
    print(f'\n{"стем":16s} {"t":>6} {"бус":4s} {"длит":>5} {"пик":>5} {"RMS":>6}  ' +
          '  '.join(f'{n:>8}' for n in BAND_NAMES))
    for name, fn, t0, bus in specs:
        tr = fn()
        x = tr.stereo()
        # §7 п.6: ФНЧ 9 кГц на всех стемах агента
        x = np.stack([lp(x[:, 0], 9000, order=3), lp(x[:, 1], 9000, order=3)], 1).astype(np.float32)
        x = L.soft_clip(x)
        path = os.path.join(STEM_DIR, name)
        save_wav(path, x)
        rms = float(np.sqrt(np.mean(x ** 2)))
        b = L.bands(x, BAND_EDGES)
        flag = '' if (b[3] <= 18.0 and b[4] <= 12.0) else '  <-- спектр!'
        print(f'{name:16s} {t0:6.2f} {bus:4s} {tr.dur:5.2f} {np.max(np.abs(x)):5.3f} '
              f'{20 * math.log10(max(rms, 1e-9)):6.1f}  ' +
              '  '.join(f'{v:7.1f}%' for v in b) + flag)
        stems.append({'file': os.path.relpath(path, ROOT), 't': round(t0, 3),
                      'gain': 1.0, 'pan': 0.0, 'bus': bus})
    write_manifest(CUE_FILE, stems)

    n = {r: sum(1 for e in EVENTS if e[1] == r) for r in ('lead', 'support', 'decor', 'bed')}
    print(f'\nманифест {os.path.relpath(CUE_FILE, ROOT)}: {len(stems)} стемов, '
          f'{len(EVENTS)} событий (lead {n["lead"]}, support {n["support"]}, '
          f'decor {n["decor"]}, bed {n["bed"]})')
    ev = [e[0] for e in EVENTS if e[1] != 'bed']
    print(f'атак в 0–25 c: {len(ev)} (норма плана <= 2.7/с = {2.7 * 25:.0f})')
    for title, bad in check_plan():
        print(f'  {title}: ' + ('ОК' if not bad else f'НАРУШЕНИЯ {bad}'))
    if _WARN:
        print('  громкость/панорама вне диапазона роли:')
        for w in _WARN:
            print('   ', w)
    if a.table:
        print('\n  сек | роль | событие на экране | звук | пик | pan')
        for t, role, what, snd, peak, pan in sorted(EVENTS):
            print(f'{t:6.2f} | {role:7s} | {what:62s} | {snd:46s} | {peak:.2f} | {pan:+.2f}')


if __name__ == '__main__':
    main()
