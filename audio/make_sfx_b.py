"""Звуковые эффекты блока B: 25.00–48.00 c (s3_code 25–34, s4_race 34–48).

    python3 audio/make_sfx_b.py

Версия 3 — по audio/PLAN.md: только реальные сэмплы из audio/samples/, событий
столько, сколько оставляет таблица §1; громкости по §4 (lead / support / bed),
плотность по §2 (П1–П9), тембр по §5.

Отличия от плана, сделанные по его же жёстким правилам (каждое — в отчёте):
  * 25.56 «ДЗЫНЬ» и 25.66 «КОД» (и 34.56/34.66) склеены в один лид — П2/П9
    не дают двум лидам стоять в 0.10 с друг от друга;
  * 32.83 свуш штампа снят — между лидом 32.80 и лидом 32.95 нет места (П1/П3);
    32.80 при этом опущен до support, как требует П7 (−6 дБ на кульминации);
  * 33.28 зап и 33.30 свуш склеены в один support (П9), 33.00 снят (П3);
  * 44.55 свуш кубка сдвинут на 44.62, 41.37 занос — на 41.33, 46.42 «оглушены» —
    на 46.55 (П3: не ближе 0.12 с к лиду);
  * 45.75 «зенит» и 45.80 «падение» склеены в один support (П9);
  * зажигание карта 36.05 не отдельное событие, а голова моторного bed — иначе
    оно стоит в 0.05 с от лида-отсчёта (П3).
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from lib_sfx_b import (add, attacks, bed, lead, pitch, set_peak, set_rms, smear, smp,  # noqa: E402
                       soft_cap, soften, support, tail_fade, take, varispeed)
from synth import SR, Track, fade, lp, load_wav, n_samples, write_manifest  # noqa: E402

OUT = os.path.join(ROOT, 'audio', 'stems', 'sfx_b')
CUES = os.path.join(ROOT, 'audio', 'cues', 'sfx_b.json')
os.makedirs(OUT, exist_ok=True)

S3, S4 = 25.0, 34.0
STEMS, EVENTS = [], []
# потолок пика по стемам — верх лида своего участка (§4 PLAN.md)
CAPS = {'s3_card': 0.50, 's3_keys': 0.18, 's3_ui': 0.20, 's3_hits': 0.50, 's3_room': 0.14,
        's4_card': 0.50, 's4_start': 0.50, 's4_race': 0.48, 's4_finish': 0.56,
        's4_after': 0.46, 's4_engines': 0.20, 's4_wind': 0.20, 's4_crowd': 0.24,
        's4_storm': 0.20}

# ---------------------------------------------------------------- сэмплы (§5: фильтры по max_hi/centroid)
SMP = dict(
    riser='oga_swosh_03.wav',                           # шорох-риз на карточке
    gong='oga_gong_02.wav',                             # «ДЗЫНЬ»
    word_slam='kenney_impactpunch_heavy_002.wav',       # слово раунда врезается
    word_jingle='kenney_jingles_steel04.wav',           # + чип-джингл в том же ударе
    flash_whoosh='oga_qubodup_megaswosh2.wav',          # вспышка-уход
    key_cl=['kenney_switch_005.wav', 'kenney_switch_003.wav', 'kenney_switch_002.wav',
            'oga_click.wav', 'kenney_click_001.wav'],
    key_cx=['kenney_select_002.wav', 'kenney_select_005.wav', 'oga_click.wav',
            'kenney_click_001.wav', 'kenney_switch_002.wav'],
    hud='kenney_confirmation_001.wav',
    popup='oga_sfx_sounds_button12.wav',
    cut_a='oga_swish_7.wav',
    cut_b='oga_swish_9.wav',
    cut_low='oga_swosh_06.wav',
    cut_air='oga_swosh_05.wav',
    cam_hit='kenney_impactwood_light_002.wav',
    done='oga_sfx_sounds_powerup13.wav',
    stamp='kenney_impactwood_heavy_002.wav',
    stamp_err='oga_sfx_sounds_error10.wav',
    zap='oga_ui_electric_02.wav',
    kart_in='kenney_impactmetal_heavy_002.wav',         # Claude запрыгивает в карт
    board='oga_whoosh_electric_02.wav',
    board_sw='kenney_switch_002.wav',
    jump='kenney_phasejump2.wav',
    ignition='oga_generic_car_startengine_1.wav',
    beep='kenney_tone1.wav',
    dirt='kenney_footstep_grass_000.wav',
    squeal='oga_tire_squeal.wav',
    pass_by='oga_qubodup_megaswosh2.wav',
    pass_hover='oga_whoosh_electric_01.wav',
    bump_a='kenney_impactmetal_heavy_004.wav',
    bump_b='kenney_impactmetal_heavy_002.wav',
    crash='kenney_explosioncrunch_000.wav',
    crash_metal='kenney_impactmetal_000.wav',
    brawl_hit1='kenney_impactpunch_medium_000.wav',
    brawl_hit2='kenney_impactpunch_medium_002.wav',
    trophy_air='oga_whoosh2_0.wav',
    shimmer='oga_sfx_coin_cluster3.wav',
    shatter='kenney_impactglass_heavy_001.wav',
    shatter_low='kenney_impactwood_heavy_004.wav',
    dazed='kenney_lowdown.wav',
    thunder='kenney_explosioncrunch_004.wav',
    thunder_low='kenney_lowfrequency_explosion_001.wav',
    drone='oga_ambience_sinister_electric_cello_loop_02.wav',
    motor='oga_motorseamless01.wav',
    hover='kenney_spaceengine_002.wav',
    wind='oga_wind.wav',
    crowd='oga_gregor_quendel_crowd_cheering_sounds_04_strong_c.wav',
    rattle='kenney_lowfrequency_explosion_000.wav',
)


def stem(name, t0, dur, bus='sfx'):
    tr = Track(dur)
    STEMS.append(dict(file=f'audio/stems/sfx_b/{name}.wav', t=round(t0, 3), gain=1.0, pan=0.0,
                      bus=bus, _tr=tr, _name=name, _dur=dur))
    return tr, t0


def ev(t, role, screen, sound, peak=None):
    EVENTS.append((round(t, 3), role, screen, sound, peak))


def glue(parts):
    """Склеить слои одного события в один сигнал с одной атакой (П9)."""
    n = max(len(p[0]) for p in parts)
    out = np.zeros(n, np.float32)
    for (x, g, off) in parts:
        p = n_samples(off)
        out[p:p + len(x)] += x[:max(0, n - p)] * g
    return out


# ================================================================ S3 (25.00–34.00)
T_FIN3 = S3 + 7.8
POPUPS = [27.40, 29.55, 31.50, 32.20]        # 4 опорных блипа вместо 12 (§1)


def pan_cl(t):
    if t < S3 + 3.30:
        return -0.45
    if t < S3 + 3.95:
        return -0.05
    if t < S3 + 4.60:
        return -0.50
    if t < S3 + 5.20:
        return 0.0
    if t < S3 + 5.80:
        return -0.50
    return -0.50


def pan_cx(t):
    if t < S3 + 3.30:
        return 0.45
    if t < S3 + 3.95:
        return 0.50
    if t < S3 + 4.60:
        return 0.05
    if t < S3 + 5.20:
        return 0.50
    if t < S3 + 5.80:
        return 0.0
    return 0.50


def card_stem(name, t0, exit_at=None, dur=1.6):
    """Карточка раунда: риз, «ДЗЫНЬ», второй «ДЗЫНЬ»+слово одним ударом, уход."""
    tr, _ = stem(name, t0, dur)
    add(tr, support(take(smp(SMP['riser']), 0.55, 0.05, 0.25), 0.20), 0.0, pan=0.0)
    add(tr, lead(take(smp(SMP['gong']), 1.0), 0.44), 0.22, pan=-0.25)
    hit = glue([(take(smp(SMP['gong']), 0.9), 1.0, 0.0),
                (take(smp(SMP['word_slam']), 0.45), 0.85, 0.06),
                (take(smp(SMP['word_jingle']), 0.5), 0.35, 0.06)])
    add(tr, lead(hit, 0.50), 0.60, pan=0.10)
    if exit_at is not None:
        add(tr, support(take(smp(SMP['flash_whoosh']), 0.5), 0.20), exit_at, pan=0.0)
    return tr


def build_s3():
    # ---------------- карточка «РАУНД 1 / КОД» (25.00–26.30)
    card_stem('s3_card', S3, exit_at=1.16)
    ev(S3 + 0.00, 'sup', 'карточка раунда, код бежит по фону', 'шорох-риз', 0.20)
    ev(S3 + 0.22, 'lead', '«ДЗЫНЬ!» слева, кольца от колокола', 'гонг', 0.44)
    ev(S3 + 0.60, 'lead', 'второй «ДЗЫНЬ!» + слово «КОД» врезается', 'гонг + панч + чип-джингл (один удар)', 0.50)
    ev(S3 + 1.16, 'sup', 'вспышка-уход с карточки', 'свуш', 0.20)

    # ---------------- дробь клавиш как непрерывный bed (П5), RMS −29
    tr, t0 = stem('s3_keys', S3 + 1.28, 6.62)
    rng = np.random.default_rng(777)
    mono = np.zeros((n_samples(6.62), 2), np.float32)
    kt = Track(6.62)
    pools = {'cl': [take(smp(n), 0.13, 0.002, 0.05) for n in SMP['key_cl']],
             'cx': [take(smp(n), 0.11, 0.002, 0.05) for n in SMP['key_cx']]}
    for who, rate, pan_fn, base_st in (('cl', 3.0, pan_cl, -3.0), ('cx', 3.4, pan_cx, 1.5)):
        t = S3 + 1.34
        i = 0
        while t < S3 + 7.76:
            x = pools[who][i % 5]
            y = soften(pitch(x, base_st + float(rng.uniform(-1.5, 1.5))), 5000, -7.0, 6.0)
            add(kt, set_peak(y, float(rng.uniform(0.55, 1.0))), t - t0,
                pan=pan_fn(t) + float(rng.uniform(-0.04, 0.04)))
            k = 1.18 if t >= S3 + 5.80 else 1.0
            t += (1.0 / (rate * k)) * float(rng.uniform(0.6, 1.45))
            i += 1
    for ch in (0, 1):
        mono[:, ch] = smear(kt.buf[:, ch], room=0.13, cutoff=2400, mix=0.78)
    for _ in range(4):        # RMS −29: на 13 дБ ниже музыки s3 и ниже порога детектора атак
        mono = set_rms(mono, -29.0)
        mono = soft_cap(mono, 0.12, 0.18)
    tr.buf[:] = mono
    ev(S3 + 1.30, 'bed', 'оба долбят по клавишам', 'реальные щелчки, размазаны свёрткой в слой: RMS −29 dBFS, пик 0.16')

    # ---------------- HUD и окна кода (26.34–32.60)
    tr, t0 = stem('s3_ui', S3 + 1.34, 6.30, bus='ui')
    hud = glue([(take(smp(SMP['cut_a']), 0.2), 1.0, 0.0),
                (take(smp(SMP['hud']), 0.3), 0.8, 0.03)])
    add(tr, support(hud, 0.18), S3 + 1.38 - t0, pan=0.0)
    ev(S3 + 1.38, 'sup', 'HUD-полосы выезжают сверху', 'свуш + один блип', 0.18)
    for k, t in enumerate(POPUPS):
        p = (t - 26.3) / 6.5
        add(tr, support(pitch(take(smp(SMP['popup']), 0.2), -2 + 5 * p), 0.14), t - t0,
            pan=(-0.35 if k % 2 == 0 else 0.35))
        ev(t, 'sup', 'всплывает окно кода, счёт растёт', 'блип окна, выше с прогрессом', 0.14)

    # ---------------- склейки и финал (28.20–34.00)
    tr, t0 = stem('s3_hits', S3 + 3.20, 5.80)
    push = glue([(take(smp(SMP['cut_a']), 0.22), 1.0, 0.0),
                 (take(smp(SMP['cam_hit']), 0.2), 0.5, 0.05)])
    add(tr, support(push, 0.20), S3 + 3.26 - t0, pan=-0.2)
    ev(S3 + 3.30, 'sup', 'быстрый наезд на Claude', 'свуш + лёгкий толчок камеры', 0.20)
    add(tr, support(take(smp(SMP['cut_b']), 0.22), 0.20), S3 + 3.91 - t0, pan=0.2)
    ev(S3 + 3.95, 'sup', 'быстрый наезд на Codex', 'свуш', 0.20)
    add(tr, support(take(smp(SMP['cut_low']), 0.3), 0.20), S3 + 5.74 - t0, pan=0.0)
    ev(S3 + 5.80, 'sup', 'камера падает к полу, темп дроби +18 %', 'низкий свуш', 0.20)
    done = glue([(pitch(take(smp(SMP['done']), 0.45), -1.0), 1.0, 0.0),
                 (pitch(take(smp(SMP['done']), 0.45), 3.0), 0.7, 0.02)])
    add(tr, support(done, 0.20), T_FIN3 - t0, pan=0.0)
    ev(T_FIN3, 'sup', 'обе полосы 100 %, вспышка', 'две «галочки» одним cue (−6 дБ по П7)', 0.20)
    stamp = glue([(take(smp(SMP['stamp']), 0.4), 1.0, 0.0),
                  (take(smp(SMP['stamp_err']), 0.3), 0.45, 0.05)])
    add(tr, lead(stamp, 0.50), T_FIN3 + 0.15 - t0, pan=0.0)
    ev(T_FIN3 + 0.15, 'lead', 'удар штампа «НИЧЬЯ!», тряска, пыль', 'тяжёлый удар + чип-«ошибка»', 0.50)
    turn = glue([(take(smp(SMP['zap']), 0.3), 1.0, 0.0),
                 (take(smp(SMP['cut_b']), 0.2), 0.6, 0.04)])
    add(tr, support(turn, 0.18), T_FIN3 + 0.50 - t0, pan=0.0)
    ev(T_FIN3 + 0.50, 'sup', 'разворот друг к другу, молнии-взгляды', 'зап + свуш одним cue', 0.18)

    # ---------------- гул голограмм (26.30–33.00), RMS −31
    tr, t0 = stem('s3_room', S3 + 1.30, 6.70, bus='amb')
    amp = [(0.0, 0.0), (0.5, 1.0), (4.5, 1.05), (6.4, 1.1), (6.7, 0.0)]
    add(tr, bed(SMP['drone'], 6.70, -31.0, [0.0, 6.7], [0.92, 1.0], amp, lp_hz=1600, hp_hz=120),
        0.0, pan=-0.2, fin=0.5, fout=0.6)
    ev(S3 + 1.30, 'bed', 'неоновая арена, две голограммы', 'электрический дрон, RMS −31', None)


# ================================================================ S4 (34.00–48.00)
T_GO = S4 + 3.00
T_PASS = S4 + 6.604
BUMP1, BUMP2 = S4 + 7.47, S4 + 8.12
T_HIT = S4 + 10.50


def build_s4():
    # ---------------- карточка «РАУНД 2 / ГОНКА» (34.00–35.20)
    card_stem('s4_card', S4, exit_at=1.08, dur=1.35)
    ev(S4 + 0.00, 'sup', 'карточка раунда, спидлайны', 'шорох-риз', 0.20)
    ev(S4 + 0.22, 'lead', '«ДЗЫНЬ!» + «РАУНД 2»', 'гонг', 0.44)
    ev(S4 + 0.60, 'lead', 'второй «ДЗЫНЬ!» + слово «ГОНКА»', 'гонг + панч + джингл (один удар)', 0.50)
    ev(S4 + 1.08, 'sup', 'вспышка-уход', 'свуш', 0.20)

    # ---------------- старт (35.20–37.60)
    tr, t0 = stem('s4_start', S4 + 1.20, 2.45)
    add(tr, lead(take(smp(SMP['kart_in']), 0.25), 0.40), S4 + 1.30 - t0, pan=-0.25)
    ev(S4 + 1.30, 'lead', 'Claude запрыгивает в карт', 'глухой удар металла', 0.40)
    brd = glue([(take(smp(SMP['board']), 0.17, 0.01, 0.06), 1.0, 0.0),
                (take(smp(SMP['board_sw']), 0.12), 0.5, 0.05)])
    add(tr, support(brd, 0.18), S4 + 1.45 - t0, pan=0.40)
    ev(S4 + 1.45, 'sup', 'ховерборд влетает справа', 'электро-свуш + щелчок захвата', 0.18)
    add(tr, support(take(smp(SMP['jump']), 0.3), 0.16), S4 + 1.64 - t0, pan=0.35)
    ev(S4 + 1.64, 'sup', 'Codex прыгает на доску (и приземляется)', 'чип-прыжок', 0.16)
    for k, lt in enumerate((2.10, 2.40, 2.70)):
        add(tr, lead(take(smp(SMP['beep']), 0.13), 0.34), S4 + lt - t0, pan=0.0)
        ev(S4 + lt, 'lead', f'светофор, отсчёт «{3 - k}»', 'чистый тон-би', 0.34)
    go = glue([(pitch(take(smp(SMP['beep']), 0.16), 7.0), 1.0, 0.0),
               (take(pitch(smp(SMP['squeal']), -5.0), 0.45, 0.01, 0.2), 0.7, 0.02),
               (take(smp(SMP['dirt']), 0.5, 0.005, 0.2), 0.5, 0.02)])
    add(tr, lead(go, 0.50), T_GO - t0, pan=0.0)
    ev(T_GO, 'lead', 'зелёный, «СТАРТ!», взрыв пыли', 'би + пробуксовка + грунт (один cue)', 0.50)

    # ---------------- моторы (35.90–44.55), RMS −27; голова слоя — зажигание карта
    tr, t0 = stem('s4_engines', S4 + 1.90, 8.65, bus='amb')
    kt = [0.0, 0.14, 0.15, 1.10, 2.00, 2.50, 2.67, 3.55, 4.71, 5.57, 5.77, 6.22, 6.42, 8.65]
    kr = [0.62, 0.62, 0.62, 0.66, 1.00, 0.98, 0.98, 1.00, 1.05, 0.92, 1.02, 0.91, 1.02, 1.06]
    ka = [(0.0, 0.0), (0.15, 0.55), (1.10, 0.6), (1.40, 1.0), (2.50, 1.0), (2.67, 0.55),
          (3.55, 0.6), (3.85, 0.95), (4.40, 1.15), (4.71, 1.5), (5.00, 0.8), (5.15, 1.0),
          (6.80, 1.0), (7.20, 0.75), (8.45, 0.7), (8.65, 0.5)]
    add(tr, bed(SMP['motor'], 8.65, -30.0, kt, kr, ka, lp_hz=1700, hp_hz=120),
        0.0, pan=-0.35, fin=0.2, fout=0.3)
    ht = [0.0, 0.30, 1.10, 2.00, 2.50, 3.55, 4.71, 5.57, 5.77, 6.22, 6.42, 8.65]
    hr = [0.55, 0.58, 0.60, 0.98, 0.96, 0.99, 1.03, 0.92, 1.00, 0.91, 1.00, 1.04]
    ha = [(0.0, 0.0), (0.10, 0.85), (0.35, 0.55), (1.10, 0.6), (1.40, 1.0), (2.50, 1.0),
          (2.67, 0.55), (3.55, 0.6), (3.85, 0.95), (4.40, 1.1), (4.71, 1.45), (5.00, 0.8),
          (5.15, 1.0), (6.80, 1.0), (7.20, 0.75), (8.45, 0.7), (8.65, 0.5)]
    add(tr, bed(SMP['hover'], 8.65, -31.0, ht, hr, ha, lp_hz=1200, hp_hz=120),
        0.0, pan=0.35, fin=0.25, fout=0.3)
    ign = soften(take(smp(SMP['ignition']), 0.95, 0.06, 0.4), 4000, -6.0, 60.0)
    add(tr, set_peak(ign, 0.17), S4 + 2.05 - t0, pan=-0.30)
    ev(S4 + 1.90, 'bed', 'зажигание, карт тарахтит, доска гудит', 'мотор + космо-гул, высота по скорости, RMS −27 сумма', None)

    # ---------------- ветер (37.00–44.55), RMS −29
    tr, t0 = stem('s4_wind', T_GO, 7.55, bus='amb')
    wa = [(0.0, 0.0), (0.35, 0.3), (0.9, 1.0), (1.55, 1.15), (2.6, 1.05), (3.3, 0.85),
          (3.5, 1.25), (3.75, 1.0), (5.6, 1.1), (5.71, 0.62), (7.3, 0.6), (7.55, 0.3)]
    add(tr, bed(SMP['wind'], 7.55, -29.0, [0.0, 0.9, 1.95, 7.55], [0.8, 0.85, 1.12, 1.15], wa,
                lp_hz=2400, hp_hz=140), 0.0, pan=0.0, fin=0.35, fout=0.5)
    ev(T_GO, 'bed', 'гонщики срываются с места, встречный ветер', 'ветер, громкость и высота по скорости, RMS −29', None)

    # ---------------- гонка (38.30–44.60)
    tr, t0 = stem('s4_race', S4 + 4.30, 6.30)
    add(tr, support(take(smp(SMP['cut_air']), 0.4), 0.18), S4 + 4.34 - t0, pan=-0.1, pan_to=0.3)
    ev(S4 + 4.40, 'sup', 'склейка на «вертолёт» сверху', 'свуш', 0.18)
    px = varispeed(take(smp(SMP['pass_by']), 0.58), 0.62, [0, 0.62], [1.45, 0.62], loop=False)
    ph = varispeed(take(smp(SMP['pass_hover']), 0.7), 0.58, [0, 0.58], [1.5, 0.7], loop=False)
    add(tr, lead(px, 0.44), T_PASS - 0.30 - t0, pan=-0.60, pan_to=0.60)
    add(tr, set_peak(soften(ph), 0.16), T_PASS - 0.26 - t0, pan=0.30, pan_to=-0.45)
    ev(T_PASS, 'lead', 'встречный проезд вплотную к камере', 'доплер: высота падает, панорама слева направо', 0.44)
    add(tr, support(take(pitch(smp(SMP['squeal']), -3.0), 0.40, 0.02, 0.18), 0.18),
        BUMP1 - 0.14 - t0, pan=-0.35)
    ev(BUMP1 - 0.14, 'sup', 'Claude уводит карт на соперника', 'скрип шин слева', 0.18)
    add(tr, lead(take(smp(SMP['bump_a']), 0.3), 0.44), BUMP1 - t0, pan=0.0)
    ev(BUMP1, 'lead', 'толчок №1, искры между картом и доской', 'удар металла', 0.44)
    add(tr, support(take(pitch(smp(SMP['squeal']), 0.0), 0.36, 0.02, 0.16), 0.18),
        BUMP2 - 0.14 - t0, pan=0.35)
    ev(BUMP2 - 0.14, 'sup', 'резкий вираж навстречу друг другу', 'скрип шин справа', 0.18)
    add(tr, lead(pitch(take(smp(SMP['bump_b']), 0.26), -3.0), 0.44), BUMP2 - t0, pan=0.0)
    ev(BUMP2, 'lead', 'толчок №2, оба подпрыгивают', 'удар металла ниже тоном', 0.44)
    add(tr, support(take(smp(SMP['cut_b']), 0.22), 0.18), S4 + 8.54 - t0, pan=0.3, pan_to=-0.25)
    ev(S4 + 8.60, 'sup', 'склейка на погоню к финишной арке', 'свуш', 0.18)

    # ---------------- толпа (43.40–46.00), RMS −28
    tr, t0 = stem('s4_crowd', S4 + 9.40, 2.60, bus='amb')
    ca = [(0.0, 0.0), (0.25, 0.3), (0.9, 0.75), (1.12, 1.3), (1.45, 1.0), (2.1, 0.5), (2.6, 0.0)]
    add(tr, bed(SMP['crowd'], 2.60, -28.0, None, None, ca, lp_hz=3000, hp_hz=180, start=4.0),
        0.0, pan=0.0, fin=0.35, fout=0.5)
    ev(S4 + 9.60, 'bed', 'финишная арка растёт в кадре, трибуны', 'реальный рёв толпы, RMS −28', None)

    # ---------------- финиш и кубок (44.40–46.90)
    tr, t0 = stem('s4_finish', S4 + 10.40, 2.50)
    bam = glue([(take(smp(SMP['crash']), 0.62, 0.002, 0.25), 1.0, 0.0),
                (take(smp(SMP['crash_metal']), 0.6), 0.5, 0.02)])
    bam[n_samples(0.16):] = lp(bam[n_samples(0.16):], 2200)   # хвост без «трещащих» осколков
    add(tr, lead(bam, 0.56), T_HIT - t0, pan=0.0)
    ev(T_HIT, 'lead', 'вспышка, «БАМ!», столкновение у пьедестала', 'взрыв-хруст + металл (один cue)', 0.56)
    up = varispeed(take(smp(SMP['trophy_air']), 2.2, 0.05, 0.2), 1.05, [0, 1.05], [0.65, 1.7], loop=False)
    add(tr, support(up, 0.20), T_HIT + 0.16 - t0, pan=0.0, fin=0.15, fout=0.25)
    ev(T_HIT + 0.16, 'sup', 'кубок взлетает, камера за ним', 'восходящий свуш', 0.20)
    rb = [(0.0, 0.0), (0.12, 1.0), (0.55, 0.85), (1.0, 0.55), (1.5, 0.25), (1.75, 0.0)]
    add(tr, bed(SMP['rattle'], 1.75, -31.0, [0, 1.75], [0.7, 0.5], rb, lp_hz=380, hp_hz=110),
        T_HIT + 0.08 - t0, pan=0.0, fin=0.1, fout=0.3)
    for (dt, nm, g, pn) in ((0.45, 'brawl_hit1', 0.13, -0.25), (0.95, 'brawl_hit2', 0.11, 0.25)):
        add(tr, set_peak(soften(take(smp(SMP[nm]), 0.28)), g), T_HIT + dt - t0, pan=pn)
    ev(T_HIT + 0.08, 'bed', 'облако драки клубится, из него летят удары', 'низкий рокот + два удара на уровне bed', None)
    fall = glue([(take(smp(SMP['shimmer']), 0.45), 1.0, 0.0),
                 (varispeed(take(smp(SMP['trophy_air']), 1.4, 0.05, 0.2), 0.5, [0, 0.5], [1.7, 0.7],
                            loop=False), 0.8, 0.10)])
    add(tr, support(fall, 0.20), T_HIT + 1.25 - t0, pan=0.0)
    ev(T_HIT + 1.25, 'sup', 'кубок в зените и падает', 'блик + нисходящий свуш (один cue)', 0.20)
    br = glue([(take(smp(SMP['shatter']), 0.5), 1.0, 0.0),
               (take(smp(SMP['shatter_low']), 0.3), 0.6, 0.01)])
    add(tr, lead(br, 0.54), T_HIT + 1.82 - t0, pan=0.0)
    ev(T_HIT + 1.82, 'lead', '«ХРЯСЬ!» — кубок раскололся надвое', 'стекло + низкий удар', 0.54)

    # ---------------- контуженные и гроза (46.30–47.98)
    tr, t0 = stem('s4_after', S4 + 12.30, 1.68)
    add(tr, support(take(smp(SMP['dazed']), 0.6), 0.16), S4 + 12.55 - t0, pan=-0.2)
    ev(S4 + 12.55, 'sup', 'оба лежат оглушённые, звёздочки кружат', 'нисходящий «lowdown»', 0.16)
    th = glue([(lp(take(smp(SMP['thunder']), 1.6), 700), 1.0, 0.0),
               (take(smp(SMP['thunder_low']), 0.9), 0.7, 0.0)])
    add(tr, lead(th, 0.44, cutoff=2500, atk_ms=8.0), S4 + 13.20 - t0, pan=0.35, fout=0.4)
    ev(S4 + 13.20, 'lead', 'первая молния справа от арки', 'раскат грома, панорама вправо', 0.44)
    tail_fade(tr, 0.12)

    tr, t0 = stem('s4_storm', S4 + 12.40, 1.58, bus='amb')
    sa = [(0.0, 0.0), (0.15, 0.3), (0.7, 0.8), (1.1, 1.0), (1.58, 0.5)]
    add(tr, bed(SMP['wind'], 1.58, -28.0, [0, 1.58], [0.62, 0.7], sa, lp_hz=1400, hp_hz=120,
                start=0.6), 0.0, pan=0.0, fin=0.25, fout=0.45)
    ev(S4 + 12.45, 'bed', 'небо темнеет, набегает гроза', 'грозовой ветер, RMS −28', None)
    tail_fade(tr, 0.12)


# ================================================================ проверки и отчёт
BANDS5 = [(20, 250), (250, 1200), (1200, 2500), (2500, 6000), (6000, 20000)]
BANDS_PLAN = [(20, 200), (200, 2500), (2500, 6000), (6000, 20000)]


def band_mix(x, bands):
    mono = x.mean(1) if x.ndim == 2 else x
    X = np.abs(np.fft.rfft(mono))
    f = np.fft.rfftfreq(len(mono), 1 / SR)
    tot = X.sum() + 1e-9
    return [100 * X[(f >= a) & (f < b)].sum() / tot for a, b in bands]


def check_rules():
    cues = [(t, r) for (t, r, _, _, _) in EVENTS if r in ('lead', 'sup')]
    cues.sort()
    leads = [t for t, r in cues if r == 'lead']
    bad2 = [(a, b, round(b - a, 2)) for a, b in zip(leads, leads[1:]) if b - a < 0.349]
    bad3 = []
    for t, r in cues:
        if r != 'sup':
            continue
        for lt in leads:
            if 0 < abs(lt - t) < 0.12:
                bad3.append((t, lt))
    clusters = []
    ts = [t for t, _ in cues]
    for i in range(len(ts)):
        k = sum(1 for t in ts if 0 <= t - ts[i] < 0.12)
        if k >= 3:
            clusters.append(ts[i])
    return leads, bad2, bad3, clusters


def main():
    build_s3()
    build_s4()
    rows = []
    for s in STEMS:
        tr = s.pop('_tr')
        name, dur = s.pop('_name'), s.pop('_dur')
        tail_fade(tr, 0.06)
        if s['bus'] != 'amb':
            tr.buf[:, 0] = lp(tr.buf[:, 0], 9000)     # §7 п.6 — ФНЧ 9 кГц
            tr.buf[:, 1] = lp(tr.buf[:, 1], 9000)
        ceil = CAPS.get(name, 0.50)
        if float(np.max(np.abs(tr.buf))) > ceil:
            tr.buf[:] = soft_cap(tr.buf, ceil * 0.78, ceil)
        path = os.path.join(OUT, name + '.wav')
        tr.save(path, limit=False)
        x = load_wav(path)
        pk = float(np.max(np.abs(x)))
        r = float(np.sqrt(np.mean(x.mean(1) ** 2)))
        rows.append((s['t'], s['t'] + dur, s['bus'], name, pk, 20 * math.log10(max(r, 1e-9)),
                     band_mix(x, BANDS5), band_mix(x, BANDS_PLAN)))
    write_manifest(CUES, STEMS)

    print(f'{len(STEMS)} стемов -> {OUT}\n')
    print(f'{"начало":>7} {"конец":>7}  {"шина":5s} {"файл":11s} {"пик":>5s} {"RMS":>6s}  '
          f'{"20-250":>7s}{"250-1.2k":>9s}{"1.2-2.5k":>9s}{"2.5-6k":>8s}{">6k":>7s}   '
          f'{"<200":>6s}{"2.5-6k":>8s}{">6k":>7s}')
    for (a, b, bus, nm, pk, rdb, b5, bp) in sorted(rows):
        print(f'{a:7.2f} {b:7.2f}  {bus:5s} {nm:11s} {pk:5.2f} {rdb:6.1f}  '
              f'{b5[0]:6.1f}%{b5[1]:8.1f}%{b5[2]:8.1f}%{b5[3]:7.1f}%{b5[4]:6.1f}%   '
              f'{bp[0]:5.1f}%{bp[2]:7.1f}%{bp[3]:6.1f}%')
    bad = [nm for (_, _, _, nm, _, _, _, bp) in rows if bp[3] > 12 or bp[2] > 18]
    print('\n§5: стемы вне нормы (2.5–6 кГц >18 % или >6 кГц >12 %):', ', '.join(bad) if bad else 'нет')

    leads, bad2, bad3, clusters = check_rules()
    n_lead = sum(1 for e in EVENTS if e[1] == 'lead')
    n_sup = sum(1 for e in EVENTS if e[1] == 'sup')
    n_bed = sum(1 for e in EVENTS if e[1] == 'bed')
    print(f'\nсобытий: {n_lead} lead + {n_sup} support + {n_bed} слоёв = {n_lead + n_sup} дискретных cue')
    print('П2 (лиды ближе 0.35 c):', bad2 if bad2 else 'нет')
    print('П3 (support ближе 0.12 c к лиду):', bad3 if bad3 else 'нет')
    print('П1 (3+ cue в окне 0.12 c):', clusters if clusters else 'нет')
    print()
    for (t, role, screen, sound, peak) in sorted(EVENTS):
        pk = f'{peak:.2f}' if peak else ' bed'
        print(f'{t:6.2f} {role:4s} {pk}  {screen:50s} | {sound}')


if __name__ == '__main__':
    main()
