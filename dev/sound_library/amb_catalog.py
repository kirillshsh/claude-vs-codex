#!/usr/bin/env python3
"""Build catalog_ambience.json, LICENSES_AMB/* and README_AMB.md from the per-category items."""
import json, os, glob, datetime, collections

SP = "/path/to/scratchpad"
PROJ = SP + "/cartoon"
SAMP = PROJ + "/audio/samples"
AMB = SAMP + "/ambience"
LIC = SAMP + "/LICENSES_AMB"

CATS = ["rain", "thunder", "wind", "sea", "birds", "insects", "nature_misc", "room_tone"]

LICENSE_NOTES = {
    "CC0-1.0": ("CC0 1.0 Universal (Public Domain Dedication)",
                "https://creativecommons.org/publicdomain/zero/1.0/",
                "The rights holder waived all copyright and related rights worldwide. "
                "No permission, attribution or share-alike obligation applies. Attribution is "
                "recorded here as a courtesy only."),
    "PublicDomainMark-1.0": ("Public Domain Mark 1.0",
                             "https://creativecommons.org/publicdomain/mark/1.0/",
                             "The work is marked as free of known copyright restrictions. It may "
                             "be copied, modified, distributed and performed, including for "
                             "commercial purposes, without asking permission. Attribution is "
                             "recorded here as a courtesy only."),
    "CC-BY-3.0": ("Creative Commons Attribution 3.0 Unported",
                  "https://creativecommons.org/licenses/by/3.0/",
                  "Free to share and adapt, including commercially, PROVIDED the original "
                  "author, the source URL and the license are credited, and changes are "
                  "indicated. The derived files here are edited excerpts - see SOURCES.md."),
    "CC-BY-SA-3.0": ("Creative Commons Attribution-ShareAlike 3.0 Unported",
                     "https://creativecommons.org/licenses/by-sa/3.0/",
                     "Free to share and adapt PROVIDED the original author, the source URL and "
                     "the license are credited, changes are indicated, AND any redistributed "
                     "derivative is released under CC BY-SA 3.0 (or a compatible license). "
                     "The derived files here are edited excerpts - see SOURCES.md."),
    "CC-BY-NC-3.0": ("Creative Commons Attribution-NonCommercial 3.0 Unported",
                     "https://creativecommons.org/licenses/by-nc/3.0/",
                     "NON-COMMERCIAL USE ONLY, with attribution. Fine for this personal film; "
                     "these files must be removed or replaced before any commercial release."),
    "CC-BY-NC-SA-3.0": ("Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported",
                        "https://creativecommons.org/licenses/by-nc-sa/3.0/",
                        "NON-COMMERCIAL USE ONLY, with attribution and ShareAlike on "
                        "derivatives. Fine for this personal film; remove before commercial use."),
}

CAT_DESC = {
    "rain": "Ливень разной плотности, дождь по листве и по крыше, капли и лужи.",
    "thunder": "Раскаты грома: близкие резкие удары и далёкие раскатистые рокоты.",
    "wind": "Ветер в поле и в деревьях, лёгкий бриз, порывы.",
    "sea": "Морской прибой, накат волн на берег и на камни.",
    "birds": "Щебет птиц днём: луг, опушка, утренний хор.",
    "insects": "Сверчки и цикады вечером и ночью.",
    "nature_misc": "Ручей и река, шелест листвы и травы, шаги, всплески воды.",
    "room_tone": "Тихая комнатная подложка для интерьерных сцен.",
}


def main():
    items = []
    for c in CATS:
        p = os.path.join(SP, "items_%s.json" % c)
        if os.path.exists(p):
            items.extend(json.load(open(p)))
    items = [i for i in items if os.path.exists(os.path.join(PROJ, i["file"]))]
    items.sort(key=lambda i: (CATS.index(i["cat"]), i["file"]))

    os.makedirs(LIC, exist_ok=True)

    # --- catalog ---
    cat_items = []
    for i in items:
        e = {k: i[k] for k in ("file", "cat", "tags", "dur", "peak_db", "centroid_hz",
                               "hi_ratio", "loop") if k in i}
        e["rms_db"] = i["rms_db"]
        e["lo_ratio"] = i["lo_ratio"]
        e["lowpass_hz"] = i["lowpass_hz"]
        e["sr"] = 48000
        e["channels"] = 2
        e["bits"] = 16
        if i.get("crossfade_s"):
            e["crossfade_s"] = i["crossfade_s"]
            e["loop_joint_db"] = i["loop_joint_db"]
        if i.get("prominence_db") is not None:
            e["prominence_db"] = i["prominence_db"]
            e["rise_s"] = i["rise_s"]
        e["src"] = i["src"]
        e["title"] = i["title"]
        e["license"] = i["license"]
        e["license_url"] = i["license_url"]
        cat_items.append(e)
    by_cat = collections.Counter(i["cat"] for i in cat_items)
    doc = {
        "generated": datetime.date.today().isoformat(),
        "format": {"sr": 48000, "channels": 2, "bits": 16, "peak_dbfs": -3.0},
        "note": ("Real field recordings from archive.org (radio-aporee-maps). Every file is "
                 "peak-normalised to -3 dBFS and gently low-passed (see lowpass_hz) so the "
                 "share of energy above 5 kHz (hi_ratio) stays at or below 0.30."),
        "counts": dict(by_cat),
        "items": cat_items,
    }
    json.dump(doc, open(SAMP + "/catalog_ambience.json", "w"), indent=1, ensure_ascii=False)

    # --- licenses ---
    used = sorted(set(i["license"] for i in items))
    for lname in used:
        title, url, note = LICENSE_NOTES.get(lname, (lname, "", ""))
        files = [i for i in items if i["license"] == lname]
        srcs = {}
        for i in files:
            srcs.setdefault(i["src"], (i["title"], i.get("creator", "")))
        body = ["%s" % title, "=" * len(title), "",
                "Canonical text: %s" % url, "",
                note, "",
                "Files in audio/samples/ambience/ under this license: %d" % len(files),
                "Source recordings: %d" % len(srcs), "",
                "Source recordings (archive.org, collection radio-aporee-maps):", ""]
        for u, (t, cr) in sorted(srcs.items()):
            body.append('  "%s"%s' % (t, (" - " + cr) if cr else ""))
            body.append("      %s" % u)
        open(os.path.join(LIC, lname + ".txt"), "w").write("\n".join(body) + "\n")

    # --- per-file attribution table ---
    rows = ["# Ambience sources and attribution", "",
            "Every file below is an excerpt of a real field recording, edited "
            "(cut, high-passed at ~28 Hz, low-passed, peak-normalised to -3 dBFS, and for "
            "`_loop` files crossfaded into a seamless loop).", "",
            "| file | license | source recording | archive.org |",
            "|---|---|---|---|"]
    for i in items:
        rows.append("| `%s` | %s | %s | %s |" % (
            os.path.relpath(i["file"], "audio/samples"), i["license"],
            i["title"].replace("|", "/")[:90], i["src"]))
    open(os.path.join(LIC, "SOURCES.md"), "w").write("\n".join(rows) + "\n")

    # --- README ---
    loops = [i for i in items if i["loop"]]
    r = ["# Библиотека атмосфер (ambience)", "",
         "Реальные полевые записи, а не синтез. Скачаны с archive.org, коллекция "
         "[radio-aporee-maps](https://archive.org/details/radio-aporee-maps) — открытый архив "
         "полевых записей со всего мира.", "",
         "Всего файлов: **%d** (из них зациклённых `_loop`: **%d**) в %d категориях." % (
             len(items), len(loops), len(by_cat)), "",
         "## Формат", "",
         "Все файлы: **48 кГц, стерео, 16 бит WAV**, пик нормирован к **−3 dBFS**.", "",
         "## Как отбиралось", "",
         "Прослушать записи нельзя, поэтому отбор численный. Исходники качались фрагментами "
         "(~80 с из середины файла, чтобы не попасть на возню с микрофоном в начале), затем по "
         "скользящему окну считались RMS, спектральный центроид, доля энергии выше 5 кГц "
         "(`hi_ratio`), доля ниже 500 Гц (`lo_ratio`) и модуляция огибающей в полосе 2–10 Гц "
         "(индикатор речи). Окно отбраковывалось, если:", "",
         "- `hi_ratio` выше порога категории — шипящий «белый» звук, тот самый, что режет уши;",
         "- центроид вне коридора категории (например, дождь — 550–3200 Гц): всё, что ниже, "
         "оказывалось инфранизким рокотом ветра в микрофон, а не дождём;",
         "- разброс RMS слишком большой — в окно попало постороннее событие;",
         "- пик/медиана RMS выше 7–12 — щелчок, удар, хлопок;",
         "- модуляция 2–10 Гц выше 0.38 — в записи говорят;",
         "- центроид «плывёт» — спектр по ходу окна меняется, значит начался другой звук.", "",
         "Дальше — мягкий срез верха (ФНЧ Баттерворта 4-го порядка, нулевая фаза), причём "
         "частота среза подбирается итеративно вниз от 11 кГц, пока итоговый `hi_ratio` не "
         "станет ≤ 0.30. Гром отбирался отдельным детектором событий: раскаты ищутся как "
         "превышение над скользящей медианой уровня (≥ 8 дБ) с заметной долей низов, затем "
         "режутся по фронту и по спаду хвоста.", "",
         "## Зацикливание", "",
         "Файлы с суффиксом `_loop` склеены бесшовно: хвост длиной `crossfade_s` (обычно 1.5 с) "
         "перекрёстно сведён с началом по равномощной кривой. Скачок на стыке проверен "
         "численно — RMS зоны стыка сравнивается с RMS середины файла, приняты только варианты "
         "с расхождением `loop_joint_db` меньше 1.5 дБ. Значение есть в каталоге для каждого "
         "зациклённого файла.", "",
         "## Категории", ""]

    def plural(n):
        if n % 10 == 1 and n % 100 != 11:
            return "файл"
        if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
            return "файла"
        return "файлов"

    for c in CATS:
        n = by_cat.get(c, 0)
        if not n:
            continue
        nl = sum(1 for i in items if i["cat"] == c and i["loop"])
        durs = [i["dur"] for i in items if i["cat"] == c]
        cens = [i["centroid_hz"] for i in items if i["cat"] == c]
        his = [i["hi_ratio"] for i in items if i["cat"] == c]
        r.append("### `ambience/%s/` — %d %s (%d зациклено)" % (c, n, plural(n), nl))
        r.append("")
        r.append(CAT_DESC[c])
        r.append("")
        r.append("Длительность %.0f–%.0f с, центроид %d–%d Гц, hi_ratio %.3f–%.3f." % (
            min(durs), max(durs), min(cens), max(cens), min(his), max(his)))
        r.append("")
    r += ["## Лицензии", ""]
    for lname in used:
        n = sum(1 for i in items if i["license"] == lname)
        title, url, _ = LICENSE_NOTES.get(lname, (lname, "", ""))
        r.append("- **%s** — %d %s. %s" % (lname, n, plural(n), url))
    r += ["",
          "Подробности и обязательства по каждой лицензии — в `LICENSES_AMB/<лицензия>.txt`. "
          "Пофайловая таблица «какой файл из какой записи» — в `LICENSES_AMB/SOURCES.md`.", "",
          "Подавляющее большинство записей помечено Public Domain Mark 1.0 — ограничений нет. "
          "Для файлов под CC BY / CC BY-SA нужна атрибуция автора и ссылка на источник "
          "(готовый текст лежит в `LICENSES_AMB/`); CC BY-SA дополнительно требует выпускать "
          "производные под той же лицензией при распространении.", "",
          "## Каталог", "",
          "`audio/samples/catalog_ambience.json` — все файлы с измеренными характеристиками "
          "(`dur`, `peak_db`, `rms_db`, `centroid_hz`, `hi_ratio`, `lo_ratio`, `lowpass_hz`, "
          "`loop`, `crossfade_s`, `loop_joint_db`, для грома — `prominence_db` и `rise_s`), "
          "тегами, ссылкой на источник и лицензией.", ""]
    open(SAMP + "/README_AMB.md", "w").write("\n".join(r))

    print("catalog: %d items" % len(cat_items))
    print(dict(by_cat), "loops:", len(loops))
    print("licenses:", used)


if __name__ == "__main__":
    main()
