# -*- coding: utf-8 -*-
"""
cleanup.py — чистка проекта herb_bot от лишних файлов.

Удаляет ТОЛЬКО отладочный мусор и старую версию бота. Рабочие файлы
(omela_bg.py, requirements.txt, README.md, browser_profile/, .git/) не трогает.

Запуск из папки проекта:
    python cleanup.py          # показать, что будет удалено (ничего не удаляет)
    python cleanup.py --yes    # реально удалить

Скрипт работает только в своей собственной папке.
"""

import os
import sys
import glob
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))

# Файлы/папки, которые НИКОГДА не удаляем.
KEEP = {
    "omela_bg.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    "cleanup.py",
    "browser_profile",
    ".git",
}

# Шаблоны мусора (относительно папки проекта).
JUNK_GLOBS = [
    "debug_*.png",        # отладочные скриншоты распознавания
    "page_full*.png",     # полные скриншоты страницы
    "page_region*.png",
    "page_dom_*.html",    # дампы DOM
    "screen_full*.png",
    "screen_region*.png",
    "omela.png",          # старый образец
    "*.log",              # herb_bot.log, omela_bg.log (создаются заново)
    "herb_bot.py",        # старая версия бота (pyautogui) — заменена omela_bg.py
]

# Отдельные файлы/папки, которые надо удалить, но которые проще перечислить явно.
JUNK_DIRS = [
    "templates",          # папка под шаблоны-картинки; в omela_bg.py не используется
]


def collect():
    files, dirs = [], []

    for pattern in JUNK_GLOBS:
        for path in glob.glob(os.path.join(BASE, pattern)):
            name = os.path.basename(path)
            if name in KEEP:
                continue
            if os.path.isfile(path):
                files.append(path)

    # Все .md кроме README.md (старые инструкции в разных кодировках).
    for path in glob.glob(os.path.join(BASE, "*.md")):
        if os.path.basename(path) != "README.md":
            files.append(path)

    for d in JUNK_DIRS:
        p = os.path.join(BASE, d)
        if os.path.isdir(p):
            dirs.append(p)

    # Убрать дубли, сохранить порядок.
    files = list(dict.fromkeys(files))
    return files, dirs


def human(n):
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} ТБ"


def main():
    do_it = "--yes" in sys.argv or "-y" in sys.argv
    files, dirs = collect()

    if not files and not dirs:
        print("Мусорных файлов не найдено — папка уже чистая.")
        return

    total = 0
    print("Будут удалены:\n")
    for f in sorted(files):
        try:
            size = os.path.getsize(f)
        except OSError:
            size = 0
        total += size
        print(f"  {human(size):>8}  {os.path.relpath(f, BASE)}")
    for d in dirs:
        dsize = 0
        for root, _, fs in os.walk(d):
            for x in fs:
                try:
                    dsize += os.path.getsize(os.path.join(root, x))
                except OSError:
                    pass
        total += dsize
        print(f"  {human(dsize):>8}  {os.path.relpath(d, BASE)}{os.sep}  (папка)")

    print(f"\nИтого освободится: {human(total)}")

    if not do_it:
        print("\nЭто предпросмотр. Чтобы реально удалить, запусти:")
        print("    python cleanup.py --yes")
        return

    print("\nУдаляю...")
    removed = 0
    for f in files:
        try:
            os.remove(f)
            removed += 1
        except OSError as e:
            print(f"  ! не удалось удалить {f}: {e}")
    for d in dirs:
        try:
            shutil.rmtree(d)
            removed += 1
        except OSError as e:
            print(f"  ! не удалось удалить {d}: {e}")

    print(f"Готово. Удалено объектов: {removed}. Освобождено ~{human(total)}.")
    print("Можешь удалить и сам cleanup.py, если он больше не нужен.")


if __name__ == "__main__":
    main()