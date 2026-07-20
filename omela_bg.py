"""
omela_bg.py — фоновый автосбор ОМЕЛЫ для браузерной игры dwar / Легенда.

Работает так: открывает СВОЁ игровое окно (Playwright) и держит ОДНУ сессию.
Ты входишь в игру прямо в этом окне, встаёшь на локацию с омелой, жмёшь ENTER —
и бот начинает собирать омелу двойным кликом ВНУТРИ страницы. Окно можно свернуть
и работать за компьютером параллельно.

⚠️  ВАЖНО:
  • Игра пускает аккаунт только в ОДНУ сессию. Перед запуском ПОЛНОСТЬЮ ЗАКРОЙ
    свой обычный Chrome с игрой, иначе будет ошибка «Пользователь восстанавливается».
  • Автоматизация нарушает правила игры и может привести к бану. На свой риск.

Режимы:
  python omela_bg.py --login   # только войти в игру (сохранить сессию)
  python omela_bg.py --debug   # войти, встать на омелу, ENTER — сохранит скриншот и
                               #   структуру страницы (для настройки; пришли их мне)
  python omela_bg.py           # рабочий режим: войти, встать на омелу, ENTER — сбор

Остановка сбора: Ctrl+C в терминале.
"""

import os
import sys
import time
import random
import argparse
import logging

import numpy as np
import cv2
from playwright.sync_api import sync_playwright


# =========================================================================
#                              НАСТРОЙКИ
# =========================================================================

URL = "https://w1.dwar.ru/main.php"

BASE = os.path.dirname(os.path.abspath(__file__))
USER_DATA = os.path.join(BASE, "browser_profile")

VIEWPORT = {"width": 1600, "height": 900}

# Область КАРТЫ внутри окна (в пикселях страницы). Настроено по скриншоту:
# только зелёная карта, без верхней панели «собрать/Омела» и без правых кнопок.
MAP_REGION = {"left": 80, "top": 150, "width": 1460, "height": 440}

# Распознавание омелы (по цвету)
OMELA_HSV_LOW  = (22, 120, 175)
OMELA_HSV_HIGH = (45, 255, 255)
BLOB_MIN_AREA = 40
BLOB_SIZE_MIN = 8
BLOB_SIZE_MAX = 40
BLOB_ASPECT   = (0.45, 2.2)
MATCH_MIN_DISTANCE = 25
MAX_PER_CYCLE = 30

# Задержки (секунды)
DOUBLECLICK_GAP = (0.08, 0.16)
GATHER_WAIT     = (2.5, 4.5)
BETWEEN_HERBS   = (0.6, 1.6)
CYCLE_PAUSE     = (2.0, 4.0)
LONG_BREAK_EVERY = (15, 30)
LONG_BREAK       = (20.0, 60.0)
MAX_RUNTIME_MIN  = 120

# =========================================================================


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("omela_bg.log", encoding="utf-8")],
)
log = logging.getLogger("omela_bg")


def open_context(p):
    ctx = p.chromium.launch_persistent_context(
        USER_DATA,
        headless=False,
        viewport=VIEWPORT,
        device_scale_factor=1,
        args=["--disable-blink-features=AutomationControlled"],
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    return ctx, page


def screenshot_bgr(page):
    png = page.screenshot(type="png")
    arr = np.frombuffer(png, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def find_omela(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(OMELA_HSV_LOW), np.array(OMELA_HSV_HIGH))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, _, stats, cent = cv2.connectedComponentsWithStats(closed, 8)
    centers = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < BLOB_MIN_AREA:
            continue
        if not (BLOB_SIZE_MIN <= w <= BLOB_SIZE_MAX and BLOB_SIZE_MIN <= h <= BLOB_SIZE_MAX):
            continue
        if not (BLOB_ASPECT[0] <= w / max(h, 1) <= BLOB_ASPECT[1]):
            continue
        cx, cy = int(cent[i][0]), int(cent[i][1])
        if all((cx - px) ** 2 + (cy - py) ** 2 >= MATCH_MIN_DISTANCE ** 2
               for px, py in centers):
            centers.append((cx, cy))
    return centers


def crop_map(full_bgr):
    m = MAP_REGION
    return full_bgr[m["top"]:m["top"] + m["height"], m["left"]:m["left"] + m["width"]]


def map_to_page(cx, cy):
    return MAP_REGION["left"] + cx, MAP_REGION["top"] + cy


def double_click(page, x, y):
    x += random.randint(-3, 3)
    y += random.randint(-3, 3)
    page.mouse.move(x, y)
    time.sleep(random.uniform(0.05, 0.15))
    page.mouse.click(x, y)
    time.sleep(random.uniform(*DOUBLECLICK_GAP))
    page.mouse.click(x, y)


def open_and_wait(p, prompt):
    """Открыть окно, дать войти в игру вручную, дождаться ENTER. Возвращает (ctx, page)."""
    ctx, page = open_context(p)
    try:
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    except Exception as e:
        log.warning("Страница открылась с задержкой/ошибкой (%s). Это ок — работай в окне.", e)
    log.info("Окно игры открыто.")
    input("\n>>> " + prompt + "\n>>> Когда готово — нажми ENTER здесь <<<\n")
    return ctx, page


def mode_login():
    with sync_playwright() as p:
        ctx, _ = open_and_wait(p, "Войди в игру (не через Google!).")
        ctx.close()
    log.info("Сессия сохранена в browser_profile.")


def mode_debug():
    with sync_playwright() as p:
        ctx, page = open_and_wait(
            p, "Войди в игру и встань на локацию с омелой.")
        stamp = time.strftime("%H%M%S")
        full = screenshot_bgr(page)
        cv2.imwrite("page_full_%s.png" % stamp, full)
        try:
            with open("page_dom_%s.html" % stamp, "w", encoding="utf-8") as f:
                f.write(page.content())
        except Exception as e:
            log.warning("DOM не сохранён: %s", e)
        pts = find_omela(crop_map(full))
        try:
            title = page.title()
        except Exception:
            title = "?"
        log.info("URL: %s | заголовок: %s", page.url, title)
        log.info("Скриншот %dx%d, найдено 'омелы': %d.", full.shape[1], full.shape[0], len(pts))
        log.info("Отметка времени %s. Пришли мне page_full_%s.png и page_dom_%s.html "
                 "(или напиши 'готово').", stamp, stamp, stamp)
        ctx.close()


def mode_run():
    with sync_playwright() as p:
        ctx, page = open_and_wait(
            p, "Войди в игру и встань на локацию с омелой. После ENTER начнётся сбор "
               "(окно можно свернуть).")
        log.info("Старт сбора. Стоп: Ctrl+C.")
        started = time.time()
        cycle = 0
        next_long = random.randint(*LONG_BREAK_EVERY)
        total = 0
        try:
            while True:
                if (time.time() - started) / 60.0 >= MAX_RUNTIME_MIN:
                    log.info("Лимит времени (%d мин). Стоп.", MAX_RUNTIME_MIN)
                    break
                cycle += 1
                pts = find_omela(crop_map(screenshot_bgr(page)))
                if len(pts) > MAX_PER_CYCLE:
                    log.info("Цикл #%d: слишком много пятен (%d) — вероятно анимация, пропускаю.",
                             cycle, len(pts))
                    time.sleep(random.uniform(*CYCLE_PAUSE))
                    continue
                log.info("Цикл #%d: вижу омелы — %d шт.", cycle, len(pts))
                for (cx, cy) in pts:
                    px, py = map_to_page(cx, cy)
                    double_click(page, px, py)
                    total += 1
                    log.info("Собрал кустик (всего: %d)", total)
                    time.sleep(random.uniform(*GATHER_WAIT))
                    time.sleep(random.uniform(*BETWEEN_HERBS))
                time.sleep(random.uniform(*CYCLE_PAUSE))
                if cycle >= next_long:
                    pause = random.uniform(*LONG_BREAK)
                    log.info("Длинный перерыв ~%.0f сек.", pause)
                    time.sleep(pause)
                    cycle = 0
                    next_long = random.randint(*LONG_BREAK_EVERY)
        except KeyboardInterrupt:
            log.info("Остановлено (Ctrl+C).")
        finally:
            ctx.close()


def main():
    ap = argparse.ArgumentParser(description="Фоновый автосбор омелы (Playwright).")
    ap.add_argument("--login", action="store_true", help="только войти (сохранить сессию)")
    ap.add_argument("--debug", action="store_true", help="сохранить скриншот и структуру страницы")
    args = ap.parse_args()
    if args.login:
        mode_login()
    elif args.debug:
        mode_debug()
    else:
        mode_run()


if __name__ == "__main__":
    main()