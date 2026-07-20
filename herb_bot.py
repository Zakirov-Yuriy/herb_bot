"""
herb_bot.py — помощник травника для браузерной игры (dwar / Легенда).
Автосбор ОМЕЛЫ.

Как работает:
  1) Делает скриншот игровой карты.
  2) Находит омелу по ЦВЕТУ И ФОРМЕ — плотные ярко-жёлто-зелёные «пёстрые»
     пятнышки (шаблоны-картинки не нужны; так бот не кликает по мобах/подписях).
  3) По каждому кустику делает ДВОЙНОЙ клик — омела начинает собираться.

⚠️  ВАЖНО. Автоматизация нарушает правила почти всех онлайн-игр и может
привести к блокировке аккаунта. Ты используешь скрипт на свой риск.

Аварийная остановка:
  • Резко уведи курсор мыши в ЛЕВЫЙ ВЕРХНИЙ угол экрана (failsafe pyautogui).
  • Или Ctrl+C в терминале.

Режимы:
  python herb_bot.py --check    # показать размеры экрана (проверка масштаба)
  python herb_bot.py --grab     # сохранить скриншоты (настройка области)
  python herb_bot.py --debug    # показать, что бот считает омелой, БЕЗ кликов
  python herb_bot.py            # рабочий режим (собирает омелу)
"""

import sys
import time
import random
import argparse
import logging
import warnings

# --- ВАЖНО для Windows: сделать процесс DPI-aware, иначе при масштабе экрана
#     125%/150% клики промахиваются мимо цели. Должно идти ДО импорта pyautogui. ---
try:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # Windows 8.1+
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()        # запасной вариант
except Exception:
    pass

import numpy as np
import cv2
import mss
import pyautogui

warnings.filterwarnings("ignore", category=DeprecationWarning)


# =========================================================================
#                              НАСТРОЙКИ
# =========================================================================

# --- Область экрана с игровой КАРТОЙ (только зелёная карта, без панели и чата) ---
# Проверь через --grab (screen_region.png). Значения под игру на весь экран 1920x1080.
GAME_REGION = {"left": 197, "top": 275, "width": 1700, "height": 435}

# --- Как собирается омела ---
#   "double" — двойной клик прямо по кустику (для dwar это оно).
#   "click"  — одиночный клик.
#   "button" — клик по кустику + клик по кнопке ACTION_BUTTON.
GATHER_MODE = "double"
ACTION_BUTTON = (700, 254)   # используется только при GATHER_MODE = "button"

# --- Распознавание омелы (по цвету) ---
OMELA_HSV_LOW  = (22, 120, 175)
OMELA_HSV_HIGH = (45, 255, 255)
BLOB_MIN_AREA = 40
BLOB_SIZE_MIN = 8
BLOB_SIZE_MAX = 40
BLOB_ASPECT   = (0.45, 2.2)

# Если в одном скане вдруг «нашлось» подозрительно много пятен — это, скорее
# всего, жёлтая вспышка-анимация во время сбора. Такой цикл пропускаем.
MAX_PER_CYCLE = 30

# --- Задержки (секунды) ---
DOUBLECLICK_GAP = (0.08, 0.16)  # интервал между двумя кликами двойного клика
CLICK_TO_BUTTON = (0.3, 0.7)    # для режима "button"
GATHER_WAIT     = (2.5, 4.5)    # ожидание, пока идёт сбор
BETWEEN_HERBS   = (0.6, 1.6)    # пауза между кустиками
CYCLE_PAUSE     = (2.0, 4.0)    # пауза между сканами
LONG_BREAK_EVERY = (15, 30)
LONG_BREAK       = (20.0, 60.0)

MATCH_MIN_DISTANCE = 25
MAX_RUNTIME_MIN = 60
START_DELAY = 5

# =========================================================================


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("herb_bot.log", encoding="utf-8")],
)
log = logging.getLogger("herb_bot")


def screen_sizes():
    with mss.mss() as sct:
        mon = sct.monitors[1]
    pag = pyautogui.size()
    return (mon["width"], mon["height"]), (pag.width, pag.height)


def log_sizes():
    (mw, mh), (pw, ph) = screen_sizes()
    log.info("Размер экрана: скриншот %dx%d, клики %dx%d", mw, mh, pw, ph)
    if (mw, mh) != (pw, ph):
        log.warning("РАЗМЕРЫ НЕ СОВПАДАЮТ! Из-за масштаба Windows клики будут "
                    "промахиваться. Поставь масштаб дисплея 100%% "
                    "(Параметры → Система → Дисплей → Масштаб = 100%%) и перезапусти.")


def grab_region(region):
    with mss.mss() as sct:
        img = np.array(sct.grab(region))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def grab_fullscreen():
    with mss.mss() as sct:
        mon = sct.monitors[1]
        img = np.array(sct.grab(mon))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), mon


def find_omela(screen_bgr):
    """Центры омелы (x, y) в координатах ОБЛАСТИ."""
    hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
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
        aspect = w / max(h, 1)
        if not (BLOB_ASPECT[0] <= aspect <= BLOB_ASPECT[1]):
            continue
        cx, cy = int(cent[i][0]), int(cent[i][1])
        if all((cx - px) ** 2 + (cy - py) ** 2 >= MATCH_MIN_DISTANCE ** 2
               for px, py in centers):
            centers.append((cx, cy))
    return centers


def move_to(sx, sy):
    pyautogui.moveTo(sx + random.randint(-3, 3), sy + random.randint(-3, 3),
                     duration=random.uniform(0.15, 0.45))
    time.sleep(random.uniform(0.05, 0.15))


def region_to_screen(cx, cy):
    return GAME_REGION["left"] + cx, GAME_REGION["top"] + cy


def gather_at(sx, sy):
    move_to(sx, sy)
    if GATHER_MODE == "double":
        pyautogui.click()
        time.sleep(random.uniform(*DOUBLECLICK_GAP))
        pyautogui.click()
    elif GATHER_MODE == "button":
        pyautogui.click()
        if ACTION_BUTTON is not None:
            time.sleep(random.uniform(*CLICK_TO_BUTTON))
            pyautogui.moveTo(*ACTION_BUTTON, duration=random.uniform(0.15, 0.4))
            pyautogui.click()
    else:  # "click"
        pyautogui.click()
    time.sleep(random.uniform(*GATHER_WAIT))


def mode_check():
    log_sizes()


def mode_grab():
    log_sizes()
    full, mon = grab_fullscreen()
    cv2.imwrite("screen_full.png", full)
    cv2.imwrite("screen_region.png", grab_region(GAME_REGION))
    log.info("Сохранены screen_full.png и screen_region.png. Проверь, что в "
             "screen_region.png попала только зелёная карта. Иначе поправь GAME_REGION.")


def mode_debug():
    log_sizes()
    screen = grab_region(GAME_REGION)
    pts = find_omela(screen)
    vis = screen.copy()
    for (cx, cy) in pts:
        cv2.circle(vis, (cx, cy), 16, (0, 0, 255), 2)
        cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
    cv2.imwrite("debug_omela.png", vis)
    log.info("Найдено кустиков омелы: %d. Разметка — в debug_omela.png", len(pts))


def mode_run():
    log_sizes()
    log.info("Старт через %d сек. Переключись в окно игры (карта видна целиком). "
             "Стоп: курсор в левый верхний угол экрана или Ctrl+C.", START_DELAY)
    time.sleep(START_DELAY)

    started = time.time()
    cycle = 0
    next_long = random.randint(*LONG_BREAK_EVERY)
    total = 0

    while True:
        if (time.time() - started) / 60.0 >= MAX_RUNTIME_MIN:
            log.info("Достигнут лимит времени (%d мин). Стоп.", MAX_RUNTIME_MIN)
            break

        cycle += 1
        pts = find_omela(grab_region(GAME_REGION))

        if len(pts) > MAX_PER_CYCLE:
            log.info("Цикл #%d: подозрительно много пятен (%d) — похоже на "
                     "анимацию сбора, пропускаю.", cycle, len(pts))
            time.sleep(random.uniform(*CYCLE_PAUSE))
            continue

        if pts:
            log.info("Цикл #%d: вижу омелы — %d шт.", cycle, len(pts))
        else:
            log.info("Цикл #%d: омелы не видно, жду.", cycle)

        for (cx, cy) in pts:
            sx, sy = region_to_screen(cx, cy)
            gather_at(sx, sy)
            total += 1
            log.info("Собрал кустик (всего за сессию: %d)", total)
            time.sleep(random.uniform(*BETWEEN_HERBS))

        time.sleep(random.uniform(*CYCLE_PAUSE))

        if cycle >= next_long:
            pause = random.uniform(*LONG_BREAK)
            log.info("Длинный перерыв ~%.0f сек.", pause)
            time.sleep(pause)
            cycle = 0
            next_long = random.randint(*LONG_BREAK_EVERY)


def main():
    ap = argparse.ArgumentParser(description="Автосбор омелы.")
    ap.add_argument("--check", action="store_true", help="показать размеры экрана")
    ap.add_argument("--grab", action="store_true", help="сохранить скриншоты")
    ap.add_argument("--debug", action="store_true", help="распознавание без кликов")
    args = ap.parse_args()

    if args.check:
        mode_check(); return
    if args.grab:
        mode_grab(); return
    if args.debug:
        mode_debug(); return
    try:
        mode_run()
    except pyautogui.FailSafeException:
        log.info("Аварийный стоп (курсор в углу экрана).")
    except KeyboardInterrupt:
        log.info("Остановлено (Ctrl+C).")


if __name__ == "__main__":
    main()
