"""
omela_bg.py — фоновый автосбор РЕСУРСОВ + авто-бой для браузерной игры dwar / Легенда.

Собирает ресурс ВЫБРАННОЙ профессии:
  • травник — омела/травы (как было, по умолчанию);
  • геолог  — драгоценные камни/руда (несколько цветов);
  • рыбак   — рыба (пресет-заготовка).
Профессия: флаг  --prof geolog  или Этап 0 мастера --calib. Цвет ресурса берётся
из пресета профессии либо снимается «пипеткой» в --calib (надёжнее для камней).

Что умеет:
  • Сбор ресурса кликом внутри страницы (окно можно свернуть). Добыча запускается
    двойным кликом; бот НЕ тыкает повторно тот же куст, пока идёт добыча
    (иначе игра отменяет добычу).
  • Прокрутка карты вверх/вниз — собирать ресурсы за пределами видимой части.
  • Авто-бой при нападении монстра: раунды «блок + атака», после победы «выход» и
    «В охоту». Боевые кнопки в canvas → кликаются по пикселям (мастер --calib).
  • Авто-закрытие окон-ошибок («Добыча не удалась», «Объект уже не существует!»,
    «нет профессии») по ОТКАЛИБРОВАННОЙ точке «закрыть» (окна в canvas, не HTML).
  • «Чёрный список» чужих ресурсов: клик привёл к ошибке → точка запоминается и
    какое-то время не кликается.

Режимы:
  python omela_bg.py --login          # только войти в игру (сохранить сессию)
  python omela_bg.py --prof geolog    # выбрать профессию и сохранить (потом можно --calib)
  python omela_bg.py --calib          # МАСТЕР: профессия/пипетка, карта, добыча, «закрыть», бой
  python omela_bg.py --debug          # скриншот + DOM + слепок боя (для диагностики)
  python omela_bg.py                  # рабочий режим

Остановка: Ctrl+C в терминале.

⚠️  Игра пускает аккаунт в ОДНУ сессию — закрой обычный Chrome с игрой перед запуском.
    Автоматизация нарушает правила игры и может привести к бану. На свой риск.
"""

import os
import sys
import json
import time
import random
import argparse
import logging
import threading

import numpy as np
import cv2
from playwright.sync_api import sync_playwright


# =========================================================================
#                              НАСТРОЙКИ
# =========================================================================

URL = "https://w1.dwar.ru/main.php"

BASE = os.path.dirname(os.path.abspath(__file__))
USER_DATA = os.path.join(BASE, "browser_profile")
ZONES_FILE = os.path.join(BASE, "fight_zones.json")   # сюда мастер --calib пишет все точки

VIEWPORT = {"width": 1600, "height": 900}

# Область КАРТЫ внутри окна (можно переопределить мастером --calib → ключ map_region).
MAP_REGION = {"left": 80, "top": 150, "width": 1460, "height": 440}

# =========================================================================
#                        ПРОФЕССИИ И РЕСУРСЫ
# =========================================================================
# Бот собирает ресурс НУЖНОЙ профессии. Ресурс распознаётся по ОДНОМУ или
# НЕСКОЛЬКИМ диапазонам цвета (HSV) + по форме пятна (размер/пропорции).
#
#   • травник (herbalist) — омела/травы: один жёлто-зелёный диапазон.
#   • геолог  (geolog)    — драгоценные камни: РАЗНЫЕ яркие цвета, поэтому
#     несколько диапазонов (красный, оранж., жёлтый, зелёный, циан, синий,
#     фиолетовый). Точнее всего — снять цвет камней «пипеткой» в --calib.
#   • рыбак   (fisher)    — рыба: пресет-заготовка (уточняется пипеткой).
#
# Профессия выбирается флагом  --prof geolog  или в мастере --calib (Этап 0),
# и сохраняется в fight_zones.json (ключи "profession", "resource_ranges").
# «Пипетка» из калибровки пишет свои диапазоны в "resource_ranges" и имеет
# приоритет над пресетом профессии.

# Диапазон HSV задаётся как [ [Hlow,Slow,Vlow], [Hhigh,Shigh,Vhigh] ].
# OpenCV: H = 0..179, S = 0..255, V = 0..255.
PROFESSIONS = {
    # Омела — как было (жёлто-зелёный куст).
    "herbalist": {
        "title": "Травник (омела/травы)",
        "ranges": [[[22, 120, 175], [45, 255, 255]]],
        "blob": {"min_area": 40, "size_min": 8, "size_max": 40,
                 "aspect": [0.45, 2.2]},
    },
    # Драгоценные камни — яркие, насыщенные, компактные пятна разных цветов.
    # Зелёный диапазон НАМЕРЕННО сужен (S/V высокие), чтобы не ловить траву.
    "geolog": {
        "title": "Геолог (драгоценные камни/руда)",
        "ranges": [
            [[0,   130, 130], [10,  255, 255]],   # красный (низ)
            [[168, 130, 130], [179, 255, 255]],   # красный (верх)
            [[11,  130, 150], [22,  255, 255]],   # оранжевый / янтарь
            [[23,  140, 175], [33,  255, 255]],   # жёлтый / топаз
            [[70,  150, 170], [92,  255, 255]],   # изумруд (ярче травы)
            [[92,  110, 160], [104, 255, 255]],   # циан / аквамарин
            [[105, 120, 140], [128, 255, 255]],   # синий / сапфир
            [[129, 100, 140], [160, 255, 255]],   # фиолетовый / аметист / розовый
        ],
        "blob": {"min_area": 22, "size_min": 6, "size_max": 40,
                 "aspect": [0.4, 2.5]},
    },
    # Рыба — заготовка. Лучше снять цвет пипеткой в --calib.
    "fisher": {
        "title": "Рыбак (рыба)",
        "ranges": [
            [[90,  90, 150], [128, 255, 255]],    # серебристо-синие рыбы
            [[15,  90, 150], [35,  255, 255]],    # золотистые рыбы
        ],
        "blob": {"min_area": 30, "size_min": 7, "size_max": 48,
                 "aspect": [0.35, 3.0]},
    },
}
DEFAULT_PROFESSION = "herbalist"

# Активные параметры распознавания (заполняются apply_saved_config / --prof).
# По умолчанию — омела, чтобы поведение «из коробки» не менялось.
ACTIVE_PROF    = DEFAULT_PROFESSION
RESOURCE_RANGES = [[np.array(lo, np.uint8), np.array(hi, np.uint8)]
                   for lo, hi in PROFESSIONS[DEFAULT_PROFESSION]["ranges"]]
_b = PROFESSIONS[DEFAULT_PROFESSION]["blob"]
BLOB_MIN_AREA = _b["min_area"]
BLOB_SIZE_MIN = _b["size_min"]
BLOB_SIZE_MAX = _b["size_max"]
BLOB_ASPECT   = tuple(_b["aspect"])

MATCH_MIN_DISTANCE = 25
MAX_PER_CYCLE = 30

# Добыча
GATHER_CLICKS   = 2            # сколько кликов запускают добычу (в этой игре — двойной)
DOUBLECLICK_GAP = (0.08, 0.16)
GATHER_WAIT     = (2.5, 4.5)
BETWEEN_HERBS   = (0.6, 1.6)
CYCLE_PAUSE     = (2.0, 4.0)
LONG_BREAK_EVERY = (15, 30)
LONG_BREAK       = (20.0, 60.0)
MAX_RUNTIME_MIN  = 120

# Не тыкать повторно куст, который только что начали добывать (иначе добыча отменится)
RECLICK_COOLDOWN = 9.0         # сек: столько не кликаем по той же точке снова
RECLICK_RADIUS   = 26          # px

# Пропуск чужих ресурсов (чёрный список): клик привёл к окну ошибки → запомнить.
SKIP_FAILED_ENABLED = True
SKIP_FAILED_RADIUS  = 22
SKIP_FAILED_TTL     = 300.0

# Прокрутка карты
MAP_SCROLL_ENABLED   = True
MAP_SCROLL_POSITIONS = 3
MAP_SCROLL_DELTA     = 320

# ---- ОКНО «закрыть» -----------------------------------------------------
# Окна-ошибки нарисованы в игровом canvas (не HTML). Надёжный способ — кликать по
# ОТКАЛИБРОВАННОЙ точке кнопки «закрыть», но ТОЛЬКО когда она сейчас красная
# (значит окно открыто). Точка снимается мастером --calib (ключ "close").
POPUP_CLOSE_TARGET = None      # (x, y) или None → берётся из fight_zones.json
POPUP_RED_FRAC     = 0.25      # какая доля пикселей вокруг точки должна быть красной
# ВАЖНО: во время добычи открывается окно «Добыча» с кнопкой «отменить» и ЗЕЛЁНОЙ
# полосой прогресса. Окно-ошибка («закрыть») — БЕЗ зелёной полосы. Обе кнопки в
# одном месте и обе красные. Отличаем по зелёной полосе прогресса НАД кнопкой:
#   • есть зелёная полоса → идёт добыча → НЕ трогаем (иначе отменим!);
#   • красная кнопка есть, а зелёной полосы нет → это ошибка → жмём «закрыть».
# Ищем САМУ полосу прогресса — широкую зелёную ГОРИЗОНТАЛЬНУЮ полосу в центре окна,
# не привязываясь к кнопке (у окна-прогресса и окна-ошибки раскладка разная).
PROGRESS_GREEN_LOW  = (33, 80, 105)  # зелёный полосы (HSV; трава темнее — не ловится)
PROGRESS_GREEN_HIGH = (92, 255, 255)
PROGRESS_BAR_W      = (45, 360)      # ширина полосы (px; > куста омелы, чтобы не путать)
PROGRESS_BAR_H      = (6, 34)        # высота полосы (px)
PROGRESS_BAR_ASPECT = 3.0            # ширина/высота ≥ этого (горизонтальная полоса)
PROGRESS_BAR_FILL   = 0.55           # плотность заливки прямоугольника (0..1)
PROGRESS_REGION_HW  = 300            # полуширина зоны поиска от центра карты (px)
PROGRESS_REGION_HH  = 150            # полувысота зоны поиска от центра карты (px)
# Ожидание завершения добычи (пока висит окно прогресса)
GATHER_POLL       = 1.0              # как часто проверять (сек)
GATHER_MAX_WAIT   = 25.0             # максимум ждать одну добычу (сек)
POPUP_RED_LOW1  = (0, 70, 40)
POPUP_RED_HIGH1 = (14, 255, 235)
POPUP_RED_LOW2  = (166, 70, 40)
POPUP_RED_HIGH2 = (180, 255, 235)
# Запасной авто-поиск окна по картинке. По умолчанию ВЫКЛ — может ложно кликать по
# карте и отменять добычу. Включай только если не хочешь калибровать точку «закрыть».
POPUP_USE_GEOMETRY = False
POPUP_TITLE_W   = (190, 440)
POPUP_BUTTON_W  = (60, 210)
POPUP_BAR_H     = (12, 48)
POPUP_FILL_MIN  = 0.45
POPUP_SEARCH_X  = (440, 1160)
POPUP_SEARCH_Y  = (230, 660)

# ---- БОЙ ----------------------------------------------------------------
FIGHT_ENABLED = True
FIGHT_UI_MARKERS = ["Введите ник цели", "Показать жизнь", "ПОКАЗАТЬ УБИТЫХ", "Показать убитых"]
FIGHT_BLOCK_TARGETS  = []
FIGHT_ATTACK_TARGETS = []
FIGHT_EXIT_TARGET    = None
FIGHT_HUNT_TARGET    = None
FIGHT_ROUND_WAIT = (1.6, 3.0)
FIGHT_MAX_ROUNDS = 60
FIGHT_POLL_AFTER_GATHER = True

# =========================================================================


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("omela_bg.log", encoding="utf-8")],
)
log = logging.getLogger("omela_bg")


def load_zones():
    try:
        with open(ZONES_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_zones(zones):
    with open(ZONES_FILE, "w", encoding="utf-8") as f:
        json.dump(zones, f, ensure_ascii=False, indent=2)


def _ranges_to_np(ranges):
    """[[[h,s,v],[h,s,v]], ...] → [[np.array(lo), np.array(hi)], ...]."""
    out = []
    for pair in ranges or []:
        try:
            lo, hi = pair
            out.append([np.array(lo, np.uint8), np.array(hi, np.uint8)])
        except Exception:
            continue
    return out


def set_active_profession(name, custom_ranges=None, custom_blob=None):
    """Сделать профессию `name` активной: задать диапазоны цвета и форму пятна.

    Приоритет диапазонов: custom_ranges (пипетка из калибровки) → пресет профессии.
    Форма пятна: custom_blob → blob профессии.
    """
    global ACTIVE_PROF, RESOURCE_RANGES
    global BLOB_MIN_AREA, BLOB_SIZE_MIN, BLOB_SIZE_MAX, BLOB_ASPECT
    prof = PROFESSIONS.get(name)
    if prof is None:
        log.warning("Профессия '%s' неизвестна. Доступны: %s. Оставляю '%s'.",
                    name, ", ".join(PROFESSIONS), ACTIVE_PROF)
        return
    ACTIVE_PROF = name
    if custom_ranges:
        RESOURCE_RANGES = _ranges_to_np(custom_ranges)
        src = "пипетка (%d диап.)" % len(RESOURCE_RANGES)
    else:
        RESOURCE_RANGES = _ranges_to_np(prof["ranges"])
        src = "пресет (%d диап.)" % len(RESOURCE_RANGES)
    b = dict(prof["blob"])
    if custom_blob:
        b.update(custom_blob)
    BLOB_MIN_AREA = b["min_area"]
    BLOB_SIZE_MIN = b["size_min"]
    BLOB_SIZE_MAX = b["size_max"]
    BLOB_ASPECT   = tuple(b["aspect"])
    log.info("Профессия: %s [%s] — цвет: %s.", name, prof["title"], src)


def apply_saved_config():
    """Подтянуть в глобальные настройки то, что снято мастером --calib."""
    global MAP_REGION, GATHER_CLICKS
    z = load_zones() or {}
    # профессия + цвет ресурса (пипетка имеет приоритет над пресетом)
    prof_name = z.get("profession", ACTIVE_PROF)
    set_active_profession(prof_name,
                          custom_ranges=z.get("resource_ranges"),
                          custom_blob=z.get("resource_blob"))
    mr = z.get("map_region")
    if mr and len(mr) == 4:
        MAP_REGION = {"left": int(mr[0]), "top": int(mr[1]),
                      "width": int(mr[2]), "height": int(mr[3])}
        log.info("Область карты из калибровки: %s", MAP_REGION)
    gc = z.get("gather_clicks")
    if gc in (1, 2, 3):
        GATHER_CLICKS = int(gc)
    log.info("Добыча: %d клик(а/ов) на ресурс.", GATHER_CLICKS)


def open_context(p):
    launch_kwargs = dict(
        user_data_dir=USER_DATA, headless=False, viewport=VIEWPORT, device_scale_factor=1,
        args=["--disable-blink-features=AutomationControlled", "--no-first-run",
              "--no-default-browser-check"],
        ignore_default_args=["--enable-automation"],
    )
    try:
        ctx = p.chromium.launch_persistent_context(channel="chrome", **launch_kwargs)
        log.info("Запущен установленный Google Chrome.")
    except Exception as e:
        log.warning("Chrome не запустился (%s). Использую встроенный Chromium.", e)
        ctx = p.chromium.launch_persistent_context(**launch_kwargs)
    try:
        ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
    except Exception:
        pass
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    return ctx, page


def screenshot_bgr(page):
    png = page.screenshot(type="png")
    arr = np.frombuffer(png, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def find_resource(img_bgr):
    """Найти ресурс активной профессии по цвету (одному/нескольким диапазонам)
    и по форме пятна. Возвращает список центров (cx, cy) внутри кадра карты."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in RESOURCE_RANGES:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    if mask is None:
        return []
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


def sample_hsv_ranges_at(full_bgr, x, y, hw=13, h_pad=10, sv_pad=70):
    """«Пипетка»: снять цвет ресурса вокруг точки (x, y) и построить диапазон(ы) HSV.

    Берём насыщенные/яркие пиксели вокруг клика (фон-траву отбрасываем), считаем
    медианный цвет и строим коридор H±h_pad, S/V — от (медиана−sv_pad) до 255.
    Красный «оборачивается» через 0 → возвращаем ДВА диапазона. Возвращает список
    вида [ [[h,s,v],[h,s,v]], ... ] или [] если не удалось.
    """
    h, w = full_bgr.shape[:2]
    x0, x1 = max(0, int(x) - hw), min(w, int(x) + hw)
    y0, y1 = max(0, int(y) - hw), min(h, int(y) + hw)
    if x1 <= x0 or y1 <= y0:
        return []
    patch = full_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    bright = hsv[(hsv[:, 1] >= 60) & (hsv[:, 2] >= 60)]
    use = bright if len(bright) >= 8 else hsv
    hm = int(np.median(use[:, 0]))
    sm = int(np.median(use[:, 1]))
    vm = int(np.median(use[:, 2]))
    s_lo = max(60, sm - sv_pad)
    v_lo = max(60, vm - sv_pad)
    h_lo, h_hi = hm - h_pad, hm + h_pad
    ranges = []
    if h_lo < 0:                      # красный, обёрнутый через 0
        ranges.append([[0, s_lo, v_lo], [h_hi, 255, 255]])
        ranges.append([[180 + h_lo, s_lo, v_lo], [179, 255, 255]])
    elif h_hi > 179:
        ranges.append([[h_lo, s_lo, v_lo], [179, 255, 255]])
        ranges.append([[0, s_lo, v_lo], [h_hi - 180, 255, 255]])
    else:
        ranges.append([[h_lo, s_lo, v_lo], [h_hi, 255, 255]])
    return ranges


def crop_map(full_bgr):
    m = MAP_REGION
    return full_bgr[m["top"]:m["top"] + m["height"], m["left"]:m["left"] + m["width"]]


def map_to_page(cx, cy):
    return MAP_REGION["left"] + cx, MAP_REGION["top"] + cy


def gather_click(page, x, y):
    """Запустить добычу: GATHER_CLICKS кликов подряд по (x, y) с лёгким разбросом."""
    x += random.randint(-3, 3)
    y += random.randint(-3, 3)
    page.mouse.move(x, y)
    time.sleep(random.uniform(0.05, 0.15))
    page.mouse.click(x, y)
    for _ in range(max(1, GATHER_CLICKS) - 1):
        time.sleep(random.uniform(*DOUBLECLICK_GAP))
        page.mouse.click(x, y)


def click_point(page, xy):
    try:
        x = int(xy[0]) + random.randint(-2, 2)
        y = int(xy[1]) + random.randint(-2, 2)
        page.mouse.move(x, y)
        time.sleep(random.uniform(0.05, 0.12))
        page.mouse.click(x, y)
        return True
    except Exception as e:
        log.warning("Клик по (%s) не удался: %s", xy, e)
        return False


def scroll_map(page, dy):
    cx = MAP_REGION["left"] + MAP_REGION["width"] // 2
    cy = MAP_REGION["top"] + MAP_REGION["height"] // 2
    try:
        page.mouse.move(cx, cy)
        time.sleep(random.uniform(0.1, 0.2))
        page.mouse.wheel(0, dy)
        time.sleep(random.uniform(0.5, 0.9))
        return True
    except Exception as e:
        log.warning("Прокрутка карты не удалась: %s", e)
        return False


def wait_enter_keep_alive(ctx):
    """Ждать ENTER, не «замораживая» браузер."""
    done = threading.Event()

    def _reader():
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        done.set()

    threading.Thread(target=_reader, daemon=True).start()
    while not done.is_set():
        try:
            pg = ctx.pages[0] if ctx.pages else None
            if pg is not None:
                pg.wait_for_timeout(150)
            else:
                time.sleep(0.15)
        except Exception:
            time.sleep(0.15)


def read_line_keep_alive(ctx, prompt):
    """Прочитать строку из терминала, не «замораживая» браузер. Возвращает str."""
    print(prompt, flush=True)
    box = {"line": ""}
    done = threading.Event()

    def _reader():
        try:
            box["line"] = input()
        except (EOFError, KeyboardInterrupt):
            pass
        done.set()

    threading.Thread(target=_reader, daemon=True).start()
    while not done.is_set():
        try:
            pg = ctx.pages[0] if ctx.pages else None
            if pg is not None:
                pg.wait_for_timeout(150)
            else:
                time.sleep(0.15)
        except Exception:
            time.sleep(0.15)
    return box["line"].strip()


def open_and_wait(p, prompt):
    ctx, page = open_context(p)
    try:
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    except Exception as e:
        log.warning("Страница открылась с задержкой/ошибкой (%s). Это ок — работай в окне.", e)
    log.info("Окно игры открыто.")
    print("\n>>> " + prompt + "\n>>> Когда готов(а) — нажми ENTER здесь <<<\n", flush=True)
    wait_enter_keep_alive(ctx)
    return ctx, page


def mode_login():
    with sync_playwright() as p:
        ctx, _ = open_and_wait(p, "Войди в игру — можно через кнопку Google.")
        ctx.close()
    log.info("Сессия сохранена в browser_profile.")


def _all_frames(page):
    try:
        return list(page.frames)
    except Exception:
        return []


def _click_first_visible(loc, limit=5):
    try:
        n = loc.count()
    except Exception:
        return False
    for i in range(min(n, limit)):
        try:
            el = loc.nth(i)
            if el.is_visible():
                el.click(timeout=1000)
                return True
        except Exception:
            continue
    return False


# --- ОКНО «закрыть» -------------------------------------------------------

def get_close_target():
    if POPUP_CLOSE_TARGET:
        return tuple(POPUP_CLOSE_TARGET)
    z = load_zones()
    if z and z.get("close"):
        return tuple(z["close"])
    return None


def _red_fraction_at(full_bgr, x, y, hw=16, hh=10):
    h, w = full_bgr.shape[:2]
    x0, x1 = max(0, int(x) - hw), min(w, int(x) + hw)
    y0, y1 = max(0, int(y) - hh), min(h, int(y) + hh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = full_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(POPUP_RED_LOW1), np.array(POPUP_RED_HIGH1))
    m2 = cv2.inRange(hsv, np.array(POPUP_RED_LOW2), np.array(POPUP_RED_HIGH2))
    mask = cv2.bitwise_or(m1, m2)
    return float((mask > 0).sum()) / float(mask.size)


def _red_rects(full_bgr):
    hsv = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(POPUP_RED_LOW1), np.array(POPUP_RED_HIGH1))
    m2 = cv2.inRange(hsv, np.array(POPUP_RED_LOW2), np.array(POPUP_RED_HIGH2))
    mask = cv2.morphologyEx(cv2.bitwise_or(m1, m2),
                            cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    rects = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx, cy = int(cent[i][0]), int(cent[i][1])
        if not (POPUP_SEARCH_X[0] <= cx <= POPUP_SEARCH_X[1]):
            continue
        if not (POPUP_SEARCH_Y[0] <= cy <= POPUP_SEARCH_Y[1]):
            continue
        if not (POPUP_BAR_H[0] <= h <= POPUP_BAR_H[1]):
            continue
        if area / float(max(w * h, 1)) < POPUP_FILL_MIN:
            continue
        rects.append((cx, cy, int(w), int(h)))
    return rects


def find_popup_close(full_bgr):
    rects = _red_rects(full_bgr)
    titles = [r for r in rects if POPUP_TITLE_W[0] <= r[2] <= POPUP_TITLE_W[1]]
    buttons = [r for r in rects if POPUP_BUTTON_W[0] <= r[2] <= POPUP_BUTTON_W[1]]
    best = None
    for (bx, by, bw, bh) in buttons:
        for (tx, ty, tw, th) in titles:
            if abs(bx - tx) <= 130 and 25 <= (by - ty) <= 170:
                if best is None or by > best[1]:
                    best = (bx, by)
    return best


def _close_via_dom(page):
    closed = False
    for fr in _all_frames(page):
        strategies = []
        try:
            strategies.append(fr.get_by_text("закрыть", exact=False))
        except Exception:
            pass
        try:
            strategies.append(fr.get_by_role("button", name="закрыть", exact=False))
        except Exception:
            pass
        try:
            strategies.append(fr.locator(
                "input[value='закрыть'], input[value='Закрыть'], "
                "a:has-text('закрыть'), button:has-text('закрыть')"))
        except Exception:
            pass
        for loc in strategies:
            if _click_first_visible(loc):
                closed = True
                log.info("Закрыл всплывающее окно (DOM).")
                time.sleep(random.uniform(0.3, 0.6))
                break
    return closed


def _popup_red_now(page):
    """Сейчас в точке «закрыть» красная кнопка? (окно открыто). Нужна калибровка."""
    target = get_close_target()
    if target is None:
        return False
    try:
        full = screenshot_bgr(page)
    except Exception:
        return False
    return _red_fraction_at(full, target[0], target[1]) >= POPUP_RED_FRAC


def progress_bar_present(full_bgr):
    """Есть ли в центре окна широкая зелёная ГОРИЗОНТАЛЬНАЯ полоса прогресса «Добыча»?

    Ищем по всему центру карты (не привязываясь к кнопке). Полоса — это залитый
    зелёный прямоугольник шириной ≥ PROGRESS_BAR_W и вытянутый по горизонтали.
    Трава (темнее и «рваная») такой сплошной полосы не образует.
    """
    cx = MAP_REGION["left"] + MAP_REGION["width"] // 2
    cy = MAP_REGION["top"] + MAP_REGION["height"] // 2
    h, w = full_bgr.shape[:2]
    x0, x1 = max(0, cx - PROGRESS_REGION_HW), min(w, cx + PROGRESS_REGION_HW)
    y0, y1 = max(0, cy - PROGRESS_REGION_HH), min(h, cy + PROGRESS_REGION_HH)
    if x1 <= x0 or y1 <= y0:
        return False
    region = full_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(PROGRESS_GREEN_LOW), np.array(PROGRESS_GREEN_HIGH))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if not (PROGRESS_BAR_W[0] <= ww <= PROGRESS_BAR_W[1]):
            continue
        if not (PROGRESS_BAR_H[0] <= hh <= PROGRESS_BAR_H[1]):
            continue
        if ww / float(max(hh, 1)) < PROGRESS_BAR_ASPECT:
            continue
        if area / float(max(ww * hh, 1)) < PROGRESS_BAR_FILL:
            continue
        return True
    return False


def window_kind(page):
    """Что сейчас в центре: 'progress' (идёт добыча), 'error' (окно с «закрыть»),
    'none' (окна нет) или 'unknown' (нельзя определить)."""
    try:
        full = screenshot_bgr(page)
    except Exception:
        return "unknown"
    # 1) полоса прогресса → идёт добыча (проверяем ПЕРВЫМ, чтобы не отменить)
    if progress_bar_present(full):
        return "progress"
    # 2) красная кнопка в откалиброванной точке без полосы → окно-ошибка
    target = get_close_target()
    if target is not None and _red_fraction_at(full, target[0], target[1]) >= POPUP_RED_FRAC:
        return "error"
    return "none"


def close_if_blocking(page):
    """Закрыть ТОЛЬКО окно-ошибку (с «закрыть»). Окно добычи (с полосой) не трогаем."""
    if window_kind(page) == "error":
        target = get_close_target()
        if target:
            click_point(page, target)
            log.info("Закрыл окно-ошибку («закрыть» по калибровке).")
            time.sleep(random.uniform(0.3, 0.6))
            return True
    return False


def close_blocking_popup(page):
    """Закрыть окно-ошибку. Главное — по откалиброванной точке «закрыть», и только
    если она сейчас красная (окно открыто) — иначе НИКОГДА не кликаем (не мешаем сбору).
    """
    target = get_close_target()
    if target is not None:
        try:
            full = screenshot_bgr(page)
        except Exception:
            full = None
        if full is not None and _red_fraction_at(full, target[0], target[1]) >= POPUP_RED_FRAC:
            click_point(page, target)
            log.info("Закрыл окно (кнопка «закрыть» по калибровке).")
            time.sleep(random.uniform(0.3, 0.6))
            return True
        # точка задана, но не красная → окна нет, ничего не жмём
        return False

    # точка «закрыть» не откалибрована
    if POPUP_USE_GEOMETRY:
        try:
            pt = find_popup_close(screenshot_bgr(page))
        except Exception:
            pt = None
        if pt:
            click_point(page, pt)
            log.info("Закрыл окно «закрыть» по картинке.")
            time.sleep(random.uniform(0.3, 0.6))
            return True
    return _close_via_dom(page)


# --- ЧЁРНЫЙ СПИСОК и КУЛДАУН ПОВТОРНОГО КЛИКА -----------------------------
_failed_points = []   # (scroll_pos, x, y, expire) — чужие ресурсы
_recent_points = []   # (x, y, expire) — недавно начатые добычи (не тыкать снова)


def _prune(lst, now, idx):
    lst[:] = [t for t in lst if t[idx] > now]


def _fp_blacklisted(pos, x, y, now):
    r2 = SKIP_FAILED_RADIUS ** 2
    for (p, bx, by, exp) in _failed_points:
        if p == pos and exp > now and (x - bx) ** 2 + (y - by) ** 2 <= r2:
            return True
    return False


def _fp_add(pos, x, y, now):
    _failed_points.append((pos, x, y, now + SKIP_FAILED_TTL))


def _recent(x, y, now):
    r2 = RECLICK_RADIUS ** 2
    for (bx, by, exp) in _recent_points:
        if exp > now and (x - bx) ** 2 + (y - by) ** 2 <= r2:
            return True
    return False


def _recent_add(x, y, now):
    _recent_points.append((x, y, now + RECLICK_COOLDOWN))


# =========================================================================
#                                  БОЙ
# =========================================================================

def in_fight(page):
    for fr in _all_frames(page):
        for marker in FIGHT_UI_MARKERS:
            try:
                loc = fr.get_by_text(marker, exact=False)
                if loc.count() > 0 and loc.first.is_visible():
                    return True
            except Exception:
                continue
    return False


def _count_text_in_frames(page, needle):
    total = 0
    for fr in _all_frames(page):
        try:
            total += fr.get_by_text(needle, exact=False).count()
        except Exception:
            continue
    return total


def resolve_fight_targets():
    block = list(FIGHT_BLOCK_TARGETS)
    attack = list(FIGHT_ATTACK_TARGETS)
    exit_t = FIGHT_EXIT_TARGET
    hunt_t = FIGHT_HUNT_TARGET
    if not (block or attack or exit_t or hunt_t):
        z = load_zones()
        if z:
            if z.get("block"):
                block = [tuple(z["block"])]
            if z.get("attack"):
                attack = [tuple(z["attack"])]
            if z.get("exit"):
                exit_t = tuple(z["exit"])
            if z.get("hunt"):
                hunt_t = tuple(z["hunt"])
    return block, attack, exit_t, hunt_t


def stats_screen_present(page):
    for fr in _all_frames(page):
        for marker in ("В охоту", "В локацию"):
            try:
                loc = fr.get_by_text(marker, exact=False)
                if loc.count() > 0 and loc.first.is_visible():
                    return True
            except Exception:
                continue
    return False


def return_to_hunt(page, hunt_t=None):
    for attempt in range(6):
        for fr in _all_frames(page):
            try:
                loc = fr.get_by_text("В охоту", exact=False)
                if loc.count() > 0 and loc.first.is_visible():
                    loc.first.click(timeout=1500)
                    log.info("Нажал «В охоту» — возвращаюсь к сбору.")
                    time.sleep(random.uniform(0.8, 1.5))
                    return True
            except Exception:
                continue
        time.sleep(0.5)
    if hunt_t:
        click_point(page, hunt_t)
        time.sleep(random.uniform(0.8, 1.5))
        return True
    log.warning("Не удалось нажать «В охоту».")
    return False


def do_fight(page):
    log.info("⚔️  Обнаружен бой — вступаю в схватку.")
    block, attack, exit_t, hunt_t = resolve_fight_targets()
    if not (block or attack):
        log.warning("Зоны колеса НЕ настроены. Прогони `python omela_bg.py --calib`.")
        waited = 0
        while in_fight(page) and waited < FIGHT_MAX_ROUNDS:
            close_blocking_popup(page)
            time.sleep(random.uniform(*FIGHT_ROUND_WAIT))
            waited += 1
        return
    end_baseline = _count_text_in_frames(page, "Окончен бой")
    rounds = 0
    while in_fight(page) and rounds < FIGHT_MAX_ROUNDS:
        rounds += 1
        if block:
            click_point(page, random.choice(block))
            time.sleep(random.uniform(0.15, 0.35))
        if attack:
            click_point(page, random.choice(attack))
        log.info("Раунд #%d: блок + атака.", rounds)
        time.sleep(random.uniform(*FIGHT_ROUND_WAIT))
        if _count_text_in_frames(page, "Окончен бой") > end_baseline:
            log.info("🏆 Бой окончен (победа). Жму «выход».")
            if exit_t:
                click_point(page, exit_t)
                time.sleep(random.uniform(0.6, 1.2))
            break
    if exit_t and in_fight(page):
        click_point(page, exit_t)
        time.sleep(random.uniform(0.6, 1.2))
    time.sleep(random.uniform(0.4, 0.9))
    if stats_screen_present(page):
        return_to_hunt(page, hunt_t)
    log.info("✅ Бой обработан (раундов: %d). Возвращаюсь к сбору.", rounds)


def dump_fight(page, stamp):
    js = r"""
    () => { const out=[]; const els=document.querySelectorAll('a,button,img,div,span,td,area,input,canvas');
      for (const el of els){ const r=el.getBoundingClientRect();
        if (r.width<4||r.height<4) continue; if (r.width>500&&r.height>500) continue;
        const cls=(el.className&&el.className.toString)?el.className.toString().slice(0,80):'';
        const txt=((el.innerText||el.alt||el.title||el.value||'')+'').trim().slice(0,50);
        out.push({tag:el.tagName,id:el.id||'',cls:cls,txt:txt,
          cx:Math.round(r.x+r.width/2),cy:Math.round(r.y+r.height/2),
          w:Math.round(r.width),h:Math.round(r.height)}); } return out; } """
    frames_data = []
    for fr in _all_frames(page):
        try:
            frames_data.append({"url": fr.url, "name": fr.name, "elements": fr.evaluate(js)})
        except Exception:
            frames_data.append({"url": "?", "name": "", "elements": []})
    path = "fight_dump_%s.json" % stamp
    try:
        save = {"in_fight": in_fight(page), "frames": frames_data}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(save, f, ensure_ascii=False, indent=2)
        log.info("Слепок боя сохранён: %s", path)
    except Exception as e:
        log.warning("Не сохранил слепок: %s", e)
    return path


def mode_debug():
    with sync_playwright() as p:
        ctx, page = open_and_wait(p, "Войди в игру (можно открыть окно-ошибку или бой).")
        apply_saved_config()
        stamp = time.strftime("%H%M%S")
        full = screenshot_bgr(page)
        cv2.imwrite("page_full_%s.png" % stamp, full)
        try:
            with open("page_dom_%s.html" % stamp, "w", encoding="utf-8") as f:
                f.write(page.content())
        except Exception:
            pass
        tgt = get_close_target()
        if tgt:
            log.info("Доля красного в точке «закрыть» %s = %.2f (окно открыто, если > %.2f).",
                     tgt, _red_fraction_at(full, tgt[0], tgt[1]), POPUP_RED_FRAC)
        log.info("Профессия: %s (%s). Диапазонов цвета: %d.",
                 ACTIVE_PROF, PROFESSIONS[ACTIVE_PROF]["title"], len(RESOURCE_RANGES))
        log.info("Найдено ресурсов (%s): %d. Файлы page_full_%s.png / page_dom_%s.html.",
                 ACTIVE_PROF, len(find_resource(crop_map(full))), stamp, stamp)
        dump_fight(page, stamp)
        ctx.close()


# =========================================================================
#                           МАСТЕР КАЛИБРОВКИ
# =========================================================================

_CALIB_JS = r"""
(() => {
  if (window.__calibInstalled) return;
  window.__calibInstalled = true;
  document.addEventListener('click', function(e){
    let x=e.clientX, y=e.clientY, w=window, g=0;
    try { while (w.frameElement && g++<10){ const r=w.frameElement.getBoundingClientRect();
      x+=r.left; y+=r.top; w=w.parent; } } catch(err){}
    console.log('CALIB '+Math.round(x)+' '+Math.round(y));
  }, true);
})();
"""


def _install_calib(page):
    n = 0
    for fr in _all_frames(page):
        try:
            fr.evaluate(_CALIB_JS)
            n += 1
        except Exception:
            continue
    return n


def _capture_point(ctx, page, clicks, label):
    _install_calib(page)
    clicks.clear()
    print("\n>>> Кликни в игре по: %s" % label)
    print(">>> потом нажми ENTER. (Просто ENTER без клика — пропустить.)\n", flush=True)
    wait_enter_keep_alive(ctx)
    if clicks:
        pt = clicks[-1]
        log.info("  → (%d, %d)", pt[0], pt[1])
        return [pt[0], pt[1]]
    log.info("  → пропущено.")
    return None


def mode_calib():
    with sync_playwright() as p:
        ctx, page = open_and_wait(p, "Мастер калибровки. Войди в игру и встань на локацию.")
        clicks = []

        def _on_console(msg):
            try:
                t = msg.text
            except Exception:
                return
            if isinstance(t, str) and t.startswith("CALIB "):
                a = t.split()
                try:
                    clicks.append((int(a[1]), int(a[2])))
                except Exception:
                    pass

        page.on("console", _on_console)
        log.info("Перехват кликов установлен во фреймах: %d.", _install_calib(page))
        zones = load_zones() or {}

        print("\n================ МАСТЕР КАЛИБРОВКИ ================")
        print("Проходи этапы по порядку. Любой этап можно пропустить (ENTER без клика).")
        print("Всё сохраняется в fight_zones.json; можно запускать --calib несколько раз.\n")

        # 0) ПРОФЕССИЯ И ЦВЕТ РЕСУРСА
        print(">>> ЭТАП 0. ПРОФЕССИЯ И ЦВЕТ РЕСУРСА (что собирать).")
        names = list(PROFESSIONS)
        for i, nm in enumerate(names, 1):
            print("   %d) %-10s — %s" % (i, nm, PROFESSIONS[nm]["title"]))
        prev_prof = zones.get("profession", DEFAULT_PROFESSION)
        ans = read_line_keep_alive(
            ctx, ">>> Номер или имя профессии (сейчас: %s; ENTER — оставить): " % prev_prof)
        chosen = prev_prof
        if ans:
            if ans.isdigit() and 1 <= int(ans) <= len(names):
                chosen = names[int(ans) - 1]
            elif ans in PROFESSIONS:
                chosen = ans
            else:
                print("   Не понял '%s' — оставляю '%s'." % (ans, prev_prof))
        zones["profession"] = chosen
        # сменили профессию → старый снятый цвет уже не подходит, сбрасываем
        if chosen != prev_prof and zones.get("resource_ranges"):
            zones.pop("resource_ranges", None)
            log.info("Профессия изменилась — сброшен старый цвет ресурса (пресет %s).", chosen)
        log.info("Профессия: %s (%s)", chosen, PROFESSIONS[chosen]["title"])

        # Пипетка — снять реальный цвет ресурса кликами по игре
        print("\n>>> ПИПЕТКА (рекомендуется для камней/рыбы: цвет берётся прямо из игры).")
        print(">>> Для камней РАЗНОГО цвета снимай по одному образцу на каждый цвет.")
        print(">>> Совет: кликай ОДИН раз ровно по центру ресурса, потом ENTER.")
        ans = read_line_keep_alive(
            ctx, ">>> Сколько образцов цвета снять? (0 — пропустить и оставить пресет; по умолчанию 0): ")
        try:
            n_samples = int(ans) if ans else 0
        except ValueError:
            n_samples = 0
        if n_samples > 0:
            collected = []
            for i in range(n_samples):
                pt = _capture_point(
                    ctx, page, clicks,
                    "образец #%d — кликни по ресурсу (ENTER без клика — стоп)" % (i + 1))
                if not pt:
                    break
                try:
                    full = screenshot_bgr(page)
                    rngs = sample_hsv_ranges_at(full, pt[0], pt[1])
                except Exception as ex:
                    log.warning("Пипетка не сработала: %s", ex)
                    rngs = []
                if rngs:
                    collected.extend(rngs)
                    log.info("  Снят цвет (HSV-диапазон): %s", rngs)
                else:
                    log.info("  Не удалось снять цвет — попробуй кликнуть точнее по ресурсу.")
            if collected:
                zones["resource_ranges"] = collected
                log.info("Пипетка: сохранено диапазонов цвета: %d.", len(collected))
            else:
                log.info("Пипетка: образцы не сняты — останется пресет профессии.")
        # применить выбор сразу, чтобы дальнейшие этапы работали с нужным ресурсом
        set_active_profession(zones["profession"], custom_ranges=zones.get("resource_ranges"))

        # 1) ОБЛАСТЬ КАРТЫ
        print(">>> ЭТАП 1. ОБЛАСТЬ КАРТЫ (где бот ищет ресурсы).")
        p1 = _capture_point(ctx, page, clicks, "ЛЕВЫЙ-ВЕРХНИЙ угол зелёной карты")
        p2 = _capture_point(ctx, page, clicks, "ПРАВЫЙ-НИЖНИЙ угол зелёной карты")
        if p1 and p2:
            left, top = min(p1[0], p2[0]), min(p1[1], p2[1])
            width, height = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
            if width > 100 and height > 100:
                zones["map_region"] = [left, top, width, height]
                log.info("Область карты: %s", zones["map_region"])

        # 2) СПОСОБ ДОБЫЧИ
        print("\n>>> ЭТАП 2. СПОСОБ ДОБЫЧИ.")
        ans = read_line_keep_alive(
            ctx, ">>> Сколько кликов запускают добычу ресурса? Введи 1 или 2 и нажми ENTER "
                 "(по умолчанию 2): ")
        if ans in ("1", "2", "3"):
            zones["gather_clicks"] = int(ans)
            log.info("Добыча: %s клик(а/ов).", ans)

        # 3) КНОПКА «закрыть»
        print("\n>>> ЭТАП 3. КНОПКА «закрыть» (окна-ошибки).")
        print(">>> Кликни по ЧУЖОМУ ресурсу (который твоя профессия НЕ добывает),")
        print(">>> чтобы вылезло окно «нет профессии».")
        close_t = _capture_point(ctx, page, clicks,
                                 "кнопку «закрыть» в этом окне (нет окна — пропусти)")
        if close_t:
            zones["close"] = close_t

        # 4) ЗОНЫ БОЯ
        print("\n>>> ЭТАП 4. ЗОНЫ БОЯ (нужно быть В БОЮ; нет боя — пропускай ENTER).")
        b = _capture_point(ctx, page, clicks, "ЗОНУ БЛОКА на колесе")
        if b:
            zones["block"] = b
        a = _capture_point(ctx, page, clicks, "ЗОНУ АТАКИ на колесе")
        if a:
            zones["attack"] = a
        e = _capture_point(ctx, page, clicks, "кнопку «выход» после победы")
        if e:
            zones["exit"] = e
        h = _capture_point(ctx, page, clicks, "кнопку «В охоту» (можно пропустить)")
        if h:
            zones["hunt"] = h

        try:
            save_zones(zones)
            print("\n================ ГОТОВО ================")
            log.info("Сохранил в %s: %s", ZONES_FILE, zones)
            log.info("Запускай: python omela_bg.py")
        except Exception as ex:
            log.warning("Не смог сохранить: %s", ex)
        ctx.close()


# =========================================================================
#                              РАБОЧИЙ РЕЖИМ
# =========================================================================

def gather_visible(page, scroll_pos, total):
    """Собрать омелу в текущем кадре. Возвращает (total, прервано_ли_боем)."""
    now = time.time()
    _prune(_failed_points, now, 3)
    _prune(_recent_points, now, 2)
    pts = find_resource(crop_map(screenshot_bgr(page)))
    if len(pts) > MAX_PER_CYCLE:
        log.info("Слишком много пятен (%d) — вероятно анимация, пропускаю кадр.", len(pts))
        return total, False
    if pts:
        log.info("Вижу ресурсов (%s): %d шт. (прокрутка %d).", ACTIVE_PROF, len(pts), scroll_pos)
    for (cx, cy) in pts:
        px, py = map_to_page(cx, cy)
        now = time.time()
        # уже добываем этот куст — не тыкаем повторно (иначе добыча отменится)
        if _recent(px, py, now):
            continue
        # чужой ресурс из чёрного списка — пропускаем
        if SKIP_FAILED_ENABLED and _fp_blacklisted(scroll_pos, px, py, now):
            continue
        gather_click(page, px, py)
        _recent_add(px, py, now)
        total += 1
        log.info("Начал добычу (всего запусков: %d)", total)
        time.sleep(1.0)   # дать окну «Добыча» появиться

        # ждём, пока добыча идёт (видно окно прогресса с зелёной полосой),
        # НЕ трогая его; закрываем только окно-ошибку
        waited = 0.0
        err_streak = 0
        while waited < GATHER_MAX_WAIT:
            if FIGHT_ENABLED and FIGHT_POLL_AFTER_GATHER and in_fight(page):
                do_fight(page)
                return total, True
            kind = window_kind(page)
            if kind == "progress":
                err_streak = 0
                time.sleep(GATHER_POLL)
                waited += GATHER_POLL
                continue
            if kind == "error":
                # «ошибка» только если подтвердилась 2 раза подряд (чтобы не спутать
                # с началом добычи, когда зелёная полоса ещё узкая)
                err_streak += 1
                if err_streak < 2:
                    time.sleep(GATHER_POLL)
                    waited += GATHER_POLL
                    continue
                if get_close_target():
                    click_point(page, get_close_target())
                total -= 1
                if SKIP_FAILED_ENABLED:
                    _fp_add(scroll_pos, px, py, time.time())
                    log.info("Чужой/неудачный ресурс — закрыл окно и запомнил.")
                break
            # 'none' → окна нет: добыча завершилась (успех) или ещё идёт в фоне
            break

        if FIGHT_ENABLED and FIGHT_POLL_AFTER_GATHER and in_fight(page):
            do_fight(page)
            return total, True
        time.sleep(random.uniform(*BETWEEN_HERBS))
    return total, False


def mode_run():
    with sync_playwright() as p:
        ctx, page = open_and_wait(
            p, "Войди в игру и встань на локацию с нужным ресурсом. После ENTER начнётся сбор.")
        apply_saved_config()
        log.info("Старт сбора. Стоп: Ctrl+C.")
        if FIGHT_ENABLED:
            block, attack, exit_t, hunt_t = resolve_fight_targets()
            log.info("Авто-бой: блок=%s атака=%s выход=%s охота=%s", block, attack, exit_t, hunt_t)
        if get_close_target():
            log.info("Кнопка «закрыть» откалибрована: %s", get_close_target())
        else:
            log.warning("Кнопка «закрыть» НЕ откалибрована — прогони --calib (этап 3).")
        started = time.time()
        cycle = 0
        scroll_pos = 0
        next_long = random.randint(*LONG_BREAK_EVERY)
        total = 0
        try:
            while True:
                if (time.time() - started) / 60.0 >= MAX_RUNTIME_MIN:
                    log.info("Лимит времени (%d мин). Стоп.", MAX_RUNTIME_MIN)
                    break
                cycle += 1
                close_if_blocking(page)
                if FIGHT_ENABLED and in_fight(page):
                    do_fight(page)
                    time.sleep(random.uniform(*CYCLE_PAUSE))
                    continue
                if FIGHT_ENABLED and stats_screen_present(page):
                    _, _, _, hunt_t = resolve_fight_targets()
                    return_to_hunt(page, hunt_t)
                    time.sleep(random.uniform(*CYCLE_PAUSE))
                    continue

                total, interrupted = gather_visible(page, scroll_pos, total)
                if interrupted:
                    time.sleep(random.uniform(*CYCLE_PAUSE))
                    continue

                if MAP_SCROLL_ENABLED and MAP_SCROLL_POSITIONS > 1:
                    if scroll_pos < MAP_SCROLL_POSITIONS - 1:
                        if scroll_map(page, MAP_SCROLL_DELTA):
                            scroll_pos += 1
                    else:
                        scroll_map(page, -MAP_SCROLL_DELTA * (MAP_SCROLL_POSITIONS - 1))
                        scroll_pos = 0

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
    ap = argparse.ArgumentParser(description="Автосбор ресурсов + авто-бой (Playwright).")
    ap.add_argument("--login", action="store_true", help="только войти (сохранить сессию)")
    ap.add_argument("--calib", action="store_true", help="мастер калибровки (профессия/карта/добыча/закрыть/бой)")
    ap.add_argument("--debug", action="store_true", help="скриншот + DOM + слепок боя")
    ap.add_argument("--prof", metavar="ИМЯ",
                    help="выбрать профессию (%s) и сохранить в конфиг" % "/".join(PROFESSIONS))
    args = ap.parse_args()

    # --prof: сохранить выбранную профессию в fight_zones.json (можно вместе с др. режимом)
    if args.prof:
        if args.prof not in PROFESSIONS:
            print("Неизвестная профессия '%s'. Доступны: %s"
                  % (args.prof, ", ".join(PROFESSIONS)))
            return
        z = load_zones() or {}
        if z.get("profession") != args.prof:
            z.pop("resource_ranges", None)   # старый цвет от другой профессии не годится
        z["profession"] = args.prof
        save_zones(z)
        log.info("Профессия сохранена: %s (%s). Цвет — пресет; уточнить: --calib (пипетка).",
                 args.prof, PROFESSIONS[args.prof]["title"])

    if args.login:
        mode_login()
    elif args.calib:
        mode_calib()
    elif args.debug:
        mode_debug()
    else:
        mode_run()


if __name__ == "__main__":
    main()
