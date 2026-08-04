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
  python omela_bg.py --sens 0.8       # чувствительность: <1 мягче (видит больше), >1 строже
  python omela_bg.py --calib          # МАСТЕР: профессия/пипетка/исключение, карта, добыча, «закрыть», бой
  python omela_bg.py --debug          # скриншот + DOM + карта распознавания + слепок боя
  python omela_bg.py --testcraft      # разово проверить авто-крафт (профессия→«Создать»→«Вернуться»)
  python omela_bg.py                  # рабочий режим (сбор + бой + авто-крафт рецептов)

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
    # blob-поля:
    #   min_area/size_*/aspect — размер и пропорции пятна (как раньше);
    #   solidity   — плотность заливки пятна (area / площадь_описанного_прямоуг.);
    #                камень «сплошной», рваная трава — нет. 0 = выключить;
    #   circularity— насколько пятно круглое (1.0 = идеальный круг). 0 = выключить;
    #   contrast   — насколько ресурс ЯРЧЕ/НАСЫЩЕННЕЕ фона вокруг (S+V, 0..255+).
    #                Главный фильтр против травы: у камня контраст высокий,
    #                у пятна травы — низкий. 0 = выключить.
    "herbalist": {
        "title": "Травник (омела/травы)",
        "ranges": [[[22, 120, 175], [45, 255, 255]]],
        "blob": {"min_area": 40, "size_min": 8, "size_max": 40,
                 "aspect": [0.45, 2.2],
                 "solidity": 0.30, "circularity": 0.0, "contrast": 8},
    },
    # Драгоценные камни — яркие, насыщенные, компактные пятна разных цветов.
    # Зелёный диапазон НАМЕРЕННО сужен (S/V высокие), чтобы не ловить траву.
    # Для камней фильтры формы/контраста включены жёстче: они реально круглые,
    # плотные и заметно ярче газона — так отсекаем ложные срабатывания по траве.
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
        "blob": {"min_area": 18, "size_min": 5, "size_max": 42,
                 "aspect": [0.4, 2.5],
                 "solidity": 0.40, "circularity": 0.30, "contrast": 18},
    },
    # Рыба — заготовка. Лучше снять цвет пипеткой в --calib.
    "fisher": {
        "title": "Рыбак (рыба)",
        "ranges": [
            [[90,  90, 150], [128, 255, 255]],    # серебристо-синие рыбы
            [[15,  90, 150], [35,  255, 255]],    # золотистые рыбы
        ],
        "blob": {"min_area": 30, "size_min": 7, "size_max": 48,
                 "aspect": [0.35, 3.0],
                 "solidity": 0.30, "circularity": 0.0, "contrast": 10},
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
# Новые фильтры формы/контраста (см. комментарий в PROFESSIONS). 0 = выключено.
BLOB_MIN_SOLIDITY    = _b.get("solidity", 0.0)
BLOB_MIN_CIRCULARITY = _b.get("circularity", 0.0)
BLOB_MIN_CONTRAST    = _b.get("contrast", 0.0)

# «Пипетка-исключение»: цвета фона/травы/чужих ресурсов, которые НЕ надо собирать.
# Пятно, чей центр попадает в один из этих диапазонов, отбрасывается. Заполняется
# в --calib (Этап 0б, «пипетка фона») и хранится в fight_zones.json → exclude_ranges.
EXCLUDE_RANGES = []

# Чувствительность распознавания: множитель ко всем фильтрам формы/контраста.
#   1.0  — как в пресете (баланс);
#   <1.0 — МЯГЧЕ: бот видит больше (и слабые камни), но чаще ловит лишнее;
#   >1.0 — СТРОЖЕ: меньше ложных кликов, но можно пропустить тусклый ресурс.
# Можно переопределить в fight_zones.json ключом "sensitivity" или флагом --sens.
DETECT_SENSITIVITY = 1.0

MATCH_MIN_DISTANCE = 25
MAX_PER_CYCLE = 40          # больше — вероятно анимация: переснимаем кадр и берём топ
DETECT_MAX_RESULTS = 12     # за цикл кликаем не больше стольких — самых «уверенных»
# Сколько циклов подряд без единого найденного ресурса, прежде чем бот один раз
# подскажет в логе, что настроить (иначе он молча крутит карту — «не собирает»).
DRY_STREAK_HINT = 6

# «Фон-детектор» для пипетки/старта. Ресурс (камень/куст/рыба) — это КОМПАКТНОЕ
# пятно: его цвет покрывает лишь малую долю карты. Если «цвет ресурса» покрывает
# большую часть карты — это почти наверняка ТРАВА/ФОН, случайно снятая пипеткой.
# Такой цвет ломает сбор: маска заливает всю карту, компактных пятен нет, и бот
# только крутит карту. Порог = максимально допустимая доля карты (0..1). Пресеты
# профессий покрывают ~1-3% карты, трава — десятки процентов, значит 0.15 разделяет
# их с большим запасом.
BG_COVERAGE_MAX = 0.15

# Добыча
GATHER_CLICKS   = 2            # сколько кликов запускают добычу (в этой игре — двойной)
DOUBLECLICK_GAP = (0.08, 0.16)
GATHER_WAIT     = (2.5, 4.5)
BETWEEN_HERBS   = (0.6, 1.6)
CYCLE_PAUSE     = (2.0, 4.0)
LONG_BREAK_EVERY = (15, 30)
LONG_BREAK       = (20.0, 60.0)
MAX_RUNTIME_MIN  = 240

# =========================================================================
#        АВТО-КРАФТ РЕЦЕПТОВ (панель профессии → кнопки «Создать»)
# =========================================================================
# Пока идёт сбор ресурсов, бот периодически заходит в панель профессии
# (кнопка «профессии» на боковой панели) и жмёт «Создать» у каждого рецепта
# из «Избранных». Рецепты обновляются раз в несколько минут, поэтому крафт
# запускается по таймеру. После каждого «Создать» экран сам перезагружается —
# бот ждёт, потом жмёт следующий рецепт. В конце — «Вернуться» к сбору.
#
# Точки снимаются в мастере --calib (ЭТАП 6) и хранятся в fight_zones.json:
#   "craft_open"    — кнопка, открывающая панель профессии;
#   "craft_creates" — список кнопок «Создать» (по одной на рецепт);
#   "craft_back"    — кнопка «Вернуться» (выход из панели к сбору).
CRAFT_ENABLED         = True          # включить авто-создание рецептов
CRAFT_EVERY_SEC       = 330           # как часто крафтить, сек (5 мин 30 сек)
CRAFT_ON_START        = True          # сразу крафтить при запуске бота
CRAFT_OPEN_WAIT       = (1.2, 2.2)    # ждём открытия панели профессии
CRAFT_RELOAD_WAIT     = (1.6, 2.6)    # ждём перезагрузки экрана после «Создать»
CRAFT_AFTER_BACK_WAIT = (1.0, 1.8)    # пауза после возврата к сбору
CRAFT_CLICK_GAP       = (0.4, 0.8)    # микропауза перед нажатием каждого «Создать»

# Внутреннее состояние авто-крафта (глобальное — чтобы цикл сбора мог прерваться
# ровно в момент, когда подошло время крафта, а не ждать конца всего цикла).
_CRAFT_ON   = False   # включён ли авто-крафт в этом запуске
_NEXT_CRAFT = None     # время (time.time()) следующего захода за «Создать»

# Не тыкать повторно куст, который только что начали добывать (иначе добыча отменится)
RECLICK_COOLDOWN = 9.0         # сек: столько не кликаем по той же точке снова
RECLICK_RADIUS   = 26          # px

# Пропуск чужих ресурсов (чёрный список): клик привёл к окну ошибки → запомнить.
SKIP_FAILED_ENABLED = True
SKIP_FAILED_RADIUS  = 22
SKIP_FAILED_TTL     = 300.0

# ---- ПРОКРУТКА КАРТЫ (вертикальная) -------------------------------------
# Прокрутка АДАПТИВНАЯ: бот прокручивает карту колёсиком шагами, после каждого
# шага проверяет, что картинка реально сдвинулась, и когда упирается в край
# (низ/верх карты) — разворачивается и идёт обратно («маятник»). Так охватывается
# ВСЯ высота карты, не завися от фиксированного числа позиций, и прокрутка не
# «проскакивает» дальше края вхолостую.
MAP_SCROLL_ENABLED   = True
MAP_SCROLL_DELTA     = 260        # на сколько px прокручивать за один шаг (>0)
MAP_SCROLL_SUBSTEPS  = 4          # дробим шаг на N мелких докруток — плавно, по-людски
MAP_SCROLL_SETTLE    = (0.45, 0.8)  # пауза после шага, чтобы карта «устоялась» (сек)
MAP_SCROLL_MAX_POS   = 10         # максимум шагов в одну сторону (страховка от зацикливания)
MAP_SCROLL_MOVE_MIN  = 2.5        # средняя разница пикселей, выше которой карта «сдвинулась»
# Устаревшее (совместимость): если задать >1, ограничит размах «маятника».
MAP_SCROLL_POSITIONS = 0
MAP_SCROLL_DELTA_LEGACY = MAP_SCROLL_DELTA

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

# ---- ПОИСК КНОПКИ «закрыть» В ПОЛОСЕ ------------------------------------
# Разные окна-ошибки («Добыча не удалась», «нет профессии», «Объект уже не
# существует!») имеют РАЗНУЮ высоту, поэтому кнопка «закрыть» оказывается на РАЗНОЙ
# высоте экрана. Одной откалиброванной точки мало: для окна другой высоты кнопка
# уезжает вниз/вверх, проверка «красноты» в точке промахивается — и окно НЕ
# закрывается (бот застревает). Поэтому кнопку ищем в вертикальной ПОЛОСЕ вокруг
# калиброванной точки (окна по центру, X кнопки почти не гуляет) и кликаем по её
# ФАКТИЧЕСКОМУ центру. Заголовок окна тоже красный, но он ВЫШЕ кнопки — берём самый
# нижний подходящий красный прямоугольник в полосе.
POPUP_BAND_UP     = 70         # на сколько px ВЫШЕ калиброванной точки искать кнопку
POPUP_BAND_DOWN   = 95         # на сколько px НИЖЕ калиброванной точки искать кнопку
POPUP_BAND_XTOL   = 95         # полуширина поиска по X вокруг калиброванной точки
POPUP_BTN_W       = (40, 320)  # ширина кнопки «закрыть» (px)
POPUP_BTN_H       = (9, 34)    # высота кнопки «закрыть» (px). НЕ выше — иначе можно
                               # спутать с красной частью полосы прогресса (там ~46px)
POPUP_BTN_ASPECT  = 1.8        # ширина/высота ≥ этого (кнопка горизонтальная)
POPUP_BTN_FILL    = 0.40       # плотность заливки красным (0..1)
POPUP_CLOSE_TRIES = 3          # сколько раз пытаться закрыть окно за один вызов
# Отличаем КНОПКУ «закрыть» от красного ЗАГОЛОВКА окна (оба красные): над кнопкой —
# светлое «тело» окна (бежевый фон), а над заголовком — тёмный игровой фон. Считаем
# долю «светлых» пикселей в полоске НАД красным прямоугольником.
POPUP_BODY_V_MIN     = 140     # яркость (V) «тела» окна ≥ этого
POPUP_BODY_S_MAX     = 120     # насыщенность (S) «тела» окна ≤ этого (бежевый бледный)
POPUP_ABOVE_STRIP    = 9       # высота полоски НАД кнопкой для проверки «тела» (px)
POPUP_ABOVE_BEIGE_MIN = 0.45   # какая доля полоски над кнопкой должна быть «телом»
# Окно-ошибка может вылезти НЕ по центру карты и на разной высоте (например, у места
# клика). Поэтому кнопку «закрыть» ищем ПО ВСЕЙ карте, а не только у калиброванной
# точки. Надёжный якорь: окна центрируются по ГОРИЗОНТАЛИ — центр кнопки близко к
# центру карты по X (± POPUP_CENTER_XTOL). Это отсекает красные ники/текст на карте.
POPUP_CENTER_XTOL    = 150     # насколько центр кнопки может отстоять от центра карты по X
# Тело окна НАД кнопкой, где ищем зелёную полосу прогресса «Добыча». Ищем только тут
# (внутри попапа трава закрыта бежевым), чтобы зелень карты не путалась с прогрессом.
POPUP_BODY_HALF_W    = 170     # полуширина тела окна от центра кнопки (px)
POPUP_BODY_UP        = 130     # на сколько px ВВЕРХ от кнопки простирается тело окна
POPUP_TITLE_MINW     = 150     # мин. ширина красного ЗАГОЛОВКА окна (px)
POPUP_TITLE_MINGAP   = 8       # мин. зазор между низом заголовка и верхом кнопки (px)
POPUP_TITLE_MAXGAP   = 170     # макс. высота тела окна (заголовок ↔ кнопка, px)
# Полосу прогресса «Добыча» отличаем от травы по тому, что рядом с ней (сверху ИЛИ
# снизу) — светлое БЕЖЕВОЕ тело окна, а у травы вокруг снова трава (бежевого 0%).
# Достаточно бежевого хотя бы с ОДНОЙ стороны: настоящую полосу не потеряем (не
# отменим добычу), а траву отсечём.
PROGRESS_BEIGE_STRIP = 6       # высота полоски над/под полосой для проверки «тела» (px)
PROGRESS_BEIGE_MIN   = 0.30    # доля «бежевого» хотя бы с одной стороны полосы
# Полоса добычи — это зелёная ЗАЛИВКА, рядом с которой красный «остаток» отсчёта
# (полоса убывает) или бежевое тело окна. По этому и опознаём прогресс. Зелёная
# заливка ПЛОТНАЯ (в отличие от тонкого зелёного текста в окне).
PROGRESS_FILL_H      = (12, 44)  # высота зелёной заливки полосы (px)
PROGRESS_FILL_WMIN   = 16        # мин. ширина зелёной заливки (px)
PROGRESS_FILL_ASPECT = 1.3       # ширина/высота заливки ≥ этого
PROGRESS_FILL_SOLID  = 0.60      # плотность зелёной заливки (0..1; текст «рыхлее»)
PROGRESS_SIDE_W      = 55        # ширина полоски сбоку/сверху/снизу для проверки
PROGRESS_RED_ADJ_MIN = 0.20      # доля красного сбоку от заливки (красный «остаток»)

# ---- ЗАНОЗА (splinter) --------------------------------------------------
# При долгой добыче персонаж получает «занозу»: в чат приходит оповещение, а
# рабочий инструмент убирается из рук в рюкзак (добывать нельзя, пока не вылечат).
# Лечение: попросить в чате игроков той же локации «дёрнуть занозу». Чат — это
# HTML, поэтому бот НАДЁЖНО и читает оповещение, и пишет просьбу сам.
# Что делает бот: заметил «занозу» → пауза сбора → пишет просьбу в чат (и вежливо
# повторяет) → зовёт тебя звуком вернуть инструмент в руки → как только добыча
# снова проходит (инструмент в руках), продолжает сам.
SPLINTER_ENABLED       = True
SPLINTER_CHAT_KEYWORD  = "заноз"     # слово-маркер занозы (регистр не важен)
# ВАЖНО: реагируем ТОЛЬКО на СИСТЕМНУЮ строку игры о СВОЕЙ занозе. Это отсекает:
#   • ники игроков (напр. «Заноза-Буля») в списке жителей и в чате;
#   • чужие и личные сообщения — в них есть знак «»» («Ник » Ник: текст»);
#   • чужие/свои просьбы «дерните/вытащите занозу»;
#   • строку лечения «Вы избавились от занозы» (в ней есть «избавил»).
# Строка засчитывается как «моя заноза», если (после отсева «»», просьб и лечения):
#   (а) содержит один из НАДЁЖНЫХ маркеров SPLINTER_SELF_MARKERS — это системная
#       строка-награда «Получено: Заноза …», которая бывает ТОЛЬКО у тебя; ЛИБО
#   (б) начинается с одного из SPLINTER_SELF_PREFIXES (запасной путь для игр,
#       где сообщение начинается с «Вы …»).
# Если у тебя другой текст — допиши маркер в SPLINTER_SELF_MARKERS (надёжнее) или
# начало строки в SPLINTER_SELF_PREFIXES.
#
# ПРИМЕР реальной строки игры (её раньше НЕ ловил префикс, т.к. она начинается со
# слова «Обнаружив», а не «Вы»):
#   «Обнаружив драгоценный камень, вы бездумно набросились на него… шипы которой
#    сразу же вонзились вам в руку. Получено: Заноза 1 шт.»
# Ключевой хвост «Получено: Заноза» и есть надёжный маркер ниже.
SPLINTER_SELF_MARKERS  = ("получено: заноз", "получено заноз", "получена заноз",
                          "получены занозы", "вонзил")   # надёжные маркеры «моей» занозы
SPLINTER_SELF_PREFIXES = ("вы ", "вы,", "вы получ", "вы посад", "вы загна",
                          "вы занозил", "у вас", "вам ")
SPLINTER_REQUEST_WORDS = ("дерн", "дёрн", "вытащ", "вытян", "помог", "прош",
                          "пожалуйста", "плиз", "плз")
SPLINTER_HEAL_WORDS    = ("избавил", "избавля", "избавилис")
SPLINTER_MESSAGE       = "дерните занозу пожалуйста"  # текст просьбы (можно добавить :mol:)
SPLINTER_PM_EACH       = True        # True: писать ЛИЧНО каждому игроку локации;
                                     # False: одно сообщение в общий чат
# Как в этой игре адресуется сообщение (впечатывается префикс в поле ввода):
#   личное:          "prv[{nick}] {msg}"
#   в общий, адресно: "to[{nick}] {msg}"
SPLINTER_PM_FORMAT     = "prv[{nick}] {msg}"
SPLINTER_REPEAT_EVERY  = 45.0        # как часто повторять просьбу (сек)
SPLINTER_POLL          = 20.0        # как часто проверять, вернулся ли инструмент (сек)
SPLINTER_MAX_WAIT      = 1200.0      # максимум ждать лечения (сек), потом продолжить пробовать
SPLINTER_ALERT_SOUND   = True        # звать звуковым сигналом
# Благодарность помощнику. Когда занозу вытащат, в чат приходит «Вы избавились от
# занозы» и строка «…аптечку, ИМЯ [ур] избавляет воина ТВОЙ_НИК от занозы». Бот
# достаёт ИМЯ и пишет благодарность в чат.
SPLINTER_THANKS         = True
# Тело благодарности. В личке (prv[ник]) имя уже адресовано префиксом, поэтому его
# в тексте не повторяем. Если хочешь имя в тексте — добавь {name}.
SPLINTER_THANKS_MESSAGE = "спасибо вам большое!"
SPLINTER_MY_NAME        = "-VeliS-"          # твой ник (напр. "-VeliS-"); пусто — определять по последней строке лечения
# Авто-возврат инструмента после лечения. Последовательность:
#   рюкзак → вкладка «вещи» → навести на кирку (появляется «надеть») → клик «надеть»
#   → режим охоты. Точки снимаются в --calib (Этап 5): bag / tab / pick / equip / hunt_mode.
# Если не откалибровано или не сработало — бот зовёт звуком и ждёт, пока наденешь сама.
SPLINTER_REEQUIP        = True
SPLINTER_HOVER_WAIT     = 1.2         # сек навести на кирку и подождать надпись «надеть»

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
# Как бот выбирает зону, если их снято несколько (Этап 4 калибровки):
#   "cycle"  — по кругу, зона за зоной (комбинация ударов: голова→грудь→живот→…)
#   "random" — случайная зона каждый раунд (труднее предсказать)
# Значения можно переопределить в fight_zones.json (attack_mode / block_mode).
FIGHT_ATTACK_MODE = "cycle"
FIGHT_BLOCK_MODE  = "random"

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
    global BLOB_MIN_SOLIDITY, BLOB_MIN_CIRCULARITY, BLOB_MIN_CONTRAST
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
    BLOB_MIN_SOLIDITY    = b.get("solidity", 0.0)
    BLOB_MIN_CIRCULARITY = b.get("circularity", 0.0)
    BLOB_MIN_CONTRAST    = b.get("contrast", 0.0)
    log.info("Профессия: %s [%s] — цвет: %s.", name, prof["title"], src)


def apply_saved_config():
    """Подтянуть в глобальные настройки то, что снято мастером --calib."""
    global MAP_REGION, GATHER_CLICKS, EXCLUDE_RANGES, DETECT_SENSITIVITY
    z = load_zones() or {}
    # профессия + цвет ресурса (пипетка имеет приоритет над пресетом)
    prof_name = z.get("profession", ACTIVE_PROF)
    set_active_profession(prof_name,
                          custom_ranges=z.get("resource_ranges"),
                          custom_blob=z.get("resource_blob"))
    # цвета-исключения (пипетка фона): что бот НЕ должен трогать.
    # ЗАЩИТА ОТ «ОТРАВЛЕННОГО» ИГНОРА: частая причина «бот перестал собирать» —
    # в игнор случайно попал цвет САМОГО ресурса (например, кликнули по камню, а не
    # по траве в Шаге 1.3). Тогда бот выбрасывает все находки этого цвета и только
    # крутит карту. Такой игнор мы на старте НЕ применяем (и пишем об этом в лог):
    # центр диапазона-исключения не должен совпадать с активным цветом ресурса.
    raw_excl = _ranges_to_np(z.get("exclude_ranges"))
    EXCLUDE_RANGES = []
    dropped = 0
    for lo, hi in raw_excl:
        center = (lo.astype(int) + hi.astype(int)) // 2
        if RESOURCE_RANGES and _pixel_in_ranges(center, RESOURCE_RANGES):
            dropped += 1
            continue
        EXCLUDE_RANGES.append([lo, hi])
    if dropped:
        log.warning("Игнор: %d цвет(ов) совпадали с ЦВЕТОМ РЕСУРСА — НЕ применяю их "
                    "(иначе бот не собирал бы ресурс). Чтобы убрать совсем — "
                    "«Очистить игнор» в launcher.py.", dropped)
    if EXCLUDE_RANGES:
        log.info("Цветов-исключений (фон/трава): %d.", len(EXCLUDE_RANGES))
    # чувствительность распознавания (если задана в конфиге)
    s = z.get("sensitivity")
    if isinstance(s, (int, float)) and 0.2 <= float(s) <= 4.0:
        DETECT_SENSITIVITY = float(s)
    if abs(DETECT_SENSITIVITY - 1.0) > 1e-6:
        log.info("Чувствительность распознавания: %.2f (%s).", DETECT_SENSITIVITY,
                 "мягче" if DETECT_SENSITIVITY < 1 else "строже")
    mr = z.get("map_region")
    if mr and len(mr) == 4:
        MAP_REGION = {"left": int(mr[0]), "top": int(mr[1]),
                      "width": int(mr[2]), "height": int(mr[3])}
        log.info("Область карты из калибровки: %s", MAP_REGION)
    gc = z.get("gather_clicks")
    if gc in (1, 2, 3):
        GATHER_CLICKS = int(gc)
    log.info("Добыча: %d клик(а/ов) на ресурс.", GATHER_CLICKS)


def apply_run_config():
    """Подтянуть настройки ЗАПУСКА из fight_zones.json (их задаёт меню launcher.py).

    Все ключи опциональны и лежат под "run_config": если ключа нет — остаётся
    значение-константа из кода. Профессия/чувствительность грузит apply_saved_config
    (ключи верхнего уровня "profession"/"sensitivity"), тут — только вкл/выкл и числа.
    """
    global MAX_RUNTIME_MIN, CRAFT_ENABLED, CRAFT_EVERY_SEC, CRAFT_ON_START
    global FIGHT_ENABLED, SPLINTER_ENABLED
    cfg = (load_zones() or {}).get("run_config")
    if not isinstance(cfg, dict):
        return
    v = cfg.get("max_runtime_min")
    if isinstance(v, (int, float)) and v > 0:
        MAX_RUNTIME_MIN = int(v)
    v = cfg.get("craft_every_sec")
    if isinstance(v, (int, float)) and v > 0:
        CRAFT_EVERY_SEC = int(v)
    for key, name in (("craft_enabled", "CRAFT_ENABLED"),
                      ("craft_on_start", "CRAFT_ON_START"),
                      ("fight_enabled", "FIGHT_ENABLED"),
                      ("splinter_enabled", "SPLINTER_ENABLED")):
        if isinstance(cfg.get(key), bool):
            globals()[name] = cfg[key]
    log.info("Настройки меню: время=%d мин | крафт=%s (кажд.%dс, старт=%s) | бой=%s | лечение=%s.",
             MAX_RUNTIME_MIN, "вкл" if CRAFT_ENABLED else "выкл", CRAFT_EVERY_SEC,
             "да" if CRAFT_ON_START else "нет",
             "вкл" if FIGHT_ENABLED else "выкл", "вкл" if SPLINTER_ENABLED else "выкл")


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


def _pixel_in_ranges(hsv_px, ranges):
    """Попадает ли HSV-пиксель (h,s,v) хотя бы в один диапазон из списка np-пар."""
    h_, s_, v_ = int(hsv_px[0]), int(hsv_px[1]), int(hsv_px[2])
    for lo, hi in ranges:
        if (lo[0] <= h_ <= hi[0] and lo[1] <= s_ <= hi[1] and lo[2] <= v_ <= hi[2]):
            return True
    return False


def _color_is_resource(full_bgr, x, y, hw=13):
    """Похож ли цвет вокруг точки (x, y) на цвет РЕСУРСА (RESOURCE_RANGES)?

    Нужно для защиты «пипетки-исключения»: если игрок случайно кликнул по камню,
    который сам же собирает, мы НЕ должны заносить этот цвет в чёрный список.
    Берём медианный HSV в маленьком пятне вокруг клика и проверяем попадание
    в активные диапазоны ресурса. Если ресурс не задан — вернём False."""
    try:
        if not RESOURCE_RANGES:
            return False
        h, w = full_bgr.shape[:2]
        x0, x1 = max(0, int(x) - hw), min(w, int(x) + hw)
        y0, y1 = max(0, int(y) - hw), min(h, int(y) + hw)
        if x1 <= x0 or y1 <= y0:
            return False
        patch = full_bgr[y0:y1, x0:x1]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        # берём яркие/насыщенные пиксели (сам ресурс), фон-траву отбрасываем
        bright = hsv[(hsv[:, 1] >= 60) & (hsv[:, 2] >= 60)]
        use = bright if len(bright) >= 8 else hsv
        med = [int(np.median(use[:, 0])), int(np.median(use[:, 1])), int(np.median(use[:, 2]))]
        return _pixel_in_ranges(med, RESOURCE_RANGES)
    except Exception:
        return False


def find_resource_scored(img_bgr, debug=False):
    """Найти ресурс активной профессии и вернуть список кандидатов со ВСЕМИ метриками:
    [{"cx","cy","w","h","area","solidity","circularity","contrast","sat","score"}, ...],
    отсортированный по «уверенности» (score) по убыванию.

    Помимо цвета и размера теперь применяются:
      • solidity   — пятно должно быть плотным (не рваная трава);
      • circularity— пятно должно быть достаточно круглым (камни округлые);
      • contrast   — ресурс должен быть заметно ЯРЧЕ/НАСЫЩЕННЕЕ фона вокруг;
      • exclude    — если центр пятна попал в «цвет фона» (пипетка-исключение) — отброс.
    Пороги масштабируются DETECT_SENSITIVITY (мягче/строже). Debug-режим не отсекает
    кандидатов, а помечает, прошёл ли каждый фильтр (для наглядной диагностики).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in RESOURCE_RANGES:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    if mask is None:
        return []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(closed, 8)

    # HSV-каналы как int для расчёта яркости/насыщенности внутри пятна и вокруг
    s_ch = hsv[:, :, 1].astype(np.int32)
    v_ch = hsv[:, :, 2].astype(np.int32)
    bright = s_ch + v_ch                       # «заметность» пикселя = S + V
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    sens = max(0.2, min(4.0, DETECT_SENSITIVITY))
    min_area = max(6.0, BLOB_MIN_AREA * sens)          # мягче → ловим мельче
    min_solidity    = BLOB_MIN_SOLIDITY * sens
    min_circularity = BLOB_MIN_CIRCULARITY * sens
    min_contrast    = BLOB_MIN_CONTRAST * sens

    cands = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx, cy = int(cent[i][0]), int(cent[i][1])
        rej = None

        # 1) размер / пропорции (как раньше)
        if area < min_area:
            rej = "area"
        elif not (BLOB_SIZE_MIN <= w <= BLOB_SIZE_MAX and BLOB_SIZE_MIN <= h <= BLOB_SIZE_MAX):
            rej = "size"
        elif not (BLOB_ASPECT[0] <= w / max(h, 1) <= BLOB_ASPECT[1]):
            rej = "aspect"

        # 2) цвет-исключение: центр пятна — это фон/трава? → выбросить
        if rej is None and EXCLUDE_RANGES and 0 <= cy < hsv.shape[0] and 0 <= cx < hsv.shape[1]:
            if _pixel_in_ranges(hsv[cy, cx], EXCLUDE_RANGES):
                rej = "exclude"

        blob = (labels[y:y + h, x:x + w] == i)
        blob_area = int(blob.sum()) or 1

        # 3) плотность заливки (solidity) = площадь пятна / площадь bbox
        solidity = blob_area / float(max(w * h, 1))
        if rej is None and min_solidity > 0 and solidity < min_solidity:
            rej = "solidity"

        # 4) округлость (circularity) через периметр контура: 4π·S / P²
        circularity = 0.0
        if min_circularity > 0 or debug:
            try:
                bm = (blob.astype(np.uint8)) * 255
                cnts, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    per = cv2.arcLength(c, True)
                    if per > 0:
                        circularity = min(1.0, 4.0 * np.pi * cv2.contourArea(c) / (per * per))
            except Exception:
                circularity = 0.0
        if rej is None and min_circularity > 0 and circularity < min_circularity:
            rej = "circularity"

        # 5) контраст с фоном: (S+V) внутри пятна vs кольцо вокруг него
        contrast = 0.0
        sat_in = 0.0
        if min_contrast > 0 or debug:
            bm_full = np.zeros(closed.shape, np.uint8)
            bm_full[y:y + h, x:x + w][blob] = 1
            ring = cv2.dilate(bm_full, ring_k) - bm_full
            in_vals = bright[bm_full == 1]
            ring_vals = bright[ring == 1]
            sat_in = float(s_ch[bm_full == 1].mean()) if in_vals.size else 0.0
            if in_vals.size and ring_vals.size:
                contrast = float(in_vals.mean() - ring_vals.mean())
        if rej is None and min_contrast > 0 and contrast < min_contrast:
            rej = "contrast"

        # score — «уверенность»: контраст + насыщенность + округлость + размер
        score = (max(contrast, 0.0) * 1.0 + sat_in * 0.25
                 + circularity * 40.0 + min(area, 300) * 0.10)

        item = {"cx": cx, "cy": cy, "w": int(w), "h": int(h), "area": int(area),
                "solidity": round(solidity, 2), "circularity": round(circularity, 2),
                "contrast": round(contrast, 1), "sat": round(sat_in, 0),
                "score": round(score, 1), "reject": rej}
        if debug or rej is None:
            cands.append(item)

    # сортируем по уверенности; для рабочего режима оставляем только прошедших фильтры
    cands.sort(key=lambda d: d["score"], reverse=True)
    if not debug:
        # дедуп по расстоянию: держим самые «уверенные», близкие дубли выкидываем.
        # Кап по количеству НЕ здесь (иначе сломается защита «слишком много → анимация»).
        picked = []
        for d in cands:
            if all((d["cx"] - p["cx"]) ** 2 + (d["cy"] - p["cy"]) ** 2 >= MATCH_MIN_DISTANCE ** 2
                   for p in picked):
                picked.append(d)
        return picked
    return cands


def find_resource(img_bgr):
    """Совместимость: список центров (cx, cy) ресурса, отсортированный по уверенности."""
    return [(d["cx"], d["cy"]) for d in find_resource_scored(img_bgr)]


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


def sample_exclude_ranges_at(full_bgr, x, y, hw=13, h_pad=10, sv_pad=55):
    """«Пипетка-исключение»: снять цвет ФОНА/травы вокруг точки и построить УЗКИЙ
    диапазон HSV именно этого цвета — включая ограниченный коридор по S/V.

    Отличие от sample_hsv_ranges_at: здесь S/V ограничены и снизу, и сверху
    (медиана±sv_pad), а не «до 255». Это важно: у камня того же оттенка, что и
    трава (например, изумруд ≈ зелёная трава), насыщенность/яркость выше — и он
    НЕ попадёт в этот узкий диапазон, значит исключение фона его не заденет.
    Возвращает список [ [[h,s,v],[h,s,v]], ... ] (красный — два диапазона) или [].
    """
    h, w = full_bgr.shape[:2]
    x0, x1 = max(0, int(x) - hw), min(w, int(x) + hw)
    y0, y1 = max(0, int(y) - hw), min(h, int(y) + hw)
    if x1 <= x0 or y1 <= y0:
        return []
    patch = full_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hm = int(np.median(hsv[:, 0]))
    sm = int(np.median(hsv[:, 1]))
    vm = int(np.median(hsv[:, 2]))
    s_lo, s_hi = max(0, sm - sv_pad), min(255, sm + sv_pad)
    v_lo, v_hi = max(0, vm - sv_pad), min(255, vm + sv_pad)
    h_lo, h_hi = hm - h_pad, hm + h_pad
    ranges = []
    if h_lo < 0:
        ranges.append([[0, s_lo, v_lo], [h_hi, s_hi, v_hi]])
        ranges.append([[180 + h_lo, s_lo, v_lo], [179, s_hi, v_hi]])
    elif h_hi > 179:
        ranges.append([[h_lo, s_lo, v_lo], [179, s_hi, v_hi]])
        ranges.append([[0, s_lo, v_lo], [h_hi - 180, s_hi, v_hi]])
    else:
        ranges.append([[h_lo, s_lo, v_lo], [h_hi, s_hi, v_hi]])
    return ranges


def crop_map(full_bgr):
    m = MAP_REGION
    return full_bgr[m["top"]:m["top"] + m["height"], m["left"]:m["left"] + m["width"]]


def color_coverage_over_map(full_bgr, ranges, map_region=None):
    """Какую ДОЛЮ карты (0..1) заливает набор цветовых диапазонов `ranges`.

    Нужно, чтобы отличить настоящий ресурс (компактное пятно, доля мала) от травы/
    фона (заливает пол-карты). `ranges` — список [lo, hi], где lo/hi — [H,S,V] либо
    np.uint8-массивы. `map_region` — dict {left,top,width,height} или None (тогда
    берётся глобальный MAP_REGION). Вернёт 0.0, если не удалось посчитать.
    """
    if not ranges:
        return 0.0
    m = map_region or MAP_REGION
    try:
        y0, x0 = int(m["top"]), int(m["left"])
        crop = full_bgr[y0:y0 + int(m["height"]), x0:x0 + int(m["width"])]
        if crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = None
        for lo, hi in ranges:
            lo = np.asarray(lo, np.uint8)
            hi = np.asarray(hi, np.uint8)
            mm = cv2.inRange(hsv, lo, hi)
            mask = mm if mask is None else cv2.bitwise_or(mask, mm)
        if mask is None:
            return 0.0
        return float((mask > 0).sum()) / float(mask.size)
    except Exception:
        return 0.0


def map_to_page(cx, cy):
    return MAP_REGION["left"] + cx, MAP_REGION["top"] + cy


def gather_click(page, x, y):
    """Запустить добычу по (x, y) с лёгким разбросом.

    ВАЖНО про двойной клик: игровой canvas запускает добычу по НАСТОЯЩЕМУ событию
    двойного клика (`dblclick`, detail=2). Два отдельных `mouse.click()` порождают
    два события `click` с detail=1, а `dblclick` — НЕ порождают, поэтому добыча
    молча не стартовала (в логе «Начал добычу», а в игре ничего). Playwright шлёт
    корректный двойной клик через `mouse.dblclick()` — им и пользуемся, когда
    GATHER_CLICKS >= 2. Для 1 клика — обычный click; для 3+ — двойной плюс добор.
    """
    x += random.randint(-3, 3)
    y += random.randint(-3, 3)
    page.mouse.move(x, y)
    time.sleep(random.uniform(0.05, 0.15))
    n = max(1, GATHER_CLICKS)
    if n >= 2:
        try:
            page.mouse.dblclick(x, y)          # настоящее событие dblclick
        except Exception:
            # запасной путь, если dblclick недоступен — два клика вплотную
            page.mouse.click(x, y)
            time.sleep(random.uniform(*DOUBLECLICK_GAP))
            page.mouse.click(x, y)
        # если по калибровке нужно больше двух кликов — дожимаем остаток
        for _ in range(n - 2):
            time.sleep(random.uniform(*DOUBLECLICK_GAP))
            page.mouse.click(x, y)
    else:
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
    """Плавно прокрутить карту колёсиком на dy px (положительное — вниз).

    Наводим курсор в центр карты (иначе колесо крутит не ту область), затем
    прокручиваем НЕ одним рывком, а MAP_SCROLL_SUBSTEPS мелкими докрутками —
    это надёжнее срабатывает в игровом canvas и выглядит естественнее.
    """
    cx = MAP_REGION["left"] + MAP_REGION["width"] // 2
    cy = MAP_REGION["top"] + MAP_REGION["height"] // 2
    subs = max(1, MAP_SCROLL_SUBSTEPS)
    step = int(round(dy / float(subs)))
    if step == 0:
        step = 1 if dy > 0 else -1
    try:
        page.mouse.move(cx + random.randint(-6, 6), cy + random.randint(-6, 6))
        time.sleep(random.uniform(0.08, 0.18))
        for _ in range(subs):
            page.mouse.wheel(0, step)
            time.sleep(random.uniform(0.05, 0.12))
        time.sleep(random.uniform(*MAP_SCROLL_SETTLE))
        return True
    except Exception as e:
        log.warning("Прокрутка карты не удалась: %s", e)
        return False


def _map_diff(page):
    """Сделать скриншот и вернуть уменьшенный ч/б кадр карты (для сравнения «до/после»)."""
    try:
        crop = crop_map(screenshot_bgr(page))
        small = cv2.resize(crop, (160, 90), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)
    except Exception:
        return None


def scroll_step_adaptive(page, direction):
    """Прокрутить карту на один шаг в сторону direction (+1 вниз / −1 вверх) и
    проверить, СДВИНУЛАСЬ ли карта. Возвращает True, если сдвинулась (значит края
    ещё не достигли), False — если картинка не изменилась (уперлись в край)."""
    before = _map_diff(page)
    ok = scroll_map(page, MAP_SCROLL_DELTA * (1 if direction >= 0 else -1))
    if not ok:
        return False
    after = _map_diff(page)
    if before is None or after is None:
        return True   # не смогли сравнить — считаем, что сдвинулись (не блокируем цикл)
    moved = float(np.abs(after - before).mean())
    return moved >= MAP_SCROLL_MOVE_MIN


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


def find_close_button_near(full_bgr, target):
    """Найти красную кнопку «закрыть» в вертикальной ПОЛОСЕ вокруг калиброванной
    точки. Возвращает (x, y) центра кнопки или None.

    Зачем: окна-ошибки разной высоты сдвигают кнопку по вертикали, и клик по одной
    статичной точке промахивается. Ищем в полосе [target ± POPUP_BAND_*]. Заголовок
    окна — тоже красный, но ВЫШЕ кнопки, поэтому среди подходящих прямоугольников
    берём САМЫЙ НИЖНИЙ (у него максимальный y)."""
    if target is None or full_bgr is None:
        return None
    tx, ty = int(target[0]), int(target[1])
    H, W = full_bgr.shape[:2]
    x0, x1 = max(0, tx - POPUP_BAND_XTOL), min(W, tx + POPUP_BAND_XTOL)
    y0, y1 = max(0, ty - POPUP_BAND_UP),   min(H, ty + POPUP_BAND_DOWN)
    if x1 <= x0 or y1 <= y0:
        return None
    region = full_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(POPUP_RED_LOW1), np.array(POPUP_RED_HIGH1))
    m2 = cv2.inRange(hsv, np.array(POPUP_RED_LOW2), np.array(POPUP_RED_HIGH2))
    mask = cv2.morphologyEx(cv2.bitwise_or(m1, m2), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    # V/S тела окна берём из того же region-hsv, что и красная маска
    reg_v = hsv[:, :, 2]
    reg_s = hsv[:, :, 1]
    cand_beige = []   # (cx, cy): красные прямоугольники со «светлым телом» над ними
    cand_all = []     # (cx, cy): все подходящие по форме красные прямоугольники
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (POPUP_BTN_W[0] <= w <= POPUP_BTN_W[1]):
            continue
        if not (POPUP_BTN_H[0] <= h <= POPUP_BTN_H[1]):
            continue
        if w / float(max(h, 1)) < POPUP_BTN_ASPECT:
            continue
        if area / float(max(w * h, 1)) < POPUP_BTN_FILL:
            continue
        cx = x0 + int(cent[i][0])
        cy = y0 + int(cent[i][1])
        cand_all.append((cx, cy))
        # полоска НАД прямоугольником: у кнопки там «тело» окна, у заголовка — фон
        sy1 = y
        sy0 = max(0, y - POPUP_ABOVE_STRIP)
        if sy1 > sy0:
            strip_v = reg_v[sy0:sy1, x:x + w]
            strip_s = reg_s[sy0:sy1, x:x + w]
            if strip_v.size:
                beige = float(((strip_v >= POPUP_BODY_V_MIN) &
                               (strip_s <= POPUP_BODY_S_MAX)).sum()) / float(strip_v.size)
                if beige >= POPUP_ABOVE_BEIGE_MIN:
                    cand_beige.append((cx, cy))
    # предпочитаем кнопки с «телом» над ними (не заголовок); среди них — самую нижнюю
    pool = cand_beige if cand_beige else cand_all
    if not pool:
        return None
    return max(pool, key=lambda p: p[1])


def _find_popup_button(full_bgr):
    """Найти нижнюю красную КНОПКУ окна (закрыть/отменить) ГДЕ УГОДНО на карте.
    Возвращает (cx, cy, w, h) или None.

    Признаки: красный ГОРИЗОНТАЛЬНЫЙ прямоугольник + светлое «тело» окна прямо НАД ним
    + центр близко к центру карты по X (окна центрируются). Отсекает красные ники/текст
    на карте и красный заголовок окна (над заголовком нет светлого тела)."""
    if full_bgr is None:
        return None
    H, W = full_bgr.shape[:2]
    try:
        map_cx = MAP_REGION["left"] + MAP_REGION["width"] // 2
        y_lo = MAP_REGION["top"] - 20
        y_hi = MAP_REGION["top"] + MAP_REGION["height"] + 20
    except Exception:
        map_cx, y_lo, y_hi = W // 2, 0, H
    hsv = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(POPUP_RED_LOW1), np.array(POPUP_RED_HIGH1))
    m2 = cv2.inRange(hsv, np.array(POPUP_RED_LOW2), np.array(POPUP_RED_HIGH2))
    mask = cv2.morphologyEx(cv2.bitwise_or(m1, m2), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    vch, sch = hsv[:, :, 2], hsv[:, :, 1]
    best, best_dx = None, 1e9
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx, cy = int(cent[i][0]), int(cent[i][1])
        if not (y_lo <= cy <= y_hi):
            continue
        if not (POPUP_BTN_W[0] <= w <= POPUP_BTN_W[1]):
            continue
        if not (POPUP_BTN_H[0] <= h <= POPUP_BTN_H[1]):
            continue
        if w / float(max(h, 1)) < POPUP_BTN_ASPECT:
            continue
        if area / float(max(w * h, 1)) < POPUP_BTN_FILL:
            continue
        dx = abs(cx - map_cx)
        if dx > POPUP_CENTER_XTOL:          # окно центрируется по горизонтали
            continue
        sy1, sy0 = y, max(0, y - POPUP_ABOVE_STRIP)   # светлое «тело» окна над кнопкой
        if sy1 <= sy0:
            continue
        strip_v = vch[sy0:sy1, x:x + w]
        strip_s = sch[sy0:sy1, x:x + w]
        if not strip_v.size:
            continue
        beige = float(((strip_v >= POPUP_BODY_V_MIN) &
                       (strip_s <= POPUP_BODY_S_MAX)).sum()) / float(strip_v.size)
        if beige < POPUP_ABOVE_BEIGE_MIN:
            continue
        if dx < best_dx:                    # самая «центральная» кнопка — она и есть
            best, best_dx = (cx, cy, int(w), int(h)), dx
    return best


def find_error_popup(full_bgr):
    """Кнопка окна ГДЕ УГОДНО на карте (совместимость): (x, y) или None."""
    b = _find_popup_button(full_bgr)
    return (b[0], b[1]) if b else None


def _find_title_above(full_bgr, cx, button_top):
    """Найти нижнюю границу красного ЗАГОЛОВКА окна над кнопкой. Возвращает y низа
    заголовка или None. Заголовок — широкий красный горизонтальный прямоугольник,
    отцентрованный примерно по X кнопки, чуть выше неё."""
    hsv = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(POPUP_RED_LOW1), np.array(POPUP_RED_HIGH1))
    m2 = cv2.inRange(hsv, np.array(POPUP_RED_LOW2), np.array(POPUP_RED_HIGH2))
    mask = cv2.morphologyEx(cv2.bitwise_or(m1, m2), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    best_bottom = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        tcx, tcy = int(cent[i][0]), int(cent[i][1])
        if w < POPUP_TITLE_MINW:                       # заголовок широкий
            continue
        if abs(tcx - cx) > POPUP_CENTER_XTOL:          # по центру над кнопкой
            continue
        bottom = y + h
        if not (button_top - POPUP_TITLE_MAXGAP <= bottom <= button_top - POPUP_TITLE_MINGAP):
            continue
        if best_bottom is None or bottom > best_bottom:  # ближайший над кнопкой
            best_bottom = bottom
    return best_bottom


def _green_bar_in_region(full_bgr, x0, y0, x1, y1):
    """Есть ли внутри прямоугольника НАСТОЯЩАЯ зелёная полоса прогресса «Добыча»?

    Полосу отличаем от травы по «бежевому боку»: у настоящей полосы прямо НАД или ПОД
    ней — светлое бежевое тело окна (у травы вокруг снова трава, бежевого 0%). Хватает
    бежевого хотя бы с одной стороны — так настоящую полосу не теряем (не отменяем
    добычу), а траву отсекаем."""
    H, W = full_bgr.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(W, int(x1)), min(H, int(y1))
    if x1 <= x0 or y1 <= y0:
        return False
    region = full_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    vch, sch = hsv[:, :, 2], hsv[:, :, 1]
    rh, rw = region.shape[:2]
    mask = cv2.inRange(hsv, np.array(PROGRESS_GREEN_LOW), np.array(PROGRESS_GREEN_HIGH))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    def _beige(yy0, yy1, xx0, xx1):
        yy0, yy1 = max(0, yy0), min(rh, yy1)
        xx0, xx1 = max(0, xx0), min(rw, xx1)
        if yy1 <= yy0 or xx1 <= xx0:
            return 0.0
        v = vch[yy0:yy1, xx0:xx1]
        s = sch[yy0:yy1, xx0:xx1]
        return float(((v >= POPUP_BODY_V_MIN) & (s <= POPUP_BODY_S_MAX)).sum()) / float(v.size)

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
        above = _beige(y - PROGRESS_BEIGE_STRIP, y, x, x + ww)
        below = _beige(y + hh, y + hh + PROGRESS_BEIGE_STRIP, x, x + ww)
        if above >= PROGRESS_BEIGE_MIN or below >= PROGRESS_BEIGE_MIN:
            return True
    return False


def _progress_bar_present(full_bgr):
    """Идёт ли добыча? Ищем зелёную ЗАЛИВКУ полосы прогресса «Добыча», рядом с которой
    (слева/справа) красный «остаток» отсчёта ЛИБО сверху/снизу бежевое тело окна.

    Полоса добычи наполовину зелёная, наполовину красная (убывает), поэтому у зелёной
    заливки почти всегда есть красный сосед. Трава так не выглядит: у зелёной травы
    вокруг снова трава (ни красного «остатка», ни бежевого тела) — она отсекается.
    Проверяем ПЕРВЫМ, чтобы никогда не отменить идущую добычу."""
    if full_bgr is None:
        return False
    H, W = full_bgr.shape[:2]
    try:
        map_cx = MAP_REGION["left"] + MAP_REGION["width"] // 2
        y_lo = MAP_REGION["top"] - 20
        y_hi = MAP_REGION["top"] + MAP_REGION["height"] + 20
    except Exception:
        map_cx, y_lo, y_hi = W // 2, 0, H
    hsv = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2HSV)
    gmask = cv2.inRange(hsv, np.array(PROGRESS_GREEN_LOW), np.array(PROGRESS_GREEN_HIGH))
    gmask = cv2.morphologyEx(gmask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    rmask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array(POPUP_RED_LOW1), np.array(POPUP_RED_HIGH1)),
        cv2.inRange(hsv, np.array(POPUP_RED_LOW2), np.array(POPUP_RED_HIGH2)))
    vch, sch = hsv[:, :, 2], hsv[:, :, 1]
    n, _, stats, cent = cv2.connectedComponentsWithStats(gmask, 8)

    def _frac(m, yy0, yy1, xx0, xx1):
        yy0, yy1 = max(0, yy0), min(H, yy1)
        xx0, xx1 = max(0, xx0), min(W, xx1)
        if yy1 <= yy0 or xx1 <= xx0:
            return 0.0
        sub = m[yy0:yy1, xx0:xx1]
        return float((sub > 0).sum()) / float(sub.size)

    def _beige(yy0, yy1, xx0, xx1):
        yy0, yy1 = max(0, yy0), min(H, yy1)
        xx0, xx1 = max(0, xx0), min(W, xx1)
        if yy1 <= yy0 or xx1 <= xx0:
            return 0.0
        v = vch[yy0:yy1, xx0:xx1]
        s = sch[yy0:yy1, xx0:xx1]
        return float(((v >= POPUP_BODY_V_MIN) & (s <= POPUP_BODY_S_MAX)).sum()) / float(v.size)

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx, cy = int(cent[i][0]), int(cent[i][1])
        if not (PROGRESS_FILL_H[0] <= h <= PROGRESS_FILL_H[1]):
            continue
        if w < PROGRESS_FILL_WMIN:
            continue
        if w / float(max(h, 1)) < PROGRESS_FILL_ASPECT:
            continue
        if area / float(max(w * h, 1)) < PROGRESS_FILL_SOLID:  # плотная (не текст)
            continue
        if not (y_lo <= cy <= y_hi):
            continue
        if abs(cx - map_cx) > POPUP_CENTER_XTOL + 90:          # у центра окна
            continue
        red_r = _frac(rmask, y, y + h, x + w, x + w + PROGRESS_SIDE_W)   # красный справа
        red_l = _frac(rmask, y, y + h, x - PROGRESS_SIDE_W, x)          # красный слева
        beige_a = _beige(y - PROGRESS_BEIGE_STRIP, y, x, x + w)         # бежевое сверху
        beige_b = _beige(y + h, y + h + PROGRESS_BEIGE_STRIP, x, x + w) # бежевое снизу
        if (max(red_r, red_l) >= PROGRESS_RED_ADJ_MIN or
                beige_a >= PROGRESS_BEIGE_MIN or beige_b >= PROGRESS_BEIGE_MIN):
            return True
    return False


def classify_center_window(full_bgr):
    """Классифицировать центральное окно. Возвращает (kind, point): 'none' (окна нет),
    'progress' (идёт добыча) или 'error' (окно-ошибка с «закрыть»). point — центр кнопки.

    ВАЖЕН ПОРЯДОК: сперва проверяем полосу прогресса (чтобы НИКОГДА не отменить идущую
    добычу), и только если её нет — ищем красную кнопку окна-ошибки."""
    if _progress_bar_present(full_bgr):
        return "progress", None
    b = _find_popup_button(full_bgr)
    if b is None:
        return "none", None
    cx, cy, w, h = b
    return "error", (cx, cy)


def _find_open_error_close(page):
    """Если сейчас открыто окно-ОШИБКА (без зелёной полосы прогресса «Добыча») —
    вернуть точку кнопки «закрыть», по которой надо кликнуть. Иначе None.

    Порядок: 1) идёт добыча (зелёная полоса) → это НЕ ошибка, вернём None (иначе
    отменим добычу); 2) ищем красную кнопку в полосе вокруг калиброванной точки;
    3) если в полосе не нашли, но прямо в калиброванной точке красно — кликнем её;
    4) точка не откалибрована → запасная геометрия (если включена)."""
    try:
        full = screenshot_bgr(page)
    except Exception:
        return None
    # ГЛАВНОЕ: тип окна определяем по КНОПКЕ окна, а зелёную полосу прогресса ищем
    # только ВНУТРИ тела окна — поэтому трава на карте больше не выдаётся за «добычу».
    kind, pt = classify_center_window(full)
    if kind == "progress":
        return None          # идёт добыча — не трогаем (иначе отменим)
    if kind == "error" and pt is not None:
        return pt
    # запас: кнопка «закрыть» в полосе вокруг калиброванной точки
    target = get_close_target()
    if target is not None:
        pt = find_close_button_near(full, target)
        if pt is not None and not _green_bar_in_region(
                full, pt[0] - POPUP_BODY_HALF_W, pt[1] - POPUP_BODY_UP,
                pt[0] + POPUP_BODY_HALF_W, pt[1]):
            return pt
        if _red_fraction_at(full, target[0], target[1]) >= POPUP_RED_FRAC:
            return tuple(int(v) for v in target)
    if target is None and POPUP_USE_GEOMETRY:
        return find_popup_close(full)
    return None


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
    'none' (окна нет) или 'unknown' (нельзя определить).

    Тип окна определяем по КНОПКЕ окна, а зелёную полосу прогресса ищем ТОЛЬКО внутри
    тела окна — трава на карте больше не выдаётся за «добычу» (это и была причина, по
    которой бот не закрывал окна-ошибки на травяных локациях)."""
    try:
        full = screenshot_bgr(page)
    except Exception:
        return "unknown"
    kind, _ = classify_center_window(full)
    if kind != "none":
        return kind
    # запас: кнопка в полосе вокруг калиброванной точки (если центр-эвристика не сошлась)
    target = get_close_target()
    if target is not None:
        if find_close_button_near(full, target) is not None:
            if _green_bar_in_region(full, target[0] - POPUP_BODY_HALF_W,
                                    target[1] - POPUP_BODY_UP,
                                    target[0] + POPUP_BODY_HALF_W, target[1]):
                return "progress"
            return "error"
        if _red_fraction_at(full, target[0], target[1]) >= POPUP_RED_FRAC:
            return "error"
    if target is None and POPUP_USE_GEOMETRY and find_popup_close(full) is not None:
        return "error"
    return "none"


def _click_close_verified(page, first_pt):
    """Кликнуть по кнопке «закрыть» и ПРОВЕРИТЬ, что окно исчезло. Если нет —
    повторить (окна разной высоты: могло вылезти следующее или клик чуть промахнулся;
    каждый раз ищем кнопку заново). Возвращает True, если попытка была сделана."""
    pt = first_pt
    for _ in range(max(1, POPUP_CLOSE_TRIES)):
        click_point(page, pt)
        log.info("Закрыл окно-ошибку (кнопка «закрыть» %s).",
                 tuple(int(v) for v in pt))
        time.sleep(random.uniform(0.35, 0.6))
        nxt = _find_open_error_close(page)   # окно ещё висит? уточним точку
        if nxt is None:
            return True
        pt = nxt
    # калиброванным путём не закрылось — последняя попытка через DOM
    if _close_via_dom(page):
        return True
    log.warning("Окно-ошибку не удалось закрыть за %d попыт(ки). Проверь калибровку "
                "точки «закрыть» (python omela_bg.py --calib, этап 3).", POPUP_CLOSE_TRIES)
    return True


def close_if_blocking(page):
    """Закрыть ТОЛЬКО окно-ошибку (с «закрыть»). Окно добычи (с зелёной полосой) не
    трогаем. Кнопку ищем в ПОЛОСЕ вокруг калиброванной точки, кликаем по её
    фактическому центру и проверяем, что окно закрылось."""
    pt = _find_open_error_close(page)
    if pt is None:
        return False
    return _click_close_verified(page, pt)


def close_blocking_popup(page):
    """Закрыть окно-ошибку. Кнопку «закрыть» ищем в ПОЛОСЕ вокруг калиброванной
    точки (окна разной высоты сдвигают кнопку по вертикали), кликаем по её
    фактическому центру и ПРОВЕРЯЕМ, что окно исчезло; если нет — повторяем. Окно
    добычи (с зелёной полосой прогресса) не трогаем."""
    pt = _find_open_error_close(page)
    if pt is not None:
        return _click_close_verified(page, pt)
    # точка «закрыть» не откалибрована и геометрия выключена — последний шанс через DOM
    if get_close_target() is None and not POPUP_USE_GEOMETRY:
        return _close_via_dom(page)
    return False


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
            if z.get("blocks"):                          # несколько зон блока
                block = [tuple(p) for p in z["blocks"]]
            elif z.get("block"):                         # одна зона (старый формат)
                block = [tuple(z["block"])]
            if z.get("attacks"):                         # несколько зон атаки
                attack = [tuple(p) for p in z["attacks"]]
            elif z.get("attack"):
                attack = [tuple(z["attack"])]
            if z.get("exit"):
                exit_t = tuple(z["exit"])
            if z.get("hunt"):
                hunt_t = tuple(z["hunt"])
    return block, attack, exit_t, hunt_t


def get_fight_modes():
    """Режимы выбора зоны: (атака, блок). Значение из fight_zones.json важнее дефолта."""
    z = load_zones() or {}
    am = z.get("attack_mode") or FIGHT_ATTACK_MODE
    bm = z.get("block_mode") or FIGHT_BLOCK_MODE
    return am, bm


def _pick_zone(zones_list, mode, idx):
    """Выбрать точку из списка: 'cycle' — по кругу (idx), иначе — случайно."""
    if not zones_list:
        return None
    if len(zones_list) == 1:
        return zones_list[0]
    if mode == "cycle":
        return zones_list[idx % len(zones_list)]
    return random.choice(zones_list)


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
    atk_mode, blk_mode = get_fight_modes()
    if len(attack) > 1 or len(block) > 1:
        log.info("Зоны: атака=%d (%s), блок=%d (%s).",
                 len(attack), atk_mode, len(block), blk_mode)
    end_baseline = _count_text_in_frames(page, "Окончен бой")
    rounds = 0
    while in_fight(page) and rounds < FIGHT_MAX_ROUNDS:
        rounds += 1
        bp = _pick_zone(block, blk_mode, rounds - 1)
        ap = _pick_zone(attack, atk_mode, rounds - 1)
        if bp:
            click_point(page, bp)
            time.sleep(random.uniform(0.15, 0.35))
        if ap:
            click_point(page, ap)
        log.info("Раунд #%d: блок %s + атака %s.", rounds,
                 tuple(bp) if bp else "—", tuple(ap) if ap else "—")
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


def annotate_detection(crop_bgr, cands, path):
    """Нарисовать на кадре карты найденные пятна: зелёный кружок = принято,
    красный = отсеяно (с подписью причины). Помогает подобрать чувствительность."""
    try:
        img = crop_bgr.copy()
        for c in cands:
            ok = c["reject"] is None
            color = (0, 200, 0) if ok else (0, 0, 230)
            r = max(6, int(max(c["w"], c["h"]) / 2) + 3)
            cv2.circle(img, (c["cx"], c["cy"]), r, color, 2)
            tag = ("%.0f" % c["score"]) if ok else c["reject"]
            cv2.putText(img, tag, (c["cx"] + r, c["cy"]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.imwrite(path, img)
    except Exception as e:
        log.warning("Не удалось сохранить аннотированный кадр: %s", e)


def save_color_mask(crop_bgr, path):
    """Сохранить бинарную цветовую маску активной профессии (что подходит по цвету)."""
    try:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mask = None
        for lo, hi in RESOURCE_RANGES:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        if mask is not None:
            cv2.imwrite(path, mask)
    except Exception as e:
        log.warning("Не удалось сохранить маску: %s", e)


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
        log.info("Профессия: %s (%s). Диапазонов цвета: %d. Исключений: %d. Чувств.: %.2f.",
                 ACTIVE_PROF, PROFESSIONS[ACTIVE_PROF]["title"], len(RESOURCE_RANGES),
                 len(EXCLUDE_RANGES), DETECT_SENSITIVITY)
        # --- визуальная диагностика распознавания ---
        crop = crop_map(full)
        cands = find_resource_scored(crop, debug=True)
        accepted = [c for c in cands if c["reject"] is None]
        annotate_detection(crop, cands, "detect_%s.png" % stamp)
        save_color_mask(crop, "mask_%s.png" % stamp)
        log.info("Найдено ресурсов (%s): ПРИНЯТО %d из %d кандидатов.",
                 ACTIVE_PROF, len(accepted), len(cands))
        # что отсеяли и почему (топ причин)
        reasons = {}
        for c in cands:
            if c["reject"]:
                reasons[c["reject"]] = reasons.get(c["reject"], 0) + 1
        if reasons:
            log.info("Отсеяно по причинам: %s",
                     ", ".join("%s=%d" % (k, v) for k, v in sorted(reasons.items())))
        for c in accepted[:8]:
            log.info("  ✔ (%d,%d) score=%.1f contrast=%.1f solid=%.2f circ=%.2f sat=%.0f",
                     c["cx"], c["cy"], c["score"], c["contrast"], c["solidity"],
                     c["circularity"], c["sat"])
        log.info("Файлы: page_full_%s.png, detect_%s.png (зелёные=принято, красные=отсев), "
                 "mask_%s.png (цветовая маска), page_dom_%s.html.",
                 stamp, stamp, stamp, stamp)
        dump_fight(page, stamp)
        ctx.close()


# =========================================================================
#                                ЗАНОЗА
# =========================================================================

def alert_beep(n=4):
    """Позвать игрока звуком. На Windows — winsound, иначе — системный «бип»."""
    if not SPLINTER_ALERT_SOUND:
        return
    try:
        import winsound
        for _ in range(n):
            winsound.Beep(1000, 300)
            time.sleep(0.12)
    except Exception:
        for _ in range(n):
            print("\a", end="", flush=True)
            time.sleep(0.2)


def _is_my_splinter_line(line):
    """True, если строка — СИСТЕМНОЕ сообщение игры о МОЕЙ занозе (а не чужой ник,
    не личное/чужое сообщение, не просьба и не строка лечения)."""
    import re
    if not line:
        return False
    low = line.lower()
    if SPLINTER_CHAT_KEYWORD not in low:          # нет слова «заноз» — точно не то
        return False
    if "»" in line or ">>" in line:               # «Ник » Ник: …» — личное/чужое, пропускаем
        return False
    for bad in SPLINTER_HEAL_WORDS:               # «Вы избавились от занозы» — это лечение
        if bad in low:
            return False
    for bad in SPLINTER_REQUEST_WORDS:            # «дерните/вытащите занозу» — просьба
        if bad in low:
            return False
    # (а) НАДЁЖНЫЙ путь: системная строка-награда «Получено: Заноза …» (только у тебя).
    # Нормализуем пробелы/двоеточие, чтобы «Получено: Заноза» и «Получено  Заноза»
    # ловились одинаково.
    compact = re.sub(r"[:\s]+", " ", low).strip()
    for mark in SPLINTER_SELF_MARKERS:
        mk = re.sub(r"[:\s]+", " ", mark.lower()).strip()
        if mk and mk in compact:
            return True
    # (б) ЗАПАСНОЙ путь: строка начинается с «Вы …» / «У вас …» / «Вам …».
    body = re.sub(r"^\s*\d{1,2}:\d{2}\s*", "", low)   # отрезать ведущее время «13:47 »
    body = body.lstrip()
    return any(body.startswith(pfx) for pfx in SPLINTER_SELF_PREFIXES)


def chat_splinter_count(page):
    """Сколько СИСТЕМНЫХ строк «Вы …заноз…» (моя заноза) сейчас видно в чате.
    Считаем ТОЛЬКО свою занозу — ники игроков, чужие сообщения и просьбы бота
    (в них есть «»») и строку лечения не учитываем."""
    try:
        text = read_chat_text(page)
    except Exception:
        text = ""
    if not text:
        return 0
    return sum(1 for line in text.splitlines() if _is_my_splinter_line(line))


def read_chat_text(page):
    """Собрать видимый текст чата (для разбора, кто вытащил занозу)."""
    chunks = []
    for fr in _all_frames(page):
        u = (getattr(fr, "url", "") or "")
        nm = (getattr(fr, "name", "") or "")
        if "cht" not in u and "chat" not in nm:
            continue
        try:
            loc = fr.locator("#content")
            if loc.count() > 0:
                chunks.append(loc.first.inner_text(timeout=1000))
        except Exception:
            continue
    return "\n".join(chunks)


def parse_splinter_helper(text, my_name=""):
    """Из строки лечения достать НИК помощника: «…аптечку, ИМЯ [ур] избавляет
    воина ТВОЙ_НИК от занозы». Возвращает ник или None."""
    import re
    if my_name:
        m = re.search(r"аптечк[^,]*,\s*(.+?)\s*\[\d+\]\s*избавля\w+\s+воин\w*\s+"
                      + re.escape(my_name), text)
        if m:
            return m.group(1).strip() or None
    m = re.search(r"аптечк[^,]*,\s*(.+?)\s*\[\d+\]\s*избавля", text)
    if not m:
        m = re.search(r"([^\n]+?)\s*\[\d+\]\s*избавля\w+\s+воин", text)
    if not m:
        return None
    name = m.group(1).strip()
    if "," in name:                    # отрезать возможный префикс «Применив …,»
        name = name.split(",")[-1].strip()
    return name or None


def post_chat(page, text):
    """Написать text в игровой чат: заполнить поле ввода и отправить (Enter/кнопка)."""
    for fr in _all_frames(page):
        try:
            inp = fr.locator("#message")
            if inp.count() == 0:
                continue
            el = inp.first
            if not el.is_visible():
                continue
            el.click(timeout=1500)
            el.fill(text)
            time.sleep(0.2)
            el.press("Enter")
            time.sleep(0.3)
            # если текст остался в поле — Enter не отправил, жмём кнопку «отправить»
            try:
                if (el.input_value() or "").strip():
                    sb = fr.locator("#send_btn")
                    if sb.count() > 0:
                        sb.first.click(timeout=1500)
            except Exception:
                pass
            return True
        except Exception:
            continue
    return False


def _user_frame(page):
    """Фрейм со списком игроков локации (cht_iframe.php?mode=user)."""
    for fr in _all_frames(page):
        u = (getattr(fr, "url", "") or "")
        nm = (getattr(fr, "name", "") or "")
        if "mode=user" in u or nm == "chat_user":
            return fr
    return None


def pm_all_in_location(page, message, my_name=""):
    """Написать ЛИЧНО каждому игроку в локации, впечатывая префикс prv[ник].
    (Формат — SPLINTER_PM_FORMAT.) Себя пропускаем. Возвращает число отправленных."""
    nicks = location_nicks(page)
    sent = 0
    for nick in nicks:
        if my_name and nick.lower() == my_name.lower():
            continue                      # себе не пишем
        line = SPLINTER_PM_FORMAT.format(nick=nick, msg=message)
        if post_chat(page, line):
            sent += 1
            time.sleep(random.uniform(0.5, 1.0))
    return sent


def pm_one_in_location(page, nick, message):
    """Написать ЛИЧНО одному игроку (впечатать prv[ник] текст). True — отправлено."""
    if not nick:
        return False
    line = SPLINTER_PM_FORMAT.format(nick=nick, msg=message)
    return post_chat(page, line)


def location_nicks(page):
    """Список ников игроков в локации (без уровня «[NN]» и служебных строк)."""
    import re
    fr = _user_frame(page)
    out = []
    if fr is None:
        return out
    try:
        items = fr.locator(".chat_user_item")
        n = items.count()
    except Exception:
        return out
    for i in range(n):
        try:
            it = items.nth(i)
            pn = it.locator(".pnick")
            raw = (pn.first.inner_text(timeout=800) if pn.count() > 0
                   else it.inner_text(timeout=800)) or ""
        except Exception:
            continue
        line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        if not line or line.startswith("- - -"):
            continue
        nick = re.sub(r"\s*\[\d+\].*$", "", line).strip()   # убрать «[20]…»
        if nick:
            out.append(nick)
    return out


def find_helper_nick(page, healing_text):
    """Кто вытащил занозу: ник из списка локации, который встречается в строке
    лечения (надёжнее разбора текста — не цепляет титул). Фолбэк — разбор строки."""
    for nick in location_nicks(page):
        if nick and nick.lower() in healing_text.lower():
            return nick
    return parse_splinter_helper(healing_text, SPLINTER_MY_NAME)


def request_splinter_help(page):
    """Попросить вытащить занозу: лично каждому (SPLINTER_PM_EACH) или в общий чат."""
    if SPLINTER_PM_EACH:
        sent = pm_all_in_location(page, SPLINTER_MESSAGE, SPLINTER_MY_NAME)
        if sent:
            log.info("Личная просьба разослана игрокам локации: %d.", sent)
            return True
        log.info("Не нашёл игроков для лички — пишу в общий чат.")
    return post_chat(page, SPLINTER_MESSAGE)


def get_reequip_targets():
    """Точки авто-возврата инструмента из калибровки (dict): bag, tab, pick, equip, hunt_mode."""
    z = load_zones() or {}
    return {k: z.get(k) for k in ("bag", "tab", "pick", "equip", "hunt_mode")}


def get_craft_targets():
    """Точки авто-крафта рецептов из калибровки (Этап 6):
    open  — кнопка, открывающая панель профессии;
    creates — список кнопок «Создать» (по одной на рецепт);
    back  — кнопка «Вернуться» к сбору."""
    z = load_zones() or {}
    return {
        "open":    z.get("craft_open"),
        "creates": z.get("craft_creates") or [],
        "back":    z.get("craft_back"),
    }


def craft_ready():
    """Настроен ли авто-крафт (есть кнопка открытия панели и хотя бы одна «Создать»)."""
    if not CRAFT_ENABLED:
        return False
    t = get_craft_targets()
    return bool(t["open"] and t["creates"])


def craft_due():
    """Подошло ли время авто-крафта (проверяется и внутри цикла сбора, чтобы не
    опаздывать на длинных добычах)."""
    return bool(_CRAFT_ON and _NEXT_CRAFT is not None and time.time() >= _NEXT_CRAFT)


def craft_recipes(page):
    """Зайти в панель профессии и нажать «Создать» у каждого рецепта.
    После каждого «Создать» экран сам перезагружается — ждём и жмём следующий.
    В конце — «Вернуться» к сбору. True — выполнено, False — не настроено."""
    t = get_craft_targets()
    if not (t["open"] and t["creates"]):
        log.info("Авто-крафт не настроен (Этап 6 калибровки) — пропускаю.")
        return False
    # если висит окно-ошибка — сначала закрыть, иначе клики уйдут «в молоко»
    close_if_blocking(page)
    log.info("Авто-крафт: открываю панель профессии…")
    click_point(page, t["open"])
    time.sleep(random.uniform(*CRAFT_OPEN_WAIT))
    made = 0
    for i, pt in enumerate(t["creates"], 1):
        time.sleep(random.uniform(*CRAFT_CLICK_GAP))
        click_point(page, pt)                       # нажать «Создать» у рецепта #i
        made += 1
        log.info("  → «Создать» рецепт #%d %s — жду перезагрузку экрана.", i, pt)
        # экран сам перезагружается после «Создать» — ждём, потом следующий рецепт
        time.sleep(random.uniform(*CRAFT_RELOAD_WAIT))
        close_if_blocking(page)
    if t["back"]:
        click_point(page, t["back"])                # «Вернуться» к сбору
        time.sleep(random.uniform(*CRAFT_AFTER_BACK_WAIT))
    log.info("Авто-крафт: нажал «Создать» %d раз, вернулся к сбору.", made)
    return True


def reequip_tool(page):
    """Рюкзак → вкладка «вещи» → навести на кирку → клик «надеть» → режим охоты.
    Работает только если точки откалиброваны (Этап 5). Возвращает False, если нет."""
    t = get_reequip_targets()
    if not (t["bag"] and t["pick"] and t["equip"]):
        return False
    click_point(page, t["bag"])                  # открыть рюкзак
    time.sleep(random.uniform(0.9, 1.5))
    if t["tab"]:                                 # перейти на вкладку «вещи»
        click_point(page, t["tab"])
        time.sleep(random.uniform(0.7, 1.2))
    # навести на кирку, чтобы появилась надпись «надеть»
    page.mouse.move(int(t["pick"][0]), int(t["pick"][1]))
    time.sleep(SPLINTER_HOVER_WAIT)
    click_point(page, t["equip"])                # кликнуть «надеть»
    time.sleep(random.uniform(0.8, 1.4))
    if t["hunt_mode"]:                           # режим охоты
        click_point(page, t["hunt_mode"])
        time.sleep(random.uniform(0.9, 1.5))
    return True


def try_reequip_verified(page, tries=2):
    """Попытаться надеть кирку и ПРОВЕРИТЬ, что добыча заработала. True — получилось."""
    t = get_reequip_targets()
    if not (t["bag"] and t["pick"] and t["equip"]):
        log.info("Авто-возврат кирки не настроен (Этап 5 калибровки) — верни вручную.")
        return False
    for attempt in range(tries):
        log.info("Надеваю кирку из рюкзака (попытка %d/%d)…", attempt + 1, tries)
        reequip_tool(page)
        time.sleep(1.2)
        close_if_blocking(page)
        if _test_can_gather(page) is True:
            return True
    return False


def _test_can_gather(page):
    """Проверить, вернулся ли инструмент: пробуем добыть один ресурс и смотрим окно.
    'progress' → добыча пошла (инструмент в руках) → True. 'error' → закрываем окно,
    инструмента нет → False. Ресурса не видно → None (проверить не удалось)."""
    try:
        pts = find_resource(crop_map(screenshot_bgr(page)))
    except Exception:
        return None
    if not pts:
        return None
    px, py = map_to_page(*pts[0])
    gather_click(page, px, py)
    time.sleep(1.2)
    kind = window_kind(page)
    if kind == "progress":
        return True
    if kind == "error":
        close_blocking_popup(page)
        return False
    return False


def handle_splinter(page):
    """Обработать занозу: пауза сбора, просьба в чат (+повтор), звук; ждём, пока
    игрок вернёт инструмент и добыча снова заработает."""
    # для прозрачности — покажем, на какую именно системную строку среагировали
    try:
        trig = [ln.strip() for ln in read_chat_text(page).splitlines()
                if _is_my_splinter_line(ln)]
        if trig:
            log.warning("🩹 ЗАНОЗА! Сработала строка: «%s»", trig[-1])
    except Exception:
        pass
    log.warning("🩹 ЗАНОЗА! Инструмент убран в рюкзак — сбор на паузе.")
    request_splinter_help(page)
    log.warning(">>> ВЕРНИ ИНСТРУМЕНТ В РУКИ (надень кирку → режим охоты), когда занозу "
                "вытащат. Бот ждёт, зовёт сигналом и сам продолжит, когда добыча пойдёт.")
    alert_beep(6)
    start = time.time()
    last_post = time.time()
    thanked = False
    reequip_tried = False
    while True:
        if (time.time() - start) > SPLINTER_MAX_WAIT:
            log.warning("Заноза не вылечена за %.0f мин — пробую продолжить сбор.",
                        SPLINTER_MAX_WAIT / 60.0)
            return
        # бой во время ожидания не пропускаем
        if FIGHT_ENABLED and in_fight(page):
            do_fight(page)
        # на всякий случай закрываем окна-ошибки
        close_if_blocking(page)

        # занозу вытащили? («Вы избавились от занозы» / «…избавляет воина … от занозы»)
        if not thanked:
            try:
                txt = read_chat_text(page)
            except Exception:
                txt = ""
            low = txt.lower()
            if "избавил" in low and "заноз" in low:
                helper = find_helper_nick(page, txt)
                if SPLINTER_THANKS and helper:
                    msg = SPLINTER_THANKS_MESSAGE.format(name=helper)
                    # благодарим ЛИЧНО помощнику; если не вышло — в общий чат
                    ok = pm_one_in_location(page, helper, msg) or post_chat(page, msg)
                    log.info("Занозу вытащил(а) %s — благодарность %s.", helper,
                             "отправлена" if ok else "НЕ отправилась")
                else:
                    log.info("Занозу вытащили (имя помощника не распознал — поблагодари сам(а)).")
                thanked = True

        # занозу вытащили → пробуем сами надеть кирку и включить охоту
        if thanked and not reequip_tried and SPLINTER_REEQUIP:
            reequip_tried = True
            if try_reequip_verified(page):
                log.info("✅ Надел кирку, включил охоту — продолжаю сбор.")
                return
            log.warning(">>> Не смог надеть кирку сам — верни её в руки вручную. Бот ждёт и зовёт.")
            alert_beep(5)

        # повторяем просьбу и зовём звуком (пока не вытащили)
        if not thanked and (time.time() - last_post) > SPLINTER_REPEAT_EVERY:
            request_splinter_help(page)
            alert_beep(3)
            last_post = time.time()
            log.info("Повторил просьбу (заноза ещё не вылечена).")
        # вернулся ли инструмент в руки?
        res = _test_can_gather(page)
        if res is True:
            log.info("✅ Инструмент снова в руках — продолжаю сбор.")
            return
        time.sleep(SPLINTER_POLL)


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


def _capture_multi(ctx, page, clicks, label, max_n=8):
    """Снять НЕСКОЛЬКО точек подряд: кликай зону → ENTER, повторяй.
    Пустой ENTER — закончить этот список. Возвращает список [[x,y], ...]."""
    pts = []
    print("\n>>> %s" % label)
    print(">>> Кликай по ОДНОЙ зоне и жми ENTER. Пустой ENTER (без клика) — закончить.\n",
          flush=True)
    while len(pts) < max_n:
        _install_calib(page)
        clicks.clear()
        print(">>> Зона #%d — кликни и ENTER (пустой ENTER — закончить):" % (len(pts) + 1),
              flush=True)
        wait_enter_keep_alive(ctx)
        if clicks:
            pt = clicks[-1]
            pts.append([pt[0], pt[1]])
            log.info("  → зона #%d (%d, %d)", len(pts), pt[0], pt[1])
        else:
            break
    if not pts:
        log.info("  → пропущено.")
    return pts


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

        print("\n==================== МАСТЕР КАЛИБРОВКИ (настройка бота) ====================")
        print("Мы вместе пройдём 7 шагов и настроим бота под твою игру.")
        print("")
        print("КАК ЭТО РАБОТАЕТ:")
        print("  - На каждом шаге бот пишет, ЧТО именно мы сейчас настраиваем и что делать.")
        print("  - Обычно нужно КЛИКНУТЬ мышкой в игре по нужному месту, потом нажать ENTER.")
        print("  - Любой шаг можно ПРОПУСТИТЬ - просто нажми ENTER, ничего не кликая.")
        print("  - Всё, что настроим, сохранится в файл fight_zones.json.")
        print("  - Настроил не так? Запусти калибровку заново - старое не потеряется.")
        print("")
        print("КРАТКО про 7 шагов:")
        print("  ШАГ 1 - профессия и какие ресурсы СОБИРАТЬ / какие ИГНОРИРОВАТЬ.")
        print("  ШАГ 2 - где на экране искать ресурсы (границы карты).")
        print("  ШАГ 3 - сколько кликов запускают добычу.")
        print("  ШАГ 4 - кнопка «закрыть» у окна-ошибки.")
        print("  ШАГ 5 - зоны боя (блок / атака / выход).")
        print("  ШАГ 6 - авто-возврат кирки после занозы.")
        print("  ШАГ 7 - авто-крафт рецептов.")
        print("===========================================================================\n")

        # =============================== ШАГ 1 из 7 ===============================
        print("\n###########################################################################")
        print("#  ШАГ 1 из 7  -  ПРОФЕССИЯ И КАКИЕ РЕСУРСЫ СОБИРАТЬ")
        print("###########################################################################")
        print(">>> ЧТО СЕЙЧАС ДЕЛАЕМ: выбираем профессию (травник / геолог / рыбак), а")
        print(">>> потом покажем боту, какие ресурсы СОБИРАТЬ и какие ИГНОРИРОВАТЬ.")
        print("")
        print(">>> 1.1 Сначала выбери ПРОФЕССИЮ - введи её номер или имя из списка ниже:")
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
        print("\n---------------------------------------------------------------------------")
        print(">>> 1.2  «ПИПЕТКА СБОРА» — ЧТО БОТ БУДЕТ СОБИРАТЬ")
        print("---------------------------------------------------------------------------")
        print(">>> Простыми словами: ты кликаешь прямо ПО РЕСУРСУ в игре, бот запоминает")
        print(">>> его ЦВЕТ и потом ищет и добывает только такие пятна этого цвета.")
        print(">>> Это самый надёжный способ настроить сбор — точнее готового пресета.")
        print(">>> ")
        print(">>> ПРАВИЛО: кликай ТОЛЬКО по тому, что ХОЧЕШЬ собирать. Один клик — один")
        print(">>> цвет. Если ресурс бывает РАЗНОГО цвета (камни: красный, синий, зелёный")
        print(">>> и т.д.) — сними по одному образцу НА КАЖДЫЙ цвет, они суммируются.")
        print(">>> ")
        print(">>> КАК ДЕЛАТЬ ПОШАГОВО:")
        print(">>>   1) реши, сколько образцов снять (1 — если ресурс одного цвета;")
        print(">>>      2-4 — если цветов несколько), и введи это число ниже;")
        print(">>>   2) на каждый образец: наведись и кликни ОДИН раз ровно в ЦЕНТР")
        print(">>>      ресурса (не по краю, не по траве рядом!), затем нажми ENTER.")
        print(">>> ")
        print(">>> СОВЕТ: бери яркий, крупный, хорошо видимый экземпляр — по нему цвет")
        print(">>> снимется чище. По тусклому/наполовину скрытому — цвет выйдет смазанным.")
        print(">>> Введёшь 0 — вручную цвет не снимаем, бот возьмёт готовый цвет профессии")
        print(">>> (для травы обычно ок; для камней/рыбы лучше снять пипеткой).")
        ans = read_line_keep_alive(
            ctx, ">>> Сколько ресурсов-образцов СНЯТЬ? (0 - оставить готовый цвет; по умолчанию 0): ")
        try:
            n_samples = int(ans) if ans else 0
        except ValueError:
            n_samples = 0
        if n_samples > 0:
            collected = []
            for i in range(n_samples):
                pt = _capture_point(
                    ctx, page, clicks,
                    "образец #%d — кликни по РЕСУРСУ, который НУЖНО собирать "
                    "(ENTER без клика — закончить)" % (i + 1))
                if not pt:
                    break
                try:
                    full = screenshot_bgr(page)
                    rngs = sample_hsv_ranges_at(full, pt[0], pt[1])
                except Exception as ex:
                    log.warning("Пипетка не сработала: %s", ex)
                    rngs = []
                # ЗАЩИТА ОТ СНЯТИЯ ФОНА: если снятый цвет заливает бо́льшую часть карты,
                # это трава/фон, а не ресурс. Сохранять его нельзя — иначе бот перестанет
                # собирать (маска на всю карту → компактных пятен нет → только крутит).
                if rngs:
                    mrgn = zones.get("map_region")
                    mrgn = ({"left": mrgn[0], "top": mrgn[1],
                             "width": mrgn[2], "height": mrgn[3]}
                            if mrgn and len(mrgn) == 4 else None)
                    cov = color_coverage_over_map(full, rngs, mrgn)
                    if cov > BG_COVERAGE_MAX:
                        log.warning("  ⚠ Этот цвет покрывает %.0f%% карты — это похоже на "
                                    "ТРАВУ/ФОН, а не на ресурс. НЕ сохраняю образец.", cov * 100)
                        print(">>> ⚠ ПРОПУЩЕНО: снятый цвет заливает почти всю карту — это фон,")
                        print(">>>   а не камень. Кликни ровно по ЦЕНТРУ яркого камня и повтори.")
                        print(">>>   (Если хочешь просто вернуть встроенный цвет геолога — введи 0.)")
                        rngs = []
                if rngs:
                    collected.extend(rngs)
                    log.info("  Снят цвет (HSV-диапазон): %s", rngs)
                else:
                    log.info("  Цвет не снят — попробуй кликнуть точнее по яркому центру ресурса.")
            if collected:
                zones["resource_ranges"] = collected
                log.info("Пипетка: сохранено диапазонов цвета: %d.", len(collected))
            else:
                log.info("Пипетка: образцы не сняты — останется пресет профессии.")

        # активируем профессию/цвет ресурса УЖЕ СЕЙЧАС, чтобы следующий шаг (игнор)
        # мог проверять: не совпадает ли выбранный «фон» с цветом РЕСУРСА.
        set_active_profession(zones["profession"], custom_ranges=zones.get("resource_ranges"))

        # ПИПЕТКА-ИСКЛЮЧЕНИЕ — снять цвета фона/травы/чужого, по которым НЕ кликать
        print("\n---------------------------------------------------------------------------")
        print(">>> 1.3  «ПИПЕТКА ИГНОРА» — ЧТО БОТ ДОЛЖЕН ПРОПУСКАТЬ (по желанию)")
        print("---------------------------------------------------------------------------")
        print(">>> Это ОБРАТНАЯ пипетка: тут кликают по тому, что бот трогать НЕ должен.")
        print(">>> Снятые цвета идут в ЧЁРНЫЙ СПИСОК — по таким пятнам бот кликать не будет.")
        print(">>> ")
        print(">>> НУЖНО ТОЛЬКО ЕСЛИ бот ошибается — тыкает по ТРАВЕ, ФОНУ или по ЧУЖИМ")
        print(">>> ресурсам (не твоей профессии). Тогда кликай прямо по этой траве/фону.")
        print(">>> Пример: бот путает жёлтую траву с омелой — кликни пару раз по такой")
        print(">>> траве, и он перестанет её собирать.")
        print(">>> ")
        print(">>> ┌─ ГЛАВНОЕ ПРАВИЛО, ЧТОБЫ НЕ СЛОМАТЬ СБОР ─────────────────────────┐")
        print(">>> │ НИКОГДА не кликай здесь по ресурсу, который САМ СОБИРАЕШЬ!         │")
        print(">>> │ Занесёшь его цвет в игнор — бот перестанет его собирать и будет   │")
        print(">>> │ только крутить карту. Кликай ТОЛЬКО по траве/фону/чужому.         │")
        print(">>> │ (Бот подстрахует и не даст занести цвет ресурса, но не рискуй.)   │")
        print(">>> └──────────────────────────────────────────────────────────────────┘")
        print(">>> ")
        print(">>> ЧТО ВВЕСТИ НИЖЕ:")
        print(">>>   0   — пропустить шаг (всё и так нормально) — так по умолчанию;")
        print(">>>   1-4 — снять столько образцов фона/травы, по которым не кликать;")
        print(">>>   c   — ОЧИСТИТЬ игнор. Введи 'c', если бот вообще НЕ собирает ресурс —")
        print(">>>         частая причина именно в том, что в игнор попало лишнее.")
        prev_excl = zones.get("exclude_ranges", [])
        if prev_excl:
            print(">>> Сейчас уже занесено в чёрный список цветов: %d." % len(prev_excl))
        ans = read_line_keep_alive(
            ctx, ">>> Сколько образцов для ИГНОРА снять? (0 - пропустить; 'c' - очистить старые): ")
        if ans.strip().lower() == "c":
            zones.pop("exclude_ranges", None)
            log.info("Список цветов-исключений очищен.")
        else:
            try:
                n_excl = int(ans) if ans else 0
            except ValueError:
                n_excl = 0
            if n_excl > 0:
                excl = list(prev_excl)
                for i in range(n_excl):
                    pt = _capture_point(
                        ctx, page, clicks,
                        "образец #%d — кликни по тому, что НУЖНО ИГНОРИРОВАТЬ "
                        "(трава/фон/чужой ресурс; ENTER без клика — закончить)" % (i + 1))
                    if not pt:
                        break
                    try:
                        full = screenshot_bgr(page)
                        rngs = sample_exclude_ranges_at(full, pt[0], pt[1])
                    except Exception as ex:
                        log.warning("Пипетка-исключение не сработала: %s", ex)
                        rngs = []
                    # ЗАЩИТА: если кликнули по цвету РЕСУРСА, не заносим его в игнор —
                    # иначе бот перестанет собирать этот ресурс (частая ошибка).
                    if rngs and _color_is_resource(full, pt[0], pt[1]):
                        log.warning("  ⚠ Этот цвет совпадает с РЕСУРСОМ (который ты собираешь) — "
                                    "НЕ добавляю в игнор. Кликай по ТРАВЕ/ФОНУ, а не по камню.")
                        print(">>> ⚠ ПРОПУЩЕНО: похоже, ты кликнул(а) по нужному ресурсу, а не по фону.")
                        print(">>>   Этот цвет НЕ занесён в игнор, иначе бот перестал бы его собирать.")
                        rngs = []
                    if rngs:
                        excl.extend(rngs)
                        log.info("  Исключён цвет (HSV): %s", rngs)
                if excl:
                    zones["exclude_ranges"] = excl
                    log.info("Пипетка-исключение: всего цветов-исключений: %d.", len(excl))

        # применить выбор сразу, чтобы дальнейшие этапы работали с нужным ресурсом
        set_active_profession(zones["profession"], custom_ranges=zones.get("resource_ranges"))

        # 2) ОБЛАСТЬ КАРТЫ
        print("\n###########################################################################")
        print("#  ШАГ 2 из 7  -  ГРАНИЦЫ КАРТЫ (где искать ресурсы)")
        print("###########################################################################")
        print(">>> ЧТО СЕЙЧАС ДЕЛАЕМ: очерчиваем прямоугольник зелёной карты. Бот будет")
        print(">>> искать ресурсы ТОЛЬКО внутри него и не полезет в интерфейс по краям.")
        print(">>> ")
        print(">>> КАК ДЕЛАТЬ: кликни по ДВУМ углам карты по очереди -")
        print(">>>   сначала левый-верхний угол, потом правый-нижний.")
        p1 = _capture_point(ctx, page, clicks, "ЛЕВЫЙ-ВЕРХНИЙ угол зелёной карты")
        p2 = _capture_point(ctx, page, clicks, "ПРАВЫЙ-НИЖНИЙ угол зелёной карты")
        if p1 and p2:
            left, top = min(p1[0], p2[0]), min(p1[1], p2[1])
            width, height = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
            if width > 100 and height > 100:
                zones["map_region"] = [left, top, width, height]
                log.info("Область карты: %s", zones["map_region"])

        # 3) СПОСОБ ДОБЫЧИ
        print("\n###########################################################################")
        print("#  ШАГ 3 из 7  -  СКОЛЬКО КЛИКОВ ЗАПУСКАЮТ ДОБЫЧУ")
        print("###########################################################################")
        print(">>> ЧТО СЕЙЧАС ДЕЛАЕМ: говорим боту, сколько раз кликнуть по ресурсу, чтобы")
        print(">>> началась добыча. В этой игре обычно НУЖНО 2 клика.")
        print(">>> Тут кликать в игре НЕ надо - просто введи число в этом окне.")
        ans = read_line_keep_alive(
            ctx, ">>> Сколько кликов запускают добычу? Введи 1 или 2 и нажми ENTER "
                 "(по умолчанию 2): ")
        if ans in ("1", "2", "3"):
            zones["gather_clicks"] = int(ans)
            log.info("Добыча: %s клик(а/ов).", ans)

        # 4) КНОПКА «закрыть»
        print("\n###########################################################################")
        print("#  ШАГ 4 из 7  -  КНОПКА «ЗАКРЫТЬ» У ОКНА-ОШИБКИ")
        print("###########################################################################")
        print(">>> ЧТО СЕЙЧАС ДЕЛАЕМ: показываем боту, где кнопка «закрыть», чтобы он сам")
        print(">>> убирал всплывающее окно «нет профессии», когда случайно тыкнет чужой ресурс.")
        print(">>> ")
        print(">>> КАК ДЕЛАТЬ:")
        print(">>>   1) кликни в игре по ЧУЖОМУ ресурсу (который твоя профессия НЕ добывает)")
        print(">>>      - выскочит окно «Ошибка... нет профессии»;")
        print(">>>   2) затем кликни по кнопке «закрыть» этого окна.")
        print(">>> Нет под рукой чужого ресурса - можно пропустить (ENTER) и снять позже.")
        close_t = _capture_point(ctx, page, clicks,
                                 "кнопку «ЗАКРЫТЬ» в окне-ошибке (нет окна — пропусти)")
        if close_t:
            zones["close"] = close_t

        # 5) ЗОНЫ БОЯ
        print("\n###########################################################################")
        print("#  ШАГ 5 из 7  -  ЗОНЫ БОЯ (блок / атака / выход)")
        print("###########################################################################")
        print(">>> ЧТО СЕЙЧАС ДЕЛАЕМ: показываем боту, куда жать в бою, чтобы он сам")
        print(">>> отбивался от монстров: ставил блок, атаковал и выходил после победы.")
        print(">>> ")
        print(">>> ВАЖНО: этот шаг нужно делать, КОГДА ТЫ УЖЕ В БОЮ (видно колесо «БЛОК»).")
        print(">>> Сейчас боя нет? Спокойно жми ENTER на каждом пункте и пройди этот шаг")
        print(">>> позже - запусти калибровку снова во время боя.")
        print(">>> Можно снять НЕСКОЛЬКО зон атаки и блока - бот будет их чередовать.")
        blocks = _capture_multi(ctx, page, clicks, "ЗОНЫ БЛОКА на колесе (одну или несколько)")
        if blocks:
            zones["blocks"] = blocks
            zones["block"] = blocks[0]                  # совместимость со старым форматом
        attacks = _capture_multi(ctx, page, clicks, "ЗОНЫ АТАКИ на колесе (одну или несколько)")
        if attacks:
            zones["attacks"] = attacks
            zones["attack"] = attacks[0]
        if attacks and len(attacks) > 1:
            ans = read_line_keep_alive(
                ctx, ">>> Как бить по зонам? 1) по кругу — комбинация  2) случайно  [1]: ")
            zones["attack_mode"] = "random" if ans.strip() == "2" else "cycle"
            log.info("Режим атаки: %s.", zones["attack_mode"])
        e = _capture_point(ctx, page, clicks, "кнопку «выход» после победы")
        if e:
            zones["exit"] = e
        h = _capture_point(ctx, page, clicks, "кнопку «В охоту» (можно пропустить)")
        if h:
            zones["hunt"] = h

        # 6) ЗАНОЗА: авто-возврат кирки (рюкзак / вкладка «вещи» / кирка / «надеть» / охота)
        print("\n###########################################################################")
        print("#  ШАГ 6 из 7  -  АВТО-ВОЗВРАТ КИРКИ ПОСЛЕ ЗАНОЗЫ (по желанию)")
        print("###########################################################################")
        print(">>> ЧТО СЕЙЧАС ДЕЛАЕМ: учим бота после лечения занозы самому снова НАДЕТЬ")
        print(">>> кирку и вернуться к добыче, чтобы не останавливаться вручную.")
        print(">>> Не нужно / лень настраивать - просто жми ENTER, шаг пропустится.")
        print(">>> ")
        print(">>> КАК ДЕЛАТЬ: пройди по порядку, кликая по нужным местам в игре:")
        print(">>>   рюкзак -> вкладка «ВЕЩИ» -> кирка -> надпись «надеть» -> режим охоты.")
        bag = _capture_point(ctx, page, clicks, "кнопку РЮКЗАК (которой открываешь рюкзак)")
        if bag:
            zones["bag"] = bag
        print(">>> Теперь ОТКРОЙ рюкзак и перейди на вкладку «ВЕЩИ» (где лежит кирка).")
        tab = _capture_point(ctx, page, clicks, "вкладку «ВЕЩИ» (вторая вкладка рюкзака)")
        if tab:
            zones["tab"] = tab
        pick = _capture_point(ctx, page, clicks,
                              "КИРКУ в рюкзаке (просто кликни по ней — сюда бот будет наводить курсор)")
        if pick:
            zones["pick"] = pick
        print(">>> Наведи курсор на кирку, чтобы появилась надпись «надеть», и кликни по ней.")
        equip = _capture_point(ctx, page, clicks, "надпись «НАДЕТЬ» (появляется при наведении на кирку)")
        if equip:
            zones["equip"] = equip
        hunt_mode = _capture_point(ctx, page, clicks,
                                   "кнопку «РЕЖИМ ОХОТЫ» (перейти к добыче после надевания)")
        if hunt_mode:
            zones["hunt_mode"] = hunt_mode

        # 7) АВТО-КРАФТ РЕЦЕПТОВ (панель профессии → «Создать» у каждого рецепта)
        print("\n###########################################################################")
        print("#  ШАГ 7 из 7  -  АВТО-КРАФТ РЕЦЕПТОВ (по желанию)")
        print("###########################################################################")
        print(">>> ЧТО СЕЙЧАС ДЕЛАЕМ: настраиваем, чтобы бот сам заходил в панель профессии")
        print(">>> (примерно каждые %d сек) и жал «Создать» у твоих рецептов, а потом" % CRAFT_EVERY_SEC)
        print(">>> возвращался к сбору. Не нужно - жми ENTER, авто-крафт останется выключен.")
        print(">>> ")
        print(">>> КАК ДЕЛАТЬ (по порядку):")
        print(">>>   1) сейчас кликни кнопку, которая ОТКРЫВАЕТ панель профессии;")
        print(">>>   2) потом сам открой панель и кликни «Создать» у КАЖДОГО рецепта;")
        print(">>>   3) в конце кликни кнопку «Вернуться».")
        c_open = _capture_point(ctx, page, clicks,
                                "кнопку «ПРОФЕССИИ» (открывает панель с рецептами)")
        if c_open:
            zones["craft_open"] = c_open
            print("\n>>> Теперь ОТКРОЙ панель профессии (нажми ту кнопку в игре),")
            print(">>> чтобы стали видны кнопки «Создать» у рецептов.")
            creates = _capture_multi(
                ctx, page, clicks,
                "кнопки «СОЗДАТЬ» у КАЖДОГО рецепта (по очереди, сверху вниз)")
            if creates:
                zones["craft_creates"] = creates
                log.info("Кнопок «Создать» снято: %d.", len(creates))
            c_back = _capture_point(
                ctx, page, clicks,
                "кнопку «ВЕРНУТЬСЯ» (выход из панели профессии к сбору)")
            if c_back:
                zones["craft_back"] = c_back
        else:
            print(">>> Кнопка профессии не снята — авто-крафт останется выключенным.")

        try:
            save_zones(zones)
            print("\n========================= ГОТОВО! =========================")
            print("Все 7 шагов пройдены, настройки сохранены в fight_zones.json.")
            print("Что-то настроил не так? Просто запусти калибровку заново -")
            print("уже снятые точки не потеряются, поправишь только нужное.")
            print("")
            print("ЗАПУСК БОТА:  python omela_bg.py   (остановить - Ctrl+C)")
            print("===========================================================")
            log.info("Сохранил в %s: %s", ZONES_FILE, zones)
            log.info("Запускай: python omela_bg.py")
        except Exception as ex:
            log.warning("Не смог сохранить: %s", ex)
        ctx.close()


# =========================================================================
#                              РАБОЧИЙ РЕЖИМ
# =========================================================================

def gather_visible(page, scroll_pos, total):
    """Собрать ресурс в текущем кадре.

    Возвращает (total, прервано_ли_боем, сколько_ресурсов_видел). Последнее число
    нужно циклу mode_run: по нему видно «бот давно ничего не находит», и тогда
    можно один раз подсказать, что настроить (профессия/карта/чувствительность),
    а не молча крутить карту.
    """
    now = time.time()
    _prune(_failed_points, now, 3)
    _prune(_recent_points, now, 2)
    scored = find_resource_scored(crop_map(screenshot_bgr(page)))
    if len(scored) > MAX_PER_CYCLE:
        # Много пятен может означать И анимацию/перезагрузку экрана, И просто
        # «густой» ресурс (частый случай у травника: куст похож на траву, поэтому
        # находок много). РАНЬШЕ бот пропускал такой кадр целиком — и если ресурса
        # реально много, он не собирал НИЧЕГО, только прокручивал карту (та самая
        # жалоба «крутит, но не собирает»). Теперь переснимаем кадр один раз: если
        # это была анимация — пятен станет мало и продолжим нормально; если пятен
        # всё ещё много — это не мелькание, а просто много ресурса, и мы всё равно
        # собираем самые уверенные (клики и так ограничены DETECT_MAX_RESULTS),
        # а не выбрасываем весь кадр.
        time.sleep(0.6)
        scored = find_resource_scored(crop_map(screenshot_bgr(page)))
        if len(scored) > MAX_PER_CYCLE:
            log.info("Пятен много (%d) — кадр НЕ пропускаю, беру только самые "
                     "уверенные (топ-%d).", len(scored), DETECT_MAX_RESULTS)
    # кликаем только самые «уверенные» пятна — так реже промахи по фону
    pts = [(d["cx"], d["cy"]) for d in scored[:DETECT_MAX_RESULTS]]
    seen = len(pts)
    if pts:
        log.info("Вижу ресурсов (%s): %d, беру топ-%d по уверенности (прокрутка %d).",
                 ACTIVE_PROF, len(scored), len(pts), scroll_pos)
    for (cx, cy) in pts:
        # подошло время авто-крафта — прерываем сбор, чтобы не опоздать
        # (иначе длинная добыча могла бы задержать крафт на минуты)
        if craft_due():
            log.info("Пора крафтить — прерываю текущий сбор.")
            break
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
                return total, True, seen
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
                # надёжно закрываем: ищем кнопку в полосе и проверяем закрытие
                close_blocking_popup(page)
                total -= 1
                if SKIP_FAILED_ENABLED:
                    _fp_add(scroll_pos, px, py, time.time())
                    log.info("Чужой/неудачный ресурс — закрыл окно и запомнил.")
                break
            # 'none' → окна нет: добыча завершилась (успех) или ещё идёт в фоне
            break

        if FIGHT_ENABLED and FIGHT_POLL_AFTER_GATHER and in_fight(page):
            do_fight(page)
            return total, True, seen
        time.sleep(random.uniform(*BETWEEN_HERBS))
    return total, False, seen


def guard_background_resource(page):
    """Самолечение «отравленного» цвета: если снятый пипеткой цвет ресурса заливает
    почти всю карту (трава/фон), сбор сломан — бот только крутит карту. Тогда на
    старте ОТКАТЫВАЕМСЯ на встроенный пресет профессии на этот запуск и громко пишем
    в лог, как починить насовсем. Возвращает True, если пришлось откатиться.

    Срабатывает только когда активны СНЯТЫЕ пипеткой диапазоны (в fight_zones.json
    есть resource_ranges). Пресеты профессий покрывают малую долю карты и не задеты.
    """
    z = load_zones() or {}
    if not z.get("resource_ranges"):
        return False
    try:
        full = screenshot_bgr(page)
    except Exception:
        return False
    cov = color_coverage_over_map(full, RESOURCE_RANGES, MAP_REGION)
    if cov <= BG_COVERAGE_MAX:
        return False
    log.warning("=" * 70)
    log.warning("ВНИМАНИЕ: снятый цвет ресурса заливает %.0f%% карты — это ТРАВА/ФОН, "
                "а не камень (порог %.0f%%).", cov * 100, BG_COVERAGE_MAX * 100)
    log.warning("Именно из-за этого бот «крутит карту, но не собирает». На ЭТОТ запуск "
                "откатываюсь на встроенный цвет профессии «%s».", ACTIVE_PROF)
    log.warning("Чтобы починить насовсем: открой launcher.py → «Калибровка», на Шаге 1 "
                "введи 0 в «пипетке сбора» (оставить встроенный цвет), ЛИБО кликни ровно "
                "по центру яркого камня. Можно и просто удалить ключ resource_ranges из "
                "fight_zones.json.")
    log.warning("=" * 70)
    # откат: игнорируем снятый цвет, берём пресет профессии
    set_active_profession(z.get("profession", ACTIVE_PROF), custom_ranges=None,
                          custom_blob=z.get("resource_blob"))
    return True


def mode_run():
    with sync_playwright() as p:
        ctx, page = open_and_wait(
            p, "Войди в игру и встань на локацию с нужным ресурсом. После ENTER начнётся сбор.")
        apply_saved_config()
        apply_run_config()
        guard_background_resource(page)
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
        scroll_pos = 0        # накопленное смещение прокрутки (ключ для чёрного списка)
        scroll_dir = 1        # текущее направление «маятника»: +1 вниз, −1 вверх
        next_long = random.randint(*LONG_BREAK_EVERY)
        total = 0
        dry_streak = 0        # подряд циклов, где бот НЕ увидел ни одного ресурса
        dry_hint_shown = False   # подсказку «ничего не вижу» показываем один раз

        # АВТО-КРАФТ: таймер захода в панель профессии за «Создать»
        # (глобальные _CRAFT_ON / _NEXT_CRAFT — чтобы gather_visible мог прерваться)
        global _CRAFT_ON, _NEXT_CRAFT
        _CRAFT_ON = craft_ready()
        do_craft = _CRAFT_ON
        _NEXT_CRAFT = None
        if do_craft:
            ct = get_craft_targets()
            log.info("Авто-крафт включён: каждые ~%d сек жму «Создать» у %d рецепт(ов).",
                     CRAFT_EVERY_SEC, len(ct["creates"]))
            if CRAFT_ON_START:
                try:
                    craft_recipes(page)          # сразу крафтим при запуске
                except Exception as e:
                    log.warning("Авто-крафт при старте не удался: %s", e)
            _NEXT_CRAFT = time.time() + CRAFT_EVERY_SEC
        elif CRAFT_ENABLED:
            log.info("Авто-крафт не настроен (Этап 6 калибровки) — крафтить не буду.")
        # базовый счётчик «занозы» в чате — реагируем только на НОВОЕ появление
        splinter_seen = 0
        if SPLINTER_ENABLED:
            try:
                splinter_seen = chat_splinter_count(page)
            except Exception:
                splinter_seen = 0
            log.info("Слежу за занозой в чате (маркер «%s»).", SPLINTER_CHAT_KEYWORD)
        try:
            while True:
                if (time.time() - started) / 60.0 >= MAX_RUNTIME_MIN:
                    log.info("Лимит времени (%d мин). Стоп.", MAX_RUNTIME_MIN)
                    break
                cycle += 1
                close_if_blocking(page)

                # ЗАНОЗА: новое оповещение в чате → пауза, просьба в чат, ждём лечения
                if SPLINTER_ENABLED:
                    try:
                        cnt = chat_splinter_count(page)
                    except Exception:
                        cnt = splinter_seen
                    if cnt > splinter_seen:
                        handle_splinter(page)
                        try:
                            splinter_seen = chat_splinter_count(page)  # перебазируемся
                        except Exception:
                            splinter_seen = cnt
                        time.sleep(random.uniform(*CYCLE_PAUSE))
                        continue
                    splinter_seen = cnt

                if FIGHT_ENABLED and in_fight(page):
                    do_fight(page)
                    time.sleep(random.uniform(*CYCLE_PAUSE))
                    continue
                if FIGHT_ENABLED and stats_screen_present(page):
                    _, _, _, hunt_t = resolve_fight_targets()
                    return_to_hunt(page, hunt_t)
                    time.sleep(random.uniform(*CYCLE_PAUSE))
                    continue

                # АВТО-КРАФТ: подошло время — заходим в профессию и жмём «Создать».
                # (В бой не лезем: бой проверен выше и делает continue, сюда не дойдём.)
                if craft_due():
                    try:
                        craft_recipes(page)
                    except Exception as e:
                        log.warning("Авто-крафт не удался: %s", e)
                    _NEXT_CRAFT = time.time() + CRAFT_EVERY_SEC
                    time.sleep(random.uniform(*CYCLE_PAUSE))
                    continue

                total, interrupted, seen = gather_visible(page, scroll_pos, total)
                if interrupted:
                    dry_streak = 0
                    time.sleep(random.uniform(*CYCLE_PAUSE))
                    continue

                # ДИАГНОСТИКА «крутит, но не собирает»: если бот много циклов подряд
                # не видит НИ ОДНОГО ресурса — почти всегда дело в настройке (не та
                # профессия, узко снят цвет, не та область карты, слишком строгая
                # чувствительность или ресурс попал в игнор). Молчать и бесконечно
                # крутить карту — плохо, поэтому один раз выводим понятную подсказку.
                if seen == 0:
                    dry_streak += 1
                else:
                    dry_streak = 0
                    dry_hint_shown = False
                if dry_streak == DRY_STREAK_HINT and not dry_hint_shown:
                    dry_hint_shown = True
                    log.warning(
                        "Уже %d циклов подряд НЕ вижу ресурсов (%s) — только кручу карту. "
                        "Скорее всего дело в настройке. Проверь по порядку:",
                        dry_streak, ACTIVE_PROF)
                    log.warning("  1) Та ли ПРОФЕССИЯ? Сейчас: %s. Сменить: "
                                "python omela_bg.py --prof <имя> или Шаг 1 калибровки.",
                                ACTIVE_PROF)
                    log.warning("  2) Область КАРТЫ (Шаг 2 калибровки) — не смотрит ли "
                                "бот мимо игрового поля.")
                    log.warning("  3) Запусти python omela_bg.py --debug — файл "
                                "detect_*.png покажет, что бот видит и почему отсеивает.")
                    log.warning("  4) Слишком СТРОГО? python omela_bg.py --sens 0.8 "
                                "(мягче, видит больше).")
                    if EXCLUDE_RANGES:
                        log.warning("  5) В ИГНОРЕ %d цвет(ов). Если случайно занёс цвет "
                                    "ресурса — «Очистить игнор» в launcher.py.",
                                    len(EXCLUDE_RANGES))

                if MAP_SCROLL_ENABLED:
                    # предел размаха: жёсткий MAP_SCROLL_MAX_POS, плюс необяз.
                    # MAP_SCROLL_POSITIONS (совместимость) — что меньше, то и предел
                    limit = MAP_SCROLL_MAX_POS
                    if MAP_SCROLL_POSITIONS and MAP_SCROLL_POSITIONS > 1:
                        limit = min(limit, MAP_SCROLL_POSITIONS - 1)
                    moved = scroll_step_adaptive(page, scroll_dir)
                    if moved and abs(scroll_pos + scroll_dir) <= limit:
                        scroll_pos += scroll_dir
                    else:
                        # уперлись в край карты (или дошли до предела) → разворот
                        if not moved:
                            log.info("Край карты (прокрутка %d) — разворачиваюсь.", scroll_pos)
                        scroll_dir = -scroll_dir
                        if scroll_step_adaptive(page, scroll_dir):
                            scroll_pos += scroll_dir

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


def mode_reequip_test():
    """Разово прогнать возврат кирки (рюкзак→вещи→навести→надеть→охота) для проверки
    калибровки Этапа 5. Смотри в игре, всё ли открывается и надевается."""
    with sync_playwright() as p:
        ctx, page = open_and_wait(
            p, "ТЕСТ ВОЗВРАТА КИРКИ. Убери кирку в рюкзак (или просто проверь навигацию), затем ENTER.")
        apply_saved_config()
        t = get_reequip_targets()
        log.info("Точки: рюкзак=%s вкладка=%s кирка=%s надеть=%s охота=%s",
                 t["bag"], t["tab"], t["pick"], t["equip"], t["hunt_mode"])
        if not (t["bag"] and t["pick"] and t["equip"]):
            log.warning("Не хватает точек (нужны минимум рюкзак/кирка/надеть). Пройди --calib Этап 5.")
            ctx.close()
            return
        log.info("Выполняю последовательность возврата кирки…")
        reequip_tool(page)
        time.sleep(1.5)
        ok = _test_can_gather(page)
        if ok is True:
            log.info("✅ Похоже, кирка надета — добыча проходит.")
        else:
            log.info("Проверь глазами: открылся ли рюкзак, та ли вкладка, появилась ли «надеть». "
                     "Если точка мимо — перепройди нужный шаг в --calib (Этап 5).")
        print("\n>>> Посмотри результат в игре. ENTER — закрыть.\n", flush=True)
        wait_enter_keep_alive(ctx)
        ctx.close()


def mode_craft_test():
    """Разово прогнать авто-крафт (панель профессии → «Создать» у рецептов → «Вернуться»)
    для проверки калибровки Этапа 6. Смотри в игре, всё ли нажимается."""
    with sync_playwright() as p:
        ctx, page = open_and_wait(
            p, "ТЕСТ АВТО-КРАФТА. Встань на локацию (как при сборе), затем ENTER.")
        apply_saved_config()
        t = get_craft_targets()
        log.info("Точки: профессии=%s создать=%s вернуться=%s",
                 t["open"], t["creates"], t["back"])
        if not (t["open"] and t["creates"]):
            log.warning("Не хватает точек (нужны кнопка профессии и хотя бы одна «Создать»). "
                        "Пройди --calib Этап 6.")
            ctx.close()
            return
        log.info("Выполняю последовательность авто-крафта…")
        craft_recipes(page)
        log.info("Готово. Проверь в игре: открылась ли панель, нажались ли все «Создать», "
                 "вернулся ли бот к сбору. Если точка мимо — перепройди --calib (Этап 6).")
        print("\n>>> Посмотри результат в игре. ENTER — закрыть.\n", flush=True)
        wait_enter_keep_alive(ctx)
        ctx.close()


def main():
    ap = argparse.ArgumentParser(description="Автосбор ресурсов + авто-бой (Playwright).")
    ap.add_argument("--login", action="store_true", help="только войти (сохранить сессию)")
    ap.add_argument("--calib", action="store_true", help="мастер калибровки (профессия/карта/добыча/закрыть/бой)")
    ap.add_argument("--debug", action="store_true", help="скриншот + DOM + слепок боя")
    ap.add_argument("--testkirka", action="store_true",
                    help="разово проверить возврат кирки (рюкзак→вещи→надеть→охота)")
    ap.add_argument("--testcraft", action="store_true",
                    help="разово проверить авто-крафт (профессия→«Создать»→«Вернуться»)")
    ap.add_argument("--prof", metavar="ИМЯ",
                    help="выбрать профессию (%s) и сохранить в конфиг" % "/".join(PROFESSIONS))
    ap.add_argument("--sens", metavar="ЧИСЛО", type=float,
                    help="чувствительность распознавания: <1 мягче (видит больше), "
                         ">1 строже (меньше ложных). Норма 1.0. Сохраняется в конфиг.")
    args = ap.parse_args()

    # --sens: сохранить чувствительность распознавания в fight_zones.json
    if args.sens is not None:
        if not (0.2 <= args.sens <= 4.0):
            print("Чувствительность должна быть в диапазоне 0.2..4.0 (норма 1.0).")
            return
        z = load_zones() or {}
        z["sensitivity"] = round(float(args.sens), 2)
        save_zones(z)
        log.info("Чувствительность распознавания сохранена: %.2f (%s).", z["sensitivity"],
                 "мягче — видит больше" if z["sensitivity"] < 1 else
                 ("строже — меньше ложных" if z["sensitivity"] > 1 else "норма"))

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
    elif args.testkirka:
        mode_reequip_test()
    elif args.testcraft:
        mode_craft_test()
    else:
        mode_run()


if __name__ == "__main__":
    main()