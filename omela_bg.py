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
MAX_PER_CYCLE = 40          # больше кадра с анимацией → пропуск (для геолога поднято)
DETECT_MAX_RESULTS = 12     # за цикл кликаем не больше стольких — самых «уверенных»

# Добыча
GATHER_CLICKS   = 2            # сколько кликов запускают добычу (в этой игре — двойной)
DOUBLECLICK_GAP = (0.08, 0.16)
GATHER_WAIT     = (2.5, 4.5)
BETWEEN_HERBS   = (0.6, 1.6)
CYCLE_PAUSE     = (2.0, 4.0)
LONG_BREAK_EVERY = (15, 30)
LONG_BREAK       = (20.0, 60.0)
MAX_RUNTIME_MIN  = 120

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
# ВАЖНО: реагируем ТОЛЬКО на СИСТЕМНУЮ строку игры о СВОЕЙ занозе — она всегда
# начинается с «Вы …» (как «Вы получили вещь…», «Вы создали…»). Это отсекает:
#   • ники игроков (напр. «Заноза-Буля») в списке жителей и в чате;
#   • чужие и личные сообщения — в них есть знак «»» («Ник » Ник: текст»);
#   • чужие/свои просьбы «дерните/вытащите занозу»;
#   • строку лечения «Вы избавились от занозы» (в ней есть «избавил»).
# Строка засчитывается как «моя заноза», если: начинается с одного из
# SPLINTER_SELF_PREFIXES, содержит «заноз», НЕ содержит «»», слов-просьб и слов
# лечения. Если у тебя другой текст сообщения о занозе — допиши сюда его начало.
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
    # цвета-исключения (пипетка фона): что бот НЕ должен трогать
    EXCLUDE_RANGES = _ranges_to_np(z.get("exclude_ranges"))
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

        # ПИПЕТКА-ИСКЛЮЧЕНИЕ — снять цвета фона/травы/чужого, по которым НЕ кликать
        print("\n>>> ПИПЕТКА-ИСКЛЮЧЕНИЕ (по желанию): цвета, которые НЕ надо собирать.")
        print(">>> Если бот тыкает по траве/фону/чужим ресурсам — кликни по ним здесь,")
        print(">>> и он занесёт эти цвета в чёрный список (яркие камни того же оттенка")
        print(">>> при этом не пострадают — учитывается и яркость).")
        prev_excl = zones.get("exclude_ranges", [])
        if prev_excl:
            print(">>> Сейчас уже есть исключений: %d." % len(prev_excl))
        ans = read_line_keep_alive(
            ctx, ">>> Сколько образцов ФОНА снять? (0 — пропустить; 'c' — очистить старые): ")
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
                        "образец фона #%d — кликни по тому, что НЕ надо собирать "
                        "(ENTER без клика — стоп)" % (i + 1))
                    if not pt:
                        break
                    try:
                        full = screenshot_bgr(page)
                        rngs = sample_exclude_ranges_at(full, pt[0], pt[1])
                    except Exception as ex:
                        log.warning("Пипетка-исключение не сработала: %s", ex)
                        rngs = []
                    if rngs:
                        excl.extend(rngs)
                        log.info("  Исключён цвет (HSV): %s", rngs)
                if excl:
                    zones["exclude_ranges"] = excl
                    log.info("Пипетка-исключение: всего цветов-исключений: %d.", len(excl))

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
        print(">>> Можно снять НЕСКОЛЬКО зон атаки и блока — бот будет их чередовать.")
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

        # 5) ЗАНОЗА: авто-возврат кирки (рюкзак / вкладка «вещи» / кирка / «надеть» / охота)
        print("\n>>> ЭТАП 5. ЗАНОЗА — АВТО-ВОЗВРАТ КИРКИ (можно пропустить ENTER).")
        print(">>> Чтобы после лечения занозы бот сам надел кирку и включил охоту.")
        print(">>> Порядок: рюкзак → вкладка «вещи» → навести на кирку → «надеть» → охота.")
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

        # 6) АВТО-КРАФТ РЕЦЕПТОВ (панель профессии → «Создать» у каждого рецепта)
        print("\n>>> ЭТАП 6. АВТО-КРАФТ РЕЦЕПТОВ (можно пропустить ENTER).")
        print(">>> Бот сам будет периодически (каждые ~%d сек) заходить в панель" % CRAFT_EVERY_SEC)
        print(">>> профессии и жать «Создать» у рецептов, потом возвращаться к сбору.")
        print(">>> Сначала укажи кнопку, которая ОТКРЫВАЕТ панель профессии.")
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
    scored = find_resource_scored(crop_map(screenshot_bgr(page)))
    if len(scored) > MAX_PER_CYCLE:
        log.info("Слишком много пятен (%d) — вероятно анимация, пропускаю кадр.", len(scored))
        return total, False
    # кликаем только самые «уверенные» пятна — так реже промахи по фону
    pts = [(d["cx"], d["cy"]) for d in scored[:DETECT_MAX_RESULTS]]
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
        scroll_pos = 0        # накопленное смещение прокрутки (ключ для чёрного списка)
        scroll_dir = 1        # текущее направление «маятника»: +1 вниз, −1 вверх
        next_long = random.randint(*LONG_BREAK_EVERY)
        total = 0

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

                total, interrupted = gather_visible(page, scroll_pos, total)
                if interrupted:
                    time.sleep(random.uniform(*CYCLE_PAUSE))
                    continue

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