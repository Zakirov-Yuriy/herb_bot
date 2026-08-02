# -*- coding: utf-8 -*-
"""
Меню запуска herb_bot.

Окно с настройками, которые задаются ПЕРЕД стартом бота: профессия,
чувствительность распознавания, время работы, авто-крафт, авто-бой, лечение.
Настройки сохраняются в fight_zones.json (тот же файл, что и калибровка),
а кнопка «Сохранить и запустить» стартует omela_bg.py в отдельном окне консоли.

Ничего ставить не надо — tkinter входит в стандартный Python.
Запуск:  python launcher.py   (или двойной клик по файлу).
"""

import os
import sys
import json
import subprocess

import tkinter as tk
from tkinter import ttk, messagebox

BASE = os.path.dirname(os.path.abspath(__file__))
ZONES_FILE = os.path.join(BASE, "fight_zones.json")
BOT_FILE = os.path.join(BASE, "omela_bg.py")

# Профессии: подпись в меню -> внутренний ключ (как в omela_bg.py -> PROFESSIONS).
PROFESSIONS = {
    "Травник (омела / травы)": "herbalist",
    "Геолог (камни / руда)":   "geolog",
    "Рыбак (рыба)":            "fisher",
}
PROF_BY_KEY = {v: k for k, v in PROFESSIONS.items()}

# Значения по умолчанию (совпадают с константами в omela_bg.py).
DEFAULTS = {
    "profession":       "herbalist",
    "sensitivity":      1.0,
    "max_runtime_min":  240,
    "craft_enabled":    True,
    "craft_on_start":   True,
    "craft_every_sec":  330,
    "fight_enabled":    True,
    "splinter_enabled": True,
}


def load_zones():
    try:
        with open(ZONES_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_zones(z):
    with open(ZONES_FILE, "w", encoding="utf-8") as f:
        json.dump(z, f, ensure_ascii=False, indent=2)


def python_exe():
    """Путь к python.exe (а не pythonw.exe — боту нужна консоль для логов и ENTER)."""
    py = sys.executable or "python"
    low = py.lower()
    if low.endswith("pythonw.exe"):
        py = py[:-len("pythonw.exe")] + "python.exe"
    return py


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Herb Bot — настройки запуска")
        self.resizable(False, False)
        try:
            self.call("tk", "scaling", 1.2)
        except Exception:
            pass

        z = load_zones()
        cfg = z.get("run_config") or {}

        def g(key):
            # профессия/чувствительность — верхний уровень, остальное — run_config
            if key in ("profession", "sensitivity"):
                return z.get(key, DEFAULTS[key])
            return cfg.get(key, DEFAULTS[key])

        # ---- переменные ----
        self.var_prof = tk.StringVar(
            value=PROF_BY_KEY.get(g("profession"), PROF_BY_KEY["herbalist"]))
        self.var_sens = tk.DoubleVar(value=float(g("sensitivity")))
        self.var_runtime = tk.IntVar(value=int(g("max_runtime_min")))
        self.var_craft = tk.BooleanVar(value=bool(g("craft_enabled")))
        self.var_craft_start = tk.BooleanVar(value=bool(g("craft_on_start")))
        self.var_craft_every = tk.IntVar(value=int(g("craft_every_sec")))
        self.var_fight = tk.BooleanVar(value=bool(g("fight_enabled")))
        self.var_splinter = tk.BooleanVar(value=bool(g("splinter_enabled")))

        pad = {"padx": 10, "pady": 6}
        root = ttk.Frame(self, padding=12)
        root.grid()

        # ---- Профессия и распознавание ----
        f1 = ttk.LabelFrame(root, text="Профессия и распознавание")
        f1.grid(row=0, column=0, sticky="ew", **pad)
        ttk.Label(f1, text="Профессия:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.OptionMenu(f1, self.var_prof, self.var_prof.get(),
                       *PROFESSIONS.keys()).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(f1, text="Чувствительность:").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        srow = ttk.Frame(f1)
        srow.grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        self.lbl_sens = ttk.Label(srow, width=5)
        self.lbl_sens.pack(side="right")
        sc = ttk.Scale(srow, from_=0.2, to=4.0, orient="horizontal",
                       variable=self.var_sens, command=self._upd_sens, length=200)
        sc.pack(side="left", fill="x", expand=True)
        self._upd_sens(None)
        ttk.Label(f1, text="<1 — мягче (видит больше)   ·   >1 — строже",
                  foreground="#666").grid(row=2, column=0, columnspan=2, sticky="w", padx=8)

        # ---- Сбор ----
        f2 = ttk.LabelFrame(root, text="Сбор ресурсов")
        f2.grid(row=1, column=0, sticky="ew", **pad)
        ttk.Label(f2, text="Время работы, мин:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Spinbox(f2, from_=5, to=1440, increment=10, width=8,
                    textvariable=self.var_runtime).grid(row=0, column=1, sticky="w", padx=8, pady=6)

        # ---- Крафт ----
        f3 = ttk.LabelFrame(root, text="Авто-крафт")
        f3.grid(row=2, column=0, sticky="ew", **pad)
        ttk.Checkbutton(f3, text="Включить авто-крафт (собирать + крафтить рецепты)",
                        variable=self.var_craft,
                        command=self._toggle_craft).grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        self.cb_craft_start = ttk.Checkbutton(f3, text="Крафтить сразу при запуске",
                                               variable=self.var_craft_start)
        self.cb_craft_start.grid(row=1, column=0, columnspan=2, sticky="w", padx=24, pady=2)
        self.lbl_every = ttk.Label(f3, text="Крафтить каждые, сек:")
        self.lbl_every.grid(row=2, column=0, sticky="w", padx=24, pady=4)
        self.sp_every = ttk.Spinbox(f3, from_=30, to=3600, increment=30, width=8,
                                    textvariable=self.var_craft_every)
        self.sp_every.grid(row=2, column=1, sticky="w", padx=8, pady=4)
        self._toggle_craft()

        # ---- Бой ----
        f4 = ttk.LabelFrame(root, text="Бой")
        f4.grid(row=3, column=0, sticky="ew", **pad)
        ttk.Checkbutton(f4, text="Авто-бой (боевой модуль во время сбора)",
                        variable=self.var_fight).grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(f4, text="Реагировать на «занозу» / лечение",
                        variable=self.var_splinter).grid(row=1, column=0, sticky="w", padx=8, pady=4)

        # ---- кнопки ----
        btns = ttk.Frame(root)
        btns.grid(row=4, column=0, sticky="ew", **pad)
        ttk.Button(btns, text="Сохранить и запустить",
                   command=self.save_and_run).pack(side="left", padx=4)
        ttk.Button(btns, text="Только сохранить",
                   command=self.save_only).pack(side="left", padx=4)
        ttk.Button(btns, text="Калибровка",
                   command=self.run_calib).pack(side="left", padx=4)
        ttk.Button(btns, text="Очистить игнор",
                   command=self.clear_excludes).pack(side="left", padx=4)

        self.status = ttk.Label(root, text="", foreground="#0a0")
        self.status.grid(row=5, column=0, sticky="w", padx=10)

    # ---- вспомогательное ----
    def _upd_sens(self, _):
        self.lbl_sens.config(text="%.2f" % self.var_sens.get())

    def _toggle_craft(self):
        state = "normal" if self.var_craft.get() else "disabled"
        for w in (self.cb_craft_start, self.lbl_every, self.sp_every):
            w.configure(state=state)

    def collect(self):
        """Собрать настройки из окна в словарь run_config + профессия/чувствит."""
        return {
            "profession": PROFESSIONS[self.var_prof.get()],
            "sensitivity": round(float(self.var_sens.get()), 2),
            "run_config": {
                "max_runtime_min":  int(self.var_runtime.get()),
                "craft_enabled":    bool(self.var_craft.get()),
                "craft_on_start":   bool(self.var_craft_start.get()),
                "craft_every_sec":  int(self.var_craft_every.get()),
                "fight_enabled":    bool(self.var_fight.get()),
                "splinter_enabled": bool(self.var_splinter.get()),
            },
        }

    def do_save(self):
        data = self.collect()
        z = load_zones()
        # смена профессии — сбросить старый цвет-пипетку от другой профессии
        if z.get("profession") and z.get("profession") != data["profession"]:
            z.pop("resource_ranges", None)
            z.pop("resource_blob", None)
        z["profession"] = data["profession"]
        z["sensitivity"] = data["sensitivity"]
        z["run_config"] = data["run_config"]
        save_zones(z)
        return data

    # ---- кнопки ----
    def save_only(self):
        try:
            self.do_save()
            self.status.config(text="Настройки сохранены в fight_zones.json ✓", foreground="#0a0")
        except Exception as e:
            messagebox.showerror("Ошибка", "Не удалось сохранить:\n%s" % e)

    def _launch_bot(self, extra_args=None):
        if not os.path.exists(BOT_FILE):
            messagebox.showerror("Ошибка", "Не найден omela_bg.py рядом с меню.")
            return False
        cmd = [python_exe(), BOT_FILE] + (extra_args or [])
        kwargs = {"cwd": BASE}
        # На Windows — открыть бота в НОВОМ окне консоли (нужен ввод ENTER и логи).
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        try:
            subprocess.Popen(cmd, **kwargs)
            return True
        except Exception as e:
            messagebox.showerror("Ошибка", "Не удалось запустить бота:\n%s" % e)
            return False

    def save_and_run(self):
        try:
            self.do_save()
        except Exception as e:
            messagebox.showerror("Ошибка", "Не удалось сохранить:\n%s" % e)
            return
        if self._launch_bot():
            self.status.config(text="Бот запущен в отдельном окне. Нажми там ENTER для старта.",
                               foreground="#0a0")

    def clear_excludes(self):
        """Очистить чёрный список цветов-исключений (exclude_ranges).

        Нужно, когда бот перестал собирать ресурс: часто причина в том, что в
        игнор случайно попал цвет самого ресурса, и бот выбрасывает все находки.
        Профессию, снятый цвет ресурса и точки калибровки НЕ трогаем."""
        z = load_zones()
        n = len(z.get("exclude_ranges") or [])
        if n == 0:
            self.status.config(text="Чёрный список уже пуст — чистить нечего.",
                               foreground="#0a0")
            return
        if not messagebox.askyesno(
                "Очистить игнор?",
                "В чёрном списке сейчас %d цвет(ов)-исключений.\n\n"
                "Очистить их? Это часто чинит проблему «бот не собирает ресурс».\n"
                "Профессия, цвет ресурса и точки калибровки останутся на месте." % n):
            return
        try:
            z.pop("exclude_ranges", None)
            save_zones(z)
            self.status.config(text="Игнор очищен (%d цвет(ов) удалено) ✓ Можно запускать." % n,
                               foreground="#0a0")
        except Exception as e:
            messagebox.showerror("Ошибка", "Не удалось очистить:\n%s" % e)

    def run_calib(self):
        # сначала сохраняем (чтобы профессия для калибровки была верной), потом --calib
        try:
            self.do_save()
        except Exception:
            pass
        if self._launch_bot(["--calib"]):
            self.status.config(text="Калибровка запущена в отдельном окне.", foreground="#0a0")


if __name__ == "__main__":
    App().mainloop()