import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# === AUDIO (pygame) ===
try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

# --- paden opbouwen vanaf de scriptmap ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "Audio")
DATASET_FILE = os.path.join(BASE_DIR, "dataset.csv")

def _audio_path(name):
    return os.path.join(AUDIO_DIR, name)

AUDIO_FILES = {
    "music": _audio_path("Music.wav"),
    "start": _audio_path("Start_Game.wav"),
    "ui":    _audio_path("UI_Normal.wav"),
    "end":   _audio_path("End_Game.wav"),
}

# === SCORING CONSTANTS ===
BETA_F = 1.41421356237   # privacy weegt ~2x t.o.v. bruikbaarheid
MIN_UTILITY = 40         # minimum bruikbaarheid, anders score 0
MIN_PRIVACY = 40         # minimum privacy, anders score 0
SUPPRESSION_PENALTY_START = 0.70  # onder 70% bewaarde rijen: straf
SUPPRESSION_MAX_PENALTY = 15      # max -15 punten bij ~40%

# === APP SETTINGS ===
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

HIGHSCORE_FILE = "highscores.json"

# ========== DATA / UTILS ==========

def _read_csv_smart(path):
    required = {"Naam", "Leeftijd", "Postcode", "Opleidingkeuze"}
    encodings = ["utf-8", "utf-8-sig"]
    # komma
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            if set(df.columns) >= required:
                return df
        except Exception:
            pass
    # puntkomma
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, sep=";")
            if set(df.columns) >= required:
                return df
        except Exception:
            pass
    return None

def show_dataframe_in_tree(parent, df, empty_text="Geen data", font_size=18):
    for child in parent.winfo_children():
        child.destroy()

    if df is None or df.empty:
        lbl = ctk.CTkLabel(parent, text=empty_text, anchor="w", font=ctk.CTkFont(size=font_size))
        lbl.pack(fill="both", expand=True, padx=15, pady=15)
        return

    frame = ctk.CTkFrame(parent)
    frame.pack(fill="both", expand=True, padx=4, pady=4)

    cols = list(df.columns)
    tree = ttk.Treeview(frame, columns=cols, show="headings", style="Custom.Treeview")
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    for c in cols:
        tree.heading(c, text=str(c))
        if not df.empty:
            max_len = max(len(str(c)), df[c].astype(str).str.len().max())
            width = min(max(max_len * 9 + 30, 150), 280)
        else:
            width = 180
        tree.column(c, width=width, anchor="w", minwidth=120)

    for _, row in df.iterrows():
        values = [str(row.get(c, "")) for c in cols]
        tree.insert("", "end", values=values)

    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

def k_anonymity(df, qi_cols):
    if not qi_cols or df is None or df.empty:
        return None, None
    grp = df.groupby(qi_cols, dropna=False).size().rename("count").reset_index()
    min_k = int(grp["count"].min()) if len(grp) else 0
    return min_k, grp

def privacy_score(min_k_val, target_k):
    if min_k_val is None or min_k_val == 0:
        return 0
    if min_k_val >= target_k:
        ratio = min(min_k_val / target_k, 3.0)
        score = 80 + (ratio - 1.0) * 10
    else:
        ratio = min_k_val / target_k
        score = ratio * 80
    return int(max(0, min(100, score)))

def utility_score(df_orig, df_transformed):
    if df_orig is None or df_transformed is None or df_orig.empty:
        return 0
    rows_kept_ratio = len(df_transformed) / len(df_orig) if len(df_orig) > 0 else 0
    base_score = rows_kept_ratio * 50
    common_cols = [c for c in df_orig.columns if c in df_transformed.columns]
    if not common_cols:
        return int(base_score)
    col_scores = []
    for c in common_cols:
        a = df_orig[c].astype(str).values
        b = df_transformed[c].astype(str).values
        n = min(len(a), len(b))
        if n == 0:
            continue
        exact_matches = (a[:n] == b[:n]).sum()
        exact_ratio = exact_matches / n
        fully_masked = sum(1 for v in b[:n] if v == '*' or v == '***')
        masked_penalty = (fully_masked / n) * 0.5
        col_score = max(0, exact_ratio - masked_penalty)
        col_scores.append(col_score)
    info_score = (sum(col_scores) / len(col_scores)) * 50 if col_scores else 0
    total = base_score + info_score
    return int(max(0, min(100, total)))

def generalize_age_from_numeric(series_numeric, bin_size, original_series):
    try:
        finite_vals = series_numeric.replace([np.inf, -np.inf], np.nan).dropna()
        if finite_vals.empty:
            return original_series
        min_v = int(np.floor(finite_vals.min() / bin_size) * bin_size)
        max_v = int(np.ceil(finite_vals.max() / bin_size) * bin_size + bin_size)
        bins = np.arange(min_v, max_v + bin_size, bin_size)
        labels = [f"{int(b)}–{int(b+bin_size-1)}" for b in bins[:-1]]
        result = pd.cut(series_numeric, bins=bins, labels=labels, include_lowest=True).astype(str)
        mask = series_numeric.isna()
        result[mask] = original_series[mask].astype(str)
        return result
    except Exception:
        return original_series

def generalize_postcode(series, keep_n):
    s = series.astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    if keep_n <= 0:
        return pd.Series(["*"] * len(s), index=series.index)
    return s.str[:keep_n]

rng = np.random.default_rng(42)

def add_noise_numeric(series_numeric, max_amount):
    s = series_numeric.copy()
    mask = s.isna()
    noise = rng.integers(-max_amount, max_amount + 1, size=len(s))
    s = s.add(noise)
    s = s.clip(lower=0)
    s[mask] = np.nan
    return s

# ----- Nieuwe eindscore helpers -----
def fbeta_score(pct_privacy, pct_utility, beta=BETA_F):
    """pct_* 0..100 → Fβ terug in 0..100. Beta>1 geeft privacy meer gewicht."""
    P = max(0.0, min(100.0, float(pct_privacy))) / 100.0
    U = max(0.0, min(100.0, float(pct_utility))) / 100.0
    if P == 0 or U == 0:
        return 0
    b2 = beta * beta
    f = (1 + b2) * P * U / (b2 * P + U)
    return int(round(f * 100))

def apply_suppression_penalty(score, df_orig, df_transformed,
                              start_ratio=SUPPRESSION_PENALTY_START,
                              max_penalty=SUPPRESSION_MAX_PENALTY):
    """Lineaire straf onder start_ratio bewaarde rijen, tot max_penalty bij ~0.4."""
    try:
        n0 = len(df_orig) or 1
        n1 = len(df_transformed)
        keep_ratio = n1 / n0
        if keep_ratio < start_ratio:
            penalty = int(round(min(max_penalty, max(0.0, (start_ratio - keep_ratio) / 0.3 * max_penalty))))
            return max(0, score - penalty)
    except Exception:
        pass
    return score

# ========== HIGHSCORES ==========

def load_highscores():
    if os.path.exists(HIGHSCORE_FILE):
        try:
            with open(HIGHSCORE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_highscores(scores):
    try:
        with open(HIGHSCORE_FILE, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("Error saving highscores:", e)

def add_highscore(name, score, privacy, utility):
    scores = load_highscores()
    scores.append({
        "name": name,
        "score": score,
        "privacy": privacy,
        "utility": utility,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    scores.sort(key=lambda x: x["score"], reverse=True)
    scores = scores[:10]
    save_highscores(scores)
    return scores

def get_rank(score):
    if score >= 90:
        return "🏆 Privacy Expert"
    elif score >= 80:
        return "🥇 Data Guardian"
    elif score >= 70:
        return "🥈 Privacy Pro"
    elif score >= 60:
        return "🥉 Anonimiseerder"
    elif score >= 50:
        return "📊 Data Masker"
    else:
        return "🔰 Beginner"

# ========== MAIN APP ==========

class DataMaskerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🧩 Data Masker Machine")
        # Start NIET fullscreen; kies een nette startgrootte
        self.geometry("1280x800")
        self.minsize(1100, 700)

        # --- Fullscreen & display state ---
        self._fullscreen = False  # <-- standaard UIT
        # geen _apply_fullscreen hier
        self.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.bind("<Shift-Escape>", lambda e: self.quit_game())

        # --- Audio state & volumes ---
        self.music_volume = 0.5   # 0.0–1.0
        self.sfx_volume   = 0.6   # 0.0–1.0

        # GROTE LETTERTYPES
        self.font_title = ctk.CTkFont(size=38, weight="bold")
        self.font_subtitle = ctk.CTkFont(size=30)
        self.font_body = ctk.CTkFont(size=26)
        self.font_small = ctk.CTkFont(size=22)

        # STATE
        self.player_name = ""
        self.player_ready = False
        self.phase = None
        self.phase_timer_id = None

        # dataset (altijd uit ./dataset.csv)
        self.df_orig = None
        self.df_transformed = None

        # scores
        self.current_privacy_score = 0
        self.current_utility_score = 100

        self.pending_apply_after = None
        self.is_admin = False

        # AUDIO cache
        self.sounds = {}

        # init audio
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                # Muziek
                music_path = AUDIO_FILES.get("music")
                if music_path and os.path.exists(music_path):
                    try:
                        pygame.mixer.music.load(music_path)
                        pygame.mixer.music.set_volume(self.music_volume)
                        pygame.mixer.music.play(loops=-1)
                        print(f"[AUDIO] Music gestart: {music_path}")
                    except Exception as e:
                        print(f"[AUDIO] Kon Music.wav niet laden: {e}")
                else:
                    print(f"[AUDIO] Music-bestand niet gevonden: {music_path}")

                # SFX preload
                for key in ("start", "ui", "end"):
                    p = AUDIO_FILES.get(key)
                    if p and os.path.exists(p):
                        try:
                            s = pygame.mixer.Sound(p)
                            s.set_volume(self.sfx_volume)
                            self.sounds[key] = s
                            print(f"[AUDIO] SFX geladen: {key} -> {p}")
                        except Exception as e:
                            print(f"[AUDIO] Kon sound {key} niet laden: {e}")
                    else:
                        print(f"[AUDIO] SFX-bestand niet gevonden voor '{key}': {p}")
            except Exception as e:
                print(f"[AUDIO] pygame mixer init failed: {e}")
                globals()["PYGAME_AVAILABLE"] = False
        else:
            print("[AUDIO] pygame niet beschikbaar; audio uitgeschakeld.")

        self._configure_treeview_style()

        # FRAMES
        self.menu_frame = ctk.CTkFrame(self)
        self.explain_frame = ctk.CTkFrame(self)
        self.admin_frame = ctk.CTkFrame(self)
        self.phase_raw_frame = ctk.CTkFrame(self)
        self.phase_settings_frame = ctk.CTkFrame(self)
        self.phase_compare_frame = ctk.CTkFrame(self)
        self.result_frame = ctk.CTkFrame(self)

        # build UI
        self._build_menu_screen()
        self._build_explain_screen()
        self._build_admin_screen()
        self._build_phase_raw()
        self._build_phase_settings()
        self._build_phase_compare()
        self._build_result_screen()

        # ensure audio stops on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # dataset verplicht inladen
        if not self.ensure_dataset_loaded():
            self.after(100, self.destroy)
            return

        self.show_menu_screen()

    # ----- Fullscreen & display helpers -----
    def _apply_fullscreen(self, enable: bool):
        try:
            self.attributes("-fullscreen", enable)
        except Exception:
            try:
                self.state('zoomed' if enable else 'normal')
            except Exception:
                pass
        self._fullscreen = enable

    def toggle_fullscreen(self):
        self._apply_fullscreen(not self._fullscreen)

    def apply_resolution(self, res_string: str):
        """
        res_string formaat: 'WIDTHxHEIGHT' bv '1920x1080'
        Let op: in fullscreen zie je geen effect; zet fullscreen uit om te testen.
        """
        try:
            w, h = res_string.lower().split('x')
            w, h = int(w.strip()), int(h.strip())
            # zorg dat we niet fullscreen/zoomed zijn voordat we geometry toepassen
            self._apply_fullscreen(False)
            self.update_idletasks()
            try:
                self.state('normal')  # ga uit 'zoomed' modus (Windows)
            except Exception:
                pass
            # (optioneel) centreren op hoofdscherm
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            x = max(0, (sw - w) // 2)
            y = max(0, (sh - h) // 2)
            self.geometry(f"{w}x{h}+{x}+{y}")
        except Exception as e:
            messagebox.showwarning("Resolutie", f"Onjuiste resolutie: {res_string}\n{e}")

    def quit_game(self):
        self._on_close()

    # ----- AUDIO HELPERS -----
    def _set_music_volume(self, vol_0_1: float):
        self.music_volume = max(0.0, min(1.0, float(vol_0_1)))
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.set_volume(self.music_volume)
            except Exception:
                pass

    def _set_sfx_volume(self, vol_0_1: float):
        self.sfx_volume = max(0.0, min(1.0, float(vol_0_1)))
        for s in self.sounds.values():
            try:
                s.set_volume(self.sfx_volume)
            except Exception:
                pass

    def play_sfx(self, key):
        if not PYGAME_AVAILABLE:
            return
        snd = self.sounds.get(key)
        if snd:
            try:
                snd.play()
            except Exception as e:
                print(f"[AUDIO] play_sfx error ({key}):", e)

    def play_start_once(self):
        self.play_sfx("start")

    def play_ui_once(self):
        self.play_sfx("ui")

    def play_end_once(self):
        self.play_sfx("end")

    def _on_close(self):
        try:
            if PYGAME_AVAILABLE:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
                try:
                    pygame.mixer.quit()
                except Exception:
                    pass
        finally:
            self.destroy()

    # ----- convenience button factory -----
    def _make_button(self, parent, text, command, play_ui=True, **kwargs):
        def wrapped_cmd(*a, **k):
            if play_ui:
                try:
                    self.play_ui_once()
                except Exception:
                    pass
            if callable(command):
                self.after(10, lambda: command())
        return ctk.CTkButton(parent, text=text, command=wrapped_cmd, **kwargs)

    # ----- COMMON -----
    def _configure_treeview_style(self):
        style = ttk.Style()
        try:
            style.theme_use('alt')
        except:
            pass
        style.configure("Custom.Treeview",
                        background="#2b2b2b",
                        foreground="white",
                        fieldbackground="#2b2b2b",
                        rowheight=40,
                        font=("Segoe UI", 16))
        style.configure("Custom.Treeview.Heading",
                        background="#404040",
                        foreground="white",
                        font=("Segoe UI", 17, "bold"))
        style.map("Custom.Treeview",
                  background=[("selected", "#0d7377")],
                  foreground=[("selected", "white")])

    def clear_all_frames(self):
        for f in (self.menu_frame, self.explain_frame, self.admin_frame,
                  self.phase_raw_frame, self.phase_settings_frame,
                  self.phase_compare_frame, self.result_frame):
            f.pack_forget()

    # ----- DATASET LOAD (verplicht) -----
    def ensure_dataset_loaded(self):
        if not os.path.exists(DATASET_FILE):
            messagebox.showerror(
                "Dataset ontbreekt",
                f"'dataset.csv' niet gevonden op:\n{DATASET_FILE}\n\n"
                "Zet het bestand naast het script en start opnieuw."
            )
            self.df_orig = None
            return False

        df = _read_csv_smart(DATASET_FILE)
        if df is not None:
            self.df_orig = df
            print(f"[DATASET] Gelezen: {DATASET_FILE} ({len(df)} rijen)")
            return True

        messagebox.showerror(
            "Dataset ongeldig",
            "Kon 'dataset.csv' niet lezen óf vereiste kolommen ontbreken.\n\n"
            "Vereiste kolommen: Naam, Leeftijd, Postcode, Opleidingkeuze\n"
            "Tip: sla het bestand op als CSV (UTF-8 of UTF-8-SIG) met komma's of puntkomma's."
        )
        self.df_orig = None
        return False

    # ----- MENU -----
    def _build_menu_screen(self):
        title = ctk.CTkLabel(self.menu_frame, text="🧩 Data Masker Machine", font=self.font_title)
        title.pack(pady=(30, 10))

        subtitle = ctk.CTkLabel(self.menu_frame, text="Hoofdmenu — bekijk de highscores en vul je naam in om te starten.", font=self.font_subtitle)
        subtitle.pack()

        self.hs_container = ctk.CTkFrame(self.menu_frame)
        self.hs_container.pack(pady=25, padx=20, fill="x")

        self.hs_title = ctk.CTkLabel(self.hs_container, text="🏆 Highscores (top 10)", font=ctk.CTkFont(size=28, weight="bold"))
        self.hs_title.pack(anchor="w", padx=10, pady=(10,5))

        self.hs_list = ctk.CTkTextbox(self.hs_container, height=250, font=ctk.CTkFont(size=22))
        self.hs_list.pack(fill="x", padx=10, pady=(0,10))
        self.hs_list.configure(state="disabled")

        name_label = ctk.CTkLabel(self.menu_frame, text="👤 Naam:", font=self.font_body)
        name_label.pack(pady=(10,2))

        self.menu_name_entry = ctk.CTkEntry(self.menu_frame, placeholder_text="bv. Luca", width=420, height=50, font=ctk.CTkFont(size=24))
        self.menu_name_entry.pack(pady=(0,10))

        start_btn = self._make_button(
            self.menu_frame,
            text="Volgende",
            command=self.go_to_explain_or_admin,
            play_ui=True,
            width=240,
            height=55,
            font=ctk.CTkFont(size=26, weight="bold")
        )
        start_btn.pack(pady=(10,10))

        self.menu_msg_lbl = ctk.CTkLabel(self.menu_frame, text="", text_color="red", font=self.font_body)
        self.menu_msg_lbl.pack()

    def refresh_highscores_in_menu(self):
        scores = load_highscores()
        self.hs_list.configure(state="normal")
        self.hs_list.delete("1.0", tk.END)
        if not scores:
            self.hs_list.insert(tk.END, "Nog geen highscores.\n")
        else:
            for i, hs in enumerate(scores, 1):
                rank = get_rank(hs["score"])
                star = "⭐️ " if i == 1 else ""
                self.hs_list.insert(
                    tk.END,
                    f"{i}. {star}{hs['name']} — {hs['score']}/100 ({rank})  [P:{hs['privacy']} U:{hs['utility']}]\n"
                )
        self.hs_list.configure(state="disabled")

    def show_menu_screen(self):
        self.clear_all_frames()
        self.refresh_highscores_in_menu()
        self.menu_frame.pack(fill="both", expand=True)

    def go_to_explain_or_admin(self):
        name = self.menu_name_entry.get().strip()
        if not name:
            self.menu_msg_lbl.configure(text="Vul eerst een naam in.")
            return
        if name.lower() == "roodkapje":
            self.is_admin = True
            self.player_name = "ADMIN"
            self.show_admin_screen()
        else:
            self.is_admin = False
            self.player_name = name
            self.player_ready = True
            self.show_explain_screen()

    # ----- ADMIN -----
    def _build_admin_screen(self):
        title = ctk.CTkLabel(self.admin_frame, text="🛠️ Admin menu", font=self.font_title)
        title.pack(pady=(30,10))

        info = ctk.CTkLabel(self.admin_frame, text="Je bent ingelogd als beheerder (roodkapje).", font=self.font_body)
        info.pack(pady=(0,10))

        # ---- Weergave sectie ----
        disp_card = ctk.CTkFrame(self.admin_frame)
        disp_card.pack(pady=(10,5), padx=10, fill="x")

        ctk.CTkLabel(disp_card, text="Weergave", font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w", padx=10, pady=(10,6))

        # checkbox start met actuele fullscreen status (nu: False)
        self.fullscreen_var = tk.IntVar(value=1 if self._fullscreen else 0)
        fs_chk = ctk.CTkCheckBox(disp_card, text="Fullscreen (F11 toggelt)", variable=self.fullscreen_var,
                                 command=self._on_fullscreen_checkbox, font=self.font_small)
        fs_chk.pack(anchor="w", padx=10, pady=(0,8))

        # Resolutie keuzelijst
        w_win = self.winfo_width() or 1280
        h_win = self.winfo_height() or 800
        w_scr = self.winfo_screenwidth()
        h_scr = self.winfo_screenheight()

        common_res = [
            f"{w_win}x{h_win}",           # huidige venstergrootte
            f"{w_scr}x{h_scr}",           # volledige schermres
            "1280x720", "1366x768", "1600x900",
            "1920x1080", "2560x1440", "3840x2160"
        ]
        # maak uniek en bewaar volgorde
        seen = set()
        res_options = []
        for r in common_res:
            if r not in seen:
                res_options.append(r); seen.add(r)

        ctk.CTkLabel(disp_card, text="Resolutie (effect alleen buiten fullscreen)", font=self.font_small).pack(anchor="w", padx=10)
        self.res_var = tk.StringVar(value=res_options[0])
        res_menu = ctk.CTkOptionMenu(disp_card, values=res_options, variable=self.res_var, font=self.font_small)
        res_menu.pack(anchor="w", padx=10, pady=(4,8))

        apply_res_btn = self._make_button(disp_card, text="Toepassen", play_ui=True,
                                          command=lambda: self.apply_resolution(self.res_var.get()),
                                          font=self.font_small, height=40, width=160)
        apply_res_btn.pack(anchor="w", padx=10, pady=(0,10))

        # ---- Audio sectie ----
        audio_card = ctk.CTkFrame(self.admin_frame)
        audio_card.pack(pady=(10,10), padx=10, fill="x")

        ctk.CTkLabel(audio_card, text="Audio", font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w", padx=10, pady=(10,6))

        # Muziek volume
        ctk.CTkLabel(audio_card, text="Muziekvolume", font=self.font_small).pack(anchor="w", padx=10)
        self.music_vol_slider = ctk.CTkSlider(audio_card, from_=0, to=100, number_of_steps=100,
                                              command=self._on_music_slider)
        self.music_vol_slider.set(int(self.music_volume * 100))
        self.music_vol_slider.pack(fill="x", padx=10, pady=(2,8))

        # SFX volume
        ctk.CTkLabel(audio_card, text="SFX-volume (UI & End Game)", font=self.font_small).pack(anchor="w", padx=10)
        self.sfx_vol_slider = ctk.CTkSlider(audio_card, from_=0, to=100, number_of_steps=100,
                                            command=self._on_sfx_slider)
        self.sfx_vol_slider.set(int(self.sfx_volume * 100))
        self.sfx_vol_slider.pack(fill="x", padx=10, pady=(2,10))

        # ---- Dataset / highscores ----
        upload_btn = self._make_button(self.admin_frame, text="Upload CSV als dataset",
                                       command=self.admin_upload_csv, play_ui=True, width=260, height=55, font=self.font_body)
        upload_btn.pack(pady=6)

        reset_btn = self._make_button(self.admin_frame, text="Highscores resetten",
                                      command=self.admin_reset_highscores, play_ui=True, width=260, height=55, font=self.font_body, fg_color="#a83232")
        reset_btn.pack(pady=6)

        # Verborgen exit knop in admin
        quit_btn = self._make_button(self.admin_frame, text="Quit game",
                                     command=self.quit_game, play_ui=True, width=260, height=55, font=self.font_body, fg_color="#8a1e1e")
        quit_btn.pack(pady=6)

        back_btn = self._make_button(self.admin_frame, text="Terug naar hoofdmenu",
                                     command=self.show_menu_screen, play_ui=True, width=260, height=55, font=self.font_body)
        back_btn.pack(pady=10)

        self.admin_status_lbl = ctk.CTkLabel(self.admin_frame, text="", font=self.font_body)
        self.admin_status_lbl.pack(pady=(0,10))

    def _on_fullscreen_checkbox(self):
        new_state = bool(self.fullscreen_var.get())
        self._apply_fullscreen(new_state)

    def _on_music_slider(self, val):
        try:
            v = float(val) / 100.0
        except Exception:
            v = 0.0
        self._set_music_volume(v)

    def _on_sfx_slider(self, val):
        try:
            v = float(val) / 100.0
        except Exception:
            v = 0.0
        self._set_sfx_volume(v)

    def show_admin_screen(self):
        self.clear_all_frames()
        self.admin_frame.pack(fill="both", expand=True)

    def admin_upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            df = _read_csv_smart(file_path)
            if df is None:
                raise ValueError("Ongeldig CSV-formaat of ontbrekende kolommen.")
            try:
                df.to_csv(DATASET_FILE, index=False)
                self.df_orig = df
                self.admin_status_lbl.configure(text="Dataset aangepast en opgeslagen als 'dataset.csv'!", text_color="green")
            except Exception as e:
                self.admin_status_lbl.configure(text=f"Kon 'dataset.csv' niet overschrijven: {e}", text_color="red")
        except Exception as e:
            self.admin_status_lbl.configure(text=f"Fout bij laden: {e}", text_color="red")

    def admin_reset_highscores(self):
        save_highscores([])
        self.admin_status_lbl.configure(text="Highscores gereset!", text_color="green")

    # ----- UITLEG -----
    def _build_explain_screen(self):
        title = ctk.CTkLabel(self.explain_frame, text="📘 Hoe werkt dit spel?", font=self.font_title)
        title.pack(pady=(30,10))

        txt = (
            "Je krijgt zo een dataset met (fictieve) persoonsgegevens.\n"
            "Jouw taak: maak de gegevens minder herkenbaar (privacy), maar zorg dat ze bruikbaar blijven.\n\n"
            "👉 Pseudonimisering (namen → ID's)\n"
            "👉 Generalisatie (waarden grover maken)\n"
            "👉 Ruis (cijfers een beetje aanpassen)\n"
            "👉 Suppressie (te unieke rijen weg)\n\n"
            "Flow:\n"
            "1) 35s: ruwe data kijken\n"
            "2) 60s: instellingen zetten (je ziet het direct)\n"
            "3) 35s: vergelijken\n"
            "\n"
            "Eindscore = Fβ(priv, util) met privacy iets zwaarder dan bruikbaarheid.\n"
            "Als privacy < 40 of bruikbaarheid < 40 → eindscore 0."
        )
        lbl = ctk.CTkLabel(self.explain_frame, text=txt, justify="left", font=self.font_body)
        lbl.pack(pady=10, padx=30)

        def start_and_play_start_sfx():
            self.play_start_once()
            self.after(50, self.start_phase_raw)

        start_btn = self._make_button(self.explain_frame, text="Start spel",
                                      command=start_and_play_start_sfx, play_ui=False,
                                      width=240, height=55, font=self.font_body)
        start_btn.pack(pady=25)

    def show_explain_screen(self):
        self.clear_all_frames()
        self.explain_frame.pack(fill="both", expand=True)

    # ----- FASE 1: RAW -----
    def _build_phase_raw(self):
        title = ctk.CTkLabel(self.phase_raw_frame, text="📄 Dit is de ruwe data", font=self.font_title)
        title.pack(pady=(20,5))

        info = ctk.CTkLabel(
            self.phase_raw_frame,
            text="Bekijk deze gegevens goed. Dit is de originele dataset die je zo gaat anonimiseren.\nJe hebt 35 seconden.",
            font=self.font_body
        )
        info.pack(pady=(0,10))

        self.phase_raw_timer_lbl = ctk.CTkLabel(self.phase_raw_frame, text="35s over", font=ctk.CTkFont(size=26, weight="bold"))
        self.phase_raw_timer_lbl.pack(pady=(0,10))

        self.phase_raw_table = ctk.CTkFrame(self.phase_raw_frame)
        self.phase_raw_table.pack(fill="both", expand=True, padx=10, pady=10)

        next_btn = self._make_button(self.phase_raw_frame, text="Volgende → instellingen",
                                     command=self.start_phase_settings, play_ui=True,
                                     width=280, height=55, font=self.font_body)
        next_btn.pack(pady=(0,15))

    def start_phase_raw(self):
        if self.df_orig is None:
            if not self.ensure_dataset_loaded():
                self.show_menu_screen()
                return
        self.current_privacy_score = 0
        self.current_utility_score = 100
        self.df_transformed = None

        self.clear_all_frames()
        self.phase = "raw_view"
        self.phase_raw_frame.pack(fill="both", expand=True)

        show_dataframe_in_tree(self.phase_raw_table, self.df_orig, font_size=18)
        self.start_phase_timer(35, self.phase_raw_timer_lbl, self.start_phase_settings)

    # ----- FASE 2: SETTINGS -----
    def _build_phase_settings(self):
        self.phase_settings_frame.columnconfigure(1, weight=1)

        self.sidebar = ctk.CTkScrollableFrame(self.phase_settings_frame, width=300, label_text="Dataset & Instellingen")
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=10, pady=10)

        info_top = ctk.CTkLabel(self.sidebar, text="Pas de instellingen aan.\nProbeer zélf in te schatten wat goed is.", font=self.font_body)
        info_top.pack(pady=(0,10), padx=10, anchor="w")

        self.phase_settings_timer_lbl = ctk.CTkLabel(self.sidebar, text="60s", font=ctk.CTkFont(size=26, weight="bold"))
        self.phase_settings_timer_lbl.pack(pady=(0,10), padx=10, anchor="w")

        ctk.CTkLabel(self.sidebar, text="Directe identificatoren", font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w", padx=12, pady=(6,2))
        self.direct_listbox = tk.Listbox(self.sidebar, selectmode=tk.MULTIPLE, exportselection=False, height=4, font=("Segoe UI", 14))
        self.direct_listbox.pack(fill="x", padx=12, pady=(0,6))
        self.direct_listbox.bind("<<ListboxSelect>>", lambda e: self.on_setting_changed())

        ctk.CTkLabel(self.sidebar, text="Quasi-identificatoren", font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w", padx=12, pady=(6,2))
        self.qi_listbox = tk.Listbox(self.sidebar, selectmode=tk.MULTIPLE, exportselection=False, height=4, font=("Segoe UI", 14))
        self.qi_listbox.pack(fill="x", padx=12, pady=(0,8))
        self.qi_listbox.bind("<<ListboxSelect>>", lambda e: self.on_setting_changed())

        ctk.CTkLabel(self.sidebar, text="Technieken", font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w", padx=12, pady=(10,4))

        self.apply_pseudo_var = tk.IntVar(value=0)
        ctk.CTkCheckBox(self.sidebar, text="Pseudonimisering", variable=self.apply_pseudo_var, command=self.on_setting_changed, font=self.font_small).pack(anchor="w", padx=12, pady=2)

        self.apply_general_age_var = tk.IntVar(value=0)
        ctk.CTkCheckBox(self.sidebar, text="Generalisatie leeftijd", variable=self.apply_general_age_var, command=self.on_setting_changed, font=self.font_small).pack(anchor="w", padx=12, pady=2)

        ctk.CTkLabel(self.sidebar, text="Leeftijdsklasse (2-30)", font=self.font_small).pack(anchor="w", padx=12)
        self.age_slider = ctk.CTkSlider(self.sidebar, from_=2, to=30, number_of_steps=28, command=self._age_slider_event)
        self.age_slider.set(2)
        self.age_slider.pack(fill="x", padx=12, pady=(2,6))
        self.age_value_lbl = ctk.CTkLabel(self.sidebar, text="2 jaar", font=self.font_small)
        self.age_value_lbl.pack(anchor="w", padx=12, pady=(0,6))

        self.apply_general_pc_var = tk.IntVar(value=0)
        ctk.CTkCheckBox(self.sidebar, text="Generalisatie postcode", variable=self.apply_general_pc_var, command=self.on_setting_changed, font=self.font_small).pack(anchor="w", padx=12, pady=2)

        ctk.CTkLabel(self.sidebar, text="Postcode tekens (0-6)", font=self.font_small).pack(anchor="w", padx=12)
        self.pc_slider = ctk.CTkSlider(self.sidebar, from_=0, to=6, number_of_steps=6, command=self._pc_slider_event)
        self.pc_slider.set(0)
        self.pc_slider.pack(fill="x", padx=12, pady=(2,6))
        self.pc_value_lbl = ctk.CTkLabel(self.sidebar, text="0", font=self.font_small)
        self.pc_value_lbl.pack(anchor="w", padx=12, pady=(0,6))

        self.apply_noise_var = tk.IntVar(value=0)
        ctk.CTkCheckBox(self.sidebar, text="Ruis toevoegen aan leeftijd", variable=self.apply_noise_var, command=self.on_setting_changed, font=self.font_small).pack(anchor="w", padx=12, pady=2)

        ctk.CTkLabel(self.sidebar, text="Max. ruis (0-10)", font=self.font_small).pack(anchor="w", padx=12)
        self.noise_slider = ctk.CTkSlider(self.sidebar, from_=0, to=10, number_of_steps=10, command=self._noise_slider_event)
        self.noise_slider.set(0)
        self.noise_slider.pack(fill="x", padx=12, pady=(2,6))
        self.noise_value_lbl = ctk.CTkLabel(self.sidebar, text="0 jaar", font=self.font_small)
        self.noise_value_lbl.pack(anchor="w", padx=12, pady=(0,6))

        self.apply_suppress_var = tk.IntVar(value=0)
        ctk.CTkCheckBox(self.sidebar, text="Suppressie (k-anonimiteit)", variable=self.apply_suppress_var, command=self.on_setting_changed, font=self.font_small).pack(anchor="w", padx=12, pady=2)

        ctk.CTkLabel(self.sidebar, text="k (2-10)", font=self.font_small).pack(anchor="w", padx=12)
        self.k_slider = ctk.CTkSlider(self.sidebar, from_=2, to=10, number_of_steps=8, command=self._k_slider_event)
        self.k_slider.set(2)
        self.k_slider.pack(fill="x", padx=12, pady=(2,6))
        self.k_value_lbl = ctk.CTkLabel(self.sidebar, text="2", font=self.font_small)
        self.k_value_lbl.pack(anchor="w", padx=12, pady=(0,6))

        done_btn = self._make_button(self.sidebar, text="Klaar met instellingen →", command=self.start_phase_compare, play_ui=True, font=self.font_small, height=46)
        done_btn.pack(fill="x", padx=12, pady=(8,12))

        # RIGHT AREA
        right = ctk.CTkFrame(self.phase_settings_frame)
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.phase_settings_frame.rowconfigure(0, weight=1)
        self.phase_settings_frame.columnconfigure(1, weight=1)

        title = ctk.CTkLabel(right, text="📊 Dataset (live)", font=self.font_subtitle)
        title.pack(anchor="nw", pady=(8,4), padx=8)

        self.settings_data_container = ctk.CTkFrame(right)
        self.settings_data_container.pack(fill="both", expand=True, padx=8, pady=(6,8))

    def start_phase_settings(self):
        self.stop_phase_timer()
        self.clear_all_frames()
        self.phase = "settings"
        self.phase_settings_frame.pack(fill="both", expand=True)

        if self.df_orig is None:
            if not self.ensure_dataset_loaded():
                self.show_menu_screen()
                return

        self.populate_column_selectors()

        self.apply_pseudo_var.set(0)
        self.apply_general_age_var.set(0)
        self.apply_general_pc_var.set(0)
        self.apply_noise_var.set(0)
        self.apply_suppress_var.set(0)
        self.age_slider.set(2); self.age_value_lbl.configure(text="2 jaar")
        self.pc_slider.set(0);  self.pc_value_lbl.configure(text="0")
        self.noise_slider.set(0); self.noise_value_lbl.configure(text="0 jaar")
        self.k_slider.set(2); self.k_value_lbl.configure(text="2")

        self.direct_listbox.selection_clear(0, tk.END)
        self.qi_listbox.selection_clear(0, tk.END)

        show_dataframe_in_tree(self.settings_data_container, self.df_orig, font_size=16)

        self.current_privacy_score = 0
        self.current_utility_score = 100

        self.start_phase_timer(60, self.phase_settings_timer_lbl, self.start_phase_compare)

    # ----- FASE 3: COMPARE -----
    def _build_phase_compare(self):
        title = ctk.CTkLabel(self.phase_compare_frame, text="🔍 Vergelijking", font=self.font_title)
        title.pack(pady=(20,5))

        info = ctk.CTkLabel(self.phase_compare_frame,
                            text="Links zie je de originele data, rechts de getransformeerde data.\nJe hebt 35 seconden om te kijken wat je instellingen gedaan hebben.",
                            font=self.font_body)
        info.pack(pady=(0,10))

        self.phase_compare_timer_lbl = ctk.CTkLabel(self.phase_compare_frame, text="35s over", font=ctk.CTkFont(size=26, weight="bold"))
        self.phase_compare_timer_lbl.pack()

        container = ctk.CTkFrame(self.phase_compare_frame)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        self.compare_orig = ctk.CTkFrame(container)
        self.compare_orig.grid(row=0, column=0, sticky="nsew", padx=(0,5))

        self.compare_trans = ctk.CTkFrame(container)
        self.compare_trans.grid(row=0, column=1, sticky="nsew", padx=(5,0))

        next_btn = self._make_button(self.phase_compare_frame, text="Naar score →", command=self.show_result_screen, play_ui=True, width=240, height=50, font=self.font_body)
        next_btn.pack(pady=10)

    def start_phase_compare(self):
        self.stop_phase_timer()
        self.clear_all_frames()
        self.phase = "compare"
        self.phase_compare_frame.pack(fill="both", expand=True)

        show_dataframe_in_tree(self.compare_orig, self.df_orig, font_size=16)
        show_dataframe_in_tree(self.compare_trans, self.df_transformed, font_size=16)

        self.start_phase_timer(35, self.phase_compare_timer_lbl, self.show_result_screen)

    # ----- RESULT -----
    def _build_result_screen(self):
        title = ctk.CTkLabel(self.result_frame, text="🎮 Resultaat", font=self.font_title)
        title.pack(pady=(30,10))

        self.result_info_lbl = ctk.CTkLabel(self.result_frame, text="", justify="left", font=self.font_body)
        self.result_info_lbl.pack(pady=10)

        submit_btn = self._make_button(self.result_frame, text="Afronden", command=self.submit_score_and_back_to_menu, play_ui=True, width=220, height=55, font=self.font_body)
        submit_btn.pack(pady=15)

    def _any_technique_enabled(self):
        return (bool(self.apply_pseudo_var.get()) or bool(self.apply_general_age_var.get()) or
                bool(self.apply_general_pc_var.get()) or bool(self.apply_noise_var.get()) or
                bool(self.apply_suppress_var.get()))

    def _any_qi_selected(self):
        return len(self.qi_listbox.curselection()) > 0

    def show_result_screen(self):
        self.stop_phase_timer()
        try:
            self.play_end_once()
        except Exception:
            pass

        # --- Nieuwe eindscore: Fβ + drempels + suppressie-straf ---
        final_score = fbeta_score(self.current_privacy_score, self.current_utility_score, beta=BETA_F)

        zero_reason = None
        if self.current_utility_score < MIN_UTILITY:
            zero_reason = f"Bruikbaarheid < {MIN_UTILITY}"
        if self.current_privacy_score < MIN_PRIVACY:
            zero_reason = (zero_reason + " & " if zero_reason else "") + f"Privacy < {MIN_PRIVACY}"

        if zero_reason:
            final_score = 0

        # optionele straf voor zware suppressie (alleen als niet al 0 door drempels)
        if final_score > 0:
            final_score = apply_suppression_penalty(final_score, self.df_orig, self.df_transformed,
                                                    start_ratio=SUPPRESSION_PENALTY_START,
                                                    max_penalty=SUPPRESSION_MAX_PENALTY)

        # als niets aan staat of geen QI's → 0
        if not self._any_technique_enabled() or not self._any_qi_selected():
            final_score = 0
            zero_reason = (zero_reason + " & " if zero_reason else "") + "Geen technieken of geen QI's geselecteerd"

        self.result_privacy = self.current_privacy_score
        self.result_utility = self.current_utility_score
        self.result_final = final_score

        self.clear_all_frames()
        self.result_frame.pack(fill="both", expand=True)

        rank = get_rank(self.result_final)
        info = (
            f"Speler: {self.player_name}\n\n"
            f"🔒 Privacy-score: {self.result_privacy}/100\n"
            f"📊 Bruikbaarheid-score: {self.result_utility}/100\n"
            f"🎯 Eindscore: {self.result_final}/100\n"
            f"🏅 Rang: {rank}"
        )
        if zero_reason:
            info += f"\n\nℹ️ Reden voor 0: {zero_reason}"

        self.result_info_lbl.configure(text=info)

    def submit_score_and_back_to_menu(self):
        add_highscore(self.player_name, self.result_final, self.result_privacy, self.result_utility)
        self.show_menu_screen()

    # ----- TIMERS -----
    def start_phase_timer(self, seconds, label_widget, on_finish):
        self.phase_time_left = seconds
        self.phase_timer_label = label_widget
        self.phase_timer_callback = on_finish
        self._tick_phase_timer()

    def _tick_phase_timer(self):
        if self.phase_time_left <= 0:
            self.phase_timer_id = None
            self.phase_timer_callback()
            return
        self.phase_timer_label.configure(text=f"{self.phase_time_left}s over")
        self.phase_time_left -= 1
        self.phase_timer_id = self.after(1000, self._tick_phase_timer)

    def stop_phase_timer(self):
        if self.phase_timer_id:
            self.after_cancel(self.phase_timer_id)
            self.phase_timer_id = None

    # ----- SETTINGS CHANGE -----
    def on_setting_changed(self):
        if self.phase != "settings":
            return
        if self.pending_apply_after is not None:
            self.after_cancel(self.pending_apply_after)
        self.pending_apply_after = self.after(200, self.apply_transformations)

    # ----- UI CALLBACKS -----
    def _age_slider_event(self, v):
        val = int(float(v))
        self.age_value_lbl.configure(text=f"{val} jaar")
        self.on_setting_changed()

    def _pc_slider_event(self, v):
        val = int(float(v))
        self.pc_value_lbl.configure(text=str(val))
        self.on_setting_changed()

    def _noise_slider_event(self, v):
        val = int(float(v))
        self.noise_value_lbl.configure(text=f"{val} jaar")
        self.on_setting_changed()

    def _k_slider_event(self, v):
        val = int(float(v))
        self.k_value_lbl.configure(text=str(val))
        self.on_setting_changed()

    # ----- DATA LOAD -----
    def populate_column_selectors(self):
        cols = list(self.df_orig.columns) if self.df_orig is not None else []
        self.direct_listbox.delete(0, tk.END)
        self.qi_listbox.delete(0, tk.END)
        for c in cols:
            self.direct_listbox.insert(tk.END, c)
            self.qi_listbox.insert(tk.END, c)

    # ----- APPLY -----
    def apply_transformations(self):
        self.pending_apply_after = None
        if self.phase != "settings":
            return
        if self.df_orig is None or self.df_orig.empty:
            return

        df = self.df_orig.copy()

        direct_ids = [self.direct_listbox.get(i) for i in self.direct_listbox.curselection()]
        qi_cols = [self.qi_listbox.get(i) for i in self.qi_listbox.curselection()]

        apply_pseudo = bool(self.apply_pseudo_var.get())
        apply_general_age = bool(self.apply_general_age_var.get())
        age_bin_size = int(round(self.age_slider.get()))
        apply_general_pc = bool(self.apply_general_pc_var.get())
        pc_digits = int(round(self.pc_slider.get()))
        apply_noise = bool(self.apply_noise_var.get())
        noise_amount = int(round(self.noise_slider.get()))
        apply_suppress = bool(self.apply_suppress_var.get())
        k = int(round(self.k_slider.get()))

        if "Leeftijd" in df.columns:
            age_num = pd.to_numeric(df["Leeftijd"], errors="coerce")
            if apply_noise and noise_amount > 0:
                age_num = add_noise_numeric(age_num, noise_amount)
            if apply_general_age:
                df["Leeftijd"] = generalize_age_from_numeric(age_num, age_bin_size, df["Leeftijd"])
            else:
                if apply_noise and noise_amount > 0:
                    df["Leeftijd"] = age_num.round().astype("Int64").astype(str)

        if "Postcode" in df.columns and apply_general_pc:
            df["Postcode"] = generalize_postcode(df["Postcode"], pc_digits)

        if apply_pseudo and direct_ids:
            for c in direct_ids:
                if c in df.columns:
                    df[c] = [f"ID-{i+1:03d}" for i in range(len(df))]

        min_k, groups = k_anonymity(df, qi_cols)

        if apply_suppress and qi_cols and groups is not None:
            small_groups = groups[groups["count"] < k]
            if len(small_groups):
                temp = df.reset_index()
                merged = temp.merge(small_groups[qi_cols], on=qi_cols, how="left", indicator=True)
                to_drop_idx = merged[merged["_merge"] == "both"]["index"].values
                df = df.drop(index=to_drop_idx).copy()
                min_k, groups = k_anonymity(df, qi_cols)

        p_score = privacy_score(min_k if min_k is not None else 0, k)
        u_score = utility_score(self.df_orig, df)

        self.current_privacy_score = p_score
        self.current_utility_score = u_score

        self.df_transformed = df
        show_dataframe_in_tree(self.settings_data_container, self.df_transformed, font_size=16)

# RUN
if __name__ == "__main__":
    app = DataMaskerApp()
    app.mainloop()
