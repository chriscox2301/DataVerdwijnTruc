# data_masker_ctk.py
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np

# Appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Sample data generator (same as jouw Streamlit voorbeeld, iets uitgebreider)
def make_sample():
    return pd.DataFrame({
        "Naam": ["Lisa Jansen", "Tom Vermeer", "Sara de Wit", "Mark Peters", "Eva Jans", "Noah Smit", "Mila Bakker", "Daan Vis", "Fleur van Lee", "Kian Vos"],
        "Leeftijd": [24, 46, 24, 33, 33, 29, 51, 51, 46, 24],
        "Postcode": ["6215 BG","6221 CD","6215 BH","6212 AA","6212 BA","6225 CC","6215 BG","6215 BH","6221 CD","6215 BG"],
        "Ziekte": ["Migraine","Astma","Allergie","Geen","Diabetes","Allergie","Astma","Geen","Migraine","Geen"]
    })

# Utility: render pandas DataFrame into a ttk.Treeview with dark styling
def show_dataframe_in_tree(tree_parent, df):
    # Clear previous treeviews
    for child in tree_parent.winfo_children():
        child.destroy()

    if df is None or df.empty:
        lbl = ctk.CTkLabel(tree_parent, text="Geen data", anchor="w", font=ctk.CTkFont(size=14))
        lbl.pack(fill="both", expand=True, padx=15, pady=15)
        return

    frame = ctk.CTkFrame(tree_parent)
    frame.pack(fill="both", expand=True, padx=4, pady=4)

    cols = list(df.columns)
    tree = ttk.Treeview(frame, columns=cols, show="headings", style="Custom.Treeview")
    
    # Simple scrollbars
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    # Configure headers and columns with better sizing
    for i, c in enumerate(cols):
        tree.heading(c, text=str(c))
        
        # Calculate better column width based on content
        if not df.empty:
            max_len = max(
                len(str(c)),  # Header length
                df[c].astype(str).str.len().max() if len(df) > 0 else 10
            )
            width = min(max(max_len * 8 + 20, 100), 200)  # Between 100-200px
        else:
            width = 120
            
        tree.column(c, width=width, anchor="w", minwidth=80)

    # Insert rows
    for i, (idx, row) in enumerate(df.iterrows()):
        values = [str(row.get(c, "")) for c in cols]
        tree.insert("", "end", values=values)

    # Grid layout
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    return tree

# k-anonimiteit calculation
def k_anonymity(df, qi_cols):
    if not qi_cols or df is None or df.empty:
        return None, None
    grp = df.groupby(qi_cols, dropna=False).size().rename("count").reset_index()
    min_k = int(grp["count"].min()) if len(grp) else 0
    return min_k, grp

# privacy / utility scoring (same heuristics)
def privacy_score(min_k_val, target_k):
    if min_k_val is None:
        return 0
    score = min(100, int((min_k_val / max(target_k,1)) * 70) + 30)
    return max(0, min(100, score))

def utility_score(df_orig, df_transformed):
    if df_orig is None or df_transformed is None:
        return 0
    common_cols = [c for c in df_orig.columns if c in df_transformed.columns]
    if not common_cols:
        return 0
    same = 0
    total = 0
    for c in common_cols:
        a = df_orig[c].astype(str).values
        b = df_transformed[c].astype(str).values
        n = min(len(a), len(b))
        same += (a[:n] == b[:n]).sum()
        total += n
    return int(100 * same / total) if total else 0

# Data transformation functions (ported from the Streamlit logic)
def generalize_age(series, bin_size):
    """Generalize age into bins, handling various input types safely"""
    try:
        # Probeer naar numeriek te converteren
        s = pd.to_numeric(series, errors="coerce")
        
        # Check if we have valid numeric data
        finite_vals = s.replace([np.inf, -np.inf], np.nan).dropna()
        if finite_vals.empty:
            print("Waarschuwing: Geen geldige numerieke waarden voor leeftijd generalisatie")
            return series
            
        # Create stable bins that cover the range
        min_v = int(np.floor(finite_vals.min() / bin_size) * bin_size)
        max_v = int(np.ceil(finite_vals.max() / bin_size) * bin_size + bin_size)
        bins = np.arange(min_v, max_v + bin_size, bin_size)
        labels = [f"{int(b)}–{int(b+bin_size-1)}" for b in bins[:-1]]
        
        # Apply binning
        result = pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(str)
        
        # Replace 'nan' strings with original values for non-numeric entries
        mask = s.isna()
        result[mask] = series[mask].astype(str)
        
        return result
    except Exception as e:
        print(f"Fout bij leeftijd generalisatie: {e}")
        return series

def generalize_postcode(series, keep_n):
    s = series.astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    if keep_n <= 0:
        return pd.Series(["*"] * len(s), index=series.index)
    return s.str[:keep_n]

# RNG with fixed seed for reproducible noise (like Streamlit example)
rng = np.random.default_rng(42)
def add_noise(series, max_amount):
    """Add noise to numeric series, handling various input types safely"""
    try:
        # Eerst proberen om naar numeriek te converteren
        s = pd.to_numeric(series, errors="coerce")
        
        # Check if we have any valid numeric values
        if s.isna().all():
            print("Waarschuwing: Geen geldige numerieke waarden gevonden voor ruis toevoegen")
            return series  # Return original series if no valid numbers
            
        # Generate noise
        noise = rng.integers(-max_amount, max_amount+1, size=len(s))
        
        # Add noise and clip to non-negative
        result = (s + noise).clip(lower=0)
        
        # Fill NaN values with original values where conversion failed
        mask = s.isna()
        result[mask] = series[mask]
        
        return result
    except Exception as e:
        print(f"Fout bij ruis toevoegen: {e}")
        return series  # Return original on any error

# Main App
class DataMaskerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🧩 Data Masker Machine — Desktop (CustomTkinter)")
        self.geometry("1200x720")
        self.minsize(1000, 600)

        # Data holders
        self.df_orig = None
        self.df_transformed = None
        
        # Configure treeview styling once
        self._configure_treeview_style()

        # --- Layout: sidebar (left) + main (right) ---
        self.sidebar = ctk.CTkScrollableFrame(self, width=250, label_text="Dataset & Instellingen")
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.main = ctk.CTkFrame(self)
        self.main.pack(side="right", fill="both", expand=True)

        # Sidebar contents
        self._build_sidebar()

        # Main area: top title + two columns frames
        self._build_main_area()

        # Load default sample data
        self.use_sample_var.set(1)
        self.load_data()
        
    def _configure_treeview_style(self):
        """Configure dark treeview styling once"""
        style = ttk.Style()
        
        try:
            style.theme_use('alt')
            
            style.configure("Custom.Treeview",
                           background="#2b2b2b",
                           foreground="white",
                           rowheight=25,
                           fieldbackground="#2b2b2b",
                           font=('Segoe UI', 10))
            
            style.configure("Custom.Treeview.Heading",
                           background="#404040",
                           foreground="white",
                           font=('Segoe UI', 10, 'bold'))
            
            style.map("Custom.Treeview",
                     background=[('selected', '#0d7377')],
                     foreground=[('selected', 'white')])
            
            style.map("Custom.Treeview.Heading",
                     background=[('active', '#505050')])
                     
        except Exception as e:
            print(f"Treeview styling error: {e}")

    def _build_sidebar(self):
        pad = {"padx": 12, "pady": 6}
        header = ctk.CTkLabel(self.sidebar, text="Dataset & Instellingen", font=ctk.CTkFont(size=16, weight="bold"))
        header.pack(anchor="w", **pad)

        # Dataset selection
        ds_frame = ctk.CTkFrame(self.sidebar)
        ds_frame.pack(fill="x", padx=10, pady=(0,8))

        self.use_sample_var = tk.IntVar(value=1)
        self.use_sample_cb = ctk.CTkCheckBox(ds_frame, text="Gebruik voorbeelddata", variable=self.use_sample_var, command=self.load_data)
        self.use_sample_cb.pack(anchor="w", pady=4)

        load_btn = ctk.CTkButton(ds_frame, text="Upload CSV", command=self.upload_csv)
        load_btn.pack(anchor="w", pady=4)

        # Placeholder for column selectors (we'll populate after data load)
        ctk.CTkLabel(self.sidebar, text="Identificatoren", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", **pad)

        self.direct_label = ctk.CTkLabel(self.sidebar, text="Directe identificatoren (multi-select)")
        self.direct_label.pack(anchor="w", padx=12)
        # Listbox for multi-select direct ids
        self.direct_listbox = tk.Listbox(self.sidebar, selectmode=tk.MULTIPLE, exportselection=False, height=5)
        self.direct_listbox.pack(fill="x", padx=12, pady=(4,8))

        self.qi_label = ctk.CTkLabel(self.sidebar, text="Quasi-identificatoren (multi-select)")
        self.qi_label.pack(anchor="w", padx=12)
        self.qi_listbox = tk.Listbox(self.sidebar, selectmode=tk.MULTIPLE, exportselection=False, height=5)
        self.qi_listbox.pack(fill="x", padx=12, pady=(4,8))

        # Techniques
        ctk.CTkLabel(self.sidebar, text="Technieken", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", **pad)
        self.apply_pseudo_var = tk.IntVar(value=1)
        ctk.CTkCheckBox(self.sidebar, text="Pseudonimisering (vervang directe ID's)", variable=self.apply_pseudo_var).pack(anchor="w", padx=12, pady=2)

        # Age generalization
        self.apply_general_age_var = tk.IntVar(value=1)
        ctk.CTkCheckBox(self.sidebar, text="Generalisatie leeftijd → klassen", variable=self.apply_general_age_var).pack(anchor="w", padx=12, pady=2)
        ctk.CTkLabel(self.sidebar, text="Grootte leeftijdsklasse (jaren)").pack(anchor="w", padx=12)
        self.age_bin_var = tk.IntVar(value=10)
        self.age_slider = ctk.CTkSlider(self.sidebar, from_=5, to=20, number_of_steps=4, command=self._age_slider_event)
        self.age_slider.set(10)
        self.age_slider.pack(fill="x", padx=12, pady=(2,6))
        self.age_value_lbl = ctk.CTkLabel(self.sidebar, text="10 jaar")
        self.age_value_lbl.pack(anchor="w", padx=12, pady=(0,8))

        # Postcode generalization
        self.apply_general_pc_var = tk.IntVar(value=1)
        ctk.CTkCheckBox(self.sidebar, text="Generalisatie postcode → minder precisie", variable=self.apply_general_pc_var).pack(anchor="w", padx=12, pady=2)
        ctk.CTkLabel(self.sidebar, text="Aantal tekens behouden (0–4)").pack(anchor="w", padx=12)
        self.pc_slider = ctk.CTkSlider(self.sidebar, from_=0, to=4, number_of_steps=4, command=self._pc_slider_event)
        self.pc_slider.set(4)
        self.pc_slider.pack(fill="x", padx=12, pady=(2,6))
        self.pc_value_lbl = ctk.CTkLabel(self.sidebar, text="4")
        self.pc_value_lbl.pack(anchor="w", padx=12, pady=(0,8))

        # Noise
        self.apply_noise_var = tk.IntVar(value=0)
        ctk.CTkCheckBox(self.sidebar, text="Ruis toevoegen aan leeftijd (±)", variable=self.apply_noise_var).pack(anchor="w", padx=12, pady=2)
        ctk.CTkLabel(self.sidebar, text="Maximale ruis (jaren)").pack(anchor="w", padx=12)
        self.noise_slider = ctk.CTkSlider(self.sidebar, from_=1, to=5, number_of_steps=4, command=self._noise_slider_event)
        self.noise_slider.set(2)
        self.noise_slider.pack(fill="x", padx=12, pady=(2,6))
        self.noise_value_lbl = ctk.CTkLabel(self.sidebar, text="2 jaar")
        self.noise_value_lbl.pack(anchor="w", padx=12, pady=(0,8))

        # Suppression / k-anonymity
        self.apply_suppress_var = tk.IntVar(value=0)
        ctk.CTkCheckBox(self.sidebar, text="Suppressie (verwijder te unieke rijen)", variable=self.apply_suppress_var).pack(anchor="w", padx=12, pady=2)
        ctk.CTkLabel(self.sidebar, text="k (k-anonimiteit)").pack(anchor="w", padx=12)
        self.k_slider = ctk.CTkSlider(self.sidebar, from_=2, to=6, number_of_steps=4, command=self._k_slider_event)
        self.k_slider.set(3)
        self.k_slider.pack(fill="x", padx=12, pady=(2,6))
        self.k_value_lbl = ctk.CTkLabel(self.sidebar, text="3")
        self.k_value_lbl.pack(anchor="w", padx=12, pady=(0,8))

        # Apply button
        self.apply_btn = ctk.CTkButton(self.sidebar, text="Apply transformations", command=self.apply_transformations)
        self.apply_btn.pack(fill="x", padx=12, pady=(8,4))
        
        # Reset button
        self.reset_btn = ctk.CTkButton(self.sidebar, text="Reset naar origineel", command=self.reset_transformations)
        self.reset_btn.pack(fill="x", padx=12, pady=(4,8))

        # Spacer + explanation button
        self.help_btn = ctk.CTkButton(self.sidebar, text="Uitleg (wat gebeurt er?)", command=self.show_help)
        self.help_btn.pack(fill="x", padx=12, pady=(0,12))

    # Slider callbacks to update displayed labels
    def _age_slider_event(self, v):
        # slider returns float
        val = int(float(v))
        # force to multiples of 5 (like original slider step)
        if val not in (5,10,15,20):
            # round to nearest 5
            val = int(round(val/5)*5)
            self.age_slider.set(val)
        self.age_value_lbl.configure(text=f"{val} jaar")

    def _pc_slider_event(self, v):
        val = int(float(v))
        self.pc_value_lbl.configure(text=str(val))

    def _noise_slider_event(self, v):
        val = int(float(v))
        self.noise_value_lbl.configure(text=f"{val} jaar")

    def _k_slider_event(self, v):
        val = int(float(v))
        self.k_value_lbl.configure(text=str(val))

    # CSV upload
    def upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            messagebox.showerror("Fout bij laden", f"Kon CSV niet laden:\n{e}")
            return
        self.df_orig = df
        self.populate_column_selectors()
        self.refresh_main_views()

    # Loads sample data or clears
    def load_data(self):
        if self.use_sample_var.get() == 1:
            self.df_orig = make_sample()
        else:
            # if user unticked sample and hasn't uploaded, clear
            if self.df_orig is None:
                self.df_orig = pd.DataFrame()
        self.populate_column_selectors()
        self.refresh_main_views()

    # Populate Listboxes for direct and quasi identifiers
    def populate_column_selectors(self):
        cols = list(self.df_orig.columns) if self.df_orig is not None else []
        # Clear listboxes
        self.direct_listbox.delete(0, tk.END)
        self.qi_listbox.delete(0, tk.END)
        for c in cols:
            self.direct_listbox.insert(tk.END, c)
            self.qi_listbox.insert(tk.END, c)
        # default selections: Naam as direct, leeftijd/postcode as qi if present
        try:
            # select "Naam" if present in direct
            if "Naam" in cols:
                idx = cols.index("Naam")
                self.direct_listbox.select_set(idx)
            # select default quasi
            defaults = [c for c in cols if c.lower() in ["leeftijd", "postcode"]]
            for d in defaults:
                idx = cols.index(d)
                self.qi_listbox.select_set(idx)
        except Exception:
            pass

    # Build main area layout
    def _build_main_area(self):
        # Title
        title = ctk.CTkLabel(self.main, text="🧩 Data Masker Machine — Anonimisering Demo", font=ctk.CTkFont(size=18, weight="bold"))
        title.pack(anchor="nw", padx=12, pady=(12,6))

        subtitle = ctk.CTkLabel(self.main, text="Speel met verschillende anonimiseringstechnieken en zie direct het effect op privacy en bruikbaarheid.")
        subtitle.pack(anchor="nw", padx=12, pady=(0,8))

        # Two columns area
        self.top_frame = ctk.CTkFrame(self.main)
        self.top_frame.pack(fill="both", expand=True, padx=12, pady=(6,12))

        # Left column (originele + transformed data)
        self.left_col = ctk.CTkFrame(self.top_frame)
        self.left_col.pack(side="left", fill="both", expand=True, padx=(0,6))

        # Right column (groups + metrics)
        self.right_col = ctk.CTkFrame(self.top_frame, width=360)
        self.right_col.pack(side="right", fill="y")

        # In left: original data (top) and transformed data (bottom)
        self.orig_label = ctk.CTkLabel(self.left_col, text="📋 Ruwe data", font=ctk.CTkFont(size=16, weight="bold"))
        self.orig_label.pack(anchor="nw", pady=(8,4), padx=8)
        self.orig_table_container = ctk.CTkFrame(self.left_col)
        self.orig_table_container.pack(fill="both", expand=True, padx=8, pady=(0,12))

        self.trans_label = ctk.CTkLabel(self.left_col, text="🔄 Getransformeerde data", font=ctk.CTkFont(size=16, weight="bold"))
        self.trans_label.pack(anchor="nw", pady=(8,4), padx=8)
        self.trans_table_container = ctk.CTkFrame(self.left_col)
        self.trans_table_container.pack(fill="both", expand=True, padx=8, pady=(0,12))

        # In right: groups area + metrics + suppressed info + groups tree
        self.k_label = ctk.CTkLabel(self.right_col, text="🔐 k-anonimiteit & groepen", font=ctk.CTkFont(size=16, weight="bold"))
        self.k_label.pack(anchor="nw", pady=(8,6), padx=8)

        self.groups_container = ctk.CTkFrame(self.right_col)
        self.groups_container.pack(fill="both", expand=False, padx=8, pady=(0,12))
        self.groups_table = None

        # Metrics
        self.metrics_frame = ctk.CTkFrame(self.right_col)
        self.metrics_frame.pack(fill="x", padx=8, pady=8)
        
        # Metrics title
        metrics_title = ctk.CTkLabel(self.metrics_frame, text="📊 Resultaten", font=ctk.CTkFont(size=14, weight="bold"))
        metrics_title.pack(fill="x", pady=(8,4))
        
        self.privacy_metric_lbl = ctk.CTkLabel(self.metrics_frame, text="🔒 Privacy-score (heuristisch): -/100", anchor="w", font=ctk.CTkFont(size=12))
        self.privacy_metric_lbl.pack(fill="x", pady=(6,4), padx=6)
        
        self.utility_metric_lbl = ctk.CTkLabel(self.metrics_frame, text="📊 Bruikbaarheid-score (heuristisch): -/100", anchor="w", font=ctk.CTkFont(size=12))
        self.utility_metric_lbl.pack(fill="x", pady=(4,6), padx=6)
        
        self.suppressed_lbl = ctk.CTkLabel(self.metrics_frame, text="🗑️ Suppressed rijen: 0", anchor="w", font=ctk.CTkFont(size=12))
        self.suppressed_lbl.pack(fill="x", pady=(6,8), padx=6)

    # Show explanation popup
    def show_help(self):
        help_text = (
            "Technieken in deze demo:\n\n"
            "- Pseudonimisering: vervangt directe identificatoren met tokens (bv. Naam -> ID-001).\n"
            "- Generalisatie: maakt waarden grover (bv. leeftijd in klassen, postcode korter).\n"
            "- Ruis toevoegen: verandert numerieke waarden lichtjes (bv. leeftijd ±2).\n"
            "- Suppressie: verwijdert records/groepen die te uniek zijn voor een gekozen k.\n\n"
            "De scores zijn heuristisch en bedoeld voor educatieve illustratie."
        )
        messagebox.showinfo("Uitleg", help_text)

    # Reset transformations to show original data
    def reset_transformations(self):
        """Reset to original data without any transformations"""
        self.df_transformed = None
        self.refresh_main_views()
        print("Data gereset naar origineel")

    # Main transformation runner (applies current UI settings)
    def apply_transformations(self):
        if self.df_orig is None or self.df_orig.empty:
            messagebox.showinfo("Geen data", "Laad voorbeelddata of upload een CSV om verder te gaan.")
            return

        # BELANGRIJK: Altijd beginnen vanaf de originele data
        df = self.df_orig.copy()

        # read selected direct ids & qi from listboxes
        direct_idxs = self.direct_listbox.curselection()
        direct_ids = [self.direct_listbox.get(i) for i in direct_idxs]

        qi_idxs = self.qi_listbox.curselection()
        qi_cols = [self.qi_listbox.get(i) for i in qi_idxs]

        # options
        apply_pseudo = bool(self.apply_pseudo_var.get())
        apply_general_age = bool(self.apply_general_age_var.get())
        age_bin_size = int(round(self.age_slider.get()))
        apply_general_pc = bool(self.apply_general_pc_var.get())
        pc_digits = int(round(self.pc_slider.get()))
        apply_noise = bool(self.apply_noise_var.get())
        noise_amount = int(round(self.noise_slider.get()))
        apply_suppress = bool(self.apply_suppress_var.get())
        k = int(round(self.k_slider.get()))

        # VOLGORDE IS BELANGRIJK: Eerst noise, dan generalisatie, dan pseudonimisering
        
        # 1. Ruis toevoegen (moet eerst, op originele numerieke waarden)
        if "Leeftijd" in df.columns and apply_noise:
            try:
                df["Leeftijd"] = add_noise(df["Leeftijd"], noise_amount)
            except Exception as e:
                print(f"Fout bij ruis toevoegen: {e}")

        # 2. Generalisatie leeftijd (na noise)
        if "Leeftijd" in df.columns and apply_general_age:
            try:
                df["Leeftijd"] = generalize_age(df["Leeftijd"], age_bin_size)
            except Exception as e:
                print(f"Fout bij leeftijd generalisatie: {e}")

        # 3. Generalisatie postcode
        if "Postcode" in df.columns and apply_general_pc:
            try:
                df["Postcode"] = generalize_postcode(df["Postcode"], pc_digits)
            except Exception as e:
                print(f"Fout bij postcode generalisatie: {e}")

        # 4. Pseudonimisering (laatst, zodat het geen andere transformaties beïnvloedt)
        if apply_pseudo and direct_ids:
            for c in direct_ids:
                if c in df.columns:
                    df[c] = [f"ID-{i+1:03d}" for i in range(len(df))]

        # k-anonymity computation
        min_k, groups = k_anonymity(df, qi_cols)

        suppressed_rows = 0
        if apply_suppress and qi_cols and groups is not None:
            small_groups = groups[groups["count"] < k]
            if len(small_groups):
                temp = df.reset_index()
                merged = temp.merge(small_groups[qi_cols], on=qi_cols, how="left", indicator=True)
                to_drop_idx = merged[merged["_merge"] == "both"]["index"].values
                suppressed_rows = len(to_drop_idx)
                df = df.drop(index=to_drop_idx).copy()
                min_k, groups = k_anonymity(df, qi_cols)

        # scores
        p_score = privacy_score(min_k if min_k is not None else 0, k)
        u_score = utility_score(self.df_orig, df)

        # store transformed df
        self.df_transformed = df

        # update displays
        self.refresh_main_views(groups=groups, min_k=min_k, p_score=p_score, u_score=u_score, suppressed_rows=suppressed_rows)

    # Refresh tables in main area
    def refresh_main_views(self, groups=None, min_k=None, p_score=None, u_score=None, suppressed_rows=0):
        # Original data
        show_dataframe_in_tree(self.orig_table_container, self.df_orig)

        # Transformed data
        show_dataframe_in_tree(self.trans_table_container, self.df_transformed)

        # Groups table
        for widget in self.groups_container.winfo_children():
            widget.destroy()

        # Show min_k
        qi_idxs = self.qi_listbox.curselection()
        qi_cols = [self.qi_listbox.get(i) for i in qi_idxs]
        if qi_cols:
            min_k_text = str(min_k) if min_k is not None else "-"
            label = ctk.CTkLabel(self.groups_container, text=f"k (minimale groepsgrootte) op [{', '.join(qi_cols)}]: {min_k_text}", 
                               anchor="w", font=ctk.CTkFont(size=12, weight="bold"))
            label.pack(fill="x", padx=8, pady=(8,6))
            if groups is not None and not groups.empty:
                # show groups in a styled treeview
                gframe = ctk.CTkFrame(self.groups_container)
                gframe.pack(fill="both", expand=True, padx=8, pady=(0,8))
                
                # Use same simplified styling
                cols = list(groups.columns)
                tree = ttk.Treeview(gframe, columns=cols, show="headings", height=8, style="Custom.Treeview")
                
                vsb = ttk.Scrollbar(gframe, orient="vertical", command=tree.yview)
                hsb = ttk.Scrollbar(gframe, orient="horizontal", command=tree.xview)
                tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
                
                for c in cols:
                    tree.heading(c, text=str(c))
                    # Better column sizing for groups
                    if c == 'count':
                        tree.column(c, width=80, anchor="center")
                    else:
                        tree.column(c, width=120, anchor="w")
                
                # sort groups by count ascending for visibility
                groups_sorted = groups.sort_values("count")
                for i, (_, row) in enumerate(groups_sorted.iterrows()):
                    vals = [str(row.get(c, '')) for c in cols]
                    tree.insert("", "end", values=vals)
                
                tree.grid(row=0, column=0, sticky="nsew")
                vsb.grid(row=0, column=1, sticky="ns")
                hsb.grid(row=1, column=0, sticky="ew")
                gframe.rowconfigure(0, weight=1)
                gframe.columnconfigure(0, weight=1)
        else:
            lbl = ctk.CTkLabel(self.groups_container, text="ℹ️ Selecteer quasi-identificatoren om k-anonimiteit te berekenen.", 
                             anchor="w", font=ctk.CTkFont(size=12))
            lbl.pack(fill="x", padx=8, pady=8)

        # Metrics - improved styling and formatting
        if p_score is None:
            p_score = "-"
        else:
            p_score = f"{p_score}/100"
        if u_score is None:
            u_score = "-"
        else:
            u_score = f"{u_score}/100"
        self.privacy_metric_lbl.configure(text=f"🔒 Privacy-score: {p_score}")
        self.utility_metric_lbl.configure(text=f"📊 Bruikbaarheid-score: {u_score}")
        self.suppressed_lbl.configure(text=f"🗑️ Suppressed rijen: {suppressed_rows}")

# Run app
if __name__ == "__main__":
    app = DataMaskerApp()
    app.mainloop()