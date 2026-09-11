"""
Feature engineering: stock growth, FDA impact, adverse-event severity,
demographic profile (Q5) and the clinical weight-loss summary (Q2).

All functions receive already-cleaned DataFrames (see src/cleaning.py).
"""
import pandas as pd


# ─────────────────────────────────────────────────────────────────
# Q4 / Q1 · BOLSA
# ─────────────────────────────────────────────────────────────────

# Hitos usados para partir el crecimiento bursátil en tramos.
# MOUNJARO viene de drugs_overview.csv (fda_first_approval_date de tirzepatide).
# ZEPBOUND NO está en el dataset (solo guarda la *primera* aprobación de cada
# genérico), por eso se añade a mano con su fuente.
HITOS_BURSATILES = {
    "MOUNJARO": {"ticker": "LLY", "fecha": "2022-05-13", "fuente": "dataset (drugs_overview.csv)"},
    "ZEPBOUND": {"ticker": "LLY", "fecha": "2023-11-08", "fuente": "FDA press release 2023-11-08 (manual)"},
}


def stock_normalized_growth(
    stocks: pd.DataFrame, ticker: str, fecha_inicio: str, fecha_fin: str
) -> pd.DataFrame:
    """
    Return a filtered stock slice with a 'crecimiento' column (base-100 index
    starting from the first close in the given date range).

    Raises ValueError if there is no data for the ticker in that range, so the
    caller gets a clear message instead of an IndexError on .iloc[0].
    """
    s = (
        stocks[stocks["ticker"] == ticker]
        .set_index("date")
        .sort_index()
        .loc[fecha_inicio:fecha_fin]
        .copy()
    )
    if s.empty:
        raise ValueError(
            f"No hay datos de '{ticker}' entre {fecha_inicio} y {fecha_fin}. "
            f"Tickers disponibles: {sorted(stocks['ticker'].unique())}"
        )
    s["crecimiento"] = (s["close"] / s["close"].iloc[0]) * 100
    return s


def max_drawdown(close: pd.Series) -> dict:
    """
    Largest peak-to-trough fall of a price series.
    Returns pct (negative), peak/trough dates and prices.
    """
    close = close.dropna()
    if close.empty:
        return {"drawdown_pct": None, "fecha_pico": None, "precio_pico": None,
                "fecha_valle": None, "precio_valle": None}
    running_max = close.cummax()
    dd = close / running_max - 1
    fecha_valle = dd.idxmin()
    fecha_pico = close.loc[:fecha_valle].idxmax()
    return {
        "drawdown_pct": round(dd.min() * 100, 1),
        "fecha_pico": str(fecha_pico.date()),
        "precio_pico": round(close.loc[fecha_pico], 2),
        "fecha_valle": str(fecha_valle.date()),
        "precio_valle": round(close.loc[fecha_valle], 2),
    }


def _growth_since(s: pd.DataFrame, fecha: str) -> float | None:
    """% change between the first close on/after `fecha` and the last close."""
    tramo = s.loc[fecha:]
    if tramo.empty:
        return None
    return round((tramo["close"].iloc[-1] / tramo["close"].iloc[0] - 1) * 100, 1)


def stock_growth_summary(
    stocks: pd.DataFrame,
    fecha_inicio: str = "2017-01-01",
    fecha_fin: str | None = None,
    hitos: dict = HITOS_BURSATILES,
) -> pd.DataFrame:
    """
    One row per ticker with the growth split into tramos:
      - Crecimiento_total_pct         : first close (fecha_inicio) → last close
      - Crecimiento_desde_<HITO>_pct  : close on hito date → last close (one col per hito)
      - Max_drawdown_pct + peak/trough dates
    The last available date is reported explicitly so the reader knows how
    recent the series is (the dataset reaches 2026).
    """
    if fecha_fin is None:
        fecha_fin = str(stocks["date"].max().date())

    rows = []
    for ticker in sorted(stocks["ticker"].unique()):
        s = stock_normalized_growth(stocks, ticker, fecha_inicio, fecha_fin)
        row = {
            "Ticker": ticker,
            "Fecha_inicio": str(s.index[0].date()),
            "Fecha_fin": str(s.index[-1].date()),
            "Precio_inicio": round(s["close"].iloc[0], 2),
            "Precio_fin": round(s["close"].iloc[-1], 2),
            "Precio_max": round(s["close"].max(), 2),
            "Fecha_max": str(s["close"].idxmax().date()),
            "Crecimiento_total_pct": round((s["close"].iloc[-1] / s["close"].iloc[0] - 1) * 100, 1),
        }
        for nombre, h in hitos.items():
            row[f"Crecimiento_desde_{nombre}_pct"] = _growth_since(s, h["fecha"])
        dd = max_drawdown(s["close"])
        row["Max_drawdown_pct"] = dd["drawdown_pct"]
        row["Drawdown_pico"] = dd["fecha_pico"]
        row["Drawdown_valle"] = dd["fecha_valle"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Ticker")


def stock_impact_at_fda(
    stocks: pd.DataFrame, drugs: pd.DataFrame, window_days: int = 30
) -> pd.DataFrame:
    """
    For each drug with fecha_en_rango_stock == True, compare the close
    `window_days` calendar days before vs after the FDA approval date.
    Column names are built from window_days so they always match the window.

    NOTE: this is a descriptive ±window comparison, not an event study. It does
    not control for market moves, earnings or expectations already priced in.
    """
    cols = [
        "Generico", "Marcas", "Ticker", "Fecha_FDA",
        f"Precio_{window_days}d_antes", "Precio_aprobacion",
        f"Precio_{window_days}d_despues", "Cambio_pct",
    ]
    if "fecha_en_rango_stock" not in drugs.columns:
        raise ValueError("drugs necesita la columna 'fecha_en_rango_stock' (ver add_fda_in_stock_range)")

    aprobados = drugs[drugs["fecha_en_rango_stock"] == True]
    if aprobados.empty:
        return pd.DataFrame(columns=cols)

    results = []
    for _, row in aprobados.iterrows():
        ticker = row["ticker"]
        fecha = row["fda_first_approval_date"]
        st = stocks[stocks["ticker"] == ticker].set_index("date").sort_index()
        if st.empty:
            continue

        before_data = st[st.index <= fecha - pd.Timedelta(days=window_days)]["close"]
        at_data = st[st.index >= fecha]["close"]
        after_data = st[st.index >= fecha + pd.Timedelta(days=window_days)]["close"]

        before = before_data.iloc[-1] if len(before_data) > 0 else None
        at_date = at_data.iloc[0] if len(at_data) > 0 else None
        after = after_data.iloc[0] if len(after_data) > 0 else None
        cambio = (
            round(((after - before) / before) * 100, 1)
            if (before is not None and after is not None and before != 0)
            else None
        )

        results.append({
            "Generico": row["generic_name"],
            "Marcas": row["brand_names"],
            "Ticker": ticker,
            "Fecha_FDA": str(fecha.date()),
            f"Precio_{window_days}d_antes": round(before, 2) if before is not None else None,
            "Precio_aprobacion": round(at_date, 2) if at_date is not None else None,
            f"Precio_{window_days}d_despues": round(after, 2) if after is not None else None,
            "Cambio_pct": cambio,
        })

    return pd.DataFrame(results, columns=cols)


def stock_impact_sensitivity(
    stocks: pd.DataFrame, drugs: pd.DataFrame, windows: tuple[int, ...] = (10, 20, 30, 45, 60)
) -> pd.DataFrame:
    """
    Cambio_pct of stock_impact_at_fda() for several windows, one column per
    window. Shows how much the "FDA impact" depends on the arbitrary window,
    which is the main reason not to read it as a causal effect.
    """
    out = None
    for w in windows:
        df = stock_impact_at_fda(stocks, drugs, w)[["Generico", "Ticker", "Fecha_FDA", "Cambio_pct"]]
        df = df.rename(columns={"Cambio_pct": f"Cambio_pct_{w}d"})
        out = df if out is None else out.merge(df, on=["Generico", "Ticker", "Fecha_FDA"], how="outer")
    return out if out is not None else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# Q3 · SEGURIDAD (FAERS)
# ─────────────────────────────────────────────────────────────────

SEVERITY_COLS = {
    "Graves (%)": "serious",
    "Muerte (%)": "seriousness_death",
    "Riesgo vital (%)": "seriousness_lifethreatening",
    "Hospitalizacion (%)": "seriousness_hospitalization",
    "Discapacidad (%)": "seriousness_disabling",
}


def resumen_gravedad(df: pd.DataFrame, nombre: str) -> dict:
    """
    Severity percentages for a subset of FAERS reports.
    Percentages are "% of *reports*", not "% of patients": FAERS is a
    spontaneous reporting system and does not give incidence.
    Returns None percentages when the subset is empty instead of dividing by 0.
    """
    total = len(df)
    out = {"Medicamento": nombre, "Total reportes": total}
    for label, col in SEVERITY_COLS.items():
        if total == 0 or col not in df.columns:
            out[label] = None
        else:
            out[label] = round(df[col].sum() / total * 100, 1)
    return out


def tabla_gravedad(ae: pd.DataFrame, marcas: list[str]) -> pd.DataFrame:
    """resumen_gravedad() for several brands, indexed by brand. Adds the
    reporting window (first/last receive_date) as exposure context."""
    rows = []
    for marca in marcas:
        sub = ae[ae["brand_queried"] == marca]
        r = resumen_gravedad(sub, marca)
        r["Primer reporte"] = str(sub["receive_date"].min().date()) if len(sub) else None
        r["Ultimo reporte"] = str(sub["receive_date"].max().date()) if len(sub) else None
        rows.append(r)
    return pd.DataFrame(rows).set_index("Medicamento")


# ─────────────────────────────────────────────────────────────────
# Q5 · PERFIL DEMOGRÁFICO (FAERS)
# ─────────────────────────────────────────────────────────────────

AGE_BINS = [0, 18, 30, 45, 60, 75, 200]
AGE_LABELS = ["<18", "18-29", "30-44", "45-59", "60-74", "75+"]


def _age_in_years(df: pd.DataFrame) -> pd.Series:
    """Age in years (only rows whose unit is 'Year'; 'Decade' and 'Unknown' → NaN)."""
    age = pd.to_numeric(df["patient_age"], errors="coerce")
    return age.where(df["patient_age_unit"] == "Year")


def demographic_profile(ae: pd.DataFrame, marcas: list[str]) -> dict[str, pd.DataFrame]:
    """
    Q5 — demographic profile of the reports for the given brands.
    Returns a dict of tidy DataFrames (all with a 'Medicamento' column):
      - 'sexo'          : n and % of reports by patient_sex
      - 'edad'          : n, mean, median, % with known age
      - 'grupo_edad'    : n and % by age band
      - 'pais'          : top countries (n and %)
      - 'gravedad_sexo' : % serious / hospitalization by sex
      - 'gravedad_edad' : % serious / hospitalization by age band
    Percentages are over reports with a known value (Unknown shown separately).
    """
    out = {k: [] for k in ["sexo", "edad", "grupo_edad", "pais", "gravedad_sexo", "gravedad_edad"]}

    for marca in marcas:
        sub = ae[ae["brand_queried"] == marca].copy()
        n = len(sub)
        if n == 0:
            continue
        sub["edad_anios"] = _age_in_years(sub)
        sub["grupo_edad"] = pd.cut(sub["edad_anios"], bins=AGE_BINS, labels=AGE_LABELS, right=False)

        # sexo
        vc = sub["patient_sex"].fillna("Unknown").value_counts()
        for sexo, cnt in vc.items():
            out["sexo"].append({"Medicamento": marca, "Sexo": sexo, "N": int(cnt), "Pct": round(cnt / n * 100, 1)})

        # edad
        edad = sub["edad_anios"].dropna()
        out["edad"].append({
            "Medicamento": marca, "N reportes": n, "N con edad": int(len(edad)),
            "Pct con edad": round(len(edad) / n * 100, 1),
            "Edad media": round(edad.mean(), 1) if len(edad) else None,
            "Edad mediana": round(edad.median(), 1) if len(edad) else None,
        })

        # grupo de edad
        vg = sub["grupo_edad"].value_counts().sort_index()
        for g, cnt in vg.items():
            out["grupo_edad"].append({"Medicamento": marca, "Grupo edad": str(g), "N": int(cnt),
                                      "Pct": round(cnt / max(len(edad), 1) * 100, 1)})

        # país
        vp = sub["country"].fillna("UNK").value_counts().head(8)
        for pais, cnt in vp.items():
            out["pais"].append({"Medicamento": marca, "Pais": pais, "N": int(cnt), "Pct": round(cnt / n * 100, 1)})

        # gravedad por sexo / edad
        for sexo, g in sub.groupby(sub["patient_sex"].fillna("Unknown")):
            out["gravedad_sexo"].append({
                "Medicamento": marca, "Sexo": sexo, "N": len(g),
                "Graves (%)": round(g["serious"].mean() * 100, 1),
                "Hospitalizacion (%)": round(g["seriousness_hospitalization"].mean() * 100, 1),
                "Muerte (%)": round(g["seriousness_death"].mean() * 100, 1),
            })
        for grupo, g in sub.dropna(subset=["grupo_edad"]).groupby("grupo_edad", observed=True):
            out["gravedad_edad"].append({
                "Medicamento": marca, "Grupo edad": str(grupo), "N": len(g),
                "Graves (%)": round(g["serious"].mean() * 100, 1),
                "Hospitalizacion (%)": round(g["seriousness_hospitalization"].mean() * 100, 1),
                "Muerte (%)": round(g["seriousness_death"].mean() * 100, 1),
            })

    return {k: pd.DataFrame(v) for k, v in out.items()}


# ─────────────────────────────────────────────────────────────────
# Q2 · EFICACIA CLÍNICA
# ─────────────────────────────────────────────────────────────────

# ⚠️  DATOS DE LITERATURA, NO DEL DATASET.
# clinical_trials.csv (ClinicalTrials.gov) NO contiene resultados de eficacia,
# solo el registro del ensayo (título, fase, n, fechas). La pérdida de peso
# se toma de las publicaciones primarias de cada ensayo y se cruza por NCT.
CLINICAL_WEIGHT_LOSS = [
    {"medicamento": "semaglutide", "marca": "WEGOVY",   "dosis": "2.4 mg/sem",  "ensayo": "STEP 1",
     "nct_id": "NCT03548935", "N": 1961, "duracion_semanas": 68, "perdida_peso_pct": -14.9, "vs_placebo_pp": -12.4,
     "poblacion": "Obesidad/sobrepeso sin DM2",
     "fuente": "Wilding et al., NEJM 2021;384:989-1002"},
    {"medicamento": "tirzepatide", "marca": "ZEPBOUND", "dosis": "15 mg/sem",   "ensayo": "SURMOUNT-1",
     "nct_id": "NCT04184622", "N": 2539, "duracion_semanas": 72, "perdida_peso_pct": -20.9, "vs_placebo_pp": -17.8,
     "poblacion": "Obesidad/sobrepeso sin DM2",
     "fuente": "Jastreboff et al., NEJM 2022;387:205-216"},
    {"medicamento": "semaglutide", "marca": "OZEMPIC",  "dosis": "1 mg/sem",    "ensayo": "SURPASS-2",
     "nct_id": "NCT03987919", "N": 1879, "duracion_semanas": 40, "perdida_peso_pct": -5.7,  "vs_placebo_pp": None,
     "poblacion": "DM2 (comparador activo)",
     "fuente": "Frias et al., NEJM 2021;385:503-515"},
    {"medicamento": "tirzepatide", "marca": "MOUNJARO", "dosis": "15 mg/sem",   "ensayo": "SURPASS-2",
     "nct_id": "NCT03987919", "N": 1879, "duracion_semanas": 40, "perdida_peso_pct": -11.2, "vs_placebo_pp": None,
     "poblacion": "DM2 (vs semaglutide 1 mg)",
     "fuente": "Frias et al., NEJM 2021;385:503-515"},
]


def weight_trials_from_dataset(trials: pd.DataFrame) -> pd.DataFrame:
    """
    Q2 (parte que SÍ sale del dataset): completed semaglutide/tirzepatide
    trials in clinical_trials.csv whose title or conditions mention weight/obesity.
    """
    pattern = "weight|obesity|obese|overweight|STEP|SURMOUNT|SURPASS"
    mask = (
        trials["drug_query"].isin(["semaglutide", "tirzepatide"])
        & trials["overall_status"].eq("COMPLETED")
        & (
            trials["brief_title"].str.contains(pattern, case=False, na=False)
            | trials["conditions"].str.contains("obes|weight|overweight", case=False, na=False)
        )
    )
    cols = ["drug_query", "nct_id", "brief_title", "phase", "enrollment", "start_date", "completion_date"]
    return trials.loc[mask, cols].drop_duplicates(subset="nct_id").reset_index(drop=True)


def clinical_weight_loss_summary(trials: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Literature table of weight-loss outcomes (CLINICAL_WEIGHT_LOSS) with a
    'origen' column that makes the provenance explicit. If `trials` is given,
    each NCT is cross-checked against clinical_trials.csv and the dataset's
    own enrollment/status are appended (en_dataset, N_dataset, estado_dataset).
    """
    df = pd.DataFrame(CLINICAL_WEIGHT_LOSS)
    df["origen"] = "literatura (publicacion primaria)"
    if trials is not None:
        ref = (
            trials.drop_duplicates(subset="nct_id")
            .set_index("nct_id")[["enrollment", "overall_status"]]
            .rename(columns={"enrollment": "N_dataset", "overall_status": "estado_dataset"})
        )
        df = df.join(ref, on="nct_id")
        df["en_dataset"] = df["nct_id"].isin(ref.index)
    return df
