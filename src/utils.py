import pandas as pd


def assert_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def assert_no_nulls(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise ValueError if any of the specified columns contains nulls."""
    for col in cols:
        n = df[col].isnull().sum()
        if n > 0:
            raise ValueError(f"Column '{col}' has {n} null values after cleaning")


def print_dataset_summary(datasets: dict) -> None:
    """Print a quick shape/nulls/duplicates summary for each dataset in the dict."""
    print(f"{'Archivo':<30} {'Filas':>8} {'Cols':>5} {'Nulos%':>8} {'Duplic.':>8}")
    print("-" * 65)
    for name, df in datasets.items():
        null_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        dups = df.duplicated().sum()
        print(f"{name:<30} {df.shape[0]:>8,} {df.shape[1]:>5} {null_pct:>7.1f}% {dups:>8}")


def build_quality_report(raw: dict, clean: dict, drugs_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Quality-check table comparing each raw dataset with its cleaned version.
    raw / clean: dicts keyed by dataset name with the *same* keys
    (e.g. {"drugs_overview": df, "stock_prices": df, "adverse_events": df}).
    Rows: filas antes/después, duplicados eliminados, nulos antes/después
    and a few dataset-specific checks (fechas FDA fuera de rango, booleanos convertidos).
    """
    rows = []
    for name, df_raw in raw.items():
        df_clean = clean.get(name, df_raw)
        rows.append({
            "Dataset": name,
            "Filas_antes": len(df_raw),
            "Filas_despues": len(df_clean),
            "Filas_eliminadas": len(df_raw) - len(df_clean),
            "Duplicados_exactos_raw": int(df_raw.duplicated().sum()),
            "Nulos_antes": int(df_raw.isnull().sum().sum()),
            "Nulos_despues": int(df_clean.isnull().sum().sum()),
            "Cols_antes": df_raw.shape[1],
            "Cols_despues": df_clean.shape[1],
        })
    report = pd.DataFrame(rows).set_index("Dataset")

    # checks específicos
    if "stock_prices" in raw:
        dups = raw["stock_prices"].duplicated(subset=["ticker", "date"]).sum()
        report.loc["stock_prices", "Duplicados_ticker_fecha"] = int(dups)
    if "adverse_events" in raw:
        bool_cols = [c for c in raw["adverse_events"].columns if c.startswith("serious")]
        report.loc["adverse_events", "Booleanos_convertidos"] = len(bool_cols)
    if "fecha_en_rango_stock" in drugs_clean.columns:
        con_fecha = drugs_clean["fda_first_approval_date"].notna() & drugs_clean["ticker"].notna()
        fuera = int((con_fecha & ~drugs_clean["fecha_en_rango_stock"]).sum())
        report.loc["drugs_overview", "FDA_fuera_rango_stock"] = fuera
        report.loc["drugs_overview", "FDA_sin_fecha"] = int(drugs_clean["fda_first_approval_date"].isna().sum())

    return report
