import pandas as pd


TICKER_MAP = {"Eli Lilly": "LLY", "Novo Nordisk": "NVO"}

BOOL_COLS = [
    "serious",
    "seriousness_death",
    "seriousness_lifethreatening",
    "seriousness_hospitalization",
    "seriousness_disabling",
]


def clean_drugs(df: pd.DataFrame) -> pd.DataFrame:
    """Parse FDA date, fill missing brand names, add ticker column."""
    out = df.copy()
    out["fda_first_approval_date"] = pd.to_datetime(
        out["fda_first_approval_date"], errors="coerce"
    )
    out["brand_names"] = out["brand_names"].fillna("Sin marca (investigacional)")
    out["ticker"] = out["manufacturer"].map(TICKER_MAP)
    return out


def clean_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, drop invalid prices and duplicates, sort."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out[out["close"] > 0]
    out = out.drop_duplicates(subset=["ticker", "date"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def clean_adverse_events(df: pd.DataFrame) -> pd.DataFrame:
    """Parse receive_date and normalize boolean severity columns to int (0/1)."""
    out = df.copy()
    out["receive_date"] = pd.to_datetime(out["receive_date"], errors="coerce")
    for col in BOOL_COLS:
        if col in out.columns:
            out[col] = (
                out[col]
                .map({True: 1, False: 0, "True": 1, "False": 0})
                .fillna(0)
                .astype(int)
            )
    return out


def add_fda_in_stock_range(drugs: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean column 'fecha_en_rango_stock' to drugs:
    True if the drug's FDA approval date falls within the available stock data window.
    """
    out = drugs.copy()
    stock_inicio = stocks.groupby("ticker")["date"].min()
    stock_fin = stocks.groupby("ticker")["date"].max()

    out["fecha_en_rango_stock"] = out.apply(
        lambda r: (
            pd.notna(r["fda_first_approval_date"])
            and pd.notna(r["ticker"])
            and stock_inicio.get(r["ticker"], pd.NaT)
            <= r["fda_first_approval_date"]
            <= stock_fin.get(r["ticker"], pd.NaT)
        ),
        axis=1,
    )
    return out
