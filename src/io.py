from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a single CSV file into a DataFrame."""
    return pd.read_csv(path)


def load_all_datasets(base_path: str | Path) -> dict[str, pd.DataFrame]:
    """Load all 8 raw CSVs from data/raw/. Returns a dict keyed by stem name."""
    base = Path(base_path)
    names = [
        "drugs_overview",
        "stock_prices",
        "adverse_events",
        "adverse_events_summary",
        "clinical_trials",
        "search_trends",
        "wikipedia_summaries",
        "data_dictionary",
    ]
    datasets = {}
    for name in names:
        path = base / f"{name}.csv"
        if path.exists():
            datasets[name] = pd.read_csv(path)
        else:
            print(f"[WARNING] {path} not found — skipped")
    return datasets
