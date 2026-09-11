"""
Pipeline end-to-end: load → quality checks → clean → features → export (CSV + PNG).
Run from the project root:  python main.py

Salidas
  data/processed/  drugs_clean.csv, stocks_clean.csv, quality_checks.csv,
                   fda_stock_impact.csv, fda_stock_impact_sensitivity.csv, stock_growth_summary.csv,
                   adverse_events_gravedad.csv, demografia_*.csv,
                   clinical_weight_loss_summary.csv, weight_trials_dataset.csv
  output/          crecimiento_stock_glp1.png, comparacion_efectos_adversos.png,
                   comparativa_perdida_peso.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin ventana: las figuras solo se guardan a disco

from src.io import load_all_datasets
from src.cleaning import clean_drugs, clean_stocks, clean_adverse_events, add_fda_in_stock_range
from src.features import (
    stock_normalized_growth, stock_growth_summary, stock_impact_at_fda, stock_impact_sensitivity,
    tabla_gravedad, demographic_profile,
    clinical_weight_loss_summary, weight_trials_from_dataset,
    HITOS_BURSATILES, AGE_LABELS,
)
from src.viz import plot_stock_comparison, plot_adverse_events, plot_weight_loss
from src.utils import print_dataset_summary, assert_columns, build_quality_report

BASE = Path(__file__).resolve().parent
DATA_RAW = BASE / "data" / "raw"
DATA_OUT = BASE / "data" / "processed"
OUTPUT = BASE / "output"

FECHA_INICIO = "2017-01-01"
MARCAS_AE = ["OZEMPIC", "MOUNJARO"]


def main():
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)

    # 1) LOAD ------------------------------------------------------------
    print("=== 1. CARGA ===")
    datasets = load_all_datasets(DATA_RAW)
    print_dataset_summary(datasets)

    # 2) CLEAN + QUALITY CHECKS -----------------------------------------
    print("\n=== 2. LIMPIEZA ===")
    drugs  = clean_drugs(datasets["drugs_overview"])
    stocks = clean_stocks(datasets["stock_prices"])
    ae     = clean_adverse_events(datasets["adverse_events"])
    aes    = datasets["adverse_events_summary"]
    trials = datasets["clinical_trials"]
    drugs  = add_fda_in_stock_range(drugs, stocks)

    assert_columns(drugs,  ["generic_name", "brand_names", "ticker", "fda_first_approval_date", "fecha_en_rango_stock"])
    assert_columns(stocks, ["ticker", "date", "close"])
    assert_columns(ae,     ["brand_queried", "serious", "seriousness_death", "patient_sex", "patient_age", "country"])
    print("  Validaciones OK")

    quality = build_quality_report(
        raw={k: datasets[k] for k in ["drugs_overview", "stock_prices", "adverse_events"]},
        clean={"drugs_overview": drugs, "stock_prices": stocks, "adverse_events": ae},
        drugs_clean=drugs,
    )
    print("\n  Quality checks (antes → después):")
    print(quality.to_string())

    # 3) FEATURES --------------------------------------------------------
    print("\n=== 3. FEATURES ===")
    fecha_fin = str(stocks["date"].max().date())

    # Q1 · impacto FDA (ventana ±30 días naturales, descriptivo)
    df_impact = stock_impact_at_fda(stocks, drugs, window_days=30)
    print("  [Q1] Impacto FDA en stock (±30d, sin controlar mercado):")
    print(df_impact[["Generico", "Ticker", "Fecha_FDA", "Cambio_pct"]].to_string(index=False))
    df_sens = stock_impact_sensitivity(stocks, drugs)
    print("  [Q1] Sensibilidad a la ventana (el signo cambia → no leer como causalidad):")
    print(df_sens.drop(columns="Fecha_FDA").to_string(index=False))

    # Q4 · crecimiento por tramos + drawdown
    df_growth = stock_growth_summary(stocks, FECHA_INICIO, fecha_fin, HITOS_BURSATILES)
    print(f"\n  [Q4] Crecimiento bursátil {FECHA_INICIO} → {fecha_fin} (último dato del dataset):")
    print(df_growth.T.to_string())

    # Q3 · gravedad FAERS
    df_gravedad = tabla_gravedad(ae, MARCAS_AE)
    print("\n  [Q3] Gravedad de reportes FAERS (% sobre reportes, no incidencia):")
    print(df_gravedad.to_string())

    # Q5 · perfil demográfico
    demo = demographic_profile(ae, MARCAS_AE)
    print("\n  [Q5] Perfil demográfico de los reportes:")
    print(demo["edad"].to_string(index=False))
    print(demo["sexo"].pivot(index="Sexo", columns="Medicamento", values="Pct").to_string())
    print(demo["gravedad_edad"].pivot(index="Grupo edad", columns="Medicamento", values="Graves (%)")
          .reindex(AGE_LABELS).to_string())

    # Q2 · eficacia clínica: registro (dataset) vs resultados (literatura)
    df_trials_ds = weight_trials_from_dataset(trials)
    df_clinical = clinical_weight_loss_summary(trials)
    print(f"\n  [Q2] Ensayos completados sobre peso/obesidad en clinical_trials.csv: {len(df_trials_ds)}")
    print(df_trials_ds.groupby("drug_query").size().to_string())
    print("  [Q2] Resultados de pérdida de peso (LITERATURA, cruzados por NCT con el dataset):")
    print(df_clinical[["marca", "ensayo", "nct_id", "N", "perdida_peso_pct", "poblacion", "en_dataset"]].to_string(index=False))

    # 4) EXPORT CSV -----------------------------------------------------
    print("\n=== 4. EXPORT CSV ===")
    drugs.to_csv(DATA_OUT / "drugs_clean.csv", index=False)
    stocks.to_csv(DATA_OUT / "stocks_clean.csv", index=False)
    quality.to_csv(DATA_OUT / "quality_checks.csv")
    df_impact.to_csv(DATA_OUT / "fda_stock_impact.csv", index=False)
    df_sens.to_csv(DATA_OUT / "fda_stock_impact_sensitivity.csv", index=False)
    df_growth.to_csv(DATA_OUT / "stock_growth_summary.csv")
    df_gravedad.to_csv(DATA_OUT / "adverse_events_gravedad.csv")
    for nombre, df in demo.items():
        df.to_csv(DATA_OUT / f"demografia_{nombre}.csv", index=False)
    df_clinical.to_csv(DATA_OUT / "clinical_weight_loss_summary.csv", index=False)
    df_trials_ds.to_csv(DATA_OUT / "weight_trials_dataset.csv", index=False)
    print(f"  Archivos guardados en: {DATA_OUT}")

    # 5) EXPORT PNG -----------------------------------------------------
    print("\n=== 5. GRÁFICOS ===")
    lly = stock_normalized_growth(stocks, "LLY", FECHA_INICIO, fecha_fin)
    nvo = stock_normalized_growth(stocks, "NVO", FECHA_INICIO, fecha_fin)
    hitos = drugs[drugs["fecha_en_rango_stock"] == True]
    plot_stock_comparison(lly, nvo, hitos, FECHA_INICIO, fecha_fin,
                          save_path=OUTPUT / "crecimiento_stock_glp1.png", show=False)

    ozempic, mounjaro = (ae[ae["brand_queried"] == m] for m in MARCAS_AE)
    ozempic_s  = aes[aes["generic_name"] == "semaglutide"]
    mounjaro_s = aes[aes["generic_name"] == "tirzepatide"]
    plot_adverse_events(ozempic, mounjaro, ozempic_s, mounjaro_s, df_gravedad,
                        save_path=OUTPUT / "comparacion_efectos_adversos.png", show=False)

    plot_weight_loss(df_clinical, save_path=OUTPUT / "comparativa_perdida_peso.png", show=False)
    for png in ["crecimiento_stock_glp1.png", "comparacion_efectos_adversos.png", "comparativa_perdida_peso.png"]:
        print(f"  {OUTPUT / png}")

    print("\n=== Pipeline completado ===")


if __name__ == "__main__":
    main()
