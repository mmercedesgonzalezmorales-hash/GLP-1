# Proyecto GLP-1: Análisis de Medicamentos Agonistas del Receptor GLP-1

## 1. Objetivo
Identificar patrones clínicos, comparar la eficacia y el perfil de reportes de seguridad entre los principales medicamentos GLP-1 (semaglutide y tirzepatide), y relacionar los hitos regulatorios de la FDA con el comportamiento bursátil de Eli Lilly y Novo Nordisk.

## 2. Dataset
- **Fuente**: FDA FAERS, ClinicalTrials.gov, Yahoo Finance, Wikipedia — [Kaggle: GLP-1 Weight Loss Drugs Master Dataset 2017-2026](https://www.kaggle.com/datasets/devtayyabsajjad/glp-1-weight-loss-drugs-master-dataset-2017-2026?resource=download)
- **Archivos**: 8 CSVs originales en `data/raw/` (sin modificar)
- **Filas totales**: ~170,000 (adverse_events domina con 149,209)
- **Rango temporal**: stocks 2017-01-03 → 2026-04-24; FAERS 2012 → 2025 (pero con ventanas distintas por marca, ver §4)
- **Variables clave**: `brand_queried`, `generic_name`, `serious`, `seriousness_*`, `patient_age`, `patient_sex`, `country`, `close`, `fda_first_approval_date`, `enrollment`

### Dos niveles de evidencia
| Nivel | Qué es | Dónde se ve |
|---|---|---|
| **Dataset (EDA reproducible)** | Todo lo que sale de los CSV: stocks, FAERS, registro de ensayos | `main.py`, notebook, `data/processed/` |
| **Literatura (contexto)** | Resultados de pérdida de peso de STEP 1, SURMOUNT-1 y SURPASS-2. `clinical_trials.csv` solo trae el *registro* del ensayo (NCT, fase, n, fechas), **no los resultados** | `src/features.py::CLINICAL_WEIGHT_LOSS`, exportado a `clinical_weight_loss_summary.csv` con columna `origen` y `fuente`, y cruzado por NCT con el dataset (`en_dataset`) |

## 3. Preguntas
- **Q1**: ¿Cómo se movió la acción de LLY y NVO alrededor de las aprobaciones de la FDA? *(descriptivo, no causal)*
- **Q2**: ¿Qué medicamento genera más pérdida de peso: semaglutide o tirzepatide? *(literatura, cruzada con el registro de ensayos del dataset)*
- **Q3**: ¿Cuál tiene un perfil de **reportes** adversos más grave en FDA FAERS? *(% sobre reportes, no incidencia)*
- **Q4**: ¿Qué compañía ha tenido mayor crecimiento bursátil desde 2017, y cómo se reparte por tramos (total / desde MOUNJARO / desde ZEPBOUND / drawdown)?
- **Q5**: ¿Cuál es el perfil demográfico (edad, sexo, país) de los reportes adversos, y cómo varía la gravedad por grupo?

## 4. Data issues & fixes
| Problema | Solución |
|---|---|
| `fda_first_approval_date` como string | `pd.to_datetime(..., errors='coerce')` |
| `brand_names` con 3 nulos (investigacionales) | `.fillna('Sin marca (investigacional)')` |
| Columnas booleanas en adverse_events guardadas como strings `'True'/'False'` | `.map({True:1, False:0, 'True':1, 'False':0})` |
| Posibles duplicados en stock_prices | `.drop_duplicates(subset=['ticker','date'])` (0 encontrados) |
| Aprobaciones FDA anteriores a 2017 (fuera del rango de stock) | Columna `fecha_en_rango_stock`; 2 fármacos excluidos del análisis de impacto |
| `patient_age` con unidad `Unknown`/`Decade` | Solo se usa edad cuando `patient_age_unit == 'Year'` |
| **Ventanas de reporte FAERS distintas por marca**: OZEMPIC hasta jul-2020, MOUNJARO concentrado en 2022 (90% de sus reportes), y reportes anteriores a la aprobación | No se corrige (es el dataset); se imprime la ventana junto a cada % y se documenta como limitación en §6 |
| ZEPBOUND no está en `drugs_overview` (solo primera aprobación por genérico) | Fecha FDA 2023-11-08 añadida a mano en `HITOS_BURSATILES` con su fuente |

### Quality checks (`data/processed/quality_checks.csv`)
| Dataset | Filas antes → después | Duplicados | Nulos | Check específico |
|---|---|---|---|---|
| drugs_overview | 10 → 10 | 0 | 6 | 2 fechas FDA fuera del rango de stock, 3 sin fecha (investigacionales) |
| stock_prices | 4,680 → 4,680 | 0 (ticker+fecha) | 0 | — |
| adverse_events | 149,209 → 149,209 | 0 | 159,310 | 5 columnas booleanas convertidas a 0/1 |

## 5. Pipeline
```
data/raw/ (8 CSVs originales, sin tocar)
    → src/io.py        load_all_datasets
    → src/cleaning.py  clean_drugs, clean_stocks, clean_adverse_events, add_fda_in_stock_range
    → src/utils.py     assert_columns, build_quality_report
    → src/features.py  stock_growth_summary, stock_impact_at_fda (+sensitivity), tabla_gravedad,
                       demographic_profile, weight_trials_from_dataset, clinical_weight_loss_summary
    → src/viz.py       plot_stock_comparison, plot_adverse_events, plot_weight_loss
    → data/processed/  CSVs procesados (15 archivos)
    → output/          PNGs
```
`python main.py` ejecuta todo esto y genera **CSVs y gráficos**; el notebook usa las mismas funciones de `src/` y añade la narrativa.

## 6. Hallazgos

### Q4 · Bolsa (del dataset, hasta 2026-04-24)
| | LLY | NVO |
|---|---|---|
| Crecimiento total desde ene-2017 | **+1,085%** | +129% |
| Desde MOUNJARO (may-2022) | +203% | −22% |
| Desde ZEPBOUND (nov-2023) | +43% | −60% |
| Máximo | $1,110 (nov-2025) | $147 (jun-2024) |
| Max drawdown | −35% | **−76%** (jun-2024 → mar-2026) |

La brecha se abre a partir de la aprobación de MOUNJARO. Es coherente con el éxito comercial de tirzepatide, pero es una **correlación temporal**, no un efecto medido.

### Q1 · Impacto FDA (descriptivo)
Con ventana ±30 días: OZEMPIC/NVO +9.1%, MOUNJARO/LLY −3.8%. **El resultado depende de la ventana** (`fda_stock_impact_sensitivity.csv`): MOUNJARO va de +8.5% (20d) a −3.8% (30d) a +20.3% (60d). Un cambio de precio alrededor de una aprobación mezcla mercado general, resultados trimestrales, guidance y expectativas ya descontadas; no aísla el efecto del hito. Para inferencia haría falta un event study con benchmark.

### Q2 · Eficacia (literatura, no del CSV)
| Fármaco | Ensayo | N | Semanas | Pérdida de peso | Población |
|---|---|---|---|---|---|
| Tirzepatide 15 mg (ZEPBOUND) | SURMOUNT-1 | 2,539 | 72 | **−20.9%** | Obesidad sin DM2 |
| Semaglutide 2.4 mg (WEGOVY) | STEP 1 | 1,961 | 68 | −14.9% | Obesidad sin DM2 |
| Tirzepatide 15 mg (MOUNJARO) | SURPASS-2 | 1,879 | 40 | −11.2% | DM2 |
| Semaglutide 1 mg (OZEMPIC) | SURPASS-2 | 1,879 | 40 | −5.7% | DM2 |

Los cuatro NCT existen en `clinical_trials.csv` como COMPLETED con el mismo `enrollment`. El dataset aporta además el registro de 128 ensayos completados sobre peso/obesidad (102 semaglutide, 26 tirzepatide). STEP 1 y SURMOUNT-1 son ensayos distintos; el único head-to-head es SURPASS-2 (en DM2, dosis de diabetes).

### Q3 · Seguridad: reportes FAERS (% sobre reportes, **no incidencia**)
| | OZEMPIC | MOUNJARO |
|---|---|---|
| Reportes | 16,530 (2014 → jul-2020) | 11,855 (2016 → dic-2022) |
| Graves | 49.8% | 19.2% |
| Hospitalización | 23.9% | 9.3% |
| Muerte | 1.2% | 0.2% |

**Cómo leerlo.** "49.8% de reportes graves" significa que la mitad de los *reportes recibidos* sobre OZEMPIC fueron marcados como graves — **no** que el 49.8% de los pacientes tenga un evento grave. FAERS es notificación espontánea:
- **Sin denominador de exposición**: no se sabe cuántos pacientes tomaron cada fármaco. Más reportes puede reflejar más uso, más tiempo en mercado o más vigilancia/atención mediática.
- **Sesgo de notificación**: los eventos graves se reportan más; la propensión a reportar cambia por fármaco y año.
- **Ventanas no comparables en este dataset**: OZEMPIC cubre 2018-2020 (fármaco ya maduro) y MOUNJARO casi solo 2022 (primeros 7 meses de comercialización, cuando dominan errores de uso).
- Las reacciones más frecuentes de MOUNJARO son **errores de dosificación** ("Incorrect dose administered", "Extra dose"), no toxicidad clínica.

Conclusión defendible: *el patrón de reportes es distinto*; **no** que un fármaco sea más seguro que otro.

### Q5 · Perfil demográfico de los reportes
| | OZEMPIC | MOUNJARO |
|---|---|---|
| Edad media (reportes con edad) | 60.3 años (67% con edad) | 49.4 años (74% con edad) |
| Mujeres | 60.5% | 74.5% |
| País principal | US | US |
| % graves por edad (grupos con n ≥ 100) | Máx. en 18-29 (84%), baja hasta 43% en 75+ | Máx. en 60-74 (31%), mín. en 30-44 (9%) |

Los reportes de MOUNJARO corresponden a pacientes más jóvenes y más mujeres (perfil obesidad vs. DM2). La gravedad por edad no sigue el mismo patrón en ambos, así que la comparación global de Q3 mezcla poblaciones y contextos de notificación distintos. Tablas completas en `data/processed/demografia_*.csv`.

## 7. Estructura del proyecto
```
Proyecto_GLP-1/
├── main.py                          # Entrypoint: pipeline completo (CSVs + PNGs)
├── README.md
├── requirements.txt
├── .gitignore                       # excluye __pycache__
├── data/
│   ├── raw/                         # 8 CSVs originales (solo lectura)
│   └── processed/                   # Salida de main.py / notebook
│       ├── drugs_clean.csv, stocks_clean.csv, quality_checks.csv
│       ├── fda_stock_impact.csv, fda_stock_impact_sensitivity.csv, stock_growth_summary.csv
│       ├── adverse_events_gravedad.csv, demografia_{sexo,edad,grupo_edad,pais,gravedad_sexo,gravedad_edad}.csv
│       └── clinical_weight_loss_summary.csv, weight_trials_dataset.csv
├── output/
│   ├── crecimiento_stock_glp1.png       # main.py y notebook
│   ├── comparacion_efectos_adversos.png # main.py y notebook
│   ├── comparativa_perdida_peso.png     # main.py y notebook
│   └── conclusiones_resumen.png         # solo notebook
├── src/
│   ├── __init__.py
│   ├── io.py                        # Carga de CSVs
│   ├── cleaning.py                  # Limpieza de drugs, stocks, adverse_events
│   ├── features.py                  # Bolsa, gravedad FAERS, demografía, tabla clínica (literatura)
│   ├── viz.py                       # Gráficos reutilizables (dark theme)
│   └── utils.py                     # Validaciones, resumen y quality report
└── notebook/
    └── Medicamentos GLP-1.ipynb     # Notebook narrativo (usa src/)
```

## 8. Cómo ejecutar
```bash
pip install -r requirements.txt

# Pipeline completo: limpia, calcula, exporta CSVs a data/processed/ y PNGs a output/
python main.py

# Notebook (misma lógica, con narrativa): abrir notebook/Medicamentos GLP-1.ipynb y "Run All"
```

## 9. Limitaciones
- FAERS no permite estimar incidencia ni comparar riesgo entre fármacos (ver §6 Q3).
- El impacto bursátil ±N días es descriptivo; no hay benchmark de mercado ni control de eventos concurrentes.
- Los resultados de eficacia no están en el dataset; vienen de literatura y se marcan como tal.
- El dataset de stocks llega a abril-2026, así que "crecimiento total" incluye datos muy recientes y sujetos a revisión; por eso se reporta también por tramos y con drawdown.
