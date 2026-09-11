import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


COLOR_LLY = "#4fc3f7"
COLOR_NVO = "#ef9a9a"
COLOR_OZ  = "#ef9a9a"
COLOR_MO  = "#4fc3f7"
DARK_BG   = "#0f1117"
CARD_BG   = "#1a1d27"


def _finish(fig, save_path, show):
    """Save (if save_path) and show (if show); always close when not showing."""
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    else:
        plt.close(fig)


def _style_ax(ax):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors="#cccccc", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color("#cccccc")
    ax.xaxis.label.set_color("#cccccc")
    ax.title.set_color("white")
    ax.grid(True, linestyle="--", alpha=0.2, color="#555555")


def plot_stock_comparison(
    lly: pd.DataFrame,
    nvo: pd.DataFrame,
    hitos: pd.DataFrame,
    fecha_inicio: str = "2017-01-01",
    fecha_fin: str = "2025-12-31",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Two-panel dark-theme chart: closing price (top) and base-100 growth (bottom).
    hitos: drugs DataFrame rows with fecha_en_rango_stock == True.
    show=False (e.g. from main.py) saves the figure without opening a window.
    """
    zona_inicio = pd.Timestamp("2022-05-01")
    zona_fin    = pd.Timestamp("2024-06-30")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor(DARK_BG)

    for ax in (ax1, ax2):
        _style_ax(ax)
        ax.axvspan(zona_inicio, zona_fin, color="#f9a825", alpha=0.07, zorder=0)
        ax.axvline(zona_inicio, color="#f9a825", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(zona_fin,    color="#f9a825", linewidth=0.8, linestyle="--", alpha=0.5)

    ax1.plot(lly.index, lly["close"], color=COLOR_LLY, linewidth=1.8, label="LLY - Eli Lilly")
    ax1.plot(nvo.index, nvo["close"], color=COLOR_NVO, linewidth=1.8, label="NVO - Novo Nordisk")
    ax1.set_title("Precio de cierre (USD)", fontsize=13, pad=10)
    ax1.set_ylabel("USD", fontsize=11)
    ax1.legend(facecolor="#1e1e2e", edgecolor="#444444", labelcolor="white", fontsize=10)
    ax1.text(
        zona_inicio + (zona_fin - zona_inicio) / 2,
        ax1.get_ylim()[1] * 0.92,
        "Periodo clave\nLLY supera a NVO",
        color="#f9a825", fontsize=8.5, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e1e2e", edgecolor="#f9a825", alpha=0.85),
    )

    ax2.plot(lly.index, lly["crecimiento"], color=COLOR_LLY, linewidth=1.8)
    ax2.plot(nvo.index, nvo["crecimiento"], color=COLOR_NVO, linewidth=1.8)
    ax2.axhline(100, color="#555555", linewidth=0.8, linestyle="--")
    ax2.set_title("Crecimiento relativo (base 100 = enero 2017)", fontsize=13, pad=10)
    ax2.set_ylabel("Indice (base 100)", fontsize=11)
    ax2.text(
        zona_inicio + (zona_fin - zona_inicio) / 2,
        ax2.get_ylim()[1] * 0.95,
        "Aprobacion MOUNJARO\nhasta pico NVO",
        color="#f9a825", fontsize=8.5, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e1e2e", edgecolor="#f9a825", alpha=0.85),
    )

    for _, row in hitos.iterrows():
        fecha  = row["fda_first_approval_date"]
        color  = COLOR_LLY if row["ticker"] == "LLY" else COLOR_NVO
        marca  = row["brand_names"].split(";")[0].strip()
        serie  = lly if row["ticker"] == "LLY" else nvo
        tramo  = serie["crecimiento"].loc[fecha:]
        if tramo.empty:          # hito fuera del rango dibujado → sin anotación
            continue
        for ax in (ax1, ax2):
            ax.axvline(fecha, color=color, linewidth=1, linestyle=":", alpha=0.7)
        y_pos = tramo.iloc[0]
        ax2.annotate(
            marca, xy=(fecha, y_pos), xytext=(10, 25), textcoords="offset points",
            color=color, fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e1e2e", edgecolor=color, alpha=0.9),
        )

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xlim(pd.Timestamp(fecha_inicio), pd.Timestamp(fecha_fin))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

    ultimo = max(lly.index.max(), nvo.index.max())
    fig.suptitle(
        f"Eli Lilly (LLY) vs Novo Nordisk (NVO)  |  {fecha_inicio[:4]}-{ultimo.year}\n"
        "Evolucion bursatil y aprobaciones FDA GLP-1",
        fontsize=14, color="white", y=1.02,
    )
    fig.text(
        0.5, -0.01,
        "Lectura descriptiva: los movimientos alrededor de una aprobacion FDA no aislan el efecto del hito "
        "(mercado general, resultados trimestrales, guidance, expectativas ya descontadas).",
        ha="center", color="#aaaaaa", fontsize=8.5, style="italic",
    )
    plt.tight_layout()
    _finish(fig, save_path, show)


def plot_adverse_events(
    ozempic: pd.DataFrame,
    mounjaro: pd.DataFrame,
    ozempic_s: pd.DataFrame,
    mounjaro_s: pd.DataFrame,
    df_gravedad: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Dark-theme figure with 3 panels:
    - Top: grouped bar chart comparing severity categories
    - Bottom-left: top reactions OZEMPIC
    - Bottom-right: top reactions MOUNJARO
    """
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    _style_ax(ax1)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.15, color="#555555")

    categorias = ["Graves (%)", "Muerte (%)", "Riesgo vital (%)", "Hospitalizacion (%)", "Discapacidad (%)"]
    etiquetas  = ["Graves", "Muerte", "Riesgo vital", "Hospitalizacion", "Discapacidad"]
    x = np.arange(len(categorias))
    ancho = 0.35
    vals_oz = [df_gravedad.loc["OZEMPIC",  c] for c in categorias]
    vals_mo = [df_gravedad.loc["MOUNJARO", c] for c in categorias]

    bars_oz = ax1.bar(x - ancho / 2, vals_oz, ancho, label="OZEMPIC",  color=COLOR_OZ, alpha=0.85)
    bars_mo = ax1.bar(x + ancho / 2, vals_mo, ancho, label="MOUNJARO", color=COLOR_MO, alpha=0.85)

    for bar in bars_oz:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom",
                 color=COLOR_OZ, fontsize=9, fontweight="bold")
    for bar in bars_mo:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom",
                 color=COLOR_MO, fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(etiquetas, fontsize=10)
    ax1.set_ylabel("% de reportes FAERS", fontsize=11)
    ax1.set_title("Gravedad de los reportes (% sobre reportes, no incidencia)", fontsize=13, pad=12)
    ax1.legend(facecolor="#1e1e2e", edgecolor="#444444", labelcolor="white", fontsize=10)

    ax2 = fig.add_subplot(gs[1, 0])
    _style_ax(ax2)
    top_oz = ozempic_s.nlargest(8, "report_count")
    ax2.barh(top_oz["reaction"], top_oz["report_count"], color=COLOR_OZ, alpha=0.85)
    ax2.set_title("Top reacciones OZEMPIC", fontsize=11, pad=8)
    ax2.set_xlabel("N reportes", fontsize=9)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.invert_yaxis()

    ax3 = fig.add_subplot(gs[1, 1])
    _style_ax(ax3)
    top_mo = mounjaro_s.nlargest(8, "report_count")
    ax3.barh(top_mo["reaction"], top_mo["report_count"], color=COLOR_MO, alpha=0.85)
    ax3.set_title("Top reacciones MOUNJARO", fontsize=11, pad=8)
    ax3.set_xlabel("N reportes", fontsize=9)
    ax3.tick_params(axis="y", labelsize=8)
    ax3.invert_yaxis()

    fig.suptitle(
        "Efectos Adversos: OZEMPIC (semaglutide) vs MOUNJARO (tirzepatide)\nFuente: FDA FAERS",
        fontsize=14, color="white", y=1.01,
    )
    fig.text(
        0.5, -0.01,
        "FAERS = notificacion espontanea: los % son sobre reportes recibidos, no sobre pacientes tratados. "
        "No controla exposicion, tiempo en mercado ni vigilancia; no permite inferir incidencia.",
        ha="center", color="#aaaaaa", fontsize=8.5, style="italic",
    )
    _finish(fig, save_path, show)   # GridSpec ya fija hspace/wspace; tight_layout no aplica


def plot_weight_loss(
    df_clinical: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Bar chart of % body-weight loss per trial arm (Q2).
    df_clinical: output of features.clinical_weight_loss_summary().
    These values come from the literature, not from the CSVs — the chart says so.
    """
    df = df_clinical.sort_values("perdida_peso_pct", ascending=False).reset_index(drop=True)
    colores = [COLOR_MO if m == "tirzepatide" else COLOR_OZ for m in df["medicamento"]]
    etiquetas = [f"{r.marca} {r.dosis}\n{r.ensayo}" for r in df.itertuples()]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax)
    ax.grid(True, axis="y", linestyle="--", alpha=0.2, color="#555555")
    ax.grid(False, axis="x")

    bars = ax.bar(etiquetas, -df["perdida_peso_pct"], color=colores, alpha=0.85, width=0.55)
    for bar, r in zip(bars, df.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{r.perdida_peso_pct}%\n(n={r.N}, {r.duracion_semanas} sem)",
            ha="center", va="bottom", color="white", fontsize=8.5, fontweight="bold",
        )

    ax.legend(
        handles=[mpatches.Patch(color=COLOR_OZ, label="Semaglutide"),
                 mpatches.Patch(color=COLOR_MO, label="Tirzepatide")],
        facecolor="#1e1e2e", edgecolor="#444444", labelcolor="white", fontsize=9,
    )
    ax.set_ylabel("% perdida de peso corporal", fontsize=11)
    ax.set_ylim(0, (-df["perdida_peso_pct"]).max() * 1.25)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_title("Perdida de peso por ensayo (dosis maxima aprobada)", fontsize=13, pad=10)
    fig.suptitle(
        "Semaglutide vs Tirzepatide  |  Resultados publicados (literatura, no dataset)",
        fontsize=14, color="white", y=1.0,
    )
    fig.text(
        0.5, -0.02,
        "Fuente: publicaciones primarias (NEJM). clinical_trials.csv solo aporta el registro del ensayo "
        "(NCT, fase, n, fechas), no los resultados.\nPoblaciones distintas: STEP 1 / SURMOUNT-1 sin DM2; "
        "SURPASS-2 con DM2 y dosis de diabetes.",
        ha="center", color="#aaaaaa", fontsize=8.5, style="italic",
    )
    plt.tight_layout()
    _finish(fig, save_path, show)
