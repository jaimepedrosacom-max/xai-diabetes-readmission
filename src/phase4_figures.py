"""
Fase 4 — Figuras de los dos hallazgos principales (solo lectura de CSV).

Este script NO entrena ni puntúa nada: lee los CSV que ya escribieron
phase4_correction.py, phase4_test_final.py y phase4_ablation.py, y dibuja.
Así las figuras no pueden depender de un recálculo distinto del de las cifras.

Figura A — brecha de precisión A − B con IC95, en val y en test.
    Val: pasos (1), (2) y (3). Test: pasos (1), (2), (3a) y (3b).
    El (3b) es el entregable SIN payer_code; no se midió en val. El (3) de val
    es el mismo modelo que el (3a) de test (corregido CON payer_code), con otros datos.
Figura B — cociente de tasa de marcado por payer_code frente a MC, con y sin la
    variable. SOLO VALIDACIÓN (top 10 %, 19.868 filas). Escala logarítmica para que
    la banda de los cuatro quintos [0,80; 1,25] sea simétrica (0,80 y 1,25 son
    recíprocos). Los grupos con menos de 20 marcados se dibujan en hueco y sin
    veredicto, según el prerregistro.

Salida: reports/figures/phase4_fig_A_brecha_val_test.png
        reports/figures/phase4_fig_B_payer_code_val.png
"""
import matplotlib

matplotlib.use("Agg")  # sin ventana: solo guardamos a fichero (como evaluation.py)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as D
import evaluation as E

REPORTS = D.PROJECT_ROOT / "reports"
FIG_DIR = E.FIG_DIR

SURFACE = "#fcfcfb"                  # mismo fondo que las figuras de la Fase 3a
INK, MUTED, GRID = E.INK, E.MUTED, E.GRID
# La paleta sale de evaluation.py. El nombre de la clave no describe el uso aquí:
# azul = modelo CON payer_code (o la brecha), naranja = modelo SIN payer_code.
BLUE = E.MODEL_COLORS["LogReg"]
ORANGE = E.MODEL_COLORS["LightGBM"]
SECONDARY = "#52514e"                # mismo gris de texto secundario que xai_shap.py
BAND = "#e6efe3"                     # banda de los cuatro quintos

MIN_MARCADOS = 20                    # mínimo del prerregistro para tener veredicto


def fmt_num(x: float, signed: bool = False) -> str:
    """Número con coma decimal, como en la memoria."""
    s = f"{x:+.2f}" if signed else f"{x:.2f}"
    return s.replace(".", ",").replace("-", "−")


def fmt_int(n: int) -> str:
    """Entero con punto de miles, como en la memoria."""
    return f"{int(n):,}".replace(",", ".")


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=SECONDARY, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


# ---------------------------------------------------------------------------
# Figura A
# ---------------------------------------------------------------------------
def load_gap_val() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "phase4_prediction_gaps.csv")


def load_gap_test() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "phase4_test_gaps.csv")


def plot_gap_panel(ax, gaps: pd.DataFrame, labels: list, title: str, contrast: str):
    x = np.arange(len(gaps))
    b = gaps["brecha_puntos"].to_numpy()
    lo = gaps["brecha_puntos_IC95_bajo"].to_numpy()
    hi = gaps["brecha_puntos_IC95_alto"].to_numpy()
    ax.axhline(0, color=INK, lw=1)
    ax.errorbar(x, b, yerr=[b - lo, hi - b], fmt="o", color=BLUE, ms=7,
                ecolor=BLUE, elinewidth=1.6, capsize=5)
    for xi, bi in zip(x, b):
        ax.text(xi + 0.08, bi, fmt_num(bi, signed=True), va="center", fontsize=9, color=INK)
    # n de cada subgrupo va dentro de la etiqueta del eje X, bajo el nombre del paso
    etiquetas = [
        f"{lab}\nn_A {fmt_int(nA)} · n_B {fmt_int(nB)}"
        for lab, nA, nB in zip(labels, gaps["n_A"], gaps["n_B"])
    ]
    ax.set_xticks(x, etiquetas, fontsize=8)
    ax.set_xlim(-0.6, len(gaps) - 0.4)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    ax.text(0.02, 0.97, contrast, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.2, color=SECONDARY,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.6))
    _style(ax)


def figure_A() -> str:
    val = load_gap_val()
    test = load_gap_test()
    pc_val = pd.read_csv(REPORTS / "phase4_paired_contrasts.csv").iloc[0]
    pc_test = pd.read_csv(REPORTS / "phase4_test_paired_contrasts.csv")
    pc_test = pc_test[pc_test["contraste"].str.startswith("brecha(2)")].iloc[0]

    # Mapeo de etiquetas: el (3) de val es el (3a) de test (mismo modelo).
    labels_val = ["(1) viejo/vieja", "(2) viejo/nueva", "(3) nuevo/nueva\n= (3a) en test"]
    labels_test = ["(1) viejo/vieja", "(2) viejo/nueva",
                   "(3a) corregido CON\n= (3) en val", "(3b) entregable SIN\nsolo en test"]
    assert list(val["paso"]) == ["(1) viejo/vieja", "(2) viejo/nueva", "(3) nuevo/nueva"]
    assert list(test["paso"]) == ["(1) viejo/vieja", "(2) viejo/nueva",
                                  "(3a) corregido CON/nueva", "(3b) entregable SIN/nueva"]

    fig, (axv, axt) = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=SURFACE,
                                   sharey=True, gridspec_kw={"width_ratios": [3, 4]})
    plot_gap_panel(
        axv, val, labels_val,
        "VAL · 19.868 filas · 3 pasos",
        f"brecha(2) − brecha(1) = {fmt_num(pc_val['observado'], True)} puntos "
        f"[{fmt_num(pc_val['IC95_bajo'], True)}; {fmt_num(pc_val['IC95_alto'], True)}]",
    )
    plot_gap_panel(
        axt, test, labels_test,
        "TEST · 19.867 filas · 4 pasos (replicación descriptiva)",
        f"brecha(2) − brecha(1) = {fmt_num(pc_test['observado'], True)} puntos "
        f"[{fmt_num(pc_test['IC95_bajo'], True)}; {fmt_num(pc_test['IC95_alto'], True)}]",
    )
    axv.set_ylabel("brecha A − B (puntos de precisión, IC95)", color=SECONDARY, fontsize=9)
    # Margen superior para que el cuadro del contraste no tape ninguna barra de error.
    axv.set_ylim(-14, 15)

    fig.suptitle("Brecha de precisión A − B en el top 10 %: ¿la ventaja sobrevive a la etiqueta limpia?",
                 x=0.01, ha="left", color=INK, fontsize=12, y=0.99)
    fig.text(0.01, 0.905,
             "A = top 10 % sin ingresos previos · B = top 10 % con ≥1 ingreso previo · "
             ">0: A por delante de B · IC95 bootstrap por paciente, 2.000 réplicas",
             ha="left", color=MUTED, fontsize=8.5)
    fig.text(0.01, 0.015,
             "Nota: el paso (3b) no se midió en val (la ablación de payer_code se decidió en val, "
             "pero su brecha solo se calculó en test). El (3) de val y el (3a) de test son el mismo\n"
             "modelo (corregido CON payer_code) sobre datos distintos. Test es la segunda y última "
             "consulta: replicación descriptiva, no se decide nada con ella.",
             ha="left", va="bottom", color=MUTED, fontsize=7.6)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    out = FIG_DIR / "phase4_fig_A_brecha_val_test.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)  # PDF vectorial, mismo nombre
    plt.close(fig)
    return out.name


# ---------------------------------------------------------------------------
# Figura B
# ---------------------------------------------------------------------------
def figure_B() -> str:
    comp = pd.read_csv(REPORTS / "phase4_ablation_composition.csv")
    mc = comp[comp["es_referencia"]].iloc[0]
    grupos = comp[~comp["es_referencia"]].copy()

    # Comprobación del denominador: cada cociente = tasa del grupo / tasa de MC del MISMO modelo.
    for suf in ("con", "sin"):
        tasa_mc = mc[f"marcados_{suf}"] / mc["n_grupo"]
        recalc = (grupos[f"marcados_{suf}"] / grupos["n_grupo"]) / tasa_mc
        assert np.allclose(recalc, grupos[f"cociente_{suf}"], atol=1e-9), f"cociente_{suf} no reconcilia"

    grupos = grupos.sort_values("cociente_con", ascending=True).reset_index(drop=True)
    y = np.arange(len(grupos))
    off = 0.16

    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
    ax.axvspan(0.80, 1.25, color=BAND, zorder=0)
    ax.axvline(1.0, color=INK, lw=1, ls="--", zorder=1)

    # CON payer_code (azul, círculo). Hueco si no llega a 20 marcados.
    yc = y + off
    ax.hlines(yc, grupos["IC_con_bajo"], grupos["IC_con_alto"], color=BLUE, lw=1.6, zorder=2)
    for yi, c, ok in zip(yc, grupos["cociente_con"], grupos["elegible_con"]):
        if ok:
            ax.plot(c, yi, "o", color=BLUE, ms=7, zorder=3)
        else:
            ax.plot(c, yi, "o", mfc=SURFACE, mec=BLUE, mew=1.6, ms=7, zorder=3)

    # SIN payer_code (naranja, cuadrado).
    ys = y - off
    ax.hlines(ys, grupos["IC_sin_bajo"], grupos["IC_sin_alto"], color=ORANGE, lw=1.6, zorder=2)
    for yi, c, ok in zip(ys, grupos["cociente_sin"], grupos["elegible_sin"]):
        if ok:
            ax.plot(c, yi, "s", color=ORANGE, ms=6.5, zorder=3)
        else:
            ax.plot(c, yi, "s", mfc=SURFACE, mec=ORANGE, mew=1.6, ms=6.5, zorder=3)

    # Etiquetas con n del grupo y marcados con | sin
    etiquetas = []
    for _, r in grupos.iterrows():
        nombre = "Missing (dato ausente)" if r["grupo"] == "Missing" else r["grupo"]
        etiquetas.append(
            f"{nombre}   n={fmt_int(r['n_grupo'])} · marcados {fmt_int(r['marcados_con'])} | {fmt_int(r['marcados_sin'])}"
        )
    ax.set_yticks(y, etiquetas, fontsize=8.5)
    ax.set_xscale("log")
    xt = [0.25, 0.5, 0.8, 1.0, 1.25, 2.0]
    ax.set_xticks(xt, [fmt_num(v) for v in xt])
    ax.minorticks_off()
    ax.set_xlim(0.2, 2.8)
    ax.set_ylim(-0.7, len(grupos) - 0.3)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("cociente de tasa de marcado frente a MC (escala logarítmica)", color=SECONDARY, fontsize=9)
    _style(ax)

    # Leyenda propia (la banda, la referencia y las dos series)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], marker="o", ls="", color=BLUE, ms=7,
               label="con payer_code (corregido, 183 árboles)"),
        Line2D([], [], marker="s", ls="", color=ORANGE, ms=6.5,
               label="sin payer_code (entregable, 123 árboles)"),
        Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=BLUE, mew=1.6, ms=7,
               label="hueco = sin veredicto (<20 marcados con payer_code)"),
        Patch(facecolor=BAND, label="banda de los cuatro quintos [0,80; 1,25]"),
    ]
    # Leyenda debajo del eje, fuera de los datos (en el área de datos tapaba UN y SP).
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=2,
              fontsize=8, frameon=False)

    fig.suptitle("Composición del top 10 % por payer_code: cociente frente a MC, con y sin la variable",
                 x=0.01, ha="left", color=INK, fontsize=12, y=0.985)
    fig.text(0.01, 0.925,
             "SOLO VALIDACIÓN · 19.868 filas de val · top 10 % del modelo · barras = IC95 bootstrap por paciente",
             ha="left", color=SECONDARY, fontsize=9, weight="bold")
    fig.text(0.01, 0.015,
             "Denominador de cada cociente: tasa de marcado del grupo / tasa de MC del mismo modelo "
             f"(MC: {fmt_int(mc['marcados_con'])} de {fmt_int(mc['n_grupo'])} con payer_code; "
             f"{fmt_int(mc['marcados_sin'])} de {fmt_int(mc['n_grupo'])} sin payer_code).\n"
             "Missing = ausencia de dato administrativo, no un tipo de seguro. MC es la referencia "
             "por ser la mayor categoría con seguro conocido.",
             ha="left", va="bottom", color=MUTED, fontsize=7.6)
    fig.tight_layout(rect=(0, 0.1, 1, 0.915))
    out = FIG_DIR / "phase4_fig_B_payer_code_val.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)  # PDF vectorial, mismo nombre
    plt.close(fig)
    return out.name


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Escrito:", FIG_DIR / figure_A())
    print("Escrito:", FIG_DIR / figure_B())


if __name__ == "__main__":
    main()
