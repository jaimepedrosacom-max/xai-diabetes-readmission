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

Idioma: figure_A() y figure_B() aceptan lang="es" (por defecto, los ficheros de siempre)
o lang="en", que escribe los mismos gráficos con sufijo _en y los rótulos en inglés para
la memoria. Lo único que cambia son cadenas y separadores de número; ver el bloque TXT.
Las versiones en inglés las genera src/figures_en.py.
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

# ---------------------------------------------------------------------------
# Idioma de los rótulos: SOLO texto y formato de número
# ---------------------------------------------------------------------------
# La memoria va en inglés, pero el notebook (notebooks/analisis.ipynb) muestra estas
# figuras en español y se decidió NO reejecutarlo. Así que el idioma es un parámetro y
# la versión inglesa se guarda con sufijo _en, dejando los ficheros en español intactos.
# Cambian únicamente cadenas y separadores de número: ningún dato, ningún CSV y ninguna
# cifra se recalculan. Todas las cifras siguen saliendo de los mismos CSV.
#
# Los "n" de los títulos de panel van escritos a mano en cada idioma (igual que estaban
# en español) porque son los tamaños fijos de los splits, no una cifra que el script lea.
TXT = {
    "es": {
        # --- Figura A ---
        "A_suptitle": "Brecha de precisión A − B en el top 10 %: ¿la ventaja sobrevive a la etiqueta limpia?",
        "A_subtitle": "A = top 10 % sin ingresos previos · B = top 10 % con ≥1 ingreso previo · "
                      ">0: A por delante de B · IC95 bootstrap por paciente, 2.000 réplicas",
        "A_ylabel": "brecha A − B (puntos de precisión, IC95)",
        "A_panel_val": "VAL · 19.868 filas · 3 pasos",
        "A_panel_test": "TEST · 19.867 filas · 4 pasos (replicación descriptiva)",
        "A_steps_val": ["(1) viejo/vieja", "(2) viejo/nueva", "(3) nuevo/nueva\n= (3a) en test"],
        "A_steps_test": ["(1) viejo/vieja", "(2) viejo/nueva",
                         "(3a) corregido CON\n= (3) en val", "(3b) entregable SIN\nsolo en test"],
        "A_n_label": "{lab}\nn_A {nA} · n_B {nB}",
        "A_contrast": "brecha(2) − brecha(1) = {obs} puntos [{lo}; {hi}]",
        # Geometría del panel: en inglés las etiquetas de paso son más largas y ocupan
        # una línea más, así que la figura es algo más alta y la letra algo menor.
        "A_figsize": (11, 5.2), "A_tick_fs": 8, "A_rect_bottom": 0.07,
        "A_note": "Nota: el paso (3b) no se midió en val (la ablación de payer_code se decidió en val, "
                  "pero su brecha solo se calculó en test). El (3) de val y el (3a) de test son el mismo\n"
                  "modelo (corregido CON payer_code) sobre datos distintos. Test es la segunda y última "
                  "consulta: replicación descriptiva, no se decide nada con ella.",
        # --- Figura B ---
        "B_suptitle": "Composición del top 10 % por payer_code: cociente frente a MC, con y sin la variable",
        "B_banner": "SOLO VALIDACIÓN · 19.868 filas de val · top 10 % del modelo · "
                    "barras = IC95 bootstrap por paciente",
        "B_xlabel": "cociente de tasa de marcado frente a MC (escala logarítmica)",
        "B_missing": "Missing (dato ausente)",
        "B_y_label": "{nombre}   n={n} · marcados {con} | {sin}",
        "B_legend_con": "con payer_code (corregido, 183 árboles)",
        "B_legend_sin": "sin payer_code (entregable, 123 árboles)",
        "B_legend_hollow": "hueco = sin veredicto (<20 marcados con payer_code)",
        "B_legend_band": "banda de los cuatro quintos [0,80; 1,25]",
        "B_note": "Denominador de cada cociente: tasa de marcado del grupo / tasa de MC del mismo modelo "
                  "(MC: {mc_con} de {n} con payer_code; {mc_sin} de {n} sin payer_code).\n"
                  "Missing = ausencia de dato administrativo, no un tipo de seguro. MC es la referencia "
                  "por ser la mayor categoría con seguro conocido.",
    },
    "en": {
        # --- Figure A ---
        "A_suptitle": "Precision gap A − B in the top 10%: does the advantage survive a clean label?",
        "A_subtitle": "A = top 10% with no previous inpatient admission · B = top 10% with at least one · "
                      ">0: A ahead of B · 95% CI, patient-level bootstrap, 2,000 replicates",
        "A_ylabel": "gap A − B (precision points, 95% CI)",
        "A_panel_val": "VALIDATION · 19,868 rows · 3 steps",
        "A_panel_test": "TEST · 19,867 rows · 4 steps (descriptive replication)",
        "A_steps_val": ["(1) old model /\nold label", "(2) old model /\nnew label",
                        "(3) new model /\nnew label\n= (3a) in test"],
        "A_steps_test": ["(1) old model /\nold label", "(2) old model /\nnew label",
                         "(3a) corrected WITH\npayer_code\n= step (3) in validation",
                         "(3b) delivered model,\nWITHOUT payer_code\ntest only"],
        "A_n_label": "{lab}\nn_A {nA} · n_B {nB}",
        "A_contrast": "gap(2) − gap(1) = {obs} points [{lo}, {hi}]",
        "A_figsize": (11.6, 6.0), "A_tick_fs": 6.8, "A_rect_bottom": 0.145,
        # "second query" se ata a ESTAS cifras, no al total de consultas: la Fase 5A fue la
        # TERCERA, y la memoria dice tres. El pie dice de dónde salen estas cifras concretas,
        # no cuántas veces se tocó test, para que no parezca una contradicción con la memoria.
        "A_note": "Note: step (3b) was not measured in validation (the payer_code ablation was decided in "
                  "validation, but its gap was only computed on test).\n"
                  "Step (3) in validation and step (3a) in test are the same model (corrected WITH "
                  "payer_code) on different data.\n"
                  "These test figures come from the second query of the held-out set (descriptive "
                  "replication; nothing is decided with them).",
        # --- Figure B ---
        "B_suptitle": "Composition of the top 10% by payer_code: ratio to Medicare, with and without the variable",
        "B_banner": "VALIDATION ONLY · 19,868 validation rows · model's top 10% · "
                    "bars = 95% CI, patient-level bootstrap",
        "B_xlabel": "flagging-rate ratio to Medicare (log scale)",
        "B_missing": "Missing (no data recorded)",
        "B_y_label": "{nombre}   n={n} · flagged {con} | {sin}",
        "B_legend_con": "with payer_code (corrected, 183 trees)",
        "B_legend_sin": "without payer_code (delivered, 123 trees)",
        "B_legend_hollow": "hollow = no verdict (<20 flagged with payer_code)",
        "B_legend_band": "four-fifths band [0.80, 1.25]",
        "B_note": "Denominator of each ratio: the group's flagging rate / Medicare's flagging rate for the "
                  "same model\n(Medicare: {mc_con} of {n} with payer_code; {mc_sin} of {n} without "
                  "payer_code).\nMissing = an absent administrative field, not a type of insurance. "
                  "Medicare is the reference because it is the largest category with a known payer.",
    },
}

# Sufijo de fichero por idioma: el español conserva el nombre original.
SUFFIX = {"es": "", "en": "_en"}


def fmt_num(x: float, signed: bool = False, lang: str = "es") -> str:
    """Número con 2 decimales. es: coma decimal. en: punto decimal (como la memoria)."""
    s = f"{x:+.2f}" if signed else f"{x:.2f}"
    if lang == "es":
        s = s.replace(".", ",")
    return s.replace("-", "−")         # signo menos tipográfico en los dos idiomas


def fmt_int(n: int, lang: str = "es") -> str:
    """Entero con separador de miles. es: punto. en: coma."""
    s = f"{int(n):,}"
    return s if lang == "en" else s.replace(",", ".")


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


def plot_gap_panel(ax, gaps: pd.DataFrame, labels: list, title: str, contrast: str,
                   lang: str = "es"):
    x = np.arange(len(gaps))
    b = gaps["brecha_puntos"].to_numpy()
    lo = gaps["brecha_puntos_IC95_bajo"].to_numpy()
    hi = gaps["brecha_puntos_IC95_alto"].to_numpy()
    ax.axhline(0, color=INK, lw=1)
    ax.errorbar(x, b, yerr=[b - lo, hi - b], fmt="o", color=BLUE, ms=7,
                ecolor=BLUE, elinewidth=1.6, capsize=5)
    for xi, bi in zip(x, b):
        ax.text(xi + 0.08, bi, fmt_num(bi, signed=True, lang=lang), va="center",
                fontsize=9, color=INK)
    # n de cada subgrupo va dentro de la etiqueta del eje X, bajo el nombre del paso
    etiquetas = [
        TXT[lang]["A_n_label"].format(lab=lab, nA=fmt_int(nA, lang), nB=fmt_int(nB, lang))
        for lab, nA, nB in zip(labels, gaps["n_A"], gaps["n_B"])
    ]
    ax.set_xticks(x, etiquetas, fontsize=TXT[lang]["A_tick_fs"])
    ax.set_xlim(-0.6, len(gaps) - 0.4)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    ax.text(0.02, 0.97, contrast, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.2, color=SECONDARY,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.6))
    _style(ax)


def figure_A(lang: str = "es") -> str:
    t = TXT[lang]
    val = load_gap_val()
    test = load_gap_test()
    pc_val = pd.read_csv(REPORTS / "phase4_paired_contrasts.csv").iloc[0]
    pc_test = pd.read_csv(REPORTS / "phase4_test_paired_contrasts.csv")
    pc_test = pc_test[pc_test["contraste"].str.startswith("brecha(2)")].iloc[0]

    # Mapeo de etiquetas: el (3) de val es el (3a) de test (mismo modelo).
    labels_val = t["A_steps_val"]
    labels_test = t["A_steps_test"]
    # Los asserts comprueban los VALORES DEL CSV, que están en español y no se traducen.
    assert list(val["paso"]) == ["(1) viejo/vieja", "(2) viejo/nueva", "(3) nuevo/nueva"]
    assert list(test["paso"]) == ["(1) viejo/vieja", "(2) viejo/nueva",
                                  "(3a) corregido CON/nueva", "(3b) entregable SIN/nueva"]

    fig, (axv, axt) = plt.subplots(1, 2, figsize=t["A_figsize"], facecolor=SURFACE,
                                   sharey=True, gridspec_kw={"width_ratios": [3, 4]})
    for ax, gaps, labels, panel, pc in (
        (axv, val, labels_val, t["A_panel_val"], pc_val),
        (axt, test, labels_test, t["A_panel_test"], pc_test),
    ):
        plot_gap_panel(
            ax, gaps, labels, panel,
            t["A_contrast"].format(obs=fmt_num(pc["observado"], True, lang),
                                   lo=fmt_num(pc["IC95_bajo"], True, lang),
                                   hi=fmt_num(pc["IC95_alto"], True, lang)),
            lang=lang,
        )
    axv.set_ylabel(t["A_ylabel"], color=SECONDARY, fontsize=9)
    # Margen superior para que el cuadro del contraste no tape ninguna barra de error.
    axv.set_ylim(-14, 15)

    fig.suptitle(t["A_suptitle"], x=0.01, ha="left", color=INK, fontsize=12, y=0.99)
    fig.text(0.01, 0.905, t["A_subtitle"], ha="left", color=MUTED, fontsize=8.5)
    fig.text(0.01, 0.015, t["A_note"], ha="left", va="bottom", color=MUTED, fontsize=7.6)
    fig.tight_layout(rect=(0, t["A_rect_bottom"], 1, 0.9))
    out = FIG_DIR / f"phase4_fig_A_brecha_val_test{SUFFIX[lang]}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)  # PDF vectorial, mismo nombre
    plt.close(fig)
    return out.name


# ---------------------------------------------------------------------------
# Figura B
# ---------------------------------------------------------------------------
def figure_B(lang: str = "es") -> str:
    t = TXT[lang]
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
        nombre = t["B_missing"] if r["grupo"] == "Missing" else r["grupo"]
        etiquetas.append(t["B_y_label"].format(
            nombre=nombre, n=fmt_int(r["n_grupo"], lang),
            con=fmt_int(r["marcados_con"], lang), sin=fmt_int(r["marcados_sin"], lang)))
    ax.set_yticks(y, etiquetas, fontsize=8.5)
    ax.set_xscale("log")
    xt = [0.25, 0.5, 0.8, 1.0, 1.25, 2.0]
    ax.set_xticks(xt, [fmt_num(v, lang=lang) for v in xt])
    ax.minorticks_off()
    ax.set_xlim(0.2, 2.8)
    ax.set_ylim(-0.7, len(grupos) - 0.3)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(t["B_xlabel"], color=SECONDARY, fontsize=9)
    _style(ax)

    # Leyenda propia (la banda, la referencia y las dos series)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], marker="o", ls="", color=BLUE, ms=7, label=t["B_legend_con"]),
        Line2D([], [], marker="s", ls="", color=ORANGE, ms=6.5, label=t["B_legend_sin"]),
        Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=BLUE, mew=1.6, ms=7,
               label=t["B_legend_hollow"]),
        Patch(facecolor=BAND, label=t["B_legend_band"]),
    ]
    # Leyenda debajo del eje, fuera de los datos (en el área de datos tapaba UN y SP).
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=2,
              fontsize=8, frameon=False)

    fig.suptitle(t["B_suptitle"], x=0.01, ha="left", color=INK, fontsize=12, y=0.985)
    fig.text(0.01, 0.925, t["B_banner"], ha="left", color=SECONDARY, fontsize=9, weight="bold")
    fig.text(0.01, 0.015,
             t["B_note"].format(mc_con=fmt_int(mc["marcados_con"], lang),
                                mc_sin=fmt_int(mc["marcados_sin"], lang),
                                n=fmt_int(mc["n_grupo"], lang)),
             ha="left", va="bottom", color=MUTED, fontsize=7.6)
    fig.tight_layout(rect=(0, 0.1, 1, 0.915))
    out = FIG_DIR / f"phase4_fig_B_payer_code_val{SUFFIX[lang]}.png"
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
