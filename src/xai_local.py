"""
Fase 3d — Explicaciones locales SHAP de los 11 casos fijados en la Fase 3a.

Técnica: TreeSHAP (Lundberg et al., 2020), el mismo explicador exacto de la Fase 3a,
pero leído paciente a paciente en vez de agregado. SHAP se basa en los valores de
Shapley (Lundberg & Lee, 2017, "A Unified Approach to Interpreting Model
Predictions"). Solo val; test no se usa.

Los casos NO se eligen aquí: vienen de `src/xai_cases.py` con una regla fija y con
semilla, que no mira las explicaciones finales (ver su docstring). Esto evita escoger
pacientes que cuenten una historia bonita. Los tres `perfil_alto_*` son centroides de
k-means sobre los vectores SHAP del top 10%, así que son representativos de su grupo
POR CONSTRUCCIÓN, no por suerte; los demás salen de reglas sobre probabilidad,
`number_inpatient` y error del modelo.

Las figuras de waterfall se dibujan a mano con matplotlib, no con `shap.plots.waterfall`,
por dos razones: hacen falta etiquetas legibles de los niveles administrativos
(`XS.LEVEL_LABELS`, de IDS_mapping.csv) y hace falta marcar en la propia figura las
variables que no se interpretan. El cálculo de los valores es el de la librería.

-----------------------------------------------------------------------------
CRITERIOS FIJADOS ANTES DE VER NINGÚN RESULTADO (no se tocan después)
-----------------------------------------------------------------------------
1. Escala. SHAP reparte LOG-ODDS, que es la escala en la que es exactamente aditivo
   (`base + suma de contribuciones = raw_score`). NO se traduce cada contribución a
   "puntos de probabilidad": la logística no es aditiva y esa frase convertiría la
   explicación en un engaño. La probabilidad del paciente se da aparte, en el título
   de cada figura y en las tablas.

2. `interpretable`. Se lee de `reports/xai_sanity_noise_probes.csv` con la regla de la
   Fase 3b: la variable supera a la mejor sonda de ruido en las 10 semillas
   (`gana_a_sonda == N_PROBE_SEEDS`). No hay ninguna lista escrita a mano.
   AVISO IMPORTANTE: las sondas de ruido son una prueba GLOBAL (comparan la media de
   |SHAP| sobre todo val). Trasladar su resultado al nivel local es una extensión
   PRUDENTE —si el modelo no usa una variable de forma distinguible del ruido en
   conjunto, no nos fiamos de su atribución en un paciente suelto— y NO una validación
   de la explicación local. En este proyecto no hay sanity check local por variable.

3. Discrepancia entre atribución y sensibilidad (SHAP frente al diagnóstico (b) de la
   Fase 3c). Definida antes de medir, sin anticipar ningún ejemplo:
     - "atribuye sin mover": variable en el top-5 local de |SHAP| cuyo mayor |delta|
       en el diagnóstico (b) es exactamente 0.
     - "mueve sin atribuir": variable con |delta| > 0 en (b) que NO está en el top-5
       local de |SHAP|.
   Solo se puede evaluar en los 6 casos elegibles de la Fase 3c, que son los que
   tienen tabla (b). Miden cosas distintas (cuánto contribuye al valor actual frente a
   cuánto se movería si se cambiara), así que NO se espera que coincidan: la tabla
   documenta la discrepancia, no la denuncia.
   Columna `delta_incluye_change`: el delta de un fármaco puede venir en parte del
   recálculo de la variable derivada `change` (ver `Scorer.derive_change` de la Fase
   3c). Se marca cuando el candidato cambia `change` respecto al original o cuando
   `change` queda ambiguo y se puntúa el peor caso.

4. `destino_contaminado`. True si el `discharge_disposition_id` del caso es 22 o 5. En
   esos casos la contribución del destino se REPORTA pero no se lee como riesgo
   clínico: la auditoría de la Fase 3b indica que parte de su efecto viene de cómo se
   define la etiqueta (quien sale a rehabilitación suele tener un siguiente encuentro
   planificado).

5. Lectura por caso. Plantilla FIJA generada a partir de los datos (top-3
   interpretables con contribución y rango global, probabilidad, grupo de selección
   con su n y su tasa real, estabilidad del top-3 entre folds y las marcas de los
   puntos 2 y 4). Nada de texto libre por caso: una narrativa escrita a mano sobre un
   paciente concreto es exactamente el tipo de sobreinterpretación que este proyecto
   quiere evitar.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as D
import evaluation as E
import preprocessing as P
import xai_counterfactuals as CF
import xai_shap as XS
from xai_sanity import N_PROBE_SEEDS

TOP_N_FIG = 10          # variables en cada waterfall; el resto se agrega en una barra
TOP_N_TABLE = 5         # top-k local que usa la definición de discrepancia (criterio 3)
TOP_N_READING = 3       # variables interpretables en la lectura por caso (criterio 5)
TOP_N_PAIR_FIG = 6      # variables en cada panel de la figura de pares contrastivos

REPORT_DIR = D.PROJECT_ROOT / "reports"
SUSPECT_CODES = CF.SUSPECT_CODES     # {"22", "5"}, la misma definición de la Fase 3c

UP, DOWN = E.MODEL_COLORS["LightGBM"], E.MODEL_COLORS["LogReg"]   # sube / baja el riesgo
MARK_NO_INTERP = "°"    # variable que no supera las sondas de ruido
MARK_CONTAM = "‡"       # destino señalado por la auditoría de fuga

# Regla por la que entró cada caso (de xai_cases.select_cases). Es metadato fijo del
# diseño, no interpretación: sirve para no leer un paciente como si fuera la población.
GROUP_DESC = {
    "perfil_alto_1": "centroide k-means 1 sobre SHAP del top 10%",
    "perfil_alto_2": "centroide k-means 2 sobre SHAP del top 10%",
    "perfil_alto_3": "centroide k-means 3 sobre SHAP del top 10%",
    "riesgo_mediano": "paciente en la mediana de riesgo de val",
    "riesgo_bajo": "mediana del 20% de menor riesgo",
    "modelo_alto_sin_ingresos_reingresa": "top 10% con number_inpatient=0, reingresó",
    "modelo_alto_sin_ingresos_no_reingresa": "top 10% con number_inpatient=0, no reingresó",
    "modelo_bajo_con_ingresos_reingresa": "bajo la mediana con number_inpatient>=1, reingresó",
    "modelo_bajo_con_ingresos_no_reingresa": "bajo la mediana con number_inpatient>=1, no reingresó",
    "falsa_alarma_mayor_riesgo": "la falsa alarma de mayor riesgo de val",
    "reingreso_menor_riesgo": "el reingreso con menor riesgo de val",
}

# Advertencia que acompaña siempre a la figura de pares: la fila de arriba sale de un
# grupo donde los códigos de destino señalados por la auditoría son mayoritarios, así
# que no demuestra que el modelo capte riesgo clínico que la regla simple no ve.
# La cifra está medida en Fase 3d (val) y recogida en CLAUDE.md.
PAIR_CAVEAT = ("Aviso: en el grupo de la fila superior (top 10% con number_inpatient=0, "
               "n=440) el 69,3% tiene destino 22/5, frente al 10,6% del resto del top 10%\n"
               "y el 3,2% de todo val. La contribución #1 de esos dos casos es esa "
               "variable contaminada, así que la fila no demuestra ventaja clínica "
               "sobre la regla simple.")

# Los dos pares que contrastan al modelo con la regla simple number_inpatient>0:
# filas = perfil del paciente, columnas = lo que pasó de verdad.
PAIR_ROWS = [("top 10% del modelo, sin ingresos previos",
              "modelo_alto_sin_ingresos_reingresa",
              "modelo_alto_sin_ingresos_no_reingresa"),
             ("bajo la mediana del modelo, con ingresos previos",
              "modelo_bajo_con_ingresos_reingresa",
              "modelo_bajo_con_ingresos_no_reingresa")]


# ---------------------------------------------------------------------------
# Lecturas de fases anteriores (nada se recalcula aquí)
# ---------------------------------------------------------------------------
def load_interpretable() -> pd.Series:
    """Variables cuya atribución se interpreta, según las sondas de ruido (criterio 2).

    La regla es la de la Fase 3b y se aplica tal cual: ganar a la mejor sonda en las
    N_PROBE_SEEDS semillas. Se lee del CSV para que no haya forma de que la lista se
    desincronice de lo que midió el sanity check.
    """
    probes = pd.read_csv(REPORT_DIR / "xai_sanity_noise_probes.csv", index_col=0)
    assert probes["gana_a_sonda"].max() <= N_PROBE_SEEDS, "más victorias que semillas"
    return probes["gana_a_sonda"] == N_PROBE_SEEDS


def load_local_stability() -> pd.DataFrame:
    """Solape del top-3 local entre folds y tamaño/tasa del grupo de cada caso.

    Lo calcula `fold_stability` en la Fase 3b; aquí solo se lee. Si falta el CSV, hay
    que volver a ejecutar `src/xai_sanity.py` (es el único sitio donde se reentrenan
    los 5 folds).
    """
    path = REPORT_DIR / "xai_sanity_local_stability.csv"
    assert path.exists(), f"falta {path.name}: ejecuta antes src/xai_sanity.py"
    return pd.read_csv(path)


def load_sensitivity() -> pd.DataFrame:
    """Tabla (b) de la Fase 3c: delta de puntuación al mover un lever a la vez."""
    sens = pd.read_csv(REPORT_DIR / "xai_counterfactuals_diag_sensibilidad.csv")
    return sens[sens["variable"] != "(ninguna)"].copy()


# ---------------------------------------------------------------------------
# Tablas de contribuciones
# ---------------------------------------------------------------------------
def value_label(col: str, value) -> str:
    """Valor legible: los IDs administrativos llevan su descripción de IDS_mapping."""
    text = str(value)
    return XS.LEVEL_LABELS.get(col, {}).get(text, text)


def contributions(values: np.ndarray, L: pd.DataFrame, cases: pd.DataFrame,
                  interp: pd.Series, rank_global: pd.Series, base: float,
                  booster) -> pd.DataFrame:
    """Una fila por (caso, variable): contribución, rango local y global, y marcas.

    Comprueba la aditividad caso a caso: base + suma de contribuciones tiene que dar
    el log-odds del modelo. Es la garantía de que lo que se dibuja es la predicción
    entera y no un resumen parcial.
    """
    pos = {enc: i for i, enc in enumerate(L.index)}
    rows = []
    for case in cases.itertuples():
        encounter = int(case.encounter_id)
        row = L.loc[[encounter]]
        phi = values[pos[encounter]]
        raw = float(booster.predict(row, raw_score=True)[0])
        err = abs(base + phi.sum() - raw)
        assert err < 1e-6, f"aditividad rota en {case.caso}: error {err:.2e}"
        dest = str(row["discharge_disposition_id"].iloc[0])
        order = np.argsort(-np.abs(phi))
        rank_local = pd.Series(np.empty(len(phi), dtype=int), index=L.columns)
        rank_local.iloc[order] = np.arange(1, len(phi) + 1)
        for col, value in zip(L.columns, phi):
            rows.append({
                "caso": case.caso, "encounter_id": encounter, "prob": case.prob,
                "reingresa": int(case.reingresa), "log_odds": raw, "base": base,
                "variable": col, "valor": value_label(col, row[col].iloc[0]),
                "shap_logodds": float(value), "abs_shap": abs(float(value)),
                "rank_local": int(rank_local[col]),
                "rank_global": int(rank_global[col]),
                "interpretable": bool(interp.get(col, False)),
                "destino_contaminado": dest in SUSPECT_CODES,
            })
    return pd.DataFrame(rows)


def top_features(contrib: pd.DataFrame, stability: pd.DataFrame,
                 k: int = TOP_N_TABLE) -> pd.DataFrame:
    """Top-k local de cada caso, con rango global, marcas y estabilidad entre folds."""
    top = (contrib.sort_values(["caso", "rank_local"])
           .groupby("caso", sort=False).head(k).copy())
    top["grupo_seleccion"] = top["caso"].map(GROUP_DESC)
    return top.merge(stability, on="caso", how="left")


# ---------------------------------------------------------------------------
# Paso 5: atribución (SHAP) frente a sensibilidad (diagnóstico (b) de la Fase 3c)
# ---------------------------------------------------------------------------
def change_recalculated(scorer: CF.Scorer, var: str, new_value: str) -> bool:
    """¿El delta de este cambio incluye el recálculo de la derivada `change`?

    Importa porque un "efecto del fármaco" puede ser en realidad el efecto de que el
    modelo vea `change` distinto. Se marca también el caso ambiguo (el candidato
    elimina todos los Up/Down del original), donde la Fase 3c puntúa el peor caso.
    """
    cand = scorer.frame({var: [new_value]}, 1)
    change, ambiguous = scorer.derive_change(cand)
    return bool(change[0] != scorer.orig_change or ambiguous[0])


def attribution_vs_sensitivity(booster, L: pd.DataFrame, contrib: pd.DataFrame,
                               sens: pd.DataFrame, k: int = TOP_N_TABLE) -> pd.DataFrame:
    """Cruza el top-k local de |SHAP| con el mayor |delta| de (b). Criterio 3.

    Devuelve una fila por (caso, variable) de la unión de ambos conjuntos, con la
    clasificación ya fijada: 'atribuye sin mover', 'mueve sin atribuir', 'coincide'
    (en el top-k y además mueve) o 'ni atribuye ni mueve'.
    """
    delta = (sens.assign(abs_delta=sens["delta"].abs())
             .groupby(["caso", "variable"])
             .agg(max_abs_delta=("abs_delta", "max"),
                  delta_min=("delta", "min"), delta_max=("delta", "max")).reset_index())

    # El flag de `change` se calcula con el Scorer de la Fase 3c, sobre el valor que
    # produce el mayor |delta| de cada (caso, variable).
    flags = []
    for (caso, var), grp in sens.groupby(["caso", "variable"]):
        best = grp.loc[grp["delta"].abs().idxmax()]
        encounter = int(best["encounter_id"])
        scorer = CF.Scorer(booster, L.loc[[encounter]])
        flags.append({"caso": caso, "variable": var,
                      "valor_max_delta": best["valor_nuevo"],
                      "delta_incluye_change": change_recalculated(
                          scorer, var, str(best["valor_nuevo"]))})
    delta = delta.merge(pd.DataFrame(flags), on=["caso", "variable"], how="left")

    evaluable = sens["caso"].unique()
    local = contrib[contrib["caso"].isin(evaluable)].copy()
    local["en_top_k"] = local["rank_local"] <= k
    merged = local.merge(delta, on=["caso", "variable"], how="left")
    # Solo interesa la unión: top-k local, o variable con sensibilidad medida.
    merged = merged[merged["en_top_k"] | merged["max_abs_delta"].notna()]

    mueve = merged["max_abs_delta"] > 0
    tiene_medida = merged["max_abs_delta"].notna()
    # Las cinco categorías son excluyentes. Distinguir "medida y vale 0" de "sin medir"
    # es imprescindible: una variable que no es lever NO tiene delta (NaN) y no dice
    # nada sobre la discrepancia; una con delta medido exactamente 0 sí.
    merged["clasificacion"] = np.select(
        [merged["en_top_k"] & tiene_medida & ~mueve,
         ~merged["en_top_k"] & mueve,
         merged["en_top_k"] & mueve,
         ~merged["en_top_k"] & tiene_medida & ~mueve],
        ["atribuye sin mover", "mueve sin atribuir", "coincide",
         "ni atribuye ni mueve"],
        default="sin sensibilidad medida (no es lever)")
    cols = ["caso", "encounter_id", "variable", "valor", "shap_logodds", "abs_shap",
            "rank_local", "en_top_k", "max_abs_delta", "delta_min", "delta_max",
            "valor_max_delta", "delta_incluye_change", "interpretable",
            "destino_contaminado", "clasificacion"]
    return merged[cols].sort_values(["caso", "rank_local"])


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def _bars(ax, sub: pd.DataFrame, rest: float, n_rest: int):
    """Barras horizontales de contribución, de mayor a menor |SHAP| de arriba abajo."""
    labels, vals = [], []
    for r in sub.itertuples():
        mark = "" if r.interpretable else MARK_NO_INTERP
        if r.variable == "discharge_disposition_id" and r.destino_contaminado:
            mark += MARK_CONTAM
        labels.append(f"{r.variable}{mark} = {r.valor}")
        vals.append(r.shap_logodds)
    if n_rest:
        labels.append(f"resto ({n_rest} variables)")
        vals.append(rest)
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=[UP if v > 0 else DOWN for v in vals], height=0.68)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.axvline(0, color=E.MUTED, lw=1)
    span = max(abs(min(vals)), abs(max(vals))) or 1.0
    for yi, v in zip(y, vals):
        ax.text(v + np.sign(v) * span * 0.02, yi, f"{v:+.3f}", va="center",
                ha="left" if v > 0 else "right", fontsize=7.5, color=E.MUTED)
    ax.set_xlim(min(0, min(vals)) - span * 0.22, max(0, max(vals)) + span * 0.22)
    XS._style(ax)
    ax.grid(axis="x", color=E.GRID, lw=0.6)
    ax.set_axisbelow(True)


def _split_top(case_rows: pd.DataFrame, n: int):
    """(top-n por |SHAP|, suma del resto, cuántas quedan en el resto)."""
    ordered = case_rows.sort_values("rank_local")
    top, rest = ordered.head(n), ordered.iloc[n:]
    return top, float(rest["shap_logodds"].sum()), len(rest)


def plot_waterfall(case_rows: pd.DataFrame, n: int = TOP_N_FIG):
    """Un waterfall por caso, en log-odds, con la probabilidad en el título."""
    first = case_rows.iloc[0]
    top, rest, n_rest = _split_top(case_rows, n)
    fig, ax = plt.subplots(figsize=(8.2, 0.33 * (len(top) + 1) + 1.9), facecolor=XS.SURFACE)
    _bars(ax, top, rest, n_rest)
    ax.set_xlabel("contribución SHAP (log-odds)", color=E.MUTED, fontsize=9)
    reingreso = "sí" if first["reingresa"] else "no"
    ax.set_title(f"{first['caso']} — encuentro {first['encounter_id']}\n"
                 f"p = {first['prob']:.4f} | log-odds {first['log_odds']:.3f} "
                 f"= base {first['base']:.3f} + contribuciones | "
                 f"reingresó en <30 d: {reingreso}",
                 loc="left", color=E.INK, fontsize=10.5)
    nota = f"{MARK_NO_INTERP} no supera las sondas de ruido (Fase 3b): no se interpreta"
    if bool(first["destino_contaminado"]):
        nota += f"\n{MARK_CONTAM} destino 22/5: contribución señalada por la auditoría de fuga"
    ax.text(0, -0.085, nota, transform=ax.transAxes, va="top", color=E.MUTED, fontsize=7.5)
    return XS._save(fig, f"shap_local_{first['caso']}.png")


def plot_contrastive_pairs(contrib: pd.DataFrame, n: int = TOP_N_PAIR_FIG):
    """Los dos pares que contrastan al modelo con la regla number_inpatient>0."""
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 7.6), facecolor=XS.SURFACE)
    for (titulo, caso_si, caso_no), row_axes in zip(PAIR_ROWS, axes):
        for ax, caso in zip(row_axes, (caso_si, caso_no)):
            rows = contrib[contrib["caso"] == caso]
            top, rest, n_rest = _split_top(rows, n)
            _bars(ax, top, rest, n_rest)
            first = rows.iloc[0]
            ax.set_title(f"{caso}\np = {first['prob']:.4f} | "
                         f"reingresó: {'sí' if first['reingresa'] else 'no'}",
                         loc="left", color=E.INK, fontsize=9.5)
        row_axes[0].set_ylabel(titulo, color=E.MUTED, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("contribución SHAP (log-odds)", color=E.MUTED, fontsize=9)
    fig.suptitle("Pares contrastivos: mismo perfil para la regla simple, resultado distinto",
                 x=0.01, ha="left", color=E.INK, fontsize=12)
    # El pie va anclado FUERA del lienzo (y negativo, va="top"): bbox_inches="tight" lo
    # incluye al guardar y así no puede solaparse con los títulos de los paneles.
    # La advertencia es obligatoria porque esta figura va suelta al anexo: sin ella se
    # lee como si demostrara que el modelo capta riesgo que la regla simple no ve.
    fig.text(0.01, -0.012,
             f"{MARK_NO_INTERP} no supera las sondas de ruido (Fase 3b): no se interpreta"
             f"   |   {MARK_CONTAM} destino 22/5 (auditoría de fuga)\n{PAIR_CAVEAT}",
             ha="left", va="top", color=E.MUTED, fontsize=8.5)
    return XS._save(fig, "shap_local_pares_contrastivos.png")


# ---------------------------------------------------------------------------
# Lectura por caso (plantilla fija, criterio 5)
# ---------------------------------------------------------------------------
def case_reading(case_rows: pd.DataFrame, stability: pd.DataFrame,
                 k: int = TOP_N_READING) -> str:
    """Texto generado a partir de los datos. Ninguna frase se escribe por caso."""
    first = case_rows.iloc[0]
    caso = first["caso"]
    st = stability.set_index("caso").loc[caso]
    ordered = case_rows.sort_values("rank_local")
    interp = ordered[ordered["interpretable"]].head(k)
    no_interp = ordered.head(TOP_N_TABLE)
    no_interp = no_interp[~no_interp["interpretable"]]["variable"].tolist()

    # Varios grupos se definen POR el resultado (reingresa / no reingresa), así que su
    # tasa real es 100% o 0% por construcción y no dice nada. Se marca para no leerla
    # como si fuera el riesgo del grupo.
    tasa = st["tasa_real_%"]
    nota_tasa = " (trivial: el grupo se define por el resultado)" if tasa in (0.0, 100.0) else ""
    # El separador de miles se aplica SOLO al número: hacerlo sobre la línea entera
    # convertía las comas del texto en puntos.
    n_grupo = f"{int(st['n_grupo']):,}".replace(",", ".")
    lineas = [f"{caso} — encuentro {int(first['encounter_id'])} | p = {first['prob']:.4f} "
              f"| reingresó <30 d: {'sí' if first['reingresa'] else 'no'}",
              f"  Grupo de selección: {GROUP_DESC[caso]} "
              f"(n={n_grupo}, tasa real {tasa:.1f}%{nota_tasa})"]
    if interp.empty:
        lineas.append(f"  Top-{k} interpretables: NINGUNA de sus variables supera las sondas "
                      "de ruido; no se lee ninguna contribución de este caso.")
    else:
        lineas.append(f"  Top-{k} interpretables (contribución en log-odds, rango global):")
        for r in interp.itertuples():
            lineas.append(f"    - {r.variable} = {r.valor}: {r.shap_logodds:+.3f} "
                          f"(local #{r.rank_local}, global #{r.rank_global})")
    lineas.append(f"  Estabilidad del top-3 local entre folds: solape medio "
                  f"{st['solape_top3_medio']:.2f}, mínimo {st['solape_top3_min']:.2f}")
    marcas = []
    if no_interp:
        marcas.append(f"{len(no_interp)} de las {TOP_N_TABLE} primeras no se interpretan "
                      f"({', '.join(no_interp)})")
    marcas.append("destino 22/5: la contribución del destino no se lee como riesgo"
                  if bool(first["destino_contaminado"]) else "destino no contaminado")
    lineas.append(f"  Marcas: {'; '.join(marcas)}")
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
def main():
    pd.set_option("display.width", 200, "display.max_colwidth", 46, "display.max_rows", 300)
    model = XS.load_final_model()
    booster, L = model["booster"], model["Lva"]
    values, base = XS.compute_shap(booster, L)          # mismas comprobaciones que la Fase 3a
    print(f"SHAP en val: {values.shape[0]:,} x {values.shape[1]} | base (log-odds) = {base:.4f}")

    interp = load_interpretable()
    print(f"Variables interpretables (ganan a la sonda en {N_PROBE_SEEDS}/{N_PROBE_SEEDS} "
          f"semillas): {int(interp.sum())} de {len(interp)}")
    print("  " + ", ".join(interp.index[interp]))
    print("  NO interpretables entre las de más peso: "
          + ", ".join(interp.index[~interp][:8]) + ", ...")

    rank_global = pd.Series(np.abs(values).mean(axis=0), index=L.columns) \
        .rank(ascending=False).astype(int)
    cases = pd.read_csv(REPORT_DIR / "xai_local_cases.csv")
    stability = load_local_stability()
    sens = load_sensitivity()

    contrib = contributions(values, L, cases, interp, rank_global, base, booster)
    print("Aditividad por caso: OK (error < 1e-6 en los 11)")
    top = top_features(contrib, stability)
    cruce = attribution_vs_sensitivity(booster, L, contrib, sens)

    contrib.to_csv(REPORT_DIR / "xai_local_contributions.csv", index=False)
    top.to_csv(REPORT_DIR / "xai_local_top_features.csv", index=False)
    cruce.to_csv(REPORT_DIR / "xai_local_attribution_vs_sensitivity.csv", index=False)

    figs = [plot_waterfall(contrib[contrib["caso"] == c.caso]) for c in cases.itertuples()]
    figs.append(plot_contrastive_pairs(contrib))

    print("\n" + "=" * 78)
    print("TOP-5 LOCAL POR CASO (contribución en log-odds; ° no se interpreta)")
    print("=" * 78)
    shown = top.copy()
    shown["variable"] = np.where(shown["interpretable"], shown["variable"],
                                 shown["variable"] + MARK_NO_INTERP)
    print(shown[["caso", "prob", "variable", "valor", "shap_logodds", "rank_local",
                 "rank_global", "destino_contaminado", "solape_top3_medio"]]
          .round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("ATRIBUCIÓN (SHAP) FRENTE A SENSIBILIDAD (diagnóstico (b) de la Fase 3c)")
    print("Miden cosas distintas; la tabla documenta la discrepancia, no la denuncia.")
    print("=" * 78)
    CF.show(cruce, "0 filas: ningún caso elegible tiene sensibilidad medida.",
            cols=["caso", "variable", "valor", "shap_logodds", "rank_local",
                  "max_abs_delta", "delta_incluye_change", "clasificacion"])
    print("\nRecuento por clasificación:")
    print(cruce["clasificacion"].value_counts().to_string())

    print("\n" + "=" * 78)
    print("LECTURA POR CASO (plantilla fija generada de los datos)")
    print("Los 3 perfil_alto_* son centroides de k-means sobre SHAP: sus explicaciones")
    print("son representativas de su grupo POR CONSTRUCCIÓN, no por casualidad.")
    print("=" * 78)
    for case in cases.itertuples():
        print()
        print(case_reading(contrib[contrib["caso"] == case.caso], stability))

    print("\nFiguras guardadas:")
    for path in figs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
