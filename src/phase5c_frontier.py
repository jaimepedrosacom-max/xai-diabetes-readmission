"""
Fase 5C, paso 1 — Espacio de acciones, distancia a la frontera y escalones de relajación.

Prerregistro: el bloque "Prerregistro de la Fase 5C" de CLAUDE.md, commiteado en `44db202`
ANTES de este script.

POR QUÉ ESTA FASE: el profesor objetó, con razón, que en teoría un contrafactual siempre
existe si alguna variable movible tiene efecto, y que la frontera del modelo está dentro de
los datos. Nuestro "no existe ningún contrafactual" de la Fase 3c era cierto SOLO dentro del
espacio de acciones restringido, y el texto no lo decía con esas palabras. Este script
convierte la ausencia en una MEDIDA: a qué distancia está la frontera, y de qué está hecha.

REDACCIÓN OBLIGATORIA, que el CSV `phase5c_ladder.csv` lleva en su columna `afirmacion`:
un "no existe ninguno" solo es una DEMOSTRACIÓN si el espacio se ha enumerado COMPLETO. Si
no, la frase correcta es "la búsqueda no encontró ninguno". Nunca "no existe ningún
contrafactual" a secas.

MONTAJE
    Modelo: el ENTREGABLE congelado (LightGBM corregido SIN `payer_code`, 123 árboles), que
    se reproduce y cuyo guardarraíl se exige con `assert` (123 árboles Y PR-AUC val 0,1868).
    Se rehace todo sobre el entregable porque la Fase 3c usó el modelo de la Fase 2: 122
    árboles, las 44 features CON `payer_code` y el target ORIGINAL. Datos: val, etiqueta
    corregida. TEST NO SE TOCA: el tercer split se descarta al cargarlo.

    Escala: PROBABILIDAD en todas las comparaciones con el umbral, nunca log-odds.

EL MARCO, explícito porque es donde nuestra propia Fase 3c se explicaba mal: la relajación
que mide algo es DESCONGELAR variables, no subir k. Subir el número de cambios simultáneos
no puede crear contrafactuales donde ≤ 2 ya agota el espacio. E1 (sin límite de k) se
mantiene precisamente porque es lo que DEMUESTRA que k no era la restricción; que no cambie
nada es el resultado.

DiCE NO se corre aquí, y la razón está prerregistrada: en un espacio de 2 a 26 combinaciones
la enumeración exhaustiva es estrictamente más fuerte que un muestreador aleatorio, porque
prueba la inexistencia y DiCE no puede probarla.
"""

import numpy as np
import pandas as pd

import data as D
import preprocessing as P
import target_correction as TC
import xai_counterfactuals as XCF
from phase4_test_final import ABL_TREES, ABL_VAL_PR_AUC, fit_with_guardrail
from phase5_lr_comparison import lr_feature_lists
from phase5b_probes import truncated_booster

REPORT_DIR = D.PROJECT_ROOT / "reports"
DISCHARGE = "discharge_disposition_id"
TOP_KS = (0.05, 0.10, 0.20)
K_PRIMARY = 0.05                 # el umbral de la fase: corte del top 5% del entregable
ALL_MED_LEVELS = ["No", "Steady", "Up", "Down"]   # con `No` se puede empezar y suspender
BATCH = 20000                    # candidatos por llamada al booster, para no cargar memoria

# Cifras publicadas de la Fase 3c con las que se AUDITA el refactor de `xai_counterfactuals`.
# Dependen solo de los DATOS de cada fila, no del modelo, así que son exigibles aunque el
# modelo haya cambiado. Fuente: reports/xai_counterfactuals_diag_minimo_alcanzable.csv.
AUDIT_CSV = "xai_counterfactuals_diag_minimo_alcanzable.csv"

# Los escalones, fijados en el prerregistro. `exhaustiva` es la columna que decide qué frase
# se puede usar en la memoria.
ESCALONES = [
    {"escalon": "E0", "descripcion": "Fase 3c: <=2 cambios, destinos 1/6/3, 22 y 5 "
                                     "congelados, solo fármacos ya prescritos",
     "freeze_suspect": True, "destinations": XCF.DESTINATIONS,
     "med_levels": XCF.MED_LEVELS, "free_meds": False, "max_changes": 2, "enumerable": True},
    {"escalon": "E1", "descripcion": "E0 SIN límite de k (producto cartesiano completo)",
     "freeze_suspect": True, "destinations": XCF.DESTINATIONS,
     "med_levels": XCF.MED_LEVELS, "free_meds": False, "max_changes": None, "enumerable": True},
    {"escalon": "E2", "descripcion": "E1 + se DESCONGELAN los destinos 22 y 5",
     "freeze_suspect": False, "destinations": XCF.DESTINATIONS,
     "med_levels": XCF.MED_LEVELS, "free_meds": False, "max_changes": None, "enumerable": True},
    {"escalon": "E3", "descripcion": "E2 + destino a TODOS sus niveles de train",
     "freeze_suspect": False, "destinations": None,
     "med_levels": XCF.MED_LEVELS, "free_meds": False, "max_changes": None, "enumerable": True},
    {"escalon": "E4", "descripcion": "E3 + los 21 fármacos libres (incluido No: empezar y "
                                     "suspender), con k <= 3",
     "freeze_suspect": False, "destinations": None,
     "med_levels": ALL_MED_LEVELS, "free_meds": True, "max_changes": 3, "enumerable": True},
]

# Escalones que NO se enumeran, y que se declaran como tales en lugar de sustituirse por una
# búsqueda heurística presentada como prueba.
NO_ENUMERABLES = [
    # El tamaño NO se estima a mano: se calcula con count_combinations y se escribe en el
    # CSV. El prerregistro traía la estimación «~4^21 * 18 ~ 8e13», que es una cota alta
    # porque supone 4 niveles en train para los 21 fármacos, y no los tienen todos.
    {"escalon": "E4-completo", "descripcion": "E4 sin límite de k",
     "motivo": "{:.3e} combinaciones: no enumerable"},
    {"escalon": "E5", "descripcion": "sin restricciones: las 43 features del entregable",
     "motivo": "contiene variables numéricas continuas, así que el espacio no es finito"},
]


def space_for(row: pd.DataFrame, esc: dict) -> dict:
    """El espacio de acciones de un escalón, con la función de la Fase 3c."""
    return XCF.action_space(row, freeze_suspect=esc["freeze_suspect"],
                            destinations=esc["destinations"],
                            med_levels=esc["med_levels"], free_meds=esc["free_meds"])


def scorer_for(booster, row: pd.DataFrame, esc: dict) -> XCF.Scorer:
    """El Scorer del escalón.

    Si el escalón permite mover fármacos que el paciente NO toma, las dos derivadas
    (`change` y `diabetesMed`) pueden cambiar, así que hay que derivarlas mirando TODAS las
    columnas de medicación. En los escalones E0-E3 ningún fármaco se mueve a `No`, así que
    el comportamiento es el de la 3c.
    """
    if esc["free_meds"]:
        return XCF.Scorer(booster, row, derive_cols=P.MED_COLS, derive_diabetes_med=True)
    return XCF.Scorer(booster, row)


def count_combinations(alt_counts, max_changes=None) -> int:
    """Nº EXACTO de combinaciones con 1..max_changes cambios, SIN enumerarlas.

    Producto de polinomios: cada variable con `a` alternativas aporta (1 + a·x), y el
    coeficiente de x^j es el nº de combinaciones con exactamente j cambios. Sirve para
    declarar el tamaño de un escalón que no vamos a enumerar (E4 completo) sin tener que
    construirlo.
    """
    poly = [1]
    for a in alt_counts:
        nxt = [0] * (len(poly) + 1)
        for j, c in enumerate(poly):
            nxt[j] += c
            nxt[j + 1] += c * a
        poly = nxt
    limit = len(alt_counts) if max_changes is None else min(max_changes, len(alt_counts))
    return int(sum(poly[1:limit + 1]))


def score_in_batches(scorer: XCF.Scorer, changes: list) -> np.ndarray:
    """`score_changes` por lotes: los escalones altos tienen decenas de miles de candidatos."""
    out = [XCF.score_changes(scorer, changes[i:i + BATCH])
           for i in range(0, len(changes), BATCH)]
    return np.concatenate(out) if out else np.array([])


# ---------------------------------------------------------------------------
# Auditoría del refactor: por REPRODUCCIÓN, no por revisión del diff
# ---------------------------------------------------------------------------
def audit_refactor(L: pd.DataFrame) -> pd.DataFrame:
    """Las llamadas SIN los argumentos nuevos reproducen las cifras publicadas de la 3c.

    `n_candidatos` y `palancas` de los 6 casos elegibles dependen solo de los datos de cada
    fila, no del modelo, así que son exigibles aunque el modelo haya cambiado. Si no cuadran,
    el refactor cambió algo y la fase NO sigue.
    """
    pub = pd.read_csv(REPORT_DIR / AUDIT_CSV)
    rows = []
    for _, r in pub.iterrows():
        row = L.loc[[int(r["encounter_id"])]]
        space = XCF.action_space(row)                 # defaults = E0 = espacio de la 3c
        changes = XCF.enumerate_changes(space)        # default MAX_CHANGES = 2
        rows.append({"caso": r["caso"], "encounter_id": int(r["encounter_id"]),
                     "n_candidatos_publicado": int(r["n_candidatos"]),
                     "n_candidatos_ahora": len(changes),
                     "palancas_publicado": r["palancas"],
                     "palancas_ahora": ", ".join(space) or "(ninguna)",
                     "n_vars_accionables": len(space),
                     "alternativas_por_var": ", ".join(str(len(v)) for v in space.values())})
    aud = pd.DataFrame(rows)
    aud["ok"] = ((aud["n_candidatos_publicado"] == aud["n_candidatos_ahora"])
                 & (aud["palancas_publicado"] == aud["palancas_ahora"]))
    print(aud[["caso", "n_candidatos_publicado", "n_candidatos_ahora", "n_vars_accionables",
               "alternativas_por_var", "ok"]].to_string(index=False))
    assert aud["ok"].all(), (
        "PARADA: el refactor de xai_counterfactuals NO reproduce las cifras publicadas de la "
        "Fase 3c (n_candidatos 18/18/8/2/2/2 y las mismas palancas). No se sigue.")
    print("Auditoría del refactor OK: los 6 casos reproducen n_candidatos y palancas.")
    return aud


# ---------------------------------------------------------------------------
# (f) Elegibilidad de los 11 casos, recalculada con el entregable
# ---------------------------------------------------------------------------
def eligibility(cases: pd.DataFrame, probs: pd.Series, cuts: dict) -> pd.DataFrame:
    """Los 11 casos NO se reeligen: se recalcula su estado con el entregable.

    Los casos vienen de `reports/xai_local_cases.csv`, fijados en la Fase 3b por
    `src/xai_cases.py` con regla y semilla. Volver a elegirlos con el entregable sería una
    selección NUEVA y rompería la cadena con las Fases 3b, 3c y 3d.
    """
    out = cases.copy().rename(columns={"prob": "prob_modelo_viejo"})
    out["prob_entregable"] = [float(probs.loc[int(e)]) for e in out["encounter_id"]]
    for k, cut in cuts.items():
        out["marcado_{:.0%}".format(k)] = out["prob_entregable"] >= cut
    cut5 = cuts[K_PRIMARY]
    out["motivo"] = np.where(
        out["prob_entregable"] >= cut5,
        "prob >= corte del top 5% del entregable ({:.4f})".format(cut5),
        ("prob < corte del top 5% del entregable ({:.4f}): el modelo NO lo marca, así que la "
         "pregunta del contrafactual NO SE PLANTEA").format(cut5))
    return out


# ---------------------------------------------------------------------------
# (a) Tabla del espacio de acciones: se dicen las variables, una por una
# ---------------------------------------------------------------------------
def freeze_reason(col: str, row: pd.DataFrame) -> str:
    """Motivo de congelación en E0, con las razones ya fijadas en la Fase 3c."""
    if col == DISCHARGE:
        if str(row[col].iloc[0]) in XCF.SUSPECT_CODES:
            return ("CONGELADA: destino 22 o 5, señalado por audit_discharge_leakage como "
                    "contaminado por la definición de la etiqueta")
        return "accionable solo hacia 1, 6 o 3"
    if col in XCF.LAB_COLS:
        return "CONGELADA: se puede pedir la prueba, no elegir su resultado"
    if col in ("change", "diabetesMed"):
        return "CONGELADA: derivada de las columnas de medicación, nunca se elige"
    if col in P.MED_COLS:
        if pd.isna(row[col].iloc[0]):
            return "CONGELADA: nivel de val que no existe en train (el modelo no lo conoce)"
        if str(row[col].iloc[0]) == "No":
            return ("CONGELADA en E0: el paciente no lo toma, y empezar un fármaco queda "
                    "fuera porque no vemos contraindicaciones")
        return "accionable entre Steady/Up/Down"
    return "CONGELADA: inmutable o ya ocurrida"


def action_space_vars(L: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """Una fila por caso x variable del entregable (43), con su motivo de congelación."""
    tipo = {c: "numérica" for c in P.NUM_COLS + P.COUNT_COLS}
    rows = []
    for _, case in cases.iterrows():
        row = L.loc[[int(case["encounter_id"])]]
        e0 = space_for(row, ESCALONES[0])
        for col in L.columns:
            opts = e0.get(col, [])
            rows.append({"caso": case["caso"], "encounter_id": int(case["encounter_id"]),
                         "variable": col, "tipo": tipo.get(col, "categórica"),
                         "valor_actual": str(row[col].iloc[0]),
                         "accionable_E0": col in e0,
                         "valores_permitidos_E0": ", ".join(map(str, opts)) or "",
                         "n_alternativas_E0": len(opts),
                         "motivo": freeze_reason(col, row)})
    return pd.DataFrame(rows)


def action_space_sizes(L: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """Nº de combinaciones por caso y escalón, con ≤2, ≤3 y el producto cartesiano completo."""
    rows = []
    for _, case in cases.iterrows():
        row = L.loc[[int(case["encounter_id"])]]
        for esc in ESCALONES:
            space = space_for(row, esc)
            alts = [len(v) for v in space.values()]
            rows.append({
                "caso": case["caso"], "encounter_id": int(case["encounter_id"]),
                "escalon": esc["escalon"], "descripcion": esc["descripcion"],
                "n_vars_accionables": len(space),
                "variables": ", ".join(space) or "(ninguna)",
                "n_comb_hasta_2": count_combinations(alts, 2),
                "n_comb_hasta_3": count_combinations(alts, 3),
                "n_comb_completo": count_combinations(alts, None),
                "n_enumeradas": count_combinations(alts, esc["max_changes"]),
                "max_changes": "sin límite" if esc["max_changes"] is None else esc["max_changes"],
                "exhaustiva": True,
                "motivo_exhaustividad": (
                    "el escalón se DEFINE con k <= {}, y se enumeran todas sus combinaciones"
                    .format(esc["max_changes"]) if esc["max_changes"] is not None else
                    "se enumera el producto cartesiano completo del espacio")})
        # Los dos escalones declarados NO enumerables: se dice su tamaño, no se construyen.
        full = space_for(row, ESCALONES[-1])
        alts_full = [len(v) for v in full.values()]
        for dec in NO_ENUMERABLES:
            n = (count_combinations(alts_full, None) if dec["escalon"] == "E4-completo"
                 else np.nan)
            motivo = dec["motivo"].format(n) if "{" in dec["motivo"] else dec["motivo"]
            rows.append({
                "caso": case["caso"], "encounter_id": int(case["encounter_id"]),
                "escalon": dec["escalon"], "descripcion": dec["descripcion"],
                "n_vars_accionables": len(full) if dec["escalon"] == "E4-completo" else len(L.columns),
                "variables": ", ".join(full) if dec["escalon"] == "E4-completo" else "las 43",
                "n_comb_hasta_2": np.nan, "n_comb_hasta_3": np.nan,
                "n_comb_completo": n, "n_enumeradas": np.nan, "max_changes": "sin límite",
                "exhaustiva": False, "motivo_exhaustividad": motivo})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (b) Distancia a la frontera y (c) escalones
# ---------------------------------------------------------------------------
def frontier_and_ladder(booster, L: pd.DataFrame, cases: pd.DataFrame, cut: float):
    """Las dos cifras del profesor y su diferencia, por caso y escalón.

    reduccion_necesaria       = prob_orig - umbral
    reduccion_maxima_alcanzable = prob_orig - prob_min_alcanzable (dentro del escalón)
    deficit                   = prob_min_alcanzable - umbral   (positivo = NO se cruza)

    Se calculan para los 11 casos, no solo para los marcados. Las filas de los casos NO
    marcados llevan `marcado_top5 = False` y su `deficit` negativo NO se puede leer como
    "sí cruza": para ellos la pregunta del contrafactual no se plantea.
    """
    dist, ladder = [], []
    for _, case in cases.iterrows():
        enc = int(case["encounter_id"])
        row = L.loc[[enc]]
        marcado = bool(float(booster.predict(row)[0]) >= cut)
        for esc in ESCALONES:
            scorer = scorer_for(booster, row, esc)
            base = float(scorer.score({}, 1)[0])
            space = space_for(row, esc)
            changes = XCF.enumerate_changes(space, esc["max_changes"])
            print("    {:38s} {:3s}  {:6d} candidatos".format(
                case["caso"], esc["escalon"], len(changes)), flush=True)
            meta = {"caso": case["caso"], "encounter_id": enc, "escalon": esc["escalon"],
                    "destino": str(row[DISCHARGE].iloc[0]), "prob_orig": base,
                    "umbral_top5": cut, "marcado_top5": marcado,
                    "n_vars_accionables": len(space), "n_candidatos": len(changes)}
            if not changes:
                dist.append({**meta, "prob_min_alcanzable": np.nan,
                             "reduccion_necesaria": base - cut,
                             "reduccion_maxima_alcanzable": 0.0, "deficit": base - cut,
                             "cruza": False, "todo_cambio_empeora": False,
                             "combinacion_minima": "(espacio vacío)"})
                ladder.append({**meta, "primer_cf": False, "n_cf": 0,
                               "min_n_cambios_del_cf": np.nan, "exhaustiva": True,
                               "afirmacion": "no existe ninguno en {} (espacio de acciones "
                                             "VACÍO)".format(esc["escalon"])})
                continue
            probs = score_in_batches(scorer, changes)
            best = int(probs.argmin())
            cf = [(c, p) for c, p in zip(changes, probs) if p < cut]
            dist.append({**meta, "prob_min_alcanzable": float(probs.min()),
                         "reduccion_necesaria": base - cut,
                         "reduccion_maxima_alcanzable": base - float(probs.min()),
                         "deficit": float(probs.min()) - cut,
                         "cruza": bool(probs.min() < cut),
                         # Negativa = TODO cambio disponible SUBE la puntuación.
                         "todo_cambio_empeora": bool(probs.min() > base),
                         "combinacion_minima": XCF.describe(changes[best], row)})
            ladder.append({**meta, "primer_cf": bool(cf), "n_cf": len(cf),
                           "min_n_cambios_del_cf": min((len(c) for c, _ in cf), default=np.nan),
                           "exhaustiva": True,
                           "afirmacion": ("existen {} contrafactuales en {}".format(
                               len(cf), esc["escalon"]) if cf else
                               "no existe ninguno en {} (enumeración completa del escalón)"
                               .format(esc["escalon"]))})
        n_full = count_combinations([len(v) for v in space_for(row, ESCALONES[-1]).values()],
                                    None)
        for dec in NO_ENUMERABLES:
            ladder.append({"caso": case["caso"], "encounter_id": enc,
                           "escalon": dec["escalon"], "destino": str(row[DISCHARGE].iloc[0]),
                           "prob_orig": np.nan, "umbral_top5": cut, "marcado_top5": marcado,
                           "n_vars_accionables": np.nan, "n_candidatos": np.nan,
                           "primer_cf": np.nan, "n_cf": np.nan,
                           "min_n_cambios_del_cf": np.nan, "exhaustiva": False,
                           "afirmacion": "NO ENUMERABLE ({}): esta fase no puede afirmar nada "
                                         "sobre inexistencia aquí".format(
                                             dec["motivo"].format(n_full)
                                             if "{" in dec["motivo"] else dec["motivo"])})
    return pd.DataFrame(dist), pd.DataFrame(ladder)


# ---------------------------------------------------------------------------
# Medida PROPIA de la 5C: marcados sin acciones posibles (dos criterios)
# ---------------------------------------------------------------------------
def no_actions(L: pd.DataFrame, probs: pd.Series, cuts: dict) -> pd.DataFrame:
    """% de marcados sin acciones posibles en E0, con DOS criterios y sus denominadores.

    Criterio 1: espacio de acciones E0 VACÍO (no hay NADA que mover).
    Criterio 2, el que responde mejor a la objeción: NINGÚN FÁRMACO accionable, es decir
    pacientes cuya única variable movible es el destino al alta, o que no tienen ninguna.
    Se cuenta aparte porque el destino NO es una palanca: refleja en buena parte la GRAVEDAD
    del paciente y no una decisión libre del hospital (Fase 3c), así que ofrecerlo como
    acción es justo el contrafactual engañoso.

    AVISO sobre cómo se cita esta cifra: NO sostiene un "a la mayoría de los marcados no se
    les puede ofrecer ninguna acción que no sea mover el destino". La medida dice que es una
    MINORÍA la que se queda sin ninguna palanca de medicación, y el resto SÍ tiene alguna.
    Que esa palanca no sirva de nada es otro argumento, sobre el EFECTO y no sobre la
    EXISTENCIA, y se apoya en este orden: PRIMERO en la enumeración de E4 de esta misma fase
    (dos de los tres casos marcados no cruzan el umbral ni con los 21 fármacos libres y
    k <= 3, con sus 29.167 candidatos enumerados al completo, que es una demostración), y
    DESPUÉS en las sondas de ruido de la Fase 5B (ningún fármaco las supera en el
    entregable), que son evidencia global y más débil que la enumeración. Los dos argumentos
    van juntos y no se confunden.

    Es el recálculo SOBRE EL ENTREGABLE del caso "edad y altura" del profesor: a una parte de
    los pacientes marcados no se le puede ofrecer NINGUNA acción, ni siquiera mala. La cifra
    del modelo viejo (12,09% en el grupo 22/5 del top 5%,
    `xai_counterfactuals_population.csv`) NO se compara con esta: son dos conjuntos de filas
    marcadas distintos, elegidos por dos modelos distintos sobre dos etiquetas distintas.

    Depende solo de los datos de cada fila; el modelo interviene únicamente en QUÉ filas se
    marcan. Sale del mismo `action_space` que todo lo demás, así que no hay un segundo
    cálculo que pueda divergir.
    """
    marcadas = probs.index[probs >= min(cuts.values())]
    vacio, sin_farmaco, grupo = {}, {}, {}
    for enc in marcadas:
        row = L.loc[[enc]]
        space = space_for(row, ESCALONES[0])
        vacio[enc] = not space
        # Sin NINGÚN fármaco accionable: o el espacio está vacío, o su única variable
        # movible es el destino al alta, que el proyecto tiene establecido como CONFUSIÓN y
        # NO palanca (Fase 3c: refleja la gravedad del paciente, no una decisión libre).
        sin_farmaco[enc] = not [v for v in space if v != DISCHARGE]
        grupo[enc] = ("22/5 (destino congelado)"
                      if str(row[DISCHARGE].iloc[0]) in XCF.SUSPECT_CODES else "otros destinos")
    vacio = pd.Series(vacio)
    sin_farmaco = pd.Series(sin_farmaco)
    grupo = pd.Series(grupo)
    criterios = [("sin NINGUNA variable accionable", vacio),
                 ("sin NINGÚN fármaco accionable (solo el destino, o nada)", sin_farmaco)]
    rows = []
    for k, cut in cuts.items():
        sel = probs.loc[marcadas] >= cut
        idx = sel.index[sel]
        for name, sub in [("TODOS los marcados", idx),
                          ("22/5 (destino congelado)", idx[grupo.loc[idx] == "22/5 (destino congelado)"]),
                          ("otros destinos", idx[grupo.loc[idx] == "otros destinos"])]:
            n = len(sub)
            for crit, flag in criterios:
                sin = int(flag.loc[sub].sum()) if n else 0
                rows.append({"top_k": "{:.0%}".format(k), "umbral": cut, "grupo": name,
                             "criterio": crit, "n_marcados": n, "n_sin_acciones": sin,
                             "pct_sin_acciones": 100 * sin / n if n else np.nan,
                             "espacio": "E0"})
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 220, "display.max_colwidth", 60)

    # --- Datos: val. El tercer split se descarta aquí mismo. --------------------
    train, val, _test = P.load_model_data()
    del _test
    flags = TC.planned_flags()
    ytr_new = TC.relabel(train, flags).to_numpy()
    yva_new = TC.relabel(val, flags).to_numpy()
    feats, _cats = lr_feature_lists()            # P.FEATURES menos payer_code
    Ltr, Lva = D.to_categorical(train[feats], val[feats])
    print(f"Val: {len(Lva)} filas, {Lva.shape[1]} variables, prevalencia "
          f"{100 * yva_new.mean():.4f}% (etiqueta corregida). TEST NO SE TOCA.")

    # --- Guardarraíl del entregable, antes de medir nada ------------------------
    model = fit_with_guardrail(Ltr, ytr_new, Lva, yva_new,
                               "ENTREGABLE (sin payer_code)", ABL_TREES, ABL_VAL_PR_AUC)
    booster = truncated_booster(model, Lva)
    probs = pd.Series(booster.predict(Lva), index=Lva.index)

    # --- Umbral: quantile(0.95), con thresholds() al lado como comprobación -----
    cuts = {k: float(probs.quantile(1 - k)) for k in TOP_KS}
    alt = XCF.thresholds(probs.to_numpy())
    print("\nUmbrales del entregable en val (probs.quantile), y el valor de "
          "xai_counterfactuals.thresholds() al lado:")
    for k in TOP_KS:
        n_marc = int((probs >= cuts[k]).sum())
        print("  top {:>3.0%}: corte {:.6f} -> {} filas marcadas | thresholds() {:.6f} "
              "(diferencia {:+.2e}); ceil(k*n) = {}".format(
                  k, cuts[k], n_marc, alt[k], cuts[k] - alt[k], int(np.ceil(k * len(probs)))))
    cut5 = cuts[K_PRIMARY]

    # --- Auditoría del refactor, por reproducción -------------------------------
    print("\n--- Auditoría del refactor de xai_counterfactuals (por REPRODUCCIÓN) ---")
    aud = audit_refactor(Lva)

    # --- (f) Elegibilidad de los 11 --------------------------------------------
    cases = pd.read_csv(REPORT_DIR / "xai_local_cases.csv")
    elig = eligibility(cases, probs, cuts)
    print("\n--- (f) Los 11 casos con el entregable (NO se reeligen) ---")
    print(elig[["caso", "prob_modelo_viejo", "prob_entregable", "marcado_5%", "marcado_10%",
                "marcado_20%"]].round(4).to_string(index=False))
    print("Marcados en el top 5% del entregable: {} de {}. En el top 10%: {}.".format(
        int(elig["marcado_5%"].sum()), len(elig), int(elig["marcado_10%"].sum())))

    # --- (a) Espacio de acciones ----------------------------------------------
    espacio_vars = action_space_vars(Lva, cases)
    tamanos = action_space_sizes(Lva, cases)
    print("\n--- (a) Tamaño del espacio de acciones por caso y escalón ---")
    print(tamanos[tamanos["escalon"].isin(["E0", "E1", "E2", "E3", "E4"])]
          [["caso", "escalon", "n_vars_accionables", "n_comb_hasta_2", "n_comb_hasta_3",
            "n_comb_completo", "n_enumeradas"]].to_string(index=False))
    print("\nVerificación del MARCO: ¿coincide ≤2 con el producto cartesiano completo en E0?")
    e0 = tamanos[tamanos["escalon"] == "E0"]
    iguales = e0[e0["n_comb_hasta_2"] == e0["n_comb_completo"]]
    print("  {} de {} casos: {}".format(len(iguales), len(e0),
                                        ", ".join(iguales["caso"])))

    # --- Medida propia: marcados sin ninguna acción ----------------------------
    print("\n--- Marcados SIN acciones posibles en E0 (medida propia de la 5C) ---")
    print("Dos criterios: sin NINGUNA variable accionable, y sin NINGÚN fármaco accionable")
    print("(el segundo cuenta a quien solo puede mover el destino al alta, que es "
          "CONFUSIÓN y no palanca).")
    sin_acc = no_actions(Lva, probs, cuts)
    print(sin_acc.round(4).to_string(index=False))

    # --- (b) y (c) ------------------------------------------------------------
    print("\n--- (b) Distancia a la frontera y (c) escalones, para los 11 casos ---")
    dist, ladder = frontier_and_ladder(booster, Lva, cases, cut5)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    aud.to_csv(REPORT_DIR / "phase5c_refactor_audit.csv", index=False)
    elig.to_csv(REPORT_DIR / "phase5c_eligibility.csv", index=False)
    espacio_vars.to_csv(REPORT_DIR / "phase5c_action_space_vars.csv", index=False)
    tamanos.to_csv(REPORT_DIR / "phase5c_action_space.csv", index=False)
    sin_acc.to_csv(REPORT_DIR / "phase5c_no_actions.csv", index=False)
    dist.to_csv(REPORT_DIR / "phase5c_frontier_distance.csv", index=False)
    ladder.to_csv(REPORT_DIR / "phase5c_ladder.csv", index=False)

    print("\n=== (b) DISTANCIA A LA FRONTERA (probabilidad; deficit > 0 = NO se cruza) ===")
    print(dist[["caso", "escalon", "marcado_top5", "prob_orig", "reduccion_necesaria",
                "reduccion_maxima_alcanzable", "deficit", "cruza",
                "todo_cambio_empeora"]].round(5).to_string(index=False))
    print("\n=== (c) ESCALONES: en cuál aparece el primer contrafactual ===")
    print(ladder[["caso", "escalon", "marcado_top5", "n_candidatos", "primer_cf", "n_cf",
                  "exhaustiva"]].to_string(index=False))
    print("\nAVISO de lectura: en los casos con marcado_top5 = False la pregunta del "
          "contrafactual NO SE PLANTEA\ny su deficit negativo NO se puede leer como «sí "
          "cruza»: el modelo no los marca.")
    print("\nEscritos: phase5c_refactor_audit.csv, phase5c_eligibility.csv, "
          "phase5c_action_space_vars.csv,\nphase5c_action_space.csv, phase5c_no_actions.csv, "
          "phase5c_frontier_distance.csv, phase5c_ladder.csv")


if __name__ == "__main__":
    main()
