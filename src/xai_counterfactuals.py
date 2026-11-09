"""
Fase 3c — Contrafactuales: ¿qué cambio accionable bajaría a este paciente del umbral?

Modelo: el ACTUAL (el "antes", sin corregir). Datos: solo val; test no se usa.

Qué es y qué NO es un contrafactual aquí
    Dice qué cambia la PUNTUACIÓN del modelo, no el riesgo real del paciente. Los datos
    son observacionales: si "insulina Steady" puntúa menos que "insulina Down", puede
    ser porque esos pacientes estaban menos enfermos. Todo lo que sale de este script
    es una hipótesis para revisión clínica, nunca una recomendación.

Dos métodos, y por qué los dos
    - DiCE (Mothilal et al., 2020, "Explaining Machine Learning Classifiers through
      Diverse Counterfactual Explanations"), method="random": no necesita gradientes,
      así que sirve para árboles. Es el método pedido en el plan.
    - Enumeración exhaustiva de todas las combinaciones de hasta 2 cambios. El espacio
      es pequeño (de media 1,2 fármacos por paciente marcado). DiCE muestrea al azar:
      si no encuentra nada, no prueba que no exista. La enumeración sí, y además audita
      a DiCE. Las cifras por población salen de aquí porque son deterministas.

Reglas de lo que se puede cambiar (plan de la Fase 3, sección 3)
    - Inmutables o ya ocurridas: todo lo demás, fuera del espacio de búsqueda.
    - discharge_disposition_id: solo hacia 1 (casa), 6 (casa con atención domiciliaria)
      o 3 (centro de enfermería). CONGELADO si el destino es 22 o 5: la auditoría
      (audit_discharge_leakage.py) indica que parte de su efecto viene de cómo se define
      la etiqueta. Quien sale a rehabilitación suele tener un siguiente encuentro
      planificado, y "mandarlo a casa" bajaría la puntuación por ese artefacto.
    - Fármacos: solo los que ya toma, entre Steady/Up/Down. Empezar uno nuevo o
      suspenderlo queda fuera: no vemos contraindicaciones.
    - A1Cresult y max_glu_serum: congeladas. Se puede pedir la prueba, no elegir su
      resultado. Para NotMeasured se puntúan aparte todos los resultados y se da el rango.
    - change y diabetesMed: derivadas, nunca se eligen (ver derive_change).

Límite conocido: el umbral se calcula con las puntuaciones originales de val. Si muchos
pacientes cambiaran a la vez, el corte del top-k% también se movería.
"""

import contextlib
import io
import itertools
import random
import warnings

import dice_ml
import numpy as np
import pandas as pd
from raiutils.exceptions import UserConfigValidationException

import data as D
import preprocessing as P
import xai_shap as XS

TOP_KS = (0.05, 0.10, 0.20)     # el k lo decide el gestor; aquí no se elige
ELIGIBLE_K = 0.10               # el plan genera casos individuales solo para el top 10%
MAX_CHANGES = 2
DESTINATIONS = ["1", "6", "3"]
SUSPECT_CODES = {"22", "5"}     # destinos señalados por la auditoría
MED_LEVELS = ["Steady", "Up", "Down"]
LAB_COLS = ["A1Cresult", "max_glu_serum"]
N_DICE_CFS = 5
N_DICE_AUDIT = 30               # pacientes del control de DiCE

# Columnas de xai_counterfactuals_cases.csv. Se fijan aquí para escribir la cabecera
# incluso con 0 filas: un CSV vacío sin cabecera no distingue "no se calculó" de
# "se calculó y no hay ninguno", y lo segundo es el hallazgo de esta fase.
DETAIL_COLS = ["caso", "encounter_id", "prob", "destino", "top_k", "umbral", "metodo",
               "cambios", "n_cambios", "prob_cf", "toca_destino"]

# Caso para el ejemplo de cómo un contrafactual puede engañar. Fijado ANTES de generar
# nada: es el único caso de los 11 con código 22, el destino con V57 estable entre
# train y val (57,2% y 57,3%). No forma parte del análisis principal.
DEMO_ENCOUNTER = 206978460
DEMO_WARNING = "EJEMPLO DE CONTRAFACTUAL ENGAÑOSO: NO ES ANÁLISIS NI RECOMENDACIÓN"

REPORT_DIR = D.PROJECT_ROOT / "reports"


def thresholds(probs: np.ndarray) -> dict:
    """Puntuación del último marcado en cada top-k, igual que evaluation.top_k_table."""
    ordered = np.sort(probs)[::-1]
    return {k: ordered[int(np.ceil(k * len(probs))) - 1] for k in TOP_KS}


def check_derived_rules(Ltr: pd.DataFrame) -> dict:
    """Comprueba en train la relación entre fármacos y variables derivadas.

    El plan suponía que change se podía recalcular a partir de los fármacos. Solo se
    cumple en un sentido: Up/Down implica Ch, pero hay Ch sin ningún Up/Down (por
    ejemplo, cambios que estas columnas no registran). De ahí derive_change.
    """
    meds = Ltr[P.MED_COLS].astype(str)
    has_updown = meds.isin(["Up", "Down"]).any(axis=1)
    taking = meds.ne("No").any(axis=1)
    change, on_meds = Ltr["change"].astype(str), Ltr["diabetesMed"].astype(str)
    assert not (has_updown & (change == "No")).any(), "Up/Down con change=No"
    assert ((on_meds == "Yes") == taking).all(), "diabetesMed no coincide con los fármacos"
    return {"filas_train": len(Ltr), "Ch_sin_UpDown": int((~has_updown & (change == "Ch")).sum())}


class Scorer:
    """Puntúa candidatos de UN paciente, partiendo siempre de su fila original.

    Todo lo que no se pasa en `overrides` se copia de la fila original, así que una
    variable inmutable no puede cambiar aunque DiCE devolviera otro valor.
    """

    def __init__(self, booster, row: pd.DataFrame):
        self.booster, self.row = booster, row
        self.taken = [m for m in P.MED_COLS if str(row[m].iloc[0]) != "No"]
        self.orig_change = str(row["change"].iloc[0])
        self.orig_updown = row[self.taken].astype(str).isin(["Up", "Down"]).any(axis=1).iloc[0]
        # Sin cambios, debe salir exactamente la puntuación original del modelo.
        assert np.isclose(self.score({}, 1)[0], booster.predict(row)[0]), "Scorer no reproduce"

    def frame(self, overrides: dict, n: int) -> pd.DataFrame:
        cand = self.row.iloc[np.zeros(n, dtype=int)].copy()
        for col, values in overrides.items():
            cand[col] = pd.Categorical(values, dtype=self.row[col].dtype)
            assert cand[col].notna().all(), f"nivel que el modelo no conoce en {col}"
        return cand

    def derive_change(self, cand: pd.DataFrame):
        """Devuelve (change si se mantiene, máscara de filas donde es ambiguo).

        - Algún Up/Down en el candidato: Ch (se cumple en todo train).
        - Sin Up/Down y original No: No.
        - Sin Up/Down y original Ch sin Up/Down: Ch; ese cambio no lo explican los fármacos.
        - El candidato elimina TODOS los Up/Down del original: AMBIGUO, porque no
          sabemos si había algo más detrás de ese Ch. Se puntúan las dos opciones.
        """
        has_updown = cand[self.taken].astype(str).isin(["Up", "Down"]).any(axis=1).to_numpy()
        change = np.where(has_updown | (self.orig_change == "Ch"), "Ch", "No")
        return change, (~has_updown) & bool(self.orig_updown)

    def score(self, overrides: dict, n: int) -> np.ndarray:
        """Probabilidad en el PEOR caso: si change es ambiguo, la mayor de las dos."""
        cand = self.frame(overrides, n)
        change, ambiguous = self.derive_change(cand)
        cand["change"] = pd.Categorical(change, dtype=self.row["change"].dtype)
        probs = self.booster.predict(cand)
        if ambiguous.any():
            cand["change"] = pd.Categorical(np.where(ambiguous, "No", change),
                                            dtype=self.row["change"].dtype)
            probs = np.maximum(probs, self.booster.predict(cand))
        return probs


def action_space(row: pd.DataFrame, freeze_suspect: bool = True) -> dict:
    """{variable: valores alternativos}. Vacío si no hay nada accionable.

    freeze_suspect=False solo lo usa el ejemplo engañoso de demo_misleading().
    """
    space = {}
    dest = str(row["discharge_disposition_id"].iloc[0])
    if not (freeze_suspect and dest in SUSPECT_CODES):
        space["discharge_disposition_id"] = [d for d in DESTINATIONS if d != dest]
    for med in P.MED_COLS:
        current = str(row[med].iloc[0])
        # NaN: nivel de val que no existe en train (el modelo no conoce su estado), no se toca.
        if current == "No" or pd.isna(row[med].iloc[0]):
            continue
        # Solo niveles que existen en train: un nivel no visto no significa nada para el modelo.
        known = set(row[med].cat.categories)
        options = [lvl for lvl in MED_LEVELS if lvl != current and lvl in known]
        if options:
            space[med] = options
    return {var: opts for var, opts in space.items() if opts}


def enumerate_changes(space: dict) -> list:
    """Todas las combinaciones de 1 a MAX_CHANGES variables, cada una con sus valores."""
    changes = []
    for size in range(1, MAX_CHANGES + 1):
        for variables in itertools.combinations(space, size):
            for values in itertools.product(*(space[v] for v in variables)):
                changes.append(dict(zip(variables, values)))
    return changes


def score_changes(scorer: Scorer, changes: list) -> np.ndarray:
    variables = {v for c in changes for v in c}
    original = {v: str(scorer.row[v].iloc[0]) for v in variables}
    overrides = {v: [c.get(v, original[v]) for c in changes] for v in variables}
    return scorer.score(overrides, len(changes))


def describe(changes: dict, row: pd.DataFrame) -> str:
    return "; ".join(f"{v}: {row[v].iloc[0]} → {new}" for v, new in changes.items())


class ThresholdModel:
    """Envoltorio para DiCE: el umbral del top-k pasa a ser 0,5.

    DiCE decide la clase comparando con 0,5, y aquí casi nadie llega a esa puntuación.
    La transformación es lineal a trozos y monótona: no cambia el orden, solo sitúa el
    umbral donde DiCE lo espera. Cada resultado se revalida luego con la puntuación real.
    """

    def __init__(self, scorer: Scorer, threshold: float, varied: list):
        self.scorer, self.threshold, self.varied = scorer, threshold, varied

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        overrides = {v: X[v].astype(str).to_numpy() for v in self.varied}
        p = self.scorer.score(overrides, len(X))
        t = self.threshold
        rescaled = np.where(p < t, 0.5 * p / t, 0.5 + 0.5 * (p - t) / (1 - t))
        return np.column_stack([1 - rescaled, rescaled])


def make_dice_data(Ltr: pd.DataFrame, ytr: np.ndarray) -> dice_ml.Data:
    """DiCE trabaja con texto; las categorías vuelven a su tipo dentro de Scorer."""
    frame = Ltr.copy()
    for col in P.CAT_COLS:
        frame[col] = frame[col].astype(str)
    frame[D.TARGET] = ytr
    return dice_ml.Data(dataframe=frame, continuous_features=P.NUM_COLS + P.COUNT_COLS,
                        outcome_name=D.TARGET)


def run_dice(dice_data, scorer: Scorer, space: dict, threshold: float) -> tuple:
    """Devuelve (lista de cambios propuestos, nº de filas devueltas por DiCE).

    Reproducibilidad: `random_seed` por sí solo NO basta. En dice_ml 0.12,
    dice_random.py solo lo propaga a get_samples, mientras que dos pasos más usan el
    estado GLOBAL de numpy sin semilla: `np.random.choice` para elegir qué variables
    varía cada muestra (línea 114) y `cfs_df.sample(total_CFs)` para quedarse con
    total_CFs de los candidatos (línea 135; pandas sin random_state también usa
    np.random global). Medido antes de arreglarlo: dos llamadas sobre el mismo paciente
    devolvían conjuntos distintos, incluso dentro del mismo proceso y con
    PYTHONHASHSEED=0 (así que no era la aleatorización del hash de cadenas). Fijar el
    estado global justo antes de cada llamada lo hace reproducible entre llamadas y
    entre procesos, sin cambiar el método (sigue siendo method="random").
    """
    if not space:
        return [], 0
    varied = list(space)
    wrapper = ThresholdModel(scorer, threshold, varied)
    explainer = dice_ml.Dice(dice_data, dice_ml.Model(model=wrapper, backend="sklearn"),
                             method="random")
    query = scorer.row.copy()
    for col in P.CAT_COLS:
        query[col] = query[col].astype(str)
    # Semillas globales antes de CADA llamada: ver el aviso del docstring. Se fijan
    # aquí, y no una vez al principio, para que el resultado de un paciente no dependa
    # de cuántas llamadas se hayan hecho antes.
    np.random.seed(D.SEED)
    random.seed(D.SEED)
    try:
        # DiCE escribe barras de progreso y avisos por consola aunque verbose=False.
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            result = explainer.generate_counterfactuals(
                query, total_CFs=N_DICE_CFS, desired_class=0, features_to_vary=varied,
                permitted_range=space, random_seed=D.SEED, verbose=False)
    except UserConfigValidationException:
        return [], 0  # DiCE avisa así cuando no encuentra nada
    cfs = result.cf_examples_list[0].final_cfs_df
    if cfs is None or cfs.empty:
        return [], 0
    proposals = []
    for _, cf in cfs.iterrows():
        changed = {v: str(cf[v]) for v in varied if str(cf[v]) != str(scorer.row[v].iloc[0])}
        if changed and changed not in proposals:
            proposals.append(changed)
    return proposals, len(cfs)


def explain_patient(booster, dice_data, row: pd.DataFrame, threshold: float,
                    freeze_suspect: bool = True) -> dict:
    """Contrafactuales de un paciente para un umbral, con los dos métodos."""
    scorer = Scorer(booster, row)
    space = action_space(row, freeze_suspect)
    changes = enumerate_changes(space)
    rows = []
    enumerated = set()
    closest = {"prob_min_alcanzable": np.nan, "cambio_mas_cercano": None}
    if changes:
        probs = score_changes(scorer, changes)
        # Lo más bajo que se puede llegar con las reglas, cruce o no el umbral.
        closest = {"prob_min_alcanzable": probs.min(),
                   "cambio_mas_cercano": describe(changes[int(probs.argmin())], row)}
        for change, prob in zip(changes, probs):
            enumerated.add(frozenset(change.items()))
            if prob < threshold:
                rows.append({"metodo": "enumeracion", "cambios": describe(change, row),
                             "n_cambios": len(change), "prob_cf": prob,
                             "toca_destino": "discharge_disposition_id" in change})

    proposals, n_returned = run_dice(dice_data, scorer, space, threshold)
    # Validación posterior: DiCE solo cuenta si cumple el límite y cruza con la puntuación real.
    too_many = [c for c in proposals if len(c) > MAX_CHANGES]
    within = [c for c in proposals if len(c) <= MAX_CHANGES]
    dice_probs = score_changes(scorer, within) if within else np.array([])
    valid = 0
    for change, prob in zip(within, dice_probs):
        assert frozenset(change.items()) in enumerated, "DiCE se salió del espacio permitido"
        if prob < threshold:
            valid += 1
            rows.append({"metodo": "dice", "cambios": describe(change, row),
                         "n_cambios": len(change), "prob_cf": prob,
                         "toca_destino": "discharge_disposition_id" in change})
    return {"rows": rows, "space": space, "n_candidatos": len(changes), **closest,
            "dice_devueltos": n_returned, "dice_distintos": len(proposals),
            "dice_rechazados_mas_de_2": len(too_many), "dice_validos": valid,
            "enum_validos": sum(r["metodo"] == "enumeracion" for r in rows)}


def lab_range(booster, row: pd.DataFrame) -> list:
    """Si una prueba no se hizo, puntúa todos sus resultados posibles.

    Sirve para ver cuánto dependería la puntuación de un resultado que no conocemos. No
    es un contrafactual. Además, A1Cresult no superó las sondas de ruido: su efecto no
    se interpreta.
    """
    scorer, out = Scorer(booster, row), []
    for lab in LAB_COLS:
        if str(row[lab].iloc[0]) != "NotMeasured":
            continue
        levels = [lvl for lvl in row[lab].cat.categories if lvl != "NotMeasured"]
        probs = scorer.score({lab: levels}, len(levels))
        out.append({"prueba": lab, "min": probs.min(), "nivel_min": levels[int(probs.argmin())],
                    "max": probs.max(), "nivel_max": levels[int(probs.argmax())]})
    return out


def population_summary(booster, L: pd.DataFrame, probs: pd.Series, cuts: dict):
    """Cobertura por enumeración sobre TODOS los marcados de val (solo reglas principales)."""
    records = []
    for encounter in probs.index[probs >= min(cuts.values())]:
        row = L.loc[[encounter]]
        dest = str(row["discharge_disposition_id"].iloc[0])
        space = action_space(row)
        changes = enumerate_changes(space)
        scored = score_changes(Scorer(booster, row), changes) if changes else np.array([])
        for k, cut in cuts.items():
            if probs.loc[encounter] < cut:
                continue
            valid = [(c, p) for c, p in zip(changes, scored) if p < cut]
            best = min(valid, key=lambda cp: (len(cp[0]), cp[1])) if valid else None
            records.append({
                "top_k": f"{k:.0%}",
                "grupo": "22/5 (destino congelado)" if dest in SUSPECT_CODES else "otros destinos",
                "sin_acciones": not changes,
                "con_cf": bool(valid),
                "con_cf_sin_destino": any("discharge_disposition_id" not in c for c, _ in valid),
                "bajada": probs.loc[encounter] - best[1] if best else np.nan,
                "palanca": ("; ".join(f"{v}→{x}" for v, x in best[0].items())) if best else None,
            })
    rec = pd.DataFrame(records)
    summary = rec.groupby(["top_k", "grupo"], sort=False).agg(
        n=("con_cf", "size"),
        pct_sin_acciones=("sin_acciones", lambda s: 100 * s.mean()),
        pct_con_cf=("con_cf", lambda s: 100 * s.mean()),
        pct_con_cf_sin_destino=("con_cf_sin_destino", lambda s: 100 * s.mean()),
        mediana_bajada=("bajada", "median")).reset_index()
    levers = (rec.dropna(subset=["palanca"]).groupby(["top_k", "grupo", "palanca"], sort=False)
              .size().rename("n").reset_index().sort_values(["top_k", "grupo", "n"],
                                                            ascending=[True, True, False]))
    return summary, levers


def case_tables(booster, dice_data, L, cases, cuts):
    """Análisis principal: casos fijados del top 10%, en cada umbral en que están marcados."""
    detail, status, labs = [], [], []
    eligible = cases[cases["prob"] >= cuts[ELIGIBLE_K]]
    for _, case in eligible.iterrows():
        encounter = int(case["encounter_id"])
        row = L.loc[[encounter]]
        meta = {"caso": case["caso"], "encounter_id": encounter, "prob": case["prob"],
                "destino": str(row["discharge_disposition_id"].iloc[0])}
        labs += [{**meta, **entry} for entry in lab_range(booster, row)]
        for k, cut in cuts.items():
            if case["prob"] < cut:
                status.append({**meta, "top_k": f"{k:.0%}", "umbral": cut, "marcado": False})
                continue
            result = explain_patient(booster, dice_data, row, cut)
            detail += [{**meta, "top_k": f"{k:.0%}", "umbral": cut, **r} for r in result["rows"]]
            status.append({**meta, "top_k": f"{k:.0%}", "umbral": cut, "marcado": True,
                           "variables_accionables": ", ".join(result["space"]),
                           **{key: result[key] for key in ("n_candidatos", "enum_validos",
                              "dice_devueltos", "dice_distintos", "dice_rechazados_mas_de_2",
                              "dice_validos", "prob_min_alcanzable", "cambio_mas_cercano")}})
    return (pd.DataFrame(detail, columns=DETAIL_COLS), pd.DataFrame(status),
            pd.DataFrame(labs))


def audit_dice(booster, dice_data, L, probs, cut) -> pd.DataFrame:
    """Control de DiCE: ¿encuentra contrafactual cuando la enumeración prueba que existe?

    Sin esto, que DiCE devuelva "nada" en un caso no significaría mucho. Pacientes
    elegidos al azar con semilla, y sin mirar el resultado de DiCE, entre los marcados
    del top 10% que sí tienen contrafactual.
    """
    rng = np.random.default_rng(D.SEED)
    flagged = probs.index[probs >= cut].to_numpy()
    rows = []
    for encounter in rng.permutation(flagged):
        row = L.loc[[encounter]]
        changes = enumerate_changes(action_space(row))
        if not changes or score_changes(Scorer(booster, row), changes).min() >= cut:
            continue
        result = explain_patient(booster, dice_data, row, cut)
        rows.append({"encounter_id": encounter, "enum_validos": result["enum_validos"],
                     "dice_distintos": result["dice_distintos"],
                     "dice_rechazados_mas_de_2": result["dice_rechazados_mas_de_2"],
                     "dice_validos": result["dice_validos"]})
        if len(rows) == N_DICE_AUDIT:
            break
    return pd.DataFrame(rows)


def demo_misleading(booster, dice_data, L, cases, cuts) -> pd.DataFrame:
    """EJEMPLO, no análisis: mismo paciente con y sin congelar el destino.

    La única regla que se levanta es la congelación del código 22/5; el resto se mantiene.
    Sus filas se guardan en un fichero aparte y no entran en ningún resumen.
    """
    row = L.loc[[DEMO_ENCOUNTER]]
    prob = float(cases.set_index("encounter_id").loc[DEMO_ENCOUNTER, "prob"])
    out = []
    for k, cut in cuts.items():
        if prob < cut:
            continue
        for label, freeze in (("responsable (destino congelado)", True),
                              ("SIN restricción (engañoso)", False)):
            result = explain_patient(booster, dice_data, row, cut, freeze_suspect=freeze)
            if not result["rows"]:
                out.append({"top_k": f"{k:.0%}", "umbral": cut, "version": label,
                            "metodo": "ninguno", "cambios": "ningún contrafactual",
                            "n_cambios": 0, "prob_cf": np.nan, "toca_destino": False})
            out += [{"top_k": f"{k:.0%}", "umbral": cut, "version": label, **r}
                    for r in result["rows"]]
    demo = pd.DataFrame(out)
    demo.insert(0, "aviso", DEMO_WARNING)
    demo.insert(1, "encounter_id", DEMO_ENCOUNTER)
    demo.insert(2, "prob_original", prob)
    return demo



# ---------------------------------------------------------------------------
# Diagnóstico de la Fase 3c: ¿"cero contrafactuales" es un bug o un resultado?
#
# Criterios prerregistrados (fijados ANTES de ver la salida, no se tocan después):
#   BUG        si falla (a), si falla (b) (ningún cambio mueve la puntuación, o se
#              pierden los dtypes/categorías) o si falla (d).
#   HALLAZGO   si (a), (b) y (d) pasan y (c) muestra que el mínimo alcanzable de cada
#              caso sigue por encima de cada umbral.
#   INCONSIST. si (c) encuentra un caso que baja del umbral pero la enumeración
#              principal o DiCE dieron 0.
# ---------------------------------------------------------------------------
def show(frame, empty_msg: str, cols=None, sort=None, decimals=4):
    """Imprime una tabla, o un mensaje explícito si está vacía.

    Ordenar o seleccionar columnas de un DataFrame vacío lanza KeyError (pandas no
    sabe qué columnas tendría), que es lo que rompía el script cuando no salía ningún
    contrafactual. Aquí el caso vacío es un resultado, no un fallo.
    """
    if frame is None or frame.empty:
        print(empty_msg)
        return
    out = frame.sort_values(sort) if sort else frame
    if cols:
        out = out[[c for c in cols if c in out.columns]]
    print(out.round(decimals).to_string(index=False))


def diag_eligibility(cases: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    """Por qué solo algunos de los casos fijados llegan a tener contrafactual.

    Un contrafactual solo tiene sentido para alguien a quien el gestor marcaría: si el
    paciente ya está por debajo del umbral, no hay nada que bajar. El corte de
    elegibilidad es el top 10% (ELIGIBLE_K), que es lo que fijaba el plan.
    """
    cut = cuts[ELIGIBLE_K]
    out = cases.copy()
    out["elegible"] = out["prob"] >= cut
    dentro = "prob >= umbral top {:.0%} ({:.4f})".format(ELIGIBLE_K, cut)
    fuera = ("prob < umbral top {:.0%} ({:.4f}): el modelo no lo marca, "
             "no hay nada que bajar").format(ELIGIBLE_K, cut)
    out["motivo"] = np.where(out["elegible"], dentro, fuera)
    return out[["caso", "encounter_id", "prob", "reingresa", "elegible", "motivo"]]


def diag_reproduction(sk_model, booster, L: pd.DataFrame, eligible: pd.DataFrame,
                      cuts: dict, tol: float = 1e-9) -> pd.DataFrame:
    """(a) El Scorer sin cambios reproduce el pipeline original, y en la misma escala.

    Tres puntuaciones de la misma fila que deben coincidir:
      - sk_model.predict_proba(row)[:, 1]: el pipeline LightGBM original de train_lgbm.
      - booster.predict(row): el booster truncado a 122 árboles que usa todo el script.
      - Scorer.score({}, 1): el camino que recorren los contrafactuales, sin cambios.
    Si el tercero no coincide con el primero, nada de lo que venga después es válido.

    Escala: booster.predict() sin raw_score devuelve PROBABILIDAD, y los umbrales
    salen de thresholds(booster.predict(Lva)), así que umbral y puntuación están en la
    misma escala. Se imprime además el log-odds para dejar constancia de que no se
    mezclan las dos escalas en ninguna comparación.
    """
    rows = []
    for _, case in eligible.iterrows():
        encounter = int(case["encounter_id"])
        row = L.loc[[encounter]]
        p_pipe = float(sk_model.predict_proba(row)[:, 1][0])
        p_boost = float(booster.predict(row)[0])
        p_scorer = float(Scorer(booster, row).score({}, 1)[0])
        raw = float(booster.predict(row, raw_score=True)[0])
        rows.append({
            "caso": case["caso"], "encounter_id": encounter,
            "prob_pipeline": p_pipe, "prob_booster": p_boost, "prob_scorer": p_scorer,
            "log_odds": raw, "prob_csv_fase3a": float(case["prob"]),
            "dif_max": max(abs(p_scorer - p_pipe), abs(p_scorer - p_boost)),
        })
    diag = pd.DataFrame(rows)
    if diag.empty:
        print("(a) FALLA: no hay casos elegibles que comprobar.")
        return diag
    diag["ok"] = diag["dif_max"] <= tol
    umbrales = {"{:.0%}".format(k): round(v, 4) for k, v in cuts.items()}
    print("Tolerancia: {:g}. Umbrales (misma escala, PROBABILIDAD): {}".format(tol, umbrales))
    print("Escala: booster.predict() -> probabilidad; el log-odds se imprime solo como "
          "referencia y NUNCA se compara con el umbral.")
    show(diag, "sin casos elegibles", decimals=10)
    print("(a) {}: {}/{} casos reproducen el pipeline original.".format(
        "PASA" if diag["ok"].all() else "FALLA", int(diag["ok"].sum()), len(diag)))
    return diag


def frame_integrity(cand: pd.DataFrame, Ltr: pd.DataFrame, var: str, new_value: str) -> list:
    """Problemas del DataFrame que llega al modelo tras un cambio. Lista vacía = bien.

    Importa porque LightGBM trata en silencio un NaN o una categoría desconocida como
    missing: el cambio "no haría nada" sin lanzar ningún error, y leeríamos ese cero
    como si el lever no moviera la puntuación.
    """
    problems = []
    if list(cand.columns) != list(Ltr.columns):
        problems.append("cambia el orden o el conjunto de columnas")
    for col in Ltr.columns:
        if cand[col].dtype != Ltr[col].dtype:
            problems.append("dtype distinto en {}: {} vs {}".format(
                col, cand[col].dtype, Ltr[col].dtype))
        elif isinstance(Ltr[col].dtype, pd.CategoricalDtype) and \
                list(cand[col].cat.categories) != list(Ltr[col].cat.categories):
            problems.append("categories distintas en {}".format(col))
    if cand[var].isna().any():
        problems.append("{} quedó NaN (LightGBM lo leería como missing)".format(var))
    if str(cand[var].iloc[0]) != str(new_value):
        problems.append("{} no aplicó el valor: {!r} != {!r}".format(
            var, cand[var].iloc[0], new_value))
    if new_value not in set(Ltr[var].cat.categories):
        problems.append("{!r} no es una categoría vista en train".format(new_value))
    return problems


def diag_sensitivity(booster, Ltr: pd.DataFrame, L: pd.DataFrame,
                     eligible: pd.DataFrame) -> pd.DataFrame:
    """(b) Un lever a la vez: ¿mueve la puntuación, y llega intacto al modelo?

    La comprobación más importante. Si ningún cambio permitido mueve la puntuación, el
    "cero contrafactuales" sería un artefacto del montaje (overrides que no se aplican,
    categorías que se vuelven missing), no un límite del modelo.
    """
    rows = []
    for _, case in eligible.iterrows():
        encounter = int(case["encounter_id"])
        row = L.loc[[encounter]]
        scorer = Scorer(booster, row)
        base = float(scorer.score({}, 1)[0])
        space = action_space(row)
        if not space:
            rows.append({"caso": case["caso"], "encounter_id": encounter,
                         "variable": "(ninguna)", "valor_orig": None, "valor_nuevo": None,
                         "prob_orig": base, "prob_nueva": np.nan, "delta": np.nan,
                         "mueve": False, "frame_ok": True,
                         "problemas": "sin variables accionables"})
            continue
        for var, options in space.items():
            for new_value in options:
                cand = scorer.frame({var: [new_value]}, 1)
                problems = frame_integrity(cand, Ltr, var, new_value)
                prob = float(scorer.score({var: [new_value]}, 1)[0])
                rows.append({
                    "caso": case["caso"], "encounter_id": encounter, "variable": var,
                    "valor_orig": str(row[var].iloc[0]), "valor_nuevo": new_value,
                    "prob_orig": base, "prob_nueva": prob, "delta": prob - base,
                    "mueve": abs(prob - base) > 0,
                    "frame_ok": not problems, "problemas": "; ".join(problems) or "-",
                })
    diag = pd.DataFrame(rows)
    show(diag, "(b) FALLA: ningún caso elegible tiene variables accionables que probar",
         cols=["caso", "variable", "valor_orig", "valor_nuevo", "prob_orig", "prob_nueva",
               "delta", "mueve", "frame_ok", "problemas"], decimals=5)
    if diag.empty:
        return diag
    probados = diag[diag["variable"] != "(ninguna)"]
    if probados.empty:
        print("(b) FALLA: no hay ningún cambio que probar.")
        return diag
    por_caso = (probados.groupby("caso")["mueve"].agg(["sum", "size"])
                .rename(columns={"sum": "cambios_que_mueven", "size": "cambios_probados"}))
    por_caso["bajada_max"] = -probados.groupby("caso")["delta"].min()
    print("\nResumen por caso (bajada_max = mayor descenso con UN solo cambio):")
    print(por_caso.round(5).to_string())
    algun_mueve = bool(probados["mueve"].any())
    frames_ok = bool(probados["frame_ok"].all())
    casos_planos = sorted(por_caso.index[por_caso["cambios_que_mueven"] == 0])
    print("\n(b) {}: {}/{} cambios mueven la puntuación; dtypes y categories intactos "
          "en {}/{}.".format("PASA" if algun_mueve and frames_ok else "FALLA",
                             int(probados["mueve"].sum()), len(probados),
                             int(probados["frame_ok"].sum()), len(probados)))
    if casos_planos:
        print("    Casos donde NINGÚN cambio mueve la puntuación: {}".format(casos_planos))
    return diag


def diag_min_reachable(booster, L: pd.DataFrame, eligible: pd.DataFrame,
                       cuts: dict) -> pd.DataFrame:
    """(c) Mínimo alcanzable por enumeración exhaustiva, frente a cada umbral.

    Separa 1 cambio y hasta 2 porque al gestor no le cuesta lo mismo una cosa que dos.
    La distancia (mínimo - umbral) es lo que decide: si es positiva en los tres
    umbrales, no existe contrafactual dentro del espacio de acciones acordado, y el
    cero no es un fallo de búsqueda sino el resultado.
    """
    rows = []
    for _, case in eligible.iterrows():
        encounter = int(case["encounter_id"])
        row = L.loc[[encounter]]
        scorer = Scorer(booster, row)
        base = float(scorer.score({}, 1)[0])
        space = action_space(row)
        changes = enumerate_changes(space)
        entry = {"caso": case["caso"], "encounter_id": encounter,
                 "destino": str(row["discharge_disposition_id"].iloc[0]),
                 "palancas": ", ".join(space) or "(ninguna)",
                 "prob_orig": base, "n_candidatos": len(changes)}
        if not changes:
            entry.update({"min_1_cambio": np.nan, "min_hasta_2": np.nan,
                          "combinacion_minima": None})
        else:
            probs = score_changes(scorer, changes)
            unos = [p for c, p in zip(changes, probs) if len(c) == 1]
            best = int(probs.argmin())
            entry.update({"min_1_cambio": float(min(unos)) if unos else np.nan,
                          "min_hasta_2": float(probs.min()),
                          "combinacion_minima": describe(changes[best], row)})
        for k, cut in cuts.items():
            entry["marcado_{:.0%}".format(k)] = base >= cut
            # Distancia positiva = el mínimo sigue por encima del umbral (no hay CF).
            entry["dist_{:.0%}".format(k)] = entry["min_hasta_2"] - cut
        rows.append(entry)
    diag = pd.DataFrame(rows)
    if diag.empty:
        print("(c) sin casos elegibles")
        return diag
    dist_cols = ["dist_{:.0%}".format(k) for k in cuts]
    marc_cols = ["marcado_{:.0%}".format(k) for k in cuts]
    show(diag, "(c) sin casos elegibles",
         cols=["caso", "destino", "palancas", "prob_orig", "n_candidatos", "min_1_cambio",
               "min_hasta_2"] + dist_cols + marc_cols, decimals=5)
    print("\nCombinación que consigue el mínimo:")
    for _, r in diag.iterrows():
        print("  {}: {}".format(r["caso"], r["combinacion_minima"] or "(no hay ninguna)"))
    # Cruza = está marcado en ese umbral Y su mínimo alcanzable baja del umbral.
    cruza = []
    for _, r in diag.iterrows():
        for k in cuts:
            if r["marcado_{:.0%}".format(k)] and r["dist_{:.0%}".format(k)] < 0:
                cruza.append((r["caso"], "{:.0%}".format(k), r["dist_{:.0%}".format(k)]))
    if cruza:
        print("\n(c) Casos que SÍ bajan del umbral según la enumeración: {}".format(cruza))
        print("    Si la enumeración principal o DiCE dieron 0 para alguno de estos, es "
              "INCONSISTENCIA (bug de filtrado o de validación).")
    else:
        print("\n(c) Ningún caso elegible baja de ningún umbral en que esté marcado: el "
              "mínimo alcanzable queda siempre por encima del corte.")
    return diag


def diag_dice_control(booster, dice_data, L: pd.DataFrame, probs: pd.Series, cut: float,
                      n_cases: int = 3) -> pd.DataFrame:
    """(d) Control positivo de DiCE sobre casos con contrafactual DEMOSTRADO.

    Partiendo de pacientes reales marcados: se eligen al azar con semilla y se queda
    solo con aquellos para los que la enumeración PRUEBA que existe un cambio concreto
    que los baja del umbral. En esos, DiCE debe encontrar al menos uno; si devolviera
    cero, el "cero" de los casos principales no sería informativo.
    """
    rng = np.random.default_rng(D.SEED)
    flagged = probs.index[probs >= cut].to_numpy()
    rows = []
    for encounter in rng.permutation(flagged):
        row = L.loc[[encounter]]
        scorer = Scorer(booster, row)
        changes = enumerate_changes(action_space(row))
        if not changes:
            continue
        scored = score_changes(scorer, changes)
        if scored.min() >= cut:
            continue
        best = int(scored.argmin())
        result = explain_patient(booster, dice_data, row, cut)
        rows.append({
            "encounter_id": int(encounter), "prob_orig": float(probs.loc[encounter]),
            "umbral": cut, "cambio_probado": describe(changes[best], row),
            "prob_probada": float(scored.min()), "enum_validos": result["enum_validos"],
            "dice_devueltos": result["dice_devueltos"],
            "dice_distintos": result["dice_distintos"],
            "dice_validos": result["dice_validos"],
            "dice_encuentra": result["dice_validos"] > 0,
        })
        if len(rows) == n_cases:
            break
    diag = pd.DataFrame(rows)
    show(diag, "(d) FALLA: la enumeración no encontró ningún paciente marcado con "
               "contrafactual demostrado, así que no se puede controlar a DiCE",
         decimals=5)
    if not diag.empty:
        ok = bool(diag["dice_encuentra"].all())
        print("(d) {}: DiCE encuentra contrafactual válido en {}/{} controles positivos; "
              "la enumeración en {}/{}.".format(
                  "PASA" if ok else "FALLA", int(diag["dice_encuentra"].sum()), len(diag),
                  int((diag["enum_validos"] > 0).sum()), len(diag)))
    return diag


def run_diagnostics(model, booster, L, Ltr, probs, cases, cuts, dice_data) -> dict:
    """Los cuatro diagnósticos, en orden, antes del análisis principal."""
    print("\n" + "=" * 78)
    print("DIAGNÓSTICO DE LA FASE 3c (criterios prerregistrados, ver el docstring)")
    print("=" * 78)

    print("\n--- (0) Elegibilidad: de los casos fijados en la Fase 3a, quiénes llegan a "
          "tener contrafactual ---")
    elig_table = diag_eligibility(cases, cuts)
    show(elig_table, "no hay casos en xai_local_cases.csv", decimals=4)
    eligible = cases[cases["prob"] >= cuts[ELIGIBLE_K]]
    print("\nElegibles: {} de {}. Descartados: {}.".format(
        len(eligible), len(cases), len(cases) - len(eligible)))
    for _, r in elig_table[~elig_table["elegible"]].iterrows():
        print("  - {} (encounter {}, prob {:.4f}): {}".format(
            r["caso"], int(r["encounter_id"]), r["prob"], r["motivo"]))

    print("\n--- (a) Reproducción: Scorer vs pipeline original ---")
    rep = diag_reproduction(model["sk_model"], booster, L, eligible, cuts)

    print("\n--- (b) Sensibilidad: un lever a la vez (la comprobación clave) ---")
    sens = diag_sensitivity(booster, Ltr, L, eligible)

    print("\n--- (c) Mínimo alcanzable por enumeración exhaustiva (1 y 2 cambios) ---")
    minr = diag_min_reachable(booster, L, eligible, cuts)

    print("\n--- (d) Control positivo de DiCE ---")
    dctl = diag_dice_control(booster, dice_data, L, probs, cuts[ELIGIBLE_K])

    return {"elegibilidad": elig_table, "reproduccion": rep, "sensibilidad": sens,
            "minimo_alcanzable": minr, "control_dice": dctl}


def main():
    pd.set_option("display.width", 200, "display.max_colwidth", 80, "display.max_rows", 200)
    model = XS.load_final_model()
    booster, Ltr, L = model["booster"], model["Ltr"], model["Lva"]
    probs = pd.Series(booster.predict(L), index=L.index)
    cuts = thresholds(probs.to_numpy())
    print("Umbrales en val:", {f"{k:.0%}": round(v, 4) for k, v in cuts.items()})
    print("Regla de derivadas en train:", check_derived_rules(Ltr))

    cases = pd.read_csv(REPORT_DIR / "xai_local_cases.csv")
    dice_data = make_dice_data(Ltr, model["ytr"])

    # Diagnóstico primero: si (a), (b) o (d) fallan, el análisis principal no vale nada.
    diags = run_diagnostics(model, booster, L, Ltr, probs, cases, cuts, dice_data)
    for name, frame in diags.items():
        frame.to_csv(REPORT_DIR / f"xai_counterfactuals_diag_{name}.csv", index=False)

    detail, status, labs = case_tables(booster, dice_data, L, cases, cuts)
    summary, levers = population_summary(booster, L, probs, cuts)
    demo = demo_misleading(booster, dice_data, L, cases, cuts)
    dice_audit = audit_dice(booster, dice_data, L, probs, cuts[ELIGIBLE_K])

    dice_audit.to_csv(REPORT_DIR / "xai_counterfactuals_dice_audit.csv", index=False)
    detail.to_csv(REPORT_DIR / "xai_counterfactuals_cases.csv", index=False)
    status.to_csv(REPORT_DIR / "xai_counterfactuals_status.csv", index=False)
    labs.to_csv(REPORT_DIR / "xai_counterfactuals_lab_range.csv", index=False)
    summary.to_csv(REPORT_DIR / "xai_counterfactuals_population.csv", index=False)
    levers.to_csv(REPORT_DIR / "xai_counterfactuals_levers.csv", index=False)
    demo.to_csv(REPORT_DIR / "xai_counterfactuals_DEMO_ENGANOSO.csv", index=False)

    print("\n=== ANÁLISIS PRINCIPAL: estado por caso y umbral ===")
    show(status, "0 casos elegibles: ningún caso fijado llega al top 10% de val")
    print("\n=== ANÁLISIS PRINCIPAL: contrafactuales (hipótesis, no recomendaciones) ===")
    show(detail, "0 contrafactuales para los casos elegibles: ninguno baja de ningún "
                 "umbral con el espacio de acciones fijado (ver el diagnóstico (c)).",
         cols=["caso", "destino", "prob", "top_k", "umbral", "metodo", "cambios",
               "n_cambios", "prob_cf", "toca_destino"],
         sort=["caso", "top_k", "metodo", "n_cambios", "prob_cf"])
    print("\n=== CONTROL DE DICE: pacientes con contrafactual probado por enumeración ===")
    if dice_audit.empty:
        print("0 pacientes: la enumeración no encontró ningún marcado con contrafactual "
              "demostrado, así que no hay con qué controlar a DiCE.")
    else:
        print(f"Pacientes: {len(dice_audit)} | DiCE encuentra alguno válido: "
              f"{(dice_audit['dice_validos'] > 0).sum()} | "
              f"propuestas con más de 2 cambios: "
              f"{dice_audit['dice_rechazados_mas_de_2'].sum()} | "
              f"propuestas distintas: {dice_audit['dice_distintos'].sum()}, "
              f"válidas: {dice_audit['dice_validos'].sum()}")
    print("\n=== Rango por resultado de pruebas no realizadas (no es contrafactual) ===")
    show(labs, "0 filas: ningún caso elegible tiene pruebas sin realizar.")
    print("\n=== POBLACIÓN: todos los marcados de val, por enumeración ===")
    show(summary, "0 filas: ningún paciente marcado en val.", decimals=3)
    print("\nPalancas más frecuentes (mejor contrafactual de cada paciente):")
    if levers.empty:
        print("0 palancas: ningún paciente marcado de val tiene contrafactual.")
    else:
        print(levers.groupby(["top_k", "grupo"], sort=False).head(5).to_string(index=False))

    print("\n" + "!" * 78)
    print(f"!!! {DEMO_WARNING}")
    print("!" * 78)
    show(demo, "0 filas: el caso de la demo no está marcado en ningún umbral.",
         cols=["top_k", "umbral", "version", "metodo", "cambios", "n_cambios", "prob_cf",
               "toca_destino"])
    audit = pd.read_csv(REPORT_DIR / "audit_discharge_leakage.csv", index_col=0, dtype={0: str})
    audit.index = audit.index.astype(str)
    print("\nPor qué engaña (auditoría, train+val):")
    print(audit.loc[["22", "1", "6", "3"], ["pct_lt30", "pct_V57_o_prog", "tasa_sin_V57_o_prog"]]
          .round(1).to_string())


if __name__ == "__main__":
    main()
