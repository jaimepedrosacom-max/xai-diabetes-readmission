"""
Fase 4, punto 5 — Análisis de equidad, con el prerregistro de CLAUDE.md.

Cuatro análisis, todos con la ETIQUETA CORREGIDA y sobre el MODELO CORREGIDO (el
entregable). El modelo viejo aparece solo como columna descriptiva, sin veredictos.

  1. Composición del top-k: tasa de marcado por grupo y cociente frente a la referencia.
     Primario `payer_code` al k = 10%. Veredicto por la regla de los cuatro quintos
     [0,80; 1,25] sobre la ESTIMACIÓN PUNTUAL; el IC95 va al lado, descriptivo.
  2. Error por grupo: FNR al mismo k (positivos reales NO marcados / positivos del
     grupo), que es a quién se le niega el seguimiento. Primario `race`. Veredicto si
     |diferencia| >= 5 puntos Y Holm dentro de la familia.
  3. Detectabilidad de proxies: ROC-AUC al predecir el atributo desde el RESTO de
     features. Primario `race` (OVR macro + por clase); `gender` secundario. Umbral 0,65.
  4. Efecto distributivo de la PROPIA corrección: tasa de reetiquetado por grupo en
     train+val. Primario `race`. Mismo criterio que el FNR.

Mínimos del prerregistro: n >= 100 en val y >= 20 marcados al k primario para
interpretar. El filtro de >= 20 marcados se evalúa AQUÍ, no antes: `marcados_grupo` es el
numerador de la tasa de selección, así que consultarlo para fijar el alcance habría sido
ver el resultado primario antes de cerrar su prerregistro.

Bootstrap: 2.000 réplicas agrupadas por `patient_nbr` (los encuentros de un paciente no
son independientes), semilla D.SEED. Holm con alfa 0,05 dentro de cada familia basada en
IC; las familias por umbral no llevan corrección porque su veredicto no es por
significación.

TEST NO SE TOCA.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import data as D
import preprocessing as P
import target_correction as TC
from evaluate_test import CHOSEN, EXPECTED_TREES, EXPECTED_VAL_PR_AUC
from train_lgbm import LEARNING_RATE, MAX_TREES, PATIENCE, fit_lgbm

REPORT_DIR = D.PROJECT_ROOT / "reports"
N_REP = 2000
ALPHA = 0.05
K_PRIMARY = 0.10
KS = (0.05, 0.10, 0.20)
MIN_N = 100
MIN_FLAGGED = 20
FOUR_FIFTHS = (0.80, 1.25)
PROXY_RECOVERABLE = 0.65
PROXY_NOT = 0.55
# Referencias fijadas en el prerregistro (payer_code: MC, no Missing).
REFERENCE = {"race": "Caucasian", "gender": "Female", "payer_code": "MC"}
ATTRS = ["payer_code", "race", "gender"]
# Guardarraíl del modelo corregido, medido en la Fase 4 punto 2.
NEW_TREES = 183
NEW_VAL_PR_AUC = 0.1893


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def codes_of(series: pd.Series) -> tuple:
    """Códigos enteros de una categórica, con sus niveles en orden."""
    cat = pd.Categorical(series.astype(str))
    return cat.codes.astype(np.int64), list(cat.categories)


def patient_blocks(patients: np.ndarray) -> list:
    """Índices de filas de cada paciente, para remuestrear pacientes enteros."""
    uniq, inv = np.unique(patients, return_inverse=True)
    return [np.flatnonzero(inv == i) for i in range(len(uniq))]


def boot_indices(blocks: list, rng) -> np.ndarray:
    elegidos = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[i] for i in elegidos])


def holm(pvals: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Holm-Bonferroni: devuelve el vector de rechazos. m = len(pvals)."""
    m = len(pvals)
    orden = np.argsort(pvals)
    rechaza = np.zeros(m, dtype=bool)
    for i, j in enumerate(orden):
        if pvals[j] <= alpha / (m - i):
            rechaza[j] = True
        else:
            break                      # Holm para en el primer no rechazo
    return rechaza


def boot_pvalue(dist: np.ndarray, nulo: float = 0.0) -> float:
    """p de dos colas por bootstrap: 2*min(P(<=nulo), P(>=nulo)), con suelo 1/n."""
    dist = np.asarray(dist)
    n = len(dist)
    p = 2 * min((dist <= nulo).mean(), (dist >= nulo).mean())
    return max(min(p, 1.0), 1.0 / n)


def rate_by_group(codes: np.ndarray, n_groups: int, num_mask: np.ndarray,
                  den_mask: np.ndarray) -> np.ndarray:
    """num/den por grupo, con NaN donde el denominador es 0."""
    num = np.bincount(codes[num_mask], minlength=n_groups).astype(float)
    den = np.bincount(codes[den_mask], minlength=n_groups).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# ---------------------------------------------------------------------------
# 1. Composición del top-k
# ---------------------------------------------------------------------------
def composition(attr: str, series: pd.Series, probs: np.ndarray, probs_old: np.ndarray,
                patients: np.ndarray, k: float, primario: bool) -> pd.DataFrame:
    codes, levels = codes_of(series)
    ng = len(levels)
    ref = levels.index(REFERENCE[attr])
    todos = np.ones(len(codes), dtype=bool)

    def ratios(c, p, idx=None):
        top = p >= np.quantile(p, 1 - k)
        r = rate_by_group(c, ng, top, todos)
        return r, r / r[ref]

    tasa, coc = ratios(codes, probs)
    tasa_old, coc_old = ratios(codes, probs_old)
    marcados = np.bincount(codes[probs >= np.quantile(probs, 1 - k)], minlength=ng)
    n_grupo = np.bincount(codes, minlength=ng)

    # IC95 del cociente, bootstrap de pacientes.
    rng = np.random.default_rng(D.SEED)
    blocks = patient_blocks(patients)
    muestras = np.full((N_REP, ng), np.nan)
    for b in range(N_REP):
        idx = boot_indices(blocks, rng)
        c, p = codes[idx], probs[idx]
        top = p >= np.quantile(p, 1 - k)
        r = rate_by_group(c, ng, top, np.ones(len(idx), dtype=bool))
        muestras[b] = r / r[ref]
    lo, hi = np.nanpercentile(muestras, [2.5, 97.5], axis=0)

    elegible = (n_grupo >= MIN_N) & (marcados >= MIN_FLAGGED)
    fuera = ~((coc >= FOUR_FIFTHS[0]) & (coc <= FOUR_FIFTHS[1]))
    return pd.DataFrame({
        "atributo": attr, "primario": primario, "top_k": f"{k:.0%}", "grupo": levels,
        "es_referencia": np.arange(ng) == ref, "n_grupo": n_grupo, "marcados": marcados,
        "tasa_marcado_%": 100 * tasa, "cociente": coc,
        "cociente_IC95_bajo": lo, "cociente_IC95_alto": hi,
        "tasa_modelo_viejo_%": 100 * tasa_old, "cociente_modelo_viejo": coc_old,
        "elegible": elegible,
        # Veredicto solo donde el prerregistro lo permite: atributo primario, k primario,
        # grupo elegible y no referencia.
        "veredicto_disparidad": np.where(
            primario & (k == K_PRIMARY) & elegible & (np.arange(ng) != ref),
            np.where(fuera, "DISPARIDAD", "no"), "sin veredicto"),
    })


# ---------------------------------------------------------------------------
# 2. FNR por grupo  /  4. Efecto distributivo (misma maquinaria: diferencia + Holm)
# ---------------------------------------------------------------------------
def diff_analysis(attr: str, series: pd.Series, num_mask: np.ndarray,
                  den_mask: np.ndarray, patients: np.ndarray, primario: bool,
                  nombre: str, elegible_extra: np.ndarray = None,
                  num_old: np.ndarray = None) -> pd.DataFrame:
    """Tasa por grupo (num/den), diferencia frente a la referencia, IC, p y Holm."""
    codes, levels = codes_of(series)
    ng = len(levels)
    ref = levels.index(REFERENCE[attr])

    tasa = rate_by_group(codes, ng, num_mask, den_mask)
    dif = tasa - tasa[ref]
    den_n = np.bincount(codes[den_mask], minlength=ng)
    n_grupo = np.bincount(codes, minlength=ng)

    rng = np.random.default_rng(D.SEED)
    blocks = patient_blocks(patients)
    muestras = np.full((N_REP, ng), np.nan)
    for b in range(N_REP):
        idx = boot_indices(blocks, rng)
        c = codes[idx]
        r = rate_by_group(c, ng, num_mask[idx], den_mask[idx])
        muestras[b] = r - r[ref]
    lo, hi = np.nanpercentile(muestras, [2.5, 97.5], axis=0)

    elegible = (n_grupo >= MIN_N) & (den_n >= MIN_FLAGGED)
    if elegible_extra is not None:
        elegible &= elegible_extra
    contrasta = elegible & (np.arange(ng) != ref)

    # p-valores y Holm SOLO en la familia primaria: el prerregistro dice que los
    # atributos secundarios se reportan como exploratorios, sin veredicto y sin
    # corrección por comparaciones múltiples.
    p = np.full(ng, np.nan)
    rechaza = np.zeros(ng, dtype=bool)
    idx_fam = np.flatnonzero(contrasta) if primario else np.array([], dtype=int)
    for g in idx_fam:
        col = muestras[:, g]
        p[g] = boot_pvalue(col[~np.isnan(col)])
    if len(idx_fam):
        rechaza[idx_fam] = holm(p[idx_fam])   # m = grupos que contrastan de verdad

    out = pd.DataFrame({
        "analisis": nombre, "atributo": attr, "primario": primario, "grupo": levels,
        "es_referencia": np.arange(ng) == ref, "n_grupo": n_grupo,
        "denominador": den_n, "tasa_%": 100 * tasa,
        "dif_vs_ref_puntos": 100 * dif, "dif_IC95_bajo": 100 * lo, "dif_IC95_alto": 100 * hi,
        "p_bootstrap": p, "holm_rechaza": rechaza, "elegible": elegible,
        "m_familia": len(idx_fam) if primario else 0,
        "rol": "primario" if primario else "secundario/exploratorio",
    })
    if num_old is not None:
        tasa_old = rate_by_group(codes, ng, num_old, den_mask)
        out["tasa_modelo_viejo_%"] = 100 * tasa_old
    out["veredicto_disparidad"] = np.where(
        primario & contrasta,
        np.where((np.abs(100 * dif) >= 5) & rechaza, "DISPARIDAD", "no"),
        "sin veredicto")
    return out


# ---------------------------------------------------------------------------
# 3. Detectabilidad de proxies
# ---------------------------------------------------------------------------
def proxy_detectability(attr: str, train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """ROC-AUC al predecir el atributo desde el RESTO de features (val como holdout)."""
    feats = [c for c in P.FEATURES if c != attr]
    Ltr, Lva = D.to_categorical(train[feats], val[feats])
    # El atributo a predecir va como entero: LightGBM acepta texto en `fit` pero no en
    # el conjunto de evaluación del early stopping. Las clases se fijan con TRAIN.
    clases = sorted(train[attr].astype(str).unique())
    code = {c: i for i, c in enumerate(clases)}
    ytr = train[attr].astype(str).map(code).to_numpy()
    yva_txt = val[attr].astype(str)
    yva = yva_txt.map(code).to_numpy()

    multiclase = len(clases) > 2
    model = lgb.LGBMClassifier(
        n_estimators=MAX_TREES, learning_rate=LEARNING_RATE,
        objective="multiclass" if multiclase else "binary",
        num_class=len(clases) if multiclase else None,
        random_state=D.SEED, verbose=-1, **CHOSEN)
    model.fit(Ltr, ytr, eval_X=(Lva,), eval_y=(yva,), eval_names=["val"],
              callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    proba = model.predict_proba(Lva)

    if multiclase:
        macro = roc_auc_score(yva, proba, multi_class="ovr", average="macro",
                              labels=list(range(len(clases))))
        por_clase = {c: roc_auc_score((yva == i).astype(int), proba[:, i])
                     for i, c in enumerate(clases)}
    else:
        macro = roc_auc_score(yva, proba[:, 1])
        por_clase = {clases[1]: macro}
    return {"atributo": attr, "arboles": model.best_iteration_, "auc_ovr_macro": macro,
            "por_clase": por_clase,
            "lectura": ("recuperable" if macro >= PROXY_RECOVERABLE else
                        "no recuperable" if macro < PROXY_NOT else "zona gris")}


# ---------------------------------------------------------------------------
def main():
    pd.set_option("display.width", 220)
    train, val, test = P.load_model_data()
    flags = TC.planned_flags()
    ytr_old, yva_old = train[D.TARGET].to_numpy(), val[D.TARGET].to_numpy()
    ytr_new = TC.relabel(train, flags).to_numpy()
    yva_new = TC.relabel(val, flags).to_numpy()
    Ltr, Lva = P.to_lgbm_frames(train, val)

    # Modelo corregido (primario) y modelo viejo (columna descriptiva).
    new = fit_lgbm(Ltr, ytr_new, Lva, yva_new, **CHOSEN)
    assert new.best_iteration_ == NEW_TREES, new.best_iteration_
    assert round(new.best_score_["val"]["average_precision"], 4) == NEW_VAL_PR_AUC
    old = fit_lgbm(Ltr, ytr_old, Lva, yva_old, **CHOSEN)
    assert old.best_iteration_ == EXPECTED_TREES
    assert round(old.best_score_["val"]["average_precision"], 4) == EXPECTED_VAL_PR_AUC
    print(f"Guardarraíles OK: corregido {new.best_iteration_} árboles / "
          f"{new.best_score_['val']['average_precision']:.4f}; "
          f"viejo {old.best_iteration_} / {old.best_score_['val']['average_precision']:.4f}")

    p_new = new.predict_proba(Lva)[:, 1]
    p_old = old.predict_proba(Lva)[:, 1]
    pac_val = val[D.GROUP_COL].to_numpy()

    # --- 1. Composición del top-k --------------------------------------
    comp = pd.concat([composition(a, val[a], p_new, p_old, pac_val, k,
                                  primario=(a == "payer_code"))
                      for a in ATTRS for k in KS], ignore_index=True)
    print("\n1. Composición del top 10% (primario: payer_code):")
    print(comp[(comp["top_k"] == "10%") & (comp["atributo"] == "payer_code")]
          .drop(columns=["atributo", "primario", "top_k"]).round(4).to_string(index=False))

    # --- 2. FNR por grupo al k primario --------------------------------
    top_new = p_new >= np.quantile(p_new, 1 - K_PRIMARY)
    top_old = p_old >= np.quantile(p_old, 1 - K_PRIMARY)
    pos_new = yva_new == 1
    fnr = pd.concat([
        diff_analysis(a, val[a], num_mask=pos_new & ~top_new, den_mask=pos_new,
                      patients=pac_val, primario=(a == "race"), nombre="FNR top 10%",
                      num_old=pos_new & ~top_old)
        for a in ATTRS], ignore_index=True)
    print("\n2. FNR al top 10% (primario: race):")
    print(fnr[fnr["atributo"] == "race"].drop(columns=["analisis", "atributo", "primario"])
          .round(4).to_string(index=False))

    # --- 3. Proxies ------------------------------------------------------
    proxies = [proxy_detectability("race", train, val),
               proxy_detectability("gender", train, val)]
    prox_tab = pd.DataFrame([{k: v for k, v in d.items() if k != "por_clase"}
                             for d in proxies])
    prox_clases = pd.DataFrame([{"atributo": d["atributo"], "clase": c, "auc": v}
                                for d in proxies for c, v in d["por_clase"].items()])
    print("\n3. Detectabilidad de proxies (primario: race):")
    print(prox_tab.round(4).to_string(index=False))
    print(prox_clases.round(4).to_string(index=False))

    # --- 4. Efecto distributivo del reetiquetado (train+val) ------------
    both = pd.concat([train, val])
    y_old_both = np.concatenate([ytr_old, yva_old]).astype(bool)
    y_new_both = np.concatenate([ytr_new, yva_new]).astype(bool)
    cambiada = y_old_both & ~y_new_both
    dist = pd.concat([
        diff_analysis(a, both[a], num_mask=cambiada, den_mask=y_old_both,
                      patients=both[D.GROUP_COL].to_numpy(), primario=(a == "race"),
                      nombre="reetiquetado train+val")
        for a in ATTRS], ignore_index=True)
    print("\n4. Efecto distributivo del reetiquetado, train+val (primario: race):")
    print(dist[dist["atributo"] == "race"].drop(columns=["analisis", "atributo", "primario"])
          .round(4).to_string(index=False))

    print("\n4b. Efecto distributivo por payer_code (SECUNDARIO/exploratorio, sin veredicto):")
    print(dist[dist["atributo"] == "payer_code"]
          .drop(columns=["analisis", "atributo", "primario", "p_bootstrap",
                         "holm_rechaza", "m_familia", "veredicto_disparidad"])
          .round(4).to_string(index=False))

    # --- Recuento de contrastes y grupos excluidos ----------------------
    def recuento(tab, nombre, cond):
        hechos = int((tab["veredicto_disparidad"] != "sin veredicto").sum())
        excl = tab[cond & ~tab["elegible"]]
        return {"analisis": nombre, "contrastes_planificados": cond.sum() - 1,
                "contrastes_hechos": hechos,
                "grupos_excluidos": ", ".join(
                    f"{r.grupo} (marcados/den={r_den})" for r, r_den in
                    zip(excl.itertuples(), excl.get("marcados", excl.get("denominador")))
                ) or "ninguno"}

    comp10 = comp[(comp["top_k"] == "10%") & (comp["atributo"] == "payer_code")]
    res = pd.DataFrame([
        recuento(comp10, "composición payer_code 10%", comp10["atributo"] == "payer_code"),
        recuento(fnr[fnr["atributo"] == "race"], "FNR race", fnr[fnr["atributo"] == "race"]["atributo"] == "race"),
        recuento(dist[dist["atributo"] == "race"], "distributivo race", dist[dist["atributo"] == "race"]["atributo"] == "race"),
    ])
    print("\nRecuento de contrastes realizados:")
    print(res.to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comp.to_csv(REPORT_DIR / "phase4_fairness_composition.csv", index=False)
    fnr.to_csv(REPORT_DIR / "phase4_fairness_fnr.csv", index=False)
    prox_tab.to_csv(REPORT_DIR / "phase4_fairness_proxies.csv", index=False)
    prox_clases.to_csv(REPORT_DIR / "phase4_fairness_proxies_by_class.csv", index=False)
    dist.to_csv(REPORT_DIR / "phase4_fairness_relabel_distribution.csv", index=False)
    res.to_csv(REPORT_DIR / "phase4_fairness_contrast_count.csv", index=False)


if __name__ == "__main__":
    main()
