"""
Fase 4, puntos 2 y 3 — Aplica la corrección del target y mide la predicción prerregistrada.

Mide los TRES pasos que fija el prerregistro (ver CLAUDE.md, "Descomposición de la
predicción prerregistrada"), en val y por subgrupo del top 10%:
  (1) modelo viejo / etiqueta vieja
  (2) modelo viejo / etiqueta nueva  -> mismo ranking y mismo corte, solo cambia la
      verdad contra la que se puntúa. Aísla el efecto PURO de la etiqueta y no requiere
      entrenar nada.
  (3) modelo nuevo / etiqueta nueva  -> etiqueta MÁS reentrenamiento.
Y de ahí: `delta = (1) - (3)` (el del criterio original), `delta_etiqueta = (1) - (2)`,
`delta_modelo = (2) - (3)`, con `delta_etiqueta + delta_modelo = delta` como comprobación.
La brecha `precision_A - precision_B` se reporta en los tres pasos en puntos Y en
cociente, porque limpiar la etiqueta baja las dos precisiones a la vez.

Criterio prerregistrado, sin cambios: CONFIRMADA si `delta_A > delta_B`, REFUTADA si
`delta_A <= delta_B`. Se reporta salga como salga, junto a la descomposición y la brecha.

PR-AUC no es comparable entre los dos targets (su línea base es la prevalencia), así que
cada PR-AUC va con su prevalencia y su `lift = PR-AUC / prevalencia`, y el titular es el
lift. Igual en la CV de 5 folds dentro de train.

Test: SOLO se le aplica la regla de reetiquetado y se reporta el conteo de filas
cambiadas. Ninguna métrica de test se calcula aquí.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import data as D
import preprocessing as P
import target_correction as TC
from cv_model_comparison import LGBM_PARAMS, N_FOLDS
from evaluate_test import CHOSEN, EXPECTED_TREES, EXPECTED_VAL_PR_AUC
from train_lgbm import LEARNING_RATE, fit_lgbm
from xai_shap import compute_shap

REPORT_DIR = D.PROJECT_ROOT / "reports"
TOP_Q = 0.9
DISCHARGE = "discharge_disposition_id"
CONTAMINATED = {"22", "5"}


# ---------------------------------------------------------------------------
# Subgrupos del top 10%
# ---------------------------------------------------------------------------
def subgroup_table(probs: pd.Series, y: np.ndarray, inpatient: np.ndarray,
                   destino: np.ndarray, paso: str) -> pd.DataFrame:
    """Precisión en A y B dentro del top 10%, con el corte sobre estas puntuaciones."""
    corte = probs.quantile(TOP_Q)
    top = (probs >= corte).to_numpy()
    rows = []
    for nombre, mask in [("A", inpatient == 0), ("B", inpatient >= 1)]:
        sel = top & mask
        rows.append({"paso": paso, "subgrupo": nombre, "corte": corte,
                     "n_marcados": int(sel.sum()), "aciertos": int(y[sel].sum()),
                     "precision_%": 100 * y[sel].mean(),
                     "pct_destino_22_o_5": 100 * np.isin(destino[sel], list(CONTAMINATED)).mean()})
    return pd.DataFrame(rows)


def gap_rows(tabla: pd.DataFrame) -> pd.DataFrame:
    """Brecha A - B por paso, en puntos y en cociente."""
    rows = []
    for paso, g in tabla.groupby("paso", sort=False):
        pa = float(g.loc[g["subgrupo"] == "A", "precision_%"].iloc[0])
        pb = float(g.loc[g["subgrupo"] == "B", "precision_%"].iloc[0])
        rows.append({"paso": paso, "precision_A_%": pa, "precision_B_%": pb,
                     "brecha_puntos": pa - pb, "cociente_A_entre_B": pa / pb})
    return pd.DataFrame(rows)


def bootstrap_gap(probs: pd.Series, y: np.ndarray, inpatient: np.ndarray,
                  groups: np.ndarray, paso: str, n_rep: int = 2000) -> dict:
    """IC del 95% de la brecha A - B (puntos y cociente) por bootstrap de PACIENTES.

    Se remuestrean pacientes con reemplazo, no filas: los encuentros de un mismo
    paciente no son independientes, y el split ya se agrupa por `patient_nbr`.
    El corte del top 10% se recalcula DENTRO de cada réplica, así que el intervalo
    refleja el procedimiento completo y no un corte fijado con la muestra entera.
    """
    rng = np.random.default_rng(D.SEED)
    p = probs.to_numpy()
    # Filas de cada paciente, para poder reconstruir la réplica de golpe.
    uniq, inv = np.unique(groups, return_inverse=True)
    por_paciente = [np.flatnonzero(inv == i) for i in range(len(uniq))]

    difs, cocientes, descartadas = [], [], 0
    for _ in range(n_rep):
        elegidos = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([por_paciente[i] for i in elegidos])
        pr, yy, inp = p[idx], y[idx], inpatient[idx]
        top = pr >= np.quantile(pr, TOP_Q)
        a, b = top & (inp == 0), top & (inp >= 1)
        if not a.any() or not b.any() or yy[b].mean() == 0:
            descartadas += 1          # réplica sin subgrupo o con B sin aciertos
            continue
        pa, pb = 100 * yy[a].mean(), 100 * yy[b].mean()
        difs.append(pa - pb)
        cocientes.append(pa / pb)

    lo_d, hi_d = np.percentile(difs, [2.5, 97.5])
    lo_c, hi_c = np.percentile(cocientes, [2.5, 97.5])
    return {"paso": paso, "replicas_usadas": len(difs), "replicas_descartadas": descartadas,
            "brecha_puntos_IC95_bajo": lo_d, "brecha_puntos_IC95_alto": hi_d,
            "cociente_IC95_bajo": lo_c, "cociente_IC95_alto": hi_c}


def bootstrap_paired(p_old: pd.Series, p_new: pd.Series, y_old: np.ndarray,
                     y_new: np.ndarray, inpatient: np.ndarray, groups: np.ndarray,
                     n_rep: int = 2000) -> pd.DataFrame:
    """IC95 de los contrastes EMPAREJADOS, calculados dentro de cada réplica.

    Los pasos (1) y (2) son los mismos pacientes con el mismo ranking y el mismo corte:
    lo único que cambia es la etiqueta. Comparar sus IC por separado ignora ese
    emparejamiento; la diferencia hay que calcularla DENTRO de cada réplica, que es lo
    que cancela la variación entre pacientes.

    El paso (3) no se empareja con el (1) para la brecha (su ranking y sus n son otros),
    pero `delta_A - delta_B` sí se calcula por réplica porque es la cantidad del criterio
    prerregistrado y ambos modelos puntúan las MISMAS filas.
    """
    rng = np.random.default_rng(D.SEED)
    po, pn = p_old.to_numpy(), p_new.to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    por_paciente = [np.flatnonzero(inv == i) for i in range(len(uniq))]

    d_brecha, r_cociente, d_delta, descartadas = [], [], [], 0
    for _ in range(n_rep):
        elegidos = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([por_paciente[i] for i in elegidos])
        o, n_, yo, yn, inp = po[idx], pn[idx], y_old[idx], y_new[idx], inpatient[idx]
        top_o, top_n = o >= np.quantile(o, TOP_Q), n_ >= np.quantile(n_, TOP_Q)
        a_o, b_o = top_o & (inp == 0), top_o & (inp >= 1)
        a_n, b_n = top_n & (inp == 0), top_n & (inp >= 1)
        if not (a_o.any() and b_o.any() and a_n.any() and b_n.any()):
            descartadas += 1
            continue
        p1a, p1b = 100 * yo[a_o].mean(), 100 * yo[b_o].mean()
        p2a, p2b = 100 * yn[a_o].mean(), 100 * yn[b_o].mean()
        p3a, p3b = 100 * yn[a_n].mean(), 100 * yn[b_n].mean()
        if p1b == 0 or p2b == 0 or p1a == 0:
            descartadas += 1          # cocientes indefinidos
            continue
        d_brecha.append((p2a - p2b) - (p1a - p1b))
        r_cociente.append((p2a / p2b) / (p1a / p1b))
        d_delta.append((p1a - p3a) - (p1b - p3b))

    # Valores OBSERVADOS: los mismos contrastes sobre la muestra completa, sin remuestrear.
    def observados():
        top_o, top_n = po >= np.quantile(po, TOP_Q), pn >= np.quantile(pn, TOP_Q)
        a_o, b_o = top_o & (inpatient == 0), top_o & (inpatient >= 1)
        a_n, b_n = top_n & (inpatient == 0), top_n & (inpatient >= 1)
        p1a, p1b = 100 * y_old[a_o].mean(), 100 * y_old[b_o].mean()
        p2a, p2b = 100 * y_new[a_o].mean(), 100 * y_new[b_o].mean()
        p3a, p3b = 100 * y_new[a_n].mean(), 100 * y_new[b_n].mean()
        return [(p2a - p2b) - (p1a - p1b), (p2a / p2b) / (p1a / p1b),
                (p1a - p3a) - (p1b - p3b)]

    nombres = ["brecha(2) - brecha(1) [puntos]", "cociente(2) / cociente(1)",
               "delta_A - delta_B [puntos]"]
    nulos = [0.0, 1.0, 0.0]          # el nulo del cociente de cocientes es 1, no 0
    rows = []
    for nombre, v, obs, nulo in zip(nombres, [d_brecha, r_cociente, d_delta],
                                    observados(), nulos):
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append({"contraste": nombre, "observado": obs, "IC95_bajo": lo,
                     "IC95_alto": hi, "nulo": nulo,
                     "excluye_nulo": not (lo <= nulo <= hi),
                     "replicas_usadas": len(v), "replicas_descartadas": descartadas})
    return pd.DataFrame(rows)


def delta_rows(tabla: pd.DataFrame) -> pd.DataFrame:
    """delta, delta_etiqueta y delta_modelo por subgrupo, en puntos de precisión."""
    piv = tabla.pivot(index="subgrupo", columns="paso", values="precision_%")
    out = pd.DataFrame({
        "delta_etiqueta": piv["(1) viejo/vieja"] - piv["(2) viejo/nueva"],
        "delta_modelo": piv["(2) viejo/nueva"] - piv["(3) nuevo/nueva"],
        "delta_total": piv["(1) viejo/vieja"] - piv["(3) nuevo/nueva"],
    })
    # La descomposición es exacta por construcción; se comprueba igualmente.
    assert np.allclose(out["delta_etiqueta"] + out["delta_modelo"], out["delta_total"])
    return out.reset_index()


# ---------------------------------------------------------------------------
# Métricas globales y CV, siempre con prevalencia y lift
# ---------------------------------------------------------------------------
def metric_row(nombre: str, y: np.ndarray, p: np.ndarray) -> dict:
    pr = average_precision_score(y, p)
    prev = y.mean()
    return {"caso": nombre, "n": len(y), "prevalencia_%": 100 * prev,
            "roc_auc": roc_auc_score(y, p), "pr_auc": pr, "lift": pr / prev}


def run_cv(flags: pd.DataFrame) -> pd.DataFrame:
    """CV agrupada de 5 folds dentro de train, con la etiqueta vieja y con la nueva.

    Mismo diseño que `cv_model_comparison.run_cv` (mismos folds, mismo colapso ajustado
    dentro de cada fold) y con `n_estimators` FIJO a los 122 de LGBM_PARAMS en los dos
    casos: así la única diferencia entre antes y después es la ETIQUETA. Si en el
    "después" se usara el early stopping del modelo nuevo, se mezclarían dos cambios.
    """
    train, val, test = D.grouped_split(D.clean(D.load_raw()), seed=D.SEED)
    D.assert_no_patient_overlap(train, val, test)
    del val, test

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=D.SEED)
    rows = []
    for fold, (tr_idx, ev_idx) in enumerate(
            sgkf.split(train, train[D.TARGET], groups=train[D.GROUP_COL]), start=1):
        tr, ev = train.iloc[tr_idx], train.iloc[ev_idx]
        tr_c, ev_c = D.collapse_rare_categories(tr, ev)
        tr_c, ev_c = P.build_feature_frame(tr_c), P.build_feature_frame(ev_c)
        Ltr, Lev = P.to_lgbm_frames(tr_c, ev_c)

        for etiqueta in ("vieja", "nueva"):
            if etiqueta == "vieja":
                ytr, yev = tr_c[D.TARGET].to_numpy(), ev_c[D.TARGET].to_numpy()
            else:
                ytr = TC.relabel(tr_c, flags).to_numpy()
                yev = TC.relabel(ev_c, flags).to_numpy()
            model = lgb.LGBMClassifier(learning_rate=LEARNING_RATE, random_state=D.SEED,
                                       verbose=-1, **LGBM_PARAMS).fit(Ltr, ytr)
            p = model.predict_proba(Lev)[:, 1]
            rows.append({"fold": fold, "etiqueta": etiqueta,
                         **metric_row(f"fold{fold}/{etiqueta}", yev, p)})
    return pd.DataFrame(rows).drop(columns="caso")


# ---------------------------------------------------------------------------
def mean_abs_shap_discharge(model: lgb.LGBMClassifier, L: pd.DataFrame) -> float:
    """SHAP medio |contribución| de discharge_disposition_id en val, booster truncado."""
    booster = lgb.Booster(model_str=model.booster_.model_to_string(
        num_iteration=model.best_iteration_))
    values, _base = compute_shap(booster, L)
    return float(np.abs(values[:, list(L.columns).index(DISCHARGE)]).mean())


def main():
    pd.set_option("display.width", 200)
    train, val, test = P.load_model_data()          # split con el TARGET ORIGINAL
    splits = {"train": train, "val": val, "test": test}
    flags = TC.planned_flags()

    # --- Comprobaciones mecánicas del reetiquetado -------------------------
    dentro = TC.assert_next_within_split(splits, flags)
    print("¿El encuentro siguiente cae siempre en el mismo split?")
    print(dentro.to_string(index=False))

    counts = TC.changed_counts(splits, flags)
    by_dest = TC.changed_by_destination(splits, flags)
    print("\nFilas reetiquetadas (1 -> 0) por split:")
    print(counts.round(3).to_string(index=False))
    print("\nDesglose por destino al alta (train+val; de test solo el conteo total):")
    print(by_dest.round(2).to_string())

    # --- Datos y etiquetas -------------------------------------------------
    (Xtr, ytr_old), (Xva, yva_old) = P.split_xy(train), P.split_xy(val)
    Ltr, Lva = P.to_lgbm_frames(train, val)
    ytr_new = TC.relabel(train, flags).to_numpy()
    yva_new = TC.relabel(val, flags).to_numpy()
    inpatient = Xva["number_inpatient"].to_numpy()
    destino = Xva[DISCHARGE].astype(str).to_numpy()

    # --- Modelo viejo: guardarraíl de la Fase 2b ---------------------------
    old = fit_lgbm(Ltr, ytr_old, Lva, yva_old, **CHOSEN)
    assert old.best_iteration_ == EXPECTED_TREES, old.best_iteration_
    assert round(old.best_score_["val"]["average_precision"], 4) == EXPECTED_VAL_PR_AUC
    print(f"\nGuardarraíl modelo viejo OK: {old.best_iteration_} árboles, "
          f"PR-AUC val {old.best_score_['val']['average_precision']:.4f}")

    # --- Modelo nuevo: mismos hiperparámetros, etiqueta corregida ----------
    new = fit_lgbm(Ltr, ytr_new, Lva, yva_new, **CHOSEN)
    print(f"Modelo nuevo (guardarraíl nuevo, anótalo): {new.best_iteration_} árboles, "
          f"PR-AUC val {new.best_score_['val']['average_precision']:.4f}")

    p_old = pd.Series(old.predict_proba(Lva)[:, 1], index=Xva.index)
    p_new = pd.Series(new.predict_proba(Lva)[:, 1], index=Xva.index)

    # --- Los tres pasos ----------------------------------------------------
    pasos = pd.concat([
        subgroup_table(p_old, yva_old, inpatient, destino, "(1) viejo/vieja"),
        subgroup_table(p_old, yva_new, inpatient, destino, "(2) viejo/nueva"),
        subgroup_table(p_new, yva_new, inpatient, destino, "(3) nuevo/nueva"),
    ], ignore_index=True)
    gaps = gap_rows(pasos)
    deltas = delta_rows(pasos)

    print("\nLos tres pasos (val, top 10%):")
    print(pasos.round(4).to_string(index=False))
    # IC del 95% de la brecha, bootstrap de pacientes (2.000 réplicas, semilla D.SEED).
    grupos = val[D.GROUP_COL].to_numpy()
    ic = pd.DataFrame([
        bootstrap_gap(p_old, yva_old, inpatient, grupos, "(1) viejo/vieja"),
        bootstrap_gap(p_old, yva_new, inpatient, grupos, "(2) viejo/nueva"),
        bootstrap_gap(p_new, yva_new, inpatient, grupos, "(3) nuevo/nueva"),
    ])
    # n de cada subgrupo al lado de su IC, como pide el prerregistro.
    ns = pasos.pivot(index="paso", columns="subgrupo", values="n_marcados")
    gaps = (gaps.merge(ic, on="paso")
                .merge(ns.rename(columns={"A": "n_A", "B": "n_B"}).reset_index(), on="paso"))

    print("\nBrecha A - B por paso, con IC95 bootstrap de pacientes (descriptivos):")
    print(gaps.round(4).to_string(index=False))

    # Contrastes EMPAREJADOS: la prueba del titular (ver prerregistro).
    pareado = bootstrap_paired(p_old, p_new, yva_old, yva_new, inpatient, grupos)
    print("\nContrastes emparejados, calculados dentro de cada réplica:")
    print(pareado.round(4).to_string(index=False))
    print("\nDeltas por subgrupo (puntos de precisión):")
    print(deltas.round(4).to_string(index=False))

    da = float(deltas.loc[deltas["subgrupo"] == "A", "delta_total"].iloc[0])
    db = float(deltas.loc[deltas["subgrupo"] == "B", "delta_total"].iloc[0])
    veredicto = "CONFIRMADA" if da > db else "REFUTADA"
    print(f"\nCriterio prerregistrado: delta_A = {da:.4f}, delta_B = {db:.4f} "
          f"-> {veredicto}")

    # --- Métricas globales en val, con prevalencia y lift ------------------
    met = pd.DataFrame([
        metric_row("(1) viejo/vieja", yva_old, p_old.to_numpy()),
        metric_row("(2) viejo/nueva", yva_new, p_old.to_numpy()),
        metric_row("(3) nuevo/nueva", yva_new, p_new.to_numpy()),
    ])
    print("\nMétricas globales en val (PR-AUC siempre con prevalencia y lift):")
    print(met.round(4).to_string(index=False))

    # --- SHAP medio del destino al alta, antes y después -------------------
    shap_old = mean_abs_shap_discharge(old, Lva)
    shap_new = mean_abs_shap_discharge(new, Lva)
    shap_tab = pd.DataFrame([
        {"modelo": "viejo (etiqueta vieja)", "shap_medio_abs_discharge": shap_old},
        {"modelo": "nuevo (etiqueta nueva)", "shap_medio_abs_discharge": shap_new},
        {"modelo": "cambio", "shap_medio_abs_discharge": shap_new - shap_old},
    ])
    print("\nSHAP medio |discharge_disposition_id| en val (prerregistro: se esperaba que BAJE):")
    print(shap_tab.round(4).to_string(index=False))

    # --- CV de 5 folds, etiqueta vieja frente a nueva ----------------------
    cv = run_cv(flags)
    resumen = cv.groupby("etiqueta").agg(
        pr_auc_media=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        prevalencia_media=("prevalencia_%", "mean"),
        lift_medio=("lift", "mean"), lift_std=("lift", "std"))
    print("\nCV agrupada de 5 folds dentro de train (n_estimators fijo a 122 en ambas):")
    print(cv.round(4).to_string(index=False))
    print("\nResumen de la CV:")
    print(resumen.round(4).to_string())

    # --- Salidas -----------------------------------------------------------
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dentro.to_csv(REPORT_DIR / "phase4_split_check.csv", index=False)
    counts.to_csv(REPORT_DIR / "phase4_target_correction.csv", index=False)
    by_dest.to_csv(REPORT_DIR / "phase4_relabel_by_destination.csv", index_label="destino")
    pasos.to_csv(REPORT_DIR / "phase4_prediction_steps.csv", index=False)
    gaps.to_csv(REPORT_DIR / "phase4_prediction_gaps.csv", index=False)
    pareado.to_csv(REPORT_DIR / "phase4_paired_contrasts.csv", index=False)
    deltas.to_csv(REPORT_DIR / "phase4_prediction_deltas.csv", index=False)
    met.to_csv(REPORT_DIR / "phase4_val_metrics.csv", index=False)
    shap_tab.to_csv(REPORT_DIR / "phase4_shap_discharge.csv", index=False)
    cv.to_csv(REPORT_DIR / "phase4_cv_lift.csv", index=False)


if __name__ == "__main__":
    main()
