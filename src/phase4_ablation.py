"""
Fase 4 — Ablación de `payer_code`: decide qué modelo es el entregable.

Regla de decisión PRERREGISTRADA (commit 8872ae6, antes de ejecutar esto). El modelo
ablacionado pasa a ser el entregable si y solo si:
  1. REDUCE el recuento de fallos ROBUSTOS (cociente fuera de [0,80; 1,25] Y IC95 que
     excluya el 1), sobre el conjunto COMÚN de grupos elegibles, y
  2. su lift en CV de 5 folds NO cae más de un 5% relativo frente al modelo con
     `payer_code`.
Si no se cumplen las dos, el entregable sigue siendo el modelo CON `payer_code`.

Hipótesis de Jaime, escrita antes de medir: la disparidad PERSISTIRÁ en buena medida,
porque `race` es recuperable del resto del formulario (AUC OVR macro 0,737). Si se
confirma, el hallazgo es que quitar la variable no arregla el problema: lo hace invisible
y no auditable. Se reporta salga como salga.

Todo en val y en CV dentro de train. TEST NO SE TOCA.
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
from evaluate_test import CHOSEN
from phase4_fairness import (FOUR_FIFTHS, K_PRIMARY, NEW_TREES, NEW_VAL_PR_AUC,
                             composition)
from train_lgbm import LEARNING_RATE, fit_lgbm

REPORT_DIR = D.PROJECT_ROOT / "reports"
ABLATED = "payer_code"
CV_TOLERANCE = 0.05          # caída relativa máxima admitida en el lift de CV


def frames(train: pd.DataFrame, val: pd.DataFrame, feats: list):
    """Frames de LightGBM con las categorías fijadas en train, para `feats`."""
    return D.to_categorical(train[feats], val[feats])


def lift(y: np.ndarray, p: np.ndarray) -> dict:
    pr = average_precision_score(y, p)
    prev = y.mean()
    return {"n": len(y), "prevalencia_%": 100 * prev, "roc_auc": roc_auc_score(y, p),
            "pr_auc": pr, "lift": pr / prev}


def run_cv(flags: pd.DataFrame, feats_by_name: dict) -> pd.DataFrame:
    """CV agrupada de 5 folds con la etiqueta corregida, con y sin `payer_code`.

    `n_estimators` FIJO (LGBM_PARAMS, 122) en los dos: así lo único que cambia entre las
    dos filas de cada fold es el conjunto de features, no el nº de árboles.
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
        ytr = TC.relabel(tr_c, flags).to_numpy()
        yev = TC.relabel(ev_c, flags).to_numpy()
        for nombre, feats in feats_by_name.items():
            Ltr, Lev = frames(tr_c, ev_c, feats)
            model = lgb.LGBMClassifier(learning_rate=LEARNING_RATE, random_state=D.SEED,
                                       verbose=-1, **LGBM_PARAMS).fit(Ltr, ytr)
            rows.append({"fold": fold, "modelo": nombre,
                         **lift(yev, model.predict_proba(Lev)[:, 1])})
    return pd.DataFrame(rows)


def robust_failures(tab: pd.DataFrame) -> pd.DataFrame:
    """Marca fallo bruto (umbral) y fallo ROBUSTO (umbral + IC que excluya el 1)."""
    t = tab.copy()
    bruto = ~t["cociente"].between(*FOUR_FIFTHS)
    ic_excluye_1 = ~((t["cociente_IC95_bajo"] <= 1) & (1 <= t["cociente_IC95_alto"]))
    t["fallo_bruto"] = bruto & t["elegible"] & ~t["es_referencia"]
    t["fallo_robusto"] = t["fallo_bruto"] & ic_excluye_1
    return t


def main():
    pd.set_option("display.width", 220)
    train, val, _test = P.load_model_data()
    flags = TC.planned_flags()
    ytr = TC.relabel(train, flags).to_numpy()
    yva = TC.relabel(val, flags).to_numpy()
    pac = val[D.GROUP_COL].to_numpy()

    feats_con = list(P.FEATURES)
    feats_sin = [c for c in P.FEATURES if c != ABLATED]

    # --- Modelo con payer_code: el actual, con su guardarraíl ---------------
    Ltr, Lva = frames(train, val, feats_con)
    con = fit_lgbm(Ltr, ytr, Lva, yva, **CHOSEN)
    assert con.best_iteration_ == NEW_TREES, con.best_iteration_
    assert round(con.best_score_["val"]["average_precision"], 4) == NEW_VAL_PR_AUC
    print(f"Modelo CON payer_code (guardarraíl OK): {con.best_iteration_} árboles, "
          f"PR-AUC val {con.best_score_['val']['average_precision']:.4f}")

    # --- Modelo ablacionado: su propio guardarraíl, que se anota -----------
    Ltr_s, Lva_s = frames(train, val, feats_sin)
    sin = fit_lgbm(Ltr_s, ytr, Lva_s, yva, **CHOSEN)
    print(f"Modelo SIN payer_code (guardarraíl NUEVO, anótalo): {sin.best_iteration_} "
          f"árboles, PR-AUC val {sin.best_score_['val']['average_precision']:.4f}")

    p_con = con.predict_proba(Lva)[:, 1]
    p_sin = sin.predict_proba(Lva_s)[:, 1]

    # --- (a) Lift en val ----------------------------------------------------
    val_tab = pd.DataFrame([{"modelo": "con payer_code", **lift(yva, p_con)},
                            {"modelo": "sin payer_code", **lift(yva, p_sin)}])
    print("\n(a) Val, etiqueta corregida:")
    print(val_tab.round(4).to_string(index=False))

    # --- (a) Lift en CV -----------------------------------------------------
    cv = run_cv(flags, {"con payer_code": feats_con, "sin payer_code": feats_sin})
    res_cv = cv.groupby("modelo").agg(lift_medio=("lift", "mean"), lift_std=("lift", "std"),
                                      pr_auc_medio=("pr_auc", "mean"))
    l_con = res_cv.loc["con payer_code", "lift_medio"]
    l_sin = res_cv.loc["sin payer_code", "lift_medio"]
    caida_rel = (l_con - l_sin) / l_con
    print("\n(a) CV de 5 folds (n_estimators fijo en los dos):")
    print(cv.round(4).to_string(index=False))
    print(res_cv.round(4).to_string())
    print(f"Caída relativa del lift en CV: {100 * caida_rel:.2f}% "
          f"(tolerancia prerregistrada: {100 * CV_TOLERANCE:.0f}%)")

    # --- (b) Composición del top 10% por payer_code, con y sin -------------
    # Se llama dos veces porque `composition` calcula el IC del primer modelo que recibe.
    comp_con = robust_failures(composition(ABLATED, val[ABLATED], p_con, p_sin, pac,
                                           K_PRIMARY, primario=True))
    comp_sin = robust_failures(composition(ABLATED, val[ABLATED], p_sin, p_con, pac,
                                           K_PRIMARY, primario=True))

    lado = pd.DataFrame({
        "grupo": comp_con["grupo"], "es_referencia": comp_con["es_referencia"],
        "n_grupo": comp_con["n_grupo"],
        "marcados_con": comp_con["marcados"], "cociente_con": comp_con["cociente"],
        "IC_con_bajo": comp_con["cociente_IC95_bajo"], "IC_con_alto": comp_con["cociente_IC95_alto"],
        "elegible_con": comp_con["elegible"], "fallo_robusto_con": comp_con["fallo_robusto"],
        "marcados_sin": comp_sin["marcados"], "cociente_sin": comp_sin["cociente"],
        "IC_sin_bajo": comp_sin["cociente_IC95_bajo"], "IC_sin_alto": comp_sin["cociente_IC95_alto"],
        "elegible_sin": comp_sin["elegible"], "fallo_robusto_sin": comp_sin["fallo_robusto"],
    })
    lado["cambio_cociente"] = lado["cociente_sin"] - lado["cociente_con"]
    print("\n(b) Composición del top 10% por payer_code, lado a lado (descriptivo):")
    print(lado.round(4).to_string(index=False))

    # --- Regla de decisión --------------------------------------------------
    comun = lado["elegible_con"] & lado["elegible_sin"] & ~lado["es_referencia"]
    rob_con = int(lado.loc[comun, "fallo_robusto_con"].sum())
    rob_sin = int(lado.loc[comun, "fallo_robusto_sin"].sum())
    bruto_con = int(comp_con["fallo_bruto"].sum())
    bruto_sin = int(comp_sin["fallo_bruto"].sum())
    entra = lado.loc[~lado["elegible_con"] & lado["elegible_sin"], "grupo"].tolist()
    sale = lado.loc[lado["elegible_con"] & ~lado["elegible_sin"], "grupo"].tolist()

    cond1 = rob_sin < rob_con
    cond2 = caida_rel <= CV_TOLERANCE
    entregable = "sin payer_code" if (cond1 and cond2) else "con payer_code"
    decision = pd.DataFrame([{
        "grupos_comunes_elegibles": int(comun.sum()),
        "fallos_robustos_con": rob_con, "fallos_robustos_sin": rob_sin,
        "fallos_brutos_con": bruto_con, "fallos_brutos_sin": bruto_sin,
        "entran_en_elegibilidad": ", ".join(entra) or "ninguno",
        "salen_de_elegibilidad": ", ".join(sale) or "ninguno",
        "lift_cv_con": l_con, "lift_cv_sin": l_sin, "caida_relativa_%": 100 * caida_rel,
        "cond1_reduce_fallos_robustos": cond1,
        "cond2_lift_dentro_de_tolerancia": cond2,
        "ENTREGABLE": entregable,
    }])
    print("\nRegla de decisión prerregistrada:")
    print(decision.T.to_string(header=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    val_tab.to_csv(REPORT_DIR / "phase4_ablation_val.csv", index=False)
    cv.to_csv(REPORT_DIR / "phase4_ablation_cv.csv", index=False)
    lado.to_csv(REPORT_DIR / "phase4_ablation_composition.csv", index=False)
    decision.to_csv(REPORT_DIR / "phase4_ablation_decision.csv", index=False)


if __name__ == "__main__":
    main()
