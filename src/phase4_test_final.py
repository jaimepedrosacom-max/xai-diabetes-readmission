"""
Fase 4, punto 4 — CONSULTA ÚNICA A TEST con el modelo entregable.

Es la SEGUNDA vez que se toca test en todo el proyecto. La primera fue la Fase 2c
(`evaluate_test.py`), con el target ORIGINAL. Esta cifra NO es comparable con el
PR-AUC 0,227 de entonces, porque el target cambió: responde a otra pregunta (cómo
predice el TARGET CORREGIDO en datos no vistos).

Nada cambia después de verla. Hiperparámetros, split, corrección del target y elección
de modelo están cerrados ANTES de ejecutar esto:
  - split congelado con el target original (Fase 4, "Split congelado");
  - corrección del target aplicada (`target_correction.py`);
  - entregable = modelo corregido SIN `payer_code`, decidido en val por la regla
    prerregistrada de la ablación (commit 8872ae6, resultados en 2f2b80c).

Guardarraíles que se comprueban ANTES de puntuar test:
  - entregable (sin `payer_code`): 123 árboles, PR-AUC val 0,1868;
  - corregido CON `payer_code`: 183 árboles, PR-AUC val 0,1893;
  - modelo viejo (Fase 2): 122 árboles, PR-AUC val 0,2308.

Qué se mide en test (etiqueta CORREGIDA):
  1. Globales del ENTREGABLE: ROC-AUC, PR-AUC, prevalencia, lift (el TITULAR) y Brier.
  2. Top-k 5/10/20% del entregable: marcados, aciertos, precisión, recall y lift, que es
     lo que el gestor aplica.
  3. Líneas base en test con la etiqueta corregida: prevalencia y la regla
     `number_inpatient > 0`. Sin referencia, el PR-AUC no se interpreta.
  4. Replicación DESCRIPTIVA de los pasos de la brecha A - B, con los cuatro pasos:
     (1) viejo/vieja, (2) viejo/nueva, (3a) corregido CON `payer_code`/nueva y
     (3b) ablacionado (entregable)/nueva.

Por qué DOS pasos (3): (3a) es el modelo que se usó en val, así que es la replicación
estricta; (3b) es el entregable. De NINGUNO de los dos sale ninguna decisión. El titular
tampoco depende de esa elección: la inversión relevante es la de (1) a (2), donde solo
interviene el modelo viejo.

DELIBERADO, no un olvido — qué NO se mide en test:
  - Las GLOBALES (ROC-AUC, PR-AUC, prevalencia, lift, Brier, top-k y líneas base) son
    SOLO las del entregable. Aquí se calculan las predicciones en test del modelo
    corregido CON `payer_code`, porque el paso (3a) las necesita, pero sus globales NO se
    calculan: tenerlas invitaría a replantear qué modelo se entrega, y eso está decidido
    en val. Lo mismo, y por la misma razón, con el modelo viejo: solo entra en los pasos
    (1) y (2) de la brecha.
  - SHAP y equidad: viven en val por diseño (de las explicaciones salen decisiones, y
    test no debe influir en ninguna).
  - La regresión logística: se excluye para no crear la tentación de un cambio de modelo
    con test ya visto.

Nada se decide con lo que salga de aquí. Si el número sale mal, se reporta igual.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import data as D
import evaluation as E
import preprocessing as P
import target_correction as TC
from evaluate_test import CHOSEN, EXPECTED_TREES, EXPECTED_VAL_PR_AUC
from phase4_ablation import ABLATED
from phase4_correction import (DISCHARGE, TOP_Q, bootstrap_gap, bootstrap_paired,
                               delta_rows, gap_rows, subgroup_table)
from phase4_fairness import NEW_TREES, NEW_VAL_PR_AUC
from train_lgbm import fit_lgbm

REPORT_DIR = D.PROJECT_ROOT / "reports"

# Guardarraíl del entregable, anotado al ejecutar la ablación (commit 2f2b80c).
ABL_TREES = 123
ABL_VAL_PR_AUC = 0.1868


def fit_with_guardrail(Ltr, ytr, Lva, yva, nombre: str, trees: int, pr_auc: float):
    """Entrena y EXIGE el guardarraíl: mismo nº de árboles y mismo PR-AUC en val.

    Si cualquiera de los dos no cuadra, el script aborta ANTES de tocar test: significa
    que el modelo no es el que se cerró en val, y entonces la consulta a test no sería
    la consulta prerregistrada.
    """
    model = fit_lgbm(Ltr, ytr, Lva, yva, **CHOSEN)
    val_pr = model.best_score_["val"]["average_precision"]
    assert model.best_iteration_ == trees, (nombre, model.best_iteration_)
    assert round(val_pr, 4) == pr_auc, (nombre, val_pr)
    print(f"Guardarraíl OK — {nombre}: {model.best_iteration_} árboles, "
          f"PR-AUC val {val_pr:.4f}")
    return model


def global_row(nombre: str, y: np.ndarray, p: np.ndarray) -> dict:
    """Globales con prevalencia y lift SIEMPRE al lado del PR-AUC."""
    pr = average_precision_score(y, p)
    prev = y.mean()
    return {"modelo": nombre, "n": len(y), "prevalencia_%": 100 * prev,
            "roc_auc": roc_auc_score(y, p), "pr_auc": pr, "lift_pr_auc": pr / prev,
            "brier": brier_score_loss(y, p)}


def topk_rows(nombre: str, y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Top-k 5/10/20% con marcados, aciertos, precisión, recall y lift."""
    tk = E.top_k_table(y, p)
    tk.insert(0, "modelo", nombre)
    tk["aciertos"] = (tk["precision"] * tk["n_marcados"]).round().astype(int)
    tk["precision_%"] = 100 * tk["precision"]
    tk["recall_%"] = 100 * tk["recall"]
    return tk[["modelo", "top_k", "n_marcados", "aciertos", "precision_%", "recall_%",
               "lift"]]


def main():
    pd.set_option("display.width", 220)
    train, val, test = P.load_model_data()        # split con el TARGET ORIGINAL
    flags = TC.planned_flags()

    (Xtr, ytr_old), (Xva, yva_old), (Xte, yte_old) = (
        P.split_xy(s) for s in (train, val, test))
    ytr_new = TC.relabel(train, flags).to_numpy()
    yva_new = TC.relabel(val, flags).to_numpy()
    yte_new = TC.relabel(test, flags).to_numpy()

    # Frames con TODAS las features y sin `payer_code`; categorías fijadas en train.
    Ltr, Lva, Lte = P.to_lgbm_frames(train, val, test)
    feats_sin = [c for c in P.FEATURES if c != ABLATED]
    Ltr_s, Lva_s, Lte_s = D.to_categorical(train[feats_sin], val[feats_sin],
                                           test[feats_sin])

    # --- Los tres modelos, cada uno con su guardarraíl ---------------------
    viejo = fit_with_guardrail(Ltr, ytr_old, Lva, yva_old, "viejo (Fase 2)",
                               EXPECTED_TREES, EXPECTED_VAL_PR_AUC)
    corr_con = fit_with_guardrail(Ltr, ytr_new, Lva, yva_new,
                                  "corregido CON payer_code", NEW_TREES, NEW_VAL_PR_AUC)
    entregable = fit_with_guardrail(Ltr_s, ytr_new, Lva_s, yva_new,
                                    "ENTREGABLE (sin payer_code)", ABL_TREES,
                                    ABL_VAL_PR_AUC)

    # A partir de aquí se toca test. Una sola vez.
    p_te_viejo = pd.Series(viejo.predict_proba(Lte)[:, 1], index=Xte.index)
    p_te_con = pd.Series(corr_con.predict_proba(Lte)[:, 1], index=Xte.index)
    p_te_ent = pd.Series(entregable.predict_proba(Lte_s)[:, 1], index=Xte.index)

    inpatient = Xte["number_inpatient"].to_numpy()
    destino = Xte[DISCHARGE].astype(str).to_numpy()
    grupos = test[D.GROUP_COL].to_numpy()

    # --- 1. Globales del ENTREGABLE y líneas base --------------------------
    # Línea base trivial: puntuación constante -> PR-AUC = prevalencia, lift 1 por
    # construcción. Se incluye para que el PR-AUC del entregable tenga referencia.
    constante = np.full(len(yte_new), yte_new.mean())
    regla = (inpatient > 0).astype(float)
    glob = pd.DataFrame([
        global_row("ENTREGABLE (sin payer_code)", yte_new, p_te_ent.to_numpy()),
        global_row("base: prevalencia (constante)", yte_new, constante),
        global_row("base: regla number_inpatient > 0", yte_new, regla),
    ])
    print("\n1. Test con la etiqueta CORREGIDA — globales del entregable y líneas base:")
    print(glob.round(4).to_string(index=False))

    # --- 2. Top-k del entregable, y de la regla para comparar --------------
    topk = pd.concat([
        topk_rows("ENTREGABLE (sin payer_code)", yte_new, p_te_ent.to_numpy()),
        topk_rows("base: regla number_inpatient > 0", yte_new, regla),
    ], ignore_index=True)
    print("\n2. Top-k en test (lo que el gestor aplica):")
    print(topk.round(4).to_string(index=False))

    # --- 4. Replicación DESCRIPTIVA de la brecha A - B ---------------------
    # Cuatro pasos. (3a) es el modelo usado en val -> replicación estricta;
    # (3b) es el entregable. Ninguna decisión sale de ninguno de los dos.
    pasos = pd.concat([
        subgroup_table(p_te_viejo, yte_old, inpatient, destino, "(1) viejo/vieja"),
        subgroup_table(p_te_viejo, yte_new, inpatient, destino, "(2) viejo/nueva"),
        subgroup_table(p_te_con, yte_new, inpatient, destino, "(3a) corregido CON/nueva"),
        subgroup_table(p_te_ent, yte_new, inpatient, destino, "(3b) entregable SIN/nueva"),
    ], ignore_index=True)
    print("\n4. Los cuatro pasos en TEST (top 10%, corte recalculado en cada paso):")
    print(pasos.round(4).to_string(index=False))

    gaps = gap_rows(pasos)
    ic = pd.DataFrame([
        bootstrap_gap(p_te_viejo, yte_old, inpatient, grupos, "(1) viejo/vieja"),
        bootstrap_gap(p_te_viejo, yte_new, inpatient, grupos, "(2) viejo/nueva"),
        bootstrap_gap(p_te_con, yte_new, inpatient, grupos, "(3a) corregido CON/nueva"),
        bootstrap_gap(p_te_ent, yte_new, inpatient, grupos, "(3b) entregable SIN/nueva"),
    ])
    ns = pasos.pivot(index="paso", columns="subgrupo", values="n_marcados")
    gaps = (gaps.merge(ic, on="paso")
                .merge(ns.rename(columns={"A": "n_A", "B": "n_B"}).reset_index(),
                       on="paso"))
    print("\nBrecha A - B en test, con IC95 bootstrap de pacientes (descriptivos):")
    print(gaps.round(4).to_string(index=False))

    # Contrastes emparejados, uno por cada modelo nuevo. `brecha(2) - brecha(1)` y el
    # cociente de cocientes NO dependen del modelo nuevo (solo usan el viejo), así que
    # salen idénticos en las dos llamadas; lo que cambia es `delta_A - delta_B`.
    pareados = []
    for etiqueta, p_new in [("(3a) corregido CON payer_code", p_te_con),
                            ("(3b) entregable sin payer_code", p_te_ent)]:
        par = bootstrap_paired(p_te_viejo, p_new, yte_old, yte_new, inpatient, grupos)
        par.insert(0, "modelo_nuevo", etiqueta)
        pareados.append(par)
    pareado = pd.concat(pareados, ignore_index=True)
    print("\nContrastes emparejados en test (replicación descriptiva):")
    print(pareado.round(4).to_string(index=False))

    # Deltas por subgrupo, con cada paso (3) por separado.
    deltas = []
    for etiqueta, paso3 in [("(3a) corregido CON payer_code", "(3a) corregido CON/nueva"),
                            ("(3b) entregable sin payer_code", "(3b) entregable SIN/nueva")]:
        sub = pasos[pasos["paso"].isin(
            ["(1) viejo/vieja", "(2) viejo/nueva", paso3])].copy()
        sub["paso"] = sub["paso"].replace({paso3: "(3) nuevo/nueva"})
        d = delta_rows(sub)
        d.insert(0, "modelo_nuevo", etiqueta)
        deltas.append(d)
    deltas = pd.concat(deltas, ignore_index=True)
    print("\nDeltas por subgrupo en test (puntos de precisión):")
    print(deltas.round(4).to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    glob.to_csv(REPORT_DIR / "phase4_test_global.csv", index=False)
    topk.to_csv(REPORT_DIR / "phase4_test_topk.csv", index=False)
    pasos.to_csv(REPORT_DIR / "phase4_test_steps.csv", index=False)
    gaps.to_csv(REPORT_DIR / "phase4_test_gaps.csv", index=False)
    pareado.to_csv(REPORT_DIR / "phase4_test_paired_contrasts.csv", index=False)
    deltas.to_csv(REPORT_DIR / "phase4_test_deltas.csv", index=False)


if __name__ == "__main__":
    main()
