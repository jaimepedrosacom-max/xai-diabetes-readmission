"""
Fase 2c — Evaluación final en test del modelo elegido (LightGBM).

Este es el ÚNICO script que predice con LightGBM en test. Se ejecuta después de
elegir el modelo con val y de confirmar la ventaja con la CV dentro de train
(src/cv_model_comparison.py). Sus cifras no deben usarse para retocar el modelo:
si se hiciera, dejarían de ser una estimación honesta.

Guardarraíl: antes de tocar test se comprueba que el modelo reentrenado es
exactamente el elegido en la Fase 2b (mismo nº de árboles y mismo PR-AUC en val).
"""

import pandas as pd

import data as D
import evaluation as E
import preprocessing as P
from train_baseline import select_logreg, trivial_baselines
from train_lgbm import fit_lgbm

# Configuración elegida en la Fase 2b (ver train_lgbm.py).
CHOSEN = {"num_leaves": 31, "min_child_samples": 20}
EXPECTED_TREES = 122
EXPECTED_VAL_PR_AUC = 0.2308

REPORT_DIR = D.PROJECT_ROOT / "reports"


def main():
    train, val, test = P.load_model_data()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (P.split_xy(s) for s in (train, val, test))
    Ltr, Lva, Lte = P.to_lgbm_frames(train, val, test)  # categorías fijadas con train

    # El early stopping sigue usando val; test no interviene en el ajuste.
    lgbm = fit_lgbm(Ltr, ytr, Lva, yva, **CHOSEN)
    val_pr_auc = lgbm.best_score_["val"]["average_precision"]
    assert lgbm.best_iteration_ == EXPECTED_TREES, lgbm.best_iteration_
    assert round(val_pr_auc, 4) == EXPECTED_VAL_PR_AUC, val_pr_auc
    print(f"Guardarraíl OK: {lgbm.best_iteration_} árboles, PR-AUC val = {val_pr_auc:.4f}")

    logreg, _ = select_logreg(Xtr, ytr, Xva, yva)
    assert logreg.named_steps["clf"].C == 1

    Xs = {"train": Xtr, "val": Xva, "test": Xte}
    Ls = {"train": Ltr, "val": Lva, "test": Lte}
    ys = {"train": ytr, "val": yva, "test": yte}
    trivial, _, _ = trivial_baselines(Xs, ytr)
    preds = {"LightGBM": {s: lgbm.predict_proba(L)[:, 1] for s, L in Ls.items()},
             "LogReg": {s: logreg.predict_proba(X)[:, 1] for s, X in Xs.items()},
             **trivial}

    results = [E.evaluate(name, p, ys) for name, p in preds.items()]
    met = pd.concat([m for m, _ in results])
    topk = pd.concat([t for _, t in results])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    met.to_csv(REPORT_DIR / "metrics_final.csv", index=False)
    topk.to_csv(REPORT_DIR / "topk_final.csv", index=False)

    pd.set_option("display.width", 160)
    print("\nMétricas por split:")
    print(met.pivot(index="modelo", columns="split", values=["roc_auc", "pr_auc", "brier"])
          .reindex(columns=["train", "val", "test"], level=1).round(4).to_string())
    print("\nTop-k% en TEST:")
    print(topk[topk["split"] == "test"].pivot_table(
        index=["modelo", "top_k"], values=["precision", "recall", "lift"], sort=False)
        [["precision", "recall", "lift"]].round(3).to_string())
    print("\nTop-k% de LightGBM por split (precisión / recall):")
    print(topk[topk["modelo"] == "LightGBM"].pivot_table(
        index="top_k", columns="split", values=["precision", "recall"], sort=False)
        .reindex(columns=["train", "val", "test"], level=1).round(3).to_string())

    path = E.plot_calibration({"LogReg": preds["LogReg"], "LightGBM": preds["LightGBM"]},
                              ys, "calibration_logreg_vs_lgbm.png")
    print(f"\nCurva de calibración actualizada en: {path}")


if __name__ == "__main__":
    main()
