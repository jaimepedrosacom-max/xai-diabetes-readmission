"""
Fase 2b — LightGBM con categóricas nativas y early stopping sobre val.

Reglas de este script (para que test siga limpio):
  - Se entrena SOLO con train. Val se usa para (a) early stopping, (b) elegir
    entre una rejilla pequeña de hiperparámetros y (c) comparar con la LR.
  - Test NO se toca aquí: ni se predice. Se evaluará una única vez al final,
    cuando ya hayamos elegido el modelo definitivo entre LR y LightGBM.
  - No reentrenamos con train+val: el modelo que se evalúa es exactamente el
    que se eligió, con el nº de árboles que fijó el early stopping.

Consecuencia que hay que reportar: como val ha decidido el nº de árboles y los
hiperparámetros, la métrica de val es algo optimista. La honesta será la de test.
"""

import itertools

import lightgbm as lgb
import pandas as pd

import data as D
import evaluation as E
import preprocessing as P
from train_baseline import select_logreg

# Rejilla pequeña a propósito: cuantas más combinaciones probemos en val, más
# "sobreajustamos a val" al elegir.
#  - num_leaves: complejidad de cada árbol (hojas máximas).
#  - min_child_samples: mínimo de pacientes por hoja (regulariza: evita hojas
#    que describan a 20 personas concretas).
GRID = {"num_leaves": [15, 31, 63], "min_child_samples": [20, 100]}
LEARNING_RATE = 0.03
MAX_TREES = 5000   # techo; el early stopping decide cuántos se usan
PATIENCE = 100     # parar si el PR-AUC de val no mejora en 100 árboles


def fit_lgbm(Xtr, ytr, Xva, yva, **params) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=MAX_TREES,
        learning_rate=LEARNING_RATE,
        # Única métrica de evaluación -> el early stopping vigila SOLO esta
        # (si no, LightGBM añade binary_logloss y podría parar por ella).
        metric="average_precision",
        random_state=D.SEED,
        verbose=-1,
        **params,
    )
    # Las columnas con dtype 'category' se tratan como categóricas nativas
    # (categorical_feature="auto" por defecto).
    # eval_X/eval_y: API de LightGBM >= 4.6 (sustituye a eval_set, obsoleto).
    # Va como tupla: una lista de DataFrames se interpretaría como datos crudos.
    model.fit(Xtr, ytr, eval_X=(Xva,), eval_y=(yva,), eval_names=["val"],
              callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    return model


def main():
    train, val, test = P.load_model_data()
    (Xtr, ytr), (Xva, yva) = P.split_xy(train), P.split_xy(val)
    Ltr, Lva = P.to_lgbm_frames(train, val)  # categorías fijadas con train
    # (test no se convierte ni se usa en este script)

    cat_cols = [c for c in Ltr.columns if str(Ltr[c].dtype) == "category"]
    print(f"Features: {Ltr.shape[1]} ({len(cat_cols)} categóricas nativas, "
          f"{Ltr.shape[1] - len(cat_cols)} numéricas)")

    # --- Rejilla + early stopping (todo sobre val) ---------------------------
    rows, models = [], {}
    for nl, mcs in itertools.product(GRID["num_leaves"], GRID["min_child_samples"]):
        m = fit_lgbm(Ltr, ytr, Lva, yva, num_leaves=nl, min_child_samples=mcs)
        models[(nl, mcs)] = m
        rows.append({"num_leaves": nl, "min_child_samples": mcs,
                     "best_iteration": m.best_iteration_,
                     "train_pr_auc": E.metrics(ytr, m.predict_proba(Ltr)[:, 1])["pr_auc"],
                     "val_pr_auc": m.best_score_["val"]["average_precision"]})
    grid = pd.DataFrame(rows)
    best = grid.loc[grid["val_pr_auc"].idxmax()]
    lgbm = models[(int(best["num_leaves"]), int(best["min_child_samples"]))]
    print("\nRejilla LightGBM (early stopping en val, PR-AUC):")
    print(grid.round(4).to_string(index=False))
    print(f"=> elegido: num_leaves={int(best['num_leaves'])}, "
          f"min_child_samples={int(best['min_child_samples'])}, "
          f"árboles={lgbm.best_iteration_}")
    # Comprobación: la métrica vigilada es solo average_precision.
    assert list(lgbm.evals_result_["val"]) == ["average_precision"]

    # --- LR elegida igual que en la Fase 2a (mismo C por val) ----------------
    logreg, _ = select_logreg(Xtr, ytr, Xva, yva)

    # --- Comparación SOLO en train/val ---------------------------------------
    ys = {"train": ytr, "val": yva}
    preds = {
        "LogReg": {"train": logreg.predict_proba(Xtr)[:, 1], "val": logreg.predict_proba(Xva)[:, 1]},
        # predict_proba usa por defecto best_iteration_ (los árboles de early stopping)
        "LightGBM": {"train": lgbm.predict_proba(Ltr)[:, 1], "val": lgbm.predict_proba(Lva)[:, 1]},
    }
    results = [E.evaluate(name, p, ys) for name, p in preds.items()]
    met = pd.concat([m for m, _ in results])
    topk = pd.concat([t for _, t in results])

    pd.set_option("display.width", 160)
    print("\nMétricas (train / val) — test NO evaluado:")
    print(met.pivot(index="modelo", columns="split", values=["roc_auc", "pr_auc", "brier"])
          .reindex(columns=["train", "val"], level=1).round(4).to_string())
    print("\nTop-k% en VAL:")
    print(topk[topk["split"] == "val"].pivot_table(
        index="top_k", columns="modelo", values=["precision", "recall", "lift"], sort=False)
        .round(3).to_string())

    # --- Calibración: LR con sus 3 splits (test de LR ya estaba gastado en 2a);
    #     LightGBM solo train/val. El panel de test lo indica.
    Xte, yte = P.split_xy(test)
    preds_plot = {"LogReg": {**preds["LogReg"], "test": logreg.predict_proba(Xte)[:, 1]},
                  "LightGBM": preds["LightGBM"]}
    path = E.plot_calibration(preds_plot, {"train": ytr, "val": yva, "test": yte},
                              "calibration_logreg_vs_lgbm.png")
    print(f"\nCurva de calibración guardada en: {path}")
    return lgbm, logreg, met, topk


if __name__ == "__main__":
    main()
