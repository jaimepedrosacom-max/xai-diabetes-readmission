"""
Fase 2a — Baselines: dos triviales + regresión logística.

Orden (y por qué):
  1) Baselines triviales: el suelo que cualquier modelo debe superar.
       - Prevalencia: predice a todos la tasa de positivos de train.
       - Regla number_inpatient > 0: "¿ha estado ingresado en el último año?".
         Su "probabilidad" es la tasa de positivos de train en cada grupo.
  2) Regresión logística: se elige C con val (PR-AUC), entrenando en train.
  3) Test se evalúa UNA vez, al final, con todo ya fijado.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

import data as D
import evaluation as E
import preprocessing as P

C_GRID = [0.001, 0.01, 0.1, 1, 10]  # C = 1/fuerza de regularización L2


def make_logreg(C: float) -> Pipeline:
    return Pipeline([("prep", P.build_lr_preprocessor()),
                     ("clf", LogisticRegression(C=C, max_iter=5000))])


def select_logreg(Xtr, ytr, Xva, yva, c_grid=C_GRID):
    """Entrena una LR por cada C en train y elige la de mejor PR-AUC en val.

    Devuelve (modelo elegido ya entrenado, tabla de la rejilla).
    """
    grid, models = [], {}
    for C in c_grid:
        models[C] = make_logreg(C).fit(Xtr, ytr)
        grid.append({"C": C,
                     "train_pr_auc": average_precision_score(ytr, models[C].predict_proba(Xtr)[:, 1]),
                     "val_pr_auc": average_precision_score(yva, models[C].predict_proba(Xva)[:, 1])})
    grid = pd.DataFrame(grid)
    best_C = grid.loc[grid["val_pr_auc"].idxmax(), "C"]
    return models[best_C], grid


def trivial_baselines(Xs: dict, ytr):
    """Predicciones de las baselines triviales; sus parámetros salen solo de train.

    Xs: {"train": X, "val": X, ...}. Devuelve (preds, prevalencia, tasa por grupo).
    """
    prev = ytr.mean()
    rate_by_group = pd.Series(ytr).groupby(Xs["train"]["number_inpatient"].to_numpy() > 0).mean()
    preds = {
        "Prevalencia": {s: [prev] * len(X) for s, X in Xs.items()},
        "Regla inpatient>0": {s: (X["number_inpatient"] > 0).map(rate_by_group).to_numpy()
                              for s, X in Xs.items()},
    }
    return preds, prev, rate_by_group


def main():
    train, val, test = P.load_model_data()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (P.split_xy(s) for s in (train, val, test))
    ys = {"train": ytr, "val": yva, "test": yte}
    Xs = {"train": Xtr, "val": Xva, "test": Xte}

    # --- 1) Baselines triviales (los parámetros salen solo de train) --------
    trivial, prev, rate_by_group = trivial_baselines(Xs, ytr)
    print(f"Prevalencia train = {prev:.4f} | P(y=1 | inpatient=0) = {rate_by_group[False]:.4f}, "
          f"P(y=1 | inpatient>0) = {rate_by_group[True]:.4f}")

    # --- 2) Regresión logística: selección de C en val ----------------------
    logreg, grid = select_logreg(Xtr, ytr, Xva, yva)
    print("\nRejilla de C (entrena en train, evalúa en val):")
    print(grid.round(4).to_string(index=False))
    print(f"=> C elegido = {logreg.named_steps['clf'].C}")

    n_feats = len(logreg.named_steps["prep"].get_feature_names_out())
    print(f"Nº de columnas tras el ColumnTransformer: {n_feats}")

    # --- 3) Evaluación en train / val / test (test: única vez) ---------------
    preds = {"LogReg": {s: logreg.predict_proba(X)[:, 1] for s, X in Xs.items()}, **trivial}
    results = [E.evaluate(name, p, ys) for name, p in preds.items()]
    met = pd.concat([m for m, _ in results])
    topk = pd.concat([t for _, t in results])

    pd.set_option("display.width", 160)
    print("\nMétricas por split:")
    print(met.pivot(index="modelo", columns="split", values=["roc_auc", "pr_auc", "brier"])
          .reindex(columns=["train", "val", "test"], level=1).round(4).to_string())
    print("\nTop-k% (precisión / recall / lift) por split:")
    print(topk.pivot_table(index=["modelo", "top_k"], columns="split",
                           values=["precision", "recall", "lift"], sort=False)
          .reindex(columns=["train", "val", "test"], level=1).round(3).to_string())

    path = E.plot_calibration({"LogReg": preds["LogReg"]}, ys, "calibration_logreg.png")
    print(f"\nCurva de calibración guardada en: {path}")
    return logreg, met, topk


if __name__ == "__main__":
    main()
