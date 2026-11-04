"""
Fase 2b — Validación cruzada agrupada DENTRO de train: LR vs. LightGBM.

Pregunta: ¿la ventaja de LightGBM en val es estable o es un efecto de ese split
concreto? Se usa SOLO train; val y test se descartan justo después del split.

Decisiones para que la comparación sea justa:
  - Configuraciones fijas, ya elegidas en val: LR con C=1; LightGBM con
    num_leaves=31, min_child_samples=20 y 122 árboles.
  - SIN early stopping por fold: parar con el fold que luego se puntúa daría a
    LightGBM una ventaja artificial (elegiría su iteración mirando esos datos).
  - Todo lo que se ajusta a los datos (colapso de categorías raras, scaler,
    one-hot, categorías de LightGBM) se ajusta con los 4 folds de entrenamiento
    de cada iteración, nunca con el fold de evaluación.
  - Los dos modelos ven las mismas particiones -> comparación por parejas.

Limitación que hay que reportar: los hiperparámetros se eligieron con modelos
entrenados en train (evaluados en val), así que esta CV no es independiente de
esa selección. Sirve para medir estabilidad, no como estimación final; esa es test.
"""

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

import data as D
import preprocessing as P
from train_baseline import make_logreg
from train_lgbm import LEARNING_RATE

N_FOLDS = 5
LR_C = 1.0
LGBM_PARAMS = {"num_leaves": 31, "min_child_samples": 20, "n_estimators": 122}


def run_cv() -> pd.DataFrame:
    # Train SIN colapsar: el colapso se reajusta dentro de cada fold.
    train, val, test = D.grouped_split(D.clean(D.load_raw()), seed=D.SEED)
    D.assert_no_patient_overlap(train, val, test)  # val y test quedan fuera de la CV
    del val, test

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=D.SEED)
    rows = []
    for fold, (tr_idx, ev_idx) in enumerate(
            sgkf.split(train, train[D.TARGET], groups=train[D.GROUP_COL]), start=1):
        tr, ev = train.iloc[tr_idx], train.iloc[ev_idx]
        D.assert_no_patient_overlap(tr, ev)

        tr, ev = D.collapse_rare_categories(tr, ev)     # ajustado solo con tr
        tr, ev = P.build_feature_frame(tr), P.build_feature_frame(ev)  # reglas fijas
        (Xtr, ytr), (Xev, yev) = P.split_xy(tr), P.split_xy(ev)

        logreg = make_logreg(LR_C).fit(Xtr, ytr)        # pipeline: se ajusta solo con Xtr
        Ltr, Lev = P.to_lgbm_frames(tr, ev)             # categorías fijadas con tr
        lgbm = lgb.LGBMClassifier(learning_rate=LEARNING_RATE, random_state=D.SEED,
                                  verbose=-1, **LGBM_PARAMS).fit(Ltr, ytr)

        rows.append({
            "fold": fold, "n_eval": len(ev), "%_pos_eval": 100 * yev.mean(),
            "LR": average_precision_score(yev, logreg.predict_proba(Xev)[:, 1]),
            "LightGBM": average_precision_score(yev, lgbm.predict_proba(Lev)[:, 1]),
        })
    res = pd.DataFrame(rows)
    res["dif_LGBM-LR"] = res["LightGBM"] - res["LR"]
    return res


if __name__ == "__main__":
    res = run_cv()
    pd.set_option("display.width", 140)
    print("PR-AUC por fold (5 folds agrupados dentro de train):")
    print(res.round(4).to_string(index=False))
    print("\nMedia ± desviación entre folds:")
    for col in ["LR", "LightGBM", "dif_LGBM-LR"]:
        print(f"  {col:<12} {res[col].mean():.4f} ± {res[col].std():.4f}")
    print(f"\nFolds en los que LightGBM supera a LR: "
          f"{(res['dif_LGBM-LR'] > 0).sum()} de {len(res)}")
