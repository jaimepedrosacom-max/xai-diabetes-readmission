"""
Experimento de leakage por paciente (Fase 1b, punto 7).

Pregunta: ¿cuánto se inflan las métricas de test si hacemos un split
aleatorio por FILAS (el mismo paciente puede estar en train y en test)
en vez de un split agrupado por PACIENTE?

Diseño (para que la comparación sea justa):
  - Mismos datos limpios (data.clean) y mismo modelo: LightGBM con
    hiperparámetros por defecto, sin tunear ni early stopping. Esto NO es
    el modelo final: solo una sonda para medir el efecto del split.
  - En ambos casos se entrena con el 80% y se evalúa en el 20% de test
    (en el agrupado, train+val del split 60/20/20 forman el 80%).
  - Se repite con 5 semillas: con una sola, una diferencia de ~0,005 de AUC
    podría ser ruido del split.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import data as D

SEEDS = [42, 0, 1, 2, 3]


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict:
    """Preprocesa (ajustado en train), entrena LightGBM por defecto y evalúa."""
    train, test = D.collapse_rare_categories(train, test)
    train, test = D.to_categorical(train, test)
    feats = D.feature_columns(train)

    model = LGBMClassifier(random_state=seed, verbose=-1)
    model.fit(train[feats], train[D.TARGET])

    p_train = model.predict_proba(train[feats])[:, 1]
    p_test = model.predict_proba(test[feats])[:, 1]
    return {
        "train_auc": roc_auc_score(train[D.TARGET], p_train),
        "test_auc": roc_auc_score(test[D.TARGET], p_test),
        "test_pr_auc": average_precision_score(test[D.TARGET], p_test),
        # % de filas de test cuyo paciente también aparece en train
        "%_test_pac_en_train": 100 * test[D.GROUP_COL].isin(set(train[D.GROUP_COL])).mean(),
    }


def run():
    df = D.clean(D.load_raw())
    rows = []
    for seed in SEEDS:
        # (1) Split aleatorio estratificado por filas: ignora patient_nbr.
        tr, te = train_test_split(df, test_size=0.2, stratify=df[D.TARGET], random_state=seed)
        rows.append({"split": "aleatorio (filas)", "seed": seed, **fit_and_score(tr, te, seed)})

        # (2) Split agrupado por paciente (el de data.py).
        tr, va, te = D.grouped_split(df, seed=seed)
        D.assert_no_patient_overlap(tr, va, te)
        rows.append({"split": "agrupado (paciente)", "seed": seed,
                     **fit_and_score(pd.concat([tr, va]), te, seed)})

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 150)
    print(res.round(4).to_string(index=False))
    print("\nMedia ± desviación sobre", len(SEEDS), "semillas:")
    summary = res.drop(columns="seed").groupby("split").agg(["mean", "std"]).round(4)
    print(summary.to_string())
    return res


if __name__ == "__main__":
    run()
