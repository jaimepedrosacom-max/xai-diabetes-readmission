"""
Fase 4, punto 1 — Cifras "ANTES" de la predicción prerregistrada.

Qué mide, literalmente lo que fija la predicción prerregistrada de la Fase 3d:
  1. Top 10% de val por puntuación del modelo, con su propio corte.
  2. Ese top 10% partido en dos subgrupos: A = number_inpatient == 0 (los que la regla
     simple NO marca) y B = number_inpatient >= 1 (los que SÍ marca).
  3. En cada subgrupo, la PRECISIÓN: % de reingresos <30 d reales entre los marcados.
     Se usa precisión y no PR-AUC porque dentro de cada subgrupo la regla es constante
     y su PR-AUC sería degenerado.
La hipótesis es que, al corregir el target, la precisión caerá MÁS en A que en B
(delta_A > delta_B). Aquí SOLO se mide el "antes": este script no reetiqueta nada.

Se reporta además la BRECHA precision_A - precision_B, que es la cantidad que contrasta
la hipótesis de verdad (ver el prerregistro en CLAUDE.md): la pregunta es si la ventaja
del modelo en A se sostiene con la etiqueta limpia. Este es el paso (1) de los tres del
prerregistro (modelo viejo / etiqueta vieja); los pasos (2) y (3) se miden al aplicar la
corrección, y no se tocan aquí.

Corte del top 10%: `probs >= probs.quantile(0.9)`, el MISMO criterio que usó
`xai_cases.select_cases` en la Fase 3b. Así estas cifras reconcilian con los artefactos
ya publicados, y de hecho eso sirve de comprobación: el subgrupo A tiene que salir
n = 440 con un 69,3% de destinos 22/5, que es lo que reporta la Fase 3d.
(No se usa `evaluation.top_k_table`, que marca exactamente ceil(k*n) filas con
desempate aleatorio: daría un conjunto ligeramente distinto al de la Fase 3b.)

Guardarraíl: antes de medir se comprueba que el modelo reentrenado es exactamente el
documentado en la Fase 2b (122 árboles, PR-AUC en val 0,2308). Los modelos no están
serializados, así que reentrenar es la única forma de partir del artefacto actual.

TEST NO SE TOCA en este script. Solo train (para ajustar) y val (para medir).
"""

import numpy as np
import pandas as pd

import data as D
import preprocessing as P
from evaluate_test import CHOSEN, EXPECTED_TREES, EXPECTED_VAL_PR_AUC
from train_lgbm import fit_lgbm

REPORT_DIR = D.PROJECT_ROOT / "reports"
TOP_Q = 0.9          # corte del top 10%, igual que en xai_cases
# Códigos de destino al alta que la auditoría de la Fase 3b señala como contaminados.
CONTAMINATED = {"22", "5"}


def subgroup_rows(marcados: pd.DataFrame, momento: str) -> list:
    """Precisión en A (number_inpatient == 0) y B (>= 1) dentro de los marcados.

    `marcados` trae una fila por encuentro marcado, con `y` (reingreso real),
    `number_inpatient` y `destino` (para la composición 22/5 que documenta la Fase 3d).
    """
    rows = []
    for nombre, mask in [("A (number_inpatient == 0)", marcados["number_inpatient"] == 0),
                         ("B (number_inpatient >= 1)", marcados["number_inpatient"] >= 1)]:
        g = marcados[mask]
        rows.append({
            "momento": momento,
            "subgrupo": nombre,
            "n_marcados": len(g),
            "aciertos": int(g["y"].sum()),
            "precision": g["y"].mean() if len(g) else np.nan,
            # Composición del subgrupo: es el dato que hace que la predicción sea
            # interpretable (si A es casi todo 22/5, la corrección le afecta más).
            "pct_destino_22_o_5": 100 * g["destino"].isin(CONTAMINATED).mean() if len(g) else np.nan,
        })
    return rows


def main():
    train, val, _test = P.load_model_data()          # test se carga pero NO se usa
    (Xtr, ytr), (Xva, yva) = (P.split_xy(s) for s in (train, val))
    Ltr, Lva = P.to_lgbm_frames(train, val)          # categorías fijadas con train

    lgbm = fit_lgbm(Ltr, ytr, Lva, yva, **CHOSEN)
    val_pr_auc = lgbm.best_score_["val"]["average_precision"]
    assert lgbm.best_iteration_ == EXPECTED_TREES, lgbm.best_iteration_
    assert round(val_pr_auc, 4) == EXPECTED_VAL_PR_AUC, val_pr_auc
    print(f"Guardarraíl OK: {lgbm.best_iteration_} árboles, PR-AUC val = {val_pr_auc:.4f}")

    probs = pd.Series(lgbm.predict_proba(Lva)[:, 1], index=Xva.index)
    corte = probs.quantile(TOP_Q)
    top = probs >= corte
    marcados = pd.DataFrame({
        "y": yva,
        "number_inpatient": Xva["number_inpatient"].to_numpy(),
        "destino": Xva["discharge_disposition_id"].astype(str).to_numpy(),
    }, index=Xva.index)[top.to_numpy()]

    print(f"\nTop 10% de val: corte = {corte:.4f}, marcados = {len(marcados)} "
          f"de {len(probs)} ({100 * len(marcados) / len(probs):.2f}%)")

    rows = subgroup_rows(marcados, "antes")
    # La regla simple como referencia: en A no marca a nadie y en B marca a todos, así
    # que su precisión en B es la tasa real del subgrupo y en A no está definida.
    tabla = pd.DataFrame(rows)
    tabla["precision_%"] = 100 * tabla["precision"]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(REPORT_DIR / "phase4_prediction.csv", index=False)

    pd.set_option("display.width", 160)
    print("\nCifras ANTES de la corrección (val, top 10%):")
    print(tabla[["momento", "subgrupo", "n_marcados", "aciertos", "precision_%",
                 "pct_destino_22_o_5"]].round(2).to_string(index=False))

    # Brecha A - B del paso (1). Se compara con la de los pasos (2) y (3) en la Fase 4.
    pa, pb = tabla.loc[0, "precision"], tabla.loc[1, "precision"]
    print(f"\nBrecha del paso (1), modelo viejo / etiqueta vieja: "
          f"precision_A - precision_B = {100 * (pa - pb):+.2f} puntos")

    # Comprobación de reconciliación con la Fase 3d, que reporta n=440 y 69,3%.
    a = tabla.iloc[0]
    print(f"\nReconciliación con la Fase 3d: subgrupo A n={int(a['n_marcados'])} "
          f"(esperado 440), 22/5 = {a['pct_destino_22_o_5']:.1f}% (esperado 69,3%)")


if __name__ == "__main__":
    main()
