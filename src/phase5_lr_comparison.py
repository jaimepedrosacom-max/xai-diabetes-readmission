"""
Fase 5A — el entregable (LightGBM corregido SIN `payer_code`) frente a una regresión
logística. Lo pide el profesor: si LightGBM es claramente superior, eso apunta a no
linealidad en los datos.

Prerregistro en CLAUDE.md, commit `c579acf`, escrito ANTES de este script.

DOS ETAPAS, en commits distintos y con una sola orden de entrada (`--stage`):

  --stage val   Verifica el guardarraíl del entregable, comprueba que la LR de la
                Fase 2a se reproduce tras parametrizar el preprocesado, elige C con la
                etiqueta CORREGIDA y sin `payer_code`, y escribe la rejilla y los
                coeficientes. NO CARGA TEST.

  --stage test  Exige con `assert` los DOS guardarraíles (el del entregable y el de la
                LR, ya anotado por la etapa val) y solo entonces puntúa test, UNA vez.
                NO ESTÁ IMPLEMENTADA EN ESTE COMMIT, a propósito.

POR QUÉ DOS ETAPAS Y NO UNA: si la elección de C tuviera un error, con una sola
ejecución lo descubriríamos con test ya mirado. Separándolas, la etapa val se revisa
antes de que exista código capaz de leer test.

EN ESTA ETAPA TEST NO SE TOCA, y no es solo una promesa del docstring: `stage_val` tira
el tercer split en cuanto `load_model_data` lo devuelve, no calcula su etiqueta
corregida y no construye sus frames. La función `stage_test` aborta.

El preprocesado de la LR es el de la Fase 2a tal cual (`P.build_lr_preprocessor`):
`StandardScaler` en las numéricas, `log1p` + `StandardScaler` en los tres contadores y
one-hot con `min_frequency=0.01`, `handle_unknown="infrequent_if_exist"` y
`drop="if_binary"` en las categóricas. Lo único que cambia respecto a la Fase 2a es que
`payer_code` sale de la lista de categóricas, para igualar las variables del entregable.
Ninguno de los dos modelos usa reponderado de clases.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import data as D
import preprocessing as P
import target_correction as TC
from phase4_ablation import ABLATED
from phase4_test_final import ABL_TREES, ABL_VAL_PR_AUC, fit_with_guardrail
from train_baseline import C_GRID, select_logreg

REPORT_DIR = D.PROJECT_ROOT / "reports"

# --- Guardarraíl de la Fase 2a, para auditar la parametrización del preprocesado ------
# Fuente: reports/metrics_final.csv, fila `LogReg,val` (PR-AUC 0,2123530892664265), y el
# `assert logreg.named_steps["clf"].C == 1` de evaluate_test.py. Condiciones de la Fase
# 2a: TODAS las features (con `payer_code`) y la etiqueta VIEJA.
LR_2A_C = 1.0
LR_2A_VAL_PR_AUC = 0.2124

# --- Guardarraíl NUEVO de la LR de la Fase 5A -----------------------------------------
# Anotados tras la PRIMERA ejecución de `--stage val` (igual que se hizo con los 123
# árboles del entregable en la ablación). De aquí en adelante se exigen con `assert`:
# si no cuadran, la LR no es la que se cerró en val y la etapa test no se ejecuta.
# C = 1 NO cae en un extremo de C_GRID, así que no hay ganador en el borde; coincide con
# el C de la Fase 2a, pero eso es una coincidencia y no una herencia: aquí se re-eligió
# con la etiqueta corregida y sin `payer_code` (rejilla en phase5_lr_c_grid.csv).
LR_C = 1.0
LR_VAL_PR_AUC = 0.1804
LR_N_COLS = 128          # columnas tras el one-hot, de 43 variables de entrada


def lr_feature_lists() -> tuple:
    """Columnas de la LR de la Fase 5A: las del entregable, es decir sin `payer_code`.

    Se devuelven las dos cosas que hacen falta: la lista de features (para recortar X) y
    la lista de categóricas (para el `ColumnTransformer`). `payer_code` se quita de las
    DOS, así que no entra ni por la puerta de atrás del `remainder`.
    """
    feats = [c for c in P.FEATURES if c != ABLATED]
    cats = [c for c in P.CAT_COLS if c != ABLATED]
    assert ABLATED not in feats and ABLATED not in cats
    return feats, cats


def original_var(name: str, cat_cols: list) -> str:
    """Variable original a la que pertenece una columna del ColumnTransformer.

    Los nombres llegan como "num__age", "counts__number_inpatient" o
    "cat__A1Cresult_>8". En el bloque `cat` hay que buscar el prefijo porque el nivel
    puede contener guiones bajos. Mismo criterio que `xai_shap.lr_contributions`.
    """
    bloque, resto = name.split("__", 1)
    if bloque in ("num", "counts"):
        return resto
    matches = [c for c in cat_cols if resto.startswith(c + "_")]
    assert len(matches) == 1, f"nombre ambiguo: {name}"
    return matches[0]


def coefficient_table(logreg, cat_cols: list) -> pd.DataFrame:
    """Coeficientes de la LR, para la Fase 5B. En la 5A NO se interpretan.

    Aviso para la 5B: las escalas no son comparables entre sí (las numéricas van por
    desviación típica, las dummies por categoría), `drop="if_binary"` solo elimina una
    columna en las binarias, así que hay colinealidad que la L2 reparte entre dummies, y
    el nivel `infrequent` actúa como referencia implícita en varias variables.
    """
    prep, clf = logreg.named_steps["prep"], logreg.named_steps["clf"]
    names = prep.get_feature_names_out()
    coef = clf.coef_[0]
    tab = pd.DataFrame({
        "columna": names,
        "bloque": [n.split("__", 1)[0] for n in names],
        "variable_original": [original_var(n, cat_cols) for n in names],
        "coef": coef,
        "odds_ratio": np.exp(coef),
    })
    tab["rank_abs"] = tab["coef"].abs().rank(ascending=False).astype(int)
    # El intercepto no es de ninguna variable, pero sin él los coeficientes no
    # reconstruyen ninguna predicción; va como fila aparte y se marca como tal.
    intercepto = pd.DataFrame([{
        "columna": "(intercepto)", "bloque": "intercepto",
        "variable_original": "(intercepto)", "coef": float(clf.intercept_[0]),
        "odds_ratio": float(np.exp(clf.intercept_[0])), "rank_abs": 0,
    }])
    return pd.concat([intercepto, tab.sort_values("rank_abs")], ignore_index=True)


def stage_val() -> None:
    """Etapa val: guardarraíles, auditoría del refactor y elección de C. Sin test."""
    pd.set_option("display.width", 220)

    # El tercer split se descarta AQUÍ MISMO: en esta etapa no se calcula su etiqueta
    # corregida, no se construyen sus frames y no se puntúa nada sobre él.
    train, val, _test = P.load_model_data()
    del _test

    flags = TC.planned_flags()
    (Xtr_all, ytr_old), (Xva_all, yva_old) = P.split_xy(train), P.split_xy(val)
    ytr_new = TC.relabel(train, flags).to_numpy()
    yva_new = TC.relabel(val, flags).to_numpy()
    feats, cats = lr_feature_lists()

    # --- 1. Guardarraíl del entregable --------------------------------------------
    # No se reentrena nada: se REPRODUCE el modelo cerrado en val y se exige que sea el
    # mismo (123 árboles Y PR-AUC val 0,1868). Si no cuadra, la Fase 5A no sigue.
    Ltr_s, Lva_s = D.to_categorical(train[feats], val[feats])
    fit_with_guardrail(Ltr_s, ytr_new, Lva_s, yva_new,
                       "ENTREGABLE (sin payer_code)", ABL_TREES, ABL_VAL_PR_AUC)

    # --- 2. Auditoría del refactor: la llamada SIN argumentos reproduce la Fase 2a --
    # Mismas condiciones que entonces: TODAS las features (con `payer_code`) y la
    # etiqueta VIEJA. Si esto falla, parametrizar el preprocesado cambió algo y ninguna
    # cifra publicada de la LR seguiría siendo la documentada.
    lr_2a, grid_2a = select_logreg(Xtr_all, ytr_old, Xva_all, yva_old)
    c_2a = lr_2a.named_steps["clf"].C
    pr_2a = float(grid_2a.loc[grid_2a["C"] == c_2a, "val_pr_auc"].iloc[0])
    assert c_2a == LR_2A_C, c_2a
    assert round(pr_2a, 4) == LR_2A_VAL_PR_AUC, pr_2a
    print(f"Refactor auditado OK — LR de la Fase 2a reproducida: C = {c_2a:g}, "
          f"PR-AUC val {pr_2a:.4f}")

    # --- 3. Elección de C: etiqueta CORREGIDA, sin `payer_code` --------------------
    # Mismo procedimiento que el entregable: se entrena SOLO con train, se elige por
    # PR-AUC en val y no se reentrena con train+val. `C_GRID` es la rejilla de la Fase
    # 2a, fijada en el prerregistro; no se amplía pase lo que pase.
    Xtr, Xva = Xtr_all[feats], Xva_all[feats]
    lr, grid = select_logreg(Xtr, ytr_new, Xva, yva_new, cat_cols=cats)
    best_c = lr.named_steps["clf"].C
    n_cols = len(lr.named_steps["prep"].get_feature_names_out())

    prev = yva_new.mean()
    grid = grid.assign(
        n_val=len(yva_new),
        val_prevalencia_pct=100 * prev,
        val_lift=grid["val_pr_auc"] / prev,
        # El one-hot no depende de C, así que la columna vale lo mismo en las 5 filas;
        # se repite para que cada fila del CSV sea legible por sí sola.
        n_columnas_tras_onehot=n_cols,
        elegido=lambda g: g["C"] == best_c,
    )
    en_el_borde = best_c in (min(C_GRID), max(C_GRID))
    grid["ganador_en_el_borde"] = en_el_borde

    print("\nRejilla de C (entrena en train, elige por PR-AUC en val, etiqueta "
          "corregida, sin payer_code):")
    print(grid.round(4).to_string(index=False))
    print(f"=> C elegido = {best_c:g}"
          + (" — GANADOR EN EL BORDE de la rejilla, que NO se amplía (prerregistro)"
             if en_el_borde else ""))
    if best_c != LR_2A_C:
        print(f"   OJO: la Fase 2a eligió C = {LR_2A_C:g} con la etiqueta vieja y con "
              f"payer_code. Cambió la etiqueta y cambiaron las features.")

    # --- 4. Guardarraíl NUEVO de la LR, para anotarlo -----------------------------
    val_pr = float(grid.loc[grid["elegido"], "val_pr_auc"].iloc[0])
    print(f"\nGuardarraíl NUEVO de la LR (ANÓTALO en el script y en CLAUDE.md): "
          f"LR_C = {best_c:g}, LR_VAL_PR_AUC = {val_pr:.4f}, LR_N_COLS = {n_cols}")
    if LR_C is not None:
        assert best_c == LR_C and round(val_pr, 4) == LR_VAL_PR_AUC and n_cols == LR_N_COLS

    # Comprobación de que la métrica es la misma que usa el early stopping de LightGBM
    # y toda la Fase 4: average_precision_score, sin variantes interpoladas.
    assert np.isclose(val_pr,
                      average_precision_score(yva_new, lr.predict_proba(Xva)[:, 1]))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(REPORT_DIR / "phase5_lr_c_grid.csv", index=False)
    coefficient_table(lr, cats).to_csv(REPORT_DIR / "phase5_lr_coefficients.csv",
                                       index=False)
    print(f"\nEscritos: phase5_lr_c_grid.csv ({len(grid)} filas) y "
          f"phase5_lr_coefficients.csv ({n_cols} columnas + intercepto)")


def stage_test() -> None:
    """Etapa test: deliberadamente NO implementada en este commit."""
    raise SystemExit(
        "La etapa test no está implementada en este commit, y es a propósito.\n"
        "El prerregistro (CLAUDE.md, commit c579acf) separa las dos etapas para que un\n"
        "error en la elección de C no se descubra con test ya mirado. Primero se revisa\n"
        "la salida de `--stage val`; la etapa test llega en el commit siguiente, y\n"
        "exigirá con assert el guardarraíl del entregable Y el de la LR anotado aquí."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stage", required=True, choices=["val", "test"],
                    help="val: elige C sin tocar test. test: consulta única a test.")
    stage = ap.parse_args().stage
    print(f"Fase 5A, etapa: {stage}")
    (stage_val if stage == "val" else stage_test)()


if __name__ == "__main__":
    main()
