"""
Fase 5B, paso 2 — Fidelidad: permutation importance frente a SHAP en el entregable.

Prerregistro en CLAUDE.md, commit `19e7ea5`. La lista de variables interpretables NO se
calcula aquí: se LEE de `reports/phase5b_interpretable_set.csv`, que fijó y commiteó el
paso 1 (`src/phase5b_probes.py`, commit `425019f`) antes de que existiera este código.

Qué se mide, todo en val y con la etiqueta corregida (TEST NO SE TOCA):
  1. mean |SHAP| del entregable, con `compute_shap` y sus dos verificaciones internas.
  2. PI con `permutation_importance` de sklearn, R = 30 repeticiones, semilla D.SEED y las
     DOS métricas en una sola llamada (`average_precision` y `neg_log_loss`) para que
     compartan permutación.
  3. Regla de discrepancia por variable, métrica PRINCIPAL PR-AUC: discrepa si (a) la
     diferencia de rangos PI-SHAP dentro del conjunto es >= 4, o (b) está en el top 5 de
     SHAP y su PI no es distinta de cero.
  4. Log-loss: la misma regla, como ROBUSTEZ descriptiva, para señalar en qué variables
     las dos métricas difieren.
  5. Concordancia de Spearman entre los dos rankings (descriptiva, sin veredicto).
  6. Dirección: Spearman valor-SHAP en las numéricas y SHAP medio por nivel en las
     categóricas. Y el SIGNO del coeficiente de la LR de la 5A, solo en las numéricas.

SOBRE EL SIGNO DE `neg_log_loss`: `permutation_importance` define la importancia como
`score_base - score_permutado`, y con `neg_log_loss` el score es `-log_loss`. Al
desarrollarlo, la importancia es `log_loss_permutado - log_loss_base`, es decir
**directamente el AUMENTO de log-loss**. Más alto = la variable importa más, igual que en
PR-AUC. Se deja escrito porque es fácil leerlo al revés.

«PI NO DISTINTA DE CERO» es el criterio ya publicado de la Fase 3a: la variable no pasa
`perm_clara`, es decir media <= 2 x desviación típica de las R repeticiones. Limitación
declarada en el prerregistro: esa desviación es la de las permutaciones con val FIJO, así
que recoge solo el ruido de la permutación y no dice nada sobre si la PI se replicaría en
otra muestra. El percentil 2,5 va al lado como columna descriptiva y no decide nada.

Este script NO emite ningún juicio clínico ni asigna explicaciones a las discrepancias:
las cuatro explicaciones candidatas están prerregistradas y la lectura la hacemos nosotros.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

import data as D
import preprocessing as P
import target_correction as TC
import xai_shap as XS
from phase4_test_final import ABL_TREES, ABL_VAL_PR_AUC, fit_with_guardrail
from phase5b_probes import MIN_SET_SIZE, truncated_booster, variable_type
from phase5_lr_comparison import lr_feature_lists

REPORT_DIR = D.PROJECT_ROOT / "reports"

N_REPEATS = 30            # R del prerregistro
RANK_GAP = 4              # umbral de la regla (a)
TOP_SHAP_FOR_RULE_B = 5   # top 5 de SHAP para la regla (b)
CLEAR_SD_FACTOR = 2       # `perm_clara` de la Fase 3a: media > 2 x desviación
METRICS = {"average_precision": "prauc", "neg_log_loss": "logloss"}


def load_interpretable_set(columns) -> pd.DataFrame:
    """Lee el conjunto interpretable del paso 1. NO lo recalcula.

    Se exige que todas estén en el entregable y que el conjunto siga pasando la parada
    prerregistrada por tamaño: si alguien editara el CSV a mano, el script falla aquí.
    """
    tab = pd.read_csv(REPORT_DIR / "phase5b_interpretable_set.csv", index_col=0)
    faltan = [c for c in tab.index if c not in columns]
    assert not faltan, f"variables del CSV que no están en el entregable: {faltan}"
    assert (tab["gana_a_sonda"] == 10).all(), "el CSV trae variables que no pasan 10/10"
    assert len(tab) >= MIN_SET_SIZE, (len(tab), MIN_SET_SIZE)
    return tab


def run_pi(model, L: pd.DataFrame, y: np.ndarray, mean_abs: pd.Series) -> dict:
    """PI con las dos métricas sobre la MISMA permutación, y sus tres comprobaciones."""
    multi = permutation_importance(model, L, y, scoring=list(METRICS), n_repeats=N_REPEATS,
                                   random_state=D.SEED, n_jobs=1)
    # Comprobación 1: la llamada multimétrica no cambia las permutaciones respecto a una
    # llamada de una sola métrica (misma semilla, mismo orden de consumo del rng).
    solo = permutation_importance(model, L, y, scoring="average_precision",
                                  n_repeats=N_REPEATS, random_state=D.SEED, n_jobs=1)
    assert np.allclose(multi["average_precision"].importances, solo.importances), \
        "la llamada multimétrica no coincide con la de una sola métrica"
    print(f"Comprobación OK — multimétrica == una sola métrica: las {N_REPEATS} "
          f"repeticiones de PR-AUC son idénticas")

    # Comprobación 2 (nulo interno): lo que el modelo no usa no puede tener PI.
    sin_usar = list(mean_abs.index[mean_abs == 0])
    for scorer in METRICS:
        imp = pd.DataFrame(multi[scorer].importances, index=L.columns)
        assert (imp.loc[sin_usar] == 0).all().all(), f"PI no nula en variables sin usar ({scorer})"
    print(f"Comprobación OK — nulo interno: las {len(sin_usar)} variables con mean |SHAP| "
          f"= 0 dan PI exactamente 0 en las {N_REPEATS} repeticiones, en las dos métricas")
    return {corto: multi[scorer].importances for scorer, corto in METRICS.items()}


def pi_table(imps: dict, columns) -> pd.DataFrame:
    """Media, desviación, percentil 2,5 y `perm_clara` por variable y métrica."""
    out = pd.DataFrame(index=columns)
    for corto, arr in imps.items():
        out[f"pi_{corto}_media"] = arr.mean(axis=1)
        out[f"pi_{corto}_sd"] = arr.std(axis=1, ddof=0)
        out[f"pi_{corto}_p2_5"] = np.percentile(arr, 2.5, axis=1)
        # `perm_clara` de la Fase 3a: media > 2 x desviación. Si NO se cumple, la PI se
        # considera «no distinta de cero» para la regla (b).
        out[f"pi_{corto}_clara"] = (out[f"pi_{corto}_media"]
                                    > CLEAR_SD_FACTOR * out[f"pi_{corto}_sd"])
    return out


def verdict_table(conjunto: list, mean_abs: pd.Series, pi: pd.DataFrame) -> pd.DataFrame:
    """Regla de discrepancia dentro del conjunto interpretable. PR-AUC es la principal."""
    t = pd.DataFrame(index=pd.Index(conjunto, name="variable"))
    t["tipo"] = [variable_type(c) for c in conjunto]
    t["mean_abs_shap"] = mean_abs.reindex(conjunto)
    # Rangos DENTRO del conjunto, 1 = más importante, empates por rango medio.
    t["rank_shap"] = t["mean_abs_shap"].rank(ascending=False)
    for corto in ("prauc", "logloss"):
        media = pi[f"pi_{corto}_media"].reindex(conjunto)
        t[f"pi_{corto}_media"] = media
        t[f"pi_{corto}_sd"] = pi[f"pi_{corto}_sd"].reindex(conjunto)
        t[f"pi_{corto}_p2_5"] = pi[f"pi_{corto}_p2_5"].reindex(conjunto)
        t[f"rank_pi_{corto}"] = media.rank(ascending=False)
        t[f"dif_rangos_{corto}"] = (t[f"rank_pi_{corto}"] - t["rank_shap"]).abs()
        distinta = pi[f"pi_{corto}_clara"].reindex(conjunto)
        t[f"pi_distinta_de_cero_{corto}"] = distinta
        a = t[f"dif_rangos_{corto}"] >= RANK_GAP
        b = (t["rank_shap"] <= TOP_SHAP_FOR_RULE_B) & ~distinta
        t[f"discrepa_rango_{corto}"] = a
        t[f"discrepa_cero_{corto}"] = b
        t[f"DISCREPA_{corto}"] = a | b
    t["coinciden_las_dos_metricas"] = t["DISCREPA_prauc"] == t["DISCREPA_logloss"]
    return t.sort_values("rank_shap")


def direction_numeric(conjunto: list, values: np.ndarray, L: pd.DataFrame) -> pd.DataFrame:
    """Signo de Spearman entre el valor de la variable y su SHAP, y signo del coef de la LR.

    Solo el SIGNO es comparable con la LR: sus numéricas van escaladas (y los contadores
    con `log1p` antes), pero las dos transformaciones son monótonas crecientes, así que no
    pueden invertir el signo.
    """
    coefs = pd.read_csv(REPORT_DIR / "phase5_lr_coefficients.csv")
    coefs = coefs[coefs["bloque"].isin(["num", "counts"])]
    rows = []
    for col in [c for c in conjunto if variable_type(c) == "numerica"]:
        sv = values[:, L.columns.get_loc(col)]
        rho = spearmanr(L[col].to_numpy(dtype=float), sv).statistic
        fila = coefs[coefs["variable_original"] == col]
        assert len(fila) == 1, (col, len(fila))
        coef = float(fila["coef"].iloc[0])
        rows.append({"variable": col, "n_val": len(L), "rho_valor_vs_shap": rho,
                     "signo_shap": int(np.sign(rho)), "bloque_lr": fila["bloque"].iloc[0],
                     "coef_lr": coef, "signo_coef_lr": int(np.sign(coef)),
                     "coinciden_signos": int(np.sign(rho)) == int(np.sign(coef))})
    return pd.DataFrame(rows)


def direction_categorical(conjunto: list, values: np.ndarray, L: pd.DataFrame) -> pd.DataFrame:
    """SHAP medio por nivel, con su n y su % de val. Reutiliza `xai_shap.level_table`."""
    cats = [c for c in conjunto if variable_type(c) == "categorica"]
    tab = XS.level_table(values, L)
    tab = tab[tab["variable"].isin(cats)].copy()
    tab["pct_de_val"] = 100 * tab["n"] / len(L)
    return tab.sort_values(["variable", "shap_medio"], ascending=[True, False])


def main():
    warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
    warnings.filterwarnings("ignore", message="LightGBM binary classifier", category=UserWarning)
    pd.set_option("display.width", 250)
    np.random.seed(D.SEED)

    # --- Datos: val. Test se descarta aquí mismo. ------------------------------
    train, val, _test = P.load_model_data()
    del _test
    flags = TC.planned_flags()
    ytr_new = TC.relabel(train, flags).to_numpy()
    yva_new = TC.relabel(val, flags).to_numpy()
    feats, _cats = lr_feature_lists()
    Ltr, Lva = D.to_categorical(train[feats], val[feats])

    # --- Guardarraíles del entregable -----------------------------------------
    model = fit_with_guardrail(Ltr, ytr_new, Lva, yva_new,
                               "ENTREGABLE (sin payer_code)", ABL_TREES, ABL_VAL_PR_AUC)
    booster = truncated_booster(model, Lva)
    values, base = XS.compute_shap(booster, Lva)
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=Lva.columns)
    print(f"compute_shap OK: aditividad y pred_contrib verificados; valor base {base:.4f}")

    # --- La lista viene del paso 1, y se comprueba que es el MISMO SHAP --------
    fijado = load_interpretable_set(Lva.columns)
    conjunto = list(fijado.index)
    assert np.allclose(mean_abs.reindex(conjunto).to_numpy(),
                       fijado["media_abs_shap"].to_numpy(), atol=1e-12), \
        "el mean |SHAP| de ahora no es el que fijó la lista en el paso 1"
    print(f"Lista leída de phase5b_interpretable_set.csv: {len(conjunto)} variables, y su "
          f"mean |SHAP| reproduce el del paso 1 (atol 1e-12)")

    # --- Línea base de la PI: tiene que ser el guardarraíl --------------------
    p_val = model.predict_proba(Lva)[:, 1]
    base_pr = average_precision_score(yva_new, p_val)
    assert round(base_pr, 4) == ABL_VAL_PR_AUC, base_pr
    print(f"Comprobación OK — línea base de la PI: PR-AUC val {base_pr:.4f} = guardarraíl "
          f"{ABL_VAL_PR_AUC}")

    # --- PI de las 43 (contexto) ----------------------------------------------
    print(f"\nPI: {Lva.shape[1]} variables x {N_REPEATS} repeticiones, semilla {D.SEED}, "
          f"dos métricas en una sola llamada.")
    imps = run_pi(model, Lva, yva_new, mean_abs)
    pi = pi_table(imps, Lva.columns)

    contexto = pi.assign(
        mean_abs_shap=mean_abs,
        rank_shap_43=mean_abs.rank(ascending=False).astype(int),
        rank_pi_prauc_43=pi["pi_prauc_media"].rank(ascending=False).astype(int),
        tipo=[variable_type(c) for c in pi.index],
        en_el_conjunto=[c in conjunto for c in pi.index],
        R=N_REPEATS, n_val=len(Lva), pr_auc_base=base_pr,
    ).sort_values("mean_abs_shap", ascending=False)
    print("\nContexto: las 43 variables, ordenadas por SHAP (primeras 15):")
    print(contexto.head(15)[["mean_abs_shap", "rank_shap_43", "pi_prauc_media",
                             "pi_prauc_sd", "pi_prauc_clara", "rank_pi_prauc_43",
                             "pi_logloss_media", "en_el_conjunto"]].round(5).to_string())

    # --- La tabla del veredicto ------------------------------------------------
    ver = verdict_table(conjunto, mean_abs, pi)
    print(f"\n{'=' * 110}\nTABLA DEL VEREDICTO — métrica PRINCIPAL PR-AUC "
          f"(regla (a) >= {RANK_GAP} puestos; regla (b) top {TOP_SHAP_FOR_RULE_B} de SHAP "
          f"y PI no distinta de cero)")
    print(ver[["tipo", "mean_abs_shap", "rank_shap", "pi_prauc_media", "pi_prauc_sd",
               "pi_prauc_p2_5", "rank_pi_prauc", "dif_rangos_prauc",
               "pi_distinta_de_cero_prauc", "discrepa_rango_prauc", "discrepa_cero_prauc",
               "DISCREPA_prauc"]].round(5).to_string())
    disc = list(ver.index[ver["DISCREPA_prauc"]])
    print(f"\nDiscrepan (PR-AUC, la principal): {', '.join(disc) if disc else 'ninguna'} "
          f"-> {len(disc)} de {len(ver)}")

    print("\nRobustez con log-loss (DESCRIPTIVA, sin veredicto):")
    print(ver[["rank_shap", "pi_logloss_media", "pi_logloss_sd", "rank_pi_logloss",
               "dif_rangos_logloss", "pi_distinta_de_cero_logloss", "DISCREPA_logloss",
               "coinciden_las_dos_metricas"]].round(5).to_string())
    difieren = list(ver.index[~ver["coinciden_las_dos_metricas"]])
    print(f"Variables donde las dos métricas dan resultados distintos: "
          f"{', '.join(difieren) if difieren else 'ninguna'}")

    # --- Concordancia global (descriptiva) ------------------------------------
    conc = []
    for nombre, idx in [("conjunto interpretable", conjunto), ("43 contexto", list(pi.index))]:
        a = mean_abs.reindex(idx)
        b = pi["pi_prauc_media"].reindex(idx)
        r = spearmanr(a, b)
        conc.append({"conjunto": nombre, "n_variables": len(idx),
                     "spearman_rho": r.statistic, "p_valor": r.pvalue,
                     "rol": "descriptivo/sin veredicto"})
    conc = pd.DataFrame(conc)
    print("\nConcordancia de Spearman entre los rankings de SHAP y de PI (descriptiva):")
    print(conc.round(5).to_string(index=False))

    # --- Dirección de los efectos --------------------------------------------
    dnum = direction_numeric(conjunto, values, Lva)
    print("\nDirección, numéricas (signo de Spearman valor-SHAP y signo del coef de la LR):")
    print(dnum.round(5).to_string(index=False))
    dcat = direction_categorical(conjunto, values, Lva)
    print("\nDirección, categóricas (SHAP medio por nivel, con su n):")
    print(dcat.round(5).to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    contexto.to_csv(REPORT_DIR / "phase5b_pi_all.csv")
    ver.to_csv(REPORT_DIR / "phase5b_faithfulness.csv")
    conc.to_csv(REPORT_DIR / "phase5b_concordance.csv", index=False)
    dnum.to_csv(REPORT_DIR / "phase5b_direction_numeric.csv", index=False)
    dcat.to_csv(REPORT_DIR / "phase5b_direction_categorical.csv", index=False)
    print("\nEscritos: phase5b_pi_all.csv, phase5b_faithfulness.csv, "
          "phase5b_concordance.csv, phase5b_direction_numeric.csv, "
          "phase5b_direction_categorical.csv")


if __name__ == "__main__":
    main()
