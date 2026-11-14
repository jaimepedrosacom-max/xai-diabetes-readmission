"""
Fase 5B, paso 1 — Sondas de ruido sobre el ENTREGABLE: fija el conjunto interpretable.

Prerregistro en CLAUDE.md, commit `19e7ea5`, escrito ANTES de este script.

POR QUÉ ESTE PASO VA SOLO, EN SU PROPIO SCRIPT Y SU PROPIO COMMIT: la lista de variables
interpretables tiene que quedar FIJADA y commiteada antes de calcular ninguna permutation
importance. Aquí NO hay ni una línea de código de PI, a propósito: así no se puede ajustar
la lista después de ver qué variables le convienen a la comparación.

Qué hace, en este orden:
  1. Reproduce el entregable y exige su guardarraíl (123 árboles Y PR-AUC val 0,1868).
  2. Calcula el mean |SHAP| del entregable en val, con `compute_shap` y sus dos
     verificaciones internas. Es la PRIMERA vez que se calcula el ranking global de SHAP
     del entregable: la Fase 4 solo guardó la columna del destino al alta.
  3. Repite las sondas de ruido con `xai_sanity.noise_probes` TAL CUAL, sin modificar
     `xai_sanity.py`: mismas 10 semillas y misma regla `gana_a_sonda == 10`. Lo único que
     cambia son los argumentos — frames SIN `payer_code`, etiqueta CORREGIDA y el ranking
     de SHAP del entregable.
  4. Compara la lista resultante con las 13 del modelo viejo: qué entra y qué sale.
  5. Comprueba la PARADA prerregistrada: si el conjunto tiene menos de 8 variables, avisa
     y la fase se para para reconsiderar el umbral de 4 puestos de la regla (a).

DESVIACIÓN DECLARADA en el prerregistro: `xai_sanity.fit_fixed` fija 122 árboles, no 123,
y NO se cambia. Los modelos de las sondas llevan dos columnas de sonda añadidas, así que
nunca son literalmente el entregable; y dejando 122 lo único que cambia respecto a las
sondas de la Fase 3b son las FEATURES y la ETIQUETA, que es la comparación que interesa.

TEST NO SE TOCA: el tercer split se descarta en cuanto `load_model_data` lo devuelve.
"""

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

import data as D
import preprocessing as P
import target_correction as TC
import xai_sanity as XSAN
import xai_shap as XS
from phase4_test_final import ABL_TREES, ABL_VAL_PR_AUC, fit_with_guardrail
from phase5_lr_comparison import lr_feature_lists

REPORT_DIR = D.PROJECT_ROOT / "reports"

# Parada prerregistrada: por debajo de esto, el umbral de 4 puestos de la regla (a)
# deja de significar lo mismo y hay que reconsiderarlo ANTES de medir la PI.
MIN_SET_SIZE = 8

# Lista de la Fase 3b, para la comparación descriptiva. Salió del modelo de la Fase 2
# (122 árboles, las 44 features CON payer_code, target ORIGINAL).
OLD_PROBES_CSV = "xai_sanity_noise_probes.csv"


def truncated_booster(model: lgb.LGBMClassifier, L: pd.DataFrame) -> lgb.Booster:
    """Booster truncado al nº de árboles de la mejor iteración, como en la Fase 3a.

    El truncado es una garantía explícita, no una corrección: se comprueba que no cambia
    ninguna predicción. Así el nº de árboles explicados no depende del comportamiento
    interno de la librería.
    """
    booster = lgb.Booster(model_str=model.booster_.model_to_string(
        num_iteration=model.best_iteration_))
    assert booster.num_trees() == ABL_TREES, booster.num_trees()
    assert np.allclose(booster.predict(L), model.predict_proba(L)[:, 1], atol=1e-12)
    print(f"Booster truncado OK: {booster.num_trees()} árboles, mismas predicciones "
          f"(atol 1e-12)")
    return booster


def variable_type(col: str) -> str:
    """Numérica o categórica, según las listas de `preprocessing` (no por dtype)."""
    if col in P.NUM_COLS or col in P.COUNT_COLS:
        return "numerica"
    assert col in P.CAT_COLS, col
    return "categorica"


def compare_with_old(nuevas: list, mean_abs: pd.Series) -> pd.DataFrame:
    """Entradas y salidas respecto a las 13 del modelo viejo, con su mean |SHAP|.

    Es DESCRIPCIÓN del cambio de modelo (features + etiqueta), no un resultado de
    fidelidad. `payer_code` sale por fuerza: el entregable no la tiene, así que su salida
    no la deciden las sondas y se marca aparte.
    """
    old = pd.read_csv(REPORT_DIR / OLD_PROBES_CSV, index_col=0)
    viejas = list(old.index[old["gana_a_sonda"] == XSAN.N_PROBE_SEEDS])
    rows = []
    for col in sorted(set(viejas) | set(nuevas)):
        en_viejas, en_nuevas = col in viejas, col in nuevas
        if en_viejas and en_nuevas:
            estado = "se mantiene"
        elif en_nuevas:
            estado = "ENTRA"
        elif col not in mean_abs.index:
            estado = "SALE (no está en el entregable)"
        else:
            estado = "SALE"
        rows.append({
            "variable": col, "estado": estado,
            "en_las_13_viejas": en_viejas, "en_el_conjunto_nuevo": en_nuevas,
            "gana_a_sonda_viejo": int(old.loc[col, "gana_a_sonda"]) if col in old.index else None,
            "media_abs_shap_viejo": old.loc[col, "media_abs_shap"] if col in old.index else None,
            "media_abs_shap_entregable": mean_abs.get(col, np.nan),
        })
    return pd.DataFrame(rows)


def main():
    # Mismos filtros de aviso que `xai_sanity.main`, para que la salida sea legible.
    warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
    warnings.filterwarnings("ignore", message="LightGBM binary classifier", category=UserWarning)
    pd.set_option("display.width", 200)
    # El SHAP de tree_path_dependent es determinista y la aleatoriedad sin semilla que se
    # detectó en `shap` estaba en el DIBUJO del beeswarm (aquí no hay figura). Se fija
    # igualmente, porque las sondas sí usan aleatoriedad.
    np.random.seed(D.SEED)

    # --- Datos: val. El tercer split se descarta aquí mismo. --------------------
    train, val, _test = P.load_model_data()
    del _test
    flags = TC.planned_flags()
    ytr_new = TC.relabel(train, flags).to_numpy()
    yva_new = TC.relabel(val, flags).to_numpy()
    feats, _cats = lr_feature_lists()           # P.FEATURES menos payer_code
    Ltr, Lva = D.to_categorical(train[feats], val[feats])
    print(f"Val: {len(Lva)} filas, {Lva.shape[1]} variables, prevalencia "
          f"{100 * yva_new.mean():.4f}% (etiqueta corregida)")

    # --- 1. Guardarraíl del entregable -----------------------------------------
    model = fit_with_guardrail(Ltr, ytr_new, Lva, yva_new,
                               "ENTREGABLE (sin payer_code)", ABL_TREES, ABL_VAL_PR_AUC)
    booster = truncated_booster(model, Lva)

    # --- 2. SHAP del entregable (primera vez que se calcula su ranking global) --
    values, base = XS.compute_shap(booster, Lva)
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=Lva.columns)
    ranking = list(mean_abs.sort_values(ascending=False).index)
    print(f"compute_shap OK: aditividad y pred_contrib verificados; valor base {base:.4f}, "
          f"{values.shape[0]} filas x {values.shape[1]} variables")
    print(f"Variables con mean |SHAP| = 0 (el entregable no las usa): "
          f"{int((mean_abs == 0).sum())}")

    # --- 3. Sondas de ruido, con el código de la Fase 3b sin tocar -------------
    print(f"\nSondas de ruido: {XSAN.N_PROBE_SEEDS} semillas, regla gana_a_sonda == "
          f"{XSAN.N_PROBE_SEEDS}. Modelos auxiliares con {XSAN.EXPECTED_TREES} árboles "
          f"fijos (desviación declarada en el prerregistro).")
    table, seeds, floors, top_ok = XSAN.noise_probes(Ltr, ytr_new, Lva, ranking)
    print(seeds.round(5).to_string(index=False))
    print(f"Suelo (mejor sonda): media {np.mean(floors):.4f}, máximo {np.max(floors):.4f}")

    out = table.assign(
        media_abs_shap=mean_abs.reindex(table.index),
        rank_shap_entregable=mean_abs.rank(ascending=False).reindex(table.index).astype(int),
        tipo=[variable_type(c) for c in table.index],
        interpretable=table["gana_a_sonda"] == XSAN.N_PROBE_SEEDS,
    )
    nuevas = list(out.index[out["interpretable"]])
    print(f"\nTabla completa de las {len(out)} variables (ordenada por SHAP del entregable):")
    print(out.round(5).to_string())
    print(f"\nCriterio de la Fase 3b sobre las top-{XSAN.TOP_N_PROBE}: "
          f"{'PASA' if top_ok else 'NO TODAS'}")

    # --- 4. El conjunto interpretable del entregable ---------------------------
    conjunto = out.loc[nuevas, ["rank_shap_entregable", "media_abs_shap", "gana_a_sonda",
                                "tipo"]].copy()
    print(f"\n{'=' * 80}\nCONJUNTO INTERPRETABLE DE LA FASE 5B: {len(conjunto)} variables")
    print(conjunto.round(5).to_string())
    print(f"Numéricas: {(conjunto['tipo'] == 'numerica').sum()} | "
          f"Categóricas: {(conjunto['tipo'] == 'categorica').sum()}")

    # --- 5. Comparación descriptiva con las 13 del modelo viejo ----------------
    comp = compare_with_old(nuevas, mean_abs)
    print(f"\nComparación con las 13 del modelo viejo (descriptivo):")
    print(comp.round(5).to_string(index=False))
    for estado in ("ENTRA", "SALE", "SALE (no está en el entregable)"):
        cols = comp.loc[comp["estado"] == estado, "variable"].tolist()
        print(f"  {estado}: {', '.join(cols) if cols else 'ninguna'}")

    # --- 6. Parada prerregistrada ----------------------------------------------
    parada = len(conjunto) < MIN_SET_SIZE
    if parada:
        print(f"\n{'!' * 80}\nPARADA PRERREGISTRADA: el conjunto tiene {len(conjunto)} "
              f"variables, menos de {MIN_SET_SIZE}.\nLa fase SE PARA aquí: hay que "
              f"reconsiderar con Jaime el umbral de 4 puestos de la regla (a)\nANTES de "
              f"calcular ninguna PI.\n{'!' * 80}")
    else:
        print(f"\nParada por tamaño: NO se activa ({len(conjunto)} >= {MIN_SET_SIZE}). "
              f"El umbral de la regla (a) sigue siendo 4 puestos.")

    resumen = pd.DataFrame([{
        "n_variables_entregable": len(out),
        "n_interpretables": len(conjunto),
        "n_numericas": int((conjunto["tipo"] == "numerica").sum()),
        "n_categoricas": int((conjunto["tipo"] == "categorica").sum()),
        "n_interpretables_viejas": int(comp["en_las_13_viejas"].sum()),
        "entran": ", ".join(comp.loc[comp["estado"] == "ENTRA", "variable"]) or "ninguna",
        "salen": ", ".join(comp.loc[comp["estado"].str.startswith("SALE"), "variable"]) or "ninguna",
        "criterio_top10_fase3b": top_ok,
        "suelo_sonda_medio": float(np.mean(floors)),
        "min_set_size": MIN_SET_SIZE,
        "PARADA_POR_TAMANO": parada,
        "n_val": len(Lva),
        "prevalencia_val_%": 100 * yva_new.mean(),
    }])
    print("\nResumen:")
    print(resumen.T.to_string(header=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT_DIR / "phase5b_noise_probes.csv")
    seeds.assign(suelo=floors).to_csv(REPORT_DIR / "phase5b_probe_seeds.csv", index=False)
    conjunto.to_csv(REPORT_DIR / "phase5b_interpretable_set.csv")
    comp.to_csv(REPORT_DIR / "phase5b_probes_vs_old.csv", index=False)
    resumen.to_csv(REPORT_DIR / "phase5b_probes_summary.csv", index=False)
    print("\nEscritos: phase5b_noise_probes.csv, phase5b_probe_seeds.csv, "
          "phase5b_interpretable_set.csv, phase5b_probes_vs_old.csv, "
          "phase5b_probes_summary.csv")


if __name__ == "__main__":
    main()
