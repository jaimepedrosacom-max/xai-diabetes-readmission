"""
Fase 5C, paso 2 — Vecino más cercano al otro lado de la frontera, con distancia de Gower.

Prerregistro: el bloque de la Fase 5C en CLAUDE.md, commiteado en `44db202`, punto (d).

PARA QUÉ: el profesor objetó que la frontera del modelo tiene que estar DENTRO de los datos.
Esto lo confirma, y sobre todo mide DE QUÉ ESTÁ HECHA la distancia hasta ella. Val para los
casos, train para el vecindario. TEST NO SE TOCA.

LO QUE NO ES EL HALLAZGO, declarado en el prerregistro ANTES de medir: el VALOR de la
distancia no dice nada por sí solo, porque que exista una fila de train por debajo del corte
es trivial (el 95% de val lo está). Lo informativo es QUÉ variables separan al caso de su
vecino, y cuántas de ellas son accionables y cuántas están congeladas. La hipótesis es que la
separación esté hecha de variables CONGELADAS (`number_inpatient`, los diagnósticos, `age`,
`number_emergency`). Se reporta salga como salga: si la separación resultara accionable, eso
CONTRADICE nuestra lectura de la Fase 3c y se dice.

LA DISTANCIA, definida en el prerregistro y no revisada aquí:
    d(i,j) = sum_v w_v * d_v(i,j) / sum_v w_v,  con w_v = 1 si la variable es COMPARABLE en
    ese par y 0 si no.
      - numéricas (P.NUM_COLS + P.COUNT_COLS, 9): d_v = |x_iv - x_jv| / rango_train(v), con
        rango = max - min calculado EN TRAIN. Una variable con rango 0 en train se declara y
        se excluye. `age` va aquí, porque la Fase 1 la dejó como ordinal 0..9, y se declara.
      - categóricas (34): 0 si el nivel es el mismo, 1 si no.
      - AUSENTES: un NaN en cualquiera de las dos filas hace la variable NO comparable en ese
        par, y queda fuera del numerador Y del denominador (Gower estándar). Se reporta
        `n_vars_comparables`. En este dataset los NaN son casi siempre niveles de val que no
        existen en train (`D.to_categorical`); los missing reales ya son niveles propios
        (`Missing`, `NotMeasured`) por la decisión de limpieza de la Fase 1.
    Rango de train y PESOS IGUALES, y no se prueban variantes: normalizar por desviación
    típica o ponderar cambiaría qué fila es «la más cercana», así que la receta se fijó antes
    de medir.

QUÉ ES «EL OTRO LADO». Son DOS cosas distintas y NO se mezclan:
    PRIMARIO, la frontera del MODELO: la fila de train con `prob_entregable < corte del top
        5% de val`. Es lo que responde a la objeción, porque la frontera es un objeto del
        modelo.
    SECUNDARIO y descriptivo: la fila de train con etiqueta CORREGIDA y = 0. Se reporta al
        lado, sin veredicto.

AVISO que hay que decir y no esconder: las puntuaciones del entregable sobre TRAIN son EN
MUESTRA (el modelo se entrenó con esas filas), así que están infladas. Eso hace el lado
«prob < corte» algo MÁS exigente de poblar, no menos, así que no favorece a nuestra
conclusión; pero se declara.

Sin veredicto, sin umbral y sin p-valor: es descripción.
"""

import numpy as np
import pandas as pd

import data as D
import preprocessing as P
import target_correction as TC
import xai_counterfactuals as XCF
from phase4_test_final import ABL_TREES, ABL_VAL_PR_AUC, fit_with_guardrail
from phase5_lr_comparison import lr_feature_lists
from phase5b_probes import truncated_booster
from phase5c_frontier import ESCALONES, K_PRIMARY, TOP_KS, space_for

REPORT_DIR = D.PROJECT_ROOT / "reports"


def numeric_ranges(Ltr: pd.DataFrame, num_cols: list) -> pd.Series:
    """Rango (max - min) de cada numérica EN TRAIN. Las de rango 0 se declaran fuera."""
    rng = Ltr[num_cols].max() - Ltr[num_cols].min()
    cero = list(rng.index[rng == 0])
    if cero:
        print(f"AVISO: variables numéricas con rango 0 en train, EXCLUIDAS de Gower: {cero}")
    return rng


class Gower:
    """Distancia de Gower de una fila contra todo un conjunto, vectorizada.

    Se precalculan las matrices una vez: las numéricas como float y las categóricas como
    CÓDIGOS enteros (-1 = NaN, que es lo que `pd.Categorical` usa para un nivel no visto en
    train). Así cada caso es una pasada sobre el array, sin bucles por fila.
    """

    def __init__(self, pool: pd.DataFrame, num_cols: list, cat_cols: list,
                 ranges: pd.Series):
        self.num_cols = [c for c in num_cols if ranges[c] > 0]
        self.excluidas = [c for c in num_cols if ranges[c] == 0]
        self.cat_cols = cat_cols
        self.ranges = ranges[self.num_cols].to_numpy(dtype=float)
        self.pool_index = pool.index
        self.num = pool[self.num_cols].to_numpy(dtype=float)
        self.cat = np.column_stack([pool[c].cat.codes.to_numpy() for c in self.cat_cols])
        self.n_vars = len(self.num_cols) + len(self.cat_cols)

    def to_vectors(self, row: pd.DataFrame) -> tuple:
        num = row[self.num_cols].to_numpy(dtype=float).ravel()
        cat = np.array([row[c].cat.codes.iloc[0] for c in self.cat_cols])
        return num, cat

    def distances(self, row: pd.DataFrame) -> tuple:
        """Devuelve (distancias, nº de variables comparables, matriz de d_v por variable)."""
        qnum, qcat = self.to_vectors(row)
        # Numéricas: |dif| / rango, y comparable si ninguno de los dos es NaN.
        dif_num = np.abs(self.num - qnum) / self.ranges
        comp_num = ~np.isnan(dif_num)
        dif_num = np.where(comp_num, dif_num, 0.0)
        # Categóricas: 0 si coinciden, 1 si no; comparable si ningún código es -1 (= NaN).
        comp_cat = (self.cat >= 0) & (qcat >= 0)
        dif_cat = np.where(self.cat != qcat, 1.0, 0.0) * comp_cat
        num_total = dif_num.sum(axis=1) + dif_cat.sum(axis=1)
        n_comp = comp_num.sum(axis=1) + comp_cat.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            dist = np.where(n_comp > 0, num_total / n_comp, np.nan)
        return dist, n_comp, (dif_num, dif_cat, comp_num, comp_cat)


def pick_neighbor(gow: Gower, row: pd.DataFrame, mask: np.ndarray) -> dict:
    """Vecino más cercano dentro de `mask`. Empates: el `encounter_id` MENOR (determinista)."""
    dist, n_comp, parts = gow.distances(row)
    cand = np.where(mask & ~np.isnan(dist))[0]
    assert len(cand) > 0, "el lado de la frontera está vacío en train"
    mejor = dist[cand].min()
    empatados = cand[dist[cand] == mejor]
    # Desempate explícito por encounter_id, para que no decida el orden de las filas.
    ids = gow.pool_index.to_numpy()[empatados]
    pos = empatados[int(np.argmin(ids))]
    return {"pos": int(pos), "dist": float(mejor), "n_comparables": int(n_comp[pos]),
            "n_empatados": len(empatados), "parts": parts}


def differing_variables(gow: Gower, row: pd.DataFrame, pool: pd.DataFrame, res: dict,
                        space: dict) -> pd.DataFrame:
    """Las variables que separan al caso de su vecino, con su contribución y si son accionables.

    Es LO INFORMATIVO de este análisis, no la distancia. `accionable_E0` sale del mismo
    `action_space` que usa todo lo demás, así que no hay dos definiciones.
    """
    pos = res["pos"]
    dif_num, dif_cat, comp_num, comp_cat = res["parts"]
    vecino = pool.index[pos]
    rows = []
    for j, col in enumerate(gow.num_cols):
        if not comp_num[pos, j] or dif_num[pos, j] == 0:
            continue
        rows.append({"variable": col, "tipo": "numérica",
                     "valor_caso": row[col].iloc[0], "valor_vecino": pool[col].iloc[pos],
                     "contribucion_gower": float(dif_num[pos, j])})
    for j, col in enumerate(gow.cat_cols):
        if not comp_cat[pos, j] or dif_cat[pos, j] == 0:
            continue
        rows.append({"variable": col, "tipo": "categórica",
                     "valor_caso": str(row[col].iloc[0]),
                     "valor_vecino": str(pool[col].iloc[pos]),
                     "contribucion_gower": float(dif_cat[pos, j])})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["accionable_E0"] = out["variable"].isin(space)
    out["vecino_encounter_id"] = int(vecino)
    # La contribución a la distancia final: cada variable pesa 1 / n_comparables.
    out["aporta_a_la_distancia"] = out["contribucion_gower"] / res["n_comparables"]
    return out.sort_values("contribucion_gower", ascending=False)


def main():
    pd.set_option("display.width", 220, "display.max_colwidth", 44)

    train, val, _test = P.load_model_data()
    del _test                                   # test se descarta aquí mismo
    flags = TC.planned_flags()
    ytr_new = TC.relabel(train, flags).to_numpy()
    yva_new = TC.relabel(val, flags).to_numpy()
    feats, _cats = lr_feature_lists()
    Ltr, Lva = D.to_categorical(train[feats], val[feats])
    print(f"Train: {len(Ltr)} filas (el vecindario). Val: {len(Lva)} filas (los casos). "
          f"TEST NO SE TOCA.")

    model = fit_with_guardrail(Ltr, ytr_new, Lva, yva_new,
                               "ENTREGABLE (sin payer_code)", ABL_TREES, ABL_VAL_PR_AUC)
    booster = truncated_booster(model, Lva)
    probs_val = pd.Series(booster.predict(Lva), index=Lva.index)
    probs_tr = pd.Series(booster.predict(Ltr), index=Ltr.index)
    cuts = {k: float(probs_val.quantile(1 - k)) for k in TOP_KS}
    cut5 = cuts[K_PRIMARY]
    print(f"\nCorte del top 5% de VAL: {cut5:.6f} ({int((probs_val >= cut5).sum())} filas "
          f"marcadas de {len(probs_val)}).")

    # Los dos lados de la frontera, que NO se mezclan.
    lado_modelo = (probs_tr < cut5).to_numpy()
    lado_etiqueta = (ytr_new == 0)
    print("Vecindarios en train: por la frontera del MODELO (prob < corte) "
          f"{int(lado_modelo.sum())} filas; por ETIQUETA corregida (y = 0) "
          f"{int(lado_etiqueta.sum())} filas.")
    print("AVISO: las puntuaciones del entregable sobre TRAIN son EN MUESTRA, así que están\n"
          "infladas. Eso hace el lado «prob < corte» más exigente de poblar, no menos.")

    num_cols = [c for c in P.NUM_COLS + P.COUNT_COLS if c in Ltr.columns]
    cat_cols = [c for c in Ltr.columns if c not in num_cols]
    ranges = numeric_ranges(Ltr, num_cols)
    print(f"\nGower sobre {len(num_cols)} numéricas (rango de train) y {len(cat_cols)} "
          f"categóricas, pesos IGUALES.")
    print("  `age` va como NUMÉRICA porque la Fase 1 la dejó ordinal 0..9.")
    print("  Rangos de train:", {c: float(ranges[c]) for c in num_cols})
    gow = Gower(Ltr, num_cols, cat_cols, ranges)

    cases = pd.read_csv(REPORT_DIR / "xai_local_cases.csv")
    filas, detalle = [], []
    for _, case in cases.iterrows():
        enc = int(case["encounter_id"])
        row = Lva.loc[[enc]]
        space = space_for(row, ESCALONES[0])
        base = {"caso": case["caso"], "encounter_id": enc,
                "prob_entregable": float(probs_val.loc[enc]),
                "marcado_top5": bool(probs_val.loc[enc] >= cut5),
                "n_vars_accionables_E0": len(space)}
        for etiqueta, mask in [("frontera del modelo (PRIMARIO)", lado_modelo),
                               ("etiqueta corregida y=0 (SECUNDARIO)", lado_etiqueta)]:
            res = pick_neighbor(gow, row, mask)
            vecino = int(Ltr.index[res["pos"]])
            dif = differing_variables(gow, row, Ltr, res, space)
            n_acc = int(dif["accionable_E0"].sum()) if not dif.empty else 0
            filas.append({**base, "lado": etiqueta, "vecino_encounter_id": vecino,
                          "gower_dist": res["dist"], "n_vars_comparables": res["n_comparables"],
                          "n_empatados": res["n_empatados"],
                          "prob_vecino": float(probs_tr.loc[vecino]),
                          "y_vecino_corregida": int(ytr_new[res["pos"]]),
                          "n_vars_distintas": len(dif),
                          "n_distintas_accionables": n_acc,
                          "n_distintas_congeladas": len(dif) - n_acc,
                          "variables_distintas_accionables": ", ".join(
                              dif.loc[dif["accionable_E0"], "variable"]) if not dif.empty else "",
                          "top3_contribuciones": "; ".join(
                              f"{r.variable}={r.contribucion_gower:.3f}"
                              for r in dif.head(3).itertuples()) if not dif.empty else ""})
            if not dif.empty:
                detalle.append(dif.assign(caso=case["caso"], encounter_id=enc, lado=etiqueta))
    vecinos = pd.DataFrame(filas)
    variables = (pd.concat(detalle, ignore_index=True) if detalle else pd.DataFrame())

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    vecinos.to_csv(REPORT_DIR / "phase5c_nearest_neighbor.csv", index=False)
    variables.to_csv(REPORT_DIR / "phase5c_nn_variables.csv", index=False)

    print("\n=== (d) VECINO MÁS CERCANO al otro lado, por Gower ===")
    print(vecinos[["caso", "lado", "marcado_top5", "gower_dist", "n_vars_comparables",
                   "n_vars_distintas", "n_distintas_accionables", "n_distintas_congeladas"]]
          .round(4).to_string(index=False))
    print("\nLas tres variables que más separan a cada caso de su vecino:")
    print(vecinos[["caso", "lado", "top3_contribuciones"]].to_string(index=False))
    print("\nVariables DISTINTAS que son ACCIONABLES en E0 (lo que decide la lectura):")
    print(vecinos[["caso", "lado", "n_distintas_accionables",
                   "variables_distintas_accionables"]].to_string(index=False))
    print("\nLECTURA fijada ANTES de medir: el VALOR de la distancia no es el hallazgo (que "
          "haya una fila de\ntrain por debajo del corte es trivial). Lo informativo es de qué "
          "está hecha: cuántas de las\nvariables que separan son accionables y cuántas están "
          "congeladas. Sin veredicto y sin p-valor.")
    print("\nEscritos: phase5c_nearest_neighbor.csv, phase5c_nn_variables.csv")


if __name__ == "__main__":
    main()
