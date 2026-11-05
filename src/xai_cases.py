"""
Selección de los casos para las explicaciones locales (plan de la Fase 3, sección 5).

La regla es fija, con semilla, y no mira las explicaciones finales de cada caso,
para no escoger pacientes que "cuenten una historia bonita". Solo val, y como
máximo un encuentro por paciente.

  - Perfiles de riesgo (5): 3 centroides de k-means sobre los vectores SHAP del top
    10% (por QUÉ se marca a alguien), más uno en la mediana de riesgo y otro en la
    mediana del 20% más bajo.
  - Discrepancia con la regla number_inpatient (4): modelo en top 10% con
    number_inpatient = 0, y modelo bajo la mediana con number_inpatient >= 1; en
    ambos, un paciente que reingresó y uno que no. El plan decía >= 2, pero ese
    grupo está vacío: el mínimo de esos pacientes queda en el percentil 68 de riesgo.
    >= 1 coincide además con la regla baseline (number_inpatient > 0).
  - Errores (2): la falsa alarma con mayor riesgo y el reingreso con menor riesgo.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import data as D


def select_cases(probs: np.ndarray, values: np.ndarray, X: pd.DataFrame, y: np.ndarray,
                 groups: pd.Series, seed: int = D.SEED):
    """Devuelve (casos, tamaño y tasa real de cada grupo del que se extraen)."""
    probs = pd.Series(probs, index=X.index)
    y = pd.Series(y, index=X.index)
    rng = np.random.default_rng(seed)
    p90, p50, p20 = probs.quantile(0.9), probs.median(), probs.quantile(0.2)
    top10, below_median = probs >= p90, probs < p50

    used_patients, cases, group_info = set(), [], {}

    def register(name, pool: pd.Index, chooser):
        """Elige de `pool` evitando pacientes ya usados y anota n y tasa del grupo."""
        available = pool[~groups.loc[pool].isin(used_patients)]
        encounter = chooser(available)
        used_patients.add(groups.loc[encounter])
        cases.append({"caso": name, "encounter_id": encounter, "prob": probs.loc[encounter],
                      "reingresa": int(y.loc[encounter])})
        group_info[name] = {"n_grupo": len(pool), "tasa_real_%": 100 * y.loc[pool].mean()}

    # --- Perfiles de riesgo ------------------------------------------------------
    top_idx = X.index[top10]
    top_vals = values[top10.to_numpy()]
    km = KMeans(n_clusters=3, n_init=10, random_state=seed).fit(top_vals)
    dist = km.transform(top_vals)
    for c in range(3):
        members = pd.Series(dist[km.labels_ == c, c], index=top_idx[km.labels_ == c])
        register(f"perfil_alto_{c + 1}", members.index,
                 lambda a, m=members: m.loc[a].idxmin())

    near = lambda target: (lambda a: (probs.loc[a] - target).abs().idxmin())
    register("riesgo_mediano", X.index, near(p50))
    low = X.index[probs <= p20]
    register("riesgo_bajo", low, near(probs.loc[low].median()))

    # --- Discrepancia con la regla simple ------------------------------------------
    pick = lambda a: a[rng.integers(len(a))]
    inp = X["number_inpatient"]
    register("modelo_alto_sin_ingresos_reingresa", X.index[top10 & (inp == 0) & (y == 1)], pick)
    register("modelo_alto_sin_ingresos_no_reingresa", X.index[top10 & (inp == 0) & (y == 0)], pick)
    register("modelo_bajo_con_ingresos_reingresa", X.index[below_median & (inp >= 1) & (y == 1)], pick)
    register("modelo_bajo_con_ingresos_no_reingresa", X.index[below_median & (inp >= 1) & (y == 0)], pick)

    # --- Errores más extremos ------------------------------------------------------
    register("falsa_alarma_mayor_riesgo", X.index[y == 0], lambda a: probs.loc[a].idxmax())
    register("reingreso_menor_riesgo", X.index[y == 1], lambda a: probs.loc[a].idxmin())

    return pd.DataFrame(cases), pd.DataFrame(group_info).T
