"""
Fase 4, punto 5 — ¿El reetiquetado explica el desplazamiento de composición?

Cuantifica con un número lo que si no sería una impresión ("encaja casi uno a uno"):
la correlación de rangos de Spearman entre
  - la tasa de reetiquetado del grupo (positivos cambiados / positivos, train+val), y
  - el factor de cambio de su tasa de marcado (tasa con el modelo corregido / tasa con el
    modelo viejo, top 10% de val),
sobre los 10 grupos de `payer_code`.

Se espera NEGATIVA: a más positivos reetiquetados, más baja su tasa de marcado.

Es un script aparte y de SOLO LECTURA: toma los dos CSV que ya escribió
`phase4_fairness.py` y no recalcula nada, así que no hace falta repetir el bootstrap.
Spearman (de rangos) y no Pearson porque solo se afirma el ORDEN, no una relación lineal,
y con 10 grupos un par de valores extremos dominarían una correlación lineal.

AVISO sobre el factor de cambio: la tasa de marcado se compara en cociente frente a MC
(el grupo de referencia), pero este análisis usa la tasa ABSOLUTA de cada grupo, no el
cociente. Si se usara el cociente, un grupo podría "subir" solo porque MC bajó, y lo que
se quiere medir es el movimiento propio de cada grupo.
"""

import pandas as pd
from scipy.stats import spearmanr

import data as D

REPORT_DIR = D.PROJECT_ROOT / "reports"
K = "10%"
ATTR = "payer_code"


def main():
    comp = pd.read_csv(REPORT_DIR / "phase4_fairness_composition.csv")
    comp = comp[(comp["atributo"] == ATTR) & (comp["top_k"] == K)].set_index("grupo")
    dist = pd.read_csv(REPORT_DIR / "phase4_fairness_relabel_distribution.csv")
    dist = dist[dist["atributo"] == ATTR].set_index("grupo")

    tab = pd.DataFrame({
        "tasa_reetiquetado_%": dist["tasa_%"],
        "positivos_train_val": dist["denominador"],
        "tasa_marcado_vieja_%": comp["tasa_modelo_viejo_%"],
        "tasa_marcado_nueva_%": comp["tasa_marcado_%"],
    })
    tab["factor_cambio"] = tab["tasa_marcado_nueva_%"] / tab["tasa_marcado_vieja_%"]
    tab = tab.sort_values("tasa_reetiquetado_%", ascending=False)

    rho, pval = spearmanr(tab["tasa_reetiquetado_%"], tab["factor_cambio"])
    print(tab.round(4).to_string())
    print(f"\nSpearman entre tasa de reetiquetado y factor de cambio de la tasa de "
          f"marcado: rho = {rho:.3f} (p = {pval:.4g}, n = {len(tab)} grupos de {ATTR})")

    tab.to_csv(REPORT_DIR / "phase4_relabel_vs_selection.csv", index_label="grupo")
    pd.DataFrame([{"atributo": ATTR, "n_grupos": len(tab), "spearman_rho": rho,
                   "p_valor": pval}]).to_csv(
        REPORT_DIR / "phase4_relabel_vs_selection_rho.csv", index=False)


if __name__ == "__main__":
    main()
