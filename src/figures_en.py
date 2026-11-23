"""
Versiones EN INGLÉS de las tres figuras que van a la memoria (cambio COSMÉTICO).

Por qué existe este script y no se traducen las figuras en el sitio: el notebook
(notebooks/analisis.ipynb) está en español y muestra esas figuras ya ejecutadas, y se
decidió NO reejecutarlo. Así que el idioma es un parámetro de las funciones de dibujo y
aquí solo se las llama con lang="en". Los ficheros en español NO se tocan: la versión
inglesa sale con sufijo _en.

NO SE RECALCULA NADA. Ni modelo, ni SHAP, ni bootstrap, ni CSV. Las tres figuras se
dibujan leyendo CSV ya publicados:
  - shap_global_bar_en      <- reports/xai_global_comparison.csv (columna shap_lgbm)
  - phase4_fig_A_..._en     <- phase4_prediction_gaps.csv, phase4_test_gaps.csv,
                               phase4_paired_contrasts.csv, phase4_test_paired_contrasts.csv
  - phase4_fig_B_..._en     <- phase4_ablation_composition.csv
Lo único que cambia respecto a las versiones en español son cadenas de texto y el
formato de número (punto decimal y coma de miles, como la memoria en inglés).

Las figuras A y B ya se dibujaban solo desde CSV (phase4_figures.py). La global de SHAP
la dibujaba xai_shap.py a partir de los valores SHAP recién calculados; aquí se le pasa
la MISMA media de |SHAP| leída de xai_global_comparison.csv, que es la que escribió esa
misma ejecución. Así no hace falta reentrenar el modelo de la Fase 2 ni recalcular
TreeSHAP para cambiar un rótulo.
"""
import pandas as pd

import data as D
import phase4_figures as F4
import preprocessing as P
import xai_shap as XS

REPORTS = D.PROJECT_ROOT / "reports"

# Filas de val. No se lee del dataset (eso sería cargar datos para un cambio de texto):
# se deriva de los tamaños de grupo de payer_code, que particionan val sin solapamiento.
# El assert deja fijo el valor conocido del split, igual que los guardarraíles del resto
# del proyecto.
N_VAL_ESPERADO = 19868


def val_rows() -> int:
    comp = pd.read_csv(REPORTS / "phase4_ablation_composition.csv")
    n = int(comp["n_grupo"].sum())
    assert n == N_VAL_ESPERADO, f"filas de val = {n}, se esperaban {N_VAL_ESPERADO}"
    return n


def shap_global_bar_en() -> str:
    """Figura global de SHAP en inglés, desde el CSV de la Fase 3a."""
    cmp = pd.read_csv(REPORTS / "xai_global_comparison.csv", index_col=0)
    # El CSV está ordenado por SHAP descendente, pero la figura original se dibujó con
    # la Series en el orden de las COLUMNAS del modelo. Importa porque 12 variables
    # empatan en |SHAP| exactamente 0 y sort_values de pandas es estable: con otro orden
    # de entrada, esas 12 barras (de longitud 0) saldrían con sus etiquetas en otro
    # orden. Se reindexa a P.FEATURES para que el empate se rompa igual que en la figura
    # en español. Verificado: así la versión es-redibujada sale idéntica byte a byte.
    mean_abs = cmp["shap_lgbm"].reindex(P.FEATURES)
    # Las 44 variables del modelo de la Fase 2 (el que dibuja esta figura).
    assert len(mean_abs) == 44, f"se esperaban 44 variables, hay {len(mean_abs)}"
    assert mean_abs.notna().all(), "alguna variable de P.FEATURES no está en el CSV"
    # plot_global_bar no usa `base`; se pasa NaN para no inventar un valor que no está
    # en el CSV y que la figura no dibuja.
    path = XS.plot_global_bar(mean_abs, base=float("nan"), n=val_rows(), lang="en")
    return path.name


def main():
    print("Escrito:", XS.E.FIG_DIR / shap_global_bar_en())
    print("Escrito:", F4.FIG_DIR / F4.figure_A(lang="en"))
    print("Escrito:", F4.FIG_DIR / F4.figure_B(lang="en"))


if __name__ == "__main__":
    main()
