"""
Fase 5B, adenda — medición DESCRIPTIVA de `admission_type_id`.

Prerregistro: la adenda de CLAUDE.md, commiteada en `425019f` ANTES de esta medición.
Va en un script APARTE de la PI, como fija la adenda.

POR QUÉ: `admission_type_id` entró en el conjunto interpretable del entregable en el
puesto 3, multiplicando por 8,7 su mean |SHAP| respecto al modelo viejo (0,00860 ->
0,07525). El mecanismo candidato es que la regla de reetiquetado define la etiqueta
corregida usando `admission_type_id == 3` (programado) del encuentro SIGUIENTE
(`src/target_correction.py`), y dentro de un mismo paciente los tipos de ingreso están
correlacionados.

Las TRES medidas de la adenda, solo en train y val, NUNCA en test:
  1. Tasa de positivos eliminados por el reetiquetado según el nivel de
     `admission_type_id` ACTUAL, con denominador = positivos del nivel en train+val.
  2. Tabla cruzada del `admission_type_id` ACTUAL frente al del encuentro SIGUIENTE del
     mismo paciente, solo para los encuentros con siguiente VISIBLE, con n por celda.
  3. SHAP medio por nivel de `admission_type_id` en el entregable, que NO se recalcula:
     se LEE de `reports/phase5b_direction_categorical.csv`, que escribe el script de la
     PI. Así no hay dos cálculos del mismo número que puedan divergir.

LECTURA FIJADA EN LA ADENDA, antes de ver esto: es una DESCRIPCIÓN DEL MECANISMO, no un
contraste. No hay veredicto, ni umbral, ni p-valor. Aunque el mecanismo se confirme, NO se
concluye que la variable sea un error: si la etiqueta corregida es correcta, una pauta
clínica real (el tipo de ingreso predice el riesgo) y un eco de la regla de reetiquetado
son compatibles, y este diseño NO los separa. Nada del entregable cambia.

Nota sobre los niveles: el `admission_type_id` del encuentro siguiente se agrupa con la
MISMA regla que usa el modelo para el actual (los códigos 5, 6 y 8 de `IDS_mapping.csv`
-> `Unknown`, `preprocessing.UNKNOWN_ADMIN_CODES`), para que las dos marginales de la
tabla cruzada sean comparables.
"""

import numpy as np
import pandas as pd

import data as D
import preprocessing as P
import target_correction as TC
from audit_discharge_leakage import add_next_flags

REPORT_DIR = D.PROJECT_ROOT / "reports"
ADMISSION = "admission_type_id"
ELECTIVE = "3"            # ingreso programado: el nivel que usa la regla de reetiquetado


def group_admin_codes(serie: pd.Series) -> pd.Series:
    """Misma regla que el modelo: los códigos 'desconocido' se unen en `Unknown`."""
    unknown = P.UNKNOWN_ADMIN_CODES[ADMISSION]
    num = pd.to_numeric(serie, errors="coerce")
    return num.where(~num.isin(unknown), np.nan).map(
        lambda v: "Unknown" if pd.isna(v) else str(int(v)))


def relabel_rate_by_level(train: pd.DataFrame, val: pd.DataFrame,
                          flags: pd.DataFrame) -> pd.DataFrame:
    """(1) Positivos reetiquetados por nivel del `admission_type_id` ACTUAL (train+val)."""
    df = pd.concat([train, val])
    y_old = df[D.TARGET].to_numpy().astype(bool)
    planned = flags["next_planned"].reindex(df.index).fillna(False).to_numpy()
    tab = pd.DataFrame({"nivel": df[ADMISSION].astype(str).to_numpy(),
                        "positivo": y_old, "reetiquetada": y_old & planned})
    out = tab.groupby("nivel").agg(n_filas=("positivo", "size"),
                                   positivos=("positivo", "sum"),
                                   reetiquetadas=("reetiquetada", "sum"))
    out["tasa_%"] = 100 * out["reetiquetadas"] / out["positivos"]
    out["pct_del_total_reetiquetado"] = 100 * out["reetiquetadas"] / out["reetiquetadas"].sum()
    return out.sort_values("positivos", ascending=False)


def next_crosstab(train: pd.DataFrame, val: pd.DataFrame) -> tuple:
    """(2) Tabla cruzada actual x siguiente, solo con siguiente VISIBLE (train+val)."""
    idx = pd.concat([train, val]).index
    nxt = add_next_flags(D.load_raw())
    sub = nxt.reindex(idx)
    visible = sub["has_next"].fillna(False).to_numpy()
    actual = group_admin_codes(sub[ADMISSION])[visible]
    siguiente = group_admin_codes(sub[f"next_{ADMISSION}"])[visible]
    cruz = pd.crosstab(actual, siguiente, dropna=False)
    cruz.index.name, cruz.columns.name = "actual", "siguiente"
    # % de "siguiente programado" por nivel actual: es la fila que mira la hipótesis.
    prog = pd.DataFrame({
        "n_con_siguiente_visible": cruz.sum(axis=1),
        "n_siguiente_programado": cruz.get(ELECTIVE, pd.Series(0, index=cruz.index)),
    })
    prog["pct_siguiente_programado"] = 100 * prog["n_siguiente_programado"] / prog["n_con_siguiente_visible"]
    return cruz, prog, int(visible.sum()), len(idx)


def shap_by_level() -> pd.DataFrame:
    """(3) SHAP medio por nivel: se LEE del CSV de la PI, no se recalcula."""
    tab = pd.read_csv(REPORT_DIR / "phase5b_direction_categorical.csv")
    out = tab[tab["variable"] == ADMISSION].copy()
    assert not out.empty, "falta admission_type_id en phase5b_direction_categorical.csv"
    return out.sort_values("shap_medio", ascending=False)


def main():
    pd.set_option("display.width", 220)
    train, val, _test = P.load_model_data()     # test se descarta aquí mismo
    del _test
    flags = TC.planned_flags()
    print(f"train+val = {len(train) + len(val)} filas. TEST NO SE TOCA.")

    print("\n(1) Positivos reetiquetados por nivel del admission_type_id ACTUAL "
          "(train+val; denominador = positivos del nivel):")
    tasa = relabel_rate_by_level(train, val, flags)
    print(tasa.round(3).to_string())

    cruz, prog, n_vis, n_tot = next_crosstab(train, val)
    print(f"\n(2) Tabla cruzada actual x siguiente, n por celda. Solo encuentros con "
          f"siguiente VISIBLE: {n_vis} de {n_tot} ({100 * n_vis / n_tot:.2f}%).")
    print(cruz.to_string())
    print(f"\n    Y la fila que mira la hipótesis: % de siguiente PROGRAMADO "
          f"(nivel {ELECTIVE}) por nivel actual, denominador = n con siguiente visible:")
    print(prog.round(3).to_string())

    print("\n(3) SHAP medio por nivel de admission_type_id en el entregable "
          "(leído de phase5b_direction_categorical.csv, no recalculado):")
    shap_lv = shap_by_level()
    print(shap_lv.round(5).to_string(index=False))

    print("\nLectura fijada en la adenda y NO revisada al ver esto: es descripción del "
          "mecanismo, no\nun contraste. Aunque el mecanismo se confirme, NO se concluye "
          "que la variable sea un error:\nuna pauta clínica real y un eco de la regla de "
          "reetiquetado son compatibles y este diseño\nno los separa. Nada del entregable "
          "cambia.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tasa.to_csv(REPORT_DIR / "phase5b_admtype_relabel_rate.csv")
    cruz.to_csv(REPORT_DIR / "phase5b_admtype_crosstab.csv")
    prog.to_csv(REPORT_DIR / "phase5b_admtype_next_elective.csv")
    shap_lv.to_csv(REPORT_DIR / "phase5b_admtype_shap_levels.csv", index=False)
    print("\nEscritos: phase5b_admtype_relabel_rate.csv, phase5b_admtype_crosstab.csv, "
          "phase5b_admtype_next_elective.csv, phase5b_admtype_shap_levels.csv")


if __name__ == "__main__":
    main()
