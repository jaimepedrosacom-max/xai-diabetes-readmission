"""
Fase 4, punto 2 — Corrección del target: reetiquetar los reingresos planificados.

Qué hace: los encuentros etiquetados `<30` cuyo SIGUIENTE encuentro tiene diagnóstico
principal V57 (rehabilitación) o es un ingreso programado (`admission_type_id == 3`)
pasan a 0. Es la opción "reetiquetar" de las tres que plantea la Fase 3b, y la elegida
porque mantiene la población: no deja sin predicción a los pacientes de esas altas.

La regla NO es nuestra invención aquí: es exactamente la que auditó la Fase 3b, y se
importa de `audit_discharge_leakage.add_next_flags` para que no haya dos copias que
puedan divergir.

TRES COSAS QUE HAY QUE TENER CLARAS PARA DEFENDERLO:

1. El SPLIT YA ESTÁ HECHO cuando se reetiqueta. `D.grouped_split` estratifica por `y`,
   así que reetiquetar antes cambiaría los folds y val dejaría de ser el mismo conjunto
   de filas: el antes/después no sería comparable. Por eso este módulo recibe los
   splits ya construidos y solo cambia la columna del target.

2. DE TEST SOLO SE APLICA LA REGLA. No se calcula ninguna estadística de test ni entra
   en ninguna decisión. Lo único que se reporta de test antes de la consulta final es el
   CONTEO de filas cambiadas, que es descripción del dataset. `changed_counts` separa a
   propósito el conteo (los tres splits) del desglose por destino (solo train y val).

3. La regla mira el encuentro SIGUIENTE, que es información futura. Es legítimo porque
   define la ETIQUETA, no una feature, pero se aplica igual en los tres splits: si se
   aplicara solo en train, los conjuntos medirían cosas distintas. Y no cruza
   información entre splits porque el siguiente encuentro es del MISMO paciente y el
   split es agrupado por `patient_nbr` (lo comprueba `assert_next_within_split`).
"""

import pandas as pd

import data as D
from audit_discharge_leakage import add_next_flags


def planned_flags() -> pd.DataFrame:
    """Banderas del encuentro siguiente para TODOS los pacientes, indexado por encuentro.

    A diferencia de `load_audit_frame`, no filtra a los pacientes de train+val: la
    corrección tiene que poder aplicarse también a test (solo la regla, ver (2) arriba).
    """
    return add_next_flags(D.load_raw())[
        ["patient_nbr", "has_next", "next_v57", "next_elective", "next_planned", "destino"]]


def assert_next_within_split(splits: dict, flags: pd.DataFrame) -> pd.DataFrame:
    """Comprueba que el encuentro siguiente nunca cae en otro split.

    Debería cumplirse por construcción (el siguiente encuentro es del mismo paciente y
    el split agrupa por `patient_nbr`), pero se verifica en vez de suponerlo: si
    fallara, la regla estaría cruzando información entre train, val y test.
    """
    owner = pd.concat([pd.Series(name, index=df.index) for name, df in splits.items()])
    nxt = add_next_flags(D.load_raw())["next_encounter_id"]
    rows = []
    for name, df in splits.items():
        sig = nxt.reindex(df.index).dropna().astype("int64")
        # Solo se pueden comparar los siguientes que sobreviven a la limpieza y están
        # en algún split; los demás son encuentros excluidos (fallecimientos, etc.).
        visibles = sig[sig.isin(owner.index)]
        destino_split = owner.reindex(visibles.to_numpy()).to_numpy()
        fuera = int((destino_split != name).sum())
        rows.append({"split": name, "n": len(df), "siguientes_en_un_split": len(visibles),
                     "siguientes_en_OTRO_split": fuera})
        assert fuera == 0, f"{name}: {fuera} encuentros siguientes caen en otro split"
    return pd.DataFrame(rows)


def relabel(df: pd.DataFrame, flags: pd.DataFrame) -> pd.Series:
    """Target corregido de un split: 0 donde el `<30` tenía siguiente encuentro planificado."""
    planned = flags["next_planned"].reindex(df.index).fillna(False).to_numpy()
    y_old = df[D.TARGET].to_numpy().astype(bool)
    return pd.Series((y_old & ~planned).astype(int), index=df.index)


def changed_counts(splits: dict, flags: pd.DataFrame) -> pd.DataFrame:
    """Filas que cambian de 1 a 0, por split, con denominador explícito."""
    rows = []
    for name, df in splits.items():
        y_old = df[D.TARGET].to_numpy()
        y_new = relabel(df, flags).to_numpy()
        cambiadas = int((y_old == 1).sum() - (y_new == 1).sum())
        rows.append({
            "split": name, "n": len(df),
            "positivos_antes": int(y_old.sum()),
            "positivos_despues": int(y_new.sum()),
            "filas_cambiadas": cambiadas,
            "pct_de_positivos_cambiados": 100 * cambiadas / y_old.sum(),
            "prevalencia_antes_%": 100 * y_old.mean(),
            "prevalencia_despues_%": 100 * y_new.mean(),
        })
    return pd.DataFrame(rows)


def changed_by_destination(splits: dict, flags: pd.DataFrame) -> pd.DataFrame:
    """Desglose por destino al alta de las filas reetiquetadas. SOLO train y val.

    Test se excluye a propósito: de test solo se reporta el conteo total (ver (2)).
    """
    use = {k: v for k, v in splits.items() if k != "test"}
    df = pd.concat(use.values())
    y_old = df[D.TARGET].to_numpy().astype(bool)
    planned = flags["next_planned"].reindex(df.index).fillna(False).to_numpy()
    destino = flags["destino"].reindex(df.index)
    tab = pd.DataFrame({"destino": destino, "pos": y_old, "cambiada": y_old & planned})
    out = tab.groupby("destino").agg(positivos=("pos", "sum"), cambiadas=("cambiada", "sum"))
    out["pct_de_sus_positivos"] = 100 * out["cambiadas"] / out["positivos"]
    out["pct_del_total_cambiado"] = 100 * out["cambiadas"] / out["cambiadas"].sum()
    return out.sort_values("cambiadas", ascending=False)
