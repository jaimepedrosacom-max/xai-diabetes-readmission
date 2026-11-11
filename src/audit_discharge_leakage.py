"""
Auditoría del destino al alta: ¿parte de los "reingresos" son cuidados programados?

Origen: SHAP señaló el alta a rehabilitación (código 22) como la señal de riesgo
más fuerte entre los destinos con n >= 50 en val. Si esos "reingresos" son continuación de cuidados registrada como un
encuentro nuevo, sería una correlación espuria por cómo se define la etiqueta.

Limitación de partida: el dataset NO tiene fechas ni días hasta el reingreso, ni
identificador de hospital. La aproximación es mirar el SIGUIENTE encuentro del
mismo paciente en los casos etiquetados "<30":
  - V57 como diagnóstico principal: ICD-9 "care involving use of rehabilitation
    procedures".
  - admission_type_id == 3: ingreso programado ("Elective" en IDS_mapping.csv).

Supuesto: encounter_id ordena los encuentros en el tiempo. No está documentado, así
que se valida indirectamente (un encuentro "NO" casi nunca debería tener otro detrás).

Son indicios, no prueba: el 28,2% de los "<30" no tiene encuentro posterior visible
(2.548 de 9.051 en train+val), y un ingreso programado puede no tener relación con el
episodio anterior. Por destino el hueco es peor justo donde está el hallazgo: 37,8% en
el código 5 y 36,0% en el 22, frente al 27,3% de las altas a casa.

Solo se usan pacientes de train y val: ninguna fila de test entra en el cálculo.
"""

import pandas as pd

import data as D
from preprocessing import UNKNOWN_ADMIN_CODES

MIN_N = 30          # destinos con menos filas en train+val no se muestran
V57 = "V57"
ELECTIVE = 3
ER_SOURCE = 7       # admission_source_id: "Emergency Room"
REFERENCE = "1"     # alta a casa


def add_next_flags(raw: pd.DataFrame) -> pd.DataFrame:
    """Añade los datos del encuentro SIGUIENTE de cada paciente y las banderas.

    Se extrajo de `load_audit_frame` para que la Fase 4 reetiquete con exactamente
    esta lógica y no con una copia (ver `src/target_correction.py`). Al extraerla se
    relanzó este script y sus tres CSV salieron idénticos byte a byte.

    `raw` debe venir del crudo (así se ven también los encuentros que la limpieza
    excluye, que igual sirven como "siguiente") y se ordena por paciente y
    encounter_id, el supuesto temporal que valida `validate_order`.
    """
    raw = raw.sort_values([D.GROUP_COL, "encounter_id"]).copy()
    by_patient = raw.groupby(D.GROUP_COL)
    for col in ["encounter_id", "admission_type_id", "admission_source_id", "diag_1"]:
        raw[f"next_{col}"] = by_patient[col].shift(-1)
    raw["has_next"] = raw["next_encounter_id"].notna()
    raw["next_v57"] = raw["next_diag_1"].astype(str).str.startswith(V57)
    raw["next_elective"] = raw["next_admission_type_id"] == ELECTIVE
    raw["next_planned"] = raw["next_v57"] | raw["next_elective"]
    raw["next_er"] = raw["next_admission_source_id"] == ER_SOURCE

    # Mismo agrupado de códigos desconocidos que usa el modelo.
    unknown = UNKNOWN_ADMIN_CODES["discharge_disposition_id"]
    raw["destino"] = raw["discharge_disposition_id"].where(
        ~raw["discharge_disposition_id"].isin(unknown), "Unknown").astype(str)
    return raw.set_index("encounter_id")


def load_audit_frame():
    """Encuentros de train y val con los datos de su encuentro siguiente."""
    train, val, _test = D.grouped_split(D.clean(D.load_raw()))
    keep = set(train[D.GROUP_COL]) | set(val[D.GROUP_COL])

    # Se parte del crudo para ver también encuentros excluidos de la limpieza,
    # pero solo de estos pacientes (el split agrupado deja fuera a los de test).
    raw = D.load_raw()
    raw = add_next_flags(raw[raw[D.GROUP_COL].isin(keep)])
    return raw, raw.loc[train.index], raw.loc[val.index]


def validate_order(raw: pd.DataFrame) -> dict:
    """Si encounter_id ordena en el tiempo, "NO" apenas debería tener encuentro detrás."""
    return {"pct_NO_si_hay_posterior": 100 * (raw.loc[raw["has_next"], "readmitted"] == "NO").mean(),
            "pct_NO_si_es_el_ultimo": 100 * (raw.loc[~raw["has_next"], "readmitted"] == "NO").mean(),
            **{f"pct_posterior_si_{k}": 100 * raw.loc[raw["readmitted"] == k, "has_next"].mean()
               for k in ("<30", ">30", "NO")}}


def summarize(df: pd.DataFrame) -> dict:
    """Fila de resultados de un grupo de encuentros."""
    pos = df[df["readmitted"] == "<30"]
    vis = pos[pos["has_next"]]
    n = len(df)
    return {
        "n": n,
        "pct_lt30": 100 * len(pos) / n,
        "n_lt30": len(pos),
        "n_visibles": len(vis),
        "pct_V57": 100 * vis["next_v57"].mean(),
        "pct_programado": 100 * vis["next_elective"].mean(),
        "pct_V57_o_prog": 100 * vis["next_planned"].mean(),
        "pct_urgencias": 100 * vis["next_er"].mean(),
        # Tasa si se descontaran los casos visibles señalados. Es una cota: los "<30"
        # sin encuentro visible se mantienen, porque no sabemos qué son.
        "tasa_sin_V57": 100 * (len(pos) - vis["next_v57"].sum()) / n,
        "tasa_sin_V57_o_prog": 100 * (len(pos) - vis["next_planned"].sum()) / n,
    }


def sweep(df: pd.DataFrame) -> pd.DataFrame:
    rows = {d: summarize(g) for d, g in df.groupby("destino") if len(g) >= MIN_N}
    return pd.DataFrame(rows).T


def main():
    pd.set_option("display.width", 190)
    raw, train, val = load_audit_frame()
    both = pd.concat([train, val])

    print("Validación del orden (pacientes de train+val):")
    for k, v in validate_order(raw).items():
        print(f"  {k}: {v:.1f}")

    # SHAP medio por destino, de la Fase 3a (solo val). Puede faltar en niveles sin SHAP.
    levels = pd.read_csv(D.PROJECT_ROOT / "reports" / "xai_shap_levels.csv", dtype={"nivel": str})
    shap_by_dest = levels[levels["variable"] == "discharge_disposition_id"].set_index("nivel")

    table = sweep(both)
    table["shap_medio_val"] = shap_by_dest["shap_medio"].reindex(table.index)
    table["n_val"] = shap_by_dest["n"].reindex(table.index)
    table = table.sort_values("shap_medio_val", ascending=False)
    (D.PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    table.to_csv(D.PROJECT_ROOT / "reports" / "audit_discharge_leakage.csv")

    cols = ["n", "n_val", "shap_medio_val", "pct_lt30", "n_lt30", "n_visibles", "pct_V57",
            "pct_programado", "pct_V57_o_prog", "pct_urgencias", "tasa_sin_V57", "tasa_sin_V57_o_prog"]
    print("\nBarrido por destino al alta (train+val), ordenado por SHAP:")
    print(table[cols].round(1).to_string())

    print("\nRéplica por split, destinos con más SHAP (pct_V57 / pct_V57_o_prog / n_visibles):")
    tr, va = sweep(train), sweep(val)
    rep = pd.DataFrame({"train_V57": tr["pct_V57"], "val_V57": va["pct_V57"],
                        "train_V57_o_prog": tr["pct_V57_o_prog"], "val_V57_o_prog": va["pct_V57_o_prog"],
                        "train_vis": tr["n_visibles"], "val_vis": va["n_visibles"]})
    print(rep.reindex([d for d in table.index if d in rep.index]).round(1).to_string())

    print("\nAlcance global sobre el target (train+val):")
    g = summarize(both)
    print(f"  Positivos <30: {int(g['n_lt30']):,} ({g['pct_lt30']:.2f}% de {int(g['n']):,}); "
          f"con encuentro visible: {int(g['n_visibles']):,}")
    print(f"  Siguiente encuentro V57: {g['pct_V57']:.1f}% | programado: {g['pct_programado']:.1f}% | "
          f"V57 o programado: {g['pct_V57_o_prog']:.1f}%")
    print(f"  Prevalencia descontándolos: sin V57 {g['tasa_sin_V57']:.2f}% | "
          f"sin V57 o programado {g['tasa_sin_V57_o_prog']:.2f}%")

    # Se persiste el mismo dict `g` que acaba de imprimirse, sin recalcular nada. Hace
    # falta porque audit_discharge_leakage.csv aplica MIN_N y NO cubre estas cifras:
    # sumando ese CSV salen 6.497 visibles y 19,8%, no los 6.503 y 19,9% de aquí.
    # pct_sin_visible es el complementario de n_visibles/n_lt30, el límite que se declara
    # en el informe: de esos "<30" no sabemos qué fueron porque no hay encuentro detrás.
    global_row = {k: g[k] for k in ["n", "pct_lt30", "n_lt30", "n_visibles"]}
    global_row["pct_sin_visible"] = 100 * (1 - g["n_visibles"] / g["n_lt30"])
    global_row.update({k: g[k] for k in ["pct_V57", "pct_programado", "pct_V57_o_prog",
                                         "tasa_sin_V57", "tasa_sin_V57_o_prog"]})
    pd.DataFrame([global_row], index=["train+val"]).to_csv(
        D.PROJECT_ROOT / "reports" / "audit_discharge_global.csv", index_label="grupo")

    pos = both[(both["readmitted"] == "<30") & both["has_next"]]
    planned = pos[pos["next_planned"]]
    share = planned["destino"].value_counts(normalize=True).mul(100).round(1).head(6)
    print("\n¿De dónde salen los positivos con siguiente V57 o programado? (% por destino)")
    print(share.to_string())
    print("\n¿Y los positivos con siguiente V57? (% por destino)")
    print(pos[pos["next_v57"]]["destino"].value_counts(normalize=True).mul(100).round(1).head(6).to_string())

    # Mismo reparto que los dos bloques de arriba, pero completo: la consola imprime
    # head(6) y redondeado a un decimal. Aquí van TODOS los destinos (sin MIN_N, que es
    # filtro de la tabla por destino, no de este reparto), para que los porcentajes
    # sumen 100. Se guardan también los conteos para que el denominador de cada
    # porcentaje sea explícito: suma de n_V57_o_prog y suma de n_V57.
    v57 = pos[pos["next_v57"]]["destino"].value_counts()
    prog = planned["destino"].value_counts()
    shares = pd.DataFrame({
        "n_V57_o_prog": prog,
        "pct_V57_o_prog": prog / prog.sum() * 100,
        "n_V57": v57,
        "pct_V57": v57 / v57.sum() * 100,
    }).fillna(0).astype({"n_V57_o_prog": int, "n_V57": int}).sort_values("n_V57_o_prog", ascending=False)
    shares.to_csv(D.PROJECT_ROOT / "reports" / "audit_discharge_shares.csv",
                  index_label="destino")


if __name__ == "__main__":
    main()
