"""
Fase 1c — Agrupación de los diagnósticos ICD-9 (diag_1, diag_2, diag_3).

El mapeo reproduce la Tabla 2 de:
    Strack, B. et al. (2014). "Impact of HbA1c Measurement on Hospital
    Readmission Rates: Analysis of 70,000 Clinical Database Patient Records".
    BioMed Research International, 2014, 781670.

Por qué no hay riesgo de leakage: la tabla de rangos es FIJA (está escrita
aquí a mano, no se calcula con nuestros datos), así que transformar train,
val y test con ella da lo mismo antes o después del split.

Detalles del paper:
  - Los códigos 785, 786, 787 y 788 (síntomas circulatorios, respiratorios,
    digestivos y genitourinarios) se asignan a su aparato, no a "Other".
  - Los códigos E (causas externas) y V (clasificación suplementaria) tienen
    su propia fila en la Tabla 2 (1,3% de encuentros) y NO van a Injury; el
    paper los agrupa en "Other" junto con todos los grupos de <3,5%.
    Ese umbral del 3,5% lo calcularon los autores sobre SU muestra; nosotros
    usamos la lista de grupos resultante como regla fija, sin recalcularlo.
  - El paper agrupa SOLO el diagnóstico principal (diag_1). Aplicar el mismo
    mapeo a diag_2 y diag_3 es decisión nuestra (extensión del paper).
"""

import numpy as np
import pandas as pd

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

# (categoría, lista de rangos [inicio, fin] INCLUSIVOS sobre la parte entera
# del código ICD-9). Ej.: "428.0" -> 428 -> Circulatory.
ICD9_GROUPS = [
    ("Circulatory",     [(390, 459), (785, 785)]),
    ("Respiratory",     [(460, 519), (786, 786)]),
    ("Digestive",       [(520, 579), (787, 787)]),
    ("Diabetes",        [(250, 250)]),           # 250.xx
    ("Injury",          [(800, 999)]),
    ("Musculoskeletal", [(710, 739)]),
    ("Genitourinary",   [(580, 629), (788, 788)]),
    ("Neoplasms",       [(140, 239)]),
]

# EXTENSIÓN NUESTRA (no está en el paper), solo para diag_2 y diag_3.
# Motivo: en los diagnósticos secundarios, 240-279 (endocrinas, nutricionales
# y metabólicas sin diabetes: hiperlipidemia, obesidad, trastornos
# hidroelectrolíticos...) es ~8-9% de los encuentros y es el bloque más grande
# dentro de "Other"; son comorbilidades clínicamente relevantes.
# Se comprueba DESPUÉS de ICD9_GROUPS: así el 250 ya ha caído en "Diabetes"
# y "Endocrine" equivale a "240-279 sin 250", como en la Tabla 2 del paper.
ENDOCRINE_GROUP = [("Endocrine", [(240, 279)])]

# diag_1 replica el paper al dígito; diag_2/3 añaden "Endocrine".
GROUPS_BY_COL = {
    "diag_1": ICD9_GROUPS,
    "diag_2": ICD9_GROUPS + ENDOCRINE_GROUP,
    "diag_3": ICD9_GROUPS + ENDOCRINE_GROUP,
}


def icd9_category(code, groups=ICD9_GROUPS) -> str:
    """Devuelve la categoría clínica de un código ICD-9 (como texto).

    - NaN -> "Missing" (el paciente no tiene ese diagnóstico secundario).
    - Empieza por 'E' o 'V' -> "Other" (ver docstring del módulo).
    - Numérico -> se compara su parte entera con los rangos de `groups`
      EN ORDEN (gana el primero que encaja); si no cae en ninguno -> "Other".
    """
    if pd.isna(code):
        return "Missing"
    code = str(code).strip()
    if code[:1].upper() in ("E", "V"):
        return "Other"
    try:
        num = int(np.floor(float(code)))
    except ValueError:
        # No debería pasar; si pasa, lo veremos en el recuento de "Other".
        return "Other"
    for group, ranges in groups:
        if any(lo <= num <= hi for lo, hi in ranges):
            return group
    return "Other"


def add_icd9_groups(df: pd.DataFrame, groups_by_col=GROUPS_BY_COL) -> pd.DataFrame:
    """Añade columnas <col>_group SIN borrar los códigos originales.

    Mantenemos los originales de momento para poder auditar el mapeo; la
    decisión de con cuáles se queda el modelo la tomaremos en el encoding.
    """
    df = df.copy()
    for col, groups in groups_by_col.items():
        df[f"{col}_group"] = df[col].map(lambda c: icd9_category(c, groups))
    return df


# --- Utilidades de auditoría (solo para revisar el mapeo) -------------------
def _other_breakdown(codes: pd.Series) -> pd.Series:
    """Desglosa los códigos que caen en "Other" por tipo, para revisar el mapeo."""
    def kind(c):
        c = str(c)
        if c[0] in "EV":
            return f"{c[0]}-codes"
        n = int(float(c))
        bands = [(1, 139, "001-139 Infecciosas"), (240, 279, "240-279 Endocrinas (sin 250)"),
                 (280, 289, "280-289 Sangre"), (290, 319, "290-319 Mentales"),
                 (320, 359, "320-359 Nervioso"), (360, 389, "360-389 Sentidos"),
                 (630, 679, "630-679 Embarazo"), (680, 709, "680-709 Piel"),
                 (740, 759, "740-759 Congénitas"), (780, 799, "780-799 Síntomas mal definidos")]
        return next((name for lo, hi, name in bands if lo <= n <= hi), "sin rango (revisar)")
    return codes.map(kind).value_counts()


if __name__ == "__main__":
    import data as D

    df = add_icd9_groups(D.clean(D.load_raw()))
    pd.set_option("display.width", 150)

    dist = pd.DataFrame({c: 100 * df[f"{c}_group"].value_counts(normalize=True) for c in DIAG_COLS})
    print("Distribución de categorías (% de encuentros):")
    print(dist.round(2).sort_values("diag_1", ascending=False).to_string())

    print("\nCódigos en 'Other' (excluye Missing):")
    for c in DIAG_COLS:
        other = df.loc[df[f"{c}_group"] == "Other", c]
        print(f"\n{c}: {len(other):,} encuentros ({100 * len(other) / len(df):.1f}%), "
              f"{other.nunique()} códigos distintos")
        print(_other_breakdown(other).to_string())

    # Antes/después de añadir "Endocrine": mapeo solo-paper vs. extendido.
    print("\n% 'Other' (solo paper -> con Endocrine):")
    for c in DIAG_COLS:
        before = 100 * (df[c].map(icd9_category) == "Other").mean()
        after = 100 * (df[f"{c}_group"] == "Other").mean()
        print(f"  {c}: {before:.2f}% -> {after:.2f}%  (Endocrine = "
              f"{100 * (df[f'{c}_group'] == 'Endocrine').mean():.2f}%)")
    assert "Endocrine" not in set(df["diag_1_group"]), "diag_1 debe replicar el paper"

    print("\nComprobación con valores conocidos (paper | con Endocrine):")
    ext = GROUPS_BY_COL["diag_2"]
    for code in ["428", "250.83", "250", "272.4", "276", "786", "V57", "E909", "782", np.nan]:
        print(f"  {str(code):>7} -> {icd9_category(code):<12} | {icd9_category(code, ext)}")
