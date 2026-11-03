"""
Fase 1 — Carga, limpieza, target y split del dataset UCI 296
(Diabetes 130-US Hospitals for Years 1999-2008).

Pipeline (ver `prepare_data`):
    load_raw -> filter_rows -> drop_columns -> fill_missing_as_category
             -> add_target -> grouped_split -> collapse_rare_categories

Referencia principal para varias decisiones:
    Strack, B. et al. (2014). "Impact of HbA1c Measurement on Hospital
    Readmission Rates: Analysis of 70,000 Clinical Database Patient Records".
    BioMed Research International, 2014, 781670.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Semilla global: la reutilizamos en el split y en los modelos para que
# todos los resultados sean reproducibles.
SEED = 42

# Rutas relativas a la raíz del proyecto (este fichero vive en src/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_CSV = RAW_DIR / "diabetes_130_raw.csv"

UCI_DATASET_ID = 296

# Marcadores de "dato faltante" que usa este dataset. ucimlrepo ya convierte
# los "?" a NaN, pero 'gender' tiene "Unknown/Invalid", que también es missing.
MISSING_MARKERS = ["?", "Unknown/Invalid"]

# discharge_disposition_id que indican fallecimiento (11, 19, 20, 21) o alta a
# cuidados paliativos/hospicio (13, 14). Estos pacientes no pueden reingresar,
# así que el target no tiene sentido para ellos (el código 11 tiene 0,00% de
# positivos). Seguimos a Strack et al. (2014), que los excluye igual.
# Significados en IDs_mapping.csv de UCI.
DEATH_HOSPICE_CODES = {11, 13, 14, 19, 20, 21}

# Columnas que se eliminan:
#  - weight: 97% missing, no aporta.
#  - examide, citoglipton: un único valor en todo el dataset (varianza cero).
#  - encounter_id: no se borra del todo, pasa a ser el índice (trazabilidad).
DROP_COLS = ["weight", "examide", "citoglipton"]

# Missing tratados como categoría propia (NO imputamos ni eliminamos).
#  - A1Cresult / max_glu_serum: en el CSV original el valor es "None" =
#    "prueba no realizada" (pandas lo lee como NaN). Que no se mida la HbA1c
#    es justo la hipótesis central de Strack et al., así que es información.
NOT_MEASURED_COLS = ["A1Cresult", "max_glu_serum"]
MISSING_CAT_COLS = ["medical_specialty", "payer_code", "race"]

# Columnas de alta cardinalidad donde agrupamos categorías raras en "Other".
RARE_COLLAPSE_COLS = ["medical_specialty", "payer_code"]
RARE_THRESHOLD = 0.01  # < 1% de las filas de train -> "Other"

# Códigos numéricos que en realidad son categorías nominales (el 6 no es
# "mayor" que el 1): se tratarán como categóricas al preparar el modelo.
CODED_CATEGORICAL_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

TARGET = "readmitted_30d"
GROUP_COL = "patient_nbr"  # se mantiene para agrupar el split, NO es feature
NON_FEATURES = [TARGET, GROUP_COL]


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load_raw(use_cache: bool = True) -> pd.DataFrame:
    """Descarga el dataset con ucimlrepo (o lo lee de la caché local).

    Usamos `data.original`, que contiene todas las columnas juntas
    (IDs + features + target), para no perder encounter_id ni patient_nbr.
    """
    if use_cache and RAW_CSV.exists():
        # low_memory=False evita que pandas infiera tipos distintos por trozos.
        return pd.read_csv(RAW_CSV, low_memory=False)

    from ucimlrepo import fetch_ucirepo  # solo hace falta si no hay caché

    repo = fetch_ucirepo(id=UCI_DATASET_ID)
    df = repo.data.original.copy()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    return df


# ---------------------------------------------------------------------------
# Limpieza (sin ajustar nada a los datos: son reglas fijas, válidas antes del split)
# ---------------------------------------------------------------------------
def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con gender inválido y altas por fallecimiento/hospicio."""
    keep = (df["gender"] != "Unknown/Invalid") & (
        ~df["discharge_disposition_id"].isin(DEATH_HOSPICE_CODES)
    )
    return df.loc[keep].copy()


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Quita columnas sin valor y usa encounter_id como índice (trazabilidad).

    Así cualquier predicción o explicación local se puede rastrear hasta su
    encuentro original, pero el modelo nunca ve el ID como feature.
    """
    return df.drop(columns=DROP_COLS).set_index("encounter_id")


def fill_missing_as_category(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte los missing en una categoría explícita."""
    df = df.copy()
    df[NOT_MEASURED_COLS] = df[NOT_MEASURED_COLS].fillna("NotMeasured")
    df[MISSING_CAT_COLS] = df[MISSING_CAT_COLS].fillna("Missing")
    # diag_1/2/3 (<1,5% missing) se dejan como NaN de momento: su tratamiento
    # irá junto con la agrupación ICD-9 en feature engineering.
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """readmitted_30d = 1 si reingresa en <30 días; 0 si '>30' o 'NO'.

    Se elimina 'readmitted' para que su información no se cuele como feature.
    """
    df = df.copy()
    df[TARGET] = (df["readmitted"] == "<30").astype(int)
    return df.drop(columns="readmitted")


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas las reglas fijas de limpieza (no dependen del split)."""
    df = filter_rows(df_raw)
    df = drop_columns(df)
    df = fill_missing_as_category(df)
    return add_target(df)


# ---------------------------------------------------------------------------
# Split agrupado por paciente
# ---------------------------------------------------------------------------
def _one_grouped_fold(df: pd.DataFrame, n_splits: int, seed: int):
    """Devuelve (resto, fold) usando el primer fold de StratifiedGroupKFold.

    StratifiedGroupKFold reparte PACIENTES enteros entre folds (nunca separa
    los encuentros de un mismo paciente) intentando a la vez que cada fold
    tenga una proporción de positivos parecida a la global. Con n_splits=5,
    un fold es aprox. el 20% de las filas.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, fold_idx = next(sgkf.split(df, df[TARGET], groups=df[GROUP_COL]))
    return df.iloc[rest_idx], df.iloc[fold_idx]


def grouped_split(df: pd.DataFrame, seed: int = SEED):
    """Split 60/20/20 agrupado por patient_nbr y estratificado por el target.

    1) test = 1 de 5 folds  -> 20% del total.
    2) del 80% restante, val = 1 de 4 folds -> 25% de 80% = 20% del total.
    """
    train_val, test = _one_grouped_fold(df, n_splits=5, seed=seed)
    train, val = _one_grouped_fold(train_val, n_splits=4, seed=seed)
    return train.copy(), val.copy(), test.copy()


def assert_no_patient_overlap(*splits: pd.DataFrame) -> None:
    """Falla si algún paciente aparece en más de una partición."""
    sets = [set(s[GROUP_COL]) for s in splits]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            assert not overlap, f"{len(overlap)} pacientes compartidos entre splits {i} y {j}"


# ---------------------------------------------------------------------------
# Transformaciones que SÍ se ajustan a los datos -> solo con train
# ---------------------------------------------------------------------------
def collapse_rare_categories(train: pd.DataFrame, *others: pd.DataFrame,
                             cols=RARE_COLLAPSE_COLS, threshold=RARE_THRESHOLD):
    """Agrupa en "Other" las categorías con < threshold de frecuencia EN TRAIN.

    Las categorías frecuentes se aprenden solo en train y luego se aplican a
    val/test; así las frecuencias de test no influyen en el preprocesado.
    Una categoría que no aparece en train también acaba en "Other".
    """
    out = [d.copy() for d in (train, *others)]
    for col in cols:
        freq = train[col].value_counts(normalize=True)
        keep = set(freq[freq >= threshold].index) | {"Missing"}
        for d in out:
            d[col] = d[col].where(d[col].isin(keep), "Other")
    return out


def to_categorical(train: pd.DataFrame, *others: pd.DataFrame):
    """Convierte las columnas de texto y los códigos nominales a dtype 'category'.

    LightGBM trata nativamente las columnas 'category'. Las categorías se fijan
    a partir de train: así el mismo código significa lo mismo en los tres
    splits, y un valor no visto en train pasa a NaN.
    """
    cat_cols = [c for c in train.columns
                if (train[c].dtype == object or c in CODED_CATEGORICAL_COLS)
                and c not in NON_FEATURES]
    out = [d.copy() for d in (train, *others)]
    for col in cat_cols:
        cats = pd.Index(train[col].dropna().unique())
        for d in out:
            d[col] = pd.Categorical(d[col], categories=cats)
    return out


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURES]


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
def prepare_data(seed: int = SEED):
    """Devuelve (train, val, test) limpios, con target y split agrupado."""
    df = clean(load_raw())
    train, val, test = grouped_split(df, seed=seed)
    assert_no_patient_overlap(train, val, test)
    train, val, test = collapse_rare_categories(train, val, test)
    return train, val, test


# ---------------------------------------------------------------------------
# Exploración (Fase 1a)
# ---------------------------------------------------------------------------
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta missing por columna: NaN reales + marcadores ('?', 'Unknown/Invalid')."""
    n_nan = df.isna().sum()
    n_marker = df.isin(MISSING_MARKERS).sum()
    total = n_nan + n_marker
    report = pd.DataFrame({"n_nan": n_nan, "n_marker": n_marker, "n_missing": total,
                           "pct_missing": (100 * total / len(df)).round(2)})
    return report[report["n_missing"] > 0].sort_values("n_missing", ascending=False)


def explore(df: pd.DataFrame) -> None:
    """Resumen exploratorio del dataset crudo, SIN modificarlo."""
    print(f"Shape: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print("\n--- Tipos de variable ---")
    print(df.dtypes.value_counts().to_string())
    print("\n--- Balance de 'readmitted' ---")
    print((100 * df["readmitted"].value_counts(normalize=True)).round(2).to_string())
    print("\n--- Missing por columna ---")
    print(missing_report(df).to_string())
    enc_per_pat = df.groupby("patient_nbr").size()
    print(f"\nEncuentros: {len(df):,} | Pacientes únicos: {len(enc_per_pat):,} | "
          f"Pacientes con >1 encuentro: {(enc_per_pat > 1).sum():,}")


def split_summary(**splits: pd.DataFrame) -> pd.DataFrame:
    """Tabla con tamaño, % del total, pacientes y tasa de positivos por split."""
    total = sum(len(s) for s in splits.values())
    rows = {name: {"n_filas": len(s), "%_total": round(100 * len(s) / total, 1),
                   "n_pacientes": s[GROUP_COL].nunique(), "n_pos": int(s[TARGET].sum()),
                   "%_pos": round(100 * s[TARGET].mean(), 2)}
            for name, s in splits.items()}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    raw = load_raw()
    df = clean(raw)
    print(f"Crudo: {raw.shape} -> limpio: {df.shape} "
          f"({len(raw) - len(df):,} filas eliminadas)")
    print("Missing restantes:\n", missing_report(df).to_string())

    train, val, test = prepare_data()
    print("\nOverlap de patient_nbr entre splits: ninguno (assert OK)")
    print(split_summary(train=train, val=val, test=test).to_string())

    for col in RARE_COLLAPSE_COLS:
        print(f"\n{col}: {raw[col].nunique()} categorías -> {train[col].nunique()} "
              f"tras colapsar (<{RARE_THRESHOLD:.0%} en train -> 'Other')")
        print((100 * train[col].value_counts(normalize=True)).round(1).to_string())
