"""
Fase 2 — Preprocesado para los modelos.

Dos niveles, separados a propósito:

1) REGLAS FIJAS (`build_feature_frame`): no se ajustan a los datos, así que da
   igual aplicarlas antes o después del split. Agrupación ICD-9, códigos
   "desconocido" de los IDs administrativos, edad ordinal.

2) TRANSFORMACIONES AJUSTADAS (`build_lr_preprocessor`, `to_lgbm_frames`):
   aprenden algo de los datos (medias, desviaciones, categorías frecuentes),
   así que se ajustan SOLO con train y luego se aplican a val/test.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import data as D
from features import DIAG_COLS, add_icd9_groups

# ---------------------------------------------------------------------------
# Reglas fijas
# ---------------------------------------------------------------------------
# Códigos que en IDs_mapping.csv (zip oficial de UCI) significan
# "NULL", "Not Available", "Not Mapped" o "Unknown/Invalid". Todos dicen lo
# mismo ("no sabemos"), así que los unimos en una sola categoría "Unknown".
UNKNOWN_ADMIN_CODES = {
    "admission_type_id": {5, 6, 8},                 # Not Available, NULL, Not Mapped
    "discharge_disposition_id": {18, 25, 26},       # NULL, Not Mapped, Unknown/Invalid
    "admission_source_id": {9, 15, 17, 20, 21},     # Not Available x2, NULL, Not Mapped, Unknown/Invalid
}

# Los tramos de edad tienen un orden real -> ordinal 0..9 (no one-hot).
AGE_ORDER = [f"[{10 * i}-{10 * (i + 1)})" for i in range(10)]

# ---------------------------------------------------------------------------
# Qué tratamiento recibe cada columna (listas explícitas para poder defenderlas)
# ---------------------------------------------------------------------------
NUM_COLS = ["time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_diagnoses", "age"]
# Visitas en el año previo: casi todo ceros y cola larga -> log1p antes de escalar
# (solo para la regresión logística; a los árboles les da igual).
COUNT_COLS = ["number_outpatient", "number_emergency", "number_inpatient"]
MED_COLS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
            "insulin", "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
CAT_COLS = (["race", "gender", "payer_code", "medical_specialty",
             "admission_type_id", "discharge_disposition_id", "admission_source_id",
             "A1Cresult", "max_glu_serum", "change", "diabetesMed"]
            + [f"{c}_group" for c in DIAG_COLS] + MED_COLS)
FEATURES = NUM_COLS + COUNT_COLS + CAT_COLS


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas fijas y deja solo FEATURES + target + grupo."""
    df = add_icd9_groups(df)
    for col, codes in UNKNOWN_ADMIN_CODES.items():
        # A texto: son categorías, y así "Unknown" convive con "1", "3", ...
        df[col] = df[col].where(~df[col].isin(codes), "Unknown").astype(str)
    df["age"] = df["age"].map({a: i for i, a in enumerate(AGE_ORDER)})
    assert df["age"].notna().all(), "tramo de edad no reconocido"

    df = df.drop(columns=DIAG_COLS)  # los sustituyen los *_group
    # Guardarraíl: cada columna tiene exactamente UN tratamiento asignado.
    assert len(FEATURES) == len(set(FEATURES))
    leftover = set(df.columns) - set(FEATURES) - set(D.NON_FEATURES)
    assert not leftover, f"columnas sin tratamiento asignado: {leftover}"
    return df[FEATURES + D.NON_FEATURES]


def load_model_data(seed: int = D.SEED):
    """train/val/test listos: limpieza + split agrupado + colapso (en train) + reglas fijas."""
    train, val, test = D.prepare_data(seed=seed)
    return tuple(build_feature_frame(s) for s in (train, val, test))


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[D.TARGET].to_numpy()


# ---------------------------------------------------------------------------
# Transformaciones ajustadas (solo con train)
# ---------------------------------------------------------------------------
def build_lr_preprocessor() -> ColumnTransformer:
    """ColumnTransformer para la regresión logística.

    Al ir dentro de un Pipeline con el clasificador, `pipeline.fit(X_train)`
    ajusta scaler y one-hot SOLO con train; val/test solo se transforman.
    """
    counts = Pipeline([
        # feature_names_out="one-to-one" mantiene los nombres de columna (útil para XAI).
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    onehot = OneHotEncoder(
        min_frequency=0.01,                    # niveles <1% en train -> "infrequent"
        handle_unknown="infrequent_if_exist",  # nivel no visto en train -> "infrequent"
        drop="if_binary",                      # binarias (gender, change...) -> 1 columna
        sparse_output=False,
    )
    return ColumnTransformer(
        [("num", StandardScaler(), NUM_COLS),
         ("counts", counts, COUNT_COLS),
         ("cat", onehot, CAT_COLS)],
        remainder="drop",              # nada fuera de FEATURES entra al modelo
        verbose_feature_names_out=True,  # nombres tipo "cat__A1Cresult_>8"
    )


def to_lgbm_frames(train: pd.DataFrame, *others: pd.DataFrame):
    """Para LightGBM: categóricas nativas con categorías fijadas en train."""
    cat_train, *cat_others = D.to_categorical(train[FEATURES], *(o[FEATURES] for o in others))
    return [cat_train, *cat_others]
