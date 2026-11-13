"""
Fase 5A — el entregable (LightGBM corregido SIN `payer_code`) frente a una regresión
logística. Lo pide el profesor: si LightGBM es claramente superior, eso apunta a no
linealidad en los datos.

Prerregistro en CLAUDE.md, commit `c579acf`, escrito ANTES de este script.

DOS ETAPAS, en commits distintos y con una sola orden de entrada (`--stage`):

  --stage val   Verifica el guardarraíl del entregable, comprueba que la LR de la
                Fase 2a se reproduce tras parametrizar el preprocesado, elige C con la
                etiqueta CORREGIDA y sin `payer_code`, y escribe la rejilla y los
                coeficientes. NO CARGA TEST.

  --stage test  Exige con `assert` los DOS guardarraíles (el del entregable y el de la
                LR, ya anotado por la etapa val) y solo entonces puntúa test, UNA vez.
                TERCERA consulta a test del proyecto. Nada cambia después de verla.

POR QUÉ DOS ETAPAS Y NO UNA: si la elección de C tuviera un error, con una sola
ejecución lo descubriríamos con test ya mirado. Separándolas, la etapa val se revisa
antes de que exista código capaz de leer test.

EN LA ETAPA VAL TEST NO SE TOCA, y no es solo una promesa del docstring: `stage_val`
tira el tercer split en cuanto `load_model_data` lo devuelve, no calcula su etiqueta
corregida y no construye sus frames.

El preprocesado de la LR es el de la Fase 2a tal cual (`P.build_lr_preprocessor`):
`StandardScaler` en las numéricas, `log1p` + `StandardScaler` en los tres contadores y
one-hot con `min_frequency=0.01`, `handle_unknown="infrequent_if_exist"` y
`drop="if_binary"` en las categóricas. Lo único que cambia respecto a la Fase 2a es que
`payer_code` sale de la lista de categóricas, para igualar las variables del entregable.
Ninguno de los dos modelos usa reponderado de clases.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import data as D
import evaluation as E
import preprocessing as P
import target_correction as TC
from phase4_ablation import ABLATED
from phase4_fairness import boot_indices, patient_blocks
from phase4_test_final import (ABL_TREES, ABL_VAL_PR_AUC, fit_with_guardrail,
                               global_row, topk_rows)
from train_baseline import C_GRID, select_logreg

REPORT_DIR = D.PROJECT_ROOT / "reports"

# --- Contraste primario, prerregistrado ------------------------------------------------
N_REP = 2000             # mismas réplicas que toda la Fase 4
MATERIALITY = 0.01       # umbral de materialidad del Delta de PR-AUC
K_SECONDARY = 0.05       # el top-k de la secundaria con IC; 10% y 20% van como contexto
DECISION_THRESHOLD = 0.5  # umbral de decisión de la accuracy, en los DOS modelos

# --- Cifras publicadas del entregable en test (Fase 4, punto 4) -----------------------
# Fuente: reports/phase4_test_global.csv, fila `ENTREGABLE (sin payer_code)`
# (pr_auc 0,18897616914934168 y roc_auc 0,6654651595328367), redondeadas a 4 decimales.
# Se exigen con `assert` justo después de puntuar test: si el entregable no reproduce sus
# propias cifras, las predicciones no son las de la Fase 4 y el contraste no se calcula.
TEST_PR_AUC_F4 = 0.1890
TEST_ROC_AUC_F4 = 0.6655

# --- Guardarraíl de la Fase 2a, para auditar la parametrización del preprocesado ------
# Fuente: reports/metrics_final.csv, fila `LogReg,val` (PR-AUC 0,2123530892664265), y el
# `assert logreg.named_steps["clf"].C == 1` de evaluate_test.py. Condiciones de la Fase
# 2a: TODAS las features (con `payer_code`) y la etiqueta VIEJA.
LR_2A_C = 1.0
LR_2A_VAL_PR_AUC = 0.2124

# --- Guardarraíl NUEVO de la LR de la Fase 5A -----------------------------------------
# Anotados tras la PRIMERA ejecución de `--stage val` (igual que se hizo con los 123
# árboles del entregable en la ablación). De aquí en adelante se exigen con `assert`:
# si no cuadran, la LR no es la que se cerró en val y la etapa test no se ejecuta.
# C = 1 NO cae en un extremo de C_GRID, así que no hay ganador en el borde; coincide con
# el C de la Fase 2a, pero eso es una coincidencia y no una herencia: aquí se re-eligió
# con la etiqueta corregida y sin `payer_code` (rejilla en phase5_lr_c_grid.csv).
LR_C = 1.0
LR_VAL_PR_AUC = 0.1804
LR_N_COLS = 128          # columnas tras el one-hot, de 43 variables de entrada


def lr_feature_lists() -> tuple:
    """Columnas de la LR de la Fase 5A: las del entregable, es decir sin `payer_code`.

    Se devuelven las dos cosas que hacen falta: la lista de features (para recortar X) y
    la lista de categóricas (para el `ColumnTransformer`). `payer_code` se quita de las
    DOS, así que no entra ni por la puerta de atrás del `remainder`.
    """
    feats = [c for c in P.FEATURES if c != ABLATED]
    cats = [c for c in P.CAT_COLS if c != ABLATED]
    assert ABLATED not in feats and ABLATED not in cats
    return feats, cats


def original_var(name: str, cat_cols: list) -> str:
    """Variable original a la que pertenece una columna del ColumnTransformer.

    Los nombres llegan como "num__age", "counts__number_inpatient" o
    "cat__A1Cresult_>8". En el bloque `cat` hay que buscar el prefijo porque el nivel
    puede contener guiones bajos. Mismo criterio que `xai_shap.lr_contributions`.
    """
    bloque, resto = name.split("__", 1)
    if bloque in ("num", "counts"):
        return resto
    matches = [c for c in cat_cols if resto.startswith(c + "_")]
    assert len(matches) == 1, f"nombre ambiguo: {name}"
    return matches[0]


def coefficient_table(logreg, cat_cols: list) -> pd.DataFrame:
    """Coeficientes de la LR, para la Fase 5B. En la 5A NO se interpretan.

    Aviso para la 5B: las escalas no son comparables entre sí (las numéricas van por
    desviación típica, las dummies por categoría), `drop="if_binary"` solo elimina una
    columna en las binarias, así que hay colinealidad que la L2 reparte entre dummies, y
    el nivel `infrequent` actúa como referencia implícita en varias variables.
    """
    prep, clf = logreg.named_steps["prep"], logreg.named_steps["clf"]
    names = prep.get_feature_names_out()
    coef = clf.coef_[0]
    tab = pd.DataFrame({
        "columna": names,
        "bloque": [n.split("__", 1)[0] for n in names],
        "variable_original": [original_var(n, cat_cols) for n in names],
        "coef": coef,
        "odds_ratio": np.exp(coef),
    })
    tab["rank_abs"] = tab["coef"].abs().rank(ascending=False).astype(int)
    # El intercepto no es de ninguna variable, pero sin él los coeficientes no
    # reconstruyen ninguna predicción; va como fila aparte y se marca como tal.
    intercepto = pd.DataFrame([{
        "columna": "(intercepto)", "bloque": "intercepto",
        "variable_original": "(intercepto)", "coef": float(clf.intercept_[0]),
        "odds_ratio": float(np.exp(clf.intercept_[0])), "rank_abs": 0,
    }])
    return pd.concat([intercepto, tab.sort_values("rank_abs")], ignore_index=True)


def stage_val() -> None:
    """Etapa val: guardarraíles, auditoría del refactor y elección de C. Sin test."""
    pd.set_option("display.width", 220)

    # El tercer split se descarta AQUÍ MISMO: en esta etapa no se calcula su etiqueta
    # corregida, no se construyen sus frames y no se puntúa nada sobre él.
    train, val, _test = P.load_model_data()
    del _test

    flags = TC.planned_flags()
    (Xtr_all, ytr_old), (Xva_all, yva_old) = P.split_xy(train), P.split_xy(val)
    ytr_new = TC.relabel(train, flags).to_numpy()
    yva_new = TC.relabel(val, flags).to_numpy()
    feats, cats = lr_feature_lists()

    # --- 1. Guardarraíl del entregable --------------------------------------------
    # No se reentrena nada: se REPRODUCE el modelo cerrado en val y se exige que sea el
    # mismo (123 árboles Y PR-AUC val 0,1868). Si no cuadra, la Fase 5A no sigue.
    Ltr_s, Lva_s = D.to_categorical(train[feats], val[feats])
    fit_with_guardrail(Ltr_s, ytr_new, Lva_s, yva_new,
                       "ENTREGABLE (sin payer_code)", ABL_TREES, ABL_VAL_PR_AUC)

    # --- 2. Auditoría del refactor: la llamada SIN argumentos reproduce la Fase 2a --
    # Mismas condiciones que entonces: TODAS las features (con `payer_code`) y la
    # etiqueta VIEJA. Si esto falla, parametrizar el preprocesado cambió algo y ninguna
    # cifra publicada de la LR seguiría siendo la documentada.
    lr_2a, grid_2a = select_logreg(Xtr_all, ytr_old, Xva_all, yva_old)
    c_2a = lr_2a.named_steps["clf"].C
    pr_2a = float(grid_2a.loc[grid_2a["C"] == c_2a, "val_pr_auc"].iloc[0])
    assert c_2a == LR_2A_C, c_2a
    assert round(pr_2a, 4) == LR_2A_VAL_PR_AUC, pr_2a
    print(f"Refactor auditado OK — LR de la Fase 2a reproducida: C = {c_2a:g}, "
          f"PR-AUC val {pr_2a:.4f}")

    # --- 3. Elección de C: etiqueta CORREGIDA, sin `payer_code` --------------------
    # Mismo procedimiento que el entregable: se entrena SOLO con train, se elige por
    # PR-AUC en val y no se reentrena con train+val. `C_GRID` es la rejilla de la Fase
    # 2a, fijada en el prerregistro; no se amplía pase lo que pase.
    Xtr, Xva = Xtr_all[feats], Xva_all[feats]
    lr, grid = select_logreg(Xtr, ytr_new, Xva, yva_new, cat_cols=cats)
    best_c = lr.named_steps["clf"].C
    n_cols = len(lr.named_steps["prep"].get_feature_names_out())

    prev = yva_new.mean()
    grid = grid.assign(
        n_val=len(yva_new),
        val_prevalencia_pct=100 * prev,
        val_lift=grid["val_pr_auc"] / prev,
        # El one-hot no depende de C, así que la columna vale lo mismo en las 5 filas;
        # se repite para que cada fila del CSV sea legible por sí sola.
        n_columnas_tras_onehot=n_cols,
        elegido=lambda g: g["C"] == best_c,
    )
    en_el_borde = best_c in (min(C_GRID), max(C_GRID))
    grid["ganador_en_el_borde"] = en_el_borde

    print("\nRejilla de C (entrena en train, elige por PR-AUC en val, etiqueta "
          "corregida, sin payer_code):")
    print(grid.round(4).to_string(index=False))
    print(f"=> C elegido = {best_c:g}"
          + (" — GANADOR EN EL BORDE de la rejilla, que NO se amplía (prerregistro)"
             if en_el_borde else ""))
    if best_c != LR_2A_C:
        print(f"   OJO: la Fase 2a eligió C = {LR_2A_C:g} con la etiqueta vieja y con "
              f"payer_code. Cambió la etiqueta y cambiaron las features.")

    # --- 4. Guardarraíl NUEVO de la LR, para anotarlo -----------------------------
    val_pr = float(grid.loc[grid["elegido"], "val_pr_auc"].iloc[0])
    print(f"\nGuardarraíl NUEVO de la LR (ANÓTALO en el script y en CLAUDE.md): "
          f"LR_C = {best_c:g}, LR_VAL_PR_AUC = {val_pr:.4f}, LR_N_COLS = {n_cols}")
    if LR_C is not None:
        assert best_c == LR_C and round(val_pr, 4) == LR_VAL_PR_AUC and n_cols == LR_N_COLS

    # Comprobación de que la métrica es la misma que usa el early stopping de LightGBM
    # y toda la Fase 4: average_precision_score, sin variantes interpoladas.
    assert np.isclose(val_pr,
                      average_precision_score(yva_new, lr.predict_proba(Xva)[:, 1]))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(REPORT_DIR / "phase5_lr_c_grid.csv", index=False)
    coefficient_table(lr, cats).to_csv(REPORT_DIR / "phase5_lr_coefficients.csv",
                                       index=False)
    print(f"\nEscritos: phase5_lr_c_grid.csv ({len(grid)} filas) y "
          f"phase5_lr_coefficients.csv ({n_cols} columnas + intercepto)")


# ---------------------------------------------------------------------------
# Métricas del contraste y bootstrap emparejado
# ---------------------------------------------------------------------------
def precision_at_k(y: np.ndarray, p: np.ndarray, k: float = K_SECONDARY) -> float:
    """Precisión marcando al k% de mayor puntuación.

    Llama a `E.top_k_table` en vez de reimplementar el corte: así el observado y cada
    réplica del bootstrap usan EXACTAMENTE la misma regla (ceil(k*n) marcados, con
    desempate aleatorio de semilla fija), y no hay dos definiciones que puedan divergir.
    """
    return float(E.top_k_table(y, p, ks=(k,))["precision"].iloc[0])


# Las seis cantidades del contraste. La PRIMERA es la única con veredicto; las otras
# cinco son descriptivas. Los dos `lift` comparten prevalencia entre los dos modelos
# dentro de cada réplica (es la misma muestra), así que su diferencia es la de la métrica
# de arriba dividida por esa prevalencia; se reportan igual porque el lift es la escala
# titular del proyecto y el prerregistro pide precisión Y lift en el top 5%.
METRICS = {
    "pr_auc": average_precision_score,
    "roc_auc": roc_auc_score,
    "lift": lambda y, p: average_precision_score(y, p) / y.mean(),
    f"precision_top{K_SECONDARY:.0%}": precision_at_k,
    f"lift_top{K_SECONDARY:.0%}": lambda y, p: precision_at_k(y, p) / y.mean(),
    "brier": brier_score_loss,
}

# Dirección de cada métrica, para que nadie lea al revés el signo de la diferencia
# ENTREGABLE - LR: el Brier es un error, así que ahí lo bueno es negativo.
BETTER_IF = {m: ("diferencia < 0" if m == "brier" else "diferencia > 0") for m in METRICS}


def bootstrap_paired(p_ent: np.ndarray, p_lr: np.ndarray, y: np.ndarray,
                     patients: np.ndarray, n_rep: int = N_REP) -> pd.DataFrame:
    """IC95 de las diferencias ENTREGABLE - LR, calculadas DENTRO de cada réplica.

    Se remuestrean PACIENTES con reemplazo, no filas: los encuentros de un mismo
    paciente no son independientes y el split ya se agrupa por `patient_nbr`. Es el
    mismo esquema (2.000 réplicas, semilla `D.SEED`) de toda la Fase 4.

    El emparejamiento es lo que importa: los dos modelos puntúan las MISMAS filas en
    cada réplica, así que la variación entre pacientes se cancela y el intervalo mide la
    diferencia, no la suma de dos incertidumbres.

    LIMITACIÓN que se declara en el informe: PR-AUC y ROC-AUC no son medias por filas,
    así que remuestrear bloques de pacientes no tiene la justificación limpia que tendría
    con una proporción. El IC se reporta como aproximado, no como exacto.
    """
    rng = np.random.default_rng(D.SEED)
    blocks = patient_blocks(patients)
    dists = {k: [] for k in METRICS}
    descartadas = 0
    for _ in range(n_rep):
        idx = boot_indices(blocks, rng)
        yy = y[idx]
        # Sin positivos (o sin negativos) el PR-AUC y el ROC-AUC no están definidos.
        if yy.sum() == 0 or yy.sum() == len(yy):
            descartadas += 1
            continue
        a, b = p_ent[idx], p_lr[idx]
        for nombre, f in METRICS.items():
            dists[nombre].append(f(yy, a) - f(yy, b))

    rows = []
    for nombre, f in METRICS.items():
        v = np.asarray(dists[nombre])
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append({
            "metrica": nombre,
            "valor_entregable": f(y, p_ent), "valor_lr": f(y, p_lr),
            "diferencia": f(y, p_ent) - f(y, p_lr),   # el observado, sin remuestrear
            "IC95_bajo": lo, "IC95_alto": hi,
            "IC_excluye_0": not (lo <= 0 <= hi),
            "replicas_usadas": len(v), "replicas_descartadas": descartadas,
        })
    return pd.DataFrame(rows)


def verdict(delta: float, lo: float, hi: float) -> tuple:
    """Veredicto prerregistrado, en el orden fijado en CLAUDE.md (commit c579acf).

    Delta = PR-AUC(entregable) - PR-AUC(LR). Las condiciones se evalúan en este orden y
    NO se tocan tras ver el resultado:
      (c) el IC incluye 0                                 -> indistinguibles
      (d) el IC queda ENTERO POR DEBAJO de 0              -> LR superior
      (a) IC ENTERO POR ENCIMA de 0 y delta >= 0,01       -> LightGBM claramente superior
      (b) IC ENTERO POR ENCIMA de 0 y delta <  0,01       -> superior pero no material
    """
    if lo <= 0 <= hi:
        return "c", "indistinguibles (el IC incluye 0)"
    if hi < 0:
        return "d", "LR superior (IC entero por debajo de 0)"
    if delta >= MATERIALITY:
        return "a", "LightGBM claramente superior (IC entero por encima de 0 y Δ >= 0,01)"
    return "b", "superior pero no material (IC entero por encima de 0 y Δ < 0,01)"


def stage_test() -> None:
    """Etapa test: los dos guardarraíles y UNA sola consulta a test.

    TERCERA consulta a test del proyecto (1.ª Fase 2c, 2.ª Fase 4 punto 4). Nada cambia
    después de verla: si el número sale mal, se reporta igual. Qué NO se mide aquí, por
    prerregistro: SHAP, equidad y cualquier variante adicional de la LR o de LightGBM.
    """
    pd.set_option("display.width", 220)
    assert None not in (LR_C, LR_VAL_PR_AUC, LR_N_COLS), (
        "sin el guardarraíl de la LR anotado no hay nada que exigir: corre antes "
        "`--stage val`")

    train, val, test = P.load_model_data()
    flags = TC.planned_flags()
    (Xtr_all, _ytr_old), (Xva_all, _yva_old), (Xte_all, _yte_old) = (
        P.split_xy(s) for s in (train, val, test))
    ytr_new = TC.relabel(train, flags).to_numpy()
    yva_new = TC.relabel(val, flags).to_numpy()
    yte_new = TC.relabel(test, flags).to_numpy()
    feats, cats = lr_feature_lists()

    # --- 1. Guardarraíl del entregable, ANTES de tocar test ------------------------
    Ltr_s, Lva_s, Lte_s = D.to_categorical(train[feats], val[feats], test[feats])
    entregable = fit_with_guardrail(Ltr_s, ytr_new, Lva_s, yva_new,
                                    "ENTREGABLE (sin payer_code)", ABL_TREES,
                                    ABL_VAL_PR_AUC)

    # --- 2. Guardarraíl de la LR, también ANTES de tocar test ----------------------
    # Se repite la elección de C completa en vez de dar por buena la de la etapa val:
    # si la rejilla no devuelve exactamente lo anotado, el script aborta aquí.
    Xtr, Xva, Xte = Xtr_all[feats], Xva_all[feats], Xte_all[feats]
    lr, grid = select_logreg(Xtr, ytr_new, Xva, yva_new, cat_cols=cats)
    best_c = lr.named_steps["clf"].C
    val_pr = float(grid.loc[grid["C"] == best_c, "val_pr_auc"].iloc[0])
    n_cols = len(lr.named_steps["prep"].get_feature_names_out())
    assert best_c == LR_C, best_c
    assert round(val_pr, 4) == LR_VAL_PR_AUC, val_pr
    assert n_cols == LR_N_COLS, n_cols
    print(f"Guardarraíl OK — LR (sin payer_code): C = {best_c:g}, PR-AUC val "
          f"{val_pr:.4f}, {n_cols} columnas")

    # ---------------------------------------------------------------------------
    # A partir de aquí se toca test. UNA vez.
    # ---------------------------------------------------------------------------
    p_ent = entregable.predict_proba(Lte_s)[:, 1]
    p_lr = lr.predict_proba(Xte)[:, 1]
    pacientes = test[D.GROUP_COL].to_numpy()
    inpatient = Xte_all["number_inpatient"].to_numpy()

    # Tercer guardarraíl, el único que SOLO se puede comprobar con test ya puntuado: el
    # entregable tiene que reproducir sus cifras publicadas en la Fase 4, punto 4
    # (reports/phase4_test_global.csv). Si no, estas predicciones no son las de entonces
    # y el contraste se calcularía sobre otro modelo. Va antes del bootstrap para que el
    # script aborte sin haber calculado el contraste ni el veredicto.
    pr_ent = average_precision_score(yte_new, p_ent)
    roc_ent = roc_auc_score(yte_new, p_ent)
    assert round(pr_ent, 4) == TEST_PR_AUC_F4, pr_ent
    assert round(roc_ent, 4) == TEST_ROC_AUC_F4, roc_ent
    print(f"Guardarraíl OK — el entregable reproduce la Fase 4 en test: "
          f"PR-AUC {pr_ent:.4f}, ROC-AUC {roc_ent:.4f}")

    # --- 3. Globales de los dos modelos y líneas base ------------------------------
    # La accuracy va con umbral 0,5 en los DOS modelos y SOLO sirve para enseñar por qué
    # no vale como métrica aquí. Con el 9,8% de prevalencia, la puntuación constante de
    # la línea base trivial queda por debajo de 0,5 y marca a cero pacientes: es decir,
    # "prevalencia constante" y "nunca reingresa" son el MISMO clasificador a ese
    # umbral, y por eso hay una fila y no dos.
    def accuracy(p: np.ndarray) -> float:
        return float(((p >= DECISION_THRESHOLD).astype(int) == yte_new).mean())

    constante = np.full(len(yte_new), yte_new.mean())
    regla = (inpatient > 0).astype(float)
    glob = pd.DataFrame([
        {**global_row("ENTREGABLE (sin payer_code)", yte_new, p_ent),
         "accuracy_%": 100 * accuracy(p_ent)},
        {**global_row(f"LR (C={best_c:g}, sin payer_code)", yte_new, p_lr),
         "accuracy_%": 100 * accuracy(p_lr)},
        {**global_row("base: prevalencia constante = nunca reingresa", yte_new, constante),
         "accuracy_%": 100 * accuracy(constante)},
        {**global_row("base: regla number_inpatient > 0", yte_new, regla),
         "accuracy_%": 100 * accuracy(regla)},
    ])
    print("\n1. Test con la etiqueta CORREGIDA — globales y líneas base:")
    print(glob.round(4).to_string(index=False))

    # --- 4. Top-k de los dos modelos ----------------------------------------------
    topk = pd.concat([
        topk_rows("ENTREGABLE (sin payer_code)", yte_new, p_ent),
        topk_rows(f"LR (C={best_c:g}, sin payer_code)", yte_new, p_lr),
    ], ignore_index=True)
    print(f"\n2. Top-k en test (el {K_SECONDARY:.0%} es la secundaria con IC; el resto, "
          f"contexto):")
    print(topk.round(4).to_string(index=False))

    # --- 5. Contraste primario y secundarias, todo emparejado ---------------------
    tab = bootstrap_paired(p_ent, p_lr, yte_new, pacientes)
    primaria = tab[tab["metrica"] == "pr_auc"].iloc[0]
    codigo, texto = verdict(primaria["diferencia"], primaria["IC95_bajo"],
                            primaria["IC95_alto"])
    contraste = pd.DataFrame([{
        "contraste": "PR-AUC(entregable) - PR-AUC(LR)",
        "pr_auc_entregable": primaria["valor_entregable"],
        "pr_auc_lr": primaria["valor_lr"],
        "prevalencia_%": 100 * yte_new.mean(),
        "observado": primaria["diferencia"],
        "IC95_bajo": primaria["IC95_bajo"], "IC95_alto": primaria["IC95_alto"],
        "nulo": 0.0, "IC_excluye_0": primaria["IC_excluye_0"],
        "umbral_materialidad": MATERIALITY,
        "supera_umbral": bool(primaria["diferencia"] >= MATERIALITY),
        "VEREDICTO": codigo, "VEREDICTO_texto": texto,
        "n_test": len(yte_new), "n_pacientes_test": len(np.unique(pacientes)),
        "replicas_usadas": primaria["replicas_usadas"],
        "replicas_descartadas": primaria["replicas_descartadas"],
    }])
    print("\n3. CONTRASTE PRIMARIO (el único con veredicto):")
    print(contraste.T.to_string(header=False))

    secundarias = tab[tab["metrica"] != "pr_auc"].assign(
        mejor_si=lambda t: t["metrica"].map(BETTER_IF),
        rol="descriptivo/sin veredicto", denominador=f"test = {len(yte_new)} filas")
    print("\n4. Secundarias, descriptivas y SIN veredicto:")
    print(secundarias.round(4).to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    glob.to_csv(REPORT_DIR / "phase5_test_global.csv", index=False)
    topk.to_csv(REPORT_DIR / "phase5_test_topk.csv", index=False)
    contraste.to_csv(REPORT_DIR / "phase5_test_primary_contrast.csv", index=False)
    secundarias.to_csv(REPORT_DIR / "phase5_test_secondary.csv", index=False)
    print("\nEscritos: phase5_test_global.csv, phase5_test_topk.csv, "
          "phase5_test_primary_contrast.csv, phase5_test_secondary.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stage", required=True, choices=["val", "test"],
                    help="val: elige C sin tocar test. test: consulta única a test.")
    stage = ap.parse_args().stage
    print(f"Fase 5A, etapa: {stage}")
    (stage_val if stage == "val" else stage_test)()


if __name__ == "__main__":
    main()
