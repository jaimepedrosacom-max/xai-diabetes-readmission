"""
Fase 5C, secundaria — La frontera de la regresión logística, por el camino EXACTO.

Prerregistro: el bloque de la Fase 5C en CLAUDE.md, commiteado en `44db202`, sección
"SECUNDARIO Y DESCRIPTIVO". **Descriptiva, SIN veredicto.**

POR QUÉ: responde al profesor en su propio terreno. Con un modelo interpretable por diseño la
frontera no hay que buscarla, se CALCULA: el logit es aditivo, así que la reducción máxima
alcanzable dentro del espacio de acciones es

    max_reduccion = sum_v [ término_v(valor actual) - min_{valores permitidos} término_v ]

un mínimo por variable y una suma, sin enumerar ninguna combinación. Enlaza con el veredicto
(c) de la Fase 5A, donde los dos modelos salieron indistinguibles. **NO se concluye nada nuevo
sobre qué modelo es mejor:** eso se cerró en val, test ya está visto y no se reabre.

MONTAJE
    LR de la Fase 5A: C = 1, sin `payer_code`, con su guardarraíl (PR-AUC val 0,1804 y 128
    columnas tras el one-hot) exigido con `assert`. Espacio de acciones **E0**, el de la Fase
    3c. Mismos 11 casos. Umbral: el cuantil 0,95 de las puntuaciones de val **de la LR**, en
    logit. Todo en val; TEST NO SE TOCA.

CÓMO SE CALCULA CADA TÉRMINO, y por qué no se leen los coeficientes a mano
    El delta de llevar la variable v al valor x se obtiene TRANSFORMANDO la fila con ese único
    cambio y restando el logit original. Es exacto porque la LR es lineal y las columnas del
    one-hot de variables distintas son disjuntas, y así no hay que mapear valores a nombres de
    dummies: el nivel `infrequent_sklearn` (donde el OneHotEncoder colapsa lo que tiene <1% en
    train) se resuelve solo, en vez de ser una fuente de error silencioso.

LAS DOS DERIVADAS, con la misma política que el `Scorer` de la Fase 3c
    `diabetesMed` no puede cambiar en E0, porque ningún fármaco se mueve a `No`.
    `change` SÍ puede: si el candidato elimina todos los Up/Down del original, es AMBIGUO, y
    entonces se puntúa el PEOR caso (el valor que da el logit más alto). Por eso se reportan
    DOS cifras: la SUMA ADITIVA de los mínimos por variable (con `change` fijo, que es la que
    el argumento del profesor predice) y el logit REALIZADO de la fila que mueve todas las
    variables a su mejor valor con `change` derivado al peor caso. Si las dos difieren, se
    dice: la diferencia es exactamente lo que aporta la derivada.

LA COTA SUPERIOR, y por qué el «no cruza» se sostiene sobre ella y no sobre la suma aditiva
    La suma aditiva NO es una cota superior de la reducción alcanzable: si mover variables
    fuerza un `change` cuyo término de logit es MENOR, la reducción real puede SUPERARLA
    (se ve en `modelo_alto_sin_ingresos_reingresa`, donde la derivada aporta -0,05909). Por eso
    se calcula aparte una COTA SUPERIOR explícita: como el logit es aditivo y `change` es una
    columna disjunta de las demás,

        logit >= logit_orig + sum_v min(0, delta_v) + min_c gamma_c

    con `gamma_c` el delta de llevar `change` a cada uno de sus niveles (0 para el original).
    Los dos sumandos se minimizan POR SEPARADO, así que la cota es válida aunque la
    combinación que la alcanzaría no sea realizable: resuelve la ambigüedad de `change` del
    modo MÁS FAVORABLE posible. **El «no cruza» de la LR se afirma sobre esta cota**, es
    decir sobre un límite de lo alcanzable, no sobre una estimación.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import data as D
import preprocessing as P
import target_correction as TC
import xai_counterfactuals as XCF
from phase5_lr_comparison import LR_C, LR_N_COLS, LR_VAL_PR_AUC, lr_feature_lists
from phase5c_frontier import ESCALONES, K_PRIMARY, space_for
from train_baseline import make_logreg

REPORT_DIR = D.PROJECT_ROOT / "reports"


def fit_lr_with_guardrail(Xtr, ytr, Xva, yva, feats: list, cats: list):
    """Reproduce la LR de la Fase 5A y EXIGE su guardarraíl antes de medir nada."""
    lr = make_logreg(LR_C, cat_cols=cats).fit(Xtr[feats], ytr)
    pr = average_precision_score(yva, lr.predict_proba(Xva[feats])[:, 1])
    n_cols = len(lr.named_steps["prep"].get_feature_names_out())
    assert round(pr, 4) == LR_VAL_PR_AUC, pr
    assert n_cols == LR_N_COLS, n_cols
    print(f"Guardarraíl OK — LR de la Fase 5A: C = {LR_C:g}, PR-AUC val {pr:.4f}, "
          f"{n_cols} columnas tras el one-hot")
    return lr


def logit(lr, frame: pd.DataFrame) -> np.ndarray:
    return lr.decision_function(frame)


def worst_case_change(row: pd.DataFrame, candidate: pd.DataFrame) -> list:
    """Valores de `change` a puntuar, con la regla de `xai_counterfactuals.derive_change`.

    Up/Down en el candidato -> Ch. Sin Up/Down y original Ch sin Up/Down -> Ch. Si el
    candidato elimina TODOS los Up/Down del original, es AMBIGUO y se devuelven los dos
    valores para quedarse con el PEOR (el de logit mayor).
    """
    taken = [m for m in P.MED_COLS if str(row[m].iloc[0]) != "No"]
    orig_change = str(row["change"].iloc[0])
    orig_updown = bool(row[taken].astype(str).isin(["Up", "Down"]).any(axis=1).iloc[0]) \
        if taken else False
    has_updown = bool(candidate[taken].astype(str).isin(["Up", "Down"]).any(axis=1).iloc[0]) \
        if taken else False
    if has_updown or orig_change == "Ch":
        base = "Ch"
    else:
        base = "No"
    if (not has_updown) and orig_updown:
        return sorted({base, "No", "Ch"})          # ambiguo: se prueban las dos opciones
    return [base]


def delta_table(lr, row: pd.DataFrame, space: dict, feats: list) -> pd.DataFrame:
    """Delta de logit de CADA valor permitido de CADA variable accionable, uno a uno.

    `change` se deja FIJO en su valor original aquí: así los deltas son los TÉRMINOS aditivos
    del logit, que es justo lo que el camino exacto suma. El efecto de la derivada se mide
    aparte, en la fila realizada.
    """
    base = float(logit(lr, row[feats])[0])
    rows = []
    for var, opciones in space.items():
        for x in opciones:
            cand = row.copy()
            cand[var] = x
            rows.append({"variable": var, "valor_orig": str(row[var].iloc[0]),
                         "valor_nuevo": str(x),
                         "delta_logit": float(logit(lr, cand[feats])[0]) - base})
    return pd.DataFrame(rows)


def exact_frontier(lr, row: pd.DataFrame, space: dict, feats: list, cut: float,
                   change_levels: list) -> tuple:
    """La reducción máxima EXACTA, sin enumerar, y la fila que la realiza.

    Por variable se toma el mejor delta, y `0` (no mover) entra como opción: si mover una
    variable solo puede SUBIR el logit, lo mejor es dejarla quieta. La suma de esos mínimos es
    la reducción máxima alcanzable en E0 por el camino aditivo.
    """
    base = float(logit(lr, row[feats])[0])
    deltas = delta_table(lr, row, space, feats)
    mejores, cand = {}, row.copy()
    for var in space:
        sub = deltas[deltas["variable"] == var]
        peor_caso = sub["delta_logit"].min()
        if sub.empty or peor_caso >= 0:
            mejores[var] = {"delta": 0.0, "valor": str(row[var].iloc[0]), "mueve": False}
            continue
        fila = sub.loc[sub["delta_logit"].idxmin()]
        mejores[var] = {"delta": float(peor_caso), "valor": fila["valor_nuevo"], "mueve": True}
        cand[var] = fila["valor_nuevo"]
    suma = sum(v["delta"] for v in mejores.values())
    # COTA SUPERIOR de la reducción: a la suma de los mínimos por variable se le añade la
    # resolución MÁS FAVORABLE posible de la derivada `change`, minimizada POR SEPARADO. Es un
    # límite válido de lo alcanzable aunque la combinación que lo tocaría no sea realizable.
    ch_orig = str(row["change"].iloc[0])
    # Los niveles de `change` vienen de TRAIN y se pasan como argumento: aquí `row` es el
    # frame CRUDO (el que se usó para ajustar la LR), donde `change` no es categórica.
    gammas = {str(c): float(logit(lr, row.assign(change=c)[feats])[0]) - base
              for c in change_levels}
    mejor_gamma = min([0.0] + list(gammas.values()))
    cota_logit = base + suma + mejor_gamma
    # La cota es un LÍMITE, no un valor alcanzable: `change` es derivada y solo se mueve
    # moviendo un fármaco, así que si el caso no tiene NINGÚN fármaco accionable la
    # resolución favorable de `change` es inalcanzable y la cota queda LAXA. Se marca.
    hay_farmaco = any(v != 'discharge_disposition_id' for v in space)
    cota_laxa = bool(mejor_gamma < 0 and not hay_farmaco)
    # La fila REALIZADA, con `change` derivado al PEOR caso (el logit más alto).
    realizado = max(float(logit(lr, cand.assign(change=ch)[feats])[0])
                    for ch in worst_case_change(row, cand))
    movidas = [f"{v}: {row[v].iloc[0]} -> {mejores[v]['valor']}"
               for v in space if mejores[v]["mueve"]]
    info = {
        "logit_orig": base, "umbral_logit": cut,
        "reduccion_necesaria": base - cut,
        "reduccion_maxima_exacta": -suma,
        "logit_min_aditivo": base + suma,
        "deficit_aditivo": (base + suma) - cut,
        "logit_realizado_peor_change": realizado,
        "deficit_realizado": realizado - cut,
        "aporte_de_la_derivada_change": realizado - (base + suma),
        "change_orig": ch_orig,
        "mejor_delta_change_posible": mejor_gamma,
        "reduccion_maxima_COTA_SUPERIOR": -(suma + mejor_gamma),
        "logit_min_COTA_INFERIOR": cota_logit,
        "deficit_COTA": cota_logit - cut,
        "cruza_COTA": bool(cota_logit < cut),
        "cota_LAXA_change_inalcanzable": cota_laxa,
        "cruza_aditivo": bool(base + suma < cut),
        "cruza_realizado": bool(realizado < cut),
        "n_vars_accionables": len(space),
        "n_vars_que_bajan": sum(v["mueve"] for v in mejores.values()),
        "combinacion_optima": "; ".join(movidas) or "(ninguna baja el logit)",
    }
    return info, deltas


def main():
    pd.set_option("display.width", 220, "display.max_colwidth", 46)

    train, val, _test = P.load_model_data()
    del _test                                   # test se descarta aquí mismo
    flags = TC.planned_flags()
    ytr_new = TC.relabel(train, flags).to_numpy()
    yva_new = TC.relabel(val, flags).to_numpy()
    feats, cats = lr_feature_lists()
    Xtr, Xva = train, val
    print(f"Val: {len(Xva)} filas, {len(feats)} variables de entrada. TEST NO SE TOCA.")

    lr = fit_lr_with_guardrail(Xtr, ytr_new, Xva, yva_new, feats, cats)

    # Umbral de la LR: cuantil 0,95 de SUS puntuaciones de val, en logit (transformación
    # monótona, así que el cuantil del logit es el logit del cuantil).
    scores = pd.Series(logit(lr, Xva[feats]), index=Xva.index)
    cut = float(scores.quantile(1 - K_PRIMARY))
    p_cut = 1 / (1 + np.exp(-cut))
    print(f"Corte del top 5% de la LR en val: logit {cut:.6f} (probabilidad {p_cut:.6f}), "
          f"{int((scores >= cut).sum())} filas marcadas.")

    # Las frames categóricas que necesita `action_space` (son las del entregable).
    Ltr, Lva = D.to_categorical(train[feats], val[feats])
    change_levels = sorted(train["change"].astype(str).unique())      # niveles de TRAIN
    print(f"Niveles de `change` vistos en train: {change_levels}")

    cases = pd.read_csv(REPORT_DIR / "xai_local_cases.csv")
    filas, todos_deltas = [], []
    for _, case in cases.iterrows():
        enc = int(case["encounter_id"])
        # El espacio sale del frame CATEGÓRICO (necesita `.cat.categories`), y la LR se
        # puntúa con el frame CRUDO, que es el que se usó para ajustarla.
        espacio = space_for(Lva.loc[[enc]], ESCALONES[0])      # E0, el mismo que LightGBM
        row = Xva.loc[[enc]]
        info, deltas = exact_frontier(lr, row, espacio, feats, cut, change_levels)
        filas.append({"caso": case["caso"], "encounter_id": enc,
                      "marcado_top5_lr": bool(scores.loc[enc] >= cut),
                      "destino": str(row["discharge_disposition_id"].iloc[0]),
                      "variables_accionables": ", ".join(espacio) or "(ninguna)", **info})
        todos_deltas.append(deltas.assign(caso=case["caso"], encounter_id=enc))

    front = pd.DataFrame(filas)
    deltas = pd.concat(todos_deltas, ignore_index=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    front.to_csv(REPORT_DIR / "phase5c_lr_frontier.csv", index=False)
    deltas.to_csv(REPORT_DIR / "phase5c_lr_deltas.csv", index=False)

    print("\n=== LA FRONTERA DE LA LR, POR EL CAMINO EXACTO (escala LOGIT, espacio E0) ===")
    print(front[["caso", "marcado_top5_lr", "logit_orig", "reduccion_necesaria",
                 "reduccion_maxima_exacta", "deficit_aditivo", "cruza_aditivo",
                 "n_vars_que_bajan"]].round(5).to_string(index=False))
    print("\nCOTA SUPERIOR de la reducción (resuelve `change` del modo MÁS favorable posible). "
          "El 'no cruza' de la LR se afirma SOBRE ESTA COTA, no sobre la suma aditiva:")
    print(front[["caso", "marcado_top5_lr", "change_orig", "mejor_delta_change_posible",
                 "reduccion_maxima_COTA_SUPERIOR", "deficit_COTA", "cruza_COTA",
                 "cota_LAXA_change_inalcanzable"]]
          .round(5).to_string(index=False))
    print("\nFila REALIZADA con `change` derivado al PEOR caso, y lo que aporta la derivada:")
    print(front[["caso", "logit_min_aditivo", "logit_realizado_peor_change",
                 "aporte_de_la_derivada_change", "deficit_realizado", "cruza_realizado"]]
          .round(5).to_string(index=False))
    print("\nCombinación óptima de cada caso (NO es una recomendación):")
    print(front[["caso", "combinacion_optima"]].to_string(index=False))
    print("\nDESCRIPTIVO y SIN VEREDICTO. Lo único que se afirma: en la LR la frontera se "
          "CALCULA en vez de\nbuscarse, porque el logit es aditivo. No se concluye nada nuevo "
          "sobre qué modelo es mejor:\neso se cerró en la Fase 5A con el veredicto (c) y no se "
          "reabre.")
    print("\nEscritos: phase5c_lr_frontier.csv, phase5c_lr_deltas.csv")


if __name__ == "__main__":
    main()
