"""
Fase 3b — Sanity checks de las explicaciones, sobre el modelo ACTUAL (el "antes").

Los criterios se fijaron ANTES de ver resultados (ver CLAUDE.md) y no se tocan. Están
aquí como constantes para que el veredicto salga de la regla, no de una lectura.

  1) Aleatorización de etiquetas ("data randomization test" de Adebayo et al., 2018,
     "Sanity Checks for Saliency Maps"): si la y de train es ruido, las explicaciones
     deberían dejar de parecerse a las del modelo real. Con 122 árboles fijos: con
     early stopping pararía enseguida y el colapso sería artificial. Se mira la
     estructura y no la magnitud, porque 122 árboles también sobreajustan ruido.
  2) Barajado guiado: barajar lo que SHAP dice que importa debe doler más que barajar
     variables al azar.
  3) Sondas de ruido: variables aleatorias fijan el suelo de lo que se distingue de
     "nada". Lo que no lo supera no se interpreta.
  4) Estabilidad: modelos de los 5 folds de train, explicando los mismos pacientes.

Qué NO comprueban: que la etiqueta sea válida. Un patrón real pero espurio (código 22)
puede pasarlos todos. Por eso hay un bloque de sensibilidad aparte (no es criterio).

Modelos auxiliares: todos se entrenan solo con train. Ninguno sustituye al definitivo.
Test no se usa en ningún punto.
"""

import itertools
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import data as D
import preprocessing as P
import xai_shap as XS
from audit_discharge_leakage import load_audit_frame
from evaluate_test import CHOSEN, EXPECTED_TREES
from train_lgbm import LEARNING_RATE
from xai_cases import select_cases

# --- Criterios fijados de antemano -----------------------------------------------
N_PERMUTATIONS = 10
MAX_RANDOM_AUC = 0.52
MAX_MEDIAN_ABS_RHO = 0.30
GLOBAL_RANK_WARNING = 0.70     # solo aviso
GUIDED_KS = (1, 3, 5, 10)
N_RANDOM_SETS = 100
GUIDED_PERCENTILE = 95
N_GUIDED_SHUFFLES = 10         # el conjunto guiado se promedia; cada aleatorio, 1 barajado
N_PROBE_SEEDS = 10
TOP_N_PROBE = 10
MIN_GLOBAL_STABILITY = 0.80
N_FOLDS = 5

# Destinos señalados por la auditoría (solo para el bloque de sensibilidad).
SUSPECT_CODES = ["22", "5"]


def fit_fixed(X: pd.DataFrame, y: np.ndarray, seed: int = D.SEED) -> lgb.Booster:
    """Misma configuración que el definitivo, con 122 árboles fijos."""
    return lgb.LGBMClassifier(n_estimators=EXPECTED_TREES, learning_rate=LEARNING_RATE,
                              random_state=seed, verbose=-1, **CHOSEN).fit(X, y).booster_


def rowwise_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman entre dos vectores de SHAP, paciente a paciente (empates: rango medio)."""
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    denom = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ra * rb).sum(axis=1) / denom


def level_shap(values: np.ndarray, X: pd.DataFrame, col: str, level: str) -> float:
    mask = (X[col] == level).to_numpy()
    return float(values[mask, X.columns.get_loc(col)].mean())


def shuffled_pr_auc(booster, X, y, cols, rng) -> float:
    """PR-AUC tras barajar `cols`, cada una por separado. .values conserva el dtype category."""
    Xp = X.copy()
    for c in cols:
        Xp[c] = X[c].values[rng.permutation(len(X))]
    return average_precision_score(y, booster.predict(Xp))


# ---------------------------------------------------------------------------
# 1) Aleatorización de etiquetas
# ---------------------------------------------------------------------------
def label_randomization(Ltr, ytr, Lva, yva, real_vals, real_mean):
    rng = np.random.default_rng(D.SEED)
    rows = []
    for i in range(N_PERMUTATIONS):
        booster = fit_fixed(Ltr, rng.permutation(ytr), seed=i)
        vals, _ = XS.compute_shap(booster, Lva)
        p = booster.predict(Lva)
        rand_mean = pd.Series(np.abs(vals).mean(axis=0), index=Lva.columns)
        rows.append({
            "perm": i, "val_roc_auc": roc_auc_score(yva, p),
            "val_pr_auc": average_precision_score(yva, p),
            "mediana_abs_rho": np.nanmedian(np.abs(rowwise_spearman(real_vals, vals))),
            "mediana_abs_rho_en_abs": np.nanmedian(np.abs(rowwise_spearman(np.abs(real_vals), np.abs(vals)))),
            "rho_ranking_global": spearmanr(real_mean, rand_mean).statistic,
            "suma_media_abs_shap": rand_mean.sum(),
            "rango_inpatient": int(rand_mean.rank(ascending=False)["number_inpatient"]),
            "rango_discharge": int(rand_mean.rank(ascending=False)["discharge_disposition_id"]),
            "shap_cod22": level_shap(vals, Lva, "discharge_disposition_id", "22"),
            "shap_cod5": level_shap(vals, Lva, "discharge_disposition_id", "5"),
        })
    res = pd.DataFrame(rows)
    passed = bool((res["val_roc_auc"] <= MAX_RANDOM_AUC).all()
                  and (res["mediana_abs_rho"] < MAX_MEDIAN_ABS_RHO).all())
    return res, passed


# ---------------------------------------------------------------------------
# 2) Barajado guiado
# ---------------------------------------------------------------------------
def guided_shuffling(booster, Lva, yva, ranking):
    rng = np.random.default_rng(D.SEED)
    base = average_precision_score(yva, booster.predict(Lva))
    cols, rows = list(Lva.columns), []
    for k in GUIDED_KS:
        guided = [base - shuffled_pr_auc(booster, Lva, yva, ranking[:k], rng)
                  for _ in range(N_GUIDED_SHUFFLES)]
        rand = [base - shuffled_pr_auc(booster, Lva, yva,
                                       list(rng.choice(cols, size=k, replace=False)), rng)
                for _ in range(N_RANDOM_SETS)]
        rows.append({"k": k, "caida_guiada": np.mean(guided), "std_guiada": np.std(guided),
                     "caida_azar_media": np.mean(rand),
                     f"caida_azar_p{GUIDED_PERCENTILE}": np.percentile(rand, GUIDED_PERCENTILE),
                     "caida_azar_max": np.max(rand)})
    res = pd.DataFrame(rows)
    res["pasa"] = res["caida_guiada"] > res[f"caida_azar_p{GUIDED_PERCENTILE}"]
    return res, base, bool(res["pasa"].all())


# ---------------------------------------------------------------------------
# 3) Sondas de ruido
# ---------------------------------------------------------------------------
def noise_probes(Ltr, ytr, Lva, ranking):
    wins = pd.Series(0, index=Lva.columns)
    floors, rows = [], []
    levels = [str(i) for i in range(10)]
    for seed in range(N_PROBE_SEEDS):
        rng = np.random.default_rng(1000 + seed)

        def with_probes(X):
            Xp = X.copy()
            Xp["sonda_num"] = rng.normal(size=len(X))
            Xp["sonda_cat"] = pd.Categorical(rng.choice(levels, size=len(X)), categories=levels)
            return Xp

        Ptr, Pva = with_probes(Ltr), with_probes(Lva)
        vals, _ = XS.compute_shap(fit_fixed(Ptr, ytr, seed=seed), Pva)
        mean_abs = pd.Series(np.abs(vals).mean(axis=0), index=Pva.columns)
        floor = mean_abs[["sonda_num", "sonda_cat"]].max()
        floors.append(floor)
        rows.append({"seed": seed, "sonda_num": mean_abs["sonda_num"], "sonda_cat": mean_abs["sonda_cat"]})
        wins += (mean_abs[Lva.columns] > floor).astype(int)
    top = ranking[:TOP_N_PROBE]
    table = pd.DataFrame({"gana_a_sonda": wins}).reindex(ranking)
    return table, pd.DataFrame(rows), floors, bool((wins[top] == N_PROBE_SEEDS).all())


# ---------------------------------------------------------------------------
# 4) Estabilidad entre folds
# ---------------------------------------------------------------------------
def fold_stability(Lva, real_vals, cases):
    """5 modelos, cada uno con 4/5 de train y su propio colapso de categorías."""
    train, val, _test = D.grouped_split(D.clean(D.load_raw()), seed=D.SEED)
    del _test
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=D.SEED)
    fold_vals, fold_means, levels = [], [], []
    for fold, (tr_idx, _) in enumerate(sgkf.split(train, train[D.TARGET], groups=train[D.GROUP_COL])):
        tr, va = D.collapse_rare_categories(train.iloc[tr_idx], val)
        tr, va = P.build_feature_frame(tr), P.build_feature_frame(va)
        Ftr, Fva = P.to_lgbm_frames(tr, va)
        assert list(Fva.columns) == list(Lva.columns) and Fva.index.equals(Lva.index)
        vals, _ = XS.compute_shap(fit_fixed(Ftr, tr[D.TARGET].to_numpy()), Fva)
        fold_vals.append(vals)
        fold_means.append(pd.Series(np.abs(vals).mean(axis=0), index=Lva.columns))
        levels.append({"fold": fold + 1,
                       "shap_cod22": level_shap(vals, Fva, "discharge_disposition_id", "22"),
                       "shap_cod5": level_shap(vals, Fva, "discharge_disposition_id", "5")})

    pairs = [spearmanr(a, b).statistic for a, b in itertools.combinations(fold_means, 2)]
    # Las variables que ningún fold usa empatan en cero e inflan la correlación:
    # se da también sin ellas.
    used = pd.concat(fold_means, axis=1).sum(axis=1) > 0
    pairs_used = [spearmanr(a[used], b[used]).statistic
                  for a, b in itertools.combinations(fold_means, 2)]

    def top3(v):
        return np.argsort(-np.abs(v), axis=1)[:, :3]

    real_top = top3(real_vals)
    overlaps = np.stack([[len(set(r) & set(f)) / 3 for r, f in zip(real_top, top3(v))]
                         for v in fold_vals])  # folds x pacientes
    pos = {e: i for i, e in enumerate(Lva.index)}
    case_rows = [{"caso": c.caso, "solape_top3_medio": overlaps[:, pos[c.encounter_id]].mean(),
                  "solape_top3_min": overlaps[:, pos[c.encounter_id]].min()}
                 for c in cases.itertuples()]
    return {"min_pairwise": min(pairs), "mean_pairwise": float(np.mean(pairs)),
            "min_pairwise_usadas": min(pairs_used), "n_usadas": int(used.sum()),
            "passed": min(pairs) >= MIN_GLOBAL_STABILITY,
            "overlap_all_median": float(np.median(overlaps.mean(axis=0))),
            "overlap_all_mean": float(overlaps.mean()),
            "cases": pd.DataFrame(case_rows), "levels": pd.DataFrame(levels)}


# ---------------------------------------------------------------------------
# Sensibilidad a lo que sabemos del código 22/5 (NO es criterio de aprobado)
# ---------------------------------------------------------------------------
def suspect_sensitivity(booster, Lva, yva, values, mean_abs):
    is_suspect = Lva["discharge_disposition_id"].isin(SUSPECT_CODES).to_numpy()
    probs = booster.predict(Lva)
    rng = np.random.default_rng(D.SEED)

    def row(name, mask):
        X, y, p = Lva[mask], yva[mask], probs[mask]
        ma = pd.Series(np.abs(values[mask]).mean(axis=0), index=Lva.columns)
        base = average_precision_score(y, p)
        drops = [base - shuffled_pr_auc(booster, X, y, ["discharge_disposition_id"], rng)
                 for _ in range(N_GUIDED_SHUFFLES)]
        return {"grupo": name, "n": int(mask.sum()), "%_pos": 100 * y.mean(),
                "roc_auc": roc_auc_score(y, p), "pr_auc": base,
                "lift_pr_auc": base / y.mean(),
                "shap_discharge": ma["discharge_disposition_id"],
                "rango_discharge": int(ma.rank(ascending=False)["discharge_disposition_id"]),
                "caida_barajar_discharge": float(np.mean(drops))}

    table = pd.DataFrame([row("todo val", np.ones(len(Lva), bool)),
                          row("sin códigos 22 y 5", ~is_suspect)])

    # ¿Qué parte de los aciertos del top-k es V57 o programado en el encuentro siguiente?
    _, _, audit_val = load_audit_frame()
    audit_val = audit_val.loc[Lva.index]
    rows = []
    order = np.argsort(-probs)
    for k in (0.05, 0.10, 0.20):
        flagged = order[:int(np.ceil(k * len(Lva)))]
        a = audit_val.iloc[flagged]
        hits = a[a["readmitted"] == "<30"]
        vis = hits[hits["has_next"]]
        planned = int(vis["next_planned"].sum())
        rows.append({"top_k": f"{k:.0%}", "marcados": len(flagged), "aciertos": len(hits),
                     "precision": len(hits) / len(flagged),
                     "%_marcados_cod_22_5": 100 * is_suspect[flagged].mean(),
                     "aciertos_visibles": len(vis), "con_sig_V57_o_prog": planned,
                     "%_visibles_V57_o_prog": 100 * planned / len(vis),
                     "precision_descontados": (len(hits) - planned) / len(flagged)})
    return table, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
    warnings.filterwarnings("ignore", message="LightGBM binary classifier", category=UserWarning)
    pd.set_option("display.width", 190)
    out = D.PROJECT_ROOT / "reports"

    m = XS.load_final_model()
    booster, Ltr, Lva, ytr, yva = m["booster"], m["Ltr"], m["Lva"], m["ytr"], m["yva"]
    values, _ = XS.compute_shap(booster, Lva)
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=Lva.columns)
    ranking = list(mean_abs.sort_values(ascending=False).index)

    # Casos locales fijados antes de ver sus explicaciones.
    groups = D.grouped_split(D.clean(D.load_raw()))[1][D.GROUP_COL].loc[Lva.index]
    cases, group_info = select_cases(booster.predict(Lva), values, Lva, yva, groups)
    cases.to_csv(out / "xai_local_cases.csv", index=False)

    print("=" * 80, "\n1) ALEATORIZACIÓN DE ETIQUETAS")
    lr, lr_ok = label_randomization(Ltr, ytr, Lva, yva, values, mean_abs)
    print(lr.round(4).to_string(index=False))
    print(f"Modelo real: suma de media |SHAP| = {mean_abs.sum():.3f}")
    print(f"Criterio: ROC-AUC <= {MAX_RANDOM_AUC} y mediana |rho| < {MAX_MEDIAN_ABS_RHO} en las "
          f"{N_PERMUTATIONS} permutaciones -> {'PASA' if lr_ok else 'FALLA'}")
    warn = lr["rho_ranking_global"].mean() >= GLOBAL_RANK_WARNING
    print(f"Aviso ranking global (rho media {lr['rho_ranking_global'].mean():.3f}, umbral "
          f"{GLOBAL_RANK_WARNING}): {'ACTIVADO' if warn else 'no'}")

    print("\n" + "=" * 80, "\n2) BARAJADO GUIADO")
    gs, base, gs_ok = guided_shuffling(booster, Lva, yva, ranking)
    print(f"PR-AUC base en val: {base:.4f}")
    print(gs.round(4).to_string(index=False))
    print(f"Criterio: caída guiada > p{GUIDED_PERCENTILE} del azar en todos los k -> "
          f"{'PASA' if gs_ok else 'FALLA'}")

    print("\n" + "=" * 80, "\n3) SONDAS DE RUIDO")
    pt, pr, floors, pt_ok = noise_probes(Ltr, ytr, Lva, ranking)
    print(pr.round(5).to_string(index=False))
    print(f"Suelo (mejor sonda): media {np.mean(floors):.4f}, máximo {np.max(floors):.4f}")
    show = pt.assign(media_abs_shap=mean_abs.reindex(pt.index)).head(25)
    print(show.round(4).to_string())
    print(f"Variables que ganan a la sonda en las {N_PROBE_SEEDS} semillas: "
          f"{(pt['gana_a_sonda'] == N_PROBE_SEEDS).sum()} de {len(pt)}")
    print(f"Criterio: las top-{TOP_N_PROBE} ganan en todas las semillas -> {'PASA' if pt_ok else 'NO TODAS'}")
    bad = pt.head(TOP_N_PROBE)
    print("Top-10 que no ganan siempre:", bad[bad['gana_a_sonda'] < N_PROBE_SEEDS].index.tolist() or "ninguna")

    print("\n" + "=" * 80, "\n4) ESTABILIDAD ENTRE FOLDS")
    st = fold_stability(Lva, values, cases)
    print(f"Spearman global entre folds: mínimo {st['min_pairwise']:.3f}, media {st['mean_pairwise']:.3f}")
    print(f"Solo variables usadas por algún fold ({st['n_usadas']}): mínimo {st['min_pairwise_usadas']:.3f}")
    print(f"Criterio: rho global >= {MIN_GLOBAL_STABILITY} -> {'PASA' if st['passed'] else 'FALLA'}")
    print(f"Solape top-3 frente al modelo definitivo, todo val: mediana {st['overlap_all_median']:.3f}, "
          f"media {st['overlap_all_mean']:.3f}")
    print("Solape top-3 por caso:")
    print(cases[["caso", "prob", "reingresa"]].merge(st["cases"], on="caso")
          .merge(group_info, left_on="caso", right_index=True).round(3).to_string(index=False))
    print("SHAP medio de los códigos 22 y 5 en cada fold:")
    print(st["levels"].round(3).to_string(index=False))

    print("\n" + "=" * 80, "\nSENSIBILIDAD A LOS CÓDIGOS 22 Y 5 (no es criterio)")
    sens, topk = suspect_sensitivity(booster, Lva, yva, values, mean_abs)
    print(sens.round(4).to_string(index=False))
    print("\nComposición de los aciertos del top-k en val:")
    print(topk.round(3).to_string(index=False))
    print("\nCuántos de los casos locales son código 22 o 5:",
          int(Lva.loc[cases['encounter_id'], 'discharge_disposition_id'].isin(SUSPECT_CODES).sum()), "de", len(cases))

    # Resultados para el informe
    lr.to_csv(out / "xai_sanity_label_randomization.csv", index=False)
    gs.to_csv(out / "xai_sanity_guided_shuffling.csv", index=False)
    pt.assign(media_abs_shap=mean_abs.reindex(pt.index)).to_csv(out / "xai_sanity_noise_probes.csv")
    sens.to_csv(out / "xai_sanity_sensitivity.csv", index=False)
    topk.to_csv(out / "xai_sanity_topk_composition.csv", index=False)

    print("\n" + "=" * 80, "\nVEREDICTO SEGÚN CRITERIOS FIJADOS")
    print(f"  Aleatorización de etiquetas: {'PASA' if lr_ok else 'FALLA'}")
    print(f"  Barajado guiado:             {'PASA' if gs_ok else 'FALLA'}")
    print(f"  Sondas de ruido:             {'PASA' if pt_ok else 'NO TODAS'}")
    print(f"  Estabilidad global:          {'PASA' if st['passed'] else 'FALLA'}")


if __name__ == "__main__":
    main()
