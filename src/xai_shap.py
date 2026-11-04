"""
Fase 3a — Explicación global con TreeSHAP sobre val, y comparación con otros métodos.

Técnica: TreeSHAP (Lundberg et al., 2020, "From local explanations to global
understanding with explainable AI for trees", Nature Machine Intelligence).
Reparte el log-odds de cada predicción entre las variables. Es exacto para árboles.

Por qué val y no test: de estas explicaciones saldrán decisiones (Fase 4), y test
no debe influir en ninguna decisión.

Por qué una barra por variable original: LightGBM divide directamente por la
variable categórica (no hay dummies), así que sale un valor SHAP por variable.

Comparación entre métodos (miden cosas distintas, no deben coincidir del todo):
  - SHAP: cuánto USA el modelo la variable (magnitud de su contribución).
  - Permutación: cuánto RENDIMIENTO se pierde al romper la variable.
  - Contribución en LR: lo mismo que SHAP, pero en el modelo lineal.
"""

import warnings

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

import data as D
import evaluation as E
import preprocessing as P
from evaluate_test import CHOSEN, EXPECTED_TREES, EXPECTED_VAL_PR_AUC
from train_baseline import select_logreg
from train_lgbm import fit_lgbm

# Niveles raros en val: su media es poco fiable, no se dibujan (sí quedan en el CSV).
MIN_LEVEL_N = 50
N_LEVEL_PANELS = 6
BLUE, SURFACE = E.MODEL_COLORS["LogReg"], "#fcfcfb"
SECONDARY = "#52514e"

# Descripciones abreviadas de IDS_mapping.csv (zip oficial de UCI).
LEVEL_LABELS = {
    "discharge_disposition_id": {
        "1": "Casa", "2": "Otro hospital (corta estancia)", "3": "Centro enfermería (SNF)",
        "4": "Centro cuidados interm. (ICF)", "5": "Otra institución hospitalaria",
        "6": "Casa con atención domiciliaria", "7": "Alta voluntaria (AMA)",
        "8": "Casa con terapia IV", "22": "Centro rehabilitación",
        "23": "Hospital larga estancia", "24": "Residencia (Medicaid)",
        "28": "Hospital psiquiátrico", "15": "Swing bed", "Unknown": "Desconocido",
    },
    "admission_type_id": {"1": "Emergencia", "2": "Urgente", "3": "Programado",
                          "4": "Recién nacido", "7": "Centro de trauma", "Unknown": "Desconocido"},
    "admission_source_id": {"1": "Derivación médica", "2": "Derivación clínica",
                            "3": "Derivación HMO", "4": "Traslado de hospital",
                            "5": "Traslado de SNF", "6": "Traslado otro centro",
                            "7": "Urgencias", "Unknown": "Desconocido"},
}


# ---------------------------------------------------------------------------
# Modelo definitivo, truncado
# ---------------------------------------------------------------------------
def load_final_model():
    """Reentrena el modelo elegido, comprueba que es el mismo y trunca el booster.

    El truncado a 122 es una garantía explícita, no una corrección: medido con
    LightGBM 4.7, el booster ya sale de fit() con exactamente los 122 árboles de la
    mejor iteración. Lo mantenemos para que el nº de árboles explicados no dependa
    del comportamiento interno de la librería, y main() imprime ambos recuentos.
    """
    train, val, _test = P.load_model_data()  # test no se usa en la Fase 3
    (Xtr, ytr), (Xva, yva) = P.split_xy(train), P.split_xy(val)
    Ltr, Lva = P.to_lgbm_frames(train, val)

    lgbm = fit_lgbm(Ltr, ytr, Lva, yva, **CHOSEN)
    assert lgbm.best_iteration_ == EXPECTED_TREES, lgbm.best_iteration_
    assert round(lgbm.best_score_["val"]["average_precision"], 4) == EXPECTED_VAL_PR_AUC

    full = lgbm.booster_
    booster = lgb.Booster(model_str=full.model_to_string(num_iteration=EXPECTED_TREES))
    assert booster.num_trees() == EXPECTED_TREES, booster.num_trees()
    # El truncado no debe cambiar ninguna predicción.
    assert np.allclose(booster.predict(Lva), lgbm.predict_proba(Lva)[:, 1], atol=1e-12)
    return {"booster": booster, "full_booster": full, "sk_model": lgbm,
            "Ltr": Ltr, "Lva": Lva, "Xtr": Xtr, "Xva": Xva, "ytr": ytr, "yva": yva}


def compute_shap(booster: lgb.Booster, X: pd.DataFrame):
    """TreeSHAP con sus comprobaciones. Devuelve (matriz n x variables, valor base).

    tree_path_dependent: no necesita datos de fondo; estima los efectos siguiendo
    las ramas de los árboles, y por eso reparte crédito según las correlaciones
    con las que se entrenó. Limitación a discutir en la reflexión crítica.
    """
    explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
    values = explainer.shap_values(X)
    if isinstance(values, list):  # algunas versiones devuelven una matriz por clase
        values = values[1]
    base = float(np.ravel(explainer.expected_value)[-1])

    assert values.shape == X.shape, f"forma inesperada: {values.shape} vs {X.shape}"
    # Aditividad: valor base + suma de contribuciones = log-odds del modelo.
    raw = booster.predict(X, raw_score=True)
    max_err = np.abs(base + values.sum(axis=1) - raw).max()
    assert max_err < 1e-6, f"aditividad rota, error máximo {max_err:.2e}"
    # Segunda fuente: las contribuciones nativas de LightGBM deben ser las mismas.
    native = booster.predict(X, pred_contrib=True)
    assert np.allclose(values, native[:, :-1], atol=1e-8), "difiere de pred_contrib"
    return values, base


def lr_contributions(logreg, Xtr: pd.DataFrame, Xva: pd.DataFrame):
    """Contribución por paciente en la LR, agregada por variable original.

    Los coeficientes crudos no son comparables entre sí: las numéricas están
    escaladas, las dummies no, y una variable se reparte entre varias dummies.
    coef x (x - media de train) es el SHAP lineal (Lundberg & Lee, 2017, asumiendo
    independencia) y queda en log-odds, igual que TreeSHAP.
    """
    prep, clf = logreg.named_steps["prep"], logreg.named_steps["clf"]
    Ztr, Zva = prep.transform(Xtr), prep.transform(Xva)
    names = prep.get_feature_names_out()
    contrib = (Zva - Ztr.mean(axis=0)) * clf.coef_[0]

    def original_var(name):
        block, rest = name.split("__", 1)
        if block in ("num", "counts"):
            return rest
        matches = [c for c in P.CAT_COLS if rest.startswith(c + "_")]
        assert len(matches) == 1, f"nombre ambiguo: {name}"
        return matches[0]

    owners = pd.Series([original_var(n) for n in names])
    by_var = pd.DataFrame({v: contrib[:, (owners == v).to_numpy()].sum(axis=1)
                           for v in owners.unique()})
    coefs = pd.DataFrame({"columna": names, "coef": clf.coef_[0]})
    return by_var.abs().mean(), coefs


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
def _style(ax):
    ax.tick_params(colors=SECONDARY, labelsize=9, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def plot_global_bar(mean_abs: pd.Series, base: float, n: int):
    """Barras de media |SHAP|: una por variable original, las 44."""
    s = mean_abs.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.235 * len(s) + 1.3), facecolor=SURFACE)
    ax.barh(s.index, s.values, color=BLUE, height=0.68)
    # Solo se rotulan las 10 mayores, para no llenar la gráfica de números.
    for i, (_, v) in enumerate(s.items()):
        if i >= len(s) - 10:
            ax.text(v + s.max() * 0.012, i, f"{v:.3f}", va="center", fontsize=8, color=SECONDARY)
    _style(ax)
    ax.grid(axis="x", color=E.GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.7, len(s) - 0.3)
    ax.set_xlabel("media de |SHAP| (log-odds)", color=E.MUTED, fontsize=9)
    n_txt = f"{n:,}".replace(",", ".")
    ax.set_title(f"Qué variables usa más LightGBM (val, n={n_txt})",
                 loc="left", color=E.INK, fontsize=12)
    return _save(fig, "shap_global_bar.png")


def plot_numeric_beeswarm(values: np.ndarray, X: pd.DataFrame):
    """Beeswarm de numéricas. Color de magnitud: un solo tono, claro a oscuro."""
    num_cols = [c for c in X.columns if c in P.NUM_COLS + P.COUNT_COLS]
    idx = [X.columns.get_loc(c) for c in num_cols]
    cmap = LinearSegmentedColormap.from_list("azul", ["#cde2fb", "#2a78d6", "#0d366b"])
    # Se pasa a float: `age` ya es ordinal 0..9.
    shap.summary_plot(values[:, idx], X[num_cols].astype(float), cmap=cmap,
                      show=False, plot_size=(8, 4.6), alpha=0.6,
                      color_bar_label="Valor de la variable")
    fig = plt.gcf()
    fig.axes[-1].set_yticklabels(["Bajo", "Alto"])
    fig.set_facecolor(SURFACE)
    ax = fig.axes[0]
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("SHAP (log-odds): a la derecha, más riesgo", color=E.MUTED, fontsize=9)
    ax.set_title("Efecto de las variables numéricas (val)", loc="left", color=E.INK, fontsize=12)
    ax.tick_params(labelsize=9)
    return _save(fig, "shap_beeswarm_numeric.png")


def level_table(values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP por nivel de cada categórica. Una fila por (variable, nivel)."""
    rows = []
    for col in [c for c in X.columns if c in P.CAT_COLS]:
        sv = pd.Series(values[:, X.columns.get_loc(col)], index=X.index)
        for level, g in sv.groupby(X[col].astype(object).fillna("(no visto en train)")):
            rows.append({"variable": col, "nivel": str(level), "n": len(g),
                         "shap_medio": g.mean(), "p25": g.quantile(0.25), "p75": g.quantile(0.75)})
    return pd.DataFrame(rows)


def plot_levels(levels: pd.DataFrame, mean_abs: pd.Series):
    """Small multiples: las categóricas con más peso. Punto = media; barra = p25-p75."""
    top = [c for c in mean_abs.sort_values(ascending=False).index if c in P.CAT_COLS][:N_LEVEL_PANELS]
    frames = {c: levels[(levels["variable"] == c) & (levels["n"] >= MIN_LEVEL_N)]
              .sort_values("shap_medio") for c in top}
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.5), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [max(len(frames[top[i]]), len(frames[top[i + 1]]))
                                                            for i in (0, 2, 4)]})
    for ax, col in zip(axes.ravel(), top):
        d = frames[col]
        labels = LEVEL_LABELS.get(col, {})
        names = [f"{labels.get(l, l)}  (n={n:,})".replace(",", ".") for l, n in zip(d["nivel"], d["n"])]
        y = np.arange(len(d))
        ax.axvline(0, color=E.MUTED, lw=1)
        ax.hlines(y, d["p25"], d["p75"], color="#86b6ef", lw=3)
        ax.plot(d["shap_medio"], y, "o", color=BLUE, ms=6, mew=0)
        ax.set_yticks(y, names)
        ax.set_ylim(-0.7, len(d) - 0.3)
        ax.grid(axis="x", color=E.GRID, lw=0.6)
        ax.set_axisbelow(True)
        _style(ax)
        hidden = (levels["variable"] == col).sum() - len(d)
        word = "nivel oculto" if hidden == 1 else "niveles ocultos"
        note = f"  ({hidden} {word} con n<{MIN_LEVEL_N})" if hidden else ""
        ax.set_title(col + note, loc="left", color=E.INK, fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("SHAP (log-odds)", color=E.MUTED, fontsize=9)
    fig.suptitle("Efecto por nivel en las categóricas con más peso: punto = media, barra = p25-p75 (val)",
                 x=0.01, ha="left", color=E.INK, fontsize=12)
    return _save(fig, "shap_levels_categorical.png")


def _save(fig, name):
    fig.tight_layout()
    E.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = E.FIG_DIR / name
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    # Niveles raros de val/test que van al bucket "infrequent" de la LR: esperado.
    warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
    # Aviso de formato de salida de shap; compute_shap() ya contempla ambos formatos.
    warnings.filterwarnings("ignore", message="LightGBM binary classifier", category=UserWarning)
    pd.set_option("display.width", 170)

    m = load_final_model()
    booster, Lva, yva = m["booster"], m["Lva"], m["yva"]
    print(f"Árboles en el booster original: {m['full_booster'].num_trees()} | "
          f"truncado: {booster.num_trees()} | predicciones idénticas: OK")

    values, base = compute_shap(booster, Lva)
    print(f"SHAP: {values.shape[0]:,} pacientes x {values.shape[1]} variables | "
          f"valor base (log-odds) = {base:.4f} -> p = {1 / (1 + np.exp(-base)):.4f}")
    print("Aditividad y coincidencia con pred_contrib: OK")

    # ¿La trampa de los árboles sobrantes era real? Medida, no supuesta.
    full_vals = shap.TreeExplainer(m["full_booster"]).shap_values(Lva.head(2000))
    full_vals = full_vals[1] if isinstance(full_vals, list) else full_vals
    diff = np.abs(full_vals - values[:2000]).max()
    print(f"SHAP con booster SIN truncar vs. truncado: diferencia máxima = {diff:.4f}")

    # --- Gráficas ---------------------------------------------------------
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=Lva.columns, name="shap")
    levels = level_table(values, Lva)
    paths = [plot_global_bar(mean_abs, base, len(Lva)),
             plot_numeric_beeswarm(values, Lva),
             plot_levels(levels, mean_abs)]

    # --- Permutación sobre val (10 repeticiones, misma métrica que la selección) ---
    pi = permutation_importance(m["sk_model"], Lva, yva, scoring="average_precision",
                                n_repeats=10, random_state=D.SEED)
    perm_mean = pd.Series(pi.importances_mean, index=Lva.columns)
    perm_std = pd.Series(pi.importances_std, index=Lva.columns)

    # --- LR (C=1, mismo proceso de selección que la Fase 2) ------------------
    logreg, _ = select_logreg(m["Xtr"], m["ytr"], m["Xva"], yva)
    lr_abs, coefs = lr_contributions(logreg, m["Xtr"], m["Xva"])

    # --- Comparación -----------------------------------------------------------
    cmp = pd.DataFrame({"shap_lgbm": mean_abs, "perm_pr_auc": perm_mean, "perm_std": perm_std,
                        "lr_contrib": lr_abs.reindex(mean_abs.index)})
    assert cmp.notna().all().all()
    for col in ("shap_lgbm", "perm_pr_auc", "lr_contrib"):
        cmp[f"rank_{col}"] = cmp[col].rank(ascending=False).astype(int)
    # Caída de rendimiento distinguible del ruido de las repeticiones.
    cmp["perm_clara"] = cmp["perm_pr_auc"] > 2 * cmp["perm_std"]
    cmp = cmp.sort_values("shap_lgbm", ascending=False)

    out = D.PROJECT_ROOT / "reports"
    cmp.to_csv(out / "xai_global_comparison.csv")
    levels.to_csv(out / "xai_shap_levels.csv", index=False)

    print("\nComparación de métodos (ordenada por SHAP):")
    print(cmp.head(20)[["shap_lgbm", "rank_perm_pr_auc", "perm_pr_auc", "perm_std", "perm_clara",
                        "rank_lr_contrib", "lr_contrib"]].round(4).to_string())
    print(f"\nVariables con caída clara de PR-AUC al permutar: {cmp['perm_clara'].sum()} de {len(cmp)}")

    print("\nSpearman entre rankings (todas las variables | top-15 de SHAP):")
    top15 = cmp.head(15)
    for a, b in [("shap_lgbm", "perm_pr_auc"), ("shap_lgbm", "lr_contrib"), ("perm_pr_auc", "lr_contrib")]:
        print(f"  {a} vs {b}: {spearmanr(cmp[a], cmp[b]).statistic:.3f} | "
              f"{spearmanr(top15[a], top15[b]).statistic:.3f}")
    print("\nCoincidencia en el top-10:")
    tops = {c: set(cmp[c].nlargest(10).index) for c in ("shap_lgbm", "perm_pr_auc", "lr_contrib")}
    print(f"  SHAP ∩ permutación: {sorted(tops['shap_lgbm'] & tops['perm_pr_auc'])}")
    print(f"  SHAP ∩ LR:          {sorted(tops['shap_lgbm'] & tops['lr_contrib'])}")

    print("\nCoeficientes crudos de LR con mayor magnitud:")
    print(coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index).head(12)
          .round(3).to_string(index=False))

    print("\nSHAP por nivel, categóricas principales:")
    for col in [c for c in cmp.index if c in P.CAT_COLS][:N_LEVEL_PANELS]:
        d = levels[(levels["variable"] == col) & (levels["n"] >= MIN_LEVEL_N)].sort_values("shap_medio")
        print(f"\n{col}:\n" + d[["nivel", "n", "shap_medio", "p25", "p75"]].round(3).to_string(index=False))

    print("\nFiguras:", *[p.name for p in paths], sep="\n  ")


if __name__ == "__main__":
    main()
