"""
Fase 2 — Métricas y gráficas comunes a todos los modelos.

- ROC-AUC: ¿ordena bien a los pacientes por riesgo? (0,5 = azar)
- PR-AUC (average precision): igual, pero centrada en los positivos; con un
  11% de prevalencia es más exigente. Referencia del azar = prevalencia.
- Brier: error cuadrático medio de la probabilidad -> mide si "un 30%" es de
  verdad un 30% (calibración + discriminación). Más bajo = mejor.
- Top-k%: si el gestor solo puede hacer seguimiento al k% de mayor riesgo,
  ¿qué precisión y qué recall consigue? No fijamos k: se lo dejamos al gestor.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin ventana: solo guardamos a fichero
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import data as D

FIG_DIR = D.PROJECT_ROOT / "reports" / "figures"
TOP_KS = (0.05, 0.10, 0.20)

# Paleta: slots 1-2 de la paleta categórica de referencia (validada para
# daltonismo); color fijo por MODELO, no por orden de aparición.
MODEL_COLORS = {"LogReg": "#2a78d6", "LightGBM": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def metrics(y, p) -> dict:
    return {"roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p)}


def top_k_table(y, p, ks=TOP_KS, seed: int = D.SEED) -> pd.DataFrame:
    """Precisión, recall y lift marcando al k% con mayor puntuación.

    Empates: un desempate aleatorio con semilla fija evita que decida el orden
    de las filas. Solo importa con puntuaciones discretas (baselines triviales);
    con probabilidades continuas casi no hay empates.
    """
    y = np.asarray(y)
    tiebreak = np.random.default_rng(seed).random(len(y))
    order = np.lexsort((tiebreak, -np.asarray(p)))  # p desc., luego desempate
    prevalence = y.mean()
    rows = []
    for k in ks:
        n_flag = int(np.ceil(k * len(y)))
        tp = y[order[:n_flag]].sum()
        rows.append({"top_k": f"{k:.0%}", "n_marcados": n_flag,
                     "precision": tp / n_flag, "recall": tp / y.sum(),
                     "lift": (tp / n_flag) / prevalence})
    return pd.DataFrame(rows)


def evaluate(name: str, preds: dict, ys: dict):
    """preds/ys: {"train": ..., "val": ..., "test": ...} -> (métricas, top-k)."""
    m = pd.DataFrame([{"modelo": name, "split": s, **metrics(ys[s], preds[s])} for s in preds])
    tk = pd.concat([top_k_table(ys[s], preds[s]).assign(modelo=name, split=s) for s in preds])
    return m, tk


def plot_calibration(model_preds: dict, ys: dict, fname: str, n_bins: int = 10) -> Path:
    """Curva de calibración: un panel por split, una línea por modelo.

    Bins por cuantiles (cada punto = mismo nº de pacientes): con 11% de
    positivos, los bins uniformes dejarían los de riesgo alto casi vacíos.
    La diagonal es la calibración perfecta. Escalas iguales en los 3 paneles.

    Si un modelo no trae predicciones para un split (p. ej. LightGBM en test,
    que aún no se ha evaluado), ese panel lo indica con una nota en vez de
    dibujar nada.
    """
    splits = list(ys)
    fig, axes = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 4.2),
                             sharex=True, sharey=True)
    hi = 0
    for ax, s in zip(axes, splits):
        missing = [name for name, preds in model_preds.items() if s not in preds]
        if missing:
            ax.text(0.98, 0.04, f"{', '.join(missing)}: sin evaluar en {s}",
                    transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8.5)
        for name, preds in model_preds.items():
            if s not in preds:
                continue
            frac_pos, mean_pred = calibration_curve(ys[s], preds[s], n_bins=n_bins,
                                                    strategy="quantile")
            hi = max(hi, frac_pos.max(), mean_pred.max())
            ax.plot(mean_pred, frac_pos, color=MODEL_COLORS[name], lw=2,
                    marker="o", ms=6, mew=0, label=name)
        ax.set_title(s, color=INK, fontsize=11, loc="left")
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    lim = min(1.0, hi * 1.1)
    for ax in axes:
        ax.plot([0, lim], [0, lim], ls="--", lw=1, color=MUTED, label="calibración perfecta")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("probabilidad predicha (media del bin)", color=MUTED, fontsize=9)
    axes[0].set_ylabel("tasa real de reingreso <30d", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Curva de calibración (10 bins por cuantiles)", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / fname
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return path
