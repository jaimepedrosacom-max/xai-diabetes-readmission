# XAI — Reingreso hospitalario a <30 días en pacientes diabéticos

Caso de estudio end-to-end de IA explicable. **Tarea:** clasificación binaria sobre el
dataset UCI *Diabetes 130-US Hospitals for Years 1999-2008*
([archive.ics.uci.edu/dataset/296](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)),
~100.000 encuentros hospitalarios: ¿reingresa el paciente en menos de 30 días tras el
alta? **Stakeholder:** un gestor de calidad hospitalaria que decide a qué pacientes hacer
seguimiento post-alta con un presupuesto limitado (top 5/10/20% de riesgo).

Este README explica **qué hay, cómo instalarlo, en qué orden ejecutarlo y qué produce
cada script**. El razonamiento de cada decisión, los prerregistros y las justificaciones
metodológicas están en [`CLAUDE.md`](CLAUDE.md), que es el cuaderno de trabajo del
proyecto; aquí solo se resume en una línea y se remite allí.

---

## 1. Entorno

```bash
python -m venv C:\venvs\xai-final
C:\venvs\xai-final\Scripts\activate
python -m pip install -r requirements.txt
```

**El venv vive FUERA del proyecto, en `C:\venvs\xai-final`.** No es una preferencia: la
ruta del proyecto en el equipo de desarrollo supera el límite de 260 caracteres de
Windows, y un venv dentro del proyecto hace fallar la instalación de paquetes con rutas
largas. En otro sistema puede crearse donde se quiera.

Las versiones de `requirements.txt` son exactas. El fichero termina con un bloque de dependencias
**solo del notebook** (jupyter, notebook, ipykernel, nbconvert, nbformat); `src/` no las importa.

Dos pines merecen explicación:

- **`dice-ml==0.12` está fijado a propósito.** El arreglo de reproducibilidad de la Fase
  3c depende de cómo está escrito `explainer_interfaces/dice_random.py` en esa versión:
  `random_seed` no se propaga a `np.random.choice` ni a `cfs_df.sample`, así que el
  script fija el estado global de `numpy` y `random` antes de cada llamada. Si se sube de
  versión, hay que volver a comprobar que dos ejecuciones dan los mismos CSV.
- `raiutils==0.4.2` solo aporta la excepción `UserConfigValidationException` que captura
  `src/xai_counterfactuals.py`.

**Datos:** no hay que descargarlos a mano. `src/data.py` los obtiene con `ucimlrepo` la
primera vez y cachea el CSV en `data/raw/`, que no está versionado.

`IDS_mapping.csv` (descripción de los IDs administrativos) viene del zip oficial de UCI del dataset 296 y no está versionado. Ningún script lo lee: se usó como referencia para las etiquetas legibles.

---

## 2. Orden de ejecución

Todos los scripts se lanzan **desde la raíz del proyecto**, por ejemplo:

```bash
python src/xai_shap.py
```

Las rutas se calculan desde la ubicación del propio fichero, así que no depende del directorio
desde el que se lance.

El orden importa: varios scripts **leen CSV que escribieron los anteriores** (la columna
"lee" lo indica). Los módulos sin `main` (`data.py` como librería, `evaluation.py`,
`preprocessing.py`, `features.py`, `xai_cases.py`, `target_correction.py`) no se ejecutan
sueltos; los importan los demás.

### Fase 1 — Datos

| # | script | qué hace | produce | lee |
|---|---|---|---|---|
| 1 | `data.py` | Descarga, limpia, construye el target y hace el split 60/20/20 agrupado por `patient_nbr`. Ejecutado directamente, imprime el resumen de la limpieza. | `data/raw/diabetes_130_raw.csv` (caché, no versionado) | — |
| 2 | `leakage_experiment.py` | Mide cuánto se inflarían las métricas con un split por filas en vez de agrupado por paciente. Justifica el split. | solo consola | — |

### Fase 2 — Modelos y evaluación

| # | script | qué hace | produce | lee |
|---|---|---|---|---|
| 3 | `train_baseline.py` | Dos baselines triviales + regresión logística (selección de `C` en val). | `reports/figures/calibration_logreg.png` | — |
| 4 | `train_lgbm.py` | LightGBM con categóricas nativas y early stopping sobre val. Fija el modelo principal. | `reports/figures/calibration_logreg_vs_lgbm.png` | — |
| 5 | `cv_model_comparison.py` | CV agrupada de 5 folds **dentro de train**: confirma que LightGBM gana a la LR sin usar val ni test. | solo consola | — |
| 6 | `evaluate_test.py` | **PRIMERA consulta a test** (target original). Métricas y top-k de los tres modelos. | `metrics_final.csv`, `topk_final.csv`, `figures/calibration_logreg_vs_lgbm.png` | — |

### Fase 3 — XAI

| # | script | qué hace | produce | lee |
|---|---|---|---|---|
| 7 | `xai_shap.py` | **Global.** TreeSHAP sobre val, comparado con importancia por permutación y con los coeficientes de la LR. | `xai_global_comparison.csv`, `xai_shap_levels.csv`, `figures/shap_global_bar.png`, `figures/shap_beeswarm_numeric.png`, `figures/shap_levels_categorical.png` | — |
| 8 | `xai_sanity.py` | **Sanity checks.** Aleatorización de etiquetas, barajado guiado, sondas de ruido y estabilidad entre folds. Además fija los 11 casos locales con `xai_cases.py`. | `xai_sanity_label_randomization.csv`, `xai_sanity_guided_shuffling.csv`, `xai_sanity_noise_probes.csv`, `xai_sanity_sensitivity.csv`, `xai_sanity_topk_composition.csv`, `xai_sanity_local_stability.csv`, `xai_local_cases.csv` | — |
| 9 | `audit_discharge_leakage.py` | Auditoría del destino al alta: ¿cuántos "reingresos" son cuidados programados? Es el origen del hallazgo central. | `audit_discharge_leakage.csv`, `audit_discharge_global.csv`, `audit_discharge_shares.csv` | `xai_shap_levels.csv` |
| 10 | `xai_counterfactuals.py` | **Local (contrafactuales).** DiCE + enumeración exhaustiva de hasta 2 cambios, con el espacio de acciones restringido. | `xai_counterfactuals_cases.csv`, `xai_counterfactuals_status.csv`, `xai_counterfactuals_population.csv`, `xai_counterfactuals_levers.csv`, `xai_counterfactuals_lab_range.csv`, `xai_counterfactuals_dice_audit.csv`, `xai_counterfactuals_DEMO_ENGANOSO.csv` y 5 `xai_counterfactuals_diag_*.csv` | `xai_local_cases.csv`, `audit_discharge_leakage.csv` |
| 11 | `xai_local.py` | **Local (SHAP).** Explicaciones de los 11 casos, en log-odds, con la marca `interpretable` leída de las sondas de ruido. | `xai_local_contributions.csv`, `xai_local_top_features.csv`, `xai_local_attribution_vs_sensitivity.csv`, `xai_local_perfiles_geometria.csv`, 11 figuras `figures/shap_local_*.png` y `figures/shap_local_pares_contrastivos.png` | `xai_local_cases.csv`, `xai_sanity_noise_probes.csv`, `xai_counterfactuals_diag_sensibilidad.csv` |

### Fase 4 — Acción: corregir el target, equidad y cierre

| # | script | qué hace | produce | lee |
|---|---|---|---|---|
| 12 | `phase4_prediction.py` | Cifras **"antes"** de la predicción prerregistrada, medidas sin tocar nada todavía. | `phase4_prediction.csv` | — |
| 13 | `phase4_correction.py` | Aplica la corrección del target (`target_correction.py`) y mide los tres pasos, la brecha, los contrastes emparejados y la CV antes/después. | `phase4_split_check.csv`, `phase4_target_correction.csv`, `phase4_relabel_by_destination.csv`, `phase4_prediction_steps.csv`, `phase4_prediction_gaps.csv`, `phase4_paired_contrasts.csv`, `phase4_prediction_deltas.csv`, `phase4_val_metrics.csv`, `phase4_shap_discharge.csv`, `phase4_cv_lift.csv` | — |
| 14 | `phase4_fairness.py` | Equidad en val: composición del top-k, FNR por grupo, detectabilidad de proxies y efecto distributivo del reetiquetado. | `phase4_fairness_composition.csv`, `phase4_fairness_fnr.csv`, `phase4_fairness_proxies.csv`, `phase4_fairness_proxies_by_class.csv`, `phase4_fairness_relabel_distribution.csv`, `phase4_fairness_contrast_count.csv` | — |
| 15 | `phase4_relabel_vs_selection.py` | Solo lectura: correlación de Spearman entre la tasa de reetiquetado de cada grupo y el cambio de su tasa de marcado. | `phase4_relabel_vs_selection.csv`, `phase4_relabel_vs_selection_rho.csv` | `phase4_fairness_composition.csv`, `phase4_fairness_relabel_distribution.csv` |
| 16 | `phase4_ablation.py` | Ablación de `payer_code` con regla de decisión prerregistrada. Decide qué modelo es el entregable. | `phase4_ablation_val.csv`, `phase4_ablation_cv.csv`, `phase4_ablation_composition.csv`, `phase4_ablation_decision.csv` | — |
| 17 | `phase4_test_final.py` | **SEGUNDA y última consulta a test**, con el modelo entregable y la etiqueta corregida. | `phase4_test_global.csv`, `phase4_test_topk.csv`, `phase4_test_steps.csv`, `phase4_test_gaps.csv`, `phase4_test_paired_contrasts.csv`, `phase4_test_deltas.csv` | — |
| 18 | `phase4_figures.py` | Figuras A (brecha A − B en val y test) y B (cocientes de `payer_code` con y sin la variable). Solo lee CSV: no entrena ni puntúa nada. | `figures/phase4_fig_A_brecha_val_test.png`, `figures/phase4_fig_B_payer_code_val.png` | `phase4_prediction_gaps.csv`, `phase4_test_gaps.csv`, `phase4_paired_contrasts.csv`, `phase4_test_paired_contrasts.csv`, `phase4_ablation_composition.csv` |

Los scripts 13 a 17 tardan varios minutos cada uno: entrenan varios modelos y, en los
pasos con intervalos de confianza, hacen 2.000 réplicas bootstrap agrupadas por paciente.

---

### Notebook: la cadena completa en un solo sitio

`notebooks/analisis.ipynb` recorre la cadena de punta a punta (datos, modelo, explicaciones,
corrección del target, equidad y lectura de test), con comentarios en cada sección. Importa las
funciones de `src/` sin reimplementarlas. Recalcula lo rápido (datos, modelo, SHAP global,
reetiquetado) y **lee de `reports/`** lo lento (bootstrap, CV, DiCE, ablación). **No consulta
test.** Tarda unos 70 s en el equipo de desarrollo y se guarda ya ejecutado, con sus salidas.

Para abrirlo, desde la raíz del proyecto con el venv activado:

```bash
jupyter notebook notebooks/analisis.ipynb
```

Los guardarraíles del notebook **no detienen la ejecución**: si una cifra no coincide con la
documentada (por ejemplo, por una versión distinta de LightGBM), imprime esperado frente a obtenido
y deja un aviso visible al final. Los scripts de `src/` sí usan `assert` y fallan.

---

## 3. Guardarraíles: qué protege cada `assert`

Antes de medir nada importante, los scripts **reentrenan el modelo y comprueban que es
exactamente el que se cerró en val**: mismo número de árboles y mismo PR-AUC en val, con
`assert`. Hay tres modelos y tres guardarraíles:

| modelo | árboles | PR-AUC val | constante | dónde se comprueba |
|---|---|---|---|---|
| Original (Fase 2, target sin corregir) | 122 | 0,2308 | `EXPECTED_TREES`, `EXPECTED_VAL_PR_AUC` en `evaluate_test.py` | `evaluate_test.py`, `xai_shap.py`, `phase4_prediction.py`, `phase4_correction.py`, `phase4_fairness.py`, `phase4_test_final.py` |
| Corregido CON `payer_code` | 183 | 0,1893 | `NEW_TREES`, `NEW_VAL_PR_AUC` en `phase4_fairness.py` | `phase4_fairness.py`, `phase4_ablation.py`, `phase4_test_final.py` |
| **Entregable: corregido SIN `payer_code`** | **123** | **0,1868** | `ABL_TREES`, `ABL_VAL_PR_AUC` en `phase4_test_final.py` | `phase4_test_final.py` |

**Qué significa que un guardarraíl falle:** el modelo reentrenado no es el que se
documentó, así que ninguna cifra posterior sería la que dice el informe. Causas típicas:
versiones distintas de LightGBM o scikit-learn (de ahí los pines exactos), o un cambio en
el preprocesado que altera el early stopping. **No se "arregla" subiendo el número
esperado:** si falla, hay que averiguar qué cambió antes de usar ninguna salida.

`xai_sanity.py` usa `EXPECTED_TREES` de otra forma: fija 122 árboles sin early stopping
en los modelos auxiliares, para que el colapso de las explicaciones bajo etiquetas
aleatorias no sea un artefacto de parar antes.

---

## 4. Aviso importante: test se consultó DOS veces, y las dos cifras no son comparables

| | cuándo | script | con qué modelo y qué etiqueta | PR-AUC en test |
|---|---|---|---|---|
| 1.ª | Fase 2c | `evaluate_test.py` | modelo original, **target original** | 0,227 |
| 2.ª | Fase 4, punto 4 | `phase4_test_final.py` | **entregable** (sin `payer_code`), **target corregido** | 0,1890 |

**Estos dos números NO son comparables.** El target cambió entre una y otra: la segunda
responde a otra pregunta (cómo predice el *target corregido* en datos no vistos) y su
línea base es otra, porque el reetiquetado bajó la prevalencia del 11,39% al 9,85%. Por
eso la cifra que se compara entre modelos en todo el proyecto es el
**lift = PR-AUC / prevalencia**, no el PR-AUC crudo.

Entre las dos consultas, de test **solo** se aplicó la regla de reetiquetado y se reportó
el conteo de filas cambiadas, que es descripción del dataset. Ninguna decisión de modelo,
de features ni de umbral salió de test. Nada cambió después de ver la segunda cifra. El
detalle de ambas decisiones y sus condiciones está en `CLAUDE.md`.

---

## 5. Citas y código externo

**Papers y referencias metodológicas** (citados en los docstrings del código, donde se
usan):

- **Strack, B. et al. (2014).** "Impact of HbA1c Measurement on Hospital Readmission
  Rates: Analysis of 70,000 Clinical Database Patient Records". *BioMed Research
  International*, 2014, 781670. — Referencia principal del dataset. De aquí salen el
  filtro de altas por fallecimiento/hospicio (`src/data.py`) y el mapeo ICD-9 de la
  Tabla 2 (`src/features.py`). La extensión `Endocrine` para `diag_2`/`diag_3` es nuestra,
  no del paper, y así se dice en el código.
- **Lundberg, S. M. & Lee, S.-I. (2017).** "A Unified Approach to Interpreting Model
  Predictions". *NeurIPS*. — Marco de los valores SHAP.
- **Lundberg, S. M. et al. (2020).** "From local explanations to global understanding
  with explainable AI for trees". *Nature Machine Intelligence*. — **TreeSHAP**, el
  explicador que usan `src/xai_shap.py` y `src/xai_local.py`
  (`feature_perturbation="tree_path_dependent"`).
- **Mothilal, R. K., Sharma, A. & Tan, C. (2020).** "Explaining Machine Learning
  Classifiers through Diverse Counterfactual Explanations". *FAT\**. — **DiCE**, usado con
  `method="random"` en `src/xai_counterfactuals.py`.
- **Adebayo, J. et al. (2018).** "Sanity Checks for Saliency Maps". *NeurIPS*. — De aquí
  viene el *data randomization test* que implementa el primer sanity check de
  `src/xai_sanity.py`. El paper es de mapas de saliencia en visión; aquí se adapta la
  idea (romper la etiqueta y ver si la explicación colapsa) a un modelo tabular, y eso se
  declara como adaptación, no como réplica.
- **EEOC (1978),** *Uniform Guidelines on Employee Selection Procedures* — la **regla de
  los cuatro quintos** ([0,80; 1,25]) que usa `src/phase4_fairness.py` como umbral de
  disparidad.
- **Holm, S. (1979).** "A Simple Sequentially Rejective Multiple Test Procedure".
  *Scandinavian Journal of Statistics*. — Corrección por comparaciones múltiples dentro de
  cada familia de contrastes de equidad.

**Librerías y uso tomado de su documentación** (no se copió código de tutoriales de
terceros; estos son usos estándar de la API documentada):

- `shap`: `shap.TreeExplainer` y `shap.summary_plot` para el beeswarm. El resto de
  figuras son matplotlib escrito para este proyecto.
- `lightgbm`: categóricas nativas, `early_stopping` y `pred_contrib`, este último usado
  como segunda validación de los valores SHAP.
- `scikit-learn`: `StratifiedGroupKFold`, `average_precision_score`, `roc_auc_score`,
  `brier_score_loss`, `calibration_curve`, `ColumnTransformer`.
- `dice-ml`: `dice_ml.Dice(..., method="random")`. **El arreglo de reproducibilidad no es
  parte de la API:** se localizó leyendo el código fuente de `dice_random.py` de la
  versión 0.12 y consiste en fijar las semillas globales de `numpy` y `random` antes de
  cada llamada. Está explicado en el docstring de `src/xai_counterfactuals.py`.
- `ucimlrepo`: descarga del dataset 296.

**Herramientas de IA generativa:** el desarrollo del código se asistió con Claude Code
(Anthropic), como permite el apartado 8 del enunciado, con revisión y supervisión fase a
fase: cada fase se aprobó antes de pasar a la siguiente y cada commit se revisó antes de
hacerlo. El análisis, las decisiones metodológicas, los prerregistros y el informe son
propios.

> Si al corregir surge una duda sobre la procedencia de alguna parte concreta del código,
> está documentada en el docstring del módulo correspondiente.

---

## 6. Qué NO está en el repositorio, y por qué

- **`data/raw/`** — caché del dataset. No se versiona porque se regenera solo con
  `src/data.py` (descarga vía `ucimlrepo`); versionarlo duplicaría ~20 MB de datos
  públicos.
- **El PDF del enunciado** (`_2026__XAI__Final_Project.pdf`) — material del profesor, no
  es obra propia y no se redistribuye.

Ambos están en [`.gitignore`](.gitignore).

---

## 7. Estructura

```
final-project/
├── src/          # todos los scripts y módulos (ver sección 2)
├── notebooks/    # analisis.ipynb: la cadena completa, narrada (ver sección 2)
├── reports/      # CSV de resultados; reports/figures/ las gráficas (PNG, y PDF vectorial cuando existe)
├── report/       # esqueleto LaTeX del informe (main.tex, references.bib); el texto lo redacta el autor
├── data/raw/     # caché del dataset (no versionado)
├── requirements.txt
├── CLAUDE.md     # cuaderno de trabajo: decisiones, prerregistros y justificaciones
└── README.md     # este fichero
```

Las dos figuras de calibración (`reports/figures/calibration_logreg*.png`) no tienen PDF: se generan
con scripts que predicen sobre test, y no se relanzan para no repetir la consulta.
