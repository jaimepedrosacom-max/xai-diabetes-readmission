# Cifras para la memoria — localizadas, con fuente y denominador

Fichero de consulta para redactar el informe. **Solo cifras**, sin interpretación ni
redacción. Cada bloque dice el CSV, la columna y el denominador. Todas las rutas son
relativas a `reports/`.

Convenio: *val* = 19.868 filas, *test* = 19.867 filas, *train* = 59.605 filas,
*train+val* = 79.473 filas.

---

## 0. Presupuesto del informe y rúbrica

### Rúbrica del enunciado (sección 7), sobre 10

| criterio | puntos |
|---|---|
| Elección y claridad del problema | 1,5 |
| Modelado y evaluación | 1,0 |
| Implementación XAI | 2,5 |
| **Uso accionable de las explicaciones** | 2,0 |
| **Reflexión crítica y evaluación de XAI** | 2,5 |
| Reproducibilidad y comunicación | 0,5 |

Máximo **8 páginas**, 5-6 recomendadas. Secciones obligatorias: Introduction ·
Data and Methods · Results · Actions and Insights · Discussion.

### Presupuesto para 6 páginas

| sección | páginas |
|---|---|
| Introduction | 0,75 |
| Data and Methods | 1,00 |
| Results | 1,50 |
| Actions and Insights | 1,25 |
| Discussion | 1,50 |
| Referencias | 0,50 |

### Dos notas de posicionamiento

- **Modelado vale 1,0 punto.** El rendimiento del modelo se posiciona en UN párrafo, con
  la comparación a la literatura, y no se defiende más allá de eso.
- **Se hicieron las tres vías de la sección 4 y solo se pide una.** Se lidera con
  **4.1 Model Improvement**, que es la que pide explícitamente *"show before/after
  metrics"*. La parte de **Fairness** va como remate del Discussion.

---

## 1. Dataset y limpieza

| cifra | valor | fuente |
|---|---|---|
| Encuentros tras la limpieza | 99.340 | suma de `n` en `phase4_target_correction.csv` (59.605 + 19.868 + 19.867) |
| Pacientes con más de un encuentro | 23,5% de los pacientes | `src/leakage_experiment.py` (consola) |
| Filas de test cuyo paciente está en train con split por filas | 41,7% de las filas de test | `src/leakage_experiment.py` (consola) |
| Efecto medido del split por filas | PR-AUC +0,006 | `src/leakage_experiment.py` (consola) |
| Tasa de positivos si solo se usara el primer encuentro | 8,98% (frente a 11,39%) | `src/data.py` (consola) |
| Pacientes recurrentes descartados en ese caso | 17,13% | `src/data.py` (consola) |

*Nota: las cifras marcadas «(consola)» en este documento no tienen CSV. Se reproducen ejecutando el script indicado.*

Reparto del split y prevalencia: `phase4_target_correction.csv`, columnas `n`,
`positivos_antes`, `prevalencia_antes_%` — train 59.605 / 6.788 / 11,3883%;
val 19.868 / 2.263 / 11,3902%; test 19.867 / 2.263 / 11,3907%.

---

## 2. Modelo original (Fase 2) — métricas con el TARGET ORIGINAL

`metrics_final.csv`, columnas `roc_auc`, `pr_auc`, `brier`; denominador = filas de cada
split.

| modelo | split | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| LightGBM | train | 0,7290 | 0,3043 | 0,0923 |
| LightGBM | val | 0,6772 | 0,2308 | 0,0959 |
| LightGBM | **test** | **0,6670** | **0,2274** | 0,0962 |
| LogReg | val | 0,6696 | 0,2124 | 0,0968 |
| LogReg | test | 0,6586 | 0,2102 | 0,0970 |
| Prevalencia (base) | test | 0,5000 | 0,1139 | 0,1009 |
| Regla `number_inpatient > 0` | test | 0,5932 | **0,1420** | 0,0993 |

Top-k con el target original: `topk_final.csv`, columnas `n_marcados`, `precision`,
`recall`, `lift`, filtrando `split == "test"`.

Hiperparámetros y CV de selección (ambos de consola de `src/cv_model_comparison.py`):
`num_leaves=31`, `min_child_samples=20`, `learning_rate=0.03`, 122 árboles por early
stopping. CV agrupada de 5 folds dentro de train: PR-AUC 0,2268 ± 0,0022 (LightGBM)
frente a 0,2046 ± 0,0040 (LR); gana en 5 de 5 folds.

---

## 3. SHAP global (Fase 3a)

`xai_global_comparison.csv`, columnas `shap_lgbm`, `perm_pr_auc`, `perm_std`,
`lr_contrib` y sus `rank_*`; denominador = 44 variables originales, calculado sobre val.

| variable | puesto SHAP | caída de PR-AUC al permutar |
|---|---|---|
| `number_inpatient` | #1 | 0,070 |
| `discharge_disposition_id` | #2 | 0,057 |
| `diag_1_group` | #3 por SHAP; puesto 6 por permutación | 0,0036 |
| `payer_code` | #4 por SHAP; tercer puesto por permutación | 0,0074 |

Niveles por categoría: `xai_shap_levels.csv`. Figuras: `figures/shap_global_bar.png`,
`figures/shap_beeswarm_numeric.png`, `figures/shap_levels_categorical.png`.

Validaciones: aditividad `base + suma = log-odds` con error < 1e-6, y coincidencia con
`pred_contrib` nativo de LightGBM (consola de `src/xai_shap.py`).

---

## 4. Sanity checks (Fase 3b)

| cifra | valor | fuente (CSV y columna) | denominador |
|---|---|---|---|
| Variables que superan las sondas de ruido | **13 de 44** | `xai_sanity_noise_probes.csv`, `gana_a_sonda == 10` | 44 variables |
| `insulin` en las sondas | 8 de 10 semillas | `xai_sanity_noise_probes.csv`, `gana_a_sonda` | 10 semillas |
| Correlación del ranking global bajo etiquetas aleatorias | 0,856 (umbral de aviso ≥ 0,7) | media de `rho_ranking_global` sobre las 10 permutaciones de `xai_sanity_label_randomization.csv` (0,8558) | 10 permutaciones |
| Puesto de `number_inpatient` bajo aleatorización | baja del 1 al 10-20 | `xai_sanity_label_randomization.csv` | 44 variables |
| Barajado guiado | supera el percentil 95 de 100 conjuntos aleatorios | `xai_sanity_guided_shuffling.csv` | 100 conjuntos |

`A1Cresult`, `race`, `gender` y todos los fármacos **no** superan las sondas
(`xai_sanity_noise_probes.csv`).

### Sensibilidad al quitar los códigos 22 y 5

`xai_sanity_sensitivity.csv`, columnas `n`, `%_pos`, `roc_auc`, `pr_auc`, `lift_pr_auc`,
`shap_discharge`, `caida_barajar_discharge`:

| grupo | n | % positivos | ROC-AUC | PR-AUC | lift | SHAP destino | caída al barajar |
|---|---|---|---|---|---|---|---|
| todo val | 19.868 | 11,3902 | 0,6772 | 0,2308 | 2,0261 | 0,2043 | 0,0566 |
| sin códigos 22 y 5 | 19.236 | 10,8443 | 0,6662 | 0,2074 | 1,9127 | 0,1799 | 0,0310 |

Filas retiradas: 632, el 3,2% de val.

---

## 5. Auditoría del destino al alta (Fase 3b) — el hallazgo

### Cifras globales
`audit_discharge_global.csv`, una fila (`grupo = train+val`). **Denominador de cada
columna, que no es el mismo:**

| columna | valor | denominador |
|---|---|---|
| `n` | 79.473 | filas de train+val |
| `n_lt30` | 9.051 | positivos `<30` de train+val |
| `n_visibles` | 6.503 | positivos con encuentro posterior visible |
| `pct_sin_visible` | 28,15% | sobre los 9.051 positivos |
| `pct_V57` | 5,41% | **sobre los 6.503 visibles** |
| `pct_programado` | 18,18% | sobre los 6.503 visibles |
| `pct_V57_o_prog` | 19,85% | sobre los 6.503 visibles |
| `tasa_sin_V57` | 10,95% | sobre las 79.473 filas |
| `tasa_sin_V57_o_prog` | 9,76% | sobre las 79.473 filas |

Positivos sin encuentro posterior visible: 2.548 de 9.051 (28,15%) en train+val.

### Por destino
`audit_discharge_leakage.csv` (train+val, con `MIN_N = 30`, deja fuera 53 filas de
destinos pequeños → suma 6.497 visibles y 19,8%, que es un **subconjunto**, no una
discrepancia con el 6.503 / 19,85% del bloque global).

| destino | cifra | valor | denominador |
|---|---|---|---|
| 22 (rehabilitación) | `pct_V57` | 57,2% | `<30` visibles del código 22 (283) |
| 22 | `pct_V57_o_prog` | 68,2% | 283 visibles |
| 22 | `n_V57_o_prog` | **193** | de los 442 `<30` del código 22 → 43,67% del total |
| 22 | tasa bruta (`pct_lt30`) | 27,76% | filas del código 22 (1.592) |
| 22 | tasa sin V57 ni programados (`tasa_sin_V57_o_prog`) | 15,64% | filas del código 22 (1.592) |
| 5 (otra institución) | `pct_V57_o_prog` | 55,2% | `<30` visibles del código 5 |
| 1 (casa) | tasa bruta (`pct_lt30`) | 9,29% | filas del código 1 (48.241) |
| 1 (casa) | tasa sin V57 ni programados (`tasa_sin_V57_o_prog`) | 8,12% | filas del código 1 (48.241) |
| 1 | `pct_V57_o_prog` | 17,3% | `<30` visibles del código 1 |
| 3 | `pct_V57_o_prog` | 16,5% | `<30` visibles del código 3 |

Ventaja de riesgo del código 22 frente a casa: 2,99x bruta (27,76 / 9,29, `pct_lt30`) → 1,93x descontando los casos
V57-o-programado.

### Visibilidad peor justo en los códigos del hallazgo
`audit_discharge_leakage.csv`, calculado como `1 - n_visibles / n_lt30`:

| destino | sin encuentro visible | denominador |
|---|---|---|
| 5 | 37,8% | 76 de 201 `<30` |
| 22 | 36,0% | 159 de 442 `<30` |
| 1 (casa) | 27,3% | 1.223 de 4.481 `<30` |
| global | 28,15% | 2.548 de 9.051 |

Si los 159 invisibles del código 22 estuvieran todos limpios, la tasa sobre el total
bajaría al 43,7% (193 de 442).

### Reparto del componente contaminado
`audit_discharge_shares.csv`, columnas `n_V57_o_prog`, `pct_V57_o_prog`, `n_V57`,
`pct_V57`; denominador = **todos** los destinos, cada columna de porcentaje suma 100.

- Código 1 = **43,61%** del componente V57-o-programado (563 de 1.291).
- Código 22 = **46,02%** del componente V57 puro (162 de 352).

### Réplica train/val (SOLO en consola de `audit_discharge_leakage.py`)
Código 5: `pct_V57` 40,0% en train frente a 14,3% en val. Código 22: 57,2% / 57,3%.

---

## 6. Contrafactuales (Fase 3c)

| cifra | valor | fuente | denominador |
|---|---|---|---|
| Casos locales elegibles | 6 de 11 | `xai_counterfactuals_diag_elegibilidad.csv`, columna `elegible` | 11 casos |
| Contrafactuales encontrados en esos 6 | **0** | `xai_counterfactuals_status.csv` | 6 casos |
| Umbral del top 10% en val | 0,1954 | `xai_counterfactuals_status.csv`, columna `umbral` (top_k 10%) | val |

Cobertura por enumeración, `xai_counterfactuals_population.csv`, columnas `n`,
`pct_con_cf`, `pct_con_cf_sin_destino` — **cada fila tiene su propio denominador**:

| top_k | grupo | n | con contrafactual | solo con fármacos |
|---|---|---|---|---|
| 5% | otros destinos | 688 | 24,85% | 3,05% |
| **10%** | **otros destinos** | **1.518** | **29,51%** (448) | **3,69%** (56) |
| 10% | 22/5 (destino congelado) | 469 | 0,64% (3) | 0,64% |
| 20% | otros destinos | 3.392 | 24,26% | 1,95% |

Top 10% completo: 451 de 1.987 marcados (22,70%) con contrafactual; 59 de 1.987 (2,97%)
sin tocar el destino.

Accionabilidad, `xai_counterfactuals_levers.csv`: de los 448 con contrafactual del grupo
"otros destinos" en el top 10%, **419** tienen `discharge_disposition_id` en su mejor
contrafactual; de ellos, **356** tienen como único cambio `destino → 1` (más 33 a `→ 6`,
20 a `→ 3` y 10 en combinaciones de dos cambios). Solo **29 de 448** no tocan el destino.

Diagnósticos (`xai_counterfactuals_diag_*.csv`): (a) Scorer reproduce
`predict_proba` con diferencia 0,0 (tol 1e-9); (b) 14 de 22 cambios de un lever mueven la
puntuación, dtypes intactos en 22/22; (c) el mínimo alcanzable queda entre 0,017 y 0,322
por encima del corte del top 10% (hasta 0,362 con todos los umbrales); (d) DiCE
encuentra contrafactual en 3/3 del control y 29/30 de la auditoría agregada.

Caso límite `falsa_alarma_mayor_riesgo` (140533200): mínimo alcanzable 0,5173 **por
encima** de su puntuación original 0,5170.

---

## 7. Explicaciones locales (Fase 3d)

`xai_local_contributions.csv` (en log-odds), `xai_local_top_features.csv`,
`xai_local_attribution_vs_sensitivity.csv`; denominador = 11 casos.

| cifra | valor |
|---|---|
| `number_inpatient` en el top-3 local | 11 de 11 casos |
| `discharge_disposition_id` en el top-3 local | 9 de 11 casos |
| Aditividad `base + suma = log-odds` | error < 1e-6 en los 11 |
| Casos con una variable NO interpretable en el top-3 | 2 de 11 |

Casos con destino 22/5 (3 de 11), contribución del destino frente a la #2 real **en valor
absoluto**:

| caso | destino | #2 | cociente |
|---|---|---|---|
| `perfil_alto_2` | +0,954 | `payer_code` +0,101 | 9,4x |
| `modelo_alto_sin_ingresos_reingresa` | +0,936 | `number_inpatient` −0,134 | 7,0x |
| `modelo_alto_sin_ingresos_no_reingresa` | +1,127 | `number_inpatient` −0,146 | 7,7x |

Perfiles (`xai_local_perfiles_geometria.csv`): los tres `perfil_alto_*` son los encuentros más cercanos a cada centroide de k-means, no el centroide. Entre `perfil_alto_1` y `perfil_alto_3`, la correlación de sus vectores SHAP es 0,987 y la distancia entre centroides es 0,650 (frente a 1,108 para 2-3 y 1,630 para 1-2). La distancia entre los dos encuentros es 0,591, y 0,575 de ella corresponde a `number_inpatient`: +1,279 con valor 5 y +0,703 con valor 4 (`xai_local_contributions.csv`).

Probabilidades dentro de cada par contrastivo: 0,2617 / 0,2303 y 0,0826 / 0,0937.
Suma de |SHAP| del top-3: 1,16 y 1,37 ("alto sin ingresos") frente a 0,49 y 0,42
("bajo con ingresos").

Cruce atribución / sensibilidad (`xai_local_attribution_vs_sensitivity.csv`): 0 "atribuye
sin mover", 5 "mueve sin atribuir", 3 "coincide", 3 "ni atribuye ni mueve"; denominador =
11 filas con delta medido. Los 5 "mueve sin atribuir" son variables no interpretables
(insulin ×3, pioglitazone, glyburide), 4 de ellos con |delta| entre 0,003 y 0,005 y solo
`pioglitazone` en `perfil_alto_2` con 0,029. `delta_incluye_change` es True en 3 de 11.

**Composición del grupo A (top 10% de val con `number_inpatient == 0`)**, de consola de
`src/xai_local.py` y reproducida en `phase4_prediction_steps.csv`:

| grupo | % destinos 22/5 | denominador |
|---|---|---|
| A: top 10% con `number_inpatient == 0` | **69,32%** | 305 de 440 |
| resto del top 10% | 10,6% | 164 de 1.547 |
| top 10% completo | 23,6% | 469 de 1.987 |
| todo val | 3,2% | 632 de 19.868 |

Estabilidad del top-3 local entre folds (`xai_sanity_local_stability.csv`):
`modelo_bajo_con_ingresos_reingresa` es el único con mínimo 1,00 (n_grupo = 62); los otros
diez tienen mínimo 0,67.

---

## 8. Corrección del target (Fase 4, puntos 2-3)

### Mecánica del reetiquetado
`phase4_target_correction.csv`, columnas `positivos_antes`, `filas_cambiadas`,
`pct_de_positivos_cambiados`, `prevalencia_antes_%`, `prevalencia_despues_%`;
denominador = positivos de cada split.

| split | n | positivos antes | cambiadas | % de positivos | prevalencia antes → después |
|---|---|---|---|---|---|
| train | 59.605 | 6.788 | 971 | 14,30% | 11,3883% → 9,7592% |
| val | 19.868 | 2.263 | 320 | 14,14% | 11,3902% → 9,7795% |
| test | 19.867 | 2.263 | 307 | 13,57% | 11,3907% → 9,8455% |

Train+val suman **1.291 filas**, que reproducen `audit_discharge_shares.csv` fila a fila.

Por destino (`phase4_relabel_by_destination.csv`, denominador = positivos del destino):
código 22 **43,67%** (193 de 442); código 5 **34,33%** (69 de 201); código 1 **12,56%**
(563 de 4.481).

El encuentro siguiente nunca cae en otro split: 0 casos en los tres
(`phase4_split_check.csv`, con `assert`).

### Los tres pasos en val
`phase4_prediction_steps.csv`, columnas `corte`, `n_marcados`, `aciertos`,
`precision_%`, `pct_destino_22_o_5`; denominador = marcados de cada subgrupo.

| paso | corte | n_A | aciertos A | precisión A | % 22/5 en A | n_B | aciertos B | precisión B | % 22/5 en B |
|---|---|---|---|---|---|---|---|---|---|
| (1) viejo/vieja | 0,1954 | 440 | 142 | 32,2727% | 69,3182% | 1.547 | 422 | 27,2786% | 10,6012% |
| (2) viejo/nueva | 0,1954 | 440 | 81 | 18,4091% | 69,3182% | 1.547 | 370 | 23,9173% | 10,6012% |
| (3) nuevo/nueva | 0,1671 | 212 | 42 | 19,8113% | 34,4340% | 1.775 | 406 | 22,8732% | 6,3662% |

Corte del top 10% con el modelo viejo: 0,1954; 1.987 marcados de 19.868; 564 aciertos en
el paso (1) = la `precision` del 10% en `xai_sanity_topk_composition.csv`.

### Brecha A − B en val
`phase4_prediction_gaps.csv`, columnas `brecha_puntos`, `brecha_puntos_IC95_*`,
`cociente_A_entre_B`, `cociente_IC95_*`, `n_A`, `n_B`. Bootstrap de 2.000 réplicas
agrupadas por `patient_nbr`, semilla 42, 0 descartadas.

| paso | brecha | IC95 | cociente | IC95 | n_A | n_B |
|---|---|---|---|---|---|---|
| (1) viejo/vieja | +4,99 | [+0,07; +9,99] | 1,183 | [1,002; 1,389] | 440 | 1.547 |
| (2) viejo/nueva | **−5,51** | [−9,80; −1,04] | 0,770 | [0,613; 0,954] | 440 | 1.547 |
| (3) nuevo/nueva | −3,06 | [−9,31; +2,44] | 0,866 | [0,609; 1,109] | 212 | 1.775 |

### Contrastes emparejados en val
`phase4_paired_contrasts.csv`, columnas `observado`, `IC95_bajo`, `IC95_alto`, `nulo`,
`excluye_nulo`.

| contraste | observado | IC95 | nulo | excluye |
|---|---|---|---|---|
| brecha(2) − brecha(1) | **−10,50 puntos** | [−13,94; −7,13] | 0 | sí |
| cociente(2) / cociente(1) | 0,651 | [0,558; 0,747] | 1 | sí |
| delta_A − delta_B | +8,06 puntos | [+2,78; +13,75] | 0 | sí |

### Deltas por subgrupo en val
`phase4_prediction_deltas.csv`, en puntos de precisión:

| subgrupo | delta_etiqueta | delta_modelo | delta_total |
|---|---|---|---|
| A | 13,86 | −1,40 | **12,46** |
| B | 3,36 | +1,04 | **4,41** |

Veredicto prerregistrado: `delta_A` 12,46 > `delta_B` 4,41 → **CONFIRMADA**.

### Métricas globales en val
`phase4_val_metrics.csv`, columnas `prevalencia_%`, `roc_auc`, `pr_auc`, `lift`;
n = 19.868 en los tres:

| caso | prevalencia | ROC-AUC | PR-AUC | lift |
|---|---|---|---|---|
| (1) viejo/vieja | 11,3902% | 0,6772 | 0,2308 | 2,0261 |
| (2) viejo/nueva | 9,7795% | 0,6693 | 0,1861 | 1,9031 |
| (3) nuevo/nueva | 9,7795% | 0,6734 | 0,1893 | 1,9359 |

### CV de 5 folds dentro de train, etiqueta vieja frente a nueva
`phase4_cv_lift.csv`, columnas `fold`, `etiqueta`, `prevalencia_%`, `pr_auc`, `lift`;
`n_estimators` fijo a 122 en ambas. Lift medio: **1,9914 ± 0,0200** (vieja) frente a
**1,9878 ± 0,0575** (nueva); la nueva gana en 2 de los 5 folds.

### SHAP del destino antes y después
`phase4_shap_discharge.csv`, columna `shap_medio_abs_discharge`, sobre val:
0,2043 → 0,1871 (cambio −0,0172, un 8,4% de bajada).

---

## 9. Equidad (Fase 4, punto 5) — todo en val salvo el distributivo

### Composición del top 10% por `payer_code` (primario, referencia MC)
`phase4_fairness_composition.csv`, filtrando `atributo == "payer_code"` y
`top_k == "10%"`; columnas `n_grupo`, `marcados`, `tasa_marcado_%`, `cociente`,
`cociente_IC95_*`, `cociente_modelo_viejo`, `elegible`, `veredicto_disparidad`.
Denominador del cociente: tasa del grupo / tasa de MC (7,2408%).

| grupo | n_grupo | marcados | tasa marcado | cociente (IC95) | cociente modelo viejo | elegible | veredicto |
|---|---|---|---|---|---|---|---|
| `Missing` | 7.933 | 1.191 | 15,0132% | **2,0734** [1,7535; 2,4629] | 1,4521 | sí | DISPARIDAD |
| HM | 1.167 | 114 | 9,7686% | 1,3491 [0,9733; 1,8387] | 0,8054 | sí | DISPARIDAD |
| MD | 661 | 52 | 7,8669% | 1,0865 [0,6673; 1,6064] | 1,2659 | sí | no |
| MC (ref) | 6.201 | 449 | 7,2408% | 1,0000 | 1,0000 | sí | — |
| Other | 547 | 39 | 7,1298% | 0,9847 [0,5286; 1,5080] | 1,0268 | sí | no |
| CM | 367 | 24 | 6,5395% | 0,9032 [0,3980; 1,5716] | 0,8120 | sí | no |
| BC | 945 | 53 | 5,6085% | 0,7746 [0,4099; 1,2154] | 0,8612 | sí | DISPARIDAD |
| UN | 475 | 17 | 3,5789% | 0,4943 [0,2725; 0,7923] | 0,8687 | **no** (17 < 20) | sin veredicto |
| CP | 497 | 18 | 3,6217% | 0,5002 [0,2677; 0,8205] | 0,5996 | **no** (18 < 20) | sin veredicto |
| SP | 1.075 | 30 | 2,7907% | **0,3854** [0,2394; 0,5685] | 0,8317 | sí | DISPARIDAD |

De los 4 veredictos de DISPARIDAD, **solo 2 tienen IC que excluya el 1**: `Missing` y
`SP`. Los de `BC` y `HM` lo incluyen.

### FNR por `race` al top 10% (primario, referencia Caucasian, m de Holm = 4)
`phase4_fairness_fnr.csv`; **denominador = positivos reales del grupo** (columna
`denominador`), no `n_grupo`.

| grupo | n_grupo | positivos | FNR | dif. vs ref | IC95 | p bootstrap | Holm | veredicto |
|---|---|---|---|---|---|---|---|---|
| Caucasian (ref) | 14.872 | 1.455 | 77,5945% | 0 | — | — | — | — |
| `Missing` | 400 | 35 | 88,5714% | +10,98 | [−1,32; +20,33] | 0,080 | no | no |
| Asian | 136 | 15 | 80,0000% | +2,41 | [−16,38; +23,46] | — | — | sin veredicto (15 < 20) |
| Other | 288 | 23 | 78,2609% | +0,67 | [−19,47; +22,49] | 0,901 | no | no |
| AfricanAmerican | 3.751 | 369 | 74,2547% | −3,34 | [−10,51; +2,94] | 0,352 | no | no |
| Hispanic | 421 | 46 | 67,3913% | −10,20 | [−25,80; +6,60] | 0,243 | no | no |

### Detectabilidad de proxies
`phase4_fairness_proxies.csv`, columnas `arboles`, `auc_ovr_macro`, `lectura`:
`race` **0,7367** con 261 árboles (recuperable); `gender` 0,6299 con 309 (zona gris).
Por clase (`phase4_fairness_proxies_by_class.csv`): `Missing` 0,813, AfricanAmerican
0,750, Asian 0,735, Caucasian 0,732, Hispanic 0,723, Other 0,667 — las 6 por encima de
0,65.

### Efecto distributivo del reetiquetado (train+val, NO val)
`phase4_fairness_relabel_distribution.csv`; **denominador = positivos del grupo en
train+val** (columna `denominador`).

Por `race` (primario, referencia Caucasian, m de Holm = 5):

| grupo | positivos | tasa reetiquetado | dif. vs ref | IC95 | p | Holm | veredicto |
|---|---|---|---|---|---|---|---|
| Caucasian (ref) | 6.825 | 14,8278% | 0 | — | — | — | — |
| Asian | 54 | 16,6667% | +1,84 | [−6,93; +10,37] | 0,680 | no | no |
| `Missing` | 151 | 15,8940% | +1,07 | [−4,68; +7,59] | 0,710 | no | no |
| Other | 110 | 15,4545% | +0,63 | [−8,07; +9,88] | 0,943 | no | no |
| **AfricanAmerican** | 1.742 | 12,1125% | **−2,72** | **[−4,54; −0,95]** | **0,002** | **sí** | **no** (< 5 puntos) |
| Hispanic | 169 | 10,6509% | −4,18 | [−8,83; +0,87] | 0,099 | no | no |

Por `payer_code` (secundario/exploratorio, sin p ni Holm, referencia MC):
**MC 15,2677%** (3.026 positivos), SP 28,2927% (+13,03, IC [+8,26; +17,87]), UN 23,9766%
(+8,71), BC 23,4940% (+8,23), CP 20,4678% (+5,20), MD 19,6923% (+4,42), Other 14,3478%,
CM 12,1622%, `Missing` 11,2881% (−3,98), HM 4,8598% (−10,41, IC [−12,68; −8,01]).

### Relación entre reetiquetado y desplazamiento de composición
`phase4_relabel_vs_selection_rho.csv`: Spearman **rho = −0,9515**, p = 2,28e-05,
n = 10 grupos de `payer_code`. Factores de cambio de la tasa de marcado ABSOLUTA
(`phase4_relabel_vs_selection.csv`, columna `factor_cambio`): SP 0,385, UN 0,472,
CP 0,692, MD 0,712, BC 0,747, Other 0,796, **MC 0,830**, CM 0,923, `Missing` 1,185,
HM 1,390.

### Recuento de contrastes
`phase4_fairness_contrast_count.csv`: planificados 20, realizados 17 (16 en el CSV; el de proxies se calcula aparte) — composición 7 de 9
(excluidos CP con 18 marcados y UN con 17), FNR 4 de 5 (excluido Asian con 15 positivos),
distributivo 5 de 5, proxies 1 de 1.

---

## 10. Ablación de `payer_code` (Fase 4)

### Rendimiento
`phase4_ablation_val.csv` (val, prevalencia 9,7795%) y `phase4_ablation_cv.csv`
(5 folds, `n_estimators` fijo):

| | val PR-AUC | val lift | CV lift medio | CV PR-AUC medio |
|---|---|---|---|---|
| con `payer_code` | 0,1893 | 1,9359 | **1,9878 ± 0,0575** | 0,1940 |
| sin `payer_code` | 0,1868 | 1,9100 | **1,9419 ± 0,0651** | 0,1895 |

Caída relativa del lift en CV: **2,31%** (`phase4_ablation_decision.csv`,
`caida_relativa_%`). El modelo con `payer_code` gana en **5 de 5 folds** (lift por fold:
2,0749/2,0539; 2,0058/1,9302; 1,9249/1,8841; 1,9816/1,9219; 1,9519/1,9193).

### Composición antes y después
`phase4_ablation_composition.csv`, columnas `cociente_con`, `IC_con_*`, `cociente_sin`,
`IC_sin_*`, `cambio_cociente`, `fallo_robusto_*`:

| grupo | n_grupo | cociente CON (IC95) | cociente SIN (IC95) | cambio |
|---|---|---|---|---|
| `Missing` | 7.933 | 2,0734 [1,7535; 2,4629] | **0,9623** [0,8288; 1,1052] | **−1,1111** |
| HM | 1.167 | 1,3491 [0,9733; 1,8387] | 0,9683 [0,6993; 1,2410] | −0,3809 |
| MD | 661 | 1,0865 [0,6673; 1,6064] | 1,0980 [0,7806; 1,4696] | +0,0115 |
| MC (ref) | 6.201 | 1,0000 | 1,0000 | 0 |
| Other | 547 | 0,9847 [0,5286; 1,5080] | 0,9405 [0,5626; 1,3795] | −0,0442 |
| CM | 367 | 0,9032 [0,3980; 1,5716] | 0,8761 [0,4731; 1,3234] | −0,0270 |
| BC | 945 | 0,7746 [0,4099; 1,2154] | 0,7583 [0,4950; 1,0619] | −0,0163 |
| CP | 497 | 0,5002 [0,2677; 0,8205] | 0,5545 [0,3343; 0,8487] | +0,0543 |
| UN | 475 | 0,4943 [0,2725; 0,7923] | 0,5415 [0,3385; 0,7821] | +0,0473 |
| SP | 1.075 | 0,3854 [0,2394; 0,5685] | 0,4444 [0,3020; 0,5969] | +0,0590 |

Marcados: CP 18 → 30 y UN 17 → 28 (entran en elegibilidad); `Missing` 1.191 → 831.

### Decisión
`phase4_ablation_decision.csv`: grupos comunes elegibles 7; fallos robustos **2 → 1**;
fallos brutos 4 → 4; entran CP y UN; no sale ninguno; cond1 True, cond2 True;
**ENTREGABLE = sin `payer_code`**. Sobre los 9 grupos no-referencia el recuento robusto
sería 2 → 3.

Guardarraíles: entregable **123 árboles / PR-AUC val 0,1868**; corregido con
`payer_code` 183 / 0,1893; viejo 122 / 0,2308.

---

## 11. Consulta única a test (Fase 4, punto 4) — etiqueta CORREGIDA

### Globales del entregable
`phase4_test_global.csv`, n = 19.867, prevalencia 9,8455%:

| modelo | ROC-AUC | PR-AUC | lift | Brier |
|---|---|---|---|---|
| **ENTREGABLE (sin `payer_code`)** | 0,6655 | 0,1890 | **1,9194** | 0,0855 |
| base: prevalencia (constante) | 0,5000 | 0,0985 | 1,0000 | 0,0888 |
| base: regla `number_inpatient > 0` | 0,5971 | 0,1246 | 1,2660 | 0,3344 |

### Top-k en test
`phase4_test_topk.csv`; denominador de la precisión = marcados, de la recall = positivos
reales (1.956 en test tras la corrección).

| k | marcados | aciertos | precisión | recall | lift |
|---|---|---|---|---|---|
| 5% | 994 | 267 | 26,8612% | 13,6503% | **2,7283** |
| 10% | 1.987 | 440 | 22,1439% | 22,4949% | 2,2491 |
| 20% | 3.974 | 728 | 18,3191% | 37,2188% | 1,8607 |

Regla `number_inpatient > 0` en test: 5% → 140 aciertos, 14,0845%, lift 1,4306;
10% → 294, 14,7962%, lift 1,5028; 20% → 607, 15,2743%, lift 1,5514.

### Los cuatro pasos en test
`phase4_test_steps.csv`:

| paso | corte | n_A | aciertos A | precisión A | % 22/5 en A | n_B | aciertos B | precisión B |
|---|---|---|---|---|---|---|---|---|
| (1) viejo/vieja | 0,1942 | 427 | 111 | 25,9953% | 63,93% | 1.560 | 443 | 28,3974% |
| (2) viejo/nueva | 0,1942 | 427 | 74 | 17,3302% | 63,93% | 1.560 | 380 | 24,3590% |
| (3a) corregido CON/nueva | 0,1682 | 210 | 41 | 19,5238% | 32,38% | 1.777 | 401 | 22,5661% |
| (3b) entregable SIN/nueva | 0,1634 | 161 | 27 | 16,7702% | 35,40% | 1.826 | 413 | 22,6177% |

### Brecha A − B en test
`phase4_test_gaps.csv`, 2.000 réplicas, 0 descartadas:

| paso | brecha | IC95 | cociente | IC95 | n_A | n_B |
|---|---|---|---|---|---|---|
| (1) viejo/vieja | **−2,40** | [−7,45; +2,41] | 0,9154 | [0,7535; 1,0889] | 427 | 1.560 |
| (2) viejo/nueva | −7,03 | [−11,46; −2,65] | 0,7115 | [0,5500; 0,8850] | 427 | 1.560 |
| (3a) | −3,04 | [−8,97; +2,65] | 0,8652 | [0,6166; 1,1211] | 210 | 1.777 |
| (3b) | −5,85 | [−11,64; +0,66] | 0,7415 | [0,4981; 1,0303] | 161 | 1.826 |

### Contrastes emparejados en test
`phase4_test_paired_contrasts.csv`:

| contraste | observado | IC95 | excluye el nulo |
|---|---|---|---|
| brecha(2) − brecha(1) | **−4,6266 puntos** | [−7,5737; −1,6545] | sí |
| cociente(2) / cociente(1) | 0,7772 | [0,6670; 0,8869] | sí |
| delta_A − delta_B con (3a) | +0,6402 | [−4,5507; +5,6376] | **no** |
| delta_A − delta_B con (3b) | +3,4454 | [−2,8967; +9,2808] | **no** |

Los dos primeros son idénticos con (3a) y (3b): solo usan el modelo viejo.

### Deltas por subgrupo en test
`phase4_test_deltas.csv`:

| modelo nuevo | subgrupo | delta_etiqueta | delta_modelo | delta_total |
|---|---|---|---|---|
| (3a) | A | 8,6651 | −2,1936 | 6,4715 |
| (3a) | B | 4,0385 | +1,7929 | 5,8313 |
| (3b) | A | 8,6651 | +0,5600 | 9,2251 |
| (3b) | B | 4,0385 | +1,7412 | 5,7797 |

---

## 12. Comparaciones que NO deben hacerse (para no citarlas mal)

- **PR-AUC 0,2274 (test, Fase 2) frente a 0,1890 (test, entregable)**: targets distintos
  y prevalencias distintas (11,39% frente a 9,85%). La comparable es el lift.
- **Análisis distributivo del reetiquetado** (train+val, 79.473 filas) **frente al resto
  del punto 5** (val, 19.868 filas): denominadores distintos.
- **68,2% del código 22** (sobre 283 visibles) **frente a 43,67%** (sobre los 442
  positivos totales): son el mismo recuento de 193 casos con dos denominadores.
- **Δ de PR-AUC de la Fase 5A** (+0,0062; target corregido, sin `payer_code`, los dos
  modelos emparejados en test) **frente a la diferencia LightGBM − LR de la Fase 2 en
  test**: targets distintos, modelos distintos y conjuntos de features distintos. La de
  la Fase 2 no es una medida previa de la misma cantidad.
- **30,32% de la Fase 5B** (24.100 de 79.473 ENCUENTROS de train+val CON siguiente
  visible, `phase5b_admtype_next_elective.csv`) **frente al 28,15% de la Fase 4** (POSITIVOS
  `<30` SIN encuentro posterior visible, `audit_discharge_global.csv`, `pct_sin_visible`):
  denominadores distintos (todos los encuentros frente a solo los positivos) y signos
  contrarios (con frente a sin). No se restan ni se comparan.
- **0,6036% de la Fase 5C** (6 de los 994 marcados del top 5% del ENTREGABLE sin ninguna
  variable accionable, `phase5c_no_actions.csv`) **frente al 12,09% de la Fase 3c**
  (`xai_counterfactuals_population.csv`, `pct_sin_acciones`, grupo 22/5 del top 5% del
  modelo VIEJO): dos conjuntos de filas marcadas distintos, elegidos por dos modelos
  distintos sobre dos etiquetas distintas. El entregable marca a muchos menos pacientes de
  22/5 (62 de 994), así que la comparación no mide nada.
- **Cualquier cifra de contrafactuales de la Fase 3c frente a la Fase 5C**: la 3c usó el
  modelo de la Fase 2 (122 árboles, las 44 features CON `payer_code`, target ORIGINAL) y la
  5C el entregable (123 árboles, 43 features, target corregido). Las de la 3c quedan como
  registro de ese modelo. Lo mismo vale para las cifras de DiCE, que en la 5C no se
  recalculan.
- **6.503 visibles / 19,85%** (`audit_discharge_global.csv`) **frente a 6.497 / 19,8%**
  (suma de `audit_discharge_leakage.csv`): la segunda aplica `MIN_N = 30` y es un
  subconjunto.

---

## 13. Fase 5A — el entregable frente a la regresión logística (TERCERA consulta a test)

Todo con la **etiqueta corregida**. Denominadores: *val* = 19.868 filas (prevalencia
9,7795%), *test* = 19.867 filas y **13.951 pacientes** (prevalencia 9,8455%),
*train* = 59.605 filas. Los dos modelos usan las MISMAS 43 variables (`P.FEATURES` menos
`payer_code`) y el mismo split congelado.

### Guardarraíles (los tres pasaron)

| qué | cifra exigida | constante | fuente |
|---|---|---|---|
| Entregable, antes de test | 123 árboles / PR-AUC val 0,1868 | `ABL_TREES`, `ABL_VAL_PR_AUC` | `phase4_ablation_val.csv` |
| LR, antes de test | `C` = 1 / PR-AUC val 0,1804 / 128 columnas | `LR_C`, `LR_VAL_PR_AUC`, `LR_N_COLS` | `phase5_lr_c_grid.csv` |
| LR de la Fase 2a reproducida (refactor) | `C` = 1 / PR-AUC val 0,2124 | `LR_2A_C`, `LR_2A_VAL_PR_AUC` | `metrics_final.csv`, fila `LogReg,val` |
| Entregable reproduce la Fase 4 en test | PR-AUC 0,1890 / ROC-AUC 0,6655 | `TEST_PR_AUC_F4`, `TEST_ROC_AUC_F4` | `phase4_test_global.csv`, fila `ENTREGABLE (sin payer_code)` |

### Elección de C (val, NO test)
`phase5_lr_c_grid.csv`, columnas `C`, `val_pr_auc`, `val_lift`; denominador val = 19.868
filas (`n_val`), prevalencia 9,7795% (`val_prevalencia_pct`):

| C | PR-AUC val | lift val |
|---|---|---|
| 0,001 | 0,1782 | 1,8225 |
| 0,01 | 0,1796 | 1,8368 |
| 0,1 | 0,1803 | 1,8434 |
| **1 (elegido)** | **0,1804** | **1,8446** |
| 10 | 0,1804 | 1,8444 |

`ganador_en_el_borde` = False en las 5 filas: C = 1 no cae en ningún extremo de la
rejilla, así que la cláusula de "no se amplía" no llegó a aplicarse.
`n_columnas_tras_onehot` = 128 (constante en las 5 filas; el one-hot no depende de C).

### Contraste primario en test (el único con veredicto)
`phase5_test_primary_contrast.csv`, una fila. Δ = `observado` = PR-AUC(entregable) −
PR-AUC(LR); IC95 por bootstrap emparejado de PACIENTES, 2.000 réplicas
(`replicas_usadas` = 2000, `replicas_descartadas` = 0), semilla 42; denominador
`n_test` = 19.867 filas, `n_pacientes_test` = 13.951:

| columna | valor |
|---|---|
| `pr_auc_entregable` | 0,188976 |
| `pr_auc_lr` | 0,182739 |
| `observado` (Δ) | **+0,006237** |
| `IC95_bajo` / `IC95_alto` | **−0,000770 / +0,013005** |
| `IC_excluye_0` | False |
| `umbral_materialidad` / `supera_umbral` | 0,01 / False |
| `VEREDICTO` | **c** — indistinguibles (el IC incluye 0) |

**"Indistinguibles" NO es "equivalentes":** `IC95_alto` = +0,0130 está POR ENCIMA del
umbral de materialidad (0,01), así que estas cifras no descartan una ventaja material de
LightGBM; solo no la detectan. Al citarlo, el extremo superior del IC va siempre con el
veredicto.

### Secundarias en test (descriptivas, SIN veredicto)
`phase5_test_secondary.csv`, mismo bootstrap; `diferencia` = entregable − LR; denominador
test = 19.867 filas (columna `denominador`). La columna `mejor_si` da la dirección:

| `metrica` | `valor_entregable` | `valor_lr` | `diferencia` | IC95 | `IC_excluye_0` | `mejor_si` |
|---|---|---|---|---|---|---|
| `roc_auc` | 0,665465 | 0,659692 | +0,005773 | [+0,000222; +0,011245] | True | dif > 0 |
| `lift` | 1,919422 | 1,856076 | +0,063346 | [−0,007992; +0,134212] | False | dif > 0 |
| `precision_top5%` | 0,268612 | 0,268612 | 0,000000 | [−0,021255; +0,022024] | False | dif > 0 |
| `lift_top5%` | 2,728276 | 2,728276 | 0,000000 | [−0,219044; +0,221888] | False | dif > 0 |
| `brier` | 0,085465 | 0,085788 | −0,000323 | [−0,000608; −0,000034] | True | dif < 0 |

Sin corrección por comparaciones múltiples (no llevan veredicto). Los dos IC que excluyen
el 0 corresponden a diferencias de 0,0058 y 0,0003, y no se citan como hallazgo.

### Globales y líneas base en test
`phase5_test_global.csv`, `n` = 19.867, `prevalencia_%` = 9,8455, umbral de decisión 0,5
en la columna `accuracy_%`:

| `modelo` | ROC-AUC | PR-AUC | `lift_pr_auc` | Brier | `accuracy_%` |
|---|---|---|---|---|---|
| ENTREGABLE (sin `payer_code`) | 0,6655 | 0,1890 | 1,9194 | 0,0855 | 90,1445 |
| LR (C=1, sin `payer_code`) | 0,6597 | 0,1827 | 1,8561 | 0,0858 | 90,1445 |
| base: prevalencia constante = nunca reingresa | 0,5000 | 0,0985 | 1,0000 | 0,0888 | **90,1545** |
| base: regla `number_inpatient > 0` | 0,5971 | 0,1246 | 1,2660 | 0,3344 | 66,5576 |

Los dos modelos quedan por DEBAJO del clasificador trivial en accuracy (90,1445% frente a
90,1545%). **El Brier de la regla (0,3344) no es comparable** con los de los dos modelos:
la regla devuelve 0/1, no probabilidades.

### Top-k en test
`phase5_test_topk.csv`; denominador de `precision_%` = `n_marcados`, de `recall_%` = los
1.956 positivos reales de test tras la corrección:

| `modelo` | `top_k` | `n_marcados` | `aciertos` | `precision_%` | `lift` |
|---|---|---|---|---|---|
| ENTREGABLE | 5% | 994 | 267 | 26,8612 | 2,7283 |
| ENTREGABLE | 10% | 1.987 | 440 | 22,1439 | 2,2491 |
| ENTREGABLE | 20% | 3.974 | 728 | 18,3191 | 1,8607 |
| LR (C=1) | 5% | 994 | **267** | 26,8612 | 2,7283 |
| LR (C=1) | 10% | 1.987 | 429 | 21,5903 | 2,1929 |
| LR (C=1) | 20% | 3.974 | 732 | 18,4197 | 1,8709 |

Los 267 aciertos coinciden en el 5%, pero los vectores de predicción son distintos: se ve
en el 10% (440 frente a 429) y en el 20% (728 frente a 732), y en que PR-AUC, ROC-AUC y
Brier difieren.

### Coeficientes de la LR (para la Fase 5B, sin interpretar en la 5A)
`phase5_lr_coefficients.csv`: 128 filas de columna del `ColumnTransformer` más una fila
`(intercepto)`, con `bloque`, `variable_original`, `coef`, `odds_ratio` y `rank_abs`.
Escalas no comparables entre sí (numéricas por desviación típica, dummies por categoría).

### Nota de procedimiento
Los valores de `acarbose` y `tolazamide` que aparecen en val y no en train afectan a 4 de
las 19.868 filas de val (0,0201%): `acarbose = Down` 3 filas y `tolazamide = Up` 1 fila,
sin solapamiento. Medido solo con train y val.


---

## 14. Fase 5B — fidelidad de SHAP frente a la permutation importance (todo en val)

Prerregistro `19e7ea5`; lista de variables fijada en `425019f`. Modelo: el **entregable**
congelado (sin `payer_code`, 123 árboles), reproducido con su guardarraíl. **Test no se toca en
ninguna parte de la fase.** Denominadores: *val* = 19.868 filas, prevalencia 9,7795% con la
etiqueta corregida; *train+val* = 79.473 filas (solo en la adenda de `admission_type_id`).
R = 30 repeticiones de permutación, semilla 42.

### Conjunto interpretable: 10 variables
`phase5b_interpretable_set.csv`, columnas `rank_shap_entregable`, `media_abs_shap`,
`gana_a_sonda`, `tipo`. Regla: `gana_a_sonda == 10` sobre 10 semillas. **6 numéricas y 4
categóricas.** Suelo de la sonda: media 0,0177, máximo 0,0276 (`phase5b_probe_seeds.csv`).
La tabla de las 43 variables está en `phase5b_noise_probes.csv`.

| # | variable | mean \|SHAP\| | tipo |
|---|---|---|---|
| 1 | `number_inpatient` | 0,29377 | numérica |
| 2 | `discharge_disposition_id` | 0,18275 | categórica |
| 3 | `admission_type_id` | 0,07525 | categórica |
| 4 | `medical_specialty` | 0,07220 | categórica |
| 5 | `diag_1_group` | 0,04981 | categórica |
| 6 | `number_diagnoses` | 0,04106 | numérica |
| 7 | `age` | 0,03996 | numérica |
| 8 | `num_lab_procedures` | 0,03781 | numérica |
| 9 | `number_emergency` | 0,03424 | numérica |
| 10 | `time_in_hospital` | 0,03294 | numérica |

Cambios respecto a las 13 del modelo viejo (`phase5b_probes_vs_old.csv`, columnas `estado`,
`media_abs_shap_viejo`, `media_abs_shap_entregable`): **ENTRA** `admission_type_id`
(0,00860 -> 0,07525; `gana_a_sonda` 0 -> 10); **SALEN** `diabetesMed` (10 -> 7/10),
`diag_3_group` y `diag_2_group` (10 -> 9/10 las dos, con su mean |SHAP| casi igual: 0,03027 ->
0,03022 y 0,02632 -> 0,02607); y `payer_code`, que la quitó la ablación y no las sondas.
Resumen en `phase5b_probes_summary.csv` (`PARADA_POR_TAMANO` = False, 10 >= 8).

Fármacos en el entregable (`phase5b_noise_probes.csv`, columna `gana_a_sonda`): **ninguno supera
las sondas.** `insulin` 6/10 (0,01917), `metformin` 0/10, 10 fármacos con mean |SHAP|
exactamente 0; `diabetesMed` 7/10 y `change` 0/10.

### Tabla del veredicto (la cifra titular de la fase)
`phase5b_faithfulness.csv`, 10 filas. Métrica PRINCIPAL PR-AUC; rangos DENTRO de las 10.
Columnas: `rank_shap`, `pi_prauc_media`, `pi_prauc_sd`, `pi_prauc_p2_5`, `rank_pi_prauc`,
`dif_rangos_prauc`, `pi_distinta_de_cero_prauc`, `DISCREPA_prauc`.

| variable | #SHAP | PI media | PI sd | PI p2,5 | #PI | dif | PI != 0 | DISCREPA |
|---|---|---|---|---|---|---|---|---|
| `number_inpatient` | 1 | 0,06303 | 0,00222 | 0,05821 | 1 | 0 | sí | no |
| `discharge_disposition_id` | 2 | 0,01864 | 0,00126 | 0,01613 | 2 | 0 | sí | no |
| `admission_type_id` | 3 | 0,00298 | 0,00091 | 0,00124 | 5 | 2 | sí | no |
| `medical_specialty` | 4 | 0,00376 | 0,00104 | 0,00153 | 4 | 0 | sí | no |
| `diag_1_group` | 5 | 0,00167 | 0,00070 | 0,00026 | 6 | 1 | sí | no |
| `number_diagnoses` | 6 | 0,00148 | 0,00050 | 0,00053 | 7 | 1 | sí | no |
| `age` | 7 | 0,00104 | 0,00086 | **-0,00027** | 9 | 2 | **NO** | no |
| `num_lab_procedures` | 8 | 0,00021 | 0,00075 | **-0,00140** | 10 | 2 | **NO** | no |
| **`number_emergency`** | 9 | 0,00440 | 0,00061 | 0,00318 | **3** | **6** | sí | **SÍ** |
| `time_in_hospital` | 10 | 0,00112 | 0,00052 | 0,00001 | 8 | 2 | sí | no |

- **Discrepa 1 de 10**: `number_emergency`, por la regla (a) (6 puestos >= 4).
- **`age` y `num_lab_procedures` tienen PI que NO se distingue de cero** (criterio `perm_clara`
  de la Fase 3a: media <= 2 sd) y percentil 2,5 negativo. La regla (b) **no las marca** porque
  solo mira el top 5 de SHAP y están en los puestos 7 y 8. Al citar «1 de 10» hay que decir que
  **ese recuento subestima lo que ve la tabla**, y que es un límite de la regla además de un
  fallo de fidelidad.
  **No es contradicción con que superen las sondas 10/10:** las sondas comparan la magnitud
  SHAP de cada variable con la de variables inventadas, así que son un criterio basado en SHAP
  y NO en rendimiento, y por construcción no pueden detectar SHAP alto con PI nula. Es la
  circularidad ya declarada en el prerregistro: el filtro de interpretabilidad y la magnitud
  auditada son la misma, así que este caso quedaba fuera de su alcance por construcción.
- **Log-loss (robustez, misma tabla, columnas `*_logloss`):** el VEREDICTO coincide en 10 de 10
  (`coinciden_las_dos_metricas` = True en las 10), pero **`pi_distinta_de_cero` NO coincide**:
  con log-loss las 10 son distintas de cero, incluidas `age` (0,00061 ± 0,00016) y
  `num_lab_procedures` (0,00029 ± 0,00014). **No se escribe «las dos métricas coinciden» a
  secas.**

Contexto de las 43 variables: `phase5b_pi_all.csv` (`mean_abs_shap`, `rank_shap_43`,
`pi_prauc_media`, `rank_pi_prauc_43`, `pi_logloss_media`, `pr_auc_base` = 0,1868, `R` = 30,
`n_val` = 19.868).

### Concordancia (descriptiva, SIN veredicto)
`phase5b_concordance.csv`, columnas `spearman_rho`, `p_valor`, `rol`: **0,673** (p 0,033) sobre
las 10 interpretables y **0,882** sobre las 43 de contexto. Con 10 variables dice poco por sí
solo y el conjunto está preseleccionado por SHAP.

### Dirección de los efectos
`phase5b_direction_numeric.csv`, n = 19.868 filas de val. Las 6 numéricas tienen Spearman
valor-SHAP positivo y los 6 coeficientes de la LR de la 5A el mismo signo: **6 de 6**. Solo se
compara el SIGNO (escalas no comparables).

| variable | rho valor-SHAP | coef LR | bloque |
|---|---|---|---|
| `number_diagnoses` | 0,87630 | 0,06699 | num |
| `number_inpatient` | 0,83528 | 0,36314 | counts |
| `num_lab_procedures` | 0,80908 | 0,05052 | num |
| `time_in_hospital` | 0,77722 | 0,00235 | num |
| `age` | 0,68189 | 0,01827 | num |
| `number_emergency` | 0,54527 | 0,08266 | counts |

`phase5b_direction_categorical.csv` (columnas `nivel`, `n`, `shap_medio`, `p25`, `p75`,
`pct_de_val`), niveles de más peso: `discharge_disposition_id` 22 +0,587 (n 376), 5 +0,432
(n 256), 1 -0,152 (n 12.027 = 60,53% de val); `admission_type_id` 3 -0,209 (n 3.695);
`medical_specialty` Nephrology +0,091 (n 332), Emergency/Trauma -0,205 (n 1.565);
`diag_1_group` Circulatory +0,054 (n 5.996), Musculoskeletal -0,099 (n 901).

### Adenda: `admission_type_id` (train+val, descriptivo y sin veredicto)
1. **Reetiquetado por nivel actual** (`phase5b_admtype_relabel_rate.csv`, columna `tasa_%`,
   denominador = `positivos` del nivel en train+val): nivel 3 (programado) **33,441%** (520 de
   1.555) y el **40,279%** de todo lo reetiquetado; nivel 1 10,377% (515 de 4.963); nivel 2
   14,390% (237 de 1.647); `Unknown` 2,147% (19 de 885). Niveles 4 y 7: 1 y 0 positivos, sin
   interpretar.
2. **Siguiente programado por nivel actual** (`phase5b_admtype_next_elective.csv`, columna
   `pct_siguiente_programado`, denominador = `n_con_siguiente_visible`): nivel 3 **39,926%**
   (1.625 de 4.070); nivel 2 17,544% (753 de 4.292); nivel 1 11,479% (1.535 de 13.372);
   `Unknown` 2,455% (58 de 2.363). Tabla cruzada completa con `n` por celda en
   `phase5b_admtype_crosstab.csv`.
   **Denominador, con su etiqueta exacta: 24.100 de 79.473 (30,32%) = ENCUENTROS de train+val
   CON encuentro siguiente VISIBLE.** No confundir con el 28,15% de la Fase 4 (ver la sección 12).
3. **SHAP medio por nivel** (`phase5b_admtype_shap_levels.csv`, leído del CSV de la PI): nivel 3
   **-0,20855** (n 3.695 = 18,598% de val), el único claramente negativo; `Unknown` +0,04920
   (n 2.010), nivel 2 +0,04529 (n 3.708), nivel 1 +0,04373 (n 10.451). Niveles 4 y 7 con n = 2.

**No es un contraste.** Aunque el mecanismo se confirme, no se concluye que la variable sea un
error: una pauta clínica real y un eco de la regla de reetiquetado son compatibles y este diseño
no los separa.

### Cifras que NO existen y que no se deben inventar
El **% de ceros de `number_emergency`** no está medido en ningún CSV del repo. La lectura de que
su mean |SHAP| se hunde por tener muchos ceros mientras la PI premia su efecto en los pocos
valores altos es **RAZONAMIENTO, no medición**, y así se cita.

## 15. Fase 5C — distancia a la frontera de los contrafactuales (todo en val)

Prerregistro `44db202`. Modelo: el **entregable** congelado (sin `payer_code`, 123 árboles),
reproducido con su guardarraíl. **Test no se toca en ninguna parte de la fase.** Denominadores:
*val* = 19.868 filas; *marcados del top 5%* = **994** filas (corte `probs.quantile(0.95)` =
0,191289, `phase5c_frontier_distance.csv`, columna `umbral_top5`); top 10% = 1.987; top 20% =
3.974. Los contrafactuales de la Fase 3c son de OTRO modelo (122 árboles, 44 features con
`payer_code`, target original) y sus cifras **no se comparan** con las de aquí (ver la sección 12).

### Umbral y su comprobación
`probs.quantile(0.95)` = **0,191289**, 994 filas marcadas = `ceil(0,05 · 19.868)`.
`xai_counterfactuals.thresholds()` daría 0,191303 (diferencia **1,4e-05**) y marca el mismo nº de
filas. Se usa el cuantil, como fija el prerregistro.

### Auditoría del refactor (`phase5c_refactor_audit.csv`)
Columnas `n_candidatos_publicado`, `n_candidatos_ahora`, `n_vars_accionables`,
`alternativas_por_var`, `ok`. Los 6 casos de la 3c reproducen **18, 18, 8, 2, 2, 2** y sus
`palancas`. Confirma la inferencia del prerregistro: **2 alternativas por variable accionable** y
**m = 3, 3, 2, 1, 1, 1**. Además, `xai_counterfactuals.py` relanzado completo deja sus **12 CSV
idénticos byte a byte**.

### Tamaño del espacio de acciones (`phase5c_action_space.csv`)
Columnas `n_vars_accionables`, `n_comb_hasta_2`, `n_comb_hasta_3`, `n_comb_completo`,
`n_enumeradas`, `exhaustiva`, `motivo_exhaustividad`.
- **En 7 de los 11 casos `n_comb_hasta_2` == `n_comb_completo`** en E0, así que ahí subir k no
  puede crear ningún contrafactual. En los otros 4, el completo son 26 u 80 frente a 18 o 32.
- **E4, idéntico para los 11 casos** (las alternativas de cada fármaco dependen de sus niveles en
  TRAIN, no del valor del paciente): **20** variables accionables, **1.747** combinaciones con
  ≤ 2, **29.167** con ≤ 3 (las enumeradas) y **4,832·10¹⁰** el producto completo.
- **E4-completo y E5: `exhaustiva = False`.** El prerregistro estimaba ~8e13 para E4-completo; el
  valor medido es 4,832·10¹⁰. Era una cota alta y no cambia el veredicto de no enumerable.

### La tabla del hallazgo (`phase5c_frontier_distance.csv` y `phase5c_ladder.csv`)
Columnas `prob_orig`, `reduccion_necesaria`, `reduccion_maxima_alcanzable`, `deficit`, `cruza`,
`todo_cambio_empeora`, `combinacion_minima`; y `primer_cf`, `n_cf`, `exhaustiva`, `afirmacion`.
Todo en PROBABILIDAD. `deficit > 0` = NO se cruza. Solo los **3 casos marcados** en el top 5%:

| caso | destino | prob | red. necesaria | déficit E0 = E1 = E2 | déficit E3 | déficit E4 |
|---|---|---|---|---|---|---|
| `perfil_alto_1` | 6 | 0,26265 | 0,07136 | +0,06941 | +0,02776 | **+0,01592** |
| `perfil_alto_3` | 6 | 0,20861 | 0,01732 | +0,01156 | **−0,02229** | −0,04112 |
| `falsa_alarma_mayor_riesgo` | 1 | 0,42681 | 0,23552 | +0,25352 | +0,16098 | **+0,14168** |

- **En E0, E1 y E2 ninguno de los 3 cruza**, y los tres escalones dan el mismo déficit.
- **`perfil_alto_1` y `falsa_alarma_mayor_riesgo` no cruzan ni en E4**, con **29.167 candidatos
  enumerados al completo dentro de k ≤ 3**: es una DEMOSTRACIÓN, no una búsqueda fallida.
- **El primer contrafactual aparece en E3** y solo en `perfil_alto_3` (`n_cf` = 12 de 59).
- **La combinación mínima de E3 y E4 es, en los tres casos, `destino → 23`** (hospital de larga
  estancia), el nivel con el SHAP más negativo del destino (−0,3234,
  `phase5b_direction_categorical.csv`). **Se cita SIEMPRE etiquetado como el contrafactual
  engañoso**, nunca como acción recomendada.
- `falsa_alarma_mayor_riesgo` tiene **`reduccion_maxima_alcanzable` NEGATIVA (−0,01800)** en
  E0–E2: su único cambio disponible (destino 1 → 6) SUBE la puntuación. La columna
  `todo_cambio_empeora` que lo marca se **añadió después de ver la salida** y no está
  prerregistrada; no cambia ningún criterio.

### Marcados sin acciones posibles (`phase5c_no_actions.csv`)
Columnas `criterio`, `n_marcados` (denominador), `n_sin_acciones` (numerador),
`pct_sin_acciones`. Espacio **E0**:

| top k | grupo | n marcados | sin NINGUNA variable | sin NINGÚN fármaco |
|---|---|---|---|---|
| 5% | TODOS | 994 | 6 = **0,6036%** | 172 = **17,3038%** |
| 5% | 22/5 (destino congelado) | 62 | 6 = 9,6774% | 6 = 9,6774% |
| 5% | otros destinos | 932 | 0 = 0% | 166 = 17,8112% |
| 10% | TODOS | 1.987 | 20 = 1,0065% | 386 = 19,4263% |
| 10% | 22/5 | 176 | 20 = 11,3636% | 20 = 11,3636% |
| 10% | otros destinos | 1.811 | 0 = 0% | 366 = 20,2098% |
| 20% | TODOS | 3.974 | 49 = 1,2330% | 722 = 18,1681% |
| 20% | 22/5 | 333 | 49 = 14,7147% | 49 = 14,7147% |
| 20% | otros destinos | 3.641 | 0 = 0% | 673 = 18,4839% |

**CÓMO SE CITA, y es importante porque la cifra refuta la frase fácil:** NO se escribe «a la
mayoría de los marcados no se les puede ofrecer ninguna acción». Es una **minoría (17,30% del top
5%)** la que se queda sin ninguna palanca de medicación. La frase correcta tiene dos partes:
**(i)** al **17,30%** no se le puede ofrecer nada que no sea mover el destino, que no es una
acción; **(ii)** al **82,70%** restante sí, pero esa palanca no lo mueve al otro lado de la
frontera. El (ii) se apoya **primero en la enumeración de E4** de esta misma fase —dos de los
tres casos marcados no cruzan ni con los 21 fármacos libres y k ≤ 3, con los 29.167 candidatos
enumerados al completo (`phase5c_ladder.csv`, `primer_cf = False` con `exhaustiva = True`), que
es una demostración— y **después en las sondas de la Fase 5B** (`phase5b_noise_probes.csv`,
columna `gana_a_sonda`: `insulin` 6/10, `metformin` 0/10, diez con mean |SHAP| exactamente 0),
que son evidencia global y más débil. El (i) es sobre la EXISTENCIA de la palanca, el (ii) sobre
su EFECTO.
En el grupo 22/5 **los dos criterios coinciden por construcción** (con el destino congelado, «sin
fármaco» = «sin nada»).

### Elegibilidad de los 11 casos (`phase5c_eligibility.csv`)
Columnas `prob_modelo_viejo`, `prob_entregable`, `marcado_5%`, `marcado_10%`, `marcado_20%`,
`motivo`. **Solo 3 de los 11 siguen marcados** en el top 5% del entregable: `perfil_alto_1`
(0,2627), `perfil_alto_3` (0,2086) y `falsa_alarma_mayor_riesgo` (0,4268); los mismos 3 en el top
10%.
- **Se declara: el análisis caso por caso descansa en 3 pacientes, que es una base estrecha.** La
  generalización se apoya en la **medida de población (994 filas marcadas)** y en los tamaños del
  espacio, no en los 3 casos.
- **Los dos casos «alto sin ingresos» caen fuera** (0,2617 → 0,1214 y 0,2303 → 0,1454), coherente
  con la Fase 4: el entregable deja de marcar a esa población y su subgrupo A bajó de 440 a 212.
- Los 11 **no se reeligen**. Los 8 no marcados llevan `marcado_top5 = False` y su `deficit`
  negativo **NO** se lee como «sí cruza»: para ellos la pregunta no se plantea.

### Por qué solo había 6 casos elegibles en la Fase 3c (punto f del profesor)
`xai_counterfactuals_diag_elegibilidad.csv`: la elegibilidad era `prob >= 0,19545` (top 10% del
modelo VIEJO), y **5 de los 11 casos se eligieron POR puntuar bajo** — `riesgo_mediano` 0,0971,
`riesgo_bajo` 0,0554, `modelo_bajo_con_ingresos_reingresa` 0,0826,
`modelo_bajo_con_ingresos_no_reingresa` 0,0937, `reingreso_menor_riesgo` 0,0338. El «solo 6» es
consecuencia del diseño de la selección, no de la búsqueda.

### Cifras que NO existen y que no se deben inventar
No hay en esta fase ninguna medida de **E5** ni de **E4 sin límite de k**: los dos están
declarados NO ENUMERABLES, y de ellos **no se reporta ningún resultado de búsqueda**. Lo único
que se afirma de E5 es que **ahí el contrafactual existe**, y eso se apoyará en el vecino más
cercano del paso 2, que todavía NO está medido: en este paso no hay ninguna cifra que lo
sostenga, así que no se cita como medido.

