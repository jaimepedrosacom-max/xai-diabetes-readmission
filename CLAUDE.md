# Proyecto: XAI — Reingreso hospitalario en diabéticos

## Contexto
Proyecto final de la asignatura de Explicabilidad de IA (máster en IA).
Dataset: UCI "Diabetes 130-US Hospitals for Years 1999-2008"
(https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
~100.000 encuentros hospitalarios, variables demográficas, diagnósticos,
medicación y tiempo de estancia.

## Tarea
Clasificación binaria: ¿el paciente reingresa en <30 días tras el alta?
Stakeholder: gestor de calidad hospitalaria que decide a quién hacer
seguimiento post-alta.

## Requisitos del proyecto (no negociables)
- Modelo principal NO overfitteado: usar train/val/test con split estratificado,
  early stopping, y reportar métricas de val y test por separado.
- Al menos 3 técnicas XAI complementarias: una global, una local, un sanity check.
- Las explicaciones deben llevar a una acción concreta (quitar features con
  leakage, o recomendación clínica).
- Reflexión crítica final: dónde pueden fallar o engañar las explicaciones.

## Cómo quiero trabajar contigo
- Antes de escribir código, explícame brevemente qué vas a hacer y por qué.
- Ve fase a fase (datos → baseline → modelo principal → XAI → acción → informe).
  No adelantes fases sin que te lo pida.
- Comenta el código pensando en que tengo que defenderlo oralmente: nada de
  "magia" sin explicar.
- Si usas un snippet o enfoque de un tutorial/paper conocido, dímelo para citarlo
  en el README (es requisito de integridad académica de la asignatura).
- Cada cifra que reportes debe venir con su fuente (CSV y columna) y con su
  denominador explícito. No estimes a ojo ni redondees "de memoria".

## Stack
Python, pandas, scikit-learn, lightgbm, shap, dice-ml (o alibi) para
counterfactuals, matplotlib/seaborn para gráficas.

## Estado del proyecto y decisiones tomadas (actualizado: 2026-11-12)

### Estado
- Hecho: datos, baseline, modelo principal y evaluación final en test
  (commits `e4a0354`, `d9a017c`, `ff22295`, `16f017b`).
- Fase 3 (XAI) COMPLETA: 3a (`src/xai_shap.py`, global), 3b (`src/xai_sanity.py`,
  sanity checks + auditoría del destino), 3c (`src/xai_counterfactuals.py`,
  contrafactuales) y 3d (`src/xai_local.py`, explicaciones locales de los 11
  casos).
- Commits de la Fase 3: `a86efe7` y `8055256` (3a), `351c415` (3b), `4f2be45`
  (3c), `b8a6270` (3d).
- Cobertura del requisito de "3 técnicas XAI complementarias": global = SHAP
  (3a); local = SHAP local (3d) y contrafactuales DiCE + enumeración (3c);
  sanity check = aleatorización de etiquetas, barajado guiado, sondas de ruido y
  estabilidad entre folds (3b).
- Fase 4 COMPLETA. Código cerrado: corrección del target, predicción prerregistrada, equidad, ablación de
  `payer_code`, consulta ÚNICA a test (la segunda y última vez que se toca test), README, notebook
  (`notebooks/analisis.ipynb`) y figuras (`src/phase4_figures.py`).
- Fase 5A EN CURSO: prerregistrada (comparar el entregable con una regresión
  logística, a petición del profesor), pendiente de etapa val y etapa test. Su etapa
  test es la TERCERA consulta a test, así que el "y última" del punto anterior deja
  de valer.
- PENDIENTE, fuera del código:
  - La memoria la redacta Jaime en LaTeX desde `report/` (`main.tex`, `references.bib`) en Overleaf. El repo
    solo tiene el esqueleto, sin contenido y sin compilar. Al terminar, copiar el `.tex` final al repo.
  - Dar acceso al repo privado a quien corrija.
  - La fecha de entrega no aparece en el PDF del enunciado: confirmarla en la plataforma del curso.
- Enunciado: máximo 8 páginas (5-6 recomendadas), con Introduction / Data and Methods / Results / Actions and
  Insights / Discussion.
- Entregable final: modelo corregido SIN `payer_code`, guardarraíl 123 árboles /
  PR-AUC val 0,1868; en test, con la etiqueta corregida, lift 1,9194.
- Entorno: venv fuera del proyecto, en `C:\venvs\xai-final` (dentro del
  proyecto la ruta supera el límite de 260 caracteres de Windows).
  Versiones fijadas en `requirements.txt`. `dice-ml` está fijado a propósito en
  0.12: el arreglo de reproducibilidad de la Fase 3c depende de detalles
  internos de esa versión.

### Limpieza (`src/data.py`, `src/features.py`, `src/preprocessing.py`)
- Filas eliminadas: `gender` inválido y altas por fallecimiento/hospicio
  (`discharge_disposition_id` 11, 13, 14, 19, 20, 21), siguiendo a Strack et
  al. (2014). Quedan 99.340 encuentros.
- Columnas eliminadas: `weight` (97% missing), `examide` y `citoglipton`
  (constantes). `encounter_id` pasa a ser el índice.
- Missing como categoría, sin imputar: `A1Cresult` y `max_glu_serum` a
  `NotMeasured` (no realizar la prueba es información);
  `medical_specialty`, `payer_code` y `race` a `Missing`.
- Categorías con <1% de frecuencia colapsadas en `Other`, calculadas solo en
  train. `payer_code` se mantiene para analizarla como posible proxy de equidad.
- ICD-9: `diag_1` replica la Tabla 2 de Strack et al.; `diag_2/3` añaden
  `Endocrine` (240-279 sin 250), que es extensión nuestra y no del paper.
- Códigos desconocidos de los IDs administrativos unidos en `Unknown`, según
  `IDS_mapping.csv`.

### Split (60/20/20, `StratifiedGroupKFold` por `patient_nbr`, SEED=42)
- Por qué agrupado: el 23,5% de los pacientes tiene varios encuentros. Con un
  split por filas, el 41,7% de las filas de test tiene a su paciente en train.
- El efecto medido es pequeño (PR-AUC +0,006 con el split aleatorio): el
  argumento es metodológico, no que las métricas se inflen mucho.
- Por qué no solo el primer encuentro: baja la tasa de positivos de 11,39% a
  8,98% y descarta a los pacientes recurrentes (17,13%), que son los que más
  interesan al gestor.
- `number_inpatient/outpatient/emergency` se refieren al año previo al
  encuentro (metadatos de UCI), así que no contienen información futura.

### Modelo definitivo
- LightGBM con categóricas nativas: `num_leaves=31`, `min_child_samples=20`,
  `learning_rate=0.03`, 122 árboles (early stopping sobre val con
  `average_precision`, patience 100). Sin `class_weight`, sin reentrenar con
  train+val.
- Elegido sobre regresión logística (C=1) por val y confirmado con CV agrupada
  de 5 folds dentro de train: PR-AUC 0,2268 ± 0,0022 frente a 0,2046 ± 0,0040,
  gana en 5 de 5 folds.
- **Test ya gastado:** LightGBM ROC-AUC 0,667 / PR-AUC 0,227 (LR: 0,659 / 0,210).
  No se usa test para decidir nada más. Si el modelo cambia en la Fase 4, el
  antes/después se mide en val y CV, y cualquier nueva consulta a test se declara
  en el informe.
- El k del seguimiento (5/10/20%) queda abierto: lo decide el gestor.
- AVISO (viene de la Fase 3d): el grupo donde el modelo discrepa de la regla simple
  `number_inpatient > 0` —top 10% de val con `number_inpatient == 0`, n=440— tiene un
  69,3% de destinos 22/5, los que la Fase 3b señala como contaminados. Buena parte de
  la ventaja sobre esa regla (0,227 frente a 0,142 de PR-AUC en test, en
  `reports/metrics_final.csv`) se apoya en ese grupo. No está medido cuánta parte de
  la ventaja es artefacto, y no se afirma que lo sea toda. Ver la Fase 3d y su
  predicción prerregistrada para la Fase 4.

### Fase 3a — SHAP global (`src/xai_shap.py`)
- TreeSHAP (Lundberg et al., 2020), `feature_perturbation="tree_path_dependent"`,
  sobre val y con el booster truncado a los 122 árboles de la mejor iteración.
  Una barra por variable original (44): LightGBM divide directamente por la
  categórica, así que no hay dummies.
- Por qué val y no test: de estas explicaciones salen decisiones (Fase 4), y test
  no debe influir en ninguna.
- Doble validación: aditividad (`valor base + suma de contribuciones = log-odds`,
  error < 1e-6) y coincidencia con `pred_contrib` nativo de LightGBM.
- Top drivers consistentes entre SHAP, permutación y contribución en LR:
  `number_inpatient` (#1) y `discharge_disposition_id` (#2), con caídas de PR-AUC
  al permutar de 0,070 y 0,057 — muy por encima del tercer puesto por permutación (`payer_code`, 0,0074; el tercero por SHAP, `diag_1_group`, tiene 0,0036). A
  partir de ahí el orden entre métodos NO es fiable: el ruido de permutación es
  del mismo orden que la señal.
- Los tres métodos miden cosas distintas y no deben coincidir del todo: SHAP mide
  cuánto USA el modelo la variable; la permutación, cuánto RENDIMIENTO se pierde
  al romperla; la LR, lo mismo que SHAP pero en el modelo lineal.
- `payer_code` sale #4 en SHAP. Se mantuvo a propósito en el modelo (ver Limpieza)
  para analizarla como posible proxy de equidad → Fase 4.
- Limitación a discutir en la reflexión crítica: `tree_path_dependent` estima los
  efectos siguiendo las ramas, así que reparte crédito según las correlaciones con
  las que se entrenó el modelo.
- Salidas: `reports/xai_global_comparison.csv`, `reports/xai_shap_levels.csv`, y
  las figuras `shap_global_bar.png`, `shap_beeswarm_numeric.png`,
  `shap_levels_categorical.png`.

### Fase 3b — Sanity checks y auditoría del destino (`src/xai_sanity.py`, `src/audit_discharge_leakage.py`)
Aquí está el HALLAZGO CENTRAL del proyecto. Los criterios de los sanity checks
están más abajo, en su propia sección; esto son los resultados.

#### Dos problemas distintos en el target `readmitted_30d`
1. **Fuga concentrada en destinos al alta concretos.** Código 22 (rehabilitación):
   de los `<30` con siguiente encuentro visible, el 57,2% tiene diagnóstico V57
   (cuidados/rehabilitación) y el 68,2% tiene V57 o es un ingreso programado —
   probable continuación de cuidados registrada como "reingreso", no un reingreso
   no planificado. Descontando esos casos, la ventaja de riesgo del código 22 baja
   de 2,99x a 1,93x frente a casa: tasa bruta 27,76 % frente a 9,29 % (`pct_lt30`); sin V57 ni programados, 15,64 % frente a 8,12 % (`tasa_sin_V57_o_prog`). El
   código 5 (otra institución) confirma el patrón a medias: 55,2% de V57 o
   programado, pero el componente V57 puro no es estable entre train y val. Los
   códigos 28 (psiquiátrico) y 15 (swing bed) tienen SHAP igual de alto SIN este
   patrón: sin explicar, posible riesgo real, muestra muy pequeña (n_val 29 y 12).
2. **Problema más profundo y general: el target cuenta cualquier reingreso, no
   solo los no planificados.** De 6.503 positivos con siguiente encuentro visible,
   el 19,9% tiene V57 o ingreso programado. El componente V57 se concentra en
   pocos destinos (46% en el código 22), pero el componente "programado" está
   REPARTIDO: el 43,6% de esos casos viene de altas a CASA. Esto no se arregla
   tratando `discharge_disposition_id`: es una limitación de construcción del
   target que afecta a todo el dataset y se declara en el informe como limitación
   independiente del hallazgo del código 22.

#### La visibilidad del siguiente encuentro es PEOR justo en los dos códigos del hallazgo
Medido en `audit_discharge_leakage.csv` (`1 - n_visibles / n_lt30`, train+val): el
37,8% de los `<30` del código 5 no tiene encuentro posterior visible (76 de 201) y el
36,0% del código 22 tampoco (159 de 442), frente al 27,3% de las altas a casa (1.223
de 4.481) y al 28,15% global (`pct_sin_visible` en `audit_discharge_global.csv`).
Dos consecuencias, fijadas ANTES de implementar la corrección de la Fase 4:
- (a) **Cota inferior y estimación son dos cosas distintas, y hay que decirlas por
  separado.** Lo que es COTA INFERIOR es el NÚMERO de positivos contaminados
  detectados: 193 de los 442 `<30` del código 22 (`n_V57_o_prog` en
  `audit_discharge_shares.csv`), porque entre los 159 casos sin encuentro visible solo
  puede haber más, nunca menos. El PORCENTAJE (68,2%) es una ESTIMACIÓN medida solo
  sobre los 283 visibles, y extrapolarla a los 442 asume que los invisibles se parecen
  a los visibles. Si los 159 invisibles estuvieran todos limpios, la tasa sobre el
  total bajaría al 43,7% (193 de 442). La dirección del sesgo es DESCONOCIDA: los
  invisibles podrían estar más o menos contaminados que los visibles y no tenemos con
  qué decidirlo. Por eso el porcentaje se reporta SIEMPRE con su denominador (casos
  visibles), nunca extrapolado al total.
- (b) **El reetiquetado de la Fase 4 corregirá MENOS precisamente donde más falta
  hace.** Solo se pueden reetiquetar los casos con encuentro posterior visible, así que
  el residuo sin corregir será mayor en los códigos 5 y 22 (37,8% y 36,0% de sus
  positivos) que en el resto. El antes/después hay que leerlo sabiendo que el grupo
  contaminado queda parcialmente contaminado después de corregir.

#### OJO al verificar estas cifras: cada una tiene su propio CSV y su denominador
- Las cifras globales del punto 2 están en `reports/audit_discharge_global.csv`
  (una fila, `grupo = train+val`): `n` 79.473, `n_lt30` 9.051, `n_visibles` 6.503,
  `pct_sin_visible` 28,15, `pct_V57` 5,41, `pct_programado` 18,18,
  `pct_V57_o_prog` 19,85, `tasa_sin_V57` 10,95, `tasa_sin_V57_o_prog` 9,76.
- El reparto por destino de esos positivos está en
  `reports/audit_discharge_shares.csv` (`n_V57_o_prog`, `pct_V57_o_prog`, `n_V57`,
  `pct_V57`): código 1 = 43,61% del componente V57-o-programado (563 de 1.291) y
  código 22 = 46,02% del componente V57 puro (162 de 352). Incluye TODOS los
  destinos, así que cada columna de porcentaje suma 100.
- **Estos dos CSV no existían hasta el commit que los añadió:** antes las cifras
  solo se imprimían por consola, y por eso el informe debe citarlas desde el CSV y
  no desde una salida de terminal. No se cambió ningún cálculo al persistirlas:
  `audit_discharge_leakage.csv` sale idéntico byte a byte tras relanzar el script.
- CUIDADO con el denominador: sumando `audit_discharge_leakage.csv` (la tabla por
  destino) salen 6.497 visibles y 19,8%, no 6.503 y 19,9%. Es un SUBCONJUNTO, no
  una discrepancia: esa tabla aplica `MIN_N = 30` y deja fuera 53 filas de destinos
  pequeños, mientras el bloque global usa todo train+val. Se nota en el reparto: el
  código 9 aparece en `audit_discharge_shares.csv` y no en la tabla por destino.
- Sigue habiendo una cifra SOLO en consola: la réplica train/val del código 5
  (`pct_V57` 40,0% en train frente a 14,3% en val, mientras el código 22 se mantiene
  en 57,2% / 57,3%), que es la base de "el componente V57 puro no es estable entre
  train y val". Verificada recalculándola, pero si se cita en el informe hay que
  decir que viene del bloque de réplica por split que el script imprime.

#### Lo que esto enseña sobre los sanity checks (material para la reflexión crítica)
- Los sanity checks PASAN todos los criterios prerregistrados y NO detectan este
  problema. Dos lecciones, que son de las partes más defendibles del trabajo:
  - **"Estabilidad no es validez":** el código 22 es estable entre folds
    precisamente porque todos los folds comparten la misma etiqueta contaminada.
  - **"La aleatorización de etiquetas no distingue señal real de artefacto de
    etiqueta":** ambas colapsan igual al romper la etiqueta.
- Aviso activo (no fallo): la correlación del ranking global bajo etiquetas
  aleatorias es 0,856 (umbral de aviso ≥ 0,7; media de `rho_ranking_global` sobre las 10 permutaciones de `xai_sanity_label_randomization.csv`, 0,8558). Indica que el orden medio de la
  tabla refleja en parte facilidad estructural de split, no solo señal. El top-1
  (`number_inpatient`) SÍ es sensible a la señal real: cae del puesto 1 al 10-20
  bajo aleatorización.
- **Solo 13 de 44 variables superan las sondas de ruido en las 10 semillas.** Las
  demás NO se interpretan, según la regla fijada antes de ver resultados. Esa
  lista es la que lee la Fase 3d para su columna `interpretable`, desde
  `reports/xai_sanity_noise_probes.csv` (`gana_a_sonda == 10`).
- Consecuencias que hay que declarar y no esquivar:
  - `A1Cresult` NO supera las sondas, así que **no se puede concluir nada sobre la
    hipótesis central del paper de referencia** (Strack et al., 2014).
  - `race` y `gender` tampoco, así que no se hacen afirmaciones directas sobre su
    peso. Eso NO descarta proxies vía otras variables, que se analizan en Fase 4.
  - Ningún fármaco llega a 10/10 (`insulin` se queda en 8/10). Esto es lo que
    sostiene, junto con el espacio de acciones de la 3c, la conclusión de que la
    única palanca accionable que el modelo usa de forma distinguible del ruido es
    el destino al alta.

#### Impacto cuantificado en las métricas del gestor
- Quitando los códigos 22 y 5 (632 filas, el 3,2% de val): PR-AUC en val baja de
  0,2308 a 0,2074 y ROC-AUC de 0,6772 a 0,6662; la tasa de positivos pasa del
  11,39% al 10,84%; el SHAP medio de `discharge_disposition_id` baja de 0,204 a
  0,180 y su caída al barajar de 0,057 a 0,031.
- Lectura: el modelo conserva señal real más allá de este problema, pero una parte
  medible de su rendimiento viene de ahí.

#### Decisión pendiente para la Fase 4 (opciones ya planteadas, sin decidir)
- **Reetiquetar a 0** los `<30` cuyo siguiente encuentro es V57 o programado.
  Preferencia de Jaime: mantiene la población, y deja incertidumbre residual en el
  28,15% de los positivos que no tiene encuentro posterior visible (2.548 de 9.051 en
  train+val), que se reporta como límite.
- **Excluir esas altas:** más simple, pero deja sin predicción a una población real.
- **Solo quitar la feature:** RECHAZADO, no arregla la etiqueta contaminada.
- Cualquier corrección cambia el problema: el 0,227 de test deja de ser comparable
  y toda nueva consulta a test se declara explícitamente.

#### Otras salidas de esta fase
- `src/xai_cases.py` fija los 11 casos locales con regla y semilla, sin mirar las
  explicaciones finales, y `xai_sanity.py` los escribe en
  `reports/xai_local_cases.csv`. De ahí los consumen la Fase 3c y la 3d.
- `reports/xai_sanity_local_stability.csv` guarda la estabilidad del top-3 local
  entre folds y el `n_grupo`/`tasa_real_%` de cada caso.
- `reports/audit_discharge_leakage.csv` tiene la auditoría por destino
  (train+val): `pct_lt30`, `pct_V57`, `pct_programado`, `pct_V57_o_prog`,
  `tasa_sin_V57_o_prog`, `shap_medio_val`.

### Fase 3c — Contrafactuales (`src/xai_counterfactuals.py`)
- Dos métodos en paralelo: DiCE (Mothilal et al., 2020, `method="random"`) y
  enumeración exhaustiva de hasta 2 cambios. Las cifras del hallazgo y de cobertura
  salen de la enumeración, que es determinista; DiCE muestrea y no prueba
  inexistencia, así que solo se usa para auditarla y como el método que pedía el
  plan (ver más abajo qué depende de DiCE y qué no).
- Espacio de acciones (NO se toca para forzar resultados): hasta 2 cambios; solo
  fármacos que el paciente ya toma, entre Steady/Up/Down; destino al alta solo
  hacia 1, 6 o 3, y CONGELADO si el destino es 22 o 5 (la auditoría de
  `audit_discharge_leakage.py` indica que parte de su efecto viene de cómo se
  define la etiqueta). `A1Cresult` y `max_glu_serum` congeladas: se puede pedir la
  prueba, no elegir el resultado. `change` y `diabetesMed` son derivadas.
- `change` solo se puede recalcular en un sentido: Up/Down implica Ch, pero hay
  11.425 filas de train con Ch sin ningún Up/Down (`check_derived_rules` en `src/xai_counterfactuals.py`, impreso por `main()`). Cuando un candidato elimina
  todos los Up/Down del original, `change` es ambiguo y se puntúa el PEOR caso.
- Reproducibilidad de DiCE: `random_seed` por sí solo NO basta, y se vio en los
  resultados (dos llamadas sobre el mismo paciente devolvían conjuntos distintos
  de propuestas, incluso dentro del mismo proceso y con `PYTHONHASHSEED=0`, así
  que no era la aleatorización del hash de cadenas). Causa localizada en el código
  de dice_ml 0.12, `explainer_interfaces/dice_random.py`: `random_seed` solo se
  propaga a `get_samples`, mientras que otros dos pasos usan el estado GLOBAL de
  numpy sin semilla — `np.random.choice` para elegir qué variables varía cada
  muestra (línea 114) y `cfs_df.sample(total_CFs)` para quedarse con `total_CFs`
  candidatos (línea 135; pandas sin `random_state` también usa `np.random`
  global). Arreglado fijando `np.random.seed(D.SEED)` y `random.seed(D.SEED)`
  justo antes de cada llamada, sin cambiar el método (sigue siendo
  `method="random"`). Comprobado: dos llamadas en el mismo proceso y dos procesos
  distintos dan lo mismo, y dos ejecuciones completas del script dan los 12 CSV
  idénticos byte a byte, incluido `xai_counterfactuals_DEMO_ENGANOSO.csv`.
- Qué depende de DiCE y qué no: el hallazgo de los 6 casos y todas las cifras de
  cobertura salen de la enumeración exhaustiva, que es determinista. El control
  positivo (d) y la auditoría agregada SÍ dependen de DiCE, y tras fijar las
  semillas son reproducibles. Sus números no cambiaron al arreglarlo (3/3 y 29/30,
  con el mismo encuentro fallido), así que los criterios prerregistrados no se
  reevaluaron.
- Reproducibilidad del beeswarm de SHAP (`src/xai_shap.py`, `plot_numeric_beeswarm`): el
  método del paquete `shap` dibuja el beeswarm con `np.random` del estado GLOBAL, sin
  semilla (`shap/plots/_beeswarm.py`, líneas 383 y 400: desempate y jitter vertical de los
  puntos con el mismo valor). Con los mismos valores SHAP, el PNG cambiaba entre ejecuciones
  (12.329 píxeles distintos al relanzar, con la misma salida numérica). Se fija
  `np.random.seed(D.SEED)` justo antes del dibujo; comprobado: dos ejecuciones dan el mismo
  hash. No cambia ningún valor SHAP ni ningún CSV.
- Observación sobre el tooling, para la reflexión crítica del informe: dice-ml (DiCE) y shap
  son dos librerías de XAI que usan aleatoriedad GLOBAL sin semilla en caminos que usamos. En
  ambas hubo que localizar la fuente leyendo código y fijar la semilla a mano. Una explicación
  que cambia entre ejecuciones sin cambiar el modelo no se puede auditar, y la reproducibilidad
  de una explicación es parte de su validez.

#### Resultado: 0 contrafactuales para los 6 casos elegibles — es un hallazgo, no un bug
- De los 11 casos locales (fijados en la Fase 3b con `src/xai_cases.py`), 6 son
  elegibles (prob >= umbral del top 10% = 0,1954 en val). Los otros 5 no los marca
  el modelo: no hay nada que bajar.
- Veredicto por los criterios prerregistrados antes de ver la salida: HALLAZGO
  REAL. Diagnósticos en `reports/xai_counterfactuals_diag_*.csv`:
  (a) el Scorer reproduce `sk_model.predict_proba` con diferencia 0,0 (tol 1e-9),
  y umbral y puntuación están ambos en PROBABILIDAD, no en log-odds;
  (b) 14 de 22 cambios de un solo lever mueven la puntuación, con dtypes y
  `categories` intactos en 22/22 (ningún valor nuevo se vuelve NaN ni categoría
  desconocida, que LightGBM leería como missing en silencio);
  (c) el mínimo alcanzable queda por encima del umbral en todos los casos
  marcados: en el top 10%, entre 0,017 y 0,322 por encima del corte (hasta 0,362
  contando todos los umbrales en que cada caso está marcado);
  (d) control positivo: DiCE encuentra contrafactual en 3/3 casos donde la
  enumeración prueba que existe, y en 29/30 de la auditoría agregada (cifras
  reproducibles tras fijar las semillas globales). El fallo es un caso con un
  único contrafactual válido: probablemente muestreo aleatorio, no verificado.
- Cobertura en val, por enumeración (ojo al denominador de cada cifra):
  top 10% completo, 451 de 1.987 marcados tienen contrafactual (22,70%; 448 + 3, de `xai_counterfactuals_population.csv`); dentro
  del grupo "otros destinos", 448 de 1.518 (29,51%); dentro del grupo 22/5 con el
  destino congelado, 3 de 469 (0,64%). Sin tocar el destino (solo fármacos):
  59 de 1.987 en el top 10% completo (2,97%), 56 de 1.518 en "otros destinos"
  (3,69%).
- Dónde está la accionabilidad: de los 448 pacientes con contrafactual del grupo
  "otros destinos" en el top 10%, en 419 el MEJOR contrafactual (menos cambios y,
  a igualdad, menor puntuación) toca `discharge_disposition_id`, y en 356 de ellos
  es concretamente destino → 1. Solo en 29 de los 448 el mejor contrafactual no
  toca el destino. Los cambios de medicación mueven la puntuación <0,03 en los 6
  casos analizados (diagnóstico (b)); no se ha medido ese rango en la población.
- Por qué los 6 casos dan 0: 3 tienen el destino congelado (códigos 5, 5, 22), 1
  ya está en destino 1, y los 2 de destino 6 solo ganan 0,002 moviéndose a 1.
  `falsa_alarma_mayor_riesgo` (140533200) es el caso límite esperado: su mínimo
  alcanzable (0,5173) está POR ENCIMA de su puntuación original (0,5170).
- Matiz para el informe: los 6 casos se eligieron en la Fase 3b por interés
  explicativo, no por accionabilidad, así que están sesgados hacia probabilidades
  altas. El límite es real, pero en estos 6 es peor que en la población.
- Lectura para el gestor: el modelo sirve para PRIORIZAR a quién seguir, no para
  decidir QUÉ hacer. La accionabilidad se concentra en `discharge_disposition_id`,
  pero el destino al alta refleja en buena parte la gravedad del paciente
  (confusión), no una decisión libre: un contrafactual del tipo "mandarlo a casa"
  baja la puntuación sin ninguna evidencia de que baje el riesgo. Esto es distinto
  del problema de fuga de los códigos 22 y 5, que están congelados y no generan
  contrafactuales; los contrafactuales de destino salen de 3, 6, 15, etc. hacia 1,
  y en `audit_discharge_leakage.csv` el código 3 tiene un `pct_V57_o_prog` (16,5)
  parecido al del código 1 (17,3).
- `reports/xai_counterfactuals_DEMO_ENGANOSO.csv` es el ejemplo de cómo un
  contrafactual puede engañar (mismo paciente con y sin congelar el destino 22).
  No es análisis ni recomendación y no entra en ningún resumen.

### Fase 3d — Explicaciones locales SHAP de los 11 casos (`src/xai_local.py`)
- Los casos NO se eligen aquí: vienen de `src/xai_cases.py` con regla fija y semilla,
  sin mirar las explicaciones finales. Los 3 `perfil_alto_*` son centroides de k-means
  sobre SHAP del top 10%, así que son representativos de su grupo por construcción.
- Escala: se reporta LOG-ODDS, donde SHAP es exactamente aditivo. NO se traduce cada
  contribución a puntos de probabilidad (la logística no es aditiva). Comprobada la
  aditividad caso a caso: `base + suma = log-odds`, error < 1e-6 en los 11.
- `interpretable` se LEE de `xai_sanity_noise_probes.csv` con la regla de la Fase 3b
  (`gana_a_sonda == 10`), sin lista escrita a mano. Salen 13 variables de 44.
  AVISO: las sondas son una prueba GLOBAL; aplicarlas al nivel local es una extensión
  prudente, NO una validación de la explicación local.
- `destino_contaminado` marca los casos con destino 22 o 5: su contribución del destino
  se reporta pero no se lee como riesgo (fuga de la etiqueta, Fase 3b).
- La estabilidad del top-3 local entre folds y el `n_grupo`/`tasa_real_%` de cada caso
  salen de `xai_sanity_local_stability.csv`, que ahora escribe `xai_sanity.py` (antes
  solo se imprimían). Al añadirlo se relanzó ese script y sus 6 CSV previos salieron
  idénticos byte a byte.
- En 6 de los 11 casos el grupo se define POR el resultado, así que su `tasa_real_%` es
  100% o 0% por construcción: se marca como trivial para no leerla como riesgo.

#### Resultados de la Fase 3d
- Concentración: `number_inpatient` está en el top-3 local de 11/11 casos y
  `discharge_disposition_id` en 9/11. Los que no siguen el patrón global son
  `perfil_alto_1` y `perfil_alto_3`, cuyos puestos 2 y 3 interpretables son
  `number_emergency` (global #9) y `diag_2_group` (global #13); como son centroides,
  describen un grupo del top 10%, no un paciente excéntrico.
- `perfil_alto_1` y `perfil_alto_3` son casi el mismo perfil, medido: sus vectores SHAP
  correlacionan 0,987 (`xai_local_perfiles_geometria.csv`) y sus centroides son el par MÁS CERCANO de los tres (distancia
  0,650, frente a 1,108 y 1,630). La diferencia está concentrada en la INTENSIDAD de
  `number_inpatient` (+1,279 con valor 5 frente a +0,703 con valor 4, que son 0,575 de
  los 0,591 de distancia entre los dos encuentros, vectores SHAP, en `xai_local_perfiles_geometria.csv`) y en el signo de `diag_2_group`
  (+0,062 Endocrine frente a -0,055 Genitourinary). Consecuencia: con k=3, dos de los
  tres perfiles cuentan la misma historia a distinta escala, y solo `perfil_alto_2`
  (el del destino contaminado) es cualitativamente distinto. Para el informe conviene
  presentarlos como dos intensidades de un mismo perfil, no como dos perfiles. Dicho
  de otro modo: la parte de "perfiles de riesgo" de la selección aporta 2 perfiles y
  no 3. `k=3` se fijó a priori en `xai_cases.py`, así que NO hay selección a
  posteriori; simplemente sobreparticionó.
- En los 3 casos con destino 22/5 el destino es la contribución #1, entre 7,0 y 9,4
  veces la #2 real de cada caso EN VALOR ABSOLUTO: `perfil_alto_2` +0,954 frente a
  `payer_code` +0,101 (9,4x); `modelo_alto_sin_ingresos_reingresa` +0,936 frente a
  `number_inpatient` -0,134 (7,0x); `modelo_alto_sin_ingresos_no_reingresa` +1,127
  frente a `number_inpatient` -0,146 (7,7x). Ojo: en los dos casos "alto sin ingresos"
  la #2 tiene SIGNO CONTRARIO al destino (empuja hacia abajo), así que el cociente
  compara magnitudes, no contribuciones que se sumen. No se lee como riesgo.
- Dos casos tienen una variable NO interpretable en el top-3, y se dice explícitamente:
  `riesgo_mediano` con `num_medications` (local #3, global #15, -0,118) y
  `modelo_bajo_con_ingresos_no_reingresa` con `metformin` (local #3, global #16, -0,082).
- Pares contrastivos: la diferencia es de escala y concentración, no de qué variable
  aparece. En "alto sin ingresos" domina una sola contribución (destino, +0,936/+1,127)
  y `number_inpatient=0` empuja hacia abajo; en "bajo con ingresos" `number_inpatient=1`
  (+0,148/+0,165) queda casi cancelado por `destino=Casa` (-0,141/-0,173). Suma de
  |SHAP| del top-3: 1,16 y 1,37 frente a 0,49 y 0,42.
- LIMITACIÓN IMPORTANTE del par "alto sin ingresos", medida: su grupo de selección es
  el top 10% de val con `number_inpatient == 0` (n=440), y el **69,3% de ese grupo
  (305 de 440) tiene destino 22 o 5**, frente al 10,6% del resto del top 10%
  (164 de 1.547), el 23,6% del top 10% completo (469 de 1.987) y el 3,2% de todo val
  (632 de 19.868). Es decir: el grupo que debía enseñar "el modelo ve riesgo donde la
  regla `number_inpatient` no lo ve" está compuesto en dos tercios por los códigos que
  la auditoría señala como contaminados, y en los dos casos concretos esa variable es
  la contribución #1. Los dos casos NO son casualidad de la muestra: son típicos de su
  grupo. Conclusión para el informe: en ese grupo, la ventaja del modelo sobre la regla
  simple se apoya en buena parte en la variable contaminada, así que no sirve como
  demostración de que el modelo capte riesgo clínico que la regla no ve.
- ESTO TOCA LA FASE 2, no solo la figura. Ese grupo de 440 es exactamente donde el
  modelo discrepa de la regla simple `number_inpatient > 0`: son los que el modelo
  marca y la regla no. Buena parte de la ventaja del modelo sobre esa regla
  (PR-AUC en test 0,227 frente a 0,142, `reports/metrics_final.csv`) se apoya en
  pacientes de un grupo donde el 69,3% tiene destino 22/5, es decir, el destino que la
  Fase 3b señala como contaminado por la definición de la etiqueta. NO está medido qué
  fracción de la ventaja es artefacto, y no se afirma que lo sea toda: lo que está
  medido es la composición del grupo. Ver la predicción prerregistrada para la Fase 4,
  más abajo.
- Dentro de cada par, SHAP explica la PREDICCIÓN, no el RESULTADO: el modelo da
  probabilidades parecidas a los dos miembros del par (0,2617 y 0,2303; 0,0826 y
  0,0937), así que la explicación no puede distinguir quién reingresó. No es un fallo
  de SHAP, es lo que SHAP hace.
- Atribución frente a sensibilidad (criterio prerregistrado, cruce con el diagnóstico
  (b) de la Fase 3c): 0 "atribuye sin mover", 5 "mueve sin atribuir", 3 "coincide",
  3 "ni atribuye ni mueve". **Los 5 "mueve sin atribuir" son variables NO
  interpretables** (insulin x3, pioglitazone, glyburide), con rangos locales 6 a 16.
  Matiz de magnitud: 4 de los 5 tienen |delta| entre 0,003 y 0,005, que es
  despreciable; solo `pioglitazone` en `perfil_alto_2` mueve de forma apreciable
  (0,029) y lo hace HACIA ARRIBA. De las 11 filas de lever con sensibilidad medida,
  solo 3 son interpretables y las 3 son `discharge_disposition_id`.
- De ahí se sigue, pero conviene citarlo bien: "la única palanca accionable que pasa
  las sondas es el destino al alta" NO se deduce de esas 11 filas, sino del espacio de
  acciones de la Fase 3c (solo fármacos ya prescritos y destino) combinado con la regla
  GLOBAL de las sondas de ruido de la Fase 3b (ningún fármaco llega a 10/10; `insulin`
  se queda en 8/10). Las 11 filas lo ilustran en estos 6 casos, no lo demuestran.
- `delta_incluye_change` es True en 3 de las 11 filas con delta medido: parte del
  "efecto del fármaco" es en realidad el recálculo de la derivada `change`.
- ENTRADA PARA EL ANÁLISIS DE EQUIDAD DE LA FASE 4: `payer_code` (rango global #4, y
  supera las sondas 10/10) aparece en el top-3 interpretable de 2 de los 11 casos:
  `perfil_alto_2` (valor `Missing`, +0,101) y `modelo_alto_sin_ingresos_no_reingresa`
  (valor `MD`, -0,093). Se mantuvo a propósito en el modelo para analizarla como
  posible proxy de equidad; aquí aparece con peso local real, así que la Fase 4 debe
  examinarla.
- Estabilidad del top-3 local entre folds: `modelo_bajo_con_ingresos_reingresa` es el
  único con mínimo 1,00 (idéntico en los 5 folds), pero es el de grupo más pequeño
  (n=62), así que estable no es generalizable. Los otros diez tienen mínimo 0,67.
- HONESTIDAD DE PROCESO: la clasificación del cruce se corrigió DESPUÉS de ver la
  primera salida. El código implementaba 3 de las 4 categorías, y las variables con
  delta medido igual a 0 fuera del top-5 caían en la misma etiqueta que las que no son
  lever (sin medida, NaN). Se añadió "ni atribuye ni mueve" y se renombró la otra a
  "sin sensibilidad medida (no es lever)". Las DOS definiciones prerregistradas de
  discrepancia no se tocaron: sus condiciones son idénticas antes y después. El
  docstring de la función ya describía las cuatro categorías, pero el fichero no estaba
  versionado y no hay prueba independiente de que ese texto sea anterior a la primera
  ejecución, así que se registra como cambio posterior a ver la salida.

#### Predicción prerregistrada para la Fase 4 (fijada ANTES de corregir el target)
- Hipótesis: al corregir el target, la ventaja del modelo sobre la regla
  `number_inpatient > 0` debe caer MÁS en el grupo `number_inpatient == 0` del top 10%
  que en el resto del top 10%.
- Métrica exacta con la que se comprobará, toda en val (test está gastado):
  1. Se toma el top 10% de val del modelo, antes y después de la corrección (cada uno
     con su propio corte, recalculado sobre sus propias puntuaciones).
  2. Se parte ese top 10% en dos subgrupos: A = `number_inpatient == 0` (los que la
     regla NO marca) y B = `number_inpatient >= 1` (los que la regla SÍ marca).
  3. En cada subgrupo se mide la PRECISIÓN: % de reingresos <30 d reales entre los
     marcados. Se usa precisión y no PR-AUC porque dentro de cada subgrupo la regla es
     constante y su PR-AUC sería degenerado.
  4. Se calcula la caída: delta_A = precision_A(antes) - precision_A(después), y
     delta_B igual.
- Criterio: la predicción se considera CONFIRMADA si delta_A > delta_B, y REFUTADA si
  delta_A <= delta_B. Se reporta el resultado salga como salga.
- Se informará además de n en cada subgrupo antes y después, porque la composición del
  top 10% puede moverse y una caída de precisión con un n muy distinto no es
  comparable sin decirlo.
- Las cifras "antes" se calculan al empezar la Fase 4, con los artefactos actuales y
  ANTES de aplicar la corrección. No se han medido todavía.
- ASIMETRÍA DE LA PRUEBA, fijada antes de medir nada (viene de la limitación de
  visibilidad de la Fase 3b): el reetiquetado solo alcanza a los casos con encuentro
  posterior visible, así que corrige MENOS en los códigos 22 y 5 (36,0% y 37,8% de sus
  positivos quedan sin ver), y el grupo A es un 69,3% de esos dos códigos. En
  consecuencia `delta_A` saldrá ATENUADO respecto a lo que daría una corrección ideal.
  Por tanto los dos resultados posibles NO son simétricos:
  - REFUTADO (`delta_A <= delta_B`) admite dos lecturas que estos datos no separan: que
    la hipótesis sea falsa, o que la corrección no haya llegado a la parte contaminada
    que no vemos. Se reportará como no concluyente entre esas dos, sin elegir una.
  - CONFIRMADO (`delta_A > delta_B`) no tiene esa ambigüedad: se obtiene a pesar de la
    atenuación, no gracias a ella.

### Fase 4 — pendiente. Entra aquí:
1. Medir las cifras "antes" de la predicción prerregistrada, antes de tocar nada.
2. Decidir y aplicar la corrección del target (opciones en la Fase 3b).
3. Medir la predicción y reportar el resultado salga como salga.
4. Consultar test UNA sola vez con el modelo corregido, en las condiciones fijadas
   más abajo (decisión ya tomada: ver "Decisiones y prerregistros de la Fase 4").
5. Análisis de equidad: `payer_code` (ver la entrada de la Fase 3d) y los posibles
   proxies de `race`/`gender` vía otras variables, que las sondas no permiten
   descartar.
6. Informe (con la reflexión crítica) y `README.md` con instrucciones de
   reproducción. Dar acceso al repo privado a quien corrija.

### Decisiones y prerregistros de la Fase 4 (fijados ANTES de medir nada)

#### Split congelado
- El split se calcula con el target ORIGINAL y el reetiquetado se aplica DESPUÉS, sobre
  filas ya asignadas. Motivo: `D.grouped_split` usa `StratifiedGroupKFold`, que
  estratifica por `y`; reetiquetar antes cambiaría los folds y val dejaría de ser el
  mismo conjunto de filas, con lo que `precision_A(antes)` y `precision_A(después)` no
  serían comparables y la predicción prerregistrada perdería sentido.
- LIMITACIÓN que se declara en la memoria y NO se corrige: ese split queda estratificado
  sobre la etiqueta contaminada, así que el modelo corregido se entrena en una partición
  optimizada para la etiqueta vieja. Corregirlo rompería la comparabilidad antes/después,
  que es justo lo que la Fase 4 mide.

#### Uso de test
- Decisión de Jaime: SÍ se consulta test, UNA sola vez, como ÚLTIMA cosa que se hace,
  con hiperparámetros, split y corrección ya cerrados. Razones: el enunciado pide
  métricas de test, el modelo corregido es el entregable, y quedarse sin estimación en
  datos no vistos es un hueco más visible que el riesgo de inflación — riesgo pequeño
  porque NINGUNA decisión sobre el modelo corregido sale de test.
- Nada cambia después de verla. Si el número sale mal, se reporta igual.
- Se declara en la memoria: es la SEGUNDA vez que se toca test, para qué, y que NO es
  comparable con el 0,227 porque el target cambió. La cifra nueva responde a otra
  pregunta: cómo predice el TARGET CORREGIDO en datos no vistos.
- Hasta esa consulta final, de test SOLO se aplica la regla de reetiquetado. No se
  calcula ninguna estadística de test ni entra en ninguna decisión; lo único que se
  reporta antes es el CONTEO de filas cambiadas por split, que es descripción del
  dataset. Queda explícito en el código.

#### Descomposición de la predicción prerregistrada: etiqueta frente a modelo
El criterio original (CONFIRMADA si `delta_A > delta_B`, con `delta = precision(antes) -
precision(después)`) NO se toca. Lo que se añade, antes de medir nada del "después", es
la descomposición que hace falta para interpretarlo, porque la precisión del "después"
se mide contra la etiqueta NUEVA y eso mezcla dos efectos: un acierto puede dejar de
contar porque su etiqueta pasó a 0, sin que el modelo haya cambiado. Como el subgrupo A
es un 69,3% de códigos 22/5, justo donde se concentra el reetiquetado, `delta_A` saldría
grande casi por construcción y un CONFIRMADO diría menos de lo que parece.
- Se miden TRES puntos, no dos (el (2) no requiere entrenar nada):
  1. **modelo viejo / etiqueta vieja** — ya medido: A 32,27% (440), B 27,28% (1.547).
  2. **modelo viejo / etiqueta nueva** — mismo ranking y mismo corte; solo cambia la
     verdad contra la que se puntúa. Aísla el efecto PURO de la etiqueta.
  3. **modelo nuevo / etiqueta nueva** — efecto de la etiqueta MÁS el reentrenamiento.
- Definiciones, por subgrupo: `delta_etiqueta = (1) - (2)` y `delta_modelo = (2) - (3)`.
  Por construcción `delta_etiqueta + delta_modelo = (1) - (3) = delta`, el del criterio
  original, así que la descomposición es exacta y se puede comprobar sumando.
- En el paso (2) el corte se recalcula sobre las puntuaciones del modelo viejo, que son
  las mismas, así que el `n` de cada subgrupo NO cambia entre (1) y (2): solo cambian
  los aciertos. Se confirma al medir.
- La descomposición se reporta SIEMPRE junto al veredicto, nunca el veredicto solo.
- Lecturas fijadas AHORA, no al ver los números:
  - Si `delta_etiqueta_A > delta_etiqueta_B` pero `delta_modelo_A ≈ delta_modelo_B`, la
    lectura es que la ventaja en A era ETIQUETA y no señal.
  - Si la diferencia aparece en `delta_modelo`, el reentrenamiento también cambió el
    ORDEN, no solo la verdad contra la que se mide.

#### La BRECHA A - B, que es lo que de verdad contrasta la hipótesis
- `delta_etiqueta` es una IDENTIDAD ARITMÉTICA, no un contraste: equivale a (aciertos
  reetiquetados del subgrupo) / (n del subgrupo). Es descriptivo. Y como A es un 69,3%
  de códigos 22/5, que A pierda más es casi tautológico. Que nadie lo lea como
  evidencia por sí solo.
- La pregunta original era si la ventaja del modelo en A se sostenía sobre etiquetas
  contaminadas, y eso se responde mirando si A sigue por delante de B con la etiqueta
  limpia. Por eso se reporta, en los TRES pasos:
  `brecha(paso) = precision_A(paso) - precision_B(paso)`.
  En el paso (1) ya vale **+4,99 puntos** (32,27 - 27,28).
- Lectura, fijada AHORA:
  - Si la brecha se acerca a cero o se INVIERTE en el paso (2), la ventaja de A sobre B
    se apoyaba en la etiqueta contaminada.
  - Si se MANTIENE en el paso (2), no se apoyaba, pase lo que pase con los deltas.
- Esto NO sustituye al criterio prerregistrado (`delta_A > delta_B` con
  `delta = (1) - (3)`), que se mantiene intacto y se reporta igual. Es una cantidad
  ADICIONAL. Si las dos lecturas apuntan en sentidos distintos, se dice explícitamente y
  no se elige la que convenga.
- La brecha se reporta de DOS formas en los tres pasos: la diferencia en puntos
  (`precision_A - precision_B`) y el COCIENTE (`precision_A / precision_B`). Motivo:
  limpiar la etiqueta baja las dos precisiones a la vez, así que la diferencia en puntos
  puede encogerse sin que la ventaja RELATIVA cambie. En el paso (1) la diferencia es
  +4,99 puntos y el cociente 1,183 (32,27 / 27,28).
- Lectura del cociente, fijada AHORA: la ventaja de A se considera apoyada en la etiqueta
  si el cociente se acerca a 1 o baja por debajo en el paso (2), no solo si se reduce la
  diferencia en puntos.
- INTERVALOS DE CONFIANZA de la brecha, en los tres pasos: bootstrap sobre los PACIENTES
  de val (agrupado por `patient_nbr`, igual que el split; no sobre filas, que trataría
  como independientes los encuentros del mismo paciente), 2.000 réplicas, semilla
  `D.SEED`. Se da el IC del 95% (percentiles 2,5 y 97,5) de la diferencia en puntos y
  del cociente, con el `n` de cada subgrupo al lado. El corte del top 10% se recalcula
  DENTRO de cada réplica, para que el intervalo refleje el procedimiento completo y no
  un corte fijado con la muestra entera.
  Es el mismo estándar que ya se exige en el análisis de equidad: sería incoherente
  pedir IC bootstrap allí y no aquí, donde está la cifra titular.
- Lectura de los IC, fijada ANTES de verlos: el cambio de signo de la brecha en el paso
  (2) se considera ESTABLECIDO si su IC excluye el cero; si lo incluye, se reporta como
  COMPATIBLE CON CERO y se dice, sin cambiar el veredicto prerregistrado. Lo mismo para
  el paso (3), donde A baja a 212 marcados y se espera un IC bastante ancho.
- CONTRASTE EMPAREJADO, que es la prueba del titular. Comparar los IC de los pasos (1) y
  (2) por separado NO es el contraste adecuado: son los MISMOS pacientes con el MISMO
  ranking y el MISMO corte, y lo único que cambia es la etiqueta. Por tanto, DENTRO de
  cada réplica bootstrap (las mismas 2.000, agrupadas por `patient_nbr`, misma semilla)
  se calcula:
  - `brecha(2) - brecha(1)` en puntos,
  - el cociente de cocientes `cociente(2) / cociente(1)`,
  - `delta_A - delta_B` en puntos,
  y se da el IC95 de cada uno. Al emparejar se cancela la variación entre pacientes, así
  que se esperan intervalos bastante más estrechos que los de cada paso por separado.
- REGLA DE LECTURA, fijada ahora: el cambio de la brecha se considera ESTABLECIDO si el
  IC de `brecha(2) - brecha(1)` excluye el cero. Esa es la prueba del titular; los IC de
  cada paso por separado se siguen reportando, pero como DESCRIPTIVOS.
- El paso (3) NO se empareja con el (1): el ranking cambia y los subgrupos tienen distinto
  `n`. Se deja como está, compatible con cero y sin usarlo para concluir.

#### Corte del top 10%, en las tres medidas
- Se usa `probs.quantile(0.9)` sobre las puntuaciones de val, el MISMO criterio que
  `xai_cases.select_cases` en la Fase 3b, y el "después" usa esa misma función. NO se
  usa `evaluation.top_k_table`, que marca exactamente `ceil(k*n)` filas con desempate
  aleatorio y podría dar un conjunto ligeramente distinto.
- Motivo: así las cifras reconcilian con los artefactos ya publicados. Con el modelo
  actual ambas coinciden (corte 0,1954, 1.987 marcados de 19.868, A n=440 con 69,3% de
  destinos 22/5, y 564 aciertos = la `precision` del 10% en
  `xai_sanity_topk_composition.csv`).

#### PR-AUC no es comparable entre los dos targets: la cifra titular es el lift
- El reetiquetado BAJA la prevalencia, y la línea base del PR-AUC es justamente la
  prevalencia: un PR-AUC más bajo con menos positivos puede ser el mismo rendimiento, o
  mejor. Comparar los PR-AUC crudos de los dos targets es comparar contra dos líneas
  base distintas.
- Prerregistro: se reportan SIEMPRE juntos (a) PR-AUC, (b) la prevalencia del conjunto y
  (c) el `lift = PR-AUC / prevalencia`. La cifra TITULAR del antes/después es el LIFT, no
  el PR-AUC crudo. Si el PR-AUC y el lift apuntan en sentidos distintos, se dice y no se
  elige el que convenga.
- Esto vale igual para la CV agrupada de 5 folds: por fold se reportan PR-AUC,
  prevalencia y lift, y la comparación antes/después se hace sobre el lift.
- Referencia de partida (val, etiqueta vieja): PR-AUC 0,2308 con prevalencia 11,39%,
  lift 2,026 (`xai_sanity_sensitivity.csv`, fila "todo val").

#### Comprobación de que la corrección hace lo que decimos
- Prerregistro: se espera que el SHAP medio (|SHAP|) de `discharge_disposition_id` BAJE
  con el target corregido, porque la etiqueta contaminada es lo que lo sostenía (val,
  0,204 antes; `xai_global_comparison.csv`). Se reporta lo que salga, incluida una
  subida, que sería evidencia contra nuestra lectura del hallazgo.

#### Prerregistro del análisis de equidad (punto 5)
Fijado antes de mirar ningún resultado por grupo. Todo en val.
- **Grupos:** `race`, `gender`, `payer_code`. `race = Missing` SE TRATA COMO GRUPO
  PROPIO, no se excluye: coherente con la decisión de limpieza (missing como
  categoría), y no registrar la raza no es un fallo aleatorio.
- **Tamaño mínimo para INTERPRETAR:** n ≥ 100 en val y ≥ 20 marcados en el top 10%. Los
  grupos por debajo se REPORTAN con su n y se marcan como no interpretables; no se
  comentan ni entran en conclusiones.
- **Grupo de referencia:** el más numeroso de cada variable en val, con UNA excepción
  decidida antes de medir:
  - `race` -> Caucasian (14.872) y `gender` -> Female (10.663): son las mayores y no hay
    nada que decidir.
  - `payer_code` -> **MC** (Medicare, 6.201), NO `Missing` (7.933), que es la mayor. Un
    cociente relativo a "seguro desconocido" no es interpretable: no es un grupo de
    pacientes con una característica común, sino la ausencia de un dato. MC es la mayor
    categoría con seguro CONOCIDO. `Missing` sigue recibiendo su propio veredicto como
    grupo no-referencia, porque no medir el seguro tampoco es aleatorio.
- **Composición del top-k (5/10/20%):** métrica = tasa de marcado del grupo
  (marcados_grupo / n_grupo) y su ratio frente al grupo de referencia. Umbral: hay
  disparidad si el ratio sale de [0,80, 1,25] (regla de los cuatro quintos, EEOC 1978).
  - El cociente va acompañado de su **IC95 bootstrap** con el mismo esquema que el resto
    del trabajo (2.000 réplicas agrupadas por `patient_nbr`, semilla `D.SEED`). El
    VEREDICTO NO CAMBIA: sigue siendo la regla de los cuatro quintos sobre la estimación
    puntual. El IC se reporta al lado para que un cociente de 0,75 sobre n = 150 no se
    lea igual que sobre n = 3.000.
  - Esta familia NO lleva corrección por comparaciones múltiples, porque su veredicto es
    por UMBRAL y no por significación; el IC es descriptivo.
  - Motivo de añadirlo: sería incoherente exigir IC emparejado en el titular del punto 3
    y no tener ninguna medida de incertidumbre en el análisis PRIMARIO de equidad.
- **Error por grupo al mismo k:** métrica = FNR (positivos reales NO marcados /
  positivos reales del grupo), que es a quien se le niega el seguimiento. Umbral: hay
  disparidad si |FNR_grupo − FNR_ref| ≥ 5 puntos porcentuales Y el IC bootstrap al 95%
  (1.000 remuestreos, semilla 42) de esa diferencia excluye 0.
- **Detectabilidad de proxies:** métrica = ROC-AUC al predecir el atributo protegido
  desde el RESTO de features. Umbral: recuperable si ≥ 0,65; no recuperable si < 0,55;
  entre ambos, zona gris que se reporta sin concluir. Especificación completa, fijada
  ahora para que no se elija después la variante que más convenga:
  - Modelo: LightGBM con la misma configuración del modelo principal
    (`num_leaves=31`, `min_child_samples=20`, `learning_rate=0.03`, `SEED=42`), ajustado
    en train con early stopping sobre val, igual que el modelo principal.
  - Features: todas las de `P.FEATURES` menos el atributo que se predice. Los demás
    atributos protegidos SE QUEDAN: la pregunta es si el atributo es recuperable del
    resto del formulario, no si lo es de un subconjunto elegido por nosotros.
  - `gender` es binaria: ROC-AUC directa en val.
  - `race` es multiclase: ROC-AUC **one-vs-rest macro** en val
    (`roc_auc_score(..., multi_class="ovr", average="macro")`), y además el AUC por
    clase, porque la macro da el mismo peso a una clase grande y a una pequeña. Se
    interpretan solo las clases que cumplen el mínimo de tamaño (n ≥ 100); las demás se
    reportan sin interpretar, y se dice qué clases quedaron fuera de la macro.
- **Redacción obligatoria del resultado de proxies:** que `race`/`gender` sean
  predecibles desde el resto de variables NO demuestra que el modelo las use; demuestra
  que QUITARLAS NO SERÍA SUFICIENTE. No se escribirá "el modelo usa race indirectamente".

#### Prerregistro del punto 5, segunda parte: alcance, múltiples comparaciones y efecto distributivo
Fijado ANTES de calcular cualquier cifra de equidad. Lo único que se consultó para
escribir esto son los TAMAÑOS de grupo en val (denominadores, no resultados): `race` 6
categorías (Caucasian 14.872, AfricanAmerican 3.751, Hispanic 421, Missing 400, Other
288, Asian 136), `gender` 2 (Female 10.663, Male 9.205) y `payer_code` 10 (Missing
7.933, MC 6.201, HM 1.167, SP 1.075, BC 945, MD 661, Other 547, CP 497, UN 475, CM 367).
Las 18 categorías pasan el mínimo de n ≥ 100.

- **Modelo:** el análisis PRIMARIO es el MODELO CORREGIDO, que es el entregable. El
  modelo viejo entra solo como COLUMNA DESCRIPTIVA de comparación, SIN veredictos propios
  de disparidad, para no duplicar los contrastes con veredicto. Si la comparación sugiere
  algo, se dice como observación exploratoria.
- **Etiqueta de verdad:** la CORREGIDA, en los tres análisis. Todo en VAL (el análisis
  distributivo del reetiquetado es la excepción: train+val, ver más abajo). Test NO se
  toca en este punto.
- **k primario = top 10%**, el mismo que se usa en toda la Fase 3 y 4. El 5% y el 20% se
  reportan como SENSIBILIDAD, descriptivos y sin veredicto.
- **Atributo primario de cada análisis** (los demás son secundarios):
  - Composición del top-k -> `payer_code`. Por qué: es el único de los tres que supera
    las sondas de ruido 10/10 y aparece en el top-3 local de 2 de los 11 casos, así que
    es donde hay señal que el modelo use de forma distinguible del ruido.
  - Error por grupo (FNR) -> `race`. Por qué: es el atributo con carga ética directa, y
    el FNR es a quién se le niega el seguimiento.
  - Detectabilidad de proxies -> `race` (`gender` secundario), por el mismo motivo.
  - Efecto distributivo del reetiquetado -> `race`.
- **Secundarios:** si un atributo secundario sale "significativo", se reporta como
  EXPLORATORIO y NO se presenta como hallazgo. No se cambia el atributo primario después
  de ver los resultados.
- **Número de contrastes CON VEREDICTO, contados ahora: 20.**
  - 9 de composición (`payer_code`, 10 grupos menos la referencia), por regla de umbral
    sobre la estimación puntual (cuatro quintos), no por significación.
  - 5 de FNR (`race`, 6 grupos menos la referencia), basados en IC.
  - 1 de proxies (`race`, AUC OVR macro frente al umbral 0,65), por umbral.
  - 5 de efecto distributivo (`race`, 6 grupos menos la referencia), basados en IC.
  - Si un grupo no llega al mínimo (n ≥ 100 y ≥ 20 marcados en el k primario), su
    contraste NO se hace: se reporta sin interpretar y el recuento final se declara.
- **Corrección por comparaciones múltiples:** se aplica **HOLM** con α = 0,05 DENTRO de
  cada familia basada en IC (la de FNR, m = 5, y la del efecto distributivo, m = 5), con
  p-valores bootstrap de dos colas (proporción de réplicas al otro lado del nulo,
  duplicada, con suelo 1/n_rep). Elegido Holm y no Benjamini-Hochberg porque cada
  contraste se enunciaría como una afirmación sobre un grupo protegido concreto: interesa
  controlar la probabilidad de equivocarse en ALGUNA (FWER), no la proporción de falsos
  descubrimientos en un cribado. Las dos familias por umbral (composición y proxies) no
  llevan corrección porque no son pruebas de significación; se dice explícitamente.
- **La `m` de cada familia de Holm es el número de grupos que PASAN el mínimo de ≥ 20
  marcados**, determinado por esta regla fijada de antemano y NO ajustado tras ver los
  p-valores. La `m` PLANIFICADA es 5 por familia (6 grupos de `race` menos la
  referencia); puede BAJAR si algún grupo no alcanza el mínimo. El valor FINAL de `m` se
  reporta junto a los resultados, con el recuento de contrastes efectivamente realizados
  y la lista de grupos excluidos con su `n` de marcados.
- El filtro de ≥ 20 marcados se evalúa EN EL MOMENTO DE MEDIR, no antes: `marcados_grupo`
  es el NUMERADOR de la tasa de selección, así que consultarlo para fijar el alcance
  habría sido ver el resultado primario del análisis de composición antes de cerrar su
  prerregistro. Por eso se deja como regla y el script declara qué grupos cayeron.

#### Efecto distributivo de la PROPIA corrección (análisis nuevo, el más relevante en ética)
- Pregunta: ¿el reetiquetado quitó positivos de forma DESIGUAL entre grupos? Métrica =
  tasa de reetiquetado del grupo = (filas cambiadas de 1 a 0) / (positivos del grupo), en
  **train+val**, con `n` y denominador explícitos, para `race`, `gender` y `payer_code`.
- Mínimo para interpretar: n ≥ 100 filas y ≥ 20 POSITIVOS en train+val (el análogo del
  mínimo de marcados, porque el denominador aquí son los positivos). Los que no lleguen
  se reportan sin interpretar.
- Criterio de desigualdad: igual que el de FNR. Diferencia de la tasa de reetiquetado
  frente al grupo de referencia (el mayor) ≥ 5 puntos porcentuales Y el IC bootstrap al
  95% de la diferencia excluye 0, con bootstrap agrupado por `patient_nbr`, 2.000
  réplicas, semilla 42, y Holm dentro de la familia.
- **Lectura fijada AHORA:** si el reetiquetado es desigual entre grupos, entonces
  CUALQUIER comparación de equidad antes/después queda CONFUNDIDA por ese cambio de
  denominador, y hay que decirlo en vez de presentar el antes/después como si midiera un
  cambio del modelo. Y en ese caso la corrección metodológica tiene consecuencias
  distributivas, que van al informe como HALLAZGO PROPIO, no como nota al pie.
- **Nota de comparabilidad para el informe:** este análisis es train+val y todos los
  demás del punto 5 son solo val, así que sus cifras NO se comparan directamente entre
  sí. Se dice al presentarlas.

### Fase 4 — resultados de la corrección del target (`src/target_correction.py`, `src/phase4_correction.py`)

#### El titular: la brecha A - B se INVIERTE con el mismo modelo y el mismo ranking
Brecha = `precision_A - precision_B` en el top 10% de val (A = `number_inpatient == 0`,
B = `>= 1`), con IC95 por bootstrap de PACIENTES (2.000 réplicas, semilla 42, 0
descartadas). Fuente: `reports/phase4_prediction_gaps.csv`.

| paso | brecha | IC95 | cociente | IC95 | n_A | n_B |
|---|---|---|---|---|---|---|
| (1) viejo/vieja | +4,99 | [+0,07; +9,99] | 1,183 | [1,002; 1,389] | 440 | 1.547 |
| (2) viejo/nueva | -5,51 | [-9,80; -1,04] | 0,770 | [0,613; 0,954] | 440 | 1.547 |
| (3) nuevo/nueva | -3,06 | [-9,31; +2,44] | 0,866 | [0,609; 1,109] | 212 | 1.775 |

- El paso (2) usa EL MISMO modelo, el MISMO ranking y el MISMO corte que el (1): lo único
  que cambia es la etiqueta contra la que se puntúa. La ventaja de A sobre B se invierte.
  Esta es la evidencia SUSTANTIVA del trabajo: donde el modelo discrepaba de la regla
  simple, su ventaja estaba sostenida por etiquetas contaminadas.
- **La prueba del titular es el CONTRASTE EMPAREJADO**, no los IC de cada paso por
  separado (que son los mismos pacientes y el mismo ranking, así que comparar sus IC
  ignora el emparejamiento). Calculado dentro de cada réplica
  (`reports/phase4_paired_contrasts.csv`):

| contraste | observado | IC95 | nulo | excluye el nulo |
|---|---|---|---|---|
| brecha(2) - brecha(1) | -10,50 puntos | [-13,94; -7,13] | 0 | sí |
| cociente(2) / cociente(1) | 0,651 | [0,558; 0,747] | 1 | sí |
| delta_A - delta_B | +8,06 puntos | [+2,78; +13,75] | 0 | sí |

  Al emparejar, el intervalo de la brecha pasa de ~10 puntos de ancho en cada paso por
  separado a ~6,8 en la diferencia, porque se cancela la variación entre pacientes. Por
  la regla prerregistrada, el cambio de la brecha queda ESTABLECIDO. El tercer contraste
  también excluye el cero, así que el veredicto prerregistrado no depende del ruido de
  muestreo, aunque siga impulsado por `delta_etiqueta`.
- El paso (3) tiene IC que INCLUYE el cero ([-9,31; +2,44]) y cociente que incluye el 1:
  se reporta como compatible con cero y no se usa para concluir. A baja a 212 marcados,
  así que el intervalo es ancho por construcción. NO se empareja con el (1), porque su
  ranking y sus `n` son otros.
- MATIZ QUE HAY QUE DECIR NOSOTROS, no esperar a que lo pregunten: el IC del paso (1) es
  [+0,07; +9,99], o sea que ROZA EL CERO. La ventaja original de A sobre B ya era
  marginal de por sí. Así que la frase correcta NO es "una ventaja sólida resultó ser un
  artefacto", sino "una ventaja ya débil se INVIERTE al limpiar la etiqueta". Encaja con
  lo que ya sabíamos de la Fase 2: la ventaja sobre la regla simple depende de k.

#### El veredicto formal, y por qué NO es el titular
- Criterio prerregistrado: `delta_A` 12,46 > `delta_B` 4,41 -> **CONFIRMADA**
  (`reports/phase4_prediction_deltas.csv`).
- Pero la descomposición dice de dónde viene: en A, `delta_etiqueta` = 13,86 y
  `delta_modelo` = -1,40 (en B, 3,36 y 1,04). Es decir, el veredicto lo impulsa
  CASI ENTERO `delta_etiqueta`, que es la identidad aritmética ya anotada en el
  prerregistro (aciertos reetiquetados / n del subgrupo) y no un contraste. Suma
  comprobada con `assert`: `delta_etiqueta + delta_modelo = delta_total`.
- Por eso el titular es el cambio de signo de la brecha y no el "CONFIRMADA".

#### Los dos hallazgos van JUNTOS y en este orden (no presentar uno sin el otro)
1. **La contaminación NO estaba inflando el rendimiento global.** En CV agrupada de 5
   folds dentro de train, con `n_estimators` fijo a 122 en ambas para que lo único que
   cambie sea la etiqueta: lift medio 1,9914 ± 0,0200 con la etiqueta vieja frente a
   1,9878 ± 0,0575 con la nueva, y el lift con etiqueta nueva GANA en 2 de los 5 folds.
   El modelo predice el target limpio casi igual de bien
   (`reports/phase4_cv_lift.csv`).
2. **Lo que la contaminación fabricaba era la ventaja CONCRETA en el grupo donde el
   modelo discrepa de la regla simple**, que es el punto 1 de esta sección.

Nota de comparabilidad: PR-AUC crudo no se compara entre targets (su línea base es la
prevalencia). En val, (1) PR-AUC 0,2308 con prevalencia 11,39% y lift 2,026; (2) 0,1861
con 9,78% y lift 1,903; (3) 0,1893 con 9,78% y lift 1,936 (`phase4_val_metrics.csv`).

#### Confirmaciones mecánicas de que la corrección hizo lo que decimos
- **El modelo nuevo deja de marcar a esa población:** A cae de 440 a 212 marcados y su
  porcentaje de destinos 22/5 pasa del 69,32% al 34,43% (`phase4_prediction_steps.csv`).
- **El SHAP medio del destino bajó, pero solo un 8%** (0,2043 -> 0,1871, val;
  `phase4_shap_discharge.csv`). Se esperaba que bajara y bajó. El matiz prerregistrado:
  una caída pequeña es COHERENTE con que el reetiquetado solo alcanza los casos con
  encuentro posterior visible, y en los códigos 22 y 5 queda fuera el 36,0% y el 37,8%
  de sus positivos.
- **El reetiquetado alcanzó el 43,67% de los positivos del código 22** (193 de 442;
  `phase4_relabel_by_destination.csv`), que es exactamente la distinción fijada antes de
  medir: 68,2% es la tasa sobre los VISIBLES y 43,67% la tasa sobre el TOTAL de sus
  positivos. Código 5: 34,33% (69 de 201). Código 1: 12,56% (563 de 4.481).

#### Mecánica del reetiquetado
- Filas que pasan de 1 a 0, con denominador (`phase4_target_correction.csv`):
  train 971 de 6.788 positivos (14,30%), val 320 de 2.263 (14,14%), test 307 de 2.263
  (13,57%). Prevalencia: 11,39% -> 9,76% en train, 11,39% -> 9,78% en val.
- Train+val suman **1.291 filas**, que reproducen `audit_discharge_shares.csv` fila a
  fila (código 1: 563 = 43,61% del total cambiado; 22: 193 = 14,95%; 3: 182 = 14,10%).
- De test SOLO se aplicó la regla y se reporta el conteo. Ninguna métrica de test se
  calculó (`phase4_correction.py` no puntúa test).
- **El encuentro siguiente nunca cae en otro split**: 0 casos en train, val y test, con
  `assert` que falla si no (`phase4_split_check.csv`). Era lo esperado por el split
  agrupado por `patient_nbr`, pero está medido, no supuesto.

#### Guardarraíl del modelo corregido
- Modelo nuevo: **183 árboles, PR-AUC en val 0,1893** (mismos hiperparámetros,
  `num_leaves=31`, `min_child_samples=20`, `learning_rate=0.03`, early stopping sobre val
  con la etiqueta nueva). Este es el guardarraíl que debe reproducirse de aquí en
  adelante para el modelo corregido.
- El guardarraíl viejo de `evaluate_test.py` (122 árboles, PR-AUC val 0,2308) NO SE TOCA:
  es el registro de lo que se hizo en la Fase 2 y sigue valiendo para el modelo original.
  `phase4_correction.py` comprueba los DOS: el viejo antes de medir el paso (1) y el
  nuevo al entrenar.

### Fase 4, punto 5 — Resultados de equidad (`src/phase4_fairness.py`)
Prerregistro commiteado ANTES de medir (`995928b`). Guardarraíles comprobados: modelo
corregido 183 árboles / PR-AUC val 0,1893; modelo viejo 122 / 0,2308. Test no se tocó.

#### HALLAZGO CENTRAL: validez y equidad tiraron en sentidos opuestos
Lo principal NO son las disparidades del modelo corregido en absoluto, sino el
DESPLAZAMIENTO de la composición del top 10% entre el modelo viejo y el corregido
(`reports/phase4_fairness_composition.csv`, cociente de tasa de marcado frente a MC):

| payer_code | cociente modelo viejo | cociente modelo corregido |
|---|---|---|
| SP | 0,832 | **0,385** |
| UN | 0,869 | **0,494** |
| BC | 0,861 | 0,775 |
| Missing | 1,452 | **2,073** |
| HM | 0,805 | **1,349** |

Con el modelo viejo los diez grupos estaban dentro o cerca de [0,80; 1,25] (el más
extremo era `Missing`, 1,452). Con el corregido, `SP` y `UN` caen por debajo de 0,50 y
`Missing` sube a 2,073. **La corrección del target, metodológicamente correcta y
defendida con un contraste emparejado cuyo IC excluye el cero, empeoró de forma
apreciable quién recibe seguimiento post-alta.** Es un hallazgo del trabajo, no un efecto
colateral, y es lo que responde más directamente a la pregunta ética de la asignatura.

- **CAUTELA que no se salta:** el modelo viejo se prerregistró como COLUMNA DESCRIPTIVA
  SIN veredictos, así que no se le asigna ninguno y el desplazamiento se reporta como
  OBSERVACIÓN DESCRIPTIVA. Un veredicto formal sobre el desplazamiento exigiría su propio
  prerregistro, que NO se hizo. Decirlo no le quita fuerza al hallazgo: se la da, porque
  la alternativa sería fijar el criterio después de ver el resultado.
- **Explicación mecánica (análisis 4b, secundario/exploratorio, sin veredicto ni Holm):**
  el reetiquetado golpeó muy distinto por seguro. Tasa de reetiquetado en train+val
  (`phase4_fairness_relabel_distribution.csv`), con el grupo de REFERENCIA incluido
  porque sin él la explicación está incompleta —los cocientes son relativos a MC, así que
  un grupo puede subir solo porque MC bajó—: **MC (referencia) 15,27%** (3.026 positivos),
  SP 28,29% (+13,03, IC [+8,26; +17,87]), UN 23,98% (+8,71), BC 23,49% (+8,23), CP 20,47%,
  MD 19,69%, `Other` 14,35%, CM 12,16%, `Missing` 11,29% (-3,98) y HM 4,86%
  (-10,41, IC [-12,68; -8,01]).
- **Cuantificado, no "encaja casi uno a uno":** la correlación de rangos de Spearman entre
  la tasa de reetiquetado del grupo y el factor de cambio de su tasa de marcado
  (tasa_nueva / tasa_vieja, en términos ABSOLUTOS y no en cociente frente a MC, para que
  el movimiento de MC no contamine el de los demás) es **rho = -0,952** (p = 2,3e-05,
  n = 10 grupos de `payer_code`; `src/phase4_relabel_vs_selection.py`,
  `reports/phase4_relabel_vs_selection*.csv`). Factores: SP 0,385, UN 0,472, CP 0,692,
  MD 0,712, BC 0,747, `Other` 0,796, **MC 0,830**, CM 0,923, `Missing` 1,185, HM 1,390.
  Es decir: a más positivos reetiquetados, más baja la tasa de marcado, y el orden se
  sostiene casi perfecto. Spearman y no Pearson porque solo se afirma el ORDEN.

#### Composición del top 10% (primario `payer_code`, referencia MC): 4 veredictos, 2 que se sostienen
- 7 contrastes realizados de 9 planificados. Excluidos por el mínimo de >= 20 marcados:
  CP (18) y UN (17). El filtro se evaluó al medir, como fija el prerregistro.
- 4 fallan la regla de los cuatro quintos: `Missing` 2,073, `SP` 0,385, `BC` 0,775 y
  `HM` 1,349. **De esos, solo 2 tienen IC que excluya la paridad:** `Missing`
  [1,754; 2,463] y `SP` [0,239; 0,569]. Los de `BC` [0,410; 1,215] y `HM`
  [0,973; 1,839] la INCLUYEN.
- Redacción correcta: veredicto por el criterio prerregistrado en los cuatro, y de esos,
  dos indistinguibles de la paridad al mirar la incertidumbre. **NO se cuentan cuatro
  hallazgos.**
- **Los dos grupos excluidos por el mínimo también están por debajo de 0,80:** CP 0,500
  y UN 0,494. Se reportan sin interpretar, como manda la regla, pero conviene decirlo
  nosotros: sin esa frase, excluir grupos pequeños parece un filtro que se quedó con los
  casos favorables, y es justo lo contrario — dejó fuera DISPARIDAD POTENCIAL, no
  paridad.

#### FNR por `race` al top 10% (primario, m de Holm = 4): ningún veredicto
- Referencia Caucasian, FNR 77,59%. Diferencias: `Missing` +10,98 (p 0,080),
  `Hispanic` -10,20 (p 0,243), `AfricanAmerican` -3,34 (p 0,352), `Other` +0,67
  (p 0,901). Ninguno cumple la conjunción (>= 5 puntos Y Holm). `Asian` excluido
  (15 positivos).
- La conclusión NO es que no haya disparidad de FNR, y se apoya en el intervalo y no en
  una máxima: con 15 a 46 positivos, los IC de los grupos pequeños admiten diferencias de
  hasta unos 25 puntos (`Hispanic` [-25,80; +6,60], `Other` [-19,47; +22,49], `Asian`
  [-16,38; +23,46]). La prueba NO PODÍA detectar disparidades de tamaño relevante en esos
  grupos, así que no se concluye nada sobre ellos.

#### Efecto distributivo del reetiquetado por `race` (primario, m de Holm = 5)
- Referencia Caucasian 14,83%. Ningún veredicto de disparidad.
- **El caso `AfricanAmerican` es el ejemplo didáctico del trabajo:** es DETECTABLE
  estadísticamente (tasa 12,11%, diferencia -2,72 puntos, IC [-4,54; -0,95] que excluye
  el cero, p = 0,002, Holm RECHAZA) pero queda POR DEBAJO del umbral de materialidad de
  5 puntos fijado antes de medir, así que por la conjunción del criterio el veredicto es
  **no**. Es el ÚNICO de los 17 contrastes realizados donde las dos mitades del criterio
  discrepan. Para qué sirve prerregistrar un umbral: sin él, 2,72 puntos con p = 0,002 se
  habría presentado como hallazgo.

#### Detectabilidad de proxies (primario `race`)
- AUC OVR macro **0,737** (261 árboles), con las 6 clases por encima de 0,65: `Missing`
  0,813, AfricanAmerican 0,750, Asian 0,735, Caucasian 0,732, Hispanic 0,723, Other
  0,667 (`phase4_fairness_proxies*.csv`).
- **Redacción obligatoria, en este orden:** `race` es RECUPERABLE desde el resto del
  formulario, o sea que QUITARLA DEL MODELO NO SERÍA SUFICIENTE. Esto **NO significa que
  el modelo la use**.
- `gender` 0,630: zona gris, y es secundario.

#### Recuento final de contrastes (`phase4_fairness_contrast_count.csv`)
- Planificados 20, realizados 17 (16 en el CSV; el de proxies se calcula aparte): composición 7 de 9, FNR 4 de 5, distributivo 5 de 5,
  proxies 1 de 1 (el de `race`). `m` final de Holm: 4 en FNR y 5 en distributivo.
- Las familias secundarias (`race`/`gender` en composición, `payer_code`/`gender` en FNR
  y distributivo) se calculan SIN p-valores ni Holm y se marcan
  `rol = secundario/exploratorio`, como fija el prerregistro.
- Comparabilidad: el análisis distributivo es train+val y el resto val, así que sus
  cifras no se comparan directamente entre sí.

### Prerregistro de la ablación de `payer_code` y de la consulta única a test
Escrito y commiteado ANTES de ejecutar cualquiera de las dos cosas.

#### A. Ablación de `payer_code` (decide qué modelo es el entregable)
- Montaje: modelo corregido SIN `payer_code`, mismo split CONGELADO, mismos
  hiperparámetros (`num_leaves=31`, `min_child_samples=20`, `learning_rate=0.03`,
  SEED=42), early stopping sobre val con la etiqueta corregida, y su PROPIO guardarraíl
  (nº de árboles y PR-AUC en val), que se anota la primera vez que corra.
- Qué se mide:
  - (a) **Lift** en val y en CV agrupada de 5 folds dentro de train, frente al modelo
    corregido CON `payer_code`. En la CV, `n_estimators` fijo en los dos para que lo
    único que cambie sea la feature.
  - (b) **Composición del top 10% por `payer_code`** en el modelo ablacionado: los grupos
    siguen existiendo en los datos aunque el modelo no vea la variable. Mismos cocientes,
    misma referencia **MC**, mismos IC bootstrap (2.000 réplicas agrupadas por
    `patient_nbr`, semilla 42) y mismos mínimos (n >= 100 y >= 20 marcados).
- **REGLA DE DECISIÓN, fijada antes de medir.** El modelo ablacionado pasa a ser el
  ENTREGABLE si y solo si se cumplen LAS DOS:
  1. REDUCE el recuento de **fallos ROBUSTOS**, y
  2. su lift en CV NO cae más de un **5% relativo** frente al modelo con `payer_code`.
  Si no se cumplen las dos, el entregable sigue siendo el modelo CON `payer_code` y la
  recomendación al gestor se documenta SIN cambiar el modelo.
- **Fallo ROBUSTO** (criterio PRIMARIO del recuento) = estimación puntual fuera de
  [0,80; 1,25] **Y** IC95 del cociente que excluya el 1. Hoy son **2**: `Missing`
  [1,754; 2,463] y `SP` [0,239; 0,569]. Por qué se cambia el recuento bruto por este: con
  el recuento bruto, la ablación podría "reducir fallos" solo porque `HM` [0,973; 1,839]
  o `BC` [0,410; 1,215] cruzaran el umbral POR RUIDO, ya que sus IC incluyen la paridad.
- **Recuento BRUTO de fallos por umbral** (el criterio original): se sigue reportando,
  pero como SECUNDARIO y descriptivo. Hoy son 4.
- Precisión para que el recuento no sea ambiguo: se compara sobre el CONJUNTO COMÚN de
  grupos elegibles en los dos modelos, porque la ablación puede cambiar qué grupos llegan
  a 20 marcados. Se reportan además los dos recuentos completos y qué grupos entran o
  salen de la elegibilidad.
- **Tabla completa lado a lado:** se reportan los cocientes de los 10 grupos con sus IC,
  antes y después de la ablación, porque el RECUENTO NO CAPTURA LA MAGNITUD. Si `SP`
  pasara de 0,385 a 0,60 seguiría fallando pero habría mejorado de verdad, y eso tiene
  que verse. El cambio de magnitud se reporta como DESCRIPTIVO, sin veredicto propio.
- **HIPÓTESIS DE JAIME, escrita antes de medir:** la disparidad PERSISTIRÁ en buena
  medida, porque el análisis de proxies ya mostró que la información protegida es
  recuperable del resto del formulario (`race` AUC OVR macro 0,737, las 6 clases sobre
  0,65). Si se confirma, **ese es el hallazgo: quitar la variable no arregla el problema,
  solo lo hace invisible y no auditable.** Se reporta salga como salga.
- Matiz obligatorio para el informe: la disparidad mayor es `Missing` (cociente 2,073),
  que son **7.933 de las 19.868 filas de val** (39,9%). Eso es AUSENCIA DE DATO
  ADMINISTRATIVO, no un tipo de seguro, y se discute como categoría DISTINTA de las
  demás, no como un seguro más.

#### B. Consulta única a test
- Una sola ejecución, con hiperparámetros, split, corrección y elección de modelo ya
  cerrados. Nada cambia después de verla; si el número sale mal, se reporta igual.
- Guardarraíl antes de tocar test: el modelo entregable reproduce su nº de árboles y su
  PR-AUC en val. NO se reentrena con train+val (igual que en la Fase 2).
- Qué se mide en test, con la etiqueta CORREGIDA:
  - ROC-AUC, PR-AUC, **prevalencia y lift** (el titular es el lift), y Brier.
  - Top-k 5/10/20%: marcados, aciertos, precisión, recall y lift, que es lo que el gestor
    aplica.
  - Líneas base en test con la etiqueta corregida: prevalencia y regla
    `number_inpatient > 0`. Sin referencia, el PR-AUC no se interpreta.
- **Replicación de los TRES pasos de la brecha A - B en test**, no solo el (3): el paso
  (3) en val salió compatible con cero, así que replicar solo ese sería replicar la
  medida más débil. El titular es la inversión entre (1) y (2), y para eso hay que
  puntuar también el MODELO VIEJO en test. Condiciones:
  - Es replicación **DESCRIPTIVA**: el criterio prerregistrado se definió y se RESOLVIÓ
    en val, y NO se redefine con test.
  - Se reporta tal como salga, incluso si CONTRADICE a val, y se dice si contradice.
  - No se toma NINGUNA decisión a partir de ella: ni de modelo, ni de features, ni de
    umbral.
  - Incluye el bootstrap EMPAREJADO (mismo esquema: 2.000 réplicas por `patient_nbr`,
    semilla 42) y el `n` de cada subgrupo. Si el `n` de A en test es pequeño, el IC será
    ancho y se dice.
- Qué NO se mide en test: SHAP, equidad y la regresión logística. Lo primero y lo segundo
  viven en val por diseño; lo tercero se excluye para no crear la tentación de un cambio
  de modelo con test ya visto.
- Se declara en la memoria: es la SEGUNDA vez que se toca test, para qué, y que NO es
  comparable con el 0,227 de la Fase 2 porque el target cambió.

### Fase 4 — Resultados de la ablación de `payer_code` (`src/phase4_ablation.py`)
Prerregistro commiteado ANTES de ejecutar (`8872ae6`). Todo en val y en CV dentro de
train. Test NO se tocó.

#### Veredicto de la regla: el entregable pasa a ser el modelo SIN `payer_code`
`reports/phase4_ablation_decision.csv`: las DOS condiciones se cumplen.
- Cond. 1 — reduce el recuento de fallos ROBUSTOS: **2 -> 1** sobre el conjunto COMÚN
  de 7 grupos elegibles. El que desaparece es `Missing`; el que queda es `SP`.
- Cond. 2 — el lift en CV no cae más de un 5% relativo: cae un **2,31%**
  (1,9878 -> 1,9419).
- **Guardarraíl del nuevo entregable: 123 árboles, PR-AUC en val 0,1868**
  (`phase4_ablation_val.csv`). El guardarraíl del modelo CON `payer_code` (183 árboles,
  PR-AUC val 0,1893) NO se toca: queda como registro de lo que se hizo, igual que el de
  122 árboles de la Fase 2.

#### Rendimiento: la pérdida es pequeña pero CONSISTENTE
Val con la etiqueta corregida, prevalencia 9,78% (`phase4_ablation_val.csv`); CV de 5
folds con `n_estimators` fijo en los dos para que lo único que cambie sea la feature
(`phase4_ablation_cv.csv`):

| | val PR-AUC | val lift | CV lift medio | CV PR-AUC medio |
|---|---|---|---|---|
| con `payer_code` | 0,1893 | 1,936 | 1,9878 ± 0,0575 | 0,1940 |
| sin `payer_code` | 0,1868 | 1,910 | 1,9419 ± 0,0651 | 0,1895 |

- El modelo CON `payer_code` gana en **5 de 5 folds** (lift por fold: 2,075/2,054;
  2,006/1,930; 1,925/1,884; 1,982/1,922; 1,952/1,919). La caída es pequeña en magnitud,
  pero no es ruido de un fold: es sistemática.

#### HALLAZGO: la hipótesis prerregistrada se REFUTA en el caso dominante y acierta en el otro extremo
La predicción escrita antes de medir era que la disparidad PERSISTIRÍA en buena medida,
porque `race` es recuperable del resto del formulario (AUC OVR macro 0,737). **Esa
predicción queda REFUTADA en el grupo que dominaba el problema:** `Missing` se desploma
de **2,073** [1,754; 2,463] a **0,962** [0,829; 1,105] y deja de ser un fallo, ni
robusto ni bruto. La mayor disparidad del sistema **ERA la variable, no un proxy**.
No se suaviza: predije lo contrario y salió lo contrario.

Pero la hipótesis acierta en el otro extremo, y la distinción es más interesante que
cualquiera de las dos predicciones por separado:
- **La disparidad por DATO AUSENTE era la variable.** `Missing` (7.933 de 19.868 filas de
  val, 39,9%) es ausencia de dato administrativo, no un tipo de seguro, y desaparece al
  quitar la columna: el modelo estaba leyendo literalmente "no consta el seguro" como
  señal de riesgo.
- **La disparidad de los pacientes SIN SEGURO no lo era.** `SP` (self-pay) sigue siendo
  fallo robusto tras la ablación (0,385 -> **0,444**, IC [0,302; 0,597]), y `CP` y `UN`
  también lo son en el modelo ablacionado (0,555 [0,334; 0,849] y 0,542 [0,339; 0,782]).
  Esa parte SÍ es recuperable del resto del formulario, así que para esos grupos se
  cumple lo que decía la hipótesis: quitar la variable no arregla el problema.

#### Los dos recuentos apuntan en sentidos distintos, y por qué (hay que decirlo)
- Sobre el conjunto COMÚN de 7 grupos elegibles: fallos robustos **2 -> 1**. Es el
  criterio prerregistrado y es el que decide.
- Sobre los 9 grupos no-referencia: **2 -> 3**. **Esto NO es un empeoramiento.** `CP` y
  `UN` pasan de 18 y 17 marcados a 30 y 28, cruzan el mínimo de >= 20 y reciben veredicto
  POR PRIMERA VEZ; y sus cocientes **MEJORARON** (0,500 -> 0,555 y 0,494 -> 0,542). El
  aumento del recuento es un artefacto del umbral de elegibilidad, no de que ningún grupo
  haya ido a peor.
- El recuento bruto por umbral (secundario, descriptivo) se queda en **4 -> 4**: sale
  `Missing`, entra `CP`, y `HM` deja de fallar mientras `UN` entra.
- Conclusión: ninguno de los recuentos responde la pregunta por sí solo, porque los tres
  se mueven por razones mecánicas distintas (conjunto común, umbral de elegibilidad,
  umbral de los cuatro quintos). **Lo que la responde es la tabla de magnitudes.**

#### Tabla de magnitudes (`phase4_ablation_composition.csv`), descriptiva y sin veredicto propio
Cociente de tasa de marcado frente a MC en el top 10% de val, con IC95 bootstrap
(2.000 réplicas agrupadas por `patient_nbr`, semilla 42):

| grupo | n_grupo | cociente CON (IC95) | cociente SIN (IC95) | cambio |
|---|---|---|---|---|
| `Missing` | 7.933 | **2,073** [1,754; 2,463] | 0,962 [0,829; 1,105] | **-1,111** |
| HM | 1.167 | 1,349 [0,973; 1,839] | 0,968 [0,699; 1,241] | -0,381 |
| MD | 661 | 1,087 [0,667; 1,606] | 1,098 [0,781; 1,470] | +0,012 |
| MC (ref) | 6.201 | 1,000 | 1,000 | 0 |
| Other | 547 | 0,985 [0,529; 1,508] | 0,941 [0,563; 1,380] | -0,044 |
| CM | 367 | 0,903 [0,398; 1,572] | 0,876 [0,473; 1,323] | -0,027 |
| BC | 945 | 0,775 [0,410; 1,215] | 0,758 [0,495; 1,062] | -0,016 |
| UN | 475 | 0,494 [0,273; 0,792] | **0,542** [0,339; 0,782] | +0,047 |
| CP | 497 | 0,500 [0,268; 0,821] | **0,555** [0,334; 0,849] | +0,054 |
| SP | 1.075 | **0,385** [0,239; 0,569] | **0,444** [0,302; 0,597] | +0,059 |

Lectura exacta, en tres movimientos distintos. **NO se dice "todos mejoraron", porque no
es cierto:**
1. Los dos grupos por ENCIMA de la paridad bajan A la paridad: `Missing` 2,073 -> 0,962 y
   `HM` 1,349 -> 0,968.
2. Los tres más INFRASELECCIONADOS suben, pero siguen fallando: `SP` 0,385 -> 0,444,
   `CP` 0,500 -> 0,555 y `UN` 0,494 -> 0,542.
3. Cuatro grupos intermedios se mueven EN CONTRA (se alejan de la paridad), con cambios
   de entre 0,012 y 0,044: `MD` +0,012 (ya estaba por encima), `BC` -0,016, `CM` -0,027 y
   `Other` -0,044. Los cuatro cambios caben dentro de sus propios IC, así que no se leen
   como efecto, pero se reportan.

#### Recomendación al gestor (la "acción" del enunciado), formulada como el intercambio medido que es
No es "el modelo es más justo sin la variable" ni "quitarla no cuesta nada". Es un
intercambio con las dos mitades medidas:
- **Lo que cuesta:** el lift baja de 1,9878 a 1,9419 en CV (2,31% relativo) y de 1,936 a
  1,910 en val, y el modelo con la variable gana en 5 de 5 folds. Pérdida pequeña, pero
  consistente y no ruido.
- **Lo que compra:** elimina la mayor disparidad del sistema, `Missing` 2,073 -> 0,962,
  que afecta al 39,9% de las filas de val y que era la variable y no un proxy.
- **Lo que NO compra:** la infraselección de los pacientes sin seguro (`SP`, `CP`, `UN`)
  sigue ahí, porque es recuperable del resto del formulario. Quitar la columna no la
  arregla; si acaso la deja menos auditable, porque ya no hay una variable a la que
  mirar.
- **La decisión de aceptar ese intercambio es del gestor, no del modelo.** Lo que aporta
  este trabajo son los dos números y su incertidumbre, no la elección. Lo mismo que con
  el k del seguimiento.

### Fase 4, punto 4 — CONSULTA ÚNICA A TEST (`src/phase4_test_final.py`)
Prerregistro commiteado ANTES de ejecutar (`8872ae6`, ampliado en `2f2b80c`). Una sola
ejecución, con hiperparámetros, split, corrección y elección de modelo ya cerrados.
**Nada ha cambiado después de verla.** Es la SEGUNDA vez que se toca test: la primera fue
la Fase 2c con el target ORIGINAL, y **estas cifras NO son comparables con el PR-AUC
0,227 de entonces**, porque el target cambió y responden a otra pregunta (cómo predice el
TARGET CORREGIDO en datos no vistos).

Guardarraíles comprobados antes de puntuar test, con `assert`: viejo 122 árboles /
PR-AUC val 0,2308; corregido CON `payer_code` 183 / 0,1893; **entregable (sin
`payer_code`) 123 / 0,1868**.

#### 1. LA DISCREPANCIA CON VAL, que va en primer plano: el paso (1) NO replica
El prerregistro obliga a declarar si la replicación contradice a val. Contradice.

| paso (1) viejo/vieja | brecha A - B | IC95 | cociente |
|---|---|---|---|
| val (`phase4_prediction_gaps.csv`) | **+4,99** | [+0,07; +9,99] | 1,183 |
| test (`phase4_test_gaps.csv`) | **-2,40** | [-7,45; +2,41] | 0,915 |

**En test el grupo A nunca estuvo por delante de B.** El IC de test incluye el cero y la
estimación puntual tiene el signo contrario al de val. Consecuencia para el relato, que
se reescribe aquí y no se deja como nota al pie:
- La afirmación correcta **NO** es "una ventaja resultó ser un artefacto".
- Es: **la supuesta ventaja de A sobre B era MARGINAL en val y AUSENTE en test, así que
  no sobrevive ni a la limpieza de la etiqueta ni a datos nuevos.**
- Encaja con el matiz que ya estaba anotado antes de ver test: el IC del paso (1) en val
  ROZABA el cero ([+0,07; +9,99]), o sea que la ventaja ya era débil de por sí. Test lo
  confirma por la vía más dura: en datos nuevos no está.

#### 2. LO QUE SÍ REPLICA: el titular prerregistrado, en signo y significación
Contraste EMPAREJADO, mismas 2.000 réplicas por `patient_nbr`, semilla 42, 0 descartadas
(`phase4_test_paired_contrasts.csv`):

| contraste | val | test | IC95 en test | excluye el nulo |
|---|---|---|---|---|
| brecha(2) - brecha(1) | -10,50 puntos | **-4,63 puntos** | [-7,57; -1,65] | **sí** |
| cociente(2) / cociente(1) | 0,651 | **0,777** | [0,667; 0,887] | **sí** |

- Lo que replica es **el SIGNO y la SIGNIFICACIÓN, no el TAMAÑO**: la magnitud en test es
  **menos de la mitad** que en val (-4,63 frente a -10,50). Se dice así y no se presenta
  como una replicación limpia.
- Los dos contrastes salen idénticos con (3a) y con (3b), porque solo intervienen el
  modelo viejo y las dos etiquetas. Está anotado en el código para que no parezca un
  duplicado por descuido.
- En test, limpiar la etiqueta baja la precisión de A de 25,995% a 17,330% y la de B de
  28,397% a 24,359% (`phase4_test_steps.csv`): el mismo ranking y el mismo corte (0,1942),
  solo cambia la verdad contra la que se puntúa.

#### 3. EL CRITERIO PRERREGISTRADO NO REPLICA CON SIGNIFICACIÓN (no se esconde detrás del emparejado)
`delta_A - delta_B` (`phase4_test_paired_contrasts.csv` y `phase4_test_deltas.csv`):

| | val | test (3a) corregido CON | test (3b) entregable SIN |
|---|---|---|---|
| `delta_A - delta_B` | **+8,06** [+2,78; +13,75] | **+0,64** [-4,55; +5,64] | **+3,45** [-2,90; +9,28] |

- En val excluía el cero; en test **los dos IC lo INCLUYEN**. La cantidad del criterio
  prerregistrado no replica con significación en datos nuevos.
- El veredicto formal sigue apuntando en la misma dirección (`delta_A > delta_B` en los
  dos casos: 6,47 > 5,83 y 9,23 > 5,78), pero sin significación, así que no se usa para
  concluir nada.
- Esto NO se presenta tapado por el contraste emparejado del punto 2. Son dos cantidades
  distintas: la del punto 2 replica, la del criterio no.

#### 4. EL MECANISMO SÍ REPLICA
`phase4_test_steps.csv`, columna `pct_destino_22_o_5` del subgrupo A del top 10%:
- **63,93% con el modelo viejo -> 32,38% con (3a) y 35,40% con (3b)**, el mismo
  movimiento que en val (69,32% -> 34,43%).
- Y **`n_A` cae de 427 a 210 (3a) y a 161 (3b)**: el modelo corregido, y más aún el
  entregable, DEJA DE MARCAR a esa población. En val `n_A` caía de 440 a 212.
- Es la confirmación mecánica de que la corrección hizo lo que decimos, y replica en
  datos no vistos.

#### 5. Los dos resultados favorables, que tampoco se entierran
- **El entregable GENERALIZA bien, sin señal de sobreajuste:** lift **1,9100 en val**
  (`phase4_ablation_val.csv`) frente a **1,9194 en test** (`phase4_test_global.csv`), con
  prevalencias 9,7795% y 9,8455%. Test queda incluso ligeramente por encima de val.
  Globales en test: ROC-AUC 0,6655, PR-AUC 0,1890, Brier 0,0855.
- **Gana a la regla `number_inpatient > 0` en test en los TRES k**, y el margen es mayor
  cuanto más estrecho es el seguimiento (`phase4_test_topk.csv`):

| k | entregable: precisión / lift | regla: precisión / lift |
|---|---|---|
| 5% (994 marcados) | 26,86% / **2,728** | 14,08% / 1,431 |
| 10% (1.987) | 22,14% / **2,249** | 14,80% / 1,503 |
| 20% (3.974) | 18,32% / **1,861** | 15,27% / 1,551 |

  Es coherente con la dependencia de k que ya estaba medida en la Fase 2: donde el modelo
  aporta es en priorizar a pocos, y la ventaja se estrecha al ampliar el seguimiento.
  Líneas base en test con la etiqueta corregida: prevalencia 9,8455% (lift 1 por
  construcción) y la regla con PR-AUC 0,1246 / lift 1,2660.

#### 6. Observación SECUNDARIA, con su cautela: el ROC-AUC se mantiene y el PR-AUC baja por prevalencia
- ROC-AUC en test: **0,667** (modelo original, etiqueta vieja, `metrics_final.csv`) ->
  **0,6655** (entregable, etiqueta corregida). Prácticamente igual.
- PR-AUC en test: **0,227 -> 0,1890**, con la prevalencia bajando del 11,39% al 9,8455%.
- **La comparación NO es estricta:** son dos tareas distintas (dos targets), dos modelos
  distintos y dos conjuntos de etiquetas distintos. No se presenta como un antes/después.
- Dicho eso, apunta en la MISMA dirección que la CV de la corrección: **la contaminación
  no sostenía el rendimiento global**. Lo que sostenía era la ventaja concreta en el grupo
  donde el modelo discrepaba de la regla simple, y eso es lo que los puntos 1 a 4
  documentan.

#### Qué NO se midió en test, deliberadamente (está escrito en el script)
- Las globales, el top-k y las líneas base son SOLO del entregable. Las predicciones del
  corregido CON `payer_code` se calculan porque el paso (3a) las necesita, pero sus
  globales NO: tenerlas invitaría a replantear qué modelo se entrega, y eso está decidido
  en val. Igual con el modelo viejo, que solo entra en los pasos (1) y (2).
- SHAP y equidad viven en val por diseño. La regresión logística se excluye para no crear
  la tentación de un cambio de modelo con test ya visto.
- Por qué DOS pasos (3): (3a) es el modelo que se usó en val, así que es la replicación
  estricta; (3b) es el entregable. De ninguno sale ninguna decisión, y el titular tampoco
  depende de esa elección, porque la inversión de (1) a (2) solo usa el modelo viejo.

### Prerregistro de la Fase 5A — el entregable frente a una regresión logística
Escrito y commiteado ANTES de ejecutar nada. Lo pide el profesor: si LightGBM es claramente
superior, eso apunta a no linealidad en los datos. Lo único que se consultó para escribir esto
son artefactos ya publicados del repo; no se ha entrenado ninguna LR nueva.

#### Principio de diseño: ninguno de los dos modelos tiene ventaja de procedimiento
La LR sigue EXACTAMENTE el procedimiento del entregable: se entrena SOLO con train, su único
hiperparámetro se elige por PR-AUC en val (el mismo criterio y los mismos datos con los que se
eligieron los de LightGBM), NO se reentrena con train+val, y se evalúa en TEST, el conjunto
donde se reportan las cifras finales del entregable.

#### Condiciones fijas (ninguna se toca después de ver resultados)
- **Partición CONGELADA**: la de `D.grouped_split` con el target original, SEED=42, agrupada
  por `patient_nbr`. El reetiquetado se aplica después, sobre filas ya asignadas
  (`TC.relabel`), igual que en toda la Fase 4.
- **Etiqueta CORREGIDA** en los tres splits.
- **Mismas variables que el entregable**: `P.FEATURES` menos `payer_code`. Nada añadido, nada
  quitado.
- `A1Cresult` y `max_glu_serum` mantienen `NotMeasured` como categoría propia: no haber hecho
  la prueba es información (decisión de la Fase 1, no de esta fase).
- **Preprocesado de la LR: el que ya existe desde la Fase 2a** (`P.build_lr_preprocessor`),
  dentro de un `Pipeline` que se ajusta SOLO con train: `StandardScaler` en las numéricas,
  `log1p` + `StandardScaler` en `number_outpatient/emergency/inpatient`, y
  `OneHotEncoder(min_frequency=0.01, handle_unknown="infrequent_if_exist", drop="if_binary")`
  en las categóricas (diagnósticos ya agrupados por `features.py`). Penalización L2, la de
  `LogisticRegression` por defecto. **No se añade ninguna transformación nueva**: el `log1p` de
  los contadores se fijó en la Fase 2a, antes de cualquier resultado de esta comparación, y no
  se eligió para favorecer a la LR.
- **Sin reponderado de clases en ninguno de los dos**: LightGBM va sin `class_weight`,
  `is_unbalance` ni `scale_pos_weight`, y la LR con `class_weight=None`.
- **Rejilla de C fijada de antemano**: `C_GRID = [0.001, 0.01, 0.1, 1, 10]`, la misma de la
  Fase 2a. Se elige el C de mayor PR-AUC en val, entrenando en train. Se reporta la rejilla
  COMPLETA, no solo el ganador.
- **Si el ganador cae en un extremo de la rejilla (0,001 o 10), la rejilla NO se amplía.** Se
  reporta como GANADOR EN EL BORDE y se declara. Ampliarla tras ver el resultado sería elegir
  el espacio de búsqueda con la salida delante, y además daría a la LR más búsqueda en val que
  la que tuvo LightGBM.
- **Por qué C se re-elige con la etiqueta corregida y no se congela en el C=1 de la Fase 2a:**
  la LR tiene que entrenarse contra la etiqueta corregida (es su `y`), así que elegir C contra
  el PR-AUC de la etiqueta vieja sería incoherente. Es además el análogo exacto de lo que hizo
  el entregable: su nº de árboles (123) se re-eligió por early stopping sobre val con la
  etiqueta corregida. Si el C elegido no sale 1, se dice y se compara con la Fase 2a, pero SOLO
  el C elegido se puntúa en test: no se mandan dos LR a test.
- **El entregable NO se reentrena**: se reproduce y se exige su guardarraíl con `assert` (123
  árboles Y PR-AUC val 0,1868) ANTES de tocar test. Motivo: las predicciones del entregable en
  test no se guardaron en ningún CSV en la Fase 4, así que hay que recalcularlas; el
  guardarraíl es lo que garantiza que son las mismas.
- **Guardarraíl NUEVO de la LR**, que se anota en la etapa val la primera vez que corra: C
  elegido, PR-AUC en val y nº de columnas tras el one-hot. De ahí en adelante se exige con
  `assert`.
- **PR-AUC es `sklearn.metrics.average_precision_score`**, exactamente la misma métrica que usa
  el early stopping de LightGBM (`metric="average_precision"` en `train_lgbm.py`) y la que se
  reporta en toda la Fase 4. No se usa ninguna variante interpolada ni el área bajo la curva PR
  trapezoidal.
- **NO hay bloque de CV y NO hay figura.** Un solo contraste, en test. Los resultados van a la
  memoria como tabla.

#### Ejecución en DOS ETAPAS, no en una (`src/phase5_lr_comparison.py --stage {val,test}`)
Motivo: si la elección de C tuviera un error, con una sola ejecución lo descubriríamos con test
ya mirado. Las dos etapas se ejecutan en commits distintos y la primera se revisa antes de
lanzar la segunda.
1. **Etapa `val`** — verifica el guardarraíl del entregable, comprueba que la LR de la Fase 2a
   se reproduce, elige C con la etiqueta corregida y sin `payer_code`, y escribe
   `reports/phase5_lr_c_grid.csv` junto con el guardarraíl de la LR (C, PR-AUC en val y nº de
   columnas tras el one-hot). **NO carga test.**
2. **Etapa `test`** — exige con `assert` los DOS guardarraíles (el del entregable y el de la LR
   ya anotado en la etapa val) y solo entonces puntúa test, UNA vez.

#### Comprobación de que el refactor no cambia nada (etapa val, no toca test)
`P.build_lr_preprocessor` y `train_baseline.make_logreg`/`select_logreg` pasan a aceptar las
listas de columnas como parámetros, con valores por defecto idénticos a los actuales, para
poder construir la LR sin `payer_code` sin duplicar el preprocesador (dos definiciones del
mismo preprocesado pueden divergir, que es justo lo que `target_correction.py` evita con la
regla de reetiquetado). Para que "idénticos" no sea una afirmación de confianza, la etapa val
lo exige con `assert`, igual que un guardarraíl de modelo: **la llamada SIN argumentos
reproduce la LR de la Fase 2a en val** — C elegido = 1 y PR-AUC val = 0,2124
(`metrics_final.csv`, fila `LogReg,val`). Si falla, el refactor cambió algo y la Fase 5A no
sigue. Esta comprobación usa train y val; test no interviene.

#### Contraste primario: UNO, con veredicto
**Δ = PR-AUC(LightGBM entregable) − PR-AUC(LR)** en test, con la etiqueta corregida.
- IC95 por **bootstrap emparejado remuestreando PACIENTES** (no filas: los encuentros de un
  mismo paciente no son independientes y el split ya agrupa por `patient_nbr`), **2.000
  réplicas, semilla `D.SEED` = 42**, el mismo esquema de toda la Fase 4. Los dos modelos se
  puntúan DENTRO de cada réplica sobre las mismas filas, así que la variación entre pacientes
  se cancela. Se reportan `replicas_usadas` y `replicas_descartadas`.
- **Umbral de materialidad: Δ ≥ 0,01** (1 punto de PR-AUC), fijado ahora.
- Equivalencia en la escala titular del proyecto: con la prevalencia de test (9,8455%),
  Δ = 0,01 de PR-AUC son **0,102 de lift**. Dentro de una réplica los dos modelos comparten
  prevalencia, así que Δlift tiene siempre el mismo signo que ΔPR-AUC.
- **Honestidad sobre el umbral:** la brecha en test con el target VIEJO y con `payer_code` ya
  era visible en `reports/metrics_final.csv` (0,2274 frente a 0,2102 = 0,0172) cuando se fijó
  el 0,01. El umbral NO se eligió a ciegas, y se declara. No es la cifra de esta fase (otro
  target, otro modelo, otra feature), pero sitúa el umbral en el mismo orden de magnitud que la
  brecha esperada, así que el veredicto puede salir ajustado y eso no es una sorpresa a
  posteriori. **Más aún: esa brecha de 0,0172 sugiere DE ANTEMANO que el veredicto (a) es
  PLAUSIBLE.** Queda escrito aquí para que nadie pueda leer un (a) como un resultado
  inesperado.

#### Veredictos, fijados ANTES de medir, con lo que se concluye y lo que NO
Se aplican en este orden sobre el Δ observado en todo test y su IC95:

**(c) "indistinguibles" — el IC incluye 0.**
Se concluye: en este problema, con estas variables y este target, el modelo complejo no ordena
mejor a los pacientes que un modelo lineal. Eso activa el argumento de **Rudin (2019)**: si el
rendimiento es equivalente, para una decisión de alto riesgo sobre personas hay razón para
preferir el modelo interpretable por diseño, y el coste de la caja negra no está pagado por
nada. **NO se concluye** que los datos sean lineales ni que no haya interacciones: un empate en
PR-AUC es compatible con que las haya pero no sean recuperables con esta señal, esta prevalencia
y este tamaño. **Y un empate NO invalida el proyecto:** el objetivo de este trabajo es evaluar
la FIABILIDAD de las explicaciones post hoc de un modelo opaco, no demostrar que el modelo
opaco gane. El argumento de Rudin entra como el CONTRAPUNTO que es —si el lineal empata, ¿para
qué explicar a posteriori una caja negra?— y se discute como tal, no como una refutación de lo
que se ha hecho.

**(a) "LightGBM claramente superior" — el IC queda ENTERO POR ENCIMA de 0 y Δ ≥ 0,01.**
Se concluye: hay estructura que el modelo lineal no captura. **NO se concluye que sea no
linealidad**, y esto es lo importante: la brecha tiene al menos tres fuentes que este diseño NO
separa — (i) no linealidad en las numéricas, (ii) **interacciones**, que la LR no tiene porque
no se le han puesto términos de interacción, y (iii) el **manejo de las categóricas de alta
cardinalidad**, donde LightGBM parte directamente por la categórica nativa mientras la LR ve un
one-hot que colapsa en `infrequent` todo nivel con <1% en train. La redacción obligatoria es
"LightGBM captura estructura que la LR con este preprocesado no captura", no "los datos son no
lineales". Separar las tres fuentes exigiría otro experimento (splines, interacciones
explícitas, target encoding), que NO se hace aquí y se nombra como trabajo futuro.

**(b) "superior pero no material" — el IC queda ENTERO POR ENCIMA de 0 y Δ < 0,01.**
Se concluye: la ventaja es detectable pero por debajo del umbral de materialidad fijado antes de
medir. Es el mismo patrón que `AfricanAmerican` en el análisis distributivo de la Fase 4:
**detectable y no material a la vez**, y por la conjunción del criterio el veredicto es que no
hay superioridad relevante. **NO se presenta como hallazgo** que LightGBM gane. La lectura
práctica se acerca a la de (c), y el argumento de Rudin se menciona como atenuado, no como
establecido.

**(d) "LR superior" — el IC queda ENTERO POR DEBAJO de 0.**
Se concluye: el modelo complejo no aporta y además pierde. **NO se atribuye causa**: no se ha
medido ninguna, así que no se dice por qué pierde. Se reporta tal cual, aunque contradiga la
elección de modelo de la Fase 2. **NO se cambia el entregable**: la Fase 5A es descriptiva, la
elección de modelo se cerró en val y test ya está visto; cambiar de modelo mirando test
invalidaría la cifra. Si saliera (d), la consecuencia va al informe como limitación y como
recomendación para una iteración futura, no como un cambio de entregable.

En los cuatro casos se reportan Δ, su IC95 y los dos PR-AUC con su prevalencia y su lift al
lado, nunca el veredicto solo.

#### Secundarias: descriptivas, SIN veredicto
Con IC95 por el mismo bootstrap de pacientes, reportadas salgan como salgan y sin corrección por
comparaciones múltiples, porque no llevan veredicto:
- **ROC-AUC**, **lift** (= PR-AUC / prevalencia) y **Brier**.
- **Precisión y lift en el top 5%**, que es lo que el gestor aplica si solo puede seguir a 994
  pacientes de 19.867. El corte se recalcula dentro de cada réplica. El 10% y el 20% se añaden
  como contexto.
- **Accuracy**, y SOLO para enseñar por qué no sirve. **Umbral de decisión 0,5 en los dos
  modelos**, dicho explícitamente: no se ajusta ningún umbral, porque el umbral operativo de
  este proyecto es el top-k y no una probabilidad fija. Se reporta junto a la del clasificador
  trivial "nunca reingresa", que acierta el **90,1545%** por construcción (100 − 9,84547 de
  prevalencia en test). Si las tres accuracies son parecidas, eso es el argumento, no un
  resultado del modelo.

#### Coeficientes de la LR
Se guardan en `reports/phase5_lr_coefficients.csv` con el nombre de columna del
`ColumnTransformer`, su bloque, su variable original, el coeficiente y `exp(coef)`. **En la 5A
no se interpretan**: son el insumo de la 5B. Aviso ya anotado para entonces: los coeficientes
están en escalas distintas (por desviación típica en las numéricas, por dummy en las
categóricas), el one-hot solo elimina una columna en las binarias (`drop="if_binary"`), así que
hay colinealidad que la L2 reparte entre dummies, y el nivel `infrequent` actúa como categoría
de referencia implícita en varias variables.

#### Consulta a test: es la TERCERA del proyecto
- 1.ª Fase 2c (`evaluate_test.py`), modelo original con target original.
- 2.ª Fase 4 punto 4 (`phase4_test_final.py`), entregable con target corregido.
- 3.ª Fase 5A (`phase5_lr_comparison.py --stage test`), entregable CONGELADO frente a la LR, con
  target corregido.
- **Matiz que se declara y no se esconde:** la LR **ya fue puntuada en test en la primera
  consulta** (`metrics_final.csv`, filas `LogReg,test`: ROC-AUC 0,6586 / PR-AUC 0,2102), con el
  target VIEJO y CON `payer_code`. Esta es la segunda vez que la LR toca test. Esas cifras NO
  son comparables con las de la 5A (otro target, otro conjunto de features), y ninguna decisión
  de la 5A sale de ellas; lo único que hacen es hacer no-ciego el umbral de materialidad, que ya
  queda declarado arriba.
- Una sola ejecución de la etapa `test`, con split, corrección, features, hiperparámetros y
  rejilla de C cerrados. **Nada cambia después de verla.** Si el número sale mal, se reporta
  igual.
- Qué NO se mide en test en esta fase: SHAP, equidad, y cualquier variante adicional de la LR o
  de LightGBM. Nada se vuelve a elegir con test visto.

#### Asimetrías de procedimiento que quedan, declaradas antes de medir
Las dos primeras no se pueden eliminar sin romper la congelación del entregable, así que se
declaran en vez de corregirse:
1. **LightGBM exploró más val que la LR.** Su `num_leaves`/`min_child_samples` salieron de una
   rejilla de 6 combinaciones sobre val (Fase 2b) más el nº de árboles por early stopping; la LR
   explora 5 valores de C. Más búsqueda en val infla la métrica DE VAL, no la de test, así que
   no favorece a LightGBM en el contraste; lo que hace es que las cifras de val de los dos
   modelos no sean igual de optimistas, y por eso el contraste se hace en test.
2. **Dirección contraria:** `num_leaves`/`min_child_samples` se eligieron con la etiqueta VIEJA
   y están congelados, mientras el C de la LR se elige con la etiqueta corregida. En ese punto
   concreto la LR va algo mejor servida. No se corrige: reoptimizar LightGBM rompería el
   guardarraíl del entregable y todas las cifras publicadas de la Fase 4.
3. **Categóricas de alta cardinalidad:** la LR ve un one-hot con los niveles <1% de train
   colapsados en `infrequent`; LightGBM parte por la categórica nativa con todos los niveles. Es
   intrínseco a cada clase de modelo y se fijó en la Fase 2a, pero es una de las tres fuentes
   posibles de la brecha y por eso entra en la redacción obligatoria del veredicto (a).
4. **Valores no vistos en train:** para LightGBM pasan a NaN (`D.to_categorical`) y el árbol los
   trata como missing; para la LR caen en `infrequent`. Tratamientos distintos del mismo caso,
   los dos fijados antes de esta fase.

### Criterios fijos de los sanity checks (no se tocan tras ver resultados)
- Aleatorización de etiquetas: ROC-AUC en val ≤ 0,52 y mediana de |ρ| de
  Spearman por paciente (SHAP real frente a aleatorio) < 0,3. Correlación del
  ranking global ≥ 0,7: solo aviso.
- Barajado guiado: la caída de PR-AUC de las top-k de SHAP supera el percentil
  95 de 100 conjuntos aleatorios del mismo tamaño.
- Sondas de ruido: las top-10 superan a la mejor sonda en todas las semillas;
  las que no, no se interpretan.
- Estabilidad entre folds: ρ global ≥ 0,8; top-3 locales, comparadas por caso.

### Reglas de git
- Repo propio en `final-project/`, rama `main`. El repo de `C:\Users\Jaime` es
  otro (CMSC414) y no se toca.
- `user.email` local: `jaimepedrosacom@gmail.com`. El global no se cambia.
- Commits siempre a mi nombre: nunca `Co-Authored-By: Claude` ni similares.
- Antes de cada commit: enseñar status/diff y mensaje propuesto, y esperar mi
  visto bueno.
- El push se puede hacer directamente, sin preguntar.
- No se versionan `data/raw/` ni el PDF del enunciado.
