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

## Estado del proyecto y decisiones tomadas (actualizado: 2026-11-11)

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
- Fase 4 en curso: puntos 1-3 y 5 hechos (cifras "antes", corrección del target,
  predicción prerregistrada y equidad). Pendiente: consulta final a test (punto 4, que va
  al final por su propia condición) e informe + `README.md` (6).
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
  al permutar de 0,070 y 0,057 — muy por encima del tercer puesto (0,007). A
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
   de ~3,0x a ~1,9x frente a casa (tasa 15,6% frente a 8,1% del código 1). El
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
  aleatorias es 0,856 (umbral de aviso ≥ 0,7). Indica que el orden medio de la
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
  ~28% de los positivos que no tiene encuentro posterior visible (2.548 de 9.051 en
  train+val; 28,6% en val), que se reporta como límite.
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
  11.425 filas de train con Ch sin ningún Up/Down. Cuando un candidato elimina
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
  top 10% completo, 451 de 1.987 marcados tienen contrafactual (22,70%); dentro
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
  correlacionan 0,987 y sus centroides son el par MÁS CERCANO de los tres (distancia
  0,650, frente a 1,108 y 1,630). La diferencia está concentrada en la INTENSIDAD de
  `number_inpatient` (+1,279 con valor 5 frente a +0,703 con valor 4, que son 0,575 de
  los 0,591 de distancia entre los dos pacientes) y en el signo de `diag_2_group`
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
  **no**. Es el ÚNICO de los 16 contrastes realizados donde las dos mitades del criterio
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
- Planificados 20, realizados 16: composición 7 de 9, FNR 4 de 5, distributivo 5 de 5,
  proxies 1 de 1 (el de `race`). `m` final de Holm: 4 en FNR y 5 en distributivo.
- Las familias secundarias (`race`/`gender` en composición, `payer_code`/`gender` en FNR
  y distributivo) se calculan SIN p-valores ni Holm y se marcan
  `rol = secundario/exploratorio`, como fija el prerregistro.
- Comparabilidad: el análisis distributivo es train+val y el resto val, así que sus
  cifras no se comparan directamente entre sí.

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
