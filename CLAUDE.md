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

## Stack
Python, pandas, scikit-learn, lightgbm, shap, dice-ml (o alibi) para
counterfactuals, matplotlib/seaborn para gráficas.

## Estado del proyecto y decisiones tomadas (actualizado: 2026-11-09)

### Estado
- Hecho: datos, baseline, modelo principal y evaluación final en test
  (commits `e4a0354`, `d9a017c`, `ff22295`, `16f017b`).
- Fase 3 (XAI) COMPLETA: 3a (`src/xai_shap.py`, global), 3b (`src/xai_sanity.py`,
  sanity checks + auditoría del destino), 3c (`src/xai_counterfactuals.py`,
  contrafactuales) y 3d (`src/xai_local.py`, explicaciones locales de los 11
  casos).
- Pendiente: Fase 4 (acción), informe y `README.md`.
- Entorno: venv fuera del proyecto, en `C:\venvs\xai-final` (dentro del
  proyecto la ruta supera el límite de 260 caracteres de Windows).
  Versiones fijadas en `requirements.txt`.

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
- De los 11 casos de la Fase 3a, 6 son elegibles (prob >= umbral del top 10% =
  0,1954 en val). Los otros 5 no los marca el modelo: no hay nada que bajar.
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
- Matiz para el informe: los 6 casos se eligieron en la Fase 3a por interés
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
