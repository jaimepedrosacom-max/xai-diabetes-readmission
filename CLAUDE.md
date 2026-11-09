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
- Fase 3 (XAI): hechas 3a (`src/xai_shap.py`, global), 3b (`src/xai_sanity.py`,
  sanity checks + auditoría del destino) y 3c (`src/xai_counterfactuals.py`,
  contrafactuales). PENDIENTE dentro de la Fase 3: las explicaciones locales SHAP
  de los 11 casos de `reports/xai_local_cases.csv`. Ese CSV es solo la selección
  (`src/xai_cases.py`), y los casos se consumen hasta ahora en `fold_stability`
  (estabilidad del top-3 local entre folds, que es el sanity check SOBRE las
  explicaciones locales, no la explicación) y en la Fase 3c. No hay waterfall ni
  tabla de contribuciones por caso en `reports/`.
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
