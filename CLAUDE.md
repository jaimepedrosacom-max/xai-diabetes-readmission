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

## Estado del proyecto y decisiones tomadas (actualizado: 2026-11-04)

### Estado
- Hecho: datos, baseline, modelo principal y evaluación final en test
  (commits `e4a0354`, `d9a017c`, `ff22295`, `16f017b`).
- En curso: Fase 3 (XAI), con plan aprobado. Orden: `src/xai_shap.py`, luego
  `src/xai_sanity.py`, luego `src/xai_counterfactuals.py`.
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
