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