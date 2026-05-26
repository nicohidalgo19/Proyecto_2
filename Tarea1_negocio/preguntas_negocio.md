# Tarea 1 — Preguntas de Negocio

## Proyecto 2 · Analítica Computacional para la Toma de Decisiones

**Cliente:** Secretaría de Educación del Huila  
**Objetivo:** Desarrollar modelos predictivos que permitan anticipar el desempeño académico de estudiantes del Huila en las pruebas Saber 11, con el fin de orientar la inversión de recursos educativos de forma focalizada y oportuna.

---

## Pregunta 1
**Responsable:** Juan Nicolás Hidalgo Parra  
**Rol:** Ciencia de Datos / Tablero de datos

### ¿Cuál es el puntaje global esperado de un estudiante dado su estrato socioeconómico, zona geográfica (urbano/rural), nivel educativo de los padres y jornada escolar?

| Campo | Detalle |
|---|---|
| **Tipo de modelo** | Regresión con Red Neuronal (Keras / TensorFlow) |
| **Variable objetivo (Y)** | `puntaje_global` — continua, rango 100–500 |
| **Variables predictoras (X)** | estrato, area_colegio, educacion_madre, educacion_padre, jornada, municipio, tiene_internet, tiene_computador, naturaleza_colegio, genero |
| **Modelo seleccionado** | Modelo Base: Input → Dense(32, ReLU) → Dense(16, ReLU) → Output(lineal) |
| **Optimizador** | Adam (lr = 0.0001) |
| **Métricas del modelo final** | RMSE: 40.44 pts · MAE: 32.44 pts · R²: 0.2748 |
| **Experimento MLflow** | `Regresion_PuntajeGlobal_Huila` |

**Justificación para la Secretaría:**  
El modelo permite estimar el puntaje esperado de un estudiante con base en su contexto socioeconómico e institucional, sin necesidad de esperar los resultados del examen. Esto facilita identificar estudiantes en riesgo de bajo desempeño antes de la prueba y simular el impacto de intervenciones concretas —como mejorar conectividad o cambiar de jornada— sobre el puntaje esperado.

**Conexión con el Proyecto 1:**  
Extiende y formaliza los hallazgos descriptivos del P1, donde se identificó que la zona geográfica y el estrato son los determinantes estructurales más relevantes del rendimiento en el Huila, construyendo sobre ellos un modelo predictivo cuantitativo.

---

## Pregunta 2
**Responsable:** Juan Sebastián Méndez Martínez  
**Rol:** Ciencia de Datos / Despliegue y mantenimiento

### ¿Es posible predecir si un estudiante asiste a una jornada no convencional (nocturna o sabatina) a partir de su perfil socioeconómico y municipio?

| Campo | Detalle |
|---|---|
| **Tipo de modelo** | Clasificación binaria con Red Neuronal (Keras / TensorFlow) |
| **Variable objetivo (Y)** | `jornada_no_convencional`: 0 = Completa / Mañana / Tarde (jornadas regulares) · 1 = Nocturna / Sabatina (jornadas no convencionales) |
| **Variables predictoras (X)** | estrato, area_colegio, educacion_madre, educacion_padre, municipio, tiene_internet, tiene_computador, naturaleza_colegio, genero |
| **Modelo seleccionado** | Modelo 2 con Dropout (mejor AUC: 0.8369) |
| **Optimizador** | Adam (lr = 0.001) |
| **Métricas del modelo final** | AUC-ROC: 0.8369 |
| **Experimento MLflow** | `Clasificacion_Jornada_No_Convencional_Huila` |

**Justificación para la Secretaría:**  
Las jornadas nocturna y sabatina presentaron el peor rendimiento académico en el P1. Identificar proactivamente qué estudiantes tienen alta probabilidad de pertenecer a estas jornadas —antes de que presenten el examen— permite intervenciones preventivas dirigidas a la población más vulnerable del sistema. El modelo invierte la pregunta típica: en lugar de explicar el bajo rendimiento, busca identificar el perfil que lo antecede.

**Conexión con el Proyecto 1:**  
Extiende el hallazgo del P1 sobre las disparidades estructurales por tipo de jornada, formalizando ese patrón descriptivo en un clasificador predictivo que puede usarse como herramienta de alerta temprana.
---

## Pregunta 3
**Responsable:** Gabriel Juan De Dios  
**Rol:** Ciencia de Datos / Despliegue del tablero

### ¿Un estudiante estará por debajo del promedio departamental dado su perfil socioeconómico y municipio?

| Campo | Detalle |
|---|---|
| **Tipo de modelo** | Clasificación binaria con Red Neuronal (Keras / TensorFlow) |
| **Variable objetivo (Y)** | `desempeno_bajo`: 1 = puntaje global < promedio departamental · 0 = igual o sobre el promedio |
| **Variables predictoras (X)** | estrato, area_colegio, educacion_madre, educacion_padre, tiene_internet, tiene_computador, naturaleza_colegio, genero, municipio |
| **Modelo seleccionado** | Modelo 2 con Dropout: Input → Dense(64, ReLU) → Dropout(30%) → Dense(32, ReLU) → Dropout(30%) → Dense(16, ReLU) → Output(sigmoide) |
| **Optimizador** | Adam (lr = 0.0001) |
| **Métricas del modelo final** | Accuracy: 65.76% · Precision: 66.59% · Recall: 69.20% · F1: 67.87% · AUC-ROC: 71.48% |
| **Experimento MLflow** | `Clasificacion_DesempenoBajo_Huila` |

**Justificación para la Secretaría:**  
El modelo convierte el índice de vulnerabilidad analizado en el P1 en una herramienta de clasificación de riesgo accionable. Al conocer la probabilidad de que un estudiante quede bajo el promedio, la Secretaría puede priorizar intervenciones en municipios y perfiles con mayor concentración de riesgo, maximizando el impacto de los recursos disponibles.

**Conexión con el Proyecto 1:**  
Formaliza y extiende el análisis de vulnerabilidad por municipio del P1 (diagrama de burbujas), transformando el índice descriptivo en un clasificador predictivo respaldado por una red neuronal.

---

## Complementariedad de los tres modelos

Los tres modelos se complementan para ofrecer una visión integral del riesgo educativo:

1. **Pregunta 1 (Nicolás):** *¿Cuánto se espera que rinda el estudiante?* → Estimación cuantitativa del puntaje
2. **Pregunta 2 (Gabriel):** *¿Está en riesgo de quedar bajo el promedio?* → Clasificación de riesgo académico
3. **Pregunta 3 (Sebastián):** *¿Pertenece a una jornada vulnerable?* → Identificación de perfiles estructuralmente desventajados

Juntos permiten a la Secretaría priorizar intervenciones por perfil de riesgo, municipio y tipo de jornada, cubriendo las tres dimensiones de inequidad identificadas en el Proyecto 1.
