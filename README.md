# Proyecto 2 — Modelos Predictivos Saber 11 · Huila

**Analítica Computacional para la Toma de Decisiones – IIND4130**  
Universidad de los Andes · Profesor Juan Fernando Pérez · 2026-10

---

## Cliente

**Secretaría de Educación del Huila**

**Objetivo:** Desarrollar modelos predictivos basados en redes neuronales que permitan anticipar el desempeño académico de estudiantes del Huila en las pruebas Saber 11, con el fin de orientar la inversión de recursos educativos de forma focalizada y oportuna.

---

## Equipo

| Nombre | Código | Pregunta | Rol adicional |
|---|---|---|---|
| Juan Nicolás Hidalgo Parra | 202122726 | Pregunta 1 — Regresión puntaje global | Tablero de datos |
| Juan Sebastián Méndez Martínez | 202222395 | Pregunta 2 — Clasificación jornada vulnerable | Tablero de datos |
| Gabriel Juan De Dios | 202220936 | Pregunta 3 — Clasificación riesgo académico | Despliegue y mantenimiento |

---

## Preguntas de Negocio

| # | Pregunta | Tipo | Modelo seleccionado | Métrica principal |
|---|---|---|---|---|
| 1 | ¿Cuál es el puntaje global esperado dado el perfil socioeconómico del estudiante? | Regresión | Red neuronal base (Dense 32→16→lineal) | RMSE: 40.45 · R²: 0.27 |
| 2 | ¿El estudiante asiste a una jornada no convencional (nocturna/sabatina)? | Clasificación binaria | Red neuronal con Dropout | AUC-ROC: 0.8369 |
| 3 | ¿El estudiante quedará por debajo del promedio departamental? | Clasificación binaria | Red neuronal con Dropout | F1: 67.87% · AUC: 71.48% |

---

## Estructura del Repositorio

```
Proyecto_2/
│
├── Tarea1_negocio/
│   └── preguntas_negocio.md          # Definición de preguntas, modelos y justificación
│
├── Tarea2_datos/
│   ├── Datos_Huila.csv               # Dataset original filtrado para el Huila
│   └── Datos_Huila_Limpio.csv        # Dataset depurado y listo para análisis
│
├── Tarea3_exploracion/
│   └── Analisis_Huila.ipynb          # Análisis exploratorio de datos (reutilizado del P1)
│
├── Tarea4_modelos/
│   ├── Modelo_Regresion_Nicolas_P2.ipynb        # Pregunta 1 — Nicolás
│   ├── Modelo_ClasificacionSebastian_P2_.ipynb  # Pregunta 2 — Sebastián
│   ├── Modelo_Clasificacion_GabrielJD.ipynb     # Pregunta 3 — Gabriel
│   └── mlruns/                                  # Experimentos registrados en MLflow
│
├── Tarea5_tablero/
│   ├── app.py                        # Tablero Dash con los 3 modelos integrados
│   ├── Dockerfile                    # Imagen Docker para despliegue
│   ├── requirements.txt              # Dependencias del proyecto
│   └── modelos/                      # Modelos serializados (.keras y .pkl)
│       ├── modelo_regresion.keras
│       ├── scaler_X_regresion.pkl
│       ├── scaler_y_regresion.pkl
│       ├── municipio_encoder.pkl
│       ├── modelo_clasificacion_1.keras
│       ├── scaler_X_clasificacion_1.pkl
│       ├── modelo_clasificacion_2.keras
│       └── scaler_X_clasificacion_2.pkl
│
└── despliegue/
    ├── app.py                        # Última versión del tablero lista para producción
    ├── Dockerfile
    ├── requirements.txt
    └── modelos/                      # Modelos serializados para producción
```

---

## Stack Tecnológico

| Categoría | Tecnología |
|---|---|
| Lenguaje | Python 3.11 |
| Modelos | TensorFlow 2.16.1 / Keras |
| Tablero | Dash · Plotly |
| Preprocesamiento | scikit-learn · pandas · numpy |
| Experimentos | MLflow |
| Servidor | Gunicorn |
| Infraestructura | AWS EC2 t2.medium · Docker · Ubuntu Server 24.04 LTS |

---

## Cómo ejecutar el tablero localmente

**1. Clonar el repositorio:**
```bash
git clone https://github.com/nicohidalgo19/Proyecto_2.git
cd Proyecto_2/Tarea5_tablero
```

**2. Instalar dependencias:**
```bash
pip install -r requirements.txt
```

**3. Ejecutar el tablero:**
```bash
python app.py
```

**4. Abrir en el navegador:**
```
http://127.0.0.1:8050
```

> **Nota:** Los archivos de modelos serializados deben estar presentes en la carpeta `modelos/` antes de ejecutar. Los modelos no están incluidos en el repositorio por su tamaño.

---

## Despliegue con Docker en AWS

**1. Construir la imagen:**
```bash
sudo docker build -t mi-dash-app:latest .
```

**2. Ejecutar el contenedor:**
```bash
sudo docker run -d -p 8050:8050 --name dash-app mi-dash-app:latest
```

**3. Verificar logs:**
```bash
sudo docker logs dash-app
```

**Configuración de la instancia EC2:**
- AMI: Ubuntu Server 24.04 LTS (HVM) SSD
- Tipo: t2.medium
- Almacenamiento: 30 GB
- Puertos abiertos: 22 (SSH) y 8050 (Custom TCP)

---

## Fuente de Datos

Los datos provienen del portal de Datos Abiertos del gobierno colombiano:  
[Resultados Únicos Saber 11 — datos.gov.co](https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe)

El subconjunto utilizado corresponde al departamento del Huila, extraído mediante AWS Glue y AWS Athena en el Proyecto 1 del curso, con 19 variables seleccionadas y aproximadamente 100.000 registros.

---

## Experimentos MLflow

Los experimentos de cada modelo están registrados en la carpeta `Tarea4_modelos/mlruns/`. Para visualizarlos:

```bash
cd Tarea4_modelos
mlflow ui
```

Abrir en el navegador: `http://127.0.0.1:5000`

Experimentos disponibles:
- `Regresion_PuntajeGlobal_Huila` — Pregunta 1
- `Clasificacion_Jornada_No_Convencional_Huila` — Pregunta 2
- `Clasificacion_DesempenoBajo_Huila` — Pregunta 3

---

## Repositorio del Proyecto 1

Este proyecto es la continuación del Proyecto 1, cuyo repositorio se encuentra en:  
[ANALITICA_DE_DATOS_Proyecto_1_Huila](https://github.com/nicohidalgo19/ANALITICA_DE_DATOS_Proyecto_1_Huila.git)
