# =============================================================================
# PROYECTO 2 - TABLERO DE MODELOS PREDICTIVOS SABER 11 - HUILA
# Juan Nicolás Hidalgo Parra / Juan Sebastián Méndez Martínez
# =============================================================================

# ── Librerías ──────────────────────────────────────────────────────────────────
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

# =============================================================================
# 1. CARGA DE MODELOS Y TRANSFORMADORES
# Todos los modelos se cargan al inicio para no repetir la carga en cada
# predicción. 
# 
# Los archivos deben estar en la carpeta modelos/
# =============================================================================

# ── Modelo de regresión (Pregunta 1 - Nicolás) ────────────────────────────────
modelo_regresion   = load_model('modelos/modelo_regresion.keras')
scaler_X_reg       = joblib.load('modelos/scaler_X_regresion.pkl')
scaler_y_reg       = joblib.load('modelos/scaler_y_regresion.pkl')
municipio_encoder  = joblib.load('modelos/municipio_encoder.pkl')