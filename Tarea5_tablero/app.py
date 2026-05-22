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