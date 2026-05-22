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

# ── Modelo clasificación 1 (Pregunta 2 - Gabriel) ─────────────────────────────
# TODO: descomentar cuando el modelo esté listo
# modelo_clasificacion_1 = load_model('modelos/modelo_clasificacion_1.keras')
# scaler_X_clas1         = joblib.load('modelos/scaler_X_clasificacion_1.pkl')

# ── Modelo clasificación 2 (Pregunta 3 - Sebastián) ───────────────────────────
# TODO: descomentar cuando el modelo esté listo
# modelo_clasificacion_2 = load_model('modelos/modelo_clasificacion_2.keras')
# scaler_X_clas2         = joblib.load('modelos/scaler_X_clasificacion_2.pkl')

# ── Constantes del departamento (para contextualizar resultados) ───────────────
PROMEDIO_HUILA    = 0  # Pendiente actualizar con el valor real
MUNICIPIOS_COUNT  = 37 # Pendiente actualizar con el valor real
ESTUDIANTES_COUNT = 0  # Pendiente actualizar con el valor real

# =============================================================================
# 2. OPCIONES DE LOS FORMULARIOS
# =============================================================================

opciones_estrato = [
    {'label': 'Sin Estrato', 'value': 0},
    {'label': 'Estrato 1',   'value': 1},
    {'label': 'Estrato 2',   'value': 2},
    {'label': 'Estrato 3',   'value': 3},
    {'label': 'Estrato 4',   'value': 4},
    {'label': 'Estrato 5',   'value': 5},
    {'label': 'Estrato 6',   'value': 6},
]

opciones_educacion = [
    {'label': 'Ninguno',                           'value': 0},
    {'label': 'Primaria incompleta',               'value': 1},
    {'label': 'Primaria completa',                 'value': 2},
    {'label': 'Secundaria incompleta',             'value': 3},
    {'label': 'Secundaria completa',               'value': 4},
    {'label': 'Técnica o tecnológica incompleta',  'value': 5},
    {'label': 'Técnica o tecnológica completa',    'value': 6},
    {'label': 'Profesional incompleta',            'value': 7},
    {'label': 'Profesional completa',              'value': 8},
    {'label': 'Postgrado',                         'value': 9},
]

opciones_zona = [
    {'label': 'Urbano', 'value': 1},
    {'label': 'Rural',  'value': 0},
]

opciones_internet = [
    {'label': 'Sí', 'value': 1},
    {'label': 'No', 'value': 0},
]

opciones_computador = [
    {'label': 'Sí', 'value': 1},
    {'label': 'No', 'value': 0},
]

opciones_naturaleza = [
    {'label': 'Oficial',     'value': 0},
    {'label': 'No Oficial',  'value': 1},
]

opciones_genero = [
    {'label': 'Masculino',  'value': 1},
    {'label': 'Femenino',   'value': 0},
]

opciones_jornada = [
    {'label': 'Completa',  'value': 'COMPLETA'},
    {'label': 'Mañana',    'value': 'MAÑANA'},
    {'label': 'Tarde',     'value': 'TARDE'},
    {'label': 'Única',     'value': 'UNICA'},
    {'label': 'Nocturna',  'value': 'NOCHE'},
    {'label': 'Sabatina',  'value': 'SABATINA'},
]

opciones_municipio = [
    {'label': mun, 'value': val}
    for mun, val in sorted(municipio_encoder.items(), key=lambda x: x[0])
]

# =============================================================================
# 3. ESTILOS
# =============================================================================

ESTILO_FONDO     = '#121212'
ESTILO_TARJETA   = '#1e1e1e'
ESTILO_BORDE     = '#333333'
COLOR_ACENTO     = '#00adb5'
COLOR_TEXTO      = '#ffffff'
COLOR_SUBTEXTO   = '#aaaaaa'

estilo_label = {
    'color': COLOR_SUBTEXTO,
    'fontSize': '13px',
    'marginBottom': '4px',
    'display': 'block'
}

estilo_dropdown = {
    'backgroundColor': '#2a2a2a',
    'color': COLOR_TEXTO,
    'borderColor': ESTILO_BORDE,
    'marginBottom': '14px'
}

estilo_tarjeta = {
    'backgroundColor': ESTILO_TARJETA,
    'border': f'1px solid {ESTILO_BORDE}',
    'borderRadius': '10px',
    'padding': '24px',
}

# =============================================================================
# 4. FUNCIONES DE COMPONENTES
# =============================================================================

def kpi_card(valor, etiqueta):
    """Genera una tarjeta KPI con valor destacado y etiqueta descriptiva."""
    return html.Div([
        html.H2(valor, style={'color': COLOR_ACENTO, 'margin': '0', 'fontSize': '32px'}),
        html.P(etiqueta, style={'color': COLOR_SUBTEXTO, 'margin': '4px 0 0 0', 'fontSize': '13px'})
    ], style={
        **estilo_tarjeta,
        'width': '28%',
        'display': 'inline-block',
        'textAlign': 'center',
        'margin': '0 2%'
    })


def seccion_pregunta(texto):
    """Genera el bloque de texto con la pregunta de negocio."""
    return html.Div([
        html.P('Pregunta de negocio', style={'color': COLOR_ACENTO, 'fontSize': '12px',
                                             'textTransform': 'uppercase', 'margin': '0 0 6px 0'}),
        html.P(texto, style={'fontSize': '16px', 'color': COLOR_TEXTO, 'margin': 0})
    ], style={**estilo_tarjeta, 'marginBottom': '20px'})


def instrucciones(texto):
    """Genera un bloque de instrucciones de uso."""
    return html.Div([
        html.Span('ℹ️  ', style={'fontSize': '16px'}),
        html.Span(texto, style={'color': COLOR_SUBTEXTO, 'fontSize': '14px'})
    ], style={'marginBottom': '20px', 'padding': '12px 16px',
              'backgroundColor': '#1a2a2a', 'borderRadius': '8px',
              'borderLeft': f'3px solid {COLOR_ACENTO}'})


def placeholder_modelo(nombre_responsable, pregunta):
    """Muestra un placeholder para modelos aún no integrados."""
    return html.Div([
        html.Div('⏳', style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '16px'}),
        html.H3(f'Modelo en desarrollo', style={'textAlign': 'center', 'color': COLOR_ACENTO}),
        html.P(pregunta, style={'textAlign': 'center', 'color': COLOR_SUBTEXTO,
                                'maxWidth': '600px', 'margin': '0 auto 16px auto'}),
        html.P(f'Responsable: {nombre_responsable}',
               style={'textAlign': 'center', 'color': COLOR_SUBTEXTO, 'fontSize': '13px'})
    ], style={**estilo_tarjeta, 'padding': '60px 24px'})