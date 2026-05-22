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

ESTILO_FONDO  = '#0d0f1a'
ESTILO_TARJETA = '#111827'
ESTILO_BORDE     = '#0f3460'
COLOR_ACENTO     = '#00d4ff'
COLOR_TEXTO      = '#e0e0e0'
COLOR_SUBTEXTO   = '#9aa5b4'

estilo_label = {
    'color': COLOR_SUBTEXTO,
    'fontSize': '12px',
    'marginBottom': '3px',
    'display': 'block',
    'fontWeight': '500'
}

estilo_dropdown = {
    'marginBottom': '12px',
    'backgroundColor': '#0f3460',
}

estilo_tarjeta = {
    'backgroundColor': ESTILO_TARJETA,
    'border': f'1px solid {ESTILO_BORDE}',
    'borderRadius': '12px',
    'padding': '24px',
}

# =============================================================================
# 4. FUNCIONES DE COMPONENTES
# =============================================================================

def kpi_card(valor, etiqueta):
    return html.Div([
        html.P(valor, style={
            'color': COLOR_ACENTO,
            'margin': '0',
            'fontSize': '22px',
            'fontWeight': 'bold'
        }),
        html.P(etiqueta, style={
            'color': COLOR_SUBTEXTO,
            'margin': '2px 0 0 0',
            'fontSize': '11px',
            'textTransform': 'uppercase',
            'letterSpacing': '0.5px'
        })
    ], style={
        **estilo_tarjeta,
        'width': '22%',
        'display': 'inline-block',
        'textAlign': 'center',
        'margin': '0 1.5%',
        'padding': '14px 20px',
    })


def seccion_pregunta(texto):
    """Genera el bloque de texto con la pregunta de negocio."""
    return html.Div([
        html.P('Pregunta de negocio', style={'color': COLOR_ACENTO, 'fontSize': '12px',
                                             'textTransform': 'uppercase', 'margin': '0 0 6px 0'}),
        html.P(texto, style={'fontSize': '16px', 'color': COLOR_TEXTO, 'margin': 0})
    ], style={**estilo_tarjeta, 'marginBottom': '20px'})


def instrucciones(texto):
    return html.Div([
        html.Span('ℹ️  ', style={'fontSize': '16px'}),
        html.Span(texto, style={'color': '#ffffff', 'fontSize': '14px'})
    ], style={'marginBottom': '20px', 'padding': '12px 16px',
              'backgroundColor': '#1a2a3a', 'borderRadius': '8px',
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

# =============================================================================
# 5. LAYOUT DEL TABLERO
# =============================================================================

app = dash.Dash(__name__, title='Modelos Predictivos Saber 11 - Huila')

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .Select-control {
                background-color: #1a2340 !important;
                border-color: #00d4ff !important;
                color: #ffffff !important;
            }
            .Select-value-label {
                color: #ffffff !important;
                font-size: 14px !important;
                font-weight: 500 !important;
            }
            .Select-placeholder {
                color: #ffffff !important;
                font-size: 14px !important;
            }
            .Select-input input {
                color: #ffffff !important;
            }
            .Select-single-value {
                color: #ffffff !important;
            }
            .Select-menu-outer {
                background-color: #111827 !important;
                border-color: #00d4ff !important;
                z-index: 9999 !important;
            }
            .Select-option {
                background-color: #111827 !important;
                color: #ffffff !important;
                font-size: 13px !important;
            }
            .Select-option:hover, .Select-option.is-focused {
                background-color: #1a2340 !important;
                color: #00d4ff !important;
            }
            .Select-option.is-selected {
                background-color: #00d4ff !important;
                color: #000000 !important;
            }
            .Select-arrow {
                border-color: #00d4ff transparent transparent !important;
            }
            .dash-tab {
                background-color: #16213e !important;
                color: #9aa5b4 !important;
                border: 1px solid #0f3460 !important;
            }
            .dash-tab--selected {
                background-color: #0f3460 !important;
                color: #00d4ff !important;
                border-bottom: 2px solid #00d4ff !important;
            }
            body {
                background-color: #1a1a2e !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([

    # ── Encabezado ──────────────────────────────────────────────────────────────
    html.Div([
        html.H1('Modelos Predictivos Saber 11 — Huila',
                style={'color': COLOR_TEXTO, 'fontSize': '22px',
                    'margin': '0 0 4px 0', 'fontWeight': 'bold',
                    'textAlign': 'center'}),
        html.P('Secretaría de Educación del Huila · Proyecto 2 · 2026',
            style={'color': COLOR_SUBTEXTO, 'fontSize': '12px',
                    'margin': '0 0 6px 0', 'textAlign': 'center'}),
        html.P(id='reloj',
            style={'color': COLOR_ACENTO, 'fontSize': '12px',
                    'margin': 0, 'textAlign': 'center'}),
    ], style={
        'marginBottom': '24px',
        'paddingBottom': '16px',
        'borderBottom': f'1px solid {ESTILO_BORDE}'
    }),

    dcc.Interval(id='intervalo-reloj', interval=1000, n_intervals=0),

    # ── KPIs ────────────────────────────────────────────────────────────────────
    html.Div([
        kpi_card(f'{PROMEDIO_HUILA:.1f} pts', 'Promedio Global Huila'),
        kpi_card(str(MUNICIPIOS_COUNT),        'Municipios Analizados'),
        kpi_card(f'{ESTUDIANTES_COUNT:,}',     'Estudiantes en el Dataset'),
    ], style={'textAlign': 'center', 'marginBottom': '30px'}),

    # ── Pestañas ────────────────────────────────────────────────────────────────
    dcc.Tabs(id='tabs-principales', value='tab-1', children=[

        # ══════════════════════════════════════════════════════════════════════
        # PESTAÑA 1 — PREGUNTA 1 (Nicolás): Regresión puntaje global
        # ══════════════════════════════════════════════════════════════════════
        dcc.Tab(label='Pregunta 1 — Puntaje Global', value='tab-1', children=[
            html.Div([

                seccion_pregunta(
                    '¿Cuál es el puntaje global esperado de un estudiante dado su estrato '
                    'socioeconómico, zona geográfica, nivel educativo de los padres y jornada escolar?'
                ),

                instrucciones(
                    'Complete el perfil socioeconómico e institucional del estudiante y presione '
                    '"Predecir". El modelo estimará el puntaje global esperado en la prueba Saber 11 '
                    'con base en variables de contexto.'
                ),

                # ── Formulario + Resultado ──────────────────────────────────
                html.Div([

                # Formulario de entrada
                html.Div([
                    html.H4('Perfil del Estudiante',
                            style={'color': COLOR_ACENTO, 'marginBottom': '20px',
                                'fontSize': '13px', 'textTransform': 'uppercase',
                                'letterSpacing': '1px'}),

                    html.Label('Estrato socioeconómico', style=estilo_label),
                    dcc.Dropdown(id='reg-estrato', options=opciones_estrato,
                                value=1, clearable=False, style=estilo_dropdown),

                    html.Label('Zona del colegio', style=estilo_label),
                    dcc.Dropdown(id='reg-zona', options=opciones_zona,
                                value=1, clearable=False, style=estilo_dropdown),

                    html.Label('Nivel educativo de la madre', style=estilo_label),
                    dcc.Dropdown(id='reg-edu-madre', options=opciones_educacion,
                                value=4, clearable=False, style=estilo_dropdown),

                    html.Label('Nivel educativo del padre', style=estilo_label),
                    dcc.Dropdown(id='reg-edu-padre', options=opciones_educacion,
                                value=4, clearable=False, style=estilo_dropdown),

                    html.Label('Jornada escolar', style=estilo_label),
                    dcc.Dropdown(id='reg-jornada', options=opciones_jornada,
                                value='MAÑANA', clearable=False, style=estilo_dropdown),

                    html.Label('Municipio', style=estilo_label),
                    dcc.Dropdown(id='reg-municipio', options=opciones_municipio,
                                value=list(municipio_encoder.values())[0],
                                clearable=False, style=estilo_dropdown),

                    html.Label('Acceso a internet en el hogar', style=estilo_label),
                    dcc.Dropdown(id='reg-internet', options=opciones_internet,
                                value=1, clearable=False, style=estilo_dropdown),

                    html.Label('Acceso a computador en el hogar', style=estilo_label),
                    dcc.Dropdown(id='reg-computador', options=opciones_computador,
                                value=1, clearable=False, style=estilo_dropdown),

                    html.Label('Naturaleza del colegio', style=estilo_label),
                    dcc.Dropdown(id='reg-naturaleza', options=opciones_naturaleza,
                                value=0, clearable=False, style=estilo_dropdown),

                    html.Label('Género', style=estilo_label),
                    dcc.Dropdown(id='reg-genero', options=opciones_genero,
                                value=1, clearable=False, style=estilo_dropdown),

                    html.Div([
                        html.Button('Predecir', id='btn-predecir-reg', style={
                            'backgroundColor': COLOR_ACENTO,
                            'color': '#000',
                            'border': 'none',
                            'padding': '12px 28px',
                            'borderRadius': '6px',
                            'cursor': 'pointer',
                            'fontWeight': 'bold',
                            'fontSize': '14px',
                            'marginRight': '10px'
                        }),
                        html.Button('Limpiar', id='btn-limpiar-reg', style={
                            'backgroundColor': 'transparent',
                            'color': COLOR_SUBTEXTO,
                            'border': f'1px solid {ESTILO_BORDE}',
                            'padding': '12px 28px',
                            'borderRadius': '6px',
                            'cursor': 'pointer',
                            'fontSize': '14px'
                        }),
                    ], style={'marginTop': '8px'})

                ], style={
                    **estilo_tarjeta,
                    'flex': '0 0 380px',
                    'minWidth': '300px',
                }),

                # Panel de resultado
                html.Div([
                    html.H4('Resultado de la Predicción',
                            style={'color': COLOR_ACENTO, 'marginBottom': '24px',
                                'fontSize': '13px', 'textTransform': 'uppercase',
                                'letterSpacing': '1px'}),

                    html.Div(id='resultado-regresion', children=[
                        html.P('Complete el formulario y presione Predecir.',
                            style={'color': COLOR_SUBTEXTO, 'textAlign': 'center',
                                    'marginTop': '80px', 'fontSize': '14px'})
                    ]),

                    dcc.Graph(id='gauge-regresion',
                            config={'displayModeBar': False},
                            style={'height': '250px', 'marginTop': '10px'}),

                    html.Div(id='interpretacion-regresion', style={'marginTop': '16px'})

                ], style={
                    **estilo_tarjeta,
                    'flex': '1',
                    'minWidth': '300px',
                })

            ], style={
                'display': 'flex',
                'flexDirection': 'row',
                'gap': '20px',
                'alignItems': 'flex-start',
                'flexWrap': 'wrap'
            })

            ], style={'padding': '20px'})
        ]),

        # ══════════════════════════════════════════════════════════════════════
        # PESTAÑA 2 — PREGUNTA 2 (Gabriel): Clasificación riesgo académico
        # ══════════════════════════════════════════════════════════════════════
        dcc.Tab(label='Pregunta 2 — Riesgo Académico', value='tab-2', children=[
            html.Div([
                seccion_pregunta(
                    '¿Un estudiante estará por debajo del promedio departamental dado su '
                    'perfil socioeconómico y municipio?'
                ),
                placeholder_modelo(
                    'Gabriel Juan De Dios',
                    'Modelo de clasificación binaria: bajo el promedio departamental / sobre el promedio.'
                )
            ], style={'padding': '20px'})
        ]),

        # ══════════════════════════════════════════════════════════════════════
        # PESTAÑA 3 — PREGUNTA 3 (Sebastián): Clasificación jornada vulnerable
        # ══════════════════════════════════════════════════════════════════════
        dcc.Tab(label='Pregunta 3 — Jornada Vulnerable', value='tab-3', children=[
            html.Div([
                seccion_pregunta(
                    '¿Es posible predecir si un estudiante asiste a jornada nocturna o sabatina '
                    'a partir de su perfil socioeconómico y municipio?'
                ),
                placeholder_modelo(
                    'Juan Sebastián Méndez Martínez',
                    'Modelo de clasificación binaria: jornada regular / jornada nocturna o sabatina.'
                )
            ], style={'padding': '20px'})
        ]),

    ], colors={
        'background': ESTILO_FONDO,
        'border': ESTILO_BORDE,
        'primary': COLOR_ACENTO
    })

], style={
    'backgroundColor': ESTILO_FONDO,
    'color': COLOR_TEXTO,
    'minHeight': '100vh',
    'padding': '30px',
    'fontFamily': 'Arial, sans-serif'
})

# =============================================================================
# 6. CALLBACKS
# =============================================================================

@app.callback(
    Output('resultado-regresion',     'children'),
    Output('gauge-regresion',         'figure'),
    Output('interpretacion-regresion','children'),
    Output('reg-estrato',             'value'),
    Output('reg-zona',                'value'),
    Output('reg-edu-madre',           'value'),
    Output('reg-edu-padre',           'value'),
    Output('reg-jornada',             'value'),
    Output('reg-municipio',           'value'),
    Output('reg-internet',            'value'),
    Output('reg-computador',          'value'),
    Output('reg-naturaleza',          'value'),
    Output('reg-genero',              'value'),
    Input('btn-predecir-reg',         'n_clicks'),
    Input('btn-limpiar-reg',          'n_clicks'),
    State('reg-estrato',              'value'),
    State('reg-zona',                 'value'),
    State('reg-edu-madre',            'value'),
    State('reg-edu-padre',            'value'),
    State('reg-jornada',              'value'),
    State('reg-municipio',            'value'),
    State('reg-internet',             'value'),
    State('reg-computador',           'value'),
    State('reg-naturaleza',           'value'),
    State('reg-genero',               'value'),
    prevent_initial_call=True
)
def manejar_formulario(n_predecir, n_limpiar,
                       estrato, zona, edu_madre, edu_padre,
                       jornada, municipio, internet, computador,
                       naturaleza, genero):

    # Valores por defecto del formulario
    defaults = (1, 1, 4, 4, 'MAÑANA',
                list(municipio_encoder.values())[0], 1, 1, 0, 1)

    # Figura vacía para el gauge cuando no hay predicción
    figura_vacia = go.Figure()
    figura_vacia.update_layout(
        paper_bgcolor=ESTILO_TARJETA,
        plot_bgcolor=ESTILO_TARJETA,
        margin=dict(l=20, r=20, t=20, b=20),
        height=220,
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[{
            'text': 'El resultado aparecerá aquí',
            'xref': 'paper', 'yref': 'paper',
            'x': 0.5, 'y': 0.5,
            'showarrow': False,
            'font': {'color': COLOR_SUBTEXTO, 'size': 13}
        }]
    )

    mensaje_vacio = html.P(
        'Complete el formulario y presione Predecir.',
        style={'color': COLOR_SUBTEXTO, 'textAlign': 'center',
               'marginTop': '80px', 'fontSize': '14px'}
    )

    # ── Si se presionó Limpiar ─────────────────────────────────────────────────
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'btn-limpiar-reg.n_clicks':
        return (mensaje_vacio, figura_vacia, html.Div(), *defaults)

    # ── Si se presionó Predecir ────────────────────────────────────────────────
    jornada_mañana   = 1 if jornada == 'MAÑANA'   else 0
    jornada_noche    = 1 if jornada == 'NOCHE'     else 0
    jornada_sabatina = 1 if jornada == 'SABATINA'  else 0
    jornada_tarde    = 1 if jornada == 'TARDE'     else 0
    jornada_unica    = 1 if jornada == 'UNICA'     else 0

    features = np.array([[
        estrato, zona, edu_madre, edu_padre, municipio,
        internet, computador, naturaleza, genero,
        jornada_mañana, jornada_noche, jornada_sabatina,
        jornada_tarde, jornada_unica
    ]])

    features_scaled = scaler_X_reg.transform(features)
    pred_scaled     = modelo_regresion.predict(features_scaled, verbose=0).flatten()
    pred_real       = scaler_y_reg.inverse_transform(
                          pred_scaled.reshape(-1, 1)
                      ).flatten()[0]
    pred_real = round(float(pred_real), 1)

    diferencia = pred_real - PROMEDIO_HUILA
    color_dif  = '#00c48c' if diferencia >= 0 else '#ff6b6b'
    signo      = '+' if diferencia >= 0 else ''

    resultado_div = html.Div([
        html.P('Puntaje Global Estimado',
               style={'color': COLOR_SUBTEXTO, 'fontSize': '13px',
                      'textAlign': 'center', 'margin': '0 0 8px 0'}),
        html.H2(f'{pred_real} pts',
                style={'color': COLOR_TEXTO, 'fontSize': '48px',
                       'textAlign': 'center', 'margin': '0 0 8px 0'}),
        html.P(f'{signo}{diferencia:.1f} pts vs promedio departamental ({PROMEDIO_HUILA} pts)',
               style={'color': color_dif, 'textAlign': 'center',
                      'fontSize': '14px', 'margin': 0})
    ])

    gauge = go.Figure(go.Indicator(
        mode='gauge+number',
        value=pred_real,
        number={'suffix': ' pts', 'font': {'color': COLOR_TEXTO}},
        gauge={
            'axis': {'range': [100, 500], 'tickcolor': COLOR_SUBTEXTO,
                     'tickfont': {'color': COLOR_SUBTEXTO}},
            'bar': {'color': COLOR_ACENTO},
            'bgcolor': ESTILO_TARJETA,
            'steps': [
                {'range': [100, 200], 'color': '#2a1a1a'},
                {'range': [200, 300], 'color': '#1a2a1a'},
                {'range': [300, 500], 'color': '#1a3a1a'},
            ],
            'threshold': {
                'line': {'color': '#ffcc00', 'width': 3},
                'thickness': 0.75,
                'value': PROMEDIO_HUILA
            }
        }
    ))
    gauge.update_layout(
        paper_bgcolor=ESTILO_TARJETA,
        font={'color': COLOR_TEXTO},
        margin=dict(l=20, r=20, t=20, b=20),
        height=220
    )

    if pred_real >= PROMEDIO_HUILA + 20:
        msg   = 'El estudiante presenta un perfil favorable. Se proyecta un desempeño destacado frente al promedio departamental.'
        color = '#00c48c'
        icono = '✅'
    elif pred_real >= PROMEDIO_HUILA:
        msg   = 'El estudiante se proyecta por encima del promedio departamental. Condiciones socioeconómicas adecuadas para un desempeño satisfactorio.'
        color = COLOR_ACENTO
        icono = '📊'
    elif pred_real >= PROMEDIO_HUILA - 20:
        msg   = 'El estudiante se proyecta ligeramente por debajo del promedio. Se recomienda seguimiento y apoyo pedagógico focalizado.'
        color = '#ffcc00'
        icono = '⚠️'
    else:
        msg   = 'El perfil del estudiante indica alto riesgo de bajo rendimiento. Se recomienda priorizar intervención temprana.'
        color = '#ff6b6b'
        icono = '🔴'

    interpretacion_div = html.Div([
        html.Span(f'{icono}  '),
        html.Span(msg, style={'color': COLOR_SUBTEXTO, 'fontSize': '14px'})
    ], style={'padding': '12px 16px', 'backgroundColor': '#0d0f1a',
              'borderRadius': '8px', 'borderLeft': f'3px solid {color}'})

    # Retorna resultado + valores actuales del formulario sin cambio
    return (resultado_div, gauge, interpretacion_div,
            estrato, zona, edu_madre, edu_padre, jornada,
            municipio, internet, computador, naturaleza, genero)

@app.callback(
    Output('reloj', 'children'),
    Input('intervalo-reloj', 'n_intervals')
)
def actualizar_reloj(n):
    from datetime import datetime
    ahora = datetime.now()
    return ahora.strftime('%A %d de %B de %Y  |  %H:%M:%S')

# =============================================================================
# 7. EJECUCIÓN DEL SERVIDOR
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True)