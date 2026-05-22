# =============================================================================
# PROYECTO 2 - TABLERO DE MODELOS PREDICTIVOS SABER 11 - HUILA
# Responsable tablero: Juan Nicolás Hidalgo Parra / Juan Sebastián Méndez Martínez
# =============================================================================

import os
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

# =============================================================================
# 1. CARGA DE MODELOS Y TRANSFORMADORES
# =============================================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELOS_DIR = os.path.join(BASE_DIR, 'modelos')

# Pregunta Nicolás
modelo_regresion  = load_model(os.path.join(MODELOS_DIR, 'modelo_regresion.keras'))
scaler_X_reg      = joblib.load(os.path.join(MODELOS_DIR, 'scaler_X_regresion.pkl'))
scaler_y_reg      = joblib.load(os.path.join(MODELOS_DIR, 'scaler_y_regresion.pkl'))
municipio_encoder = joblib.load(os.path.join(MODELOS_DIR, 'municipio_encoder.pkl'))

# TODO: descomentar cuando los modelos estén listos

# Pregunta Gabriel
# modelo_clasificacion_1 = load_model(os.path.join(MODELOS_DIR, 'modelo_clasificacion_1.keras'))
# scaler_X_clas1         = joblib.load(os.path.join(MODELOS_DIR, 'scaler_X_clasificacion_1.pkl'))

# Pregunta Sebastián
# modelo_clasificacion_2 = load_model(os.path.join(MODELOS_DIR, 'modelo_clasificacion_2.keras'))
# scaler_X_clas2         = joblib.load(os.path.join(MODELOS_DIR, 'scaler_X_clasificacion_2.pkl'))

# Actualizar con valores reales del dataset
PROMEDIO_HUILA    = 250.0
MUNICIPIOS_COUNT  = 37
ESTUDIANTES_COUNT = 80283

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
    {'label': 'Ninguno',                          'value': 0},
    {'label': 'Primaria incompleta',              'value': 1},
    {'label': 'Primaria completa',                'value': 2},
    {'label': 'Secundaria incompleta',            'value': 3},
    {'label': 'Secundaria completa',              'value': 4},
    {'label': 'Técnica o tecnológica incompleta', 'value': 5},
    {'label': 'Técnica o tecnológica completa',   'value': 6},
    {'label': 'Profesional incompleta',           'value': 7},
    {'label': 'Profesional completa',             'value': 8},
    {'label': 'Postgrado',                        'value': 9},
]

opciones_zona        = [{'label': 'Urbano', 'value': 1}, {'label': 'Rural',      'value': 0}]
opciones_internet    = [{'label': 'Sí',     'value': 1}, {'label': 'No',         'value': 0}]
opciones_computador  = [{'label': 'Sí',     'value': 1}, {'label': 'No',         'value': 0}]
opciones_naturaleza  = [{'label': 'Oficial','value': 0}, {'label': 'No Oficial', 'value': 1}]
opciones_genero      = [{'label': 'Masculino', 'value': 1}, {'label': 'Femenino','value': 0}]

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

ESTILO_FONDO   = '#0d0f1a'
ESTILO_TARJETA = '#111827'
ESTILO_BORDE   = '#0f3460'
COLOR_ACENTO   = '#00d4ff'
COLOR_TEXTO    = '#e0e0e0'
COLOR_SUBTEXTO = '#9aa5b4'

estilo_label = {
    'color': COLOR_SUBTEXTO,
    'fontSize': '12px',
    'marginBottom': '3px',
    'display': 'block',
    'fontWeight': '500'
}

estilo_tarjeta = {
    'backgroundColor': ESTILO_TARJETA,
    'border': f'1px solid {ESTILO_BORDE}',
    'borderRadius': '12px',
    'padding': '24px',
}

# Estilo base de todos los dropdowns
dd = {'marginBottom': '12px'}

# =============================================================================
# 4. COMPONENTES REUTILIZABLES
# =============================================================================

def kpi_card(valor, etiqueta):
    return html.Div([
        html.P(valor, style={'color': COLOR_ACENTO, 'margin': '0',
                             'fontSize': '22px', 'fontWeight': 'bold'}),
        html.P(etiqueta, style={'color': COLOR_SUBTEXTO, 'margin': '2px 0 0 0',
                                'fontSize': '11px', 'textTransform': 'uppercase',
                                'letterSpacing': '0.5px'})
    ], style={**estilo_tarjeta, 'width': '22%', 'display': 'inline-block',
              'textAlign': 'center', 'margin': '0 1.5%', 'padding': '14px 20px'})


def seccion_pregunta(texto):
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
    return html.Div([
        html.Div('⏳', style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '16px'}),
        html.H3('Modelo en desarrollo', style={'textAlign': 'center', 'color': COLOR_ACENTO}),
        html.P(pregunta, style={'textAlign': 'center', 'color': COLOR_SUBTEXTO,
                                'maxWidth': '600px', 'margin': '0 auto 16px auto'}),
        html.P(f'Responsable: {nombre_responsable}',
               style={'textAlign': 'center', 'color': COLOR_SUBTEXTO, 'fontSize': '13px'})
    ], style={**estilo_tarjeta, 'padding': '60px 24px'})


def dropdown(id_, opciones, valor):
    return dcc.Dropdown(
        id=id_,
        options=opciones,
        value=valor,
        clearable=False,
        style={
            'marginBottom': '12px',
            'backgroundColor': '#1a2340',
            'border': '1px solid #00d4ff',
            'borderRadius': '6px',
        },
        className='custom-dd'
    )

# =============================================================================
# 5. APP Y CSS
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
            body { background-color: #0d0f1a !important; font-family: Arial, sans-serif; }

            /* ── Dropdown Dash 4.x (clases reales) ── */
            .dash-dropdown-trigger {
                background-color: #1a2340 !important;
                border: 1px solid #00d4ff !important;
                border-radius: 6px !important;
                min-height: 38px !important;
            }
            .dash-dropdown-value,
            .dash-dropdown-value-item {
                color: #ffffff !important;
                font-size: 13px !important;
            }
            .dash-dropdown-placeholder {
                color: #aaaaaa !important;
                font-size: 13px !important;
            }
            .dash-dropdown-trigger-icon,
            .dash-dropdown-search-icon {
                color: #00d4ff !important;
                fill: #00d4ff !important;
            }
            .dash-dropdown-content,
            .dash-dropdown-options {
                background-color: #111827 !important;
                border: 1px solid #00d4ff !important;
                border-radius: 6px !important;
            }
            .dash-dropdown-option {
                background-color: #111827 !important;
                color: #ffffff !important;
                font-size: 13px !important;
            }
            .dash-dropdown-option:hover {
                background-color: #1a2340 !important;
                color: #00d4ff !important;
            }
            .dash-dropdown-search,
            .dash-dropdown-search-container {
                background-color: #111827 !important;
                color: #ffffff !important;
            }
            .dash-dropdown-search input {
                color: #ffffff !important;
                background-color: #1a2340 !important;
            }
            .dash-dropdown-clear {
                color: #00d4ff !important;
            }

            /* ── Tabs ── */
            .dash-tab { background-color: #111827 !important; color: #9aa5b4 !important; border: 1px solid #0f3460 !important; }
            .dash-tab--selected { background-color: #0f3460 !important; color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }
        </style>
        <script>
        function fixDropdowns() {
            document.querySelectorAll('.custom-dd').forEach(function(dd) {
                dd.querySelectorAll('div').forEach(function(el) {
                    var cn = el.className || '';
                    if (typeof cn !== 'string') return;
                    // Fondo del control
                    if (cn.match(/control|ValueContainer|container/i)) {
                        el.style.setProperty('background-color', '#1a2340', 'important');
                        el.style.setProperty('border-color', '#00d4ff', 'important');
                    }
                    // Color del texto seleccionado
                    if (cn.match(/singleValue|placeholder|value/i)) {
                        el.style.setProperty('color', '#ffffff', 'important');
                        el.style.setProperty('font-size', '13px', 'important');
                    }
                    // Input de búsqueda
                    if (cn.match(/Input/)) {
                        el.style.setProperty('color', '#ffffff', 'important');
                    }
                    // Menú desplegable
                    if (cn.match(/menu|MenuList|option/i)) {
                        el.style.setProperty('background-color', '#111827', 'important');
                        el.style.setProperty('color', '#ffffff', 'important');
                    }
                });
                // Input directo
                dd.querySelectorAll('input').forEach(function(inp) {
                    inp.style.setProperty('color', '#ffffff', 'important');
                });
                // SVG de la flecha
                dd.querySelectorAll('svg').forEach(function(svg) {
                    svg.style.setProperty('fill', '#00d4ff', 'important');
                });
            });
        }
        // Corre cada 150ms para capturar cambios de React
        setInterval(fixDropdowns, 150);
        </script>
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

# =============================================================================
# 6. LAYOUT
# =============================================================================

app.layout = html.Div([

    # ── Encabezado ──────────────────────────────────────────────────────────────
    html.Div([
        html.H1('Modelos Predictivos Saber 11 — Huila',
                style={'color': COLOR_TEXTO, 'fontSize': '32px',
                       'margin': '0 0 6px 0', 'fontWeight': 'bold',
                       'textAlign': 'center'}),
        html.P('Secretaría de Educación del Huila · Proyecto 2 · 2026',
               style={'color': COLOR_SUBTEXTO, 'fontSize': '19px',
                      'margin': '0 0 6px 0', 'textAlign': 'center'}),
        html.P(id='reloj', style={'color': COLOR_ACENTO, 'fontSize': '12px',
                                  'margin': 0, 'textAlign': 'center'}),
    ], style={'marginBottom': '24px', 'paddingBottom': '16px',
              'borderBottom': f'1px solid {ESTILO_BORDE}'}),

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
        # PESTAÑA 1 — Regresión puntaje global (Nicolás)
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

                html.Div([

                    # ── Formulario ─────────────────────────────────────────
                    html.Div([
                        html.H4('Perfil del Estudiante',
                                style={'color': COLOR_ACENTO, 'marginBottom': '20px',
                                       'fontSize': '13px', 'textTransform': 'uppercase',
                                       'letterSpacing': '1px'}),

                        html.Label('Estrato socioeconómico', style=estilo_label),
                        dropdown('reg-estrato', opciones_estrato, 1),

                        html.Label('Zona del colegio', style=estilo_label),
                        dropdown('reg-zona', opciones_zona, 1),

                        html.Label('Nivel educativo de la madre', style=estilo_label),
                        dropdown('reg-edu-madre', opciones_educacion, 4),

                        html.Label('Nivel educativo del padre', style=estilo_label),
                        dropdown('reg-edu-padre', opciones_educacion, 4),

                        html.Label('Jornada escolar', style=estilo_label),
                        dropdown('reg-jornada', opciones_jornada, 'MAÑANA'),

                        html.Label('Municipio', style=estilo_label),
                        dropdown('reg-municipio', opciones_municipio,
                                 list(municipio_encoder.values())[0]),

                        html.Label('Acceso a internet en el hogar', style=estilo_label),
                        dropdown('reg-internet', opciones_internet, 1),

                        html.Label('Acceso a computador en el hogar', style=estilo_label),
                        dropdown('reg-computador', opciones_computador, 1),

                        html.Label('Naturaleza del colegio', style=estilo_label),
                        dropdown('reg-naturaleza', opciones_naturaleza, 0),

                        html.Label('Género', style=estilo_label),
                        dropdown('reg-genero', opciones_genero, 1),

                        html.Div([
                            html.Button('Predecir', id='btn-predecir-reg', style={
                                'backgroundColor': COLOR_ACENTO, 'color': '#000',
                                'border': 'none', 'padding': '12px 28px',
                                'borderRadius': '6px', 'cursor': 'pointer',
                                'fontWeight': 'bold', 'fontSize': '14px',
                                'marginRight': '10px'
                            }),
                            html.Button('Limpiar', id='btn-limpiar-reg', style={
                                'backgroundColor': 'transparent', 'color': COLOR_SUBTEXTO,
                                'border': f'1px solid {ESTILO_BORDE}', 'padding': '12px 28px',
                                'borderRadius': '6px', 'cursor': 'pointer', 'fontSize': '14px'
                            }),
                        ], style={'marginTop': '8px'})

                    ], style={**estilo_tarjeta, 'flex': '0 0 380px', 'minWidth': '300px'}),

                    # ── Panel de resultado ──────────────────────────────────
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

                        html.Div(id='gauge-container', children=[
                            dcc.Graph(id='gauge-regresion',
                                      config={'displayModeBar': False},
                                      style={'height': '250px', 'marginTop': '10px'})
                        ], style={'display': 'none'}),

                        html.Div(id='interpretacion-regresion', style={'marginTop': '16px'})

                    ], style={**estilo_tarjeta, 'flex': '1', 'minWidth': '300px'})

                ], style={'display': 'flex', 'flexDirection': 'row',
                          'gap': '20px', 'alignItems': 'flex-start', 'flexWrap': 'wrap'})

            ], style={'padding': '20px'})
        ]),

        # ══════════════════════════════════════════════════════════════════════
        # PESTAÑA 2 — Clasificación riesgo académico (Gabriel)
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
        # PESTAÑA 3 — Clasificación jornada vulnerable (Sebastián)
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

    ], colors={'background': ESTILO_FONDO, 'border': ESTILO_BORDE, 'primary': COLOR_ACENTO})

], style={'backgroundColor': ESTILO_FONDO, 'color': COLOR_TEXTO,
          'minHeight': '100vh', 'padding': '30px', 'fontFamily': 'Arial, sans-serif'})

# =============================================================================
# 7. CALLBACKS
# =============================================================================

@app.callback(
    Output('resultado-regresion',      'children'),
    Output('gauge-regresion',          'figure'),
    Output('interpretacion-regresion', 'children'),
    Output('gauge-container',          'style'),
    Output('reg-estrato',              'value'),
    Output('reg-zona',                 'value'),
    Output('reg-edu-madre',            'value'),
    Output('reg-edu-padre',            'value'),
    Output('reg-jornada',              'value'),
    Output('reg-municipio',            'value'),
    Output('reg-internet',             'value'),
    Output('reg-computador',           'value'),
    Output('reg-naturaleza',           'value'),
    Output('reg-genero',               'value'),
    Input('btn-predecir-reg',          'n_clicks'),
    Input('btn-limpiar-reg',           'n_clicks'),
    State('reg-estrato',               'value'),
    State('reg-zona',                  'value'),
    State('reg-edu-madre',             'value'),
    State('reg-edu-padre',             'value'),
    State('reg-jornada',               'value'),
    State('reg-municipio',             'value'),
    State('reg-internet',              'value'),
    State('reg-computador',            'value'),
    State('reg-naturaleza',            'value'),
    State('reg-genero',                'value'),
    prevent_initial_call=True
)
def manejar_formulario(n_predecir, n_limpiar,
                       estrato, zona, edu_madre, edu_padre,
                       jornada, municipio, internet, computador,
                       naturaleza, genero):

    defaults = (1, 1, 4, 4, 'MAÑANA',
                list(municipio_encoder.values())[0], 1, 1, 0, 1)

    figura_vacia = go.Figure()
    figura_vacia.update_layout(
        paper_bgcolor=ESTILO_TARJETA, plot_bgcolor=ESTILO_TARJETA,
        margin=dict(l=20, r=20, t=20, b=20), height=220,
        xaxis={'visible': False}, yaxis={'visible': False}
    )

    mensaje_vacio = html.P(
        'Complete el formulario y presione Predecir.',
        style={'color': COLOR_SUBTEXTO, 'textAlign': 'center',
               'marginTop': '80px', 'fontSize': '14px'}
    )

    # ── Limpiar ───────────────────────────────────────────────────────────────
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'btn-limpiar-reg.n_clicks':
        return (mensaje_vacio, figura_vacia, html.Div(),
                {'display': 'none'}, *defaults)

    # ── Predecir ──────────────────────────────────────────────────────────────
    features = np.array([[
        estrato, zona, edu_madre, edu_padre, municipio,
        internet, computador, naturaleza, genero,
        1 if jornada == 'MAÑANA'   else 0,
        1 if jornada == 'NOCHE'    else 0,
        1 if jornada == 'SABATINA' else 0,
        1 if jornada == 'TARDE'    else 0,
        1 if jornada == 'UNICA'    else 0,
    ]])

    features_scaled = scaler_X_reg.transform(features)
    pred_scaled     = modelo_regresion.predict(features_scaled, verbose=0).flatten()
    pred_real       = float(scaler_y_reg.inverse_transform(
                         pred_scaled.reshape(-1, 1)).flatten()[0])
    pred_real       = round(pred_real, 1)

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
        html.P(f'{signo}{diferencia:.1f} pts vs promedio departamental ({PROMEDIO_HUILA:.0f} pts)',
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
            'threshold': {'line': {'color': '#ffcc00', 'width': 3},
                          'thickness': 0.75, 'value': PROMEDIO_HUILA}
        }
    ))
    gauge.update_layout(paper_bgcolor=ESTILO_TARJETA, font={'color': COLOR_TEXTO},
                        margin=dict(l=20, r=20, t=20, b=20), height=220)

    if pred_real >= PROMEDIO_HUILA + 20:
        msg, color, icono = ('El estudiante presenta un perfil favorable. Se proyecta un desempeño destacado frente al promedio departamental.',
                             '#00c48c', '✅')
    elif pred_real >= PROMEDIO_HUILA:
        msg, color, icono = ('El estudiante se proyecta por encima del promedio departamental. Condiciones socioeconómicas adecuadas para un desempeño satisfactorio.',
                             COLOR_ACENTO, '📊')
    elif pred_real >= PROMEDIO_HUILA - 20:
        msg, color, icono = ('El estudiante se proyecta ligeramente por debajo del promedio. Se recomienda seguimiento y apoyo pedagógico focalizado.',
                             '#ffcc00', '⚠️')
    else:
        msg, color, icono = ('El perfil del estudiante indica alto riesgo de bajo rendimiento. Se recomienda priorizar intervención temprana.',
                             '#ff6b6b', '🔴')

    interpretacion_div = html.Div([
        html.Span(f'{icono}  '),
        html.Span(msg, style={'color': COLOR_SUBTEXTO, 'fontSize': '14px'})
    ], style={'padding': '12px 16px', 'backgroundColor': '#0d0f1a',
              'borderRadius': '8px', 'borderLeft': f'3px solid {color}'})

    return (resultado_div, gauge, interpretacion_div,
            {'display': 'block'},
            estrato, zona, edu_madre, edu_padre, jornada,
            municipio, internet, computador, naturaleza, genero)


@app.callback(
    Output('reloj', 'children'),
    Input('intervalo-reloj', 'n_intervals')
)
def actualizar_reloj(n):
    from datetime import datetime
    return datetime.now().strftime('%A %d de %B de %Y  |  %H:%M:%S')


# =============================================================================
# 8. EJECUCIÓN
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True)
