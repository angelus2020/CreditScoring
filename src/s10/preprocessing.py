# =============================================================================
#  preprocessing.py
#  Sesión 10 — Programa Especializado en Credit Scoring con Python
#
#  Uso en el notebook principal:
#
#    from preprocessing import aplicar_pipeline, guardar_artefactos,
#                              cargar_artefactos, guardar_por_mes
#
#    # ── Guardar después de procesar train ──
#    guardar_artefactos(artefactos, path='data/s10/artefactos.pkl')
#
#    # ── Aplicar a val y test ───────────────
#    X_val_proc  = aplicar_pipeline(X_val,  artefactos)
#    X_test_proc = aplicar_pipeline(X_test, artefactos)
#
#    # ── Guardar por mes ───────────────────
#    guardar_por_mes(X_train,     y_train, train['FECHA_ALTA'], 'train', 'data/s10')
#    guardar_por_mes(X_val_proc,  y_val,   val['FECHA_ALTA'],   'val',   'data/s10')
#    guardar_por_mes(X_test_proc, y_test,  oot['FECHA_ALTA'],   'test',  'data/s10')
# =============================================================================

import os
import json
import joblib
import numpy  as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def aplicar_pipeline(X: pd.DataFrame, artefactos: dict) -> pd.DataFrame:
    """
    Aplica el pipeline completo de preprocesamiento sobre un split crudo
    (val o test/OOT), usando ÚNICAMENTE estadísticos aprendidos en train.

    Orden de pasos
    --------------
    1. Indicadores de faltante  (ANTES de imputar)
    2. Imputación               (KNN + mediana + cero semántico)
    3. Encoding categórico      (ordinal + binario)
    4. Winsorización            (clips calculados en train)
    5. Feature Engineering      (ratios, interacciones, log-transforms)
    6. Escalado                 (StandardScaler fiteado en train)

    Parámetros
    ----------
    X          : DataFrame crudo del split (sin DEFAULT ni columnas de fecha)
    artefactos : dict devuelto por `cargar_artefactos` o construido en el notebook

    Retorna
    -------
    pd.DataFrame con las mismas columnas que X_train procesado
    """
    X = X.copy()

    # ── 1. Indicadores de faltante ────────────────────────────────────────────
    #    Se crean ANTES de imputar; si se crean después siempre valen 0
    X['IND_INGRESO_MISSING']   = X['INGRESO_MENSUAL'].isnull().astype(int)
    X['IND_HISTORIAL_MISSING'] = X['DIAS_MORA_MAX'].isnull().astype(int)

    # ── 2. Imputación ─────────────────────────────────────────────────────────

    # 2a. KNN — transform only, nunca fit sobre val/test
    VARS_KNN = ['INGRESO_MENSUAL', 'ANTIGUEDAD_LABORAL']
    X[VARS_KNN] = artefactos['knn_imputer'].transform(X[VARS_KNN])

    # 2b. Mediana de train
    X['RATIO_UTILIZACION'] = X['RATIO_UTILIZACION'].fillna(artefactos['median_util'])

    # 2c. Cero semántico
    X['DIAS_MORA_MAX']      = X['DIAS_MORA_MAX'].fillna(0)
    X['ANTIGUEDAD_LABORAL'] = X['ANTIGUEDAD_LABORAL'].fillna(0)
    X['RATIO_LTV']          = X['RATIO_LTV'].fillna(0)

    # ── 3. Encoding categórico ────────────────────────────────────────────────

    # Binaria
    X['CASADO'] = (X['ESTADO_CIVIL'] == 'Casado').astype(int)
    X.drop(columns=['ESTADO_CIVIL'], inplace=True)

    X['GENERO'] = (X['GENERO'] == 'M').astype(int)

    # Ordinal (el orden refleja nivel de riesgo ascendente)
    X['NIVEL_EDUCATIVO'] = X['NIVEL_EDUCATIVO'].map(
        {'Secundaria': 3, 'Técnico': 2, 'Universitario': 1, 'Postgrado': 0}
    )
    X['TIPO_EMPLEO'] = X['TIPO_EMPLEO'].map(
        {'Sin empleo': 4, 'Independiente': 3, 'Dependiente': 2,
         'Empresario': 1, 'Jubilado': 0}
    )
    X['TIPO_CREDITO'] = X['TIPO_CREDITO'].map(
        {'Consumo': 3, 'Vehicular': 2, 'Empresarial': 1, 'Hipotecario': 0}
    )
    X['TIPO_VIVIENDA'] = X['TIPO_VIVIENDA'].map(
        {'Arrendada': 3, 'Familiar': 2, 'Hipotecada': 1, 'Propia': 0}
    )

    # ── 4. Winsorización con clips de train ───────────────────────────────────
    for col, (p1, p99) in artefactos['winsor_clips'].items():
        X[col] = X[col].clip(lower=p1, upper=p99)

    # ── 5. Feature Engineering ────────────────────────────────────────────────

    # Cuota mensual (fórmula de amortización francesa)
    tasa_m = (X['TASA_INTERES'] / 100) / 12
    X['CUOTA_ESTIMADA'] = np.where(
        tasa_m > 0,
        X['MONTO_CREDITO'] * tasa_m / (1 - (1 + tasa_m) ** (-X['PLAZO_MESES'])),
        X['MONTO_CREDITO'] / X['PLAZO_MESES']
    ).round(2)

    # DTI — usa mediana de INGRESO de train para no dividir por 0
    X['RATIO_ENDEUDAMIENTO'] = (
        X['CUOTA_ESTIMADA'] / X['INGRESO_MENSUAL'].replace(0, np.nan)
    ).clip(0, 5).fillna(
        X['CUOTA_ESTIMADA'] / artefactos['median_ingreso']
    ).round(4)

    # Segmento de riesgo por DTI
    X['SEGMENTO_RIESGO'] = pd.cut(
        X['RATIO_ENDEUDAMIENTO'],
        bins=[-np.inf, 0.20, 0.35, 0.50, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Interacción edad × mora
    X['INTERACCION_EDAD_MORA'] = (
        (1 / X['EDAD'].clip(18, 75)) * X['DIAS_MORA_MAX']
    ).round(6)

    # Log-transform y drop de originales
    for col in ['INGRESO_MENSUAL', 'MONTO_CREDITO']:
        X[f'LOG_{col}'] = np.log1p(X[col])
        X.drop(columns=col, inplace=True)

    # ── 6. Escalado — transform only ─────────────────────────────────────────
    X[artefactos['num_cols']] = artefactos['scaler'].transform(
        X[artefactos['num_cols']]
    )

    return X


# ─────────────────────────────────────────────────────────────────────────────
#  GESTIÓN DE ARTEFACTOS
# ─────────────────────────────────────────────────────────────────────────────

def guardar_artefactos(artefactos: dict, path: str) -> None:
    """
    Serializa el dict de artefactos en dos archivos:
      - <path>.pkl  → objetos sklearn (knn_imputer, scaler)
      - <path>.json → valores numéricos y listas (legibles sin sklearn)

    Parámetros
    ----------
    artefactos : dict construido en el notebook tras procesar train
    path       : ruta base sin extensión, e.g. 'data/s10/artefactos'
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    # ── .pkl: objetos que no son serializables en JSON ────────────────────────
    joblib.dump(artefactos, path + '.pkl')

    # ── .json: valores inspeccionables sin cargar sklearn ────────────────────
    meta = {
        'median_util'   : float(artefactos['median_util']),
        'median_ingreso': float(artefactos['median_ingreso']),
        'winsor_clips'  : {
            col: [float(p1), float(p99)]
            for col, (p1, p99) in artefactos['winsor_clips'].items()
        },
        'num_cols'      : list(artefactos['num_cols']),
    }
    with open(path + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f'✅ Artefactos guardados:')
    print(f'   {path}.pkl   — objetos sklearn serializados')
    print(f'   {path}.json  — valores numéricos y listas (inspeccionable)')
    print(f'\n   Contenido del .json:')
    print(f'     median_util    = {meta["median_util"]:.6f}')
    print(f'     median_ingreso = {meta["median_ingreso"]:.4f}')
    for col, (p1, p99) in meta['winsor_clips'].items():
        print(f'     winsor {col:<25} p1={p1:.4f}  p99={p99:.4f}')
    print(f'     num_cols ({len(meta["num_cols"])}): {meta["num_cols"]}')


def cargar_artefactos(path: str) -> dict:
    """
    Carga los artefactos desde el .pkl generado por guardar_artefactos.

    Parámetros
    ----------
    path : ruta base sin extensión, e.g. 'data/s10/artefactos'

    Retorna
    -------
    dict con todos los objetos listos para pasar a aplicar_pipeline
    """
    artefactos = joblib.load(path + '.pkl')
    print(f'✅ Artefactos cargados desde {path}.pkl')
    return artefactos
