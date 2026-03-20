# ─────────────────────────────────────────────────────────────────────────────
#  GENERACIÓN DEL DATASET SEMI-SINTÉTICO DE CRÉDITO
#
#  Diseño inspirado en la estructura de "Home Credit Default Risk" (Kaggle)
#  con relaciones realistas entre variables y ruido controlado.
#
#  Variables incluidas:
#   - Sociodemográficas:  edad, género, estado civil, nivel educativo
#   - Laborales:          tipo de empleo, antigüedad laboral, ingreso
#   - Crediticias:        monto del crédito, plazo, tasa, historial de pagos
#   - Comportamentales:   días de mora máxima, número de créditos previos
#   - Colateral/garantía: tipo de vivienda, ratio LTV
#   - Target:             DEFAULT (1 = incumplimiento en los próximos 12 meses)
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 30_000  # Número de observaciones

# ── 1. Variables sociodemográficas ───────────────────────────────────────────
edad = np.random.normal(loc=42, scale=11, size=N).clip(18, 75).astype(int)
genero = np.random.choice(['M', 'F'], size=N, p=[0.52, 0.48])
estado_civil = np.random.choice(
    ['Soltero', 'Casado', 'Divorciado', 'Viudo'],
    size=N, p=[0.35, 0.45, 0.15, 0.05]
)
nivel_educativo = np.random.choice(
    ['Secundaria', 'Técnico', 'Universitario', 'Postgrado'],
    size=N, p=[0.25, 0.30, 0.35, 0.10]
)
num_dependientes = np.random.choice([0, 1, 2, 3, 4], size=N, p=[0.30, 0.30, 0.25, 0.10, 0.05])

# ── 2. Variables laborales ────────────────────────────────────────────────────
tipo_empleo = np.random.choice(
    ['Dependiente', 'Independiente', 'Empresario', 'Jubilado', 'Sin empleo'],
    size=N, p=[0.55, 0.25, 0.10, 0.07, 0.03]
)

# El ingreso varía según nivel educativo (relación realista)
ingreso_base = np.where(nivel_educativo == 'Postgrado', 6000,
               np.where(nivel_educativo == 'Universitario', 3500,
               np.where(nivel_educativo == 'Técnico', 2000, 1200)))
ingreso = (ingreso_base * np.random.lognormal(mean=0, sigma=0.4, size=N)).round(0)
ingreso = np.clip(ingreso, 500, 50000)

# Ingresos no declarados (~8% → NaN)
mask_missing_ingreso = np.random.random(N) < 0.08
ingreso = ingreso.astype(float)
ingreso[mask_missing_ingreso] = np.nan

# Antigüedad laboral (años)
antiguedad_laboral = np.random.exponential(scale=7, size=N).clip(0, 40).round(1)
mask_missing_antiguedad = np.random.random(N) < 0.05
antiguedad_laboral = antiguedad_laboral.astype(float)
antiguedad_laboral[mask_missing_antiguedad] = np.nan

# ── 3. Variables del crédito ──────────────────────────────────────────────────
monto_credito = np.random.lognormal(mean=9.5, sigma=1.0, size=N).round(0)
monto_credito = np.clip(monto_credito, 1000, 300000)

plazo_meses = np.random.choice([12, 24, 36, 48, 60, 84, 120], size=N,
                                p=[0.10, 0.20, 0.25, 0.20, 0.15, 0.07, 0.03])

tasa_interes = np.random.normal(loc=18, scale=6, size=N).clip(5, 50).round(2)

tipo_credito = np.random.choice(
    ['Consumo', 'Hipotecario', 'Vehicular', 'Empresarial'],
    size=N, p=[0.40, 0.30, 0.20, 0.10]
)

# ── 4. Variables de historial crediticio ──────────────────────────────────────
num_creditos_previos = np.random.poisson(lam=3, size=N).clip(0, 15)

dias_mora_max = np.random.choice(
    [0, 30, 60, 90, 120, 180],
    size=N, p=[0.60, 0.15, 0.10, 0.08, 0.04, 0.03]
)
# Sin historial (~6% → NaN)
mask_missing_mora = np.random.random(N) < 0.06
dias_mora_max = dias_mora_max.astype(float)
dias_mora_max[mask_missing_mora] = np.nan

ratio_utilizacion = np.random.beta(a=2, b=5, size=N).round(4)  # 0–1
mask_missing_util = np.random.random(N) < 0.04
ratio_utilizacion[mask_missing_util] = np.nan

# ── 5. Variables de colateral ─────────────────────────────────────────────────
tipo_vivienda = np.random.choice(
    ['Propia', 'Arrendada', 'Familiar', 'Hipotecada'],
    size=N, p=[0.35, 0.30, 0.25, 0.10]
)
ratio_ltv = np.where(
    tipo_credito == 'Hipotecario',
    np.random.uniform(0.5, 0.95, N).round(3),
    np.nan  # Solo aplica para hipotecarios
)

# ── 6. Construcción del TARGET con relaciones realistas ───────────────────────
#
#  El logit incluye efectos CATEGÓRICOS explícitos calibrados sobre
#  literatura real de credit scoring (Siddiqi 2017, Basel II studies).
#
#  Efectos esperados (PD aproximada por segmento):
#   tipo_credito : Consumo ~18%, Vehicular ~14%, Empresarial ~9%, Hipotecario ~5%
#   tipo_empleo  : Sin empleo ~22%, Independiente ~17%, Dependiente ~11%, Empresario ~9%, Jubilado ~7%
#   tipo_vivienda: Arrendada ~17%, Familiar ~14%, Hipotecada ~10%, Propia ~8%
#   nivel_educ.  : Secundaria ~17%, Técnico ~13%, Universitario ~10%, Postgrado ~6%
#   estado_civil : Soltero ~16%, Divorciado ~15%, Viudo ~12%, Casado ~9%
#   género       : M ~14%, F ~11%  (diferencia moderada, no determinante)

# ── Efectos por tipo de crédito ───────────────────────────────────────────────
#    Hipotecario y Empresarial tienen garantías reales → menor PD
#    Consumo y Vehicular son créditos sin garantía sólida → mayor PD
ef_tipo_credito = np.select(
    [tipo_credito == 'Consumo',
     tipo_credito == 'Vehicular',
     tipo_credito == 'Empresarial',
     tipo_credito == 'Hipotecario'],
    [0.65, 0.25, -0.40, -0.80],
    default=0.0
)

# ── Efectos por tipo de empleo ─────────────────────────────────────────────────
#    Estabilidad laboral como proxy de capacidad de pago
ef_tipo_empleo = np.select(
    [tipo_empleo == 'Sin empleo',
     tipo_empleo == 'Independiente',
     tipo_empleo == 'Dependiente',
     tipo_empleo == 'Empresario',
     tipo_empleo == 'Jubilado'],
    [1.00, 0.45, 0.0, -0.25, -0.55],
    default=0.0
)

# ── Efectos por tipo de vivienda ──────────────────────────────────────────────
#    Vivienda propia como proxy de patrimonio y estabilidad
ef_tipo_vivienda = np.select(
    [tipo_vivienda == 'Arrendada',
     tipo_vivienda == 'Familiar',
     tipo_vivienda == 'Hipotecada',
     tipo_vivienda == 'Propia'],
    [0.50, 0.20, -0.10, -0.45],
    default=0.0
)

# ── Efectos por nivel educativo ───────────────────────────────────────────────
#    Educación como proxy de ingreso permanente y cultura financiera
ef_nivel_educ = np.select(
    [nivel_educativo == 'Secundaria',
     nivel_educativo == 'Técnico',
     nivel_educativo == 'Universitario',
     nivel_educativo == 'Postgrado'],
    [0.55, 0.20, -0.15, -0.65],
    default=0.0
)

# ── Efectos por estado civil ──────────────────────────────────────────────────
#    Casados y viudos tienden a mayor estabilidad financiera
ef_estado_civil = np.select(
    [estado_civil == 'Soltero',
     estado_civil == 'Divorciado',
     estado_civil == 'Viudo',
     estado_civil == 'Casado'],
    [0.35, 0.30, 0.05, -0.30],
    default=0.0
)

# ── Efectos por género ────────────────────────────────────────────────────────
#    Efecto moderado, consistente con literatura empírica
ef_genero = np.where(genero == 'M', 0.15, -0.15)

# ── Score lineal (logit) completo ─────────────────────────────────────────────
logit = (
    -2.20                                                    # intercepto base
    # Variables continuas
    + 0.022  * (40 - edad.clip(20, 65))                      # jóvenes: más riesgo
    - 0.00007 * np.nan_to_num(ingreso, nan=1500)             # ingreso: menos riesgo
    + 0.016  * np.nan_to_num(dias_mora_max, nan=0)           # mora histórica: más riesgo
    + 0.90   * np.nan_to_num(ratio_utilizacion, nan=0.3)     # uso tarjeta: más riesgo
    - 0.045  * np.nan_to_num(antiguedad_laboral, nan=3)      # antigüedad: menos riesgo
    + 0.12   * num_dependientes                              # dependientes: más riesgo
    # Variables categóricas (efectos diferenciados)
    + ef_tipo_credito
    + ef_tipo_empleo
    + ef_tipo_vivienda
    + ef_nivel_educ
    + ef_estado_civil
    + ef_genero
    # Ruido (reducido para no aplanar las diferencias)
    + np.random.normal(0, 0.30, N)
)

prob_default = 1 / (1 + np.exp(-logit))
DEFAULT = (np.random.uniform(0, 1, N) < prob_default).astype(int)

# ── 7. Fecha de alta del crédito (para split temporal) ────────────────────────
#
#  Simulamos una cartera que crece linealmente entre 2025 y 2026.
#  El crecimiento suave replica el comportamiento típico de una cartera
#  de consumo en expansión: más originaciones cada año, sin saltos bruscos.
#
#  Esto permite hacer un split TEMPORAL en lugar de aleatorio:
#    Train : originaciones < 2023-01-01  (~67%)
#    Val   : originaciones 2023 H1       (~16%)
#    Test  : originaciones 2023 H2       (~17%)
#
#  ¿Por qué importa el split temporal?
#    - En producción nunca entrenamos con datos futuros
#    - Permite detectar drift de población (PSI) entre períodos
#    - Más cercano a la validación regulatoria bajo IFRS 9 / Basilea

fecha_inicio = pd.Timestamp('2025-01-01')
fecha_fin    = pd.Timestamp('2026-03-31')
total_dias   = (fecha_fin - fecha_inicio).days  # 1460 días

dias_idx = np.arange(total_dias + 1)
# Pesos con crecimiento lineal suave (más originaciones en años recientes)
pesos_fecha = 1 + (dias_idx / total_dias) * 1.5
pesos_fecha /= pesos_fecha.sum()

dias_offset = np.random.choice(dias_idx, size=N, p=pesos_fecha)
fecha_alta  = fecha_inicio + pd.to_timedelta(dias_offset, unit='D')

# ── 8. Ensamblar el DataFrame ──────────────────────────────────────────────────
df = pd.DataFrame({
    'ID_CLIENTE': [f'CLI{str(i).zfill(6)}' for i in range(1, N+1)],
    'FECHA_ALTA': fecha_alta,            
    # Sociodemográficas
    'EDAD': edad,
    'GENERO': genero,
    'ESTADO_CIVIL': estado_civil,
    'NIVEL_EDUCATIVO': nivel_educativo,
    'NUM_DEPENDIENTES': num_dependientes,
    # Laborales
    'TIPO_EMPLEO': tipo_empleo,
    'INGRESO_MENSUAL': ingreso,
    'ANTIGUEDAD_LABORAL': antiguedad_laboral,
    # Crédito
    'TIPO_CREDITO': tipo_credito,
    'MONTO_CREDITO': monto_credito,
    'PLAZO_MESES': plazo_meses,
    'TASA_INTERES': tasa_interes,
    # Historial
    'NUM_CREDITOS_PREVIOS': num_creditos_previos,
    'DIAS_MORA_MAX': dias_mora_max,
    'RATIO_UTILIZACION': ratio_utilizacion,
    # Colateral
    'TIPO_VIVIENDA': tipo_vivienda,
    'RATIO_LTV': ratio_ltv,
    # Target
    'DEFAULT': DEFAULT
})

out = 'data\s10'
os.makedirs(out, exist_ok=True)
df.to_csv(out + '\credit_data_raw.csv', index=False)
print(f'✅ Dataset raw guardado en {out}\credit_data_raw.csv')