import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 15_000  # Total de créditos en la cartera

# --- Variables del deudor ---
edad            = np.random.normal(38, 10, N).clip(22, 70).astype(int)
ingreso_mensual = np.random.lognormal(mean=8.5, sigma=0.5, size=N).clip(500, 20000)
score_buro      = np.random.normal(620, 80, N).clip(300, 850).astype(int)
deuda_ingreso   = np.random.beta(2, 5, N)           # Ratio deuda/ingreso (0 a 1)
num_productos   = np.random.choice([1, 2, 3, 4], N, p=[0.4, 0.3, 0.2, 0.1])

# --- Score de la cartera: combina variables para producir un logit ---
logit = (
    -3.0
    + 0.02  * (score_buro - 600)   # Mejor score → menor riesgo
    - 0.008 * (ingreso_mensual / 100)
    + 2.5   * deuda_ingreso         # Mayor endeudamiento → más riesgo
    - 0.3   * num_productos          # Más productos → relación con banco, menos riesgo
    + 0.01  * (40 - edad)           # Jóvenes son ligeramente más riesgosos
    + np.random.normal(0, 0.3, N)   # Ruido
)

# PD "verdadera" generada por el proceso (PD TTC subyacente)
pd_verdadera = 1 / (1 + np.exp(-logit))

# PD estimada por el modelo de scoring (con algo de error de estimación)
pd_modelo = pd_verdadera * np.exp(np.random.normal(0, 0.1, N))
pd_modelo = pd_modelo.clip(0.001, 0.999)

# Default observado (realización estocástica basada en la PD verdadera)
default_observado = (np.random.uniform(0, 1, N) < pd_verdadera).astype(int)

# Año de originación del crédito (distribución a lo largo de 3 años)
anio_originacion = np.random.choice([2021, 2022, 2023], N, p=[0.35, 0.40, 0.25])

# Plazo restante del crédito en meses (entre 6 y 60 meses)
plazo_meses = np.random.choice([12, 24, 36, 48, 60], N, p=[0.15, 0.30, 0.30, 0.15, 0.10])

# --- Construcción del DataFrame ---
df = pd.DataFrame({
    'id_credito'        : np.arange(1, N + 1),
    'anio_originacion'  : anio_originacion,
    'plazo_meses'       : plazo_meses,
    'edad'              : edad,
    'ingreso_mensual'   : ingreso_mensual.round(0),
    'score_buro'        : score_buro,
    'deuda_ingreso'     : deuda_ingreso.round(4),
    'num_productos'     : num_productos,
    'pd_modelo'         : pd_modelo.round(6),      # PD output del modelo (TTC)
    'default_obs'       : default_observado
})

out = 'data\s15'
os.makedirs(out, exist_ok=True)
df.to_csv(out + '\credit_data_raw.csv', index=False)
print(f'✅ Dataset raw guardado en {out}\credit_data_raw.csv')