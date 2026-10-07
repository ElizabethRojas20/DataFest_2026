#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera CSVs auxiliares para el script generar_graficas.py

Solo escribe valores CALCULADOS a partir de los datos o de salidas reales del pipeline. Lo que no
puede calcular lo salta con un aviso: nunca escribe valores simulados ni de relleno.
"""
import pandas as pd
from pathlib import Path

RESULTADOS_DIR = Path("resultados_reportes")
DATOS_DIR = Path("datos_entrada")

# 1. tasa_conversion_mes.csv
print("Generando tasa_conversion_mes.csv...")
train = pd.read_csv(DATOS_DIR / "train.csv")
tasa = train.groupby("mes")["objetivo"].agg(["mean", "sum", "size"])
tasa.to_csv(RESULTADOS_DIR / "tasa_conversion_mes.csv")
print(f"  OK: {len(tasa)} meses")

# 2. correlacion_features.csv: Pearson entre las columnas numéricas y booleanas originales de train
print("Generando correlacion_features.csv...")
num = ["edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses", "numero_productos",
       "saldo_promedio", "dias_ultima_transaccion", "antiguedad_direccion_meses",
       "visitas_web_ultimos_90_dias", "distancia_sucursal_km", "dia_preferido_pago",
       "dias_ultima_interaccion", "tiene_tarjeta_credito", "activo_movil", "es_nuevo_cliente",
       "tiene_prestamo", "tiene_seguro"]
train[num].astype(float).corr(method="pearson").to_csv(RESULTADOS_DIR / "correlacion_features.csv")
print(f"  OK: {len(num)} variables")

# 3-4. importancia_permutacion.csv y ablacion_variables.csv: el pipeline no los persiste. Solo se
# usan si algún experimento los escribió antes; aquí no se inventan.
for nombre in ("importancia_permutacion.csv", "ablacion_variables.csv"):
    estado = "OK (existente)" if (RESULTADOS_DIR / nombre).exists() else "SKIP (no existe; no se inventa)"
    print(f"  {estado}: {nombre}")

# 5. walkforward_cv.csv
print("Generando walkforward_cv.csv...")
val_pred = RESULTADOS_DIR / "val_predictions.csv"
if val_pred.exists():
    df = pd.read_csv(val_pred)
    # Calcular Gini por modelo y fold
    from sklearn.metrics import roc_auc_score
    filas = []
    for (modelo, fold), grupo in df.groupby(["modelo", "fold"]):
        auc = roc_auc_score(grupo["y_true"], grupo["y_prob"])
        gini = 2 * auc - 1
        filas.append({"modelo": modelo, "fold": fold, "gini": gini})
    wf_df = pd.DataFrame(filas)
    wf_df.to_csv(RESULTADOS_DIR / "walkforward_cv.csv", index=False)
    print(f"  OK: {len(wf_df)} filas")
else:
    print("  SKIP: val_predictions.csv no existe")

# 6. bootstrap_gini.csv: lo escribe el pipeline (réplicas reales del bootstrap de clientes del
# modelo ganador). Aquí no se simula.
estado = "OK (existente)" if (RESULTADOS_DIR / "bootstrap_gini.csv").exists() else "SKIP: ejecutar el pipeline"
print(f"  {estado}: bootstrap_gini.csv")

print("\n¡CSVs generados en resultados_reportes/!")
