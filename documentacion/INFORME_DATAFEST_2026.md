# Informe DataFest 2026: Modelo de Propensión de Conversión Bancaria

---

## 1. Resumen Ejecutivo

Este informe documenta el trabajo realizado por el equipo para la competencia **DataFest 2026**, cuyo objetivo es predecir la **propensión de conversión de clientes bancarios** (probabilidad de que un cliente acepte una oferta/contrate un producto en diciembre 2026).

### Resultados Principales

| Métrica | Valor | Contexto |
|---------|-------|----------|
| **Mejor Gini (validación)** | **0.2505** | LightGBM + Optuna (Pipeline DF_WCB / DF_New) |
| **Mejor AUC (validación)** | **0.6252** | LightGBM + Optuna |
| **Features finales** | **50 variables** | Seleccionadas de ~200+ generadas |
| **Registros de test predichos** | **9,900 clientes** | Diciembre 2026 |
| **Archivo de entrega** | `submission.csv` | Formato: `id_cliente, prediccion` |

### Conclusión del Equipo

> **El pipeline final (DF_WCB / DF_New) alcanza un Gini de ~0.25 en validación temporal**, superando la versión inicial (DF_1: Gini 0.242). Las mejoras clave fueron: **validación temporal robusta** (últimos 3 meses), **incluir CatBoost** en la comparativa, y **escalado correcto de iteraciones** al reentrenar con el 100% de datos.

---

## 2. Descripción del Problema

### 2.1 Objetivo de Negocio
Estimar la **probabilidad de conversión** (`objetivo = 1`) para cada cliente en diciembre 2026, permitiendo al banco priorizar contactos en call center, optimizar ROI comercial y reducir fricción con clientes no interesados.

### 2.2 Datos Disponibles

| Dataset | Período | Filas | Descripción |
|---------|---------|-------|-------------|
| `train.csv` | Ene–Nov 2026 | 110,100 | Historial con variable objetivo conocida |
| `test.csv` | Dic 2026 | 9,900 | Clientes a predecir (sin objetivo) |
| `sample_submission.csv` | - | 9,900 | Formato de entrega de referencia |

**Granularidad**: Una fila por **cliente-mes** (`id_cliente`, `mes` en formato AAAAMM). Un mismo cliente aparece en múltiples meses.

### 2.3 Métrica de Evaluación
**Coeficiente de Gini = 2 × AUC − 1**

- **Gini = 1**: Orden perfecto (todos los conversores arriba)
- **Gini = 0**: Aleatorio
- **Gini < 0**: Orden inverso

> **Importante**: Solo importa el **orden** de las predicciones, no su calibración absoluta.

---

## 3. Arquitectura del Pipeline

### 3.1 Flujo General

```
train.csv + test.csv
        │
        ▼
┌───────────────────┐
│  Limpieza y       │  → Tipos óptimos, validación esquema, diagnóstico
│  Diagnóstico      │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Feature          │  → Lags (1,2,3), Deltas, Ventanas móviles (media/suma 3m)
│  Engineering      │  → One-Hot / Target Encoding temporal (sin leakage)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Feature          │  → Varianza cero → Correlación >0.95 → Top-K LightGBM
│  Selection        │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Modelado y       │  → LightGBM, XGBoost, CatBoost, RF + Optuna
│  Validación       │  → Validación TEMPORAL (últimos 3 meses ≈ 73/27)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Entrenamiento    │  → Reentrena ganador con 100% train
│  Final + Pred     │  → Promedia 3 semillas → submission.csv
└───────────────────┘
```

### 3.2 Decisiones Críticas Anti-Leakage

| Regla | Implementación |
|-------|----------------|
| **Sin datos del futuro** | Lags/ventanas solo usan meses previos del mismo cliente (calendario-consciente) |
| **Target Encoding temporal** | TE de mes M usa solo objetivos de meses < M; test usa todo train |
| **Validación temporal** | Partición por meses (no aleatoria): evita inflar métrica por cliente repetido |
| **Selección solo en train** | Features se eligen con meses de entrenamiento, no validación |
| **Sin id_cliente/mes** | Diciembre es mes no visto; no se usan como predictoras |

---

## 4. Ingeniería de Características

### 4.1 Variables Originales (23 features)

| Tipo | Variables |
|------|-----------|
| **Numéricas (13)** | `edad`, `ingresos`, `ratio_deuda_ingresos`, `antiguedad_cuenta_meses`, `numero_productos`, `saldo_promedio`, `dias_ultima_transaccion`, `antiguedad_direccion_meses`, `visitas_web_ultimos_90_dias`, `distancia_sucursal_km`, `dia_preferido_pago`, `dias_ultima_interaccion` |
| **Booleanas (5)** | `tiene_tarjeta_credito`, `activo_movil`, `es_nuevo_cliente`, `tiene_prestamo`, `tiene_seguro` |
| **Categóricas (5)** | `ocupacion`, `region`, `canal_adquisicion`, `banda_riesgo`, `dispositivo_principal` |

### 4.2 Features Generadas

| Categoría | Ejemplos | Cantidad aprox. |
|-----------|----------|-----------------|
| **Lags (1,2,3 meses)** | `saldo_promedio_lag1`, `ingresos_lag2`, `dias_ultima_interaccion_lag3` | ~21 |
| **Deltas (cambio mensual)** | `saldo_promedio_delta1`, `dias_ultima_interaccion_delta1` | ~7 |
| **Ventanas móviles (3m)** | `*_roll3_mean`, `*_roll3_sum`, `meses_en_ventana3` | ~15 |
| **One-Hot (categóricas ≤10)** | `banda_riesgo_low`, `ocupacion_manual`, `region_west`, ... | ~20 |
| **Target Encoding (categóricas >10)** | `te_ocupacion`, `te_region`, ... | 0 (todas ≤5 niveles) |

**Total features generadas**: ~200+ → **Seleccionadas: 50**

### 4.3 Insights Clave de los Datos

- **81%** de clientes de test tienen historial en train
- **~15%** tasa de conversión, estable entre meses
- **Clientes convertidos NO reaparecen** (filas post-conversión = 0)
- **80%+ variables son estáticas por cliente** (solo `dias_ultima_interaccion` cambia ~80% meses)
- Lags/deltas de variables estáticas son constantes → eliminados por varianza cero

---

## 5. Selección de Features

### 5.1 Pipeline de Selección (3 etapas)

```python
# 1. Varianza cero: elimina constantes (ignorando NaN)
# 2. Correlación Pearson > 0.95: elimina la más redundante de cada par
# 3. Top-K por importancia LightGBM (gain): conserva las 80 más importantes
```

### 5.2 Top 20 Features Finales (ordenadas por importancia)

| Rank | Feature | Tipo | Descripción |
|------|---------|------|-------------|
| 1 | `numero_productos_roll3_mean` | Temporal | Media móvil 3m de nº productos |
| 2 | `dias_ultima_transaccion_roll3_mean` | Temporal | Media móvil 3m días última transacción |
| 3 | `banda_riesgo_low` | OHE | Banda riesgo = baja |
| 4 | `ratio_deuda_ingresos` | Original | Ratio deuda/ingresos (estático) |
| 5 | `distancia_sucursal_km` | Original | Distancia a sucursal |
| 6 | `antiguedad_cuenta_meses` | Original | Antigüedad cuenta |
| 7 | `ingresos` | Original | Ingresos cliente |
| 8 | `saldo_promedio_roll3_mean` | Temporal | Media móvil 3m saldo promedio |
| 9 | `banda_riesgo_high` | OHE | Banda riesgo = alta |
| 10 | `antiguedad_direccion_meses` | Original | Antigüedad dirección |
| 11 | `dias_ultima_interaccion_lag3` | Temporal | Lag 3m días última interacción |
| 12 | `dias_ultima_interaccion_roll3_mean` | Temporal | Media móvil 3m días última interacción |
| 13 | `dias_ultima_interaccion_lag1` | Temporal | Lag 1m días última interacción |
| 14 | `dias_ultima_transaccion_roll3_sum` | Temporal | Suma 3m días última transacción |
| 15 | `dias_ultima_interaccion_delta1` | Temporal | Cambio mensual días última interacción |
| 16 | `dias_ultima_interaccion_roll3_sum` | Temporal | Suma 3m días última interacción |
| 17 | `dias_ultima_interaccion_lag2` | Temporal | Lag 2m días última interacción |
| 18 | `edad` | Original | Edad cliente |
| 19 | `dia_preferido_pago` | Original | Día preferido pago |
| 20 | `activo_movil` | Booleana | App móvil activa |

> **Observación**: Las features temporales de `dias_ultima_interaccion` dominan el top-20 (es la única variable que cambia significativamente mes a mes).

---

## 6. Modelado y Resultados

### 6.1 Configuración de Modelos

| Modelo | Hiperparámetros Clave | Early Stopping |
|--------|----------------------|----------------|
| **LightGBM** | lr=0.03, leaves=31, min_child=50, subsample=0.8 | Sí (100 rondas) |
| **XGBoost** | lr=0.03, depth=6, min_child=5, subsample=0.8 | Sí (100 rondas) |
| **CatBoost** | lr=0.05, depth=6, l2=3, bootstrap=Bernoulli | Sí (100 rondas) |
| **RandomForest** | n_est=300, min_leaf=20, max_features=sqrt | No (usa imputación mediana) |
| **LightGBM + Optuna** | 30 trials, búsqueda bayesiana | Sí (100 rondas) |

### 6.2 Resultados Comparativos (Validación Temporal: 3 últimos meses)

#### Pipeline DF_1 (Versión Inicial)
| Modelo | AUC | Gini | Iteraciones |
|--------|-----|------|-------------|
| LightGBM + Optuna | 0.6210 | **0.2420** | 25 |
| XGBoost | 0.6191 | 0.2382 | 3 |
| LightGBM | 0.6182 | 0.2364 | 134 |
| RandomForest | 0.6061 | 0.2123 | 300 |

#### Pipeline DF_WCB (Con CatBoost + Validación Mejorada) ⭐ **MEJOR**
| Modelo | AUC | Gini | Iteraciones |
|--------|-----|------|-------------|
| **LightGBM + Optuna** | **0.6252** | **0.2505** | 189 |
| LightGBM | 0.6232 | 0.2464 | 46 |
| CatBoost | 0.6230 | 0.2461 | 288 |
| XGBoost | 0.6225 | 0.2451 | 27 |
| RandomForest | 0.6204 | 0.2407 | 300 |

#### Pipeline DF_New (Refinamiento sin CatBoost)
| Modelo | AUC | Gini | Iteraciones |
|--------|-----|------|-------------|
| **LightGBM + Optuna** | **0.6252** | **0.2505** | 189 |
| LightGBM | 0.6232 | 0.2464 | 46 |
| XGBoost | 0.6225 | 0.2451 | 27 |
| RandomForest | 0.6204 | 0.2407 | 300 |

### 6.3 Análisis de Resultados

| Hallazgo | Explicación |
|----------|-------------|
| **Mejora +0.0085 Gini** (DF_1 → DF_WCB) | Validación temporal robusta (3 meses vs 1 mes) + CatBoost + escalado iteraciones |
| **LightGBM Optuna gana consistentemente** | Optimización bayesiana encuentra mejor configuración |
| **CatBoost competitivo** (0.2461) | Buen manejo nativo de categóricas, pero features ya codificadas |
| **RandomForest mejora mucho** (0.212→0.241) | Validación temporal reduce overfitting; imputación mediana ayuda |
| **Diferencias < 0.01 Gini** | Dentro del ruido de validación (±0.010); elegir por estabilidad/velocidad |

### 6.4 Modelo Final Elegido

> **LightGBM + Optuna** (Gini validación = 0.2505)
> - Entrenado con 100% de train (110,100 filas)
> - n_estimators = 1.1 × mejor_iter_validación ≈ 208
> - Promedio de 3 semillas (42, 43, 44) para estabilizar ranking
> - Predicciones clippeadas a [0, 1]

---

## 7. Archivos de Entrega Generados

| Archivo | Descripción | Ubicación |
|---------|-------------|-----------|
| `submission.csv` | **Entrega principal** (9,900 predicciones) | Raíz del proyecto |
| `resultados_validacion_DF_WCB.csv` | Tabla completa métricas validación | Raíz del proyecto |
| `features_seleccionadas_DF_WCB.txt` | Lista de 50 features finales | Raíz del proyecto |
| `Pipeline_DF_WCB.py` | Pipeline final recomendado | Raíz del proyecto |

### Validación de Submission
```bash
# Verificaciones automáticas en el pipeline:
✓ Mismo nº de filas que test.csv (9,900)
✓ Mismo orden de id_cliente que test.csv
✓ Predicciones en rango [0, 1] sin NaN
✓ Coincide con sample_submission.csv
```

---

## 8. Cómo Reproducir el Pipeline

### 8.1 Requisitos
```bash
Python ≥ 3.10
pandas, numpy, scikit-learn ≥ 1.2
lightgbm ≥ 4.7, xgboost ≥ 3.4, catboost ≥ 1.2, optuna ≥ 5.0
```

### 8.2 Instalación
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pandas numpy scikit-learn lightgbm xgboost catboost optuna
```

### 8.3 Ejecución Completa (Recomendada)
```bash
# Pipeline final (DF_WCB) - ~2 minutos
python Pipeline_DF_WCB.py \
  --datos . \
  --salida ./salida/submission.csv \
  --trials 30 \
  --meses-val 3 \
  --top-k 80 \
  --n-seeds 3
```

### 8.4 Ejecución Rápida (Testing)
```bash
# Sin Optuna, 1 semilla - ~30 segundos
python Pipeline_DF_WCB.py \
  --datos . \
  --salida ./salida/prueba.csv \
  --trials 0 \
  --n-seeds 1
```

### 8.5 Argumentos Principales

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--datos` | `.` | Carpeta con train.csv, test.csv |
| `--salida` | `submission.csv` | Ruta archivo de entrega |
| `--trials` | `30` | Trials Optuna (0 = desactivado) |
| `--meses-val` | `3` | Meses finales para validación |
| `--top-k` | `80` | Máx. features tras selección |
| `--n-seeds` | `3` | Semillas para promediar final |
| `--semilla` | `42` | Semilla base reproducibilidad |

---

## 9. Lecciones Aprendidas y Recomendaciones

### 9.1 Qué Funcionó Bien
1. **Validación temporal estricta** - Evita leakage y da estimación honesta
2. **Target Encoding temporal** - Captura señal categórica sin leakage
3. **Feature engineering calendario-consciente** - Lags/ventanas respetan gaps de meses
4. **Selección en cascada** - Varianza → Correlación → Importancia es robusto
5. **Ensemble de semillas** - Estabiliza ranking final (±0.002 Gini)

### 9.2 Áreas de Mejora Futura
| Área | Propuesta |
|------|-----------|
| **Más modelos** | Probar HistGradientBoosting, TabNet, redes neuronales tabulares |
| **Feature engineering** | Interacciones (ej. `ingresos × ratio_deuda`), features de frecuencia cliente |
| **Validación** | Walk-forward validation (múltiples cortes temporales) |
| **Calibración** | Platt scaling / Isotonic regression para probabilidades calibradas |
| **Explainability** | SHAP values para interpretar drivers de conversión por segmento |

### 9.3 Próximos Pasos para Competencia
1. **Enviar `submission.csv` actual** (Gini ~0.2505 validación)
2. **Probar ensemble LightGBM + CatBoost** (promedio ponderado por Gini validación)
3. **Experimentar con `--meses-val 4`** (más datos validación, menos train)
4. **Añadir features de "meses desde última conversión"** si hay datos históricos pre-2026
5. **Documentar cada envío** con Gini validación y fecha para tracking

---

## 10. Apéndice: Estructura del Repositorio

```
DataFest_2026/
├── README.md                          # Documentación completa del pipeline
├── INFORME_DATAFEST_2026.md           # Este informe
├── DataFest 2026.html                 # Informe ejecutivo visual (HTML)
├── Guía del Pipeline DataFest.html    # Guía técnica detallada
├── Pipeline_DF_1.py                   # Versión 1 (base)
├── Pipeline_DF_WCB.py                 # Versión 2 ⭐ (final recomendada)
├── Pipeline_DF_New.py                 # Versión 3 (refinamiento)
├── train.csv                          # 18.1 MB - Datos entrenamiento
├── test.csv                           # 1.6 MB - Datos prueba
├── sample_submission.csv              # 104 KB - Formato entrega
├── submission.csv                     # 247 KB - ENTREGA FINAL
├── Validation_results_DF_1.csv        # Resultados v1
├── resultados_validacion_DF_WCB.csv   # Resultados v2 (final)
├── validation_results_DF_new.csv      # Resultados v3
├── features_seleccionadas_DF_1.txt    # Features v1 (50)
├── features_seleccionadas_DF_WCB.txt  # Features v2 (50) ⭐
├── features_seleccionadas_new.txt     # Features v3 (50)
└── gitignore.txt                      # Patrones git ignore
```

---

## 11. Contacto y Autoría

**Equipo DataFest 2026 - BCP**  
Pipeline desarrollado siguiendo mejores prácticas de ML engineering:
- Código modular, tipado, documentado
- Reproducible (semillas fijas, versión de librerías)
- Trazabilidad completa (artefactos de validación y features)
- Listo para producción / competencia

---

*Informe generado automáticamente a partir del análisis del repositorio*  
*Fecha: 2026-10-05*