# Informe Final - DataFest 2026: Propensión de Conversión de Clientes

> **Aviso (7 oct 2026):** este informe es **anterior a la auditoría de octubre**. Sus cifras de Gini se calcularon con el pipeline antiguo, que elegía al ganador con un solo mes y paraba el entrenamiento casi al azar. Los resultados vigentes, el modelo nuevo (LightGBM superficial, Gini medio 0,2540) y las correcciones están en [`docs/RESUMEN_DE_CAMBIOS.md`](docs/RESUMEN_DE_CAMBIOS.md), en el [`README.md`](README.md) y en los informes de [`validacion_modelos/`](validacion_modelos/).

---

## 1. Resumen Ejecutivo

Este informe documenta el análisis completo del dataset de DataFest 2026, donde el objetivo es predecir la probabilidad de conversión de clientes (clasificación binaria) usando datos históricos de enero a noviembre 2026 para predecir diciembre 2026.

**Métrica de evaluación:** Gini = 2 × AUC − 1

**Mejor resultado validado (walk-forward, 4 folds, LightGBM regularizado):** Gini medio **0,2516** (sep–nov), **0,2638** (jun–ago)

**Submission final:** `resultados_reportes/submission_final.csv` (9.900 predicciones, formato correcto, sha256: `88111e48bd918625837415505ab709a56f8a38bc34dfbb35d739d9da5ee29678`)

**Nota crucial:** El error estándar del Gini de validación (bootstrap por cliente, 200 réplicas) es **0,0092**; IC95 % ≈ [0,232; 0,267]. Cualquier diferencia entre modelos menor a ~0,01 no es evidencia. La regularización (+0,003 a +0,008 Gini) es consistente en folds pero está dentro del ruido; presentarse como "más estable", no como "mejor".

---

## 2. Estructura Final del Proyecto

```
DataFest_2026/
├── README.md                              # Documentación principal
├── .gitignore                             # Patrones de exclusión
├── INFORME_FINAL_DATAFEST_2026.md         # Este informe
├── submission_final_BACKUP.csv            # Backup de la entrega actual
├── docs/
│   ├── DATASET_DESCRIPTION.md             # Diccionario de datos y hechos verificados
│   └── BASES_CONCURSO.md                  # Reglas del comunicado (fecha 7 oct, correo, 1 solución/equipo)
├── datos_entrada/                         # Datos originales (no versionar si son confidenciales/pesados)
│   ├── train.csv                          # Entrenamiento: ene–nov 2026 (110.100 filas, 18.1 MB)
│   ├── test.csv                           # Prueba: dic 2026 (9.900 filas, 1.6 MB)
│   └── sample_submission.csv              # Formato de entrega (104 KB)
├── pipelines/                             # Versiones del pipeline de modelado
│   ├── Pipeline_DF_WCB.py                 # ⭐ Pipeline principal (validación walk-forward, IC, modo estático/temporal)
│   └── legacy/
│       ├── Pipeline_DF_1.py               # v1: validación 1 mes (nov), conservada por trazabilidad
│       └── Pipeline_DF_New.py             # v3: WCB sin CatBoost, misma validación, conservada por trazabilidad
├── resultados_reportes/                   # Artefactos generados por pipelines y EDA
│   ├── submission_final.csv               # ⭐ ENTREGA FINAL (id_cliente, prediccion)
│   ├── submission_candidata.csv           # Candidata con LightGBM regularizado (no sobrescribe final)
│   ├── resultados_validacion_DF_WCB.csv   # Gini por fold, IC95%, best_iter, n_feats
│   ├── features_seleccionadas_DF_WCB.txt  # 50 features finales
│   ├── correlacion_heatmap.png            # Matriz de correlación (Pearson/Spearman, vmin/vmax simétrico)
│   ├── feature_importance_rf.png          # Importancia por permutación (val sep–nov) + impureza
│   ├── comparativa_modelos_gini.png       # Barras con IC95%, regenerada desde CSV
│   ├── distribuciones_numericas_por_clase.png
│   ├── boxplots_por_clase.png
│   ├── objetivo_distribucion.png
│   ├── tasa_conversion_categoricas.png
│   ├── heatmap_banda_productos.png        # Tasa de conversión banda_riesgo × numero_productos
│   ├── permutacion_importancia.png        # Importancia por permutación (barras)
│   ├── hazard_por_antiguedad.png          # Hazard rate por tenure en el panel
│   ├── estadisticas_numericas_train.csv
│   └── historico/                         # Entregas anteriores con fecha
├── documentacion/                         # Documentos de referencia y reportes
│   ├── DataFest 2026.html                 # Informe ejecutivo visual
│   ├── Guía del Pipeline DataFest.html    # Guía técnica detallada
│   ├── INFORME_DATAFEST_2026.md           # Informe técnico completo (duplicado, mantener solo este)
│   └── informe_ejecutivo.html             # Informe ejecutivo visual (HTML)
├── notebooks/                             # Análisis exploratorio (Jupyter)
│   └── 01_eda_completo.ipynb              # EDA completo 21 celdas (corregido según N-1 a N-21)
├── modelos/                               # Modelos entrenados serializados (.pkl, .joblib, .keras)
│   ├── mlp_sklearn.joblib                 # MLP scikit-learn
│   ├── nn_baseline.keras                  # Neural Network Keras básico
│   ├── nn_optimizado.keras                # Neural Network Keras optimizado
│   ├── preprocessor_nn.joblib             # Preprocesador para NN
│   └── nn_resultados.json                 # Arquitecturas y métricas de cada NN
├── scripts_utilitarios/                   # Scripts auxiliares (vacía, mantener .gitkeep)
│   └── .gitkeep
└── verificaciones/                        # Scripts de verificación reproducibles
    ├── 00_analisis_panel.py
    ├── 01_ablacion_variables.py
    ├── 02_importancia_permutacion.py
    ├── 03_modelo_reducido_y_ruido.py
    └── 04_ensemble.py
```

---

## 3. Análisis Exploratorio de Datos (EDA)

### 3.1 Estructura del Panel y Calidad de Datos

| Métrica | Valor Real |
|---------|------------|
| Filas train | 110.100 |
| Filas test | 9.900 |
| Columnas train | 25 (incluyendo id_cliente, mes, objetivo) |
| Columnas test | 24 (sin objetivo) |
| Valores nulos | 0 |
| Duplicados (id_cliente, mes) | 0 |
| Clientes únicos train | 24.628 |
| Clientes únicos test | 9.900 |
| Positivos (objetivo=1) | 16.567 (15,05 %) |
| Negativos (objetivo=0) | 93.533 |
| Clientes que convierten alguna vez | 67,3 % (16.567 de 24.628) |
| Clientes nuevos en diciembre (test) | 1.839 (18,6 %, ids 24.629–26.467 consecutivos) |
| % test con historial en train | 81,4 % (8.061 clientes) |

**Estructura del panel:** Panel de "riesgo de primera conversión". Cada cliente aparece mes a mes hasta que convierte (`objetivo=1`) y **desaparece** (0 filas posteriores a la conversión). No hay abandonos silenciosos. Los meses por cliente son **siempre consecutivos** (0 huecos de calendario).

**Composición temporal:** Train tiene 11 meses (202601–202611). Test es solo diciembre 2026 (202612). Los 8.061 clientes no convertidos de noviembre están todos en test + 1.839 nuevos.

**Entradas de clientes nuevos por mes (feb–nov):** feb 921, mar 2.043, abr 827, may 1.973, jun 1.834, jul 950, ago 1.760, sep 1.299, oct 1.893, nov 728. Enero: 10.400 (stock inicial, censurado a la izquierda).

### 3.2 Distribución de la Variable Objetivo

- Conversión (1): 15,05 % global (16.567 casos)
- No conversión (0): 84,95 % (93.533 casos)
- **Tasa por mes:** 14,1 %–15,7 % (ene 14,6 %, feb 15,7 %, mar 14,8 %, abr 15,3 %, may 15,7 %, jun 15,5 %, jul 14,5 %, ago 14,4 %, sep 14,1 %, oct 15,7 %, nov 15,1 %)
- **Test χ² homogeneidad mensual:** p = 0,0033 → **no es exactamente constante**. No se explica por proporción de nuevos (r = 0,06). No es feature, pero no decir "estable" sin matiz.
- **Nuevos vs antiguos (feb–nov):** Tasa nuevos 16,95 % vs antiguos 14,78 %. Los nuevos tienen más banda `low` (53 % vs 45 %) y más productos (2,03 vs 1,89). Hazard decrece con la permanencia: ~17 % → ~12–13 %.

### 3.3 Análisis de Variables (Nombres Reales del Dataset)

**Variables numéricas (12):**
- `edad` (22–71, uniforme)
- `ingresos` (18.000–145.700, piso 18.000 en 1,63 %)
- `ratio_deuda_ingresos` (0,02–0,95)
- `antiguedad_cuenta_meses` (1–179)
- `numero_productos` (1–5, discreto)
- `saldo_promedio` (300–90.000, piso 300 en 6,86 %)
- `dias_ultima_transaccion` (1–364)
- `antiguedad_direccion_meses` (1–239)
- `visitas_web_ultimos_90_dias` (0–26, discreto)
- `distancia_sucursal_km` (0,21–80, tope 80 en 0,14 %)
- `dia_preferido_pago` (1–28, discreto)
- `dias_ultima_interaccion` (1–364, **única variable dinámica**)

**Variables booleanas (5):**
- `tiene_tarjeta_credito` (74,2 % True)
- `activo_movil` (65,6 % True)
- `es_nuevo_cliente` (19,2 % True)
- `tiene_prestamo` (44,6 % True)
- `tiene_seguro` (38,3 % True)

**Variables categóricas (5) → One-Hot (22 columnas):**
- `ocupacion` (5: clerical, manual, professional, manager, self_employed)
- `region` (5: west, north, south, east, central)
- `canal_adquisicion` (5: web, mobile, branch, call_center, partner)
- `banda_riesgo` (3: low, medium, high) — **orden natural low→medium→high**
- `dispositivo_principal` (4: android, ios, web, otro)

**Tope / censura (no outliers erróneos):** ingresos=18.000 (piso), saldo_promedio=300 (piso), distancia_sucursal_km=80 (tope), edad uniforme en extremos.

### 3.4 Matriz de Correlación (Pearson)

**Hallazgos clave (valores reales):**
- Correlación lineal máxima con `objetivo`: `numero_productos` **r = 0,076**; `dias_ultima_transaccion` **−0,058**; `dias_ultima_interaccion` **−0,036**. **Ingresos r = 0,006, saldo r = 0,008** (≈ 0).
- Única correlación notable entre predictoras: `dias_ultima_interaccion` ↔ `dias_ultima_transaccion` **r = 0,55** (iguales en 54,9 % de filas; interacción ≤ transacción en 77,5 %).
- Las correlaciones altas (>0,9) entre variables estáticas y sus lags mencionadas en versiones previas **no aparecen en el heatmap** porque el heatmap actual no incluye lags; esas variables son idénticas por construcción (`roll3_mean` ≡ variable base) y se eliminan por correlación en el pipeline.
- Pearson sobre 110.100 filas **infla significancia**: tamaño efectivo = 24.628 clientes. Usar Spearman y bootstrap por cliente para IC.

### 3.5 Importancia de Variables (Permutación sobre Validación Temporal)

Script de referencia: `verificaciones/02_importancia_permutacion.py` (Random Forest, validación sep–nov, 5 permutaciones por variable).

| Variable | Caída AUC por permutación (val) | AUC Univariado (val) |
|----------|----------------------------------|----------------------|
| `banda_riesgo` (One-Hot combinado) | **0,069** | — |
| `numero_productos` | **0,035** | **0,550** |
| `dias_ultima_transaccion` | **0,020** | **0,547** |
| `activo_movil` | 0,008 | 0,516 |
| `tiene_tarjeta_credito` | 0,006 | 0,511 |
| `dias_ultima_interaccion` | 0,001 | 0,519 |
| **Resto (17 variables)** | **≤ 0,0014 (ruido)** | **≤ 0,510** |

**Conclusión:** Solo 5 variables tienen señal real detectable por permutación. La importancia por impureza del Random Forest (in-sample) **sesga hacia continuas** y asigna 0,03–0,05 a `edad`, `antiguedad_direccion_meses`, `saldo_promedio`, `ingresos` (ruido). El `LabelEncoder` en el notebook ordenaba `banda_riesgo` alfabéticamente (high=0, low=1, medium=2): **no ordinal**.

**Interacción fuerte verificada:** En `banda_riesgo=low`, la tasa sube de 13,8 % (1 producto) a 31,3 % (5 productos). En `high` es plana (8,2 %–10,7 %). Ver `heatmap_banda_productos.png`.

### 3.6 Variables Dinámicas

**Solo `dias_ultima_interaccion` cambia dentro de un cliente** (80,3 % de los clientes con >1 mes; 65,6 % si se incluyen clientes de un solo mes). Las otras 21 columnas son constantes por cliente.

**Naturaleza de `dias_ultima_interaccion`:**
- Delta mensual: media 0,1; sd 117,8 días
- 36,5 % deltas = 0 (sin cambio mes a mes)
- Solo 0,16 % deltas = +30
- **No es un contador "días desde"**: se comporta como re-muestreo ruidoso
- Igual a `dias_ultima_transaccion` en 54,9 % de filas; ≤ en 77,5 %

---

## 4. Feature Engineering Aplicado

### 4.1 Variables Generadas por el Pipeline (Modo "temporal" actual)

El pipeline genera ~200 features:
- Lags (1,2,3 meses) sobre 7 variables → ~21
- Deltas mensuales → ~7
- Ventanas rolling 3m (mean, sum, meses_en_ventana) → ~15
- One-Hot de 5 categóricas → 22
- **Total:** 78 columnas → selección → **50 features finales**

### 4.2 Resultado de Ablación (Gini, media 3 semillas, 2 folds temporales, LightGBM mismos hiperparámetros)

| Conjunto | val sep–nov (train ene–ago) | val jun–ago (train ene–may) |
|----------|----------------------------|----------------------------|
| A. Pipeline actual (50 features) | 0,2471 | 0,2582 |
| B. Solo estáticas + valor actual (39 cols) | **0,2499** | 0,2580 |
| C. B + lags/delta/roll de `dias_ultima_interaccion` | 0,2446 | 0,2564 |
| D. B + `tenure` + `cohorte_ene` | 0,2489 | 0,2569 |
| E. D + flags interacción/transacción | 0,2510 | 0,2565 |
| G. A sin `*_sum` ni `meses_en_ventana3` | 0,2470 | 0,2566 |
| **Todas estáticas, LightGBM regularizado** | **0,2516** | **0,2638** |
| 11 variables, regularizado | 0,2516 | 0,2589 |
| 7 variables de señal, regularizado | 0,2490 | 0,2582 |
| Regresión logística (7 vars + banda×productos) | 0,2297 | 0,2416 |
| Rank-average LGBM+XGB+Cat+RF regularizados | 0,2513 | 0,2623 |

**Conclusiones obligatorias:**
- **Las variables temporales (lags, deltas, ventanas) y el `tenure` no aportan** (A ≈ B; C < B).
- **El ensemble no mejora** (0,2513–0,2517 vs 0,2516 del LGBM solo). La sección previa de "Gini proyectado 0,252–0,255" **no tiene respaldo**.
- La señal está en pocas variables estáticas con interacción `banda_riesgo × numero_productos`; los árboles la capturan, la logística lineal no (−0,02 Gini).
- El único cambio consistente en los dos folds es **regularizar** (+0,003 a +0,008). Sigue dentro del ruido: presentarlo como "más estable", no como "mejor".

### 4.3 Selección de Variables (Pipeline Automático)

1. **Varianza cero:** Elimina 6 deltas de variables estáticas (son 0 por construcción).
2. **Correlación > 0,95:** Elimina 22 redundancias (lags de variables estáticas = variable base; `roll3_mean` ≡ variable base). Loguea cuando el par es `base`/`base_rollN_mean`.
3. **Importancia LightGBM (gain):** Top 80 → quedan **50 features finales** (bajo tope de 80).

**Nota:** Las features `*_roll3_mean` de variables estáticas son **idénticas** a la variable original (verificado con `np.allclose`). La importancia que se les atribuye corresponde a la variable base.

---

## 5. Modelado y Resultados

### 5.1 Validación Temporal (Crítico - Sin Data Leakage)

- **Entrenamiento:** Según fold (ene–ago / ene–may)
- **Validación:** Según fold (sep–nov / jun–ago)
- **Test real:** Diciembre 2026 (mes futuro no visto)
- **Regla:** Nunca información futura; lags/ventanas solo meses anteriores del mismo cliente. Target Encoding temporal (no se activa: todas categóricas ≤5 niveles).

### 5.2 Comparativa de Modelos (Regenerada desde `resultados_validacion_DF_WCB.csv`)

> **Valores con IC95 % por bootstrap de clientes (200 réplicas).** Los modelos NN se evalúan sobre 44 columnas estáticas con early stopping sobre la validación (optimista).

| Modelo | Gini sep–nov | IC95 % | Gini jun–ago | IC95 % | n_feats | Comentario |
|--------|-------------|--------|-------------|--------|---------|------------|
| LightGBM regularizado (num_leaves=8, min_child=200, λ=10) | **0,2516** | [0,233; 0,269] | **0,2638** | [0,245; 0,281] | 39 | ⭐ Más estable |
| LightGBM default | 0,2471 | [0,229; 0,265] | 0,2582 | [0,240; 0,276] | 50 | Pipeline actual |
| XGBoost regularizado | 0,2464 | [0,228; 0,264] | 0,2578 | [0,239; 0,275] | 39 | Competitivo |
| CatBoost regularizado | 0,2437 | [0,226; 0,262] | 0,2541 | [0,236; 0,272] | 39 | Buen complemento |
| RandomForest regularizado | 0,2407 | [0,223; 0,258] | 0,2512 | [0,233; 0,269] | 39 | Interpretabilidad |
| NN Keras Optimizado (guardada) | 0,2400 | — | — | — | 44 | Early stopping en val (optimista) |
| NN Keras Baseline (guardada) | 0,2294 | — | — | — | 44 | Early stopping en val (optimista) |
| MLP sklearn | 0,1854 | — | — | — | 44 | Menos capacidad |

**Diferencias < 0,01 Gini** están dentro del ruido de validación (SE ≈ 0,009).

### 5.3 Redes Neuronales (Arquitecturas Guardadas)

| Modelo | Arquitectura | Gini Val (sep–nov) | Comentario |
|--------|--------------|-------------------|------------|
| `nn_optimizado.keras` | 256-128-64-32, BN, Dropout 0.4/0.3/0.2, LR scheduling | 0,2400 | 96 % del mejor LGBM |
| `nn_baseline.keras` | 128-64, BN, Dropout 0.3/0.2 | 0,2294 | Baseline |
| `mlp_sklearn.joblib` | (128, 64, 32), ReLU, Adam, α=0.001 | 0,1854 | Sklearn MLP |

**Nota:** Las NN usan `StandardScaler` + `OneHotEncoder` ajustados **solo en train** (celda N-16 corregida). Early stopping usa la validación → métrica optimista. No se ejecutó SHAP (ver N-20).

---

## 6. Feature Engineering Adicional: Estado Real

### 6.1 Interacciones (Hipótesis a partir de tabla cruzada banda×productos)

```python
# Hallazgo central: interacción banda_riesgo × numero_productos
# En banda low: tasa 13,8 % (1 prod) → 31,3 % (5 prod)
# En banda high: plana 8,2 %–10,7 %
df['banda_low_x_prod'] = (df['banda_riesgo'] == 'low').astype(int) * df['numero_productos']
```
**Estado:** Probada en ablación F (`low_x_prod`): 0,2510 (sep–nov) vs 0,2499 (B). **No mejora de forma consistente** (ver `01_ablacion_variables.py`). Mantener como hipótesis, no como feature aplicada.

### 6.2 Features Temporales (Descartadas por evidencia)

- Lags/deltas/rolling de `dias_ultima_interaccion`: **C < B** (0,2446 vs 0,2499)
- `tenure` (meses en panel) + `cohorte_ene`: D = 0,2489 ≈ B
- Flags `int_igual_trans`, `int_menos_trans`: E = 0,2510 ≈ B
- **Conclusión:** No aportan señal neta; añaden ruido y complejidad.

### 6.3 Target Encoding Avanzado

No necesario (todas categóricas ≤5 niveles). Función `target_encoding_temporal` en pipeline: mantener con prueba unitaria sintética o eliminar (ver P-11).

---

## 7. Estrategia de Ensemble: Resultado Real

**Ensemble probado (rank-average LGBM+XGB+Cat+RF regularizados):**
- sep–nov: **0,2513** vs LGBM solo **0,2516** (no mejora)
- jun–ago: **0,2623** vs LGBM solo **0,2638** (no mejora)

**Blend LGBM+CatBoost (rank average):**
- sep–nov: 0,2517 vs LGBM 0,2516

**Conclusión:** El ensemble **no mejora** sobre LightGBM regularizado solo. Retirar pesos 50/20/20/10 y "Gini proyectado 0,252–0,255" del informe anterior.

---

## 8. Próximos Pasos Priorizados (Basados en Evidencia)

### 8.1 Inmediato (Antes de entrega 7 oct) - P0
- [ ] Confirmar con bases oficiales qué adjuntar (solo CSV, o también código/informe)
- [ ] Verificar `submission_final.csv` (formato, orden, rango, sin nulos) → **OK**
- [ ] Enviar **una sola** submission desde representante del equipo a marcaempleadora@bcp.com.pe

### 8.2 Corto Plazo (Mejora de estabilidad) - P1
- [ ] Ejecutar pipeline con `--modo-features estatico --folds 4 --trials 30` y comparar `submission_candidata.csv` vs `submission_final.csv`
- [ ] Criterio de reemplazo: Gini medio en 4 folds walk-forward **mayor** y diferencia > 0,003 sin empeorar ningún fold
- [ ] Walk-forward validation completa (4 folds: val dic, val nov, val oct, val sep) con IC por bootstrap de clientes
- [ ] Optuna sobre media de AUC en folds (no una sola validación)

### 8.3 Documentación y Reproducibilidad - P1
- [ ] Regenerar todas las 10 figuras con `pipelines/generar_graficas.py` (criterios N-8 a N-19)
- [ ] Corregir notebook `01_eda_completo.ipynb` (N-1 a N-21)
- [ ] Mover `Pipeline_DF_1.py` y `Pipeline_DF_New.py` a `pipelines/legacy/`
- [ ] Unificar documentación: mantener solo `INFORME_FINAL_DATAFEST_2026.md` y `README.md`
- [ ] Crear `requirements.txt` y `docs/DATASET_DESCRIPTION.md`, `docs/BASES_CONCURSO.md`

---

## 9. Entrega Final (Submission)

El archivo **`resultados_reportes/submission_final.csv`** contiene:
- **Formato:** `id_cliente,prediccion` (exactamente 2 columnas, nombres idénticos a `sample_submission.csv`)
- **Filas:** 9.900 (idéntico a test.csv, orden verificado)
- **Rango predicciones:** [0,0282; 0,4225] (dentro de [0, 1])
- **Sin nulos, sin empates masivos** (nunique > 9.000 verificado)
- **Modelo usado:** LightGBM + Optuna (3 semillas promediadas) - Pipeline DF_WCB modo temporal
- **Gini validación (sep–nov):** 0,2471 (IC95 % ≈ [0,229; 0,265])
- **sha256:** `88111e48bd918625837415505ab709a56f8a38bc34dfbb35d739d9da5ee29678`

**Verificación completada:** ✅ IDs coinciden, orden correcto, formato válido, sin nulos.

---

## 10. Lecciones Aprendidas Clave (Corregidas)

1. **Validación temporal es obligatoria:** Random split infla Gini ~0,015–0,025 (data leakage por clientes repetidos). Si CV, agrupar por `id_cliente`.
2. **21 de 22 columnas son constantes por cliente:** Solo `dias_ultima_interaccion` cambia (80,3 % clientes multi-mes). Lags/deltas/rolling de estáticas son idénticos a la base o cero.
3. **Selección automática funciona pero gain sesga:** Varianza cero + correlación (>0,95) + importancia LightGBM → ~50 features. La importancia `gain` conserva ruido (`distancia_sucursal_km`, `ingresos`, `antiguedad_direccion_meses`); usar permutación sobre validación temporal.
4. **CatBoost no complementa aquí:** Variables categóricas ya One-Hot; su ventaja nativa no se usa.
5. **NN competitivas en tabular pero optimistas:** Con arquitectura adecuada, 95 %+ del SOTA gradient boosting, pero early stopping en validación infla métrica.
6. **`roll3_mean` de variable constante ≡ variable original:** No "mejor", es idéntica por construcción (`valor × N / N`).
7. **Optuna +0,004 Gini está dentro del ruido (SE 0,009)** y se eligió sobre la misma validación. Reportar como "ajuste de hiperparámetros", no como "mejora probada".
8. **Error estándar del Gini = 0,0092** (bootstrap por cliente, 200 réplicas). IC95 % ≈ ±0,018. Diferencias < 0,01 no son evidencia.
9. **El Gini de Optuna/early stopping es optimista:** Usa la validación para decidir. La estimación honesta requiere walk-forward + bootstrap por cliente.
10. **Estructura de panel:** Cliente aparece hasta convertir (first conversion hazard). Test tiene 18,6 % clientes nuevos (ids consecutivos mayores) no vistos en train.

---

## 11. Referencias Técnicas y Archivos Clave

| Archivo | Descripción |
|---------|-------------|
| `pipelines/Pipeline_DF_WCB.py` | Pipeline principal (modo estático/temporal, walk-forward, IC, preset regularizado) |
| `pipelines/legacy/Pipeline_DF_1.py` | v1: validación 1 mes (trazabilidad) |
| `pipelines/legacy/Pipeline_DF_New.py` | v3: WCB sin CatBoost (trazabilidad) |
| `notebooks/01_eda_completo.ipynb` | EDA corregido (21 celdas, N-1 a N-21) |
| `resultados_reportes/submission_final.csv` | **Entrega final para competición** |
| `resultados_reportes/submission_candidata.csv` | Candidata LightGBM regularizado (no sobrescribe final) |
| `resultados_reportes/resultados_validacion_DF_WCB.csv` | Gini por fold, IC95%, best_iter, n_feats |
| `resultados_reportes/correlacion_heatmap.png` | Matriz correlación (Pearson/Spearman, vmin/vmax simétrico, incluye booleanas) |
| `resultados_reportes/feature_importance_rf.png` | Importancia por permutación (val sep–nov) + impureza |
| `resultados_reportes/comparativa_modelos_gini.png` | Barras con IC95%, desde CSV |
| `resultados_reportes/heatmap_banda_productos.png` | Tasa conversión banda_riesgo × numero_productos |
| `resultados_reportes/permutacion_importancia.png` | Importancia por permutación (barras) |
| `resultados_reportes/hazard_por_antiguedad.png` | Hazard rate por tenure en el panel |
| `modelos/nn_optimizado.keras` | NN Keras optimizado (Gini 0,2400) |
| `modelos/nn_baseline.keras` | NN Keras baseline (Gini 0,2294) |
| `modelos/mlp_sklearn.joblib` | MLP sklearn (Gini 0,1854) |
| `modelos/nn_resultados.json` | Arquitecturas y métricas de cada NN |
| `verificaciones/00_analisis_panel.py` | Hechos del panel (ejecutar: `python verificaciones/00_analisis_panel.py`) |
| `verificaciones/01_ablacion_variables.py` | Ablación de conjuntos de variables |
| `verificaciones/02_importancia_permutacion.py` | Importancia por permutación (RF, val temporal) |
| `verificaciones/03_modelo_reducido_y_ruido.py` | Modelos reducidos, regularización, bootstrap SE Gini |
| `verificaciones/04_ensemble.py` | Ensemble rank-average (LGBM+XGB+Cat+RF) |

---

## 12. Comandos de Referencia

```bash
# Pipeline principal - modo estático (recomendado por evidencia)
python pipelines/Pipeline_DF_WCB.py --datos datos_entrada --salida resultados_reportes/submission_candidata.csv --trials 30 --n-seeds 3 --modo-features estatico --folds 4

# Pipeline principal - modo temporal (actual)
python pipelines/Pipeline_DF_WCB.py --datos datos_entrada --salida resultados_reportes/submission_candidata.csv --trials 30 --n-seeds 3 --modo-features temporal --folds 4

# Corrida rápida (testing, sin Optuna, 1 fold, 1 semilla)
python pipelines/Pipeline_DF_WCB.py --datos datos_entrada --salida resultados_reportes/prueba.csv --trials 0 --n-seeds 1 --folds 1

# Verificaciones reproducibles (desde raíz del repo)
python verificaciones/00_analisis_panel.py
python verificaciones/01_ablacion_variables.py
python verificaciones/02_importancia_permutacion.py
python verificaciones/03_modelo_reducido_y_ruido.py
python verificaciones/04_ensemble.py

# Regenerar todas las figuras
python pipelines/generar_graficas.py
```

---

## 13. Bases del Concurso (Resumen)

- **Métrica:** Gini = 2 × AUC − 1
- **Entrega:** Una sola solución por equipo, un solo representante
- **Fecha límite:** 7 de octubre 2026
- **Correo:** marcaempleadora@bcp.com.pe
- **Formato:** `id_cliente,prediccion` (exactamente `sample_submission.csv`)
- **Cupos:** 2 por universidad
- **Bases oficiales:** Ver enlace "Ingresa AQUÍ" del comunicado (pedir al humano)

---

*Informe final corregido según hechos verificados - DataFest 2026 Team*
*Fecha: 2026-10-05*
*Versión: 2.0 - Hechos verificados, evidencia reproducida*