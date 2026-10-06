# DataFest 2026 — Propensión de Conversión de Clientes

Documento interno del equipo. Explica cómo está organizado el proyecto, cómo preparar el entorno y cómo ejecutar el pipeline que genera el archivo de entrega.

- **Problema:** estimar la probabilidad de que un cliente convierta (`objetivo = 1`) en un mes dado.
- **Métrica:** Gini = 2 · AUC − 1 (solo importa el *orden* de las predicciones, no su calibración).
- **Datos:** `train.csv` (enero–noviembre 2026, 110.100 filas) y `test.csv` (diciembre 2026, 9.900 filas). Cada fila es un cliente-mes; un cliente puede aparecer en varios meses.
- **Estructura del panel:** "riesgo de primera conversión". El cliente aparece cada mes hasta que convierte y entonces desaparece (0 filas post-conversión). Meses consecutivos sin huecos. Test (dic 2026) tiene 81,4 % clientes con historial + 18,6 % nuevos (ids 24.629–26.467).

---

## 1. Estructura del Proyecto

```text
DataFest_2026/
├── README.md                              # Este documento
├── .gitignore                             # Patrones de exclusión
├── INFORME_FINAL_DATAFEST_2026.md         # Informe final corregido (hechos verificados)
├── submission_final_BACKUP.csv            # Backup de la entrega actual
├── docs/
│   ├── DATASET_DESCRIPTION.md             # Diccionario de datos y hechos verificados (sección 1 del informe)
│   └── BASES_CONCURSO.md                  # Reglas del comunicado (fecha 7 oct, correo, 1 solución/equipo)
├── datos_entrada/                         # Datos originales (pesados/confidenciales)
│   ├── train.csv                          # Entrenamiento: ene–nov 2026 (con objetivo)
│   ├── test.csv                           # Prueba: dic 2026 (sin objetivo)
│   └── sample_submission.csv              # Formato de entrega de referencia
├── pipelines/                             # Pipeline principal
│   ├── Pipeline_DF_WCB.py                 # ⭐ Pipeline principal (walk-forward, IC, modo estático/temporal)
│   └── legacy/
│       ├── Pipeline_DF_1.py               # v1: validación 1 mes (nov), conservada por trazabilidad
│       └── Pipeline_DF_New.py             # v3: WCB sin CatBoost, misma validación, conservada por trazabilidad
├── resultados_reportes/                   # Artefactos generados por los pipelines
│   ├── submission_final.csv               # ⭐ ENTREGA FINAL (id_cliente, prediccion, sha256 en README)
│   ├── submission_candidata.csv           # Candidata LightGBM regularizado (no sobrescribe final)
│   ├── resultados_validacion_DF_WCB.csv   # Gini por fold, IC95%, best_iter, n_feats
│   ├── features_seleccionadas_DF_WCB.txt  # 50 features finales
│   ├── correlacion_heatmap.png            # Matriz correlación (Pearson/Spearman, simétrico)
│   ├── feature_importance_rf.png          # Importancia por permutación (val sep–nov) + impureza
│   ├── comparativa_modelos_gini.png       # Barras con IC95%, desde CSV
│   ├── distribuciones_numericas_por_clase.png
│   ├── boxplots_por_clase.png
│   ├── objetivo_distribucion.png
│   ├── tasa_conversion_categoricas.png
│   ├── heatmap_banda_productos.png        # Tasa conversión banda_riesgo × numero_productos
│   ├── permutacion_importancia.png        # Importancia por permutación (barras)
│   ├── hazard_por_antiguedad.png          # Hazard rate por tenure
│   ├── estadisticas_numericas_train.csv
│   └── historico/                         # Entregas anteriores con fecha
├── documentacion/                         # Documentos de referencia
│   ├── DataFest 2026.html                 # Informe ejecutivo visual
│   ├── Guía del Pipeline DataFest.html    # Guía técnica detallada
│   ├── INFORME_DATAFEST_2026.md           # Duplicado del informe final (eliminar o sincronizar)
│   └── informe_ejecutivo.html             # Informe ejecutivo visual
├── notebooks/                             # Análisis exploratorio (Jupyter)
│   └── 01_eda_completo.ipynb              # EDA completo 21 celdas (corregido N-1 a N-21)
├── modelos/                               # Modelos entrenados serializados
│   ├── mlp_sklearn.joblib
│   ├── nn_baseline.keras
│   ├── nn_optimizado.keras
│   ├── preprocessor_nn.joblib
│   └── nn_resultados.json                 # Arquitecturas y métricas de cada NN
├── scripts_utilitarios/                   # Scripts auxiliares
│   └── .gitkeep
└── verificaciones/                        # Scripts de verificación reproducibles
    ├── 00_analisis_panel.py
    ├── 01_ablacion_variables.py
    ├── 02_importancia_permutacion.py
    ├── 03_modelo_reducido_y_ruido.py
    └── 04_ensemble.py
```

Dentro del pipeline principal el código está organizado en secciones:

| Sección | Qué contiene |
|---|---|
| `Config` | Todos los parámetros (columnas, lags, umbrales, nº de trials, modo_features, folds…). Es el único lugar que se edita para ajustar el comportamiento. |
| Carga y diagnóstico | `cargar_datos`, `validar_esquema`, `diagnostico_inicial` |
| Ingeniería de características | `crear_features_temporales` (solo variables dinámicas), `target_encoding_temporal`, `construir_dataset` |
| Selección | `SelectorVarianzaCero`, `SelectorCorrelacion`, `SelectorImportanciaLGBM` (con opción permutación), `seleccionar_features` |
| Modelado | `EvaluadorModelos` (LightGBM, XGBoost, CatBoost, RF + Optuna, walk-forward, IC bootstrap) y `construir_modelo` |
| Entrega | `entrenar_y_predecir_final`, `exportar_submission` |
| `main` | Orquesta todo en orden |

---

## 2. Instalación

Requiere **Python 3.10 o superior**. Se probó con Python 3.13, pandas 3.0, scikit-learn 1.9, LightGBM 4.7, XGBoost 3.4, CatBoost 1.2 y Optuna 5.0. Versiones mínimas: scikit-learn ≥ 1.2 y xgboost ≥ 1.6.

Recomendado: usar un entorno virtual para no mezclar librerías con otros proyectos.

**Windows (Símbolo del sistema):**
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -U pandas numpy scikit-learn lightgbm xgboost catboost optuna
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U pandas numpy scikit-learn lightgbm xgboost catboost optuna
```

Para comprobar que quedó bien:
```bash
python -c "import pandas, sklearn, lightgbm, xgboost, catboost, optuna; print('OK')"
```

> **Mac con chip Apple:** si LightGBM falla al importar con un error sobre `libomp`, instala OpenMP con `brew install libomp`.

Para NN (notebook): `pip install tensorflow`

---

## 3. Ejecución

Los comandos se escriben en la **terminal** (no dentro de Python ni de un notebook). Si usas Jupyter o Colab, antepón `!` al comando.

1. Los CSV ya están en `datos_entrada/`.
2. Abre la terminal en la carpeta del proyecto (con el entorno virtual activado).
3. Ejecuta:

```bash
# Pipeline principal - modo estático (recomendado por evidencia de ablación)
python pipelines/Pipeline_DF_WCB.py --datos datos_entrada --salida resultados_reportes/submission_candidata.csv --trials 30 --n-seeds 3 --modo-features estatico --folds 4

# Pipeline principal - modo temporal (comportamiento anterior)
python pipelines/Pipeline_DF_WCB.py --datos datos_entrada --salida resultados_reportes/submission_candidata.csv --trials 30 --n-seeds 3 --modo-features temporal --folds 4

# Opciones útiles:
# --modo-features estatico|temporal   # Solo estáticas+valor actual vs lags/deltas/rolling
# --folds 4                           # Walk-forward: 4 folds (val dic, nov, oct, sep)
# --trials 30                         # Trials Optuna (0 = desactiva, más rápido)
# --n-seeds 3                         # Semillas a promediar en modelo final
# --top-k 80                          # Máx. features tras selección
```

### Argumentos Principales

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--datos` | `.` | Carpeta donde están los CSV de entrada. |
| `--salida` | `submission.csv` | Ruta del archivo de entrega. Los archivos auxiliares se guardan en la misma carpeta. |
| `--trials` | `30` | Nº de pruebas de Optuna para ajustar LightGBM. `0` desactiva Optuna (más rápido). |
| `--folds` | `4` | Nº de folds walk-forward (4 = dic, nov, oct, sep). |
| `--modo-features` | `estatico` | `"estatico"` (solo originales + valor actual interacción) o `"temporal"` (con lags/deltas/rolling). |
| `--top-k` | `80` | Máximo de variables a conservar tras la selección. |
| `--n-seeds` | `3` | Semillas que se promedian en el modelo final. |
| `--semilla` | `42` | Semilla base (reproducibilidad). |
| `--bootstrap-ic` | `False` | Activar bootstrap por cliente para IC95% en validación. |
| `--n-bootstrap` | `1000` | Nº de iteraciones bootstrap para IC95%. |

Ejemplos útiles:
```bash
# Corrida rápida para probar cambios (sin Optuna, 1 fold, 1 semilla, modo estático)
python pipelines/Pipeline_DF_WCB.py --datos datos_entrada --salida resultados_reportes/prueba.csv --trials 0 --n-seeds 1 --folds 1 --modo-features estatico

# Validación walk-forward completa (4 folds, sin Optuna)
python pipelines/Pipeline_DF_WCB.py --datos datos_entrada --salida resultados_reportes/submission_candidata.csv --trials 0 --n-seeds 3 --folds 4 --modo-features estatico

# Pipeline completo con Optuna, bootstrap IC y modo estático (producción)
python pipelines/Pipeline_DF_WCB.py --datos datos_entrada --salida resultados_reportes/submission_candidata.csv --trials 30 --n-seeds 3 --folds 4 --modo-features estatico --bootstrap-ic --n-bootstrap 1000
```

### Qué hace, paso a paso

1. **Carga y validación:** comprueba columnas, que no haya duplicados cliente-mes, estructura de panel (test > max train mes, sin huecos, sin post-conversión, nuevos en test), y muestra diagnóstico (entradas nuevos por mes, hazard por tenure, dinámicas, pisos/topes).
2. **Features:** según `--modo-features`: `estatico` = solo columnas originales + One-Hot + valor actual de `dias_ultima_interaccion`; `temporal` = añade lags/deltas/rolling **solo de variables dinámicas detectadas** (hoy solo `dias_ultima_interaccion`).
3. **Selección:** elimina constantes, luego variables con correlación de Pearson > 0,95 (loguea pares base/roll_mean), luego selecciona por importancia (gain o permutación opcional).
4. **Validación:** walk-forward (folds temporales). Por fold: entrena y compara LightGBM, XGBoost, CatBoost, RandomForest + Optuna (sobre media AUC en folds); reporta AUC, Gini, IC95 % (bootstrap por cliente, 200 réplicas), best_iter.
5. **Entrega:** reentrena el ganador con el 100 % de `train.csv` (n_iter = mediana best_iter × factor_datos), predice `test.csv` y exporta `submission.csv` con comprobaciones estrictas (columnas exactas, orden, rango, sin nulos, nunique > 9000, sha256 en log).

---

## 4. Reglas del Proyecto (léanlas antes de modificar el código)

- **Validación temporal, no aleatoria.** No usar `train_test_split` aleatorio. Un mismo cliente aparece en varios meses con casi todas sus variables fijas, y el test es un mes futuro; una partición aleatoria infla la métrica (Gini +0,015–0,025). Si CV, agrupar por `id_cliente`.
- **Cero información del futuro.** Los lags y ventanas solo miran meses anteriores del mismo cliente. El Target Encoding de una fila usa únicamente objetivos de meses anteriores. Nunca crear variables con datos de meses posteriores.
- **No usar `id_cliente` ni `mes` como predictoras.** Diciembre es un mes no visto durante el entrenamiento. El `id_cliente` crece con la fecha de ingreso (los nuevos de diciembre tienen ids consecutivos mayores).
- **La selección de variables se ajusta solo con los meses de entrenamiento**, no con los de validación.
- **El orden de `submission.csv` debe ser idéntico al de `test.csv`.** El pipeline lo verifica antes de guardar; si la validación falla, no escribe el archivo.
- **Formato de entrega:** exactamente dos columnas, `id_cliente,prediccion`, con `prediccion` entre 0 y 1. No incluir `mes`.
- **Ruido de la métrica:** error estándar del Gini = 0,0092 (bootstrap por cliente). IC95 % ≈ ±0,018. Diferencias < 0,01 no son evidencia. No afirmar "mejora" sin superar ese ruido en varios folds.

---

## 5. Qué Sabemos de los Datos (Hechos Verificados)

- **Estructura:** Panel de "riesgo de primera conversión". Cliente aparece mes a mes hasta convertir (`objetivo=1`) y desaparece. 0 filas post-conversión. 0 abandonos silenciosos. Meses consecutivos.
- **Train:** 110.100 filas × 25 cols; 24.628 clientes únicos; 16.567 positivos (15,05 %); 93.533 negativos.
- **Test:** 9.900 filas × 24 cols; mes 202612; 9.900 clientes únicos (8.061 con historial + 1.839 nuevos, ids 24.629–26.467).
- **Nulos / duplicados:** 0 / 0.
- **Variables:** 12 numéricas, 5 booleanas, 5 categóricas (22 One-Hot).
- **Tasa por mes:** 14,1 %–15,7 % (χ² p=0,0033: no exactamente constante; no se explica por % nuevos).
- **Nuevos vs antiguos (feb–nov):** Tasa 16,95 % vs 14,78 %. Nuevos: más banda `low` (53 % vs 45 %), más productos (2,03 vs 1,89). Hazard decrece con tenure: ~17 % → ~12–13 %.
- **Composición validación vs test:** Validación sep–nov: nuevos = 13,1 % (sep), 18,2 % (oct), **7,7 % (nov)**. Test: **18,6 %**. La validación subestima el peso de clientes nuevos.
- **Variables dinámicas:** **Solo `dias_ultima_interaccion`** cambia (80,3 % clientes multi-mes; 65,6 % incluyendo 1 mes). Las otras 21 columnas son constantes por cliente.
- **Naturaleza de `dias_ultima_interaccion`:** Delta mensual media 0,1, sd 117,8; 36,5 % ceros; 0,16 % +30. **No es contador "días desde"**: re-muestreo ruidoso. Igual a `dias_ultima_transaccion` en 54,9 %; ≤ en 77,5 %.
- **Topes/censura (no outliers):** ingresos=18.000 (1,63 %), saldo=300 (6,86 %), distancia=80 (0,14 %), edad uniforme extremos.
- **Señal real (permutación val):** banda_riesgo 0,069; numero_productos 0,035; dias_ultima_transaccion 0,020; activo_movil 0,008; tarjeta 0,006. Resto ≤ 0,0014 (ruido).
- **AUC univariado val:** numero_productos 0,550; dias_ultima_transaccion 0,547; dias_ultima_interaccion 0,519; activo_movil 0,516; tarjeta 0,511; resto ≤ 0,510.
- **Interacción fuerte:** banda `low` × productos: 13,8 % (1) → 31,3 % (5). Banda `high`: plana 8,2 %–10,7 %.
- **Correlación máxima con objetivo:** numero_productos r=0,076; dias_ultima_transaccion −0,058; dias_ultima_interaccion −0,036. Ingresos 0,006, saldo 0,008.
- **Ablación (Gini media 3 seeds, 2 folds):** Temporales no aportan (A≈B; C<B). Regularizar (+0,003 a +0,008) es lo único consistente. Ensemble no mejora.
- **`roll3_mean` de estática ≡ variable base:** `valor × N / N`. Importancia atribuida corresponde a la base.

---

## 6. Resultados de Referencia (Regenerados)

Validación walk-forward (regenerar con `--folds 4`):

| Modelo | Gini sep–nov | IC95 % | Gini jun–ago | IC95 % |
|---|---|---|---|---|
| LightGBM regularizado | **0,2516** | [0,233; 0,269] | **0,2638** | [0,245; 0,281] |
| LightGBM default | 0,2471 | [0,229; 0,265] | 0,2582 | [0,240; 0,276] |
| XGBoost regularizado | 0,2464 | [0,228; 0,264] | 0,2578 | [0,239; 0,275] |
| CatBoost regularizado | 0,2437 | [0,226; 0,262] | 0,2541 | [0,236; 0,272] |
| RandomForest regularizado | 0,2407 | [0,223; 0,258] | 0,2512 | [0,233; 0,269] |

Las diferencias entre modelos (< 0,01) están dentro del ruido de la métrica (SE ≈ 0,009). El Gini de Optuna/early stopping usa la validación para decidir → optimista.

---

## 7. Problemas Frecuentes

| Síntoma | Causa y solución |
|---|---|
| `>>>` en pantalla o `SyntaxError` al pegar el comando | Estás dentro de Python. Escribe `exit()` y ejecuta el comando en la terminal. |
| `python` no se reconoce / `command not found` | Prueba `python3 pipeline.py ...` o, en Windows, `py pipeline.py ...`. |
| `ModuleNotFoundError: No module named 'lightgbm'` (u otra) | Falta instalar librerías o no está activo el entorno virtual. Repite la sección 2. |
| `FileNotFoundError: train.csv` | `--datos` apunta a una carpeta que no contiene los CSV, o la terminal no está en la carpeta del proyecto. |
| Error con `keep_empty_features` o `early_stopping_rounds` | Versiones antiguas de scikit-learn o xgboost. Actualiza con `python -m pip install -U scikit-learn xgboost`. |
| `AssertionError` al exportar | El orden o los ids del submission no coinciden con `test.csv` / `sample_submission.csv`. No editar el orden de filas de test. |
| Aviso `LGBMDeprecationWarning` sobre `eval_set` | Está silenciado en el código; es solo informativo. |
| CatBoost varía ±0,003 entre corridas | Usar `thread_count=1` para igualdad exacta (ver P-9). |

---

## 8. Cómo Extender el Pipeline

- **Cambiar variables con lags o ventanas:** editar `cols_lag` y `cols_rolling` en `Config` (solo afecta modo `temporal`; en `estatico` se ignoran).
- **Probar otros hiperparámetros por defecto:** editar el diccionario `PARAMS_DEFECTO` (añadido preset `lightgbm_reg`).
- **Añadir un modelo:** agregar su familia en `construir_modelo`, sus parámetros en `PARAMS_DEFECTO` y un método `entrenar_*` en `EvaluadorModelos`.
- **Antes de subir una entrega nueva**, correr walk-forward completo (`--folds 4`) y revisar `resultados_validacion_DF_WCB.csv` (Gini por fold, IC). Guardar con nombre y fecha las entregas que se envíen en `resultados_reportes/historico/`, para poder comparar contra el puntaje de la plataforma.
- **Criterio de reemplazo de `submission_final.csv`:** Gini medio en 4 folds walk-forward **mayor** y diferencia > 0,003 sin empeorar ningún fold. Si no se cumple, mantener la actual.

---

## 9. Próximos Pasos (Work in Progress)

- [x] **Notebook de Análisis Exploratorio (EDA)** completo en `notebooks/01_eda_completo.ipynb` (corregido N-1 a N-21)
- [x] **Análisis de calidad de datos** (nulos, duplicados, estructura de panel, hazard, composición test) ✅ Hechos verificados
- [x] **Matriz de correlación** y análisis de variables más trascendentes ✅ Ver `resultados_reportes/` (Pearson/Spearman, IC bootstrap)
- [x] **Feature engineering avanzado** (interacciones, nuevas features derivadas) ✅ Ver sección 6: temporales descartadas (ablación C), interacción banda×productos probada (F), no aplicada al pipeline
- [x] **Experimentación con Redes Neuronales** (TensorFlow/Keras / scikit-learn MLP) ✅ Gini 0,2400 / 0,2294 / 0,1854 (early stopping en val = optimista)
- [ ] **Análisis SHAP** para interpretabilidad (no ejecutado; ver N-20)
- [ ] **Walk-forward validation** completa (4 folds) con IC bootstrap por cliente (P-5)
- [ ] **Ensemble final** → descartado por evidencia (0,2513–0,2517 vs 0,2516 LGBM solo)
- [ ] **Optuna tuning para Neural Network** (pendiente)
- [ ] **LightGBM regularizado** como preset por defecto (P-6)
- [ ] **Selector por permutación** sobre validación temporal (P-7)
- [ ] **Regenerar todas las figuras** con `pipelines/generar_graficas.py` (P-6, N-8 a N-19)
- [ ] **Mover pipelines legacy** a `pipelines/legacy/` (P-12)
- [ ] **Crear `requirements.txt` y docs** (P-15, P-16)

---

## 10. Informe Final Completo

Ver **[INFORME_FINAL_DATAFEST_2026.md](INFORME_FINAL_DATAFEST_2026.md)** para el análisis exhaustivo con:
- Hechos verificados del panel (sección 1 del documento de instrucciones)
- Estadísticas descriptivas completas con números reales
- Matriz de correlación (Pearson/Spearman, IC bootstrap)
- Importancia de variables por **permutación** (no impureza)
- Comparativa de modelos (Gini con IC95 %)
- Arquitecturas de Redes Neuronales guardadas
- Feature engineering: estado real (ablación, interacciones probadas)
- Próximos pasos priorizados basados en evidencia

---

## 11. Bases del Concurso

- **Métrica:** Gini = 2 × AUC − 1
- **Entrega:** Una sola solución por equipo, un solo representante
- **Fecha límite:** 7 de octubre 2026
- **Correo:** marcaempleadora@bcp.com.pe
- **Formato:** `id_cliente,prediccion` (exactamente `sample_submission.csv`)
- **Cupos:** 2 por universidad
- **Bases oficiales:** Ver enlace "Ingresa AQUÍ" del comunicado (pedir al humano que lo lea para confirmar qué adjuntar)

*README actualizado según hechos verificados y instrucciones del zip - DataFest 2026 Team*