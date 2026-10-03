# DataFest — Propensión de conversión de clientes

Documento interno del equipo. Explica cómo está organizado el proyecto, cómo preparar el entorno y cómo ejecutar el pipeline que genera el archivo de entrega.

- **Problema:** estimar la probabilidad de que un cliente convierta (`objetivo = 1`) en un mes dado.
- **Métrica:** Gini = 2 · AUC − 1 (solo importa el *orden* de las predicciones, no su calibración).
- **Datos:** `train.csv` (enero–noviembre 2026, 110.100 filas) y `test.csv` (diciembre 2026, 9.900 filas). Cada fila es un cliente-mes; un cliente puede aparecer en varios meses.

---

## 1. Estructura del proyecto

```text
.
├── README.md                   # este documento
├── pipeline.py                 # TODO el pipeline (features -> selección -> modelos -> submission)
├── data/                       # entradas (no se versionan si pesan o son confidenciales)
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── metaData.csv
│   └── DATASET_DESCRIPTION.md
└── salida/                     # lo genera el pipeline
    ├── submission.csv          # archivo a subir a la competencia
    ├── resultados_validacion.csv   # AUC y Gini de cada modelo en validación
    └── features_seleccionadas.txt  # variables finales usadas por el modelo
```

Dentro de `pipeline.py` el código está organizado en secciones:

| Sección | Qué contiene |
|---|---|
| `Config` | Todos los parámetros (columnas, lags, umbrales, nº de trials…). Es el único lugar que se edita para ajustar el comportamiento. |
| Carga y diagnóstico | `cargar_datos`, `validar_esquema`, `diagnostico_inicial` |
| Ingeniería de características | `crear_features_temporales` (lags, deltas, ventanas), `target_encoding_temporal`, `construir_dataset` |
| Selección | `SelectorVarianzaCero`, `SelectorCorrelacion`, `SelectorImportanciaLGBM`, `seleccionar_features` |
| Modelado | `EvaluadorModelos` (LightGBM, XGBoost, RandomForest + Optuna) y `construir_modelo` |
| Entrega | `entrenar_y_predecir_final`, `exportar_submission` |
| `main` | Orquesta todo en orden |

---

## 2. Instalación

Requiere **Python 3.10 o superior**. Se probó con Python 3.13, pandas 3.0, scikit-learn 1.9, LightGBM 4.7, XGBoost 3.4 y Optuna 5.0. Versiones mínimas: scikit-learn ≥ 1.2 y xgboost ≥ 1.6.

Recomendado: usar un entorno virtual para no mezclar librerías con otros proyectos.

**Windows (Símbolo del sistema):**
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -U pandas numpy scikit-learn lightgbm xgboost optuna
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U pandas numpy scikit-learn lightgbm xgboost optuna
```

Para comprobar que quedó bien:
```bash
python -c "import pandas, sklearn, lightgbm, xgboost, optuna; print('OK')"
```

> **Mac con chip Apple:** si LightGBM falla al importar con un error sobre `libomp`, instala OpenMP con `brew install libomp`.

---

## 3. Ejecución

Los comandos se escriben en la **terminal** (no dentro de Python ni de un notebook). Si usas Jupyter o Colab, antepón `!` al comando.

1. Coloca `train.csv`, `test.csv` y `sample_submission.csv` en `data/`.
2. Abre la terminal en la carpeta del proyecto (con el entorno virtual activado).
3. Ejecuta:

```bash
python pipeline.py --datos ./data --salida ./salida/submission.csv --trials 30
```

Tarda unos 2 minutos. Al terminar deja en `salida/` el `submission.csv` y los archivos de trazabilidad.

### Argumentos

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--datos` | `.` | Carpeta donde están los CSV de entrada. |
| `--salida` | `submission.csv` | Ruta del archivo de entrega. Los archivos auxiliares se guardan en la misma carpeta. |
| `--trials` | `30` | Nº de pruebas de Optuna para ajustar LightGBM. `0` desactiva Optuna (más rápido). |
| `--meses-val` | `3` | Nº de últimos meses de `train` usados para validar. Con 3, la partición es ≈ 73 % / 27 %. |
| `--top-k` | `80` | Máximo de variables a conservar tras la selección. |
| `--umbral-ohe` | `10` | Categóricas con ≤ este nº de niveles van a One-Hot; con más, a Target Encoding temporal. |
| `--n-seeds` | `3` | Semillas que se promedian en el modelo final. |
| `--semilla` | `42` | Semilla base (reproducibilidad). |

Ejemplos útiles:
```bash
# Corrida rápida para probar cambios (sin Optuna, una semilla)
python pipeline.py --datos ./data --salida ./salida/prueba.csv --trials 0 --n-seeds 1

# Validar con más meses
python pipeline.py --datos ./data --salida ./salida/submission.csv --meses-val 4
```

### Qué hace, paso a paso

1. **Carga y validación:** comprueba columnas, que no haya duplicados cliente-mes y muestra un diagnóstico del dataset.
2. **Features:** rezagos de 1, 2 y 3 meses, cambio mensual (delta), medias y sumas de ventana de 3 meses, One-Hot / Target Encoding temporal.
3. **Selección:** elimina constantes, luego variables con correlación de Pearson > 0,95, luego se queda con las más importantes según un LightGBM rápido.
4. **Validación:** entrena y compara LightGBM, XGBoost, RandomForest y LightGBM con Optuna; reporta AUC y Gini.
5. **Entrega:** reentrena el ganador con el 100 % de `train.csv`, predice `test.csv` y exporta `submission.csv`.

---

## 4. Reglas del proyecto (léanlas antes de modificar el código)

- **Validación temporal, no aleatoria.** No usar `train_test_split` aleatorio. Un mismo cliente aparece en varios meses con casi todas sus variables fijas, y el test es un mes futuro; una partición aleatoria infla la métrica (en pruebas dio un Gini 0,015–0,025 más alto que la partición temporal).
- **Cero información del futuro.** Los lags y ventanas solo miran meses anteriores del mismo cliente. El Target Encoding de una fila usa únicamente objetivos de meses anteriores. Nunca crear variables con datos de meses posteriores.
- **No usar `id_cliente` ni `mes` como predictoras.** Diciembre es un mes no visto durante el entrenamiento.
- **La selección de variables se ajusta solo con los meses de entrenamiento**, no con los de validación.
- **El orden de `submission.csv` debe ser idéntico al de `test.csv`.** El pipeline lo verifica antes de guardar; si la validación falla, no escribe el archivo.
- **Formato de entrega:** exactamente dos columnas, `id_cliente,prediccion`, con `prediccion` entre 0 y 1. No incluir `mes`.

---

## 5. Qué sabemos de los datos

- Ningún cliente reaparece después de convertir (`objetivo = 1`).
- La tasa de conversión ronda el 15 % y es estable entre meses.
- El 81 % de los clientes de diciembre tiene historial en `train`.
- **Casi todas las variables son estáticas por cliente.** Solo `dias_ultima_interaccion` cambia entre meses (en ~80 % de los clientes con más de un mes). El resto (saldo, ingresos, productos, categóricas, banderas, etc.) no varía, por lo que sus lags y deltas no aportan: el filtro de varianza cero y el de correlación los descartan automáticamente.
- Tras los filtros quedan unas 50 variables, por debajo del tope de 80.
- Las 5 categóricas tienen entre 3 y 5 niveles, así que con el umbral por defecto todas van a One-Hot.

## 6. Resultados de referencia

Validación temporal (entrenamiento: enero–agosto; validación: septiembre–noviembre):

| Modelo | Gini |
|---|---|
| LightGBM + Optuna | ≈ 0,250 |
| XGBoost | ≈ 0,246 |
| LightGBM | ≈ 0,246 |
| RandomForest | ≈ 0,241 |

Las diferencias entre modelos (< 0,01) están dentro del ruido de la métrica con esta validación (≈ ±0,010 de Gini). Además, Optuna y el early stopping usan los meses de validación, así que estas cifras son algo optimistas respecto al resultado real en la competencia.

---

## 7. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `>>>` en pantalla o `SyntaxError` al pegar el comando | Estás dentro de Python. Escribe `exit()` y ejecuta el comando en la terminal. |
| `python` no se reconoce / `command not found` | Prueba `python3 pipeline.py ...` o, en Windows, `py pipeline.py ...`. |
| `ModuleNotFoundError: No module named 'lightgbm'` (u otra) | Falta instalar librerías o no está activo el entorno virtual. Repite la sección 2. |
| `FileNotFoundError: train.csv` | `--datos` apunta a una carpeta que no contiene los CSV, o la terminal no está en la carpeta del proyecto. |
| Error con `keep_empty_features` o `early_stopping_rounds` | Versiones antiguas de scikit-learn o xgboost. Actualiza con `python -m pip install -U scikit-learn xgboost`. |
| `AssertionError` al exportar | El orden o los ids del submission no coinciden con `test.csv` / `sample_submission.csv`. No editar el orden de filas de test. |
| Aviso `LGBMDeprecationWarning` sobre `eval_set` | Está silenciado en el código; es solo informativo. |

---

## 8. Cómo extender el pipeline

- **Cambiar variables con lags o ventanas:** editar `cols_lag` y `cols_rolling` en `Config`.
- **Probar otros hiperparámetros por defecto:** editar el diccionario `PARAMS_DEFECTO`.
- **Añadir un modelo:** agregar su familia en `construir_modelo` y un método `entrenar_*` en `EvaluadorModelos`.
- **Antes de subir una entrega nueva**, correr una vez completo (`--trials 30`) y revisar `resultados_validacion.csv`. Guardar con nombre y fecha las entregas que se envíen, para poder comparar contra el puntaje de la plataforma.
