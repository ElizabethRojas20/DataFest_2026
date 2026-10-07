# DataFest 2026: Propensión de Conversión de Clientes

Modelo que estima **qué clientes de un banco tienen más probabilidad de contratar un producto en diciembre de 2026**, para contactar primero a los más prometedores.

> **¿No eres técnico?** Lee primero [`docs/RESUMEN_DE_CAMBIOS.md`](docs/RESUMEN_DE_CAMBIOS.md): explica en lenguaje sencillo qué hicimos, qué encontramos y qué significa.

---

## En pocas palabras

| | |
|---|---|
| **Qué predice** | La probabilidad de que cada cliente "convierta" (contrate) en diciembre de 2026 |
| **Con qué datos** | 11 meses de historia (enero–noviembre 2026): 110 100 registros de 24 628 clientes |
| **Cómo se califica** | Gini: 0 = ordenar al azar, 1 = orden perfecto |
| **Resultado** | Gini ≈ **0,254** (estimado con agosto–noviembre; diciembre no se puede medir porque el concurso no da las respuestas) |
| **Qué significa** | En el 10 % de clientes con mayor probabilidad, **30 de cada 100** contratan, frente a 15 de cada 100 si se contacta al azar |
| **Modelo elegido** | LightGBM con árboles "superficiales" (pequeños y regularizados) |

## Novedades (octubre 2026)

Una auditoría del proyecto encontró y corrigió varios problemas. Resumen:

1. **Se elegía al mejor modelo mirando un solo mes.** Ahora se usa el promedio de los cuatro meses de prueba.
2. **El margen de error estaba mal calculado** (decía más certeza de la real). Ahora se calcula bien.
3. **El tamaño del modelo se decidía casi al azar.** Ahora se elige con una regla estable.
4. **Una variable (`dias_ultima_interaccion`) deja de tener sentido en diciembre.** Ahora se excluye.
5. **Algunos archivos de resultados tenían números de relleno.** Se retiraron y el script que los generaba ya no inventa valores.
6. **Modelo nuevo:** árboles más pequeños, que aprenden lo importante sin memorizar ruido. Mejora un poco el Gini (de 0,249 a 0,254) y es más estable.
7. **Se comprobó que la mejora no es suerte**, con pruebas de robustez y de sobreoptimización.

Detalle en lenguaje sencillo: [`docs/RESUMEN_DE_CAMBIOS.md`](docs/RESUMEN_DE_CAMBIOS.md). Detalle técnico: los informes de [`validacion_modelos/`](validacion_modelos/).

## Dónde está cada cosa

| Si quieres… | Mira |
|---|---|
| Entender el proyecto sin tecnicismos | [`docs/RESUMEN_DE_CAMBIOS.md`](docs/RESUMEN_DE_CAMBIOS.md) |
| Ver la entrega actual | `resultados_reportes/submission_final.csv` |
| Ver la entrega candidata (modelo nuevo) y su ficha | `resultados_reportes/candidata_20261007/` (incluye un reporte HTML) |
| Saber qué significa cada dato | [`docs/DATASET_DESCRIPTION.md`](docs/DATASET_DESCRIPTION.md) |
| Las reglas del concurso | [`docs/BASES_CONCURSO.md`](docs/BASES_CONCURSO.md) |
| Las pruebas de que el modelo funciona | [`validacion_modelos/`](validacion_modelos/) (3 informes) |
| El informe final del equipo (versión anterior a la auditoría) | [`INFORME_FINAL_DATAFEST_2026.md`](INFORME_FINAL_DATAFEST_2026.md) |
| El código que entrena el modelo | `pipelines/Pipeline_DF_WCB.py` |

---

# Parte técnica

## 1. Estructura del proyecto

```text
DataFest_2026/
├── README.md                          # Este documento
├── INFORME_FINAL_DATAFEST_2026.md     # Informe final del equipo (previo a la auditoría de octubre)
├── requirements.txt                   # Librerías necesarias
├── docs/
│   ├── RESUMEN_DE_CAMBIOS.md          # Resumen no técnico de la auditoría y los cambios
│   ├── DATASET_DESCRIPTION.md         # Diccionario de datos y hechos verificados
│   └── BASES_CONCURSO.md              # Reglas del concurso
├── datos_entrada/                     # train.csv, test.csv, sample_submission.csv
├── pipelines/
│   ├── Pipeline_DF_WCB.py             # ⭐ Pipeline principal (el único activo)
│   ├── generar_graficas.py            # Figuras a partir de los CSV de resultados
│   └── legacy/                        # Versiones antiguas del pipeline (solo trazabilidad)
├── resultados_reportes/
│   ├── submission_final.csv           # ⭐ Entrega actual
│   ├── candidata_20261007/            # ⭐ Entrega candidata del pipeline nuevo + ficha HTML
│   ├── historico/                     # Entregas anteriores con fecha
│   ├── resultados_validacion_DF_WCB.csv, val_predictions.csv, walkforward_cv.csv
│   └── fig*.png, *.csv                # Figuras y tablas auxiliares (de la corrida anterior)
├── validacion_modelos/                # Auditoría: validación, robustez y sobreoptimización
│   ├── INFORME_VALIDACION.md          # ¿El modelo se cae en diciembre?
│   ├── INFORME_ROBUSTEZ.md            # Pruebas R1–R9 de los candidatos
│   ├── INFORME_SOBREOPTIMIZACION.md   # ¿La mejora es real o sesgo de selección?
│   ├── scripts/                       # Scripts 01–23 (ver validacion_modelos/README.md)
│   ├── resultados/                    # CSV de cada prueba
│   └── figuras/                       # Gráficos de las pruebas
├── verificaciones/                    # Experimentos iniciales (panel, ablación, permutación, ensemble)
├── notebooks/01_eda_completo.ipynb    # Análisis exploratorio
├── modelos/                           # Redes neuronales guardadas (descartadas)
├── scripts_utilitarios/               # Comparar entregas, generar CSV de figuras, ejecución final
└── documentacion/                     # Informes ejecutivos y guía del pipeline (HTML)
```

Secciones del pipeline principal:

| Sección | Qué contiene |
|---|---|
| `Config` | Todos los parámetros (columnas, exclusiones, rejilla de árboles, folds, bootstrap…) |
| Carga y diagnóstico | `cargar_datos`, `validar_esquema`, `diagnostico_inicial` |
| Features | `crear_features_temporales`, `target_encoding_temporal`, `construir_dataset` |
| Selección | Varianza cero → correlación > 0,95 → top-K por importancia de permutación |
| Modelado | `EvaluadorModelos` (walk-forward, curva de árboles, regla 1-SE, bootstrap de clientes), `construir_modelo`, `PARAMS_DEFECTO` |
| Entrega | `entrenar_y_predecir_final`, `exportar_submission` |

## 2. Instalación

Requiere **Python 3.10 o superior** (probado con 3.12 y 3.13).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Mac/Linux: source .venv/bin/activate)
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Para los scripts de `validacion_modelos/`, el entorno exacto está en `validacion_modelos/requirements-validacion.txt` (incluye `interpret` para el EBM).

> **Mac con chip Apple:** si LightGBM falla al importar con un error sobre `libomp`, instala OpenMP con `brew install libomp`.

## 3. Ejecución

Desde la carpeta raíz del proyecto, con el entorno activado:

```bash
# Prueba rápida (1 fold, 1 semilla, sin Optuna): ~1 min
python pipelines/Pipeline_DF_WCB.py --datos datos_entrada --salida pruebas/prueba.csv --trials 0 --n-seeds 1 --n-folds 1

# Validación completa y entrega candidata (4 folds, 3 semillas, sin Optuna): ~2-3 min
python pipelines/Pipeline_DF_WCB.py --datos datos_entrada --salida resultados_reportes/candidata_AAAAMMDD/submission_candidata.csv --trials 0 --n-seeds 3 --n-folds 4
```

> Los archivos auxiliares (tabla de validación, predicciones, curvas, bootstrap) se guardan **en la misma carpeta que `--salida`**. Usa una carpeta propia para cada corrida, para no sobrescribir resultados anteriores. **Nunca** escribas directamente en `submission_final.csv`.

### Argumentos

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--datos` | `.` | Carpeta con los CSV de entrada |
| `--salida` | `submission_DF_WCB.csv` | Archivo de entrega; los auxiliares van a la misma carpeta |
| `--trials` | `30` | Pruebas de Optuna para LightGBM (`0` = sin Optuna; recomendado, ver informe de sobreoptimización) |
| `--n-folds` | `4` | Meses de validación walk-forward (4 = ago, sep, oct, nov) |
| `--n-seeds` | `3` | Semillas promediadas en el modelo final |
| `--excluir` | `dias_ultima_interaccion` | Columnas a quitar (y sus derivadas). `--excluir` sin valores no quita nada |
| `--grid-arboles` | `25 50 75 100 150 200 300 400 600 800` | Nº de árboles donde se mide la curva Gini |
| `--n-bootstrap` | `500` | Réplicas del bootstrap de clientes (regla 1-SE e IC95 %) |
| `--modo-features` | `estatico` | `estatico` (recomendado) o `temporal` (lags/deltas/rolling) |
| `--top-k` | `80` | Máximo de variables tras la selección |
| `--semilla` | `42` | Semilla base |
| `--bootstrap-ic`, `--meses-val` | — | Legacy, sin efecto |

### Qué hace, paso a paso

1. **Carga y valida** el panel: sin duplicados cliente-mes, sin huecos, sin filas posteriores a la conversión.
2. **Construye las variables:** originales + One-Hot de las categóricas, sin `dias_ultima_interaccion` por defecto.
3. **Selecciona variables** solo con meses anteriores a la validación.
4. **Valida** con walk-forward: cada uno de los 4 últimos meses es un examen, y se entrena con todos los anteriores.
   - Sin early stopping: cada fold se entrena hasta 800 árboles y se mide el Gini en cada punto de la rejilla.
   - El Gini de cada mes usa el nº de árboles elegido con la **regla 1-SE** sobre la curva media de los **otros** meses.
   - Compara LightGBM (regularizado, sin regularizar, superficial), XGBoost, CatBoost y RandomForest.
5. **Elige el ganador** por Gini **medio** de los 4 meses. La tabla muestra además el peor mes, el Gini de cada mes, el IC95 % (bootstrap de clientes con reemplazo, pareado entre modelos) y los árboles por mes.
6. **Entrena el modelo final** con todo train (nº de árboles = el de la curva media × filas totales / filas por fold), promedia 3 semillas y exporta la entrega con comprobaciones estrictas.

### Comparar una entrega nueva con la actual

```bash
python scripts_utilitarios/comparar_submissions.py resultados_reportes/candidata_20261007/submission_candidata.csv resultados_reportes/submission_final.csv
```

## 4. Reglas del proyecto (léanlas antes de modificar el código)

- **Validación temporal, nunca aleatoria.** Un mismo cliente aparece en varios meses con casi todas sus variables fijas. Una partición aleatoria infla la métrica. Si se usa otra CV, agrupar por `id_cliente`.
- **Cero información del futuro** en variables y codificaciones.
- **No usar `id_cliente` ni `mes` como predictoras.** `id_cliente` crece con la fecha de ingreso.
- **Ruido de la métrica:** el SE del Gini de un mes es ≈ 0,016, y el de la media de 4 meses ≈ 0,008. No afirmar una mejora menor a ~0,01 ni una que no se repita en los meses.
- **Reemplazar `submission_final.csv`** requiere un Gini medio de 4 folds mayor en más de 0,003, sin empeorar ningún mes. Archivar la entrega anterior con fecha en `resultados_reportes/historico/`.
- **No escribir valores inventados** en ningún archivo de resultados. Si algo no se puede calcular, se omite y se avisa.

## 5. Qué sabemos de los datos (hechos verificados)

- **Panel de "primera conversión":** cada cliente aparece mes a mes hasta que convierte y entonces desaparece. Meses consecutivos, sin huecos ni filas posteriores a la conversión.
- **Train:** 110 100 filas, 24 628 clientes, 15,05 % de positivos. **Test:** 9 900 clientes de diciembre (81,4 % con historial y 18,6 % nuevos).
- **Solo `dias_ultima_interaccion` cambia con el tiempo**; el resto de columnas es constante por cliente.
- **`dias_ultima_interaccion` es un artefacto:** es igual a `dias_ultima_transaccion` en el 100 % de las filas de enero, y ese porcentaje baja 9 puntos por mes (9 % en noviembre). En diciembre es una permutación sin señal (0,26 %).
- **Señal real concentrada en tres variables:** `banda_riesgo`, `numero_productos` y `dias_ultima_transaccion`. Hay una interacción fuerte: en banda `low`, la conversión sube del 13,8 % (1 producto) al 31,3 % (5 productos).
- **Datos sintéticos:** variables que en banca real estarían ligadas aquí son independientes (`es_nuevo_cliente` y antigüedad, `tiene_prestamo` y ratio de deuda, nº de productos y tipo de producto).
- **La permanencia en el panel predice:** 17 % de conversión en el primer mes frente a ~13 % tras 7 meses. Por sí sola tiene un Gini de 0,077 en noviembre y no es una variable del modelo.

Diccionario completo: [`docs/DATASET_DESCRIPTION.md`](docs/DATASET_DESCRIPTION.md).

## 6. Resultados de referencia (pipeline nuevo)

Walk-forward ago–nov, sin `dias_ultima_interaccion` salvo donde se indica (`resultados_reportes/candidata_20261007/resultados_validacion_DF_WCB.csv`):

| Modelo | Gini medio | IC95 % | Peor mes | Árboles por mes |
|---|---|---|---|---|
| **LightGBM superficial** (ganador) | **0,2540** | [0,238; 0,268] | 0,2375 | 100/200/200/200 |
| CatBoost | 0,2519 | [0,236; 0,267] | 0,2317 | 150/150/150/150 |
| LightGBM sin regularizar | 0,2518 | [0,236; 0,266] | 0,2390 | 25/25/25/25 |
| LightGBM regularizado | 0,2513 | [0,236; 0,267] | 0,2362 | 25/100/25/25 |
| XGBoost | 0,2494 | [0,234; 0,265] | 0,2315 | 50/25/50/50 |
| RandomForest | 0,2320 | [0,215; 0,248] | 0,2100 | 300 |
| *LightGBM regularizado con dui (modelo anterior)* | *0,2487* | *[0,233; 0,265]* | *0,2325* | *50/150/150/50* |

- **Frente al modelo anterior:** +0,0053 (+0,0049 en condiciones de diciembre, con los 4 meses mejores).
- **Comprobaciones:** el walk-forward anidado confirma que no es sesgo de selección (optimismo 0,0005), y el SPA de Hansen da p = 0,025. Ver [`INFORME_SOBREOPTIMIZACION.md`](validacion_modelos/INFORME_SOBREOPTIMIZACION.md).
- **Diferencias dentro del ruido:** entre los modelos de boosting no son significativas. La elección del superficial se apoya en su estabilidad y en las pruebas de robustez.

## 7. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `>>>` en pantalla o `SyntaxError` al pegar el comando | Estás dentro de Python. Escribe `exit()` y ejecuta el comando en la terminal |
| `python` no se reconoce | Prueba `python3` o, en Windows, `py` |
| `ModuleNotFoundError` | Falta instalar librerías o no está activo el entorno virtual (sección 2) |
| `FileNotFoundError: train.csv` | `--datos` no apunta a la carpeta correcta, o la terminal no está en la raíz del proyecto |
| `error: unrecognized arguments: --folds` | El argumento correcto es `--n-folds` |
| `AssertionError` al exportar | El orden o los ids no coinciden con `test.csv`; no reordenar las filas de test |

## 8. Cómo extender el pipeline

- **Añadir un modelo:** agregarlo en `construir_modelo` y `predecir_rejilla`, poner sus parámetros en `PARAMS_DEFECTO` y crear un método `entrenar_*` en `EvaluadorModelos`.
- **No modificar** las entradas existentes de `PARAMS_DEFECTO`, ni `construir_modelo`, `ROUNDS_MAX` o `ES_ROUNDS`: los scripts de `validacion_modelos/` las importan.
- **Probar una variable nueva** (por ejemplo, la permanencia en el panel): calcularla solo con el pasado y validarla con el walk-forward anidado (`validacion_modelos/scripts/23_anidado_placebo.py`).

## 9. Bases del concurso

- **Métrica:** Gini = 2 × AUC − 1.
- **Entrega:** una sola solución por equipo, con un solo representante. Formato `id_cliente,prediccion` (igual que `sample_submission.csv`).
- **Fecha límite:** 7 de octubre de 2026.
- Detalle: [`docs/BASES_CONCURSO.md`](docs/BASES_CONCURSO.md).
