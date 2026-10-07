# Validación de modelos: ¿el modelo aguanta en diciembre, es robusto y la mejora es real?

Esta carpeta contiene la auditoría de los modelos. Ningún script escribe en `resultados_reportes/` ni toca `submission_final.csv`.

> **En una frase:** ningún modelo "se cae" en diciembre. El LightGBM superficial sin `dias_ultima_interaccion` mejora al modelo anterior en ≈ +0,005 de Gini, y esa mejora no se explica por haber probado muchas versiones. Resumen no técnico: [`../docs/RESUMEN_DE_CAMBIOS.md`](../docs/RESUMEN_DE_CAMBIOS.md).

## Los tres informes

| Informe | Pregunta | Conclusión |
|---|---|---|
| [`INFORME_VALIDACION.md`](INFORME_VALIDACION.md) | ¿Los modelos se caen al aplicarlos a diciembre? | No. La única ruptura es `dias_ultima_interaccion`, que en diciembre no tiene señal |
| [`INFORME_ROBUSTEZ.md`](INFORME_ROBUSTEZ.md) | ¿Qué candidato es mejor y es robusto? (R1–R9) | `lgb_sup_sin_dui`: meseta de hiperparámetros, estable y pasa el placebo. Hecho con el esquema antiguo (early stopping en M−1) |
| [`INFORME_SOBREOPTIMIZACION.md`](INFORME_SOBREOPTIMIZACION.md) | ¿La mejora es real o sesgo de selección? | El régimen superficial es mejor (anidado: optimismo 0,0005; SPA p = 0,025). La variante concreta no es identificable |

`test.csv` no tiene `objetivo`, así que el Gini de diciembre no se puede medir. Todo se estima con los folds ago–nov y con diagnósticos sin etiquetas: drift, ranking y escenarios que imitan diciembre.

## Scripts (`scripts/`, ejecutar desde la raíz del repo)

| Scripts | Qué hacen | Informe |
|---|---|---|
| `01`–`08` | Walk-forward honesto, modelos finales, redes neuronales, drift de variables y de predicciones, Gini en condiciones de diciembre, veredicto | VALIDACION |
| `09`–`19` | Pruebas R1–R9: comparación formal, nº de árboles, hiperparámetros, semillas y bootstrap, segmentos, perturbación, degradación, placebo, curva de aprendizaje, tabla y submissions | ROBUSTEZ |
| `20_pipeline_nuevo.py` | C1–C3 y R4 sobre los modelos del pipeline nuevo (curva 1-SE, sin early stopping) | SOBREOPTIMIZACION (contexto) |
| `21_registro_candidatos.py` | Predicciones honestas de 37 candidatos sobre las mismas filas | SOBREOPTIMIZACION |
| `22_pruebas_seleccion.py` | PBO/CSCV, Reality Check, SPA, Romano-Wolf, Gini deflactado, MCS (sin reentrenar) | SOBREOPTIMIZACION |
| `23_anidado_placebo.py` | Walk-forward anidado y placebo de la receta (reentrenan) | SOBREOPTIMIZACION |

Módulos de apoyo: `comun.py` (rutas, criterios del veredicto, Gini ponderado, bootstrap de clientes), `modelos.py`, `robustez_comun.py`, `criterios_robustez.py` (pre-registro de R1–R9) y `sobreopt_comun.py` (candidatos y criterios de 21–23).

```bash
.venv/Scripts/python.exe validacion_modelos/scripts/<script>.py
```

Tiempos aproximados (12 núcleos):

| Scripts | Tiempo |
|---|---|
| 01–08 | ~6 min |
| R1–R9 | ~1 h |
| 20 | 2 min |
| 21 | 9 min |
| 22 | 2 min |
| 23 | ~1 h 30 (anidado ~22 min + placebo de 20 réplicas) |

## Otras carpetas

- `resultados/`: CSV de cada prueba y `logs/`. Los archivos de predicciones son los más pesados.
- `figuras/`: gráficos de los informes de validación y robustez.
- `cache_modelos/`: modelos de R1 guardados para no reentrenar.
- `requirements-validacion.txt`: entorno exacto (Python 3.12.10, `.venv`), incluye `interpret` para el EBM.

## Reglas

- Validación temporal siempre. El remuestreo es por cliente: un cliente entra o sale con todos sus meses.
- Los criterios de cada bloque de pruebas se fijan en código antes de ejecutarlas (`comun.CRITERIOS`, `criterios_robustez.py`, `sobreopt_comun.CRITERIOS_SO`).
- Nada escribe fuera de esta carpeta.
