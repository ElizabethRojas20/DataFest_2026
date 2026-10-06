# Validación de modelos sobre datos nuevos (test.csv, diciembre 2026)

Objetivo: comprobar si los modelos entrenados con ene–nov "se caen" al aplicarlos a diciembre (`datos_entrada/test.csv`).

## Limitación de partida
`test.csv` no tiene `objetivo`, así que el Gini de diciembre no se puede medir directamente. Se usan diagnósticos sin etiquetas y estimaciones indirectas:

| Bloque | Pregunta | Método |
|---|---|---|
| 1. Drift de variables | ¿Diciembre se parece a lo que vio el modelo? | PSI por variable (nov→dic y ene–nov→dic) y validación adversarial train vs test |
| 2. Drift de predicciones | ¿Cambia la distribución de scores? | PSI de scores en validación (nov, modelo entrenado ≤ oct) frente a diciembre (modelo final); nuevos vs antiguos |
| 3. Estabilidad del ranking | ¿Los modelos siguen de acuerdo en diciembre? | Spearman entre modelos y entre semillas en nov frente a dic; solapamiento del top-10 % / 20 % |
| 4. Gini esperado en condiciones de diciembre | ¿Cuánto Gini perderíamos? | Folds ago–nov (a) tal cual, (b) permutando `dias_ultima_interaccion` dentro del mes como en diciembre, (c) reponderados por el ratio de densidad adversarial y por la proporción de clientes nuevos (18,6 %) |

## Modelos evaluados
- Boosting/árboles del pipeline (`PARAMS_DEFECTO`): LightGBM regularizado y sin regularizar, XGBoost, CatBoost, RandomForest. Se reentrenan porque no estaban guardados.
- Modelos guardados en `modelos/`: MLP de scikit-learn y redes Keras (`nn_baseline`, `nn_optimizado`) con `preprocessor_nn.joblib`.
- La entrega actual `resultados_reportes/submission_final.csv`.

## Estructura
- `scripts/`: scripts numerados, ejecutables desde la raíz del repo.
- `resultados/`: CSV generados por los scripts.
- `figuras/`: gráficos.
- `INFORME_VALIDACION.md`: conclusiones.
- `requirements-validacion.txt`: entorno exacto (Python 3.12.10, `.venv`).

## Ejecución
```bash
.venv/Scripts/python.exe validacion_modelos/scripts/<script>.py
```

Reglas: validación temporal; nada escribe en `resultados_reportes/`; nunca se toca `submission_final.csv`.
