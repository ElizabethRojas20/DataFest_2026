# Bases del Concurso - DataFest 2026

> **IMPORTANTE:** Este documento resume las reglas clave. **Leer el comunicado oficial y el enlace "Ingresa AQUÍ" para confirmar qué adjuntar exactamente** (solo CSV, o también código/informe). Pedir al humano que lo revise.

---

## Resumen de Reglas Críticas

| Regla | Detalle |
|-------|---------|
| **Métrica** | Gini = 2 × AUC − 1 (solo ordena, no calibra) |
| **Entregas** | **Una sola solución por equipo** |
| **Representante** | **Un solo representante** envía por el equipo |
| **Fecha límite** | **7 de octubre 2026** |
| **Correo de entrega** | **marcaempleadora@bcp.com.pe** |
| **Formato archivo** | `id_cliente,prediccion` (exactamente `sample_submission.csv`) |
| **Cupos por universidad** | 2 equipos |
| **Bases oficiales** | Enlace "Ingresa AQUÍ" del comunicado (leer completo) |

---

## Qué debe contener el correo de entrega

Confirmar con el enlace oficial. Posibilidades típicas:
1. **Solo el archivo `submission.csv`** adjunto
2. **Código + informe + submission** (zip)
3. **Enlace a repositorio** + submission

> **Acción requerida:** Pedir al humano que lea las bases completas y confirme qué adjuntar.

---

## Validación de la Submission (Checklist técnico)

Antes de enviar, verificar con el pipeline:

```bash
python pipelines/Pipeline_DF_WCB.py --datos datos_entrada --salida resultados_reportes/submission_candidata.csv --trials 0 --n-seeds 1 --folds 1 --modo-features estatico
```

El pipeline verifica automáticamente:
- ✅ Mismo número de filas que `test.csv` (9.900)
- ✅ Mismo orden de `id_cliente` que `test.csv`
- ✅ Columnas exactas: `id_cliente,prediccion` (nombres de `sample_submission.csv`)
- ✅ Predicciones en rango [0, 1] sin NaN
- ✅ `nunique(prediccion) > 9000` (sin empates masivos)
- ✅ `sha256` del archivo escrito en log

**Submission actual (`submission_final.csv`):**
- sha256: `88111e48bd918625837415505ab709a56f8a38bc34dfbb35d739d9da5ee29678`
- Modelo: LightGBM + Optuna (3 semillas, modo temporal, validación sep–nov)
- Gini validación: 0,2471 (IC95 % ≈ [0,229; 0,265])

---

## Criterio para Reemplazar la Entrega Actual

Si se genera una candidata (ej. LightGBM regularizado modo estático con walk-forward 4 folds):

```bash
python pipelines/Pipeline_DF_WCB.py --datos datos_entrada --salida resultados_reportes/submission_candidata.csv --trials 30 --n-seeds 3 --modo-features estatico --folds 4
```

**Reemplazar `submission_final.csv` SOLO si:**
1. Gini medio en 4 folds walk-forward es **mayor** que el actual
2. Diferencia **> 0,003** (supera ruido)
3. **No empeora ningún fold** individual

Si no se cumple → mantener `submission_final.csv` actual.

---

## Recordatorios de Ingeniería (No negociables)

- ❌ No usar `id_cliente` ni `mes` como predictoras
- ❌ No partición aleatoria ni CV por filas (agrupar por `id_cliente` si CV)
- ❌ No información del futuro en lags/encoding
- ❌ No afirmar "mejora" con diferencia < 0,01 sin IC
- ❌ No ensemble 50/20/20/10 ni "Gini proyectado 0,252–0,255" (sin evidencia)
- ✅ Validación temporal / walk-forward obligatoria
- ✅ Bootstrap por cliente para IC del Gini
- ✅ Una sola submission final en `resultados_reportes/submission_final.csv`

---

*Bases del concurso resumidas - DataFest 2026 Team*
*Confirmar detalles en el comunicado oficial antes del 7 de octubre*