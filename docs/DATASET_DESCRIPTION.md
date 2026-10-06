# Descripción del Dataset - DataFest 2026

Fuente: Hechos verificados con `verificaciones/00_analisis_panel.py` (ejecutar desde raíz del repo).

---

## Estructura General

| Archivo | Filas | Columnas | Descripción |
|---------|-------|----------|-------------|
| `train.csv` | 110.100 | 25 | Enero–Noviembre 2026, con `objetivo` |
| `test.csv` | 9.900 | 24 | Diciembre 2026, sin `objetivo` |
| `sample_submission.csv` | 9.900 | 2 | Formato: `id_cliente,prediccion` |

**Granularidad:** 1 fila = 1 cliente-mes (`id_cliente`, `mes` formato AAAAMM). Panel longitudinal.

---

## Panel de "Riesgo de Primera Conversión"

- Cada cliente aparece mes a mes **hasta que convierte** (`objetivo=1`) y **desaparece** (0 filas posteriores a la conversión).
- **No hay abandonos silenciosos** (cliente deja de aparecer sin convertir).
- **Meses consecutivos por cliente** (0 huecos de calendario).
- Un cliente que convierte contribuye con su mes de conversión + meses previos como negativos.

---

## Variables (25 en train, 24 en test)

### Identificadores y tiempo
| Variable | Tipo | Descripción |
|----------|------|-------------|
| `id_cliente` | int | Identificador único. **Crece con fecha de ingreso** (nuevos de diciembre: 24.629–26.467). **Nunca usar como predictor**. |
| `mes` | int | AAAAMM (202601–202611 en train, 202612 en test). **No usar como predictor** (dic no visto en train). |
| `objetivo` | int (0/1) | 1 = convirtió ese mes (primera vez). Solo en train. |

### Numéricas (12)
| Variable | Rango | Notas |
|----------|-------|-------|
| `edad` | 22–71 | Uniforme (1,9 % en cada extremo 22 y 71) |
| `ingresos` | 18.000–145.700 | **Piso 18.000 en 1,63 %** (no outlier) |
| `ratio_deuda_ingresos` | 0,02–0,95 | Acotado |
| `antiguedad_cuenta_meses` | 1–179 | |
| `numero_productos` | 1–5 | **Discreto** (1, 2, 3, 4, 5) |
| `saldo_promedio` | 300–90.000 | **Piso 300 en 6,86 %** |
| `dias_ultima_transaccion` | 1–364 | |
| `antiguedad_direccion_meses` | 1–239 | |
| `visitas_web_ultimos_90_dias` | 0–26 | **Discreto** |
| `distancia_sucursal_km` | 0,21–80 | **Tope 80 en 0,14 %** |
| `dia_preferido_pago` | 1–28 | **Discreto** |
| `dias_ultima_interaccion` | 1–364 | **Única variable dinámica** (ver abajo) |

### Booleanas (5)
| Variable | % True |
|----------|--------|
| `tiene_tarjeta_credito` | 74,2 % |
| `activo_movil` | 65,6 % |
| `es_nuevo_cliente` | 19,2 % |
| `tiene_prestamo` | 44,6 % |
| `tiene_seguro` | 38,3 % |

### Categóricas (5) → One-Hot (22 columnas)
| Variable | Niveles | Orden natural |
|----------|---------|---------------|
| `ocupacion` | 5: clerical, manual, professional, manager, self_employed | — |
| `region` | 5: west, north, south, east, central | — |
| `canal_adquisicion` | 5: web, mobile, branch, call_center, partner | — |
| `banda_riesgo` | 3: **low, medium, high** | **low → medium → high** |
| `dispositivo_principal` | 4: android, ios, web, otro | — |

---

## Hechos Verificados Clave

### 1. Clientes y composición
- Clientes únicos train: **24.628**
- Clientes únicos test: **9.900**
- Positivos train: **16.567 (15,05 %)**
- Clientes que convierten alguna vez: **67,3 %** (16.567 / 24.628)
- Test: **8.061 con historial (81,4 %)** + **1.839 nuevos (18,6 %, ids consecutivos mayores)**

### 2. Entradas de clientes nuevos por mes (feb–nov)
| Mes | Entradas nuevas |
|-----|-----------------|
| feb | 921 |
| mar | 2.043 |
| abr | 827 |
| may | 1.973 |
| jun | 1.834 |
| jul | 950 |
| ago | 1.760 |
| sep | 1.299 |
| oct | 1.893 |
| nov | 728 |
| **Enero** | **10.400 (stock inicial, censurado a la izquierda)** |

### 3. Tasa de conversión por mes
| Mes | Tasa | n |
|-----|------|---|
| 202601 | 14,6 % | 10.400 |
| 202602 | 15,7 % | 9.800 |
| 202603 | 14,8 % | 10.300 |
| 202604 | 15,3 % | 9.600 |
| 202605 | 15,7 % | 10.100 |
| 202606 | 15,5 % | 10.350 |
| 202607 | 14,5 % | 9.700 |
| 202608 | 14,4 % | 10.050 |
| 202609 | 14,1 % | 9.900 |
| 202610 | 15,7 % | 10.400 |
| 202611 | 15,1 % | 9.500 |

**χ² homogeneidad:** p = 0,0033 → **no es exactamente constante**. No se explica por % nuevos (r = 0,06).

### 4. Hazard por antigüedad en el panel (tenure)
- Mes 1 (nuevos feb–nov): **16,95 %**
- Mes 2+: **~14,78 %** (decrece a ~12–13 % en tenure alto)

### 5. Variables dinámicas
**Solo `dias_ultima_interaccion`** cambia dentro de un cliente:
- 80,3 % de clientes con >1 mes
- 65,6 % si se incluyen clientes de 1 solo mes
- Las otras 21 columnas son **constantes por cliente**

**Naturaleza del delta mensual de `dias_ultima_interaccion`:**
- Media: 0,1 días
- SD: 117,8 días
- 36,5 % deltas = 0 (sin cambio)
- 0,16 % deltas = +30
- **No es un contador "días desde"**: se comporta como re-muestreo ruidoso
- Igual a `dias_ultima_transaccion` en 54,9 % de filas
- ≤ `dias_ultima_transaccion` en 77,5 % de filas

### 6. Relación con el objetivo
| Variable | Efecto |
|----------|--------|
| `banda_riesgo` | low 18,6 %, medium 14,0 %, high 8,7 % |
| `numero_productos` | 1→5: 12,7 %, 14,8 %, 19,2 %, 19,9 %, 21,1 % |
| **Interacción banda × productos** | **low**: 13,8 % (1 prod) → **31,3 % (5 prod)**; **high**: plana 8,2 %–10,7 % |

### 7. Correlación lineal (Pearson) con objetivo
| Variable | r |
|----------|---|
| `numero_productos` | +0,076 |
| `dias_ultima_transaccion` | −0,058 |
| `dias_ultima_interaccion` | −0,036 |
| `ratio_deuda_ingresos` | −0,012 |
| `antiguedad_cuenta_meses` | +0,011 |
| `saldo_promedio` | +0,008 |
| `edad` | +0,007 |
| `ingresos` | +0,006 |
| `distancia_sucursal_km` | +0,005 |
| Resto | ≈ 0 |

### 8. Importancia por permutación (val sep–nov, RF, 5 permutaciones)
| Variable | Caída AUC |
|----------|-----------|
| `banda_riesgo` (combinado) | **0,069** |
| `numero_productos` | **0,035** |
| `dias_ultima_transaccion` | **0,020** |
| `activo_movil` | 0,008 |
| `tiene_tarjeta_credito` | 0,006 |
| Resto (17 vars) | **≤ 0,0014 (ruido)** |

### 9. AUC Univariado (val sep–nov)
| Variable | AUC |
|----------|-----|
| `numero_productos` | 0,550 |
| `dias_ultima_transaccion` | 0,547 |
| `dias_ultima_interaccion` | 0,519 |
| `activo_movil` | 0,516 |
| `tiene_tarjeta_credito` | 0,511 |
| Resto | ≤ 0,510 |

### 10. Ablación de features (Gini, media 3 seeds, LightGBM mismos hiperparámetros)
| Conjunto | val sep–nov | val jun–ago |
|----------|-------------|-------------|
| A. Pipeline actual (50 feats) | 0,2471 | 0,2582 |
| B. Solo estáticas + valor actual | **0,2499** | 0,2580 |
| C. B + lags/delta/roll interacción | 0,2446 | 0,2564 |
| D. B + tenure + cohorte_ene | 0,2489 | 0,2569 |
| E. D + flags interacción | 0,2510 | 0,2565 |
| **Todas estáticas, LightGBM regularizado** | **0,2516** | **0,2638** |

**Conclusión:** Temporales no aportan (C < B). Regularizar es lo único consistente (+0,003 a +0,008). Ensemble no mejora (0,2513–0,2517 vs 0,2516).

---

## Validación y Métrica

- **Métrica:** Gini = 2 × AUC − 1
- **Validación honesta:** Walk-forward temporal (folds: val dic, nov, oct, sep). Bootstrap por cliente (200 réplicas) para IC95 % del Gini.
- **SE del Gini:** 0,0092 → IC95 % ≈ ±0,018
- **Partición aleatoria infla Gini** +0,015–0,025 (data leakage por clientes repetidos)

---

*Diccionario de datos generado desde hechos verificados - DataFest 2026 Team*