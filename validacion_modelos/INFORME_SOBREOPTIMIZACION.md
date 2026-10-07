# Informe de sobreoptimización: ¿la ventaja del LightGBM superficial es real o sesgo de selección?

Fecha: 7 de octubre de 2026. Carpeta: `validacion_modelos/`. Scripts: `sobreopt_comun.py` (candidatos y criterios), `21_registro_candidatos.py`, `22_pruebas_seleccion.py`, `23_anidado_placebo.py`. Métrica: Gini = 2·AUC − 1 sobre los folds walk-forward ago–nov 2026, con el esquema del pipeline nuevo (sin early stopping; nº de árboles de cada fold por regla 1-SE sobre los demás folds).

Etiquetas: **CONFIRMADO** = calculado en esta carpeta; **PLAUSIBLE** = inferencia.

## 0. Resumen

- **El régimen "LightGBM superficial" supera al modelo actual de forma robusta a la selección.** CONFIRMADO.
  - Walk-forward anidado: la receta completa, decidida solo con el pasado de cada mes, da +0,0081 frente al actual. Su optimismo frente al Gini aparente es de **0,0005**, por debajo del criterio de 0,002.
  - SPA de Hansen sobre los 37 candidatos: p = **0,025**.
  - CSCV: el ganador dentro de muestra pierde contra el actual fuera de muestra solo en el **2,3 %** de las particiones.
- **No se puede identificar cuál configuración superficial concreta es la mejor, ni afirmar que `lgb_sup_sin_dui` valga exactamente +0,005.** CONFIRMADO.
  - Romano-Wolf: p = 0,21 para `lgb_sup_sin_dui`.
  - z deflactado = 1,26 < 1,645.
  - P(ser el mejor) ≈ 0 entre 37 candidatos casi idénticos.
  - La receta anidada elige una variante distinta cada mes.
- **Tres pruebas no pasan, y por motivos identificados.** Ninguna de ellas niega la conclusión anterior:
  - El Reality Check de White (p = 0,075) no está estudentizado y tiene menos potencia que el SPA.
  - El MCS (34 de 37 modelos dentro) tiene muy poca potencia con candidatos tan parecidos.
  - El placebo de la receta (p = 0,86) resultó **no ser un nulo válido**: los árboles grandes memorizan clientes y recuperan el efecto de permanencia aun con las variables barajadas (§3.2).
- **Recomendación:** mantener `lgb_sup_sin_dui`, el preset central de la meseta, fijado antes de este análisis. No sustituirlo por el ganador del registro (`r3_learning_rate_alto`): sería elegir el máximo de 37 y caer en la maldición del ganador. La mejora esperada en diciembre es de **+0,004 a +0,006**, con una estimación central de +0,0054 (anidado con el preset fijo). PLAUSIBLE.

## 1. Diseño

### 1.1 Candidatos (`sobreopt_comun.candidatos`)

| Grupo | Candidatos | N |
|---|---|---|
| Presets del pipeline × {con, sin dui} | lgb_reg, lgb_sin_reg, lgb_sup, xgb, catboost, rf | 12 |
| Vecindad R3 del superficial (sin dui) | 8 cambios de un factor (num_leaves 4/16, min_child 100/400, λ 3/30, lr 0,015/0,06) + 16 esquinas ×½/×2 | 24 |
| Otro | ebm_sin_dui (solo en el registro y en las pruebas 22) | 1 |
| **Total** | | **37** |

- **Referencia:** `lgb_reg_con_dui`, el modelo actual.
- **Quedan fuera** del historial, porque no se pueden reproducir con este esquema: las redes neuronales, los ensembles y Optuna. El número real de intentos es mayor que 37, así que las correcciones por multiplicidad de este informe son un **mínimo**.

### 1.2 Esquema común

- **Registro (21):** las predicciones honestas de cada candidato sobre las **mismas** filas de ago–nov (40 300). Reproduce el pipeline: lgb_sup_sin_dui 0,2540 y lgb_reg_con_dui 0,2487, iguales a `Pipeline_DF_WCB.py`.
- **Unidad de remuestreo:** el cliente. Un cliente entra o sale con todas sus filas de los 4 meses.
- **Bootstrap:** 2 000 réplicas con reemplazo, comunes a todos los candidatos, así que todas las diferencias son pareadas.

### 1.3 Criterios

Los criterios se fijaron en `sobreopt_comun.CRITERIOS_SO` antes de ejecutar 21–23. El analista ya conocía los resultados de R1–R9 y del script 20, igual que en el pre-registro anterior.

## 2. Resultados

### P1. Walk-forward anidado (`sobreopt_anidado_{candidatos,resumen}.csv`) — **PASA**. CONFIRMADO

Para cada mes externo M, todo se decide con meses < M:
- el orden de variables;
- la evaluación de los 36 candidatos (sin EBM), con los 3 meses anteriores como folds internos;
- el número de árboles por regla 1-SE;
- el ganador por Gini medio interno.

Después se entrena con todo lo anterior a M y se evalúa una vez en M.

| Mes externo | Ganador interno | Rango externo (de 36) | Receta | Actual | Δ receta | lgb_sup_sin_dui fijo | Δ fijo |
|---|---|---|---|---|---|---|---|
| ago | lgb_sup_sin_dui | 2 | 0,2705 | 0,2536 | +0,0169 | 0,2705 | +0,0169 |
| sep | r3_reg_lambda_alto | 17 | 0,2479 | 0,2477 | +0,0002 | 0,2450 | −0,0026 |
| oct | r3_min_child_samples_bajo | 5 | 0,2660 | 0,2605 | +0,0055 | 0,2649 | +0,0044 |
| nov | r3_learning_rate_alto | 1 | 0,2427 | 0,2328 | +0,0099 | 0,2359 | +0,0031 |
| **media** | | 6,25 | **0,2568** | 0,2486 | **+0,0081** | 0,2541 | +0,0054 |

- **Optimismo** = Gini aparente del mejor del registro (0,2573) − Gini anidado de la receta (0,2568) = **0,0005**.
- **Fundamento:** la CV que selecciona y evalúa en los mismos datos está sesgada al alza (Varma y Simon, 2006). En el anidado, el mes evaluado nunca interviene en la selección, y además solo se usa el pasado. Que el optimismo sea casi nulo indica que elegir dentro de este conjunto **no** infla el resultado: los candidatos son tan parecidos que el ganador aparente y el anidado rinden igual.
- **La receta siempre elige un modelo del régimen superficial y nunca queda por debajo del actual.** El preset fijo pierde en septiembre (−0,0026). Su +0,0054 es la estimación más directa de lo que aportará en diciembre.
- **Limitación:** solo hay 4 meses externos (SE de la media ≈ 0,008). Es una comprobación de sesgo, no una medida precisa de nivel.

### P2. PBO por CSCV (`sobreopt_pbo.csv`) — **DUDOSO**. CONFIRMADO

- **Construcción:** 12 bloques de clientes (cada cliente, con todos sus meses, en un solo bloque), lo que da las 924 particiones en dos mitades. En cada partición:
  - el ganador de la mitad dentro de muestra se ordena entre los 37 en la otra mitad;
  - λ = logit del rango relativo.
- **Fundamento:** Bailey, Borwein, López de Prado y Zhu (2014, 2017). Si la selección no tuviera información, el rango fuera de muestra del ganador sería uniforme y PBO ≈ 0,5.

| Medida | Valor | Criterio | Resultado |
|---|---|---|---|
| PBO = P(λ ≤ 0) | **0,137** | < 0,10 fiable; > 0,30 ruido | zona dudosa |
| P(el ganador dentro de muestra pierde contra el actual fuera de muestra) | **0,023** | < 0,05 | pasa |
| Pendiente OOS ~ IS del ganador | −0,97 | > 0 | **criterio mal definido** (§3.1) |

Ganadores dentro de muestra más frecuentes: r3_learning_rate_alto (421 de 924), r3_reg_lambda_bajo (207), r3_esq_bAbA (99), r3_min_child_samples_alto (76). Todos son superficiales.

**Lectura:** en 1 de cada 7 particiones, el mejor dentro de muestra cae por debajo de la mediana fuera de muestra. Elegir **el mejor exacto** dentro de la meseta es poco fiable. Pero casi nunca (2,3 %) el elegido pierde contra el modelo actual. La selección del **régimen** es fiable; la de la **configuración concreta**, no.

### P3. Data snooping: White, Hansen y Romano-Wolf (`sobreopt_spa.csv`) — **PASA (SPA)**. CONFIRMADO

| Contraste | Estadístico | p | Resultado |
|---|---|---|---|
| Reality Check de White (2000) | max Δ = 0,0086 | 0,075 | no pasa |
| SPA consistente de Hansen (2005) | max t = 2,83 | **0,025** | **pasa** |
| Romano-Wolf (2005), lgb_sup_sin_dui | t = 1,82 (p individual 0,037) | 0,21 | no pasa |
| Romano-Wolf: candidatos con p < 0,05 | — | — | 4 (r3_learning_rate_alto 0,029, r3_reg_lambda_bajo 0,028, r3_min_child_samples_alto 0,036, r3_learning_rate_bajo 0,040) |

- **Fundamento:** el máximo de N diferencias ruidosas es positivo aunque ningún candidato sea mejor. El bootstrap recentrado da la distribución de ese máximo bajo H0, conservando la correlación real entre candidatos.
- **El SPA** estudentiza cada diferencia y no deja que los candidatos claramente peores (RF) inflen la distribución nula. Por eso tiene más potencia que White.
- **Romano-Wolf** (stepdown) controla el error de toda la familia para identificar qué candidatos concretos son mejores.
- **Lectura:** se rechaza que ningún candidato supere al actual (SPA). Pero no se puede afirmar para `lgb_sup_sin_dui` en particular, tras corregir por las 36 comparaciones.

### P4. Gini deflactado — **PASA el máximo, NO PASA lgb_sup_sin_dui**. CONFIRMADO

- **N efectivo** = 2,10, calculado como el ratio de participación de los autovalores de la correlación de los Δ bootstrap. Los 37 candidatos equivalen a unos 2 independientes.
- **Máximo esperado bajo H0:**
  - con la fórmula de Bailey y López de Prado (2014): 0,0017;
  - con el bootstrap recentrado: media 0,0043, percentil 95 0,0093.
- **z deflactado:**
  - del máximo (r3_learning_rate_alto, Δ = 0,0086): **2,23** > 1,645;
  - de lgb_sup_sin_dui (Δ = 0,0053): **1,26** < 1,645.
- **Lectura:** la fórmula supone candidatos de igual varianza y subestima el máximo nulo frente al bootstrap. El bootstrap es la referencia: el Δ máximo observado (0,0086) queda justo **por debajo** de su percentil 95 (0,0093), lo que coincide con el p = 0,075 de White.

### P5. Placebo de la receta (`sobreopt_placebo.csv`) — **NO PASA, nulo no válido**. CONFIRMADO

- **Construcción:** se permutan los vectores de variables fijas entre clientes, manteniendo intactos etiquetas, meses y trayectorias, y dui se permuta dentro del mes. Se corre la receta con los 32 candidatos LightGBM y XGBoost, 20 réplicas.
- **Resultado:** Δ placebo (ganador − actual) medio 0,0159, percentil 95 0,0275; Δ real (mismo subconjunto) 0,0086; p = 0,857.
- **Ganadores placebo:** siempre árboles grandes. lgb_sin_reg_sin_dui en 10 de 20, xgb_sin_dui en 6, lgb_reg_sin_dui en 2 y lgb_sin_reg_con_dui en 2. Ningún superficial.

**Este nulo no es válido para la pregunta planteada** (§3.2). El Gini medio de los candidatos placebo es 0,012–0,035, no ≈ 0: queda señal.

### P6. Probabilidad de ser el mejor y MCS (`sobreopt_prob_mejor.csv`, `sobreopt_mcs.csv`) — **sin poder discriminante**. CONFIRMADO

- **P(ser el mejor) en el bootstrap:**
  - r3_learning_rate_alto 0,41; r3_reg_lambda_bajo 0,19; r3_esq_AbAA 0,09; r3_esq_bAbA 0,07…
  - lgb_sup_sin_dui 0,00; actual 0,0005.
- **MCS al 90 % (Hansen, Lunde y Nason, 2011; estadístico T_max):** solo elimina rf_sin_dui (p_MCS 0,000), rf_con_dui (0,002) y xgb_con_dui (0,050). Los otros 34 quedan dentro (p_MCS = 0,21), **incluido el actual**.
- **Lectura:**
  - La probabilidad de ser el mejor se reparte entre variantes casi idénticas, y es ≈ 0 para cualquier configuración concreta, incluida la elegida.
  - El MCS compara cada modelo con la **media** del conjunto y corrige por 37. Con diferencias de 0,003–0,008 y SE ≈ 0,003, no tiene potencia para separar el actual. El contraste dirigido (SPA, frente al actual) sí la tiene.

## 3. Desviaciones y hallazgos

### 3.1 La pendiente OOS ~ IS de CSCV estaba mal planteada como criterio

En CSCV, las dos mitades de cada partición son **complementarias**. Si la mitad dentro de muestra recibe clientes "fáciles de ordenar", la de fuera recibe los difíciles, así que el Gini dentro y fuera de muestra del mismo modelo están negativamente correlacionados **por construcción**. La pendiente −0,97 refleja esa composición, no sobreajuste. Debió definirse sobre rendimientos relativos (frente a la mediana de cada mitad). Se informa, pero no se usa para decidir.

### 3.2 El placebo conserva señal: los árboles grandes memorizan clientes

Comprobación aparte, un ajuste sobre datos placebo con fold noviembre:

| Modelo sobre datos placebo | Gini nov | Spearman(predicción, permanencia en el panel) |
|---|---|---|
| LightGBM de 31 hojas (lgb_sin_reg) | 0,044 | −0,35 |
| LightGBM de 8 hojas (superficial) | 0,006 | −0,15 |
| Solo la permanencia (−meses previos), sin modelo | 0,077 | — |

**Mecanismo:**
- El vector de variables de cada cliente es único y constante.
- Un cliente presente en el mes M aparece en meses anteriores siempre con etiqueta 0, porque al convertir sale del panel.
- Un árbol con hojas pequeñas aprende "este vector → 0" y baja la probabilidad de los clientes con más historia.
- Como el hazard real cae con la permanencia (17 % en el primer mes frente a ~13 % tras 7 meses), eso produce Gini real aun sin señal en las variables.

**Consecuencias:**
1. El Δ placebo mide sobre todo la **capacidad de memorizar** de los modelos grandes frente al actual, no la suerte de elegir. Por eso no es un nulo válido para el sesgo de selección.
2. **Hallazgo nuevo:** la permanencia en el panel tiene por sí sola un Gini de 0,077 en noviembre y **no es una variable del modelo**. La ablación D (tenure + cohorte_ene) no mejoró, pero se evaluó con el esquema antiguo y los presets de 31 hojas, que ya la capturan en parte por memorización. Con árboles superficiales, que no memorizan, podría aportar. Es una línea para probar, con validación temporal y una variable calculada solo con el pasado. **PLAUSIBLE.**
3. Un placebo correcto tendría que romper también la identidad del cliente (por ejemplo, re-sortear el donante en cada mes), pero eso altera la naturaleza constante de las variables. Para la pregunta de selección, el contraste adecuado es el SPA, que usa la correlación real entre candidatos.

## 4. Conclusión y recomendación

| Pregunta | Respuesta | Evidencia |
|---|---|---|
| ¿La mejora frente al actual se explica por sesgo de selección? | **No** | Anidado: optimismo 0,0005; SPA p = 0,025; CSCV 2,3 % de pérdida |
| ¿Es identificable el mejor candidato concreto? | **No** | P(mejor) ≈ 0 para cualquiera; RW p = 0,21; ganador anidado distinto cada mes; PBO 0,14 |
| ¿Cuánto vale la mejora? | +0,004 a +0,006 (central +0,0054) | Anidado del preset fijo: +0,0054, peor mes −0,0026 |

**Recomendación:**
1. **Entregar `lgb_sup_sin_dui`** (el preset central): se fijó antes de este análisis, está en el centro de la meseta y es el ganador por defecto del pipeline. Elegir ahora `r3_learning_rate_alto` (lr 0,06, ~94 árboles) porque encabeza el registro sería el error que mide este informe: su ventaja sobre el preset (+0,003) está dentro del ruido y no se repite fuera de muestra (en el anidado ganó en noviembre, pero en septiembre la receta quedó en el puesto 17).
2. **Comunicar la mejora con su incertidumbre:** "≈ +0,005 de Gini, consistente con la evidencia, pero por debajo del umbral de ~0,01 para declararla con firmeza".
3. **Opcional, sin probar:** promediar varias variantes de la meseta (por ejemplo, las 5–8 mejores del registro) eliminaría el riesgo de elegir una concreta. Si se prueba, debe hacerse con el walk-forward anidado.
4. **Línea nueva:** una variable de permanencia en el panel (§3.2).

## 5. Reproducción

Desde la raíz del repo, con `.venv`:
```
.venv/Scripts/python.exe validacion_modelos/scripts/21_registro_candidatos.py     # ~9 min
.venv/Scripts/python.exe validacion_modelos/scripts/22_pruebas_seleccion.py       # ~2 min
.venv/Scripts/python.exe validacion_modelos/scripts/23_anidado_placebo.py todo 20 # ~22 min anidado + ~70 min placebo
```
Salidas: `resultados/sobreopt_*.csv`, `resultados/sobreopt_registro.npz` y los logs en `resultados/logs/2{1,2,3}_*.log`. No se escribió nada fuera de `validacion_modelos/`.
