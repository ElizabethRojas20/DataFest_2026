# Informe de validación de modelos sobre datos nuevos (diciembre 2026)

Responsable: ingeniero de ML. Fecha: 2026-10-05. Todo lo de este informe se reproduce con los scripts de `validacion_modelos/scripts/` (semillas fijas; se verificó que repetir `01_walkforward.py` da predicciones idénticas, diferencia máxima 0,0).

Etiquetas: **CONFIRMADO** = calculado en esta carpeta; **PLAUSIBLE** = inferencia.

## 1. Conclusión

**Ningún modelo se cae en diciembre.** Con los criterios fijados antes de ver los resultados (§3), los 9 modelos evaluados salen como "NO SE CAE":

- Las variables estáticas no cambian entre noviembre y diciembre: PSI < 0,01 en todas y AUC adversarial de 0,50.
- Los scores de un mismo modelo apenas se mueven entre noviembre y diciembre (PSI ≤ 0,025).
- En diciembre los modelos coinciden en el ranking igual o más que en noviembre.
- El Gini estimado en condiciones de diciembre está dentro del ruido del Gini actual.

El único cambio real es que `dias_ultima_interaccion` (dui) pierde toda su señal en diciembre. Los modelos dependen tan poco de ella que la pérdida estimada es de 0,001–0,003 de Gini, menos de 2 SE pareados. **CONFIRMADO**

Lo que sí distingue a los modelos es su **nivel**, no su estabilidad. RandomForest (−0,010 frente a LightGBM regularizado), MLP (−0,059) y la red Keras baseline (−0,016) son peores de forma significativa y lo seguirán siendo en diciembre. LightGBM (con y sin regularizar), XGBoost, CatBoost y LightGBM sin dui son indistinguibles entre sí. **CONFIRMADO**

## 2. Diseño

`test.csv` no tiene `objetivo`, así que el Gini de diciembre no se puede medir. Se combinan cuatro pruebas indirectas: drift de variables, drift de scores, estabilidad del ranking y Gini en folds pasados manipulados para que se parezcan a diciembre.

**Modelos evaluados.**
- Reentrenados con `PARAMS_DEFECTO` (`pipelines/Pipeline_DF_WCB.py:523-534`) sobre las 39 columnas del modo `estatico`, que coinciden exactamente con `features_seleccionadas_DF_WCB.txt`: `lgb_reg`, `lgb_sin_reg`, `xgb`, `catboost` y `random_forest`. Se añade `lgb_reg_sin_dui` (los mismos parámetros que `lgb_reg`, sin dui).
- Guardados, sin reentrenar: `mlp_sklearn`, `nn_baseline` y `nn_optimizado`.
- La entrega `resultados_reportes/submission_final.csv`, que solo se lee.

**Walk-forward honesto.** No se usa `EvaluadorModelos`. Para cada fold M ∈ {ago, sep, oct, nov}:
1. Se entrena con los meses anteriores a M−1 y se hace early stopping sobre M−1 (semilla 42) para obtener `best_iter`.
2. Se reentrena con todos los meses anteriores a M, usando `best_iter × filas(<M)/filas(<M−1)` árboles y 3 semillas (42, 43 y 44).
3. Se evalúa en M.

Como comparación se calcula también el **esquema del pipeline**: early stopping sobre el mismo M que se evalúa (`Pipeline_DF_WCB.py:614-617`), con semilla 42.

**IC95 %.** Bootstrap de clientes con reemplazo (1000 réplicas). Un cliente remuestreado entra con todas sus filas de los 4 folds, y las réplicas son las mismas para todos los modelos, así que las diferencias son pareadas.

**Modelo final.** Se entrena con todo ene–nov. Las iteraciones son la mediana sobre los folds de `best_iter_k × filas_totales/filas_ES_k`, y se promedian 3 semillas.

## 3. Criterios del veredicto (fijados antes de ejecutar; `scripts/comun.py`, `CRITERIOS`)

| Criterio | Medida | Umbral |
|---|---|---|
| C1 drift de scores | PSI de scores del **mismo** modelo (entrenado ≤ oct) en nov frente a dic | 0,10–0,25 moderado; ≥ 0,25 severo |
| C2 caída de Gini | Gini medio en el escenario "diciembre" (e: dui permutada + reponderación adversarial + 18,6 % de nuevos) frente a tal cual | significativa si es > 2·SE pareado; material si además es > 0,02 |
| C3 ranking | Spearman medio con los demás modelos (dic − nov) y entre semillas (dic − nov) | ≤ −0,05 (modelos) o ≤ −0,02 (semillas) |
| C4 covariate shift (datos) | AUC adversarial | > 0,60 relevante |

Regla: **SE CAE** si C2 es material o C1 ≥ 0,25. **DEGRADACIÓN MODERADA** si C2 es significativa, C1 ≥ 0,10 o se cumple C3. En otro caso, **NO SE CAE**.

## 4. Resultados

### 4.1 Rendimiento de referencia: walk-forward honesto (`resultados/wf_resumen.csv`, `wf_gini_por_fold.csv`) **CONFIRMADO**

Gini por fold (honesto, promedio de 3 semillas):

| Modelo | ago | sep | oct | nov | Media ago–nov | IC95 % | Δ frente a lgb_reg (SE pareado) | Folds mejores que lgb_reg |
|---|---|---|---|---|---|---|---|---|
| lgb_reg | 0,2545 | 0,2430 | 0,2650 | 0,2262 | **0,2472** | [0,232; 0,264] | — | — |
| lgb_sin_reg | 0,2542 | 0,2484 | 0,2637 | 0,2290 | 0,2488 | [0,233; 0,265] | +0,0016 (0,0018) | 2/4 |
| xgb | 0,2521 | 0,2384 | 0,2608 | 0,2302 | 0,2454 | [0,229; 0,261] | −0,0018 (0,0023) | 1/4 |
| catboost | 0,2599 | 0,2471 | 0,2610 | 0,2355 | 0,2509 | [0,236; 0,267] | +0,0037 (0,0026) | 3/4 |
| random_forest | 0,2514 | 0,2312 | 0,2503 | 0,2142 | 0,2368 | [0,220; 0,253] | **−0,0104 (0,0044)**, IC [−0,019; −0,002] | 0/4 |
| lgb_reg_sin_dui | 0,2675 | 0,2430 | 0,2561 | 0,2320 | 0,2496 | [0,234; 0,265] | +0,0025 (0,0024) | 3/4 |

- **Varianza.** La SE del Gini medio de 4 folds es ≈ 0,008 y la de un fold ≈ 0,016. La dispersión entre folds (0,226–0,265) es la esperable por ruido más una variación real entre meses. Entre semillas, el Gini medio varía ≤ 0,006 (CatBoost) y ≤ 0,002 en LightGBM.
- **Diferencias entre boosting.** Ninguna es significativa: todas las diferencias pareadas están por debajo de 2 SE. RandomForest es peor de forma significativa.
- **Optimismo del esquema del pipeline** (early stopping sobre el fold evaluado). Frente a la semilla 42 sola es de +0,0064 en lgb_reg, +0,0045 en lgb_sin_reg, +0,0076 en xgb, +0,0044 en catboost, +0,0054 en lgb_reg_sin_dui y 0 en RF (no usa ES). En lgb_reg, frente al promedio de 3 semillas, es de +0,0057 con IC [0,0002; 0,0113]. Es decir, el Gini que reporta el pipeline está inflado unas **0,005**: del orden de 0,6 SE, no cambia el orden de los modelos, pero se come el margen de 0,003 de la regla de reemplazo.
- **`best_iter` muy inestable** (`wf_iteraciones.csv`). En lgb_reg, el ES honesto da 25, 141, 29 y 8 árboles según el fold, y el esquema del pipeline 141, 29, 8 y 9. El modelo final de lgb_reg acaba con 43 árboles, mientras que `lgb_reg_sin_dui`, con los mismos hiperparámetros, acaba con 224. La curva de AUC frente al número de árboles es plana y el ES elige casi al azar. **CONFIRMADO**. Que el Gini sea poco sensible al número de árboles en ese rango es **PLAUSIBLE**: el Gini honesto y el del pipeline se parecen aunque los árboles difieran en un factor de hasta 10.

### 4.2 Redes neuronales guardadas: procedencia y rendimiento (`nn_procedencia.csv`) **CONFIRMADO**

- `X_train_nn.npy` tiene 80 300 filas, que son exactamente ene–ago, y `X_val_nn.npy` tiene 29 800, que son sep–nov. Ambas se reproducen con diferencia 0 aplicando `preprocessor_nn.joblib`. Las redes se entrenaron con ene–ago y **agosto está dentro de su entrenamiento**: su Gini en agosto, de 0,35–0,46, es dentro de muestra y no se usa.
- El `StandardScaler` del preprocesador se ajustó con **todo train** (`n_samples_seen_ = 110 100`). Es una fuga no supervisada leve y contradice la nota de `INFORME_FINAL_DATAFEST_2026.md` §5.3 ("ajustados solo en train").
- `nn_baseline` y `nn_optimizado` (Keras) usan early stopping y ReduceLROnPlateau sobre `val_auc` de **sep–nov** (celda 17 del notebook), así que su Gini en sep, oct y nov **no es fuera de muestra** y es optimista. `mlp_sklearn` usa un 10 % aleatorio de ene–ago para el early stopping, así que en sep–nov sí está fuera de muestra, salvo por el escalado.
- Discrepancias de documentación: `nn_baseline.keras` guardada es 128-64 y no la 256-128-64-32 del código de la celda 17. El MLP guardado es (256,128,64,32) con α = 1e-4, no (128,64,32) con α = 1e-3 como dice `INFORME_FINAL` §5.3. No hay código en el repo que entrene `nn_optimizado`; que use el mismo split ene–ago es **PLAUSIBLE**.

Gini en sep–nov (3 folds) frente a lgb_reg en los mismos meses (0,2447):

| Modelo | sep | oct | nov | Media | IC95 % | Δ frente a lgb_reg (SE) | Nota |
|---|---|---|---|---|---|---|---|
| mlp_sklearn | 0,1965 | 0,1794 | 0,1800 | 0,1853 | [0,167; 0,203] | **−0,0594 (0,0079)** | fuera de muestra |
| nn_baseline | 0,2347 | 0,2378 | 0,2134 | 0,2287 | [0,211; 0,246] | **−0,0161 (0,0056)** | optimista |
| nn_optimizado | 0,2420 | 0,2488 | 0,2277 | 0,2395 | [0,223; 0,257] | −0,0052 (0,0055) | optimista |

Incluso con su ventaja optimista, las redes no superan a LightGBM. El MLP sobreajusta mucho (0,455 dentro de muestra en agosto frente a 0,18 fuera). Encaja con lo esperado en datos tabulares pequeños, con señal débil y variables heterogéneas: los árboles cortan umbrales en variables individuales, mientras que el MLP tiene que aprender esas discontinuidades con ~56 000 parámetros a partir de ~80 000 filas.

### 4.3 Drift de variables (`drift_psi_variables.csv`, `drift_dui_por_mes.csv`, `adversarial_auc.csv`; fig02, fig03) **CONFIRMADO**

- **PSI por variable.** Las 22 columnas originales tienen PSI < 0,01, tanto nov→dic (máximo 0,0004) como ene–nov→dic (máximo 0,0086, `banda_riesgo`). La propia `dias_ultima_interaccion` tiene un PSI marginal de 0,0003: **su distribución no cambia**, lo que cambia es su relación con `dias_ultima_transaccion`. El indicador derivado `dui == dut` tiene PSI 0,34 (nov→dic) y 3,35 (ene–nov→dic).
- **Ruptura de dui por mes.** El porcentaje de filas con dui == dut es 100, 91, 82, 73, 64, 55, 46, 37, 28, 18 y 9 de enero a noviembre, y **0,26 en diciembre**. La correlación dui–dut en diciembre es −0,003.
- **Clientes nuevos (% de filas).** Agosto 17,5, septiembre 13,1, octubre 18,2, noviembre 7,7 y diciembre 18,6.

Validación adversarial (LightGBM, StratifiedGroupKFold por cliente, 5 folds):

| Comparación | sin dui | con dui | con dui + indicador dui==dut |
|---|---|---|---|
| nov frente a dic | **0,501** | 0,518 | 0,532 |
| ago–nov frente a dic | 0,501 | 0,594 | 0,606 |
| ene–nov frente a dic | 0,522 | **0,754** | 0,775 |

Sin dui, diciembre es **indistinguible** de noviembre (0,501). Con dui, el árbol detecta la ruptura **por sí solo**, cruzando dui con dut: en ene–nov frente a dic, la importancia de dut es 0,38 y la de dui 0,27. El indicador explícito apenas añade (0,754 → 0,775, con una importancia del indicador de 0,69). C4 solo se cumple para ene–nov con dui, es decir, el único desplazamiento es la ruptura conocida de dui.

### 4.4 Drift de predicciones (`drift_predicciones.csv`, `escala_scores.csv`; fig04) **CONFIRMADO**

| Modelo | PSI mismo modelo nov→dic (C1) | PSI nuevos | PSI antiguos | PSI nov (≤ oct) → dic (final), informativo | Media nov | Media dic final |
|---|---|---|---|---|---|---|
| lgb_reg | 0,0017 | 0,011 | 0,003 | 0,675 | 0,148 | 0,144 |
| lgb_sin_reg | 0,0016 | 0,004 | 0,003 | 0,544 | 0,148 | 0,145 |
| xgb | 0,0045 | 0,027 | 0,004 | 0,052 | 0,137 | 0,140 |
| catboost | 0,0013 | 0,029 | 0,004 | 0,006 | 0,139 | 0,140 |
| random_forest | 0,0245 | 0,015 | 0,005 | 0,013 | 0,130 | 0,134 |
| lgb_reg_sin_dui | 0,0024 | 0,012 | 0,002 | 0,005 | 0,138 | 0,138 |
| mlp_sklearn | 0,0018 | 0,025 | 0,002 | (mismo modelo) | 0,137 | 0,140 |
| nn_baseline | 0,0011 | 0,024 | 0,002 | (mismo modelo) | 0,133 | 0,135 |
| nn_optimizado | 0,0016 | 0,011 | 0,002 | (mismo modelo) | 0,134 | 0,136 |

La tasa real es de 15,15 % en noviembre y de 14–16 % en el histórico. Las medias de los scores están entre 13 y 15 %: no hay deriva de nivel.

- **Con el mismo modelo, los scores no cambian** (PSI ≤ 0,025). El PSI algo mayor de RF (0,025) se explica por la composición: sus scores para clientes nuevos son más altos (0,179 frente a 0,126) y los nuevos pasan del 7,7 al 18,6 %. Dentro de cada segmento, el PSI es ≤ 0,015.
- **Los PSI altos "modelo final" de lgb_reg (0,68), lgb_sin_reg (0,54) y la entrega (1,01) no son drift de datos.** Se deben a la **escala** del modelo. El modelo de lgb_reg entrenado hasta octubre tiene 10 árboles (desviación típica de los scores 0,0125), y el final tiene 43 (0,038). Con un learning rate de 0,03, pocos árboles comprimen los scores alrededor de la tasa base. El ranking no se ve afectado; el PSI sobre scores brutos no es comparable entre modelos con distinto número de árboles.

### 4.5 Estabilidad del ranking (`ranking_resumen.csv`, `ranking_topk.csv`, `dependencia_dui.csv`; fig05) **CONFIRMADO**

| Modelo | Spearman medio con los otros 8, nov | Ídem, dic (finales) | Δ (C3) | Spearman entre semillas, nov | Ídem, dic | Δ (C3) | Spearman con la entrega (dic) | Top-10 % compartido con la entrega |
|---|---|---|---|---|---|---|---|---|
| lgb_reg | 0,849 | 0,870 | +0,021 | 0,886 | 0,964 | +0,078 | 0,934 | 92 % |
| lgb_sin_reg | 0,845 | 0,864 | +0,019 | 0,889 | 0,967 | +0,078 | 0,930 | 92 % |
| xgb | 0,870 | 0,881 | +0,010 | 0,959 | 0,966 | +0,006 | 0,925 | 91 % |
| catboost | 0,873 | 0,882 | +0,010 | 0,970 | 0,963 | −0,007 | 0,943 | 92 % |
| random_forest | 0,810 | 0,813 | +0,003 | 0,973 | 0,976 | +0,002 | 0,772 | 82 % |
| lgb_reg_sin_dui | 0,870 | 0,878 | +0,007 | 0,968 | 0,966 | −0,002 | 0,924 | 90 % |
| mlp_sklearn | 0,664 | 0,680 | +0,016 | — | — | — | 0,637 | 58 % |
| nn_baseline | 0,810 | 0,817 | +0,007 | — | — | — | 0,819 | 79 % |
| nn_optimizado | 0,798 | 0,810 | +0,012 | — | — | — | 0,826 | 80 % |

- **El acuerdo entre modelos no baja en diciembre en ningún caso.** Sube un poco, porque los modelos finales tienen más datos y más árboles. La subida de 0,89 a 0,96 en el Spearman entre semillas de LightGBM se explica igual: el modelo de noviembre solo tenía 10 árboles.
- **Entrega actual.** Su ranking se parece mucho al de los boosting finales: Spearman 0,92–0,94 y 82–86 % del top-20 % compartido. Su distribución es distinta (bimodal, fig04), lo que indica que no es exactamente el lgb_reg de 43 árboles reproducido aquí (**PLAUSIBLE**: otro número de árboles u otra corrida).
- **Dependencia de dui en diciembre.** El Spearman entre la predicción y la misma predicción con dui permutada es 0,979 en lgb_reg, 0,982 en lgb_sin_reg, 0,981 en xgb, 0,989 en catboost, 0,971 en RF, 0,928 en mlp y 0,97 en las Keras. En diciembre, dui solo añade ruido: la fracción de la varianza del ranking que se debe a dui (1−ρ²) es del 2 % en catboost, el 4 % en LightGBM y XGBoost, el 6 % en RF y las Keras, y el 14 % en el MLP. El Spearman entre la predicción y dui en diciembre sigue siendo de −0,11 en lgb_reg: el modelo sigue "usando" una variable que ya no informa.

### 4.6 Gini esperado en condiciones de diciembre (`gini_dic_escenarios.csv`, `gini_dic_por_fold.csv`, `gini_dic_placebo_adversarial.csv`; fig06) **CONFIRMADO**

Escenarios sobre los folds honestos (media de folds; Δ = cambio frente a "a"; entre paréntesis, la SE pareada):
- (a) tal cual.
- (b) dui permutada dentro del mes, media de 3 permutaciones: es lo que ocurre en diciembre.
- (c) reponderación adversarial p/(1−p), recortada entre los percentiles 1 y 99.
- (d) clientes nuevos reponderados al 18,6 %.
- (e) b+c+d (criterio C2).
- (f) b+d (sensibilidad, sin el adversario).

| Modelo (folds) | a | b | Δb | d | f | Δf | c | e | Δe |
|---|---|---|---|---|---|---|---|---|---|
| lgb_reg (ago–nov) | 0,2472 | 0,2464 | −0,0008 (0,0012) | 0,2474 | 0,2465 | −0,0007 (0,0022) | 0,2517 | 0,2509 | +0,0037 (0,0026) |
| lgb_sin_reg | 0,2488 | 0,2485 | −0,0003 (0,0012) | 0,2486 | 0,2481 | −0,0008 (0,0022) | 0,2528 | 0,2521 | +0,0032 (0,0026) |
| xgb | 0,2454 | 0,2445 | −0,0009 (0,0014) | 0,2469 | 0,2460 | +0,0006 (0,0023) | 0,2496 | 0,2500 | +0,0046 (0,0027) |
| catboost | 0,2509 | 0,2497 | −0,0012 (0,0011) | 0,2508 | 0,2494 | −0,0015 (0,0022) | 0,2545 | 0,2531 | +0,0022 (0,0026) |
| random_forest | 0,2368 | 0,2380 | +0,0012 (0,0018) | 0,2393 | 0,2403 | +0,0035 (0,0026) | 0,2399 | 0,2434 | +0,0066 (0,0030) |
| lgb_reg_sin_dui | 0,2496 | 0,2496 | 0 | 0,2505 | 0,2505 | +0,0009 (0,0019) | 0,2544 | **0,2554** | +0,0058 (0,0024) |
| lgb_reg (sep–nov) | 0,2447 | 0,2420 | −0,0027 (0,0014) | 0,2449 | 0,2420 | −0,0027 (0,0029) | 0,2502 | 0,2475 | +0,0028 (0,0033) |
| mlp_sklearn (sep–nov) | 0,1853 | 0,1847 | −0,0006 (0,0032) | 0,1883 | 0,1876 | +0,0023 (0,0042) | 0,1891 | 0,1910 | +0,0058 (0,0045) |
| nn_baseline (sep–nov) | 0,2287 | 0,2260 | −0,0026 (0,0020) | 0,2273 | 0,2242 | −0,0045 (0,0033) | 0,2323 | 0,2280 | −0,0007 (0,0037) |
| nn_optimizado (sep–nov) | 0,2395 | 0,2364 | −0,0031 (0,0018) | 0,2374 | 0,2350 | −0,0045 (0,0031) | 0,2446 | 0,2401 | +0,0006 (0,0035) |

- **Quitarle a dui su señal (b) cuesta poco.** En ago–nov la pérdida es de 0,000–0,001 y en sep–nov de unos 0,003 (en sep–nov hay más dui coincidente). Ninguna llega a 2 SE pareados. Las redes Keras son las que más pierden (−0,003).
- **El cambio de composición hacia el 18,6 % de nuevos (d) no perjudica.** Los modelos ordenan a los nuevos igual de bien que a los antiguos: lgb_reg tiene un Gini de 0,242 en ambos segmentos (`wf_gini_segmentos.csv`).
- **La reponderación adversarial (c) sube el Gini en +0,003–0,005 en todos los modelos, pero hay que tomarla con cautela.** El adversario no tiene señal (AUC 0,501) y los pesos son casi uniformes: el tamaño efectivo es el 97 % de las filas. Con 5 adversarios placebo (etiqueta barajada), el cambio para lgb_reg va de −0,0029 a +0,0007, y el real es de +0,0046. Que haya un ligero sesgo favorable de composición hacia diciembre (clientes supervivientes) es **PLAUSIBLE**, pero no es material. Por eso se informa también del escenario (f), que es conservador y no depende del adversario: en él ningún modelo cae más de 0,0045, y ninguno de forma significativa.
- **Ranking en el escenario "diciembre".** En (e), `lgb_reg_sin_dui` queda +0,0045 por encima de lgb_reg (SE 0,0025, 3/4 folds). En (f) queda +0,0041 (SE 0,0025, 3/4). CatBoost queda +0,0022 en (e) y +0,0030 en (f). Ninguna diferencia llega a 2 SE y hay un fold peor en cada caso.

## 5. Veredicto por modelo (`resultados/veredicto_modelos.csv`) **CONFIRMADO**

| Modelo | Gini tal cual | Gini escenario dic (IC95 %) | C2: caída e (2·SE) | Sensibilidad f: caída (2·SE) | C1 PSI | C3 Δ modelos / semillas | **Veredicto** |
|---|---|---|---|---|---|---|---|
| lgb_reg | 0,2472 | 0,2509 [0,235; 0,267] | −0,0037 (0,0052) | 0,0007 (0,0044) | 0,0017 | +0,021 / +0,078 | **NO SE CAE** |
| lgb_sin_reg | 0,2488 | 0,2521 [0,236; 0,268] | −0,0032 (0,0052) | 0,0008 (0,0044) | 0,0016 | +0,019 / +0,078 | **NO SE CAE** |
| xgb | 0,2454 | 0,2500 [0,235; 0,266] | −0,0046 (0,0054) | −0,0006 (0,0046) | 0,0045 | +0,010 / +0,006 | **NO SE CAE** |
| catboost | 0,2509 | 0,2531 [0,237; 0,269] | −0,0022 (0,0052) | 0,0015 (0,0044) | 0,0013 | +0,010 / −0,007 | **NO SE CAE** |
| random_forest | 0,2368 | 0,2434 [0,227; 0,260] | −0,0066 (0,0060) | −0,0035 (0,0052) | 0,0245 | +0,003 / +0,002 | **NO SE CAE** (pero es peor en nivel) |
| lgb_reg_sin_dui | 0,2496 | 0,2554 [0,239; 0,271] | −0,0058 (0,0048) | −0,0009 (0,0038) | 0,0024 | +0,007 / −0,002 | **NO SE CAE** |
| mlp_sklearn | 0,1853 | 0,1910 [0,172; 0,208] | −0,0058 (0,0090) | −0,0023 (0,0084) | 0,0018 | +0,016 / — | **NO SE CAE** (pero es muy inferior) |
| nn_baseline | 0,2287* | 0,2280 [0,209; 0,246] | 0,0007 (0,0074) | 0,0045 (0,0066) | 0,0011 | +0,007 / — | **NO SE CAE** (inferior; Gini optimista) |
| nn_optimizado | 0,2395* | 0,2401 [0,221; 0,258] | −0,0006 (0,0070) | 0,0045 (0,0062) | 0,0016 | +0,012 / — | **NO SE CAE** (Gini optimista) |

Las caídas negativas son subidas. \* Early stopping sobre sep–nov: el nivel real es algo menor.

**Entrega actual (`submission_final.csv`).** No se puede aplicar C1 ni C2 porque no hay predicciones suyas fuera de muestra en noviembre. Su ranking coincide al 0,93–0,94 con los boosting finales, un acuerdo igual o mayor que el que tienen entre sí los boosting en noviembre (0,92–0,98). Que la entrega tampoco se cae es **PLAUSIBLE**, por ser un modelo de la misma familia sobre los mismos datos.

## 6. Limitaciones

1. **No hay etiquetas de diciembre.** Todo es indirecto. El escenario (b) reproduce exactamente el mecanismo conocido de diciembre (dui es una permutación de dut dentro del mes), pero cualquier cambio en la relación entre las variables y `objetivo` (concept drift) es invisible para todas estas pruebas.
2. **El bootstrap trata como fijos los modelos, las permutaciones y los pesos.** No incluye la varianza por semilla, que es pequeña (≤ 0,006, §4.1), ni la del adversario, que el placebo acota en ±0,003. Las SE de las diferencias de escenario son, por tanto, algo optimistas. Aun así, ninguna caída se acerca a 2 SE.
3. **Redes.** Solo hay 3 folds evaluables (sep–nov), las Keras tienen un Gini optimista y el escalador tiene una fuga leve. No se reentrenaron, por diseño: se valida lo que hay guardado.
4. **El número de árboles del modelo final es arbitrario**, porque `best_iter` es inestable (§4.1). Las predicciones de diciembre de lgb_reg y lgb_reg_sin_dui usan 43 y 224 árboles. El efecto de esa elección sobre el Gini no se ha medido (queda fuera de alcance).
5. **Muestras pequeñas por segmento.** En noviembre solo hay 728 filas de clientes nuevos, y el Gini por segmento y fold tiene una SE de unos 0,04.
6. **Entrega.** No se reprodujo su modelo y solo se compara su ranking.

## 7. Recomendaciones (priorizadas)

1. **Mantener la entrega o un boosting regularizado: no hay riesgo de caída.** Impacto esperado: 0. Para validarlo, basta con `veredicto_modelos.csv`.
2. **Valorar `lgb_reg_sin_dui` como alternativa más robusta.** En diciembre, dui es ruido puro y quitarla elimina en torno a un 4 % de varianza espuria del ranking de LightGBM. Impacto esperado: +0,003–0,004 en condiciones de diciembre, dentro del ruido (SE 0,0025). **No cumple** la regla de reemplazo (en el walk-forward honesto, +0,0025 con el fold de octubre peor, −0,009), así que solo serviría para desempatar. Para validarlo: `.venv/Scripts/python.exe validacion_modelos/scripts/07_gini_condiciones_diciembre.py` y revisar `dif_vs_lgb_reg` y los folds en `gini_dic_escenarios.csv`.
3. **Fijar el número de árboles del modelo final con un criterio estable** en lugar de la mediana de un ES ruidoso. Por ejemplo, el máximo de la curva media de AUC entre folds, o ES solo sobre AUC (con `first_metric_only` o `metric="auc"`). Impacto esperado: probablemente < 0,005 (**PLAUSIBLE**), pero eliminaría la arbitrariedad de 43 frente a 224 árboles. Para validarlo: Gini honesto de `01_walkforward.py` con n_iter fijo en una rejilla (p. ej. 50, 100, 200 y 400).
4. **Informar siempre del Gini con el esquema honesto.** El del pipeline está inflado unas 0,005 (§4.1), lo que equivale a más que el umbral de 0,003 de la regla de reemplazo.
5. **Descartar RandomForest, MLP y `nn_baseline` como candidatos.** Son peores de forma significativa. `nn_optimizado` solo tendría sentido en un ensemble, y ya se probó (`verificaciones/04_ensemble.py`).
6. **Documentación**, a corregir por quien corresponda (aquí no se ha tocado): `INFORME_FINAL` §5.3 dice que el escalador se ajustó solo en train y da arquitecturas de MLP y redes que no coinciden con los artefactos guardados.

## 8. Reproducción y archivos

Orden de ejecución, desde la raíz del repo (~6 min en total con 12 núcleos):
```
.venv/Scripts/python.exe validacion_modelos/scripts/01_walkforward.py              # ~3 min
.venv/Scripts/python.exe validacion_modelos/scripts/02_modelos_finales.py          # ~1 min (usa wf_iteraciones.csv)
.venv/Scripts/python.exe validacion_modelos/scripts/03_redes_neuronales.py
.venv/Scripts/python.exe validacion_modelos/scripts/04_metricas_walkforward.py
.venv/Scripts/python.exe validacion_modelos/scripts/05_drift_variables.py          # genera pesos_adversariales.csv
.venv/Scripts/python.exe validacion_modelos/scripts/06_drift_predicciones_ranking.py
.venv/Scripts/python.exe validacion_modelos/scripts/07_gini_condiciones_diciembre.py
.venv/Scripts/python.exe validacion_modelos/scripts/08_veredicto_y_figuras.py
```
- `scripts/comun.py`: rutas, constantes, **criterios del veredicto**, PSI, bootstrap de clientes, Gini ponderado rápido (verificado igual a `roc_auc_score` con `sample_weight`).
- `scripts/modelos.py`: definición de los 6 modelos de árboles y ajuste con y sin ES (reutiliza `construir_modelo` del pipeline).
- `resultados/`: CSV de cada paso (§4) y `logs/`. Los CSV de predicciones (`wf_predicciones.csv`, `final_pred_dic.csv`, `nn_*`) ocupan la mayor parte de los ~60 MB.
- `figuras/`: fig01 (Gini por fold, honesto frente a pipeline), fig02 (ruptura de dui y % de nuevos por mes), fig03 (PSI y AUC adversarial), fig04 (distribución de scores), fig05 (Spearman nov frente a dic), fig06 (Gini por escenario).

No se escribió nada fuera de `validacion_modelos/`. `submission_final.csv` solo se leyó.
