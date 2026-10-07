# Informe de robustez R1–R9: comparación de candidatos (DataFest 2026)

Fecha: 6 de octubre de 2026. Carpeta: `validacion_modelos/`. Pre-registro: `scripts/criterios_robustez.py` (23:31, con ENMIENDA_1 a las 23:34, ambas anteriores a cualquier `robustez_*`). Métrica: Gini = 2·AUC − 1 en walk-forward honesto (ago–nov 2026).

## 0. Resumen ejecutivo

- **`lgb_sup_sin_dui`** (LightGBM de 8 hojas, min_child 200, λ = 10, sin `dias_ultima_interaccion`) es el **único candidato estadísticamente MEJOR que la referencia lgb_reg**:
  - Gini 0,2564 frente a 0,2472.
  - Δ = **+0,0092** [IC95 +0,0044; +0,0141].
  - Bootstrap pareado por cliente sobre los 4 folds conjuntos, **p_Holm = 0,005** (10 pares), **4/4 folds** a favor.
  - Lo mismo en el escenario de diciembre (dui permutada + 18,6 % de nuevos): +0,0095.

  **CONFIRMADO.**
- **Cumple la regla de reemplazo de CLAUDE.md** (+0,0092 > +0,003 sin empeorar ningún fold), y la prueba formal R1 confirma que la diferencia no es ruido. Ningún otro candidato la cumple: catboost, catboost_sin_dui y ebm_sin_dui ganan +0,004–0,005 de media, pero pierden en octubre y no son significativos tras Holm. **CONFIRMADO.**
- **Su robustez es la mejor del grupo:**
  - Meseta de hiperparámetros plana; las 25 configuraciones vecinas superan a lgb_reg (R3).
  - Es el menos sensible al número de árboles (R2).
  - Supera el ruido de entrada (R6) y no muestra degradación significativa (R7).
  - Pasa el placebo con z = 11,4 (R8).
  - Mejora frente a lgb_reg en los 10 segmentos (R5).
  - Solo suspende R4, que suspenden todos: el top-10 % de diciembre es estable ante la semilla (95 %), pero no ante el remuestreo de clientes (83 % < 85 %). Es una limitación del problema (señal débil, Gini ≈ 0,25), no del modelo. **CONFIRMADO.**
- **Por qué funciona:** el problema tiene señal débil y la complejidad óptima es baja. Los presets por defecto (31 hojas, depth 6) sobreajustan pronto y hacen el early stopping inestable; quitar dui elimina una variable que en diciembre es ruido puro. De los +0,009, aproximadamente +0,003 vienen de un ES más estable (R2), +0,0025 de quitar dui (informe previo: lgb_reg_sin_dui) y +0,004 de la menor capacidad por árbol. **PLAUSIBLE** (descomposición aproximada y no aditiva).
- **Mejora esperada en diciembre:** entre +0,004 y +0,009 de Gini sobre lgb_reg. El rango está descontado por la selección entre 5 candidatos y por el sesgo de segundo orden. **PLAUSIBLE.**
- **Recomendación:** `validacion_modelos/resultados/submission_candidata_lgb_sup_sin_dui.csv`. La decisión de reemplazar `resultados_reportes/submission_final.csv` es del usuario; aquí no se ha tocado.
## 1. Diseño

### 1.1 Candidatos (`scripts/robustez_comun.py::definir_candidatos`)
| Candidato | Definición | Variables |
|---|---|---|
| lgb_reg (referencia) | `PARAMS_DEFECTO["lightgbm"]` (31 hojas, min_child 100, λ = 5, lr 0,03) | 39 (con dui) |
| catboost | `PARAMS_DEFECTO["catboost"]` (depth 6, l2 = 3, lr 0,05, Bernoulli 0,8, rsm 0,7) | 39 |
| catboost_sin_dui | igual que catboost | 38 |
| lgb_sup_sin_dui | preset lightgbm con num_leaves = 8, min_child_samples = 200, reg_lambda = 10, colsample_bytree = 0,8 | 38 |
| ebm_sin_dui | `ExplainableBoostingClassifier(interactions=10)` (interpret 0.7.8; outer_bags = 14; lr 0,015; ES interno con 15 % aleatorio y paciencia 100) | 38 |

### 1.2 Esquema de evaluación (idéntico a `01_walkforward.py`)
Folds M ∈ {ago, sep, oct, nov}. Para los modelos boosting:
1. Se entrena con los meses < M−1 y se hace early stopping (ES) sobre M−1 (paciencia 100, máx. 2000, semilla 42). De ahí sale `best`.
2. Se toma $n_{iter}=\max(10,\operatorname{round}(best\cdot\frac{filas(<M)}{filas(<M-1)}))$.
3. Se reentrena con < M usando las semillas 42, 43 y 44, y se promedian sus predicciones.

El mes M no interviene en ninguna decisión. El EBM se ajusta directamente con < M usando su ES interno (ver la limitación en R1). El modelo final se entrena con todo train y $n_{iter}=\operatorname{mediana}_k(best_k\cdot filas(train)/filas(<M_k-1))$.

### 1.3 Pre-registro y desviaciones
- `scripts/criterios_robustez.py` (23:31:41) fija las hipótesis, estadísticos y reglas de R1–R9 antes de cualquier resultado. Dice explícitamente que los resultados previos de 01–08 ya eran conocidos.
- **ENMIENDA_1** (23:34, añadida al mismo archivo antes de que existiera ningún `robustez_*`, por indicación de la auditoría de código):
  - E1: predicciones sin redondear; métricas sobre float64 (npz) o en memoria.
  - E2: ningún peso adversarial; el escenario secundario de R1 es el f (dui permutada + 18,6 % de nuevos).
  - E3: Holm en toda familia de comparaciones.
  - E4: fundamento del ES.
  - E5: sesgo de selección de segundo orden.
  - E6: el efecto de dui cambia de signo por fold.
- **Ejecuciones interrumpidas** (no afectan a los resultados):
  1. La primera ejecución de `09_` se detuvo a mano antes de escribir nada, porque redondeaba las predicciones (E1).
  2. La segunda murió en silencio (~23:40) durante el EBM. Coincide con un `taskkill /F /IM python.exe` lanzado por otro agente. Se añadió un punto de control por modelo a `09_` y se relanzó la cadena completa con `scripts/ejecutar_robustez.sh` (códigos de salida en `resultados/logs/ejecutar_robustez.log`).

  Todos los logs `robustez` terminan en su línea "Listo". La segunda ejecución reproduce exactamente los Gini por fold de lgb_reg y catboost de `01_walkforward.py`.
- **Recortes respecto al encargo** (pre-registrados, por presupuesto):
  - R3 de CatBoost con 1 semilla y solo variando un factor cada vez.
  - R4 con 10 semillas y 20 bootstraps.
  - R6 con 5 sorteos de ruido, 3 permutaciones y 200 réplicas bootstrap.
  - R7 con 1 semilla en CatBoost/EBM.
  - R8 con B = 100.
  - R9 sin EBM ni catboost con dui, y con 2–3 réplicas.

### 1.4 Herramientas estadísticas comunes
- **Unidad de remuestreo = cliente.** Cada cliente aporta una fila por mes hasta que convierte, y sus variables (salvo dui) son constantes. Las filas del mismo cliente en meses distintos están, por tanto, fuertemente correlacionadas. El bootstrap de conglomerados remuestrea clientes con todas sus filas en *todos* los folds o celdas a la vez, lo que da inferencia conjunta válida para medias de Gini de varios meses. `GiniRapido` evalúa el AUC ponderado en O(n) por réplica, con los pesos iguales a la multiplicidad del cliente.
- **SE_REF = 0,0080:** SE bootstrap del Gini medio de 4 folds de lgb_reg. Es el "1 SE" de R3, R6 y R9. Es un umbral **absoluto**, y por eso es generoso para cambios *pareados* (mismo modelo y mismas filas), cuyo SE es 3–4 veces menor. Por eso, junto a cada criterio, se reporta el SE pareado.

## R1. Comparación formal de candidatos

**Hipótesis.** Para cada par $(a,b)$ de los 5 candidatos: $H_0:\ \bar G_a=\bar G_b$ frente a $H_1:\ \bar G_a\neq\bar G_b$, donde $\bar G$ es la media de los Gini de los 4 folds (ago–nov) con la predicción media de 3 semillas.

**Fundamento.**
- *AUC como estadístico U de Mann-Whitney.* Con $m$ positivos $x_i$ y $n$ negativos $y_j$, $\widehat{AUC}=\frac{1}{mn}\sum_i\sum_j \psi(x_i,y_j)$, con $\psi=1$ si $x>y$, $\tfrac12$ si hay empate y $0$ si no. Es un U-estadístico de dos muestras, insesgado para $P(X>Y)+\tfrac12P(X=Y)$. $Gini=2\,AUC-1$, así que $\Delta Gini=2\,\Delta AUC$ y los contrastes son equivalentes.
- *DeLong (1988).* Las componentes de colocación $V_{10}(x_i)=\frac1n\sum_j\psi(x_i,y_j)$ y $V_{01}(y_j)=\frac1m\sum_i\psi(x_i,y_j)$ permiten estimar la matriz de covarianzas de las AUC de dos modelos sobre las MISMAS filas: $S=\frac{S_{10}}{m}+\frac{S_{01}}{n}$, con $S_{10}$ y $S_{01}$ las covarianzas muestrales entre modelos de $V_{10}$ y $V_{01}$. Así $\operatorname{Var}(\widehat{AUC}_a-\widehat{AUC}_b)=S_{aa}+S_{bb}-2S_{ab}$ y $z=\Delta/\sqrt{\cdot}\sim N(0,1)$. El término $-2S_{ab}$ explica por qué el SE pareado (≈ 0,004–0,008 por fold) es mucho menor que el SE de cada Gini (≈ 0,015). DeLong supone observaciones independientes: vale *dentro* de un fold (cada cliente aparece una sola vez en un mes), pero **no** para combinar los folds, porque los clientes supervivientes aparecen en varios meses. Implementación rápida de Sun y Xu (2014) en `robustez_comun.delong`, comprobada frente a un bootstrap (varianza 9,3e-5 frente a 1,0e-4).
- *Bootstrap por conglomerados (cliente).* La unidad de muestreo independiente es el cliente, no la fila. Se remuestrean clientes con reemplazo y cada cliente entra con todas sus filas de los 4 folds a la vez. Los pesos de las filas son $w_r=$ multiplicidad del cliente. Así se conserva la correlación intra-cliente entre meses y el estimador $\bar\Delta^*$ refleja la dependencia real entre folds. Remuestrear filas trataría como independientes las filas repetidas del mismo cliente e infraestimaría el SE. Se usan $B=2000$ réplicas. El p-valor está centrado bajo $H_0$: $p=\frac{1+\#\{|\Delta^*-\hat\Delta|\ge|\hat\Delta|\}}{1+B}$, con mínimo $1/2001=0{,}0005$.
- *Holm (FWER).* Se ordenan los $m=10$ p-valores, $p_{(1)}\le\dots\le p_{(m)}$, y se define $p^{Holm}_{(i)}=\max_{j\le i}\min\{1,(m-j+1)p_{(j)}\}$. Controla la probabilidad de algún falso positivo en la familia ($\le 5\,\%$) bajo cualquier dependencia entre contrastes, y es uniformemente más potente que Bonferroni.
- *Regla pre-registrada:* $a$ es MEJOR si $p_{Holm}<0{,}05$, $\Delta>0$ y $\Delta_{fold}>0$ en al menos 3 de 4 folds.

**Resultados** (`robustez_r1_resumen.csv`, `robustez_r1_pares.csv`, `robustez_r1_delong_fold.csv`, `figuras/fig10_r1_comparacion_formal.png`). **CONFIRMADO.**

| Modelo | Gini medio [IC95] | ago | sep | oct | nov | sd entre semillas | n_iter por fold |
|---|---|---|---|---|---|---|---|
| lgb_reg | 0,2472 [0,2318; 0,2627] | 0,2545 | 0,2430 | 0,2650 | 0,2262 | 0,0008 | 29, 161, 33, 10 |
| catboost | 0,2509 [0,2353; 0,2659] | 0,2599 | 0,2471 | 0,2610 | 0,2355 | 0,0034 | 95, 242, 331, 171 |
| catboost_sin_dui | 0,2524 [0,2368; 0,2673] | 0,2683 | 0,2452 | 0,2603 | 0,2359 | 0,0011 | 249, 238, 144, 246 |
| **lgb_sup_sin_dui** | **0,2564** [0,2408; 0,2710] | 0,2718 | 0,2480 | 0,2661 | 0,2395 | 0,0012 | 201, 297, 121, 203 |
| ebm_sin_dui | 0,2524 [0,2372; 0,2671] | 0,2616 | 0,2555 | 0,2572 | 0,2354 | 0,0000 | ES interno |

El SE bootstrap del Gini medio de lgb_reg es **SE_REF = 0,0080**, que se usa como "1 SE" en R3, R6 y R9.

Δ frente a lgb_reg (candidato − lgb_reg), bootstrap conjunto con 2000 réplicas y Holm sobre los 10 pares:

| Candidato | Δ | SE | IC95 | p_boot | p_Holm | Folds mejor | Veredicto | Regla CLAUDE.md |
|---|---|---|---|---|---|---|---|---|
| catboost | +0,0037 | 0,0026 | [−0,0012; 0,0088] | 0,147 | 0,885 | 3/4 | sin diferencia | no (oct −0,0040) |
| catboost_sin_dui | +0,0052 | 0,0028 | [−0,0001; 0,0108] | 0,058 | 0,406 | 3/4 | sin diferencia | no (oct −0,0048) |
| **lgb_sup_sin_dui** | **+0,0092** | 0,0025 | **[+0,0044; +0,0141]** | 0,0005 | **0,005** | **4/4** | **MEJOR** | **CUMPLE** |
| ebm_sin_dui | +0,0053 | 0,0036 | [−0,0016; 0,0127] | 0,148 | 0,885 | 3/4 | sin diferencia | no (oct −0,0078) |

- Δ por fold de lgb_sup_sin_dui frente a lgb_reg: +0,0173 / +0,0051 / +0,0011 / +0,0133. DeLong por fold: z = 3,38 / 1,19 / 0,27 / 2,09, con p_Holm dentro de cada fold de 0,007 / 1 / 1 / 0,36. Solo agosto es significativo por sí solo, pero el signo es positivo en los 4 folds y el contraste conjunto (z_boot = 3,72) es claro.
- Entre los otros pares no hay diferencias significativas tras Holm. Los más cercanos son lgb_sup_sin_dui frente a catboost (+0,0055, p = 0,024, p_Holm = 0,21, 4/4 folds) y frente a catboost_sin_dui (+0,0040, p = 0,051, p_Holm = 0,40, 4/4 folds). Frente a ebm_sin_dui la diferencia es de +0,0039 (p_Holm = 0,88, 3/4 folds).
- **Escenario f** (dui permutada + 18,6 % de nuevos; sin pesos adversariales, enmienda E2): las conclusiones no cambian. lgb_sup_sin_dui queda +0,0095 [0,0046; 0,0142], con p_Holm = 0,005 y 4/4 folds. Los modelos con dui pierden poco: lgb_reg −0,0003 y catboost −0,0011.
- El 2.º puesto es un empate: ebm_sin_dui 0,25244 frente a catboost_sin_dui 0,25239 (Δ = 0,0001, p = 0,99).

**Interpretación.** **Pasa:** lgb_sup_sin_dui es el único candidato MEJOR que la referencia según la regla pre-registrada, y además cumple la regla de reemplazo de CLAUDE.md (+0,0092 > 0,003 sin empeorar ningún fold). Los demás candidatos no se distinguen de lgb_reg ni entre sí.

¿De dónde viene la ventaja? Hay tres piezas aproximadas:
- Con el mismo número de árboles fijo (curva de R2, n = 150), lgb_sup_sin_dui (0,2567) supera a lgb_reg (0,2504) en +0,0063. El resto, hasta +0,0092 (≈ +0,003), se debe a que el ES de lgb_reg es inestable (R2).
- Quitar dui por sí solo aportaba ≈ +0,0025 (lgb_reg_sin_dui en el informe previo).
- Queda ≈ +0,004 atribuible a la **menor capacidad por árbol** (8 hojas, ≥ 200 filas por hoja, λ = 10).

**PLAUSIBLE** (descomposición aproximada, no aditiva).

**Limitaciones de R1.**
1. *Sesgo de selección de segundo orden* (enmienda E5): el preset de lgb_reg y el modo `estatico` se eligieron con experimentos sobre sep–nov y jun–ago. Los folds ago–nov no son vírgenes para esas decisiones, y el sesgo favorece a la *referencia*. Los hiperparámetros de lgb_sup_sin_dui los fijó el orquestador al definir los candidatos, sin búsqueda sobre estos folds que yo conozca. Aun así, elegir 1 de 5 candidatos introduce un sesgo de "ganador" que Holm corrige en la inferencia, pero no en la magnitud: el Δ = +0,0092 es probablemente algo optimista (**PLAUSIBLE**). R3 comprueba si el preset es un pico aislado o una meseta.
2. El p_boot de lgb_sup_sin_dui está en el suelo de resolución (1/2001). El p real puede ser menor; z ≈ 3,7 sugiere p ≈ 2·10⁻⁴.
3. El ES interno del EBM usa una partición aleatoria por filas del train del fold. No toca el mes evaluado, pero mezcla filas del mismo cliente entre su entrenamiento y su validación interna. Su parada puede ser algo tardía (optimista dentro del train); no contamina la evaluación.

## R2. Número de árboles

**Hipótesis.** $H_0$: el Gini es insensible al número de árboles $n$ en un rango amplio, es decir, hay una meseta ancha. $H_1$: hay un óptimo estrecho, y un $n$ mal elegido por el early stopping cuesta Gini.

**Fundamento.**
- En boosting, $F_n=F_{n-1}+\eta f_n$. Cada árbol se ajusta al gradiente $g=p-y$ y al hessiano $h=p(1-p)$ de la log-loss, y el peso óptimo de cada hoja es $w^*=-G/(H+\lambda)$. Al crecer $n$ baja el sesgo y sube la varianza. La velocidad a la que se sobreajusta depende de la capacidad de cada árbol: hojas, filas mínimas por hoja (`min_child_samples` acota $H$ por debajo) y $\lambda$, que encoge $w^*$. Con señal débil (AUC univariado máx. ≈ 0,55), un árbol de 31 hojas empieza pronto a ajustar ruido.
- *Por qué el ES es inestable* (enmienda E4): el ES de LightGBM ya para por AUC. El problema es que el AUC de validación es **plano y ruidoso**: entre el mejor $n$ y $n=200$ hay solo 0,002–0,003, mientras que el SE del Gini de un mes es ≈ 0,015 (en AUC, ≈ 0,0075). El argmax de una curva plana más ruido tiene una varianza enorme: cualquier punto de la meseta puede ganar. De ahí salen los 8–161 árboles de lgb_reg.
- *Regla 1-SE* (Breiman et al., 1984): dentro de un SE del óptimo no se distinguen los modelos, así que se elige el más simple (el menor $n$). La meseta se define como $\{n:\bar G(n)\ge\bar G(n^*)-SE(n^*)\}$. Además se usa la variante **pareada** (SE de $\bar G(n)-\bar G(n^*)$, mucho menor), que es la que de verdad discrimina.
- *Aviso de sesgo de selección:* $n^*$ y $n_{1SE}$ se eligen con los mismos folds en los que se mide el Gini, así que $\bar G(n^*)$ está sesgado al alza. Solo sirve para comparar formas de curva, no como estimación del rendimiento.

**Resultados** (`robustez_r2_curva.csv`, `robustez_r2_resumen.csv`, `figuras/fig11_r2_num_arboles.png`; 3 semillas; entrenamiento con < M hasta 1000 árboles). **CONFIRMADO.**

| Modelo | n* | G(n*) | Meseta 1-SE (ratio) | Meseta pareada (ratio) | G(1000) − G(n*) | n_iter honesto (R1) | G honesto (R1) |
|---|---|---|---|---|---|---|---|
| lgb_reg | 150 | 0,2504 | 5–500 (×100) | 30–150 (×5) | −0,0180 | 29, 161, 33, 10 | 0,2472 |
| catboost | 200 | 0,2529 | 70–500 (×7) | 150–300 (×2) | −0,0173 | 95–331 | 0,2509 |
| catboost_sin_dui | 200 | 0,2535 | 70–400 (×5,7) | 150–200 (×1,3) | −0,0225 | 144–249 | 0,2524 |
| lgb_sup_sin_dui | 300–400 | 0,2569 | 70–1000 (×14) | 150–400 (×2,7) | **−0,0051** | 121–297 | 0,2564 |

- **Los 4 pasan el criterio pre-registrado** (meseta 1-SE de al menos ×3). Pero ese criterio es poco exigente: con SE_REF ≈ 0,008, toda la curva de lgb_reg entre 5 y 500 árboles cae "dentro de 1 SE".
- Con el SE pareado, la diferencia de forma es clara. lgb_sup_sin_dui tiene la curva más plana: pierde solo 0,005 con 1000 árboles, frente a 0,017–0,023 de los demás. Su Gini honesto (0,2564) está a 0,0005 del óptimo a posteriori. lgb_reg pierde 0,0032 por la inestabilidad del ES: en noviembre el ES eligió 10 árboles, y G(10) = 0,2451.
- Con el mismo $n$ (n = 150) para todos, el orden se mantiene: lgb_sup_sin_dui 0,2567 > catboost_sin_dui 0,2529 ≈ catboost 0,2525 > lgb_reg 0,2504.

**Interpretación.** **Pasa** en los 4 modelos según el criterio pre-registrado. En la versión estricta (pareada), solo lgb_reg (×5) supera ×3, y lo hace porque su curva es plana *y baja*; lgb_sup_sin_dui se queda cerca (×2,7). El hallazgo práctico es que **lgb_sup_sin_dui es el menos sensible a sobreentrenar**: un error de ×2 en el número de árboles le cuesta < 0,001. En cambio, lgb_reg y CatBoost se degradan rápido por encima de 300 árboles. Esto hace robusto el factor de escalado del modelo final (mediana de best_iter × filas). **CONFIRMADO.**

## R3. Hiperparámetros

**Hipótesis.** $H_0$: el Gini medio es estable en una vecindad del preset, con rango ≤ 1 SE. $H_1$: el preset es un pico estrecho, posiblemente fruto del azar o de haberlo seleccionado.

**Fundamento.** Si el preset es bueno "por suerte", al moverse un factor ×2 en cualquier dirección el Gini debería caer más que el ruido. Una superficie plana alrededor del preset indica que el rendimiento procede del *régimen* de capacidad y no de un valor concreto, lo que limita la "maldición del ganador" (el sesgo de elegir el máximo de varias estimaciones ruidosas, $E[\max_k \hat G_k]\ge\max_k G_k$). Los parámetros controlan la capacidad efectiva de cada árbol:
- `num_leaves` y `depth`: orden máximo de interacción y número de regiones.
- `min_child_samples`: cota inferior de $H$ por hoja.
- $\lambda$: encoge $w^*=-G/(H+\lambda)$ hacia 0, con más fuerza en hojas pequeñas.
- `learning_rate` $\eta$: con $\eta$ pequeño se necesitan más árboles; el producto $\eta\cdot n$ es aproximadamente constante.

El rango observado incluye el ruido de semilla (1 semilla en CatBoost), así que el criterio es conservador.

**Resultados** (`robustez_r3_hiperparametros.csv`, `robustez_r3_resumen.csv`, `figuras/fig12_r3_hiperparametros.png`; walk-forward honesto completo). **CONFIRMADO.**

*lgb_sup_sin_dui* (3 semillas; centro 0,2564):

| Configuración (un factor cada vez) | Gini | Δ vs centro (SE pareado) |
|---|---|---|
| num_leaves 4 / 16 | 0,2539 / 0,2551 | −0,0025 (0,0015) / −0,0012 (0,0013) |
| min_child 100 / 400 | 0,2560 / 0,2564 | −0,0004 / 0,0000 |
| λ 3 / 30 | 0,2552 / 0,2567 | −0,0012 / +0,0003 |
| lr 0,015 / 0,06 | 0,2555 / 0,2568 | −0,0009 / +0,0004 |

- Rango en la vecindad (un factor cada vez) = **0,0029 ≤ SE_REF (0,0080) → pasa**. Ninguna variación mejora al centro en más de 0,0004.
- En la vecindad extendida (16 esquinas ×½/×2, con todos los factores cambiados a la vez) el rango es de 0,0074 (0,2494–0,2568). Las peores combinan 16 hojas con lr 0,015: el ES elige 10 árboles en octubre, otra vez la inestabilidad de R2.
- **Las 25 configuraciones (mínimo 0,2494) superan a lgb_reg (0,2472).** La ventaja de R1 no depende del valor exacto del preset, sino del régimen "LightGBM poco profundo, sin dui".

*catboost_sin_dui* (1 semilla; centro 0,2527):
- depth 4 / 8: 0,2541 / 0,2477.
- l2 1 / 10: 0,2514 / 0,2540.
- lr 0,025 / 0,1: 0,2535 / 0,2459.

Rango = **0,0082 > 0,0080 → falla por muy poco**. Las dos variaciones que más pierden son las de *más* capacidad (depth 8: −0,0050, SE 0,0027; lr 0,1: −0,0068, SE 0,0028). Las de menos capacidad mejoran ligeramente (depth 4: +0,0014; l2 10: +0,0014).

**Interpretación.**
- **lgb_sup_sin_dui pasa** con holgura: está en una meseta, no en un pico.
- **catboost_sin_dui falla** (en el límite y con 1 semilla). La asimetría que se observa (más capacidad empeora, menos capacidad mejora un poco) es la firma de un problema de **señal débil**: la complejidad óptima es baja y los presets por defecto (31 hojas, depth 6) están en el lado sobreajustado. **PLAUSIBLE** (coherente con R2 y R9).

## R4. Varianza por semilla y por muestra (modelo final, diciembre)

**Hipótesis.** $H_0$: el ranking de diciembre del modelo final es estable, es decir, al menos el 85 % del top-10 % se mantiene en ≥ 90 % de las réplicas. Se contrasta por separado ante (a) la semilla y (b) la muestra de entrenamiento.

**Fundamento.**
- (a) *Semilla.* Mide la varianza algorítmica: submuestreo de filas y columnas y bolsas externas del EBM. Promediar $k$ semillas divide esta varianza por $k$; la submission usa la media de 10 semillas.
- (b) *Bootstrap de clientes.* Mide la varianza de estimación $\operatorname{Var}_D[\hat f_D(x)]$, es decir, cuánto cambiaría el score con otra muestra de train de la misma población. Esta varianza no se reduce promediando semillas.
- Para un cliente con score medio $s_i$, desviación $\sigma_i$ y umbral $t$, $P(i\in top)\approx\Phi((s_i-t)/\sigma_i)$. Los clientes cercanos al umbral son inestables. Con señal débil (Gini ≈ 0,25), los scores están concentrados y hay muchos clientes cerca del umbral.
- Medidas: Jaccard $|A\cap B|/|A\cup B|$ entre pares de réplicas, sd del percentil y Spearman.
- **Desviación (v2):** el bootstrap usa pesos de multiplicidad (`sample_weight`) en lugar de duplicar filas. La v1 no terminaba con el EBM: las copias de un cliente caían a la vez en su train y en su validación interna, así que el ES no paraba. Sus salidas `robustez_r4_*` quedan obsoletas. El proceso v1 siguió consumiendo CPU sin que se pudiera detener (permiso denegado), lo que ralentizó R6–R9.

**Resultados** (`robustez_r4v2_resumen.csv`, `robustez_r4v2_pred_dic_<modelo>.csv`, `figuras/fig13_r4v2_estabilidad_top10.png`). Modelos: los 2 primeros de R1 (lgb_sup_sin_dui y ebm_sin_dui, este último 2.º por 0,00005 frente a catboost_sin_dui) más lgb_reg. **CONFIRMADO.**

| Modelo | Esquema | % top-10 % estable | Jaccard medio (mín.) | sd percentil media / p95 | Spearman | Pasa |
|---|---|---|---|---|---|---|
| lgb_sup_sin_dui | 10 semillas | **95,5 %** | 0,945 (0,934) | 0,025 / 0,058 | 0,988 | sí |
| lgb_sup_sin_dui | 20 bootstraps | 82,5 % | 0,842 (0,782) | 0,065 / 0,130 | 0,929 | **no** |
| ebm_sin_dui | 10 semillas | **98,2 %** | 0,982 (0,974) | 0,007 / 0,015 | 0,999 | sí |
| ebm_sin_dui | 20 bootstraps | 83,5 % | 0,842 (0,765) | 0,065 / 0,131 | 0,930 | **no** |
| lgb_reg | 10 semillas | 87,5 % | 0,877 (0,832) | 0,049 / 0,115 | 0,955 | sí |
| lgb_reg | 20 bootstraps | 76,3 % | 0,808 (0,749) | 0,077 / 0,147 | 0,902 | **no** |

**Interpretación.** **Ninguno pasa R4** según el criterio conjunto: todos son estables ante la semilla e inestables ante la muestra.
- El fallo es del *problema*: con Gini ≈ 0,25, la varianza de estimación mueve a ~1 de cada 6 clientes del borde del top-10 %.
- El orden relativo es claro: lgb_sup_sin_dui y ebm_sin_dui (82–84 %) son más estables que lgb_reg (76 %, y solo 87,5 % ante la semilla). Encaja con R2: los 43 árboles finales de 31 hojas de lgb_reg salen de un $n$ ruidoso.
- La submission promedia 10 semillas, así que su varianza por semilla es ≈ 1/10 de la de una réplica. **PLAUSIBLE.**
- El umbral del 85 % ante el bootstrap era exigente para señal débil; se mantiene el veredicto pre-registrado.

## R5. Segmentos (solo cálculo; la interpretación de negocio la hace `analista-datos`)

**Hipótesis.** $H_0$: el orden de los candidatos y el Gini del candidato recomendado son homogéneos entre segmentos. Contraste por segmento: $\Delta_s=\bar G_{s,c}-\bar G_{s,\text{lgb\_reg}}=0$, con Holm sobre 4 candidatos × 10 segmentos = **40** contrastes (la enmienda E3 decía 36 por un error de cuenta; con 40 la corrección es más conservadora).

**Fundamento.**
- El Gini *dentro* de un segmento solo compara pares positivo–negativo del mismo segmento: $AUC_s=P(X>Y\mid s_X=s_Y=s)$. El AUC global es una mezcla ponderada de los AUC intra-segmento y de los pares *entre* segmentos. Si la variable que define el segmento es predictiva por sí misma, el Gini intra-segmento es menor que el global. Esto es un efecto aritmético, no un fallo del modelo, y explica que la banda *medium* tenga un Gini ≈ 0,08.
- Con pocos positivos por segmento (808 en *high*), el SE crece aproximadamente como $1/\sqrt{\min(m,n)}$. Por eso las diferencias por segmento tienen IC de ±0,01–0,02.
- El orden se compara con la $\tau$ de Kendall entre el vector global de los 5 Gini y el del segmento.
- Estadístico: media de los 4 Gini por fold dentro del segmento. Bootstrap de clientes conjunto con 1000 réplicas.

**Resultados** (`robustez_r5_segmentos.csv`, `robustez_r5_orden.csv`, `figuras/fig14_r5_segmentos.png`). **CONFIRMADO.**

| Segmento (filas; positivos) | lgb_reg | catboost | catboost_sin_dui | **lgb_sup_sin_dui** [IC95] | ebm_sin_dui | τ |
|---|---|---|---|---|---|---|
| nuevo (5680; 1011) | 0,2416 | 0,2453 | 0,2455 | **0,2481** [0,207; 0,292] | 0,2371 | 0,4 |
| antiguo (34170; 4898) | 0,2418 | 0,2455 | 0,2474 | **0,2520** [0,235; 0,269] | 0,2487 | 1,0 |
| banda low (17626; 3216) | 0,2354 | 0,2390 | 0,2385 | **0,2408** [0,219; 0,264] | 0,2392 | 0,8 |
| banda medium (12861; 1885) | 0,0697 | 0,0743 | 0,0754 | **0,0845** [0,057; 0,112] | 0,0758 | 1,0 |
| banda high (9363; 808) | 0,3120 | 0,3143 | **0,3246** | 0,3179 [0,282; 0,356] | 0,2929 | 0,2 |
| productos 1 (18791; 2418) | 0,1403 | 0,1407 | 0,1417 | **0,1462** [0,121; 0,170] | 0,1431 | 1,0 |
| productos 2 (11761; 1730) | 0,2470 | 0,2541 | 0,2569 | **0,2613** [0,234; 0,287] | 0,2558 | 0,8 |
| productos 3+ (9298; 1761) | 0,3340 | **0,3410** | 0,3392 | 0,3409 [0,312; 0,370] | 0,3402 | 0,4 |
| activo_movil True (25833; 4028) | 0,2612 | 0,2638 | 0,2642 | **0,2669** [0,249; 0,285] | 0,2659 | 1,0 |
| activo_movil False (14017; 1881) | 0,2113 | 0,2193 | 0,2217 | **0,2284** [0,202; 0,255] | 0,2194 | 0,8 |

- **Peor segmento** para los 5 candidatos: banda_riesgo = medium (lgb_sup_sin_dui 0,0845; lgb_reg 0,0697).
- **lgb_sup_sin_dui es el mejor en 8 de 10 segmentos.** En los otros dos (high y 3+) la diferencia con el mejor del segmento no es significativa: −0,0067 [−0,016; +0,003] y −0,0001 [−0,006; +0,006].
- Δ frente a lgb_reg: es positivo en los 10 segmentos para lgb_sup_sin_dui. Es significativo tras Holm en *antiguo* (+0,0103, p_Holm = 0,04) y en *activo_movil = False* (+0,0170, p_Holm = 0,04). Ningún otro candidato es significativo en ningún segmento.
- ebm_sin_dui es el más irregular: es el peor en *nuevo* (−0,0045 frente a lgb_reg) y en *high* (−0,0191 [−0,040; +0,003]).
- $\tau$ mínima = 0,2 (banda high), donde el orden se reorganiza dentro del ruido.

**Interpretación (estadística).** El orden global se mantiene en la mayoría de segmentos (τ ≥ 0,8 en 7 de 10). En ningún segmento el candidato recomendado es significativamente peor que otro. Que la ganancia sea homogénea en signo (10/10 segmentos) indica que no procede de un subgrupo concreto. No hay criterio de aprobado pre-registrado; resultado **favorable** para lgb_sup_sin_dui.

## R6. Perturbación de entradas e importancia por permutación en validación

**Hipótesis.**
- (a) $H_0$: un error de medida multiplicativo del 10 % al predecir no baja el Gini más de 1 SE.
- (b) $H_0$: ninguna variable concentra más del 50 % de la importancia.

**Fundamento.**
- *Ruido log-normal:* $x'=x\,e^{sZ-s^2/2}$, con $Z\sim N(0,1)$. Cumple $E[x']=x$, tiene coeficiente de variación ≈ $s$, conserva el signo y es simétrico en escala logarítmica (error de medida proporcional). Las variables enteras se redondean y los valores se recortan al rango de train. Se aplica solo al predecir.
- *Atenuación:* si $x'=x+u$, en el caso lineal la señal se reduce por el factor $\lambda=\sigma_x^2/(\sigma_x^2+\sigma_u^2)$, así que la pérdida es ∝ $\sigma_u^2$. En árboles, las observaciones cercanas a cada umbral cruzan al lado equivocado con probabilidad creciente en $\sigma_u/$distancia al corte.
- *Permutación en validación:* $I_j=G-E_\pi[G(x_j\text{ permutada})]$, medida en el mes no visto, es decir, la contribución *generalizable*. La importancia por impureza o ganancia es *in-sample* y está sesgada hacia variables continuas o con muchos cortes. SHAP reparte $f(x)$ con valores de Shapley, pero describe el modelo, no su generalización. Las dummies OHE de una categórica se permutan juntas, y las variables correlacionadas se reparten la importancia. IC por bootstrap de clientes (200).
- **Desviación (E3):** con B = 200, el p bootstrap mínimo es 1/201, y Holm sobre 22–23 variables no baja de 0,11. Se añadió `15b_r6_holm_normal.py`, con p unilateral normal $1-\Phi(I/SE)$ y Holm por modelo.

**Resultados** (`robustez_r6_ruido.csv`, `robustez_r6_importancia.csv`, `robustez_r6_importancia_holm_normal.csv`, `robustez_r6_resumen.csv`, `figuras/fig15_r6_perturbacion.png`; modelos de cada fold con semilla 42). **CONFIRMADO.**

| Modelo | Caída 5 % | Caída 10 % [IC95] | Caída 20 % | Variable top (cuota) | Pasa |
|---|---|---|---|---|---|
| lgb_reg | −0,0004 | 0,0028 [0,0000; 0,0056] | 0,0080 | banda_riesgo (52,4 %) | **no** (concentración) |
| catboost | 0,0002 | 0,0029 [0,0007; 0,0052] | 0,0080 | banda_riesgo (51,4 %) | **no** (concentración) |
| catboost_sin_dui | −0,0009 | 0,0008 [−0,0013; 0,0026] | 0,0061 | banda_riesgo (49,2 %) | sí |
| lgb_sup_sin_dui | 0,0011 | 0,0027 [0,0006; 0,0047] | 0,0095 | banda_riesgo (47,3 %) | sí |
| ebm_sin_dui | 0,0000 | 0,0023 [0,0007; 0,0037] | 0,0078 | banda_riesgo (49,0 %) | sí |

- **Ruido:** con 10 %, todas las caídas (≤ 0,003) quedan muy por debajo de SE_REF (0,008). Con 20 % llegan a ≈ 1 SE. La caída crece aproximadamente como $s^2$, como predice la atenuación. **PLAUSIBLE.**
- **Importancia en lgb_sup_sin_dui:**
  - banda_riesgo 0,150 [0,136; 0,163]
  - numero_productos 0,071 [0,063; 0,081]
  - dias_ultima_transaccion 0,050 [0,041; 0,061]
  - activo_movil 0,018
  - tiene_tarjeta_credito 0,010
  - ratio_deuda_ingresos 0,007

  El resto es < 0,004. Con Holm normal hay 6 variables significativas en lgb_reg y en ambos CatBoost, 7 en lgb_sup_sin_dui (+ canal_adquisicion) y 5 en el EBM. El orden de las 6 primeras es idéntico en los 5 modelos.
- **dui en validación:** lgb_reg −0,0013 [−0,0033; +0,0013]; catboost +0,0017 [−0,0004; +0,0041]. No aporta Gini generalizable, coherente con que su efecto cambie de signo por fold (E6).

**Interpretación.**
- (a) **Pasan los 5.**
- (b) La regla del 50 % separa a los modelos por márgenes dentro del ruido (47–52 %). Lo sustantivo es común a todos: **banda_riesgo aporta cerca de la mitad de la señal**, así que cualquier cambio de definición de esa variable en diciembre afectaría a cualquier modelo.

## R7. Degradación temporal

**Hipótesis.** $H_0:\ \beta=0$, es decir, el Gini no cae con el desfase $h$ entre el último mes de entrenamiento y el mes evaluado. $H_1:\ \beta\neq0$.

**Fundamento.**
- Se usan 8 orígenes $k$ (mar–oct). Cada modelo se entrena con ≤ k, con ES honesto en $k$, y se evalúa en $k+h$ para $h=1..4$ (26 celdas).
- Modelo: $G_{kh}=\alpha_k+\beta h+\varepsilon_{kh}$. Los efectos fijos de origen $\alpha_k$ absorben el tamaño y la calidad de cada modelo, así que $\beta$ es el "envejecimiento" del *mismo* modelo.
- Problema de identificación edad–periodo–cohorte: como $k+h$ = mes evaluado, no se pueden incluir a la vez efectos fijos de origen y de mes evaluado junto con $h$. Con efectos de origen, $\beta$ se confunde con la dificultad propia de cada mes. Por ejemplo, noviembre es difícil para todos los modelos (R1), y solo se alcanza con $h$ grande desde orígenes antiguos.
- Como control se estima la variante secundaria con efectos fijos de mes evaluado: compara modelos de distinta antigüedad sobre el mismo mes, y se confunde con el tamaño de train, que R9 muestra que importa poco.
- Inferencia: bootstrap de clientes conjunto sobre las 26 celdas (1000 réplicas). Un cliente aparece en varias celdas, así que el remuestreo debe ser conjunto. Holm sobre los 5 modelos (E3).

**Resultados** (`robustez_r7_celdas.csv`, `robustez_r7_resumen.csv`, `figuras/fig16_r7_degradacion.png`). **CONFIRMADO.**

| Modelo | β (EF origen) [IC95] | p | p_Holm | β (EF mes evaluado) [IC95] | G medio h = 1 → h = 4 | Criterio pre-registrado |
|---|---|---|---|---|---|---|
| lgb_reg | −0,0044 [−0,0093; +0,0006] | 0,086 | 0,34 | −0,0005 [−0,0020; +0,0010] | 0,2556 → 0,2501 | pasa |
| catboost | −0,0055 [−0,0106; −0,0003] | 0,038 | 0,19 | −0,0031 [−0,0048; −0,0014] | 0,2560 → 0,2451 | **falla** (IC sin 0) |
| catboost_sin_dui | −0,0043 [−0,0092; +0,0009] | 0,089 | 0,34 | −0,0022 [−0,0038; −0,0005] | 0,2557 → 0,2476 | pasa |
| lgb_sup_sin_dui | −0,0043 [−0,0090; +0,0010] | 0,096 | 0,34 | −0,0012 [−0,0023; −0,0000] | 0,2621 → 0,2565 | pasa |
| ebm_sin_dui | −0,0043 [−0,0094; +0,0004] | 0,093 | 0,34 | −0,0019 [−0,0033; −0,0005] | 0,2532 → 0,2452 | pasa |

- Con efectos de origen, la pendiente es casi idéntica en todos (≈ −0,0044 por mes). Esto indica que la domina un factor común, la dificultad del mes evaluado, y no la degradación propia de cada modelo. **PLAUSIBLE.**
- Con efectos de mes evaluado, las pendientes son menores: −0,0005 (lgb_reg) a −0,0031 (catboost) por mes. Sus IC excluyen 0 en catboost, catboost_sin_dui, ebm_sin_dui y, por muy poco, lgb_sup_sin_dui.
- Nota técnica: la versión del script que generó el CSV marcaba `robusto` solo con `ic95_inf ≤ 0`. Ya está corregido en `16_r7_degradacion.py`, y la tabla del 19 aplica el criterio correcto.

**Interpretación.**
- Por el criterio pre-registrado (IC sin ajustar), **catboost falla** y los demás pasan. Con Holm (E3), **ningún** modelo muestra una degradación significativa.
- Hay un envejecimiento leve y plausible de ≈ 0,001–0,003 de Gini por mes. Para diciembre es irrelevante, porque el modelo final se entrena hasta noviembre ($h=1$), pero aconseja reentrenar cada mes.
- lgb_sup_sin_dui mantiene el Gini más alto en todo horizonte (0,2565 con $h=4$, por encima de lgb_reg con $h=1$).
- Los modelos con dui no se degradan más que sus versiones sin dui en este rango. Aun así, la fracción de filas con dui = dut cae de forma lineal del 100 % en enero al 0 % en diciembre (`drift_dui_por_mes.csv`), así que la relación que aprende el modelo con dui deja de existir exactamente en diciembre.

## R8. Placebo (prueba de permutación)

**Hipótesis.** $H_0$: el modelo no aprende señal, es decir, su Gini en validación es compatible con haber entrenado con etiquetas sin relación con $x$.

**Fundamento.**
- Al permutar `objetivo` dentro de cada mes de train se rompe cualquier relación $x\to y$, pero se conservan la tasa de positivos por mes, la estructura de panel y todo el pipeline: mismas variables, mismo $n_{iter}$ por fold (misma capacidad) y semilla 42.
- Se evalúa contra las etiquetas *reales* de M. Bajo $H_0$, el Gini real es intercambiable con los placebo, así que el p-valor de permutación exacto (Monte Carlo) es $p=\frac{1+\#\{G^{(b)}\ge G_{real}\}}{1+B}$. Es válido en muestra finita porque cuenta el propio dato observado entre las $B+1$ realizaciones; su mínimo con B = 100 es 1/101 ≈ 0,0099.
- Además se calcula $z=(G_{real}-\bar G_{plac})/s_{plac}$, que mide la distancia más allá de la resolución del p.
- La prueba detecta fugas de información: si el placebo diera Gini > 0 de forma sistemática, el pipeline estaría usando información de $y$ por otra vía (por ejemplo, codificaciones de objetivo o el orden temporal). También calibra el ruido.

**Resultados** (`robustez_r8_placebo.csv`, `robustez_r8_resumen.csv`, `figuras/fig17_r8_placebo.png`; lgb_sup_sin_dui, B = 100, pre-registrado). **CONFIRMADO.**
- Placebo: media **+0,0025** y sd 0,0223 (rango −0,050 a +0,044). Está centrado en 0: $|0{,}0025|<2\cdot0{,}0223/\sqrt{100}=0{,}0045$.
- Real (semilla 42): **0,2567**, por encima del máximo placebo (0,0443). $z=11{,}4$ y $p=1/101=0{,}0099$ (el mínimo posible).
- La sd por fold de los placebos (0,025–0,028) supera la teórica de un puntuador aleatorio, $\sqrt{(m+n+1)/(12mn)}\cdot2\approx0{,}016$. Además, la sd del Gini medio de 4 folds (0,022) apenas baja respecto de la de un fold. Las dos cosas encajan con que un placebo es una función *estructurada* de $x$, fija entre folds y evaluada sobre clientes que se repiten. **PLAUSIBLE.**

**Interpretación.** **Pasa:** no hay fuga y la señal es real. Ojo, que la señal sea real no quiere decir que sea fuerte: el Gini de 0,26 equivale a unas 11 sd del placebo, pero solo a unas 30 SE_REF del 0.

## R9. Curva de aprendizaje (submuestreo por cliente)

**Hipótesis.** $H_0$: el modelo está saturado, es decir, pasar del 75 % al 100 % de los clientes de train no mejora el Gini en más de 1 SE. Contraste adicional (E3): ganancia = 0, con Holm sobre 3 modelos.

**Fundamento.**
- Se submuestrean *clientes*, no filas, para no partir la historia de un cliente y para que $n$ cuente unidades independientes. Los subconjuntos son anidados ($U_i<f$) para que las diferencias entre fracciones sean pareadas.
- En problemas con riesgo de estimación decreciente, el exceso de error suele seguir una ley potencia: $G(n)\approx G_\infty-a\,n^{-b}$, con $b\approx0{,}5$ en regímenes paramétricos y $b<0{,}5$ en no paramétricos. $G_\infty$ es el techo alcanzable con más datos de la misma distribución, limitado por la señal (el error de Bayes).
- Con 4 puntos y 3 parámetros, el ajuste es casi exacto y $G_\infty$ queda mal identificado, así que se informa como indicativo.
- Una curva plana entre 75 % y 100 % significa que la varianza ya no es el cuello de botella: el límite es el sesgo, o la información de $x$ sobre $y$.

**Resultados** (`robustez_r9_curva.csv`, `robustez_r9_resumen.csv`, `figuras/fig18_r9_curva_aprendizaje.png`; semilla 42; 3/2 réplicas). **CONFIRMADO.**

| Modelo | 25 % | 50 % | 75 % | 100 % | Ganancia 75→100 [IC95] | p_Holm | Ganancia 25→100 | Ley potencia: $G_\infty$ (SE) |
|---|---|---|---|---|---|---|---|---|
| lgb_reg | 0,2343 | 0,2444 | 0,2465 | 0,2465 | −0,0000 [−0,0054; +0,0043] | 1,00 | +0,0122 | 0,248 (0,001), *a* en el límite: degenerada |
| lgb_sup_sin_dui | 0,2445 | 0,2533 | 0,2528 | 0,2567 | +0,0038 [+0,0008; +0,0067] | 0,030 | +0,0121 | 0,258 (0,008) |
| catboost_sin_dui | 0,2345 | 0,2456 | 0,2468 | 0,2527 | +0,0059 [+0,0017; +0,0103] | 0,030 | +0,0182 | 0,275 (0,088), no identificada |

- **Los 3 cumplen el criterio pre-registrado de "saturado"** (ganancia < SE_REF = 0,008).
- Con el contraste pareado, lgb_reg está plano. lgb_sup_sin_dui y catboost_sin_dui todavía ganan algo (+0,004 y +0,006, p_Holm = 0,03).
- La ley potencia predice que duplicar los datos daría +0,0003 (lgb_sup_sin_dui) y +0,0046 (catboost_sin_dui), con mucha incertidumbre. **PLAUSIBLE.**
- Hay ruido de réplica apreciable. En lgb_sup_sin_dui al 75 %, las réplicas van de 0,2442 a 0,2589, lo que muestra la varianza de la muestra ya vista en R4.

**Interpretación.** **Pasan los 3**: el problema está dominado por la señal y no por el tamaño de la muestra. Con 25 % de los clientes, lgb_sup_sin_dui (0,2445) ya iguala a lgb_reg con el 100 %. Que el modelo de 8 hojas sea más eficiente con pocos datos es lo esperable de un estimador de menor varianza. CatBoost es el que más podría beneficiarse de más datos (pendiente mayor), pero sin datos adicionales eso no sirve para diciembre.

## 2. Tabla comparativa candidatos × pruebas (`resultados/robustez_tabla_comparativa.csv`)

"—" = prueba no aplicada a ese candidato (pre-registrado). **CONFIRMADO** (todas las cifras calculadas).

| Prueba | lgb_reg (ref.) | catboost | catboost_sin_dui | **lgb_sup_sin_dui** | ebm_sin_dui |
|---|---|---|---|---|---|
| R1 Gini medio [IC95] | 0,2472 [0,232; 0,263] | 0,2509 [0,235; 0,266] | 0,2524 [0,237; 0,267] | **0,2564** [0,241; 0,271] | 0,2524 [0,237; 0,267] |
| R1 Δ vs ref. (p_Holm; folds) | — | +0,0037 (0,88; 3/4) | +0,0052 (0,41; 3/4) | **+0,0092 (0,005; 4/4) MEJOR** | +0,0053 (0,88; 3/4) |
| R1 escenario f (dic.) | 0,2469 | 0,2498 | 0,2527 | **0,2563** | 0,2511 |
| Regla CLAUDE.md | — | no | no | **cumple** | no |
| R2 n.º de árboles (meseta 1-SE; pareada) | pasa ×100; ×5 | pasa ×7; ×2 | pasa ×5,7; ×1,3 | **pasa ×14; ×2,7** (−0,005 con 1000 árboles) | — |
| R3 hiperparámetros (rango un factor cada vez) | — | — | **falla** 0,0082 | **pasa** 0,0029 | — |
| R4 top-10 % estable (semilla / bootstrap) | falla 87 % / 76 % | — | — | falla 95 % / 83 % | falla 98 % / 84 % |
| R5 peor segmento (Gini); n.º de segmentos en que es el mejor | medium (0,070); 0 | medium (0,074); 1 | medium (0,075); 1 | medium (0,085); **8** | medium (0,076); 0 |
| R6 caída con 10 % de ruido; cuota de la variable top | falla 0,0028; 52 % | falla 0,0029; 51 % | pasa 0,0008; 49 % | **pasa** 0,0027; 47 % | pasa 0,0023; 49 % |
| R7 β por mes [IC95] (p_Holm) | pasa −0,0044 [−0,009; +0,001] (0,34) | **falla** −0,0055 [−0,011; −0,000] (0,19) | pasa −0,0043 (0,34) | **pasa** −0,0043 (0,34) | pasa −0,0043 (0,34) |
| R8 placebo | — | — | — | **pasa** z = 11,4; p = 1/101 | — |
| R9 ganancia 75→100 % (saturado si < 0,008) | saturado −0,000 | — | saturado +0,006 | saturado +0,004 | — |
| **Pruebas suspendidas / aplicadas** | 2/4 (R4, R6) | 2/3 (R6, R7) | 1/4 (R3) | **1/6 (R4)** | 1/3 (R4) |
| Complejidad (1 = más simple) | 3 | 5 | 4 | 2 | 1 |

Lectura: lgb_sup_sin_dui es el candidato más probado (6 pruebas con criterio) y solo suspende la que suspenden todos. Los suspensos de lgb_reg y catboost en R6 son marginales (51–52 % frente al límite del 50 %), y el de catboost en R7 desaparece con Holm.

## 3. Recomendación

**Modelo recomendado: `lgb_sup_sin_dui`**: LightGBM con num_leaves = 8, min_child_samples = 200, reg_lambda = 10, colsample_bytree = 0,8, lr = 0,03 y subsample 0,7; sin `dias_ultima_interaccion`. El modelo final usa 268 árboles (mediana de best_iter × filas) y la media de 10 semillas.

**Por qué.**
1. **R1, prueba formal:** es el único candidato MEJOR que lgb_reg según la regla pre-registrada. Δ = +0,0092 [+0,0044; +0,0141], p_Holm = 0,005 sobre 10 pares y 4/4 folds a favor. Se mantiene en el escenario f de diciembre (+0,0095, p_Holm = 0,005, 4/4). **CONFIRMADO.** Al haber un ganador, no hizo falta aplicar la regla de parsimonia, aunque esta también lo habría elegido: es el que menos pruebas suspende (solo R4, que suspenden todos), no usa dui y es el 2.º en complejidad.
2. **No es un pico afortunado:** su vecindad de hiperparámetros es plana (R3: rango 0,0029) y las 25 configuraciones vecinas superan a lgb_reg. **CONFIRMADO.**
3. **Es el menos sensible al número de árboles:** con 1000 árboles pierde 0,005, frente a 0,017–0,023 de los demás (R2). Por eso el ES ruidoso y el factor de escalado del modelo final le afectan poco. **CONFIRMADO.**
4. **Elimina dui:** en diciembre dui es una permutación de `dias_ultima_transaccion` (0,26 % de coincidencia), así que no aporta Gini generalizable en validación (R6) y es la variable con la ruptura estructural más clara. **CONFIRMADO.**
5. **Ganancia homogénea por segmentos:** es positiva frente a lgb_reg en los 10 segmentos de R5 y no es significativamente peor que nadie en ninguno. **CONFIRMADO.**
6. **Es más estable que lgb_reg** ante la semilla (95 % frente a 87 %) y ante la muestra (83 % frente a 76 %), aunque ningún modelo alcanza el 85 % ante la muestra (R4). **CONFIRMADO.**

**¿Cumple la regla de reemplazo de CLAUDE.md** (Gini medio en 4 folds > +0,003 sin empeorar ningún fold)? **Sí:** +0,0092, con Δ por fold +0,0173 / +0,0051 / +0,0011 / +0,0133. **CONFIRMADO.**
- Es la única de las cuatro alternativas que la cumple. catboost, catboost_sin_dui y ebm_sin_dui superan el +0,003 de media, pero empeoran octubre.
- Además, la prueba formal R1 confirma que la diferencia no es ruido, cosa que la regla de CLAUDE.md por sí sola no garantiza: un +0,003 equivale a ~1,2 SE pareados.

Matices:
- (i) La regla compara con `submission_final.csv`. Aquí la referencia es lgb_reg reentrenado con el esquema honesto. La Spearman entre la submission_final actual y nuestra lgb_reg es de 0,934, y con lgb_sup_sin_dui de 0,968, así que la entrega actual no es idéntica a la referencia evaluada. **CONFIRMADO.**
- (ii) Por la selección de 1 entre 5 candidatos y el sesgo de segundo orden, la mejora esperada en diciembre es probablemente algo menor que +0,009. Un rango razonable es +0,004 a +0,009 de Gini. **PLAUSIBLE.**
- (iii) Reemplazar la entrega es decisión del usuario. Si se hace, CLAUDE.md pide archivar la actual con fecha en `resultados_reportes/historico/`. Este trabajo no ha escrito nada fuera de `validacion_modelos/`.

**Archivo candidato:** `validacion_modelos/resultados/submission_candidata_lgb_sup_sin_dui.csv`. Formato comprobado: `id_cliente,prediccion`, 9900 filas, mismo orden que test.csv, [0,0244; 0,4061], sin nulos. Como referencia, `submission_candidata_lgb_reg.csv`.

**Cómo validarlo antes de entregar:**
```bash
python scripts_utilitarios/comparar_submissions.py validacion_modelos/resultados/submission_candidata_lgb_sup_sin_dui.csv resultados_reportes/submission_final.csv --gini-nuevo 0.2564 --gini-actual 0.2472
```
Opcionalmente, para tenerlo dentro del pipeline oficial: añadir el preset a `PARAMS_DEFECTO` y excluir dui en `Config`, y luego ejecutar el comando completo de CLAUDE.md con `--salida resultados_reportes/submission_candidata.csv`.

## 4. Limitaciones, avisos y correcciones

- **Sin pesos adversariales** (E2). Ninguna prueba usa `pesos_adversariales.csv` ni los escenarios c/e de `07_`: los pesos tienen una Spearman ≈ −0,14 con el score y dan un bono mecánico en noviembre, porque diciembre son los supervivientes de noviembre. El escenario de diciembre usado es el f.
- **Holm** (E3) en todas las familias: R1 (10 pares; DeLong dentro de cada fold), R5 (40), R6 (por modelo, con p normal por la resolución del bootstrap), R7 (5) y R9 (3). Al aplicarlo, desaparecen varias "significancias" sin ajustar (p. ej. lgb_sup_sin_dui frente a catboost, p = 0,024 → p_Holm = 0,21; catboost en R7, p = 0,038 → 0,19).
- **Sesgo de selección de segundo orden** (E5). El preset de lgb_reg y el modo `estatico` se eligieron con experimentos sobre sep–nov y jun–ago, así que ago–nov no son folds vírgenes. Ese sesgo favorece a la referencia; aun así, elegir 1 de 5 candidatos infla algo el Δ del ganador.
- **Signo de dui por fold** (E6). Permutar dui *mejora* agosto (+0,005) y *empeora* sep–nov (−0,002 a −0,004). En R1, quitar dui (catboost frente a catboost_sin_dui) da +0,0083 en agosto y ≈ 0 en sep–nov. La ventaja de las variantes sin dui, por tanto, no es homogénea por fold. La de lgb_sup_sin_dui sí lo es (4/4), porque procede sobre todo de la menor capacidad (R1, R2, R3).
- **Corrección a `INFORME_VALIDACION.md:164`** (no modificado): dice que "en sep–nov hay más dui coincidente". Es al revés: el % de filas con dui = dut baja de forma monótona (ago 36,5 %, sep 27,6 %, oct 18,4 %, nov 9,4 %, dic 0,26 %; `drift_dui_por_mes.csv`). sep–nov tiene *menos* coincidencia que ago–nov, así que la mayor pérdida al permutar en sep–nov no se explica por eso. **CONFIRMADO** (dato); la causa es **PLAUSIBLE**: el modelo se apoya en dui de forma distinta según cuántas filas coincidían en su train.
- **ES del EBM:** usa una partición aleatoria por filas dentro del train del fold. En los modelos sin dui, las filas de un mismo cliente en meses distintos tienen *idénticas* variables, así que la validación interna comparte "copias" con el entrenamiento y el ES puede parar tarde. No contamina el mes evaluado.
- **Umbral "1 SE":** SE_REF (0,008) es un SE absoluto, generoso para cambios pareados (SE 0,001–0,003). R2, R3 y R6 se aprueban con holgura según ese umbral; por eso se reportan también los SE pareados.
- **Incidencias de cómputo:**
  - Un `taskkill` ajeno cortó la segunda ejecución de `09_`, que se relanzó completa.
  - El R4 v1 (filas duplicadas) no terminó con el EBM y no se pudo detener (permiso denegado). Siguió consumiendo CPU y ralentizó R6–R9 (R8 tardó 82 min en lugar de ~6).
  - Por esa contención, R7 y R9 se lanzaron en una cadena paralela (`logs/*_c3.log`). La cadena original los vuelve a ejecutar después con las mismas semillas y resultados idénticos.
  - Cómputo total ≈ 2,5 h de reloj, por encima de los ~90 min previstos, sobre todo por la contención.
- **Archivos obsoletos:** `robustez_r4_pred_dic_lgb_sup_sin_dui.csv` y, si llega a escribirse, `robustez_r4_resumen.csv` (v1). Las cifras válidas son las de `robustez_r4v2_*`. `cache_modelos/` (≈ 2 MB) puede borrarse; solo lo usa R6.

## 5. Archivos y reproducción

Desde la raíz del repo: `bash validacion_modelos/scripts/ejecutar_robustez.sh [desde]`, y después `.venv/Scripts/python.exe validacion_modelos/scripts/19_tabla_y_submissions.py`.

| Script | Prueba | Salidas en `resultados/` | Figura |
|---|---|---|---|
| `criterios_robustez.py` | pre-registro + ENMIENDA_1 | — | — |
| `robustez_comun.py` | candidatos, ajuste honesto, DeLong, Holm, bootstrap | — | — |
| `09_r1_walkforward_candidatos.py` | R1 (entrenamiento) | `robustez_r1_predicciones.csv/.npz`, `robustez_r1_iteraciones.csv` | — |
| `10_r1_comparacion_formal.py` | R1 (inferencia) | `robustez_r1_resumen.csv`, `_pares.csv`, `_delong_fold.csv` | `fig10_r1_comparacion_formal.png` |
| `11_r2_num_arboles.py` | R2 | `robustez_r2_curva.csv`, `_resumen.csv` | `fig11_r2_num_arboles.png` |
| `12_r3_hiperparametros.py` | R3 | `robustez_r3_hiperparametros.csv`, `_resumen.csv` | `fig12_r3_hiperparametros.png` |
| `13_r4_semillas_bootstrap.py` (v2) | R4 | `robustez_r4v2_resumen.csv`, `robustez_r4v2_pred_dic_<m>.csv` | `fig13_r4v2_estabilidad_top10.png` |
| `14_r5_segmentos.py` | R5 | `robustez_r5_segmentos.csv`, `_orden.csv` | `fig14_r5_segmentos.png` |
| `15_r6_perturbacion.py`, `15b_r6_holm_normal.py` | R6 | `robustez_r6_ruido.csv`, `_importancia.csv`, `_importancia_holm_normal.csv`, `_resumen.csv` | `fig15_r6_perturbacion.png` |
| `16_r7_degradacion.py` | R7 | `robustez_r7_celdas.csv`, `_resumen.csv` | `fig16_r7_degradacion.png` |
| `17_r8_placebo.py` | R8 | `robustez_r8_placebo.csv`, `_resumen.csv` | `fig17_r8_placebo.png` |
| `18_r9_curva_aprendizaje.py` | R9 | `robustez_r9_curva.csv`, `_resumen.csv` | `fig18_r9_curva_aprendizaje.png` |
| `19_tabla_y_submissions.py` | tabla + regla + submissions | `robustez_tabla_comparativa.csv`, `submission_candidata_lgb_sup_sin_dui.csv`, `submission_candidata_lgb_reg.csv` | — |

Logs: `resultados/logs/` (`ejecutar_robustez*.log` con códigos de salida; `*_c3.log` = cadena paralela de R7/R9).
