# -*- coding: utf-8 -*-
"""
PRE-REGISTRO de las pruebas de robustez R1-R9 (validacion_modelos/, octubre 2026).

Escrito ANTES de ejecutar cualquier script 09_ en adelante y antes de que exista ningun
resultados/robustez_*.csv. NO SE MODIFICA DESPUES (la marca de tiempo del archivo lo acredita).

Transparencia: ya se conocian los resultados previos de 01_-08_ (wf_resumen.csv): boostings
indistinguibles (lgb_reg 0,2472; catboost 0,2509; lgb_reg_sin_dui 0,2496), early stopping
inestable, dui = ruido en diciembre. Por eso las hipotesis de R1 NO son confirmatorias "a ciegas"
para lgb_reg/catboost; lo nuevo y pre-registrado son los candidatos lgb_sup_sin_dui, catboost_sin_dui
y ebm_sin_dui, y todas las reglas de decision de R1-R9.

Unico ensayo previo: medida de TIEMPOS de ajuste (EBM ~25 s, CatBoost 1000 arboles ~12 s,
LightGBM superficial <1 s) para dimensionar rejillas/replicas. No se miro ningun Gini.

ESQUEMA COMUN (identico a 01_walkforward.py):
  folds M in {ago, sep, oct, nov 2026}. Boosting: ES (paciencia 100, max 2000) entrenando con
  meses < M-1 y validando en M-1 (semilla 42) -> best; n_iter = max(10, round(best * filas(<M)/filas(<M-1)));
  reentreno con meses < M y n_iter fijo, semillas 42, 43, 44; prediccion media de las 3 semillas.
  EBM: ajuste directo con meses < M (su ES interno usa una particion ALEATORIA por filas del 15 %
  del propio train del fold: no toca el mes evaluado, pero mezcla filas del mismo cliente entre su
  train y su validacion interna -> su ES puede ser algo optimista; se documenta).
  Modelo final: todo train; n_iter = mediana_k(best_k * filas(train)/filas(<M_k - 1)).

SE_REF ("1 SE" de R2, R3, R6, R9): SE por bootstrap de clientes (2000 replicas, conjunto sobre
los 4 folds) de la MEDIA de los 4 Gini por fold de lgb_reg en R1. Esperado ~0,008. Se calcula en
10_ y se guarda en robustez_r1_resumen.csv. (En R2 se usa el SE propio del candidato en n*).
"""

SEMILLA = 42
SEMILLAS = (42, 43, 44)

CANDIDATOS = {
    "lgb_reg":          "PARAMS_DEFECTO['lightgbm'] con dui (referencia)",
    "catboost":         "PARAMS_DEFECTO['catboost'] con dui",
    "catboost_sin_dui": "PARAMS_DEFECTO['catboost'] sin dias_ultima_interaccion",
    "lgb_sup_sin_dui":  "preset lightgbm con num_leaves=8, min_child_samples=200, reg_lambda=10, "
                        "colsample_bytree=0.8; sin dui",
    "ebm_sin_dui":      "ExplainableBoostingClassifier(interactions=10, outer_bags=14 (defecto), "
                        "random_state=semilla, resto por defecto); sin dui",
}
REFERENCIA = "lgb_reg"
CON_DUI = ("lgb_reg", "catboost")

# Orden de complejidad/restriccion de la clase funcional (para desempates; 1 = mas simple):
# EBM = GA2M (orden de interaccion <= 2); lgb_sup (8 hojas); lgb_reg (31 hojas);
# catboost (arboles simetricos de profundidad 6 = 64 hojas, interacciones de orden 6).
COMPLEJIDAD = {"ebm_sin_dui": 1, "lgb_sup_sin_dui": 2, "lgb_reg": 3,
               "catboost_sin_dui": 4, "catboost": 5}

CRITERIOS = {
    # ---------------------------------------------------------------- R1
    "R1": dict(
        H0="Gini_a = Gini_b (media de los 4 folds) para cada par (a, b)",
        H1="Gini_a != Gini_b (bilateral)",
        estadistico="Delta = media_folds(Gini_a - Gini_b) con predicciones medias de 3 semillas",
        p_principal="bootstrap PAREADO por cliente, conjunto sobre los 4 folds, B=2000, centrado en "
                    "H0: p = (1 + #{|D* - D| >= |D|}) / (1 + B)",
        p_secundario="DeLong por fold (AUC correlacionadas, mismas filas; valido dentro de un mes "
                     "porque cada cliente aparece una sola vez). No se combina entre folds "
                     "(no son independientes).",
        familia_holm="los 10 pares de los 5 candidatos (FWER 5 %)",
        alpha=0.05, n_boot=2000, min_folds_mismo_signo=3,
        regla_mejor="a es MEJOR que b si p_Holm < 0,05, Delta > 0 y Delta_fold > 0 en >= 3/4 folds",
        escenario_secundario="condiciones de diciembre: dui permutada dentro del mes evaluado "
                             "(3 permutaciones) para lgb_reg y catboost; mismo test (sensibilidad)",
        regla_claude_md="reemplazo si Delta medio vs lgb_reg > 0,003 y Delta_fold >= 0 en los 4 folds",
        parsimonia=(
            "Si ningun candidato es MEJOR que lgb_reg ni que el resto: S = {c : c es el de mayor "
            "Gini medio, o p_Holm(c vs ese) >= 0,05}. Dentro de S se elige por orden: "
            "(1) menos pruebas R2-R9 SUSPENDIDAS entre las aplicadas a ese candidato; "
            "(2) no usar dui (variable rota en diciembre: es una permutacion de "
            "dias_ultima_transaccion); (3) menor COMPLEJIDAD; (4) menor varianza entre semillas "
            "del Gini de validacion."),
    ),
    # ---------------------------------------------------------------- R2
    "R2": dict(
        modelos=("lgb_reg", "catboost", "catboost_sin_dui", "lgb_sup_sin_dui"),
        H0="el Gini es insensible al n de arboles en un rango amplio (meseta ancha)",
        rejilla=(5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 500, 700, 1000),
        esquema="por fold: entrenar con < M hasta 1000 arboles (3 semillas), Gini en M con los "
                "primeros n arboles; G(n) = media de los 4 folds",
        meseta="{n : G(n) >= G(n*) - SE(n*)}, n* = argmax G, SE(n*) bootstrap de clientes (500)",
        regla_1se="n_1SE = menor n de la meseta",
        robusto="max(meseta)/min(meseta) >= 3",
        secundario="meseta estricta con SE PAREADO de G(n) - G(n*) (300 replicas)",
        aviso="n* y n_1SE se eligen con los mismos folds: su Gini esta sesgado al alza",
    ),
    # ---------------------------------------------------------------- R3
    "R3": dict(
        H0="el Gini medio es estable en una vecindad del preset",
        lgb_sup_sin_dui=dict(centro=dict(num_leaves=8, min_child_samples=200, reg_lambda=10.0,
                                         learning_rate=0.03),
                             vecindad="un-factor-cada-vez (x0,5 y x2 de cada parametro) -> 9 configs",
                             extendida="16 esquinas del factorial 2^4 (todos los factores a la vez)",
                             semillas=SEMILLAS),
        catboost_sin_dui=dict(centro=dict(depth=6, l2_leaf_reg=3.0, learning_rate=0.05),
                              vecindad="depth {4, 8}, l2_leaf_reg {1, 10}, learning_rate {0,025; 0,1} "
                                       "un-factor-cada-vez -> 7 configs",
                              nota="min_data_in_leaf no aplica a arboles simetricos (SymmetricTree)",
                              semillas=(42,)),
        esquema="walk-forward honesto completo (ES en M-1, reentreno < M)",
        robusto="max - min del Gini medio en la vecindad un-factor-cada-vez <= SE_REF",
        nota="el rango incluye el ruido de semilla: criterio conservador",
    ),
    # ---------------------------------------------------------------- R4
    "R4": dict(
        modelos="los 2 de mayor Gini medio en R1 (+ lgb_reg como referencia si no esta)",
        semillas=tuple(range(42, 52)), n_boot_clientes=20, semilla_modelo_boot=42,
        metricas="en diciembre: percentil de cada cliente por replica; sd del percentil; top-10 % "
                 "(990 clientes) por replica; Jaccard medio entre pares de replicas; "
                 "top-10 % de referencia = top-10 % de la media de percentiles; estable = "
                 "pertenencia en >= 90 % de las replicas",
        robusto="% estable del top-10 % de referencia >= 85 %; se evalua por separado para "
                "semillas y para bootstrap de clientes; pasa R4 si cumple ambos",
    ),
    # ---------------------------------------------------------------- R5
    "R5": dict(
        segmentos={"nuevo": "primer mes del cliente en el panel (sin historia) vs resto",
                   "banda_riesgo": "low / medium / high",
                   "numero_productos": "1 / 2 / 3+",
                   "activo_movil": "True / False"},
        estadistico="media de los 4 Gini por fold dentro del segmento; IC95 percentil bootstrap "
                    "de clientes (1000) conjunto",
        orden="tau de Kendall entre el orden global de los 5 candidatos y el del segmento; "
              "Delta vs lgb_reg por segmento con IC",
        criterio="solo calculo (sin interpretacion de negocio)",
    ),
    # ---------------------------------------------------------------- R6
    "R6": dict(
        H0="un error de medida moderado en las entradas no cambia el Gini",
        ruido="multiplicativo log-normal x' = x * exp(s*Z - s^2/2), Z~N(0,1), s in {0,05; 0,10; 0,20} "
              "(CV del error ~ s; media preservada; conserva el signo); enteros redondeados; "
              "recorte al rango de train; solo al PREDECIR (modelos de cada fold, semilla 42); "
              "5 sorteos por nivel",
        columnas="numericas no binarias del modelo (las 12 originales; sin dui en los *_sin_dui)",
        importancia="permutacion por variable ORIGINAL (dummies OHE de una categorica se permutan "
                    "juntas) dentro del mes de validacion, 3 permutaciones, media de 4 folds; "
                    "IC95 bootstrap de clientes (200)",
        robusto="caida media de Gini con s=0,10 < SE_REF Y ninguna variable > 50 % de la suma de "
                "importancias positivas",
    ),
    # ---------------------------------------------------------------- R7
    "R7": dict(
        H0="pendiente de Gini frente al desfase = 0 (no hay envejecimiento del modelo)",
        origenes="k in mar..oct 2026; entrenar con <= k (ES honesto en k entrenando < k, factor de "
                 "filas); evaluar en k+h, h = 1..4, k+h <= nov",
        modelos="los 5 candidatos; semillas 42-44 en LightGBM, 42 en CatBoost y EBM",
        estadistico="beta de OLS en G_kh = alpha_k + beta*h (efectos fijos de origen)",
        ic="bootstrap de clientes (1000) conjunto sobre todas las celdas (k, h)",
        robusto="IC95 de beta incluye 0 (o beta > 0)",
        secundario="efectos fijos de mes evaluado en lugar de origen (confunde con tamano de train)",
    ),
    # ---------------------------------------------------------------- R8
    "R8": dict(
        modelo="lgb_sup_sin_dui",
        H0="el modelo no aprende senal (Gini en validacion compatible con etiquetas permutadas)",
        placebo="permutar objetivo DENTRO de cada mes de train (< M); n_iter = el del modelo real en "
                "ese fold (misma capacidad); semilla 42; evaluar contra las etiquetas REALES de M",
        B=100, estadistico="media de los 4 Gini por fold",
        p="(1 + #{G_placebo >= G_real}) / (1 + B)",
        valido="placebo centrado en 0 (|media| < 2*sd/sqrt(B)) Y G_real > max(placebo) Y "
               "z = (G_real - media)/sd > 3",
    ),
    # ---------------------------------------------------------------- R9
    "R9": dict(
        modelos=("lgb_reg", "lgb_sup_sin_dui", "catboost_sin_dui"),
        fracciones=(0.25, 0.5, 0.75, 1.0),
        submuestreo="por CLIENTE (U ~ uniforme por cliente; entra si U < f; subconjuntos anidados); "
                    "replicas: 3 (LightGBM) y 2 (CatBoost) para f < 1; walk-forward honesto, semilla 42",
        saturado="G(100 %) - G(75 %) < SE_REF",
        ley_potencia="G(n) = G_inf - a * n^(-b) por minimos cuadrados (n = filas de train medias); "
                     "solo se reporta si converge con b > 0",
    ),
}

# Presupuesto: ~90 min de computo en total; si se recorta algo se documenta en el informe.


# =============================================================================
# ENMIENDA previa a resultados (2026-10-05 23:34, antes de que exista ningun robustez_*.csv).
# Origen: auditoria de codigo de 01-08 (orquestador). La primera ejecucion de 09_ se DETUVO
# antes de escribir nada (solo habia impreso Gini por fold de lgb_reg, catboost y 2 folds de
# catboost_sin_dui en el log; ninguna regla de abajo depende de ellos).
# =============================================================================
ENMIENDA_1 = dict(
    E1_redondeo="Las predicciones NO se redondean (redondear a 6 decimales crea empates y altera el "
                "AUC). Se guardan en float32 sin redondear (CSV) y, para el calculo de metricas entre "
                "scripts, en float64 (npz). Las metricas de cada script se calculan sobre predicciones "
                "en memoria / float64.",
    E2_escenario_dic="Ninguna prueba usa los pesos adversariales (pesos_adversariales.csv; escenarios c/e "
                     "de 07_): estan correlacionados con el score (Spearman ~ -0,14) y dan un bono mecanico "
                     "en noviembre. El escenario secundario de R1 pasa a ser el f de 07_: dui permutada "
                     "dentro del mes (3 permutaciones, solo modelos con dui) + pesos por fila para que los "
                     "clientes nuevos sean el 18,6 % de cada fold. Para el escenario f solo hay p bootstrap "
                     "(DeLong no admite pesos); DeLong se reporta en tal cual.",
    E3_holm="Correccion de Holm (FWER 5 %) en TODA familia de comparaciones multiples: R1 (10 pares por "
            "escenario; DeLong: 10 pares dentro de cada fold); R5 (Delta vs lgb_reg: 4 candidatos x 9 "
            "segmentos = 36 contrastes); R6 (importancia > 0: una familia por modelo con todas sus "
            "variables, p bootstrap unilateral); R7 (beta = 0 en los 5 candidatos); R9 (ganancia "
            "75 %->100 % = 0 en los 3 modelos). Los p bootstrap son centrados: (1+#{|D*-D|>=|D|})/(1+B).",
    E4_es="El ES de LightGBM ya para por AUC (eval_metric='auc' primero). La inestabilidad del n de "
          "arboles se atribuye a un AUC de validacion plano y ruidoso; R2 lo cuantifica.",
    E5_limitacion="Sesgo de seleccion de segundo orden: el preset de lgb_reg y el modo 'estatico' se "
                  "eligieron con experimentos sobre sep-nov y jun-ago; ago-nov no son folds virgenes. "
                  "Se declara como limitacion de R1 (afecta sobre todo a lgb_reg, a favor de la referencia).",
    E6_dui="El efecto de dui cambia de signo por fold (permutarla mejora agosto y empeora sep-nov); "
           "se tendra en cuenta al interpretar las variantes sin dui (no es una mejora homogenea).",
)
