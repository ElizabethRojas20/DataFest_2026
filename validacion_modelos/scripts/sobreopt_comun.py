# -*- coding: utf-8 -*-
"""
Utilidades y CRITERIOS de las pruebas de sobreoptimización (scripts 21-23).

Pregunta: ¿la ventaja del ganador (LightGBM superficial) es real o es sesgo de selección por
haber probado muchos candidatos sobre los mismos folds ago-nov?

Candidatos = "historial" reproducible con el esquema del pipeline nuevo (EvaluadorModelos: sin
early stopping, nº de árboles por regla 1-SE con los demás folds):
  * 6 presets (lgb_reg, lgb_sin_reg, lgb_sup, xgb, catboost, rf) x {con, sin dui}  = 12
  * vecindad R3 del superficial, sin dui: 8 cambios de un factor + 16 esquinas ×½/×2 = 24
  * ebm_sin_dui (solo en el registro y las pruebas 22; demasiado lento para 23)    = 1
Quedan fuera (no reproducibles con este esquema): redes neuronales, ensembles y Optuna. El
historial real es por tanto MAYOR que N y las correcciones de este análisis son un mínimo.
Referencia ("modelo actual"): lgb_reg_con_dui.
"""
from __future__ import annotations

from itertools import product

from comun import COL_DUI

REF = "lgb_reg_con_dui"
GANADOR_PIPELINE = "lgb_sup_sin_dui"

# =============================================================================
# CRITERIOS (fijados ANTES de ejecutar 21-23; no tocar después)
# =============================================================================
CRITERIOS_SO = {
    "alpha": 0.05,
    # Prueba 1 (anidado): la mejora se considera libre de sesgo de selección si el optimismo
    # (Gini aparente del ganador - Gini anidado de la receta) es < 0,002 y la receta anidada
    # mejora a la referencia anidada en media.
    "optimismo_max": 0.002,
    # Prueba 2 (PBO/CSCV): S bloques de CLIENTES (cada cliente entero en un bloque, todos sus meses).
    "cscv_bloques": 12,
    "pbo_fiable": 0.10, "pbo_dudoso": 0.30,
    # Prueba 3 (SPA de Hansen / Reality Check de White / Romano-Wolf): bootstrap de clientes.
    "n_boot": 2000,
    # Prueba 4 (Gini deflactado): z deflactado > 1,645 (unilateral 5 %).
    "z_deflactado": 1.645,
    # Prueba 5 (placebo de la receta): Δ real del ganador > percentil 95 de los Δ placebo.
    "placebo_percentil": 95, "placebo_replicas": 20,
    # Prueba 6 (MCS): nivel 10 % (Hansen, Lunde y Nason, 2011).
    "mcs_alpha": 0.10,
}
SEMILLA_SO = 2026


def candidatos(wcb, columnas, incluir_ebm=True, solo_rapidos=False):
    """Diccionario nombre -> dict(familia, params, cols, grupo).
    solo_rapidos=True: solo LightGBM y XGBoost (receta del placebo, por coste)."""
    P = wcb.PARAMS_DEFECTO
    con = list(columnas)
    sin = [c for c in columnas if c != COL_DUI]
    presets = [("lgb_reg", "lightgbm", P["lightgbm"]),
               ("lgb_sin_reg", "lightgbm", P["lightgbm_sin_regularizar"]),
               ("lgb_sup", "lightgbm", P["lightgbm_superficial"]),
               ("xgb", "xgboost", P["xgboost"]),
               ("catboost", "catboost", P["catboost"]),
               ("rf", "random_forest", P["random_forest"])]
    c = {}
    for nombre, fam, prm in presets:
        if solo_rapidos and fam not in ("lightgbm", "xgboost"):
            continue
        c[f"{nombre}_con_dui"] = dict(familia=fam, params=dict(prm), cols=con, grupo="preset")
        c[f"{nombre}_sin_dui"] = dict(familia=fam, params=dict(prm), cols=sin, grupo="preset")
    base = dict(P["lightgbm_superficial"])
    factores = {"num_leaves": (4, 16), "min_child_samples": (100, 400),
                "reg_lambda": (3.0, 30.0), "learning_rate": (0.015, 0.06)}
    for k, (bajo, alto) in factores.items():
        for v, et in ((bajo, "bajo"), (alto, "alto")):
            c[f"r3_{k}_{et}"] = dict(familia="lightgbm", params={**base, k: v}, cols=sin, grupo="r3_un_factor")
    for combo in product(*[(0, 1)] * len(factores)):
        p = dict(base)
        for (k, vals), i in zip(factores.items(), combo):
            p[k] = vals[i]
        et = "".join("A" if i else "b" for i in combo)
        c[f"r3_esq_{et}"] = dict(familia="lightgbm", params=p, cols=sin, grupo="r3_esquina")
    if incluir_ebm and not solo_rapidos:
        c["ebm_sin_dui"] = dict(familia="ebm", params=dict(interactions=10), cols=sin, grupo="otro")
    return c


def ajustar_ebm(params, X, y, seed):
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(**params, random_state=seed, n_jobs=-1).fit(X, y)
