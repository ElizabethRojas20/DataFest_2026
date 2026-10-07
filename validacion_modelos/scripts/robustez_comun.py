# -*- coding: utf-8 -*-
"""Utilidades de las pruebas de robustez R1-R9 (scripts 09_ en adelante).
Reutiliza comun.py / modelos.py (sin modificarlos). Los criterios estan en criterios_robustez.py."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats

from comun import (COL_DUI, DIR_FIG, DIR_RES, MESES_FOLD, SEMILLA, SEMILLAS, GiniRapido,
                   cargar_todo, es_nuevo_train, gini, permutar_en_mes, pesos_bootstrap_clientes)
from modelos import ajustar_con_es, ajustar_fijo, predecir

logging.getLogger("interpret").setLevel(logging.WARNING)
for _n in list(logging.root.manager.loggerDict):
    if _n.startswith("interpret"):
        logging.getLogger(_n).setLevel(logging.WARNING)

DIR_CACHE = DIR_RES.parent / "cache_modelos"
DIR_CACHE.mkdir(exist_ok=True)
CANDIDATOS = ("lgb_reg", "catboost", "catboost_sin_dui", "lgb_sup_sin_dui", "ebm_sin_dui")
CON_DUI = ("lgb_reg", "catboost")
NUM_ORIG = ["edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses", "numero_productos",
            "saldo_promedio", "dias_ultima_transaccion", "antiguedad_direccion_meses",
            "visitas_web_ultimos_90_dias", "distancia_sucursal_km", "dia_preferido_pago", COL_DUI]
CATEG_OHE = ["ocupacion", "region", "canal_adquisicion", "banda_riesgo", "dispositivo_principal"]


def definir_candidatos(wcb, columnas):
    P = wcb.PARAMS_DEFECTO
    sin = [c for c in columnas if c != COL_DUI]
    sup = dict(P["lightgbm"]); sup.update(num_leaves=8, min_child_samples=200, reg_lambda=10.0,
                                         colsample_bytree=0.8)
    return {
        "lgb_reg": dict(familia="lightgbm", params=dict(P["lightgbm"]), cols=list(columnas)),
        "catboost": dict(familia="catboost", params=dict(P["catboost"]), cols=list(columnas)),
        "catboost_sin_dui": dict(familia="catboost", params=dict(P["catboost"]), cols=sin),
        "lgb_sup_sin_dui": dict(familia="lightgbm", params=sup, cols=sin),
        "ebm_sin_dui": dict(familia="ebm", params=dict(interactions=10), cols=sin),
    }


def ajustar_ebm(params, X, y, seed, w=None):
    from interpret.glassbox import ExplainableBoostingClassifier
    m = ExplainableBoostingClassifier(**params, random_state=seed, n_jobs=-1)
    m.fit(X, y, sample_weight=w)
    return m


def fit_honesto(wcb, spec, X, y, mes, M, seeds=SEMILLAS, filas_mask=None, seed_es=SEMILLA):
    """Walk-forward honesto del fold M. filas_mask: subconjunto de filas permitido en train (R9).
    Devuelve (lista_modelos, info)."""
    fam, params, cols = spec["familia"], spec["params"], spec["cols"]
    ok = np.ones(len(y), bool) if filas_mask is None else filas_mask
    meses = np.sort(np.unique(mes))
    Mm1 = meses[int(np.searchsorted(meses, M)) - 1]
    tr_M = (mes < M) & ok
    if fam == "ebm":
        mods = [ajustar_ebm(params, X.loc[tr_M, cols], y[tr_M], s) for s in seeds]
        return mods, dict(best=np.nan, factor=np.nan, n_iter=np.nan, filas_es=np.nan,
                          filas_M=int(tr_M.sum()))
    tr_es, va_es = (mes < Mm1) & ok, mes == Mm1
    _, best = ajustar_con_es(wcb, fam, params, X.loc[tr_es, cols], y[tr_es],
                             X.loc[va_es, cols], y[va_es], seed_es)
    factor = tr_M.sum() / tr_es.sum()
    n_iter = max(10, int(round(best * factor)))
    mods = [ajustar_fijo(wcb, fam, params, n_iter, X.loc[tr_M, cols], y[tr_M], s) for s in seeds]
    return mods, dict(best=best, factor=factor, n_iter=n_iter, filas_es=int(tr_es.sum()),
                      filas_M=int(tr_M.sum()))


def fit_fijo(wcb, spec, X, y, n_iter, seed, w=None):
    fam, params = spec["familia"], spec["params"]
    if fam == "ebm":
        return ajustar_ebm(params, X, y, seed, w)
    m = wcb.construir_modelo(fam, params, int(n_iter), seed)
    if w is None:
        m.fit(X, y)
    else:
        m.fit(X, y, sample_weight=w)
    return m


def pred(m, X):
    return m.predict_proba(X)[:, 1]


def f32(x):
    return np.asarray(x, dtype=np.float32)   # SIN redondeo (enmienda E1)


# ------------------------------------------------------------------ estadistica
def _midrank(x):
    return stats.rankdata(x, method="average")


def delong(y, p_a, p_b):
    """Test de DeLong (1988), implementacion rapida de Sun & Xu (2014).
    Devuelve (auc_a, auc_b, var_dif, z, p_bilateral)."""
    y = np.asarray(y).astype(bool)
    P = np.vstack([p_a, p_b])
    m, n = y.sum(), (~y).sum()
    aucs, V10, V01 = [], [], []
    for k in range(2):
        pos, neg = P[k, y], P[k, ~y]
        tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(np.r_[pos, neg])
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        V10.append((tz[:m] - tx) / n)          # componente de colocacion de cada positivo
        V01.append(1 - (tz[m:] - ty) / m)      # componente de colocacion de cada negativo
        aucs.append(auc)
    S10, S01 = np.cov(np.vstack(V10)), np.cov(np.vstack(V01))
    S = S10 / m + S01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return aucs[0], aucs[1], var, z, 2 * stats.norm.sf(abs(z))


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[j])); adj[j] = run
    return adj


def p_boot_centrado(dstar, d):
    dstar = np.asarray(dstar)
    return (1 + np.sum(np.abs(dstar - d) >= abs(d))) / (1 + len(dstar))


class BootFolds:
    """Bootstrap de clientes CONJUNTO sobre varios bloques (folds/celdas) que comparten clientes.
    bloques: lista de arrays de posiciones (en un vector global de ids)."""

    def __init__(self, ids_global, n_boot, seed=SEMILLA):
        self.ids = np.asarray(ids_global); self.n_boot = n_boot; self.seed = seed

    def pesos(self):
        return pesos_bootstrap_clientes(self.ids, self.n_boot, seed=self.seed)


def se_boot(vals):
    return float(np.std(vals, ddof=1))


def ic(vals, a=0.05):
    return float(np.quantile(vals, a / 2)), float(np.quantile(vals, 1 - a / 2))


def leer_se_ref():
    r = pd.read_csv(DIR_RES / "robustez_r1_resumen.csv")
    return float(r.loc[r.modelo == "lgb_reg", "se_boot"].iloc[0])
