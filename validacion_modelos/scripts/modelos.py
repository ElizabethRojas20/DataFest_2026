# -*- coding: utf-8 -*-
"""Definicion de los modelos evaluados y funciones de ajuste (reutiliza construir_modelo del pipeline)."""
from __future__ import annotations

import lightgbm as lgb

from comun import COL_DUI


def definir_modelos(wcb, columnas):
    P = wcb.PARAMS_DEFECTO
    sin_dui = [c for c in columnas if c != COL_DUI]
    return {
        "lgb_reg":         dict(familia="lightgbm", params=P["lightgbm"], cols=list(columnas)),
        "lgb_sin_reg":     dict(familia="lightgbm", params=P["lightgbm_sin_regularizar"], cols=list(columnas)),
        "xgb":             dict(familia="xgboost", params=P["xgboost"], cols=list(columnas)),
        "catboost":        dict(familia="catboost", params=P["catboost"], cols=list(columnas)),
        "random_forest":   dict(familia="random_forest", params=P["random_forest"], cols=list(columnas)),
        "lgb_reg_sin_dui": dict(familia="lightgbm", params=P["lightgbm"], cols=sin_dui),
    }


def ajustar_con_es(wcb, familia, params, Xtr, ytr, Xva, yva, seed):
    """Early stopping identico al del pipeline (Pipeline_DF_WCB.py:614-634).
    Devuelve (modelo, best_iter). RandomForest no tiene ES: best_iter = n_estimators."""
    if familia == "lightgbm":
        m = wcb.construir_modelo("lightgbm", params, wcb.ROUNDS_MAX, seed)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(wcb.ES_ROUNDS, verbose=False)])
        return m, int(m.best_iteration_)
    if familia == "xgboost":
        m = wcb.construir_modelo("xgboost", params, wcb.ROUNDS_MAX, seed, early_stopping=True)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m, int(m.best_iteration) + 1
    if familia == "catboost":
        m = wcb.construir_modelo("catboost", params, wcb.ROUNDS_MAX, seed, early_stopping=True)
        m.fit(Xtr, ytr, eval_set=(Xva, yva))
        return m, int(m.get_best_iteration()) + 1
    if familia == "random_forest":
        m = wcb.construir_modelo("random_forest", params, 0, seed)
        m.fit(Xtr, ytr)
        return m, int(params["n_estimators"])
    raise ValueError(familia)


def ajustar_fijo(wcb, familia, params, n_iter, X, y, seed):
    """Modelo con n_iter fijo (sin early stopping), como entrenar_y_predecir_final."""
    m = wcb.construir_modelo(familia, params, int(n_iter), seed)
    m.fit(X, y)
    return m


def predecir(m, X):
    return m.predict_proba(X)[:, 1]
