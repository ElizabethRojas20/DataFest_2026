# -*- coding: utf-8 -*-
"""
Utilidades comunes de validacion_modelos/ (se importan desde los scripts 01_..07_).

* Carga el pipeline por importlib (sin modificarlo) y reutiliza Config / cargar_datos /
  construir_dataset en modo 'estatico' (39 columnas; sin id_cliente ni mes).
* NO se usa EvaluadorModelos / resumen() / main del pipeline (ver INFORME_VALIDACION.md).
* Contiene los CRITERIOS DEL VEREDICTO, fijados antes de ejecutar ningun experimento.

Ejecutar siempre desde la raiz del repo:
    .venv/Scripts/python.exe validacion_modelos/scripts/0X_....py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

RAIZ = Path(__file__).resolve().parents[2]
DIR_VAL = RAIZ / "validacion_modelos"
DIR_RES = DIR_VAL / "resultados"
DIR_FIG = DIR_VAL / "figuras"
DIR_RES.mkdir(parents=True, exist_ok=True)
DIR_FIG.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
SEMILLAS = (42, 43, 44)          # semillas de los modelos reentrenados
MESES_FOLD = (202608, 202609, 202610, 202611)
MES_TEST = 202612
COL_DUI = "dias_ultima_interaccion"
COL_DUT = "dias_ultima_transaccion"
PROP_NUEVOS_DIC = 0.186          # % de filas de clientes nuevos en diciembre (verificado)
N_BOOT = 1000

# =============================================================================
# CRITERIOS DEL VEREDICTO (definidos ANTES de mirar resultados; no tocar despues)
# =============================================================================
CRITERIOS = {
    # C1. Drift de scores del MISMO modelo (entrenado <= oct) entre nov y dic
    "psi_moderado": 0.10,        # 0.10 <= PSI < 0.25 -> cambio moderado
    "psi_severo": 0.25,          # PSI >= 0.25 -> cambio severo
    # C2. Gini esperado en condiciones de diciembre (escenario combinado:
    #     dui permutado + reponderacion adversarial + 18,6 % nuevos), media de 4 folds
    "k_se_pareado": 2.0,         # caida significativa si caida > 2 * SE pareado (bootstrap)
    "caida_material": 0.02,      # caida material si ademas supera 0,02 de Gini (~2 SE de un fold)
    # C3. Estabilidad del ranking
    "delta_spearman_modelos": 0.05,  # Spearman medio con los demas modelos en dic >= 0,05 menor que en nov
    "delta_spearman_semillas": 0.02, # Spearman medio entre semillas en dic >= 0,02 menor que en nov
    # C4. Covariate shift (nivel datos, no por modelo)
    "auc_adversarial_relevante": 0.60,
}
# Reglas de veredicto por modelo:
#   SE CAE                 : C2 material (caida significativa y > 0,02) o PSI >= 0,25
#   DEGRADACION MODERADA   : C2 significativa pero <= 0,02, o 0,10 <= PSI < 0,25,
#                            o C3 (modelos o semillas) se cumple
#   NO SE CAE              : ninguna de las anteriores
# =============================================================================


def cargar_pipeline():
    spec = importlib.util.spec_from_file_location("wcb", RAIZ / "pipelines" / "Pipeline_DF_WCB.py")
    wcb = importlib.util.module_from_spec(spec)
    sys.modules["wcb"] = wcb
    spec.loader.exec_module(wcb)
    import logging
    logging.getLogger("pipeline").setLevel(logging.WARNING)
    return wcb


def cargar_todo():
    """Devuelve (wcb, datos, train_crudo, test_crudo). train/test crudos en el orden de los CSV."""
    wcb = cargar_pipeline()
    cfg = wcb.Config(ruta_datos=RAIZ / "datos_entrada")
    tr, te = wcb.cargar_datos(cfg)
    d = wcb.construir_dataset(tr, te, cfg)
    # construir_dataset ordena por (id, mes): reordenamos los crudos de train igual para alinear
    tr_ord = tr.sort_values(["id_cliente", "mes"], kind="mergesort").reset_index(drop=True)
    assert (tr_ord["id_cliente"].to_numpy() == d.id_train).all()
    assert (tr_ord["mes"].to_numpy() == d.mes_train).all()
    assert (te["id_cliente"].to_numpy() == d.id_test).all()
    return wcb, d, tr_ord, te


def es_nuevo_train(ids: np.ndarray, meses: np.ndarray) -> np.ndarray:
    """True si la fila es el primer mes del cliente en el panel (ene = todos, no se usa)."""
    s = pd.DataFrame({"id": ids, "mes": meses})
    primero = s.groupby("id")["mes"].transform("min").to_numpy()
    return meses == primero


def gini(y, p, w=None) -> float:
    return 2.0 * roc_auc_score(y, p, sample_weight=w) - 1.0


def psi(ref: np.ndarray, act: np.ndarray, n_bins: int = 10, categorica: bool = False,
        eps: float = 1e-4) -> float:
    """Population Stability Index. Numericas: cortes por cuantiles de la referencia."""
    ref = np.asarray(ref); act = np.asarray(act)
    if categorica or len(np.unique(ref)) <= n_bins:
        cats = np.union1d(np.unique(ref), np.unique(act))
        r = np.array([(ref == c).mean() for c in cats])
        a = np.array([(act == c).mean() for c in cats])
    else:
        cortes = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
        r = np.bincount(np.searchsorted(cortes, ref, side="right"), minlength=len(cortes) + 1) / len(ref)
        a = np.bincount(np.searchsorted(cortes, act, side="right"), minlength=len(cortes) + 1) / len(act)
    r = np.clip(r, eps, None); a = np.clip(a, eps, None)
    return float(np.sum((a - r) * np.log(a / r)))


def pesos_bootstrap_clientes(ids: np.ndarray, n_boot: int, seed: int = SEMILLA):
    """Generador de pesos por fila = n de veces que su cliente sale en un remuestreo CON
    reemplazo de clientes (bootstrap de cluster; un cliente con varias filas entra entero)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(ids, return_inverse=True)
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        yield cnt[inv].astype(float)


def permutar_en_mes(x: np.ndarray, meses: np.ndarray, seed: int = SEMILLA) -> np.ndarray:
    """Permuta x dentro de cada mes (simula la ruptura de dui en diciembre)."""
    rng = np.random.default_rng(seed)
    out = np.array(x, copy=True)
    for m in np.unique(meses):
        idx = np.where(meses == m)[0]
        out[idx] = x[rng.permutation(idx)]
    return out


class GiniRapido:
    """Gini ponderado exacto (igual a 2*roc_auc_score(y, p, sample_weight=w)-1, empates = 1/2)
    precalculando el orden de p una vez; cada evaluacion con pesos nuevos es O(n).
    Sirve para el bootstrap de clusters (pesos = multiplicidad del cliente)."""

    def __init__(self, y, p):
        y = np.asarray(y).astype(float); p = np.asarray(p, dtype=float)
        self.orden = np.argsort(p, kind="mergesort")
        ps = p[self.orden]
        self.ys = y[self.orden]
        self.grupo = np.r_[0, np.cumsum(ps[1:] != ps[:-1])]
        self.ng = self.grupo[-1] + 1

    def __call__(self, w=None):
        w = np.ones(len(self.ys)) if w is None else np.asarray(w, dtype=float)[self.orden]
        wp = np.bincount(self.grupo, w * self.ys, self.ng)
        wn = np.bincount(self.grupo, w * (1 - self.ys), self.ng)
        neg_antes = np.cumsum(wn) - wn
        auc = np.sum(wp * (neg_antes + 0.5 * wn)) / (wp.sum() * wn.sum())
        return 2 * auc - 1
