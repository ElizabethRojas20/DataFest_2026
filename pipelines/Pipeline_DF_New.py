#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de propensión de conversión (métrica: Gini = 2*AUC - 1)
================================================================

Etapas
------
1. Ingeniería de características (lags, ventanas móviles, One-Hot, Target Encoding temporal).
2. Selección de características (varianza cero -> correlación de Pearson -> top-K por LightGBM).
3. Modelado y validación (clase ``EvaluadorModelos``: LightGBM, XGBoost, RandomForest + Optuna).
4. Entrenamiento final con el 100 % de train y generación de ``submission.csv``.

Decisiones de diseño para evitar data leakage
---------------------------------------------
* Los rezagos y ventanas usan SOLO información del pasado del mismo cliente (nunca "leads").
  Son *calendario-conscientes*: un lag de k meses solo se considera válido si el registro
  anterior corresponde exactamente a ``mes - k``; si el cliente no aparecía ese mes, queda NaN.
* Los datos de test (diciembre) se concatenan con train SOLO para poder calcular lags/ventanas
  de las variables predictoras (el historial de noviembre alimenta a diciembre). El objetivo de
  test nunca se usa.
* El Target Encoding de una fila del mes M se calcula únicamente con los objetivos de meses < M
  (media acumulada suavizada). Para test se usan todos los meses de train.
* La validación es TEMPORAL (últimos N meses de train; por defecto 3 ≈ 73/27 en filas), no un
  ``train_test_split`` aleatorio: el test es un mes futuro y un mismo cliente aparece en varios
  meses con casi todas sus variables fijas, lo que inflaría la métrica con una partición aleatoria.
* La selección de variables se ajusta solo con los meses de entrenamiento, de modo que la métrica
  de validación sea una estimación honesta.
* No se usan ``id_cliente`` ni ``mes`` como predictoras (diciembre es un mes no visto).

Uso
---
    python pipeline.py --datos ./data --salida ./salida/submission.csv --trials 30

Dependencias: pandas, numpy, scikit-learn, lightgbm, xgboost, optuna.
"""
from __future__ import annotations

import argparse
import gc
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pipeline")
optuna.logging.set_verbosity(optuna.logging.WARNING)
# LightGBM >= 4.7 avisa que `eval_set` quedará obsoleto; se mantiene por compatibilidad con
# versiones anteriores y se silencia solo ese aviso.
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")


# =============================================================================
# 0. CONFIGURACIÓN
# =============================================================================
@dataclass
class Config:
    """Parámetros centrales del pipeline (todo se ajusta desde aquí)."""

    ruta_datos: Path = Path(".")
    ruta_salida: Path = Path("submission.csv")

    # Columnas clave
    id_col: str = "id_cliente"
    mes_col: str = "mes"
    target: str = "objetivo"

    # Variables según DATASET_DESCRIPTION.md (no se inventa ninguna)
    cols_numericas: List[str] = field(default_factory=lambda: [
        "edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses",
        "numero_productos", "saldo_promedio", "dias_ultima_transaccion",
        "antiguedad_direccion_meses", "visitas_web_ultimos_90_dias",
        "distancia_sucursal_km", "dia_preferido_pago", "dias_ultima_interaccion"])
    cols_booleanas: List[str] = field(default_factory=lambda: [
        "tiene_tarjeta_credito", "activo_movil", "es_nuevo_cliente",
        "tiene_prestamo", "tiene_seguro"])
    cols_categoricas: List[str] = field(default_factory=lambda: [
        "ocupacion", "region", "canal_adquisicion", "banda_riesgo", "dispositivo_principal"])

    # Variables con rezagos y ventanas móviles
    cols_lag: List[str] = field(default_factory=lambda: [
        "saldo_promedio", "ingresos", "dias_ultima_transaccion", "ratio_deuda_ingresos",
        "numero_productos", "visitas_web_ultimos_90_dias", "dias_ultima_interaccion"])
    lags: Tuple[int, ...] = (1, 2, 3)          # debe incluir 1 (se usa para los deltas)
    cols_rolling: List[str] = field(default_factory=lambda: [
        "saldo_promedio", "dias_ultima_transaccion", "visitas_web_ultimos_90_dias",
        "numero_productos", "dias_ultima_interaccion"])
    ventana: int = 3                            # meses (mes actual + 2 anteriores)

    # Codificación
    umbral_ohe: int = 10                        # nunique <= umbral -> One-Hot; si no -> Target Encoding
    suavizado_te: float = 50.0                  # fuerza del prior en el Target Encoding

    # Selección
    corr_umbral: float = 0.95
    top_k: int = 80

    # Modelado
    n_meses_validacion: int = 3                 # últimos N meses de train para validar (3 ≈ 73/27 en filas)
    n_trials_optuna: int = 30
    n_seeds_final: int = 3
    seed: int = 42


# =============================================================================
# 1. CARGA, VALIDACIÓN Y DIAGNÓSTICO
# =============================================================================
def optimizar_memoria(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memoria: int64 -> int32 y float64 -> float32 (los booleanos se tratan después)."""
    for c in df.select_dtypes(include="integer").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include="floating").columns:
        df[c] = df[c].astype("float32")
    return df


def cargar_datos(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lee train.csv y test.csv con tipos compactos."""
    train = optimizar_memoria(pd.read_csv(cfg.ruta_datos / "train.csv"))
    test = optimizar_memoria(pd.read_csv(cfg.ruta_datos / "test.csv"))
    log.info("train=%s | test=%s | memoria train=%.1f MB",
             train.shape, test.shape, train.memory_usage(deep=True).sum() / 1e6)
    return train, test


def validar_esquema(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> None:
    """Comprueba que existan las columnas descritas y que (cliente, mes) sea único."""
    esperadas = ([cfg.id_col, cfg.mes_col] + cfg.cols_numericas + cfg.cols_booleanas
                 + cfg.cols_categoricas)
    faltan_tr = [c for c in esperadas + [cfg.target] if c not in train.columns]
    faltan_te = [c for c in esperadas if c not in test.columns]
    if faltan_tr or faltan_te:
        raise ValueError(f"Columnas faltantes -> train: {faltan_tr} | test: {faltan_te}")
    for nombre, d in (("train", train), ("test", test)):
        dup = d.duplicated([cfg.id_col, cfg.mes_col]).sum()
        if dup:
            raise ValueError(f"{nombre}: {dup} filas duplicadas en (id_cliente, mes); "
                             "la lógica de lags calendario asume unicidad.")
    if 1 not in cfg.lags:
        raise ValueError("cfg.lags debe incluir el rezago 1 (se usa para los deltas).")
    if train[esperadas].isna().any().any() or test[esperadas].isna().any().any():
        log.warning("Se detectaron NaN en las columnas originales (se esperaban 0).")


def diagnostico_inicial(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> None:
    """Imprime hechos del dataset que condicionan el diseño (sin modificar nada)."""
    tasa = train.groupby(cfg.mes_col)[cfg.target].agg(["mean", "sum", "size"])
    log.info("Tasa de conversión por mes:\n%s", tasa.round(4).to_string())
    conv = train.loc[train[cfg.target] == 1].groupby(cfg.id_col)[cfg.mes_col].min().rename("mes_conv")
    post = train[[cfg.id_col, cfg.mes_col]].merge(conv, on=cfg.id_col, how="inner")
    log.info("Filas de clientes DESPUÉS de su conversión: %d (si es 0, convertidos no reaparecen)",
             int((post[cfg.mes_col] > post["mes_conv"]).sum()))
    log.info("%% de clientes de test con historial en train: %.1f%%",
             100 * test[cfg.id_col].isin(train[cfg.id_col]).mean())
    for c in cfg.cols_categoricas:
        log.info("Cardinalidad %-22s = %d", c, train[c].nunique())


# =============================================================================
# 2. INGENIERÍA DE CARACTERÍSTICAS
# =============================================================================
def preparar_base(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Concatena train+test y ORDENA por (id_cliente, mes).

    Se añaden columnas auxiliares (prefijo '_'):
      _es_test     : True para filas de test.
      _orden_test  : posición original en test.csv (para restaurar el orden al exportar).
      _mes_idx     : índice entero de mes (año*12 + mes) para medir brechas calendario.
    """
    tr = train.copy()
    tr["_es_test"] = False
    tr["_orden_test"] = -1
    te = test.copy()
    te[cfg.target] = np.nan
    te["_es_test"] = True
    te["_orden_test"] = np.arange(len(te))
    df = pd.concat([tr, te], ignore_index=True)
    df = df.sort_values([cfg.id_col, cfg.mes_col], kind="mergesort").reset_index(drop=True)
    df["_mes_idx"] = (df[cfg.mes_col] // 100) * 12 + (df[cfg.mes_col] % 100)
    return df


def crear_features_temporales(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Rezagos, deltas y ventanas móviles calendario-conscientes (``df`` ya ordenado).

    * ``{col}_lag{k}``        : valor del mes ``mes-k``; NaN si el cliente no estaba ese mes.
    * ``{col}_delta1``        : valor actual - lag1 (cambio mensual).
    * ``{col}_roll{w}_mean``  : promedio de los últimos w meses disponibles (incluye el actual).
    * ``{col}_roll{w}_sum``   : suma de los últimos w meses disponibles (incluye el actual).
    * ``meses_en_ventana{w}`` : cuántos de los w meses existen (para interpretar la suma).

    Nota: en variables tipo "últimos 90 días" las sumas de ventanas se solapan; aun así
    resumen la intensidad reciente y el modelo/filtros decidirán si aportan.
    """
    w = cfg.ventana
    k_max = max(max(cfg.lags), w - 1)
    g_mes = df.groupby(cfg.id_col, sort=False)["_mes_idx"]
    # mascara[k][i] = True si la fila k posiciones atrás es exactamente k meses antes
    mascara = {k: ((df["_mes_idx"] - g_mes.shift(k)).to_numpy() == k) for k in range(1, k_max + 1)}

    nuevas: Dict[str, np.ndarray] = {}
    for col in dict.fromkeys(cfg.cols_lag + cfg.cols_rolling):
        actual = df[col].to_numpy(dtype="float32")
        g_col = df.groupby(cfg.id_col, sort=False)[col]
        ks = set()
        if col in cfg.cols_lag:
            ks |= set(cfg.lags) | {1}
        if col in cfg.cols_rolling:
            ks |= set(range(1, w))
        lag_arr: Dict[int, np.ndarray] = {}
        for k in sorted(ks):
            v = g_col.shift(k).to_numpy(dtype="float32", na_value=np.nan)
            lag_arr[k] = np.where(mascara[k], v, np.float32(np.nan)).astype("float32")

        if col in cfg.cols_lag:
            for k in cfg.lags:
                nuevas[f"{col}_lag{k}"] = lag_arr[k]
            nuevas[f"{col}_delta1"] = (actual - lag_arr[1]).astype("float32")
        if col in cfg.cols_rolling:
            pila = np.column_stack([actual] + [lag_arr[k] for k in range(1, w)])
            nuevas[f"{col}_roll{w}_mean"] = np.nanmean(pila, axis=1).astype("float32")
            nuevas[f"{col}_roll{w}_sum"] = np.nansum(pila, axis=1).astype("float32")
        del lag_arr
    nuevas[f"meses_en_ventana{w}"] = (
        1 + sum(mascara[k].astype("int8") for k in range(1, w))).astype("int8")
    return pd.DataFrame(nuevas, index=df.index)


def target_encoding_temporal(df: pd.DataFrame, cols: Sequence[str], cfg: Config) -> pd.DataFrame:
    """
    Target Encoding con corte temporal (sin leakage).

    Para una fila de train del mes M:  TE = (S_<M + prior_<M * m) / (N_<M + m)
      S_<M, N_<M : suma de objetivos y nº de filas de la categoría en meses estrictamente < M.
      prior_<M   : tasa global de meses < M.   m = ``cfg.suavizado_te``.
    El primer mes de train no tiene pasado -> NaN (los modelos lo manejan).
    Para filas de test se usa TODO train (equivale a "todo lo anterior a diciembre").
    Las categorías no vistas en train reciben la tasa global.
    """
    es_test = df["_es_test"].to_numpy()
    tr = df.loc[~es_test]
    meses = np.sort(tr[cfg.mes_col].unique())
    m = cfg.suavizado_te

    g = tr.groupby(cfg.mes_col)[cfg.target].agg(["sum", "count"]).reindex(meses)
    cs_g, cc_g = g["sum"].cumsum().to_numpy(float), g["count"].cumsum().to_numpy(float)
    prior_pasado = np.r_[np.nan, cs_g[:-1] / cc_g[:-1]]       # tasa global con meses < M
    prior_total = cs_g[-1] / cc_g[-1]
    idx_mes = np.searchsorted(meses, df[cfg.mes_col].to_numpy())

    salida: Dict[str, np.ndarray] = {}
    for col in cols:
        agg = tr.groupby([cfg.mes_col, col])[cfg.target].agg(["sum", "count"])
        s = agg["sum"].unstack(fill_value=0).reindex(meses, fill_value=0)
        c = agg["count"].unstack(fill_value=0).reindex(meses, fill_value=0)
        cats = s.columns
        cs, cc = s.cumsum().to_numpy(float), c.cumsum().to_numpy(float)
        # acumulados ESTRICTAMENTE anteriores al mes de cada fila (desplaza una fila)
        pasado_s = np.vstack([np.zeros((1, len(cats))), cs[:-1]])
        pasado_c = np.vstack([np.zeros((1, len(cats))), cc[:-1]])
        enc_pasado = (pasado_s + prior_pasado[:, None] * m) / (pasado_c + m)
        enc_total = (cs[-1] + prior_total * m) / (cc[-1] + m)

        idx_cat = cats.get_indexer(df[col])
        res = np.full(len(df), np.nan, dtype="float32")
        ok_tr = (~es_test) & (idx_cat >= 0)
        res[ok_tr] = enc_pasado[idx_mes[ok_tr], idx_cat[ok_tr]]
        ok_te = es_test & (idx_cat >= 0)
        res[ok_te] = enc_total[idx_cat[ok_te]]
        res[es_test & (idx_cat < 0)] = prior_total
        salida[f"te_{col}"] = res
    return pd.DataFrame(salida, index=df.index)


@dataclass
class Datos:
    """Contenedor del resultado de la ingeniería de características."""
    X_train: pd.DataFrame
    y_train: pd.Series
    mes_train: np.ndarray
    X_test: pd.DataFrame            # en el ORDEN ORIGINAL de test.csv
    id_test: np.ndarray             # id_cliente en el orden original de test.csv


def construir_dataset(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> Datos:
    """Orquesta toda la ingeniería de características y devuelve matrices listas para modelar."""
    t0 = time.time()
    df = preparar_base(train, test, cfg)

    # --- Lags / ventanas ---
    temporales = crear_features_temporales(df, cfg)
    log.info("Features temporales: %d columnas", temporales.shape[1])

    # --- Categóricas: baja cardinalidad -> OHE ; alta -> Target Encoding temporal ---
    card = {c: df[c].nunique() for c in cfg.cols_categoricas}
    baja = [c for c, n in card.items() if n <= cfg.umbral_ohe]
    alta = [c for c, n in card.items() if n > cfg.umbral_ohe]
    log.info("One-Hot: %s | Target Encoding: %s", baja, alta)
    ohe = (pd.get_dummies(df[baja], prefix=baja, dtype="int8") if baja
           else pd.DataFrame(index=df.index))
    te = target_encoding_temporal(df, alta, cfg) if alta else pd.DataFrame(index=df.index)

    # --- Ensamblado final (excluye id_cliente y mes) ---
    X = pd.concat([
        df[cfg.cols_numericas].astype("float32"),
        df[cfg.cols_booleanas].astype("int8"),
        temporales, ohe, te], axis=1)
    es_test = df["_es_test"].to_numpy()
    y = df.loc[~es_test, cfg.target].astype("int8")
    mes_train = df.loc[~es_test, cfg.mes_col].to_numpy()

    X_train = X.loc[~es_test].reset_index(drop=True)
    X_test = X.loc[es_test]
    orden = df.loc[es_test, "_orden_test"].to_numpy()
    reorden = np.argsort(orden)                                  # restaura el orden de test.csv
    X_test = X_test.iloc[reorden].reset_index(drop=True)
    id_test = df.loc[es_test, cfg.id_col].to_numpy()[reorden]

    log.info("Dataset: X_train=%s | X_test=%s | %.1f MB | %.1fs",
             X_train.shape, X_test.shape,
             (X_train.memory_usage().sum() + X_test.memory_usage().sum()) / 1e6, time.time() - t0)
    del df, temporales, ohe, te, X
    gc.collect()
    return Datos(X_train, y.reset_index(drop=True), mes_train, X_test, id_test)


# =============================================================================
# 3. SELECCIÓN DE CARACTERÍSTICAS
# =============================================================================
class SelectorVarianzaCero(BaseEstimator, TransformerMixin):
    """Elimina columnas constantes (un único valor distinto, ignorando NaN)."""

    def fit(self, X: pd.DataFrame, y=None):
        n_unicos = X.nunique(dropna=True)
        self.cols_constantes_ = n_unicos[n_unicos <= 1].index.tolist()
        self.cols_conservadas_ = [c for c in X.columns if c not in set(self.cols_constantes_)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.cols_conservadas_]


class SelectorCorrelacion(BaseEstimator, TransformerMixin):
    """
    Elimina variables con |Pearson| > ``umbral``. De cada par correlacionado se descarta la que
    tiene mayor correlación media absoluta con el resto (la más redundante). La matriz se
    calcula sobre una muestra para ahorrar tiempo y memoria.
    """

    def __init__(self, umbral: float = 0.95, max_filas: int = 50_000, seed: int = 42):
        self.umbral, self.max_filas, self.seed = umbral, max_filas, seed

    def fit(self, X: pd.DataFrame, y=None):
        muestra = X if len(X) <= self.max_filas else X.sample(self.max_filas, random_state=self.seed)
        corr = muestra.corr(method="pearson").abs().to_numpy(copy=True)  # copia: pandas 3 devuelve solo-lectura
        np.fill_diagonal(corr, 0.0)
        cols = list(X.columns)
        media = np.nanmean(corr, axis=1)
        eliminar, pares = set(), []
        for i in range(len(cols)):
            if cols[i] in eliminar:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in eliminar or not corr[i, j] > self.umbral:
                    continue
                pares.append((cols[i], cols[j], float(corr[i, j])))
                if media[i] >= media[j]:
                    eliminar.add(cols[i])
                    break
                eliminar.add(cols[j])
        self.cols_eliminadas_ = sorted(eliminar)
        self.pares_ = pares
        self.cols_conservadas_ = [c for c in cols if c not in eliminar]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.cols_conservadas_]


class SelectorImportanciaLGBM(BaseEstimator, TransformerMixin):
    """Entrena un LightGBM rápido y conserva las ``top_k`` variables por importancia (gain)."""

    def __init__(self, top_k: int = 80, seed: int = 42):
        self.top_k, self.seed = top_k, seed

    def fit(self, X: pd.DataFrame, y):
        modelo = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, importance_type="gain",
            random_state=self.seed, n_jobs=-1, verbose=-1)
        modelo.fit(X, y)
        self.importancias_ = (pd.Series(modelo.feature_importances_, index=X.columns)
                              .sort_values(ascending=False))
        self.cols_conservadas_ = self.importancias_.head(self.top_k).index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.cols_conservadas_]


def seleccionar_features(X_fit: pd.DataFrame, y_fit: pd.Series, cfg: Config) -> List[str]:
    """Cadena: varianza cero -> correlación > 0.95 -> top-K LightGBM. Devuelve la lista final."""
    n0 = X_fit.shape[1]
    s1 = SelectorVarianzaCero().fit(X_fit)
    X1 = s1.transform(X_fit)
    log.info("Varianza cero: -%d (%s)", n0 - X1.shape[1], s1.cols_constantes_)

    s2 = SelectorCorrelacion(cfg.corr_umbral, seed=cfg.seed).fit(X1)
    X2 = s2.transform(X1)
    log.info("Correlación > %.2f: -%d (%s)", cfg.corr_umbral, X1.shape[1] - X2.shape[1],
             s2.cols_eliminadas_)

    s3 = SelectorImportanciaLGBM(cfg.top_k, cfg.seed).fit(X2, y_fit)
    log.info("Top-%d por importancia: quedan %d de %d iniciales", cfg.top_k,
             len(s3.cols_conservadas_), n0)
    log.info("Top 10 importancias (gain):\n%s", s3.importancias_.head(10).round(1).to_string())
    return s3.cols_conservadas_


# =============================================================================
# 4. MODELADO
# =============================================================================
PARAMS_DEFECTO: Dict[str, dict] = {
    "lightgbm": dict(learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0),
    "xgboost": dict(learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8,
                    colsample_bytree=0.7, reg_lambda=1.0),
    "random_forest": dict(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                          max_samples=0.7),
}
ROUNDS_MAX = 2000        # tope de árboles con early stopping
ES_ROUNDS = 100


def construir_modelo(familia: str, params: dict, n_iter: int, seed: int, early_stopping: bool = False):
    """
    Fábrica única de modelos (la usan el evaluador y el entrenamiento final).

    n_iter : nº de árboles (ignorado en random_forest, que lleva n_estimators en params).
    early_stopping : True solo en validación; en el modelo final se fija n_iter.
    """
    if familia == "lightgbm":
        return lgb.LGBMClassifier(**params, n_estimators=n_iter, random_state=seed,
                                  n_jobs=-1, verbose=-1)
    if familia == "xgboost":
        extra = dict(early_stopping_rounds=ES_ROUNDS) if early_stopping else {}
        return xgb.XGBClassifier(**params, n_estimators=n_iter, tree_method="hist",
                                 eval_metric="auc", random_state=seed, n_jobs=-1, **extra)
    if familia == "random_forest":
        # RandomForest no admite NaN en todas las versiones -> imputación por mediana
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(**params, random_state=seed, n_jobs=-1))
    raise ValueError(f"Familia desconocida: {familia}")


class EvaluadorModelos:
    """
    Entrena y evalúa LightGBM, XGBoost y RandomForest sobre una partición temporal
    (entrenamiento: meses previos; validación: mes siguiente) y reporta AUC y Gini.

    Los modelos de boosting usan early stopping sobre el conjunto de validación; por eso el
    Gini reportado es ligeramente optimista (es el mismo criterio para todos los boosting).
    ``optimizar_lightgbm`` ajusta hiperparámetros con Optuna y se reporta como fila aparte
    ("LightGBM_Optuna") para que la comparación sea transparente.
    """

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series, seed: int = 42):
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.seed = seed
        self.params = {k: dict(v) for k, v in PARAMS_DEFECTO.items()}
        self.modelos_: Dict[str, object] = {}
        self.resultados_: Dict[str, dict] = {}

    # ---- utilidades ---------------------------------------------------------
    @staticmethod
    def metricas(y_true, y_prob) -> Tuple[float, float]:
        """Devuelve (AUC, Gini) con Gini = 2*AUC - 1."""
        auc = roc_auc_score(y_true, y_prob)
        return auc, 2 * auc - 1

    def _registrar(self, nombre: str, familia: str, modelo, params: dict, mejor_iter: int) -> dict:
        prob = modelo.predict_proba(self.X_val)[:, 1]
        auc, gini = self.metricas(self.y_val, prob)
        self.modelos_[nombre] = modelo
        self.resultados_[nombre] = dict(familia=familia, auc=auc, gini=gini,
                                        mejor_iter=mejor_iter, params=params)
        log.info("%-14s AUC=%.4f | Gini=%.4f | iter=%s", nombre, auc, gini, mejor_iter)
        return self.resultados_[nombre]

    # ---- entrenadores -------------------------------------------------------
    def entrenar_lightgbm(self, params: Optional[dict] = None, nombre: str = "LightGBM") -> dict:
        """LightGBM con early stopping sobre AUC de validación."""
        params = params or self.params["lightgbm"]
        m = construir_modelo("lightgbm", params, ROUNDS_MAX, self.seed)
        m.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], eval_metric="auc",
              callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
        return self._registrar(nombre, "lightgbm", m, params, int(m.best_iteration_))

    def entrenar_xgboost(self, params: Optional[dict] = None, nombre: str = "XGBoost") -> dict:
        """XGBoost (hist) con early stopping sobre AUC de validación."""
        params = params or self.params["xgboost"]
        m = construir_modelo("xgboost", params, ROUNDS_MAX, self.seed, early_stopping=True)
        m.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
        return self._registrar(nombre, "xgboost", m, params, int(m.best_iteration) + 1)

    def entrenar_random_forest(self, params: Optional[dict] = None,
                               nombre: str = "RandomForest") -> dict:
        """RandomForest con imputación por mediana (calculada solo con el set de entrenamiento)."""
        params = params or self.params["random_forest"]
        m = construir_modelo("random_forest", params, 0, self.seed)
        m.fit(self.X_train, self.y_train)
        return self._registrar(nombre, "random_forest", m, params, params["n_estimators"])

    # ---- Optuna -------------------------------------------------------------
    def optimizar_lightgbm(self, n_trials: int = 30) -> dict:
        """Busca hiperparámetros de LightGBM maximizando el AUC de validación."""
        def objetivo(trial: optuna.Trial) -> float:
            p = dict(
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                num_leaves=trial.suggest_int("num_leaves", 8, 256, log=True),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                subsample_freq=1,
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
            m = construir_modelo("lightgbm", p, ROUNDS_MAX, self.seed)
            m.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)],
                  eval_metric="auc", callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
            trial.set_user_attr("best_iter", int(m.best_iteration_))
            return roc_auc_score(self.y_val, m.predict_proba(self.X_val)[:, 1])

        estudio = optuna.create_study(direction="maximize",
                                      sampler=optuna.samplers.TPESampler(seed=self.seed))
        estudio.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)
        mejor = dict(estudio.best_params, subsample_freq=1)
        log.info("Optuna: mejor AUC=%.4f en %d trials", estudio.best_value, n_trials)
        return mejor

    # ---- orquestación -------------------------------------------------------
    def evaluar_todos(self, n_trials_optuna: int = 0) -> pd.DataFrame:
        """Entrena los tres modelos (y Optuna si n_trials_optuna > 0). Devuelve tabla ordenada."""
        self.entrenar_lightgbm()
        self.entrenar_xgboost()
        self.entrenar_random_forest()
        if n_trials_optuna > 0:
            mejor = self.optimizar_lightgbm(n_trials_optuna)
            self.entrenar_lightgbm(mejor, nombre="LightGBM_Optuna")
        return self.resumen()

    def resumen(self) -> pd.DataFrame:
        """Tabla con AUC, Gini y nº de iteraciones por modelo, ordenada por Gini."""
        tabla = pd.DataFrame(
            {n: {k: r[k] for k in ("auc", "gini", "mejor_iter")} for n, r in self.resultados_.items()}
        ).T
        return tabla.sort_values("gini", ascending=False)


# =============================================================================
# 5. ENTRENAMIENTO FINAL Y SUBMISSION
# =============================================================================
def entrenar_y_predecir_final(familia: str, params: dict, mejor_iter: int,
                              X_all: pd.DataFrame, y_all: pd.Series, X_test: pd.DataFrame,
                              n_seeds: int, seed: int, factor_datos: float = 1.1) -> np.ndarray:
    """
    Reentrena el modelo ganador con el 100 % de train y predice test.

    * Boosting: n_iter = factor_datos * mejor_iter, donde factor_datos = filas totales / filas
      de entrenamiento de la validación (más datos admiten algo más de árboles).
    * Se promedian ``n_seeds`` semillas para estabilizar el ranking.
    """
    n_iter = max(10, int(round(mejor_iter * factor_datos)))
    preds = []
    for i in range(n_seeds):
        m = construir_modelo(familia, params, n_iter, seed + i)
        m.fit(X_all, y_all)
        preds.append(m.predict_proba(X_test)[:, 1])
        log.info("Modelo final %s: semilla %d/%d lista", familia, i + 1, n_seeds)
    return np.clip(np.mean(preds, axis=0), 0.0, 1.0)


def exportar_submission(id_test: np.ndarray, pred: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Valida y escribe submission.csv con columnas EXACTAS: id_cliente, prediccion."""
    sub = pd.DataFrame({cfg.id_col: id_test, "prediccion": pred})
    test_orig = pd.read_csv(cfg.ruta_datos / "test.csv", usecols=[cfg.id_col])
    assert len(sub) == len(test_orig), "El nº de filas no coincide con test.csv"
    assert (sub[cfg.id_col].to_numpy() == test_orig[cfg.id_col].to_numpy()).all(), \
        "El orden de id_cliente no coincide con test.csv"
    assert sub["prediccion"].between(0, 1).all() and sub["prediccion"].notna().all(), \
        "Predicciones fuera de [0,1] o con NaN"
    muestra = cfg.ruta_datos / "sample_submission.csv"
    if muestra.exists():
        ids_muestra = pd.read_csv(muestra, usecols=[cfg.id_col])[cfg.id_col].to_numpy()
        assert (ids_muestra == sub[cfg.id_col].to_numpy()).all(), "Difiere de sample_submission.csv"
    cfg.ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(cfg.ruta_salida, index=False)
    log.info("Submission guardado en %s (%d filas)", cfg.ruta_salida, len(sub))
    return sub


# =============================================================================
# 6. MAIN
# =============================================================================
def main(cfg: Config) -> None:
    t0 = time.time()
    train, test = cargar_datos(cfg)
    validar_esquema(train, test, cfg)
    diagnostico_inicial(train, test, cfg)

    # 1) Features
    datos = construir_dataset(train, test, cfg)
    del train, test
    gc.collect()

    # 2) Partición TEMPORAL (no aleatoria): validación = últimos N meses de train; entrenamiento =
    #    meses anteriores. Imita la situación real (predecir un mes futuro) y evita que un mismo
    #    cliente quede a ambos lados de la partición.
    meses = np.sort(np.unique(datos.mes_train))
    if not 1 <= cfg.n_meses_validacion < len(meses):
        raise ValueError(f"n_meses_validacion debe estar entre 1 y {len(meses) - 1}")
    mes_corte = int(meses[-cfg.n_meses_validacion])
    es_val = datos.mes_train >= mes_corte
    es_tr = ~es_val
    factor_datos = len(datos.mes_train) / es_tr.sum()      # para escalar nº de árboles al final
    log.info("Entrenamiento: meses %d-%d (%d filas, %.0f%%) | Validación: meses %d-%d (%d filas, %.0f%%)",
             meses[0], meses[-cfg.n_meses_validacion - 1], es_tr.sum(), 100 * es_tr.mean(),
             mes_corte, meses[-1], es_val.sum(), 100 * es_val.mean())

    # 3) Selección de variables (SOLO con el periodo de entrenamiento)
    features = seleccionar_features(datos.X_train.loc[es_tr], datos.y_train.loc[es_tr], cfg)

    # 4) Comparación de modelos
    ev = EvaluadorModelos(
        datos.X_train.loc[es_tr, features], datos.y_train.loc[es_tr],
        datos.X_train.loc[es_val, features], datos.y_train.loc[es_val], seed=cfg.seed)
    tabla = ev.evaluar_todos(cfg.n_trials_optuna)
    log.info("Resultados en validación (meses %d-%d):\n%s", mes_corte, meses[-1],
             tabla.round(4).to_string())

    # 5) Modelo ganador -> entrenamiento con el 100 % de train -> predicción de test
    ganador = tabla.index[0]
    r = ev.resultados_[ganador]
    log.info("Modelo ganador: %s (Gini=%.4f)", ganador, r["gini"])
    pred = entrenar_y_predecir_final(
        r["familia"], r["params"], int(r["mejor_iter"]),
        datos.X_train[features], datos.y_train, datos.X_test[features],
        cfg.n_seeds_final, cfg.seed, factor_datos)
    exportar_submission(datos.id_test, pred, cfg)

    # Artefactos auxiliares para trazabilidad
    carpeta = cfg.ruta_salida.parent
    tabla.to_csv(carpeta / "validation_results_DF_new.csv")
    (carpeta / "features_seleccionadas_new.txt").write_text("\n".join(features), encoding="utf-8")
    log.info("Pipeline completo en %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline de propensión de conversión (Gini)")
    ap.add_argument("--datos", type=Path, default=Path("."), help="carpeta con train/test/sample")
    ap.add_argument("--salida", type=Path, default=Path("submission_DF_new.csv"))
    ap.add_argument("--trials", type=int, default=30, help="trials de Optuna (0 = sin Optuna)")
    ap.add_argument("--top-k", type=int, default=80)
    ap.add_argument("--meses-val", type=int, default=3,
                    help="nº de últimos meses de train usados para validar (3 ≈ 73/27)")
    ap.add_argument("--umbral-ohe", type=int, default=10,
                    help="nunique <= umbral -> One-Hot; si no -> Target Encoding temporal")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()
    main(Config(ruta_datos=a.datos, ruta_salida=a.salida, n_trials_optuna=a.trials,
                top_k=a.top_k, umbral_ohe=a.umbral_ohe, n_seeds_final=a.n_seeds, seed=a.semilla,
                n_meses_validacion=a.meses_val))
