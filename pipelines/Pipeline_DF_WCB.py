#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de propensión de conversión (métrica: Gini = 2*AUC - 1)
================================================================

Etapas
------
1. Ingeniería de características (lags, ventanas móviles, One-Hot, Target Encoding temporal).
2. Selección de características (varianza cero -> correlación de Pearson -> top-K por LightGBM).
3. Modelado y validación (clase ``EvaluadorModelos``: LightGBM, XGBoost, CatBoost, RandomForest + Optuna).
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
* La validación es TEMPORAL (walk-forward: cada uno de los últimos N meses de train es un fold;
  por defecto 4), no un ``train_test_split`` aleatorio: el test es un mes futuro y un mismo cliente
  aparece en varios meses con casi todas sus variables fijas, lo que inflaría la métrica.
* El nº de árboles no se elige con early stopping sobre el mes evaluado: el Gini de cada fold usa
  el nº elegido con los demás folds (ver ``EvaluadorModelos``).
* La selección de variables se ajusta solo con los meses de entrenamiento, de modo que la métrica
  de validación sea una estimación honesta.
* No se usan ``id_cliente`` ni ``mes`` como predictoras (diciembre es un mes no visto).

Uso
---
    python pipeline.py --datos ./data --salida ./salida/submission.csv --trials 30

Dependencias: pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, optuna.
"""
from __future__ import annotations

import argparse
import gc
import json
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
from catboost import CatBoostClassifier
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
    # Columnas que se quitan de X (y sus derivadas "{col}_*"). Vacío por defecto para que los
    # scripts de verificaciones/ y validacion_modelos/ que usan Config() sigan igual; la CLI
    # excluye dias_ultima_interaccion (copia degradada de dias_ultima_transaccion en ene–nov y
    # permutación sin señal en diciembre; ver validacion_modelos/INFORME_VALIDACION.md §4.3).
    cols_excluir: List[str] = field(default_factory=list)

    # Variables con rezagos y ventanas móviles
    cols_lag: List[str] = field(default_factory=lambda: [
        "saldo_promedio", "ingresos", "dias_ultima_transaccion", "ratio_deuda_ingresos",
        "numero_productos", "visitas_web_ultimos_90_dias", "dias_ultima_interaccion"])
    lags: Tuple[int, ...] = (1, 2, 3)          # debe incluir 1 (se usa para los deltas)
    cols_rolling: List[str] = field(default_factory=lambda: [
        "saldo_promedio", "dias_ultima_transaccion", "visitas_web_ultimos_90_dias",
        "numero_productos", "dias_ultima_interaccion"])
    ventana: int = 3                            # meses (mes actual + 2 anteriores)

    # Modo de variables (P-1): "estatico" = solo columnas originales + OHE + valor actual de dias_ultima_interaccion
    # "temporal" = comportamiento actual con lags/deltas/rolling
    modo_features: str = "estatico"

    # Codificación
    umbral_ohe: int = 10                        # nunique <= umbral -> One-Hot; si no -> Target Encoding
    suavizado_te: float = 50.0                  # fuerza del prior en el Target Encoding

    # Selección
    corr_umbral: float = 0.95
    top_k: int = 80

    # Modelado
    n_folds: int = 4                      # P-3: folds walk-forward (últimos N meses de train)
    n_meses_validacion: int = 3           # legacy: mantenido por compatibilidad
    n_trials_optuna: int = 30
    n_seeds_final: int = 3
    seed: int = 42
    # Nº de árboles: cada fold se entrena sin early stopping hasta max(grid) y se mide el Gini en
    # cada punto de la rejilla; el nº se elige con la curva media de los folds (regla 1-SE).
    grid_arboles: Tuple[int, ...] = (25, 50, 75, 100, 150, 200, 300, 400, 600, 800)
    bootstrap_ic: bool = False            # legacy: el IC95% se calcula siempre
    n_bootstrap: int = 500                # réplicas del bootstrap de clientes (SE e IC95%)
    n_bootstrap_optuna: int = 100         # réplicas por trial de Optuna (solo para la regla 1-SE)


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

    Según análisis del panel (verificaciones/00_analisis_panel.py), **solo** ``dias_ultima_interaccion``
    varía mes a mes para un mismo cliente. El resto de variables son estáticas (90 %+ iguales).

    * Si ``cfg.modo_features == "estatico"``: NO crea lags/deltas/rolling; devuelve DataFrame vacío.
    * Si ``cfg.modo_features == "temporal"``: crea features temporales SOLO para las columnas que
      realmente varían en el panel (detección dinámica). Actualmente solo ``dias_ultima_interaccion``.

    * ``{col}_lag{k}``        : valor del mes ``mes-k``; NaN si el cliente no estaba ese mes.
    * ``{col}_delta1``        : valor actual - lag1 (cambio mensual).
    * ``{col}_roll{w}_mean``  : promedio de los últimos w meses disponibles (incluye el actual).
    * ``{col}_roll{w}_sum``   : suma de los últimos w meses disponibles (incluye el actual).
    * ``meses_en_ventana{w}`` : cuántos de los w meses existen (para interpretar la suma).
    """
    # P-2: En modo estático no se crean features temporales (ablación A ≈ B)
    if cfg.modo_features == "estatico":
        log.info("Modo 'estatico': sin features temporales (solo variables originales + OHE + TE)")
        return pd.DataFrame(index=df.index)

    # Modo temporal: detectar columnas dinámicas (que varían mes a mes por cliente)
    # Según verificación, solo dias_ultima_interaccion varía; detectamos programáticamente
    log.info("Modo 'temporal': detectando columnas dinámicas...")
    columnas_candidatas = list(dict.fromkeys(cfg.cols_lag + cfg.cols_rolling))
    dinamicas = []
    for col in columnas_candidatas:
        # Una columna es dinámica si tiene más de 1 valor único por cliente (en promedio)
        nunique_por_cliente = df.groupby(cfg.id_col)[col].nunique()
        if (nunique_por_cliente > 1).any():
            dinamicas.append(col)
    log.info("Columnas dinámicas detectadas: %s", dinamicas)

    if not dinamicas:
        log.warning("Ninguna columna dinámica detectada; devolviendo DataFrame vacío")
        return pd.DataFrame(index=df.index)

    w = cfg.ventana
    k_max = max(max(cfg.lags), w - 1)
    g_mes = df.groupby(cfg.id_col, sort=False)["_mes_idx"]
    # mascara[k][i] = True si la fila k posiciones atrás es exactamente k meses antes
    mascara = {k: ((df["_mes_idx"] - g_mes.shift(k)).to_numpy() == k) for k in range(1, k_max + 1)}

    nuevas: Dict[str, np.ndarray] = {}
    for col in dinamicas:
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
    id_train: np.ndarray             # id_cliente de train (para bootstrap P-4)
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
    if cfg.cols_excluir:
        quitar = [c for c in X.columns
                  if any(c == e or c.startswith(f"{e}_") for e in cfg.cols_excluir)]
        X = X.drop(columns=quitar)
        log.info("Columnas excluidas: %s", quitar)
    es_test = df["_es_test"].to_numpy()
    y = df.loc[~es_test, cfg.target].astype("int8")
    mes_train = df.loc[~es_test, cfg.mes_col].to_numpy()
    id_train = df.loc[~es_test, cfg.id_col].to_numpy()

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
    return Datos(X_train, y.reset_index(drop=True), mes_train, id_train, X_test, id_test)


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


class SelectorImportanciaPermutacion(BaseEstimator, TransformerMixin):
    """
    Selección por importancia de permutación (P-7): insesgada vs. gain/impurity.
    Entrena un LightGBM regularizado y mide caída en AUC al permutar cada feature.
    Conserva top_k features. Más lento pero correcto (no sesga hacia cardinalidad alta).
    """

    def __init__(self, top_k: int = 80, n_repeats: int = 3, seed: int = 42):
        self.top_k, self.n_repeats, self.seed = top_k, n_repeats, seed

    def fit(self, X: pd.DataFrame, y):
        # Modelo base regularizado (rápido)
        modelo = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=100,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
            random_state=self.seed, n_jobs=-1, verbose=-1)
        modelo.fit(X, y)

        # AUC baseline
        from sklearn.metrics import roc_auc_score
        y_prob = modelo.predict_proba(X)[:, 1]
        auc_base = roc_auc_score(y, y_prob)

        # Importancia por permutación
        importancias = []
        rng = np.random.default_rng(self.seed)
        for col in X.columns:
            caidas = []
            for _ in range(self.n_repeats):
                X_perm = X.copy()
                X_perm[col] = rng.permutation(X_perm[col].values)
                y_prob_perm = modelo.predict_proba(X_perm)[:, 1]
                auc_perm = roc_auc_score(y, y_prob_perm)
                caidas.append(auc_base - auc_perm)
            importancias.append(np.mean(caidas))

        self.importancias_ = pd.Series(importancias, index=X.columns).sort_values(ascending=False)
        self.cols_conservadas_ = self.importancias_.head(self.top_k).index.tolist()
        log.info("Top-10 importancia permutación:\n%s", self.importancias_.head(10).round(6).to_string())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.cols_conservadas_]


def seleccionar_features(X_fit: pd.DataFrame, y_fit: pd.Series, cfg: Config) -> List[str]:
    """Cadena: varianza cero -> correlación > 0.95 -> top-K por permutación (P-7, insesgada)."""
    n0 = X_fit.shape[1]
    s1 = SelectorVarianzaCero().fit(X_fit)
    X1 = s1.transform(X_fit)
    log.info("Varianza cero: -%d (%s)", n0 - X1.shape[1], s1.cols_constantes_)

    s2 = SelectorCorrelacion(cfg.corr_umbral, seed=cfg.seed).fit(X1)
    X2 = s2.transform(X1)
    log.info("Correlación > %.2f: -%d (%s)", cfg.corr_umbral, X1.shape[1] - X2.shape[1],
             s2.cols_eliminadas_)

    # P-7: Selector por permutación (insesgado) en lugar de gain/impurity
    s3 = SelectorImportanciaPermutacion(cfg.top_k, n_repeats=3, seed=cfg.seed).fit(X2, y_fit)
    log.info("Top-%d por importancia permutación: quedan %d de %d iniciales", cfg.top_k,
             len(s3.cols_conservadas_), n0)
    log.info("Top 10 importancias (permutación):\n%s", s3.importancias_.head(10).round(6).to_string())
    return s3.cols_conservadas_


# =============================================================================
# 4. MODELADO
# =============================================================================
# P-5: Preset regularizado fijo para LightGBM (ablación D: +0.012 Gini consistente)
PARAMS_DEFECTO: Dict[str, dict] = {
    "lightgbm": dict(learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.7,
                     subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0),
    "lightgbm_sin_regularizar": dict(learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0),
    # Árboles poco profundos: con señal débil la curva Gini-vs-árboles es una meseta ancha y el nº
    # de árboles deja de importar (= lgb_sup_sin_dui de validacion_modelos/INFORME_ROBUSTEZ.md R1-R3).
    "lightgbm_superficial": dict(learning_rate=0.03, num_leaves=8, min_child_samples=200, subsample=0.7,
                                 subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0),
    "xgboost": dict(learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8,
                    colsample_bytree=0.7, reg_lambda=1.0),
    "catboost": dict(learning_rate=0.05, depth=6, l2_leaf_reg=3.0, bootstrap_type="Bernoulli",
                     subsample=0.8, rsm=0.7),
    "random_forest": dict(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                          max_samples=0.7),
}
# El pipeline ya no usa early stopping (ver EvaluadorModelos); se mantienen porque los scripts de
# validacion_modelos/ los importan.
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
    if familia == "catboost":
        # allow_writing_files=False evita que CatBoost cree la carpeta catboost_info/ en disco
        extra = dict(early_stopping_rounds=ES_ROUNDS) if early_stopping else {}
        return CatBoostClassifier(**params, iterations=n_iter, eval_metric="AUC",
                                  random_seed=seed, thread_count=-1, verbose=False,
                                  allow_writing_files=False, **extra)
    if familia == "random_forest":
        # RandomForest no admite NaN en todas las versiones -> imputación por mediana
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(**params, random_state=seed, n_jobs=-1))
    raise ValueError(f"Familia desconocida: {familia}")


class GiniPonderado:
    """
    Gini exacto con pesos por fila: igual a ``2*roc_auc_score(y, p, sample_weight=w) - 1``
    (empates = 1/2). Ordena ``p`` una sola vez y cada evaluación con pesos nuevos es O(n), lo que
    hace barato el bootstrap de clientes.
    """

    def __init__(self, y, p):
        y = np.asarray(y, dtype=float)
        p = np.asarray(p, dtype=float)
        self.orden = np.argsort(p, kind="mergesort")
        ps = p[self.orden]
        self.ys = y[self.orden]
        self.grupo = np.r_[0, np.cumsum(ps[1:] != ps[:-1])]     # grupos de scores empatados
        self.n_grupos = int(self.grupo[-1]) + 1

    def __call__(self, w: Optional[np.ndarray] = None) -> float:
        w = np.ones(len(self.ys)) if w is None else np.asarray(w, dtype=float)[self.orden]
        wp = np.bincount(self.grupo, w * self.ys, self.n_grupos)
        wn = np.bincount(self.grupo, w * (1 - self.ys), self.n_grupos)
        neg_antes = np.cumsum(wn) - wn
        auc = np.sum(wp * (neg_antes + 0.5 * wn)) / (wp.sum() * wn.sum())
        return float(2 * auc - 1)


def conteos_bootstrap_clientes(ids: np.ndarray, n_boot: int,
                               seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap de clientes CON reemplazo. Devuelve ``(conteos, inv)``: el peso de la fila i en la
    réplica b es ``conteos[b, inv[i]]``, el nº de veces que salió su cliente. Un cliente sorteado
    dos veces pesa 2 en todas sus filas; uno no sorteado pesa 0.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(ids, return_inverse=True)
    conteos = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
                        for _ in range(n_boot)]).astype(np.int16)
    return conteos, inv


def elegir_n_1se(gini_folds: np.ndarray, boot_folds: np.ndarray) -> int:
    """
    Índice de la rejilla de árboles elegido con la regla 1-SE sobre la curva media de los folds.

    gini_folds : (K, G) Gini de cada fold en cada punto de la rejilla.
    boot_folds : (B, K, G) el mismo Gini en cada réplica bootstrap (pesos de clientes compartidos).
    Se toma el n más pequeño cuya media no esté más de 1 SE pareado por debajo de la mejor: si dos
    n no se distinguen del ruido, se prefiere el modelo más simple.
    """
    media = gini_folds.mean(axis=0)
    g_mejor = int(np.argmax(media))
    media_boot = boot_folds.mean(axis=1)                              # (B, G)
    se = (media_boot - media_boot[:, [g_mejor]]).std(axis=0, ddof=1)  # SE pareado frente al mejor
    return int(np.flatnonzero(media >= media[g_mejor] - se)[0])


def predecir_rejilla(familia: str, modelo, X: pd.DataFrame, grid: Sequence[int]) -> np.ndarray:
    """Predicciones del mismo modelo usando solo sus primeros n árboles, para cada n de ``grid``."""
    if familia == "lightgbm":
        return np.stack([modelo.predict_proba(X, num_iteration=n)[:, 1] for n in grid])
    if familia == "xgboost":
        return np.stack([modelo.predict_proba(X, iteration_range=(0, n))[:, 1] for n in grid])
    if familia == "catboost":
        return np.stack([modelo.predict_proba(X, ntree_end=n)[:, 1] for n in grid])
    if familia == "random_forest":
        return modelo.predict_proba(X)[:, 1][None, :]
    raise ValueError(f"Familia desconocida: {familia}")


class EvaluadorModelos:
    """
    Compara LightGBM, XGBoost, CatBoost y RandomForest con validación walk-forward: cada uno de
    los últimos ``n_folds`` meses de train es un fold y se entrena con todos los meses anteriores.

    Nº de árboles, sin early stopping sobre el mes evaluado:
      1. En cada fold el modelo se entrena hasta ``max(grid_arboles)`` árboles y se mide el Gini del
         mes validado en cada punto de la rejilla (predicciones por etapas del mismo modelo).
      2. Gini honesto del fold k: n se elige con la regla 1-SE sobre la curva media de los DEMÁS
         folds y se lee la curva de k en ese n, así que el objetivo del mes k no decide su propio n.
         Los demás folds incluyen meses posteriores a k; se usan solo para fijar un hiperparámetro.
      3. Nº de árboles final: regla 1-SE sobre la curva media de todos los folds.

    El resumen da el Gini medio de los folds, el peor fold y un IC95% por bootstrap de clientes con
    reemplazo. El remuestreo es el mismo para todos los modelos, así que sus diferencias son pareadas.
    """

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series,
                 mes_train: np.ndarray, id_train: np.ndarray, cfg: Config):
        self.X_train, self.y_train = X_train, y_train
        self.mes_train, self.id_train = mes_train, id_train
        self.grid = tuple(sorted(cfg.grid_arboles))
        self.n_bootstrap = max(2, cfg.n_bootstrap)
        self.n_bootstrap_optuna = max(2, cfg.n_bootstrap_optuna)
        self.seed = cfg.seed
        self.params = {k: dict(v) for k, v in PARAMS_DEFECTO.items()}
        self.meses_val = np.sort(np.unique(mes_train))[-cfg.n_folds:]
        self.mascaras_ = [(mes_train < m, mes_train == m) for m in self.meses_val]

        # Pesos bootstrap de las filas de validación de todos los folds a la vez: un cliente
        # sorteado entra con sus filas de todos los meses validados.
        es_val = np.isin(mes_train, self.meses_val)
        self.conteos_, inv = conteos_bootstrap_clientes(
            id_train[es_val], max(self.n_bootstrap, self.n_bootstrap_optuna), cfg.seed)
        pos = np.cumsum(es_val) - 1                     # posición de cada fila dentro de es_val
        self.inv_folds_ = [inv[pos[va]] for _, va in self.mascaras_]

        # Más datos en el modelo final -> algo más de árboles (filas totales / filas medias por fold)
        self.factor_datos_ = len(y_train) / np.mean([tr.sum() for tr, _ in self.mascaras_])
        self.resultados_: Dict[str, dict] = {}
        self.val_predictions_: List[pd.DataFrame] = []  # P-11: predicciones de validación para auditoría

    # ---- utilidades ---------------------------------------------------------
    @staticmethod
    def metricas(y_true, y_prob) -> Tuple[float, float]:
        """Devuelve (AUC, Gini) con Gini = 2*AUC - 1."""
        auc = roc_auc_score(y_true, y_prob)
        return auc, 2 * auc - 1

    def _curvas(self, familia: str, params: dict, n_boot: int):
        """
        Entrena un modelo por fold y devuelve ``(grid, preds, gini, boot)``:
        preds[k] (G, n_k) predicciones del fold k en cada n; gini (K, G); boot (B, K, G).
        """
        grid = (int(params["n_estimators"]),) if familia == "random_forest" else self.grid
        K, G = len(self.mascaras_), len(grid)
        preds, gini, boot = [], np.empty((K, G)), np.empty((n_boot, K, G))
        for k, (tr, va) in enumerate(self.mascaras_):
            m = construir_modelo(familia, params, max(grid), self.seed)
            m.fit(self.X_train.loc[tr], self.y_train.loc[tr])
            p = predecir_rejilla(familia, m, self.X_train.loc[va], grid)
            y_va = self.y_train.loc[va].to_numpy()
            pesos = self.conteos_[:n_boot, self.inv_folds_[k]]
            for g in range(G):
                f = GiniPonderado(y_va, p[g])
                gini[k, g] = f()
                boot[:, k, g] = [f(w) for w in pesos]
            preds.append(p)
            del m
        return grid, preds, gini, boot

    def _evaluar_modelo_cv(self, familia: str, params: dict, nombre: str,
                           registrar: bool = True, n_boot: Optional[int] = None) -> dict:
        """Walk-forward con elección honesta del nº de árboles (ver docstring de la clase)."""
        n_boot = n_boot or self.n_bootstrap
        grid, preds, gini, boot = self._curvas(familia, params, n_boot)
        K = len(self.meses_val)
        if K == 1:
            log.warning("Con 1 fold, el nº de árboles se elige con el mismo mes evaluado: "
                        "el Gini de %s es optimista", nombre)
        g_fold = []
        for k in range(K):
            otros = [j for j in range(K) if j != k] or [k]
            g_fold.append(elegir_n_1se(gini[otros], boot[:, otros]))
        gini_fold = np.array([gini[k, g] for k, g in enumerate(g_fold)])
        gini_boot = np.mean([boot[:, k, g] for k, g in enumerate(g_fold)], axis=0)  # (B,)
        g_final = elegir_n_1se(gini, boot)

        res = dict(familia=familia, params=params,
                   auc=float(np.mean((gini_fold + 1) / 2)), gini=float(gini_fold.mean()),
                   gini_peor_fold=float(gini_fold.min()),
                   gini_ic_lower=float(np.percentile(gini_boot, 2.5)),
                   gini_ic_upper=float(np.percentile(gini_boot, 97.5)),
                   mejor_iter=int(grid[g_final]),
                   arboles_folds=[int(grid[g]) for g in g_fold],
                   gini_folds=gini_fold.tolist(), gini_boot=gini_boot,
                   curva=pd.Series(gini.mean(axis=0), index=list(grid)))
        if not registrar:
            return res

        for k, mes in enumerate(self.meses_val):
            log.info("  %-24s fold %d (mes %d): Gini=%.4f con %d árboles",
                     nombre, k + 1, mes, gini_fold[k], grid[g_fold[k]])
            va = self.mascaras_[k][1]
            self.val_predictions_.append(pd.DataFrame({
                "id_cliente": self.id_train[va], "mes": self.mes_train[va],
                "y_true": self.y_train.loc[va].to_numpy(), "y_prob": preds[k][g_fold[k]],
                "modelo": nombre, "fold": k + 1}))
        log.info("%-24s Gini medio=%.4f [IC95 %.4f; %.4f] | peor fold=%.4f | árboles finales=%d",
                 nombre, res["gini"], res["gini_ic_lower"], res["gini_ic_upper"],
                 res["gini_peor_fold"], res["mejor_iter"])
        self.resultados_[nombre] = res
        return res

    # ---- entrenadores ---------------------------------------------------------
    def entrenar_lightgbm(self, params: Optional[dict] = None, nombre: str = "LightGBM") -> dict:
        return self._evaluar_modelo_cv("lightgbm", params or self.params["lightgbm"], nombre)

    def entrenar_xgboost(self, params: Optional[dict] = None, nombre: str = "XGBoost") -> dict:
        return self._evaluar_modelo_cv("xgboost", params or self.params["xgboost"], nombre)

    def entrenar_catboost(self, params: Optional[dict] = None, nombre: str = "CatBoost") -> dict:
        return self._evaluar_modelo_cv("catboost", params or self.params["catboost"], nombre)

    def entrenar_random_forest(self, params: Optional[dict] = None,
                               nombre: str = "RandomForest") -> dict:
        return self._evaluar_modelo_cv("random_forest", params or self.params["random_forest"],
                                       nombre)

    # ---- Optuna -------------------------------------------------------------
    def optimizar_lightgbm(self, n_trials: int = 30, log_path: Optional[Path] = None) -> dict:
        """Busca hiperparámetros de LightGBM maximizando el Gini medio honesto del walk-forward.
        Los trials no se registran en ``resultados_``. El Gini de "LightGBM_Optuna" sigue siendo
        optimista: la búsqueda elige entre muchos candidatos con los mismos folds.
        P-12: Logging estructurado JSONL de trials de Optuna.
        """
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
            res = self._evaluar_modelo_cv("lightgbm", p, "Optuna_Trial", registrar=False,
                                          n_boot=self.n_bootstrap_optuna)

            # P-12: Log JSONL
            if log_path:
                log_entry = {
                    "trial": trial.number,
                    "params": p,
                    "auc_mean": res["auc"],
                    "gini_mean": res["gini"],
                    "timestamp": time.time()
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

            return res["gini"]

        # P-12: Callback para logging automático
        def log_callback(study: optuna.Study, trial: optuna.Trial):
            if log_path:
                log_entry = {
                    "trial": trial.number,
                    "params": trial.params,
                    "value": trial.value,
                    "state": trial.state.name,
                    "timestamp": time.time()
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

        estudio = optuna.create_study(direction="maximize",
                                      sampler=optuna.samplers.TPESampler(seed=self.seed))
        callbacks = [log_callback] if log_path else None
        estudio.optimize(objetivo, n_trials=n_trials, show_progress_bar=False, callbacks=callbacks)
        mejor = dict(estudio.best_params, subsample_freq=1)
        log.info("Optuna: mejor Gini medio CV=%.4f en %d trials", estudio.best_value, n_trials)
        return mejor

    # ---- orquestación -------------------------------------------------------
    def evaluar_todos(self, n_trials_optuna: int = 0) -> pd.DataFrame:
        """Evalúa todos los modelos (y Optuna si n_trials_optuna > 0) con walk-forward CV.
        P-10: Comparación obligatoria LightGBM_regularizado vs LightGBM_sin_regularizar.
        """
        self.entrenar_lightgbm(nombre="LightGBM_Regularizado")  # usa params["lightgbm"] (regularizado)
        self.entrenar_lightgbm(self.params["lightgbm_sin_regularizar"], nombre="LightGBM_Sin_Regularizar")
        self.entrenar_lightgbm(self.params["lightgbm_superficial"], nombre="LightGBM_Superficial")
        self.entrenar_xgboost()
        self.entrenar_catboost()
        self.entrenar_random_forest()
        if n_trials_optuna > 0:
            log_path = Path("optuna_trials.jsonl")  # P-12
            mejor = self.optimizar_lightgbm(n_trials_optuna, log_path=log_path)
            self.entrenar_lightgbm(mejor, nombre="LightGBM_Optuna")
        return self.resumen()

    def resumen(self) -> pd.DataFrame:
        """
        Tabla por modelo, ordenada por Gini medio de los folds: AUC y Gini medios, peor fold, IC95%,
        nº de árboles del modelo final (``mejor_iter``), árboles por fold y Gini de cada mes.
        """
        filas = {}
        for nombre, r in self.resultados_.items():
            fila = {k: r[k] for k in ("auc", "gini", "gini_peor_fold", "gini_ic_lower",
                                      "gini_ic_upper", "mejor_iter")}
            fila["arboles_folds"] = "/".join(map(str, r["arboles_folds"]))
            fila.update({f"gini_{mes}": g for mes, g in zip(self.meses_val, r["gini_folds"])})
            filas[nombre] = fila
        tabla = pd.DataFrame.from_dict(filas, orient="index")
        return tabla.sort_values("gini", ascending=False)


# =============================================================================
# 5. ENTRENAMIENTO FINAL Y SUBMISSION
# =============================================================================
def entrenar_y_predecir_final(familia: str, params: dict, mejor_iter: int,
                              X_all: pd.DataFrame, y_all: pd.Series, X_test: pd.DataFrame,
                              n_seeds: int, seed: int, factor_datos: float = 1.1) -> np.ndarray:
    """
    Reentrena el modelo ganador con el 100 % de train y predice test.

    * Boosting: n_iter = factor_datos * mejor_iter, donde mejor_iter sale de la regla 1-SE sobre
      la curva media de los folds y factor_datos = filas totales / filas medias de entrenamiento
      por fold (``EvaluadorModelos.factor_datos_``; más datos admiten algo más de árboles).
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

    # 2) Validación walk-forward (P-3): n_folds meses finales, cada fold valida un mes distinto
    meses = np.sort(np.unique(datos.mes_train))
    if not 1 <= cfg.n_folds < len(meses):
        raise ValueError(f"n_folds debe estar entre 1 y {len(meses) - 1}")
    log.info("Walk-forward CV: %d folds, meses de validación: %s", cfg.n_folds, meses[-cfg.n_folds:])

    # 3) Selección de variables (SOLO con datos de entrenamiento del primer fold = todos menos último mes)
    # Usamos una selección preliminar con el 75% más antiguo para no hacer leakage
    mes_corte_seleccion = int(meses[-cfg.n_folds])
    es_tr_sel = datos.mes_train < mes_corte_seleccion
    features = seleccionar_features(datos.X_train.loc[es_tr_sel], datos.y_train.loc[es_tr_sel], cfg)

    # 4) Comparación de modelos con walk-forward CV
    ev = EvaluadorModelos(
        datos.X_train[features], datos.y_train, datos.mes_train, datos.id_train, cfg)
    tabla = ev.evaluar_todos(cfg.n_trials_optuna)
    log.info("Resultados en walk-forward CV (%d folds):\n%s", cfg.n_folds, tabla.round(4).to_string())

    # 5) Modelo ganador -> entrenamiento con el 100 % de train -> predicción de test
    ganador = tabla.index[0]
    r = ev.resultados_[ganador]
    log.info("Modelo ganador: %s (Gini medio=%.4f | peor fold=%.4f | %d árboles en CV x %.2f)",
             ganador, r["gini"], r["gini_peor_fold"], r["mejor_iter"], ev.factor_datos_)
    pred = entrenar_y_predecir_final(
        r["familia"], r["params"], int(r["mejor_iter"]),
        datos.X_train[features], datos.y_train, datos.X_test[features],
        cfg.n_seeds_final, cfg.seed, ev.factor_datos_)
    exportar_submission(datos.id_test, pred, cfg)

    # P-11: Guardar predicciones de validación para auditoría
    if ev.val_predictions_:
        val_pred_df = pd.concat(ev.val_predictions_, ignore_index=True)
        carpeta = cfg.ruta_salida.parent
        val_pred_df.to_csv(carpeta / "val_predictions.csv", index=False)
        log.info("Predicciones de validación guardadas en %s (%d filas)", carpeta / "val_predictions.csv", len(val_pred_df))

    # Artefactos auxiliares para trazabilidad
    carpeta = cfg.ruta_salida.parent
    tabla.to_csv(carpeta / "resultados_validacion_DF_WCB.csv")
    (carpeta / "features_seleccionadas_DF_WCB.txt").write_text("\n".join(features), encoding="utf-8")
    # Réplicas bootstrap reales del Gini medio del ganador (fig9) y curvas Gini-vs-árboles
    pd.DataFrame({"gini_bootstrap": r["gini_boot"]}).to_csv(carpeta / "bootstrap_gini.csv", index=False)
    curvas = pd.concat({n: res["curva"] for n, res in ev.resultados_.items()
                        if res["familia"] != "random_forest"}, names=["modelo", "n_arboles"])
    curvas.rename("gini_medio_folds").to_csv(carpeta / "curvas_arboles_DF_WCB.csv")
    log.info("Pipeline completo en %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline de propensión de conversión (Gini)")
    ap.add_argument("--datos", type=Path, default=Path("."), help="carpeta con train/test/sample")
    ap.add_argument("--salida", type=Path, default=Path("submission_DF_WCB.csv"))
    ap.add_argument("--trials", type=int, default=30, help="trials de Optuna (0 = sin Optuna)")
    ap.add_argument("--top-k", type=int, default=80)
    ap.add_argument("--n-folds", type=int, default=4,
                    help="nº de folds walk-forward (últimos N meses de train, 1 mes por fold)")
    ap.add_argument("--meses-val", type=int, default=3,
                    help="nº de últimos meses de train usados para validar (3 ≈ 73/27) [legacy]")
    ap.add_argument("--umbral-ohe", type=int, default=10,
                    help="nunique <= umbral -> One-Hot; si no -> Target Encoding temporal")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--modo-features", choices=["estatico", "temporal"], default="estatico",
                    help="Modo de variables: 'estatico' (solo originales + OHE + dias_ultima_interaccion actual) o 'temporal' (lags/deltas/rolling)")
    ap.add_argument("--bootstrap-ic", action="store_true", default=False,
                    help="[legacy, sin efecto] el IC95%% por bootstrap de clientes se calcula siempre")
    ap.add_argument("--n-bootstrap", type=int, default=500,
                    help="réplicas del bootstrap de clientes para SE (regla 1-SE) e IC95%% (default: 500)")
    ap.add_argument("--grid-arboles", type=int, nargs="+", default=list(Config.grid_arboles),
                    help="rejilla de nº de árboles donde se mide la curva Gini (máx. = árboles por fold)")
    ap.add_argument("--excluir", nargs="*", default=["dias_ultima_interaccion"],
                    help="columnas a quitar de X junto con sus derivadas (vacío = no quitar nada). "
                         "En modo 'temporal', quitar dias_ultima_interaccion elimina también sus lags")
    a = ap.parse_args()
    main(Config(ruta_datos=a.datos, ruta_salida=a.salida, n_trials_optuna=a.trials,
                top_k=a.top_k, umbral_ohe=a.umbral_ohe, n_seeds_final=a.n_seeds, seed=a.semilla,
                n_folds=a.n_folds, n_meses_validacion=a.meses_val, modo_features=a.modo_features,
                bootstrap_ic=a.bootstrap_ic, n_bootstrap=a.n_bootstrap,
                grid_arboles=tuple(a.grid_arboles), cols_excluir=list(a.excluir)))
