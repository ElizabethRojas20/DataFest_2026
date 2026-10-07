# -*- coding: utf-8 -*-
"""
12. R3: estabilidad del Gini en una vecindad de hiperparametros (criterios_robustez.py, R3).
lgb_sup_sin_dui: centro (num_leaves=8, min_child_samples=200, reg_lambda=10, lr=0,03);
  vecindad un-factor-cada-vez (x0,5 / x2) = 9 configs + 16 esquinas del factorial 2^4 (extendida);
  semillas 42-44.
catboost_sin_dui: centro (depth=6, l2=3, lr=0,05); depth {4,8}, l2 {1,10}, lr {0,025; 0,1}
  un-factor-cada-vez = 7 configs; semilla 42 (coste).
Walk-forward honesto completo. Robusto si max-min del Gini medio (vecindad OAT) <= SE_REF.
Delta frente al centro con SE pareado (bootstrap de clientes, 300 replicas).
Salidas: robustez_r3_hiperparametros.csv, robustez_r3_resumen.csv, figuras/fig12_r3_hiperparametros.png
"""
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robustez_comun import (DIR_FIG, DIR_RES, MESES_FOLD, GiniRapido, cargar_todo, definir_candidatos,
                            fit_honesto, gini, leer_se_ref, pesos_bootstrap_clientes, pred)

NB = 300


def configs():
    cl = dict(num_leaves=8, min_child_samples=200, reg_lambda=10.0, learning_rate=0.03)
    alt = dict(num_leaves=(4, 16), min_child_samples=(100, 400), reg_lambda=(3.0, 30.0), learning_rate=(0.015, 0.06))
    out = [("lgb_sup_sin_dui", "centro", cl)]
    for k, (lo, hi) in alt.items():
        for v, t in ((lo, "x0.5"), (hi, "x2")):
            c = dict(cl); c[k] = v; out.append(("lgb_sup_sin_dui", f"oat_{k}_{t}", c))
    for combo in itertools.product(*[(0, 1)] * 4):
        c = {k: alt[k][b] for k, b in zip(alt, combo)}
        out.append(("lgb_sup_sin_dui", "esquina_" + "".join(map(str, combo)), c))
    cc = dict(depth=6, l2_leaf_reg=3.0, learning_rate=0.05)
    altc = dict(depth=(4, 8), l2_leaf_reg=(1.0, 10.0), learning_rate=(0.025, 0.1))
    out.append(("catboost_sin_dui", "centro", cc))
    for k, (lo, hi) in altc.items():
        for v, t in ((lo, "bajo"), (hi, "alto")):
            c = dict(cc); c[k] = v; out.append(("catboost_sin_dui", f"oat_{k}_{t}", c))
    return out


def main():
    t0 = time.time()
    wcb, d, _, _ = cargar_todo()
    X, y, mes, ids = d.X_train, d.y_train.to_numpy(), d.mes_train, d.id_train
    base = definir_candidatos(wcb, X.columns)
    se_ref = leer_se_ref()
    val = np.isin(mes, MESES_FOLD); pos = np.where(val)[0]
    folds = [np.searchsorted(pos, np.where(mes == M)[0]) for M in MESES_FOLD]
    yv, idv = y[val], ids[val]
    filas, preds = [], {}
    for modelo, etiqueta, cfg in configs():
        sp = dict(base[modelo]); sp["params"] = {**base[modelo]["params"], **cfg}
        seeds = (42, 43, 44) if modelo.startswith("lgb") else (42,)
        p = np.zeros(len(yv)); its = []
        for M, f in zip(MESES_FOLD, folds):
            mods, info = fit_honesto(wcb, sp, X, y, mes, M, seeds=seeds)
            Xv = X.loc[mes == M, sp["cols"]]
            p[f] = np.mean([pred(m, Xv) for m in mods], axis=0); its.append(info["n_iter"])
        preds[(modelo, etiqueta)] = p
        G = [gini(yv[f], p[f]) for f in folds]
        filas.append(dict(modelo=modelo, config=etiqueta, **cfg, gini_medio=np.mean(G),
                          **{f"gini_{M}": g for M, g in zip(MESES_FOLD, G)}, n_iter_folds=str(its),
                          vecindad="extendida" if etiqueta.startswith("esquina") else "oat"))
        print(f"{modelo} {etiqueta:28s} G={np.mean(G):.4f} n_iter={its} ({time.time()-t0:.0f}s)", flush=True)
    res = pd.DataFrame(filas)
    # Delta vs centro con SE pareado (bootstrap de clientes conjunto)
    gr = {k: [GiniRapido(yv[f], v[f]) for f in folds] for k, v in preds.items()}
    acc = {k: [] for k in preds}
    for w in pesos_bootstrap_clientes(idv, NB, seed=42):
        for k in preds:
            acc[k].append(np.mean([g(w[f]) for g, f in zip(gr[k], folds)]))
    res["delta_vs_centro"] = np.nan; res["se_delta"] = np.nan
    for i, r in res.iterrows():
        c = np.array(acc[(r.modelo, "centro")]); v = np.array(acc[(r.modelo, r.config)])
        res.loc[i, "delta_vs_centro"] = r.gini_medio - res[(res.modelo == r.modelo) & (res.config == "centro")].gini_medio.iloc[0]
        res.loc[i, "se_delta"] = (v - c).std(ddof=1)
    res.to_csv(DIR_RES / "robustez_r3_hiperparametros.csv", index=False)
    resumen = []
    for modelo, g in res.groupby("modelo", sort=False):
        oat = g[g.vecindad == "oat"]
        rng_oat = oat.gini_medio.max() - oat.gini_medio.min()
        r = dict(modelo=modelo, n_oat=len(oat), gini_centro=g[g.config == "centro"].gini_medio.iloc[0],
                 gini_min_oat=oat.gini_medio.min(), gini_max_oat=oat.gini_medio.max(), rango_oat=rng_oat,
                 config_max_oat=oat.loc[oat.gini_medio.idxmax(), "config"],
                 config_min_oat=oat.loc[oat.gini_medio.idxmin(), "config"], se_ref=se_ref,
                 robusto=rng_oat <= se_ref)
        ext = g[g.vecindad == "extendida"]
        if len(ext):
            r.update(rango_extendida=g.gini_medio.max() - g.gini_medio.min(),
                     config_max_total=g.loc[g.gini_medio.idxmax(), "config"], gini_max_total=g.gini_medio.max())
        resumen.append(r)
    rs = pd.DataFrame(resumen); rs.to_csv(DIR_RES / "robustez_r3_resumen.csv", index=False)
    print(res.round(4).to_string()); print(rs.round(4).to_string())
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for k, (modelo, g) in enumerate(res.groupby("modelo", sort=False)):
        g = g.sort_values("gini_medio"); cen = g[g.config == "centro"].gini_medio.iloc[0]
        col = ["tab:red" if c == "centro" else ("tab:blue" if v == "oat" else "tab:grey") for c, v in zip(g.config, g.vecindad)]
        ax[k].barh(g.config, g.gini_medio - cen, xerr=g.se_delta, color=col, alpha=.8)
        ax[k].axvline(-se_ref, ls=":", color="k"); ax[k].axvline(se_ref, ls=":", color="k", label="+-SE_REF")
        ax[k].set_title(f"R3 {modelo}: Gini - centro ({cen:.4f}); azul OAT, gris esquinas"); ax[k].tick_params(labelsize=7)
        ax[k].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(DIR_FIG / "fig12_r3_hiperparametros.png", dpi=130); plt.close(fig)
    print(f"Listo en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
