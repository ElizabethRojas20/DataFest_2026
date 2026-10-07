# -*- coding: utf-8 -*-
"""
18. R9: curva de aprendizaje submuestreando CLIENTES de train (criterios_robustez.py, R9).
Fracciones 25/50/75/100 %; por replica r cada cliente recibe U ~ U(0,1) y entra si U < f
(subconjuntos anidados). Replicas: 3 (LightGBM) / 2 (CatBoost) para f < 1. Walk-forward honesto
(ES en M-1 dentro de la submuestra), semilla 42; se evalua en el fold M completo.
Saturado si G(100 %) - G(75 %) < SE_REF; p bootstrap (300) de la ganancia y Holm sobre los 3 modelos.
Ley potencia G(n) = G_inf - a n^-b (n = filas medias de train) si converge con b > 0.
Salidas: robustez_r9_curva.csv, robustez_r9_resumen.csv, figuras/fig18_r9_curva_aprendizaje.png
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from criterios_robustez import CRITERIOS
from robustez_comun import (DIR_FIG, DIR_RES, MESES_FOLD, GiniRapido, cargar_todo, definir_candidatos,
                            fit_honesto, gini, holm, ic, leer_se_ref, p_boot_centrado,
                            pesos_bootstrap_clientes, pred)

C = CRITERIOS["R9"]; NB = 300


def ley(n, ginf, a, b):
    return ginf - a * np.power(n, -b)


def main():
    t0 = time.time()
    wcb, d, _, _ = cargar_todo()
    X, y, mes, ids = d.X_train, d.y_train.to_numpy(), d.mes_train, d.id_train
    specs = definir_candidatos(wcb, X.columns); se_ref = leer_se_ref()
    val = np.isin(mes, MESES_FOLD); pos = np.where(val)[0]
    folds = [np.searchsorted(pos, np.where(mes == M)[0]) for M in MESES_FOLD]
    yv, idv = y[val], ids[val]
    uniq, inv = np.unique(ids, return_inverse=True)
    pesos = list(pesos_bootstrap_clientes(idv, NB, seed=42))
    curva, res = [], []
    for nombre in C["modelos"]:
        sp = specs[nombre]; nrep = 3 if sp["familia"] == "lightgbm" else 2
        preds = {}
        for f in C["fracciones"]:
            for r in (range(nrep) if f < 1 else [0]):
                U = np.random.default_rng(900 + r).random(len(uniq))[inv]
                mask = U < f if f < 1 else np.ones(len(y), bool)
                p = np.zeros(len(yv)); filas = []
                for M, fo in zip(MESES_FOLD, folds):
                    mods, info = fit_honesto(wcb, sp, X, y, mes, M, seeds=(42,), filas_mask=mask)
                    p[fo] = pred(mods[0], X.loc[mes == M, sp["cols"]]); filas.append(info["filas_M"])
                G = [gini(yv[fo], p[fo]) for fo in folds]
                preds[(f, r)] = p
                curva.append(dict(modelo=nombre, fraccion=f, replica=r, filas_train_media=np.mean(filas),
                                  gini_medio=np.mean(G), **{f"gini_{M}": g for M, g in zip(MESES_FOLD, G)}))
                print(f"{nombre} f={f} r={r} G={np.mean(G):.4f} ({time.time()-t0:.0f}s)", flush=True)
        cu = pd.DataFrame([c for c in curva if c["modelo"] == nombre])
        gm = cu.groupby("fraccion").gini_medio.mean(); nm = cu.groupby("fraccion").filas_train_media.mean()
        gan = gm[1.0] - gm[0.75]
        r100 = [GiniRapido(yv[fo], preds[(1.0, 0)][fo]) for fo in folds]
        r75 = {r: [GiniRapido(yv[fo], preds[(0.75, r)][fo]) for fo in folds] for r in range(nrep)}
        gb = np.array([np.mean([g(w[fo]) for g, fo in zip(r100, folds)]) -
                       np.mean([np.mean([g(w[fo]) for g, fo in zip(r75[r], folds)]) for r in range(nrep)])
                       for w in pesos])
        rr = dict(modelo=nombre, **{f"gini_{int(f*100)}": gm[f] for f in gm.index},
                  ganancia_75_100=gan, ganancia_se=gb.std(ddof=1), ganancia_ic95_inf=ic(gb)[0],
                  ganancia_ic95_sup=ic(gb)[1], p_boot=p_boot_centrado(gb, gan),
                  ganancia_50_100=gm[1.0] - gm[0.5], ganancia_25_100=gm[1.0] - gm[0.25],
                  se_ref=se_ref, saturado=gan < se_ref)
        try:
            par, cov = curve_fit(ley, nm.to_numpy(), gm.to_numpy(), p0=[gm.max() + 0.01, 1.0, 0.5],
                                 bounds=([0, 0, 1e-3], [1, 1e6, 5]), maxfev=20000)
            rr.update(ley_ginf=par[0], ley_a=par[1], ley_b=par[2], ley_converge=True,
                      ley_ginf_se=float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan,
                      ley_pred_doble=ley(2 * nm[1.0], *par) - gm[1.0])
        except Exception as e:  # noqa: BLE001
            rr.update(ley_converge=False, ley_error=str(e)[:80])
        res.append(rr); print(rr, flush=True)
    rs = pd.DataFrame(res); rs["p_holm"] = holm(rs.p_boot.to_numpy())
    rs.to_csv(DIR_RES / "robustez_r9_resumen.csv", index=False)
    cu = pd.DataFrame(curva); cu.to_csv(DIR_RES / "robustez_r9_curva.csv", index=False)
    pd.set_option("display.width", 250); print(rs.round(4).to_string())
    fig, ax = plt.subplots(figsize=(8, 5))
    for nombre, g in cu.groupby("modelo", sort=False):
        m = g.groupby("fraccion").agg(n=("filas_train_media", "mean"), G=("gini_medio", "mean"),
                                      lo=("gini_medio", "min"), hi=("gini_medio", "max"))
        l, = ax.plot(m.n, m.G, "o-", label=nombre)
        ax.vlines(m.n, m.lo, m.hi, color=l.get_color(), alpha=.5)
        r = rs[rs.modelo == nombre].iloc[0]
        if r.get("ley_converge", False):
            nn = np.linspace(m.n.min(), 2 * m.n.max(), 100)
            ax.plot(nn, ley(nn, r.ley_ginf, r.ley_a, r.ley_b), ":", color=l.get_color())
    ax.set_xlabel("filas de train (media de los 4 folds)"); ax.set_ylabel("Gini medio ago-nov")
    ax.set_title("R9 - curva de aprendizaje por clientes (punteado: ley potencia)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(DIR_FIG / "fig18_r9_curva_aprendizaje.png", dpi=130); plt.close(fig)
    print(f"Listo en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
