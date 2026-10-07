# -*- coding: utf-8 -*-
"""
11. R2: curva Gini vs n de arboles (criterios_robustez.py, R2).
Por fold M y semilla (42-44): entrenar con < M hasta 1000 arboles; Gini en M usando los primeros n
arboles (n en rejilla log). G(n) = media de los 4 folds (prediccion media de 3 semillas).
Meseta = {n : G(n) >= G(n*) - SE(n*)}; regla 1-SE; robusto si max/min de la meseta >= 3.
Secundario: meseta con SE pareado de G(n) - G(n*). Bootstrap de clientes conjunto (500 replicas).
Salidas: robustez_r2_curva.csv, robustez_r2_resumen.csv, figuras/fig11_r2_num_arboles.png
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

from criterios_robustez import CRITERIOS
from robustez_comun import (DIR_FIG, DIR_RES, MESES_FOLD, SEMILLAS, GiniRapido, cargar_todo,
                            definir_candidatos, fit_fijo, gini, pesos_bootstrap_clientes)

C = CRITERIOS["R2"]
GRID = list(C["rejilla"]); NMAX = max(GRID); NB = 500


def predecir_n(m, fam, X, n):
    if fam == "lightgbm":
        return m.predict_proba(X, num_iteration=n)[:, 1]
    return m.predict_proba(X, ntree_end=n)[:, 1]


def main():
    t0 = time.time()
    wcb, d, _, _ = cargar_todo()
    X, y, mes, ids = d.X_train, d.y_train.to_numpy(), d.mes_train, d.id_train
    specs = definir_candidatos(wcb, X.columns)
    it = pd.read_csv(DIR_RES / "robustez_r1_iteraciones.csv")
    val = np.isin(mes, MESES_FOLD); pos = np.where(val)[0]
    folds = [np.searchsorted(pos, np.where(mes == M)[0]) for M in MESES_FOLD]
    yv, idv = y[val], ids[val]
    curva, resumen = [], []
    for nombre in C["modelos"]:
        sp = specs[nombre]; fam = sp["familia"]
        P = np.zeros((len(GRID), len(yv)))
        for M, f in zip(MESES_FOLD, folds):
            trm = mes < M; Xv = X.loc[mes == M, sp["cols"]]
            for s in SEMILLAS:
                m = fit_fijo(wcb, sp, X.loc[trm, sp["cols"]], y[trm], NMAX, s)
                for k, n in enumerate(GRID):
                    P[k, f] += predecir_n(m, fam, Xv, n) / len(SEMILLAS)
            print(f"{nombre} {M} ({time.time()-t0:.0f}s)", flush=True)
        G = np.array([[gini(yv[f], P[k, f]) for f in folds] for k in range(len(GRID))])
        Gm = G.mean(axis=1)
        gr = [[GiniRapido(yv[f], P[k, f]) for f in folds] for k in range(len(GRID))]
        Gb = np.empty((NB, len(GRID)))
        for b, w in enumerate(pesos_bootstrap_clientes(idv, NB, seed=42)):
            Gb[b] = [np.mean([gr[k][j](w[f]) for j, f in enumerate(folds)]) for k in range(len(GRID))]
        ks = int(np.argmax(Gm)); se_star = Gb[:, ks].std(ddof=1)
        se_par = np.array([(Gb[:, k] - Gb[:, ks]).std(ddof=1) for k in range(len(GRID))])
        mes_ = [n for k, n in enumerate(GRID) if Gm[k] >= Gm[ks] - se_star]
        mes_p = [n for k, n in enumerate(GRID) if Gm[k] >= Gm[ks] - se_par[k]]
        contig = all(GRID[k] in mes_ for k in range(GRID.index(min(mes_)), GRID.index(max(mes_)) + 1))
        n1 = min(mes_); k1 = GRID.index(n1)
        ri = it[it.modelo == nombre]
        for k, n in enumerate(GRID):
            curva.append(dict(modelo=nombre, n_arboles=n, gini_medio=Gm[k], se_boot=Gb[:, k].std(ddof=1),
                              se_pareado_vs_nstar=se_par[k], **{f"gini_{M}": G[k, j] for j, M in enumerate(MESES_FOLD)}))
        resumen.append(dict(modelo=nombre, n_star=GRID[ks], gini_n_star=Gm[ks], se_n_star=se_star,
                            meseta_min=min(mes_), meseta_max=max(mes_), ratio_meseta=max(mes_) / min(mes_),
                            meseta_contigua=contig, n_1se=n1, gini_n_1se=Gm[k1],
                            robusto=max(mes_) / min(mes_) >= 3,
                            meseta_pareada_min=min(mes_p), meseta_pareada_max=max(mes_p),
                            ratio_meseta_pareada=max(mes_p) / min(mes_p),
                            gini_1000=Gm[-1], perdida_1000_vs_nstar=Gm[ks] - Gm[-1],
                            n_iter_honesto_folds=str(list(ri.n_iter)), gini_honesto_r1=ri.gini_hon_3sem.mean()))
        print(resumen[-1], flush=True)
    pd.DataFrame(curva).to_csv(DIR_RES / "robustez_r2_curva.csv", index=False)
    rs = pd.DataFrame(resumen); rs.to_csv(DIR_RES / "robustez_r2_resumen.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5)); cu = pd.DataFrame(curva)
    for nombre, g in cu.groupby("modelo", sort=False):
        r = rs[rs.modelo == nombre].iloc[0]
        l, = ax.plot(g.n_arboles, g.gini_medio, "o-", ms=3, label=f"{nombre} (meseta {r.meseta_min}-{r.meseta_max})")
        ax.fill_between(g.n_arboles, g.gini_medio - g.se_boot, g.gini_medio + g.se_boot, alpha=.08, color=l.get_color())
        ax.axvspan(r.meseta_min, r.meseta_max, ymin=0, ymax=0.03, color=l.get_color(), alpha=.6)
    ax.set_xscale("log"); ax.set_xlabel("n de arboles"); ax.set_ylabel("Gini medio ago-nov (3 semillas)")
    ax.set_title("R2 - Gini vs n de arboles (banda = +-1 SE bootstrap)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(DIR_FIG / "fig11_r2_num_arboles.png", dpi=130); plt.close(fig)
    print(f"Listo en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
