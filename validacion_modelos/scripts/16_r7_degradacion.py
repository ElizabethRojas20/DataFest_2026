# -*- coding: utf-8 -*-
"""
16. R7: degradacion temporal (criterios_robustez.py, R7).
Origenes k = mar..oct 2026: ES honesto (entrenar < k, validar en k), reentreno con <= k y
n_iter = best * filas(<=k)/filas(<k); evaluar en k+h, h = 1..4, k+h <= nov.
Semillas 42-44 en LightGBM; 42 en CatBoost y EBM.
Estadistico: beta de OLS en G_kh = alpha_k + beta*h (efectos fijos de origen); IC95 y p bootstrap
de clientes (1000) conjunto sobre todas las celdas; Holm sobre los 5 modelos.
Secundario: efectos fijos de mes evaluado. Robusto si IC95 de beta incluye 0 (o beta > 0).
Salidas: robustez_r7_celdas.csv, robustez_r7_resumen.csv, figuras/fig16_r7_degradacion.png
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

from robustez_comun import (CANDIDATOS, DIR_FIG, DIR_RES, GiniRapido, cargar_todo, definir_candidatos,
                            fit_honesto, gini, holm, ic, p_boot_centrado, pesos_bootstrap_clientes, pred)

NB = 1000; HMAX = 4


def beta_fe(G, k_idx, h, fe):
    """OLS de G sobre h con efectos fijos (fe = vector de etiquetas de grupo). Devuelve beta."""
    grupos = np.unique(fe)
    D = np.column_stack([h] + [(fe == g).astype(float) for g in grupos])
    return np.linalg.lstsq(D, G, rcond=None)[0][0]


def main():
    t0 = time.time()
    wcb, d, _, _ = cargar_todo()
    X, y, mes, ids = d.X_train, d.y_train.to_numpy(), d.mes_train, d.id_train
    specs = definir_candidatos(wcb, X.columns)
    meses = list(np.sort(np.unique(mes)))
    origenes = [m for m in meses if 202603 <= m <= 202610]
    ev = mes >= meses[meses.index(origenes[0]) + 1]; pos = np.where(ev)[0]; yv, idv = y[ev], ids[ev]
    celdas = []
    for k in origenes:
        ik = meses.index(k)
        for h in range(1, HMAX + 1):
            if ik + h < len(meses):
                celdas.append((k, h, meses[ik + h]))
    kk = np.array([c[0] for c in celdas]); hh = np.array([c[1] for c in celdas], float)
    tt = np.array([c[2] for c in celdas])
    filas_c = {t: np.searchsorted(pos, np.where(mes == t)[0]) for t in set(tt)}
    pesos = list(pesos_bootstrap_clientes(idv, NB, seed=42))
    cel_out, res = [], []
    for nombre in CANDIDATOS:
        sp = specs[nombre]; seeds = (42, 43, 44) if sp["familia"] == "lightgbm" else (42,)
        G = np.empty(len(celdas)); gr = []
        for k in origenes:
            M = meses[meses.index(k) + 1]
            mods, info = fit_honesto(wcb, sp, X, y, mes, M, seeds=seeds)
            for j, (k2, h, t) in enumerate(celdas):
                if k2 != k:
                    continue
                p = np.mean([pred(m, X.loc[mes == t, sp["cols"]]) for m in mods], axis=0)
                G[j] = gini(y[mes == t], p); gr.append((j, GiniRapido(y[mes == t], p)))
                cel_out.append(dict(modelo=nombre, origen=k, h=h, mes_eval=t, gini=G[j], n_iter=info["n_iter"]))
        gr = [g for _, g in sorted(gr, key=lambda z: z[0])]
        b_o = beta_fe(G, kk, hh, kk); b_t = beta_fe(G, kk, hh, tt)
        Bo, Bt = np.empty(NB), np.empty(NB)
        for b, w in enumerate(pesos):
            Gb = np.array([gr[j](w[filas_c[t]]) for j, t in enumerate(tt)])
            Bo[b] = beta_fe(Gb, kk, hh, kk); Bt[b] = beta_fe(Gb, kk, hh, tt)
        res.append(dict(modelo=nombre, beta_fe_origen=b_o, se=Bo.std(ddof=1), ic95_inf=ic(Bo)[0],
                        ic95_sup=ic(Bo)[1], p_boot=p_boot_centrado(Bo, b_o),
                        beta_fe_mes_eval=b_t, se_mes_eval=Bt.std(ddof=1), ic95_inf_mes_eval=ic(Bt)[0],
                        ic95_sup_mes_eval=ic(Bt)[1], gini_h1_medio=G[hh == 1].mean(), gini_h4_medio=G[hh == 4].mean()))
        print(res[-1], f"({time.time()-t0:.0f}s)", flush=True)
    rs = pd.DataFrame(res); rs["p_holm"] = holm(rs.p_boot.to_numpy())
    rs["robusto"] = ((rs.ic95_inf <= 0) & (rs.ic95_sup >= 0)) | (rs.beta_fe_origen > 0)   # IC contiene 0
    rs.to_csv(DIR_RES / "robustez_r7_resumen.csv", index=False)
    ce = pd.DataFrame(cel_out); ce.to_csv(DIR_RES / "robustez_r7_celdas.csv", index=False)
    pd.set_option("display.width", 250); print(rs.round(5).to_string())
    fig, ax = plt.subplots(figsize=(8, 5))
    for nombre, g in ce.groupby("modelo", sort=False):
        g = g.assign(c=g.gini - g.groupby("origen").gini.transform("mean"))
        m = g.groupby("h").c.mean(); b = rs[rs.modelo == nombre].iloc[0]
        ax.plot(m.index, m.values, "o-", label=f"{nombre}: beta={b.beta_fe_origen:+.4f} [{b.ic95_inf:+.4f}, {b.ic95_sup:+.4f}]")
    ax.axhline(0, color="grey", ls="--"); ax.set_xlabel("desfase h (meses entre el corte de train y el mes evaluado)")
    ax.set_ylabel("Gini centrado por origen (media de origenes)"); ax.set_title("R7 - degradacion temporal")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(DIR_FIG / "fig16_r7_degradacion.png", dpi=130); plt.close(fig)
    print(f"Listo en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
