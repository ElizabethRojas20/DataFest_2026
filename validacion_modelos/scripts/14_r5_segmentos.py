# -*- coding: utf-8 -*-
"""
14. R5: Gini por segmento de los 5 candidatos (criterios_robustez.py, R5; solo calculo).
Segmentos: nuevo (primer mes en el panel) / antiguo; banda_riesgo low/medium/high;
numero_productos 1/2/3+; activo_movil True/False.
Estadistico: media de los 4 Gini por fold dentro del segmento; IC95 percentil por bootstrap de
clientes conjunto (1000). Delta vs lgb_reg con p bootstrap centrado y Holm sobre 36 contrastes.
Orden: tau de Kendall entre el orden global (R1) y el del segmento.
Salidas: robustez_r5_segmentos.csv, robustez_r5_orden.csv, figuras/fig14_r5_segmentos.png
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
from scipy import stats

from robustez_comun import (CANDIDATOS, DIR_FIG, DIR_RES, MESES_FOLD, GiniRapido, gini, holm, ic,
                            p_boot_centrado, pesos_bootstrap_clientes)

NB = 1000


def main():
    t0 = time.time()
    df = pd.read_csv(DIR_RES / "robustez_r1_predicciones.csv",
                     usecols=["id_cliente", "mes", "y", "nuevo", "banda_riesgo", "numero_productos", "activo_movil"])
    P = dict(np.load(DIR_RES / "robustez_r1_predicciones.npz"))
    y, mes, ids = df.y.to_numpy(), df.mes.to_numpy(), df.id_cliente.to_numpy()
    nprod = np.where(df.numero_productos >= 3, "3+", df.numero_productos.astype(str))
    segs = {("nuevo", "nuevo"): df.nuevo.astype(str).str.lower().eq("true").to_numpy()}
    segs[("nuevo", "antiguo")] = ~segs[("nuevo", "nuevo")]
    for v in ("low", "medium", "high"):
        segs[("banda_riesgo", v)] = (df.banda_riesgo == v).to_numpy()
    for v in ("1", "2", "3+"):
        segs[("numero_productos", v)] = nprod == v
    for v in ("True", "False"):
        segs[("activo_movil", v)] = df.activo_movil.astype(str).str.capitalize().eq(v).to_numpy()
    celdas = {s: [np.where(msk & (mes == M))[0] for M in MESES_FOLD] for s, msk in segs.items()}
    G = {(s, m): np.array([gini(y[c], P[f"{m}__hon"][c]) for c in cs]) for s, cs in celdas.items() for m in CANDIDATOS}
    gr = {(s, m): [GiniRapido(y[c], P[f"{m}__hon"][c]) for c in cs] for s, cs in celdas.items() for m in CANDIDATOS}
    Gb = {k: np.empty(NB) for k in G}
    for b, w in enumerate(pesos_bootstrap_clientes(ids, NB, seed=42)):
        for (s, m), gg in gr.items():
            Gb[(s, m)][b] = np.mean([g(w[c]) for g, c in zip(gg, celdas[s])])
    print(f"bootstrap ({time.time()-t0:.0f}s)", flush=True)
    filas = []
    for s, cs in celdas.items():
        n = sum(len(c) for c in cs); npos = int(sum(y[c].sum() for c in cs))
        for m in CANDIDATOS:
            r = dict(variable=s[0], segmento=s[1], modelo=m, filas=n, positivos=npos, tasa=npos / n,
                     gini_medio=G[(s, m)].mean(), se=Gb[(s, m)].std(ddof=1),
                     ic95_inf=ic(Gb[(s, m)])[0], ic95_sup=ic(Gb[(s, m)])[1],
                     **{f"gini_{M}": g for M, g in zip(MESES_FOLD, G[(s, m)])})
            if m != "lgb_reg":
                d = G[(s, m)].mean() - G[(s, "lgb_reg")].mean(); db = Gb[(s, m)] - Gb[(s, "lgb_reg")]
                r.update(delta_vs_lgb_reg=d, delta_ic95_inf=ic(db)[0], delta_ic95_sup=ic(db)[1],
                         p_boot_delta=p_boot_centrado(db, d))
            filas.append(r)
    res = pd.DataFrame(filas)
    msk = res.p_boot_delta.notna()
    res.loc[msk, "p_holm_delta"] = holm(res.loc[msk, "p_boot_delta"].to_numpy())
    res.to_csv(DIR_RES / "robustez_r5_segmentos.csv", index=False)
    r1 = pd.read_csv(DIR_RES / "robustez_r1_resumen.csv"); r1 = r1[r1.escenario == "tal_cual"].set_index("modelo")
    glob = r1.loc[list(CANDIDATOS), "gini_medio"].to_numpy(); mejor_glob = CANDIDATOS[int(np.argmax(glob))]
    orden = []
    for (v, s), g in res.groupby(["variable", "segmento"], sort=False):
        g = g.set_index("modelo").loc[list(CANDIDATOS)]
        tau = stats.kendalltau(glob, g.gini_medio.to_numpy())[0]
        mejor_seg = g.gini_medio.idxmax()
        db = Gb[((v, s), mejor_glob)] - Gb[((v, s), mejor_seg)]
        orden.append(dict(variable=v, segmento=s, tau_kendall_vs_global=tau,
                          orden_segmento=" > ".join(g.gini_medio.sort_values(ascending=False).index),
                          mejor_global=mejor_glob, mejor_segmento=mejor_seg,
                          delta_mejorglobal_vs_mejorseg=g.gini_medio[mejor_glob] - g.gini_medio[mejor_seg],
                          ic95_inf=ic(db)[0], ic95_sup=ic(db)[1]))
    od = pd.DataFrame(orden); od.to_csv(DIR_RES / "robustez_r5_orden.csv", index=False)
    peor = res.loc[res.groupby("modelo").gini_medio.idxmin(), ["modelo", "variable", "segmento", "gini_medio"]]
    pd.set_option("display.width", 250)
    print(res.round(4).to_string()); print(od.round(4).to_string()); print("peor segmento:\n", peor.round(4).to_string())
    fig, ax = plt.subplots(figsize=(12, 5)); segl = [f"{v}={s}" for v, s in celdas]
    x = np.arange(len(segl)); wd = 0.16
    for k, m in enumerate(CANDIDATOS):
        g = res[res.modelo == m]
        ax.errorbar(x + (k - 2) * wd, g.gini_medio, yerr=[g.gini_medio - g.ic95_inf, g.ic95_sup - g.gini_medio],
                    fmt="o", ms=4, capsize=2, label=m)
    ax.set_xticks(x); ax.set_xticklabels(segl, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Gini medio ago-nov (IC95)"); ax.set_title("R5 - Gini por segmento"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(DIR_FIG / "fig14_r5_segmentos.png", dpi=130); plt.close(fig)
    print(f"Listo en {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
