# -*- coding: utf-8 -*-
"""
10. R1 (inferencia): comparacion formal de los 5 candidatos (criterios_robustez.py, R1 + ENMIENDA_1).
  * Gini por fold y medio; SE e IC95 por bootstrap de clientes CONJUNTO sobre los 4 folds (B=2000).
  * Los 10 pares: Delta medio, IC95, p bootstrap pareado centrado, Holm (FWER 5 %), n de folds con
    el mismo signo, regla "MEJOR"; DeLong por fold (solo tal cual) con Holm dentro de cada fold;
    regla de reemplazo de CLAUDE.md frente a lgb_reg.
  * Escenario principal: folds tal cual. Secundario (E2): f = dui permutada (3 perm., solo modelos
    con dui) + pesos para que los nuevos sean el 18,6 % de cada fold. Sin pesos adversariales.
Requiere: robustez_r1_predicciones.npz/.csv, robustez_r1_iteraciones.csv
Salidas: robustez_r1_resumen.csv, robustez_r1_pares.csv, robustez_r1_delong_fold.csv,
         figuras/fig10_r1_comparacion_formal.png
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

from comun import PROP_NUEVOS_DIC
from criterios_robustez import CRITERIOS
from robustez_comun import (CANDIDATOS, CON_DUI, DIR_FIG, DIR_RES, MESES_FOLD, GiniRapido, delong, gini,
                            holm, ic, p_boot_centrado, pesos_bootstrap_clientes)

C = CRITERIOS["R1"]


def pesos_nuevos(mes, nuevo):
    w = np.empty(len(mes))
    for M in MESES_FOLD:
        i = mes == M; pn = nuevo[i].mean()
        w[i] = np.where(nuevo[i], PROP_NUEVOS_DIC / pn, (1 - PROP_NUEVOS_DIC) / (1 - pn))
    return w


def main():
    t0 = time.time()
    df = pd.read_csv(DIR_RES / "robustez_r1_predicciones.csv", usecols=["id_cliente", "mes", "y", "nuevo"])
    P = dict(np.load(DIR_RES / "robustez_r1_predicciones.npz"))
    it = pd.read_csv(DIR_RES / "robustez_r1_iteraciones.csv")
    y = df.y.to_numpy(); mes = df.mes.to_numpy(); ids = df.id_cliente.to_numpy()
    w_f = pesos_nuevos(mes, df.nuevo.to_numpy().astype(bool))
    folds = [np.where(mes == M)[0] for M in MESES_FOLD]
    esc = {"tal_cual": ({m: f"{m}__hon" for m in CANDIDATOS}, np.ones(len(y))),
           "f_dui_perm_18_6_nuevos": ({m: (f"{m}__perm" if m in CON_DUI else f"{m}__hon") for m in CANDIDATOS}, w_f)}
    nm, nf, B = len(CANDIDATOS), len(folds), C["n_boot"]
    res_res, res_pares, res_delong = [], [], []
    for e, (colmap, wesc) in esc.items():
        G = np.array([[gini(y[f], P[colmap[m]][f], wesc[f]) for f in folds] for m in CANDIDATOS])
        gr = [[GiniRapido(y[f], P[colmap[m]][f]) for f in folds] for m in CANDIDATOS]
        Gb = np.empty((B, nm, nf))
        for b, w in enumerate(pesos_bootstrap_clientes(ids, B, seed=42)):
            w = w * wesc
            for j, f in enumerate(folds):
                for i in range(nm):
                    Gb[b, i, j] = gr[i][j](w[f])
        for i, m in enumerate(CANDIDATOS):
            mb = Gb[:, i, :].mean(axis=1)
            r = dict(escenario=e, modelo=m, gini_medio=G[i].mean(), se_boot=mb.std(ddof=1),
                     ic95_inf=ic(mb)[0], ic95_sup=ic(mb)[1],
                     **{f"gini_{M}": G[i, j] for j, M in enumerate(MESES_FOLD)})
            if e == "tal_cual":
                ri = it[it.modelo == m]
                por_sem = [ri[f"gini_s{s}"].mean() for s in (42, 43, 44)]
                r.update(gini_medio_s42=por_sem[0], gini_medio_s43=por_sem[1], gini_medio_s44=por_sem[2],
                         sd_semillas=float(np.std(por_sem, ddof=1)), n_iter_folds=str(list(ri.n_iter)))
            res_res.append(r)
        filas = []
        for a, b_ in itertools.combinations(range(nm), 2):
            dfold = G[a] - G[b_]; d = dfold.mean()
            db = (Gb[:, a, :] - Gb[:, b_, :]).mean(axis=1)
            filas.append(dict(escenario=e, a=CANDIDATOS[a], b=CANDIDATOS[b_], delta=d, se=db.std(ddof=1),
                              ic95_inf=ic(db)[0], ic95_sup=ic(db)[1], z_boot=d / db.std(ddof=1),
                              p_boot=p_boot_centrado(db, d),
                              folds_a_mejor=int((dfold > 0).sum()), folds_b_mejor=int((dfold < 0).sum()),
                              **{f"delta_{M}": dfold[j] for j, M in enumerate(MESES_FOLD)}))
            if e != "tal_cual":
                continue
            for M, f in zip(MESES_FOLD, folds):
                auc_a, auc_b, var, z, p = delong(y[f], P[colmap[CANDIDATOS[a]]][f], P[colmap[CANDIDATOS[b_]]][f])
                res_delong.append(dict(escenario=e, a=CANDIDATOS[a], b=CANDIDATOS[b_], fold=M,
                                       delta_gini=2 * (auc_a - auc_b), se_gini=2 * np.sqrt(var), z=z, p=p))
        fp = pd.DataFrame(filas)
        fp["p_holm"] = holm(fp.p_boot.to_numpy())
        k, al = C["min_folds_mismo_signo"], C["alpha"]
        fp["veredicto"] = np.where((fp.p_holm < al) & (fp.delta > 0) & (fp.folds_a_mejor >= k), "a MEJOR",
                                   np.where((fp.p_holm < al) & (fp.delta < 0) & (fp.folds_b_mejor >= k),
                                            "b MEJOR", "sin diferencia"))
        dcols = [f"delta_{M}" for M in MESES_FOLD]
        refa = fp.a == "lgb_reg"
        fp["regla_claude_md_vs_lgb_reg"] = ""
        fp.loc[refa, "regla_claude_md_vs_lgb_reg"] = [
            "CUMPLE" if (-r.delta > 0.003 and all(-r[c] >= 0 for c in dcols)) else "no cumple"
            for _, r in fp[refa].iterrows()]
        res_pares.append(fp)
        print(f"escenario {e} ({time.time()-t0:.0f}s)", flush=True)
    rr = pd.DataFrame(res_res); rr.to_csv(DIR_RES / "robustez_r1_resumen.csv", index=False)
    pares = pd.concat(res_pares); pares.to_csv(DIR_RES / "robustez_r1_pares.csv", index=False)
    dl = pd.DataFrame(res_delong); dl["p_holm_en_fold"] = np.nan
    for M, g in dl.groupby("fold"):
        dl.loc[g.index, "p_holm_en_fold"] = holm(g.p.to_numpy())
    dl.to_csv(DIR_RES / "robustez_r1_delong_fold.csv", index=False)
    figura(pares)
    pd.set_option("display.width", 250)
    print(rr.round(4).to_string()); print(pares.round(4).to_string()); print(dl.round(4).to_string())
    print(f"Listo en {time.time()-t0:.0f}s")


def figura(pares):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for k, e in enumerate(["tal_cual", "f_dui_perm_18_6_nuevos"]):
        p = pares[(pares.escenario == e) & (pares.a == "lgb_reg")]
        yy = np.arange(len(p)); dd = -p.delta.to_numpy()
        ax[k].errorbar(dd, yy, xerr=[dd + p.ic95_sup.to_numpy(), -p.ic95_inf.to_numpy() - dd],
                       fmt="o", color="k", capsize=4, label="media 4 folds (IC95 bootstrap)")
        for j, M in enumerate(MESES_FOLD):
            ax[k].scatter(-p[f"delta_{M}"], yy + 0.12 * (j - 1.5), s=14, alpha=.6, label=str(M))
        ax[k].axvline(0, color="grey", ls="--"); ax[k].axvline(0.003, color="r", ls=":", label="+0,003 CLAUDE.md")
        ax[k].set_yticks(yy); ax[k].set_yticklabels([f"{b}\np_Holm={h:.3f}" for b, h in zip(p.b, p.p_holm)])
        ax[k].set_xlabel("Delta Gini (candidato - lgb_reg)"); ax[k].set_title(f"R1 - escenario {e}")
    ax[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(DIR_FIG / "fig10_r1_comparacion_formal.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
