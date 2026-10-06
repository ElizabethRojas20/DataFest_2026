# -*- coding: utf-8 -*-
"""
15. R6: perturbacion de entradas e importancia por permutacion EN VALIDACION (criterios_robustez.py, R6).
Modelos de cada fold (semilla 42, cache de 09_). Ruido multiplicativo log-normal al PREDECIR:
x' = x * exp(s Z - s^2/2), Z ~ N(0,1), s in {0,05; 0,10; 0,20}; columnas enteras redondeadas;
recorte al rango de train; 5 sorteos por nivel. Caida = Gini limpio - media de Gini ruidosos.
Importancia: permutacion por variable ORIGINAL (las dummies OHE de una categorica juntas) dentro
del mes, 3 permutaciones; IC95 y p unilateral (importancia > 0) por bootstrap de clientes (200),
Holm por modelo. Robusto si caida(s=0,10) < SE_REF y ninguna variable > 50 % de la importancia total.
Salidas: robustez_r6_ruido.csv, robustez_r6_importancia.csv, robustez_r6_resumen.csv,
         figuras/fig15_r6_perturbacion.png
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robustez_comun import (CANDIDATOS, CATEG_OHE, DIR_CACHE, DIR_FIG, DIR_RES, MESES_FOLD, NUM_ORIG,
                            GiniRapido, cargar_todo, definir_candidatos, gini, holm, ic, leer_se_ref,
                            pesos_bootstrap_clientes, pred)

NIVELES = (0.05, 0.10, 0.20); N_SORTEOS = 5; N_PERM = 3; NB = 200


def grupos_variables(cols):
    g = {}
    for c in cols:
        base = next((cat for cat in CATEG_OHE if c.startswith(cat + "_")), c)
        g.setdefault(base, []).append(c)
    return g


def main():
    t0 = time.time()
    wcb, d, _, _ = cargar_todo()
    X, y, mes, ids = d.X_train, d.y_train.to_numpy(), d.mes_train, d.id_train
    specs = definir_candidatos(wcb, X.columns)
    se_ref = leer_se_ref()
    val = np.isin(mes, MESES_FOLD); pos = np.where(val)[0]
    folds = [np.searchsorted(pos, np.where(mes == M)[0]) for M in MESES_FOLD]
    yv, idv = y[val], ids[val]
    lo, hi = X.min(), X.max()
    entera = {c: bool(np.all(np.mod(X[c].dropna(), 1) == 0)) for c in NUM_ORIG}
    pesos = list(pesos_bootstrap_clientes(idv, NB, seed=42))
    ruido_f, imp_f, resumen = [], [], []
    for nombre in CANDIDATOS:
        cols = specs[nombre]["cols"]; numc = [c for c in NUM_ORIG if c in cols]
        grupos = grupos_variables(cols)
        p_clean = np.zeros(len(yv)); p_noise = {s: np.zeros((N_SORTEOS, len(yv))) for s in NIVELES}
        p_perm = {v: np.zeros((N_PERM, len(yv))) for v in grupos}
        for j, (M, f) in enumerate(zip(MESES_FOLD, folds)):
            m = joblib.load(DIR_CACHE / f"r1_{nombre}_{M}.joblib")
            Xv = X.loc[mes == M, cols].reset_index(drop=True)
            p_clean[f] = pred(m, Xv)
            rng = np.random.default_rng(1000 + j)
            for s in NIVELES:
                for k in range(N_SORTEOS):
                    Xn = Xv.copy()
                    for c in numc:
                        z = rng.standard_normal(len(Xn))
                        v = Xn[c].to_numpy(dtype=float) * np.exp(s * z - s * s / 2)
                        if entera[c]:
                            v = np.round(v)
                        Xn[c] = np.clip(v, lo[c], hi[c]).astype(Xv[c].dtype)
                    p_noise[s][k, f] = pred(m, Xn)
            for v, cs in grupos.items():
                for k in range(N_PERM):
                    Xp = Xv.copy(); perm = rng.permutation(len(Xp))
                    Xp[cs] = Xv[cs].to_numpy()[perm]
                    p_perm[v][k, f] = pred(m, Xp)
        print(f"{nombre} predicciones ({time.time()-t0:.0f}s)", flush=True)
        g_clean = np.array([gini(yv[f], p_clean[f]) for f in folds])
        r_clean = [GiniRapido(yv[f], p_clean[f]) for f in folds]
        bc = np.array([[g(w[f]) for g, f in zip(r_clean, folds)] for w in pesos])     # (NB, 4)
        for s in NIVELES:
            gn = np.array([[gini(yv[f], p_noise[s][k, f]) for f in folds] for k in range(N_SORTEOS)])
            caida_f = g_clean - gn.mean(axis=0); caida = caida_f.mean()
            rr = [[GiniRapido(yv[f], p_noise[s][k, f]) for f in folds] for k in range(N_SORTEOS)]
            bn = np.array([[np.mean([rr[k][j](w[f]) for k in range(N_SORTEOS)]) for j, f in enumerate(folds)] for w in pesos])
            cb = (bc - bn).mean(axis=1)
            ruido_f.append(dict(modelo=nombre, s=s, gini_limpio=g_clean.mean(), gini_ruido=gn.mean(),
                                caida=caida, caida_se=cb.std(ddof=1), caida_ic95_inf=ic(cb)[0],
                                caida_ic95_sup=ic(cb)[1], caida_relativa=caida / g_clean.mean(),
                                **{f"caida_{M}": c for M, c in zip(MESES_FOLD, caida_f)}))
        filas = []
        for v in grupos:
            gp = np.array([[gini(yv[f], p_perm[v][k, f]) for f in folds] for k in range(N_PERM)])
            dr = (g_clean - gp.mean(axis=0)).mean()
            rr = [[GiniRapido(yv[f], p_perm[v][k, f]) for f in folds] for k in range(N_PERM)]
            bp = np.array([[np.mean([rr[k][j](w[f]) for k in range(N_PERM)]) for j, f in enumerate(folds)] for w in pesos])
            db = (bc - bp).mean(axis=1)
            filas.append(dict(modelo=nombre, variable=v, n_columnas=len(grupos[v]), caida_gini=dr,
                              se=db.std(ddof=1), ic95_inf=ic(db)[0], ic95_sup=ic(db)[1],
                              p_unilateral=(1 + np.sum(db - dr >= dr)) / (1 + NB)))
        imp = pd.DataFrame(filas)
        imp["p_holm"] = holm(imp.p_unilateral.to_numpy())
        pos_ = imp.caida_gini.clip(lower=0); imp["cuota"] = pos_ / pos_.sum()
        imp = imp.sort_values("caida_gini", ascending=False); imp_f.append(imp)
        c10 = [r for r in ruido_f if r["modelo"] == nombre and r["s"] == 0.10][0]
        resumen.append(dict(modelo=nombre, caida_s10=c10["caida"], caida_s10_se=c10["caida_se"], se_ref=se_ref,
                            pasa_ruido=c10["caida"] < se_ref, variable_top=imp.variable.iloc[0],
                            cuota_top=imp.cuota.iloc[0], pasa_concentracion=imp.cuota.iloc[0] <= 0.5,
                            n_var_significativas_holm=int((imp.p_holm < 0.05).sum()),
                            robusto=(c10["caida"] < se_ref) and (imp.cuota.iloc[0] <= 0.5)))
        print(resumen[-1], flush=True)
    rd = pd.DataFrame(ruido_f); rd.to_csv(DIR_RES / "robustez_r6_ruido.csv", index=False)
    im = pd.concat(imp_f); im.to_csv(DIR_RES / "robustez_r6_importancia.csv", index=False)
    rs = pd.DataFrame(resumen); rs.to_csv(DIR_RES / "robustez_r6_resumen.csv", index=False)
    pd.set_option("display.width", 250)
    print(rd.round(4).to_string()); print(im.round(4).to_string())
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for nombre, g in rd.groupby("modelo", sort=False):
        ax[0].errorbar(g.s * 100, g.caida, yerr=1.96 * g.caida_se, marker="o", capsize=3, label=nombre)
    ax[0].axhline(se_ref, ls=":", color="r", label="SE_REF"); ax[0].set_xlabel("ruido s (%)")
    ax[0].set_ylabel("caida de Gini (media 4 folds)"); ax[0].legend(fontsize=8); ax[0].set_title("R6 ruido multiplicativo")
    top = im.groupby("variable").caida_gini.mean().sort_values(ascending=False).index[:12]
    piv = im.pivot(index="variable", columns="modelo", values="caida_gini").loc[top[::-1]]
    piv.plot.barh(ax=ax[1], width=.8); ax[1].set_title("R6 importancia por permutacion en validacion (caida de Gini)")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(DIR_FIG / "fig15_r6_perturbacion.png", dpi=130); plt.close(fig)
    print(f"Listo en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
