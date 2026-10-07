# -*- coding: utf-8 -*-
"""
13. R4: varianza por semilla y por muestra del modelo FINAL en diciembre (criterios_robustez.py, R4).
Modelos: los 2 de mayor Gini medio (tal cual) en R1 + lgb_reg como referencia.
Modelo final: todo train; n_iter = mediana_k(best_k * filas(train) / filas(train ES_k)) (R1).
 (a) 10 semillas (42..51) con los datos completos.
 (b) 20 bootstraps de CLIENTES de train (se duplican todas las filas del cliente), semilla 42.
Metricas en diciembre: percentil de cada cliente; sd del percentil; top-10 % (990) por replica;
Jaccard medio entre pares; top-10 % de referencia = top-10 % de la media de percentiles;
% de ese top con pertenencia >= 90 % de las replicas. Robusto si >= 85 % en (a) y en (b).
Salidas: robustez_r4_resumen.csv, robustez_r4_pred_dic_<modelo>.csv (float32),
         figuras/fig13_r4v2_estabilidad_top10.png
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
from scipy import stats

from criterios_robustez import CRITERIOS
from robustez_comun import DIR_FIG, DIR_RES, cargar_todo, definir_candidatos, f32, fit_fijo, pred

C = CRITERIOS["R4"]


def n_iter_final(it, nombre, n_total):
    r = it[it.modelo == nombre]
    if r.best_iter_es_Mm1.isna().all():
        return None
    return max(10, int(round(np.median(r.best_iter_es_Mm1 * n_total / r.filas_train_es))))


def metricas(P, top=0.10):
    """P: (replicas, clientes). Devuelve dict de estabilidad del ranking."""
    R, n = P.shape; k = int(round(top * n))
    pct = np.vstack([stats.rankdata(p) / n for p in P])
    tops = [set(np.argsort(-p)[:k]) for p in P]
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(tops, 2)]
    ref = np.argsort(-pct.mean(axis=0))[:k]
    freq = np.array([np.mean([i in t for t in tops]) for i in ref])
    rho = [stats.spearmanr(P[i], P[j])[0] for i, j in itertools.combinations(range(R), 2)]
    sd = pct.std(axis=0, ddof=1)
    return dict(replicas=R, sd_percentil_media=sd.mean(), sd_percentil_p95=np.quantile(sd, .95),
                sd_percentil_top10=sd[ref].mean(), jaccard_top10_medio=np.mean(jac),
                jaccard_top10_min=np.min(jac), pct_top10_estable_90=100 * (freq >= 0.9).mean(),
                pct_top10_siempre=100 * (freq == 1).mean(), spearman_medio=np.mean(rho))


def main():
    t0 = time.time()
    wcb, d, _, _ = cargar_todo()
    X, y, ids = d.X_train, d.y_train.to_numpy(), d.id_train
    specs = definir_candidatos(wcb, X.columns)
    it = pd.read_csv(DIR_RES / "robustez_r1_iteraciones.csv")
    rr = pd.read_csv(DIR_RES / "robustez_r1_resumen.csv")
    rr = rr[rr.escenario == "tal_cual"].sort_values("gini_medio", ascending=False)
    modelos = list(rr.modelo[:2]) + ([] if "lgb_reg" in list(rr.modelo[:2]) else ["lgb_reg"])
    print("R4 modelos:", modelos, flush=True)
    uniq, inv = np.unique(ids, return_inverse=True)
    res = []
    for nombre in modelos:
        sp = specs[nombre]; cols = sp["cols"]
        n_it = n_iter_final(it, nombre, len(X))
        Xt = d.X_test[cols]; out = {"id_cliente": d.id_test}
        Ps = []
        for s in C["semillas"]:
            p = pred(fit_fijo(wcb, sp, X[cols], y, n_it, s), Xt); Ps.append(p); out[f"s{s}"] = f32(p)
        print(f"{nombre} semillas ({time.time()-t0:.0f}s)", flush=True)
        Pb = []
        rng = np.random.default_rng(2026)
        for b in range(C["n_boot_clientes"]):
            cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
            w = cnt[inv].astype(float); idx = np.where(w > 0)[0]
            # v2: pesos = multiplicidad del cliente (no se duplican filas: las copias caerian a la
            # vez en el train y en la validacion interna del ES del EBM y su ES no pararia)
            p = pred(fit_fijo(wcb, sp, X[cols].iloc[idx], y[idx], n_it, C["semilla_modelo_boot"], w=w[idx]), Xt)
            Pb.append(p); out[f"boot{b:02d}"] = f32(p)
        print(f"{nombre} bootstrap ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(out).to_csv(DIR_RES / f"robustez_r4v2_pred_dic_{nombre}.csv", index=False)
        for esquema, P in (("semillas", np.array(Ps)), ("bootstrap_clientes", np.array(Pb))):
            m = metricas(P)
            res.append(dict(modelo=nombre, esquema=esquema, n_iter_final=n_it, **m,
                            robusto=m["pct_top10_estable_90"] >= 85))
            print(res[-1], flush=True)
    rs = pd.DataFrame(res); rs.to_csv(DIR_RES / "robustez_r4v2_resumen.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, met in enumerate(["pct_top10_estable_90", "jaccard_top10_medio"]):
        piv = rs.pivot(index="modelo", columns="esquema", values=met)
        piv.plot.bar(ax=ax[k], rot=0)
        if k == 0:
            ax[k].axhline(85, ls=":", color="r", label="criterio 85 %")
        ax[k].set_title(f"R4 {met}"); ax[k].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(DIR_FIG / "fig13_r4v2_estabilidad_top10.png", dpi=130); plt.close(fig)
    print(f"Listo en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
