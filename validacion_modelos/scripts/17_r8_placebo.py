# -*- coding: utf-8 -*-
"""
17. R8: prueba placebo / de permutacion (criterios_robustez.py, R8).
lgb_sup_sin_dui. En cada replica b = 1..100 se permuta `objetivo` DENTRO de cada mes de train
(una permutacion por mes y replica), y para cada fold M se entrena con meses < M y el MISMO n_iter
que el modelo real de ese fold (R1), semilla 42; se evalua contra las etiquetas REALES de M.
Estadistico: media de los 4 Gini. p = (1 + #{G_placebo >= G_real}) / (1 + B).
Valido si |media placebo| < 2 sd/sqrt(B), G_real > max placebo y z > 3.
Salidas: robustez_r8_placebo.csv, robustez_r8_resumen.csv, figuras/fig17_r8_placebo.png
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
from robustez_comun import DIR_FIG, DIR_RES, MESES_FOLD, cargar_todo, definir_candidatos, fit_fijo, gini, pred

C = CRITERIOS["R8"]


def main():
    t0 = time.time()
    wcb, d, _, _ = cargar_todo()
    X, y, mes = d.X_train, d.y_train.to_numpy(), d.mes_train
    nombre = C["modelo"]; sp = definir_candidatos(wcb, X.columns)[nombre]; cols = sp["cols"]
    it = pd.read_csv(DIR_RES / "robustez_r1_iteraciones.csv"); it = it[it.modelo == nombre].set_index("fold")
    g_real_f = np.array([it.loc[M, "gini_s42"] for M in MESES_FOLD]); g_real = g_real_f.mean()
    idx_mes = {m: np.where(mes == m)[0] for m in np.unique(mes)}
    rng = np.random.default_rng(8008)
    filas = []
    for b in range(C["B"]):
        yp = y.copy()
        for m, ix in idx_mes.items():
            yp[ix] = y[rng.permutation(ix)]
        G = []
        for M in MESES_FOLD:
            tr = mes < M
            mod = fit_fijo(wcb, sp, X.loc[tr, cols], yp[tr], int(it.loc[M, "n_iter"]), 42)
            G.append(gini(y[mes == M], pred(mod, X.loc[mes == M, cols])))
        filas.append(dict(replica=b, gini_medio=np.mean(G), **{f"gini_{M}": g for M, g in zip(MESES_FOLD, G)}))
        if b % 10 == 9:
            print(f"placebo {b+1}/{C['B']} ({time.time()-t0:.0f}s)", flush=True)
    pl = pd.DataFrame(filas); pl.to_csv(DIR_RES / "robustez_r8_placebo.csv", index=False)
    v = pl.gini_medio.to_numpy(); B = len(v)
    mu, sd = v.mean(), v.std(ddof=1); z = (g_real - mu) / sd
    centrado = abs(mu) < 2 * sd / np.sqrt(B)
    r = dict(modelo=nombre, B=B, gini_real_s42=g_real, placebo_media=mu, placebo_sd=sd,
             placebo_min=v.min(), placebo_max=v.max(), placebo_p95=np.quantile(v, .95),
             z=z, p_permutacion=(1 + np.sum(v >= g_real)) / (1 + B), placebo_centrado=centrado,
             real_supera_max=g_real > v.max(), valido=bool(centrado and g_real > v.max() and z > 3),
             **{f"placebo_sd_{M}": pl[f"gini_{M}"].std(ddof=1) for M in MESES_FOLD},
             **{f"gini_real_{M}": g for M, g in zip(MESES_FOLD, g_real_f)})
    pd.DataFrame([r]).to_csv(DIR_RES / "robustez_r8_resumen.csv", index=False)
    print(pd.Series(r).to_string())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(v, bins=25, color="grey", alpha=.8, label=f"placebo (B={B}): media {mu:+.4f}, sd {sd:.4f}")
    ax.axvline(g_real, color="r", lw=2, label=f"real {g_real:.4f} (z={z:.1f})")
    ax.set_xlabel("Gini medio ago-nov"); ax.set_title(f"R8 - placebo con objetivo permutado dentro del mes ({nombre})")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(DIR_FIG / "fig17_r8_placebo.png", dpi=130); plt.close(fig)
    print(f"Listo en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
