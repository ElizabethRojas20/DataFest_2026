# -*- coding: utf-8 -*-
"""
04. Metricas del walk-forward: Gini por fold (honesto y esquema del pipeline), media de 4 folds,
IC95 % por bootstrap de CLIENTES con reemplazo (1000 replicas, mismas replicas para todos los
modelos -> diferencias pareadas), optimismo del early stopping sobre el fold evaluado y
Gini por segmento (clientes nuevos / antiguos).

Redes guardadas: solo sep-nov (ago esta en su entrenamiento); Keras con early stopping sobre
sep-nov -> optimista. Se comparan con lgb_reg en los mismos 3 meses.

Requiere: wf_predicciones.csv (01) y nn_predicciones.csv (03)
Salidas: wf_gini_por_fold.csv, wf_resumen.csv, wf_gini_segmentos.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from comun import DIR_RES, MESES_FOLD, N_BOOT, SEMILLA, SEMILLAS, GiniRapido, gini, pesos_bootstrap_clientes

ARBOLES = ["lgb_reg", "lgb_sin_reg", "xgb", "catboost", "random_forest", "lgb_reg_sin_dui"]
REDES = ["mlp_sklearn", "nn_baseline", "nn_optimizado"]
REF = "lgb_reg"


def resumir(boot, punto, nombre_cols):
    """boot: (n_boot, k) ; punto: (k,) -> DataFrame con estimacion, SE e IC95."""
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return pd.DataFrame({"estimacion": punto, "se_boot": boot.std(axis=0, ddof=1),
                         "ic95_inf": lo, "ic95_sup": hi}, index=nombre_cols)


def main():
    wf = pd.read_csv(DIR_RES / "wf_predicciones.csv")
    nn = pd.read_csv(DIR_RES / "nn_predicciones.csv")
    wf = wf.merge(nn.drop(columns="y"), on=["id_cliente", "mes"], how="left", validate="1:1")
    meses = wf["mes"].to_numpy(); ids = wf["id_cliente"].to_numpy(); y = wf["y"].to_numpy()
    folds = {M: np.where(meses == M)[0] for M in MESES_FOLD}

    # columnas a evaluar: (modelo, variante) -> nombre de columna
    series = {}
    for m in ARBOLES:
        series[(m, "honesto")] = f"{m}__hon"
        series[(m, "pipeline")] = f"{m}__pipe"
        for s in SEMILLAS:
            series[(m, f"semilla_{s}")] = f"{m}__s{s}"
    for m in REDES:
        series[(m, "honesto")] = f"{m}__hon"

    # Gini puntual por fold
    filas = []
    for (m, var), col in series.items():
        for M, idx in folds.items():
            filas.append(dict(modelo=m, variante=var, fold=M, gini=gini(y[idx], wf[col].to_numpy()[idx])))
    punt = pd.DataFrame(filas)

    # Bootstrap de clientes (con reemplazo), comun a todos los modelos
    claves = [k for k in series if k[1] in ("honesto", "pipeline")]
    calc = {(k, M): GiniRapido(y[idx], wf[series[k]].to_numpy()[idx]) for k in claves for M, idx in folds.items()}
    B = {(k, M): np.empty(N_BOOT) for k in claves for M in MESES_FOLD}
    for b, w in enumerate(pesos_bootstrap_clientes(ids, N_BOOT, SEMILLA)):
        for (k, M), g in calc.items():
            B[(k, M)][b] = g(w[folds[M]])

    # Tabla por fold
    tab = punt.pivot_table(index=["modelo", "fold"], columns="variante", values="gini").reset_index()
    for i, r in tab.iterrows():
        k = (r.modelo, "honesto")
        lo, hi = np.percentile(B[(k, r.fold)], [2.5, 97.5])
        tab.loc[i, "ic95_inf_honesto"], tab.loc[i, "ic95_sup_honesto"] = lo, hi
        if (r.modelo, "pipeline") in series:
            tab.loc[i, "optimismo_pipeline_menos_honesto"] = r.pipeline - r.honesto
        if r.modelo != REF:
            d = B[(k, r.fold)] - B[((REF, "honesto"), r.fold)]
            tab.loc[i, "dif_vs_lgb_reg"] = r.honesto - tab[(tab.modelo == REF) & (tab.fold == r.fold)].honesto.iloc[0]
            tab.loc[i, "se_dif_vs_lgb_reg"] = d.std(ddof=1)
    tab["nota"] = ""
    tab.loc[tab.modelo.isin(REDES) & (tab.fold == MESES_FOLD[0]), "nota"] = "DENTRO de muestra (ago en el entrenamiento de la red)"
    tab.loc[tab.modelo.isin(["nn_baseline", "nn_optimizado"]) & (tab.fold > MESES_FOLD[0]), "nota"] = "optimista: early stopping sobre sep-nov"
    tab.loc[(tab.modelo == "mlp_sklearn") & (tab.fold > MESES_FOLD[0]), "nota"] = "fuera de muestra (escalador ajustado con ene-nov)"
    tab.to_csv(DIR_RES / "wf_gini_por_fold.csv", index=False)

    # Resumen: media de folds (arboles: 4 folds ago-nov; redes: 3 folds sep-nov) + pareadas
    res = []
    for m in ARBOLES + REDES:
        for nombre_set, fs in (("ago-nov", MESES_FOLD), ("sep-nov", MESES_FOLD[1:])):
            if m in REDES and nombre_set == "ago-nov":
                continue
            k = (m, "honesto")
            bm = np.mean([B[(k, M)] for M in fs], axis=0)
            pm = punt[(punt.modelo == m) & (punt.variante == "honesto") & punt.fold.isin(fs)].gini.mean()
            fila = dict(modelo=m, folds=nombre_set, gini_medio_honesto=pm, se=bm.std(ddof=1),
                        ic95_inf=np.percentile(bm, 2.5), ic95_sup=np.percentile(bm, 97.5))
            if (m, "pipeline") in series:
                pp = punt[(punt.modelo == m) & (punt.variante == "pipeline") & punt.fold.isin(fs)].gini.mean()
                bo = np.mean([B[((m, "pipeline"), M)] - B[(k, M)] for M in fs], axis=0)
                fila.update(gini_medio_pipeline=pp, optimismo=pp - pm,
                            optimismo_ic95_inf=np.percentile(bo, 2.5), optimismo_ic95_sup=np.percentile(bo, 97.5))
                semi = [punt[(punt.modelo == m) & (punt.variante == f"semilla_{s}") & punt.fold.isin(fs)].gini.mean()
                        for s in SEMILLAS]
                fila.update(gini_medio_por_semilla=str([round(float(v), 4) for v in semi]), rango_semillas=max(semi) - min(semi))
            if m != REF:
                bd = np.mean([B[(k, M)] - B[((REF, "honesto"), M)] for M in fs], axis=0)
                pr = punt[(punt.modelo == REF) & (punt.variante == "honesto") & punt.fold.isin(fs)].gini.mean()
                gm = punt[(punt.modelo == m) & (punt.variante == "honesto")].set_index("fold").gini
                gr = punt[(punt.modelo == REF) & (punt.variante == "honesto")].set_index("fold").gini
                fila.update(dif_vs_lgb_reg=pm - pr, se_dif=bd.std(ddof=1),
                            dif_ic95_inf=np.percentile(bd, 2.5), dif_ic95_sup=np.percentile(bd, 97.5),
                            folds_mejor_que_lgb_reg=f"{int(sum(gm[M] > gr[M] for M in fs))}/{len(fs)}")
            res.append(fila)
    res = pd.DataFrame(res)
    res.to_csv(DIR_RES / "wf_resumen.csv", index=False)

    # Gini por segmento nuevos / antiguos (honesto)
    seg = []
    nuevo = wf["nuevo"].to_numpy().astype(bool)
    for m in ARBOLES + REDES:
        for M, idx in folds.items():
            if m in REDES and M == MESES_FOLD[0]:
                continue
            p = wf[f"{m}__hon"].to_numpy()
            for nom, msk in (("nuevos", nuevo[idx]), ("antiguos", ~nuevo[idx])):
                ii = idx[msk]
                seg.append(dict(modelo=m, fold=M, segmento=nom, n=len(ii), tasa=y[ii].mean(), gini=gini(y[ii], p[ii])))
    seg = pd.DataFrame(seg)
    seg.to_csv(DIR_RES / "wf_gini_segmentos.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(tab.round(4).to_string(index=False))
    print(res.round(4).to_string(index=False))
    print(seg.pivot_table(index="modelo", columns="segmento", values="gini").round(4))


if __name__ == "__main__":
    main()
