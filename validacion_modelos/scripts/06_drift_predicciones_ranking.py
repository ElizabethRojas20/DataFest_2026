# -*- coding: utf-8 -*-
"""
06. Drift de predicciones y estabilidad del ranking (nov vs dic).

Drift de scores (PSI, 10 cuantiles de la referencia = scores de nov):
  * mismo_modelo : modelo entrenado <= oct (fold nov de 01) aplicado a nov y a dic -> aisla el
                   cambio de los DATOS (criterio C1 del veredicto).
  * modelo_final : nov (modelo <= oct) frente a dic (modelo final, todo train) -> lo que vera la entrega.
  * por segmento : nuevos y antiguos por separado (mismo modelo).
  Redes: el mismo modelo guardado (entrenado ene-ago) en nov y en dic.
Ranking:
  * Spearman entre modelos en nov (modelos <= oct) y en dic (modelos finales; criterio C3) y en dic
    con los modelos <= oct (mismo modelo, referencia).
  * Spearman entre semillas (42/43/44) en nov y en dic.
  * Solapamiento del top-10 % y top-20 % con lgb_reg y con la entrega.
  * Spearman con resultados_reportes/submission_final.csv (solo lectura).
  * Dependencia de dui: Spearman(pred, dui) y Spearman(pred, pred con dui permutada) en nov y dic.

Requiere: 01, 02, 03.  Salidas: drift_predicciones.csv, ranking_spearman_{nov,dic,dic_modelos_oct}.csv,
          ranking_resumen.csv, ranking_topk.csv, dependencia_dui.csv
"""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from comun import DIR_RES, RAIZ, SEMILLAS, psi

ARBOLES = ["lgb_reg", "lgb_sin_reg", "xgb", "catboost", "random_forest", "lgb_reg_sin_dui"]
REDES = ["mlp_sklearn", "nn_baseline", "nn_optimizado"]
TODOS = ARBOLES + REDES
REF = "lgb_reg"


def sp(a, b):
    return float(spearmanr(a, b).statistic)


def solape(a, b, q):
    k = int(round(q * len(a)))
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k


def matriz(P):
    k = list(P)
    return pd.DataFrame([[sp(P[a], P[b]) for b in k] for a in k], index=k, columns=k)


def main():
    wf = pd.read_csv(DIR_RES / "wf_predicciones.csv")
    nn = pd.read_csv(DIR_RES / "nn_predicciones.csv")
    wf = wf.merge(nn.drop(columns="y"), on=["id_cliente", "mes"], how="left", validate="1:1")
    nov = wf[wf.mes == 202611].reset_index(drop=True)
    dic_oct = pd.read_csv(DIR_RES / "wf_pred_dic_modelo_oct.csv")
    fin = pd.read_csv(DIR_RES / "final_pred_dic.csv")
    nnd = pd.read_csv(DIR_RES / "nn_pred_dic.csv")
    sub = pd.read_csv(RAIZ / "resultados_reportes" / "submission_final.csv")
    ids_tr = pd.read_csv(RAIZ / "datos_entrada" / "train.csv", usecols=["id_cliente"]).id_cliente
    for t in (fin, nnd, dic_oct):
        assert (t.id_cliente.values == sub.id_cliente.values).all()
    nuevo_dic = (~fin.id_cliente.isin(ids_tr)).to_numpy()
    nuevo_nov = nov.nuevo.to_numpy().astype(bool)
    s = sub.prediccion.to_numpy()

    P_nov = {m: nov[f"{m}__hon"].to_numpy() for m in TODOS}
    P_dic_oct = {**{m: dic_oct[f"{m}__hon"].to_numpy() for m in ARBOLES}, **{m: nnd[f"{m}__final"].to_numpy() for m in REDES}}
    P_dic = {**{m: fin[f"{m}__final"].to_numpy() for m in ARBOLES}, **{m: nnd[f"{m}__final"].to_numpy() for m in REDES}}

    # ---- drift de scores
    filas = []
    for m in TODOS:
        a, b, c = P_nov[m], P_dic_oct[m], P_dic[m]
        filas.append(dict(
            modelo=m, tasa_real_nov=nov.y.mean(), media_nov=a.mean(), media_dic_mismo_modelo=b.mean(),
            media_dic_final=c.mean(), psi_mismo_modelo=psi(a, b), psi_modelo_final=psi(a, c),
            psi_mismo_modelo_nuevos=psi(a[nuevo_nov], b[nuevo_dic]),
            psi_mismo_modelo_antiguos=psi(a[~nuevo_nov], b[~nuevo_dic]),
            media_nov_nuevos=a[nuevo_nov].mean(), media_nov_antiguos=a[~nuevo_nov].mean(),
            media_dic_final_nuevos=c[nuevo_dic].mean(), media_dic_final_antiguos=c[~nuevo_dic].mean()))
    filas.append(dict(modelo="submission_final", tasa_real_nov=nov.y.mean(), media_dic_final=s.mean(),
                      psi_modelo_final=psi(P_nov[REF], s),
                      media_dic_final_nuevos=s[nuevo_dic].mean(), media_dic_final_antiguos=s[~nuevo_dic].mean()))
    drift = pd.DataFrame(filas)
    drift.to_csv(DIR_RES / "drift_predicciones.csv", index=False)

    # ---- Spearman entre modelos
    S_nov = matriz(P_nov)
    S_dic = matriz({**P_dic, "submission_final": s})
    S_dic_oct = matriz({**P_dic_oct, "submission_final": s})
    S_nov.to_csv(DIR_RES / "ranking_spearman_nov.csv")
    S_dic.to_csv(DIR_RES / "ranking_spearman_dic.csv")
    S_dic_oct.to_csv(DIR_RES / "ranking_spearman_dic_modelos_oct.csv")
    res = []
    for m in TODOS:
        otros = [o for o in TODOS if o != m]
        f = dict(modelo=m, spearman_medio_otros_nov=S_nov.loc[m, otros].mean(),
                 spearman_medio_otros_dic_final=S_dic.loc[m, otros].mean(),
                 spearman_medio_otros_dic_modelos_oct=S_dic_oct.loc[m, otros].mean(),
                 spearman_lgb_reg_nov=S_nov.loc[m, REF], spearman_lgb_reg_dic_final=S_dic.loc[m, REF],
                 spearman_submission_final_dic_final=S_dic.loc[m, "submission_final"],
                 spearman_submission_final_dic_modelos_oct=S_dic_oct.loc[m, "submission_final"])
        f["delta_spearman_otros"] = f["spearman_medio_otros_dic_final"] - f["spearman_medio_otros_nov"]
        if m in ARBOLES:
            sn = np.mean([sp(nov[f"{m}__s{a}"], nov[f"{m}__s{b}"]) for a, b in combinations(SEMILLAS, 2)])
            sd = np.mean([sp(fin[f"{m}__s{a}"], fin[f"{m}__s{b}"]) for a, b in combinations(SEMILLAS, 2)])
            f.update(spearman_semillas_nov=sn, spearman_semillas_dic=sd, delta_spearman_semillas=sd - sn)
        res.append(f)
    res = pd.DataFrame(res)
    res.to_csv(DIR_RES / "ranking_resumen.csv", index=False)

    # ---- solapamiento top-k
    tk = []
    for m in TODOS:
        for q in (0.10, 0.20):
            tk.append(dict(modelo=m, top=f"{int(q * 100)}%",
                           solape_con_lgb_reg_nov=solape(P_nov[m], P_nov[REF], q),
                           solape_con_lgb_reg_dic_final=solape(P_dic[m], P_dic[REF], q),
                           solape_con_submission_final_dic=solape(P_dic[m], s, q)))
    tk = pd.DataFrame(tk)
    tk.to_csv(DIR_RES / "ranking_topk.csv", index=False)

    # ---- dependencia de dui
    dep = []
    for m in TODOS:
        src = fin if m in ARBOLES else nnd
        perm_nov = [nov[f"{m}__perm{k}"].to_numpy() for k in range(3)]
        perm_dic = [src[f"{m}__perm{k}"].to_numpy() for k in range(3)]
        dep.append(dict(modelo=m,
                        spearman_pred_dui_nov=sp(P_nov[m], nov.dui), spearman_pred_dui_dic=sp(P_dic[m], fin.dui),
                        spearman_pred_vs_perm_dui_nov=np.mean([sp(P_nov[m], p) for p in perm_nov]),
                        spearman_pred_vs_perm_dui_dic=np.mean([sp(P_dic[m], p) for p in perm_dic]),
                        mad_pred_perm_dic=np.mean([np.abs(P_dic[m] - p).mean() for p in perm_dic])))
    dep = pd.DataFrame(dep)
    dep.to_csv(DIR_RES / "dependencia_dui.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for t in (drift, res, S_nov, S_dic, tk, dep):
        print(t.round(4).to_string()); print()


if __name__ == "__main__":
    main()
