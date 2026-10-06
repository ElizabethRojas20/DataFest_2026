# -*- coding: utf-8 -*-
"""
05. Drift de variables entre train y diciembre.

* PSI por variable (22 columnas originales): nov -> dic y ene-nov -> dic (cuantiles de la referencia).
* Ruptura de dias_ultima_interaccion (dui): % de filas con dui == dias_ultima_transaccion (dut)
  y correlacion dui-dut por mes; % de clientes nuevos por mes.
* Validacion adversarial (LightGBM, StratifiedGroupKFold por cliente, 5 folds, AUC OOF):
  comparaciones nov vs dic, ene-nov vs dic y ago-nov vs dic; variantes de variables
  'sin_dui', 'con_dui' y 'con_dui_e_indicador' (dui == dut).
* Pesos de reponderacion para 07: p/(1-p) del adversario ago-nov vs dic SIN dui (covariate shift
  de las variables estaticas), recortados a los percentiles 1-99 y normalizados a media 1 por mes.

Salidas: drift_psi_variables.csv, drift_dui_por_mes.csv, adversarial_auc.csv,
         adversarial_importancia.csv, pesos_adversariales.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from comun import COL_DUI, COL_DUT, DIR_RES, MESES_FOLD, MES_TEST, RAIZ, SEMILLA, cargar_todo, es_nuevo_train, psi

PARAMS_ADV = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  n_jobs=-1, verbose=-1)


def adversarial(X, etiqueta, grupos, seed=SEMILLA):
    oof = np.zeros(len(X)); imp = np.zeros(X.shape[1])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for k, (a, b) in enumerate(cv.split(X, etiqueta, grupos)):
        m = lgb.LGBMClassifier(**PARAMS_ADV, random_state=seed + k, importance_type="gain")
        m.fit(X.iloc[a], etiqueta[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
        imp += m.feature_importances_ / 5
    return roc_auc_score(etiqueta, oof), oof, pd.Series(imp / imp.sum(), index=X.columns)


def main():
    wcb, d, tr, te = cargar_todo()
    raw_tr = pd.read_csv(RAIZ / "datos_entrada" / "train.csv")
    raw_te = pd.read_csv(RAIZ / "datos_entrada" / "test.csv")
    cfg = wcb.Config()
    num, boo, cat = cfg.cols_numericas, cfg.cols_booleanas, cfg.cols_categoricas

    # --- PSI por variable
    filas = []
    nov = raw_tr[raw_tr.mes == 202611]
    for c in num + boo + cat:
        es_cat = c in boo + cat
        filas.append(dict(variable=c, tipo="categorica" if es_cat else "numerica",
                          psi_nov_dic=psi(nov[c].to_numpy(), raw_te[c].to_numpy(), categorica=es_cat),
                          psi_ene_nov_dic=psi(raw_tr[c].to_numpy(), raw_te[c].to_numpy(), categorica=es_cat),
                          media_ene_nov=raw_tr[c].astype(float).mean() if not c in cat else np.nan,
                          media_nov=nov[c].astype(float).mean() if not c in cat else np.nan,
                          media_dic=raw_te[c].astype(float).mean() if not c in cat else np.nan))
    # PSI de la variable derivada "dui == dut"
    ind_tr = (raw_tr[COL_DUI] == raw_tr[COL_DUT]).to_numpy().astype(int)
    ind_te = (raw_te[COL_DUI] == raw_te[COL_DUT]).to_numpy().astype(int)
    filas.append(dict(variable="indicador_dui_igual_dut", tipo="derivada",
                      psi_nov_dic=psi(ind_tr[raw_tr.mes == 202611], ind_te, categorica=True),
                      psi_ene_nov_dic=psi(ind_tr, ind_te, categorica=True),
                      media_ene_nov=ind_tr.mean(), media_nov=ind_tr[raw_tr.mes == 202611].mean(), media_dic=ind_te.mean()))
    psi_tab = pd.DataFrame(filas).sort_values("psi_nov_dic", ascending=False)
    psi_tab.to_csv(DIR_RES / "drift_psi_variables.csv", index=False)

    # --- dui por mes
    todo = pd.concat([raw_tr.assign(nuevo=es_nuevo_train(raw_tr.id_cliente.to_numpy(), raw_tr.mes.to_numpy())),
                      raw_te.assign(nuevo=~raw_te.id_cliente.isin(raw_tr.id_cliente))], ignore_index=True)
    dm = todo.groupby("mes").apply(lambda g: pd.Series(dict(
        filas=len(g), pct_dui_igual_dut=100 * (g[COL_DUI] == g[COL_DUT]).mean(),
        corr_dui_dut=np.corrcoef(g[COL_DUI], g[COL_DUT])[0, 1],
        pct_nuevos=100 * g.nuevo.mean(), tasa_conversion=g.objetivo.mean() if g.objetivo.notna().any() else np.nan)),
        include_groups=False).reset_index()
    dm.to_csv(DIR_RES / "drift_dui_por_mes.csv", index=False)

    # --- validacion adversarial
    Xall = pd.concat([d.X_train, d.X_test], ignore_index=True)
    mes_all = np.r_[d.mes_train, np.full(len(d.X_test), MES_TEST)]
    id_all = np.r_[d.id_train, d.id_test]
    Xall["indicador_dui_igual_dut"] = (Xall[COL_DUI] == Xall[COL_DUT]).astype("int8")
    base_cols = list(d.X_train.columns)
    variantes = {"sin_dui": [c for c in base_cols if c != COL_DUI], "con_dui": base_cols,
                 "con_dui_e_indicador": base_cols + ["indicador_dui_igual_dut"]}
    comparaciones = {"nov_vs_dic": mes_all >= 202611, "ene-nov_vs_dic": mes_all > 0,
                     "ago-nov_vs_dic": mes_all >= MESES_FOLD[0]}
    res, imps = [], []
    for nc, msk in comparaciones.items():
        idx = np.where(msk)[0]
        et = (mes_all[idx] == MES_TEST).astype(int)
        for nv, cols in variantes.items():
            auc, oof, imp = adversarial(Xall.iloc[idx][cols].reset_index(drop=True), et, id_all[idx])
            res.append(dict(comparacion=nc, variables=nv, auc_adversarial=auc, n_ref=int((et == 0).sum()), n_dic=int(et.sum()),
                            top5=", ".join(f"{k} ({v:.2f})" for k, v in imp.sort_values(ascending=False).head(5).items())))
            imps.append(imp.rename(f"{nc}|{nv}"))
            print(f"{nc:15s} {nv:22s} AUC={auc:.4f}", flush=True)
            if nc == "ago-nov_vs_dic" and nv == "sin_dui":
                p = np.clip(oof[et == 0], 1e-6, 1 - 1e-6)
                w = p / (1 - p)
                lo, hi = np.percentile(w, [1, 99]); w = np.clip(w, lo, hi)
                pesos = pd.DataFrame({"id_cliente": id_all[idx][et == 0], "mes": mes_all[idx][et == 0],
                                      "p_dic": p, "w_adv": w})
                pesos["w_adv"] = pesos.w_adv / pesos.groupby("mes").w_adv.transform("mean")
                pesos.to_csv(DIR_RES / "pesos_adversariales.csv", index=False)
                ess = pesos.groupby("mes").w_adv.apply(lambda v: v.sum() ** 2 / (v ** 2).sum() / len(v))
                print("Tamano muestral efectivo relativo por mes (pesos adversariales):", ess.round(3).to_dict())
    pd.DataFrame(res).to_csv(DIR_RES / "adversarial_auc.csv", index=False)
    pd.concat(imps, axis=1).to_csv(DIR_RES / "adversarial_importancia.csv")

    pd.set_option("display.width", 250)
    print(psi_tab.round(4).to_string(index=False))
    print(dm.round(3).to_string(index=False))
    print(pd.DataFrame(res).drop(columns="top5").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
