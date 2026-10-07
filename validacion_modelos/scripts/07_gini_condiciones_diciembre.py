# -*- coding: utf-8 -*-
"""
07. Gini esperado en condiciones de diciembre, estimado sobre los folds ago-nov (honestos).

Escenarios (cada uno por fold y media de folds):
  a_tal_cual      : predicciones honestas, sin pesos
  b_perm_dui      : dui permutada dentro del mes al predecir (media de 3 permutaciones) -> en dic
                    dui es una permutacion exacta de dut (sin senal)
  c_adversarial   : pesos p/(1-p) del adversario ago-nov vs dic sin dui (05), recortados p1-p99
  d_nuevos_18_6   : pesos para que los clientes nuevos sean el 18,6 % de cada mes (como dic)
  e_combinado     : b + c + d (escenario "diciembre"; criterio C2 del veredicto)
  f_perm_y_nuevos : b + d (sin reponderacion adversarial; analisis de sensibilidad)
Placebo de (c): 5 adversarios con la etiqueta dic/no-dic barajada -> distribucion del cambio de
Gini que produce reponderar con un adversario SIN senal (sirve para interpretar c y e).
Bootstrap de clientes con reemplazo (1000 replicas comunes) -> IC95 % y SE PAREADO de
(escenario - tal cual) y de (modelo - lgb_reg) dentro de cada escenario.
Redes: solo sep-nov (ago dentro de su entrenamiento); se reportan tambien para lgb_reg en sep-nov.

Requiere: 01, 03, 05.  Salidas: gini_dic_escenarios.csv, gini_dic_por_fold.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from comun import DIR_RES, MESES_FOLD, N_BOOT, PROP_NUEVOS_DIC, SEMILLA, GiniRapido, pesos_bootstrap_clientes

ARBOLES = ["lgb_reg", "lgb_sin_reg", "xgb", "catboost", "random_forest", "lgb_reg_sin_dui"]
REDES = ["mlp_sklearn", "nn_baseline", "nn_optimizado"]
REF = "lgb_reg"
ESC = ["a_tal_cual", "b_perm_dui", "c_adversarial", "d_nuevos_18_6", "e_combinado", "f_perm_y_nuevos"]
N_PLACEBO = 5


def placebo(wf, calc, folds):
    """Reponderacion con adversarios placebo (etiqueta barajada): cambio de Gini (c - a) por modelo."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold
    from comun import COL_DUI, MES_TEST, cargar_todo
    _, d, _, _ = cargar_todo()
    cols = [c for c in d.X_train.columns if c != COL_DUI]
    ref = np.isin(d.mes_train, MESES_FOLD)
    X = pd.concat([d.X_train.loc[ref, cols], d.X_test[cols]], ignore_index=True)
    g = np.r_[d.id_train[ref], d.id_test]
    clave = pd.DataFrame({"id_cliente": d.id_train[ref], "mes": d.mes_train[ref]})
    et_real = np.r_[np.zeros(ref.sum(), int), np.ones(len(d.X_test), int)]
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1)
    pos = clave.merge(wf[["id_cliente", "mes"]].reset_index(), on=["id_cliente", "mes"], how="left")["index"].to_numpy()
    filas = []
    for r in range(N_PLACEBO):
        rng = np.random.default_rng(SEMILLA + 500 + r)
        et = rng.permutation(et_real)
        oof = np.zeros(len(X))
        for k, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA + r).split(X, et, g)):
            oof[b] = lgb.LGBMClassifier(**params, random_state=SEMILLA + k).fit(X.iloc[a], et[a]).predict_proba(X.iloc[b])[:, 1]
        p = np.clip(oof[:ref.sum()], 1e-6, 1 - 1e-6); w = p / (1 - p)
        lo, hi = np.percentile(w, [1, 99]); w = np.clip(w, lo, hi)
        wfull = np.empty(len(wf)); wfull[pos] = w
        for M, idx in folds.items():
            wfull[idx] /= wfull[idx].mean()
        for m in ARBOLES:
            cambio = np.mean([calc[(m, M, "hon")](wfull[idx]) - calc[(m, M, "hon")]() for M, idx in folds.items()])
            filas.append(dict(placebo=r, modelo=m, cambio_gini_c_menos_a=cambio))
    t = pd.DataFrame(filas)
    t.to_csv(DIR_RES / "gini_dic_placebo_adversarial.csv", index=False)
    print("Placebo de la reponderacion adversarial (cambio medio de Gini ago-nov, c - a):")
    print(t.pivot(index="modelo", columns="placebo", values="cambio_gini_c_menos_a").round(4))


def main():
    wf = pd.read_csv(DIR_RES / "wf_predicciones.csv")
    nn = pd.read_csv(DIR_RES / "nn_predicciones.csv")
    pw = pd.read_csv(DIR_RES / "pesos_adversariales.csv")
    wf = wf.merge(nn.drop(columns="y"), on=["id_cliente", "mes"], how="left", validate="1:1")
    wf = wf.merge(pw[["id_cliente", "mes", "w_adv"]], on=["id_cliente", "mes"], how="left", validate="1:1")
    assert wf.w_adv.notna().all()
    mes = wf.mes.to_numpy(); y = wf.y.to_numpy(); ids = wf.id_cliente.to_numpy()
    nuevo = wf.nuevo.to_numpy().astype(bool)
    w_nuevo = np.empty(len(wf))
    for M in MESES_FOLD:
        i = mes == M; pn = nuevo[i].mean()
        w_nuevo[i] = np.where(nuevo[i], PROP_NUEVOS_DIC / pn, (1 - PROP_NUEVOS_DIC) / (1 - pn))
    w_adv = wf.w_adv.to_numpy()
    w_esc = {"a_tal_cual": None, "b_perm_dui": None, "c_adversarial": w_adv, "d_nuevos_18_6": w_nuevo,
             "e_combinado": w_adv * w_nuevo, "f_perm_y_nuevos": w_nuevo}
    usa_perm = {"b_perm_dui", "e_combinado", "f_perm_y_nuevos"}
    folds = {M: np.where(mes == M)[0] for M in MESES_FOLD}
    modelos = ARBOLES + REDES

    calc = {}
    for m in modelos:
        for M, idx in folds.items():
            calc[(m, M, "hon")] = GiniRapido(y[idx], wf[f"{m}__hon"].to_numpy()[idx])
            for k in range(3):
                calc[(m, M, f"perm{k}")] = GiniRapido(y[idx], wf[f"{m}__perm{k}"].to_numpy()[idx])

    def evaluar(m, M, e, wb):
        idx = folds[M]
        w = wb[idx] if wb is not None else np.ones(len(idx))
        if w_esc[e] is not None:
            w = w * w_esc[e][idx]
        if e in usa_perm:
            return np.mean([calc[(m, M, f"perm{k}")](w) for k in range(3)])
        return calc[(m, M, "hon")](w)

    punt = {(m, M, e): evaluar(m, M, e, None) for m in modelos for M in MESES_FOLD for e in ESC}
    B = {k: np.empty(N_BOOT) for k in punt}
    for b, wb in enumerate(pesos_bootstrap_clientes(ids, N_BOOT, SEMILLA)):
        for (m, M, e) in punt:
            B[(m, M, e)][b] = evaluar(m, M, e, wb)

    pd.DataFrame([dict(modelo=m, fold=M, escenario=e, gini=v) for (m, M, e), v in punt.items()]) \
        .pivot_table(index=["modelo", "fold"], columns="escenario", values="gini").reset_index() \
        .to_csv(DIR_RES / "gini_dic_por_fold.csv", index=False)

    filas = []
    for m in modelos:
        for nombre_set, fs in (("ago-nov", MESES_FOLD), ("sep-nov", MESES_FOLD[1:])):
            if m in REDES and nombre_set == "ago-nov":
                continue
            media = lambda mm, e: np.mean([punt[(mm, M, e)] for M in fs])
            bmed = lambda mm, e: np.mean([B[(mm, M, e)] for M in fs], axis=0)
            base_b = bmed(m, "a_tal_cual")
            for e in ESC:
                bb = bmed(m, e); cambio = bb - base_b
                f = dict(modelo=m, folds=nombre_set, escenario=e, gini=media(m, e), se=bb.std(ddof=1),
                         ic95_inf=np.percentile(bb, 2.5), ic95_sup=np.percentile(bb, 97.5),
                         cambio_vs_tal_cual=media(m, e) - media(m, "a_tal_cual"),
                         se_pareado_cambio=cambio.std(ddof=1),
                         cambio_ic95_inf=np.percentile(cambio, 2.5), cambio_ic95_sup=np.percentile(cambio, 97.5),
                         folds_con_caida=int(sum(punt[(m, M, e)] < punt[(m, M, "a_tal_cual")] for M in fs)))
                if m != REF:
                    dv = bb - bmed(REF, e)
                    f.update(dif_vs_lgb_reg=media(m, e) - media(REF, e), se_dif_vs_lgb_reg=dv.std(ddof=1),
                             dif_ic95_inf=np.percentile(dv, 2.5), dif_ic95_sup=np.percentile(dv, 97.5),
                             folds_mejor_que_lgb_reg=int(sum(punt[(m, M, e)] > punt[(REF, M, e)] for M in fs)))
                filas.append(f)
    res = pd.DataFrame(filas)
    res.to_csv(DIR_RES / "gini_dic_escenarios.csv", index=False)
    ess = {M: (lambda w: w.sum() ** 2 / (w ** 2).sum() / len(w))(w_esc["e_combinado"][idx]) for M, idx in folds.items()}
    placebo(wf, calc, folds)
    print("Tamano efectivo relativo (pesos combinados):", {k: round(v, 3) for k, v in ess.items()})
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
