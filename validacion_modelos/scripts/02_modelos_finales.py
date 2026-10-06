# -*- coding: utf-8 -*-
"""
02. Modelos finales (todo train ene-nov) y prediccion de diciembre.

n_iter final = mediana sobre los 4 folds de  best_iter_k * filas(train completo) / filas(train del ES_k)
(best_iter_k sale del early stopping honesto de 01_walkforward.py: entrenado < M-1, ES sobre M-1).
Se promedian 3 semillas (42, 43, 44). RandomForest: 300 arboles fijos.
Ademas, para medir la dependencia de dias_ultima_interaccion (dui) en diciembre, se predice
diciembre con dui permutada (3 permutaciones).

Requiere: resultados/wf_iteraciones.csv
Salidas: resultados/final_pred_dic.csv, resultados/final_iteraciones.csv
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from comun import COL_DUI, DIR_RES, SEMILLA, SEMILLAS, cargar_todo
from modelos import ajustar_fijo, definir_modelos, predecir

N_PERM = 3
SOLO = sys.argv[1:]


def main():
    t0 = time.time()
    wcb, d, tr, te = cargar_todo()
    X, y = d.X_train, d.y_train.to_numpy()
    it = pd.read_csv(DIR_RES / "wf_iteraciones.csv")
    modelos = definir_modelos(wcb, X.columns)
    if SOLO:
        modelos = {k: v for k, v in modelos.items() if k in SOLO}
    rng = np.random.default_rng(SEMILLA + 200)
    perms = [rng.permutation(len(d.X_test)) for _ in range(N_PERM)]
    out = {"id_cliente": d.id_test, "dui": d.X_test[COL_DUI].to_numpy()}
    filas = []
    for nombre, spec in modelos.items():
        fam, params, cols = spec["familia"], spec["params"], spec["cols"]
        r = it[it.modelo == nombre]
        escalado = r.best_iter_es_Mm1 * len(X) / r.filas_train_es
        n_iter = int(params["n_estimators"]) if fam == "random_forest" else max(10, int(round(np.median(escalado))))
        Xt = d.X_test[cols]
        preds, preds_perm = [], [[] for _ in range(N_PERM)]
        for s in SEMILLAS:
            m = ajustar_fijo(wcb, fam, params, n_iter, X[cols], y, s)
            p = predecir(m, Xt); preds.append(p); out[f"{nombre}__s{s}"] = p
            for k in range(N_PERM):
                Xp = Xt.copy()
                if COL_DUI in cols:
                    Xp[COL_DUI] = Xt[COL_DUI].to_numpy()[perms[k]]
                preds_perm[k].append(predecir(m, Xp))
        out[f"{nombre}__final"] = np.mean(preds, axis=0)
        for k in range(N_PERM):
            out[f"{nombre}__perm{k}"] = np.mean(preds_perm[k], axis=0)
        filas.append(dict(modelo=nombre, n_iter_final=n_iter,
                          best_iter_folds=list(r.best_iter_es_Mm1),
                          escalados=[round(v, 1) for v in escalado]))
        print(f"{nombre:16s} n_iter_final={n_iter} ({time.time()-t0:.0f}s)", flush=True)
    suf = "" if not SOLO else "_" + "_".join(SOLO)
    pd.DataFrame(out).to_csv(DIR_RES / f"final_pred_dic{suf}.csv", index=False)
    pd.DataFrame(filas).to_csv(DIR_RES / f"final_iteraciones{suf}.csv", index=False)
    print(f"Listo en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
