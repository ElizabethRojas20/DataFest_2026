# -*- coding: utf-8 -*-
"""
01. Walk-forward HONESTO vs esquema del pipeline (folds ago, sep, oct, nov de 2026).

Esquema honesto (fold M):
  1) entrenar con meses < M-1 y hacer early stopping sobre M-1 -> best_iter (semilla 42)
  2) n_iter = round(best_iter * filas(<M) / filas(<M-1))
  3) reentrenar con TODOS los meses < M con n_iter fijo, 3 semillas (42, 43, 44)
  4) evaluar en M: (a) tal cual y (b) con dias_ultima_interaccion permutada dentro del mes
     (3 permutaciones distintas) -> simula diciembre
  5) en el fold de noviembre (modelo entrenado <= oct) se predice tambien diciembre
Esquema del pipeline (para cuantificar el optimismo): entrenar con < M, early stopping sobre M
y evaluar en el mismo M (semilla 42).

Uso: python 01_walkforward.py [modelo1 modelo2 ...]   (sin argumentos = todos)
Salidas (validacion_modelos/resultados/, sufijo _<modelos> si se pasan argumentos):
  wf_predicciones.csv          una fila por fila de validacion (ago-nov), columnas por modelo
  wf_pred_dic_modelo_oct.csv   predicciones de dic de los modelos del fold nov (<= oct)
  wf_iteraciones.csv           best_iter / factor / n_iter por modelo y fold
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from comun import (COL_DUI, COL_DUT, DIR_RES, MESES_FOLD, SEMILLA, SEMILLAS, cargar_todo,
                   es_nuevo_train, gini, permutar_en_mes)
from modelos import ajustar_con_es, ajustar_fijo, definir_modelos, predecir

N_PERM = 3
SOLO = sys.argv[1:]


def main():
    t0 = time.time()
    wcb, d, tr, te = cargar_todo()
    X, y, mes = d.X_train, d.y_train.to_numpy(), d.mes_train
    nuevo = es_nuevo_train(d.id_train, mes)
    meses = np.sort(np.unique(mes))
    modelos = definir_modelos(wcb, X.columns)
    if SOLO:
        modelos = {k: v for k, v in modelos.items() if k in SOLO}

    es_val = np.isin(mes, MESES_FOLD)
    pos_val = np.where(es_val)[0]
    cols_out = {}
    base = pd.DataFrame({"id_cliente": d.id_train[es_val], "mes": mes[es_val], "y": y[es_val],
                         "nuevo": nuevo[es_val], "dui": X.loc[es_val, COL_DUI].to_numpy(),
                         "dut": X.loc[es_val, COL_DUT].to_numpy()})
    dic = {"id_cliente": d.id_test}
    perms = {k: permutar_en_mes(X[COL_DUI].to_numpy(), mes, seed=SEMILLA + 100 + k) for k in range(N_PERM)}

    def guardar(col, filas, valores):
        if col not in cols_out:
            cols_out[col] = np.full(len(base), np.nan)
        cols_out[col][filas] = valores

    filas_iter = []
    for nombre, spec in modelos.items():
        fam, params, cols = spec["familia"], spec["params"], spec["cols"]
        for M in MESES_FOLD:
            i = int(np.searchsorted(meses, M)); Mm1 = meses[i - 1]
            tr_es, va_es = mes < Mm1, mes == Mm1
            tr_M, va_M = mes < M, mes == M
            filas = np.searchsorted(pos_val, np.where(va_M)[0])
            t1 = time.time()
            _, best = ajustar_con_es(wcb, fam, params, X.loc[tr_es, cols], y[tr_es],
                                     X.loc[va_es, cols], y[va_es], SEMILLA)
            factor = tr_M.sum() / tr_es.sum()
            n_iter = best if fam == "random_forest" else max(10, int(round(best * factor)))
            Xv = X.loc[va_M, cols]
            Xv_perm = []
            for k in range(N_PERM):
                Xp = Xv.copy()
                if COL_DUI in cols:
                    Xp[COL_DUI] = perms[k][va_M]
                Xv_perm.append(Xp)
            preds, preds_perm, preds_dic = [], [[] for _ in range(N_PERM)], []
            for s in SEMILLAS:
                m = ajustar_fijo(wcb, fam, params, n_iter, X.loc[tr_M, cols], y[tr_M], s)
                p = predecir(m, Xv); preds.append(p)
                guardar(f"{nombre}__s{s}", filas, p)
                for k in range(N_PERM):
                    preds_perm[k].append(predecir(m, Xv_perm[k]))
                if M == MESES_FOLD[-1]:
                    pdic = predecir(m, d.X_test[cols]); preds_dic.append(pdic)
                    dic[f"{nombre}__s{s}"] = pdic
            guardar(f"{nombre}__hon", filas, np.mean(preds, axis=0))
            for k in range(N_PERM):
                guardar(f"{nombre}__perm{k}", filas, np.mean(preds_perm[k], axis=0))
            if preds_dic:
                dic[f"{nombre}__hon"] = np.mean(preds_dic, axis=0)
            if fam == "random_forest":
                best_pipe = best
                guardar(f"{nombre}__pipe", filas, preds[0])   # RF no usa ES: igual al honesto (semilla 42)
            else:
                mp, best_pipe = ajustar_con_es(wcb, fam, params, X.loc[tr_M, cols], y[tr_M], Xv, y[va_M], SEMILLA)
                guardar(f"{nombre}__pipe", filas, predecir(mp, Xv))
            g_h = gini(y[va_M], cols_out[f"{nombre}__hon"][filas])
            g_p = gini(y[va_M], cols_out[f"{nombre}__pipe"][filas])
            g_s42 = gini(y[va_M], cols_out[f"{nombre}__s{SEMILLA}"][filas])
            filas_iter.append(dict(modelo=nombre, fold=M, mes_es=Mm1, best_iter_es_Mm1=best,
                                   factor=factor, n_iter=n_iter, best_iter_pipeline=best_pipe,
                                   filas_train_es=int(tr_es.sum()), filas_train_M=int(tr_M.sum()),
                                   gini_honesto_3sem=g_h, gini_honesto_s42=g_s42, gini_pipeline_s42=g_p,
                                   seg=time.time() - t1))
            print(f"{nombre:16s} {M} best={best:5d} n_iter={n_iter:5d} pipe_iter={best_pipe:5d} "
                  f"Gini hon={g_h:.4f} hon42={g_s42:.4f} pipe={g_p:.4f} ({time.time()-t1:.0f}s)", flush=True)

    base = pd.concat([base, pd.DataFrame(cols_out)], axis=1)
    suf = "" if not SOLO else "_" + "_".join(SOLO)
    base.to_csv(DIR_RES / f"wf_predicciones{suf}.csv", index=False)
    pd.DataFrame(dic).to_csv(DIR_RES / f"wf_pred_dic_modelo_oct{suf}.csv", index=False)
    pd.DataFrame(filas_iter).to_csv(DIR_RES / f"wf_iteraciones{suf}.csv", index=False)
    print(f"Listo en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
