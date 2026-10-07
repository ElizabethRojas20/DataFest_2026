# -*- coding: utf-8 -*-
"""
09. R1 (entrenamiento): walk-forward honesto de los 5 candidatos (criterios_robustez.py).
Folds ago-nov; boosting: ES en M-1 entrenando < M-1, reentreno < M con best*factor, semillas 42-44.
EBM: ajuste directo con < M (ES interno aleatorio dentro del train del fold), semillas 42-44.
Para los modelos con dui se predice tambien con dui permutada dentro del mes (3 permutaciones):
escenario "condiciones de diciembre".
Salidas: resultados/robustez_r1_predicciones.csv (float32 sin redondear) y robustez_r1_predicciones.npz (float64, para metricas),
         resultados/robustez_r1_iteraciones.csv, cache_modelos/r1_<modelo>_<fold>.joblib (semilla 42)
Uso: python 09_r1_walkforward_candidatos.py [modelo ...]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib
import numpy as np
import pandas as pd

from robustez_comun import (CANDIDATOS, COL_DUI, CON_DUI, DIR_CACHE, DIR_RES, MESES_FOLD, SEMILLA,
                            SEMILLAS, cargar_todo, definir_candidatos, es_nuevo_train, fit_honesto,
                            gini, permutar_en_mes, pred, f32)

N_PERM = 3
SOLO = sys.argv[1:] or list(CANDIDATOS)


def main():
    t0 = time.time()
    wcb, d, tr, te = cargar_todo()
    X, y, mes = d.X_train, d.y_train.to_numpy(), d.mes_train
    specs = definir_candidatos(wcb, X.columns)
    es_val = np.isin(mes, MESES_FOLD)
    pos_val = np.where(es_val)[0]
    base = pd.DataFrame({"id_cliente": d.id_train[es_val], "mes": mes[es_val], "y": y[es_val],
                         "nuevo": es_nuevo_train(d.id_train, mes)[es_val],
                         "banda_riesgo": tr.loc[es_val, "banda_riesgo"].to_numpy(),
                         "numero_productos": tr.loc[es_val, "numero_productos"].to_numpy(),
                         "activo_movil": tr.loc[es_val, "activo_movil"].to_numpy()})
    perms = [permutar_en_mes(X[COL_DUI].to_numpy(), mes, seed=SEMILLA + 100 + k) for k in range(N_PERM)]
    out, filas_it = {}, []
    for nombre in SOLO:
        spec = specs[nombre]
        ck = DIR_CACHE / f"r1_parcial_{nombre}.joblib"      # punto de control por modelo
        if ck.exists():
            o, fi = joblib.load(ck); out.update(o); filas_it.extend(fi)
            print(f"{nombre}: recuperado del punto de control", flush=True)
            continue
        n_prev = len(filas_it)
        for M in MESES_FOLD:
            t1 = time.time()
            va = mes == M
            filas = np.searchsorted(pos_val, np.where(va)[0])
            mods, info = fit_honesto(wcb, spec, X, y, mes, M)
            Xv = X.loc[va, spec["cols"]]
            ps = [pred(m, Xv) for m in mods]
            for s, p in zip(SEMILLAS, ps):
                out.setdefault(f"{nombre}__s{s}", np.full(len(base), np.nan))[filas] = p
            out.setdefault(f"{nombre}__hon", np.full(len(base), np.nan))[filas] = np.mean(ps, axis=0)
            if nombre in CON_DUI:
                pp = []
                for k in range(N_PERM):
                    Xp = Xv.copy(); Xp[COL_DUI] = perms[k][va]
                    pp.append(np.mean([pred(m, Xp) for m in mods], axis=0))
                out.setdefault(f"{nombre}__perm", np.full(len(base), np.nan))[filas] = np.mean(pp, axis=0)
            joblib.dump(mods[0], DIR_CACHE / f"r1_{nombre}_{M}.joblib", compress=3)
            g = gini(y[va], np.mean(ps, axis=0))
            gs = [gini(y[va], p) for p in ps]
            filas_it.append(dict(modelo=nombre, fold=M, best_iter_es_Mm1=info["best"], factor=info["factor"],
                                 n_iter=info["n_iter"], filas_train_es=info["filas_es"],
                                 filas_train_M=info["filas_M"], gini_hon_3sem=g,
                                 gini_s42=gs[0], gini_s43=gs[1], gini_s44=gs[2], seg=time.time() - t1))
            print(f"{nombre:17s} {M} best={info['best']} n_iter={info['n_iter']} Gini={g:.4f} "
                  f"sem={np.round(gs, 4)} ({time.time()-t1:.0f}s)", flush=True)
        joblib.dump(({k: v for k, v in out.items() if k.startswith(nombre + "__")}, filas_it[n_prev:]), ck)
    suf = "" if len(SOLO) == len(CANDIDATOS) else "_" + "_".join(SOLO)
    pred_df = pd.concat([base, pd.DataFrame({k: f32(v) for k, v in out.items()})], axis=1)
    pred_df.to_csv(DIR_RES / f"robustez_r1_predicciones{suf}.csv", index=False)
    np.savez_compressed(DIR_RES / f"robustez_r1_predicciones{suf}.npz", **out)   # float64 para metricas
    pd.DataFrame(filas_it).to_csv(DIR_RES / f"robustez_r1_iteraciones{suf}.csv", index=False)
    print(f"Listo en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
