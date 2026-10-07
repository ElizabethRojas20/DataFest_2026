# -*- coding: utf-8 -*-
"""
21. Registro unificado de candidatos (base de las pruebas de sobreoptimización 22-23).

Para cada candidato de sobreopt_comun.candidatos() calcula las predicciones HONESTAS de los folds
ago-nov con el esquema del pipeline nuevo (EvaluadorModelos: selección de variables con meses < ago
como en main(), semilla 42, sin early stopping, nº de árboles de cada fold con la regla 1-SE sobre
los otros 3 folds). El EBM no tiene rejilla de árboles: se ajusta con < M y su ES interno.
Todas las predicciones son sobre las MISMAS filas, en el mismo orden.

Salidas: resultados/sobreopt_registro.npz (P[candidato, fila], y, ids, mes, nuevo, nombres)
         resultados/sobreopt_registro.csv (Gini por fold y medio, árboles, grupo, parámetros)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from comun import DIR_RES, MESES_FOLD, RAIZ, SEMILLA, GiniRapido, cargar_todo, es_nuevo_train
from sobreopt_comun import ajustar_ebm, candidatos


def main():
    t0 = time.time()
    wcb, d, _, _ = cargar_todo()
    X, y, mes, ids = d.X_train, d.y_train, d.mes_train, d.id_train
    meses = np.sort(np.unique(mes))
    cfg = wcb.Config(ruta_datos=RAIZ / "datos_entrada", n_folds=len(MESES_FOLD))
    sel = mes < meses[-cfg.n_folds]
    orden_val = np.concatenate([np.flatnonzero(mes == M) for M in MESES_FOLD])   # orden de EvaluadorModelos
    cands = candidatos(wcb, X.columns)

    orden_cols, evaluadores = {}, {}
    P, filas = [], []
    for nombre, c in cands.items():
        clave = tuple(c["cols"])
        if clave not in orden_cols:          # misma selección/orden de variables que main()
            orden_cols[clave] = wcb.seleccionar_features(X.loc[sel, c["cols"]], y.loc[sel], cfg)
            evaluadores[clave] = wcb.EvaluadorModelos(X[orden_cols[clave]], y, mes, ids, cfg)
        cols = orden_cols[clave]
        if c["familia"] == "ebm":
            p = np.empty(len(orden_val))
            ini = 0
            for M in MESES_FOLD:
                tr, va = mes < M, mes == M
                m = ajustar_ebm(c["params"], X.loc[tr, cols], y.loc[tr], SEMILLA)
                p[ini:ini + va.sum()] = m.predict_proba(X.loc[va, cols])[:, 1]
                ini += va.sum()
            arboles = "ES interno"
        else:
            ev = evaluadores[clave]
            ev.val_predictions_ = []
            r = ev._evaluar_modelo_cv(c["familia"], c["params"], nombre)
            p = pd.concat(ev.val_predictions_, ignore_index=True).y_prob.to_numpy()
            arboles = "/".join(map(str, r["arboles_folds"]))
        ginis = []
        ini = 0
        for M in MESES_FOLD:
            n = int((mes == M).sum())
            ginis.append(GiniRapido(y.to_numpy()[orden_val][ini:ini + n], p[ini:ini + n])())
            ini += n
        P.append(p.astype(np.float64))
        filas.append(dict(nombre=nombre, grupo=c["grupo"], familia=c["familia"], n_cols=len(cols),
                          con_dui="dias_ultima_interaccion" in cols, arboles_folds=arboles,
                          **{f"gini_{M}": g for M, g in zip(MESES_FOLD, ginis)},
                          gini_medio=float(np.mean(ginis)), params=json.dumps(c["params"])))
        print(f"{nombre:<22} Gini medio {np.mean(ginis):.4f} | {arboles} ({time.time() - t0:.0f}s)", flush=True)

    reg = pd.DataFrame(filas)
    reg.to_csv(DIR_RES / "sobreopt_registro.csv", index=False)
    np.savez_compressed(DIR_RES / "sobreopt_registro.npz", P=np.vstack(P), nombres=np.array(reg.nombre),
                        y=y.to_numpy()[orden_val], ids=ids[orden_val], mes=mes[orden_val],
                        nuevo=es_nuevo_train(ids, mes)[orden_val])
    print(reg.sort_values("gini_medio", ascending=False)[["nombre", "grupo", "gini_medio", "arboles_folds"]]
          .round(4).to_string(index=False))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
