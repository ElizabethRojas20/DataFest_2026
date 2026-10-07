# -*- coding: utf-8 -*-
"""
23. Pruebas de sobreoptimización que REENTRENAN.

  P1  Walk-forward anidado (Varma y Simon, 2006). Para cada mes externo M en ago-nov se repite la
      receta completa usando SOLO meses < M:
        * orden de variables: seleccionar_features con meses anteriores al primer fold interno;
        * cada candidato (sin EBM) se evalúa con EvaluadorModelos con los 3 meses anteriores a M
          como folds internos (Gini honesto interno y nº de árboles por regla 1-SE);
        * ganador = mayor Gini medio interno; se entrena con todo < M y n = mejor_iter * factor_datos_;
        * se evalúa una sola vez en M.
      Todos los candidatos se evalúan también en M (rango externo del ganador interno).
      Optimismo = Gini aparente del ganador del registro (elegido y medido en ago-nov)
                  - Gini anidado medio de la receta.
  P5  Placebo de la receta (permutación). Se rompe la relación variables-objetivo conservando el
      panel: los vectores de variables fijas se permutan entre CLIENTES (etiquetas, meses y
      trayectorias intactas) y dui se permuta dentro del mes. Se corre la receta del registro
      restringida a LightGBM + XGBoost (por coste) en ago-nov y se anota
      Δ = Gini(ganador) - Gini(referencia). Δ real comparable: el del registro (21) con el mismo
      subconjunto de candidatos.

Semilla 42 en todos los modelos. Uso:
    .venv/Scripts/python.exe validacion_modelos/scripts/23_anidado_placebo.py [anidado|placebo|todo] [replicas]
Salidas: resultados/sobreopt_anidado_{candidatos,resumen}.csv, resultados/sobreopt_placebo.csv
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from comun import COL_DUI, DIR_RES, MESES_FOLD, RAIZ, SEMILLA, GiniRapido, cargar_todo, permutar_en_mes
from sobreopt_comun import CRITERIOS_SO as C, GANADOR_PIPELINE, REF, SEMILLA_SO, candidatos


def evaluar_candidatos(wcb, cands, X, y, mes, ids, n_folds, orden_cols):
    """EvaluadorModelos (registrar=False) para cada candidato. Devuelve {nombre: (res, cols, factor)}."""
    cfg = wcb.Config(ruta_datos=RAIZ / "datos_entrada", n_folds=n_folds)
    evs, out = {}, {}
    for nombre, c in cands.items():
        cols = orden_cols[tuple(c["cols"])]
        if tuple(cols) not in evs:
            evs[tuple(cols)] = wcb.EvaluadorModelos(X[cols], y, mes, ids, cfg)
        ev = evs[tuple(cols)]
        out[nombre] = (ev._evaluar_modelo_cv(c["familia"], c["params"], nombre, registrar=False),
                       cols, ev.factor_datos_)
    return out


def anidado(wcb, d):
    t0 = time.time()
    X, y, mes, ids = d.X_train, d.y_train, d.mes_train, d.id_train
    meses = np.sort(np.unique(mes))
    cands = candidatos(wcb, X.columns, incluir_ebm=False)
    cfg_sel = wcb.Config(ruta_datos=RAIZ / "datos_entrada")
    filas = []
    for M in MESES_FOLD:
        pos = int(np.searchsorted(meses, M))
        primer_interno = meses[pos - 3]
        dentro = mes < M
        Xi, yi = X.loc[dentro].reset_index(drop=True), y.loc[dentro].reset_index(drop=True)
        mi, ii = mes[dentro], ids[dentro]
        sel = mi < primer_interno
        orden_cols = {}
        for c in cands.values():
            k = tuple(c["cols"])
            if k not in orden_cols:
                orden_cols[k] = wcb.seleccionar_features(Xi.loc[sel, c["cols"]], yi.loc[sel], cfg_sel)
        res = evaluar_candidatos(wcb, cands, Xi, yi, mi, ii, 3, orden_cols)
        va = mes == M
        for nombre, c in cands.items():
            r, cols, factor = res[nombre]
            n = max(10, int(round(r["mejor_iter"] * factor)))
            m = wcb.construir_modelo(c["familia"], c["params"], n, SEMILLA).fit(Xi[cols], yi)
            g_ext = GiniRapido(y.loc[va].to_numpy(), m.predict_proba(X.loc[va, cols])[:, 1])()
            filas.append(dict(mes_externo=M, nombre=nombre, gini_interno=r["gini"], n_final=n,
                              gini_externo=g_ext))
        print(f"anidado {M} listo ({time.time() - t0:.0f}s)", flush=True)
    t = pd.DataFrame(filas)
    t["ganador_interno"] = t.groupby("mes_externo").gini_interno.transform("max") == t.gini_interno
    t["rango_externo"] = t.groupby("mes_externo").gini_externo.rank(ascending=False)
    t.to_csv(DIR_RES / "sobreopt_anidado_candidatos.csv", index=False)

    reg = pd.read_csv(DIR_RES / "sobreopt_registro.csv")
    reg = reg[reg.nombre.isin(cands)]
    aparente = reg.gini_medio.max()
    gan = t[t.ganador_interno]
    fija = lambda n: t[t.nombre == n].set_index("mes_externo").gini_externo
    receta = gan.set_index("mes_externo").gini_externo
    res = pd.DataFrame(dict(ganador_interno=gan.set_index("mes_externo").nombre, receta=receta,
                            rango_externo_ganador=gan.set_index("mes_externo").rango_externo,
                            referencia=fija(REF), lgb_sup_sin_dui=fija(GANADOR_PIPELINE),
                            oraculo=t.groupby("mes_externo").gini_externo.max()))
    res["receta_menos_ref"] = res.receta - res.referencia
    res["sup_menos_ref"] = res.lgb_sup_sin_dui - res.referencia
    media = res.drop(columns="ganador_interno").mean().rename("media")
    res = pd.concat([res, media.to_frame().T])
    res.loc["media", "ganador_interno"] = f"{reg.loc[reg.gini_medio.idxmax(), 'nombre']} (aparente {aparente:.4f})"
    optimismo = aparente - media.receta
    res["optimismo_vs_aparente"] = np.nan
    res.loc["media", "optimismo_vs_aparente"] = optimismo
    res.to_csv(DIR_RES / "sobreopt_anidado_resumen.csv")
    print(res.round(4).to_string())
    print(f"Optimismo = {optimismo:.4f} (criterio < {C['optimismo_max']}) -> "
          f"{'PASA' if optimismo < C['optimismo_max'] and media.receta_menos_ref > 0 else 'NO PASA'}")


def placebo(wcb, d, replicas):
    t0 = time.time()
    X, y, mes, ids = d.X_train, d.y_train, d.mes_train, d.id_train
    cands = candidatos(wcb, X.columns, solo_rapidos=True)
    fijas = [c for c in X.columns if c != COL_DUI]
    uniq, inv = np.unique(ids, return_inverse=True)
    primera = pd.Series(np.arange(len(ids))).groupby(inv).first().to_numpy()       # fila 1 de cada cliente
    estatico = X[fijas].to_numpy()[primera]                                         # (clientes, vars)
    orden_cols = {tuple(c["cols"]): list(c["cols"]) for c in cands.values()}
    reg = pd.read_csv(DIR_RES / "sobreopt_registro.csv").set_index("nombre").loc[list(cands)]
    dif_real = reg.gini_medio.max() - reg.loc[REF, "gini_medio"]
    filas = []
    for b in range(replicas):
        rng = np.random.default_rng(SEMILLA_SO + b)
        donante = rng.permutation(len(uniq))
        Xp = pd.DataFrame(estatico[donante][inv], columns=fijas)
        Xp[COL_DUI] = permutar_en_mes(X[COL_DUI].to_numpy(), mes, SEMILLA_SO + b)
        Xp = Xp[list(X.columns)].astype("float32")
        res = evaluar_candidatos(wcb, cands, Xp, y, mes, ids, len(MESES_FOLD), orden_cols)
        g = pd.Series({n: r[0]["gini"] for n, r in res.items()})
        filas.append(dict(replica=b, ganador=g.idxmax(), gini_ganador=g.max(), gini_ref=g[REF],
                          dif_ganador_vs_ref=g.max() - g[REF], gini_medio_candidatos=g.mean()))
        print(f"placebo {b + 1}/{replicas}: Δ={filas[-1]['dif_ganador_vs_ref']:.4f} "
              f"ganador={filas[-1]['ganador']} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(filas).to_csv(DIR_RES / "sobreopt_placebo.csv", index=False)  # guardado incremental
    t = pd.DataFrame(filas)
    p95 = np.percentile(t.dif_ganador_vs_ref, C["placebo_percentil"])
    p = (1 + (t.dif_ganador_vs_ref >= dif_real).sum()) / (1 + len(t))
    print(t.round(4).to_string(index=False))
    print(f"N candidatos = {len(cands)} | Δ real (mismo subconjunto) = {dif_real:.4f} | "
          f"Δ placebo media = {t.dif_ganador_vs_ref.mean():.4f}, p95 = {p95:.4f} | p = {p:.3f} -> "
          f"{'PASA' if dif_real > p95 else 'NO PASA'}")


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "todo"
    replicas = int(sys.argv[2]) if len(sys.argv) > 2 else C["placebo_replicas"]
    wcb, d, _, _ = cargar_todo()
    if modo in ("anidado", "todo"):
        anidado(wcb, d)
    if modo in ("placebo", "todo"):
        placebo(wcb, d, replicas)


if __name__ == "__main__":
    main()
