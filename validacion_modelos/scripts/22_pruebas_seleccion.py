# -*- coding: utf-8 -*-
"""
22. Pruebas de sobreoptimización sin reentrenar, sobre el registro de 21 (mismas filas ago-nov).

  P2  PBO por CSCV (Bailey, Borwein, López de Prado y Zhu, 2014/2017). S bloques de CLIENTES
      (cada cliente con todas sus filas en un solo bloque, así IS y OOS no comparten clientes).
      Para cada partición en dos mitades: ganador IS -> rango relativo OOS w -> logit.
      PBO = P(logit <= 0). Además: degradación (regresión OOS ~ IS del ganador IS) y P(pérdida
      OOS frente a la referencia).
  P3  Data snooping sobre los N candidatos frente a la referencia:
      Reality Check de White (2000), SPA consistente de Hansen (2005) y stepdown de Romano y
      Wolf (2005) con p-valores ajustados. Bootstrap de clientes con reemplazo, recentrado.
  P4  Gini deflactado (Bailey y López de Prado, 2014): máximo esperado de N_ef candidatos sin
      ventaja, con N_ef = ratio de participación de los autovalores de la correlación de los Δ
      bootstrap; más el máximo nulo empírico del bootstrap recentrado.
  P6  Probabilidad de ser el mejor (frecuencia de argmax en el bootstrap) y Model Confidence Set
      (Hansen, Lunde y Nason, 2011), estadístico T_max, nivel 10 %.

Unidad de remuestreo: el cliente (sus filas de los 4 meses entran o salen juntas).
Salidas: resultados/sobreopt_{spa,pbo,mcs,prob_mejor,resumen}.csv
"""
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats

from comun import DIR_RES, MESES_FOLD, GiniRapido, pesos_bootstrap_clientes
from sobreopt_comun import CRITERIOS_SO as C, GANADOR_PIPELINE, REF, SEMILLA_SO

EULER = 0.5772156649


def main():
    t0 = time.time()
    z = np.load(DIR_RES / "sobreopt_registro.npz", allow_pickle=True)
    P, nombres, y, ids, mes = z["P"], list(z["nombres"]), z["y"], z["ids"], z["mes"]
    N, K = len(nombres), len(MESES_FOLD)
    r, g_pipe = nombres.index(REF), nombres.index(GANADOR_PIPELINE)
    folds = [np.flatnonzero(mes == M) for M in MESES_FOLD]
    calc = [[GiniRapido(y[f], P[k, f]) for f in folds] for k in range(N)]

    # ---------------------------------------------------------------- bootstrap común
    B = C["n_boot"]
    G0 = np.array([[calc[k][j]() for j in range(K)] for k in range(N)])          # (N, K)
    Gb = np.empty((B, N, K))
    for b, w in enumerate(pesos_bootstrap_clientes(ids, B, SEMILLA_SO)):
        for k in range(N):
            for j, f in enumerate(folds):
                Gb[b, k, j] = calc[k][j](w[f])
    print(f"bootstrap listo ({time.time() - t0:.0f}s)", flush=True)
    Gm0, Gmb = G0.mean(axis=1), Gb.mean(axis=2)                                   # (N,), (B, N)
    D0, Db = Gm0 - Gm0[r], Gmb - Gmb[:, [r]]
    cand = [k for k in range(N) if k != r]
    g_max = int(cand[np.argmax(D0[cand])])

    # ---------------------------------------------------------------- P3: RC, SPA, Romano-Wolf
    om = Db[:, cand].std(axis=0, ddof=1)
    d0 = D0[cand]
    t = d0 / om
    Zc = Db[:, cand] - d0                                                          # recentrado (H0: Δ=0)
    n_cli = len(np.unique(ids))
    umbral = -om * np.sqrt(2 * np.log(np.log(n_cli)))
    g_c = np.where(d0 >= umbral, d0, 0.0)                                         # SPA consistente
    T_spa = max(t.max(), 0.0)
    T_spa_b = np.maximum(((Db[:, cand] - g_c) / om).max(axis=1), 0.0)
    p_spa = float((T_spa_b >= T_spa).mean())
    T_rc = max(d0.max(), 0.0)
    p_rc = float((np.maximum(Zc.max(axis=1), 0.0) >= T_rc).mean())
    orden = np.argsort(-t)
    p_rw = np.empty(len(cand)); prev = 0.0
    for pos, j in enumerate(orden):
        resto = orden[pos:]
        prev = max(prev, float(((Zc[:, resto] / om[resto]).max(axis=1) >= t[j]).mean()))
        p_rw[j] = prev
    spa = pd.DataFrame(dict(nombre=[nombres[k] for k in cand], gini_medio=Gm0[cand], dif_vs_ref=d0, se=om,
                            t=t, p_individual=[(Db[:, k] <= 0).mean() for k in cand], p_romano_wolf=p_rw))
    spa = spa.sort_values("dif_vs_ref", ascending=False)
    spa.to_csv(DIR_RES / "sobreopt_spa.csv", index=False)

    # ---------------------------------------------------------------- P4: Gini deflactado
    lam = np.clip(np.linalg.eigvalsh(np.corrcoef(Zc.T)), 0, None)
    n_ef = float(lam.sum() ** 2 / (lam ** 2).sum())
    def max_esperado(n, s):
        return s * ((1 - EULER) * stats.norm.ppf(1 - 1 / n) + EULER * stats.norm.ppf(1 - 1 / (n * np.e)))
    defl = {}
    for etiqueta, k in (("max", g_max), ("pipeline", g_pipe)):
        s = om[cand.index(k)]
        em = max_esperado(n_ef, s)
        defl[etiqueta] = dict(dif=D0[k], se=s, e_max_formula=em, z=(D0[k] - em) / s)
    nulo_max = Zc.max(axis=1)                                                      # máximo bajo H0 (empírico)

    # ---------------------------------------------------------------- P6: prob. de ser el mejor y MCS
    best = np.bincount(Gmb.argmax(axis=1), minlength=N) / B
    top3 = np.mean(np.argsort(-Gmb, axis=1)[:, :3][:, :, None] == np.arange(N), axis=(0, 1)) * 3
    prob = pd.DataFrame(dict(nombre=nombres, gini_medio=Gm0, p_mejor=best, p_top3=top3)) \
        .sort_values("gini_medio", ascending=False)
    prob.to_csv(DIR_RES / "sobreopt_prob_mejor.csv", index=False)

    vivos, p_acum, elim = list(range(N)), 0.0, []
    while len(vivos) > 1:
        dk0 = -(Gm0[vivos] - Gm0[vivos].mean())                                    # pérdida relativa
        dkb = -(Gmb[:, vivos] - Gmb[:, vivos].mean(axis=1, keepdims=True))
        se = dkb.std(axis=0, ddof=1)
        tk = dk0 / se
        p = float((((dkb - dk0) / se).max(axis=1) >= tk.max()).mean())
        p_acum = max(p_acum, p)
        if p >= C["mcs_alpha"]:
            break
        peor = vivos[int(np.argmax(tk))]
        elim.append(dict(nombre=nombres[peor], gini_medio=Gm0[peor], p_mcs=p_acum, en_mcs=False))
        vivos.remove(peor)
    mcs = pd.DataFrame(elim + [dict(nombre=nombres[k], gini_medio=Gm0[k], p_mcs=max(p_acum, p) if len(vivos) > 1 else 1.0,
                                    en_mcs=True) for k in vivos]).sort_values("gini_medio", ascending=False)
    mcs.to_csv(DIR_RES / "sobreopt_mcs.csv", index=False)
    print(f"P3, P4, P6 listos ({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- P2: PBO por CSCV
    S = C["cscv_bloques"]
    uniq, inv = np.unique(ids, return_inverse=True)
    bloque = np.random.default_rng(SEMILLA_SO).permutation(np.arange(len(uniq)) % S)[inv]
    filas = []
    for J in combinations(range(S), S // 2):
        en_is = np.isin(bloque, J).astype(float)
        R_is = np.array([np.mean([calc[k][j](en_is[f]) for j, f in enumerate(folds)]) for k in range(N)])
        R_oos = np.array([np.mean([calc[k][j](1 - en_is[f]) for j, f in enumerate(folds)]) for k in range(N)])
        ks = int(np.argmax(R_is))
        w_rel = stats.rankdata(R_oos)[ks] / (N + 1)
        filas.append(dict(ganador_is=nombres[ks], gini_is=R_is[ks], gini_oos=R_oos[ks],
                          rango_rel_oos=w_rel, logit=np.log(w_rel / (1 - w_rel)),
                          dif_oos_vs_ref=R_oos[ks] - R_oos[r], mediana_oos=np.median(R_oos)))
    pbo_t = pd.DataFrame(filas)
    pbo_t.to_csv(DIR_RES / "sobreopt_pbo.csv", index=False)
    pbo = float((pbo_t.logit <= 0).mean())
    pend, inter = np.polyfit(pbo_t.gini_is, pbo_t.gini_oos, 1)
    p_perdida = float((pbo_t.dif_oos_vs_ref < 0).mean())
    print(f"P2 listo ({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- resumen con criterios
    def fila(prueba, medida, valor, criterio, pasa):
        return dict(prueba=prueba, medida=medida, valor=valor, criterio=criterio, pasa=pasa)
    rs = [
        fila("info", "N candidatos (incl. referencia)", N, "", None),
        fila("info", "ganador por Gini medio", nombres[g_max], "", None),
        fila("info", "dif ganador vs referencia", D0[g_max], "", None),
        fila("info", "dif lgb_sup_sin_dui vs referencia", D0[g_pipe], "", None),
        fila("P2 PBO/CSCV", "PBO", pbo, f"< {C['pbo_fiable']} fiable; > {C['pbo_dudoso']} ruido", pbo < C["pbo_fiable"]),
        fila("P2 PBO/CSCV", "pendiente OOS~IS del ganador IS", pend, "> 0 (sin degradación sistemática)", pend > 0),
        fila("P2 PBO/CSCV", "P(ganador IS pierde OOS frente a referencia)", p_perdida, "< 0,05", p_perdida < 0.05),
        fila("P3 snooping", "p Reality Check (White)", p_rc, "< 0,05", p_rc < C["alpha"]),
        fila("P3 snooping", "p SPA consistente (Hansen)", p_spa, "< 0,05", p_spa < C["alpha"]),
        fila("P3 snooping", "p Romano-Wolf lgb_sup_sin_dui", float(spa.set_index("nombre").loc[GANADOR_PIPELINE, "p_romano_wolf"]),
             "< 0,05", float(spa.set_index("nombre").loc[GANADOR_PIPELINE, "p_romano_wolf"]) < C["alpha"]),
        fila("P3 snooping", "candidatos con p Romano-Wolf < 0,05", int((spa.p_romano_wolf < C["alpha"]).sum()), "", None),
        fila("P4 deflactado", "N efectivo", n_ef, "", None),
        fila("P4 deflactado", "E[max dif] bajo H0 (fórmula)", defl["max"]["e_max_formula"], "", None),
        fila("P4 deflactado", "E[max dif] bajo H0 (bootstrap) / p95", f"{nulo_max.mean():.4f} / {np.percentile(nulo_max, 95):.4f}", "", None),
        fila("P4 deflactado", "z deflactado del ganador", defl["max"]["z"], f"> {C['z_deflactado']}", defl["max"]["z"] > C["z_deflactado"]),
        fila("P4 deflactado", "z deflactado de lgb_sup_sin_dui", defl["pipeline"]["z"], f"> {C['z_deflactado']}", defl["pipeline"]["z"] > C["z_deflactado"]),
        fila("P6 MCS", "modelos en el MCS al 90 %", int(mcs.en_mcs.sum()), "", None),
        fila("P6 MCS", "referencia fuera del MCS", not bool(mcs.set_index("nombre").loc[REF, "en_mcs"]), "True", not bool(mcs.set_index("nombre").loc[REF, "en_mcs"])),
        fila("P6 MCS", "lgb_sup_sin_dui dentro del MCS", bool(mcs.set_index("nombre").loc[GANADOR_PIPELINE, "en_mcs"]), "True", bool(mcs.set_index("nombre").loc[GANADOR_PIPELINE, "en_mcs"])),
        fila("P6 prob. mejor", "P(lgb_sup_sin_dui es el mejor)", best[g_pipe], "", None),
        fila("P6 prob. mejor", "P(referencia es la mejor)", best[r], "", None),
    ]
    rs = pd.DataFrame(rs)
    rs.to_csv(DIR_RES / "sobreopt_resumen.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 80)
    print("\n=== RESUMEN\n", rs.to_string(index=False))
    print("\n=== SPA / ROMANO-WOLF (top 15)\n", spa.head(15).round(4).to_string(index=False))
    print("\n=== MCS\n", mcs[mcs.en_mcs].round(4).to_string(index=False))
    print("\n=== PROB. DE SER EL MEJOR (top 10)\n", prob.head(10).round(4).to_string(index=False))
    print("\n=== PBO: ganadores IS más frecuentes\n", pbo_t.ganador_is.value_counts().head(8).to_string())
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
