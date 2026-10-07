# -*- coding: utf-8 -*-
"""
20. Validación en condiciones de diciembre y robustez de los modelos del pipeline NUEVO
(Pipeline_DF_WCB.py: sin early stopping; nº de árboles por curva media de folds con regla 1-SE).

Reproduce sobre esos modelos, con los mismos criterios, las pruebas de:
  * 07 (C2): Gini en condiciones de diciembre sobre los folds ago-nov. Escenarios
      a tal cual | b dui permutada dentro del mes (3 permutaciones) | d clientes nuevos al 18,6 %
      | f = b + d. Sin reponderación adversarial (enmienda E2 de criterios_robustez.py).
  * 06 (C1, C3): PSI de scores del MISMO modelo (fold nov, entrenado <= oct) entre nov y dic;
      Spearman entre semillas y entre modelos, nov frente a dic; dependencia de dui en dic.
  * 13 (R4): estabilidad del top-10 % de diciembre del modelo final (10 semillas y 20 bootstraps
      de clientes con pesos de multiplicidad).

Modelos, con los árboles que elige el pipeline nuevo:
  lgb_reg_con_dui : PARAMS_DEFECTO["lightgbm"], 39 columnas -> referencia (modelo actual)
  lgb_sup_sin_dui : PARAMS_DEFECTO["lightgbm_superficial"], 38 columnas -> ganador del pipeline
  lgb_sup_con_dui : PARAMS_DEFECTO["lightgbm_superficial"], 39 columnas

Las predicciones de validación salen de EvaluadorModelos igual que en main(): selección de variables
con meses < ago, semilla 42 y n de cada fold elegido con los otros 3 folds. El modelo final usa
n = mejor_iter * factor_datos_ y las semillas 42, 43 y 44, como entrenar_y_predecir_final.
Criterios: comun.CRITERIOS (C1-C3) y criterios_robustez.CRITERIOS["R4"], sin cambios. Regla de
reemplazo de CLAUDE.md: Gini medio > referencia + 0,003 y ningún fold peor.

Salidas nuevas (no sobrescriben nada): resultados/pipeline_nuevo_*.csv
Ejecutar desde la raíz del repo:
    .venv/Scripts/python.exe validacion_modelos/scripts/20_pipeline_nuevo.py
"""
import importlib.util
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats

from comun import (COL_DUI, CRITERIOS, DIR_RES, MESES_FOLD, N_BOOT, PROP_NUEVOS_DIC, RAIZ, SEMILLA,
                   SEMILLAS, GiniRapido, cargar_todo, es_nuevo_train, permutar_en_mes,
                   pesos_bootstrap_clientes, psi)
from criterios_robustez import CRITERIOS as CRIT_R

MODELOS = {
    "lgb_reg_con_dui": ("lightgbm", True),
    "lgb_sup_sin_dui": ("lightgbm_superficial", False),
    "lgb_sup_con_dui": ("lightgbm_superficial", True),
}
REF = "lgb_reg_con_dui"
N_PERM = 3
ESC = ("a_tal_cual", "b_perm_dui", "d_nuevos_18_6", "f_perm_y_nuevos")
C_R4 = CRIT_R["R4"]


def cargar_metricas_r4():
    """Reutiliza metricas() de 13_r4_semillas_bootstrap.py (su main está protegido)."""
    ruta = Path(__file__).resolve().parent / "13_r4_semillas_bootstrap.py"
    spec = importlib.util.spec_from_file_location("r4", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.metricas


def sp(a, b):
    return float(stats.spearmanr(a, b).statistic)


def solape(a, b, q=0.10):
    k = int(round(q * len(a)))
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k


def predecir(m, X):
    return m.predict_proba(X)[:, 1]


def main():
    t0 = time.time()
    wcb, d, _, _ = cargar_todo()
    X, y, mes, ids = d.X_train, d.y_train, d.mes_train, d.id_train
    meses = np.sort(np.unique(mes))
    nuevo_tr = es_nuevo_train(ids, mes)
    nuevo_dic = ~np.isin(d.id_test, ids)
    metricas_r4 = cargar_metricas_r4()

    info, val, nov_sem, dic_oct, dic_fin, dic_perm, r4 = {}, {}, {}, {}, {}, {}, []
    for nombre, (preset, con_dui) in MODELOS.items():
        cfg = wcb.Config(ruta_datos=RAIZ / "datos_entrada", n_folds=len(MESES_FOLD))
        cols0 = [c for c in X.columns if con_dui or c != COL_DUI]
        sel = mes < meses[-cfg.n_folds]
        cols = wcb.seleccionar_features(X.loc[sel, cols0], y.loc[sel], cfg)   # mismo orden que main()
        params = dict(wcb.PARAMS_DEFECTO[preset])
        ev = wcb.EvaluadorModelos(X[cols], y, mes, ids, cfg)
        r = ev._evaluar_modelo_cv("lightgbm", params, nombre)
        vp = pd.concat(ev.val_predictions_, ignore_index=True)
        n_final = max(10, int(round(r["mejor_iter"] * ev.factor_datos_)))
        info[nombre] = dict(n_cols=len(cols), arboles_folds="/".join(map(str, r["arboles_folds"])),
                            mejor_iter=r["mejor_iter"], n_final=n_final)
        print(f"{nombre}: Gini medio {r['gini']:.4f} | árboles {info[nombre]['arboles_folds']} "
              f"| final {n_final} ({time.time() - t0:.0f}s)", flush=True)

        # --- refit de cada fold con su n: predicción normal (verificada) y con dui permutada
        filas = []
        for k, M in enumerate(MESES_FOLD):
            tr, va = mes < M, mes == M
            n_k = r["arboles_folds"][k]
            m = wcb.construir_modelo("lightgbm", params, n_k, SEMILLA).fit(X.loc[tr, cols], y.loc[tr])
            Xv = X.loc[va, cols]
            p = predecir(m, Xv)
            dif = np.abs(p - vp.loc[vp.fold == k + 1, "y_prob"].to_numpy()).max()
            assert dif < 1e-6, f"{nombre} fold {M}: el refit no reproduce el pipeline ({dif})"
            perms = []
            for j in range(N_PERM):
                if COL_DUI in cols:
                    Xp = Xv.copy()
                    Xp[COL_DUI] = permutar_en_mes(Xv[COL_DUI].to_numpy(), mes[va], SEMILLA + j)
                    perms.append(predecir(m, Xp))
                else:
                    perms.append(p)
            filas.append(pd.DataFrame({"id_cliente": ids[va], "mes": mes[va], "y": y.loc[va].to_numpy(),
                                       "nuevo": nuevo_tr[va], "hon": p,
                                       **{f"perm{j}": perms[j] for j in range(N_PERM)}}))
            if M == MESES_FOLD[-1]:      # modelo entrenado <= oct: mismo modelo en nov y en dic
                dic_oct[nombre] = predecir(m, d.X_test[cols])
                nov_sem[nombre] = [p] + [predecir(wcb.construir_modelo("lightgbm", params, n_k, s)
                                                  .fit(X.loc[tr, cols], y.loc[tr]), Xv) for s in SEMILLAS[1:]]
        val[nombre] = pd.concat(filas, ignore_index=True)

        # --- modelo final: semillas 42..51 (las 3 primeras = entrenar_y_predecir_final) + R4
        Xt = d.X_test[cols]
        P_sem, m42 = [], None
        for s in C_R4["semillas"]:
            m = wcb.construir_modelo("lightgbm", params, n_final, s).fit(X[cols], y)
            P_sem.append(predecir(m, Xt))
            m42 = m if s == SEMILLA else m42
        dic_fin[nombre] = P_sem
        if COL_DUI in cols:
            perm = []
            for j in range(N_PERM):
                Xp = Xt.copy()
                Xp[COL_DUI] = np.random.default_rng(SEMILLA + j).permutation(Xt[COL_DUI].to_numpy())
                perm.append(predecir(m42, Xp))
            dic_perm[nombre] = perm
        uniq, inv = np.unique(ids, return_inverse=True)
        rng = np.random.default_rng(2026)
        P_boot = []
        for b in range(C_R4["n_boot_clientes"]):
            w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv].astype(float)
            idx = np.flatnonzero(w > 0)
            m = wcb.construir_modelo("lightgbm", params, n_final, C_R4["semilla_modelo_boot"])
            m.fit(X[cols].iloc[idx], y.iloc[idx], sample_weight=w[idx])
            P_boot.append(predecir(m, Xt))
        for esquema, P in (("semillas", np.array(P_sem)), ("bootstrap_clientes", np.array(P_boot))):
            met = metricas_r4(P)
            r4.append(dict(modelo=nombre, esquema=esquema, n_iter_final=n_final, **met,
                           robusto=met["pct_top10_estable_90"] >= 85))
        print(f"{nombre}: final y R4 listos ({time.time() - t0:.0f}s)", flush=True)

    # =========================================================== C2: Gini en condiciones de dic
    base = val[REF]
    for nombre in MODELOS:
        assert (val[nombre][["id_cliente", "mes", "y"]].to_numpy() == base[["id_cliente", "mes", "y"]].to_numpy()).all()
    mes_v, y_v, ids_v, nuevo_v = base.mes.to_numpy(), base.y.to_numpy(), base.id_cliente.to_numpy(), base.nuevo.to_numpy()
    w_nuevo = np.empty(len(base))
    for M in MESES_FOLD:
        i = mes_v == M
        pn = nuevo_v[i].mean()
        w_nuevo[i] = np.where(nuevo_v[i], PROP_NUEVOS_DIC / pn, (1 - PROP_NUEVOS_DIC) / (1 - pn))
    folds = {M: np.flatnonzero(mes_v == M) for M in MESES_FOLD}
    calc = {(n, M, c): GiniRapido(y_v[idx], val[n][c].to_numpy()[idx])
            for n in MODELOS for M, idx in folds.items() for c in ["hon"] + [f"perm{j}" for j in range(N_PERM)]}

    def evaluar(n, M, e, wb):
        idx = folds[M]
        w = np.ones(len(idx)) if wb is None else wb[idx]
        if e in ("d_nuevos_18_6", "f_perm_y_nuevos"):
            w = w * w_nuevo[idx]
        if e in ("b_perm_dui", "f_perm_y_nuevos"):
            return np.mean([calc[(n, M, f"perm{j}")](w) for j in range(N_PERM)])
        return calc[(n, M, "hon")](w)

    punt = {(n, M, e): evaluar(n, M, e, None) for n in MODELOS for M in MESES_FOLD for e in ESC}
    B = {k: np.empty(N_BOOT) for k in punt}
    for b, wb in enumerate(pesos_bootstrap_clientes(ids_v, N_BOOT, SEMILLA)):
        for (n, M, e) in punt:
            B[(n, M, e)][b] = evaluar(n, M, e, wb)
    media = lambda n, e: np.mean([punt[(n, M, e)] for M in MESES_FOLD])
    bmed = lambda n, e: np.mean([B[(n, M, e)] for M in MESES_FOLD], axis=0)

    pd.DataFrame([dict(modelo=n, fold=M, escenario=e, gini=v) for (n, M, e), v in punt.items()]) \
        .pivot_table(index=["modelo", "fold"], columns="escenario", values="gini").reset_index() \
        .to_csv(DIR_RES / "pipeline_nuevo_gini_por_fold.csv", index=False)
    esc = []
    for n in MODELOS:
        for e in ESC:
            bb, ba = bmed(n, e), bmed(n, "a_tal_cual")
            f = dict(modelo=n, escenario=e, gini=media(n, e), ic95_inf=np.percentile(bb, 2.5),
                     ic95_sup=np.percentile(bb, 97.5), cambio_vs_tal_cual=media(n, e) - media(n, "a_tal_cual"),
                     se_pareado_cambio=(bb - ba).std(ddof=1))
            if n != REF:
                dv = bb - bmed(REF, e)
                difs = [punt[(n, M, e)] - punt[(REF, M, e)] for M in MESES_FOLD]
                f.update(dif_vs_ref=media(n, e) - media(REF, e), se_dif=dv.std(ddof=1),
                         dif_ic95_inf=np.percentile(dv, 2.5), dif_ic95_sup=np.percentile(dv, 97.5),
                         p_dif_menor_igual_0=float((dv <= 0).mean()),
                         folds_mejor=int(sum(x > 0 for x in difs)), peor_dif_fold=min(difs),
                         cumple_regla_reemplazo=bool(media(n, e) - media(REF, e) > 0.003 and min(difs) >= 0))
            esc.append(f)
    esc = pd.DataFrame(esc)
    esc.to_csv(DIR_RES / "pipeline_nuevo_gini_escenarios.csv", index=False)

    # =========================================================== C1 / C3: drift y ranking
    nov = base.mes.to_numpy() == MESES_FOLD[-1]
    nuevo_nov = nuevo_v[nov]
    sub = pd.read_csv(RAIZ / "resultados_reportes" / "submission_final.csv")
    cand = pd.read_csv(DIR_RES / "submission_candidata_lgb_sup_sin_dui.csv")
    assert (sub.id_cliente.to_numpy() == d.id_test).all() and (cand.id_cliente.to_numpy() == d.id_test).all()
    entrega = {n: np.mean(dic_fin[n][:3], axis=0) for n in MODELOS}       # = submission del pipeline
    P_nov = {n: nov_sem[n][0] for n in MODELOS}
    P_dic = {n: dic_fin[n][0] for n in MODELOS}
    dr = []
    for n in MODELOS:
        a, b = P_nov[n], dic_oct[n]
        otros = [o for o in MODELOS if o != n]
        s_nov = np.mean([sp(nov_sem[n][i], nov_sem[n][j]) for i, j in itertools.combinations(range(3), 2)])
        s_dic = np.mean([sp(dic_fin[n][i], dic_fin[n][j]) for i, j in itertools.combinations(range(3), 2)])
        m_nov = np.mean([sp(P_nov[n], P_nov[o]) for o in otros])
        m_dic = np.mean([sp(P_dic[n], P_dic[o]) for o in otros])
        f = dict(modelo=n, media_nov=a.mean(), media_dic_mismo_modelo=b.mean(), media_dic_final=entrega[n].mean(),
                 psi_mismo_modelo=psi(a, b), psi_mismo_modelo_nuevos=psi(a[nuevo_nov], b[nuevo_dic]),
                 psi_mismo_modelo_antiguos=psi(a[~nuevo_nov], b[~nuevo_dic]),
                 spearman_semillas_nov=s_nov, spearman_semillas_dic=s_dic, delta_spearman_semillas=s_dic - s_nov,
                 spearman_modelos_nov=m_nov, spearman_modelos_dic=m_dic, delta_spearman_modelos=m_dic - m_nov,
                 spearman_submission_final=sp(entrega[n], sub.prediccion), top10_submission_final=solape(entrega[n], sub.prediccion.to_numpy()),
                 spearman_candidata_robustez=sp(entrega[n], cand.prediccion),
                 top10_candidata_robustez=solape(entrega[n], cand.prediccion.to_numpy()))
        if n in dic_perm:
            f["spearman_pred_vs_dui_permutada_dic"] = np.mean([sp(P_dic[n], p) for p in dic_perm[n]])
        dr.append(f)
    dr = pd.DataFrame(dr)
    dr.to_csv(DIR_RES / "pipeline_nuevo_drift_ranking.csv", index=False)
    r4 = pd.DataFrame(r4)
    r4.to_csv(DIR_RES / "pipeline_nuevo_r4.csv", index=False)
    pd.DataFrame({"id_cliente": d.id_test, **{n: entrega[n] for n in MODELOS}}) \
        .to_csv(DIR_RES / "pipeline_nuevo_pred_dic.csv", index=False)

    # =========================================================== veredicto (reglas de comun.py)
    ver = []
    for n in MODELOS:
        fa = esc[(esc.modelo == n) & (esc.escenario == "f_perm_y_nuevos")].iloc[0]
        caida, se = -fa.cambio_vs_tal_cual, fa.se_pareado_cambio
        signif = caida > CRITERIOS["k_se_pareado"] * se
        material = signif and caida > CRITERIOS["caida_material"]
        q = dr[dr.modelo == n].iloc[0]
        c3 = (q.delta_spearman_modelos <= -CRITERIOS["delta_spearman_modelos"]
              or q.delta_spearman_semillas <= -CRITERIOS["delta_spearman_semillas"])
        if material or q.psi_mismo_modelo >= CRITERIOS["psi_severo"]:
            v = "SE CAE"
        elif signif or q.psi_mismo_modelo >= CRITERIOS["psi_moderado"] or c3:
            v = "DEGRADACION MODERADA"
        else:
            v = "NO SE CAE"
        rr = r4[r4.modelo == n].set_index("esquema")
        ver.append(dict(modelo=n, **info[n], gini_tal_cual=media(n, "a_tal_cual"), gini_escenario_f=media(n, "f_perm_y_nuevos"),
                        caida_f=caida, dos_se_pareado=2 * se, c1_psi=q.psi_mismo_modelo,
                        c3_delta_modelos=q.delta_spearman_modelos, c3_delta_semillas=q.delta_spearman_semillas,
                        r4_semillas_pct=rr.loc["semillas", "pct_top10_estable_90"],
                        r4_bootstrap_pct=rr.loc["bootstrap_clientes", "pct_top10_estable_90"],
                        r4_pasa=bool(rr.robusto.all()), veredicto=v))
    ver = pd.DataFrame(ver)
    ver.to_csv(DIR_RES / "pipeline_nuevo_veredicto.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    for titulo, t in (("ESCENARIOS (C2 y comparación con la referencia)", esc), ("DRIFT Y RANKING (C1, C3)", dr),
                      ("R4", r4), ("VEREDICTO", ver)):
        print(f"\n=== {titulo}\n{t.round(4).to_string(index=False)}")
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
