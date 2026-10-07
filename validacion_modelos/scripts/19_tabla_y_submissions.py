# -*- coding: utf-8 -*-
"""
19. Tabla comparativa candidatos x pruebas (R1-R9), regla de decision pre-registrada
(criterios_robustez.py, R1 'parsimonia') y submissions candidatas de diciembre.
Submission = media de las 10 semillas del modelo final de R4 (si el modelo no esta en R4 se
entrenan aqui las 10 semillas con el mismo n_iter final). Formato estricto: id_cliente,prediccion,
mismo orden que datos_entrada/test.csv, [0,1], sin nulos. SOLO en validacion_modelos/resultados/.
Uso: python 19_tabla_y_submissions.py [modelo_recomendado]  (por defecto, el de la regla)
Salidas: robustez_tabla_comparativa.csv, submission_candidata_<modelo>.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from criterios_robustez import COMPLEJIDAD
from comun import RAIZ
from robustez_comun import CANDIDATOS, CON_DUI, DIR_RES, cargar_todo, definir_candidatos, fit_fijo, pred


def leer(n):
    p = DIR_RES / n
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def tabla():
    r1 = leer("robustez_r1_resumen.csv"); pa = leer("robustez_r1_pares.csv")
    r2, r3, r4 = leer("robustez_r2_resumen.csv"), leer("robustez_r3_resumen.csv"), leer("robustez_r4v2_resumen.csv")
    r5, r5o = leer("robustez_r5_segmentos.csv"), leer("robustez_r5_orden.csv")
    r6, r7, r8, r9 = leer("robustez_r6_resumen.csv"), leer("robustez_r7_resumen.csv"), leer("robustez_r8_resumen.csv"), leer("robustez_r9_resumen.csv")
    filas = []
    for m in CANDIDATOS:
        t = r1[(r1.escenario == "tal_cual") & (r1.modelo == m)].iloc[0]
        f = r1[(r1.escenario != "tal_cual") & (r1.modelo == m)].iloc[0]
        r = dict(modelo=m, r1_gini=t.gini_medio, r1_ic=f"[{t.ic95_inf:.4f}, {t.ic95_sup:.4f}]",
                 r1_gini_esc_f=f.gini_medio, r1_sd_semillas=t.sd_semillas, complejidad=COMPLEJIDAD[m])
        if m != "lgb_reg":
            for e, s in (("tal_cual", ""), ("f_dui_perm_18_6_nuevos", "_f")):
                q = pa[(pa.escenario == e) & (pa.a == "lgb_reg") & (pa.b == m)].iloc[0]
                r[f"r1_delta_vs_lgb_reg{s}"] = -q.delta; r[f"r1_p_holm{s}"] = q.p_holm
                r[f"r1_folds_mejor{s}"] = q.folds_b_mejor
                if s == "":
                    r["regla_claude_md"] = q.regla_claude_md_vs_lgb_reg
        pas, fal = [], []
        def reg(clave, ok, txt):
            r[clave] = txt; (pas if ok else fal).append(clave.split("_")[0])
        if len(r2) and m in set(r2.modelo):
            q = r2[r2.modelo == m].iloc[0]
            reg("r2_arboles", bool(q.robusto), f"{'PASA' if q.robusto else 'FALLA'} meseta {q.meseta_min}-{q.meseta_max} (x{q.ratio_meseta:.0f})")
        if len(r3) and m in set(r3.modelo):
            q = r3[r3.modelo == m].iloc[0]
            reg("r3_hiperparametros", bool(q.robusto), f"{'PASA' if q.robusto else 'FALLA'} rango {q.rango_oat:.4f}")
        if len(r4) and m in set(r4.modelo):
            q = r4[r4.modelo == m].set_index("esquema")
            ok = bool(q.robusto.all())
            reg("r4_estabilidad", ok, f"{'PASA' if ok else 'FALLA'} estable sem {q.loc['semillas','pct_top10_estable_90']:.0f} % / boot {q.loc['bootstrap_clientes','pct_top10_estable_90']:.0f} %")
        if len(r5):
            g = r5[r5.modelo == m]; w = g.loc[g.gini_medio.idxmin()]
            r["r5_peor_segmento"] = f"{w.variable}={w.segmento} ({w.gini_medio:.3f})"
            r["r5_tau_min"] = r5o.tau_kendall_vs_global.min() if len(r5o) else np.nan
        if len(r6) and m in set(r6.modelo):
            q = r6[r6.modelo == m].iloc[0]
            reg("r6_perturbacion", bool(q.robusto), f"{'PASA' if q.robusto else 'FALLA'} caida10 {q.caida_s10:.4f}; top {q.variable_top} {100*q.cuota_top:.0f} %")
        if len(r7) and m in set(r7.modelo):
            q = r7[r7.modelo == m].iloc[0]
            ok7 = bool((q.ic95_inf <= 0 <= q.ic95_sup) or q.beta_fe_origen > 0)   # criterio pre-registrado
            reg("r7_degradacion", ok7, f"{'PASA' if ok7 else 'FALLA'} (p_Holm {q.p_holm:.2f}) beta {q.beta_fe_origen:+.4f} [{q.ic95_inf:+.4f}, {q.ic95_sup:+.4f}]")
        if len(r8) and m in set(r8.modelo):
            q = r8[r8.modelo == m].iloc[0]
            reg("r8_placebo", bool(q.valido), f"{'PASA' if q.valido else 'FALLA'} z={q.z:.1f}, p={q.p_permutacion:.3f}")
        if len(r9) and m in set(r9.modelo):
            q = r9[r9.modelo == m].iloc[0]
            r["r9_curva"] = f"{'saturado' if q.saturado else 'NO saturado'} ganancia {q.ganancia_75_100:+.4f}"
        r.update(pruebas_aplicadas=len(pas) + len(fal), pruebas_falladas=len(fal), falladas=",".join(fal))
        filas.append(r)
    return pd.DataFrame(filas)


def decidir(tb, pa):
    """Regla pre-registrada (criterios_robustez.py, R1 'regla_mejor' y 'parsimonia')."""
    p = pa[pa.escenario == "tal_cual"]
    mejores = [m for m in CANDIDATOS if m != "lgb_reg" and
               (p[(p.a == "lgb_reg") & (p.b == m)].veredicto == "b MEJOR").any()]
    if mejores:
        g = tb.set_index("modelo").loc[mejores].r1_gini
        return g.idxmax(), f"MEJOR que lgb_reg por R1: {mejores}", mejores
    top = tb.loc[tb.r1_gini.idxmax(), "modelo"]
    S = [top]
    for m in CANDIDATOS:
        if m == top:
            continue
        q = p[((p.a == top) & (p.b == m)) | ((p.a == m) & (p.b == top))].iloc[0]
        if q.p_holm >= 0.05:
            S.append(m)
    s = tb.set_index("modelo").loc[S].copy()
    s["usa_dui"] = [m in CON_DUI for m in s.index]
    s = s.sort_values(["pruebas_falladas", "usa_dui", "complejidad", "r1_sd_semillas"])
    return s.index[0], f"parsimonia sobre S={S}; orden {list(s.index)}", S


def escribir_submission(wcb, d, nombre):
    r4 = DIR_RES / f"robustez_r4v2_pred_dic_{nombre}.csv"
    if r4.exists():
        q = pd.read_csv(r4); assert (q.id_cliente.to_numpy() == d.id_test).all()
        p = q[[c for c in q.columns if c.startswith("s")]].to_numpy(dtype=float).mean(axis=1)
    else:
        X, y = d.X_train, d.y_train.to_numpy(); sp = definir_candidatos(wcb, X.columns)[nombre]
        it = pd.read_csv(DIR_RES / "robustez_r1_iteraciones.csv"); r = it[it.modelo == nombre]
        n_it = None if r.best_iter_es_Mm1.isna().all() else max(10, int(round(np.median(r.best_iter_es_Mm1 * len(X) / r.filas_train_es))))
        p = np.mean([pred(fit_fijo(wcb, sp, X[sp["cols"]], y, n_it, s), d.X_test[sp["cols"]]) for s in range(42, 52)], axis=0)
    test = pd.read_csv(RAIZ / "datos_entrada" / "test.csv", usecols=["id_cliente"])
    sub = pd.DataFrame({"id_cliente": d.id_test, "prediccion": p})
    assert list(sub.columns) == ["id_cliente", "prediccion"]
    assert (sub.id_cliente.to_numpy() == test.id_cliente.to_numpy()).all(), "orden distinto a test.csv"
    assert sub.prediccion.notna().all() and sub.prediccion.between(0, 1).all()
    ruta = DIR_RES / f"submission_candidata_{nombre}.csv"
    assert "resultados_reportes" not in str(ruta)
    sub.to_csv(ruta, index=False); print("escrito", ruta, len(sub))
    return sub


def main():
    tb = tabla(); pa = leer("robustez_r1_pares.csv")
    elegido, motivo, S = decidir(tb, pa)
    tb["recomendado_regla"] = tb.modelo == elegido
    tb.to_csv(DIR_RES / "robustez_tabla_comparativa.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
    print(tb.T.to_string()); print("ELEGIDO:", elegido, "|", motivo)
    rec = sys.argv[1] if len(sys.argv) > 1 else elegido
    wcb, d, _, _ = cargar_todo()
    subs = {m: escribir_submission(wcb, d, m) for m in dict.fromkeys([rec, "lgb_reg"])}
    act = pd.read_csv(RAIZ / "resultados_reportes" / "submission_final.csv")
    from scipy.stats import spearmanr
    for m, s in subs.items():
        assert (act.id_cliente.to_numpy() == s.id_cliente.to_numpy()).all()
        print(m, "Spearman con submission_final:", round(spearmanr(act.prediccion, s.prediccion)[0], 4))


if __name__ == "__main__":
    main()
