# -*- coding: utf-8 -*-
"""
08. Veredicto por modelo (criterios fijados en comun.CRITERIOS antes de ver resultados) y figuras.

Criterios aplicados:
  C1 PSI de scores del MISMO modelo (<= oct) nov -> dic          (drift_predicciones.csv)
  C2 caida del Gini medio en el escenario e_combinado frente a a_tal_cual
     (arboles: ago-nov; redes: sep-nov), significativa si > 2 * SE pareado, material si ademas > 0,02
  C3 Spearman medio con los demas modelos (dic final - nov) <= -0,05; semillas (dic - nov) <= -0,02
Ademas (sensibilidad, no cambia la regla): escenario f_perm_y_nuevos (sin reponderacion adversarial).
Tambien se guarda la escala de los scores (explica el PSI alto "modelo final" por n de arboles).

Salidas: veredicto_modelos.csv, escala_scores.csv, figuras/*.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import CRITERIOS as C, DIR_FIG, DIR_RES, MESES_FOLD, RAIZ

ARBOLES = ["lgb_reg", "lgb_sin_reg", "xgb", "catboost", "random_forest", "lgb_reg_sin_dui"]
REDES = ["mlp_sklearn", "nn_baseline", "nn_optimizado"]
TODOS = ARBOLES + REDES


def veredicto():
    dp = pd.read_csv(DIR_RES / "drift_predicciones.csv").set_index("modelo")
    rk = pd.read_csv(DIR_RES / "ranking_resumen.csv").set_index("modelo")
    gd = pd.read_csv(DIR_RES / "gini_dic_escenarios.csv")
    filas = []
    for m in TODOS:
        fs = "sep-nov" if m in REDES else "ago-nov"
        g = gd[(gd.modelo == m) & (gd.folds == fs)].set_index("escenario")
        e, f = g.loc["e_combinado"], g.loc["f_perm_y_nuevos"]
        caida = -e.cambio_vs_tal_cual
        sig = caida > C["k_se_pareado"] * e.se_pareado_cambio
        material = sig and caida > C["caida_material"]
        psi_ = dp.loc[m, "psi_mismo_modelo"]
        d_mod = rk.loc[m, "delta_spearman_otros"]
        d_sem = rk.loc[m, "delta_spearman_semillas"] if m in ARBOLES else np.nan
        c3 = (d_mod <= -C["delta_spearman_modelos"]) or (not np.isnan(d_sem) and d_sem <= -C["delta_spearman_semillas"])
        if material or psi_ >= C["psi_severo"]:
            v = "SE CAE"
        elif sig or psi_ >= C["psi_moderado"] or c3:
            v = "DEGRADACION MODERADA"
        else:
            v = "NO SE CAE"
        caida_f = -f.cambio_vs_tal_cual
        filas.append(dict(
            modelo=m, folds=fs, gini_tal_cual=g.loc["a_tal_cual", "gini"], gini_escenario_dic=e.gini,
            ic95_escenario_dic=f"[{e.ic95_inf:.3f}; {e.ic95_sup:.3f}]",
            C2_caida=caida, C2_se_pareado=e.se_pareado_cambio, C2_significativa=bool(sig), C2_material=bool(material),
            sens_f_caida=caida_f, sens_f_se=f.se_pareado_cambio,
            sens_f_significativa=bool(caida_f > C["k_se_pareado"] * f.se_pareado_cambio),
            C1_psi_mismo_modelo=psi_, C1_psi_modelo_final_informativo=dp.loc[m, "psi_modelo_final"],
            C3_delta_spearman_modelos=d_mod, C3_delta_spearman_semillas=d_sem, C3_cumple=bool(c3),
            veredicto=v))
    t = pd.DataFrame(filas)
    t.to_csv(DIR_RES / "veredicto_modelos.csv", index=False)
    return t


def escala():
    wf = pd.read_csv(DIR_RES / "wf_predicciones.csv"); nov = wf[wf.mes == 202611]
    oct_ = pd.read_csv(DIR_RES / "wf_pred_dic_modelo_oct.csv"); fin = pd.read_csv(DIR_RES / "final_pred_dic.csv")
    sub = pd.read_csv(RAIZ / "resultados_reportes" / "submission_final.csv")
    it = pd.read_csv(DIR_RES / "wf_iteraciones.csv"); itf = pd.read_csv(DIR_RES / "final_iteraciones.csv").set_index("modelo")
    filas = []
    for m in ARBOLES:
        n_nov = int(it[(it.modelo == m) & (it.fold == 202611)].n_iter.iloc[0])
        for nombre, v, n in (("nov_modelo_oct", nov[f"{m}__hon"], n_nov), ("dic_modelo_oct", oct_[f"{m}__hon"], n_nov),
                             ("dic_modelo_final", fin[f"{m}__final"], int(itf.loc[m, "n_iter_final"]))):
            filas.append(dict(modelo=m, conjunto=nombre, n_arboles=n, media=v.mean(), desv=v.std(),
                              p05=v.quantile(.05), p95=v.quantile(.95)))
    s = sub.prediccion
    filas.append(dict(modelo="submission_final", conjunto="dic", media=s.mean(), desv=s.std(),
                      p05=s.quantile(.05), p95=s.quantile(.95)))
    t = pd.DataFrame(filas); t.to_csv(DIR_RES / "escala_scores.csv", index=False)
    return t


def figuras_1():
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
    # 1. Gini por fold: honesto vs pipeline
    t = pd.read_csv(DIR_RES / "wf_gini_por_fold.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, m in enumerate(ARBOLES):
        r = t[t.modelo == m].sort_values("fold"); x = np.arange(4) + (i - 2.5) * 0.12
        ax.errorbar(x, r.honesto, yerr=[r.honesto - r.ic95_inf_honesto, r.ic95_sup_honesto - r.honesto],
                    fmt="o", ms=4, capsize=2, label=m)
        ax.scatter(x, r.pipeline, marker="x", color="k", s=14, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(["ago", "sep", "oct", "nov"]); ax.set_ylabel("Gini")
    ax.set_title("Gini por fold: walk-forward honesto (o, IC95 %) vs esquema del pipeline (x, ES sobre el fold)")
    ax.legend(ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(DIR_FIG / "fig01_gini_por_fold.png"); plt.close(fig)
    # 2. ruptura de dui y % nuevos por mes
    dm = pd.read_csv(DIR_RES / "drift_dui_por_mes.csv")
    fig, ax = plt.subplots(figsize=(9, 4)); x = dm.mes.astype(str).str[-2:]
    ax.plot(x, dm.pct_dui_igual_dut, "o-", label="% filas con dui == dut")
    ax.plot(x, dm.pct_nuevos, "s--", label="% filas de clientes nuevos")
    ax.axvline(len(dm) - 1.5, color="grey", ls=":"); ax.set_xlabel("mes (12 = test)"); ax.set_ylabel("%")
    ax.set_title("La señal de dias_ultima_interaccion desaparece en diciembre"); ax.legend()
    fig.tight_layout(); fig.savefig(DIR_FIG / "fig02_dui_y_nuevos_por_mes.png"); plt.close(fig)
    # 3. PSI por variable y AUC adversarial
    ps = pd.read_csv(DIR_RES / "drift_psi_variables.csv").sort_values("psi_nov_dic")
    ad = pd.read_csv(DIR_RES / "adversarial_auc.csv")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].barh(ps.variable, ps.psi_nov_dic, label="nov→dic")
    axs[0].barh(ps.variable, ps.psi_ene_nov_dic, alpha=.4, label="ene-nov→dic")
    axs[0].axvline(0.1, color="orange", ls="--"); axs[0].axvline(0.25, color="red", ls="--")
    axs[0].set_xscale("symlog", linthresh=0.01); axs[0].set_title("PSI por variable"); axs[0].legend()
    pv = ad.pivot(index="comparacion", columns="variables", values="auc_adversarial")
    pv.plot.bar(ax=axs[1], rot=0); axs[1].axhline(0.5, color="k", lw=.8)
    axs[1].axhline(C["auc_adversarial_relevante"], color="red", ls="--")
    axs[1].set_ylim(.45, .8); axs[1].set_title("AUC de la validación adversarial")
    fig.tight_layout(); fig.savefig(DIR_FIG / "fig03_drift_variables.png"); plt.close(fig)


def figuras_2():
    # 4. distribucion de scores nov vs dic
    wf = pd.read_csv(DIR_RES / "wf_predicciones.csv"); nov = wf[wf.mes == 202611]
    oct_ = pd.read_csv(DIR_RES / "wf_pred_dic_modelo_oct.csv"); fin = pd.read_csv(DIR_RES / "final_pred_dic.csv")
    nnv = pd.read_csv(DIR_RES / "nn_predicciones.csv"); nnv = nnv[nnv.mes == 202611]
    nnd = pd.read_csv(DIR_RES / "nn_pred_dic.csv")
    sub = pd.read_csv(RAIZ / "resultados_reportes" / "submission_final.csv")
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6)); b = np.linspace(0, .4, 60)
    for ax, m in zip(axs[:3], ["lgb_reg", "catboost", "lgb_reg_sin_dui"]):
        ax.hist(nov[f"{m}__hon"], b, density=True, histtype="step", label="nov (modelo ≤ oct)")
        ax.hist(oct_[f"{m}__hon"], b, density=True, histtype="step", label="dic (modelo ≤ oct)")
        ax.hist(fin[f"{m}__final"], b, density=True, histtype="step", ls="--", label="dic (modelo final)")
        ax.set_title(m)
    axs[0].hist(sub.prediccion, b, density=True, histtype="step", color="k", label="submission_final")
    axs[0].legend(fontsize=7)
    b = np.linspace(0, .5, 60)
    axs[3].hist(nnv["nn_optimizado__hon"], b, density=True, histtype="step", label="nov")
    axs[3].hist(nnd["nn_optimizado__final"], b, density=True, histtype="step", label="dic")
    axs[3].set_title("nn_optimizado (mismo modelo)"); axs[3].legend(fontsize=7)
    fig.suptitle("Scores: mismo modelo nov vs dic (sin drift) y efecto del nº de árboles del modelo final")
    fig.tight_layout(); fig.savefig(DIR_FIG / "fig04_distribucion_scores.png"); plt.close(fig)
    # 5. Spearman entre modelos nov vs dic
    sn = pd.read_csv(DIR_RES / "ranking_spearman_nov.csv", index_col=0)
    sd = pd.read_csv(DIR_RES / "ranking_spearman_dic.csv", index_col=0)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, s, tt in ((axs[0], sn, "nov (modelos ≤ oct)"), (axs[1], sd, "dic (modelos finales + entrega)")):
        im = ax.imshow(s.values, vmin=.6, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(s))); ax.set_yticks(range(len(s)))
        ax.set_xticklabels(s.columns, rotation=60, ha="right", fontsize=7); ax.set_yticklabels(s.index, fontsize=7)
        ax.set_title(f"Spearman {tt}")
        for i in range(len(s)):
            for j in range(len(s)):
                ax.text(j, i, f"{s.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="w" if s.values[i, j] < .85 else "k")
    fig.colorbar(im, ax=axs, shrink=.8); fig.savefig(DIR_FIG / "fig05_spearman_nov_vs_dic.png", bbox_inches="tight")
    plt.close(fig)
    # 6. Gini por escenario
    gd = pd.read_csv(DIR_RES / "gini_dic_escenarios.csv")
    gd = gd[(gd.modelo.isin(ARBOLES) & (gd.folds == "ago-nov")) | (gd.modelo.isin(REDES) & (gd.folds == "sep-nov"))]
    esc = ["a_tal_cual", "b_perm_dui", "d_nuevos_18_6", "f_perm_y_nuevos", "c_adversarial", "e_combinado"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, e in enumerate(esc):
        r = gd[gd.escenario == e].set_index("modelo").reindex(TODOS); x = np.arange(len(TODOS)) + (i - 2.5) * 0.12
        ax.errorbar(x, r.gini, yerr=[r.gini - r.ic95_inf, r.ic95_sup - r.gini], fmt="o", ms=3, capsize=1.5, label=e)
    ax.set_xticks(range(len(TODOS))); ax.set_xticklabels(TODOS, rotation=20); ax.set_ylabel("Gini medio")
    ax.set_title("Gini esperado en condiciones de diciembre (árboles: ago-nov; redes: sep-nov)")
    ax.legend(ncol=3, fontsize=7); fig.tight_layout(); fig.savefig(DIR_FIG / "fig06_gini_escenarios_diciembre.png")
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(veredicto().round(4).to_string(index=False))
    print(escala().round(4).to_string(index=False))
    figuras_1(); figuras_2()
    print("figuras ok")
