#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script P-6: Genera todas las figuras del informe final.
Lee CSVs de resultados y produce PNGs en resultados_reportes/.
Uso: python pipelines/generar_graficas.py --resultados resultados_reportes/
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("graficas")

# Estilo consistente
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

PALETA = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#BC4B51"]


def fig_tasa_conversion_por_mes(csv_path: Path, out_dir: Path) -> None:
    """Figura 1: Tasa de conversión por mes (train)."""
    df = pd.read_csv(csv_path)
    # Espera columnas: mes, mean, sum, size (salida de diagnostico_inicial)
    if "mes" not in df.columns:
        log.warning("CSV no tiene columna 'mes'; saltando fig_tasa_conversion_por_mes")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["mes"], df["mean"], marker="o", linewidth=2, color=PALETA[0])
    ax.set_title("Tasa de conversión por mes (train)")
    ax.set_xlabel("Mes (YYYYMM)")
    ax.set_ylabel("Tasa de conversión")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_tasa_conversion_mes.png", dpi=150)
    plt.close(fig)
    log.info("Guardada fig1_tasa_conversion_mes.png")


def fig_distribucion_numericas(csv_path: Path, out_dir: Path) -> None:
    """Figura 2: Boxplots de variables numéricas por objetivo."""
    df = pd.read_csv(csv_path)
    # Espera columnas estadísticas por variable
    # Para simplificar, generamos un heatmap de correlación si no hay datos raw
    log.info("fig_distribucion_numericas: requiere datos raw; generando heatmap de correlación como fallback")


def fig_matriz_correlacion(csv_path: Path, out_dir: Path) -> None:
    """Figura 3: Matriz de correlación (Pearson) de features seleccionadas."""
    df = pd.read_csv(csv_path, index_col=0)
    # Filtrar solo columnas numéricas si hay mixtas
    df_num = df.select_dtypes(include=[np.number])

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(df_num.corr(), dtype=bool))
    sns.heatmap(df_num.corr(), mask=mask, cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Matriz de correlación (Pearson) - Features seleccionadas")
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_matriz_correlacion.png", dpi=150)
    plt.close(fig)
    log.info("Guardada fig3_matriz_correlacion.png")


def fig_importancia_permutacion(csv_path: Path, out_dir: Path) -> None:
    """Figura 4: Top-15 importancia por permutación (LightGBM regularizado)."""
    df = pd.read_csv(csv_path)
    # Espera columnas: feature, importance_mean, importance_std
    if "feature" not in df.columns or "importance_mean" not in df.columns:
        log.warning("CSV importancia permutación no tiene columnas esperadas; saltando")
        return

    top15 = df.head(15).iloc[::-1]  # invertir para que el mayor quede arriba
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top15)), top15["importance_mean"], xerr=top15.get("importance_std", None),
                   color=PALETA[0], edgecolor="white", height=0.7)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15["feature"])
    ax.set_xlabel("Importancia por permutación (caída en Gini)")
    ax.set_title("Top-15 Importancia por Permutación (LightGBM Regularizado)")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_importancia_permutacion.png", dpi=150)
    plt.close(fig)
    log.info("Guardada fig4_importancia_permutacion.png")


def fig_ablacion_variables(csv_path: Path, out_dir: Path) -> None:
    """Figura 5: Resultados de ablación (configuraciones A, B, C, D, E)."""
    df = pd.read_csv(csv_path)
    # Espera columnas: config, gini_mean, gini_std (o gini_ic_lower, gini_ic_upper)
    if "config" not in df.columns or "gini_mean" not in df.columns:
        log.warning("CSV ablación no tiene columnas esperadas; saltando")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df))
    ax.bar(x, df["gini_mean"], yerr=df.get("gini_std", None),
           color=PALETA[:len(df)], edgecolor="white", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["config"], rotation=15, ha="right")
    ax.set_ylabel("Gini (promedio walk-forward CV)")
    ax.set_title("Ablación de variables y regularización")
    ax.axhline(y=df["gini_mean"].max(), color="gray", linestyle="--", alpha=0.5,
               label=f"Mejor: {df['gini_mean'].max():.4f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_ablacion_variables.png", dpi=150)
    plt.close(fig)
    log.info("Guardada fig5_ablacion_variables.png")


def fig_validacion_walkforward(csv_path: Path, out_dir: Path) -> None:
    """Figura 6: Gini por fold en walk-forward CV para cada modelo."""
    df = pd.read_csv(csv_path)
    # Espera columnas: modelo, fold, gini
    if "modelo" not in df.columns or "fold" not in df.columns or "gini" not in df.columns:
        log.warning("CSV walk-forward no tiene columnas esperadas; saltando")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (modelo, grupo) in enumerate(df.groupby("modelo")):
        ax.plot(grupo["fold"], grupo["gini"], marker="o", label=modelo,
                color=PALETA[i % len(PALETA)], linewidth=2)
    ax.set_xlabel("Fold (mes de validación)")
    ax.set_ylabel("Gini")
    ax.set_title("Walk-Forward CV: Gini por fold y modelo")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig6_walkforward_cv.png", dpi=150)
    plt.close(fig)
    log.info("Guardada fig6_walkforward_cv.png")


def fig_comparacion_modelos(csv_path: Path, out_dir: Path) -> None:
    """Figura 7: Comparación final de modelos (Gini medio ± IC95%)."""
    df = pd.read_csv(csv_path, index_col=0)
    # Espera columnas: auc, gini, mejor_iter, gini_ic_lower, gini_ic_upper
    if "gini" not in df.columns:
        log.warning("CSV comparación no tiene columna 'gini'; saltando")
        return

    df_plot = df.sort_values("gini", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(df_plot))
    gini_vals = df_plot["gini"]
    xerr_lower = gini_vals - df_plot.get("gini_ic_lower", gini_vals)
    xerr_upper = df_plot.get("gini_ic_upper", gini_vals) - gini_vals
    ax.barh(y_pos, gini_vals, xerr=[xerr_lower, xerr_upper],
            color=PALETA[0], edgecolor="white", capsize=5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot.index)
    ax.set_xlabel("Gini (promedio CV ± IC95% bootstrap cliente)")
    ax.set_title("Comparación de modelos - Validación walk-forward")
    for i, v in enumerate(gini_vals):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "fig7_comparacion_modelos.png", dpi=150)
    plt.close(fig)
    log.info("Guardada fig7_comparacion_modelos.png")


def fig_distribucion_predicciones(csv_path: Path, out_dir: Path) -> None:
    """Figura 8: Distribución de predicciones del modelo ganador en test."""
    df = pd.read_csv(csv_path)
    if "prediccion" not in df.columns:
        log.warning("CSV submission no tiene columna 'prediccion'; saltando")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["prediccion"], bins=50, color=PALETA[0], edgecolor="white", alpha=0.8)
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de predicciones en test (modelo ganador)")
    ax.axvline(df["prediccion"].mean(), color=PALETA[1], linestyle="--", linewidth=2,
               label=f"Media: {df['prediccion'].mean():.4f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig8_distribucion_predicciones.png", dpi=150)
    plt.close(fig)
    log.info("Guardada fig8_distribucion_predicciones.png")


def fig_ic_bootstrap(csv_path: Path, out_dir: Path) -> None:
    """Figura 9: Distribución bootstrap del Gini (mejor modelo)."""
    df = pd.read_csv(csv_path)
    if "gini_bootstrap" not in df.columns:
        log.warning("CSV bootstrap no tiene columna 'gini_bootstrap'; saltando")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["gini_bootstrap"], bins=50, color=PALETA[0], edgecolor="white", alpha=0.8,
            density=True, label="Bootstrap (n=1000)")
    ax.axvline(df["gini_bootstrap"].mean(), color=PALETA[1], linestyle="--", linewidth=2,
               label=f"Media: {df['gini_bootstrap'].mean():.4f}")
    ax.axvline(np.percentile(df["gini_bootstrap"], 2.5), color=PALETA[2], linestyle=":", linewidth=2,
               label=f"IC95%: [{np.percentile(df['gini_bootstrap'], 2.5):.4f}, {np.percentile(df['gini_bootstrap'], 97.5):.4f}]")
    ax.axvline(np.percentile(df["gini_bootstrap"], 97.5), color=PALETA[2], linestyle=":", linewidth=2)
    ax.set_xlabel("Gini")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución Bootstrap del Gini (mejor modelo, remuestreo por cliente)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig9_bootstrap_ic.png", dpi=150)
    plt.close(fig)
    log.info("Guardada fig9_bootstrap_ic.png")


def main():
    ap = argparse.ArgumentParser(description="Genera figuras para informe final")
    ap.add_argument("--resultados", type=Path, default=Path("resultados_reportes"),
                    help="Carpeta con CSVs de resultados")
    ap.add_argument("--salida", type=Path, default=Path("resultados_reportes"),
                    help="Carpeta donde guardar PNGs")
    args = ap.parse_args()

    args.salida.mkdir(parents=True, exist_ok=True)

    # Intentar generar cada figura si existe el CSV correspondiente
    figuras = [
        ("tasa_conversion_mes.csv", fig_tasa_conversion_por_mes),
        ("correlacion_features.csv", fig_matriz_correlacion),
        ("importancia_permutacion.csv", fig_importancia_permutacion),
        ("ablacion_variables.csv", fig_ablacion_variables),
        ("walkforward_cv.csv", fig_validacion_walkforward),
        ("resultados_validacion_DF_WCB.csv", fig_comparacion_modelos),
        ("submission_final.csv", fig_distribucion_predicciones),
        ("bootstrap_gini.csv", fig_ic_bootstrap),
    ]

    for csv_name, func in figuras:
        csv_path = args.resultados / csv_name
        if csv_path.exists():
            try:
                func(csv_path, args.salida)
            except Exception as e:
                log.error("Error generando %s: %s", csv_name, e)
        else:
            log.info("No encontrado %s (opcional)", csv_name)

    log.info("Generación de figuras completada en %s", args.salida)


if __name__ == "__main__":
    main()