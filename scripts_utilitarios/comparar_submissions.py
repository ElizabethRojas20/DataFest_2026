#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparar dos submissions y decidir cuál enviar.
Criterio de reemplazo:
- Si Gini_validación_nuevo > Gini_validación_actual + 0.002 (≈ 2x error estándar)
- O si Gini similar pero IC95% no se solapan y nuevo es mejor
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def comparar_submissions(nuevo: str, actual: str, gini_nuevo: float = None, gini_actual: float = None,
                         ic_nuevo: tuple = None, ic_actual: tuple = None):
    """
    Compara dos submissions.

    Args:
        nuevo: path a submission candidato
        actual: path a submission actual (producción)
        gini_nuevo: Gini en validación del modelo nuevo
        gini_actual: Gini en validación del modelo actual
        ic_nuevo: (lower, upper) IC95% del modelo nuevo
        ic_actual: (lower, upper) IC95% del modelo actual
    """
    print(f"\n{'='*60}")
    print(f"COMPARACIÓN DE SUBMISSIONS")
    print(f"{'='*60}")
    print(f"NUEVO:  {nuevo}")
    print(f"ACTUAL: {actual}")
    print()

    # Verificar formato
    df_n = pd.read_csv(nuevo)
    df_a = pd.read_csv(actual)

    print(f"Filas nuevo:  {len(df_n)}")
    print(f"Filas actual: {len(df_a)}")

    # Verificar columnas
    assert list(df_n.columns) == ["id_cliente", "prediccion"], "Columnas incorrectas en nuevo"
    assert list(df_a.columns) == ["id_cliente", "prediccion"], "Columnas incorrectas en actual"
    print("✓ Columnas correctas: id_cliente, prediccion")

    # Verificar orden id_cliente
    if not (df_n["id_cliente"].values == df_a["id_cliente"].values).all():
        print("⚠ ADVERTENCIA: Orden de id_cliente difiere entre submissions")
    else:
        print("✓ Mismo orden de id_cliente")

    # Estadísticas de predicciones
    print(f"\nEstadísticas predicciones:")
    print(f"  Nuevo:  media={df_n['prediccion'].mean():.4f}, std={df_n['prediccion'].std():.4f}, "
          f"min={df_n['prediccion'].min():.4f}, max={df_n['prediccion'].max():.4f}")
    print(f"  Actual: media={df_a['prediccion'].mean():.4f}, std={df_a['prediccion'].std():.4f}, "
          f"min={df_a['prediccion'].min():.4f}, max={df_a['prediccion'].max():.4f}")

    # Correlación entre predicciones
    corr = df_n["prediccion"].corr(df_a["prediccion"])
    print(f"  Correlación (Pearson): {corr:.4f}")

    # Diferencias
    diff = df_n["prediccion"] - df_a["prediccion"]
    print(f"  Diff media: {diff.mean():.6f}")
    print(f"  Diff std:   {diff.std():.6f}")
    print(f"  Max |diff|: {diff.abs().max():.6f}")

    # Decisión basada en Gini validación
    print(f"\n{'='*60}")
    print(f"DECISIÓN BASADA EN VALIDACIÓN")
    print(f"{'='*60}")

    if gini_nuevo is not None and gini_actual is not None:
        print(f"Gini nuevo:  {gini_nuevo:.4f}")
        print(f"Gini actual: {gini_actual:.4f}")
        diff_gini = gini_nuevo - gini_actual
        print(f"Diferencia:  {diff_gini:+.4f}")

        # Error estándar ≈ 0.0092 (del informe)
        umbral = 0.002  # ~2x error estándar
        print(f"Umbral (2×SE): {umbral:.4f}")

        if diff_gini > umbral:
            print(f"\n✅ REEMPLAZAR: Nuevo supera actual por > {umbral:.4f} Gini")
            decision = "REEMPLAZAR"
        elif diff_gini < -umbral:
            print(f"\n❌ MANTENER: Actual supera nuevo por > {umbral:.4f} Gini")
            decision = "MANTENER"
        else:
            print(f"\n⚖️  SIMILARES: Diferencia < {umbral:.4f} Gini")
            decision = "SIMILARES"

            # Si hay IC95%, usarlos para desempatar
            if ic_nuevo and ic_actual:
                print(f"IC95% nuevo:  [{ic_nuevo[0]:.4f}, {ic_nuevo[1]:.4f}]")
                print(f"IC95% actual: [{ic_actual[0]:.4f}, {ic_actual[1]:.4f}]")

                # Verificar solapamiento
                if ic_nuevo[0] > ic_actual[1]:
                    print("✅ REEMPLAZAR: IC95% no se solapan, nuevo es mejor")
                    decision = "REEMPLAZAR"
                elif ic_actual[0] > ic_nuevo[1]:
                    print("❌ MANTENER: IC95% no se solapan, actual es mejor")
                    decision = "MANTENER"
                else:
                    print("⚖️  IC95% se solapan: mantener actual por parsimonia")
                    decision = "MANTENER (pars.)"
    else:
        print("No se proporcionaron métricas de validación.")
        print("Proporcione --gini-nuevo --gini-actual para decisión automática.")
        decision = "MANUAL"

    print(f"\n>>> DECISIÓN FINAL: {decision}")
    print(f"{'='*60}\n")

    return decision


def main():
    ap = argparse.ArgumentParser(description="Comparar dos submissions")
    ap.add_argument("nuevo", type=Path, help="Submission candidato (nuevo)")
    ap.add_argument("actual", type=Path, help="Submission actual (producción)")
    ap.add_argument("--gini-nuevo", type=float, help="Gini validación modelo nuevo")
    ap.add_argument("--gini-actual", type=float, help="Gini validación modelo actual")
    ap.add_argument("--ic-nuevo", type=float, nargs=2, metavar=("LOWER", "UPPER"),
                    help="IC95% modelo nuevo")
    ap.add_argument("--ic-actual", type=float, nargs=2, metavar=("LOWER", "UPPER"),
                    help="IC95% modelo actual")
    args = ap.parse_args()

    comparar_submissions(
        args.nuevo, args.actual,
        gini_nuevo=args.gini_nuevo,
        gini_actual=args.gini_actual,
        ic_nuevo=tuple(args.ic_nuevo) if args.ic_nuevo else None,
        ic_actual=tuple(args.ic_actual) if args.ic_actual else None
    )


if __name__ == "__main__":
    main()