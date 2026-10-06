#!/bin/bash
# =============================================================================
# SCRIPT DE EJECUCIÓN FINAL - DataFest 2026
# =============================================================================
# Ejecuta el pipeline completo con la configuración óptima validada.
# Genera submission_final.csv y todos los artefactos de trazabilidad.
#
# Tiempo estimado: ~30-45 minutos (30 trials Optuna, 4 folds, bootstrap 1000)
# =============================================================================

set -e  # Salir en error

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}DataFest 2026 - Pipeline Final de Propensión de Conversión${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "pipelines/Pipeline_DF_WCB.py" ]; then
    echo -e "${RED}Error: No se encuentra pipelines/Pipeline_DF_WCB.py${NC}"
    echo "Ejecutar desde la raíz del repositorio DataFest_2026"
    exit 1
fi

# Verificar datos de entrada
if [ ! -f "datos_entrada/train.csv" ] || [ ! -f "datos_entrada/test.csv" ]; then
    echo -e "${RED}Error: No se encuentran train.csv y/o test.csv en datos_entrada/${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Directorio y datos verificados${NC}"
echo ""

# Configuración óptima validada
echo -e "${YELLOW}Configuración:${NC}"
echo "  - Modo features: estatico (ablación A ≈ B, C no mejora)"
echo "  - Walk-forward CV: 4 folds (últimos 4 meses)"
echo "  - Selector: Importancia por permutación (insesgado)"
echo "  - LightGBM: Preset regularizado (reg_lambda=5, min_child_samples=100, subsample=0.7, colsample=0.7)"
echo "  - Comparación obligatoria: LightGBM_Regularizado vs LightGBM_Sin_Regularizar"
echo "  - Optuna: 30 trials para LightGBM"
echo "  - Bootstrap IC95%: 1000 iteraciones por cliente"
echo "  - Seeds finales: 3 para estabilizar ranking"
echo ""

# Preguntar confirmación
read -p "¿Continuar con la ejecución completa? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelado por el usuario."
    exit 0
fi

echo ""
echo -e "${BLUE}Iniciando pipeline...${NC}"
echo ""

# Crear directorio de salida
mkdir -p resultados_reportes

# Ejecutar pipeline
# Nota: --bootstrap-ic añade ~2 min por modelo; descomenta si lo necesitas
python pipelines/Pipeline_DF_WCB.py \
    --datos datos_entrada \
    --salida resultados_reportes/submission_final.csv \
    --trials 30 \
    --n-folds 4 \
    --modo-features estatico \
    --top-k 80 \
    --umbral-ohe 10 \
    --n-seeds 3 \
    --semilla 42 \
    --bootstrap-ic \
    --n-bootstrap 1000

echo ""
echo -e "${GREEN}===============================================================================${NC}"
echo -e "${GREEN}✓ Pipeline completado exitosamente${NC}"
echo -e "${GREEN}===============================================================================${NC}"
echo ""
echo -e "${YELLOW}Artefactos generados en resultados_reportes/:${NC}"
echo "  - submission_final.csv          (archivo para enviar al concurso)"
echo "  - resultados_validacion_DF_WCB.csv  (tabla comparativa modelos)"
echo "  - features_seleccionadas_DF_WCB.txt (lista de features finales)"
echo "  - val_predictions.csv           (predicciones OOF para auditoría)"
echo "  - optuna_trials.jsonl           (trazas de Optuna)"
echo ""
echo -e "${YELLOW}Próximos pasos:${NC}"
echo "  1. Verificar submission_final.csv:"
echo "     head resultados_reportes/submission_final.csv"
echo "  2. Generar figuras del informe:"
echo "     python pipelines/generar_graficas.py --resultados resultados_reportes"
echo "  3. Comparar con submission anterior (si existe):"
echo "     python scripts_utilitarios/comparar_submissions.py \\"
echo "       resultados_reportes/submission_final.csv \\"
echo "       resultados_reportes/submission_final.csv.anterior"
echo ""
echo -e "${BLUE}¡Listo para enviar al concurso!${NC}"