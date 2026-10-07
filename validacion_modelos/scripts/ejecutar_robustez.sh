#!/usr/bin/env bash
# Ejecuta R1-R9 en orden desde la raiz del repo: bash validacion_modelos/scripts/ejecutar_robustez.sh [desde]
# (desde = numero de script inicial, p. ej. 11). Cada log en resultados/logs/ con su codigo de salida.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
DESDE=${1:-09}
for s in 09_r1_walkforward_candidatos 10_r1_comparacion_formal 11_r2_num_arboles 12_r3_hiperparametros \
         13_r4_semillas_bootstrap 14_r5_segmentos 15_r6_perturbacion 16_r7_degradacion 17_r8_placebo \
         18_r9_curva_aprendizaje; do
  [ "${s:0:2}" \< "$DESDE" ] && continue
  inicio=$(date +%s)
  $PY -X faulthandler validacion_modelos/scripts/$s.py > validacion_modelos/resultados/logs/$s.log 2>&1
  rc=$?
  echo "$s rc=$rc $(( $(date +%s) - inicio ))s" | tee -a validacion_modelos/resultados/logs/ejecutar_robustez.log
  if [ $rc -ne 0 ]; then echo "FALLO en $s"; exit $rc; fi
done
echo "TODO OK"
