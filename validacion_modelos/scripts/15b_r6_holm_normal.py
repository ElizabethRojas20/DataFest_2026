# -*- coding: utf-8 -*-
"""15b. R6 (correccion declarada): con NB=200 el p bootstrap minimo es 1/201 y Holm sobre ~22
variables no puede bajar de ~0,11. Se recalcula p unilateral normal p = 1 - Phi(caida/SE) con Holm
por modelo. Salida: robustez_r6_importancia_holm_normal.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from scipy import stats
from robustez_comun import DIR_RES, holm

im = pd.read_csv(DIR_RES / "robustez_r6_importancia.csv")
im["z"] = im.caida_gini / im.se
im["p_normal"] = stats.norm.sf(im.z)
im["p_holm_normal"] = im.groupby("modelo").p_normal.transform(lambda p: holm(p.to_numpy()))
im.to_csv(DIR_RES / "robustez_r6_importancia_holm_normal.csv", index=False)
print(im[im.p_holm_normal < 0.05].groupby("modelo").variable.apply(list).to_string())
