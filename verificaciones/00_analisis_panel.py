"""Estructura del panel cliente-mes de DataFest 2026. Ejecutar desde la RAÍZ del repo:
    python verificaciones/00_analisis_panel.py
"""
import numpy as np, pandas as pd
from scipy import stats
tr = pd.read_csv('datos_entrada/train.csv').sort_values(['id_cliente', 'mes']).reset_index(drop=True)
te = pd.read_csv('datos_entrada/test.csv')
mi = {m: i for i, m in enumerate(sorted(tr.mes.unique()))}
tr['m'] = tr.mes.map(mi)
tr['first'] = tr.groupby('id_cliente').m.transform('min')
tr['tenure'] = tr.m - tr['first'] + 1
cl = tr.groupby('id_cliente').agg(first=('first', 'first'), last=('m', 'max'), conv=('objetivo', 'max'), n=('m', 'size'))
print('filas', tr.shape, '| clientes', len(cl), '| positivos', tr.objetivo.sum())
print('entradas por mes (índice 0 = ene):\n', cl['first'].value_counts().sort_index().to_string())
print('clientes que convierten alguna vez: %d (%.1f%%)' % (cl.conv.sum(), 100 * cl.conv.mean()))
print('abandono silencioso (sin conversión y última fila < nov):', int(((cl.conv == 0) & (cl['last'] < 10)).sum()))
surv = set(tr[(tr.mes == 202611) & (tr.objetivo == 0)].id_cliente)
print('no convertidos de nov:', len(surv), '| todos en dic:', surv <= set(te.id_cliente), '| nuevos en dic:', int((~te.id_cliente.isin(tr.id_cliente)).sum()))
print('hazard por tenure:\n', tr.groupby('tenure').objetivo.agg(['mean', 'size']).round(4).to_string())
print('chi2 tasa por mes p=%.4f' % stats.chi2_contingency(pd.crosstab(tr.mes, tr.objetivo))[1])
g = tr.groupby('id_cliente')
multi = g.size() > 1
for c in [c for c in tr.columns if c not in ('id_cliente', 'mes', 'm', 'first', 'tenure', 'objetivo')]:
    pct = (g[c].nunique() > 1)[multi].mean() * 100
    if pct > 0: print(f'VARIABLE DINÁMICA: {c} cambia en {pct:.1f}% de clientes con >1 mes')
d = g.dias_ultima_interaccion.diff().dropna()
print('delta interacción: media %.1f sd %.1f | ==0: %.3f | ==+30: %.4f' % (d.mean(), d.std(), (d == 0).mean(), (d == 30).mean()))
print('interacción == transacción: %.3f | interacción <= transacción: %.3f' % ((tr.dias_ultima_interaccion == tr.dias_ultima_transaccion).mean(), (tr.dias_ultima_interaccion <= tr.dias_ultima_transaccion).mean()))
