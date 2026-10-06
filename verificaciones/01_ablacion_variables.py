import sys, importlib.util, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
spec=importlib.util.spec_from_file_location('wcb','pipelines/Pipeline_DF_WCB.py'); wcb=importlib.util.module_from_spec(spec); sys.modules['wcb']=wcb; spec.loader.exec_module(wcb)
import logging; wcb.log.setLevel(logging.ERROR)
from pathlib import Path
cfg=wcb.Config(ruta_datos=Path("datos_entrada")); tr,te=wcb.cargar_datos(cfg)
d=wcb.construir_dataset(tr,te,cfg)
X=d.X_train.copy(); y=d.y_train.values; mes=d.mes_train
# antigüedad en el panel SIN mirar el futuro: nº de filas del cliente hasta el mes actual
df=wcb.preparar_base(tr,te,cfg); esT=df['_es_test'].to_numpy()
df['tenure']=df.groupby('id_cliente').cumcount()+1
df['cohorte_ene']=df.groupby('id_cliente')['mes'].transform('min').eq(202601).astype(int)
Xt=df.loc[~esT,['tenure','cohorte_ene']].reset_index(drop=True)
X['tenure']=Xt.tenure.values; X['cohorte_ene']=Xt.cohorte_ene.values
X['int_igual_trans']=(X.dias_ultima_interaccion==X.dias_ultima_transaccion).astype(int)
X['int_menos_trans']=X.dias_ultima_interaccion-X.dias_ultima_transaccion
X['low_x_prod']=X.banda_riesgo_low*X.numero_productos
orig=cfg.cols_numericas+cfg.cols_booleanas; ohe=[c for c in X.columns if any(c.startswith(p+'_') for p in cfg.cols_categoricas)]
static=orig+ohe
temporal_int=[c for c in X.columns if c.startswith('dias_ultima_interaccion_')]
sel=open('resultados_reportes/features_seleccionadas_DF_WCB.txt').read().split('\n')
sets={
 'A pipeline actual (50 feats)':sel,
 'B solo estáticas+valor actual':static,
 'C B + lags/delta/roll de interacción':static+temporal_int,
 'D B + tenure(+cohorte ene)':static+['tenure','cohorte_ene'],
 'E D + flags interacción/transacción':static+['tenure','cohorte_ene','int_igual_trans','int_menos_trans'],
 'F E + low_x_prod':static+['tenure','cohorte_ene','int_igual_trans','int_menos_trans','low_x_prod'],
 'G A sin roll3_sum ni meses_en_ventana':[c for c in sel if not c.endswith('_sum') and c!='meses_en_ventana3'],
 'H A + tenure':sel+['tenure','cohorte_ene'],
}
P=wcb.PARAMS_DEFECTO['lightgbm']
meses=np.sort(np.unique(mes))
folds={'val sep-nov (train ene-ago)':meses[-3],'val jun-ago (train ene-may)':meses[-6]}
rows=[]
for name,cols in sets.items():
    r={'modelo':name,'n_feats':len(cols)}
    for fn,corte in folds.items():
        hi={'val sep-nov (train ene-ago)':None,'val jun-ago (train ene-may)':meses[-3]}[fn]
        tr_i=mes<corte; va_i=(mes>=corte)&((mes<hi) if hi else True)
        g=[]
        for s in (42,43,44):
            m=lgb.LGBMClassifier(**P,n_estimators=2000,random_state=s,n_jobs=-1,verbose=-1)
            m.fit(X.loc[tr_i,cols],y[tr_i],eval_set=[(X.loc[va_i,cols],y[va_i])],eval_metric='auc',callbacks=[lgb.early_stopping(100,verbose=False)])
            g.append(2*roc_auc_score(y[va_i],m.predict_proba(X.loc[va_i,cols])[:,1])-1)
        r[fn]=round(float(np.mean(g)),4)
    rows.append(r); print(r,flush=True)
