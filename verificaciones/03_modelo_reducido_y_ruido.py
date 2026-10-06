import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
tr=pd.read_csv('datos_entrada/train.csv')
D=pd.get_dummies(tr,columns=['banda_riesgo'],dtype=int,prefix='br')
for c in D.select_dtypes('bool'): D[c]=D[c].astype(int)
D=pd.get_dummies(D,columns=['ocupacion','region','canal_adquisicion','dispositivo_principal'],dtype=int)
base=['br_high','br_low','br_medium','numero_productos','dias_ultima_transaccion','activo_movil','tiene_tarjeta_credito']
R2=base+['dias_ultima_interaccion','ratio_deuda_ingresos','antiguedad_cuenta_meses','tiene_prestamo']
allf=[c for c in D.columns if c not in ['id_cliente','mes','objetivo']]
meses=np.sort(D.mes.unique()); y=D.objetivo.values
def run(cols,params,nm):
    out=[];its=[]
    for corte,hi in [(meses[-3],None),(meses[-6],meses[-3])]:
        tri=(D.mes<corte).values; vai=((D.mes>=corte)&((D.mes<hi) if hi else True)).values; g=[]
        for s in (42,43,44):
            m=lgb.LGBMClassifier(**params,n_estimators=3000,random_state=s,n_jobs=-1,verbose=-1)
            m.fit(D.loc[tri,cols],y[tri],eval_set=[(D.loc[vai,cols],y[vai])],eval_metric='auc',callbacks=[lgb.early_stopping(100,verbose=False)])
            g.append(2*roc_auc_score(y[vai],m.predict_proba(D.loc[vai,cols])[:,1])-1); its.append(m.best_iteration_)
        out.append(round(float(np.mean(g)),4))
    print(f'{nm:55s} sep-nov={out[0]}  jun-ago={out[1]}  iters~{int(np.mean(its))}',flush=True)
P0=dict(learning_rate=0.03,num_leaves=31,min_child_samples=50,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=1.0)
P1=dict(learning_rate=0.03,num_leaves=8,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=10.0)
run(allf,P0,'todas estáticas (OHE) / params defecto')
run(base,P0,'7 feats señal / params defecto')
run(base,P1,'7 feats señal / params regularizados (8 hojas)')
run(R2,P1,'11 feats / params regularizados')
run(allf,P1,'todas estáticas / params regularizados')
# logística con interacción banda x productos
def logit(cols_extra=True):
    out=[]
    for corte,hi in [(meses[-3],None),(meses[-6],meses[-3])]:
        tri=(D.mes<corte).values; vai=((D.mes>=corte)&((D.mes<hi) if hi else True)).values
        X=D[base].copy().astype(float)
        for b in ['br_low','br_medium']: X[b+'_x_prod']=X[b]*X.numero_productos
        X['trans_z']=X.dias_ultima_transaccion/365
        Xs=X.drop(columns=['dias_ultima_transaccion']); Xs['trans']=X.trans_z
        mdl=LogisticRegression(C=1.0,max_iter=2000).fit((Xs[tri]-Xs[tri].mean())/Xs[tri].std(),y[tri])
        p=mdl.predict_proba((Xs[vai]-Xs[tri].mean())/Xs[tri].std())[:,1]; out.append(round(2*roc_auc_score(y[vai],p)-1,4))
    print(f'{"Regresión logística (7 feats + banda×productos)":55s} sep-nov={out[0]}  jun-ago={out[1]}')
logit()
# ruido: SE de Gini por bootstrap de clientes (val sep-nov) con el modelo de 7 feats
tri=(D.mes<meses[-3]).values; vai=~tri
m=lgb.LGBMClassifier(**P1,n_estimators=400,random_state=1,n_jobs=-1,verbose=-1).fit(D.loc[tri,base],y[tri]); p=m.predict_proba(D.loc[vai,base])[:,1]
V=pd.DataFrame({'id':D.id_cliente[vai].values,'y':y[vai],'p':p}); ids=V.id.unique(); grp={k:v for k,v in V.groupby('id')}
rng=np.random.RandomState(0); gs=[]
for _ in range(200):
    s=rng.choice(ids,len(ids)); B=pd.concat([grp[i] for i in s]); gs.append(2*roc_auc_score(B.y,B.p)-1)
print('Gini val (400 árboles, 7 feats)=%.4f ; SE bootstrap por cliente=%.4f ; IC95%%=[%.3f,%.3f]'%(2*roc_auc_score(V.y,V.p)-1,np.std(gs),*np.percentile(gs,[2.5,97.5])))
