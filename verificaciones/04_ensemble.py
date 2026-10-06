import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb, warnings
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
tr=pd.read_csv('datos_entrada/train.csv')
D=pd.get_dummies(tr,columns=['banda_riesgo','ocupacion','region','canal_adquisicion','dispositivo_principal'],dtype=int)
for c in D.select_dtypes('bool'): D[c]=D[c].astype(int)
F=[c for c in D.columns if c not in ['id_cliente','mes','objetivo']]; y=D.objetivo.values; meses=np.sort(D.mes.unique())
P1=dict(learning_rate=0.03,num_leaves=8,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=10.0)
for nm,(corte,hi) in {'sep-nov':(meses[-3],None),'jun-ago':(meses[-6],meses[-3])}.items():
    tri=(D.mes<corte).values; vai=((D.mes>=corte)&((D.mes<hi) if hi else True)).values
    Xt,Xv,yt,yv=D.loc[tri,F],D.loc[vai,F],y[tri],y[vai]; P={}
    m=lgb.LGBMClassifier(**P1,n_estimators=2000,random_state=1,n_jobs=-1,verbose=-1).fit(Xt,yt,eval_set=[(Xv,yv)],eval_metric='auc',callbacks=[lgb.early_stopping(100,verbose=False)]); P['lgbm_reg']=m.predict_proba(Xv)[:,1]
    m=xgb.XGBClassifier(learning_rate=0.03,max_depth=3,min_child_weight=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=10,n_estimators=2000,early_stopping_rounds=100,eval_metric='auc',tree_method='hist',n_jobs=-1).fit(Xt,yt,eval_set=[(Xv,yv)],verbose=False); P['xgb_reg']=m.predict_proba(Xv)[:,1]
    m=CatBoostClassifier(learning_rate=0.05,depth=4,l2_leaf_reg=10,iterations=2000,eval_metric='AUC',early_stopping_rounds=100,verbose=False,allow_writing_files=False,thread_count=-1).fit(Xt,yt,eval_set=(Xv,yv)); P['cat_reg']=m.predict_proba(Xv)[:,1]
    m=RandomForestClassifier(n_estimators=300,min_samples_leaf=100,max_features=0.5,max_depth=8,n_jobs=-1,random_state=1).fit(Xt,yt); P['rf_reg']=m.predict_proba(Xv)[:,1]
    g={k:round(2*roc_auc_score(yv,v)-1,4) for k,v in P.items()}
    R={k:rankdata(v)/len(v) for k,v in P.items()}
    g['blend_rank_4']=round(2*roc_auc_score(yv,sum(R.values()))-1,4); g['blend_lgbm+cat']=round(2*roc_auc_score(yv,R['lgbm_reg']+R['cat_reg'])-1,4)
    print(nm,g,flush=True)
