import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
tr=pd.read_csv('datos_entrada/train.csv')
cats=['ocupacion','region','canal_adquisicion','banda_riesgo','dispositivo_principal']
D=tr.copy()
for c in D.select_dtypes('bool'): D[c]=D[c].astype(int)
for c in cats: D[c]=LabelEncoder().fit_transform(D[c].astype(str))
feats=[c for c in D.columns if c not in ['id_cliente','mes','objetivo']]
tri=D.mes<202609; vai=~tri
# 1) AUC univariado en validación (signo irrelevante)
uni={c:max(roc_auc_score(D.objetivo[vai],D.loc[vai,c]),1-roc_auc_score(D.objetivo[vai],D.loc[vai,c])) for c in feats if c not in cats}
print('AUC univariado (val sep-nov):'); print(pd.Series(uni).sort_values(ascending=False).round(4).to_string())
# 2) RF como en el notebook: importancia por impureza (train completo, in-sample) vs permutación (val)
rf=RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=20,random_state=42,n_jobs=-1).fit(D.loc[tri,feats],D.objetivo[tri])
imp=pd.Series(rf.feature_importances_,index=feats)
base=roc_auc_score(D.objetivo[vai],rf.predict_proba(D.loc[vai,feats])[:,1]); rng=np.random.RandomState(0); perm={}
for c in feats:
    dl=[]
    for _ in range(5):
        Xv=D.loc[vai,feats].copy(); Xv[c]=rng.permutation(Xv[c].values); dl.append(base-roc_auc_score(D.objetivo[vai],rf.predict_proba(Xv)[:,1]))
    perm[c]=np.mean(dl)
out=pd.DataFrame({'imp_impureza_RF':imp,'perm_AUC_drop_val':pd.Series(perm)}).sort_values('perm_AUC_drop_val',ascending=False)
print('\nRF AUC val=%.4f (Gini %.4f)'%(base,2*base-1)); print(out.round(4).to_string())
# 3) ¿banda_riesgo label-encoded? orden alfabético
print('\nLabelEncoder banda_riesgo:', dict(zip(sorted(tr.banda_riesgo.unique()),range(3))), '(high=0, low=1, medium=2 -> NO ordinal)')
