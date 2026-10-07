# -*- coding: utf-8 -*-
"""
03. Aplicacion de las redes GUARDADAS (no se reentrenan) a ago-nov y a diciembre.

Procedencia (notebooks/01_eda_completo.ipynb, celdas 16-18, y comprobaciones de este script):
  * preprocessor_nn.joblib: StandardScaler + OneHotEncoder ajustados con TODO train (ene-nov,
    n_samples_seen_ = 110 100): fuga no supervisada leve (medias/escalas de sep-nov).
  * Entrenamiento de las redes con ene-ago (X_train_nn = 80 300 filas); "validacion" = sep-nov.
  * nn_baseline / nn_optimizado (Keras): EarlyStopping(restore_best_weights) y ReduceLROnPlateau
    sobre val_auc de sep-nov -> su Gini en sep-nov NO es fuera de muestra (optimista).
  * mlp_sklearn: early_stopping con 10 % aleatorio de ene-ago -> sep-nov si es fuera de muestra
    (salvo el escalado).
  * Agosto esta DENTRO del entrenamiento de las tres redes -> no se usa para evaluarlas.

Salidas: resultados/nn_predicciones.csv (ago-nov, + dui permutada), resultados/nn_pred_dic.csv,
         resultados/nn_procedencia.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib
import numpy as np
import pandas as pd

from comun import COL_DUI, DIR_RES, MESES_FOLD, RAIZ, SEMILLA, permutar_en_mes

N_PERM = 3


def main():
    import keras
    tr = pd.read_csv(RAIZ / "datos_entrada" / "train.csv")
    te = pd.read_csv(RAIZ / "datos_entrada" / "test.csv")
    pre = joblib.load(RAIZ / "modelos" / "preprocessor_nn.joblib")
    cols = list(pre.feature_names_in_)

    # --- procedencia: comprobar meses de entrenamiento / validacion y ajuste del escalador
    Xtr_nn = np.load(RAIZ / "modelos" / "X_train_nn.npy")
    Xva_nn = np.load(RAIZ / "modelos" / "X_val_nn.npy")
    m_ago = tr["mes"] <= 202608
    dif_tr = np.abs(pre.transform(tr.loc[m_ago, cols]) - Xtr_nn).max()
    dif_va = np.abs(pre.transform(tr.loc[~m_ago, cols]) - Xva_nn).max()
    sc = pre.named_transformers_["num"]
    proc = pd.DataFrame([
        dict(chequeo="filas X_train_nn (= ene-ago)", valor=Xtr_nn.shape[0], esperado=int(m_ago.sum())),
        dict(chequeo="filas X_val_nn (= sep-nov)", valor=Xva_nn.shape[0], esperado=int((~m_ago).sum())),
        dict(chequeo="max |transform(ene-ago) - X_train_nn|", valor=float(dif_tr), esperado=0),
        dict(chequeo="max |transform(sep-nov) - X_val_nn|", valor=float(dif_va), esperado=0),
        dict(chequeo="scaler n_samples_seen_ (110100 = todo train)", valor=float(sc.n_samples_seen_), esperado=int(m_ago.sum())),
    ])
    proc.to_csv(DIR_RES / "nn_procedencia.csv", index=False)
    print(proc.to_string(index=False))

    # --- mismo orden que construir_dataset (id, mes) para alinear con 01_walkforward.py
    tr = tr.sort_values(["id_cliente", "mes"], kind="mergesort").reset_index(drop=True)
    perms = {k: permutar_en_mes(tr[COL_DUI].to_numpy(), tr["mes"].to_numpy(), seed=SEMILLA + 100 + k)
             for k in range(N_PERM)}
    val = tr["mes"].isin(MESES_FOLD).to_numpy()
    redes = {
        "mlp_sklearn": joblib.load(RAIZ / "modelos" / "mlp_sklearn.joblib"),
        "nn_baseline": keras.models.load_model(RAIZ / "modelos" / "nn_baseline.keras"),
        "nn_optimizado": keras.models.load_model(RAIZ / "modelos" / "nn_optimizado.keras"),
    }

    def pred(nombre, m, df):
        Z = pre.transform(df[cols])
        if nombre == "mlp_sklearn":
            return m.predict_proba(Z)[:, 1]
        return m.predict(Z, verbose=0, batch_size=4096).ravel()

    out = pd.DataFrame({"id_cliente": tr.loc[val, "id_cliente"].to_numpy(), "mes": tr.loc[val, "mes"].to_numpy(),
                        "y": tr.loc[val, "objetivo"].to_numpy()})
    dic = pd.DataFrame({"id_cliente": te["id_cliente"].to_numpy()})
    for nombre, m in redes.items():
        out[f"{nombre}__hon"] = pred(nombre, m, tr.loc[val])
        for k in range(N_PERM):
            dfp = tr.loc[val].copy(); dfp[COL_DUI] = perms[k][val]
            out[f"{nombre}__perm{k}"] = pred(nombre, m, dfp)
        dic[f"{nombre}__final"] = pred(nombre, m, te)
        rng = np.random.default_rng(SEMILLA + 200)
        for k in range(N_PERM):
            tp = te.copy(); tp[COL_DUI] = te[COL_DUI].to_numpy()[rng.permutation(len(te))]
            dic[f"{nombre}__perm{k}"] = pred(nombre, m, tp)
        print(nombre, "ok", flush=True)
    out.to_csv(DIR_RES / "nn_predicciones.csv", index=False)
    dic.to_csv(DIR_RES / "nn_pred_dic.csv", index=False)


if __name__ == "__main__":
    main()
