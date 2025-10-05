import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb


# IMPORT DES DONNEES D APPRENTISSAGE
registrations = # FILE PATH
calendar = # FILE PATH
actuals = # FILE PATH



# FONCTION FULL RANGE DE DATE
def prep_sales(df,mode='train'):
  
    # Conversion des dates au bon format
    to_datetime_cols = ["CONSUMPTION_DATE","SALES_DATE"]
    if to_datetime_cols:
        for col in to_datetime_cols:
            if not pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors="coerce")
  
    # reconstitution de la range complète par jour de consommation pour avoir un "pattern" continu
    all_dates = (
        df.groupby("CONSUMPTION_DATE")["SALES_DATE"]
          .agg(["min", "max"])
          .apply(lambda x: pd.date_range(x["min"], x["max"]), axis=1)
    )
  
    full_index = (
        all_dates.explode()
                 .reset_index()
                 .rename(columns={0: "SALES_DATE"})
    )

    df_full = full_index.merge(df, on=["CONSUMPTION_DATE", "SALES_DATE"], how="left")


    # remplissage des nouveaux jours sans données (car rajoutés à l'étape précédente) par les volumes et revenus cumulés
    for col in ["Sales", "Revenues"]:
        df_full[col] = df_full[col].fillna(0)
        df_full[f"{col}_cum"] = (
            df_full.groupby("CONSUMPTION_DATE")[col]
                   .cumsum()
                   .groupby(df_full["CONSUMPTION_DATE"])
                   .ffill()
                   .fillna(0)
        )


    # On garde que la range pertinente (longueur du pattern)
    df_full['datedif'] = (df_full['CONSUMPTION_DATE'] - df_full['SALES_DATE']).dt.days
    df_full = df_full[(df_full["datedif"]>="A DETERMINER")&(df_full["datedif"]<="A DETERMINER")]


    # Mode train uniquement
    # Selection de la période si besoin pour isoler des tendances récentes etc
    if mode == "train":
        df_full = df_full[
            (df_full["CONSUMPTION_DATE"] >= datetime(2010, 1, 1)) &
            (df_full["CONSUMPTION_DATE"] < datetime(2024, 12, 31))
        ]


    # rajout du prix
    df_full["daily_price"] = df_full["Revenues_cum"] / df_full["Sales_cum"]

    # garde fou pour empêcher les valeurs impossibles
    df_full['daily_price'] = df_full['daily_price'].replace(0,pd.NA)
    df_full['daily_price'] = df_full['daily_price'].replace([np.inf,-np.inf],pd.NA)

    # remplacement des valeurs aberrantes par des NaNs pour par la suite les interpoler
    # identification par Zscore
    avg_price, std_price = df_full['daily_price'].mean(), df_full['daily_price'].std()
  
    df_full['zscore_price'] = (df_full['daily_price'] - avg_price) / std_price
    df_full['daily_price'] = np.where(
        (df_full['zscore_price'] >= 4) | (df_full['zscore_price'] <= -2),
        np.nan,
        df_full['daily_price']
    )

    # on fill les nouveaux NA par linéarisation avec les valeurs voisines
    df_full["daily_price"] = (df_full.groupby("CONSUMPTION_DATE")["daily_price"].transform(lambda x: x.interpolate(method="linear", limit_direction="both").bfill().ffill()))

    return df_full





# FONCTION AJOUT DE LA TARGET POUR LES JEUX TRAIN TEST SEULEMENT
def target(df):
  
    # Merge avec actuals pour ajouter la target
    df = pd.merge(df, actuals, how="left", left_on="CONSUMPTION_DATE", right_on="Date")
    df.drop(columns="Date", inplace=True)

    # Identification des cas où Target < Sales_cum au datedif=0
    mask_outlier = (df['datedif'] == 0) & (df['Target'] < df['Sales_cum'])
    corrections = df.loc[mask_outlier, ['CONSUMPTION_DATE', 'Sales_cum']].rename(
        columns={'Sales_cum': 'Target_corrected'}
    )
    df = df.merge(corrections, on="CONSUMPTION_DATE", how="left")
    df['Target'] = np.where(df['Target_corrected'].notna(), df['Target_corrected'], df['Target'])
    df.drop(columns=['Target_corrected'], inplace=True)

    # Suppression outliers Target avec Zscore
    target_outlier = df.groupby('CONSUMPTION_DATE')['Target'].mean().reset_index()  
    avg_target, std_target = target_outlier['Target'].mean(), target_outlier['Target'].std()
  
    target_outlier['Zscore'] = (target_outlier['Target'] - avg_target) / std_target
    df = df.merge(target_outlier[['CONSUMPTION_DATE', 'Zscore']], on='CONSUMPTION_DATE')
    df = df[(df['Zscore'] >= -2) & (df['Zscore'] <= 2)].drop(columns=['Zscore'])

    return df








# FONCTION AJOUT DES DONNEES CALENDAIRES
def add_calendar(df):
    df = pd.merge(df,calendar,how="left",left_on="CONSUMPTION_DATE",right_on="Date")
    df.drop(columns="Date",inplace=True)

    # ajout du mois et jour
    df['month'] = df['CONSUMPTION_DATE'].dt.month
    df['day'] = df['CONSUMPTION_DATE'].dt.weekday

    # colonne conditionnelle pour journées non travaillées pour combiner vacances + jours fériés + weekend
    df['weekend'] = np.where(df['day'].isin([5,6]),1,0)
    df['Holidays'] = np.where(df['Holidays']=="No Holidays",0,1)
    df['Bank_holidays'] = np.where(df['Bank_holidays'] == "No_bank_holidays", 0, 1)
    df['non_working_days'] = np.where(df['Holidays']+df['Bank_holidays']>0,1,0)

    return df




target_scaler_path = "scaler_target.save"

def prepare_features(df, scaler_path="scaler_features.save", mode="train"):

    # colonne conditionnelle pour journées non travaillées pour combiner vacances + jours fériés + weekend
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)

    # choix des colonnes à standardiser 
    continuous_features = ['Sales_cum', 'daily_price']
    if mode == "train":
        scaler = StandardScaler()
        df[continuous_features] = scaler.fit_transform(df[continuous_features])
        joblib.dump(scaler, scaler_path)

        target_scaler = StandardScaler()
        df['Target'] = target_scaler.fit_transform(df[['Target']])
        joblib.dump(target_scaler, target_scaler_path)

    elif mode == "predict":
        scaler = joblib.load(scaler_path)
        df[continuous_features] = scaler.transform(df[continuous_features])
    else:
        raise ValueError("mode doit être 'train' ou 'predict'")

    # Choix des colonnes pour les modèles
    feature_cols = (
        ['CONSUMPTION_DATE','datedif', 'Registrations_cum', 'daily_price', 'non_working_days']
        + ['month_sin', 'month_cos', 'day_sin', 'day_cos']
    )

    X = df[feature_cols]

    if mode == "train":
        y = df['Target']
        return df, X, y
    else:
        return df, X




def pipeline_ML(df):
    df = prep_registrations(df, mode='train')
    df = add_calendar(df)
    df = target(df)
    df, X, y = prepare_features(df, scaler_path="scaler_features.save", mode="train")
    return df, X, y



# PARTIE LSTM
def create_sequences(df, X, y, seq_length="A DETERMINER"):
    sequences, targets, cons_dates = [], [], []
    X_seq = X.copy()
    X_seq["CONSUMPTION_DATE"] = df["CONSUMPTION_DATE"]
    X_seq["datedif"] = df["datedif"]

    # Choix des colonnes pour le modèle de deep learning (en l'occurence enlever datedif qui ne sert plus à rien après la création de la séquence + prix qui est mal interprété par le modèle
    lstm_features = [
        'Registrations_cum','month_sin', 'month_cos', 'day_sin', 'day_cos'
    ]
    X_seq = X_seq[lstm_features + ["CONSUMPTION_DATE", "datedif"]]

    y_seq = y.values if isinstance(y, pd.Series) else y

    for cons_date, group in X_seq.groupby("CONSUMPTION_DATE"):
        group = group.sort_values("datedif", ascending=False)

        # Suppression des colonnes inutiles et conversion en array numpy
        features_only = group.drop(columns=["CONSUMPTION_DATE", "datedif"]).values

        # Récupération des targets correspondantes à chaque séquence
        target_values = y_seq[df["CONSUMPTION_DATE"] == cons_date]


        # taille de la séquence
        for i in range(len(features_only) - seq_length + 1):
            sequences.append(features_only[i:i+seq_length])
            targets.append(target_values[i+seq_length-1])
            cons_dates.append(group.iloc[i+seq_length-1]["CONSUMPTION_DATE"])

    return np.array(sequences), np.array(targets), cons_dates


def train_lstm(df, X, y, seq_length="A DETERMINER", epochs=50, batch_size=32):
    X_seq, y_seq, cons_dates = create_sequences(df, X, y, seq_length=seq_length)

    # Paramètre du modèle LSTM
    model = Sequential([
        LSTM(128, activation="tanh", return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dropout(0.3),
        LSTM(64, activation="tanh", return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(X_seq, y_seq, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])

    y_pred = model.predict(X_seq)
    pred_df = pd.DataFrame({
        "CONSUMPTION_DATE": cons_dates,
        "Prediction": y_pred.flatten()
    }).drop_duplicates(subset=["CONSUMPTION_DATE"])

    return model, pred_df



# ENTRAÎNEMENT
data, X, y = pipeline_ML(registrations)

# Sous-ensemble sans datedif pour LSTM
X_lstm = X[['Registrations_cum','month_sin', 'month_cos', 'day_sin', 'day_cos']]

model_lstm, pred_df = train_lstm(data, X_lstm, y, seq_length="A DETERMINER", epochs=50, batch_size=32)

# Merge des résultats avec la base de données pour XGBoost (pour qu'il prenne le relais et corrige les output de LSTM)
X = pd.merge(X, pred_df, how='left', on='CONSUMPTION_DATE')
X = X.rename(columns={'Prediction': 'LSTM_Prediction'})
X = X.drop(columns=["CONSUMPTION_DATE",'Registrations_cum'])

# Split en train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


#  PARTIE XGBOOST
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

param_grid = {
    "n_estimators": [300, 600],
    'max_depth': [5, 7],
    'min_child_weight': [1, 3],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'learning_rate': [0.05, 0.1]
}

grid = GridSearchCV(xgb_model, param_grid, cv=3, scoring="neg_mean_absolute_percentage_error", n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Meilleurs paramètres :", grid.best_params_)


# Évaluation
target_scaler = joblib.load(target_scaler_path)

preds_train = target_scaler.inverse_transform(best_model.predict(X_train).reshape(-1, 1)).flatten()
preds_test = target_scaler.inverse_transform(best_model.predict(X_test).reshape(-1, 1)).flatten()

y_train_denorm = target_scaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_denorm = target_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

metrics = {
    "Train_RMSE": np.sqrt(mean_squared_error(y_train_denorm, preds_train)),
    "Train_MAPE": mean_absolute_percentage_error(y_train_denorm, preds_train),
    "Train_R2": r2_score(y_train_denorm, preds_train),
    "Test_RMSE": np.sqrt(mean_squared_error(y_test_denorm, preds_test)),
    "Test_MAPE": mean_absolute_percentage_error(y_test_denorm, preds_test),
    "Test_R2": r2_score(y_test_denorm, preds_test),
}

print("\nPerformances du modèle :")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")








# APPLICATION SUR NOUVELLES DONNÉES
projections = # FILE PATH

def pipeline_projections(df):
    df = prep_registrations(df,mode='predict')
    df = add_calendar(df)
    df, X_proj = prepare_features(df, scaler_path="scaler_features.save", mode="predict")
    return df, X_proj

df_proj, X_proj = pipeline_projections(projections)

# Création des séquences pour LSTM
seq_length = 90
X_seq_proj, _, cons_dates_proj = create_sequences(df_proj, X_proj, np.zeros(len(X_proj)), seq_length=seq_length)

# Prédictions LSTM
pred_df_proj = pd.DataFrame({
    "CONSUMPTION_DATE": cons_dates_proj,
    "LSTM_Prediction": model_lstm.predict(X_seq_proj).flatten()
}).drop_duplicates(subset=["CONSUMPTION_DATE"])

# Merge avec X_proj
X_proj = pd.merge(X_proj, pred_df_proj, how='left', on='CONSUMPTION_DATE')

X_proj=X_proj.drop(columns=['Sales_cum'])

# Supprimer colonne date
if "CONSUMPTION_DATE" in X_proj.columns:
    X_proj = X_proj.drop(columns=["CONSUMPTION_DATE"])

# Prédictions XGBoost
df_proj["Pred_Target"] = target_scaler.inverse_transform(best_model.predict(X_proj).reshape(-1, 1))

df_proj = df_proj.set_index(["SALES_DATE", "CONSUMPTION_DATE"]).copy()
df_proj.to_csv(# FILE PATH)
