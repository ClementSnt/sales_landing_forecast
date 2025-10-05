import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt

# IMPORT DES DONNEES D APPRENTISSAGE
sales = # FILE PATH
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
    df_full['datedif'] = (df_full['CONSUMPTION_DATE'] - df_full['OBSERVATION_DATE']).dt.days
    df_full = df_full[(df_full["datedif"]>=0)&(df_full["datedif"]<=90)]


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


def add_lag_features(df):
    lags = # A DETERMINER EN FONCTION DES DONNES#

    df = df.sort_values(['CONSUMPTION_DATE', 'datedif'], ascending=[True, False]).reset_index(drop=True)

    for lag in lags:
        df[f"ATTENDANCE_lag_{lag}"] = df.groupby("CONSUMPTION_DATE")["Sales_cum"].shift(lag)
        df[f"att_var%_{lag}"] = (df["Sales_cum"] - df[f"ATTENDANCE_lag_{lag}"]) / df[f"ATTENDANCE_lag_{lag}"]

    datedif_range = # SELECTION DE LA RANGE PERTINENTE (0,90)
    df = df[(df['datedif'] >= datedif_range[0]) & (df['datedif'] <= datedif_range[1])]
    df = df.dropna().reset_index(drop=True)

    return df


def prepare_features(df, scaler_path="scaler_features.save", mode="train"):

    lags = # A DETERMINER EN FONCTION DES DONNES#

    # Encodage cyclique pour les mois et jours de la semaine 
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)

    # choix des colonnes à standardiser 
    continuous_features = ['Sales_cum', 'daily_price'] + [f'ATTENDANCE_lag_{lag}' for lag in lags]

    if mode == "train":
        scaler = StandardScaler()
        df[continuous_features] = scaler.fit_transform(df[continuous_features])
        joblib.dump(scaler, scaler_path)
    elif mode in ["predict"]:
        scaler = joblib.load(scaler_path)
        df[continuous_features] = scaler.transform(df[continuous_features])
    else:
        raise ValueError("mode doit être 'train' ou 'predict'")

    # Choix des colonnes pour le modèle
    feature_cols = (
        ['datedif', 'daily_price', 'Sales_cum','non_working_days']
        + [f'att_var%_{lag}' for lag in lags]
        + ['month_sin', 'month_cos', 'day_sin', 'day_cos']
    )

    X = df[feature_cols]

    if mode == "train":
        y = df['Target']
        return df, X, y
    else:
        return df, X

# Créaltion du pipeline
def pipeline_ML(df):
    df= prep_sales(df,mode='train')
    df = target(df)
    df = add_calendar(df)
    df = add_lag_features(df)
    df,X,y = prepare_features(df, scaler_path="scaler_features.save", mode="train")

    return df, X,y


data,X,y =pipeline_ML(sales)

# Fonction d'évaluation du modèle
def evaluate_model(model, X_train, y_train, X_test, y_test):
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    metrics = {
        "Train_RMSE": np.sqrt(mean_squared_error(y_train, preds_train)),
        "Train_MAPE": mean_absolute_percentage_error(y_train, preds_train),
        "Train_R2": r2_score(y_train, preds_train),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, preds_test)),
        "Test_MAPE": mean_absolute_percentage_error(y_test, preds_test),
        "Test_R2": r2_score(y_test, preds_test),
    }
    return metrics



# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)



# GridSearch
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



# Affichage des meilleurs paramètres
print("Meilleurs paramètres:")
print(grid.best_params_)

# Évaluation
metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test)
print("Performances du modèle :")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Ré-entraînement final sur tout l’historique
final_model = xgb.XGBRegressor(**grid.best_params_, random_state=42, n_jobs=-1)
final_model.fit(X, y)
joblib.dump(final_model, "xgb_final_model.save")







# Analyse des performances par mois avec biais
# Prédictions sur le jeu de test uniquement
df_eval = X_test.copy()
df_eval["Target"] = y_test
df_eval["Pred_Target"] = best_model.predict(X_test)

# Ajout du mois depuis l’index original
df_eval["CONSUMPTION_DATE"] = data.loc[X_test.index, "CONSUMPTION_DATE"]
df_eval["month"] = df_eval["CONSUMPTION_DATE"].dt.month

# MAPE + biais par mois
perf_by_month = df_eval.groupby("month").apply(
    lambda x: pd.Series({
        "MAPE": mean_absolute_percentage_error(x["Target"], x["Pred_Target"]),
        "Bias": (x["Pred_Target"].mean() - x["Target"].mean()) / x["Target"].mean()
    })
).reset_index()

print("\nPerformances par mois:")
print(perf_by_month)



# 📊 Visualisation

fig, ax1 = plt.subplots(figsize=(9,5))

# Courbe MAPE
ax1.plot(perf_by_month["month"], perf_by_month["MAPE"]*100, marker="o", color="tab:blue", label="MAPE (%)")
ax1.set_xlabel("Mois")
ax1.set_ylabel("MAPE (%)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_xticks(range(1,13))

# Deuxième axe pour le biais
ax2 = ax1.twinx()
ax2.bar(perf_by_month["month"], perf_by_month["Bias"]*100, alpha=0.3, color="tab:red", label="Biais (%)")
ax2.set_ylabel("Biais (%) (positif = surestimation, négatif = sous-estimation)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

plt.title("MAPE et biais moyen par mois")
fig.tight_layout()
plt.show()



# APPLICATION SUR NOUVELLES DONNEES
projections = # FILE PATH


def pipeline_projections(df):
    df = prep_sales(df,mode='predict')
    df = add_calendar(df)
    df = add_lag_features(df)
    df, X = prepare_features(df, scaler_path="scaler_features.save", mode="predict")
    return df, X


df_proj, X_proj = pipeline_projections(projections)


df_proj["Pred_Target"] = final_model.predict(X_proj)


# Index avec dates
df_proj = df_proj.set_index(["SALES_DATE", "CONSUMPTION_DATE"]).copy()

df_proj.to_csv(# FILE PATH)
