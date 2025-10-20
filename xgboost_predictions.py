import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import shap

##################################################################################
# IMPORT DES DONNEES
##################################################################################
sales_data = pd.read_csv("sales_data.csv")
calendar_data = pd.read_excel("calendar_data.xlsx")
target_data = pd.read_excel("target_data.xlsx")
price_data = pd.read_csv("price_data.csv")
future_data = pd.read_csv("future_data.csv")

##################################################################################
# FONCTIONS DE PREPARATION DES DONNEES
##################################################################################

# Conversion des colonnes de dates
def format_dates(df):
    for col in ["Date", "Transaction_date"]:
        if not pd.api.types.is_datetime64_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

sales_data = format_dates(sales_data)
price_data = format_dates(price_data)
future_data = format_dates(future_data)

# Calcul du prix journalier moyen et nettoyage initial
price_data = price_data[price_data['Quantity'] > 0]
price_data['daily_price'] = price_data['Revenue'] / price_data['Quantity']
price_data.drop(columns=['Quantity','Revenue'], inplace=True)

# Reconstitution de la série complète par date de transaction
def prep_sales(df):
    all_dates = (
        df.groupby("Date")["Transaction_date"]
          .agg(["min", "max"])
          .apply(lambda x: pd.date_range(x["min"], x["max"]), axis=1)
    )
    full_index = all_dates.explode().reset_index().rename(columns={0: "Transaction_date"})
    df_full = full_index.merge(df, on=["Date", "Transaction_date"], how="left")

    # Remplissage des valeurs cumulées
    for col in ["Quantity", "Revenue"]:
        df_full[col] = df_full[col].fillna(0)
        df_full[f"{col}_cum"] = (
            df_full.groupby("Date")[col]
                   .cumsum()
                   .groupby(df_full["Date"])
                   .ffill().bfill()
                   .fillna(0)
        )

    # Calcul du délai entre vente et transaction
    df_full['day_diff'] = (df_full['Date'] - df_full['Transaction_date']).dt.days

    # Prix cumulé avec interpolation
    df_full["price_cum"] = df_full["Revenue_cum"] / df_full["Quantity_cum"]
    df_full['price_cum'] = df_full['price_cum'].replace([0, np.inf, -np.inf], pd.NA)
    df_full['price_cum'] = df_full.groupby('Date')['price_cum']\
                                  .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))\
                                  .bfill().ffill()
    
    # Merge avec prix journaliers et interpolation
    df_full = pd.merge(df_full, price_data, how='left', on=['Transaction_date','Date'])
    df_full['daily_price'] = df_full.groupby('Date')['daily_price']\
                                    .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))\
                                    .bfill().ffill()
    
    # Variation relative du prix journalier par rapport au prix cumulé
    df_full['var_price'] = (df_full['daily_price'] - df_full['price_cum']) / (df_full['price_cum'] + 1e-8)
    return df_full

# Ajout de la Target et filtrage des outliers
def add_target(df):
    df = pd.merge(df, target_data, how="left", left_on="Date", right_on="Date")
    df = df.rename(columns={'Actual': 'Target'})
    return df

# Ajout des informations calendrier et flags
def add_calendar(df):
    df = pd.merge(df, calendar_data, how="left", left_on="Date", right_on="Date")
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.weekday
  
    # Flags
    df['Weekend_flag'] = np.where(df['day'].isin([5,6]), 1, 0)
    df['Holiday_flag'] = np.where(df['Holiday']=="No", 0, 1)
    df['Bank_holiday_flag'] = np.where(df['Bank_holiday']=="No", 0, 1)
    df['Non_working_day_flag'] = np.where(df['Holiday_flag'] + df['Bank_holiday_flag'] + df['Weekend_flag'] > 0, 1, 0)
    
    return df

# Préparation des features pour XGBoost
def prepare_features(df, scaler_path="scaler_features.save", mode="train"):
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)
    
    continuous_features = ['Quantity_cum', 'price_cum']
    
    if mode == "train":
        scaler = StandardScaler()
        df[continuous_features] = scaler.fit_transform(df[continuous_features])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        df[continuous_features] = scaler.transform(df[continuous_features])
    
    return df

##################################################################################
# PIPELINE PRINCIPALE
##################################################################################
def pipeline_main(df):
    df = prep_sales(df)
    df = add_target(df)
    df = add_calendar(df)
    return df

data = pipeline_main(sales_data)

##################################################################################
# AJOUT DES MOYENNES HISTORIQUES + CLIPPING
##################################################################################
clip_val = 
theorical_avg = data.groupby(['day_diff','month','Weekend_flag','Non_working_day_flag'])[['Quantity_cum','price_cum']]\
                    .mean().reset_index().round(0).rename(columns={'Quantity_cum':'avg_quantity','price_cum':'avg_price'})

data = pd.merge(data, theorical_avg, how='left',
                on=['day_diff','month','Weekend_flag','Non_working_day_flag'])

data['var_quantity'] = np.clip(data['Quantity_cum']/data['avg_quantity']-1, -clip_val, clip_val)
data['var_price_cum'] = np.clip(data['price_cum']/data['avg_price']-1, -clip_val, clip_val)
data['var_daily_price'] = np.clip(data['daily_price']/data['price_cum']-1, -clip_val, clip_val)

data_boost = prepare_features(data[['day_diff','month','day','Non_working_day_flag','Quantity_cum','price_cum',
                                    'var_quantity','var_price_cum','var_daily_price','Target']], mode="train")

X = data_boost.drop(columns=['month','day','Target'])
y = data_boost['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=, random_state=42, shuffle=True)

##################################################################################
# MODELE XGBOOST
##################################################################################
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
param_grid = {
    'n_estimators': [],
    'learning_rate': [],
    'max_depth': [],
    'min_child_weight': [],
    'subsample': [],
    'colsample_bytree': []
}
grid = GridSearchCV(xgb_model, param_grid, cv=,
                    scoring="neg_mean_absolute_percentage_error", n_jobs=-1)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Meilleurs paramètres:", grid.best_params_)

metrics = {
    "Train_RMSE": np.sqrt(mean_squared_error(y_train, best_model.predict(X_train))),
    "Train_MAPE": mean_absolute_percentage_error(y_train, best_model.predict(X_train)),
    "Train_R2": r2_score(y_train, best_model.predict(X_train)),
    "Test_RMSE": np.sqrt(mean_squared_error(y_test, best_model.predict(X_test))),
    "Test_MAPE": mean_absolute_percentage_error(y_test, best_model.predict(X_test)),
    "Test_R2": r2_score(y_test, best_model.predict(X_test)),
}
for k,v in metrics.items():
    print(f"{k}: {v:.4f}")

# Entrainement final
final_model = xgb.XGBRegressor(**grid.best_params_, random_state=42, n_jobs=-1)
final_model.fit(X, y)
joblib.dump(final_model, "xgboost_final_model.save")

##################################################################################
# PREDICTIONS SUR NOUVELLES DONNEES
##################################################################################
def pipeline_future(df):
    df = prep_sales(df)
    df = add_calendar(df)
    df = pd.merge(df, theorical_avg, how='left',
                  on=['day_diff','month','Weekend_flag','Non_working_day_flag'])
    df['var_quantity'] = np.clip(df['Quantity_cum']/df['avg_quantity']-1, -clip_val, clip_val)
    df['var_price_cum'] = np.clip(df['price_cum']/df['avg_price']-1, -clip_val, clip_val)
    df['var_daily_price'] = np.clip(df['daily_price']/df['price_cum']-1, -clip_val, clip_val)
    df = prepare_features(df, mode="predict")
    return df

future_prepared = pipeline_future(future_data)
X_future = future_prepared[X.columns]
future_prepared["Pred_Target"] = final_model.predict(X_future)

##################################################################################
# INTERPRETABILITE AVEC SHAP
##################################################################################
explainer = shap.Explainer(final_model, X_future, algorithm="tree")
shap_values = explainer(X_future)
shap_df = pd.DataFrame(shap_values.values, columns=X_future.columns)
shap_df["SHAP_sum"] = shap_df.sum(axis=1)

df_output = pd.concat([future_prepared.reset_index(drop=True), shap_df], axis=1)
df_output.to_excel("Future_Projections_with_SHAP.xlsx", index=False)
print("✅ Prédiction terminée et fichier sauvegardé : Future_Projections_with_SHAP.xlsx")

