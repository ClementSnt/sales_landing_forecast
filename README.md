# 🧠 Sales Landing Forecast  

Machine learning models for daily **sales landing forecasting**.  
This project demonstrates a complete forecasting pipeline used to predict **short/mid term sales landings** in hospitality and event management contexts.  

---

## 📈 Overview  

The goal is to forecast the final sales landing of an event or stay date based on partial booking data (observations made X days before the consumption date).  
The models use historical booking and pricing data to learn demand patterns and predict final outcomes.

Two model architectures are compared:  

1. **XGBoost Baseline**  
   - Gradient boosting model trained on cumulative bookings, prices, and calendar features.  
   - Strong benchmark for structured tabular data.  

2. **Hybrid LSTM + XGBoost**  
   - LSTM network captures temporal booking patterns across observation dates.  
   - XGBoost model refines the prediction using additional features (price, calendar effects, etc.).  
   - Designed to combine **temporal trend extraction** (via LSTM) and **feature-based correction** (via XGBoost).

---

## ⚙️ Pipeline  

1. **Data preparation**  
   - Generation of observation sequences for each consumption date.  
   - Calculation of cumulative bookings, prices, and derived temporal features.  
   - Integration of calendar features (month, weekday, holidays).  

2. **Model training**  
   - Separate pipelines for baseline and hybrid models.  
   - Hyperparameter tuning via `GridSearchCV`.  
   - Evaluation using RMSE, MAPE, and R² metrics.  

3. **Forecasting**  
   - Application on new unseen data (future observation dates).  
   - LSTM predictions are merged into the XGBoost feature space for hybrid inference.  

---

## 🧩 Example Input (simplified)


---

## 🧠 Technologies  

- **Python 3.10+**  
- **Pandas**, **NumPy**, **Scikit-learn**  
- **TensorFlow / Keras** (for LSTM)  
- **XGBoost**  
- **Joblib**
