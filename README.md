# 🧠 Prévision des ventes  

Modèle de machine learning pour la prévision quotidienne des ventes.  
Ce projet présente une pipeline complète permettant de prédire les ventes finales à court/moyen terme dans le secteur de l’événementiel (festival, event sportif etc).  

---

## 📈 Présentation  

L’objectif est de prévoir les ventes finales pour une date d’événement. Le modèle utilise les données historiques de ventes et de prix x jour avant l'événement pour apprendre les tendances de la demande et prédire les résultats finaux.  

**Architecture du modèle :**  

- **Moyenne historiques**
  - La première partie du modèle consiste à calculer les moyennes historiques de nombres de ventes et de prix moyens à chaque jour précédent l'évennement, par jours et mois. Ces nouvelles features serviront d'input pour le deuxième niveau du model. 

- **Régression XGBoost**  
  - Modèle de gradient boosting entraîné sur les ventes cumulés, les prix, les variables calendaires (mois, jours, jours férié) et les moyennes obtenues précédemment.  

- **Interprétabilité avec SHAP**  
  - Calcul des valeurs SHAP pour comprendre l’impact de chaque feature sur les prédictions.  
  - Permet d’identifier les facteurs principaux qui influencent les ventes.  

---

## ⚙️ Pipeline  

1. **Préparation des données**  
   - Calcul des cumuls de ventes, prix et features dérivées (variations relatives, métriques cumulées).  
   - Ajout des informations calendrier (mois, jour de la semaine, jours fériés, week-end, jours non travaillés).  
   - Gestion des outliers avec clipping et interpolation pour assurer la cohérence des données.  

2. **Feature engineering**
   - Encodage cyclique pour les mois et les jours de la semaine.  
   - Normalisation des variables continues avec `StandardScaler`.  
   - Ajout de features dérivées comme les variations relatives par rapport aux moyennes historiques.  

4. **Entraînement du modèle**  
   - Entraînement du modèle XGBoost sur les features préparées.  
   - Recherche des meilleurs hyperparamètres via `GridSearchCV`.  
   - Évaluation avec RMSE, MAPE et R² sur un split train/test.  

5. **Prévision**  
   - Application sur de nouvelles données pour prédire les ventes futures.  
   - Fusion des prédictions avec les valeurs SHAP pour interprétation.  

---

## 🔧 Dépendances  

- Python 3.x  
- pandas, numpy  
- xgboost  
- scikit-learn  
- shap  
- joblib  

---

## 🧩 Example Input (simplified)

Exemple Sales.csv
| Transaction_date | Date       | Sales | Revenues |
| ---------------- | ---------- | ----- | -------- |
| 2024-09-01       | 2024-08-01 | 10    | 500      |
| 2024-09-01       | 2024-08-02 | 15    | 750      |
| 2024-09-01       | 2024-08-03 | 20    | 1000     |
| 2024-09-02       | 2024-08-01 | 5     | 250      |
| 2024-09-02       | 2024-08-02 | 8     | 400      |


Exemple Calendar.csv 
| Date       | Holidays    | Bank_holidays    |
| ---------- | ----------- | ---------------- |
| 2024-09-01 | No Holidays | No_bank_holidays |
| 2024-09-02 | Holidays    | No_bank_holidays |
| 2024-09-03 | No Holidays | Bank_holidays    |
| 2024-09-04 | Holidays    | Bank_holidays    |


Exemple Actuals.csv
| Date       | Target |
| ---------- | ------ |
| 2024-09-01 | 50     |
| 2024-09-02 | 40     |
| 2024-09-03 | 60     |
| 2024-09-04 | 30     |

---

## 🔍 Résultats

Obtention d'un MAPE de 2.2% sur le jeu de test. Performance également validée lors de l'application sur de nouvelles données futures.
