## Project overview
This project builds a complete customer churn prediction system for the UCI Online Retail dataset.
Using a combination of feature engineering, RFM modelling, customer clustering, and machine learning, the system predicts whether a customer will churn (stop purchasing) in the next 3 months.

The project also includes a fully interactive Streamlit app that supports:

1. Single customer prediction

2. Batch prediction via CSV upload

3. Interactive customer analytics dashboard

4. Radar charts, probability distributions, cluster insights

5. Downloadable prediction results

This repository is structured for production-quality machine learning, with clear modular code, CRISP-DM documentation, and deployment readiness.

## CRISP-DM Workflow
This project follows the CRISP-DM framework end-to-end
### 1. Business understanding
The objective is to predict customer churn to support:

Targeted marketing

Personalised retention offers

Reduction in lost revenue

Higher customer lifetime value (CLV)

A customer is labelled as churned if they make no purchases during the final 3-month window.
### 2. Data Understanding
------


### 3. Data Preparation

Creating 6 engineered features:
| **Feature** | **Description** |
|-----------|--------------|
| **Recency** | Day since last purchase | 
| **Frequency** | Number of invoices | 
| **Monetary** | Total spend | 
| **CustomerLifetime** | Day from first to last purchase | 
| **AvgBasketSize** | items per description | 
| **AvgPurchaseInterval** | Estimated days between purchases | 
### 4. Modelling
Three models were trained:
1. Logistic Regression

2. Random Forest

3. XGBoost Classifier

Evaluation metrics:
- ROC-AUC

- Precision

- Recall

- F1 Score

- Accuracy

- Confusion Matrix

XGBOOST was selected as the best model based on ROC-AUC

Final artefacts saved:

  best_churn_model_1.pkl
  
  scaler_1.pkl

### 5. Evaluation
Model performance summary:
| **Model** | **ROC-AUC** |**Accuracy**|**Precision**|**Recall**|**F1**|
|-----------|--------------|-----------|--------------|-----------|--------------|
| **Logestic Regression** | 0.746340 |0.680999	|0.680999	|0.779661|0.705882|
| **Random Forest** |0.741329  |0.683773	|0.674033|0.689266|0.681564|
| **XGBoost** | 0.752355 |0.704577|0.684073	|0.740113|0.710991|

### 6.Deployment 
A fully interactive streamlit application was deployed containing:

1.Single Prediction Mode

- Enter customer values manually and get churn prediction + radar chart.

2 Batch Prediction Mode (CSV Upload)

- Upload a CSV containing customer features

- App outputs prediction for each row

- Downloadable results file

3 Customer Dashboard

- Churn probability distribution

- Cluster distribution

- Prediction summary

- Behaviour scatter plots
## Streamlit app review
The app live https://online-retail-customer-churn-prediction-2.onrender.com/

## How to run the project locally
### 1. Clone the repo
git clone https://github.com/Roy16Keane/online-retail-churn.git
cd online-retail-churn/streamlit_app
### 2. Install dependencies
pip install -r requirements.txt
### 3. Run the streamlit app 
streamlit run app.py
## Technologies used

| **Category** | **Tools** |
|-----------|--------------|
| **Data Processing** | Pandas, Numpy| 
| **Machine Learning** | Scikit-Learn,XGBoost | 
| **Clustering** | KMeans, PCA | 
| **Visualization** | Matplotlib, Seaborn | 
| **Deployment** | Render | 
| **Notebook Dev** | Google Colab | 

## Future Improvements
- Build an API endpoint (FastAPI + Docker)

- Create automated CI/CD pipeline

- Integrate interactive feature importance

- Add SHAP-based explainability







