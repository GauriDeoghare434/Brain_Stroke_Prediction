# Brain_Stroke_Prediction
Overview
The Stroke Prediction Pipeline was a machine learning project developed to predict stroke risk using the brain_stroke.csv dataset. The pipeline integrated advanced feature engineering, robust imbalance handling, multiple machine learning models, a stacked ensemble approach, and comprehensive explainability to deliver accurate and clinically relevant predictions. It addressed the challenges of severe class imbalance (~5% stroke cases), ineffective feature selection, and limited minority class performance in the original implementation, achieving significant improvements in F1-score (~0.6 for stroke == 1) and AUC (~0.87).

The project was built using Python and leveraged libraries such as scikit-learn, xgboost, tensorflow, imbalanced-learn, and shap. The final pipeline was containerized with Docker for scalability and included interactive visualizations via a Plotly dashboard.

Features
Dataset: Processed brain_stroke.csv (4,981 samples, 11 features, binary target stroke).
Feature Engineering: Created novel features like age_glucose_interaction, bmi_glucose_ratio, and smoking_risk.
Imbalance Handling: Used a hybrid SMOTEENN approach with class weights and focal loss.
Models: Trained Logistic Regression, Random Forest, XGBoost, a Deep Neural Network (DNN), and a stacked ensemble (XGBoost + DNN with Random Forest meta-model).
Feature Selection: Employed tree-based importance and Recursive Feature Elimination (RFE) to select 10 clinically validated features.
Explainability: Integrated SHAP and LIME for global and local prediction insights, validated by domain experts.
Visualizations: Included confusion matrices, ROC curves, class distribution plots, pair plots, histograms, boxplots, correlation matrices, and a Plotly dashboard.
Requirements
Python: 3.11
Libraries:
pandas, numpy
scikit-learn==1.2.2
xgboost==2.0.0
tensorflow==2.15
imbalanced-learn, shap, optuna
seaborn, matplotlib, plotly
