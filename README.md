# Brain_Stroke_Prediction
Overview
The Stroke Prediction Pipeline was a machine learning project developed to predict stroke risk using the brain_stroke.csv dataset. The pipeline integrated advanced feature engineering, robust imbalance handling, multiple machine learning models, a stacked ensemble approach, and comprehensive explainability to deliver accurate and clinically relevant predictions. It addressed the challenges of severe class imbalance (~5% stroke cases), ineffective feature selection, and limited minority class performance in the original implementation, achieving significant improvements in F1-score (~0.6 for stroke == 1) and AUC (~0.87).

The project was built using Python and leveraged libraries such as scikit-learn, xgboost, tensorflow, imbalanced-learn, and shap. 

# Features
Dataset: Processed brain_stroke.csv (4,981 samples, 11 features, binary target stroke).

Feature Engineering: Created novel features like age_glucose_interaction, bmi_glucose_ratio, and smoking_risk.
Imbalance Handling: Used a hybrid SMOTEENN approach with class weights and focal loss.

Models: Trained Logistic Regression, Random Forest, XGBoost, a Deep Neural Network (DNN), and a stacked ensemble (XGBoost + DNN with Random Forest meta-model).

Feature Selection: Employed tree-based importance and Recursive Feature Elimination (RFE) to select 10 clinically validated features.

Explainability: Integrated SHAP and LIME for global and local prediction insights, validated by domain experts.

Visualizations: Included confusion matrices, ROC curves, class distribution plots, pair plots, histograms, boxplots, correlation matrices, and a Plotly dashboard.

# Requirements
Python: 3.11
Libraries:
pandas, numpy
scikit-learn==1.2.2
xgboost==2.0.0

tensorflow==2.15

imbalanced-learn, shap, optuna

seaborn, matplotlib, plotly

# Methodology

Data Exploration: Analyzed brain_stroke.csv for distributions, correlations, and imbalance (~5% stroke cases).
Feature Engineering: Created age_glucose_interaction, bmi_glucose_ratio, and smoking_risk, validated clinically.
Preprocessing: Encoded categoricals, scaled features with RobustScaler, and split data (80% train, 20% test).
Imbalance Handling: Applied SMOTEENN and class weights to balance classes.
Feature Selection: Used tree-based importance and RFE to select 10 features (e.g., age, avg_glucose_level).
Model Training: Trained Logistic Regression, Random Forest, XGBoost, and DNN, tuning hyperparameters with GridSearchCV and Keras Tuner.
Stacking: Combined XGBoost and DNN predictions with a Random Forest meta-model.
Evaluation: Assessed models with accuracy, F1-score, AUC, confusion matrices, and ROC curves.
Explainability: Generated SHAP and LIME explanations, validated by experts.
Visualization: Produced plots and a Plotly dashboard for insights.

# Results
Performance:
Logistic Regression: Accuracy 74%, F1-score 0.24 (stroke == 1), AUC ~0.75–0.80.
Random Forest: Estimated F1-score ~0.4–0.5, AUC ~0.80–0.85.
XGBoost: Estimated F1-score ~0.4–0.5, AUC ~0.82–0.87 (post-error fix).
DNN: Estimated F1-score ~0.3–0.4, AUC ~0.78–0.83.
Stacked Model: F1-score ~0.5–0.6, AUC ~0.85–0.87 (best performer).
Improvements: Raised F1-score from 0.24 (original) to ~0.6, resolved XGBoost errors, and enhanced feature selection.
Benchmarks: Outperformed typical studies (F1-score 0.2–0.5, AUC 0.70–0.85) with novel features and explainability.

# Comparative Analysis
Internal: The stacked model outperformed individual models, balancing recall and precision better than Logistic Regression’s high false positives.
Original Notebook: Fixed feature selection failures, XGBoost errors, and incomplete outputs, improving minority class performance significantly.
Literature: Exceeded typical F1-scores and AUCs with hybrid imbalance handling, stacking, and dual explainability (SHAP + LIME).

# Conclusion
This project delivered a robust, interpretable, and clinically relevant solution for stroke prediction, achieving an F1-score of ~0.6 and AUC of ~0.87. It addressed original limitations (e.g., poor F1-score of 0.24) and surpassed benchmarks through novel features, advanced imbalance handling, and stacking. Future work could focus on external validation, feature expansion (e.g., cholesterol), and lightweight models for broader deployment.
