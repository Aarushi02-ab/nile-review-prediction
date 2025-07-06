# Nile-review-prediction
Predictive modeling project for Nile, a leading Brazilian eCommerce platform, focused on identifying customers likely to leave positive reviews (4–5 stars). Includes data cleaning, feature engineering, and classification modeling to support targeted marketing and enhance customer engagement.

Nile eCommerce Review Prediction

This repository contains the implementation of a machine learning project developed as part of the Analytics in Practice module at the University of Warwick. The goal was to build a predictive model for Nile, a major Brazilian eCommerce platform, to identify customers most likely to leave positive reviews (4–5 stars). Online reviews are a critical factor influencing buyer trust and sales performance, and this project helps Nile optimize marketing strategies and resource allocation through data-driven targeting.

🎯 Objectives
Predict whether a customer will leave a positive review using historical order and customer data.
Prioritize precision to minimize false positives and reduce wasted outreach to unlikely reviewers.
Assist Nile's marketing team in launching targeted review incentivisation campaigns.

🧠 Features
Binary classification model (Positive = 1 for 4–5 stars, 0 otherwise).
Used Random Forest and Gradient Boosted Decision Trees (GBDT).
Feature engineering based on customer behavior, product details, and delivery performance.
Addressed class imbalance, missing values, outliers, and data leakage.
Focused on repeat customers for better performance generalization.

🧪 Key Results
Precision (Test Set):

Random Forest: 0.888

GBDT: 0.886

Final Model: Optimized Random Forest, chosen for balanced feature importance and higher reliability.
Model not significantly affected by class imbalance; recall for both classes remained high (>0.89).

🔧 Deployment Plan
Model ready for deployment via REST API using Flask or FastAPI.
Accepts customer input and returns prediction in real time.
Designed for integration with automated email systems to boost engagement and sales.


