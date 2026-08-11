# CardioRiskML: Heart Disease Risk Prediction

Predicts heart disease risk using Logistic Regression and Random Forest, trained on a clinical dataset of 1025 patient records across 13 features.

## Overview
This project explores which clinical factors are most predictive of heart disease and compares two classification models to find the best-performing one.

## Dataset
- **Source:** Heart disease dataset (UCI-style), `heart.csv`
- **Size:** 1025 rows, 13 features + target
- **Features:** age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, rest ECG, max heart rate, exercise-induced angina, oldpeak, slope, vessels colored by flourosopy, thalassemia
- **Target:** presence (1) or absence (0) of heart disease

## Methods
1. Data cleaning — checked for missing values and duplicates
2. Exploratory visualization — disease distribution, age/gender/chest pain breakdowns, correlation heatmap
3. Train/test split — 80/20
4. Models — Logistic Regression and Random Forest (100 trees)
5. Evaluation — confusion matrix, accuracy, ROC/AUC

## Results
| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression | [X]% | [X] |
| Random Forest | [X]% | [X] |

**Best model:** [fill in based on your actual output]

## Key Findings
- Top predictive features (from Random Forest importance): [list top 3-4, e.g. chest pain type, max heart rate, vessels colored by flourosopy]
- [Any other notable pattern you observed, e.g. age/cholesterol trend]

## Visuals
![Correlation Heatmap](plots/correlation_heatmap.png)
![ROC Curves](plots/roc_curves.png)
![Feature Importance](plots/feature_importance.png)

## How to Run
1. Clone this repo
2. Open `heart_disease_analysis.R` in RStudio
3. Install required packages: `tidyverse`, `caret`, `randomForest`, `corrplot`, `pROC`
4. Run the script — `heart.csv` must be in the same directory

## Tech Stack
R, tidyverse, caret, randomForest, corrplot, pROC
