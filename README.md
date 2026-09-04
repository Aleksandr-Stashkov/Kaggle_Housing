# Kaggle Housing Prices Competition
## 📊 Project Overview
The goal of this project is to generate the best prediction of the final price of homes (`SalePrice`) based on explanatory variables describing residential properties in Ames, Iowa.

This repository contains an advanced Machine Learning pipeline implemented in Python. The project was developed for the **Housing Prices Competition for Kaggle Learn Users**, focusing on structured feature transformations, custom scikit-learn transformers, target encoding, and ensemble regression. Rather than relying on basic automated preprocessing, this notebook implements a meticulous, data-driven approach leveraging feature hierarchies, multi-stage column transformers, and robust data isolation to eliminate data leakage.

🏆 **Best Leaderboard Score:**  13948.46082

## 🛠️ Tech Stack & Libraries
**Language:** Python

**Exploratory Data Analysis:** `seaborn`, `matplotlib`

**Data Manipulation:** `pandas`, `numpy`

**Advanced Encoding:** `category_encoders` (`MEstimateEncoder`)

**Machine Learning & Pipelines:** `scikit-learn` (`RandomForestRegressor`, `Pipeline`, `ColumnTransformer`)

**Gradient Boosting (Evaluated):** `xgboost`

## 🚀 Pipeline Architecture
### 1. Robust Outlier Mitigation
To shield down-stream models from extreme statistical noise, a multi-conditional thresholding logic was implemented during the data-profiling phase, successfully identifying and dropping structural outliers based on:
* Disproportionate basement areas (`TotalBsmtSF` > 3000 sq ft)
* Sub-market anomalies (`SalePrice` < $50,000)
* High-end luxury deviations (`SalePrice` > $700,000)

### 2. Isolated Target Encoding (M-Estimate)
To capture high-cardinality categorical relationships without exploding feature dimensionality, M-Estimate Target Encoding (`MEstimateEncoder` with $m=2.0$) was deployed on the `Neighborhood` feature.

*Leakage Prevention:* A random 15% sample fraction was strictly isolated to fit the encoder, preventing target leakage into the final validation and testing sets.

### 3. Multi-Stage Feature Engineering & Data Preparation
The data is processed through an advanced, granular `ColumnTransformer` topology:

**Custom Imputation Pipelines:** Created a specialized `BsmtDetailsReplacer` imputer inheriting from scikit-learn's classes. It handles structural conditional dependencies (imputing specific basement sub-features dynamically based on the basement presence).

**Categorical Consolidation:** Handled rare qualitative classes (e.g., pooling rare `RoofMatl`, `Foundation`, or `Exterior1st` configurations into uniform `'Wood'`, `'Other'`, or `'Rare'` identifiers) using custom mapped functions to streamline vector alignment.

**Feature Creation:** Engineered synthetic indicators including binary domain flags (e.g., `HasGarage`, `Has2ndFloor`), proportional indicators, and additive metrics.

**Strict Structural Encoding:** Ordinal categories (like qualities and conditions) were manually mapped via a strict hierarchical `OrdinalEncoder` setup to respect data weightings, while remaining qualitative descriptors were mapped via `OneHotEncoder`.

### 4. Model Optimization & Cross-Validation
Built a final, nested `Pipeline` tracking feature flows into the fine-tuned model. Optimized both **Random Forest** and **XGBoost Regressor** models with the latter showing the best score in the end.

Performance validation was governed via robust cross-validation tracking `neg_mean_absolute_error` to evaluate true out-of-sample scaling.

---

## 🔬 Exploratory & Development Insights
During the analytical evaluation phase, several iterative methodologies were prototyped, tested, and recorded within the codebase for model exploration:


**Hyperparameter Sweeps:** Conducted parametric sweeps over continuous ranges to find optimal variance-bias tradeoffs.

**Feature Normalization:** Analyzed math transformations of numeric features to investigate variance shifts.
