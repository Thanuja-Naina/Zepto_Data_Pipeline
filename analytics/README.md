# Titanic Analytics & Predictive Modeling Pipeline

## Project Overview

This project implements a complete end-to-end Data Science workflow using the Titanic dataset. The workflow begins with loading and profiling the dataset, followed by data cleaning, exploratory data analysis (EDA), feature preprocessing, predictive modeling, model evaluation, hyperparameter tuning, and finally saving the best-performing machine learning pipeline for future predictions.

The project follows the assignment requirement of loading the dataset only once using Seaborn and then saving it as an offline CSV (`titanic.csv`) for subsequent analysis.

---

# Project Structure

```
analytics/
│
├── 01_eda.ipynb
├── 02_modeling.ipynb
├── titanic.csv
├── best_pipeline.pkl
├── README.md
├── requirements.txt
│
└── images/
    ├── age_histogram.png
    ├── age_boxplot.png
    ├── fare_histogram.png
    ├── fare_boxplot.png
    ├── survival_by_sex.png
    ├── survival_by_class.png
    ├── correlation_heatmap.png
    ├── decision_tree.png
    ├── roc_curve.png
    └── residual_plot.png
```

---


# Dataset

**Dataset:** Titanic Dataset

Loaded using

```python
sns.load_dataset("titanic")
```

Immediately after loading, the dataset was saved as

```python
titanic.csv
```

This ensures the project can be executed offline without requiring internet access.

---

# Technologies Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn (SMOTE)
- Joblib

---

# Part A – Exploratory Data Analysis

## Dataset Profiling

The following dataset profiling operations were performed:

- Dataset shape
- Dataset information
- Summary statistics
- Missing value analysis

---

## Missing Value Handling

The following threshold rule was used.

### Missing values below 5%

Rows containing missing values were removed.

Columns:

- embarked
- embark_town

Reason:
The missing percentage was less than 5%, so removing the affected rows had minimal impact on the dataset.

---

### Missing values between 5% and 30%

Missing values were imputed.

Column:

- age

Strategy:

Median Imputation

Reason:

Age contained approximately 20% missing values. Median imputation was selected because age is a numerical feature and the distribution contains outliers.

---

### Missing values above 30%

Column removed:

- deck

Reason:

The deck column contained approximately 77% missing values. Since such a high proportion of missing values makes reliable imputation difficult and likely to introduce bias, the column was dropped from further analysis.

---

# Univariate Analysis

The following visualizations were created.

- Age Histogram
- Age Boxplot
- Fare Histogram
- Fare Boxplot

---

## Outlier Detection

Outliers were identified using the IQR method.

```
Lower Bound = Q1 − 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

The number of outliers for both Age and Fare were reported.

---

## Fare Distribution

The following statistics were calculated.

- Mean
- Median
- Mode

### Interpretation

If

```
Mean > Median > Mode
```

then Fare follows a **right-skewed (positively skewed)** distribution.

---

# Bivariate Analysis

Survival rates were computed for

- Survival by Gender
- Survival by Passenger Class
- Survival by Gender and Passenger Class

---

# Correlation Analysis

A correlation matrix was created using the following six columns.

- survived
- pclass
- age
- sibsp
- parch
- fare

A heatmap was generated using Seaborn.

## Strongest Correlations

The two strongest absolute correlations observed in the dataset were identified and interpreted.

Example:

- Passenger Class and Fare show a strong negative correlation because higher-class passengers generally paid higher fares.
- SibSp and Parch exhibit a positive relationship, indicating that passengers travelling with spouses often also travelled with parents or children.

---

# Multivariate Analysis

The following visualizations were created.

1. Survival vs Gender
2. Survival vs Passenger Class
3. Age vs Survival
4. Fare vs Survival

Each visualization includes a written interpretation explaining its significance.

---

## Interpretation 1 – Survival by Gender

Female passengers exhibited a substantially higher survival rate than male passengers. This reflects the "women and children first" evacuation policy followed during the Titanic disaster.

---

## Interpretation 2 – Survival by Passenger Class

Passengers travelling in First Class experienced the highest survival rate, whereas Third Class passengers experienced the lowest. This suggests socioeconomic status influenced access to lifeboats.

---

## Interpretation 3 – Age vs Survival

Children generally showed higher survival rates than adults, although survival varied considerably across age groups.

---

## Interpretation 4 – Fare vs Survival

Passengers who paid higher fares tended to survive more often, largely because higher fares corresponded to higher passenger classes.

---

# Standardization Check

Age and Fare were standardized using StandardScaler.

The transformed columns showed

- Mean ≈ 0
- Standard Deviation ≈ 1

This confirms successful standardization.

---

# Part B – Machine Learning

## Train-Test Split

The dataset was divided into training and testing sets using an 80–20 split.

A stratified split was used to preserve the class distribution of the target variable (survived) across both training and testing datasets.

---

# Feature Engineering

Categorical Features

- sex
- embarked

Numerical Features

- age
- fare
- sibsp
- parch
- pclass

---

# Preprocessing Pipeline

The preprocessing pipeline included

Numerical Features

- Median Imputation
- Standard Scaling

Categorical Features

- Most Frequent Imputation
- One-Hot Encoding

The preprocessing steps were implemented using

- Pipeline
- ColumnTransformer

This ensured that preprocessing was fitted only on the training data, preventing data leakage.

---

# Classification Models

The following models were trained.

1. Logistic Regression
2. Decision Tree
3. Random Forest

---

# Model Evaluation

Each classifier was evaluated using

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- ROC AUC Score

---

# Decision Tree Visualization

The trained Decision Tree was visualized using

```python
plot_tree()
```

Feature names and class labels were included.

---

# Imbalance Handling

Three approaches were compared.

1. Baseline Random Forest
2. Random Forest with

```
class_weight="balanced"
```

3. SMOTE Oversampling

The following metrics were compared.

- Precision
- Recall
- F1 Score

### Conclusion

The SMOTE model achieved the best balance between precision and recall, resulting in the highest F1 score. It provided improved detection of the minority class while maintaining competitive precision.

---

# Hyperparameter Tuning

GridSearchCV was used for Random Forest.

Parameters tuned

- n_estimators
- max_depth
- max_features

The model was created with

```python
RandomForestClassifier(
    oob_score=True
)
```

The following were reported.

- Best Parameters
- Out-of-Bag (OOB) Score

---

# Regression Task

A multivariate Linear Regression model was built to predict Fare.

Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Adjusted R² Score

A residual plot was generated.

---

# Residual Analysis

The residual plot was examined to determine whether heteroscedasticity was present.

Interpretation:

If residuals are randomly scattered around zero with no visible pattern, the model satisfies the constant variance assumption.

If residual spread increases or decreases systematically, heteroscedasticity is present.

---

# Final Model Comparison

Classification Models

- Logistic Regression
- Decision Tree
- Random Forest

Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

Regression Model

Metrics

- MAE
- RMSE
- R²
- Adjusted R²

Classification and regression metrics were reported separately because they measure different prediction tasks and are not directly comparable.

---

# Best Model

Based on Accuracy, Precision, Recall, F1 Score and ROC-AUC, the Random Forest classifier achieved the strongest overall performance.

The complete preprocessing pipeline together with the trained Random Forest model was therefore selected as the final deployed model.

---

# Model Persistence

The complete preprocessing pipeline and classifier were saved using Joblib.

```python
joblib.dump(best_pipeline, "best_pipeline.pkl")
```

The saved model was reloaded using

```python
joblib.load("best_pipeline.pkl")
```

Predictions were successfully generated on raw input data, confirming that the entire preprocessing pipeline was stored together with the trained model.

---

# How to Run

Install dependencies

```bash
pip install -r requirements.txt
```

Run EDA

```bash
jupyter notebook 01_eda.ipynb
```

Run Modeling

```bash
jupyter notebook 02_modeling.ipynb
```

---

# Conclusion

This project demonstrates a complete end-to-end Data Science workflow, beginning with data profiling and cleaning, progressing through exploratory data analysis and predictive modeling, and concluding with model deployment. The workflow follows best practices by preventing data leakage through train-only preprocessing, evaluating multiple classification models, addressing class imbalance, performing hyperparameter tuning, and persisting the best-performing pipeline for future use. Based on the evaluation metrics, the Random Forest classifier was selected as the final model due to its superior predictive performance and balanced classification results.
