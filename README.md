# House Prices — Machine Learning Regression

Predicting residential property prices using classical machine learning algorithms.

**Kaggle Competition:** [House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Project Overview

The goal of this project is to predict house sale prices based on 79 features describing residential properties in Ames, Iowa.

The project covers the complete machine learning workflow: exploratory data analysis, preprocessing, model comparison, cross-validation, hyperparameter tuning, error analysis, and submission to Kaggle.

## Results

| Metric | Result |
|---|---|
| Best model | GradientBoostingRegressor |
| Kaggle Public Score (RMSLE) | **0.13223** |
| Kaggle leaderboard position | **2059** at submission time |
| Validation RMSLE | 0.13559 |
| Best cross-validation RMSE (log1p) | 0.12812 |

The best configuration used 200 boosting estimators with a maximum tree depth of 2.

## Dataset

The dataset contains:

- 1,460 labeled training samples
- 1,459 test samples
- 79 predictive features
- Target variable: `SalePrice`

Features include property size, construction year, neighborhood, building quality, garage characteristics, and other attributes.

## Methodology

### Exploratory Data Analysis

Investigated target distribution, missing values, numerical correlations, categorical features, and relationships between property characteristics and sale prices.

Applied a `log1p` transformation to the target variable to reduce skewness.

### Preprocessing

Built a preprocessing pipeline using scikit-learn:

- Numerical features: median imputation and standard scaling
- Nominal features: missing-value imputation and one-hot encoding
- Ordinal features: missing-value imputation and ordinal encoding

### Model Comparison

Compared several regression algorithms using cross-validation.

| Model | Mean 5-Fold CV RMSE (log1p) |
|---|---:|
| Ridge | 0.14245 |
| RandomForestRegressor | 0.14849 |
| GradientBoostingRegressor | **0.13145** |

### Hyperparameter Tuning

Used `GridSearchCV` to tune GradientBoostingRegressor.

Best parameters:

```python
{
    "n_estimators": 200,
    "max_depth": 2,
    "learning_rate": 0.1
}
```

The tuned model achieved a cross-validation error of **0.12812**.

### Error Analysis

Analyzed prediction errors using tables and an Actual vs Predicted scatter plot.

The model showed a tendency to overestimate some low-priced properties and underestimate some high-priced properties.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Kaggle

## How to Run

1. Download the dataset from the Kaggle competition page.
2. Open the notebook in Kaggle Notebooks.
3. Attach the competition dataset.
4. Run the notebook cells in order.
5. Generate `submission.csv` for Kaggle submission.

The notebook uses Kaggle-specific dataset paths.

## Limitations and Future Improvements

- Ordinal categorical features are encoded without explicitly defining their semantic order.
- Missing values are handled using general strategies for each feature group.
- Predictions for properties at the extremes of the price range can have substantial errors.
- Future improvements may include feature engineering, refined preprocessing, and additional hyperparameter tuning.

## Project Status

**Completed — October 2026**

This project was developed as part of my machine learning studies and portfolio.
