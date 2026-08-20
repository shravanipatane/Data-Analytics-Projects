# Week 3 — Statistical Analysis & Predictive Modeling in R 📊

## 📌 Project Overview

This project is part of my **R Data Analytics journey** and focuses on applying statistical analysis and predictive modeling techniques to a real-world dataset.

The objective of this task was to perform an end-to-end statistical analysis workflow — from dataset exploration and hypothesis testing to regression modeling, cross-validation, model diagnostics, and performance evaluation.

The analysis uses the **Restaurant Tipping (`tips`) dataset**, containing information about restaurant bills, tips, party size, day, time, smoker status, and other customer-related variables.

---

## 🎯 Objectives

* Explore and understand the dataset
* Check data structure, summary statistics, and missing values
* Perform distributional and normality testing
* Analyze relationships between numerical variables
* Conduct hypothesis tests
* Build a multiple linear regression model
* Perform stepwise feature selection
* Validate the model using 5-fold cross-validation
* Evaluate performance on a hold-out test set
* Perform regression diagnostic analysis
* Check multicollinearity using VIF
* Interpret model results and identify possible improvements

---

## 📂 Dataset

**Dataset:** Restaurant Tipping Dataset (`tips.csv`)

The dataset contains **244 restaurant transactions** with the following variables:

| Variable     | Description                         |
| ------------ | ----------------------------------- |
| `total_bill` | Total bill amount in USD            |
| `tip`        | Tip amount in USD — target variable |
| `sex`        | Gender of the bill payer            |
| `smoker`     | Whether the party included a smoker |
| `day`        | Day of the week                     |
| `time`       | Lunch or Dinner                     |
| `size`       | Number of people in the party       |

## A derived variable, `tip_pct`, was also created to represent the tip as a percentage of the total bill.

## 🔬 Statistical Analysis

The following statistical techniques were applied:

### 1. Shapiro-Wilk Normality Test

Used to test whether `tip` and `total_bill` follow a normal distribution.

**Result:** Both variables showed strong evidence against normality and were right-skewed.

### 2. Pearson Correlation

Used to examine the relationship between `total_bill` and `tip`.

**Result:**

* Correlation: **r = 0.676**
* **p < 2.2e-16**

This indicates a strong and statistically significant positive relationship between bill amount and tip.

### 3. Welch Two-Sample t-Test

Two t-tests were conducted:

* Tip amount: **Lunch vs Dinner**
* Tip percentage: **Smoker vs Non-smoker**

The Lunch vs Dinner comparison was borderline (**p = 0.0586**), while smoker status did not significantly affect tip percentage (**p = 0.321**).

### 4. Chi-Square Test

Tested the relationship between smoker status and day of the week.

**Result:**

* χ² = **25.79**
* df = **3**
* **p < 0.001**

The analysis found a statistically significant association between smoker status and day of the week.

### 5. One-Way ANOVA

Tested whether average tip amounts differed across days of the week.

**Result:**

* F(3, 240) = **1.15**
* **p = 0.331**

There was no statistically significant difference in average tip amount across the four days.

---

## 🤖 Predictive Modeling

### Multiple Linear Regression

The target variable was:

```text
tip
```

The initial model used:

```text
total_bill + size + sex + smoker + day + time
```

An **80/20 train-test split** was used:

* Training set: **195 observations**
* Test set: **49 observations**

AIC-based stepwise feature selection was then applied to simplify the model.

### Final Model

The stepwise procedure selected:

```text
tip ~ total_bill + size + smoker
```

`total_bill` and `size` were statistically significant positive predictors, while `smoker` was retained by the AIC-based selection procedure but was not statistically significant at the 0.05 level.

---

## 🔄 Model Validation

### 5-Fold Cross-Validation

Manual **5-fold cross-validation** was performed using base R.

The mean cross-validated R² was:

```text
0.419
```

This was close to both the training R² and hold-out test R², suggesting reasonably consistent generalization.

---

## 📈 Model Performance

| Metric            | Result |
| ----------------- | -----: |
| Training R²       |  0.472 |
| Mean 5-Fold CV R² |  0.419 |
| Test R²           |  0.417 |
| Test RMSE         |  1.168 |
| Test MAE          |  0.777 |

## The model explains approximately **42–47% of the variation in tip amount**, with similar cross-validation and test-set performance.

## 🔎 Diagnostic Analysis

The regression model was evaluated using:

* Residuals vs Fitted
* Normal Q-Q
* Scale-Location
* Residuals vs Leverage
* Predicted vs Actual plot
* Manual VIF analysis

The diagnostic analysis indicated:

* Mild non-linearity
* Right-skewed residual behavior
* Mild heteroscedasticity
* No observation with excessive Cook's distance
* No major multicollinearity issue

The calculated VIF values for the numeric predictors were approximately **1.57**, well below the common threshold of 5.

---

## 💡 Key Findings

### 1. Total Bill is the Strongest Predictor

The total bill amount was the most important predictor of tip amount. The analysis found that each additional $1 in the bill was associated with roughly **$0.087 more in tip**, holding other variables constant.

### 2. Party Size Matters

Larger groups were associated with higher absolute tip amounts even after accounting for bill size.

### 3. Other Variables Have Limited Independent Effects

After accounting for bill size, variables such as:

* smoker status
* day
* sex
* time

did not show strong independent effects on tip amount in the final model.

### 4. Model Generalizes Reasonably Well

The similarity between training, cross-validation, and test-set R² suggests that the model is not strongly overfitting the dataset.

---

## 📁 Project Structure

```text
Week-3-Statistical-Analysis/
│
├── analysis.R
│
├── Report_Statistical_Analysis_Predictive_Modeling.docx
│
├── data/
│   └── tips.csv
│
└── outputs/
    ├── 01_histograms.png
    ├── 02_boxplots.png
    ├── 03_scatter_correlation.png
    ├── 04_diagnostic_plots.png
    ├── 05_predicted_vs_actual.png
    └── model_formula.txt
```

---

## 🛠️ Tools & Technologies

* **R**
* Base R statistical functions
* Base R graphics
* Linear Regression
* Hypothesis Testing
* Cross-Validation
* Model Diagnostics
* GitHub

The complete analysis was implemented using **base R functionality without requiring external packages**.

---

## 🚀 Potential Improvements

Future improvements could include:

* Applying log transformations to address right-skew
* Using robust standard errors or weighted least squares for heteroscedasticity
* Testing interaction effects such as `total_bill × size`
* Exploring polynomial relationships
* Comparing the linear model with tree-based models
* Testing regularized regression such as Ridge or Lasso
* Using a larger and more diverse dataset

These improvements could potentially capture non-linear relationships and improve predictive performance.

---

## 📌 Conclusion

This project demonstrates a complete statistical modeling workflow in R, starting with exploratory analysis and hypothesis testing and progressing to regression modeling, cross-validation, diagnostic analysis, and evaluation.

The results show that **bill amount is the dominant predictor of tipping behavior**, with party size providing additional explanatory power. The final model provides a transparent and interpretable baseline while also identifying clear opportunities for future improvement.

---

### 👩‍💻 Author

**Shravani Patane**

**Week 3 — R Data Analytics Journey**

August 2026

