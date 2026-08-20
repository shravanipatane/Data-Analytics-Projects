# Employee Attrition Analysis — R Data Analyst Internship (Week 4 Capstone)

Comprehensive R-based analysis pipeline that cleans a raw HR dataset, explores the
drivers of employee attrition, and builds predictive models to flag at-risk employees.
Completed as the Week 4 final project for the **Virtual R Data Analyst Internship**
(Yuva Intern / NSDC).

## Project Overview

| | |
|---|---|
| **Domain** | Human Resources Analytics |
| **Objective** | Identify key drivers of employee attrition and build a model to predict which employees are at risk of leaving |
| **Dataset** | 1,000 synthetic employee records (19 features) modeled on common HR attrition datasets |
| **Tools** | R, tidyverse, ggplot2, corrplot, caret, randomForest, pROC |
| **Deliverable** | `Week4_Final_Report.docx` — full written report with methodology, results, and recommendations |

## Repository Structure

```
Week-4-Final-Capstone/
│
├── data/
│   ├── employee_attrition_raw.csv
│   └── employee_attrition_cleaned.csv
│
├── scripts/
│   ├── 01_data_cleaning.R
│   ├── 02_eda_visualization.R
│   └── 03_predictive_modeling.R
│
├── outputs/
│   ├── attrition_by_department.png
│   ├── income_by_attrition.png
│   ├── overtime_attrition.png
│   ├── satisfaction_attrition.png
│   ├── correlation_matrix.png
│   ├── tenure_attrition.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── Week4_Final_Report.docx
│
└── README.md
```

## How to Run

```r
install.packages(c("tidyverse", "janitor", "corrplot", "scales",
                    "caret", "randomForest", "pROC"))

source("scripts/01_data_cleaning.R")
source("scripts/02_eda_visualization.R")
source("scripts/03_predictive_modeling.R")
```

## Pipeline Summary

1. **Data Cleaning** — removed 14 duplicate records, standardized inconsistent text
   formatting, corrected invalid `Age` entries, imputed missing values (median/mode),
   and capped outliers in `MonthlyIncome` using the IQR method.
2. **Exploratory Data Analysis** — visualized attrition rate by department, income
   distribution, overtime status, job satisfaction, and feature correlations.
3. **Predictive Modeling** — trained and compared a Logistic Regression baseline
   against a Random Forest classifier (75/25 train-test split).

## Key Results

| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 65.7% | 0.666 |
| **Random Forest** | **67.7%** | **0.701** |

The strongest predictors of attrition were **Monthly Income**, **Age**, **Job
Satisfaction**, **Distance From Home**, and **OverTime status**.

## Author

Shravani Patane — B.E. Artificial Intelligence & Data Science
Virtual R Data Analyst Intern, Yuva Intern (NSDC) | July–August 2026

## License

This project is submitted as academic/internship coursework and is shared for
portfolio purposes.
