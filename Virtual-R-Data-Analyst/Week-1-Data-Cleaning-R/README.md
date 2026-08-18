# 📊 Week 1 — Data Cleaning and Preliminary Analysis with R

## 📌 Overview

This project was completed as part of my **Virtual R Data Analyst Internship**.

The objective of Week 1 was to perform **data cleaning, preprocessing, and preliminary exploratory analysis using R**. The analysis uses the publicly available **Titanic dataset**, which contains a combination of numerical and categorical variables along with missing values, making it suitable for practicing real-world data preparation techniques.

---

## 🎯 Objectives

The main objectives of this task were to:

* Understand the structure and characteristics of the dataset
* Identify and handle missing values
* Detect and analyze potential outliers
* Perform data transformation and preprocessing
* Apply appropriate normalization techniques
* Encode categorical variables
* Generate descriptive statistics
* Analyze relationships and correlations between variables
* Create initial data visualizations
* Extract preliminary insights from the cleaned dataset

---

## 📂 Dataset

### Titanic Dataset

The Titanic dataset contains passenger-level information from the RMS Titanic, including demographic, travel, and survival-related variables.

### Key Variables

| Variable      | Description                       | Type                |
| ------------- | --------------------------------- | ------------------- |
| `PassengerId` | Unique passenger identifier       | Numerical           |
| `Survived`    | Survival status                   | Categorical/Binary  |
| `Pclass`      | Passenger class                   | Categorical/Ordinal |
| `Name`        | Passenger name                    | Categorical         |
| `Sex`         | Passenger gender                  | Categorical         |
| `Age`         | Passenger age                     | Numerical           |
| `SibSp`       | Number of siblings/spouses aboard | Numerical           |
| `Parch`       | Number of parents/children aboard | Numerical           |
| `Fare`        | Passenger fare                    | Numerical           |
| `Embarked`    | Port of embarkation               | Categorical         |

---

## 🧹 Data Cleaning & Preprocessing

The dataset was examined and prepared for analysis through several steps.

### 1. Data Inspection

The dataset structure and summary statistics were examined using R functions such as:

```r
str(data)
summary(data)
head(data)
```

This helped understand the variables, data types, distributions, and potential data-quality issues.

### 2. Missing Value Analysis

Missing values were identified and analyzed before applying appropriate treatment methods.

The analysis focused particularly on variables containing missing observations, such as:

* `Age`
* `Embarked`
* Other incomplete fields where applicable

### 3. Missing Value Handling

Missing observations were handled using suitable preprocessing techniques while considering the characteristics of each variable.

### 4. Outlier Detection

Numerical variables such as `Age` and `Fare` were examined for potential outliers using statistical methods and visualizations such as boxplots.

### 5. Data Transformation

Variables were transformed where necessary to make them suitable for analysis and visualization.

### 6. Categorical Variable Encoding

Categorical variables were converted into appropriate formats for analysis and potential modeling.

### 7. Normalization

Normalization/scaling techniques were considered for numerical variables where required by the analytical approach.

---

## 📊 Exploratory Data Analysis

After cleaning and preprocessing, preliminary exploratory analysis was performed to understand the dataset and identify meaningful patterns.

The analysis included:

* Descriptive statistics
* Distribution analysis
* Survival analysis
* Group comparisons
* Correlation analysis
* Missing-value comparison
* Data visualization

---

## 📈 Visualizations

The project includes the following visualizations:

### 1. Age Distribution

Shows the distribution of passenger ages and helps identify the central tendency and spread of the age variable.

### 2. Fare Distribution

A boxplot was used to examine the distribution of passenger fares and identify potential outliers.

### 3. Survival by Sex

Compares survival outcomes between male and female passengers.

### 4. Survival by Passenger Class

Examines differences in survival across passenger classes.

### 5. Correlation Heatmap

Displays relationships between selected numerical variables.

### 6. Missing Value Analysis

Shows the presence of missing observations before data cleaning.

All visualizations are available in the [`plots`](./plots/) folder.

---

## 🔍 Preliminary Insights

The exploratory analysis provided several initial observations:

* Passenger survival varied considerably across gender groups.
* Passenger class was associated with differences in survival outcomes.
* Fare values showed a wide distribution with potential high-value outliers.
* Age had missing observations that required preprocessing.
* Numerical variables showed varying degrees of correlation.
* Data cleaning and preprocessing were necessary before performing reliable analysis.

These observations were used as the foundation for further analytical work.

---

## 🛠️ Tools & Technologies

* **R**
* **RStudio**
* **ggplot2**
* **dplyr**
* **readr**
* **Base R**

---

## 📁 Project Structure

```text
Week-1-Data-Cleaning-R/
│
├── README.md
│
├── data_cleaning.R
│
├── titanic_cleaned.csv
│
├── Week-1-Data-Cleaning-R-Report.pdf
│
└── plots/
    ├── 01_age_hist.png
    ├── 02_fare_boxplot.png
    ├── 03_survival_by_sex.png
    ├── 04_survival_by_class.png
    ├── 05_corr_heatmap.png
    └── 06_missing_before.png
```

---

## 📄 Deliverables

The project contains:

* **R Script** — Complete data cleaning and preliminary analysis code
* **Cleaned Dataset** — Processed Titanic dataset
* **Visualizations** — Graphs generated during exploratory analysis
* **Report** — Detailed documentation of the methodology, code, outputs, and findings

---

## 🎓 Learning Outcomes

Through this task, I gained practical experience in:

* Working with real-world datasets
* Data cleaning and preprocessing
* Missing-value handling
* Outlier detection
* Categorical data processing
* Descriptive statistics
* Exploratory data analysis
* Data visualization using R
* Interpreting analytical results
* Documenting data analysis workflows

---

## 🔗 Internship

**Virtual R Data Analyst Internship — Week 1**

This project is part of a four-week internship covering:

1. Data Cleaning and Preliminary Analysis
2. Data Visualization and Insight Communication
3. Statistical Analysis and Predictive Modeling
4. Comprehensive Data Analysis Reporting and Presentation

---

⭐ **This project demonstrates the first stage of the data analysis lifecycle: transforming raw data into a clean and analysis-ready dataset.**

