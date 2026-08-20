# =============================================================
# 01_data_cleaning.R
# Employee Attrition Analysis — Data Cleaning & Preparation
# Virtual R Data Analyst Internship | Week 4 Final Project
# =============================================================
# Purpose: Load the raw HR dataset, diagnose data quality issues,
# and produce a clean, analysis-ready dataset.

library(tidyverse)   # dplyr, readr, stringr, ggplot2
library(janitor)     # duplicate & name cleaning helpers

# ---- 1. Load raw data ----------------------------------------
raw_data <- read_csv("data/employee_attrition_raw.csv")

cat("Initial rows:", nrow(raw_data), " Columns:", ncol(raw_data), "\n")
glimpse(raw_data)

# ---- 2. Remove duplicate records -------------------------------
dupe_count <- sum(duplicated(raw_data))
data <- raw_data %>% distinct()
cat("Duplicate rows removed:", dupe_count, "-> rows now", nrow(data), "\n")

# ---- 3. Standardize categorical text formatting -----------------
data <- data %>%
  mutate(
    Gender     = str_to_title(str_trim(Gender)),
    OverTime   = str_to_title(str_trim(OverTime)),
    Department = str_to_title(str_trim(Department)),
    Department = str_replace(Department, "And", "&")
  )

# ---- 4. Correct invalid values ----------------------------------
invalid_age <- sum(data$Age < 18, na.rm = TRUE)
data$Age[data$Age < 18] <- NA
cat("Invalid Age values (<18) set to NA:", invalid_age, "\n")

# ---- 5. Handle missing values -----------------------------------
# Numeric columns: median imputation (robust to skew/outliers)
numeric_impute_cols <- c("MonthlyIncome", "DistanceFromHome", "WorkLifeBalance", "Age")
for (col in numeric_impute_cols) {
  n_missing <- sum(is.na(data[[col]]))
  med_val <- median(data[[col]], na.rm = TRUE)
  data[[col]][is.na(data[[col]])] <- med_val
  if (n_missing > 0) {
    cat("Imputed", n_missing, "missing values in", col, "with median (", med_val, ")\n")
  }
}

# Categorical columns: mode imputation
get_mode <- function(x) {
  ux <- na.omit(unique(x))
  ux[which.max(tabulate(match(x, ux)))]
}
n_missing_edu <- sum(is.na(data$EducationField))
mode_val <- get_mode(data$EducationField)
data$EducationField[is.na(data$EducationField)] <- mode_val
cat("Imputed", n_missing_edu, "missing values in EducationField with mode (", mode_val, ")\n")

# ---- 6. Outlier treatment (IQR capping on MonthlyIncome) --------
Q1 <- quantile(data$MonthlyIncome, 0.25)
Q3 <- quantile(data$MonthlyIncome, 0.75)
IQR_val <- Q3 - Q1
upper_bound <- Q3 + 1.5 * IQR_val
n_outliers <- sum(data$MonthlyIncome > upper_bound)
data$MonthlyIncome[data$MonthlyIncome > upper_bound] <- upper_bound
cat("Capped", n_outliers, "outlier MonthlyIncome values above upper bound (",
    round(upper_bound), ") using IQR method\n")

# ---- 7. Final type conversions -----------------------------------
data <- data %>%
  mutate(
    Age = as.integer(Age),
    Attrition = factor(Attrition, levels = c("No", "Yes")),
    Department = factor(Department),
    JobRole = factor(JobRole),
    OverTime = factor(OverTime, levels = c("No", "Yes"))
  )

cat("Final cleaned rows:", nrow(data), " Columns:", ncol(data), "\n")

# ---- 8. Export cleaned dataset -----------------------------------
write_csv(data, "data/employee_attrition_cleaned.csv")
cat("Cleaned dataset saved to data/employee_attrition_cleaned.csv\n")
