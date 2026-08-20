# =============================================================
# 02_eda_visualization.R
# Employee Attrition Analysis — Exploratory Data Analysis
# Virtual R Data Analyst Internship | Week 4 Final Project
# =============================================================

library(tidyverse)
library(corrplot)
library(scales)

data <- read_csv("data/employee_attrition_cleaned.csv") %>%
  mutate(Attrition = factor(Attrition, levels = c("No", "Yes")))

theme_set(theme_gray(base_size = 11))

# ---- 1. Attrition rate by department -----------------------------
dept_rate <- data %>%
  group_by(Department) %>%
  summarise(AttritionRate = mean(Attrition == "Yes") * 100) %>%
  arrange(AttritionRate)

p1 <- ggplot(dept_rate, aes(x = reorder(Department, AttritionRate), y = AttritionRate)) +
  geom_col(fill = "#F8766D") +
  coord_flip() +
  labs(title = "Attrition Rate by Department", x = NULL, y = "Attrition Rate (%)")
ggsave("outputs/attrition_by_department.png", p1, width = 6, height = 4, dpi = 140)

# ---- 2. Monthly income distribution by attrition -------------------
p2 <- ggplot(data, aes(x = Attrition, y = MonthlyIncome, fill = Attrition)) +
  geom_boxplot() +
  scale_fill_manual(values = c("No" = "#00BFC4", "Yes" = "#F8766D")) +
  labs(title = "Monthly Income Distribution by Attrition Status")
ggsave("outputs/income_by_attrition.png", p2, width = 6, height = 4, dpi = 140)

# ---- 3. Attrition rate by overtime status --------------------------
p3 <- data %>%
  count(OverTime, Attrition) %>%
  group_by(OverTime) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = OverTime, y = pct, fill = Attrition)) +
  geom_col() +
  scale_fill_manual(values = c("No" = "#00BFC4", "Yes" = "#F8766D")) +
  labs(title = "Attrition Rate by OverTime Status", y = "% of Employees")
ggsave("outputs/overtime_attrition.png", p3, width = 6, height = 4, dpi = 140)

# ---- 4. Attrition by job satisfaction level -------------------------
p4 <- data %>%
  count(JobSatisfaction, Attrition) %>%
  group_by(JobSatisfaction) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = factor(JobSatisfaction), y = pct, fill = Attrition)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = c("No" = "#00BFC4", "Yes" = "#F8766D")) +
  labs(title = "Attrition by Job Satisfaction Level",
       x = "Job Satisfaction (1 = Low, 4 = High)", y = "% of Employees")
ggsave("outputs/satisfaction_attrition.png", p4, width = 6, height = 4, dpi = 140)

# ---- 5. Correlation matrix of numeric predictors ---------------------
numeric_cols <- data %>%
  select(Age, MonthlyIncome, YearsAtCompany, YearsInCurrentRole, DistanceFromHome,
         JobSatisfaction, WorkLifeBalance, PerformanceRating,
         NumCompaniesWorked, TrainingTimesLastYear)

png("outputs/correlation_matrix.png", width = 900, height = 700, res = 140)
corrplot(cor(numeric_cols), method = "color", type = "upper",
         addCoef.col = "black", number.cex = 0.6, tl.col = "black",
         title = "Correlation Matrix of Numeric Features", mar = c(0,0,2,0))
dev.off()

# ---- 6. Tenure distribution by attrition ------------------------------
p6 <- ggplot(data, aes(x = YearsAtCompany, fill = Attrition)) +
  geom_histogram(bins = 15, position = "stack") +
  scale_fill_manual(values = c("No" = "#00BFC4", "Yes" = "#F8766D")) +
  labs(title = "Years at Company Distribution by Attrition")
ggsave("outputs/tenure_attrition.png", p6, width = 6, height = 4, dpi = 140)

cat("All EDA visualizations saved to /outputs\n")
