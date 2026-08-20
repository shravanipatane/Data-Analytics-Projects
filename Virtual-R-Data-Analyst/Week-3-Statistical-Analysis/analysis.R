## =============================================================
## Week 3 Task: Statistical Analysis and Predictive Modeling in R
## Dataset: Restaurant Tipping Dataset (tips.csv)
## Author : Shravani Patane
## =============================================================

## ---- 0. Setup ----
set.seed(42)
options(width = 100)
dir.create("outputs", showWarnings = FALSE)

tips <- read.csv("data/tips.csv", stringsAsFactors = TRUE)

## =============================================================
## 1. DATASET OVERVIEW
## =============================================================
cat("\n==================== DATASET STRUCTURE ====================\n")
str(tips)

cat("\n==================== FIRST 6 ROWS ====================\n")
print(head(tips))

cat("\n==================== MISSING VALUES PER COLUMN ====================\n")
print(colSums(is.na(tips)))

cat("\n==================== SUMMARY STATISTICS ====================\n")
print(summary(tips))

## Derived feature used later in modelling
tips$tip_pct <- tips$tip / tips$total_bill * 100

## =============================================================
## 2. EXPLORATORY STATISTICAL ANALYSIS
## =============================================================

## ---- 2.1 Normality tests (Shapiro-Wilk) ----
cat("\n==================== SHAPIRO-WILK NORMALITY: tip ====================\n")
print(shapiro.test(tips$tip))

cat("\n==================== SHAPIRO-WILK NORMALITY: total_bill ====================\n")
print(shapiro.test(tips$total_bill))

## ---- 2.2 Correlation test: total_bill vs tip ----
cat("\n==================== PEARSON CORRELATION: total_bill vs tip ====================\n")
cor_test <- cor.test(tips$total_bill, tips$tip, method = "pearson")
print(cor_test)

## ---- 2.3 Correlation matrix (numeric variables) ----
cat("\n==================== CORRELATION MATRIX (numeric vars) ====================\n")
num_vars <- tips[, c("total_bill", "tip", "size")]
print(round(cor(num_vars), 3))

## ---- 2.4 Hypothesis Test A: Welch two-sample t-test ----
## H0: mean tip (Lunch) = mean tip (Dinner)
## H1: mean tip (Lunch) != mean tip (Dinner)
cat("\n==================== T-TEST: tip by time (Lunch vs Dinner) ====================\n")
t_time <- t.test(tip ~ time, data = tips)
print(t_time)

## ---- 2.5 Hypothesis Test B: Welch two-sample t-test ----
## H0: mean tip % (Smoker) = mean tip % (Non-smoker)
cat("\n==================== T-TEST: tip percentage by smoker status ====================\n")
t_smoker <- t.test(tip_pct ~ smoker, data = tips)
print(t_smoker)

## ---- 2.6 Hypothesis Test C: Chi-square test of independence ----
## H0: smoker status and day of week are independent
cat("\n==================== CHI-SQUARE TEST: smoker vs day ====================\n")
chi_tbl <- table(tips$smoker, tips$day)
print(chi_tbl)
chi_test <- chisq.test(chi_tbl)
print(chi_test)

## ---- 2.7 One-way ANOVA: tip across days ----
cat("\n==================== ANOVA: tip across day ====================\n")
anova_day <- aov(tip ~ day, data = tips)
print(summary(anova_day))

## ---- 2.8 Visual EDA (saved as PNG for report) ----
png("outputs/01_histograms.png", width = 900, height = 450)
par(mfrow = c(1, 2))
hist(tips$total_bill, col = "#4472C4", border = "white",
     main = "Distribution of Total Bill", xlab = "Total Bill ($)")
hist(tips$tip, col = "#ED7D31", border = "white",
     main = "Distribution of Tip", xlab = "Tip ($)")
dev.off()

png("outputs/02_boxplots.png", width = 900, height = 450)
par(mfrow = c(1, 2))
boxplot(tip ~ day, data = tips, col = "#70AD47",
        main = "Tip by Day", xlab = "Day", ylab = "Tip ($)")
boxplot(tip ~ time, data = tips, col = "#FFC000",
        main = "Tip by Time", xlab = "Time", ylab = "Tip ($)")
dev.off()

png("outputs/03_scatter_correlation.png", width = 650, height = 500)
plot(tips$total_bill, tips$tip, pch = 19, col = rgb(0.2, 0.3, 0.7, 0.5),
     main = "Total Bill vs Tip", xlab = "Total Bill ($)", ylab = "Tip ($)")
abline(lm(tip ~ total_bill, data = tips), col = "red", lwd = 2)
dev.off()

## =============================================================
## 3. MODEL BUILDING — MULTIPLE LINEAR REGRESSION
## =============================================================

## ---- 3.1 Train / Test split (80 / 20) ----
n <- nrow(tips)
train_idx <- sample(seq_len(n), size = floor(0.8 * n))
train <- tips[train_idx, ]
test  <- tips[-train_idx, ]

cat("\n==================== TRAIN / TEST SPLIT ====================\n")
cat("Training rows:", nrow(train), " | Testing rows:", nrow(test), "\n")

## ---- 3.2 Full model ----
full_model <- lm(tip ~ total_bill + size + sex + smoker + day + time, data = train)
cat("\n==================== FULL MODEL SUMMARY ====================\n")
print(summary(full_model))

## ---- 3.3 Stepwise selection (AIC-based, base R) ----
step_model <- step(full_model, direction = "both", trace = FALSE)
cat("\n==================== STEPWISE-SELECTED MODEL SUMMARY ====================\n")
print(summary(step_model))
cat("\nFormula selected by stepwise AIC:\n")
print(formula(step_model))

## ---- 3.4 5-fold Cross-Validation (manual, base R) ----
cat("\n==================== 5-FOLD CROSS-VALIDATION (base model formula) ====================\n")
k <- 5
folds <- sample(rep(1:k, length.out = nrow(train)))
cv_rmse <- numeric(k)
cv_r2   <- numeric(k)
formula_used <- formula(step_model)

for (i in 1:k) {
  cv_train <- train[folds != i, ]
  cv_test  <- train[folds == i, ]
  cv_fit   <- lm(formula_used, data = cv_train)
  preds    <- predict(cv_fit, newdata = cv_test)
  resid    <- cv_test$tip - preds
  cv_rmse[i] <- sqrt(mean(resid^2))
  ss_res <- sum(resid^2)
  ss_tot <- sum((cv_test$tip - mean(cv_test$tip))^2)
  cv_r2[i] <- 1 - ss_res / ss_tot
}

cv_results <- data.frame(Fold = 1:k, RMSE = round(cv_rmse, 4), R_squared = round(cv_r2, 4))
print(cv_results)
cat("\nMean CV RMSE:", round(mean(cv_rmse), 4), " | Mean CV R-squared:", round(mean(cv_r2), 4), "\n")

## ---- 3.5 Final evaluation on held-out test set ----
test_preds <- predict(step_model, newdata = test)
test_resid <- test$tip - test_preds

rmse_test <- sqrt(mean(test_resid^2))
mae_test  <- mean(abs(test_resid))
ss_res_t  <- sum(test_resid^2)
ss_tot_t  <- sum((test$tip - mean(test$tip))^2)
r2_test   <- 1 - ss_res_t / ss_tot_t

cat("\n==================== HOLD-OUT TEST SET PERFORMANCE ====================\n")
cat("RMSE:", round(rmse_test, 4), "\n")
cat("MAE :", round(mae_test, 4), "\n")
cat("R^2 :", round(r2_test, 4), "\n")

## =============================================================
## 4. DIAGNOSTIC ANALYSIS
## =============================================================

## ---- 4.1 Base diagnostic plots (residuals, QQ, scale-location, leverage) ----
png("outputs/04_diagnostic_plots.png", width = 900, height = 900)
par(mfrow = c(2, 2))
plot(step_model)
dev.off()

## ---- 4.2 Predicted vs Actual (test set) ----
png("outputs/05_predicted_vs_actual.png", width = 650, height = 500)
plot(test$tip, test_preds, pch = 19, col = rgb(0.8, 0.2, 0.2, 0.6),
     main = "Predicted vs Actual Tip (Test Set)",
     xlab = "Actual Tip ($)", ylab = "Predicted Tip ($)")
abline(0, 1, col = "blue", lwd = 2, lty = 2)
dev.off()

## ---- 4.3 Manual multicollinearity check (VIF without external packages) ----
cat("\n==================== MANUAL VIF (1 / (1 - R^2) of auxiliary regressions) ====================\n")
predictors <- attr(terms(step_model), "term.labels")
numeric_predictors <- predictors[predictors %in% c("total_bill", "size")]
if (length(numeric_predictors) > 1) {
  vif_vals <- sapply(numeric_predictors, function(v) {
    other <- setdiff(numeric_predictors, v)
    aux_formula <- as.formula(paste(v, "~", paste(other, collapse = "+")))
    aux_model <- lm(aux_formula, data = train)
    1 / (1 - summary(aux_model)$r.squared)
  })
  print(round(vif_vals, 3))
} else {
  cat("Only one numeric predictor retained in final model — VIF not applicable.\n")
}

## =============================================================
## 5. SAVE KEY OBJECTS FOR REPORT
## =============================================================
sink("outputs/model_formula.txt")
cat("Final model formula:\n")
print(formula(step_model))
cat("\nCoefficients:\n")
print(round(coef(step_model), 4))
sink()

cat("\n\n===== SCRIPT COMPLETE — all outputs saved in outputs/ =====\n")
