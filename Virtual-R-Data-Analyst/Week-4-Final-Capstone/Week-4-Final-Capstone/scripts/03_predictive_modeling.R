# =============================================================
# 03_predictive_modeling.R
# Employee Attrition Analysis — Predictive Modeling
# Virtual R Data Analyst Internship | Week 4 Final Project
# =============================================================
# Builds and compares a Logistic Regression model and a Random
# Forest classifier to predict employee Attrition (Yes/No).

library(tidyverse)
library(caret)
library(randomForest)
library(pROC)

set.seed(42)

data <- read_csv("data/employee_attrition_cleaned.csv") %>%
  mutate(across(c(Gender, Department, JobRole, EducationField, MaritalStatus,
                   BusinessTravel, OverTime, Attrition), as.factor)) %>%
  select(-EmployeeID)

# ---- 1. Train / test split (75 / 25, stratified) --------------------
train_idx <- createDataPartition(data$Attrition, p = 0.75, list = FALSE)
train_data <- data[train_idx, ]
test_data  <- data[-train_idx, ]

# ---- 2. Logistic Regression -------------------------------------------
logit_model <- glm(Attrition ~ ., data = train_data, family = "binomial")
logit_prob  <- predict(logit_model, test_data, type = "response")
logit_pred  <- factor(ifelse(logit_prob > 0.5, "Yes", "No"), levels = c("No", "Yes"))

logit_cm <- confusionMatrix(logit_pred, test_data$Attrition, positive = "Yes")
logit_roc <- roc(response = test_data$Attrition, predictor = logit_prob, levels = c("No","Yes"))

cat("=== Logistic Regression ===\n")
print(logit_cm)
cat("AUC:", auc(logit_roc), "\n\n")

# ---- 3. Random Forest ---------------------------------------------------
rf_model <- randomForest(Attrition ~ ., data = train_data,
                          ntree = 300, maxnodes = 30, importance = TRUE)
rf_prob  <- predict(rf_model, test_data, type = "prob")[, "Yes"]
rf_pred  <- predict(rf_model, test_data, type = "class")

rf_cm <- confusionMatrix(rf_pred, test_data$Attrition, positive = "Yes")
rf_roc <- roc(response = test_data$Attrition, predictor = rf_prob, levels = c("No","Yes"))

cat("=== Random Forest ===\n")
print(rf_cm)
cat("AUC:", auc(rf_roc), "\n\n")

# ---- 4. ROC curve comparison plot ----------------------------------------
png("outputs/roc_curve.png", width = 800, height = 650, res = 140)
plot(logit_roc, col = "#F8766D", lwd = 2, main = "ROC Curve: Attrition Prediction Models")
lines(rf_roc, col = "#00BFC4", lwd = 2)
legend("bottomright",
       legend = c(sprintf("Logistic Regression (AUC=%.2f)", auc(logit_roc)),
                  sprintf("Random Forest (AUC=%.2f)", auc(rf_roc))),
       col = c("#F8766D", "#00BFC4"), lwd = 2)
dev.off()

# ---- 5. Feature importance (Random Forest) --------------------------------
importance_df <- as.data.frame(importance(rf_model)) %>%
  rownames_to_column("Feature") %>%
  arrange(desc(MeanDecreaseGini)) %>%
  slice_head(n = 10)

p_imp <- ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(fill = "#00BFC4") +
  coord_flip() +
  labs(title = "Top 10 Feature Importances (Random Forest)", x = NULL, y = "Mean Decrease in Gini")
ggsave("outputs/feature_importance.png", p_imp, width = 6, height = 5, dpi = 140)

# ---- 6. Print model comparison summary -------------------------------------
results <- tibble(
  Model = c("Logistic Regression", "Random Forest"),
  Accuracy = c(logit_cm$overall["Accuracy"], rf_cm$overall["Accuracy"]),
  Sensitivity = c(logit_cm$byClass["Sensitivity"], rf_cm$byClass["Sensitivity"]),
  Specificity = c(logit_cm$byClass["Specificity"], rf_cm$byClass["Specificity"]),
  AUC = c(auc(logit_roc), auc(rf_roc))
)
print(results)
