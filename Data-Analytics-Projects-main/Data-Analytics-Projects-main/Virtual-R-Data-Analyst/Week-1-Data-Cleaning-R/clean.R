# ============================================================
# Week 1: Data Cleaning and Preliminary Analysis with R
# Dataset: Titanic Passenger Data (Kaggle / public domain)
# ============================================================

df <- read.csv("titanic.csv", stringsAsFactors = FALSE, na.strings = c("", "NA"))

cat("\n---- 1. STRUCTURE OF RAW DATA ----\n")
str(df)

cat("\n---- 2. DIMENSIONS ----\n")
print(dim(df))

cat("\n---- 3. SUMMARY (RAW) ----\n")
print(summary(df))

cat("\n---- 4. MISSING VALUES PER COLUMN ----\n")
na_counts <- colSums(is.na(df))
na_pct <- round(100 * na_counts / nrow(df), 2)
missing_table <- data.frame(Column = names(na_counts), Missing = na_counts, Percent = na_pct)
missing_table <- missing_table[order(-missing_table$Missing), ]
print(missing_table, row.names = FALSE)

# ============================================================
# DATA CLEANING
# ============================================================

df_clean <- df

## 4a. Drop Cabin (77% missing -> not usable for imputation)
df_clean$Cabin <- NULL

## 4b. Impute Age (numeric, ~20% missing) with MEDIAN grouped by Pclass+Sex
cat("\n---- Age missing before imputation:", sum(is.na(df_clean$Age)), "----\n")
age_medians <- aggregate(Age ~ Pclass + Sex, data = df_clean, median, na.rm = TRUE)
for (i in which(is.na(df_clean$Age))) {
  m <- age_medians$Age[age_medians$Pclass == df_clean$Pclass[i] & age_medians$Sex == df_clean$Sex[i]]
  df_clean$Age[i] <- m
}
cat("Age missing after imputation:", sum(is.na(df_clean$Age)), "\n")

## 4c. Impute Embarked (categorical, 2 missing) with MODE
mode_val <- names(sort(table(df_clean$Embarked), decreasing = TRUE))[1]
cat("\nEmbarked mode used for imputation:", mode_val, "\n")
df_clean$Embarked[is.na(df_clean$Embarked)] <- mode_val

## 4d. Impute Fare (if any missing) with median
if (any(is.na(df_clean$Fare))) {
  df_clean$Fare[is.na(df_clean$Fare)] <- median(df_clean$Fare, na.rm = TRUE)
}

cat("\n---- Missing values AFTER cleaning ----\n")
print(colSums(is.na(df_clean)))

# ============================================================
# OUTLIER DETECTION (IQR method) on Fare and Age
# ============================================================
detect_outliers <- function(x) {
  q1 <- quantile(x, 0.25); q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr; upper <- q3 + 1.5 * iqr
  sum(x < lower | x > upper)
}
cat("\n---- OUTLIER COUNTS (IQR RULE) ----\n")
cat("Fare outliers:", detect_outliers(df_clean$Fare), "\n")
cat("Age outliers :", detect_outliers(df_clean$Age), "\n")

# Cap Fare outliers at the 99th percentile (winsorize) to reduce skew impact
fare_cap <- quantile(df_clean$Fare, 0.99)
df_clean$Fare_capped <- ifelse(df_clean$Fare > fare_cap, fare_cap, df_clean$Fare)
cat("Fare 99th percentile cap applied at:", round(fare_cap, 2), "\n")

# ============================================================
# NORMALIZATION (min-max scaling for Age and Fare_capped)
# ============================================================
minmax <- function(x) (x - min(x)) / (max(x) - min(x))
df_clean$Age_norm  <- minmax(df_clean$Age)
df_clean$Fare_norm <- minmax(df_clean$Fare_capped)

cat("\n---- Normalized Age/Fare (first 5 rows) ----\n")
print(head(df_clean[, c("Age", "Age_norm", "Fare_capped", "Fare_norm")], 5))

# ============================================================
# ENCODING CATEGORICAL VARIABLES
# ============================================================
# Label encoding for Sex (binary)
df_clean$Sex_encoded <- ifelse(df_clean$Sex == "male", 1, 0)

# One-hot encoding for Embarked (3 levels: C, Q, S)
df_clean$Embarked_C <- as.integer(df_clean$Embarked == "C")
df_clean$Embarked_Q <- as.integer(df_clean$Embarked == "Q")
df_clean$Embarked_S <- as.integer(df_clean$Embarked == "S")

# Pclass and Survived as factors for analysis
df_clean$Pclass   <- factor(df_clean$Pclass, levels = c(1,2,3), labels = c("1st","2nd","3rd"))
df_clean$Survived_label <- factor(df_clean$Survived, levels = c(0,1), labels = c("No","Yes"))

cat("\n---- Encoded columns (first 5 rows) ----\n")
print(head(df_clean[, c("Sex","Sex_encoded","Embarked","Embarked_C","Embarked_Q","Embarked_S")], 5))

write.csv(df_clean, "titanic_cleaned.csv", row.names = FALSE)
cat("\nCleaned dataset saved as titanic_cleaned.csv\n")

# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

cat("\n---- STRUCTURE OF CLEANED DATA ----\n")
str(df_clean)

cat("\n---- SUMMARY OF CLEANED DATA ----\n")
print(summary(df_clean[, c("Age","Fare_capped","SibSp","Parch")]))

cat("\n---- Survival rate overall ----\n")
print(round(prop.table(table(df_clean$Survived_label)) * 100, 2))

cat("\n---- Survival rate by Sex ----\n")
print(round(prop.table(table(df_clean$Sex, df_clean$Survived_label), margin = 1) * 100, 2))

cat("\n---- Survival rate by Pclass ----\n")
print(round(prop.table(table(df_clean$Pclass, df_clean$Survived_label), margin = 1) * 100, 2))

cat("\n---- Correlation matrix (numeric vars) ----\n")
num_vars <- df_clean[, c("Survived","Age","Fare_capped","SibSp","Parch","Sex_encoded")]
cor_mat <- round(cor(num_vars, use = "complete.obs"), 2)
print(cor_mat)

# ============================================================
# VISUALIZATIONS
# ============================================================

png("plots/01_age_hist.png", width = 800, height = 600)
hist(df_clean$Age, breaks = 30, col = "steelblue", border = "white",
     main = "Distribution of Passenger Age (after imputation)",
     xlab = "Age", ylab = "Frequency")
dev.off()

png("plots/02_fare_boxplot.png", width = 800, height = 600)
boxplot(Fare ~ Pclass, data = df_clean, col = c("#66c2a5","#fc8d62","#8da0cb"),
        main = "Fare Distribution by Passenger Class (before capping)",
        xlab = "Passenger Class", ylab = "Fare")
dev.off()

png("plots/03_survival_by_sex.png", width = 800, height = 600)
counts <- table(df_clean$Sex, df_clean$Survived_label)
barplot(counts, beside = TRUE, col = c("#e78ac3","#a6d854"),
        main = "Survival Count by Sex", xlab = "Survived", ylab = "Count",
        legend.text = rownames(counts), args.legend = list(x = "topright"))
dev.off()

png("plots/04_survival_by_class.png", width = 800, height = 600)
counts2 <- table(df_clean$Pclass, df_clean$Survived_label)
barplot(counts2, beside = TRUE, col = c("#66c2a5","#fc8d62","#8da0cb"),
        main = "Survival Count by Passenger Class", xlab = "Survived", ylab = "Count",
        legend.text = rownames(counts2), args.legend = list(x = "topright"))
dev.off()

png("plots/05_corr_heatmap.png", width = 800, height = 700)
heatmap(cor_mat, symm = TRUE, main = "Correlation Heatmap", margins = c(8,8))
dev.off()

png("plots/06_missing_before.png", width = 800, height = 600)
barplot(na_counts[na_counts > 0], col = "tomato", las = 2,
        main = "Missing Values by Column (Raw Data)", ylab = "Count of Missing")
dev.off()

cat("\nAll plots saved to plots/ directory.\n")
cat("\n==== SCRIPT COMPLETE ====\n")
