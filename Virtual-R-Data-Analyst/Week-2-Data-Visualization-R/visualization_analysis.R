# ============================================================
# Visualization Analysis - mtcars Dataset
# Week 2: Data Visualization and Insight Communication using R
# ============================================================

# -----------------------------
# 1. Install packages if needed
# -----------------------------
# install.packages(c("ggplot2", "dplyr", "reshape2"))

library(ggplot2)
library(dplyr)
library(reshape2)

# -----------------------------
# 2. Load the built-in dataset
# -----------------------------
data(mtcars)

# Inspect the dataset
str(mtcars)
summary(mtcars[, c("mpg", "cyl", "disp", "hp", "wt", "qsec")])

# Create output folder
if (!dir.exists("plots")) {
  dir.create("plots")
}

# ============================================================
# 3. Visualization 1 - Average MPG by Cylinder Count
# ============================================================

cyl_summary <- mtcars %>%
  group_by(cyl) %>%
  summarise(avg_mpg = mean(mpg))

p1 <- ggplot(cyl_summary,
             aes(x = factor(cyl), y = avg_mpg, fill = factor(cyl))) +
  geom_col(width = 0.6, color = "black") +
  geom_text(aes(label = round(avg_mpg, 1)),
            vjust = -0.5, fontface = "bold") +
  scale_fill_manual(values = c(
    "4" = "#4C72B0",
    "6" = "#DD8452",
    "8" = "#55A868"
  )) +
  labs(
    title = "Average Fuel Efficiency by Engine Cylinder Count",
    x = "Number of Cylinders",
    y = "Average MPG"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

print(p1)

ggsave(
  "plots/01_avg_mpg_by_cylinders.png",
  plot = p1,
  width = 8,
  height = 5,
  dpi = 150
)

# Analysis:
# 4-cylinder cars average about 26.7 MPG, 6-cylinder cars about 19.7 MPG,
# and 8-cylinder cars about 15.1 MPG. MPG decreases sharply as cylinder
# count increases, suggesting that larger engines are associated with
# lower fuel efficiency in this dataset.

# ============================================================
# 4. Visualization 2 - Horsepower Distribution
# ============================================================

p2 <- ggplot(mtcars, aes(x = hp)) +
  geom_histogram(
    bins = 8,
    fill = "#4C72B0",
    color = "black",
    alpha = 0.9
  ) +
  geom_vline(
    aes(xintercept = mean(hp)),
    color = "red",
    linetype = "dashed",
    linewidth = 1
  ) +
  labs(
    title = "Distribution of Horsepower Across 32 Automobiles",
    x = "Horsepower (hp)",
    y = "Number of Cars"
  ) +
  theme_minimal(base_size = 13)

print(p2)

ggsave(
  "plots/02_horsepower_distribution.png",
  plot = p2,
  width = 8,
  height = 5,
  dpi = 150
)

# Analysis:
# Most cars have horsepower below 200 hp, with many observations between
# approximately 60 and 180 hp. The distribution is right-skewed because
# a small number of high-performance cars have very high horsepower.
# The average horsepower is about 147 hp.

# ============================================================
# 5. Visualization 3 - Vehicle Weight vs Fuel Efficiency
# ============================================================

p3 <- ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point(size = 3, alpha = 0.9) +
  geom_smooth(
    method = "lm",
    se = FALSE,
    color = "black",
    linetype = "dashed"
  ) +
  scale_color_manual(
    values = c(
      "4" = "#4C72B0",
      "6" = "#DD8452",
      "8" = "#55A868"
    ),
    name = "Cylinders"
  ) +
  labs(
    title = "Vehicle Weight vs. Fuel Efficiency",
    x = "Weight (1000 lbs)",
    y = "Miles per Gallon (MPG)"
  ) +
  theme_minimal(base_size = 13)

print(p3)

ggsave(
  "plots/03_weight_vs_mpg.png",
  plot = p3,
  width = 8,
  height = 5,
  dpi = 150
)

# Analysis:
# Weight and MPG have a strong negative relationship, with a correlation
# of approximately -0.87. Heavier cars tend to have lower fuel efficiency.
# 8-cylinder cars are generally concentrated in the heavier, lower-MPG area,
# while 4-cylinder cars tend to be lighter and more fuel-efficient.

# ============================================================
# 6. Visualization 4 - MPG by Transmission Type
# ============================================================

p4 <- ggplot(
  mtcars,
  aes(
    x = factor(am, labels = c("Automatic", "Manual")),
    y = mpg,
    fill = factor(am)
  )
) +
  geom_boxplot(width = 0.5, alpha = 0.85) +
  geom_jitter(width = 0.05, alpha = 0.5) +
  scale_fill_manual(values = c("#4C72B0", "#DD8452")) +
  labs(
    title = "Fuel Efficiency by Transmission Type",
    x = "Transmission",
    y = "Miles per Gallon (MPG)"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

print(p4)

ggsave(
  "plots/04_mpg_by_transmission.png",
  plot = p4,
  width = 8,
  height = 5,
  dpi = 150
)

# Analysis:
# Manual-transmission cars show a higher median MPG than automatic cars
# in this sample. However, the manual group also has wider variability.
# The result should not be interpreted as a transmission-only effect because
# transmission type overlaps with vehicle weight and cylinder count.

# ============================================================
# 7. Visualization 5 - Average MPG by Number of Gears
# ============================================================

gear_summary <- mtcars %>%
  group_by(gear) %>%
  summarise(avg_mpg = mean(mpg))

p5 <- ggplot(gear_summary, aes(x = gear, y = avg_mpg)) +
  geom_line(color = "#55A868", linewidth = 1.3) +
  geom_point(size = 4, color = "#55A868") +
  geom_text(
    aes(label = round(avg_mpg, 1)),
    vjust = -1,
    fontface = "bold"
  ) +
  scale_x_continuous(breaks = c(3, 4, 5)) +
  labs(
    title = "Average MPG Trend Across Number of Forward Gears",
    x = "Number of Gears",
    y = "Average MPG"
  ) +
  theme_minimal(base_size = 13)

print(p5)

ggsave(
  "plots/05_avg_mpg_by_gears.png",
  plot = p5,
  width = 8,
  height = 5,
  dpi = 150
)

# Analysis:
# Cars with 3 gears average about 16.1 MPG, while 4-gear cars average
# about 24.5 MPG. The average falls to about 21.4 MPG for 5-gear cars.
# This does not mean that more gears automatically reduce efficiency:
# the groups contain different types of vehicles, including high-performance
# sports cars.

# ============================================================
# 8. Visualization 6 - Correlation Heatmap
# ============================================================

num_vars <- mtcars %>%
  select(mpg, disp, hp, drat, wt, qsec)

cor_matrix <- round(cor(num_vars), 2)

cor_melt <- melt(cor_matrix)

p6 <- ggplot(
  cor_melt,
  aes(x = Var1, y = Var2, fill = value)
) +
  geom_tile(color = "white") +
  geom_text(aes(label = value), size = 3) +
  scale_fill_gradient2(
    low = "#2166AC",
    mid = "white",
    high = "#B2182B",
    midpoint = 0,
    limit = c(-1, 1),
    name = "Correlation"
  ) +
  labs(
    title = "Correlation Matrix of Key Numeric Variables"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

print(p6)

ggsave(
  "plots/06_correlation_heatmap.png",
  plot = p6,
  width = 8,
  height = 6,
  dpi = 150
)

# Analysis:
# Weight (-0.87) and displacement (-0.85) have the strongest negative
# relationships with MPG. Weight and displacement are also strongly
# positively correlated (0.89), meaning larger engines tend to occur
# in heavier vehicles.
#
# Qsec has a positive correlation with MPG (0.42) and a negative
# correlation with horsepower (-0.71).

# ============================================================
# 9. Overall Visualization Analysis
# ============================================================

cat("\n============================================================\n")
cat("OVERALL VISUALIZATION ANALYSIS\n")
cat("============================================================\n")
cat("1. Engine size strongly influences fuel efficiency.\n")
cat("2. Weight is the strongest single relationship with MPG (r = -0.87).\n")
cat("3. Displacement is also strongly negatively related to MPG (r = -0.85).\n")
cat("4. Manual cars have higher MPG in this sample, but the effect is confounded.\n")
cat("5. Gear count shows a non-linear group pattern rather than a simple increase.\n")
cat("6. Horsepower is right-skewed because a small number of cars are high-performance.\n")
cat("7. Lightweight cars such as the Toyota Corolla and Fiat 128 are highly efficient.\n")
cat("8. Overall, bigger and heavier cars tend to have lower MPG in this 1973-74 sample.\n")
cat("============================================================\n")

# End of script
