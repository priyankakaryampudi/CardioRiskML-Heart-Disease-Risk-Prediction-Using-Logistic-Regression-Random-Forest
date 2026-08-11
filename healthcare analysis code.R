# ============================================================
# CARDIORISKML - HEART DISEASE RISK PREDICTION
# Logistic Regression vs Random Forest
# ============================================================

# ============================================================
# 1. LOAD REQUIRED PACKAGES
# ============================================================

library(tidyverse)
library(caret)
library(randomForest)
library(corrplot)
library(pROC)

# ============================================================
# 2. LOAD DATASET
# ============================================================

data <- read.csv("heart.csv")

cat("========================================\n")
cat("       DATASET INFORMATION\n")
cat("========================================\n")

cat("Rows:", nrow(data), "\n")
cat("Columns:", ncol(data), "\n\n")

cat("First 6 rows:\n")
print(head(data))

cat("\nDataset Structure:\n")
str(data)

cat("\nSummary Statistics:\n")
print(summary(data))


# ============================================================
# 3. DATA CLEANING
# ============================================================

cat("\n========================================\n")
cat("       DATA CLEANING\n")
cat("========================================\n")

# Missing values
cat("Missing values per column:\n")
print(colSums(is.na(data)))

# Duplicate records
duplicate_count <- sum(duplicated(data))

cat("\nDuplicate rows:", duplicate_count, "\n")

# Remove duplicate records
data <- data[!duplicated(data), ]

cat("Rows after removing duplicates:", nrow(data), "\n")

# Convert target into factor
data$target <- as.factor(data$target)

cat("\nTarget Distribution:\n")
print(table(data$target))


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

# ------------------------------------------------------------
# Plot 1: Heart Disease Distribution
# ------------------------------------------------------------

p1 <- ggplot(data, aes(x = target, fill = target)) +
  geom_bar() +
  labs(
    title = "Heart Disease Distribution",
    x = "Heart Disease (0 = No, 1 = Yes)",
    y = "Count"
  ) +
  theme_minimal()

print(p1)

ggsave("Rplot.png", p1, width = 8, height = 6)


# ------------------------------------------------------------
# Plot 2: Age vs Heart Disease
# ------------------------------------------------------------

p2 <- ggplot(data, aes(x = target, y = age, fill = target)) +
  geom_boxplot() +
  labs(
    title = "Age vs Heart Disease",
    x = "Heart Disease",
    y = "Age"
  ) +
  theme_minimal()

print(p2)

ggsave("Rplot01.png", p2, width = 8, height = 6)


# ------------------------------------------------------------
# Plot 3: Correlation Heatmap
# ------------------------------------------------------------

numeric_data <- data %>%
  select(where(is.numeric))

png("Rplot02.png", width = 900, height = 700)

corrplot(
  cor(numeric_data),
  method = "color",
  tl.cex = 0.8,
  title = "Correlation Matrix",
  mar = c(0, 0, 2, 0)
)

dev.off()


# ------------------------------------------------------------
# Plot 4: Chest Pain Type vs Disease
# ------------------------------------------------------------

p4 <- ggplot(
  data,
  aes(x = as.factor(chest_pain_type), fill = target)
) +
  geom_bar(position = "dodge") +
  labs(
    title = "Chest Pain Type vs Heart Disease",
    x = "Chest Pain Type",
    y = "Count"
  ) +
  theme_minimal()

print(p4)

ggsave("Rplot03.png", p4, width = 8, height = 6)


# ------------------------------------------------------------
# Plot 5: Gender vs Disease
# ------------------------------------------------------------

p5 <- ggplot(
  data,
  aes(x = as.factor(sex), fill = target)
) +
  geom_bar(position = "dodge") +
  labs(
    title = "Gender vs Heart Disease",
    x = "Sex (0 = Female, 1 = Male)",
    y = "Count"
  ) +
  theme_minimal()

print(p5)

ggsave("Rplot04.png", p5, width = 8, height = 6)


# ------------------------------------------------------------
# Plot 6: Maximum Heart Rate vs Disease
# ------------------------------------------------------------

p6 <- ggplot(
  data,
  aes(x = target, y = Max_heart_rate, fill = target)
) +
  geom_violin() +
  labs(
    title = "Maximum Heart Rate vs Heart Disease",
    x = "Heart Disease",
    y = "Maximum Heart Rate"
  ) +
  theme_minimal()

print(p6)

ggsave("Rplot05.png", p6, width = 8, height = 6)


# ------------------------------------------------------------
# Plot 7: Age Distribution by Disease
# ------------------------------------------------------------

p7 <- ggplot(
  data,
  aes(x = age, fill = target)
) +
  geom_histogram(
    bins = 30,
    alpha = 0.7,
    position = "identity"
  ) +
  labs(
    title = "Age Distribution by Heart Disease",
    x = "Age",
    y = "Count"
  ) +
  theme_minimal()

print(p7)

ggsave("Rplot06.png", p7, width = 8, height = 6)


# ------------------------------------------------------------
# Plot 8: Cholesterol vs Age
# ------------------------------------------------------------

p8 <- ggplot(
  data,
  aes(x = age, y = cholestoral, color = target)
) +
  geom_point(alpha = 0.7, size = 3) +
  labs(
    title = "Cholesterol vs Age",
    x = "Age",
    y = "Cholesterol"
  ) +
  theme_minimal()

print(p8)

ggsave("Rplot07.png", p8, width = 8, height = 6)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

set.seed(42)

trainIndex <- createDataPartition(
  data$target,
  p = 0.80,
  list = FALSE
)

trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

cat("\n========================================\n")
cat("       TRAIN / TEST SPLIT\n")
cat("========================================\n")

cat("Training rows:", nrow(trainData), "\n")
cat("Testing rows:", nrow(testData), "\n")


# ============================================================
# 6. LOGISTIC REGRESSION MODEL
# ============================================================

log_model <- glm(
  target ~ .,
  data = trainData,
  family = binomial
)

# Probability prediction
log_prob <- predict(
  log_model,
  testData,
  type = "response"
)

# Class prediction
log_class <- ifelse(
  log_prob >= 0.5,
  1,
  0
)

# Convert to factor
log_class_factor <- factor(
  log_class,
  levels = c(0, 1)
)

# Accuracy
actual_target <- as.numeric(
  as.character(testData$target)
)

log_accuracy <- mean(
  log_class == actual_target
)


# ============================================================
# 7. RANDOM FOREST MODEL
# ============================================================

set.seed(42)

rf_model <- randomForest(
  target ~ .,
  data = trainData,
  ntree = 100,
  importance = TRUE
)

# Class prediction
rf_pred <- predict(
  rf_model,
  testData
)

# Probability prediction
rf_prob <- predict(
  rf_model,
  testData,
  type = "prob"
)[, "1"]

# Accuracy
rf_accuracy <- mean(
  rf_pred == testData$target
)


# ============================================================
# 8. MODEL ACCURACY
# ============================================================

cat("\n========================================\n")
cat("       MODEL ACCURACY\n")
cat("========================================\n")

cat(
  "Logistic Regression Accuracy:",
  round(log_accuracy * 100, 2),
  "%\n"
)

cat(
  "Random Forest Accuracy:",
  round(rf_accuracy * 100, 2),
  "%\n"
)


# ============================================================
# 9. CONFUSION MATRICES
# ============================================================

cat("\n========================================\n")
cat(" LOGISTIC REGRESSION CONFUSION MATRIX\n")
cat("========================================\n")

log_cm <- confusionMatrix(
  log_class_factor,
  testData$target,
  positive = "1"
)

print(log_cm)


cat("\n========================================\n")
cat(" RANDOM FOREST CONFUSION MATRIX\n")
cat("========================================\n")

rf_cm <- confusionMatrix(
  rf_pred,
  testData$target,
  positive = "1"
)

print(rf_cm)


# ============================================================
# 10. ROC-AUC ANALYSIS
# ============================================================

roc_log <- roc(
  actual_target,
  log_prob,
  quiet = TRUE
)

roc_rf <- roc(
  actual_target,
  rf_prob,
  quiet = TRUE
)

log_auc <- as.numeric(auc(roc_log))
rf_auc <- as.numeric(auc(roc_rf))


# ============================================================
# 11. ROC CURVE
# ============================================================

png("Rplot08.png", width = 900, height = 700)

plot(
  roc_log,
  main = "ROC Curve Comparison",
  lwd = 2
)

lines(
  roc_rf,
  lwd = 2
)

legend(
  "bottomright",
  legend = c(
    paste0(
      "Logistic Regression (AUC = ",
      round(log_auc, 3),
      ")"
    ),
    paste0(
      "Random Forest (AUC = ",
      round(rf_auc, 3),
      ")"
    )
  ),
  lwd = 2
)

dev.off()


# Display ROC curve
plot(
  roc_log,
  main = "ROC Curve Comparison",
  lwd = 2
)

lines(
  roc_rf,
  lwd = 2
)

legend(
  "bottomright",
  legend = c(
    paste0(
      "Logistic Regression (AUC = ",
      round(log_auc, 3),
      ")"
    ),
    paste0(
      "Random Forest (AUC = ",
      round(rf_auc, 3),
      ")"
    )
  ),
  lwd = 2
)


# ============================================================
# 12. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance_values <- importance(rf_model)

importance_df <- data.frame(
  Feature = rownames(importance_values),
  Importance = importance_values[, "MeanDecreaseGini"]
)

importance_df <- importance_df[
  order(
    -importance_df$Importance
  ),
]

print(importance_df)


# ============================================================
# 13. FEATURE IMPORTANCE PLOT
# ============================================================

p9 <- ggplot(
  importance_df,
  aes(
    x = reorder(Feature, Importance),
    y = Importance
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Random Forest Feature Importance",
    x = "Features",
    y = "Importance Score"
  ) +
  theme_minimal()

print(p9)

ggsave(
  "Rplot09.png",
  p9,
  width = 9,
  height = 7
)


# ============================================================
# 14. FINAL RESULTS
# ============================================================

best_model <- ifelse(
  rf_auc > log_auc,
  "Random Forest",
  "Logistic Regression"
)

cat("\n")
cat("========================================\n")
cat("       CARDIORISKML FINAL REPORT\n")
cat("========================================\n")

cat(
  "Original Dataset Rows:",
  nrow(read.csv("heart.csv")),
  "\n"
)

cat(
  "Duplicate Rows Removed:",
  duplicate_count,
  "\n"
)

cat(
  "Final Dataset Rows:",
  nrow(data),
  "\n"
)

cat(
  "Heart Disease Cases:",
  sum(data$target == "1"),
  "\n"
)

cat(
  "No Disease Cases:",
  sum(data$target == "0"),
  "\n"
)

cat("\n")

cat(
  "Logistic Regression Accuracy:",
  round(log_accuracy * 100, 2),
  "%\n"
)

cat(
  "Random Forest Accuracy:",
  round(rf_accuracy * 100, 2),
  "%\n"
)

cat(
  "Logistic Regression AUC:",
  round(log_auc, 3),
  "\n"
)

cat(
  "Random Forest AUC:",
  round(rf_auc, 3),
  "\n"
)

cat("\n")

cat(
  "Best Model:",
  best_model,
  "\n"
)

cat("========================================\n")
cat("             ANALYSIS COMPLETE\n")
cat("========================================\n")
