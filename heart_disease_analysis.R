# ============================================================
# CARDIORISKML - HEART DISEASE RISK PREDICTION
# Logistic Regression and Random Forest
# ============================================================

# -----------------------------
# 1. Load Required Packages
# -----------------------------

library(tidyverse)
library(caret)
library(randomForest)
library(corrplot)
library(pROC)

# -----------------------------
# 2. Load Dataset
# -----------------------------

data <- read.csv("heart.csv")

cat("Dataset Dimensions:\n")
print(dim(data))

cat("\nFirst 6 Rows:\n")
print(head(data))

cat("\nDataset Structure:\n")
str(data)

# -----------------------------
# 3. Data Cleaning
# -----------------------------

cat("\nMissing Values:\n")
print(colSums(is.na(data)))

cat("\nDuplicate Rows:", sum(duplicated(data)), "\n")

# Remove duplicate rows
data <- data[!duplicated(data), ]

# Convert target to factor
data$target <- as.factor(data$target)

cat("\nCleaned Dataset Dimensions:\n")
print(dim(data))

cat("\nTarget Distribution:\n")
print(table(data$target))


# ============================================================
# 4. DATA VISUALIZATION
# ============================================================

# Create plots folder if it does not exist
if (!dir.exists("plots")) {
  dir.create("plots")
}

# -----------------------------
# Plot 1: Heart Disease Distribution
# -----------------------------

p1 <- ggplot(data, aes(x = target, fill = target)) +
  geom_bar() +
  labs(
    title = "Heart Disease Distribution",
    x = "Heart Disease (0 = No, 1 = Yes)",
    y = "Number of Patients"
  ) +
  theme_minimal()

print(p1)

ggsave(
  "plots/Rplot.png",
  p1,
  width = 8,
  height = 6
)


# -----------------------------
# Plot 2: Age vs Heart Disease
# -----------------------------

p2 <- ggplot(data, aes(x = target, y = age, fill = target)) +
  geom_boxplot() +
  labs(
    title = "Age vs Heart Disease",
    x = "Heart Disease",
    y = "Age"
  ) +
  theme_minimal()

print(p2)

ggsave(
  "plots/Rplot01.png",
  p2,
  width = 8,
  height = 6
)


# -----------------------------
# Plot 3: Correlation Matrix
# -----------------------------

num_data <- data %>%
  select(where(is.numeric))

png(
  "plots/Rplot02.png",
  width = 900,
  height = 700
)

corrplot(
  cor(num_data),
  method = "color",
  tl.cex = 0.8,
  title = "Correlation Matrix",
  mar = c(0, 0, 2, 0)
)

dev.off()


# -----------------------------
# Plot 4: Chest Pain vs Disease
# -----------------------------

p4 <- ggplot(
  data,
  aes(x = chest_pain_type, fill = target)
) +
  geom_bar(position = "dodge") +
  labs(
    title = "Chest Pain Type vs Heart Disease",
    x = "Chest Pain Type",
    y = "Number of Patients"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

print(p4)

ggsave(
  "plots/Rplot03.png",
  p4,
  width = 9,
  height = 6
)


# -----------------------------
# Plot 5: Gender vs Disease
# -----------------------------

p5 <- ggplot(
  data,
  aes(x = sex, fill = target)
) +
  geom_bar(position = "dodge") +
  labs(
    title = "Gender vs Heart Disease",
    x = "Gender",
    y = "Number of Patients"
  ) +
  theme_minimal()

print(p5)

ggsave(
  "plots/Rplot04.png",
  p5,
  width = 8,
  height = 6
)


# -----------------------------
# Plot 6: Maximum Heart Rate
# -----------------------------

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

ggsave(
  "plots/Rplot05.png",
  p6,
  width = 8,
  height = 6
)


# -----------------------------
# Plot 7: Age Distribution
# -----------------------------

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
    y = "Number of Patients"
  ) +
  theme_minimal()

print(p7)

ggsave(
  "plots/Rplot06.png",
  p7,
  width = 8,
  height = 6
)


# -----------------------------
# Plot 8: Cholesterol vs Age
# -----------------------------

p8 <- ggplot(
  data,
  aes(
    x = age,
    y = cholestoral,
    color = target
  )
) +
  geom_point(alpha = 0.7, size = 3) +
  labs(
    title = "Cholesterol vs Age",
    x = "Age",
    y = "Cholesterol"
  ) +
  theme_minimal()

print(p8)

ggsave(
  "plots/Rplot07.png",
  p8,
  width = 8,
  height = 6
)


# ============================================================
# 5. MACHINE LEARNING MODELS
# ============================================================

set.seed(42)

# 80% training, 20% testing
trainIndex <- createDataPartition(
  data$target,
  p = 0.8,
  list = FALSE
)

trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

cat("\nTraining Rows:", nrow(trainData), "\n")
cat("Testing Rows:", nrow(testData), "\n")


# -----------------------------
# Logistic Regression
# -----------------------------

log_model <- glm(
  target ~ .,
  data = trainData,
  family = binomial
)

log_prob <- predict(
  log_model,
  testData,
  type = "response"
)

log_class <- factor(
  ifelse(log_prob >= 0.5, 1, 0),
  levels = c(0, 1)
)

log_accuracy <- mean(
  log_class == testData$target
)


# -----------------------------
# Random Forest
# -----------------------------

rf_model <- randomForest(
  target ~ .,
  data = trainData,
  ntree = 100,
  importance = TRUE
)

rf_class <- predict(
  rf_model,
  testData
)

rf_prob <- predict(
  rf_model,
  testData,
  type = "prob"
)[, "1"]

rf_accuracy <- mean(
  rf_class == testData$target
)


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

cat("\n========================================\n")
cat("MODEL PERFORMANCE\n")
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


# -----------------------------
# Confusion Matrices
# -----------------------------

cat("\n=== Logistic Regression Confusion Matrix ===\n")

log_cm <- confusionMatrix(
  log_class,
  testData$target,
  positive = "1"
)

print(log_cm)


cat("\n=== Random Forest Confusion Matrix ===\n")

rf_cm <- confusionMatrix(
  rf_class,
  testData$target,
  positive = "1"
)

print(rf_cm)


# -----------------------------
# ROC and AUC
# -----------------------------

actual <- as.numeric(
  as.character(testData$target)
)

roc_log <- roc(
  actual,
  log_prob,
  quiet = TRUE
)

roc_rf <- roc(
  actual,
  rf_prob,
  quiet = TRUE
)

log_auc <- auc(roc_log)
rf_auc <- auc(roc_rf)

cat(
  "\nLogistic Regression AUC:",
  round(log_auc, 3),
  "\n"
)

cat(
  "Random Forest AUC:",
  round(rf_auc, 3),
  "\n"
)


# -----------------------------
# Plot 9: ROC Curve
# -----------------------------

png(
  "plots/Rplot08.png",
  width = 900,
  height = 700
)

plot(
  roc_log,
  main = "ROC Curve Comparison",
  print.auc = FALSE
)

lines(
  roc_rf,
  lty = 2
)

legend(
  "bottomright",
  legend = c(
    paste0("Logistic Regression AUC = ", round(log_auc, 3)),
    paste0("Random Forest AUC = ", round(rf_auc, 3))
  ),
  lty = c(1, 2)
)

dev.off()


# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

importance_values <- importance(rf_model)

importance_df <- data.frame(
  Feature = rownames(importance_values),
  Importance = importance_values[, "MeanDecreaseGini"]
)

importance_df <- importance_df[
  order(-importance_df$Importance),
]


# -----------------------------
# Plot 10: Feature Importance
# -----------------------------

p10 <- ggplot(
  importance_df,
  aes(
    x = reorder(Feature, Importance),
    y = Importance
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Feature Importance - Random Forest",
    x = "Features",
    y = "Importance Score"
  ) +
  theme_minimal()

print(p10)

ggsave(
  "plots/Rplot09.png",
  p10,
  width = 9,
  height = 7
)


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

best_model <- ifelse(
  rf_accuracy > log_accuracy,
  "Random Forest",
  "Logistic Regression"
)

cat("\n========================================\n")
cat("       CARDIORISKML FINAL REPORT\n")
cat("========================================\n")

cat(
  "Total Patients:",
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

cat(
  "Best Model:",
  best_model,
  "\n"
)

cat("========================================\n")
