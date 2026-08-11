# ============================================
# HEALTH DATA ANALYSIS - DISEASE PREDICTION
# Step 1: Install & Load Packages
# ============================================

install.packages("tidyverse")
install.packages("caret")
install.packages("randomForest")
install.packages("corrplot")
install.packages("pROC")

# Load libraries
library(tidyverse)
library(caret)
library(randomForest)
library(corrplot)
library(pROC)

# ============================================
# Step 2: Load the Dataset
# ============================================

data <- read.csv("heart.csv")

# First look at the data
head(data)        # first 6 rows
str(data)         # structure
summary(data)     # statistics
dim(data)         # rows and columns
# Load all libraries
library(tidyverse)
library(caret)
library(randomForest)
library(corrplot)
library(pROC)

# Load the dataset
data <- read.csv("heart.csv")

# First look at the data
head(data)
str(data)
summary(data)
dim(data)
# ============================================
# Step 3: Data Cleaning
# ============================================

# Check for missing values
cat("Missing values per column:\n")
colSums(is.na(data))

# Check for duplicates
cat("Number of duplicate rows:", sum(duplicated(data)), "\n")

# Remove duplicates if any
data <- data[!duplicated(data), ]

# Convert target to factor (category)
data$target <- as.factor(data$target)

# Check the cleaned data
cat("Final dataset size:\n")
dim(data)

cat("Target distribution:\n")
table(data$target)
# ============================================
# Step 4: Data Visualization
# ============================================

# Plot 1: Disease Distribution
ggplot(data, aes(x=target, fill=target)) +
  geom_bar() +
  labs(title="Heart Disease Distribution", x="0=No Disease, 1=Disease", y="Count") +
  scale_fill_manual(values=c("skyblue","tomato")) +
  theme_minimal()

# Plot 2: Age vs Disease
ggplot(data, aes(x=target, y=age, fill=target)) +
  geom_boxplot() +
  labs(title="Age vs Heart Disease", x="0=No Disease, 1=Disease", y="Age") +
  scale_fill_manual(values=c("skyblue","tomato")) +
  theme_minimal()

# Plot 3: Correlation Heatmap
num_data <- data %>% select_if(is.numeric)
corrplot(cor(num_data), method="color", tl.cex=0.8, title="Correlation Matrix")
# ============================================
# More Visualizations (Fixed Column Names)
# ============================================

# Plot 4: Chest Pain Type vs Disease
ggplot(data, aes(x=as.factor(chest_pain_type), fill=target)) +
  geom_bar(position="dodge") +
  labs(title="Chest Pain Type vs Heart Disease", x="Chest Pain Type", y="Count") +
  scale_fill_manual(values=c("skyblue","tomato")) +
  theme_minimal()

# Plot 5: Gender vs Disease
ggplot(data, aes(x=as.factor(sex), fill=target)) +
  geom_bar(position="dodge") +
  labs(title="Gender vs Heart Disease", x="0=Female, 1=Male", y="Count") +
  scale_fill_manual(values=c("skyblue","tomato")) +
  theme_minimal()

# Plot 6: Max Heart Rate vs Disease
ggplot(data, aes(x=target, y=Max_heart_rate, fill=target)) +
  geom_violin() +
  labs(title="Max Heart Rate vs Disease", x="0=No Disease, 1=Disease", y="Max Heart Rate") +
  scale_fill_manual(values=c("skyblue","tomato")) +
  theme_minimal()

# Plot 7: Age Distribution by Disease
ggplot(data, aes(x=age, fill=target)) +
  geom_histogram(bins=30, alpha=0.7, position="identity") +
  labs(title="Age Distribution by Disease", x="Age", y="Count") +
  scale_fill_manual(values=c("skyblue","tomato")) +
  theme_minimal()

# Plot 8: Cholesterol vs Age
ggplot(data, aes(x=age, y=cholestoral, color=target)) +
  geom_point(alpha=0.7, size=3) +
  labs(title="Cholesterol vs Age", x="Age", y="Cholesterol") +
  scale_color_manual(values=c("skyblue","tomato")) +
  theme_minimal()
# ============================================
# Step 5: Build Prediction Models
# ============================================

set.seed(42)

# Split data: 80% training, 20% testing
trainIndex <- createDataPartition(data$target, p=0.8, list=FALSE)
trainData <- data[trainIndex, ]
testData  <- data[-trainIndex, ]

cat("Training rows:", nrow(trainData), "\n")
cat("Testing rows:", nrow(testData), "\n")

# --- Model 1: Logistic Regression ---
log_model <- glm(target ~ ., data=trainData, family=binomial)
log_pred <- predict(log_model, testData, type="response")
log_class <- ifelse(log_pred > 0.5, 1, 0)
log_acc <- mean(log_class == as.numeric(as.character(testData$target)))
cat("Logistic Regression Accuracy:", round(log_acc*100, 2), "%\n")

# --- Model 2: Random Forest ---
rf_model <- randomForest(target ~ ., data=trainData, ntree=100)
rf_pred <- predict(rf_model, testData)
rf_acc <- mean(rf_pred == testData$target)
cat("Random Forest Accuracy:", round(rf_acc*100, 2), "%\n")
# ============================================
# Step 6: Model Evaluation
# ============================================

# Confusion Matrix - Logistic Regression
cat("=== Logistic Regression Confusion Matrix ===\n")
confusionMatrix(as.factor(log_class), testData$target)

# Confusion Matrix - Random Forest
cat("=== Random Forest Confusion Matrix ===\n")
confusionMatrix(rf_pred, testData$target)

# ROC Curve - Logistic Regression
roc_log <- roc(as.numeric(testData$target), log_pred)
plot(roc_log, col="blue", main="ROC Curves", print.auc=TRUE)

# ROC Curve - Random Forest
rf_prob <- predict(rf_model, testData, type="prob")[,2]
roc_rf <- roc(as.numeric(testData$target), rf_prob)
plot(roc_rf, col="red", add=TRUE, print.auc=TRUE, print.auc.y=0.4)

legend("bottomright", legend=c("Logistic Regression","Random Forest"),
       col=c("blue","red"), lwd=2)
# ============================================
# Step 7: Feature Importance & Final Summary
# ============================================

# Feature Importance from Random Forest
importance_df <- data.frame(
  Feature = rownames(importance(rf_model)),
  Importance = importance(rf_model)[,1]
)
importance_df <- importance_df[order(-importance_df$Importance),]

ggplot(importance_df, aes(x=reorder(Feature, Importance), y=Importance, fill=Importance)) +
  geom_bar(stat="identity") +
  coord_flip() +
  labs(title="Feature Importance - Random Forest",
       x="Features", y="Importance Score") +
  scale_fill_gradient(low="skyblue", high="tomato") +
  theme_minimal()

# ============================================
# Step 7: Feature Importance & Final Summary
# ============================================

# Feature Importance from Random Forest
importance_df <- data.frame(
  Feature = rownames(importance(rf_model)),
  Importance = importance(rf_model)[,1]
)
importance_df <- importance_df[order(-importance_df$Importance),]

ggplot(importance_df, aes(x=reorder(Feature, Importance), y=Importance, fill=Importance)) +
  geom_bar(stat="identity") +
  coord_flip() +
  labs(title="Feature Importance - Random Forest",
       x="Features", y="Importance Score") +
  scale_fill_gradient(low="skyblue", high="tomato") +
  theme_minimal()

# Final Summary
cat("========================================\n")
cat("   HEALTH DATA ANALYSIS - FINAL REPORT  \n")
cat("========================================\n")
cat("Total Patients:", nrow(data), "\n")
cat("Heart Disease Cases:", sum(data$target==1), "\n")
cat("No Disease Cases:", sum(data$target==0), "\n")
cat("Logistic Regression Accuracy: 83.05%\n")
cat("Random Forest Accuracy: 83.05%\n")
cat("Logistic Regression AUC: 0.874\n")
cat("Random Forest AUC: 0.898\n")
cat("Best Model: Random Forest\n")
cat("========================================\n")