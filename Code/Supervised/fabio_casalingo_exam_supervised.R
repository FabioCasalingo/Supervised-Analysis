library(rpart)
library(rpart.plot)
library(caret)
library(dplyr)
library(ggcorrplot)
library(randomForest)
library(ranger)
library(gbm)
library(car) 
library(ggthemes)
library(yarr)
library(coefplot)
library(reshape2)
library(ggplot2)
library(pROC)
library(tidyr)
library(caTools)  
library(kableExtra)
library(reshape2)
library(MASS)

#Import of dataset and manipulation
data <- read.csv("Datasets/Dataset_supervised/credit_risk_dataset.csv")

#data <- na.omit(data)

numeric_columns <- sapply(data, is.numeric)
 
data[, numeric_columns] <- lapply(data[, numeric_columns], function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))


  data <- data[, !(names(data) %in% c("person_emp_length", "cb_person_default_on_file"))] %>%
    mutate(person_home_ownership = case_when(
    person_home_ownership == "RENT" ~ 1,
    person_home_ownership == "OWN" ~ 2,
    person_home_ownership == "MORTGAGE" ~ 3,
    person_home_ownership == "OTHER" ~ 4,
    TRUE ~ NA_integer_  # Handling missing values
  )) %>%
  mutate(loan_intent = case_when(
    loan_intent == "PERSONAL" ~ 1,
    loan_intent == "EDUCATION" ~ 2,
    loan_intent == "MEDICAL" ~ 3,
    loan_intent == "VENTURE" ~ 4,
    loan_intent == "HOMEIMPROVEMENT" ~ 5,
    loan_intent == "DEBTCONSOLIDATION" ~ 6,
    TRUE ~ NA_integer_  
  ))

first_10_rows <- head(data, 10)

kable(first_10_rows, "html") %>%
  kable_styling(full_width = FALSE) 


boxplot(data, main = "Boxplot del dataset whit outliers", ylab = "Valore")

#Outlier removing
remove_outliers <- function(data, variable) {
  Q1 <- quantile(data[[variable]], 0.25)
  Q3 <- quantile(data[[variable]], 0.75)
  
  IQR <- Q3 - Q1
  
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  
  data <- data[data[[variable]] >= lower_bound & data[[variable]] <= upper_bound, ]
  
  return(data)
}

data <- remove_outliers(data, "person_income")
data <- remove_outliers(data, "loan_amnt")

boxplot(data, main = "Boxplot del dataset without outliers", ylab = "Valore")

#Correlation matrix
corr_matrix <- cor(data)
ggcorrplot(corr_matrix, lab = TRUE)


#########################################

#DECISION TREE
#DECISION TREE INTERPRETATION ON THE WHOLE DATASET

tree <- rpart(loan_status~., data = data,
              method  = "class")

predictions <- predict(tree, newdata = data, type = "class")
prp(tree)

conf_matrix <- table(data$loan_status, predictions)

print(conf_matrix)

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

precision <- diag(conf_matrix) / rowSums(conf_matrix)

recall <- diag(conf_matrix) / colSums(conf_matrix)

f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Accuracy:", accuracy, "\n")
cat("Precision:", mean(precision, na.rm = TRUE), "\n")
cat("Recall:", mean(recall, na.rm = TRUE), "\n")
cat("F1 Score:", mean(f1_score, na.rm = TRUE), "\n")

#plot the performance
performance_df <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score"),
  Value = c(accuracy, mean(precision, na.rm = TRUE), mean(recall, na.rm = TRUE), mean(f1_score, na.rm = TRUE)),
  Label = c(paste0(round(accuracy, 2)*100, "%"), paste0(round(mean(precision, na.rm = TRUE), 2)*100, "%"), 
            paste0(round(mean(recall, na.rm = TRUE), 2)*100, "%"), paste0(round(mean(f1_score, na.rm = TRUE), 2)*100, "%"))
)

ggplot(performance_df, aes(x = Metric, y = Value, label = Label)) +
  geom_bar(stat = "identity", fill = "orange2") +
  geom_text(size = 3, position = position_stack(vjust = 0.5)) +  
  labs(title = "Performance Metrics",
       y = "Value",
       x = "Metric") +
  theme_minimal()


#TRAINING THE DECISION TREE
set.seed(131)
#Partitioning Train and Test
intrain_tree<-createDataPartition(data$loan_status,p=0.80,list=FALSE)
train_set<-data[intrain_tree,]
test_set<-data[-intrain_tree,]
testing_newvar_names<-test_set[,1:2]

tree_train <- rpart(loan_status~., data = train_set,
                    method  = "class")

predictions <- predict(tree_train, newdata = data, type = "class")
prp(tree_train)

#PRUNINING THE TREE

(b <- tree_train$cptable[which.min(tree_train$cptable[, "xerror"]), "CP"])

pruned_model <- prune(tree_train, cp = b)

prp(pruned_model)

#PREDICTIONS AND EVALUATION OF THE PRUNED MODEL
tree_pred_pruned<- predict(pruned_model,test_set,type="vector")

actual_values <- test_set$loan_status

#Confusion Matrix
conf_matrix <- table(actual_values, tree_pred_pruned)

print(conf_matrix)

conf_matrix_df <- as.data.frame.matrix(conf_matrix)

colnames(conf_matrix_df) <- c("Predicted 0", "Predicted 1")

conf_matrix_df$Actual <- rownames(conf_matrix_df)

conf_matrix_df_melted <- melt(conf_matrix_df, id.vars = "Actual")

ggplot(conf_matrix_df_melted, aes(x = Actual, y = variable, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = value), vjust = 1) +
  theme_minimal() +
  labs(x = "Actual", y = "Predicted", fill = "Count") +
  scale_fill_gradient(low = "antiquewhite1", high = "darkorange") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Compute algorithm metrics
tree_pred_pruned_char <- as.character(tree_pred_pruned)
actual_values_char <- as.character(actual_values)

conf_matrix <- table(actual_values_char, tree_pred_pruned_char)


TP <- sum(diag(conf_matrix))
TN <- sum(diag(conf_matrix)) - sum(rowSums(conf_matrix) - diag(conf_matrix))
FP <- sum(colSums(conf_matrix) - diag(conf_matrix))
FN <- sum(rowSums(conf_matrix) - diag(conf_matrix))

accuracy <- (TP + TN) / (TP + TN + FP + FN)
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * precision * recall / (precision + recall)

print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1 Score:", f1_score))

performance_df <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score"),
  Value = c(accuracy, mean(precision, na.rm = TRUE), mean(recall, na.rm = TRUE), mean(f1_score, na.rm = TRUE)),
  Label = c(paste0(round(accuracy, 2)*100, "%"), paste0(round(mean(precision, na.rm = TRUE), 2)*100, "%"), 
            paste0(round(mean(recall, na.rm = TRUE), 2)*100, "%"), paste0(round(mean(f1_score, na.rm = TRUE), 2)*100, "%"))
)

ggplot(performance_df, aes(x = Metric, y = Value, label = Label)) +
  geom_bar(stat = "identity", fill = "orange2") +
  geom_text(size = 3, position = position_stack(vjust = 0.5)) + 
  labs(title = "Performance Metrics",
       y = "Value",
       x = "Metric") +
  theme_minimal()

# Get predicted probabilities
predicted_probs <- predict(pruned_model, newdata = test_set, type = "prob")

#ROC curve
roc_curve <- roc(test_set$loan_status, predicted_probs[, "1"])
plot(roc_curve, col = "blue", main = "ROC Curve", lwd = 2)
auc <- auc(roc_curve)
auc

#############################
#RANDOM FOREST
set.seed(4543)

# Split data into training and testing sets
index <- sample(1:nrow(data), 0.8*nrow(data))
train_data <- data[index, ]
test_data <- data[-index, ]

# Train the random forest model
train_data$loan_status <- factor(train_data$loan_status)
test_data$loan_status <- factor(test_data$loan_status)

rf <- randomForest(formula = loan_status ~ ., data = train_data, ntree = 1000, importance = TRUE)

# Make predictions on the test set
predictions <- predict(rf, newdata = test_data)

#Compute the metrics of model
accuracy <- mean(predictions == test_data$loan_status)
cat("Accuracy:", accuracy, "\n")

precision <- sum(predictions[test_data$loan_status == "1"] == "1") / sum(predictions == "1")
cat("Precision:", precision, "\n")

recall <- sum(predictions[test_data$loan_status == "1"] == "1") / sum(test_data$loan_status == "1")
cat("Recall:", recall, "\n")

f1_score <- 2 * precision * recall / (precision + recall)
cat("F1 Score:", f1_score, "\n")

performance_df <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score"),
  Value = c(accuracy, mean(precision, na.rm = TRUE), mean(recall, na.rm = TRUE), mean(f1_score, na.rm = TRUE)),
  Label = c(paste0(round(accuracy, 2)*100, "%"), paste0(round(mean(precision, na.rm = TRUE), 2)*100, "%"), 
            paste0(round(mean(recall, na.rm = TRUE), 2)*100, "%"), paste0(round(mean(f1_score, na.rm = TRUE), 2)*100, "%"))
)

ggplot(performance_df, aes(x = Metric, y = Value, label = Label)) +
  geom_bar(stat = "identity", fill = "orange2") +
  geom_text(size = 3, position = position_stack(vjust = 0.5)) + 
  labs(title = "Performance Metrics",
       y = "Value",
       x = "Metric") +
  theme_minimal()

# Get predicted probabilities
predicted_probs <- predict(rf, newdata = test_data, type = "prob")

#ROC curve
roc_curve <- roc(test_data$loan_status, predicted_probs[, "1"])
plot(roc_curve, col = "blue", main = "ROC Curve", lwd = 2)
auc <- auc(roc_curve)
auc

predictions <- predict(rf, newdata = test_data)

#Compute the confusion Matrix
confusion_mtx <- confusionMatrix(predictions, test_data$loan_status)

print(confusion_mtx)

conf_matrix_df <- as.data.frame(as.table(confusion_mtx$table))

colnames(conf_matrix_df) <- c("Predicted", "Actual", "Count")

ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), vjust = 1) +
  theme_minimal() +
  labs(x = "Actual", y = "Predicted", fill = "Count") +
  scale_fill_gradient(low = "antiquewhite1", high = "darkorange") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#################################

# Fit logistic regression model whole dataset
model <- glm(loan_status ~ person_age + log(person_income) + person_home_ownership + 
               loan_intent + loan_amnt + loan_int_rate + loan_percent_income + 
               cb_person_cred_hist_length, data = data, family = binomial)

summary(model)
 
predictions <- predict(model, type = "response")

predictions_factor <- factor(predictions > 0.5, levels = c(FALSE, TRUE), labels = c("0", "1"))

data$loan_status <- factor(data$loan_status, levels = levels(predictions_factor))

#Compute Confusion Matrix
conf_matrix <- confusionMatrix(data = predictions_factor, reference = data$loan_status)

print(conf_matrix)

conf_matrix_df <- as.data.frame(as.table(conf_matrix$table))

colnames(conf_matrix_df) <- c("Predicted", "Actual", "Count")

ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), vjust = 1) +
  theme_minimal() +
  labs(x = "Actual", y = "Predicted", fill = "Count") +
  scale_fill_gradient(low = "antiquewhite1", high = "darkorange") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

predictions_numeric <- as.numeric(predictions_factor)

data$loan_status <- as.numeric(data$loan_status)

roc_curve <- roc(data$loan_status, predictions_numeric)
plot(roc_curve, main = "ROC Curve")

data$loan_status <- as.factor(ifelse(data$loan_status == 2, 1, 0))


#ROC curve
auc <- auc(roc_curve)
auc

#####################
#K-cross validation
ctrl <- trainControl(method = "cv",  # Use k-fold cross-validation
                     number = 10,  # Number of folds
                     summaryFunction = twoClassSummary,  # For binary classification
                     classProbs = TRUE,  
                     verboseIter = TRUE)  


levels(data$loan_status)

valid_levels <- make.names(levels(data$loan_status))

data$loan_status <- factor(data$loan_status, levels = levels(data$loan_status), labels = valid_levels)

#train your model
model <- train(loan_status ~ .,
               data = data,
               method = "glm",
               trControl = ctrl)

print(model)

print(model$results)

#########################################

#LOGISTIC MODEL TREAIN and TEST

# Set seed for reproducibility
set.seed(123)

#split Dataset in Train and Test
split <- sample.split(data$loan_status, SplitRatio = 0.8)
train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)

#Logistic Regression on Train no log
model <- glm(loan_status ~ person_age + person_income + person_home_ownership + 
               loan_intent + loan_amnt + loan_int_rate + loan_percent_income + 
               cb_person_cred_hist_length, data = train_data, family = binomial)
summary(model)

#Logistic Regression on Train no log(person_income)
model <- glm(loan_status ~ person_age + log(person_income) + person_home_ownership + 
               loan_intent + loan_amnt + loan_int_rate + loan_percent_income + 
               cb_person_cred_hist_length, data = train_data, family = binomial)
summary(model)

predictions <- predict(model,newdata = test_data, type = "response")

predictions_factor <- factor(predictions > 0.5, levels = c(FALSE, TRUE), labels = c("0", "1"))

test_data$loan_status <- factor(test_data$loan_status, levels = levels(predictions_factor))

#Compute confusion Matrix
conf_matrix <- confusionMatrix(data = predictions_factor, reference = test_data$loan_status)

print(conf_matrix)

conf_matrix_df <- as.data.frame(as.table(conf_matrix$table))

colnames(conf_matrix_df) <- c("Predicted", "Actual", "Count")

ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), vjust = 1) +
  theme_minimal() +
  labs(x = "Actual", y = "Predicted", fill = "Count") +
  scale_fill_gradient(low = "antiquewhite1", high = "darkorange") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Roc Curve
roc_curve <- roc(test_data$loan_status, predictions)
plot(roc_curve, main = "ROC Curve")

auc <- auc(roc_curve)
auc

###############
#Linear Discriminant Analysis

split <- sample.split(data$loan_status, SplitRatio = 0.8)
train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)

#Compute LDA model on Train
lda_model <- lda(loan_status ~ ., data = train_data)

lda_predictions <- predict(lda_model, newdata = test_data)

#Compute confusion Matrix
conf_matrix_lda <- table(lda_predictions$class, test_data$loan_status)

print(conf_matrix_lda)

conf_matrix_df <- as.data.frame.matrix(conf_matrix_lda)

colnames(conf_matrix_df) <- c("Predicted 0", "Predicted 1")

conf_matrix_df$Actual <- rownames(conf_matrix_df)

conf_matrix_df_melted <- melt(conf_matrix_df, id.vars = "Actual")

ggplot(conf_matrix_df_melted, aes(x = Actual, y = variable, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = value), vjust = 1) +
  theme_minimal() +
  labs(x = "Actual", y = "Predicted", fill = "Count") +
  scale_fill_gradient(low = "antiquewhite1", high = "darkorange") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

conf_matrix_lda <- table(lda_predictions$class, test_data$loan_status)

#Compute metrix of model
accuracy_lda <- sum(diag(conf_matrix_lda)) / sum(conf_matrix_lda)

precision_lda <- diag(conf_matrix_lda) / rowSums(conf_matrix_lda)

recall_lda <- diag(conf_matrix_lda) / colSums(conf_matrix_lda)

f1_score_lda <- 2 * (precision_lda * recall_lda) / (precision_lda + recall_lda)

cat("Accuracy (LDA):", accuracy_lda, "\n")
cat("Precision (LDA):", mean(precision_lda, na.rm = TRUE), "\n")
cat("Recall (LDA):", mean(recall_lda, na.rm = TRUE), "\n")
cat("F1 Score (LDA):", mean(f1_score_lda, na.rm = TRUE), "\n")

lda_probabilities <- as.vector(lda_predictions$posterior[, 2])
#Roc Curve
roc_curve_lda <- roc(test_data$loan_status, lda_probabilities)
plot(roc_curve_lda, col = "blue", main = "ROC Curve (LDA)", lwd = 2)

