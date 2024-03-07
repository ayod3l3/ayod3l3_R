setwd("/Users/ayodeleogundele/Desktop/Uni_Sunderland/CETM72/CETM72_R")

set.seed(123)

#Load Libraries
library(readr)
library(dplyr)
library(caret)
library(randomForest)
library(pROC)
library(corrplot)
library(rpart)
library(rpart.plot)
library(DMwR2)
library(RColorBrewer)
library(rattle)
library(rpart.plot)

#Load dataset from working directory
bcdata <- read_csv("wisconsin.csv")

#Data inspection 
head(bcdata)
summary(bcdata)
str(bcdata)


#Identify columns with missing values
missing_values <- colSums(is.na(bcdata))
columns_with_missingV <- names(missing_values[missing_values > 0])

columns_with_missingV

#Handle missing values with mean
bcdata$Bare.nuclei[is.na(bcdata$Bare.nuclei)] <- mean(bcdata$Bare.nuclei, 
                                                      na.rm = TRUE)

#Check that there are no more missing values
sum_missing <- sum(colSums(is.na(bcdata)))
sum_missing

#Convert target variable to factors
bcdata$Class <- factor(bcdata$Class)

#Plotting class distribution using a bar plot
barplot(table(bcdata$Class), 
        main = "Class Distribution in bcdata",
        xlab = "Class",
        ylab = "Frequency")


#Splitting the data into 70% train and 30% test sets
bc_index <- sample(nrow(bcdata), 0.7 * nrow(bcdata))
bctrain <- bcdata[bc_index, ]
bctest <- bcdata[-bc_index, ]

#View class distribution for train and test subset
table(bctrain$Class)
table(bctest$Class)

#Handle imbalanced in bctrain using undersampling 
#Count the class distribution in bctrain
class_distribution <- table(bctrain$Class)

#Identify class with the majority and minority class
majority_class <- names(which.max(class_distribution))
minority_class <- names(which.min(class_distribution))

#Identify the indices of majority class instances
majority_indices <- which(bctrain$Class == majority_class)

#Random sample from the majority class to match minority class instances and undersampled majority class
undersampled_majority_indices <- sample(majority_indices, sum(bctrain$Class == minority_class))
undersampled_indices <- c(which(bctrain$Class == minority_class), undersampled_majority_indices)

# Create an undersampled dataset
bctrain_undersampled <- bctrain[undersampled_indices, ]
table(bctrain_undersampled$Class)

#Perform classification using Random Forest
rf_model <- train(Class ~ ., data = bctrain_undersampled, method = "rf", 
                  trControl = trainControl(method = "cv", 
                                           number = 5, verboseIter = TRUE))

#Use the trained model to predict on the test set
rf_predictions <- predict(rf_model, newdata = bctest)

#Assess model performance on the test set
rf_confusion <- confusionMatrix(rf_predictions, bctest$Class)
rf_confusion

#Extract other metrics
rf_precision <- rf_confusion$byClass['Precision']
rf_recall <- rf_confusion$byClass['Recall']
rf_F1 <- rf_confusion$byClass['F1']
rf_precision 
rf_recall 
rf_F1 


# Perform classification using Decision Tree
tree_model <- rpart(Class ~ ., data = bctrain_undersampled, method = "class")

# Make predictions on the test set
tree_predictions <- predict(tree_model, newdata = bctest, type = "class")

#Assess model performance and extract metrics
tree_confusion <- confusionMatrix(tree_predictions, bctest$Class)
tree_confusion

# Extract other metrics
tree_precision <- tree_confusion$byClass['Precision']
tree_recall <- tree_confusion$byClass['Recall']
tree_F1 <- tree_confusion$byClass['F1']
tree_precision 
tree_recall 
tree_F1 


# Plot the decision tree
fancyRpartPlot(tree_model)
table(tree_predictions, bctest$Class)


#Compare the performance of Random Forest predictions VS Decision Tree predictions 
#Calculate ROC for Random Forest
rf_roc <- roc(ifelse(rf_predictions == "benign", 1, 0), 
              ifelse(bctest$Class == "benign", 1, 0))

#Calculate ROC for Decision Tree 
tree_roc <- roc(ifelse(tree_predictions == "benign", 1, 0), 
                ifelse(bctest$Class == "benign", 1, 0))


#Plotting ROC curves for both models
plot(rf_roc, col = "blue", legacy.axes = TRUE, print.auc = TRUE, main = "ROC Curves - Random Forest vs. Decision Tree")
lines(tree_roc, col = "red")

legend("bottomright", legend = c("Random Forest", "Decision Tree"),
       col = c("blue", "red"), lty = 1)