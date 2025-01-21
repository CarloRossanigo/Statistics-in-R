# 2. The adiposity index adipos is also known as BMI (Body Mass Index). 
# When BMI is greater than 25, the person is considered overweight. 
# Create a binary variable in the dataset indicating whether the person is overweight or not.

# Which other variables are most influential in predicting whether a person is overweight? 
# Are they the same ones identified in point 1, or not? 
# Which classification method works best for this prediction?

# Load the required libraries
library(caret)
library(pROC) # ROC Curve
require('faraway')
library(nortest) # Normality test 
library(MASS)
require(tidyverse)
require(ggpubr)

help(fat)
data <- fat
str(data)
# Convert height from inches to centimeters
data$height <- round(data$height * 2.54, 2)  # Height in cm

# Convert weight from pounds to kilograms
data$weight <- round(data$weight * 0.453592, 2)  # Weight in kg

# Recalculate adiposity index: Weight / Height^2
data$adipos <- round(data$weight * 10000 / (data$height * data$height), 2)  # Height is in cm, not meters

# Add descriptions
attr(data$height, "label") <- "Height in centimeters"
attr(data$weight, "label") <- "Weight in kilograms"

# Remove the outlier (adipos = 165.62)
data <- data[data$adipos != 165.62, ]
# Removing the collinear variables (Height, Weight, Siri, Brozek, and Density) from the dataset

data <- data[, !(names(data) %in% c("height", "weight", "density", "siri", "brozek",'free'))]

# Create a binary variable for overweight based on adipos 
data$overweight <- ifelse(data$adipos > 25, 1, 0)  # Overweight = 1; Not Overweight = 0

# 1. Logistic Regression - Fit a logistic regression model with all variables
lr <- glm(overweight ~ .-adipos, 
          data = data, family = binomial)  # Fit logistic regression with all variables except 'adipos'
# Warning Message: fitted probabilities numerically 0 or 1 occurred 
# 2. Display the summary of the full model
summary(lr)

#Chest,Abdom are the most significant variables. 
# From the Null deviance -residual deviance good adaptability
# 3. Visualize the relationship between "abdom" and the dependent variable "overweight"
ggplot(data, aes(abdom, overweight)) +
  geom_point() + 
  geom_smooth(method = 'glm', method.args = list(family = "binomial"), se = TRUE)

# 4. Model Selection using Stepwise Method (both directions)
model_full <- glm(overweight ~ .-adipos, 
                  family = binomial, data = data)  # Full model

# Stepwise selection (both forward and backward) to improve the model
model_step <- step(model_full, direction = "both")
summary(model_step)
# Similar result: smaller AIC but higher residual deviance
#Significant variables: Chest Abdom Thigh

# 5. Split the dataset into training and test sets (70% for training, 30% for test)
set.seed(132)  # Initialize random number generator for reproducibility
prop_train <- 0.7  # Proportion of data to use for the training set
ndat <- length(data$overweight)  # Total number of observations
n_train <- round(ndat * prop_train)  # Calculate the number of training samples
index_train <- sample(c(1:ndat), size = n_train)  # Randomly sample indices for training set
train.data <- data[index_train, ]  # Training dataset
test.data <- data[-index_train, ]  # Test dataset

# 6. Train the logistic regression model on the training data (full model)
logitM <- glm(overweight~ .-adipos, 
              data = train.data, family = binomial)  # Fit model on training set
summary(logitM)

# Better AIC and residual deviance

# 7. Make predictions on the training set to evaluate accuracy (full model)
probabilities_train <- predict(logitM, train.data, type = "response")  # Compute probabilities
predicted_classes_train <- ifelse(probabilities_train > 0.5, 1, 0)  # Classify as 1 if probability > 0.5
train_accuracy <- mean(predicted_classes_train == train.data$overweight)  # Calculate accuracy
print(paste("Training set accuracy (full model): ", train_accuracy))
# 0.90
# 8. Make predictions on the test set to evaluate accuracy (full model)
probabilities_test <- predict(logitM, test.data, type = "response")  
predicted_classes_test <- ifelse(probabilities_test > 0.5, 1, 0)  
test_accuracy <- mean(predicted_classes_test == test.data$overweight) 
print(paste("Test set accuracy (full model): ", test_accuracy))
# 0.92
# Good accuracy on training and test set: 0.90 and 0.92. No overfitting

# 10. Calculate confusion matrix for the test set (full model)
conf_matrix_test <- confusionMatrix(as.factor(predicted_classes_test), as.factor(test.data$overweight))
print(conf_matrix_test)
# Good result of Kappa, accuracy, Sensitivity, Detection Rate
conf_matrix_train <- confusionMatrix(as.factor(predicted_classes_train), as.factor(train.data$overweight))
print(conf_matrix_train)
# Similar result, especially the Specificity
# 11. Calculate the ROC curve and AUC for the full model
roc_curve <- roc(test.data$overweight, probabilities_test)  # Compute ROC curve
plot(roc_curve)  # Plot ROC curve
auc_value <- auc(roc_curve)  # Calculate AUC
print(paste("AUC (full model): ", auc_value))

# The AUC is near 1 so the model fit well
# 12. Train the logistic regression model on the training data (stepwise model)
logit_model_step <- glm(overweight ~ .-adipos, 
                        data = train.data, family = binomial)  # Fit stepwise model on training set
summary(logit_model_step)
logit_model_step<-step(logit_model_step,direction='both')

# 13. Make predictions on the training set to evaluate accuracy (stepwise model)
probabilities_train_step <- predict(logit_model_step, train.data, type = "response")  
predicted_classes_train_step <- ifelse(probabilities_train_step > 0.5, 1, 0)  
train_accuracy_step <- mean(predicted_classes_train_step == train.data$overweight)  
print(paste("Training set accuracy (stepwise model): ", train_accuracy_step))
#Accuracy: 0.9
# 14. Make predictions on the test set to evaluate accuracy (stepwise model)
probabilities_test_step <- predict(logit_model_step, test.data, type = "response")  
predicted_classes_test_step <- ifelse(probabilities_test_step > 0.5, 1, 0)  
test_accuracy_step <- mean(predicted_classes_test_step == test.data$overweight)  
print(paste("Test set accuracy (stepwise model): ", test_accuracy_step))
#Accuracy on test : 0.94


# 16. Calculate confusion matrix for the test set (stepwise model)
conf_matrix_step <- confusionMatrix(as.factor(predicted_classes_test_step), as.factor(test.data$overweight))
print(conf_matrix_step)
# High Sensitivity, good Detection Rate
# 17. Calculate the ROC curve and AUC for the stepwise model
roc_curve_step <- roc(test.data$overweight, probabilities_test_step)  # Compute ROC curve for stepwise model
plot(roc_curve_step)  # Plot ROC curve
auc_value_step <- auc(roc_curve_step)  # Calculate AUC for stepwise model
print(paste("AUC (stepwise model): ", auc_value_step))

# AUC: 0.975-> good

