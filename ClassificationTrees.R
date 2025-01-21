# 2. The adiposity index adipos is also known as BMI (Body Mass Index). 
# When BMI is greater than 25, the person is considered overweight. 
# Create a binary variable in the dataset indicating whether the person is overweight or not.

# Which other variables are most influential in predicting whether a person is overweight? 
# Are they the same ones identified in point 1, or not? 
# Which classification method works best for this prediction?

# Load the required libraries
library(ResourceSelection)
library(caret)
require('faraway')
library(MASS)
require(tidyverse)

library(rpart)       # For classification trees
library(rpart.plot)  # For plotting decision trees
library(randomForest) # For random forests

data <- fat
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

# Classification Trees with Entropy (Information Gain) 
tree_model_entropy <- rpart(overweight ~ .-adipos, 
                            data = train.data, 
                            method = "class", 
                            parms = list(split = "information"))

# Plot the tree
plot(tree_model_entropy)
text(tree_model_entropy)
rpart.plot(tree_model_entropy, type = 1, fallen.leaves = FALSE)
# Chest, Abdom, Biceps
# Print detailed information about the tree
print(tree_model_entropy)
printcp(tree_model_entropy)  # Print complexity parameters table: CP=0.01,Number of split=3 better paramaters of error 
plotcp(tree_model_entropy)   # Plot complexity parameter

# Classification Trees with Gini Index
tree_model_gini <- rpart(overweight ~ . -adipos, 
                         data = train.data, 
                         method = "class", 
                         parms = list(split = "gini"))

# Plot the tree
rpart.plot(tree_model_gini, type = 1, fallen.leaves = FALSE)
# Chest, Abdom, Biceps
# Print detailed information about the tree
printcp(tree_model_gini)  # Print complexity parameters table
plotcp(tree_model_gini)   # Plot complexity parameter
# Gini index, number of split=3 higher xerror and so xstd
# Model Evaluation on Training Data
# Predictions on training set for entropy model
predictions_entropy_train <- predict(tree_model_entropy, newdata = train.data, type = "class")
accuracy_entropy_train <- mean(predictions_entropy_train == train.data$overweight)
cat("Training Accuracy (Entropy): ", accuracy_entropy_train)
# 0.9375
# Predictions on training set for Gini model
predictions_gini_train <- predict(tree_model_gini, newdata = train.data, type = "class")
accuracy_gini_train <- mean(predictions_gini_train == train.data$overweight)
cat("Training Accuracy (Gini): ", accuracy_gini_train )
# 0.9375
# Model Evaluation on Test Data 
# Predictions on test set for entropy model
predictions_entropy_test <- predict(tree_model_entropy, newdata = test.data, type = "class")
accuracy_entropy_test <- mean(predictions_entropy_test == test.data$overweight)
cat("Test Accuracy (Entropy): ", accuracy_entropy_test)
# 0.93->similar to training set
# Predictions on test set for Gini model
predictions_gini_test <- predict(tree_model_gini, newdata = test.data, type = "class")
accuracy_gini_test <- mean(predictions_gini_test == test.data$overweight)
cat("Test Accuracy (Gini): ", accuracy_gini_test )
# 0.92-> lesser than Entropy

#  Random Forest Models 
# Ensure the response variable is a factor for classification
train.data$overweight <- factor(train.data$overweight, levels = c(0, 1))

# Train random forest models: Gini Index

random_forest <- randomForest(overweight ~ .-adipos, 
                              data = train.data, 
                              importance = TRUE,  # Show feature importance
                              mtry = 3) # Number of variables for the decision  

#  test: mtry=3, mtry=5, mtry=7.
#Better result wiht mtry=3: class error lower (especially non overweight), OOB lower
# mtry=5 lower class error for overweight
print(random_forest)

# Print feature importance
print(importance(random_forest, type = 1))
# Importance: Chest, Abdom, Thigh, Biceps
# Model Evaluation on Training Data (Random Forests) 

# Predictions on training set for Random Forest Gini
predictions_rf_gini <- predict(random_forest, newdata = train.data, type = "class")
accuracy_rf_gini <- mean(predictions_rf_gini == train.data$overweight)
cat("Training Accuracy (Random Forest - Gini): ", accuracy_rf_gini)
# 1-> Overfitting

# Predictions on test set for Random Forest Gini
predictions_rf_gini_test <- predict(random_forest, newdata = test.data, type = "class")
accuracy_rf_gini_test <- mean(predictions_rf_gini_test == test.data$overweight)
cat("Test Accuracy (Random Forest - Gini): ", accuracy_rf_gini_test, "\n")
# 0.95
# Tune random forest model to prevent overfitting (limiting variables and tree size)
random_forest_tuned <- randomForest(overweight ~ . -adipos, 
                                    data = train.data, 
                                    mtry = 3,    # Number of variables to consider for each split
                                    ntree = 500,  # Number of trees in the forest
                                    nodesize = 10, # Min number of samples required to split a node
                                    maxnodes = 50) # Limit the maximum number of terminal nodes

# Print feature importance
print(importance(random_forest_tuned))
# Chest, Abdom, Hip, Biceps, Thigh
print(random_forest_tuned)
# Better OOB and class Error than mtry=5,7 with overfitting

# Evaluate tuned Random Forest on training set
predictions_rf_tuned_train <- predict(random_forest_tuned, newdata = train.data, type = "class")
accuracy_rf_tuned_train <- mean(predictions_rf_tuned_train == train.data$overweight)
cat("Training Accuracy (Tuned Random Forest): ", accuracy_rf_tuned_train)
#  0.97
# Evaluate tuned Random Forest on test set
predictions_rf_tuned_test <- predict(random_forest_tuned, newdata = test.data, type = "class")
accuracy_rf_tuned_test <- mean(predictions_rf_tuned_test == test.data$overweight)
cat("Test Accuracy (Tuned Random Forest): ", accuracy_rf_tuned_test)
# 0.96
# Pruning Classification Trees
# Create complex classification trees (Entropy and Gini) for pruning demonstration: try to obtain the same tree
complexTree_entropy <- rpart(overweight ~ .-adipos, 
                             data=train.data, 
                             method = 'class', 
                             parms = list(split = "information"), 
                             control = rpart.control(xval = 10, minbucket = 2, cp = 0.00))

# Plot and prune the entropy tree
printcp(complexTree_entropy)
plotcp(complexTree_entropy)
# cp=0.0176-> 3 split and less cross validation error / relative error
rpart.plot(complexTree_entropy, type = 1, fallen.leaves = FALSE)
complexTree_entropy <- prune(complexTree_entropy, cp = 0.018)
# Visualize the pruned entropy tree
rpart.plot(complexTree_entropy, type = 1, fallen.leaves = FALSE, main = "Pruned Entropy Tree")

# Create complex classification tree using Gini index and prune
complexTree_gini <- rpart(overweight ~ .-adipos, 
                          data=train.data, 
                          method = 'class', 
                          parms = list(split = "gini"), 
                          control = rpart.control(xval = 10, minbucket = 2, cp = 0.00))

# Plot and prune the Gini tree
printcp(complexTree_gini)
plotcp(complexTree_gini)
rpart.plot(complexTree_gini, type = 1, fallen.leaves = FALSE)
# Cp=0.018
complexTree_gini <- prune(complexTree_gini, cp = 0.018)
# Visualize the pruned Gini tree
rpart.plot(complexTree_gini, type = 1, fallen.leaves = FALSE, main = "Pruned Gini Tree")

# Evaluate the pruned trees on the training set
predictions_entropy_train <- predict(complexTree_entropy, newdata = train.data, type = "class")
accuracy_entropy_train <- mean(predictions_entropy_train == train.data$overweight)
cat("Training Accuracy (Pruned Entropy): ", accuracy_entropy_train)
# 0.9375
predictions_gini_train <- predict(complexTree_gini, newdata = train.data, type = "class")
accuracy_gini_train <- mean(predictions_gini_train == train.data$overweight)
cat("Training Accuracy (Pruned Gini): ", accuracy_gini_train)
# 0.9375
# Evaluate the pruned trees on the test set
predictions_entropy_test <- predict(complexTree_entropy, newdata = test.data, type = "class")
accuracy_entropy_test <- mean(predictions_entropy_test == test.data$overweight)
cat("Test Accuracy (Pruned Entropy): ", accuracy_entropy_test)
# 0.93
predictions_gini_test <- predict(complexTree_gini, newdata = test.data, type = "class")
accuracy_gini_test <- mean(predictions_gini_test == test.data$overweight)
cat("Test Accuracy (Pruned Gini): ", accuracy_gini_test)
# 0.92
# I obtain the same tree ( this was a demonstration, one could change the code: 
# For instance change the cp parameter or not pruning, change the mtry and other parameter
