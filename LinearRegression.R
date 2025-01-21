# 1. Which variables are most influential in predicting the adiposity index (adipos)? 
# Consider only the variables that are not directly used in the calculation of adipos, and therefore not collinear with adipos.
# Load the required libraries
library(lmtest)
require('faraway')
library(nortest) # Normality test 
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

# Build the regression model
model <- lm(adipos ~ ., data = data)
summary(model)

# Breusch-Pagan test for heteroscedasticity
bp_test <- bptest(model) #H0: no heteroscedasticity

# Display the test results
bp_test #Reject H0: heteroscedasticity


# Calculating the residuals
residuals <- resid(model)

# Calculating the predicted values
predicted_values <- predict(model)

# Residuals vs Predicted Values plot
plot(predicted_values, residuals, 
     main = "Residuals vs Predicted Values", 
     xlab = "Predicted Values", 
     ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)  # Horizontal line at zero

# Shapiro-Wilk test for normality of residuals
shapiro.test(model$residuals)

# Anderson-Darling test for normality of residuals
ad.test(model$residuals)

# Do not reject the hypothesis of residuals being normal
# Q-Q plot of residuals
qqnorm(model$residuals)
qqline(model$residuals, col = 'red')

# Condition number of the model matrix
kappa(model.matrix(model)) 
kappa(t(model.matrix(model)) %*% model.matrix(model))
#High condition number
# Variance Inflation Factor (VIF) to check for multicollinearity
vif(model)  # Abdom and Hip are correlated?
cor.test(data$hip, data$abdom)
plot(data$hip, data$abdom, main = "Scatter Plot between Hip and Abdom", xlab = "Hip", ylab = "Abdom", pch = 19)
#High positive correlation between hip and abdom (this makes sense)-> possible multicollinearity
# Significant variables
significant_vars <- model$coefficients[summary(model)$coefficients[, "Pr(>|t|)"] < 0.05]
significant_vars
#Chest  Abdom Hip Thigh  Knee

# One could try to remove Hip/ Abdom/ Other Variables from the model: in line 32
# model <- lm(adipos ~ .-abdom.hip, data = data)
# But the result are more or less the same: high condition number (lower without abdom but pretty high)
# and sometimes losing normality

#Improve the model with Box-Cox
bc<-boxcox(model)
lambda_opt <- bc$x[which.max(bc$y)]
lambda_opt #0.02020202: near 0, I try Log and this value

lambdamodel<-lm( (((adipos^lambda_opt) -1)/lambda_opt) ~ ., data = data)
summary(lambdamodel)

# Breusch-Pagan test for heteroscedasticity
bp_test_lambda <- bptest(lambdamodel)

bp_test_lambda # Don't reject H0: can't reject the hypotesys of homoscedasticity

#Normality
shapiro.test(lambdamodel$residuals)
ad.test(lambdamodel$residuals)
# Do not reject the hypothesis of residuals being normal: The tests have got good p-value
qqnorm(lambdamodel$residuals)
qqline(lambdamodel$residuals,col='red')

kappa(model.matrix(lambdamodel))
kappa(t(model.matrix(lambdamodel))%*%model.matrix(lambdamodel))
#Again, high condition number and multicollinearity

vif(lambdamodel) #Abdom and hip are high positive correlated
vars_significant <- lambdamodel$coefficients[summary(lambdamodel)$coefficients[, "Pr(>|t|)"] < 0.05]
vars_significant
#Chest Abdom Thigh Knee Forearm
# Again, removing hip / Abdom / other variables won't improve the model: stil high condition number and losing homoscedasticity

# Logarithmic transformation
logmodel <- lm(log(adipos) ~ ., data = data)
summary(logmodel)

# Breusch-Pagan test for heteroscedasticity on the logarithmic model
bp_test_log <- bptest(logmodel)


bp_test_log
# Don't reject the hypotesys of homoscedasticity

# Shapiro-Wilk test for normality of residuals
shapiro.test(logmodel$residuals)

# Anderson-Darling test for normality of residuals
ad.test(logmodel$residuals)
# Don't reject the hypotesys of normality
# Q-Q plot of residuals
qqnorm(logmodel$residuals)
qqline(logmodel$residuals, col = 'red')

# Residuals are normal

# Condition number of the model matrix for the logarithmic model
kappa(model.matrix(logmodel))
kappa(t(model.matrix(logmodel)) %*% model.matrix(logmodel))

# Variance Inflation Factor (VIF) for multicollinearity check
vif(logmodel)

#Same problem: High condition number, multicollinearity and Hip Abdom correlated.

# Significant variables in the logarithmic model
significant_vars_log <- logmodel$coefficients[summary(logmodel)$coefficients[, "Pr(>|t|)"] < 0.05]
significant_vars_log

#Chest Abdom Thigh Knee Forearm
#Similar result to lambda regression model

# Perform PCA on the independent variables:
pca <- prcomp(data[, c("neck", "chest", "hip", "thigh", "knee", "ankle", "biceps", "forearm", "wrist", "age")], scale = TRUE)

# Display the variance explained by the principal components
summary(pca)

# Use the first principal components in the regression model
pc_data <- data.frame(pca$x[, 1:5])  # Select the first 5 principal components
model_pca <- lm((data$adipos) ~ ., data = pc_data)
summary(model_pca)
bp_test_pca <- bptest(model_pca)
bp_test_pca # Heteroscedasticity
# Low  Adjusted R-squared, no overfitting
shapiro.test(model_pca$residuals) 
ad.test(model_pca$residuals)
# Reject H0: residuals aren't normal
qqnorm(model_pca$residuals)
qqline(model_pca$residuals, col = 'red')
kappa(model.matrix(model_pca))
#Low condition number->this was the goal
kappa(t(model.matrix(model_pca))%*%model.matrix(model_pca))
# No collinearity->this was the goal
vif(model_pca)  # Check how each combination of variables influences the dependent variable
pca$rotation
summary(pca)

# PC1 explains 63% of the variance, dominated by variables like hip, thigh, chest, knee, neck and biceps.
# PC2 is mainly driven by age (0.8851), and the first 5 components explain 90.75% of the total variance.
# The most important varibale between PC1 and PC5 are: Chest, Neck, Abdom, Biceps
