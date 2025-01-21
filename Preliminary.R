# Analyze the dataset using linear regressions, logistic regressions, and classification trees, 
# particularly addressing the following questions:

# 1. Which variables are most influential in predicting the adiposity index (adipos)? 
# Consider only the variables that are not directly used in the calculation of adipos, and therefore not collinear with adipos.

# 2. The adiposity index adipos is also known as BMI (Body Mass Index). 
# When BMI is greater than 25, the person is considered overweight. 
# Create a binary variable in the dataset indicating whether the person is overweight or not.

# Which other variables are most influential in predicting whether a person is overweight? 
# Are they the same ones identified in point 1, or not? 
# Which classification method works best for this prediction?

# Install the necessary packages
install.packages(c("rpart.plot","randomForest","caret","ggpubr","nortest","lmtest","ResourceSelection"))
# Load the required libraries

require('faraway')

library(nortest)  # Normality test
library(ggpubr)

help(fat)
data <- fat
str(data)
pairs(data)

# Preliminary analysis: Are there any missing values?
summary(data)
colSums(is.na(data))  # All 0 -> no missing data
# Convert height from inches to centimeters

data$height <- round(data$height * 2.54, 2)  # Height in cm

# Convert weight from pounds to kilograms
data$weight <- round(data$weight * 0.453592, 2)  # Weight in kg

# Recalculate adiposity index: Weight / Height^2
data$adipos <- round(data$weight * 10000 / (data$height * data$height), 2)  # Height is in cm, not meters

# Add descriptions
attr(data$height, "label") <- "Height in centimeters"
attr(data$weight, "label") <- "Weight in kilograms"

# Descriptive Analysis
# adipos: Does the variable follow a normal distribution?
summary(data$adipos)
# There is one data point, the maximum, that seems excessive compared to the expected values:
# From the histogram, I see the presence of an outlier: I will investigate it
hist(data$adipos, main = "Adiposity Index Distribution", xlab = "Adiposity Index", col = "lightblue", border = "black", breaks = 100)

# The qqplot indeed shows the presence of an outlier:
qqnorm(data$adipos, 
       main = "Adiposity Index", 
       ylab = "Observed Quantiles", 
       xlab = "Theoretical Quantiles", 
       pch = 19, col = "blue")

qqline(data$adipos, col = "red", lwd = 2)

# Using boxplot to confirm the outlier
boxplot(data$adipos, 
        main = "Adiposity Index (BMI) Boxplot", 
        ylab = "Adiposity Index", 
        col = "lightblue", 
        border = "black")

# Calculate Z-scores to possibly remove the outlier
z_scores <- scale(data$adipos)
# Display outliers (values with Z > 3 or Z < -3)
outliers <- data$adipos[abs(z_scores) > 3]
print(outliers)

# Remove the outlier (adipos = 165.62)
data <- data[data$adipos != 165.62, ]

# Display BoxPlot and QQplot again to visually assess the difference:
boxplot(data$adipos, 
        main = "Adiposity Index (BMI) Boxplot", 
        ylab = "Adiposity Index", 
        col = "lightblue", 
        border = "black")

qqnorm(data$adipos, 
       main = "Adiposity Index", 
       ylab = "Observed Quantiles", 
       xlab = "Theoretical Quantiles", 
       pch = 19, col = "blue")
qqline(data$adipos, col = "red", lwd = 2)

# The distribution seems similar to a normal distribution in its central part:
# Perform two tests (Shapiro-Wilk and Anderson-Darling) to accept or reject the hypothesis of normality

shapiro.test(data$adipos) # Reject the hypothesis of normality for adipos

ad.test(data$adipos)  # Reject the hypothesis of normality for adipos

# Curiosity:
# 1) What is the age range of our sample? Is the age distribution normal? Is there any correlation between age and obesity?
summary(data$age)
hist(data$age, main = "Age Distribution", xlab = "Age", col = "lightgreen", border = "black", breaks = 20)
shapiro.test(data$age)  # From the results, reject H0: age is not normally distributed

# Descriptive statistics for some variables

summary(data$height)
summary(data$weight)

# Look for initial correlations between adipos and body measurements 

plot(data$density, data$adipos, main = "Scatter Plot between Density and Adiposity", xlab = "Density", ylab = "Adiposity Index", pch = 19, col = "red")

plot(data$chest, data$adipos, main = "Scatter Plot between Chest and Adiposity", xlab = "Chest", ylab = "Adiposity Index", pch = 19)

plot(data$abdom, data$adipos, main = "Scatter Plot between Abdomen and Adiposity", xlab = "Abdomen", ylab = "Adiposity Index", pch = 19)

plot(data$hip, data$adipos, main = "Scatter Plot between Hip and Adiposity", xlab = "Hip", ylab = "Adiposity Index", pch = 19)

plot(data$biceps, data$adipos, main = "Scatter Plot between Biceps and Adiposity", xlab = "Biceps", ylab = "Adiposity Index", pch = 19)

plot(data$free, data$adipos, main = "Scatter Plot between Free and Adiposity", xlab = "Free", ylab = "Adiposity Index", pch = 19)

plot(data$ankle, data$adipos, main = "Scatter Plot between Ankle and Adiposity", xlab = "Ankle", ylab = "Adiposity Index", pch = 19)

plot(data$brozek, data$siri, main = "Scatter Plot between Brozek and Siri", xlab = "Brozek", ylab = "Siri", pch = 19)

plot(data$density, data$siri, main = "Scatter Plot between Density and Siri", xlab = "Density", ylab = "Siri", pch = 19)

cor_matrix <- cor(data)
cor_matrix
# Siri and Brozek are correlated with density, which is correlated with adipos and free: remove them from the model
# Correlation test between some variables and adiposity, H0 = no correlation. 
cor.test(data$age, data$adipos)  # No evident correlation between age and adipos
cor.test(data$abdom, data$adipos) 
cor.test(data$density, data$adipos)
cor.test(data$chest, data$adipos)
cor.test(data$hip, data$adipos)
cor.test(data$free, data$weight)
# Correlation between abdom and adipos, chets and adipos, hip and adipos, negative correlation between density and adipos
# Correlation between free and weight: weight is collinear with free
# As requested, I remove from the model the variables that are collinear with Adipos: Height and Weight.
# Additionally, Siri and Brozek are highly correlated with density, which appears in their equations.
# Density also appears in the Adipos equation: I remove these three variables.
# Weight appears in the Free equation, so it will also be removed.
# https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset/discussion/303555
# Removing the collinear variables (Height, Weight, Siri, Brozek, and Density) from the dataset

# Create a binary variable for overweight based on adipos 
data$overweight <- ifelse(data$adipos > 25, 1, 0)  # Overweight = 1; Not Overweight = 0

#Box-Plot
boxplot(chest ~ overweight, data = data, main = "Chest Circumference Distribution for Overweight and Non-Overweight", ylab = "Chest", xlab = "Overweight", col = c("lightblue", "lightgreen"))

# There are some outliers, but most of the overweight chest's circumference is in 100-110, with the median in 105 approx

boxplot(abdom ~ overweight, data = data, main = "Abdomen Circumference Distribution for Overweight and Non-Overweight", ylab = "Abdom", xlab = "Overweight", col = c("lightblue", "lightgreen"))

# Curiosity: the median overweight has approx 10-15 cm more of abdom's circumference 
# The maximum abdominal circumference among non-overweight individuals (a tall or fit man?) is approximately the median of those in the overweight group

boxplot(hip ~ overweight, data = data, main = "Body Density Distribution for Overweight and Non-Overweight", ylab = "Hip", xlab = "Overweight", col = c("lightblue", "lightgreen"))
# Overweight individuals have a larger range of hip circumferences, with two outliers
# Interesting to note that the minimum value is smaller than the median of the non-overweight group
# while similar to the abdomen  the maximum value of the non-overweight group is approximately at the median of the overweight group


boxplot(age ~ overweight, data = data, main = "Age Distribution for Overweight and Non-Overweight", ylab = "Age", xlab = "Overweight", col = c("lightblue", "lightgreen"))

# The average age of overweight individuals is around 45 years, slightly higher than that of non-overweight individuals
# The outlier for the non-overweight group is 81
# while it can be observed that both the minimum and maximum values, as well as the first and third quartiles, are slightly shifted higher with age for the overweight group

