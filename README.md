# Statistical Analysis of BMI and Overweight Predictors

## Project Overview

This project focuses on the analysis of the **fat** dataset with the objective of identifying key variables that predict Body Mass Index (BMI) and the likelihood of being overweight. Specifically, the analysis explores variables that are not collinear with BMI (such as **adipos**) and evaluates their significance in predicting BMI and the condition of being overweight.

## Project Files

The repository contains four R scripts, each corresponding to different stages of the analysis:

1. **Preliminary.R**  
   This script performs initial data inspection and preparation, including data cleaning and exploration. It sets up the dataset for further analysis.

2. **LinearRegression.R**  
   This script applies linear regression to identify continuous predictors of BMI. It provides insights into how different variables relate to BMI using statistical models.

3. **LogisticRegression.R**  
   This script performs logistic regression to model the likelihood of being overweight based on various predictors. It focuses on a binary outcome (overweight vs. not overweight).

4. **ClassificationTrees.R**  
   This script uses classification trees to explore the relationship between predictors and the classification of individuals as overweight or not. It provides a visual representation of the decision-making process.

## Dataset

The dataset used in this project is the **fat** dataset, which can be loaded in R using the following commands:

```r
require('faraway')
help(fat)
data <- fat
```
For a clearer understanding of the dataset, please refer to the help documentation (help(fat)) and  https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset for detailed descriptions of the variables.

## Running the Scripts

To run the R scripts, you need to have R installed on your machine. You can download and install R from [https://cran.r-project.org](https://cran.r-project.org). Additionally, I suggest to install RStudio, a popular IDE for R, from [https://rstudio.com](https://rstudio.com) for a more user-friendly experience.


