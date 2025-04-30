# Income Classification Using Spark Machine Learning (SparkML)
A Predictive Analysis of Census Data

## By
**Akshara Kumari, Christian Abreu, Dipanjan Ghosh, Zayviana Singletary**

## Introduction
Income classification helps organizations understand the factors influencing income thresholds. This project uses Apache Spark's MLlib to classify individuals’ income levels based on demographic data from the UCI Adult Census dataset.

## Dataset
- Source: UCI Machine Learning Repository
- Size: 32,561 records, 15 attributes
- Target: Income (<=\$50K or >\$50K)

## Data Preparation
- Cleaned missing values (replacing '?' with null)
- Standardized string formatting
- Indexed and encoded categorical variables
- Assembled features into a modeling vector

## Exploratory Data Analysis (EDA)
Performed SQL and Python-based EDA:
- Marital Status vs Age Group
- Workclass by Age
- Median Age by Education & Occupation
- Income Distribution Visualization

## Models Trained
1. Logistic Regression (Baseline & Tuned)
2. Decision Tree
3. Random Forest
4. Gradient Boosted Trees (GBT)
5. Linear SVC
6. Neural Network (MLP)

## Results

| Model | Training Accuracy | Testing Accuracy | Notes |
|-------|-------------------|------------------|-------|
| Logistic Regression (Baseline) | 85.04% | 84.93% | Good generalization |
| Logistic Regression (Tuned) | 84.71% | 84.62% | Minor change |
| Decision Tree | 84.67% | 83.71% | Slight overfitting |
| Random Forest | 86.09% | 85.00% | Strong performance |
| GBT | 86.92% | 86.14% | Best overall model |
| Linear SVC | 83.90% | 83.33% | Acceptable |
| Neural Network | 79.79% | 79.00% | Underfit |

## Visualizations
- Accuracy comparison bar chart
- Error bars for model variance
- Scatter plot: actual vs predicted income

## Conclusion
Gradient Boosted Trees was the top model, showing the best generalization and accuracy. Future enhancements may include:
- More hyperparameter tuning
- Addressing class imbalance
- Using PCA or feature selection
- Cross-validation for reliability

## References
- UCI Machine Learning Repository – Adult Dataset: https://archive.ics.uci.edu/ml/datasets/Adult
- Apache Spark MLlib: https://spark.apache.org/mllib
