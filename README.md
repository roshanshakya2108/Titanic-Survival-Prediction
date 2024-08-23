# Titanic-Survival-Prediction

## Overview
This project aims to predict whether a passenger survived the Titanic disaster based on various characteristics using different machine learning models. The dataset used is the famous Titanic dataset, which provides information on the passengers aboard the Titanic, including attributes like age, sex, passenger class, fare, and more.

## Project Structure
- Data Preprocessing: Handling missing values, encoding categorical variables, and feature -engineering.
- Exploratory Data Analysis (EDA): Visualizing and understanding the data using various plots and statistical methods.
- Modeling: Building and evaluating multiple machine learning models to predict passenger survival.
- Prediction: Using the best-performing model to predict whether a given passenger survived.


## Dataset
The Titanic dataset is obtained using the seaborn library:

python
Copy code
import seaborn as sns

# Load the Titanic dataset
dataset = sns.load_dataset('titanic')


#### The dataset contains the following features:

- pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- sex: Gender of the passenger
- age: Age of the passenger
- sibsp: Number of siblings/spouses aboard
- parch: Number of parents/children aboard
- fare: Fare paid for the ticket
- embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- And other features related to the passenger's details and survival status.

## Data Preprocessing
- Handling Missing Values: Filled missing values in the age, embarked, and embark_town columns using the mean and mode.
- Encoding Categorical Variables: Converted categorical variables like sex, embarked, and who into numerical formats for model compatibility.
- Feature Engineering: Removed unnecessary columns and created one-hot encodings for categorical features.


## Exploratory Data Analysis (EDA)
Conducted various visualizations to understand the distribution of features and the relationship between different attributes and survival rates. Some visualizations included:

Distribution of passenger ages
Heatmap of missing values
Pair plots to understand feature relationships


## Modeling
Several machine learning models were trained and evaluated to predict passenger survival:

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost
- Linear Regression


## Model Evaluation
The performance of each model was evaluated using accuracy as the metric. The models were tested on a test set to check their predictive capabilities.
- Logistic Regression achieved the best accuracy of 83.33% compared to other algorithms.


## Prediction
The best-performing model, Logistic Regression, was used to predict the survival of a passenger given a set of features. The model predicted that the sample passenger survived.


## Conclusion
This project demonstrates the application of machine learning techniques to predict survival in a real-world dataset. The best model, Logistic Regression, suggests that it effectively captures the factors influencing survival in the Titanic disaster.


## Acknowledgements
- Seaborn Library
- Scikit-Learn Library
- XGBoost Library
