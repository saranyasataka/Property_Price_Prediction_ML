# Property Price Prediction using Machine Learning

## 1. Project Overview

This project focuses on predicting property prices in different districts of California using Machine Learning techniques. The main objective is to build a regression model that can estimate house prices based on features such as income, number of rooms, population, housing age, and proximity to the ocean.

The project demonstrates the complete Machine Learning workflow including:

* Data loading
* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model training
* Prediction
* Model evaluation

The project was implemented using Python and commonly used Machine Learning libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

---

# 2. Problem Statement

House prices are influenced by multiple factors including location, income levels, population density, and housing characteristics. Predicting house prices manually can be difficult because many variables affect the final value.

The objective of this project is to develop a Machine Learning model capable of predicting median house values using district-level housing information.

The project uses Multiple Linear Regression to identify relationships between input features and house prices.

---

# 3. Objectives

The main objectives of this project are:

* To understand and analyze housing data
* To preprocess and clean the dataset
* To visualize relationships between features
* To train a regression model for price prediction
* To evaluate model performance using regression metrics
* To understand the overall Machine Learning workflow

---

# 4. Technologies Used

| Technology       | Purpose                         |
| ---------------- | ------------------------------- |
| Python           | Programming Language            |
| Pandas           | Data handling and preprocessing |
| NumPy            | Numerical operations            |
| Matplotlib       | Data visualization              |
| Seaborn          | Advanced visualizations         |
| Scikit-learn     | Machine Learning model building |
| Jupyter Notebook | Development environment         |

---

# 5. Dataset Description

The dataset contains housing-related information from California districts.

## Dataset Features

| Feature Name       | Description                   |
| ------------------ | ----------------------------- |
| longitude          | Geographic longitude          |
| latitude           | Geographic latitude           |
| housing_median_age | Median age of houses          |
| total_rooms        | Total number of rooms         |
| total_bedrooms     | Total number of bedrooms      |
| population         | Population in the district    |
| households         | Number of households          |
| median_income      | Median income of residents    |
| ocean_proximity    | Distance from ocean           |
| median_house_value | Target variable (house price) |

---

# 6. Project Workflow

The complete workflow of the project is shown below:

```text
Data Collection
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Encoding
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Prediction
        ↓
Model Evaluation
        ↓
Conclusion
```

---

# 7. Data Preprocessing

Data preprocessing is one of the most important phases in Machine Learning.

## 7.1 Handling Missing Values

The dataset contained missing values in the `total_bedrooms` column.

The missing values were handled using median imputation:

```python
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
```

Median was chosen because it is less affected by outliers.

---

## 7.2 Encoding Categorical Data

The `ocean_proximity` column contained categorical text values.

Machine Learning models cannot directly process text data, so One-Hot Encoding was applied.

```python
df = pd.get_dummies(df, columns=["ocean_proximity"])
```

This converted categorical values into numerical binary columns.

---

# 8. Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset and identify patterns.

## Histograms

Histograms were used to analyze the distribution of numerical features.

```python
df.hist(figsize=(15,10))
plt.show()
```

### Observations

* Most districts had moderate median income values
* House prices were not evenly distributed
* Some features showed skewed distributions

---

## Correlation Heatmap

A heatmap was used to understand relationships between variables.

```python
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
```

### Key Insight

`median_income` showed the strongest positive correlation with `median_house_value`.

This indicates that districts with higher income generally have higher property prices.

---

# 9. Feature and Target Selection

## Features (X)

All columns except `median_house_value` were used as input features.

```python
X = df.drop("median_house_value", axis=1)
```

## Target Variable (y)

`median_house_value` was selected as the target variable.

```python
y = df["median_house_value"]
```

---

# 10. Train-Test Split

The dataset was divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Purpose

* Training data is used to train the model
* Testing data is used to evaluate model performance

80% of the data was used for training and 20% for testing.

---

# 11. Model Development

## Linear Regression Model

Multiple Linear Regression was used for this project.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

The model learns relationships between housing features and house prices.

---

# 12. Prediction

Predictions were generated using test data.

```python
y_pred = model.predict(X_test)
```

The model predicted house prices for unseen data.

---

# 13. Model Evaluation

The model was evaluated using regression metrics.

## Evaluation Metrics Used

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

## Results

| Metric   | Value         |
| -------- | ------------- |
| MAE      | 50670.74      |
| MSE      | 4908476721.15 |
| RMSE     | 70060.52      |
| R² Score | 0.625         |

---

# 14. Result Analysis

## R² Score Interpretation

The model achieved an R² score of approximately 0.625.

This means the model explains around 62.5% of the variation in housing prices.

For a beginner Machine Learning regression project, this is considered a decent result.

---

## Scatter Plot Analysis

A scatter plot was used to compare actual and predicted prices.

```python
plt.scatter(y_test, y_pred)
```

### Observation

The scatter plot showed a positive relationship between actual and predicted prices, indicating that the model captured housing price trends reasonably well.

---

# 15. Advantages of the Project

* Demonstrates complete ML workflow
* Helps understand regression problems
* Uses real-world housing data
* Provides practical experience in data preprocessing and model evaluation
* Beginner-friendly Machine Learning implementation

---

# 16. Limitations

* Linear Regression assumes linear relationships
* The model cannot capture highly complex patterns
* Some important external factors affecting house prices are not included in the dataset
* Prediction accuracy can be further improved using advanced algorithms

---

# 17. Future Scope

The project can be improved by:

* Using advanced regression models
* Applying feature engineering techniques
* Hyperparameter tuning
* Adding more real-world features
* Deploying the model as a web application using Flask or Streamlit

---

# 18. Conclusion

This project successfully demonstrated how Machine Learning can be used for property price prediction.

The project covered all important stages of a Machine Learning workflow including preprocessing, visualization, model training, prediction, and evaluation.

Multiple Linear Regression was successfully implemented to predict house prices using housing-related features.

The project also provided valuable practical understanding of data analysis and Machine Learning concepts.

---

# 19. Author

## Developed By

Saranya sataka

Machine Learning with AI Project

---

# 20. GitHub Repository Structure

```text
Property_Price_Prediction/
│
├── dataset/
├── notebooks/
├── screenshots/
├── report/
└── README.md
```

---

# GitHub README.md Content

````markdown
# Property Price Prediction using Machine Learning

## Project Overview

This project predicts California housing prices using Multiple Linear Regression. The project demonstrates a complete Machine Learning workflow including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, prediction, and evaluation.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Project Workflow

```text
Data Collection
↓
Data Preprocessing
↓
Exploratory Data Analysis
↓
Feature Encoding
↓
Train-Test Split
↓
Model Training
↓
Prediction
↓
Model Evaluation
````

---

## Dataset Features

* Longitude
* Latitude
* Housing Median Age
* Total Rooms
* Total Bedrooms
* Population
* Households
* Median Income
* Ocean Proximity
* Median House Value

---

## Data Preprocessing

* Handled missing values using median imputation
* Applied One-Hot Encoding for categorical data
* Performed Exploratory Data Analysis using histograms and heatmaps

---

## Machine Learning Model

### Multiple Linear Regression

The model was trained to predict house prices using housing-related features.

---

## Model Evaluation Results

| Metric   | Value         |
| -------- | ------------- |
| MAE      | 50670.74      |
| MSE      | 4908476721.15 |
| RMSE     | 70060.52      |
| R² Score | 0.625         |

---

## Visualizations

### Correlation Heatmap

Add your heatmap screenshot here.

### Actual vs Predicted Prices

Add your prediction scatter plot here.

---

## Project Structure

```text
Property_Price_Prediction_ML/
│
├── dataset/
├── notebooks/
├── screenshots/
├── report/
├── README.md
└── requirements.txt
```

---

## Conclusion

This project successfully demonstrates how Machine Learning can be used for property price prediction using regression techniques. The project also provides practical understanding of preprocessing, visualization, model training, and evaluation.

---

## Author

Saranya

```

---

# 21. References

- Scikit-learn Documentation
- Pandas Documentation
- NumPy Documentation
- Matplotlib Documentation
- California Housing Dataset

```
