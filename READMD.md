# Sales Forecasting and Business Analytics Project

## Overview

This project focuses on analyzing retail sales data and predicting future sales trends using time series and machine learning techniques.

The goal is to generate useful insights that can help in inventory planning, demand forecasting, and improving business decisions.

---

## Objectives

* Perform exploratory data analysis
* Understand sales and profit patterns
* Identify loss-making products and regions
* Build time series forecasting models using ARIMA
* Apply machine learning models for prediction

---

## Dataset

The dataset contains retail transaction details such as:

* Order ID, Order Date, Ship Date
* Customer and regional information
* Product category and sub-category
* Sales, quantity, discount, and profit

---

## Project Workflow

### 1. Data Preprocessing

* Converted date columns into proper datetime format
* Created additional features like year, month, and weekday
* Calculated profit margin for better analysis

---

### 2. Exploratory Data Analysis

* Analyzed sales distribution across categories
* Studied the relationship between discount and profit
* Compared regional performance
* Identified top-selling and loss-making products

---

### 3. Data Visualization

* Sales trends over time
* Category and sub-category analysis
* Correlation between important variables
* Profit distribution

---

### 4. Time Series Forecasting

* Applied moving average for smoothing
* Built ARIMA model
* Predicted future sales based on historical data

---

### 5. Machine Learning Models

* Linear Regression
* Random Forest Regressor

Features used:

* Quantity
* Discount
* Month and Year

---

### 6. Model Evaluation

* Evaluated performance using Mean Absolute Error
* Compared actual and predicted values

---

## Key Insights

* Higher discounts often reduce profit
* Some product categories consistently generate losses
* Sales show seasonal patterns
* Certain regions perform better than others

---

## Technologies Used

* Python
* Pandas and NumPy
* Matplotlib and Seaborn
* Scikit-learn
* Statsmodels

---

## How to Run

```bash
git clone https://github.com/sandip-0/sales-forecasting.git
cd sales-forecasting
pip install -r requirements.txt
python main.py
```

---

## Business Impact

This project can help in:

* Forecasting future demand
* Improving inventory management
* Optimizing pricing strategies
* Reducing losses

---

## Future Improvements

* Seasonal ARIMA (SARIMA)
* Prophet model
* Deep learning approach (LSTM)
* Interactive dashboard using Streamlit or Power BI

---

## Author

Sandip Adak
[sandipadak000@gmail.com](mailto:sandipadak000@gmail.com)

---

## Note

Add graphs and outputs in the repository to make the project more clear and easier to understand.
