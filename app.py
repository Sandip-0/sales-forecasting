import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Forecasting Dashboard")
st.write("Sales analysis and Random Forest regression")


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "stores_sales_forecasting.csv",
        encoding="latin-1"
    )

    # Convert dates
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])

    # Feature engineering
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Day"] = df["Order Date"].dt.date
    df["Weekday"] = df["Order Date"].dt.day_name()

    # Profit Margin
    df["Profit Margin"] = df["Profit"] / df["Sales"]

    return df


df = load_data()


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("Dashboard Filters")

# Region
regions = ["All"] + sorted(
    df["Region"].dropna().unique().tolist()
)

selected_region = st.sidebar.selectbox(
    "Select Region",
    regions
)


# Year
years = ["All"] + sorted(
    df["Year"].dropna().unique().tolist()
)

selected_year = st.sidebar.selectbox(
    "Select Year",
    years
)


# Apply filters
filtered_df = df.copy()

if selected_region != "All":
    filtered_df = filtered_df[
        filtered_df["Region"] == selected_region
    ]

if selected_year != "All":
    filtered_df = filtered_df[
        filtered_df["Year"] == selected_year
    ]


# =========================================================
# BUSINESS OVERVIEW
# =========================================================

st.header("Business Overview")

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = len(filtered_df)

if total_sales != 0:
    profit_margin = total_profit / total_sales
else:
    profit_margin = 0


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Sales",
        f"${total_sales:,.2f}"
    )

with col2:
    st.metric(
        "Total Profit",
        f"${total_profit:,.2f}"
    )

with col3:
    st.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

with col4:
    st.metric(
        "Profit Margin",
        f"{profit_margin:.2%}"
    )


st.divider()


# =========================================================
# TOP 10 PRODUCTS
# =========================================================

st.header("Top 10 Products by Sales")

top_products = (
    filtered_df
    .groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))

top_products.sort_values().plot(
    kind="barh",
    ax=ax
)

ax.set_xlabel("Sales")
ax.set_ylabel("Product")
ax.set_title("Top 10 Products by Sales")

st.pyplot(fig)

plt.close(fig)


# =========================================================
# LOSS-MAKING PRODUCTS
# =========================================================

st.header("Top 10 Loss-Making Products")

loss = filtered_df[
    filtered_df["Profit"] < 0
]

loss_products = (
    loss
    .groupby("Product Name")["Profit"]
    .sum()
    .sort_values()
    .head(10)
)

if not loss_products.empty:

    fig, ax = plt.subplots(figsize=(10, 5))

    loss_products.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_xlabel("Profit")
    ax.set_ylabel("Product")
    ax.set_title("Top 10 Loss-Making Products")

    st.pyplot(fig)

    plt.close(fig)

else:
    st.success("No loss-making products found.")


# =========================================================
# SALES BY REGION
# =========================================================

st.header("Sales by Region")

region_sales = (
    filtered_df
    .groupby("Region")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(8, 5))

region_sales.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Region")
ax.set_ylabel("Sales")
ax.set_title("Sales by Region")

plt.xticks(rotation=30)

st.pyplot(fig)

plt.close(fig)


# =========================================================
# DAILY SALES TIME SERIES
# =========================================================

st.header("Daily Sales Trend")

ts = (
    filtered_df
    .groupby("Order Date")["Sales"]
    .sum()
    .sort_index()
)

# Same logic as your notebook
ts = ts.asfreq("D")
ts = ts.fillna(0)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    ts.index,
    ts.values
)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Daily Sales Trend")

plt.xticks(rotation=45)

st.pyplot(fig)

plt.close(fig)


# =========================================================
# 7-DAY MOVING AVERAGE
# =========================================================

st.header("7-Day Moving Average")

ts_ma = ts.rolling(window=7).mean()

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    ts.index,
    ts.values,
    label="Daily Sales"
)

ax.plot(
    ts_ma.index,
    ts_ma.values,
    label="7-Day Moving Average"
)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Sales Trend with 7-Day Moving Average")

ax.legend()

plt.xticks(rotation=45)

st.pyplot(fig)

plt.close(fig)


# =========================================================
# MONTHLY SALES
# =========================================================

st.header("Monthly Sales")

monthly_sales = (
    filtered_df
    .groupby(["Year", "Month"])["Sales"]
    .sum()
    .reset_index()
)

monthly_sales["Month-Year"] = (
    monthly_sales["Year"].astype(str)
    + "-"
    + monthly_sales["Month"].astype(str).str.zfill(2)
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(
    monthly_sales["Month-Year"],
    monthly_sales["Sales"]
)

ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.set_title("Monthly Sales")

plt.xticks(rotation=45)

st.pyplot(fig)

plt.close(fig)


# =========================================================
# SALES BY WEEKDAY
# =========================================================

st.header("Sales by Weekday")

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_sales = (
    filtered_df
    .groupby("Weekday")["Sales"]
    .sum()
    .reindex(weekday_order)
)

fig, ax = plt.subplots(figsize=(10, 5))

weekday_sales.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Weekday")
ax.set_ylabel("Sales")
ax.set_title("Sales by Weekday")

plt.xticks(rotation=30)

st.pyplot(fig)

plt.close(fig)


# =========================================================
# SALES VS PROFIT
# =========================================================

st.header("Sales vs Profit")

fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(
    filtered_df["Sales"],
    filtered_df["Profit"],
    alpha=0.5
)

ax.axhline(
    y=0,
    linestyle="--"
)

ax.set_xlabel("Sales")
ax.set_ylabel("Profit")
ax.set_title("Sales vs Profit")

st.pyplot(fig)

plt.close(fig)


# =========================================================
# MACHINE LEARNING
# =========================================================

st.divider()

st.header("🤖 Random Forest Regression")

st.write(
    "Predicting Sales using Quantity, Discount, Month and Year."
)


# Same features as your notebook
X = filtered_df[
    [
        "Quantity",
        "Discount",
        "Month",
        "Year"
    ]
]

y = filtered_df["Sales"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Random Forest
model = RandomForestRegressor(
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# Prediction
pred = model.predict(X_test)


# =========================================================
# MAE
# =========================================================

mae = mean_absolute_error(
    y_test,
    pred
)

st.subheader("Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Mean Absolute Error",
        f"{mae:,.2f}"
    )

with col2:
    st.metric(
        "Test Samples",
        f"{len(X_test):,}"
    )


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

st.subheader("Actual vs Predicted Sales")

prediction_df = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": pred
})

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    prediction_df["Actual Sales"],
    label="Actual Sales"
)

ax.plot(
    prediction_df["Predicted Sales"],
    label="Predicted Sales"
)

ax.set_xlabel("Test Sample")
ax.set_ylabel("Sales")
ax.set_title("Actual vs Predicted Sales")

ax.legend()

st.pyplot(fig)

plt.close(fig)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader("Feature Importance")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

ax.set_xlabel("Feature")
ax.set_ylabel("Importance")
ax.set_title("Random Forest Feature Importance")

plt.xticks(rotation=30)

st.pyplot(fig)

plt.close(fig)


# =========================================================
# DATASET
# =========================================================

st.divider()

st.header("Dataset Preview")

st.dataframe(
    filtered_df.head(20),
    use_container_width=True
)


# =========================================================
# DATASET INFORMATION
# =========================================================

with st.expander("Dataset Information"):

    st.write(
        "Rows:",
        len(filtered_df)
    )

    st.write(
        "Columns:",
        len(filtered_df.columns)
    )

    st.write(
        "Column Names:"
    )

    st.write(
        filtered_df.columns.tolist()
    )