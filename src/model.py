import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# load data
df=pd.read_csv("data/stores_sales_forecasting.csv",encoding='latin-1')
# print(df.head())

# convert dates
df['Order Date']=pd.to_datetime(df['Order Date'])
df['Ship Date']=pd.to_datetime(df['Ship Date'])

# add fiture 
df['Year']=df['Order Date'].dt.year
df['Month']=df['Order Date'].dt.month
df['Day']=df['Order Date'].dt.date
df['Weekday']=df['Order Date'].dt.day_name()

# profit margin
df['Profit Margin']=df['Profit']/df['Sales']

# Business Insights (EDA)
# top products
top_products=df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
print(top_products)

# loos making products
loss=df[df['Profit']<0]
print(loss[['Product Name','Profit']].sort_values(by='Profit', ascending=True).head(10))

# sales regios
region_sales=df.groupby('Region')['Sales'].sum()
region_sales.plot(kind='bar',title='Sales by Region',rot=30)
plt.ylabel('Sales')
plt.show()

# time serise data
ts=df.groupby('Order Date')['Sales'].sum()
ts=ts.asfreq('D')
ts = ts.fillna(0)

# moving avg
ts_ma = ts.rolling(window=7).mean()

# # arima model
# ...


# mechin learning model 
# Features
X=df[['Quantity','Discount','Month','Year']]
y=df['Sales']
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
model=RandomForestRegressor()
model.fit(X_train,y_train)
pred=model.predict(X_test)

# evalution
print("MAE:", mean_absolute_error(y_test, pred))