import pandas as pd
import matplotlib.pyplot as plt



df=pd.read_csv("data/stores_sales_forecasting.csv",encoding='latin-1')
df['Order Date']=pd.to_datetime(df['Order Date'])
df_subset=df[['Order Date','Sales']]
df_daily=df_subset.groupby('Order Date').sum()
df_weekly=df_daily['Sales'].resample('W').sum()
plt.figure(figsize=(12,6))
plt.plot(df_weekly)
plt.title('Weekly Sales Volume')
plt.show()