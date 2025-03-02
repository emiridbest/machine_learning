import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


df  = pd.read_csv("/content/house_prices.csv") 

df_float = df.select_dtypes(include=['float64']).copy()
df_float.info()


df_float.describe()

df_float['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=False)
df_float['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean(), inplace=False)
df_float['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean(), inplace=False)

df_float.info()

lotFrontage = df_float[['LotFrontage']]
salePrice = df['SalePrice']

lr1 = LinearRegression()
lr1.fit(lotFrontage, salePrice)

print(lr1.coef_)
print(lr1.intercept_)
print(mean_squared_error(salePrice, lr1.predict(lotFrontage)))

masVnrAre = df_float[['MasVnrArea']]
salePrice = df['SalePrice']

