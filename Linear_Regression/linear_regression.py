import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Import dataset
df = pd.read_csv("/content/house_prices.csv")

# Load datasets => only attributes that are measurements
df_float = df.select_dtypes(include=["float64"]).copy()
df_float.info()
df_float.describe()

# Fill in na values
df_float["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].mean(), inplace=False)
df_float["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].mean(), inplace=False)
df_float["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].mean(), inplace=False)
df_float.info()

# Regression LotFrontage
# Define values for the attributes
lotFrontage = df_float[["LotFrontage"]]
salePrice = df["SalePrice"]

# Set instance and fit model
lr1 = LinearRegression()
lr1.fit(lotFrontage, salePrice)
predict1 = lr1.predict(lotFrontage)

# Print coefficient, intercept and mean squared error
print(lr1.coef_)
print(lr1.intercept_)
print(mean_squared_error(salePrice, lr1.predict(lotFrontage)))

# Regression MasVnrArea
# Define values for the attributes
masVnrArea = df_float[["MasVnrArea"]]
salePrice = df["SalePrice"]

# Set instance and fit model
lr2 = LinearRegression()
lr2.fit(masVnrArea, salePrice)
predict2 = lr2.predict(masVnrArea)

# Print coefficient, intercept and mean squared error
print(lr2.coef_)
print(lr2.intercept_)
print(mean_squared_error(salePrice, lr2.predict(masVnrArea)))


# Regression GarageYrBlt
# Define values for the attributes
garageYrBlt = df_float[["GarageYrBlt"]]
salePrice = df["SalePrice"]

# Set instance and fit model
lr3 = LinearRegression()
lr3.fit(garageYrBlt, salePrice)
predict3 = lr3.predict(garageYrBlt)

# Print coefficient, intercept and mean squared error
print(lr3.coef_)
print(lr3.intercept_)
print(mean_squared_error(salePrice, lr3.predict(garageYrBlt)))

# Visualise
f = plt.figure()
f, ax = plt.subplots(1, 3, figsize=(30, 8))

ax = plt.subplot(1, 3, 1)
plt.ylabel("Sale Price")
plt.xlabel("LotFrontage")
ax = plt.scatter(lotFrontage, salePrice)
ax = plt.plot(lotFrontage, predict1, linewidth=5.0, color="red")

ax = plt.subplot(1, 3, 2)
plt.ylabel("SalePrice")
plt.xlabel("MasVnrArea")
ax = plt.scatter(masVnrArea, salePrice)
ax = plt.plot(masVnrArea, predict2, linewidth=5.0, color="red")

ax = plt.subplot(1, 3, 3)
plt.ylabel("Sale Price")
plt.xlabel("GarageYrBlt")
ax = plt.scatter(garageYrBlt, salePrice)
ax = plt.plot(garageYrBlt, predict3, linewidth=5.0, color="red")

ax = plt.show()


# Compare mean squared error using bar chart

names = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]
heights = [
    mean_squared_error(salePrice, predict1),
    mean_squared_error(salePrice, predict2),
    mean_squared_error(salePrice, predict3)]

f = plt.figure(figsize=(8,8))
ax = plt.bar(names, heights)
