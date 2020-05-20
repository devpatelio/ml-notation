
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
from sklearn import linear_model


df = pd.read_csv("FuelConsumption.csv")
# print(df.head())
# print(df.describe())

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

def enginesize():
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
    plt.xlabel('Engine Size')
    plt.ylabel('Emissions')
    plt.show()

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)

print(f'Coefficients/Gradient: {regr.coef_}')



regr1 = linear_model.LinearRegression()
x1 = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
regr1.fit(x1, y)
print(f'Coefficients/Gradient: {regr1.coef_}')



def multi_regression():
    y_prediction = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
    x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
    y = np.asanyarray(test[['CO2EMISSIONS']])
    print(f"Residual sum of squares: {np.mean((y_prediction-y)**2)}")
    print(f'Variance score (1 is perfect prediction): {regr.score(x, y)}')


def different_multi_regression():
    y_prediction1 = regr1.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
    x1 = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
    y = np.asanyarray(test[['CO2EMISSIONS']])
    print(f"Residual sum of squares: {np.mean((y_prediction1 - y) ** 2)}")
    print(f'Variance score (1 is perfect prediction): {regr1.score(x1, y)}')


multi_regression()
different_multi_regression()
