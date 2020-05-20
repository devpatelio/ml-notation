
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import wget
from sklearn import linear_model
from sklearn.metrics import r2_score


df = pd.read_csv("FuelConsumption.csv")
# print(df.head())
# print(df.describe())

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# print(cdf.head())

def fuelconsumptioncomb():
    plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel("FUELCONSUMPTION")
    plt.ylabel("Emission")
    plt.show()

def enginesize():
    plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel("Engine Size")
    plt.ylabel("Emission")
    plt.show()

def cylinder():
    plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel("Cylinder Size")
    plt.ylabel("Emission")
    plt.show()


fuelconsumptioncomb()
# cylinder()
# enginesize()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

#Coefficients (Intercept and Gradient)
print('Coefficients:', regr.coef_)
print('Intercept: ', regr.intercept_)

def training_data():
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
    plt.plot(train_x, regr.coef_[0][0]*train_x+regr.intercept_[0], '-r')
    plt.xlabel("Engine Size")
    plt.ylabel("CO2 Emissions")
    plt.show()

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

#Evaluation Methods
print(f"Mean absolute error: {np.mean(np.absolute(test_y_-test_y))}.2f")
print(f"Residual sum of squares (MSE): {np.mean((test_y_ - test_y) ** 2)}.2f")
print(f"R2-score: {r2_score(test_y_, test_y)}.2f")

