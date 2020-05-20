
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


df = pd.read_csv("FuelConsumption.csv")
# print(df.head())

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])


train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x)
clf = linear_model.LinearRegression()
train_y_predicted = clf.fit(train_x_poly, train_y)
# print(f"Coeficcient/Gradient: {clf.coef_}")
# print(f"Intercept: {clf.intercept_}")

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )


#
#
#
# poly_1 = PolynomialFeatures(degree=3)
# train_x_1 = poly_1.fit_transform(train_x)
# cdf = linear_model.LinearRegression()
# train_y_1 = cdf.fit(train_x_1, train_y)
# test_x_1 = poly.fit_transform(test_x)
# test_y_1 = cdf.predict(test_x_1)
# print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_1 - test_y)))
# print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_1 - test_y) ** 2))
# print("R2-score: %.2f" % r2_score(test_y_1 , test_y) )




def poopoo():
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    XX = np.arange(0.0, 10.0, 0.1)
    yy = clf.intercept_[0] + clf.coef_[0][1]*XX+clf.coef_[0][2]*np.power(XX, 2)
    plt.plot(XX, yy, '-r')
    plt.show()

# def peepee():
#     plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
#     plt.xlabel("Engine size")
#     plt.ylabel("Emission")
#     XX = np.arange(0.0, 10.0, 0.1)
#     yy = cdf.intercept_[0] + cdf.coef_[0][1]*XX+cdf.coef_[0][2]*np.power(XX, 2)+cdf.coef_[0][3]*np.power(XX, 3)
#     plt.plot(XX, yy, '-r')
#     plt.show()

poopoo()