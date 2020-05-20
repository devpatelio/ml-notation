import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def cubic():
    x = np.arange(-5.0, 5.0, 0.1)
    ##You can adjust the slope and intercept to verify the changes in the graph
    y = 1*(x**3) + 1*(x**2) + 1*x + 3
    y_noise = 20 * np.random.normal(size=x.size)
    ydata = y + y_noise
    plt.plot(x, ydata,  'bo')
    plt.plot(x,y, 'r')
    plt.ylabel('Dependent Variable')
    plt.xlabel('Indepdendent Variable')
    plt.show()

def exponential_euler():
    X = np.arange(-5.0, 5.0, 0.1)

    ##You can adjust the slope and intercept to verify the changes in the graph

    Y= np.exp(X)

    plt.plot(X,Y)
    plt.ylabel('Dependent Variable')
    plt.xlabel('Indepdendent Variable')
    plt.show()

df = pd.read_csv("china_gdp.csv")
# print(df.head(10))


def sigmoid(x, Beta_1, Beta_2):
    y = 1/(1+np.exp(-Beta_1*(x-Beta_2)))
    return y

def china():
    # plt.figure(figsize=(8, 5))
    # plt.xlabel("Year")
    # plt.ylabel("GDP")
    # beta_1 = 0.10
    # beta_2 = 1990.0
    # Y_pred = sigmoid(x_data, beta_1, beta_2)
    # plt.plot(x_data, Y_pred*15**12)
    # plt.plot(x_data, y_data, 'ro')

    x_data, y_data = (df["Year"].values, df["Value"].values)

    # xdata = (x_data - min(x_data))/(max(x_data) - min(x_data))
    # ydata = (y_data - min(y_data))/(max(y_data) - min(y_data))
    xdata = x_data/max(x_data)
    ydata = y_data/max(y_data)

    # print(f"beta_1 = {popt[0]}, beta_2 = {popt[1]}")

    msk = np.random.rand(len(df)) < 0.8
    train_x = xdata[msk]
    test_x = xdata[~msk]
    train_y = ydata[msk]
    test_y = ydata[~msk]

    popt, pcov = curve_fit(sigmoid, train_x, train_y)
    x = np.linspace(1960, 2015, 55)
    x = x/max(x)
    plt.figure(figsize=(8, 5))
    y = sigmoid(x, popt[0], popt[1])
    y_predict = sigmoid(test_x, popt[0], popt[1])
    plt.plot(xdata, ydata, 'ro', color='blue', label='data')
    plt.plot(x, y, label='fit')
    plt.ylabel('GDP')
    plt.xlabel('Year')
    plt.show()
    print("Mean absolute error: %.2f" % np.mean(np.absolute(y_predict - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((y_predict - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(y_predict, test_y))


china()



