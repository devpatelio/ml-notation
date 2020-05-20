import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv("teleCust1000t.csv")
# print(df.head(10))
# print(df['custcat'].value_counts())

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values
# print(X[:5])

y = df['custcat'].values
# print(y[:5])

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
# print(X[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2)
# print(f'Train Test: {X_train.shape}, {y_train.shape}')
# print(f'Test Test: {X_test.shape}, {y_test.shape}')


k = 6
neighbour = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

y_prediction = neighbour.predict(X_test)
# print(y_prediction[:7])
#
# print(f"train set accuracy: {metrics.accuracy_score(y_train, neighbour.predict(X_train))}")
# print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_prediction))

Ks = 35
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

# print(mean_acc)
# print(std_acc)
ConfusionMx = []
for n in range(1, Ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    y_predict = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, y_predict)
    std_acc[n-1] = np.std(y_predict==y_test)/np.sqrt(y_predict.shape[0])


def plotter():
    plt.plot(range(1, Ks), mean_acc, 'g')
    plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
    plt.legend(('Accuracy', '+/- 3xstd'))
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Neighbours (K)')
    plt.tight_layout()
    plt.show()

# plotter()
print(f"Best Accuracy was with {mean_acc.max()} with k={mean_acc.argmax()+1}")
