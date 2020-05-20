import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree



my_data = pd.read_csv('drug200.csv', delimiter=",")
# print(my_data[0:5])
# print(len(my_data))
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values


le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])

le_Cholesterol = preprocessing.LabelEncoder()
le_Cholesterol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Cholesterol.transform(X[:, 3])

# print(X[:5])
y = my_data["Drug"]
# print(y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=3)

# print(f'Dimensions of training data (X, y) are {X_train.shape} and {y_train.shape}.')
# print(f'Dimensions of test data (X, y) are {X_test.shape} and {y_test.shape}.')

drugtree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
# print(drugtree)
drugtree.fit(X_train, y_train)

predtree = drugtree.predict(X_test)
# print(predtree[:5])
# print(y_test[:5])
print(f'DecisionTree Accuracy: {metrics.accuracy_score(y_test, predtree)}')


dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out = tree.export_graphviz(drugtree, feature_names=featureNames, out_file=dot_data,
                              class_names=np.unique(y_train), filled=True, special_characters=True, rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')

# visualization does not work with graphviz
