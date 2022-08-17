import pandas as pd
import numpy as np
from custom_knn import CustomKNN
# import random


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])


clf = CustomKNN()
X_train, X_test, y_train, y_test = clf.split_data(X, y, test_size=0.2)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(score)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)

print("Prediction: ", prediction)
