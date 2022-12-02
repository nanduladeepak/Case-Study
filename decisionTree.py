import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data_banknote_authentication.txt", sep=",", header=None)
X_raw = df.iloc[:, 0:3]
X_normalized=(X_raw-X_raw.min())/(X_raw.max()-X_raw.min())
y = df.iloc[:, 4]


# Creating Train and Test datasets
X_train, X_test, y_train, y_test = train_test_split(X_normalized,y, random_state = 50, test_size = 0.25)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(y_pred)
print(accuracy_score(y_test,y_pred))