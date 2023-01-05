from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("data_banknote_authentication.txt", sep=",", header=None)
X_raw = df.iloc[:, 0:3]
X_normalized=(X_raw-X_raw.min())/(X_raw.max()-X_raw.min())
y = df.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X_normalized,y, random_state = 50, test_size = 0.25)

logistic_regression_classifier = LogisticRegression(max_iter = 1000, random_state=0).fit(X_train,y_train)

y_pred = logistic_regression_classifier.predict(X_test)
print(y_pred)

print(accuracy_score(y_test,y_pred))