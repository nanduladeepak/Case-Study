import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC



df = pd.read_csv("data_banknote_authentication.txt", sep=",", header=None)
X_raw = df.iloc[:, 0:3]
# X_normalized=(X_raw-X_raw.min())/(X_raw.max()-X_raw.min())
y = df.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X_raw,y, random_state = 50, test_size = 0.25)

SVM_classifier = SVC(kernel = 'linear', random_state = 0)


SVM_classifier.fit(X_train,y_train)
y_pred = SVM_classifier.predict(X_test)


print(y_pred)

print(accuracy_score(y_test,y_pred))