import pandas as pd
from pandas_profiling import ProfileReport
from matplotlib import axes, pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from numpy import set_printoptions
from pickle import dump
from pickle import load
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from IPython.display import Image

#Uploded data file using read.csv
df=pd.read_csv("data_banknote_authentication.txt",sep=',',header=None)
print(df)

#Assign name to the colums
df.columns=['Variance','Skewness','Kurtosis','Entropy','Output']
print(df)

#Generate Analysis Report of Data
"""Report=ProfileReport(df)
Report.to_file(output_file="Banknote_Authentication.html")"""

#Standardizing the data
arr=df.values
#separate the Input and Output
x=arr[:,0:4]
y=arr[:,4]
scaler=StandardScaler().fit(x)
rescale_x=scaler.fit_transform(x)
set_printoptions(precision=3)
print(rescale_x[0:5,:])

#split dataset
x=arr[:,0:4]
y=arr[:,4]
validation_siz=0.20

seed=55
x_train , x_validation , y_train , y_validation = train_test_split(x, y, test_size=validation_siz, random_state=seed)
print (y)

#check models
models=[]
models.append(('LR', LogisticRegression()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('DTC',DecisionTreeClassifier()))
models.append(('SVM',SVC()))
models.append(('RF',RandomForestClassifier()))
#Evaluate models
result=[]
names=[]
for names, model in models:
    kFold=KFold(n_splits=10,shuffle=True, random_state=seed)
    CvResults=cross_val_score(model,x_train,y_train,cv=kFold,scoring='accuracy')
    result.append(CvResults)
    
    ms="%s=%f (%f)"%(names,CvResults.mean(),CvResults.std())
    print(ms)

#compare model
"""fig=plt.figure()
fig.suptitle('Models Comparison')

sx=fig.add_subplot(111)
plt.boxplot(CvResults)
axes.set_xticklabels(names)
a=plt.show()
print(a)"""

#Confusion matrix
Rf=RandomForestClassifier ()
Rf.fit(x_train,y_train)
pred=Rf.predict(x_validation)
print(accuracy_score(y_validation,pred))
print(confusion_matrix(y_validation,pred))
print(classification_report(y_validation,pred))

#finalize model
arr=df.values
x=arr[:,0:4]
y=arr[:,4]
x_train , x_validation , y_train , y_validation = train_test_split(x, y, test_size=0.30, random_state=5)
#fit model on 30%
model=RandomForestClassifier()
model.fit(x_train,y_train)

#save
filename= 'best_fit.sav'
dump(model,open(filename,'wb'))

#load model
loaded_model=load(open(filename,'rb'))
result=loaded_model.score(x,y)
print('Accuracy',result*100)