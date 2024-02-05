import pandas as pd

#read the data set
df = pd.read_csv('diabetes.csv')
df.head()
df.info()
df.shape
from sklearn.model_selection import train_test_split
x=df.iloc[:,df.columns!='Outcome']
y=df.iloc[:,df.columns=='Outcome']
print(y)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(xtrain,ytrain.values.ravel())
predict_output=model.predict(xtest)
print(predict_output)
from sklearn.metrics import accuracy_score
acc=accuracy_score(predict_output,ytest)
print('the accuracy score for random forest:',acc)

























