import pandas as pd

dataset=pd.read_csv("breast-cancer.csv")


dataset.drop('id',axis=1,inplace=True)

print(dataset['diagnosis'].value_counts())

dataset['diagnosis'].replace(['M','B'],[0,1],inplace=True)
print(dataset)



y=dataset['diagnosis']
x=dataset.drop('diagnosis',axis=1)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)



model=LogisticRegression()
model.fit(x_train,y_train)
y_train_pred=model.predict(x_train)
print("acc of training set:",accuracy_score(y_train,y_train_pred)*100)
y_pred=model.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
print("Accuracy of predicting: ",accuracy_score(y_test,y_pred)*100)