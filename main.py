from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

data=load_iris()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23)


pipeline=Pipeline([("Scaller",StandardScaler()),
                   ("Model",RandomForestClassifier())])

pipeline.fit(x_train,y_train)

pred=pipeline.predict(x_test)


print(f"pipe accuracy is {accuracy_score(y_test,pred)}")



