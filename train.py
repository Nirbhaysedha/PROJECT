
### EXPERIMENTATION TRACKING USING DVC

import random # for random numbers 
import sys
from dvclive import Live
import yaml 
with Live(save_dvc_exp=True) as live: # context manager of dvc live
    train = yaml.safe_load(open('params.yaml'))['train'] # reading train parameters form param.yaml
    epochs=train['epochs'] # reading epochs from train (this can be accessed as a key value pair)
    live.log_param("epochs", epochs) # LOGGING THE PARAMETER 
    for epoch in range(epochs):
        live.log_metric("train/accuracy", epoch + random.random())
        live.log_metric("train/loss", epochs - epoch - random.random())
        live.log_metric("val/accuracy",epoch + random.random() )
        live.log_metric("val/loss", epochs - epoch - random.random()) # HERE WE LOGGED MULTIPLE METRICS
        live.log_metric("ALWAYS WANTED",100)
        live.next_step()
    const=100
    live.log_param("constant",const)
# to delete a specific experiment we can do that using (select dvc there we have experiment block down)
# i can log paraams where i want to experiments 




# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.datasets import load_iris
# from sklearn.metrics import accuracy_score
# data=load_iris()

# x=data.data
# y=data.target

# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23)

# with Live(save_dvc_exp=True) as live:
#     N=10
#     live.log_param("No of neighbours",N)
#     Model=KNeighborsClassifier(n_neighbors=N)
#     Model.fit(x_train,y_train)
#     y_pred=Model.predict(x_test)
#     live.log_metric("Accuracy ",accuracy_score(y_test,y_pred))
