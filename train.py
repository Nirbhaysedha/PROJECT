
### EXPERIMENTATION TRACKING USING DVC

import random # for random numbers 
import sys
from dvclive import Live

with Live(save_dvc_exp=True) as live: # context manager of dvc 
    epochs = 10
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




