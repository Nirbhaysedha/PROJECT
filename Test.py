from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import random # for random numbers 
import sys
from dvclive import Live
# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with a StandardScaler and a RandomForestClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the features
    ('classifier', RandomForestClassifier(random_state=42))  # Step 2: Random Forest Classifier
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)




with Live(save_dvc_exp=True) as live: # context manager of dvc 
    epochs = 2
    live.log_param("epochs", epochs) # LOGGING THE PARAMETER 
    for epoch in range(epochs):
        live.log_metric("Model Accuracy",accuracy)
        live.next_step()


