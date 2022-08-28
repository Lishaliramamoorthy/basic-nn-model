# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
data1 = pd.read_csv('datadl1.csv')
data1.head()
X = data1[['Input']].values
X
Y = data1[["Output"]].values
Y
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scalar=MinMaxScaler()
scalar.fit(X_train)
scalar.fit(X_test)
X_train=scalar.transform(X_train)
X_test=scalar.transform(X_test)
import tensorflow as tf
model=tf.keras.Sequential([tf.keras.layers.Dense(4,activation='relu'),
                          tf.keras.layers.Dense(4,activation='relu'),
                          tf.keras.layers.Dense(1)])
model.compile(loss="mae",optimizer="rmsprop",metrics=["mse"])
history=model.fit(X_train,Y_train,epochs=1000)
import numpy as np
X_test
preds=model.predict(X_test)
np.round(preds)
tf.round(model.predict([[20]]))
pd.DataFrame(history.history).plot()
r=tf.keras.metrics.RootMeanSquaredError()
r(Y_test,preds)
```

## Dataset Information

![dataset](https://user-images.githubusercontent.com/75237886/187087553-3281fa32-1698-439c-b90c-9c0c2c9475ed.jpg)


## OUTPUT

### Training Loss Vs Iteration Plot

![plot](https://user-images.githubusercontent.com/75237886/187087575-0f9d5068-435c-4c90-b3cd-ac610b3fd5ac.jpg)


### Test Data Root Mean Squared Error

![rootmean](https://user-images.githubusercontent.com/75237886/187087584-f08788d1-2a5f-4ed5-98a7-18ef70d9dc52.jpg)


### New Sample Data Prediction

![predict](https://user-images.githubusercontent.com/75237886/187087597-03d355ee-403e-4f28-927e-77658c62ff21.jpg)


## RESULT

Thus to develop a neural network model for the given dataset has been implemented successfully.
