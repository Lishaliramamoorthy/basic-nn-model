# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries that are necessary . Import the dataset and check the types of the columns. Now build your training and test set from the dataset.Networks
typically consists of input,hidden and output layer.We can see that there are three inputs and thus there are three input nodes. Here we are making the neural network 2 hidden layer with activation layer as relu and with their nodes in them. These hidden layers have 4 nodes in each hidden layer.The output layer gives one output therefore,there is one output node.The nodes in the input layer take input and forward it for further process, in the diagram above the nodes in the input layer forwards their output to each of the four nodes in the hidden layer, and in the same way, the hidden layer processes the information and passes it to the output layer.  Now we will fit our dataset and then predict the value.

## Neural Network Model

![diagram](https://user-images.githubusercontent.com/75237886/187088165-0292532e-16ab-4e9f-8aed-ce3a1ec9a011.jpg)


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
