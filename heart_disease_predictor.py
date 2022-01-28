# The accuracy of model recorded is 0.91.
# activation functions used are relu for hidden layers and sigmoid for output.
# each hidden layer have 12 neurons.
# the network required 100 epochs to achieve the output.
# the dataset is taken from kaggle.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

dataset = pd.read_csv("data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
from sklearn.compose import * 

x_train,x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(12, activation = 'relu'))
ann.add(tf.keras.layers.Dense(12, activation = 'relu'))
ann.add(tf.keras.layers.Dense(1, activation='sigmoid'))

ann.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

ann.fit(x_train,y_train, batch_size = 32, epochs = 100)

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_pred,y_test)
a = accuracy_score(y_pred,y_test)
print("Obtained Confusion Matrix: ")
print(cm)
print("The accuracy obtained by the model is: ", a)