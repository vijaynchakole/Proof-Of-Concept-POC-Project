# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 23:40:30 2020

@author: Vijay Narsing Chakole
Topic : ANN for Titanic survival

"""

# import libraries
# For data loading and manipulation
import pandas as pd
# For maths calculations
import numpy as np

# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For splitting data into train set and test set
from sklearn.model_selection import train_test_split



dataset = pd.read_csv('TitanicDataset.csv')

#dataset.head(20)
# drop na values (delete rows)
dataset.dropna(inplace = True, axis = 0) 

# creating X and y
X = dataset.drop('Survived', axis = 1)
y = dataset.Survived


# spliting dataset into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)



# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)



# Importing the Keras libraries and packages
import tensorflow.compat.v1 as tf
import keras 
from tensorflow.compat.v1.keras.models import Sequential # To initialise the neural networks
from tensorflow.compat.v1.keras.layers import Dense # to create layers in neural networks

# create your classifier here
# Initialising the ANN
classifier = Sequential()

"""
AttributeError: module 'tensorflow' has no attribute 'get_default_graph' 
= due to using wrong version of tensprflow

from tensorflow.compat.v1.keras.models import Sequential
use this
classifier = tf.keras.models.Sequential()
"""

X_train.shape
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, activation = 'relu', input_dim = 9))

# Adding second hidden layer
classifier.add(Dense(units = 6, activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)


# Predicting the test set Results
y_pred = classifier.predict(X_test) # its gives us probability so we have to convert it into 1 or 0 form

y_pred = (y_pred > 0.5)


# making confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
cm

score = accuracy_score(y_test, y_pred)
score # accuracy : 86% 


