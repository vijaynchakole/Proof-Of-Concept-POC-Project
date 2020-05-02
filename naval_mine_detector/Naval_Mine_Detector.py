# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:01:40 2020

@author: Vijay Narsing Chakole

"""

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#################################################################

#Reading dataset

def read_dataset(data):
    df = pd.read_csv(data)
    print("dataset loaded successfully")
    print("number of columns",len(df.columns))
    
    #Features of dataset
    X = df[df.columns[0:60]].values
    
    #Target variable of dataset
    Y = df[df.columns[60]].values
    
    #Encode the depedent variable
    encoder = LabelEncoder()
    
    #Encode character label into integer ie 1 or 0 (one hot encode)
    encoder.fit(Y)
    Y = encoder.transform(Y)
    Y = one_hot_encode(Y)
    
    print("shape of X :", X.shape)
    
    return(X,Y)
    
    
###########################################################################
    
#define encoder function  to set M=> 1 , R=>0

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels),labels] = 1
    return one_hot_encode

##############################################################################

#model for training
    
def multilayer_perceptron(x,weights,biases):
    
    #Hidden layers with RELU activations
    
    #First layer performs  matrix multiplication of input  with weights
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    #Hidden layer with sigmoid activations
    #Second layer performs matrix multiplication of layer 1 with weights
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    
    #Hidden layer with sigmoid activations
    #Third layer performs matrix multiplication of layer 2 with weights
    layer_3 = tf.add(tf.matmul(layer_2,weights['h3']),biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    
    #Hidden layer with RELU activations
    #Forth layer performs matrix multiplication of layer 3 with weights
    layer_4 = tf.add(tf.add(layer_3,weights['h4']),biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    
    #output layer with linear activations
    out_layer  = tf.matmul(layer_4,weights['out']) + biases['out']

    return out_layer

    
    
    

###############################################################################
#Read dataset
path = "C:/Users/hp/PycharmProjects/Project/Sonar.csv"
X,Y = read_dataset(path)
X
    
#shuffle the dataset to mix up the rows
X,Y = shuffle(X,Y,random_state = 1)
X
Y  
    
#convert data into train and test datasets
#for testing we use 30% and for training we use 70%

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = .30,random_state = 42)

#check shape of data of train and test
print(f"X train shape : {X_train.shape}")
print(f"X test shape : {X_test.shape}")
print(f"y train shape : {y_train.shape}")
print(f"y test shape : {y_test.shape}")

#define parameters which are required for tensors

#change in a variable in each iteration
learning_rate = 0.30

#Total number of iteration to minimize error
training_epochs = 1000

cost_history = np.empty(shape=[1],dtype = float)


#number of features = number of columns
n_dim = X.shape[1]
print(f"number of columns are : {n_dim}")


#as we have two classes as M and R
n_class = 2

#path which contains model file

model_path = "C:/Users/hp/PycharmProjects/Project"

#define number of hidden layers and numbers of neurons for each layer 

#number of hidden layers are 4
#and number of nuerons in each layer are 60

n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

#Placeholder to store Inputs
#RuntimeError: tf.placeholder() is not compatible with eager execution.
#tf.placeholder() is meant to be fed to the session that when run recieve the values from feed dict and perform the required operation. 
#Generally you would create a Session() with 'with' keyword and run it.
#But this might not favour all situations due to which you would require immediate execution.
#This is called eager execution.

tf.compat.v1.disable_eager_execution()


x = tf.compat.v1.placeholder(tf.float32,[None,n_dim])
#Model parameters
W = tf.Variable(tf.zeros([n_dim,n_class]))
W
b = tf.Variable(tf.zeros([n_class]))
b

#define weights and bias for each layer
#Create variable which contents random values

weights = {
        'h1':tf.Variable(tf.random.truncated_normal([n_dim, n_hidden_1])),
        'h2':tf.Variable(tf.random.truncated_normal([n_hidden_1, n_hidden_2])),
        'h3':tf.Variable(tf.random.truncated_normal([n_hidden_2, n_hidden_3])),
        'h4':tf.Variable(tf.random.truncated_normal([n_hidden_3, n_hidden_4])),
        'out':tf.Variable(tf.random.truncated_normal([n_hidden_4, n_class]))
        
        }


#create Variable which contents random bias values

biases = {
        'b1':tf.Variable(tf.random.truncated_normal([n_hidden_1])),
        'b2':tf.Variable(tf.random.truncated_normal([n_hidden_2])),
        'b3':tf.Variable(tf.random.truncated_normal([n_hidden_3])),
        'b4':tf.Variable(tf.random.truncated_normal([n_hidden_4])),
        'out':tf.Variable(tf.random.truncated_normal([n_class]))
        }


##################################################################################


#initialize all variables
init = tf.global_variables_initializer()

#define cost function
y_ = tf.compat.v1.placeholder(tf.float32,[None,n_class])
y = tf.nn.softmax(tf.matmul(x,W)+b)
cost_function = tf.reduce_mean(-tf.reduce_sum((y_ *  tf.log(y)),reduction_indices=[1]))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

#initialize the session
sess = tf.Session()
sess.run(init)
mse_history = []


#calculate the cost for each epoch
for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={x:X_train, y_:y_train})
    cost = sess.run(cost_function,feed_dict={x:X_train, y_:y_train})
    cost_history = np.append(cost_history, cost)
    print(f"epoch : {epoch} \n cost : {cost}")

pred_y = sess.run(y, feed_dict = {x: X_test})
pred_y
pred_y.shape    

#Calculate Accuracy
correct_prediction = tf.equal(tf.argmax(pred_y,1),tf.argmax(y_test,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(f"Accuracy : {sess.run(accuracy)}")

plt.plot(range(len(cost_history)),cost_history)
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()
