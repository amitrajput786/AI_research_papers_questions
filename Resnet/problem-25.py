#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 15:31:20 2026

@author: amit
"""
""" 
problem-25 
Single Neuron with Backpropagation:
    Write a Python function that simulates a single neuron with sigmoid activation, and implements backpropagation to update the neuron's weights and bias. The function should take a list of feature vectors, associated true binary labels, initial weights, initial bias, a learning rate, and the number of epochs. The function should update the weights and bias using gradient descent based on the MSE loss, and return the updated weights, bias, and a list of MSE values for each epoch, each rounded to four decimal places.
Example:
Input:

features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]], labels = [1, 0, 0], initial_weights = [0.1, -0.2], initial_bias = 0.0, learning_rate = 0.1, epochs = 2

Output:

updated_weights = [0.1036, -0.1425], updated_bias = -0.0167, mse_values = [0.3033, 0.2942]

Reasoning:

The neuron receives feature vectors and computes predictions using the sigmoid activation. Based on the predictions and true labels, the gradients of MSE loss with respect to weights and bias are computed and used to update the model parameters across epochs.
"""

features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]
labels = [1, 0, 0]
initial_weights = [0.1, -0.2]
initial_bias = 0.0
learning_rate = 0.1
epochs = 2
import numpy as np
def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    w=initial_weights
    w=np.array(w)
    y=labels
    y=np.array(y)
    b=initial_bias
    X=features
    X=np.array(X)
    m=X.shape[0]
    mse_values=[]
    for i in range(epochs):
        y_pred=X@(w)+b
        error=y_pred-y
        z=1/(1+np.exp(-y_pred))
        mse=np.mean((z-y)**2)
        mse_values.append(mse)
        sig_d=z*(1-z)
        dw=(2/m)*(X.T@(error*sig_d))
        db=(1/m)*(np.sum(error*sig_d))
        w=w-learning_rate*dw
        b=b-learning_rate*db
    return np.round(w,4).tolist(),round(b,4),mse_values
x=train_neuron(features,labels,initial_weights,initial_bias,learning_rate,epochs)
print(x)
        



labels=[1,0,0]
c=np.array(labels)
print(c.shape)
features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]
d=np.array(features)
print(d.shape[0])
no_sample, no_feature =d.shape
print(no_sample,no_feature)
""" no_sample==no of rows , no_feature == no of columns , """
