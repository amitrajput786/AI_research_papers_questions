#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 23:06:12 2026

@author: amit
"""
""" global average pooling 
problem;114"""
import numpy as np 

x = np.array([[[1, 2, 3], 
               [4, 5, 6]], 
              [[7, 8, 9], 
               [10, 11, 12]]])

# Shape: (2, 2, 3)
#         │  │  └── 3 channels
#         │  └───── 2 width  
#         └──────── 2 height

# np.mean(x, axis=(0, 1)) explanation:
# ═════════════════════════════════════

# STEP 1: Average over axis 0 (height)
temp = np.mean(x, axis=1)
# temp = [[(1+7)/2, (2+8)/2, (3+9)/2],     = [[4, 5, 6],
#         [(4+10)/2, (5+11)/2, (6+12)/2]]     [7, 8, 9]]
# Shape: (2, 3)
print(temp)
# STEP 2: Average over axis 1 (width) of temp
result = np.mean(temp, axis=0)  
# result = [(4+7)/2, (5+8)/2, (6+9)/2] = [5.5, 6.5, 7.5]
# Shape: (3,)
print(result)
# OR in one step:
result = np.mean(x, axis=(0, 1))  # [5.5, 6.5, 7.5]

"""problem:115
 
Implement Batch Normalization for BCHW Input
Easy
Deep Learning

Implement a function that performs Batch Normalization on a 4D NumPy array representing a batch of feature maps in the BCHW format (batch, channels, height, width). The function should normalize the input across the batch and spatial dimensions for each channel, then apply scale (gamma) and shift (beta) parameters. Use the provided epsilon value to ensure numerical stability."""

import numpy as np 
B, C, H, W = 2, 2, 2, 2
np.random.seed(42)
X = np.random.randn(B, C, H, W)
gamma = np.ones(C).reshape(1, C, 1, 1)
beta = np.zeros(C).reshape(1, C, 1, 1)
def batch_normal(x,gamma,beta,epsilon=1e-2):
    x=np.array(x)
    gamma=np.array(gamma)
    beta=np.array(beta)
    mean=np.mean(x,axis=(0,2,3),keepdims=True)
    variance=np.var(x,axis=(0,2,3),keepdims=True)
    x_normalized=(x-mean)/np.sqrt(variance+epsilon)
    output=gamma*x_normalized+beta
    return print(output)

c=batch_normal(X, gamma, beta)
    
    
    
    
    
    
    
    
    
    
    