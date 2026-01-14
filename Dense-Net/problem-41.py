#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 05:11:15 2026

@author: amit
"""


"""
Simple Convolutional 2D Layer
Medium
Deep Learning

In this problem, you need to implement a 2D convolutional layer in Python. This function will process an input matrix using a specified convolutional kernel, padding, and stride.

"""
import numpy as np 
def simple_conv(matrix, kernel,padding,stride):
    matrix=np.array(matrix)
    kernel=np.array(kernel)
    m_h,m_w=matrix.shape
    k_h,k_w=kernel.shape
    if padding>0:
        padded=np.pad(matrix,pad_width=padding, mode='constant',constant_values=0)
    else:
        padded=matrix.copy()
    output_h=(m_h+2*padding-k_h)//stride +1
    output_w=(m_w+2*padding-k_w)//stride+1
    output=np.zeros((output_h,output_w))
    for i in range(output_h):
        for j in range(output_w):
            row_start=i*stride
            col_start=j*stride
            region=padded[
                row_start:row_start+k_h,
                col_start:col_start+k_w]
            
            output[i,j]=np.sum(region*kernel)
    return output 
input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

output = simple_conv(input_matrix, kernel, padding, stride)
print(output)