#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 22:34:52 2026

@author: amit
"""
import numpy as np
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

def simple_conv2d(input_matrix,kernel,padding,strides):
    matrix=np.array(input_matrix)
    kernel=np.array(kernel)
    input_hieght,input_width=matrix.shape
    kernel_hiegth,kernel_width=kernel.shape
    if padding>0:
        padded_input=np.pad(
            input_matrix,pad_width=padding,mode='constant',
            constant_values=0)
    else:
        padded_input=input_matrix.copy()
    output_hieght=(input_hieght + 2*padding-kernel_hiegth)//stride +1
    output_width=(input_hieght + 2*padding-kernel_width)//stride +1
    output=np.zeros((output_hieght,output_width))
    for i in range(output_hieght):
        for j in range(output_width):
            row_start=i*stride
            col_start=j*stride
            region=padded_input[
                row_start:row_start+kernel_hiegth,
                col_start:col_start+kernel_width]
            output[i,j]=np.sum(region*kernel)
    return output

output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)