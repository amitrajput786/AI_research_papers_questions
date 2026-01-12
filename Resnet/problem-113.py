#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 22:48:23 2026

@author: amit
"""
import numpy as np 
def residual_block(x,w1,w2,):
    x=np.array(x)
    w1=np.array(w1)
    w2=np.array(w2)
    l=w1@(x)
    z=np.maximum(0,l)
    l2=w2@(z)
    a=l2+x
    z=np.maximum(0,a)
    return z
x=[[1,1],[2,2]]
w1=[[1,0],[0,2]]
w2=[[0,1],[3,0]]
print(residual_block(x,w1,w2))
