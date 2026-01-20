#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 20:18:51 2026

@author: amit
"""



"""Implement Self-Attention Mechanism
Medium
Deep Learning

Implement the self-attention mechanism, a fundamental component of transformer models used in NLP and computer vision.

Your task is to implement the self_attention function that computes attention output given Query (Q), Key (K), and Value (V) matrices.

The self-attention formula is: Attention(Q, K, V) = softmax(Q Ã K^T / sqrt(d_k)) Ã V

where d_k is the dimensionality of the key vectors (number of columns in K).

Input:

    Q: Query matrix of shape (seq_len, d_k)
    K: Key matrix of shape (seq_len, d_k)
    V: Value matrix of shape (seq_len, d_v)

Output:

    Attention output matrix of shape (seq_len, d_v)

Steps:

    Compute attention scores: scores = Q Ã K^T / sqrt(d_k)
    Apply softmax row-wise to get attention weights
    Compute output: output = attention_weights Ã V

Note: The helper function compute_qkv is provided to compute Q, K, V from input X and weight matrices."""





import numpy as np

Q = np.array([[1, 0], [0, 1]])
K = np.array([[1, 0], [0, 1]])
V = np.array([[1, 2], [3, 4]])


def compute_qkv(X,W_q,W_k,W_v):
    Q=X@(W_q)
    K=X@(W_k)
    V=X@(W_v)
    return Q,K,V
def softmax(x):
    exp_x=np.exp(x-np.max(x,axis=-1,keepdims=True))
    return exp_x/np.sum(exp_x,axis=-1,keepdims=True)

def self_attention(Q, K, V):
    
    d_k=K.shape[-1]
    scores=Q@(K.T)
    scaled_scores=scores/np.sqrt(d_k)
    attention_weights=softmax(scaled_scores)
    attention_output=attention_weights@(V)
    return attention_output
output = self_attention(Q, K, V)
print(output)