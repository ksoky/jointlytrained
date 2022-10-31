#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Kak Soky, SAP, Kyoto University
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Gradient Reversal Layer
     @InProceedings{pmlr-v37-ganin15,
     title = {Unsupervised Domain Adaptation by Backpropagation},
     author = {Yaroslav Ganin and Victor Lempitsky},
     pages = {1180--1189},
     year = {2015},
     editor = {Francis Bach and David Blei},
     volume = {37}, series = {Proceedings of Machine Learning Research},
     address = {Lille, France}, month = {07--09 Jul}, publisher = {PMLR},
     pdf = {http://proceedings.mlr.press/v37/ganin15.pdf}, url = {http://proceedings.mlr.press/v37/ganin15.html}
     } 
"""

import torch

class GradReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        #return grad_output * 0.2
        return grad_output.neg()
        

def grad_reverse(x):
    return GradReverseLayer.apply(x)

#def grad_reverse(x):
#    return GradReverseLayer()(x)