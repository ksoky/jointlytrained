#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Kak Soky, SAP, Kyoto University
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from espnet.nets.pytorch_backend.adversarial.grl import grad_reverse


class Adversarial(nn.Module):

    def __init__(self,idim, odim):
        super(Adversarial, self).__init__()
        self.idim = idim #output from the encoder
        self.odim = odim #the number of output, i.e. 6 for speaker in ECCC Khmer data, 2 for gender, and 2 for age.

        # Linears for Speaker Detection
        self.lin_sp = torch.nn.Linear(self.idim, self.odim)
        self.lin_sp_hidden = torch.nn.Linear(self.odim, self.odim)
        self.relu = torch.nn.ReLU()
        #self.grl = GradReverseLayer.apply

    def forward(self, hx):
        
        # for Speaker Adversarial Training
        out = torch.mean(hx, dim=1) # (B x 2*HIDDEN_SIZE)
        #out = torch.sigmoid(hx.view(1))
        # mean pooling. you can change here to any pooling style as you like
        # Here Gradients are reversed
        #out = grad_reverse(out)
        out = self.lin_sp(out) # (B x 512)
        out = self.relu(out)
        out = self.lin_sp_hidden(out) # (B x SPEAKER_NUM)
        #out = F.softmax(out, dim=-1)
        
        return out
