#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Hang Le
# hangtp.le@gmail.com

"""Dual-decoder self-attention layer definition."""
import logging
import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DualDecoderLayer(nn.Module):
    """Single decoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention src_attn: source attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward layer module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(self, size, sre, 
                 self_attn, src_attn, feed_forward, 
                 cross_self_attn, 
                 cross_src_attn,
                 dropout_rate, 
                 normalize_before=True, 
                 concat_after=False, 
                 cross_operator="concat",
                 cross_weight_learnable=False, 
                 cross_weight=0.0,
                 cross_to_asr=True):
        """Construct an DecoderLayer object."""
        super(DualDecoderLayer, self).__init__()

        #SRE
        self.idim = size #output from the encoder
        self.sre = sre #the number of output, i.e. 6 for speaker in ECCC Khmer data, 2 for gender, and 2 for age.
        # Linears for Speaker Detection
        #self.lin_sp = torch.nn.Linear(self.idim, self.odim)
        #self.lin_sp_hidden = torch.nn.Linear(self.odim, self.odim)
        #self.relu = torch.nn.ReLU()
        #ASR
        self.size = size
        #self.size_sre = size_sre

        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        self.cross_self_attn = cross_self_attn
        self.cross_src_attn = cross_src_attn
        
        self.cross_to_asr = cross_to_asr
        
        self.cross_operator = cross_operator
        if cross_operator == "concat":
            if cross_self_attn is not None: 
                self.cross_concat_linear1 = nn.Linear(size + size, size)
            if cross_src_attn is not None:
                self.cross_concat_linear2 = nn.Linear(size + size, size)
        elif cross_operator == "sum":
            if cross_weight_learnable:
                assert float(cross_weight) > 0.0
                if self.cross_to_asr:
                    self.cross_weight = torch.nn.Parameter(torch.tensor(cross_weight))
            else:
                if self.cross_to_asr:
                    self.cross_weight = cross_weight

        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)

        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, cross_self=False, cross_src=False, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out, max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, 1, max_time_in)
            cache (torch.Tensor): cached output (batch, max_time_out-1, size)
            cross (torch.Tensor): decoded previous target from another decoder (batch, max_time_out, size)
        """

        #SRE
        #out_sp = torch.mean(tgt, dim=1) # (B x 2*HIDDEN_SIZE)
        #out_sp = self.lin_sp(out_sp) # (B x 512)
        #out_sp = self.relu(out_sp)
        #out_sp = self.lin_sp_hidden(out_sp) # (B x SPEAKER_NUM)
        
        #out = F.softmax(out, dim=-1) 
        #ASR
        residual = tgt
        
        print(cache)
        if self.normalize_before:
            tgt = self.norm1(tgt)
        
        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (tgt.shape[0], tgt.shape[1] - 1, self.size), \
                f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        
        

        # Self-attention
        if self.concat_after:
            tgt_concat = torch.cat((tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1)
            x = self.concat_linear1(tgt_concat)
        else:
            x = self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        
        # Cross-self attention
        if cross_self: # and cross_self_from == "before-self":
            #if self.cross_to_asr:
            z = self.dropout(self.cross_self_attn(tgt_q, self.sre, self.sre))
            #z = self.dropout(self.cross_self_attn(tgt_q, out_sp, out_sp, cross_mask))
            if self.cross_operator == 'sum':
                x = x + self.cross_weight * z
            elif self.cross_operator == 'concat':
                x = self.cross_concat_linear1(torch.cat((x, z), dim=-1))
            else:
                raise NotImplementedError

        x = x + residual
        
        if not self.normalize_before:
            x = self.norm1(x)
            
        # Source attention
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        y = x
        
        if self.concat_after:
            x_concat = torch.cat((x, self.src_attn(x, memory, memory, memory_mask)), dim=-1)
            x = self.concat_linear2(x_concat)
        else:
            x = self.dropout(self.src_attn(x, memory, memory, memory_mask))
            
        # Cross-source attention
        if cross_src: #and cross_src_from == "before-src":
            #if self.cross_to_asr:
            z = self.dropout(self.cross_src_attn(y, self.sre, self.sre))
            if self.cross_operator == 'sum':
                x = x + self.cross_weight * z
            elif self.cross_operator == 'concat':
                x = self.cross_concat_linear2(torch.cat((x, z), dim=-1))
            else:
                raise NotImplementedError
        
        x = x + residual
        
        if not self.normalize_before:
            x = self.norm2(x)
            
        # Feed forward
        residual = x
        
        if self.normalize_before:
            x = self.norm3(x)

        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)
        
        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        
        return x, tgt_mask, self.sre, \
                memory, memory_mask, \
                cross_self, cross_src