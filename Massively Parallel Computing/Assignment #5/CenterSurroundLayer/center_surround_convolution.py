#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
import os
import torch
# import center_surround_cuda as csc
from torch.utils.cpp_extension import load

# Use the Just In Time compiler from pytorch to load the module that you
# exported from c++.
csc = load(name="center_surround_cuda",
           sources=["center_surround_convolution.cu"],
           extra_cuda_cflags=["-lineinfo", "--resource-usage"],
           verbose=True)

# Load your the exported python module in center surround convolution.py and
# implement the torch.autograd.Function class center surround convolution.
class center_surround_convolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                I: torch.Tensor,
                w_c: torch.Tensor,
                w_s: torch.Tensor,
                w_b: torch.Tensor) -> torch.Tensor:
        outputs = csc.forward(I, w_c, w_s, w_b)
        output = outputs[0]
        ctx.save_for_backward(I, w_c, w_s, w_b)
        return output

    @staticmethod
    def backward(ctx, dL_dO: torch.Tensor) -> Tuple[torch.Tensor]:
        I, w_c, w_s, w_b = ctx.saved_tensors
        dL_dI, dL_dw_c, dL_dw_s, dL_dw_b = csc.backward(dL_dO, I, w_c, w_s, w_b)
        return dL_dI, dL_dw_c, dL_dw_s, dL_dw_b


# f) In the same file create a new torch.nn.Module called
# CenterSurroundConvolution which can be used as layer in a neural network.
class CenterSurroundConvolution(torch.nn.Module):
    def __init__(self, in_chanels: int, out_chanels: int):
        super(CenterSurroundConvolution, self).__init__()
        self.w_c = torch.nn.Parameter(torch.empty([in_chanels, out_chanels]))
        self.w_s = torch.nn.Parameter(torch.empty([in_chanels, out_chanels]))
        self.w_b = torch.nn.Parameter(torch.empty([out_chanels]))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights
        # torch.nn.init.uniform_(self.w_c, -0.5, 0.5)
        # torch.nn.init.uniform_(self.w_s, -0.05, 0.05)
        torch.nn.init.xavier_normal_(self.w_c)
        torch.nn.init.xavier_normal_(self.w_s, gain=0.125)
        torch.nn.init.zeros_(self.w_b)

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        return center_surround_convolution.apply(I, self.w_c, self.w_s, self.w_b)
