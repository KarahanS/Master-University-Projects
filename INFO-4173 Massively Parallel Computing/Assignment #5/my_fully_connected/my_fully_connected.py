#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
import os
import torch
from torch.utils.cpp_extension import load
import numpy as np

# Import the exported cuda functions into the center_surround module
dirname = os.path.dirname(os.path.relpath(__file__))
sources = [os.path.join(dirname, f) for f in
        ["my_fully_connected.cu"]]
my_fully_connected = load(name="my_fully_connected",
        sources=sources,
        verbose=True)


# Create a binding to pytorch
# This is the center surround function with the definition of its derivative:
class myfullyconnected(torch.autograd.Function):

    # Forward pass
    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor) -> torch.Tensor:
        outputs = my_fully_connected.forward(input, weight, bias) # Use the C++-based module to compute the function's output
        ctx.save_for_backward(input, weight, bias)                # Remember own inputs in context for the backward pass

        return outputs[0]

    # Backward pass
    @staticmethod
    def backward(ctx, dL_doutput: torch.Tensor) -> Tuple[torch.Tensor]:
        input, weight, bias = ctx.saved_tensors                   # Retrieve inputs for which a derivative should be computed from ctx
        # Use the C++-based module to compute all input derivatives
        dL_dinput, dL_dweight, dL_dbias = my_fully_connected.backward(dL_doutput, input, weight, bias)

        return dL_dinput, dL_dweight, dL_dbias


# This is the layer that you use in a network
class MyFullyConnected(torch.nn.Module):
    def __init__(self, in_chanels: int, out_chanels: int):
        super(MyFullyConnected, self).__init__()                                 # Initialize the base class
        self.weight = torch.nn.Parameter(torch.empty([in_chanels, out_chanels])) # Initialize the parameter weights
        self.bias = torch.nn.Parameter(torch.empty([out_chanels]))               # Initialize the bias weights
        self.reset_parameters()                                                  # Reset all parameters (see below)

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, a=-0.02, b=0.02)                     # Initialize weights uniformly in [-0.02, 0.02]
        torch.nn.init.uniform_(self.bias, a=-0.02, b=0.02)                       # Initialize biases uniformly in [-0.02, 0.02]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # In the forward pass, our autograd function is used to compute the layers output from input, weights and biases
        return myfullyconnected.apply(input, self.weight, self.bias)


class Net(torch.nn.Module):
    def __init__(self, I):
        super(Net, self).__init__()
        # Define all submodules here
        #self.fc = torch.nn.Linear(I, 10)
        self.fc = MyFullyConnected(I, 10)
        self.sm = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.sm(x)
        return x
    
    def num_flat_features(self, x):
        return x.size(1)


def main():
    # Get GPU device
    device = torch.device("cuda")
    
    # Set sizes:
    # Number of all samples
    NN = 60000
    # Batch size
    N = 32
    # Input size of one sample
    I = 28*28
    
    # Load all the inputs and labels from disk
    all_input_raw = np.fromfile("train-images-idx3-ubyte", dtype=np.uint8, count=NN*I, offset=16)
    all_labels_raw = np.fromfile("train-labels-idx1-ubyte", dtype=np.uint8, count=NN, offset=8)
    all_input = all_input_raw.reshape([NN, I])
    all_input = all_input.astype(np.single) / 255.0
    all_labels = np.zeros([NN, 10], dtype=np.single)
    for i in range(NN):
        all_labels[i, all_labels_raw[i]] = 1.0
    
    # Setup network modules
    net = Net(I)
    net.to(device)
    bcel = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    # Training
    for it in range(int(NN / N)):
        optimizer.zero_grad()
        
        input = torch.from_numpy(all_input[it*N:(it+1)*N,:]).to(device)
        target = torch.from_numpy(all_labels[it*N:(it+1)*N,:]).to(device)
        prediction = net(input)
        loss = bcel(prediction, target)
        loss.backward()
        
        optimizer.step()
        
        print("Loss: {}".format(loss))
    
    print("Done")

if __name__ == '__main__':
    main()
