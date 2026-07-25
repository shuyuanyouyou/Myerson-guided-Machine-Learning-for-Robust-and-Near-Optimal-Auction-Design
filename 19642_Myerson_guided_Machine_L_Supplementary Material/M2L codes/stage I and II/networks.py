import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import torch.optim as optim
from torch.optim import Optimizer
import time
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VectView(nn.Module):
    def __init__(self, nAgent, nObject):
        super(VectView, self).__init__()
        self.nAgent  = nAgent
        self.nObject = nObject
    def forward(self, input):
        l = list(input.shape)[:-2]
        l.append(self.nAgent*self.nObject)
        return input.view(* tuple(l))

class MatrixView(nn.Module):
    def __init__(self, nAgent, nObject):
        super(MatrixView, self).__init__()
        self.nAgent  = nAgent
        self.nObject = nObject
    def forward(self, input):
        l = list(input.shape)[:-1]
        l.append([self.nAgent,self.nObject])
        return input.view(* tuple(l))

class MatrixToTensor(nn.Module):
    def __init__(self, nAgent, nObject):
        super(MatrixToTensor, self).__init__()
        self.nAgent  = nAgent
    def forward(self, input):
        input = input.unsqueeze(1)
        input = input.repeat(1,self.nAgent,1,1)

        for i in range(self.nAgent):
            p    = list(range(self.nAgent))
            p[0] = i
            p[i] = 0
            input[:,i,:,:] = input[:,i,np.array(p),:]

        return input

class MLP(nn.Module):
    def __init__(self, dimInput, dimOutput, nLayers, width):
        super(MLP, self).__init__()

        self.dimInput    = dimInput
        self.dimOutput   = dimOutput
        self.nLayers     = nLayers
        self.width       = width

        self.layers                               = collections.OrderedDict()

        self.layers["fc1"]                        = nn.Linear(dimInput, width).to(device)
        torch.nn.init.xavier_uniform_(self.layers["fc1"].weight, gain=nn.init.calculate_gain('tanh'))

        for i in range(2,nLayers):
            self.layers["tanh"+str(i-1)]                      = nn.Tanh().to(device)
            self.layers["fc"+str(i)]                          = nn.Linear(width, width).to(device)
            torch.nn.init.xavier_uniform_(self.layers["fc"+str(i)].weight, gain=nn.init.calculate_gain('tanh'))


        self.layers["tanh"+str(nLayers-1)]        = nn.Tanh().to(device)
        self.layers["fc"+str(nLayers)]            = nn.Linear(width, dimOutput).to(device)
        torch.nn.init.xavier_uniform_(self.layers["fc"+str(nLayers)].weight, gain=1)

        self.model                      = nn.Sequential(self.layers).to(device)

    def forward(self, input):
        input = input.to(device)
        return(self.model(input))


class AllocationNet(nn.Module):
    def __init__(self, nAgent, nObject, nLayer, width):
        super().__init__()

        self.nAgent     = nAgent
        self.nObject    = nObject
        self.nLayer     = nLayer
        self.width      = width

        self.layersAllocation                     = collections.OrderedDict()
        self.layersAllocation["tensorize"]        = MatrixToTensor(nAgent, nObject)
        self.layersAllocation["view1"]            = VectView(nAgent, nObject)
        self.layersAllocation["MLP"]              = MLP(nAgent*nObject, nObject, nLayer, width)
        self.layersAllocation["Softmax"]          = nn.Softmax(dim=-2)

        self.modelAllocation                      = nn.Sequential(self.layersAllocation).to(device)

        self.layersProbability                    = collections.OrderedDict()
        self.layersProbability["view1"]           = VectView(nAgent, nObject)
        self.layersProbability["MLP"]             = MLP(nAgent*nObject, nObject, nLayer, width)
        self.layersProbability["Softmax"]         = nn.Sigmoid()

        self.modelProbability                     = nn.Sequential(self.layersProbability).to(device)



    def forward(self, input):
        input = input.to(device)
        allocation = self.modelAllocation(input)*self.modelProbability(input).unsqueeze(1)
        return allocation.to(device)
    
class PaymentNet(nn.Module):
    def __init__(self, nAgent, nObject, nLayer, width):
        super().__init__()

        self.nAgent     = nAgent
        self.nObject    = nObject
        self.nLayer     = nLayer
        self.width      = width


        self.layersPayment                        = collections.OrderedDict()
        self.layersPayment["tensorize"]           = MatrixToTensor(nAgent, nObject)
        self.layersPayment["view1"]               = VectView(nAgent, nObject)
        self.layersPayment["MLP"]                 = MLP(nAgent*nObject, nObject, nLayer, width)
        self.layersPayment["Softmax"]             = nn.Sigmoid()

        self.modelPayment                         = nn.Sequential(self.layersPayment).to(device)

    def forward(self, input, allocation):
        input = input.to(device)
        allocation=allocation.to(device)
        pp=self.modelPayment(input)
        payment    = allocation*input*pp
        return payment.to(device)

class AdditiveMechanism(nn.Module):
    """
    一个薄封装：把两个子网串起来，得到:
      alloc_prob (B,A,O)
      payments   (B,A)
      pp         (B,O)
    """
    def __init__(self, nAgent, nObject, nLayer, width):
        super().__init__()
        self.alloc_net  = AllocationNet(nAgent, nObject, nLayer, width).to(device)
        self.payment_net= PaymentNet(nAgent, nObject, nLayer, width).to(device)
        self.device = device

    def forward(self, values):
        values = values.to(self.device)               # (B,A,O)
        alloc_prob = self.alloc_net(values)     # (B,A,O), (B,A,O), (B,O)
        payments = self.payment_net(values, alloc_prob)  # (B,O), (B,A)
        return alloc_prob.to(device), torch.sum(payments,dim=-1).to(device)
    
class Misreports(nn.Module):

    def __init__(self, nAgent, nObject, nLayers, width):

        super(Misreports, self).__init__()

        self.nLayers                              = nLayers
        self.width                                = width
        self.nAgent                               = nAgent
        self.nObject                              = nObject

        self.layers                               = collections.OrderedDict()
        self.layers["tensorize"]                  = MatrixToTensor(nAgent, nObject)
        self.layers["view1"]                      = VectView(self.nAgent, self.nObject)
        self.layers["MLP"]                        = MLP(nAgent*nObject, nObject, nLayers, width)
        # self.layers["Sigmoid"]                    = nn.Sigmoid()

        self.model                                = nn.Sequential(self.layers).to(device)

    def forward(self, input):
        input = input.to(device)
        return nn.Sigmoid()(self.model(input))
