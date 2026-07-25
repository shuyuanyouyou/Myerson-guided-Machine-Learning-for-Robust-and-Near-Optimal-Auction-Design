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
from restrictedAdam import Adam

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

def utility(valuations, allocation, pay):
    """ Given input valuation , payment  and allocation , computes utility
            Input params:
                valuation : [num_batches, num_agents, num_items]
                allocation: [num_batches, num_agents, num_items]
                pay       : [num_batches, num_agents]
            Output params:
                utility: [num_batches, num_agents]
    """
    return (torch.sum(valuations*allocation, dim=-1) - pay)


def misreportUtility(mechanism,batchTrueValuations,batchMisreports):
    """ This function takes the valuation and misreport batches
        and returns a tensor constaining all the misreported utilities

        Input :
            batchTrueValuations is a [num_batches, num_agents, num_items] tensor
            batchMisreports     is a [num_batches, num_initializations, num_agents, num_items] tensor
        Output:
            advUtility is a [num_batches, num_initializations, num_agents] tensor

        advUtility[b,k,i] = utility of bidder i in batch b when the valuations
                            of bidder i are replaced by the misreport for the kth initialization


    """
    nAgent             = batchTrueValuations.shape[-2]
    nObjects           = batchTrueValuations.shape[-1]
    batchSize          = batchTrueValuations.shape[0]
    nbrInitializations = batchMisreports.shape[1]

    V  = batchTrueValuations.unsqueeze(1)
    V  = V.repeat(1,nbrInitializations, 1, 1)
    V  = V.unsqueeze(0)
    V  = V.repeat(nAgent, 1, 1, 1, 1)

    M  = batchMisreports.unsqueeze(0)
    M  = M.repeat(nAgent,1, 1, 1, 1)


    mask1                                           = np.zeros((nAgent,nAgent,nObjects))
    mask1[np.arange(nAgent),np.arange(nAgent),:]    = 1.0
    mask2                                           = np.ones((nAgent,nAgent,nObjects))
    mask2                                           = mask2-mask1

    mask1       = (torch.tensor(mask1).float()).to(device)
    mask2       = (torch.tensor(mask2).float()).to(device)

    V  = V.permute(1, 2, 0, 3, 4)
    M  = M.permute(1, 2, 0, 3, 4)

    tensor      =  M*mask1 + V*mask2

    tensor      = tensor.permute(2, 0, 1, 3, 4)

    V  = V.permute(2, 0, 1, 3, 4)
    M  = M.permute(2, 0, 1, 3, 4)

    tensor = View(-1,nAgent, nObjects)(tensor)

    allocation, payment = mechanism(tensor)

    allocation    =  View(nAgent,batchSize,nbrInitializations,nAgent, nObjects)(allocation)
    payment       =  View(nAgent,batchSize,nbrInitializations,nAgent)(payment)

    advUtilities    = torch.sum(allocation*V, dim=-1)-payment

    advUtility      = advUtilities[np.arange(nAgent),:,:,np.arange(nAgent)]

    return(advUtility.permute(1, 2, 0))


def misreportOptimization(mechanism,batch,trueValuations, misreports, R, gamma, minimum=0, maximum=1):

    """ This function takes the valuation and misreport batches
        and R the number of optimization step and modifies the misreport array

        Input :
            batch which is a numpy array of indices
        Output:
            no Output, just modifies the array

        """
    localMisreports     = misreports[:]
    batchMisreports     = torch.tensor(misreports[batch]).float().to(device)
    batchTrueValuations = torch.tensor(trueValuations[batch]).float().to(device)
    batchMisreports.requires_grad = True

    opt = Adam([batchMisreports], lr=gamma)

    for k in range(R):
        advU         = misreportUtility(mechanism,batchTrueValuations,batchMisreports)
        loss         =  -1*torch.sum(advU).to(device)
        loss.backward()
        opt.step(restricted= True, min=minimum, max=maximum)
        opt.zero_grad()

    mechanism.zero_grad()

    localMisreports[batch,:,:,:] = batchMisreports.cpu().detach().numpy()
    return(localMisreports)

def trueUtility(mechanism,batchTrueValuations):

    """ This function takes the valuation batches
        and returns a tensor constaining the utilities

        Input :
            batchTrueValuations is a [num_batches, num_agents, num_items] tensor
        Output:
            trueUtility is a [num_batches, num_agents]
    """

    allocation, payment = mechanism(batchTrueValuations)

    return utility(batchTrueValuations, allocation, payment)

def loss(payment, regret):
    """
    This function tackes a batch which is a numpy array of indices and computes
    the loss function                                                             : los
    the average regret per agent which is a tensor of size [nAgent]               : rMean
    the average revenue  which is a tensor of size [1]                            : -paymentLoss
    """
    batchSize            = payment.shape[0]
    paymentLoss          = -torch.sum(payment)/batchSize

    rMean                = torch.sum(torch.mean(regret, dim=0)).to(device)

    los = -1*(torch.sqrt(-paymentLoss + 1e-20) - torch.sqrt(rMean + 1e-20)) + rMean

    return(los, rMean, -paymentLoss)
