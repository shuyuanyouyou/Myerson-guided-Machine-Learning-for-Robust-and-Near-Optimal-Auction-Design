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
    
@torch.no_grad()
def myerson_itemwise_allocation_payment(values, reserve=0.5):
    """
    values: [B, I, M]
    返回:
      alloc_myr: [B, I, M] （winner one-hot；最高价>=reserve 才成交）
      pay_myr:   [B, I]    （逐 item 清算价相加）
    清算价 = max(reserve, 次高价)。最高价<reserve 则不成交。
    """
    B, I, M = values.shape
    top2 = values.topk(k=2, dim=-2)
    v1, idx1 = top2.values[:, 0, :], top2.indices[:, 0, :]
    v2 = top2.values[:, 1, :]
    win = (v1 >= reserve).float()
    price = torch.maximum(v2, torch.full_like(v2, float(reserve))) * win  # [B,M]
    alloc = torch.zeros(B, I, M, device=values.device)
    alloc.scatter_(1, idx1.unsqueeze(1), win.unsqueeze(1))
    pay_myr = (alloc * price.unsqueeze(1)).sum(-1)  # [B,I]
    return alloc, pay_myr
class MixedWrapper(torch.nn.Module):
    """
    与原机制同签名的薄包装器：
      forward(vals) -> (alloc_mix, pay_mix, pp)
    规则：对每个样本，比较 Net vs Myerson 的总收益，谁高用谁（硬选择）。
    训练期可开启直通：前向硬选，反向只回到 Net（Myerson不更新）。
    """
    def __init__(self, base_mech, reserve=0.5, straight_through=True):
        super().__init__()
        self.base = base_mech
        self.reserve = float(reserve)
        self.st = bool(straight_through)

    def forward(self, vals):
        alloc_net, pay_net = self.base(vals)                   # [B,I,M], [B,I]
        alloc_myr, pay_myr = myerson_itemwise_allocation_payment(vals, reserve=self.reserve)

        rev_net = pay_net.sum(dim=-1)                              # [B]
        rev_myr = pay_myr.sum(dim=-1)                              # [B]
        choose_net = (rev_net > rev_myr)                           # [B]
        mA = choose_net.view(-1,1,1); mP = choose_net.view(-1,1)

        alloc_hard = torch.where(mA, alloc_net, alloc_myr)
        pay_hard   = torch.where(mP, pay_net,   pay_myr)

        if not self.st:
            return alloc_hard, pay_hard  # 评估/测试：纯硬选择

        # 训练：直通估计器（保持硬选前向，梯度只回 Net）
        alloc_mix = alloc_hard.detach() + alloc_net - alloc_net.detach()
        pay_mix   = pay_hard.detach()   + pay_net   - pay_net.detach()
        return alloc_mix, pay_mix
    