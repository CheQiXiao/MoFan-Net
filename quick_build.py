#!/user/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

#method 1
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.output=torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.output(x)
        return x
net1=Net(1,10,1)


#method2(代码简单)
net2=torch.nn.Sequential(   #直接向神经网络nn中叠加
    torch.nn.Linear(1,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,1)
)
print(net1)
print(net2)