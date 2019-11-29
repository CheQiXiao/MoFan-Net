#!/user/bin/env python
# -*- coding:utf-8 -*-

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# 画图
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x=F.relu(self.hidden(x))#隐藏层线性输出
        x=self.predict(x)#输出层线性输出
        return x
net=Net(n_feature=1,n_hidden=10,n_output=1)
print(net)#net的结构


#训练网络
#optimizer是训练的工具
optimizer=torch.optim.SGD(net.parameters(),lr=0.2)#SGD是随机梯度下降，Ir学习率，传入 net 的所有参数, 学习率
loss_func=torch.nn.MSELoss()# # 预测值和真实值的误差计算公式 (均方差)
#MSELoss（）多用于回归问题，也可以用于one_hotted编码形式，
plt.ion()

for t in range(200):
    prediction=net(x) #喂给net训练数据x,输出预测值
    loss=loss_func(prediction,y)#预测值和标签值，计算出损失值

    optimizer.zero_grad()#清空上一步残余
    loss.backward() #反向传播，计算参数更新值
    optimizer.step()#将参数更新值施加到net的parameters上

    if t%5==0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
