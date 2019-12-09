#!/user/bin/env python
# -*- coding:utf-8 -*-
import torch
import torchvision
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data

EPOCH=1
BATCH_SIZE=64
TIME_STEP=28  #时间步数/图片高度
INPUT_SIZE=28  #每步输入的值，图片每行像素
LR=0.01
DOWNLOAD_MNIST=False

train_data=dsets.MNIST(
    root='./mnist',train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST)
train_loader=torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True  #每个epoch开始的时候，对数据进行重新排序
)

# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# test_y = test_data.test_labels.numpy()[:2000]
test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy()[:2000]    # covert to numpy array

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.LSTM(
            input_size=INPUT_SIZE,
             hidden_size=64,
            num_layers=1,# 有几层 RNN layers
            batch_first=True # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)  当(time_step，batch,input_size)时batch_first=FALSE
        )
        self.out=nn.Linear(64,10)
    def forward(self, x):
        r_out,(h_n,h_c)=self.rnn(x,None)#h_n 是分线, h_c 是主线
        out=self.out(r_out[:,-1,:])        # 选取最后一个时间点的 r_out 输出. 这里 r_out[:, -1, :] 的值也是 h_n 的值
        return out
rnn=RNN()
print(rnn)

optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        b_x=b_x.view(-1,28,28)
        output=rnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if step % 50 == 0:
            test_output = rnn(test_x)  # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

#测试
test_output=rnn(test_x[:10].view(-1,28,28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y,'prediction number')
print(test_y[:10],'real number')