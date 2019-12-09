#!/user/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.utils.data as Data

torch.manual_seed(1)
BATCH_SIZE = 8  # 批处理数据5个5个的来
x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)  # torch_dataset 所建立的数据库
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 在epoch过程中要不要打乱数据
    num_workers=2,  # 提取batch_x,batch_y时多线程来读数据
)


def show_batch():


    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            # training...
            print('Epoch:', epoch, '| Step:', step, '|batch x:', batch_x.numpy(), '|batch y:', batch_y.numpy())

if __name__=='__main__':
    show_batch()