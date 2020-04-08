# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 05:31:13 2020
    https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.3_linear-regression-pytorch
@author: bejin
"""

import torch
import numpy as np
import torch.nn as nn

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

import torch.utils.data as Data
batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break



# 3.3.3 定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
        
    def forward(self, x):
        y = self.linear(x)
        return y
    
net = LinearNet(num_inputs)
print(net)

# 写法一
net = nn.Sequential(
        nn.Linear(num_inputs, 1)
    )

'''
# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))

# 写法三
from  collections import OrderedDict
net = nn.Sequential(OrderedDict([
        ('linear', nn.Linear(num_inputs, 1))
    ]))
'''

print(net)
print(net[0])


for param in net.parameters():
    print(param)
    


# 3.3.4 初始化模型参数
'''
PyTorch在init模块中提供了多种参数初始化方法。
这里的init是initializer的缩写形式。
我们通过init.normal_将权重参数每个元素初始化为
随机采样于均值为0、标准差为0.01的正态分布。
偏差会初始化为零。
'''

from torch.nn import init
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0) # 也可以直接修改bias的data: net[0].bias.data.fill_(0)


loss = nn.MSELoss()


import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

'''
optimizer =optim.SGD([
                # 如果对某个参数不指定学习率，就使用最外层的默认学习率
                {'params': net.subnet1.parameters()}, # lr=0.03
                {'params': net.subnet2.parameters(), 'lr': 0.01}
            ], lr=0.03)
'''

# 调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
    

# 3.3.7 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))



dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)



































    