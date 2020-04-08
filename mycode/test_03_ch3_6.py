# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 06:16:55 2020
    3.8 多层感知机
    https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.8_mlp
@author: bejin
"""

#%matplotlib inline

import torch
import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(12, 8))
    d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')


x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')


y.sum().backward()
xyplot(x, x.grad, 'grad of relu')


y = x.sigmoid()
xyplot(x, y, 'sigmoid')


x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')


y = x.tanh()
xyplot(x, y, 'tanh')


x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')



'''
3.9 多层感知机的从零开始实现
https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.9_mlp-scratch
'''

import torch
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)


params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)



def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))



def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2


loss = torch.nn.CrossEntropyLoss()



num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)



'''
3.10 多层感知机的简洁实现
https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.10_mlp-pytorch
'''

import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l



num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs), 
        )

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)



batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)













