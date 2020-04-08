# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:42:34 2020

@author: bejin
"""


import torch
import numpy as np


# index 
x = torch.randn((5,3))
y = x[0, :]
y += 1

z= torch.empty(1,1)
torch.index_select(x, 0, 2, out = z)


# np.matmul(), *, np.multiply, np.dot 


# np.matmul   - broadcasting

# *  -  matice product

# np.multiply - element wise produt

# np.dot   - dot product, inner product




# 2.3.1

import torch
import numpy as np

x = torch.ones(2,2,requires_grad=True)
print(x)
print(x.grad_fn)


y = x + 2
print(y)
print(y.grad_fn)


print(x.is_leaf, y.is_leaf) 


z = y * y * 3
out = z.mean()
print(z, out)



a = torch.randn(2,2)
a = ((a * 3)/(a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)


# 2.3.2
out.backward()
print(x.grad)

#
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad = True)
y = 2*x
z = y.view(2,2)
print(z)

#现在 y 不是一个标量，所以在调用backward时需要传入一个和y同形的权重向量进行加权求和得到一个标量。
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad) # 注意，x.grad是和x同形的张量。


# 再来看看中断梯度追踪的例子:
x = torch.tensor(1.0, requires_grad = True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2

print(x.requires_grad)
print(y1, y1.requires_grad) # True
print(y2, y2.requires_grad) # False
print(y3, y3.requires_grad) # True


y3.backward()
print(x.grad)


# 此外，如果我们想要修改tensor的数值，但是又不希望被autograd记录（即不会影响反向传播），
# 那么我么可以对tensor.data进行操作。

x = torch.ones(1,requires_grad=True)

print(x.data) # 还是一个tensor
print(x.data.requires_grad) # 但是已经是独立于计算图之外

y = 2 * x
x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播

y.backward()
print(x) # 更改data的值也会影响tensor的值
print(x.grad)











































