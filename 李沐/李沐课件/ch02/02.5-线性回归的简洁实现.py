import random
from d2l import torch as d2l
import  torch
import numpy as np
from torch.utils import data    #一些处理数据的模块
from torch import nn

true_w=torch.tensor([2,-3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
#直接在d2l里面 给你创建数据的函数

d2l.plt.scatter(features[:,1].detach().numpy(), #detach().numpy(),在某些版本里面,数据要先detach才能转到numpy里面
                    labels.detach().numpy(),1)  #lebels()用于画图,数据也是如上所述,1为圆点直径
d2l.plt.show()
#和前面一样

def load_array(data_arrays,batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    #＊号在python 语言中调用函数时参数前使用表示解包
    #列表前加星号是把列表元素分别当做参数穿进去
    return data.DataLoader(dataset,batch_size,shuffle = is_train)
#代替read_data函数

batch_size = 10
data_iter=load_array((features,labels),batch_size)

next(iter(data_iter) )
#一般来说，list，tuple多为可迭代对象，为了访问里面的元素，可以使用iter()，再通过next()来获取下一个元素。

net=nn.Sequential(nn.Linear(2,1))#使用全连接层,输入为2,输出为1

net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

loss=nn.MSELoss()#均方差作为误差

trainer=torch.optim.SGD(net.parameters(),lr=0.03)
#SGD随机梯度下降,net.parameters() net里面所有的参数,learning_rate=0.03

num_epochs= 3

for epoch in range(num_epochs):
    for X, y in (data_iter):
        l = loss (net(X), y)   #net自带了模型参数,不再需要w和b

        trainer.zero_grad() #给优化器清零
        l.backward()    #直接求backward
        trainer.step()  #用step更新模型(之前sgd函数里面有这步)

    with torch.no_grad():
        train_l = loss(net(features), labels)
        print(f'epoch{epoch +1},loss{l:f}')
    #{l : f} -> 打印l，格式为浮点型


