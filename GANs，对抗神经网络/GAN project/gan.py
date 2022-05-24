import torch
from torch import nn,optim,autograd #导入autograd手动求导
import numpy as np
#import  visdom
import  random


h_dim = 400
batchsz = 512   #数据量少，大一点也没关系
#viz = visdom.Visdom()


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            #z的输入维度是[b,2]（自己设置的噪声的维度）
            nn.Linear(2, h_dim),
            #hidden layer dimention，隐藏层大小
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2),
            # z的输出维度是[b,2]（学习分布的真实维度）
        )

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        #判别器的输入=生成器的输出[b,2]
        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
            #输出是概率,即[b,1]，用Sigmoid()来使得概率分布为(0或1)
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)

def data_generator():
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    # 8个高斯分布的均值点
    centers = [(scale * x, scale * y) for x, y in centers]
    #(在（0，1）分布中放缩一下)
    while True:
        dataset=[]
        for i in range(batchsz):

            point=np.random.rand(2)*0.02
            center= random.choice(centers)#在8个点里面随便选一个
            # N（0，1）+center_x1/x2

            point[0]+=center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset=np.array(dataset).astype(np.float32)
        dataset/=1.414#放缩一下，使得值在

        yield dataset
        #循环迭代生成器，每一次运行后都保存状态，下一次运行又继续有现在的状态

def main():
    torch.manual_seed(23)
    np.random.seed(23)
    #固定种子，便于测试结果，复现过程
    data_iter=data_generator()
    x=next(data_iter)
    #[b,2]
    #print(x)

    G=Generator().cuda()
    D=Discriminator().cuda()

    #
    #两个优化器
    # Model.parameters()    用于获取模型参数
    optim_G= optim.Adam(G.parameters(),lr=5e-4,betas=(0.5,0.9))
    optim_D = optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))
    #

    #输出看看网络结构
    # print(G)
    # print(D)

    # TODO：GAN核心：对抗
    for epoch in  range(50000):
        #对D和G交换change直至平衡,先D后G


        #1. train D first
        for _ in range(5):

            # 1.1 train on real data
            x=next(data_iter)#读取真实数据xr
            xr=torch.from_numpy(x).cuda()
            #简单说一下，就是torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。,
            #然后再放进cuda里面

            #[b,2]=>[b,1]
            predr=(D(xr))

            # 1.2 真实数据的输出,lossr当然是越大越好,又因为这里用的是梯度下降,所以loss取负值
            lossr=-(torch.mean(predr))

            # 1.2 train on generate data
            # 用随机数z和伪造函数生成虚构数据xf
            z=torch.randn(batchsz,2).cuda()
            xf=G(z).detach()
            # TODO 使用detach能截至反向传播时(这里训练的是D不是G),等价于tf.stop_gradient()
            # xf是假数据

            # 简单说一下，就是torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。,
            # 然后再放进cuda里面

            # [b,2]=>[b,1]
            predf = D(xf)
            #  假数据的输出,lossf当然是越小越好,所以
            lossf = torch.mean(predf)
            # aggregate all
            loss_D=lossr+lossf
            # optimize
            #梯度清零（同时也清零下面tarin G产生的梯度）
            optim_D.zero_grad()
            #反向传播
            loss_D.backward()
            #修正参数
            optim_D.step()


        # 2. train G
        z = torch.randn(batchsz, 2).cuda()
        xf = G(z)
        predf = (D(xf)) #这里是不能像上面一样加.detach()的，因为G（z)在上面
        #其余和上面训练D时差不多

        # max predf，取负使得predf趋于最大
        loss_G = - (predf.mean())

        #optim_G
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
     #       viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')

        #    generate_image(D, G, xr, epoch)

            print(loss_D.item(), loss_G.item())


if __name__ == '__main__':
    main()