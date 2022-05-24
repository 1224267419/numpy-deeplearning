PyTorch:动态，好用    
Gpu加速：节省时间，快速并行运算

自动求导：比如DL的back propagation

这三个函数的作用是先将梯度归零（optimizer.zero_grad()），然后反向传播计算得到每个参数的梯度值（loss.backward()），最后通过梯度下降执行一步参数更新（optimizer.step()）,基本每次训练都要用这三个函数(自动计算省精力，不用手动通过函数更新)
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
# 数学知识部分

#### norm
向量范数的通用公式为L-P范数

记住该公式其他公式都是该公式的引申。
L-0范数：用来统计向量中非零元素的个数。
L-1范数：向量中所有元素的绝对值之和。可用于优化中去除没有取值的信息，又称稀疏规则算子。
L-2范数：典型应用——欧式距离。可用于优化正则化项，避免过拟合。
L-∞范数：计算向量中的最大值。

![范数](D:\numpy%2Bdeep%20learning\范数.png)
# real learning
梯度下降：x``=x-αΔx    阿尔法是函数的学习率，一般较小，步长太大难以逼近最优点，小学习率慢一点，但是值更接近最优点
对以上公式添加各种约束条件效果更好，更快更精准

我们想要实际值y 更接近 预测值 y_hat=wx+b，即最小化   wx+b-y 
## 损失函数loss=(wx+b-y)^2    
有极小值时最好，但是有时候函数有多个极小值点

对于确定的(X,Y)  (矩阵)    ,必有确定的(w,b)使得∑loss=∑(wx+b-y)^2最小(损失函数不一定)  
广播易得此       自己凸优化的优化效果可能比预设优化器效果更好

**目的：通过优化计算后的w和b，可以很好地通过新的x预测新的y**

Liner Regression ：连续的x和y，得到在连续域(比较大)上的值

Logistics Regression：类似sigmoid 函数，取值在[0,1]，用于二分类问题或者概率问题（∑p=1，Pn∈[0,1]）


通过多个连续空间中的数据建造线性模型得到的预测值往往具有很好的鲁棒性，有效应对高斯噪声的影响，发现其中的关键点

#### 传统欧式空间
传统欧式空间编码：在回归，分类，聚类等机器学习算法中，特征之间距离计算 或 相似度计算是非常重要的，而我们常用的距离或相似度的计算都是在欧式空间的相似度计算，计算余弦相似性，基于的就是欧式空间。

#### one-hot编码：使用 N位 状态寄存器来对 N个状态 进行编码，每个状态都有它独立的寄存器位，并且在任意时候，其中只有一位有效。
将类别变量转换为机器学习算法易于利用的一种形式的过程。
使得数字大小本身的意义消失，仅代表编号or位置，无大小之分

优点：
(1) 解决了 分类器不好处理离散数据 的问题。
(2) 在一定程度上也起到了 扩充特征 的作用。
缺点：
在**文本特征**表示上有些缺点就非常突出了。

(1) 它是一个词袋模型，不考虑 词与词之间的顺序（文本中词的顺序信息也是很重要的）；
(2) 它 假设词与词相互独立（在大多数情况下，词与词是相互影响的）；
(3) 它得到的 特征是离散稀疏 的 (这个问题最严重)。

对于one-hot矩阵，依然有最优解  loss=
![0f187b40dfb57fa71b991a566939292664bd8788](0f187b40dfb57fa71b991a566939292664bd8788.png)
即矩阵欧式距离的算法，从而与其他深度学习一样进行优化 


### pytorch中的数据类型：
int      IntTensor of size()       8(ByteTensor，16(ShortTensor，32(IntTensor,   64(Longtensor(仅signed)bit
float   FloatTensor of size()   16(HalfTensor，32(FloatTensor，64bit(DoubleTensor
int array   IntTensor of size[d1,d2……]
float array   FloatTensor of size[d1,d2……]

pytorch表达string
1.one-hot方法（少量字母可以）
2.embedding     word2vec     glove  （）了解即可


torch.tensor初始化数据是16bit的浮点数，
torch.tensor初始化仅指定list大小时，矩阵内数值随机

对于pytorch里面的数据进行数据类型判断和输出时，一般有三种方法：
(1)print(a.type):输出数据a的详细数据类型；
(2)print(type(a)):输出数据a的基本数据类型，没有(1)中那么详尽；
(3)print(isinstance(a,torch.FloatTensor)):用来输出数据a是否为torch.Tensor数据类型，即返回值为True或者False.
对于pytorch的张量Tensor数据类型，在不同的平台上是不一样的，如果在CPU上即为正常的张量Tensor数据类型，如果在GPU上面，则需要将其数据类型转换:data=data.cuda()，此时data的数据类型从torch.FlaotTensor转换为了torch.cuda.FloatTensor，它可以在cuda上面进行算法的加速实现。

5、对于pytorch里面的**标量数据a**，进行相关的数据定义时,一般将其定义为torch.tensor(a),则输出时返回为tensor(a)
6、对于标量的数据类型，其数据shape输出一般为**a.shape=tensor.size([])**,对于其长度输出**len(a.shape)=0**,另外，对于**a.size()也是等于tensor.size([])等于a.shape[]的。**      **a.size() 的括号一定不要忘记**
#### 7、对于pytorch里面的任何一个张量数据torch.tensor([d1,d2,d3,d4])DIM和size以及shape三种数据属性的含义解析与区分如下：
DIM是指张量数据的长度(即数据的层数)=len(a.shape)，size和shape都是指张量数据的形状;   rank：维度数量 
另外，a.numel()是指数据的大小为d1*d2*d3*d4

**通过list(a.shape)可以轻松得到表达a形状的矩阵**

(1)DIM=2:
**a=torch.tensor([4,784])**
**其中4是指数据图片的数目，而784是指每一张图片的特征维度**
举例：对于a=torch.tensor([1,2,3])
适用于普通的机器学习数据
##### (2)DIM=3:  RNN
1)a.size/shape=tensor.size([1,2,3])
2)a.size(0)=1
3)a.shape[2]=3
4)a[0].shape=[2,3]
**适用于RNN神经网络的数据类型[length,num,feature]**
例如，对于RNN神经网络进行语音识别与处理时[10,20,100]表示:每个单词包含100个特征，一句话一共有10个单词，而每次输20句话
##### (3)DIM=4:
**一般适用于CNN卷积神经网络[b,c,h,w]:图像处理中图片的信息**
torch.tensor.rand(2,3,28,28):
1)2是指每次输入的图片的个数
2)3是指每张图片的**基本特征通道类型**
3)28,28是指每张图片的像素特征：**长和宽**
#### 8、创建Tensor数据的方法主要有以下几种：
(1)
##### 从numpy中导入
Import from numpy：
```python

a=np.array([1.1,2.1)
b=torch.from_numpy(a)
a=np.ones([2,3]) #定义矩阵的方式
b=torch.from_numpy(a)
```
注：从numpy中导入的数据float类型其实是double类型的。

(2)Import from List:
```python
a=torch.tensor([[1.1,2.1],[1.5,1.2]])  #这里的小写tensor中的list数据就是指data本身数据
b=torch.FloatTensor/Tensor(d1,d2,d3)  #这里的大写Tensor中为数据的shape，即数据的维度组成,尽量用上面的
```
9、生成未初始化的数据uninitialized：
(1)torch.empty()
(2)torch.FloatTensor(d1,d2,d3)
(3)torch.IntTensor(d1,d2,d3)

10、tensor数据的随机初始化的方式—rand/rand_like(0-1),randint(整数数据类型),randn(正态分布数据)：
(1)torch.rand()：产生0-1之间的均匀的随机数据
(2)torch.rand_like(a):a为一个tensor数据类型，产生一个和a数据**shape相同的随机tensor数据类型** 
(3)torch.randint(min,max,[d1,d2,d3]):产生一个shape类型为[d1,d2,d3]的tensor数据，数据最小和最大分别为min和max
(4)torch.randn:产生一个**正态分布的数据类型N(0,1)** ，对于自定义的正态分布的数据N(mean,std),一般需要用到torch.normal()函数，一般需要两步步骤进行，其具体的用法如下举例所示：
a=torch.normal(mean=torch.full([10],0)),std=torch.arange(1,0,-0.1))        得到的形状是[10]
b=a.reshape(2,5)

std表示方差范围
(5)生成数据时使用默认的FloatTensor
```
torch.set_default_tensor_type(torch.DoubleTensor)
```
用此方法改变默认数据类型
11、生成一个全部填充相同的数据：torch.full([d1,d2,de3],a)其中填充数据为a
[]为空，输出0维数据
[]为1维数组，输出1维数据
[]为n维数组，输出n维数据

12、递增或者递减函数API：arange/range
torch.arange(min,max,distance):左闭右开区间，不包含最大值
torch.range(min,max,distance)：全闭区间，包含最大值，不推荐使用

13、linspace/logspace:线性空间
(1)torch.linspace(min,max,steps=data number)：返回的是等间距的数据，其中左右数据均包括，数据个数为steps，数据间隔为(max-min)/(steps-1)
**等差数列**

(2)torch.logspace(min,max,steps=data number):返回的是10的各个线性空间次方的数值
10^（等差数列），也就是等比数列

14、torch中一些零、一和单位张量数据生成API：
torch.zeros(3,4) #零张量数据
torch.ones(3,4) #1张量数据
torch.eye(3,4) #单位张量数据
输出有点类似hot-one的单位张量数据
[[1,0,0,0]
 [0,1,0,0,]
 [0,0,1,0]]

以上加_like可以实现得到相同形状的tensor
15、randperm:主要是产生随机的索引值：
torch.randperm(10):在[0,10)，即0-9产生随机的10个索引

16. y = torch.randperm(n)
　　y是把1到n这些数随机打乱得到的一个数字序列。

定位a[]准确坐标时，得到的是tensor类型而非普通数据类型

17. 切片用法与numpy切片一致

18. a.index_select(dim,index)
dim：表示从第几维挑选数据，类型为int值；
index：表示从第一个参数维度中的哪个位置挑选数据,

比如
a=torch.Size([4,3,28,28])
第一维所有数  a.index_select(0,)
 c = torch.index_select(a, 1, torch.tensor([1, 3])),第一个参数是索引的对象，第二个参数0表示按行索引，1表示按列进行索引，c里面tensor[1, 3]表示第一维的第1列和第3列。
b= torch.index_select(a,2, torch.arange(8)), 取所有维度中的第三维的前八列  b.shape=[4,3,8,28]
a[0,...].shape=[s,28,28]   ...剩下的全取
a[...,:2].shape=[4,3,28,2]

20. torch.ge(a,b)比较a，b的大小，a为张量，b可以为和a相同形状的张量，也可以为一个常数。
mask=torch.ge(a,b)
a大于b为1，小于为0
再用 b=torch.masked_select(x,mask)从而得到一个一维矩阵放置了a中所有大于b的数值（类似cv中的掩膜思想）

21. torch.take(a,tensor())
如：torch.take(a,tensor([0,2,5])) 会将矩阵变为一维，然后再去取第0，第2，第5位


##### a.numel() :   获取tensor中一共包含多少个元素


## 维度变换
1. a.reshape( , ) 和a.view( , ) 是完全一样的，没有区别，老版可能没有reshape，用法和numpy一致

2. a.squeeeze(-1) 删除最后一个维度
    a.unsqueeeze(0)增添一个维度在第0维度的位置上（rank变了）
    注意，这俩函数不改变a本身，仅产生新的值，需要其他变量去接收 


a=torch.array([4,1,28,28])
a.unsqueeeze(0)        [1,4,1,28,28]
a.squeeeze(-1)           [1,4,1,28,1]
for example
展示不同维度的例子
```python
a=torch.tensor([1,2])
print(a.unsqueeeze(-1).shape)   #[2,1]
print(a.unsqueeeze(0).shape)    #[1,2] 

```

输出结果：
[[1],[2]]

[[1,2]]


用途：增加维度

```python
b=torch.array([1,32,1,1])
print(b.squeeeze().shape)   #自动删掉所有可以删除的维度
print(b.squeeeze(0).shape)
print(b.squeeeze(1).shape)  #有数据的删不掉
print(b.squeeeze(-1).shape)

```

输出结果：
[32]

[32,1,1]

 [1,32,1]

[1,32,1,1]


3. 维度扩展
expand  or repeat
使用expand()和repeat()函数的时候，x自身不会改变，因此需要将结果重新赋值。

repeat()函数会直接复制数组不会判断什么时候需要复制，容易浪费内存

```python
b.shape=([1,32,1,1])
print(b.expand(1,32,1,1).shape)
print(b.expand(4,32,14,14).shape)         
print(b.expand(1,33,-1,1).shape)          #-1表示维度不变，而维度非1的部分是不能扩展的，会报错
print(b.expand(4,32,14,-4).shape)         #-4是可以生成的。都是没有意义
print(b.repeat(4,32,1,1).shape)           #repeat和expand的不同在于他的维度数字是倍数而非数字
```

[1,32,1,1]
[4,32,14,14]
[4,32,14,-4]
[4,1024,1,1]

4. 维度交换
转置： a.t()     小写  仅二维

普遍维度交换：
API ：  a1=a.transpose(x1,x2)
   x1,x2分别为需要更改的维度，这种方法只能交换其中两个维度

API：y.permute(1,0,2）
permute()可以一次操作多维数据，且必须传入所有维度数，可以一次更改多个维度



5. 广播 
broadcast
可广播的一对张量需满足以下规则：
每个张量至少有一个维度。
迭代维度尺寸时，从***尾部的维度开始，维度尺寸***
* 或者**相等**，
* 或者其中一个张量的**维度尺寸为 1** ，
*  或者其中一个张量前面不存在这个维度。  如   [32,1,1]是可以被[9,32, 2,32]广播的 
```python
import torch

# 示例1：相同形状的张量总是可广播的，因为总能满足以上规则。
x = torch.empty(5, 7, 3)
y = torch.empty(5, 7, 3)


# 示例2：不可广播（ a 不满足第一条规则）。
a = torch.empty((0,))
b = torch.empty(2, 2)


# 示例3：m 和 n 可广播：
m = torch.empty(5, 3, 4, 1)
n = torch.empty(   3, 1, 1)
# 倒数第一个维度：两者的尺寸均为1
# 倒数第二个维度：n尺寸为1
# 倒数第三个维度：两者尺寸相同
# 倒数第四个维度：n该维度不存在


# 示例4：不可广播，因为倒数第三个维度：2 != 3
p = torch.empty(5, 2, 4, 1)
q = torch.empty(   3, 1, 1)
```
6. 合并与分割

(1). torch.cat([a,b],dim=0)
cat用于合并矩阵，**不增加新的维度**
被拼接的维度可以不一致
```
a1=torch.rand(4,3,5,6)
a2=torch.rand(4,3,5,6)

a3=torch.cat([a1,a2],dim=0)
print(a3.shape)

print(torch.cat([a1,a2],dim=1).shape)

print(torch.cat([a1,a2],dim=2).shape)
```

输出：
[8,3,5,6]

[4,6,5,6]

[4,3,10,6]


(2)torch.stack([a,b],dim=0)
stack合并矩阵，**会增加新的维度在第dim的位置上**，即合并但有分组概念
且a b维度必须相同

分割：

```python
a=torch.rand(32,8)
b=torch.rand(32,8)
c=torch.stack([a,b],dim=0)
d=torch.rand(3,32,8)

aa,bb = c.split([1,1],dim=0)
print(aa.shape,bb.shape)

aa,bb = c.split(1,dim=0)   #前后维度完全相等时才用这个
print(aa.shape,bb.shape)

aa,bb=aa,bb = d.split([2,1],dim=0)
print(aa.shape,bb.shape)
```
输出：
[1,32,8]           [1,32,8]
[1,32,8]           [1,32,8]
[2,32,8]           [1,32,8]


加减乘除：
a+b   和torch.add(a,b) 一致
a-b   和torch.sub(a,b) 一致
a*b   和torch.mul(a,b) 一致
a/b   和torch.div(a,b) 一致

矩阵乘法
torch.mm(a,b)   #仅2维，不推荐
torch.matmul(a,b)     和a@b     一致，都是矩阵乘法 
多维矩阵乘法，后两维执行矩阵乘法，前n维不变
这就要求：

a和b除了最后两个维度可以不一致，其他维度的宽度要相同(比如上面代码第一维和第二维分别都是1,2)
a和b最后两维的维度要符合矩阵乘法的要求（比如a的(3,4)能和b的(4,6)进行矩阵乘法）


次方
a**(2)    和    a.pow(2)   和 power(a,2)    一致


倒数

a.rsqrt()

torch.exp(a)
e^x

a.floor()舍去小数    a.ceil()进一     a.trunc()取整数部分      a.frac()取小数部分  a.eound 四舍五入

取值：

a.max()  最大值   a.median()   中间值   a.clamp(10) 小于10的都变成10，大于的照写      a.clamp(0,10)   所有值限定在0到10中间
a.min()  最小值

#### 统计属性

##### 范数：
1. 矩阵范数与向量范数并不完全一致

API: a.nprm(x,dim= 0)    意为求a矩阵的x范数,dim=0 or 1,对应对行矩阵求范数or列矩阵求范数


```python
a=torch.full([8],1)
b=a.reshape(2,4)


print(a.norm(1))

print(b.norm(1))

print(b.norm(1，dim=0))

print(a.norm(1,dim=1))
```


输出：
8
8
[2,2,2,2]
[4.4]


2. a.sum()    求和
a.mean() 求平均值 即 a.sum()/sizeof(a)   
a.max()    最大值    a.min()   最小值
a.argmax(dim=,,keepdim=)    最大值所在索引       a.argmin(dim=,keepdim=)    最小值索引     注意：不给定dim=会打平矩阵然后给索引
keepdim=True 时默认保持二维性，即给出的矩阵必为二维矩阵，默认为false
```python

a=torch.randn(4,10)
print(a)
print(a.argmax())
print(a.argmax(dim=0))
print(a.argmax(dim=1))
#torch.argmax(dim)会返回dim维度上张量最大值的索引
```

3. topk   找出矩阵中最大的n个数
API : a.topk(n,dim=,largest=)         dim=0or1为行or列 默认1       largest=True or False 为最大or最小，默认True
返回**值函数以及所在位置**，所在位置部分可用于hot one算法

4. kthvalue   找出第n小的数   只能小 

API：a.kthvalue(n,dim=)                    dim=0or1为行or列 默认1  

5. 各类相等

eq – 逐元素判断
原型：x.eq(y)
比较两个张量tensor中，每一个对应位置上元素是否相等–对应位置相等，就返回一个True；否则返回一个False.
返回对应位置为True 或者False的矩阵

a.equal(b)要求整个列表完全相同才是True



6. where操作

对于形状相同的A，B矩阵，当condition_i=1时，取a；condition_i=1
`判断概率问题时用GPU，十分好用`
API；
torch.where(condition_i,a,b)
```python
import torch
a = torch.tensor([[0.0349,  0.0670, -0.0612, 0.0280, -0.0222,  0.0422],
         [-1.6719,  0.1242, -0.6488, 0.3313, -1.3965, -0.0682],
         [-1.3419,  0.4485, -0.6589, 0.1420, -0.3260, -0.4795]])
b = torch.tensor([[-0.0658, -0.1490, -0.1684, 0.7188,  0.3129, -0.1116],
         [-0.2098, -0.2980,  0.1126, 0.9666, -0.0178,  0.1222],
         [ 0.1179, -0.4622, -0.2112, 1.1151,  0.1846,  0.4283]])
cc = torch.where(a>0,a,b)     #合并a,b两个tensor，如果a中元素大于0，则c中与a对应的位置取a的值，否则取b的值
print(cc)
```


7. gather(input(tensor),dim(int),index(LongTensor),out=None)
 查表对应元素，调用**GPU**,one hot 算法常用，即用a矩阵的元素对应b矩阵的元素，进行一个映射

torch.gather
作用：收集输入的特定维度指定位置的数值
参数：
input(tensor):   待操作数。不妨设其维度为（x1, x2, …, xn）
dim(int):   待操作的维度。
index(LongTensor):   如何对input进行操作。其维度有限定，例如当dim=i时，index的维度为（x1, x2, …y, …,xn），既是将input的第i维的大小更改为y，且要满足y>=1（除了第i维之外的其他维度，大小要和input保持一致）。
out:   注意输出和index的维度是一致的

```python
out[i][j][k] = input[index[i][j][k]][j][k]]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```

`举例`:**很重要，认真看**

```python
idx=[[7,4,9],
	[8,1,3],
	[8,6,0]]

idx=torch.tensor(idx)

label=torch.arange(10)+100

print(torch.gather(label.expand(3,10),dim=1,index=idx.long())) 

```

输出：
tensor([[107,104,109],
			[108,101,103],
			[108,106,100])

# 梯度下降法
#### θ_1=θ_1-αd(f(θ_1,θ_2))/dθ_1
阿尔法是学习率

### 凸优化
for example:
寻找全局最优解：借助resnet-56的shot-cut模块使得粗糙表面变得相对平滑，更容易找到最优解

##### saddle point(鞍点)
取得了一个维度的局部极小这和另一个维度的局部极大值

##### 局部极小值 intialization

1. 修改初始值范围（何凯琳初始方法）
2. learning rate 步长太大:达不到最小值，不收敛，训练精度低    步长太小:训练慢
3. escape minima逃离局部最小值：添加训练惯性，即总的调整方向也会对接下来的训练产生影响，使得训练不至于落入局部极小值

![逃离局部极小值——训练惯性](D:\numpy%2Bdeep%20learning\逃离局部极小值——训练惯性.png)

## 常用激活函数
##### 阶梯函数(生物神经元)
   123
             {1   Σ符合条件
a=	     {
		     {0         others               

然而这玩意不可导.....,基本只能用于输出了


所有引入了另外一个经典函数

##### torch.sigmoid(a)函数
f(x)=1/(1+e^(-x))
![sigmoid](D:\numpy%2Bdeep%20learning\sigmoid.png)

值域在0~1

**f`(x)=f(1-f)**
然而，导数在x过大或过小时，参数容易长时间不更新或者迭代缓慢


##### torch.tanh(a)
**rnn**循环神经网络中适用
tanh=((e^x)-e^(-x))/((e^x)+e^(-x))=2sigmoid(2x)-1![tanh](D:\numpy%2Bdeep%20learning\tanh.png)

f`(x)=1-f^2
值域在-1~1

函数和sigmoid类似，同样存在倒数问题


所以，再有新函数:

##### torch.relu(a)    or  nn.ReLU(implace=bool)
```
		{ 0      x<0 
f(x)={
   		{x       x≥0


		{ 0      x<0 
f`(x)={
   		{1      x≥0

```
导数固定，计算方便，避免梯度弥散和梯度爆炸

**优先使用**

###### nn.LeakyReLU(implace=bool)
0<λ<1,不至于出现<0导致Loss卡住不动
```
		{λx      x<0 
f(x)={
   		{x       x≥0


		{λ      x<0 
f`(x)={
   		{1      x≥0

```

![leaky relu](D:\numpy%2Bdeep%20learning\leaky%20relu.png)


##### SELU
SELU   修正不连续点(用途较少，了解一下吧)
![SELU](D:\numpy%2Bdeep%20learning\SELU.png)

##### softplus
具体看图，relu光滑修正_2
![softplus](D:\numpy%2Bdeep%20learning\softplus.png)

#### 常见Loss

##### Mean Squared Error: 均方差

loss=∑[y-(wx+b)]^2=(l2-norm)^2


##### F.mse_loss(a,b)

就是求a，b矩阵(or标量)的均方差，且
允许广播
　这里注意一下两个入参：

　　Ａ reduce = False，返回向量形式的 loss　

　　Ｂ　reduce = True， 返回标量形式的loss

       C  size_average = True，返回 loss.mean();

　　D  如果 size_average = False，返回 loss.sum()
#### 自动求导:
torch.autograd.grad(loss, [w1, w2,…])    需要时可以一次输出多个参数的导数
使用F函数前，记得import torch.nn.functional as F
```
import torch
x=torch.one(1)
w=torch.full([1],3.)  #注意，整型没有梯度，至少是float
mse=F.mse_loss(torch.ones(1),x*w)

# torch.autograd.grad(mse,[w]) 直接调用函数是错误的，需要告诉torch {w}需要梯度信息

w.requires_grad_()  #单告诉需要梯度信息还不行，还要刷新mse


mse=F.mse_loss(torch.ones(1),x*w)
print(torch.autograd.grad(mse,[w]))  
```

输出：
(tensor([4.]),)
就是w=3时loss函数对w的梯度

#### loss.backward
```python
x=torch.ones(1)
w=torch.full([1],3.)
mse=F.mse_loss(torch.ones(1),x*w)
print(w)
#torch.autograd.grad(mse,[w]) # 直接调用函数是错误的，需要告诉torch {w}需要梯度信息

w.requires_grad_()  #单告诉需要梯度信息还不行，还要刷新mse

mse=F.mse_loss(torch.ones(1),x*w)

mse.backward(retain_graph=True)#使用mse.backward后，参数.grad 可以直接输出导数，相当于上面的求导,需要手动输出
 
print(w.grad)
```




#### softmax

放大最大值与其他值的倍数差距，压缩其他值，最终和为1


![softmax](D:\numpy%2Bdeep%20learning\softmax.jpg)

p=(e^x_i)/∑(e^x_ j)
 					1
					{p_i(1-p_ j)    if  i=j
(σp_i)/(σa_j)={
					{-p_i*p_j    	if	i≠jpg

对此函数，仅当i=j时导数取正

softmax改进:因为e^x容易导致数据溢出，所以作出如下图的修正，能有效避免数据溢出问题

![sigmoid改进方法](D:\numpy%2Bdeep%20learning\sigmoid改进方法.jpg)
for example:

```python
a=torch.rand(3)
a.requires_grad_()
print(a)
c=
p=F.softmax(a,dim=0,)
  #有retain_graph=True参数，就可以二次back，但是仅限一次，想要继续backward请再输一次,而这里显然是不行的
#p.backward()
#p.backward()     #试试就逝世

p=F.softmax(a,dim=0)    #重新建图   上面那玩意都玩成啥样了
print(torch.autograd.grad(p[1],[a],retain_graph=True))     #同上，求

print('\n')

print(torch.autograd.grad(p[2],[a]))
```


#### 感知机

##### 单层感知机尝试

```python
x=torch.randn(1,10)
w=torch.randn(2,10,requires_grad=Ture)

o=torch.sigmoid(z@w.t())   #output部分
print(o.shape)                                 #正向传播
						
loss=F.mse_loss(torch.ones(1,1),o)     #本来应该是(1,2)的，但是借用广播机制，全一可以直接适配，不用考虑矩阵形状
print(loss)

loss.backward()

print(w.grad)             #反向传播

```

![单一向量机](D:\numpy%2Bdeep%20learning\单一向量机.png)


现在仍是感知机，但多来两层使求导更复杂：如下图
![f54fef68f3c6f467715bd7e6eb74a2c7dd04ef50](f54fef68f3c6f467715bd7e6eb74a2c7dd04ef50.png)

```python
x=torch.tensor(1.)
w1=torch.tensor(2.,requires_grad=True)
b1=torch.tensor(1.)
w2=torch.tensor(2.,requires_grad=True)
b2=torch.tensor(1.)



y1=x* w1+b1
y2=y1*w2+b2           #正向运算部分


dy2_dy1=torch.autograd.grad(y2,[y1],retain_graph=True)[0]

dy1_dw1=torch.autograd.grad(y1,[w1],retain_graph=True)[0]

dy2_dw1=torch.autograd.grad(y2,[w1],retain_graph=True)[0]


print(torch.mul(dy2_dy1,dy1_dw1))

print(dy2_dw1)
```


###  MLP反向传播

 ![多个反向传播](D:\numpy%2Bdeep%20learning\多个反向传播.png)

```							
             k∈K
如图：dE/dWi=∑dE/dw_ij
```

从最后层输出到j层节点的所有变量

![2](D:\numpy%2Bdeep%20learning\2.png)
化为
![1](D:\numpy%2Bdeep%20learning\1.png)

δi->δn    即n在i的上一层
通过δi和On(下一层输出)可以得到任意一层的梯度dE/dWi


### 2D函数极小值优化
1.用python的matlab模块画图

```python
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def himmelblau(x):
	return 	((x[0] ** 2+ x[1] -11)**2 +(x[0] ** 2+ x[1] -7)**2)

x=np.arange(-6,6,0.1)
y=np.arange(-6,6,0.1)
print('X,Y range:',x.shape,y.shape)
X,Y =np.meshgrid(x,y)
print('X,Y maps:',X.shape,Y.shape)
Z=himmelblau([X,Y])

fig=plt.figure('himmelblau')
ax= fig.gca(projection='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```


##### 梯度下降的计算：
```python
x=torch.tensor([0.,0.],requires_grad=True)
optimizer =torch.optim.Adam([x], lr=1e-3)
for step in range(20000):
	pred =himmelblau(x)

	optimizer.zero_grad()
	pred.backward()
	optimizer.step()
	if step%2000 == 0 :
		print('step {} : x={} , f(x)= {}'
			  .format(step,x.tolist(),pred.item()))
```

optimizer = optim.Adam(model.parameters(), lr=learning_rate) #优化函数，model.parameters()为该实例中可优化的参数，lr为参数优化的选项（学习率等）
## Logistic Regression 逻辑回归
##### for regression问题：预测值问题
目标goal:pred = y  预测与真实值接近
 approach: minimize dist(pred,y)   二范数

##### for classification

Goal：maximize benchmark,  e.g.accuracy       并不直接优化accuracy
approach1：minimize dist(p_θ(y|x),p_r(y|x)      给出x得到的分布和真实情况下的分布作二范数等处理使得p_θ(预测值)和p_r(真实值)接近
approach2:minimize divergence(p_θ(y|x),p_r(y|x)


**为什么classification中train的目标和最终test的目标不一样？也就是为什么不能直接maximize accuracy?**

直接的maximize accuracy会有两个问题：
issues 1.gradient=0 if accuracy unchanged but weights changed
结果是非0即1的，概率大于0.5为1，小于0.5为0，若一个本该划为1的情况，w在计算中为0.4，w可能会发生由0.4到0.45的改变而并没有改变最终结果，结果仍为0，分类错误。
issues 2.gradient not continuous since the number of correct is not continuous
也有可能会出现从0.499到0.501的改变，w只改变了0.002，但结果发生了很大的变化，变得不连续，预测变化很大(即使参数调整很小)。


MSELoss（）多用于回归问题，也可以用于one_hotted编码形式，

CrossEntropyLoss()名字为交叉熵损失函数，不用于one_hotted编码形式

MSELoss（）要求batch_x与batch_y的tensor都是FloatTensor类型

CrossEntropyLoss（）要求batch_x为Float，batch_y为LongTensor类型



当然大多数情况下Classification问题可以取代regression问题



#### Multi_class classification  二分类问题
多类：f:x -> p(y|x)
∑p=1   p∈[0,1]

##### 故可以使用softmax函数
放大大的pred
d
###### 熵：Entropy  (不确定性(uncertainty)，惊喜度(measure of surprise )的衡量度

这段可见[zhihu]https://www.zhihu.com/question/65288314

**熵是随机变量不确定度的度量。**

![熵1](D:\numpy%2Bdeep%20learning\熵1.png)


###### 熵越大，越稳定，同时越混乱，因为混乱才是最稳定的，有序是最不稳定的

### cross  entropy 交叉熵 

**不稳定度**：
		
		H(p,q)=-∑p(x)logq（x）
		H(p,q)=H(p)+D_kl(p|q)


![5219191b4a9138289ec2c63e645366f1086fd273](5219191b4a9138289ec2c63e645366f1086fd273.png)

通过熵，我们可以知道我们学习的目标(能达到的最优效果)，但是没法知道具体方法，只能一步步进行学习


相对熵（relative entropy），又被称为Kullback-Leibler散度
**相对熵是一些优化算法，例如最大期望算法（Expectation-Maximization algorithm, EM）的损失函数 。此时参与计算的一个概率分布为真实分布，另一个为理论（拟合）分布，相对熵表示使用理论分布拟合真实分布时产生的信息损耗，重合部分越大，相对熵越小，完全重合接近于0**

所以当P=Q时，

	D_kl(p|q)=0,H(p,q)=H(p)
###### 而当使用one-hot encoding时
```
E(p)=1log1=0
所以 H（p,q）=D_kl(p|q)

```

所以，如果有学习到的P_r(y|x)和Q_θ(y|x)的D_kl接近0，则p=q，解释了我们优化的目标，即H(P,Q)趋近于0  

for example:

###### 对一个二分类问题
P(dog)=1-P(cat)
![试求二分类问题](D:\numpy%2Bdeep%20learning\试求二分类问题.png)

所以函数最终的目标是
H(P,Q)=-ΣP(i)logQ(i)趋近于0
交叉熵最小实质上就是似然值最大

### 综上，为什么不用MSELoss：
1. sigmoid+MSE 饱和，梯度弥散
2. Log Loss的梯度较大
3. 不要死板，有时候MSELoss好用，因为梯度求导简单，浅显算法好用

![主流神经网络结构](D:\numpy%2Bdeep%20learning\主流神经网络结构.png)


如图，一般而言：
得到LoGic后直接一次完成处理(F.cross_entropy)，熟练以后再进行处理，新手容易出现数据不稳定

```python
)
x=torch.randn(1,784)
w=torch.randn(10,784)

logits=x@w.t()
print(logits.shape)

pred=F.softmax(logits,dim=1)
print(pred.shape)

pred_log=torch.log(pred)

print(F.nll_loss(pred_log,torch.tensor([3])))
print(F.cross_entropy(logits,torch.tensor([3])

```


训练一个模型:

运行模型时如果没有初始化（即仅仅简单用随机数进行初始化，会导致出现梯度弥散，Loss长时间不更新）
下面的模型训练有所体现
```python
w1, b1 = torch.randn(200，784，requires_grad=True),\
		torch.zeros(200,requires_grad=True)
w2, b2 = torch.randn(200，200，requires_grad=True),\
		torch.zeros(200,requires_grad=True)
w3, b3 = torch.randn(10，200，requires_grad=True),\
		torch.zeros(10,requires_grad=True)

def forward(x):
	x=x@w1.t() +b1
	x=F.relu(x)
	x=x@w2.t() +b2
	x=F.relu(x)
	x=x@w3.t() +b3
	x=F.relu(x)          #最后结尾用不用relu都行
	return x


optimizer =optin.SGD([w1,b1,w2,b2,w3,b3],lr=learning_rate）
criteon=nn.CrossentropyLoss()

for epoch in range(epochs)
	for batch_idx , (data ,target) in enumerate(train_loader):
		date=data.view(-1,28*28)
		
		logits=forward(data)
		loss=criteon(logits,target)
		
		optimizer.zero_grad()
		loss.backward
		#print(w1.grad.norm(),w2.grad.norm())
		optimizer.step()
```

### 工程实现：

 ```一版
X=torch.ones([1,784])
				#入，出
layer1=nn.Liner(784,200)
layer2=nn.Liner(200,200)
layer3=nn.Liner(200,10)



x=layer1(x)
print(x.shape)

x=layer2(x)
print(x.shape)

x=layer3(x)
print(x.shape)   #每一层都使用上一层输出作为本层输入，要保留其中的某个x时直接加后缀就行
 ```

```二版
X=torch.ones([1,784])
				#入，出
layer1=nn.Liner(784,200)
layer2=nn.Liner(200,200)
layer3=nn.Liner(200,10)



x=layer1(x)
x=F.relu(x,inplace=True)          #inplace=True  不保留原值，省内存
print(x.shape)

x=layer2(x)
x=F.relu(x,inplace=True)
print(x.shape)

x=layer3(x)
x=F.relu(x,inplace=True)
print(x.shape)   #每一层都使用上一层输出(relu(layer(x)))作为本层输入，要保留其中的某个x时直接加后缀就行

```

### 高层接口：nn.
1. 可以将多层网络串在一起，封装性更强
2. init layer in _init_()    带上layer自己的参数  
3. 自己forward()即可，backward和autograd 自动帮你完成 

``` 
使用nn.Module 创建神经网络会很方便
torcn.nn是专门为神经网络设计的模块化接口. nn构建于autograd之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类，包含网络各层的定义及forward方法。

如何定义自己的网络：

需要继承nn.Module类，并实现forward方法。继承nn.Module类之后，在构造函数中要调用Module的构造函数, super(Linear, self).init()
一般把网络中具有可学习参数的层放在构造函数__init__()中。
不具有可学习参数的层（如ReLU）可放在构造函数中，也可不放在构造函数中（而在forward中使用nn.functional来代替）。可学习参数放在构造函数中，并且通过nn.Parameter()使参数以parameters（一种tensor,默认是自动求导）的形式存在Module中，并且通过parameters()或者named_parameters()以迭代器的方式返回可学习参数。
只要在nn.Module中定义了forward函数，backward函数就会被自动实现（利用Autograd)。而且一般不是显式的调用forward(layer.forward), 而是layer(input), 会自执行forward().
在forward中可以使用任何Variable支持的函数，毕竟在整个pytorch构建的图中，是Varible在流动。还可以使用if, for, print, log等python语法。
```


使用nn.Module封装:
```python
class MLP(nn.Moudle):
    def __init__(self):
        super(MLP,self).__init__()

        selp.model =nn.Sequential(
            nn.Linear(784,200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True),
        )

	 def __format__(self, x):
        x=self.model(x)
    
   	  return x
```

class-style API:类风格API，必须先实例化再调用，名字大写，如：nn.ReLU
function -style API：F.relu等，可以与上面对比一下
train 训练模块：  注意：初始化已经交给Linear，所以训练效果也不错
```
net =MLP()
optimizer = optim.SGD(net.parameters(),lr=learning_rate)
criteon =nn.CrossEntropyLoss()
for epoch in range(epochs):
    for batch_idx ,(data,target) in enumerate(train_loader):
        data=data.view(-1,28*28)
        logits=net(data)
        loss =criteon(logits,target)
        
        optimizer.zero_grad()
        loss.backward()
        print(w1.grad.norm(),w2.grad.norm())
        optimizer.step()
```



41. 6:34 GPU加速
```
device = torch.device('cida:0')
net =MLP ().to(device)   整个网络模块搬到gpu上




```
.to()方法得到的实例化是原先的放到GPU上的模块
.to()方法得到的tensor和原先的不一样，求梯度会得到GPU版本和CPU版本

#### 测试方法

1. 过拟合overfitting：训练过度，记住了其他特征,泛化能力弱
![5ac3b384fc01c8504c6586909ea201ab3f91b7f9](5ac3b384fc01c8504c6586909ea201ab3f91b7f9.png)
训练集性能好，text性能差 ，用text来修正防止过拟合现象
![Loss!=Accuracy](D:\numpy%2Bdeep%20learning\Loss!=Accuracy.png)
欠拟合：underfitting
继续训练效果不变好：增加模型复杂度 
![b46e2da774a4583d6bda45dd86eebc9e565d1cbf](b46e2da774a4583d6bda45dd86eebc9e565d1cbf.png)
![ec4e84998a921f5a3b83b6c349009a8ecad34427](ec4e84998a921f5a3b83b6c349009a8ecad34427.png)
![395f7c2fd0a151cd3d450217a4ec67fd51a83f58](395f7c2fd0a151cd3d450217a4ec67fd51a83f58.png)
保存多个，选取最好的text acquance：最高的状态作为模型最终值
2. 验证正确率:
对于hot_one,
用torch.eq(P,Q）来比对数据
用.sum().float().item()/.len()来计算准确度
3. 一些常用名词解释：
Epoch（时期）：
当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一次>epoch。（也就是说，所有训练样本在神经网络中都 进行了一次正向传播 和一次反向传播 ）
再通俗一点，一个Epoch就是将所有训练样本训练一次的过程。
然而，当一个Epoch的样本（也就是所有的训练样本）数量可能太过庞大（对于计算机而言），就需要把它分成多个小块，也就是就是分成多个Batch 来进行训练。**

Batch（批 / 一批样本）：
将整个训练样本分成若干个Batch。

Batch_Size（批大小）：
每批样本的大小。

Iteration（一次迭代）：
训练一个Batch就是一次Iteration（这个概念跟程序语言中的迭代器相似）。


一般而言，train完n个batch后就可以进行一次test，在test_loader中分出数据集，然后送到net(data)网络中，一般而言可以不计算test_loss
下面是一个test案例：
![test](D:\numpy%2Bdeep%20learning\test.png)
### 可视化：
```python
from tensorboardX import SummaryWriter
import numpy as np

writer = SummaryWriter()   #新建实例
writer.add_scalar('data/scalar1',dummy_s1[0], n_iter)					#x坐标，自变量名称
writer.add_scalar('data/scalar_grop',{'xsinx': n_iter *np.sin(n_iter),    #（因变量名称，坐标位置）
                                    'xcosx': n_iter *np.cos(n_iter),
                                    'xarctanx': n_iter *np.arctan(n_iter)},n_iter)


writer.add_image('Image',x,n_iter)
writer.add_text('Text','text logged at step:' + str(n_iter),n_iter) #输出字符

for name , param in resnet18.name_parameters():
    writer.add_histogram(name, param.close().cpu.data.numpy(),n_iter)  #直方图可视化过程中只能使用cpu中的numpy数据

writer.close()
```

```
开启方式：终端运行： python -m visdom.server

from visdom import Visdom
viz=Visdom()
viz.line([0.],[0.],win='train_loss',dict(title='train loss'))  #win指定窗口，不指定默认main吧 

viz.line([loss.item()],[global_step],win='train_loss,update='append'')
```
如图:
![fca7ca62a3760398a7385fa3cce397e6620609c9](fca7ca62a3760398a7385fa3cce397e6620609c9.png)
![17a681b758d546406680bebd0fa8b651876a2264](17a681b758d546406680bebd0fa8b651876a2264.png)
注意:得到的两条线会放在同一坐标系中，值域不同的值最好不要放一起，否则很难看
![1cba6fbc21e3368c4bc8d03810bbded0cb0fb9ba](1cba6fbc21e3368c4bc8d03810bbded0cb0fb9ba.png)
对于visdom，不需要转化为numpy格式即可画图，可以自动转换，省资源


#### training
训练：![35a9845256728c5e81b5abeda3c7ba6b5d5abb66](35a9845256728c5e81b5abeda3c7ba6b5d5abb66.png)
如图，测试集放入val set，test set作为验证集用于验收


##### 数据划分
如图，比赛中会给你很多数据，你能用的数据集应先划分为train（多）和val（少），然后再训练和验算
![fd1fa54fbb24c36b309836858b3300dec65b258f](fd1fa54fbb24c36b309836858b3300dec65b258f.png)
一定不能用test来训练，会破坏泛化能力
##### K-Fold 交叉验证 (Cross-Validation)
在机器学习建模过程中，通行的做法通常是将数据分为训练集和测试集。测试集是与训练独立的数据，完全不参与训练，用于最终模型的评估。在训练过程中，经常会出现过拟合的问题，就是模型可以很好的匹配训练数据，却不能很好在预测训练集外的数据。如果此时就使用测试数据来调整模型参数，就相当于在训练时已知部分测试数据的信息，会影响最终评估结果的准确性。通常的做法是在训练数据再中分出一部分做为验证(Validation)数据，用来评估模型的训练效果。

验证数据取自训练数据，但不参与训练，这样可以相对客观的评估模型对于训练集之外数据的匹配程度。模型在验证数据中的评估常用的是交叉验证，又称循环验证。它将原始数据分成K组(K-Fold)，将每个子集数据分别做一次验证集，其余的K-1组子集数据作为训练集，这样会得到K个模型。这K个模型分别在验证集中评估结果，最后的误差MSE(Mean Squared Error)加和平均就得到交叉验证误差。交叉验证有效利用了有限的数据，并且评估结果能够尽可能接近模型在测试集上的表现，可以做为模型优化的指标使用。

数据总量较小时，其他方法无法继续提升性能，可以尝试K-Fold。其他情况就不太建议了，例如数据量很大，就没必要更多训练数据，同时训练成本也要扩大K倍（主要指的训练时间）


#### 防止过拟合:
1. 模型选择不要太深(好)，以免学习到不正确的特征（有的时候表达能力不需要这么好）
2. 利用合适的函数进行提前终结

#### 交叉验证

类似的train_loader，test_loader其实都不是真正的test，仅仅是用于防止过拟合来提前终止训练的方法
```python
train_loader = torch.utils.data.DataLoader(
    train_db,
    batch_size=batch_size, shuffle=True)

test_db = datasets.MNIST('../data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))
test_loader = torch.utils.data.DataLoader(test_db,
    batch_size=batch_size, shuffle=True)
```

```python
for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.cuda()

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
                #完成一次训练

    test_loss = 0
    correct = 0
    for data, target in val_loader:   
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()    
        #检测是否过拟合，如果是就停止训练 
    test_loss /= len(val_loader.dataset)
    print('\nVAL set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
```


 torch.utils.data.DataLoader
API：
```python
# 训练数据集的加载器，自动将数据(train_dataset)分割成batch，顺序随机打乱
 # 从数据库中每次抽出batch size个样本
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                            drop_last = True ,      
                                           shuffle=True)
```
for example
```
"""
    批训练，把数据变成一小批一小批数据进行训练。
    DataLoader就是用来包装所使用的数据，每次抛出一批数据
"""
import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
# 把数据放在数据库中
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)


def show_batch():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            # training


            print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))


if __name__ == '__main__':
    show_batch()
```

K-Fold 交叉验证 (Cross-Validation)的理解与应用

参考: https://www.cnblogs.com/xiaosongshine/p/10557891.html

#### 防止过拟合(overfit)

1. 更多数据(代价高)
2. 差一点的模型，降低复杂度(shallow相对的)   regularization
3. Dropout，能增加鲁棒性
4. Data arguentation 数据增强
5. Early Stopping 提前终结

##### regularization 

![正则化2](D:\numpy%2Bdeep%20learning\正则化2.png)
λ是参数(较小)
后面的|θ|是范数(1范数)
当范数减少时，模型复杂度会减少，即考虑loss的同时也将模型复杂度列为模型评判标准之一(越简单越好)

参考：https://www.cnblogs.com/jianxinzhou/p/4083921.html
利用正则化或者更直接的大参数惩罚，从而降低模型复杂度，防止过拟合以及被部分噪声污染


![正则化降低表达能力](D:\numpy%2Bdeep%20learning\正则化降低表达能力.png)

下面是使用二范数修正模型
![regularization](D:\numpy%2Bdeep%20learning\regularization.png)


```python
optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.01)
criteon = nn.CrossEntropyLoss().to(device)
```
weight_decay就是λ参数
仅在已经over fitting后才使用正则化(因为正则化本身就是降低与training_data的匹配度的，还没过拟合就正则化只会让模型更糟)

```
class torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)[source]
实现随机梯度下降算法（momentum可选）。
```
参数:
params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
lr (float) – 学习率
momentum (float, 可选) – 动量因子（默认：0）
weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认：0）
dampening (float, 可选) – 动量的抑制因子（默认：0）
nesterov (bool, 可选) – 使用Nesterov动量（默认：False）

利用如下公式，将范数加入loss中，使得模型复杂度成为考量之一
![θ范数](D:\numpy%2Bdeep%20learning\θ范数.png)


#### 动量
逃出局部最小值，防止多次震荡达不到最小 值 

原公式：w[k+1]=w[k]-αΔf(w[k])

现公式：z[k+1]=βz[k]+Δf(w[k])
			 w[k+1]=w[k]-αz[k+1]

z[k]上一次梯度方向


###### learning rate的选择
如下图也是一种learning rate的选择方法，随着不断梯度下降，learning rate也逐渐减小 
![learning rate decay](D:\numpy%2Bdeep%20learning\learning%20rate%20decay.png)

如图实现学习率的逐渐下降(每十次参数调整学习率下降一个数量级)
![scheme1](D:\numpy%2Bdeep%20learning\scheme1.png)

![scheme2](D:\numpy%2Bdeep%20learning\scheme2.png)


5. Early Stopping 提前终结
在到达临界点(over fitting)时截断训练
1. validation set to select parameters（模型筛选）
2. Monitor validation performance
3. stop at the hightest val perf （极值点点截止(接下来一段时间都不上升则视为最大值)，经验参考）
![early stop](D:\numpy%2Bdeep%20learning\early%20stop.png)

3. Dropout，能增加鲁棒性
学的多不一定学的好，有效的才是好的
使用概率断掉部分神经元，从而在training时调用少部分的神经元，使得曲线更加平滑，不易学习到噪声
![dropout](D:\numpy%2Bdeep%20learning\dropout.png)

```
cet_dropped = torch.nn.Sequential(
	torch.nn.Linear(784,200),
	torch.nn.Dropout(0.5),
	torch.nn.ReLU(),
	torch.nn.Linear(200,200),
	torch.nn.Dropout(0.5),
	torch.nn.ReLU(),
	torch.nn.Linear(200,10),
)
```
如上,使用
>torch.nn.Dropout(0.5),

可以轻松实现dropout(不加dropout就是直连了(图左))

>tf.nn.dropout(keep_prob)#这里是保留率而不是·上面·的·丢失率(1-p=dropout_prob)

### stochastic gradient descent
随机梯度下降(不是真随机，而是类似于正态分布之类的规则内随机)

批量梯度下降：在每次更新时用所有样本，要留意，在梯度下降中，对于  的更新，所有的样本都有贡献，也就是参与调整  .其计算得到的是一个标准梯度，对于最优化问题，凸问题，也肯定可以达到一个全局最优。因而理论上来说一次更新的幅度是比较大的。如果样本不多的情况下，当然是这样收敛的速度会更快啦。但是很多时候，样本很多，更新一次要很久，这样的方法就不合适啦。下图是其更新公式
![批量梯度下降](D:\numpy%2Bdeep%20learning\批量梯度下降.png)
随机梯度下降：在每次更新时用1个样本，可以看到多了随机两个字，随机也就是说我们用样本中的一个例子来近似我所有的样本，来调整θ，因而随机梯度下降是会带来一定的问题，因为计算得到的并不是准确的一个梯度，对于最优化问题，凸问题，虽然不是每次迭代得到的损失函数都向着全局最优方向， 但是大的整体的方向是向全局最优解的，最终的结果往往是在全局最优解附近。但是相比于批量梯度，这样的方法更快，更快收敛，虽然不是全局最优，但很多时候是我们可以接受的，所以这个方法用的也比上面的多。下图是其更新公式：
(节省资源，训练时间短)
![随机梯度下降](D:\numpy%2Bdeep%20learning\随机梯度下降.png)
mini-batch梯度下降：在每次更新时用b个样本,其实批量的梯度下降就是一种折中的方法，他用了一些小样本来近似全部的，其本质就是我1个指不定不太准，那我用个30个50个样本那比随机的要准不少了吧，而且批量的话还是非常可以反映样本的一个分布情况的。在深度学习中，这种方法用的是最多的，因为这个方法收敛也不会很慢，收敛的局部最优也是更多的可以接受！
![mini-batch梯度下降](D:\numpy%2Bdeep%20learning\mini-batch梯度下降.png)

4. Data arguentation 数据增强

# 卷积
非常好的视频，值得看
![卷积举例](D:\numpy%2Bdeep%20learning\卷积举例.png)

上面例子：f(t)是一个人在每个时间点吃东西的数量，g(t_0-t)是t时间以后食物被消化剩下的数量，那每部分食物剩余的**积分**，就是卷积
为什么说公式是如下的
![卷积基本公式](D:\numpy%2Bdeep%20learning\卷积基本公式.png)
，也就很好理解了		（此处的理解并不准确，下面cnn更准

当然，对于离散型的数据，我们可以直接求和而不是积分

卷积的物理意义：如果有个系统输入不稳定f(x)，输出是稳定的g(t-x)，那我们就可以用卷积求系统存量。

### 卷积神经网络
先对图片卷积，再交给神经网络处理(卷积后节点大大减少，节省资源)
图像卷积：详见opencv
##### 平滑卷积(容易理解，可以拿这个入门)
对卷积核对应点乘再求∑平均，得到一个新的像素值作为中心点的新像素值，剩下的外围空白部分填0使得原图像和卷积后图像大小相同,图像更平滑更朦胧而一圈(3*3)不行就两圈(5*5)，当然现实计算考虑性价比 

对应的固定的卷积核g()和变化的图片像素f()
![卷积具体理解](D:\numpy%2Bdeep%20learning\卷积具体理解.png)

更进一步，前面的值对后面也产生影响，即f(x)=∫g(t-x)f(x)dt,即前面对后面影响 
现在如下图，g(x)的180°旋转，更能体现出**卷** 的含义

![二维图像卷积示意图(3x3)](D:\numpy%2Bdeep%20learning\二维图像卷积示意图(3x3).png)

![g函数和卷积核](D:\numpy%2Bdeep%20learning\g函数和卷积核.png)
g函数180°旋转得到卷积核(不完全一致)

####  过滤器
本质上也是一种卷积核，调用不同的矩阵进行卷积可以得到不同的结果(比如检测水平边缘或者垂直边缘)，例子如下：
![水平和垂直过滤器](D:\numpy%2Bdeep%20learning\水平和垂直过滤器.png)
筛选了一些特征，并对此进行计算
![多个卷积核得到多个特征值](D:\numpy%2Bdeep%20learning\多个卷积核得到多个特征值.png)
以上是多个过滤器计算得到的矩阵

>卷积：1.不稳定输入，稳定输出，求系统存量
			2.周围像素点如何产生影响
			3.一个像素点和周围像素点的总特征符合原要求的程度

而多个二维矩阵叠加，就合成了三维矩阵，如下：
![卷积得到的矩阵叠加得到三维矩阵](D:\numpy%2Bdeep%20learning\卷积得到的矩阵叠加得到三维矩阵.png)
类比求多个特征值(猫，人脸，狗，边缘)，


input_channels:输入通道数，灰度图：1，彩色：3or4
Kernel_channels：输出通道数（一个卷积核对应一个通道）
Kernel_size:卷积核大小:3,5,7等奇数，卷积核的边长
stride：每次卷积的步长（移动的格数）
padding：在卷积的矩阵外围加的0的圈数

H_out和W_out为图像高度，图像宽度，在卷积后可能有所变化，公式如下，当然图里也有
H_out = (H_in-H_k+2padding)/stride + 1
W_out = (W_in-W_k+2padding)/stride + 1
![卷积核参数解释](D:\numpy%2Bdeep%20learning\卷积核参数解释.png)
x：b张图片，3个通道(rgb)，28* 28个像素点	
样本数目，图像通道数，图像高度，图像宽度

one k:3_1:对应输入![123](D:\numpy%2Bdeep%20learning\123.gif)通道数得到的Kernel数		3_2·3对应：3* 3的卷积核

multi-k:	16：要观察的特征数(filter,kernel,weight都是它)（edge，blue...共计16个）3的定义同上
权重矩阵（卷积核的格式:输出通道数（卷积核的个数）输入通道数（以RGB为例，每个通道对应自己的一个权重矩阵）卷积核高度，卷积核宽度，，

bias偏置：输出通道数（一个卷积核对应一个偏置）

out:卷积之后矩阵格式: 样本数目，图像通道数，图像高度，图像宽度 （后三个维度在卷积之后会发生变化）
![Kernel参数分辨范例](D:\numpy%2Bdeep%20learning\Kernel参数分辨范例.png)

如图：理解上面的各项值。
![123](D:\numpy%2Bdeep%20learning\123.gif)

[1 0 1
 0 1 0
 1 0 1]如此矩阵，做一次卷积运算如下
![x型检测](D:\numpy%2Bdeep%20learning\x型检测.png)

如图，不断做卷积convolution和子抽样subsampling
![卷积神经网络1](D:\numpy%2Bdeep%20learning\卷积神经网络1.png)

对于神经网络，从底层特征(边缘，颜色)向高层特征(车轮，车顶等等)的不断提取
![卷积神经网络2](D:\numpy%2Bdeep%20learning\卷积神经网络2.png)

```python
layer=nn.Conv2d(1,3,kernel_size=3,stride=1,padding=0)   #这里的3是kernel的通道数
x=torch.rand(1,1,28,28)  #模拟图片数量，通道数，宽和高

out=layer.forward(x)
print(layer.size)
##看看卷积一次后的宽和高

layer=layer=nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1)

out=layer.forward(x)
print(layer.size)
#补一圈

layer=layer=nn.Conv2d(1,3,kernel_size=3,stride=,padding=1)
out=layer.forward(x)
print(layer.size)
out=layer(x) #这么用可以调用__call__方法，从而调用hooks的方法，.forward是用不了的，所以用实例更好
print(layer.size)

layer.weight
Parameter containing:
tensor([[[[0.2727,-0.0923,-0.15.0],  #以下略，看图



									,requires_grad=True)

print（layer.weight.shape)
print（layer.bias.shape)

w=torch.rand(16,3,5,5)
b=torch.rand(16)

#out=F.conv2d(x,w,b.stride=1,padding=1)  #使用自定义的卷积核和偏置进行二维卷积直接用会报错，w的通道和上面不一样
x=torch.randn(1,3,28,28)
out=F.conv2d(x,w,b.stride=1,padding=1)
print(out.shape)
out=F.conv2d(x,w,b.stride=2,padding=2)
print(out.shape)
```
输出：
[1,3,26,26]
[1,3,28,28]
[1,3,14,14]
[1,3,14,14]

[3,1,3,3]
[3]

[ 1,16,26,26]
[1,16,14,14]
通过调节stride可以

### 池化和采样
pooling池化
upsample上采样
 缩小图像（或称为下采样（subsampled）或降采样（downsampled））的主要目的有两个：1、使得图像符合显示区域的大小；2、生成对应图像的缩略图。

>放大图像（或称为上采样（upsampling）或图像插值（interpolating））的主要目的是放大原图像,从而可以显示在更高分辨率的显示设备上。对图像的缩放操作并不能带来更多关于该图像的信息, 因此图像的质量将不可避免地受到影响。然而，确实有一些缩放方法能够增加图像的信息，从而使得缩放后的图像质量超过原图质量的。


如图做一个下采样(即数据量变少，仅取最大值 )
![类下采样(采样的函数为max函数)](D:\numpy%2Bdeep%20learning\类下采样(采样的函数为max函数).png)
使用nn.和F.都可以，例子如下 
```
print(x.shape)
layer=nn.MaxPool2d(2,stride=2)

out=layer(x)
print(out.shape)

out=F.avg_pool2d(x,2,stride=2)
print(out.shape)

```


![部分池化方法](D:\numpy%2Bdeep%20learning\部分池化方法.png)
输出：
[1,16,14,14]
[1,16,7,7]
[1,16,7,7]

pytorch也有自己的上采样(为了在gpu上运行)
```
print(x.shape)

out=F.interpolate(x,scale_factor=3,mode='nearest')#此处采用近邻插值
print(out.shape)

```
输出：
[1,16,7,7]
[1,16,21,21]

同样，对数据进行ReLU，能去掉负反应点
```python

layer=nn.ReLU(inplace=True)  #此参数可节省内存空间
out=layer(x)

out=F.relu(x)
```
以上俩API等价


### Batch-Norm(olization)
![Batch-Norm](D:\numpy%2Bdeep%20learning\Batch-Norm.png)


训练深度网络的时候经常发生训练困难的问题，因为，每一次参数迭代更新后，上一层网络的输出数据经过这一层网络计算后，数据的分布会发生变化，为下一层网络的学习带来困难（神经网络本来就是要学习数据的分布，要是分布一直在变，学习就很难了），此现象称之为Internal Covariate Shift，而为此使用Batch-Norm(批规范)，使得数据集中于某一区域

如上图：使数据集中于梯度较大的范围内，从而避免出现梯度弥散

BatchNorm对训练过程有着更根本的影响：它能使优化问题的解空间更加平滑，而这种平滑性确保了梯度更具预测性和稳定性，因此可以使用更大范围的学习速率并获得更快的网络收敛。
 具体公式：	  z_i`=(z_i-μ)/σ
					z_i``=γx_i`+β

μ：均值   σ方差(通过每一批数据计算得到的值)    γ ：缩放变量 β:平移变量	（多次学习后得到的值）

```
x=torch.rand(100,16,784)
#对于28*28是一维的，所以选用1d
layer=nn.BatchNorm1d(16)
out=layer(x)

print(layer.running_main)#看均值
print(layer.running_var)#看方差
```
 ε是为了调整误差而生的，一般很小如10^-8
![规范化的batch_norm](D:\numpy%2Bdeep%20learning\规范化的batch_norm.png)
```
x=torch.rand(100,16,784)
#对于28*28是一维的，所以选用1d
layer=nn.BatchNorm1d(16)
out=layer(x)

print(layer.weight.shape)#就是上面的γ
print(layer.bias.shape)#上面的β，都是需要梯度的

```
![全局梯度信息](D:\numpy%2Bdeep%20learning\全局梯度信息.png)
affine:β和γ参数是否需要自动学习
training：区分学习模式还是训练模式
而这俩在test时记得切换为False

有了BN可以**放心的使用大学习率**，但是使用了BN，就不用小心的调参了，**较大的学习率极大的提高了学习速度，(超参可以调大一点)**
Batchnorm本身上也是一种正则的方式，可以**代替其他正则方式如dropout等**
另外，个人认为，**batchnorm降低了数据之间的绝对差异，有一个去相关的性质，更多的考虑相对差异性，因此在分类任务上具有更好的效果。**
注意：BN并不是适用于所有任务的，在image-to-image这样的任务中，尤其是超分辨率上，图像的绝对差异显得尤为重要，所以batchnorm的scale并不适合。
### 经典的神经网络
 LeNet-5
![LeNet-5](D:\numpy%2Bdeep%20learning\LeNet-5.png)
传统，不大好用 

对于手写数字识别这种简单任务正确率高，不用多想

#####  AlexNet
8layers(5+3)
引入relu和dropout
因为显存不够，所以分开两部分进行计算
![AlexNet](D:\numpy%2Bdeep%20learning\AlexNet.png)

VGG
有VGG11 VGG16 VGG19多个版本
卷积 pooling 全连接
小窗口带来同样感受野的同时还能减少数据量(5*5->3 *3的数据量远少于7 *7的，而感受野相同)
小窗口的计算更快
1 *1小窗口可用于调整channal的数量，避免用3 *3及更大的卷积层来调整channal，节省时间，空间和参数
![VGG](D:\numpy%2Bdeep%20learning\VGG.png)

###### GoogleNet
多种卷积方式运算得到same类型的output，然后再直接连接起来
https://blog.csdn.net/jufengwudi/article/details/79102024

部分中间层也设置output，可以防止过拟合
### ResNet
参考隔壁 deeplearning的笔记
减轻了网络过深导致继续训练然而性能没有提升的问题，使得网络得以进一步加深(并增加性能)
![ResNet对比·](D:\numpy%2Bdeep%20learning\ResNet对比·.png)

如图，捷径带来的好处：即使后面的层数难以训练完成，但是仍然能通过shortcut来训练前面的参数，即最差也能把前面的层训练好，如果其中有一层训练不好，那么就直接走shortcut，所以resnet的层数加深结果是一个至少是不会降低识别率的做法

每一个res module跳过两到三个卷积-relu层

加上shortcut从数据上来说，使得模型更平滑，容易找到全局最优解(凸优化)

本文件夹中的RESNET(2).py中记录了部分笔记，可以参考一下

### 序列表示方法

一般而言，自然界中大多数信息表示与序列有关(声音，对话文字)，而与一般的图片数据不同(图片仅有x，y，channal)

![语音信号示例](D:\numpy%2Bdeep%20learning\语音信号示例.png)
如图是一个语音输入的信号展示，再不同时间输入的信号
然而，pytorch中是没有string类型的(都是数字类型)
###### 序列表征sequence representation
将string表示成另外一种数据类型
如：[seq_len,feature_len]


1. one-hot 稀疏，占用空间大，维度高
2. glove (或者是 word2vec)利用语义相关性对数字编码进行转换(近，反，同义词之间遵循一定的数学逻辑)

```python
word={"hello":0,"word":1}
lookup=torch.tensor([word["hello"]],dtype=torch.long)
embeds=nn.Embedding(2,5)  #编码格式为[2,5]
hello_embed=embeds(lookup)
print(hello_embed)

```
如上，先用字典记录单词
将其转换为向量
使用[2,5]的编码格式(两个单词，每个单词用五个数字表示)
将转换得到的向量使用embed处理(制作对应的表格) 
输出(随机的，临时生成的)

glove：下载一份表格，固定单词有固定的对应数字

# RNN循环神经网络
1. 共享权值，减少负荷
2. 一个贯穿始终的consistent memory用来记录语境
3.   h_t(memory)会跟着不断接受的信息不断更新(而不是完成了一次传递再更新，更新取决于h_t-1和x_t)，随着接受一次信息就改变一次语境信息
4. 
h_t=tanh(W_hh *h_(t-1)+W_xh *xt)
y_t=W_hy *h_t

使用tanh作为激活函数 

```python
nn.RNN.input_size




```



