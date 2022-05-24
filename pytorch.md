PyTorch:��̬������    
Gpu���٣���ʡʱ�䣬���ٲ�������

�Զ��󵼣�����DL��back propagation

�������������������Ƚ��ݶȹ��㣨optimizer.zero_grad()����Ȼ���򴫲�����õ�ÿ���������ݶ�ֵ��loss.backward()�������ͨ���ݶ��½�ִ��һ���������£�optimizer.step()��,����ÿ��ѵ����Ҫ������������(�Զ�����ʡ�����������ֶ�ͨ����������)
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
# ��ѧ֪ʶ����

#### norm
����������ͨ�ù�ʽΪL-P����

��ס�ù�ʽ������ʽ���Ǹù�ʽ�����ꡣ
L-0����������ͳ�������з���Ԫ�صĸ�����
L-1����������������Ԫ�صľ���ֵ֮�͡��������Ż���ȥ��û��ȡֵ����Ϣ���ֳ�ϡ��������ӡ�
L-2����������Ӧ�á���ŷʽ���롣�������Ż�������������ϡ�
L-�޷��������������е����ֵ��

![����](D:\numpy%2Bdeep%20learning\����.png)
# real learning
�ݶ��½���x``=x-����x    �������Ǻ�����ѧϰ�ʣ�һ���С������̫�����Աƽ����ŵ㣬Сѧϰ����һ�㣬����ֵ���ӽ����ŵ�
�����Ϲ�ʽ��Ӹ���Լ������Ч�����ã��������׼

������Ҫʵ��ֵy ���ӽ� Ԥ��ֵ y_hat=wx+b������С��   wx+b-y 
## ��ʧ����loss=(wx+b-y)^2    
�м�Сֵʱ��ã�������ʱ�����ж����Сֵ��

����ȷ����(X,Y)  (����)    ,����ȷ����(w,b)ʹ�á�loss=��(wx+b-y)^2��С(��ʧ������һ��)  
�㲥�׵ô�       �Լ�͹�Ż����Ż�Ч�����ܱ�Ԥ���Ż���Ч������

**Ŀ�ģ�ͨ���Ż�������w��b�����Ժܺõ�ͨ���µ�xԤ���µ�y**

Liner Regression ��������x��y���õ���������(�Ƚϴ�)�ϵ�ֵ

Logistics Regression������sigmoid ������ȡֵ��[0,1]�����ڶ�����������߸������⣨��p=1��Pn��[0,1]��


ͨ����������ռ��е����ݽ�������ģ�͵õ���Ԥ��ֵ�������кܺõ�³���ԣ���ЧӦ�Ը�˹������Ӱ�죬�������еĹؼ���

#### ��ͳŷʽ�ռ�
��ͳŷʽ�ռ���룺�ڻع飬���࣬����Ȼ���ѧϰ�㷨�У�����֮�������� �� ���ƶȼ����Ƿǳ���Ҫ�ģ������ǳ��õľ�������ƶȵļ��㶼����ŷʽ�ռ�����ƶȼ��㣬�������������ԣ����ڵľ���ŷʽ�ռ䡣

#### one-hot���룺ʹ�� Nλ ״̬�Ĵ������� N��״̬ ���б��룬ÿ��״̬�����������ļĴ���λ������������ʱ������ֻ��һλ��Ч��
��������ת��Ϊ����ѧϰ�㷨�������õ�һ����ʽ�Ĺ��̡�
ʹ�����ִ�С�����������ʧ����������orλ�ã��޴�С֮��

�ŵ㣺
(1) ����� ���������ô�����ɢ���� �����⡣
(2) ��һ���̶���Ҳ���� �������� �����á�
ȱ�㣺
��**�ı�����**��ʾ����Щȱ��ͷǳ�ͻ���ˡ�

(1) ����һ���ʴ�ģ�ͣ������� �����֮���˳���ı��дʵ�˳����ϢҲ�Ǻ���Ҫ�ģ���
(2) �� ���������໥�������ڴ��������£���������໥Ӱ��ģ���
(3) ���õ��� ��������ɢϡ�� �� (�������������)��

����one-hot������Ȼ�����Ž�  loss=
![0f187b40dfb57fa71b991a566939292664bd8788](0f187b40dfb57fa71b991a566939292664bd8788.png)
������ŷʽ������㷨���Ӷ����������ѧϰһ�������Ż� 


### pytorch�е��������ͣ�
int      IntTensor of size()       8(ByteTensor��16(ShortTensor��32(IntTensor,   64(Longtensor(��signed)bit
float   FloatTensor of size()   16(HalfTensor��32(FloatTensor��64bit(DoubleTensor
int array   IntTensor of size[d1,d2����]
float array   FloatTensor of size[d1,d2����]

pytorch���string
1.one-hot������������ĸ���ԣ�
2.embedding     word2vec     glove  �����˽⼴��


torch.tensor��ʼ��������16bit�ĸ�������
torch.tensor��ʼ����ָ��list��Сʱ����������ֵ���

����pytorch��������ݽ������������жϺ����ʱ��һ�������ַ�����
(1)print(a.type):�������a����ϸ�������ͣ�
(2)print(type(a)):�������a�Ļ����������ͣ�û��(1)����ô�꾡��
(3)print(isinstance(a,torch.FloatTensor)):�����������a�Ƿ�Ϊtorch.Tensor�������ͣ�������ֵΪTrue����False.
����pytorch������Tensor�������ͣ��ڲ�ͬ��ƽ̨���ǲ�һ���ģ������CPU�ϼ�Ϊ����������Tensor�������ͣ������GPU���棬����Ҫ������������ת��:data=data.cuda()����ʱdata���������ʹ�torch.FlaotTensorת��Ϊ��torch.cuda.FloatTensor����������cuda��������㷨�ļ���ʵ�֡�

5������pytorch�����**��������a**��������ص����ݶ���ʱ,һ�㽫�䶨��Ϊtorch.tensor(a),�����ʱ����Ϊtensor(a)
6�����ڱ������������ͣ�������shape���һ��Ϊ**a.shape=tensor.size([])**,�����䳤�����**len(a.shape)=0**,���⣬����**a.size()Ҳ�ǵ���tensor.size([])����a.shape[]�ġ�**      **a.size() ������һ����Ҫ����**
#### 7������pytorch������κ�һ����������torch.tensor([d1,d2,d3,d4])DIM��size�Լ�shape�����������Եĺ���������������£�
DIM��ָ�������ݵĳ���(�����ݵĲ���)=len(a.shape)��size��shape����ָ�������ݵ���״;   rank��ά������ 
���⣬a.numel()��ָ���ݵĴ�СΪd1*d2*d3*d4

**ͨ��list(a.shape)�������ɵõ����a��״�ľ���**

(1)DIM=2:
**a=torch.tensor([4,784])**
**����4��ָ����ͼƬ����Ŀ����784��ָÿһ��ͼƬ������ά��**
����������a=torch.tensor([1,2,3])
��������ͨ�Ļ���ѧϰ����
##### (2)DIM=3:  RNN
1)a.size/shape=tensor.size([1,2,3])
2)a.size(0)=1
3)a.shape[2]=3
4)a[0].shape=[2,3]
**������RNN���������������[length,num,feature]**
���磬����RNN�������������ʶ���봦��ʱ[10,20,100]��ʾ:ÿ�����ʰ���100��������һ�仰һ����10�����ʣ���ÿ����20�仰
##### (3)DIM=4:
**һ��������CNN���������[b,c,h,w]:ͼ������ͼƬ����Ϣ**
torch.tensor.rand(2,3,28,28):
1)2��ָÿ�������ͼƬ�ĸ���
2)3��ָÿ��ͼƬ��**��������ͨ������**
3)28,28��ָÿ��ͼƬ������������**���Ϳ�**
#### 8������Tensor���ݵķ�����Ҫ�����¼��֣�
(1)
##### ��numpy�е���
Import from numpy��
```python

a=np.array([1.1,2.1)
b=torch.from_numpy(a)
a=np.ones([2,3]) #�������ķ�ʽ
b=torch.from_numpy(a)
```
ע����numpy�е��������float������ʵ��double���͵ġ�

(2)Import from List:
```python
a=torch.tensor([[1.1,2.1],[1.5,1.2]])  #�����Сдtensor�е�list���ݾ���ָdata��������
b=torch.FloatTensor/Tensor(d1,d2,d3)  #����Ĵ�дTensor��Ϊ���ݵ�shape�������ݵ�ά�����,�����������
```
9������δ��ʼ��������uninitialized��
(1)torch.empty()
(2)torch.FloatTensor(d1,d2,d3)
(3)torch.IntTensor(d1,d2,d3)

10��tensor���ݵ������ʼ���ķ�ʽ��rand/rand_like(0-1),randint(������������),randn(��̬�ֲ�����)��
(1)torch.rand()������0-1֮��ľ��ȵ��������
(2)torch.rand_like(a):aΪһ��tensor�������ͣ�����һ����a����**shape��ͬ�����tensor��������** 
(3)torch.randint(min,max,[d1,d2,d3]):����һ��shape����Ϊ[d1,d2,d3]��tensor���ݣ�������С�����ֱ�Ϊmin��max
(4)torch.randn:����һ��**��̬�ֲ�����������N(0,1)** �������Զ������̬�ֲ�������N(mean,std),һ����Ҫ�õ�torch.normal()������һ����Ҫ����������У��������÷����¾�����ʾ��
a=torch.normal(mean=torch.full([10],0)),std=torch.arange(1,0,-0.1))        �õ�����״��[10]
b=a.reshape(2,5)

std��ʾ���Χ
(5)��������ʱʹ��Ĭ�ϵ�FloatTensor
```
torch.set_default_tensor_type(torch.DoubleTensor)
```
�ô˷����ı�Ĭ����������
11������һ��ȫ�������ͬ�����ݣ�torch.full([d1,d2,de3],a)�����������Ϊa
[]Ϊ�գ����0ά����
[]Ϊ1ά���飬���1ά����
[]Ϊnά���飬���nά����

12���������ߵݼ�����API��arange/range
torch.arange(min,max,distance):����ҿ����䣬���������ֵ
torch.range(min,max,distance)��ȫ�����䣬�������ֵ�����Ƽ�ʹ��

13��linspace/logspace:���Կռ�
(1)torch.linspace(min,max,steps=data number)�����ص��ǵȼ������ݣ������������ݾ����������ݸ���Ϊsteps�����ݼ��Ϊ(max-min)/(steps-1)
**�Ȳ�����**

(2)torch.logspace(min,max,steps=data number):���ص���10�ĸ������Կռ�η�����ֵ
10^���Ȳ����У���Ҳ���ǵȱ�����

14��torch��һЩ�㡢һ�͵�λ������������API��
torch.zeros(3,4) #����������
torch.ones(3,4) #1��������
torch.eye(3,4) #��λ��������
����е�����hot-one�ĵ�λ��������
[[1,0,0,0]
 [0,1,0,0,]
 [0,0,1,0]]

���ϼ�_like����ʵ�ֵõ���ͬ��״��tensor
15��randperm:��Ҫ�ǲ������������ֵ��
torch.randperm(10):��[0,10)����0-9���������10������

16. y = torch.randperm(n)
����y�ǰ�1��n��Щ��������ҵõ���һ���������С�

��λa[]׼ȷ����ʱ���õ�����tensor���Ͷ�����ͨ��������

17. ��Ƭ�÷���numpy��Ƭһ��

18. a.index_select(dim,index)
dim����ʾ�ӵڼ�ά��ѡ���ݣ�����Ϊintֵ��
index����ʾ�ӵ�һ������ά���е��ĸ�λ����ѡ����,

����
a=torch.Size([4,3,28,28])
��һά������  a.index_select(0,)
 c = torch.index_select(a, 1, torch.tensor([1, 3])),��һ�������������Ķ��󣬵ڶ�������0��ʾ����������1��ʾ���н���������c����tensor[1, 3]��ʾ��һά�ĵ�1�к͵�3�С�
b= torch.index_select(a,2, torch.arange(8)), ȡ����ά���еĵ���ά��ǰ����  b.shape=[4,3,8,28]
a[0,...].shape=[s,28,28]   ...ʣ�µ�ȫȡ
a[...,:2].shape=[4,3,28,2]

20. torch.ge(a,b)�Ƚ�a��b�Ĵ�С��aΪ������b����Ϊ��a��ͬ��״��������Ҳ����Ϊһ��������
mask=torch.ge(a,b)
a����bΪ1��С��Ϊ0
���� b=torch.masked_select(x,mask)�Ӷ��õ�һ��һά���������a�����д���b����ֵ������cv�е���Ĥ˼�룩

21. torch.take(a,tensor())
�磺torch.take(a,tensor([0,2,5])) �Ὣ�����Ϊһά��Ȼ����ȥȡ��0����2����5λ


##### a.numel() :   ��ȡtensor��һ���������ٸ�Ԫ��


## ά�ȱ任
1. a.reshape( , ) ��a.view( , ) ����ȫһ���ģ�û�������ϰ����û��reshape���÷���numpyһ��

2. a.squeeeze(-1) ɾ�����һ��ά��
    a.unsqueeeze(0)����һ��ά���ڵ�0ά�ȵ�λ���ϣ�rank���ˣ�
    ע�⣬�����������ı�a�����������µ�ֵ����Ҫ��������ȥ���� 


a=torch.array([4,1,28,28])
a.unsqueeeze(0)        [1,4,1,28,28]
a.squeeeze(-1)           [1,4,1,28,1]
for example
չʾ��ͬά�ȵ�����
```python
a=torch.tensor([1,2])
print(a.unsqueeeze(-1).shape)   #[2,1]
print(a.unsqueeeze(0).shape)    #[1,2] 

```

��������
[[1],[2]]

[[1,2]]


��;������ά��

```python
b=torch.array([1,32,1,1])
print(b.squeeeze().shape)   #�Զ�ɾ�����п���ɾ����ά��
print(b.squeeeze(0).shape)
print(b.squeeeze(1).shape)  #�����ݵ�ɾ����
print(b.squeeeze(-1).shape)

```

��������
[32]

[32,1,1]

 [1,32,1]

[1,32,1,1]


3. ά����չ
expand  or repeat
ʹ��expand()��repeat()������ʱ��x������ı䣬�����Ҫ��������¸�ֵ��

repeat()������ֱ�Ӹ������鲻���ж�ʲôʱ����Ҫ���ƣ������˷��ڴ�

```python
b.shape=([1,32,1,1])
print(b.expand(1,32,1,1).shape)
print(b.expand(4,32,14,14).shape)         
print(b.expand(1,33,-1,1).shape)          #-1��ʾά�Ȳ��䣬��ά�ȷ�1�Ĳ����ǲ�����չ�ģ��ᱨ��
print(b.expand(4,32,14,-4).shape)         #-4�ǿ������ɵġ�����û������
print(b.repeat(4,32,1,1).shape)           #repeat��expand�Ĳ�ͬ��������ά�������Ǳ�����������
```

[1,32,1,1]
[4,32,14,14]
[4,32,14,-4]
[4,1024,1,1]

4. ά�Ƚ���
ת�ã� a.t()     Сд  ����ά

�ձ�ά�Ƚ�����
API ��  a1=a.transpose(x1,x2)
   x1,x2�ֱ�Ϊ��Ҫ���ĵ�ά�ȣ����ַ���ֻ�ܽ�����������ά��

API��y.permute(1,0,2��
permute()����һ�β�����ά���ݣ��ұ��봫������ά����������һ�θ��Ķ��ά��



5. �㲥 
broadcast
�ɹ㲥��һ���������������¹���
ÿ������������һ��ά�ȡ�
����ά�ȳߴ�ʱ����***β����ά�ȿ�ʼ��ά�ȳߴ�***
* ����**���**��
* ��������һ��������**ά�ȳߴ�Ϊ 1** ��
*  ��������һ������ǰ�治�������ά�ȡ�  ��   [32,1,1]�ǿ��Ա�[9,32, 2,32]�㲥�� 
```python
import torch

# ʾ��1����ͬ��״���������ǿɹ㲥�ģ���Ϊ�����������Ϲ���
x = torch.empty(5, 7, 3)
y = torch.empty(5, 7, 3)


# ʾ��2�����ɹ㲥�� a �������һ�����򣩡�
a = torch.empty((0,))
b = torch.empty(2, 2)


# ʾ��3��m �� n �ɹ㲥��
m = torch.empty(5, 3, 4, 1)
n = torch.empty(   3, 1, 1)
# ������һ��ά�ȣ����ߵĳߴ��Ϊ1
# �����ڶ���ά�ȣ�n�ߴ�Ϊ1
# ����������ά�ȣ����߳ߴ���ͬ
# �������ĸ�ά�ȣ�n��ά�Ȳ�����


# ʾ��4�����ɹ㲥����Ϊ����������ά�ȣ�2 != 3
p = torch.empty(5, 2, 4, 1)
q = torch.empty(   3, 1, 1)
```
6. �ϲ���ָ�

(1). torch.cat([a,b],dim=0)
cat���ںϲ�����**�������µ�ά��**
��ƴ�ӵ�ά�ȿ��Բ�һ��
```
a1=torch.rand(4,3,5,6)
a2=torch.rand(4,3,5,6)

a3=torch.cat([a1,a2],dim=0)
print(a3.shape)

print(torch.cat([a1,a2],dim=1).shape)

print(torch.cat([a1,a2],dim=2).shape)
```

�����
[8,3,5,6]

[4,6,5,6]

[4,3,10,6]


(2)torch.stack([a,b],dim=0)
stack�ϲ�����**�������µ�ά���ڵ�dim��λ����**�����ϲ����з������
��a bά�ȱ�����ͬ

�ָ

```python
a=torch.rand(32,8)
b=torch.rand(32,8)
c=torch.stack([a,b],dim=0)
d=torch.rand(3,32,8)

aa,bb = c.split([1,1],dim=0)
print(aa.shape,bb.shape)

aa,bb = c.split(1,dim=0)   #ǰ��ά����ȫ���ʱ�������
print(aa.shape,bb.shape)

aa,bb=aa,bb = d.split([2,1],dim=0)
print(aa.shape,bb.shape)
```
�����
[1,32,8]           [1,32,8]
[1,32,8]           [1,32,8]
[2,32,8]           [1,32,8]


�Ӽ��˳���
a+b   ��torch.add(a,b) һ��
a-b   ��torch.sub(a,b) һ��
a*b   ��torch.mul(a,b) һ��
a/b   ��torch.div(a,b) һ��

����˷�
torch.mm(a,b)   #��2ά�����Ƽ�
torch.matmul(a,b)     ��a@b     һ�£����Ǿ���˷� 
��ά����˷�������άִ�о���˷���ǰnά����
���Ҫ��

a��b�����������ά�ȿ��Բ�һ�£�����ά�ȵĿ��Ҫ��ͬ(������������һά�͵ڶ�ά�ֱ���1,2)
a��b�����ά��ά��Ҫ���Ͼ���˷���Ҫ�󣨱���a��(3,4)�ܺ�b��(4,6)���о���˷���


�η�
a**(2)    ��    a.pow(2)   �� power(a,2)    һ��


����

a.rsqrt()

torch.exp(a)
e^x

a.floor()��ȥС��    a.ceil()��һ     a.trunc()ȡ��������      a.frac()ȡС������  a.eound ��������

ȡֵ��

a.max()  ���ֵ   a.median()   �м�ֵ   a.clamp(10) С��10�Ķ����10�����ڵ���д      a.clamp(0,10)   ����ֵ�޶���0��10�м�
a.min()  ��Сֵ

#### ͳ������

##### ������
1. ����������������������ȫһ��

API: a.nprm(x,dim= 0)    ��Ϊ��a�����x����,dim=0 or 1,��Ӧ���о�������or�о�������


```python
a=torch.full([8],1)
b=a.reshape(2,4)


print(a.norm(1))

print(b.norm(1))

print(b.norm(1��dim=0))

print(a.norm(1,dim=1))
```


�����
8
8
[2,2,2,2]
[4.4]


2. a.sum()    ���
a.mean() ��ƽ��ֵ �� a.sum()/sizeof(a)   
a.max()    ���ֵ    a.min()   ��Сֵ
a.argmax(dim=,,keepdim=)    ���ֵ��������       a.argmin(dim=,keepdim=)    ��Сֵ����     ע�⣺������dim=���ƽ����Ȼ�������
keepdim=True ʱĬ�ϱ��ֶ�ά�ԣ��������ľ����Ϊ��ά����Ĭ��Ϊfalse
```python

a=torch.randn(4,10)
print(a)
print(a.argmax())
print(a.argmax(dim=0))
print(a.argmax(dim=1))
#torch.argmax(dim)�᷵��dimά�����������ֵ������
```

3. topk   �ҳ�����������n����
API : a.topk(n,dim=,largest=)         dim=0or1Ϊ��or�� Ĭ��1       largest=True or False Ϊ���or��С��Ĭ��True
����**ֵ�����Լ�����λ��**������λ�ò��ֿ�����hot one�㷨

4. kthvalue   �ҳ���nС����   ֻ��С 

API��a.kthvalue(n,dim=)                    dim=0or1Ϊ��or�� Ĭ��1  

5. �������

eq �C ��Ԫ���ж�
ԭ�ͣ�x.eq(y)
�Ƚ���������tensor�У�ÿһ����Ӧλ����Ԫ���Ƿ���ȨC��Ӧλ����ȣ��ͷ���һ��True�����򷵻�һ��False.
���ض�Ӧλ��ΪTrue ����False�ľ���

a.equal(b)Ҫ�������б���ȫ��ͬ����True



6. where����

������״��ͬ��A��B���󣬵�condition_i=1ʱ��ȡa��condition_i=1
`�жϸ�������ʱ��GPU��ʮ�ֺ���`
API��
torch.where(condition_i,a,b)
```python
import torch
a = torch.tensor([[0.0349,  0.0670, -0.0612, 0.0280, -0.0222,  0.0422],
         [-1.6719,  0.1242, -0.6488, 0.3313, -1.3965, -0.0682],
         [-1.3419,  0.4485, -0.6589, 0.1420, -0.3260, -0.4795]])
b = torch.tensor([[-0.0658, -0.1490, -0.1684, 0.7188,  0.3129, -0.1116],
         [-0.2098, -0.2980,  0.1126, 0.9666, -0.0178,  0.1222],
         [ 0.1179, -0.4622, -0.2112, 1.1151,  0.1846,  0.4283]])
cc = torch.where(a>0,a,b)     #�ϲ�a,b����tensor�����a��Ԫ�ش���0����c����a��Ӧ��λ��ȡa��ֵ������ȡb��ֵ
print(cc)
```


7. gather(input(tensor),dim(int),index(LongTensor),out=None)
 ����ӦԪ�أ�����**GPU**,one hot �㷨���ã�����a�����Ԫ�ض�Ӧb�����Ԫ�أ�����һ��ӳ��

torch.gather
���ã��ռ�������ض�ά��ָ��λ�õ���ֵ
������
input(tensor):   ������������������ά��Ϊ��x1, x2, ��, xn��
dim(int):   ��������ά�ȡ�
index(LongTensor):   ��ζ�input���в�������ά�����޶������統dim=iʱ��index��ά��Ϊ��x1, x2, ��y, ��,xn�������ǽ�input�ĵ�iά�Ĵ�С����Ϊy����Ҫ����y>=1�����˵�iά֮�������ά�ȣ���СҪ��input����һ�£���
out:   ע�������index��ά����һ�µ�

```python
out[i][j][k] = input[index[i][j][k]][j][k]]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```

`����`:**����Ҫ�����濴**

```python
idx=[[7,4,9],
	[8,1,3],
	[8,6,0]]

idx=torch.tensor(idx)

label=torch.arange(10)+100

print(torch.gather(label.expand(3,10),dim=1,index=idx.long())) 

```

�����
tensor([[107,104,109],
			[108,101,103],
			[108,106,100])

# �ݶ��½���
#### ��_1=��_1-��d(f(��_1,��_2))/d��_1
��������ѧϰ��

### ͹�Ż�
for example:
Ѱ��ȫ�����Ž⣺����resnet-56��shot-cutģ��ʹ�ôֲڱ��������ƽ�����������ҵ����Ž�

##### saddle point(����)
ȡ����һ��ά�ȵľֲ���С�����һ��ά�ȵľֲ�����ֵ

##### �ֲ���Сֵ intialization

1. �޸ĳ�ʼֵ��Χ���ο��ճ�ʼ������
2. learning rate ����̫��:�ﲻ����Сֵ����������ѵ�����ȵ�    ����̫С:ѵ����
3. escape minima����ֲ���Сֵ�����ѵ�����ԣ����ܵĵ�������Ҳ��Խ�������ѵ������Ӱ�죬ʹ��ѵ������������ֲ���Сֵ

![����ֲ���Сֵ����ѵ������](D:\numpy%2Bdeep%20learning\����ֲ���Сֵ����ѵ������.png)

## ���ü����
##### ���ݺ���(������Ԫ)
   123
             {1   ����������
a=	     {
		     {0         others               

Ȼ�������ⲻ�ɵ�.....,����ֻ�����������


��������������һ�����亯��

##### torch.sigmoid(a)����
f(x)=1/(1+e^(-x))
![sigmoid](D:\numpy%2Bdeep%20learning\sigmoid.png)

ֵ����0~1

**f`(x)=f(1-f)**
Ȼ����������x������Сʱ���������׳�ʱ�䲻���»��ߵ�������


##### torch.tanh(a)
**rnn**ѭ��������������
tanh=((e^x)-e^(-x))/((e^x)+e^(-x))=2sigmoid(2x)-1![tanh](D:\numpy%2Bdeep%20learning\tanh.png)

f`(x)=1-f^2
ֵ����-1~1

������sigmoid���ƣ�ͬ�����ڵ�������


���ԣ������º���:

##### torch.relu(a)    or  nn.ReLU(implace=bool)
```
		{ 0      x<0 
f(x)={
   		{x       x��0


		{ 0      x<0 
f`(x)={
   		{1      x��0

```
�����̶������㷽�㣬�����ݶ���ɢ���ݶȱ�ը

**����ʹ��**

###### nn.LeakyReLU(implace=bool)
0<��<1,�����ڳ���<0����Loss��ס����
```
		{��x      x<0 
f(x)={
   		{x       x��0


		{��      x<0 
f`(x)={
   		{1      x��0

```

![leaky relu](D:\numpy%2Bdeep%20learning\leaky%20relu.png)


##### SELU
SELU   ������������(��;���٣��˽�һ�°�)
![SELU](D:\numpy%2Bdeep%20learning\SELU.png)

##### softplus
���忴ͼ��relu�⻬����_2
![softplus](D:\numpy%2Bdeep%20learning\softplus.png)

#### ����Loss

##### Mean Squared Error: ������

loss=��[y-(wx+b)]^2=(l2-norm)^2


##### F.mse_loss(a,b)

������a��b����(or����)�ľ������
����㲥
������ע��һ��������Σ�

������ reduce = False������������ʽ�� loss��

�����¡�reduce = True�� ���ر�����ʽ��loss

       C  size_average = True������ loss.mean();

����D  ��� size_average = False������ loss.sum()
#### �Զ���:
torch.autograd.grad(loss, [w1, w2,��])    ��Ҫʱ����һ�������������ĵ���
ʹ��F����ǰ���ǵ�import torch.nn.functional as F
```
import torch
x=torch.one(1)
w=torch.full([1],3.)  #ע�⣬����û���ݶȣ�������float
mse=F.mse_loss(torch.ones(1),x*w)

# torch.autograd.grad(mse,[w]) ֱ�ӵ��ú����Ǵ���ģ���Ҫ����torch {w}��Ҫ�ݶ���Ϣ

w.requires_grad_()  #��������Ҫ�ݶ���Ϣ�����У���Ҫˢ��mse


mse=F.mse_loss(torch.ones(1),x*w)
print(torch.autograd.grad(mse,[w]))  
```

�����
(tensor([4.]),)
����w=3ʱloss������w���ݶ�

#### loss.backward
```python
x=torch.ones(1)
w=torch.full([1],3.)
mse=F.mse_loss(torch.ones(1),x*w)
print(w)
#torch.autograd.grad(mse,[w]) # ֱ�ӵ��ú����Ǵ���ģ���Ҫ����torch {w}��Ҫ�ݶ���Ϣ

w.requires_grad_()  #��������Ҫ�ݶ���Ϣ�����У���Ҫˢ��mse

mse=F.mse_loss(torch.ones(1),x*w)

mse.backward(retain_graph=True)#ʹ��mse.backward�󣬲���.grad ����ֱ������������൱���������,��Ҫ�ֶ����
 
print(w.grad)
```




#### softmax

�Ŵ����ֵ������ֵ�ı�����࣬ѹ������ֵ�����պ�Ϊ1


![softmax](D:\numpy%2Bdeep%20learning\softmax.jpg)

p=(e^x_i)/��(e^x_ j)
 					1
					{p_i(1-p_ j)    if  i=j
(��p_i)/(��a_j)={
					{-p_i*p_j    	if	i��jpg

�Դ˺���������i=jʱ����ȡ��

softmax�Ľ�:��Ϊe^x���׵������������������������ͼ������������Ч���������������

![sigmoid�Ľ�����](D:\numpy%2Bdeep%20learning\sigmoid�Ľ�����.jpg)
for example:

```python
a=torch.rand(3)
a.requires_grad_()
print(a)
c=
p=F.softmax(a,dim=0,)
  #��retain_graph=True�������Ϳ��Զ���back�����ǽ���һ�Σ���Ҫ����backward������һ��,��������Ȼ�ǲ��е�
#p.backward()
#p.backward()     #���Ծ�����

p=F.softmax(a,dim=0)    #���½�ͼ   ���������ⶼ���ɶ����
print(torch.autograd.grad(p[1],[a],retain_graph=True))     #ͬ�ϣ���

print('\n')

print(torch.autograd.grad(p[2],[a]))
```


#### ��֪��

##### �����֪������

```python
x=torch.randn(1,10)
w=torch.randn(2,10,requires_grad=Ture)

o=torch.sigmoid(z@w.t())   #output����
print(o.shape)                                 #���򴫲�
						
loss=F.mse_loss(torch.ones(1,1),o)     #����Ӧ����(1,2)�ģ����ǽ��ù㲥���ƣ�ȫһ����ֱ�����䣬���ÿ��Ǿ�����״
print(loss)

loss.backward()

print(w.grad)             #���򴫲�

```

![��һ������](D:\numpy%2Bdeep%20learning\��һ������.png)


�������Ǹ�֪��������������ʹ�󵼸����ӣ�����ͼ
![f54fef68f3c6f467715bd7e6eb74a2c7dd04ef50](f54fef68f3c6f467715bd7e6eb74a2c7dd04ef50.png)

```python
x=torch.tensor(1.)
w1=torch.tensor(2.,requires_grad=True)
b1=torch.tensor(1.)
w2=torch.tensor(2.,requires_grad=True)
b2=torch.tensor(1.)



y1=x* w1+b1
y2=y1*w2+b2           #�������㲿��


dy2_dy1=torch.autograd.grad(y2,[y1],retain_graph=True)[0]

dy1_dw1=torch.autograd.grad(y1,[w1],retain_graph=True)[0]

dy2_dw1=torch.autograd.grad(y2,[w1],retain_graph=True)[0]


print(torch.mul(dy2_dy1,dy1_dw1))

print(dy2_dw1)
```


###  MLP���򴫲�

 ![������򴫲�](D:\numpy%2Bdeep%20learning\������򴫲�.png)

```							
             k��K
��ͼ��dE/dWi=��dE/dw_ij
```

�����������j��ڵ�����б���

![2](D:\numpy%2Bdeep%20learning\2.png)
��Ϊ
![1](D:\numpy%2Bdeep%20learning\1.png)

��i->��n    ��n��i����һ��
ͨ����i��On(��һ�����)���Եõ�����һ����ݶ�dE/dWi


### 2D������Сֵ�Ż�
1.��python��matlabģ�黭ͼ

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


##### �ݶ��½��ļ��㣺
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

optimizer = optim.Adam(model.parameters(), lr=learning_rate) #�Ż�������model.parameters()Ϊ��ʵ���п��Ż��Ĳ�����lrΪ�����Ż���ѡ�ѧϰ�ʵȣ�
## Logistic Regression �߼��ع�
##### for regression���⣺Ԥ��ֵ����
Ŀ��goal:pred = y  Ԥ������ʵֵ�ӽ�
 approach: minimize dist(pred,y)   ������

##### for classification

Goal��maximize benchmark,  e.g.accuracy       ����ֱ���Ż�accuracy
approach1��minimize dist(p_��(y|x),p_r(y|x)      ����x�õ��ķֲ�����ʵ����µķֲ����������ȴ���ʹ��p_��(Ԥ��ֵ)��p_r(��ʵֵ)�ӽ�
approach2:minimize divergence(p_��(y|x),p_r(y|x)


**Ϊʲôclassification��train��Ŀ�������test��Ŀ�겻һ����Ҳ����Ϊʲô����ֱ��maximize accuracy?**

ֱ�ӵ�maximize accuracy�����������⣺
issues 1.gradient=0 if accuracy unchanged but weights changed
����Ƿ�0��1�ģ����ʴ���0.5Ϊ1��С��0.5Ϊ0����һ�����û�Ϊ1�������w�ڼ�����Ϊ0.4��w���ܻᷢ����0.4��0.45�ĸı����û�иı����ս���������Ϊ0���������
issues 2.gradient not continuous since the number of correct is not continuous
Ҳ�п��ܻ���ִ�0.499��0.501�ĸı䣬wֻ�ı���0.002������������˺ܴ�ı仯����ò�������Ԥ��仯�ܴ�(��ʹ����������С)��


MSELoss���������ڻع����⣬Ҳ��������one_hotted������ʽ��

CrossEntropyLoss()����Ϊ��������ʧ������������one_hotted������ʽ

MSELoss����Ҫ��batch_x��batch_y��tensor����FloatTensor����

CrossEntropyLoss����Ҫ��batch_xΪFloat��batch_yΪLongTensor����



��Ȼ����������Classification�������ȡ��regression����



#### Multi_class classification  ����������
���ࣺf:x -> p(y|x)
��p=1   p��[0,1]

##### �ʿ���ʹ��softmax����
�Ŵ���pred
d
###### �أ�Entropy  (��ȷ����(uncertainty)����ϲ��(measure of surprise )�ĺ�����

��οɼ�[zhihu]https://www.zhihu.com/question/65288314

**�������������ȷ���ȵĶ�����**

![��1](D:\numpy%2Bdeep%20learning\��1.png)


###### ��Խ��Խ�ȶ���ͬʱԽ���ң���Ϊ���Ҳ������ȶ��ģ���������ȶ���

### cross  entropy ������ 

**���ȶ���**��
		
		H(p,q)=-��p(x)logq��x��
		H(p,q)=H(p)+D_kl(p|q)


![5219191b4a9138289ec2c63e645366f1086fd273](5219191b4a9138289ec2c63e645366f1086fd273.png)

ͨ���أ����ǿ���֪������ѧϰ��Ŀ��(�ܴﵽ������Ч��)������û��֪�����巽����ֻ��һ��������ѧϰ


����أ�relative entropy�����ֱ���ΪKullback-Leiblerɢ��
**�������һЩ�Ż��㷨��������������㷨��Expectation-Maximization algorithm, EM������ʧ���� ����ʱ��������һ�����ʷֲ�Ϊ��ʵ�ֲ�����һ��Ϊ���ۣ���ϣ��ֲ�������ر�ʾʹ�����۷ֲ������ʵ�ֲ�ʱ��������Ϣ��ģ��غϲ���Խ�������ԽС����ȫ�غϽӽ���0**

���Ե�P=Qʱ��

	D_kl(p|q)=0,H(p,q)=H(p)
###### ����ʹ��one-hot encodingʱ
```
E(p)=1log1=0
���� H��p,q��=D_kl(p|q)

```

���ԣ������ѧϰ����P_r(y|x)��Q_��(y|x)��D_kl�ӽ�0����p=q�������������Ż���Ŀ�꣬��H(P,Q)������0  

for example:

###### ��һ������������
P(dog)=1-P(cat)
![�������������](D:\numpy%2Bdeep%20learning\�������������.png)

���Ժ������յ�Ŀ����
H(P,Q)=-��P(i)logQ(i)������0
��������Сʵ���Ͼ�����Ȼֵ���

### ���ϣ�Ϊʲô����MSELoss��
1. sigmoid+MSE ���ͣ��ݶ���ɢ
2. Log Loss���ݶȽϴ�
3. ��Ҫ���壬��ʱ��MSELoss���ã���Ϊ�ݶ��󵼼򵥣�ǳ���㷨����

![����������ṹ](D:\numpy%2Bdeep%20learning\����������ṹ.png)


��ͼ��һ����ԣ�
�õ�LoGic��ֱ��һ����ɴ���(F.cross_entropy)�������Ժ��ٽ��д����������׳������ݲ��ȶ�

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


ѵ��һ��ģ��:

����ģ��ʱ���û�г�ʼ����������������������г�ʼ�����ᵼ�³����ݶ���ɢ��Loss��ʱ�䲻���£�
�����ģ��ѵ����������
```python
w1, b1 = torch.randn(200��784��requires_grad=True),\
		torch.zeros(200,requires_grad=True)
w2, b2 = torch.randn(200��200��requires_grad=True),\
		torch.zeros(200,requires_grad=True)
w3, b3 = torch.randn(10��200��requires_grad=True),\
		torch.zeros(10,requires_grad=True)

def forward(x):
	x=x@w1.t() +b1
	x=F.relu(x)
	x=x@w2.t() +b2
	x=F.relu(x)
	x=x@w3.t() +b3
	x=F.relu(x)          #����β�ò���relu����
	return x


optimizer =optin.SGD([w1,b1,w2,b2,w3,b3],lr=learning_rate��
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

### ����ʵ�֣�

 ```һ��
X=torch.ones([1,784])
				#�룬��
layer1=nn.Liner(784,200)
layer2=nn.Liner(200,200)
layer3=nn.Liner(200,10)



x=layer1(x)
print(x.shape)

x=layer2(x)
print(x.shape)

x=layer3(x)
print(x.shape)   #ÿһ�㶼ʹ����һ�������Ϊ�������룬Ҫ�������е�ĳ��xʱֱ�ӼӺ�׺����
 ```

```����
X=torch.ones([1,784])
				#�룬��
layer1=nn.Liner(784,200)
layer2=nn.Liner(200,200)
layer3=nn.Liner(200,10)



x=layer1(x)
x=F.relu(x,inplace=True)          #inplace=True  ������ԭֵ��ʡ�ڴ�
print(x.shape)

x=layer2(x)
x=F.relu(x,inplace=True)
print(x.shape)

x=layer3(x)
x=F.relu(x,inplace=True)
print(x.shape)   #ÿһ�㶼ʹ����һ�����(relu(layer(x)))��Ϊ�������룬Ҫ�������е�ĳ��xʱֱ�ӼӺ�׺����

```

### �߲�ӿڣ�nn.
1. ���Խ�������紮��һ�𣬷�װ�Ը�ǿ
2. init layer in _init_()    ����layer�Լ��Ĳ���  
3. �Լ�forward()���ɣ�backward��autograd �Զ�������� 

``` 
ʹ��nn.Module �����������ܷ���
torcn.nn��ר��Ϊ��������Ƶ�ģ�黯�ӿ�. nn������autograd֮�ϣ�����������������������硣
nn.Module��nn��ʮ����Ҫ���࣬�����������Ķ��弰forward������

��ζ����Լ������磺

��Ҫ�̳�nn.Module�࣬��ʵ��forward�������̳�nn.Module��֮���ڹ��캯����Ҫ����Module�Ĺ��캯��, super(Linear, self).init()
һ��������о��п�ѧϰ�����Ĳ���ڹ��캯��__init__()�С�
�����п�ѧϰ�����Ĳ㣨��ReLU���ɷ��ڹ��캯���У�Ҳ�ɲ����ڹ��캯���У�����forward��ʹ��nn.functional�����棩����ѧϰ�������ڹ��캯���У�����ͨ��nn.Parameter()ʹ������parameters��һ��tensor,Ĭ�����Զ��󵼣�����ʽ����Module�У�����ͨ��parameters()����named_parameters()�Ե������ķ�ʽ���ؿ�ѧϰ������
ֻҪ��nn.Module�ж�����forward������backward�����ͻᱻ�Զ�ʵ�֣�����Autograd)������һ�㲻����ʽ�ĵ���forward(layer.forward), ����layer(input), ����ִ��forward().
��forward�п���ʹ���κ�Variable֧�ֵĺ������Ͼ�������pytorch������ͼ�У���Varible��������������ʹ��if, for, print, log��python�﷨��
```


ʹ��nn.Module��װ:
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

class-style API:����API��������ʵ�����ٵ��ã����ִ�д���磺nn.ReLU
function -style API��F.relu�ȣ�����������Ա�һ��
train ѵ��ģ�飺  ע�⣺��ʼ���Ѿ�����Linear������ѵ��Ч��Ҳ����
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



41. 6:34 GPU����
```
device = torch.device('cida:0')
net =MLP ().to(device)   ��������ģ��ᵽgpu��




```
.to()�����õ���ʵ������ԭ�ȵķŵ�GPU�ϵ�ģ��
.to()�����õ���tensor��ԭ�ȵĲ�һ�������ݶȻ�õ�GPU�汾��CPU�汾

#### ���Է���

1. �����overfitting��ѵ�����ȣ���ס����������,����������
![5ac3b384fc01c8504c6586909ea201ab3f91b7f9](5ac3b384fc01c8504c6586909ea201ab3f91b7f9.png)
ѵ�������ܺã�text���ܲ� ����text��������ֹ���������
![Loss!=Accuracy](D:\numpy%2Bdeep%20learning\Loss!=Accuracy.png)
Ƿ��ϣ�underfitting
����ѵ��Ч������ã�����ģ�͸��Ӷ� 
![b46e2da774a4583d6bda45dd86eebc9e565d1cbf](b46e2da774a4583d6bda45dd86eebc9e565d1cbf.png)
![ec4e84998a921f5a3b83b6c349009a8ecad34427](ec4e84998a921f5a3b83b6c349009a8ecad34427.png)
![395f7c2fd0a151cd3d450217a4ec67fd51a83f58](395f7c2fd0a151cd3d450217a4ec67fd51a83f58.png)
��������ѡȡ��õ�text acquance����ߵ�״̬��Ϊģ������ֵ
2. ��֤��ȷ��:
����hot_one,
��torch.eq(P,Q�����ȶ�����
��.sum().float().item()/.len()������׼ȷ��
3. һЩ�������ʽ��ͣ�
Epoch��ʱ�ڣ���
��һ�����������ݼ�ͨ����������һ�β��ҷ�����һ�Σ�������̳�Ϊһ��>epoch����Ҳ����˵������ѵ���������������ж� ������һ�����򴫲� ��һ�η��򴫲� ��
��ͨ��һ�㣬һ��Epoch���ǽ�����ѵ������ѵ��һ�εĹ��̡�
Ȼ������һ��Epoch��������Ҳ�������е�ѵ����������������̫���Ӵ󣨶��ڼ�������ԣ�������Ҫ�����ֳɶ��С�飬Ҳ���Ǿ��Ƿֳɶ��Batch ������ѵ����**

Batch���� / һ����������
������ѵ�������ֳ����ɸ�Batch��

Batch_Size������С����
ÿ�������Ĵ�С��

Iteration��һ�ε�������
ѵ��һ��Batch����һ��Iteration�������������������еĵ��������ƣ���


һ����ԣ�train��n��batch��Ϳ��Խ���һ��test����test_loader�зֳ����ݼ���Ȼ���͵�net(data)�����У�һ����Կ��Բ�����test_loss
������һ��test������
![test](D:\numpy%2Bdeep%20learning\test.png)
### ���ӻ���
```python
from tensorboardX import SummaryWriter
import numpy as np

writer = SummaryWriter()   #�½�ʵ��
writer.add_scalar('data/scalar1',dummy_s1[0], n_iter)					#x���꣬�Ա�������
writer.add_scalar('data/scalar_grop',{'xsinx': n_iter *np.sin(n_iter),    #����������ƣ�����λ�ã�
                                    'xcosx': n_iter *np.cos(n_iter),
                                    'xarctanx': n_iter *np.arctan(n_iter)},n_iter)


writer.add_image('Image',x,n_iter)
writer.add_text('Text','text logged at step:' + str(n_iter),n_iter) #����ַ�

for name , param in resnet18.name_parameters():
    writer.add_histogram(name, param.close().cpu.data.numpy(),n_iter)  #ֱ��ͼ���ӻ�������ֻ��ʹ��cpu�е�numpy����

writer.close()
```

```
������ʽ���ն����У� python -m visdom.server

from visdom import Visdom
viz=Visdom()
viz.line([0.],[0.],win='train_loss',dict(title='train loss'))  #winָ�����ڣ���ָ��Ĭ��main�� 

viz.line([loss.item()],[global_step],win='train_loss,update='append'')
```
��ͼ:
![fca7ca62a3760398a7385fa3cce397e6620609c9](fca7ca62a3760398a7385fa3cce397e6620609c9.png)
![17a681b758d546406680bebd0fa8b651876a2264](17a681b758d546406680bebd0fa8b651876a2264.png)
ע��:�õ��������߻����ͬһ����ϵ�У�ֵ��ͬ��ֵ��ò�Ҫ��һ�𣬷�����ѿ�
![1cba6fbc21e3368c4bc8d03810bbded0cb0fb9ba](1cba6fbc21e3368c4bc8d03810bbded0cb0fb9ba.png)
����visdom������Ҫת��Ϊnumpy��ʽ���ɻ�ͼ�������Զ�ת����ʡ��Դ


#### training
ѵ����![35a9845256728c5e81b5abeda3c7ba6b5d5abb66](35a9845256728c5e81b5abeda3c7ba6b5d5abb66.png)
��ͼ�����Լ�����val set��test set��Ϊ��֤����������


##### ���ݻ���
��ͼ�������л����ܶ����ݣ������õ����ݼ�Ӧ�Ȼ���Ϊtrain���ࣩ��val���٣���Ȼ����ѵ��������
![fd1fa54fbb24c36b309836858b3300dec65b258f](fd1fa54fbb24c36b309836858b3300dec65b258f.png)
һ��������test��ѵ�������ƻ���������
##### K-Fold ������֤ (Cross-Validation)
�ڻ���ѧϰ��ģ�����У�ͨ�е�����ͨ���ǽ����ݷ�Ϊѵ�����Ͳ��Լ������Լ�����ѵ�����������ݣ���ȫ������ѵ������������ģ�͵���������ѵ�������У���������ֹ���ϵ����⣬����ģ�Ϳ��Ժܺõ�ƥ��ѵ�����ݣ�ȴ���ܺܺ���Ԥ��ѵ����������ݡ������ʱ��ʹ�ò�������������ģ�Ͳ��������൱����ѵ��ʱ��֪���ֲ������ݵ���Ϣ����Ӱ���������������׼ȷ�ԡ�ͨ������������ѵ���������зֳ�һ������Ϊ��֤(Validation)���ݣ���������ģ�͵�ѵ��Ч����

��֤����ȡ��ѵ�����ݣ���������ѵ��������������Կ͹۵�����ģ�Ͷ���ѵ����֮�����ݵ�ƥ��̶ȡ�ģ������֤�����е��������õ��ǽ�����֤���ֳ�ѭ����֤������ԭʼ���ݷֳ�K��(K-Fold)����ÿ���Ӽ����ݷֱ���һ����֤���������K-1���Ӽ�������Ϊѵ������������õ�K��ģ�͡���K��ģ�ͷֱ�����֤��������������������MSE(Mean Squared Error)�Ӻ�ƽ���͵õ�������֤��������֤��Ч���������޵����ݣ�������������ܹ������ܽӽ�ģ���ڲ��Լ��ϵı��֣�������Ϊģ���Ż���ָ��ʹ�á�

����������Сʱ�����������޷������������ܣ����Գ���K-Fold����������Ͳ�̫�����ˣ������������ܴ󣬾�û��Ҫ����ѵ�����ݣ�ͬʱѵ���ɱ�ҲҪ����K������Ҫָ��ѵ��ʱ�䣩


#### ��ֹ�����:
1. ģ��ѡ��Ҫ̫��(��)������ѧϰ������ȷ���������е�ʱ������������Ҫ��ô�ã�
2. ���ú��ʵĺ���������ǰ�ս�

#### ������֤

���Ƶ�train_loader��test_loader��ʵ������������test�����������ڷ�ֹ���������ǰ��ֹѵ���ķ���
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
                #���һ��ѵ��

    test_loss = 0
    correct = 0
    for data, target in val_loader:   
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()    
        #����Ƿ����ϣ�����Ǿ�ֹͣѵ�� 
    test_loss /= len(val_loader.dataset)
    print('\nVAL set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
```


 torch.utils.data.DataLoader
API��
```python
# ѵ�����ݼ��ļ��������Զ�������(train_dataset)�ָ��batch��˳���������
 # �����ݿ���ÿ�γ��batch size������
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                            drop_last = True ,      
                                           shuffle=True)
```
for example
```
"""
    ��ѵ���������ݱ��һС��һС�����ݽ���ѵ����
    DataLoader����������װ��ʹ�õ����ݣ�ÿ���׳�һ������
"""
import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
# �����ݷ������ݿ���
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    # �����ݿ���ÿ�γ��batch size������
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

K-Fold ������֤ (Cross-Validation)�������Ӧ��

�ο�: https://www.cnblogs.com/xiaosongshine/p/10557891.html

#### ��ֹ�����(overfit)

1. ��������(���۸�)
2. ��һ���ģ�ͣ����͸��Ӷ�(shallow��Ե�)   regularization
3. Dropout��������³����
4. Data arguentation ������ǿ
5. Early Stopping ��ǰ�ս�

##### regularization 

![����2](D:\numpy%2Bdeep%20learning\����2.png)
���ǲ���(��С)
�����|��|�Ƿ���(1����)
����������ʱ��ģ�͸��ӶȻ���٣�������loss��ͬʱҲ��ģ�͸��Ӷ���Ϊģ�����б�׼֮һ(Խ��Խ��)

�ο���https://www.cnblogs.com/jianxinzhou/p/4083921.html
�������򻯻��߸�ֱ�ӵĴ�����ͷ����Ӷ�����ģ�͸��Ӷȣ���ֹ������Լ�������������Ⱦ


![���򻯽��ͱ������](D:\numpy%2Bdeep%20learning\���򻯽��ͱ������.png)

������ʹ�ö���������ģ��
![regularization](D:\numpy%2Bdeep%20learning\regularization.png)


```python
optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.01)
criteon = nn.CrossEntropyLoss().to(device)
```
weight_decay���Ǧ˲���
�����Ѿ�over fitting���ʹ������(��Ϊ���򻯱�����ǽ�����training_data��ƥ��ȵģ���û����Ͼ�����ֻ����ģ�͸���)

```
class torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)[source]
ʵ������ݶ��½��㷨��momentum��ѡ����
```
����:
params (iterable) �C ���Ż�������iterable�����Ƕ����˲������dict
lr (float) �C ѧϰ��
momentum (float, ��ѡ) �C �������ӣ�Ĭ�ϣ�0��
weight_decay (float, ��ѡ) �C Ȩ��˥����L2�ͷ�����Ĭ�ϣ�0��
dampening (float, ��ѡ) �C �������������ӣ�Ĭ�ϣ�0��
nesterov (bool, ��ѡ) �C ʹ��Nesterov������Ĭ�ϣ�False��

�������¹�ʽ������������loss�У�ʹ��ģ�͸��Ӷȳ�Ϊ����֮һ
![�ȷ���](D:\numpy%2Bdeep%20learning\�ȷ���.png)


#### ����
�ӳ��ֲ���Сֵ����ֹ����𵴴ﲻ����С ֵ 

ԭ��ʽ��w[k+1]=w[k]-����f(w[k])

�ֹ�ʽ��z[k+1]=��z[k]+��f(w[k])
			 w[k+1]=w[k]-��z[k+1]

z[k]��һ���ݶȷ���


###### learning rate��ѡ��
����ͼҲ��һ��learning rate��ѡ�񷽷������Ų����ݶ��½���learning rateҲ�𽥼�С 
![learning rate decay](D:\numpy%2Bdeep%20learning\learning%20rate%20decay.png)

��ͼʵ��ѧϰ�ʵ����½�(ÿʮ�β�������ѧϰ���½�һ��������)
![scheme1](D:\numpy%2Bdeep%20learning\scheme1.png)

![scheme2](D:\numpy%2Bdeep%20learning\scheme2.png)


5. Early Stopping ��ǰ�ս�
�ڵ����ٽ��(over fitting)ʱ�ض�ѵ��
1. validation set to select parameters��ģ��ɸѡ��
2. Monitor validation performance
3. stop at the hightest val perf ����ֵ����ֹ(������һ��ʱ�䶼����������Ϊ���ֵ)������ο���
![early stop](D:\numpy%2Bdeep%20learning\early%20stop.png)

3. Dropout��������³����
ѧ�Ķ಻һ��ѧ�ĺã���Ч�Ĳ��Ǻõ�
ʹ�ø��ʶϵ�������Ԫ���Ӷ���trainingʱ�����ٲ��ֵ���Ԫ��ʹ�����߸���ƽ��������ѧϰ������
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
����,ʹ��
>torch.nn.Dropout(0.5),

��������ʵ��dropout(����dropout����ֱ����(ͼ��))

>tf.nn.dropout(keep_prob)#�����Ǳ����ʶ����ǡ����桤�ġ���ʧ��(1-p=dropout_prob)

### stochastic gradient descent
����ݶ��½�(�����������������������̬�ֲ�֮��Ĺ��������)

�����ݶ��½�����ÿ�θ���ʱ������������Ҫ���⣬���ݶ��½��У�����  �ĸ��£����е��������й��ף�Ҳ���ǲ������  .�����õ�����һ����׼�ݶȣ��������Ż����⣬͹���⣬Ҳ�϶����Դﵽһ��ȫ�����š������������˵һ�θ��µķ����ǱȽϴ�ġ�����������������£���Ȼ�������������ٶȻ�����������Ǻܶ�ʱ�������ܶ࣬����һ��Ҫ�ܾã������ķ����Ͳ�����������ͼ������¹�ʽ
![�����ݶ��½�](D:\numpy%2Bdeep%20learning\�����ݶ��½�.png)
����ݶ��½�����ÿ�θ���ʱ��1�����������Կ���������������֣����Ҳ����˵�����������е�һ�����������������е��������������ȣ��������ݶ��½��ǻ����һ�������⣬��Ϊ����õ��Ĳ�����׼ȷ��һ���ݶȣ��������Ż����⣬͹���⣬��Ȼ����ÿ�ε����õ�����ʧ����������ȫ�����ŷ��� ���Ǵ������ķ�������ȫ�����Ž�ģ����յĽ����������ȫ�����Ž⸽������������������ݶȣ������ķ������죬������������Ȼ����ȫ�����ţ����ܶ�ʱ�������ǿ��Խ��ܵģ�������������õ�Ҳ������Ķࡣ��ͼ������¹�ʽ��
(��ʡ��Դ��ѵ��ʱ���)
![����ݶ��½�](D:\numpy%2Bdeep%20learning\����ݶ��½�.png)
mini-batch�ݶ��½�����ÿ�θ���ʱ��b������,��ʵ�������ݶ��½�����һ�����еķ�����������һЩС����������ȫ���ģ��䱾�ʾ�����1��ָ������̫׼�������ø�30��50�������Ǳ������Ҫ׼�����˰ɣ����������Ļ����Ƿǳ����Է�ӳ������һ���ֲ�����ġ������ѧϰ�У����ַ����õ������ģ���Ϊ�����������Ҳ��������������ľֲ�����Ҳ�Ǹ���Ŀ��Խ��ܣ�
![mini-batch�ݶ��½�](D:\numpy%2Bdeep%20learning\mini-batch�ݶ��½�.png)

4. Data arguentation ������ǿ

# ���
�ǳ��õ���Ƶ��ֵ�ÿ�
![�������](D:\numpy%2Bdeep%20learning\�������.png)

�������ӣ�f(t)��һ������ÿ��ʱ���Զ�����������g(t_0-t)��tʱ���Ժ�ʳ�ﱻ����ʣ�µ���������ÿ����ʳ��ʣ���**����**�����Ǿ��
Ϊʲô˵��ʽ�����µ�
![���������ʽ](D:\numpy%2Bdeep%20learning\���������ʽ.png)
��Ҳ�ͺܺ������		���˴�����Ⲣ��׼ȷ������cnn��׼

��Ȼ��������ɢ�͵����ݣ����ǿ���ֱ����Ͷ����ǻ���

������������壺����и�ϵͳ���벻�ȶ�f(x)��������ȶ���g(t-x)�������ǾͿ����þ����ϵͳ������

### ���������
�ȶ�ͼƬ������ٽ��������紦��(�����ڵ�����٣���ʡ��Դ)
ͼ���������opencv
##### ƽ�����(������⣬�������������)
�Ծ���˶�Ӧ��������ƽ�����õ�һ���µ�����ֵ��Ϊ���ĵ��������ֵ��ʣ�µ���Χ�հײ�����0ʹ��ԭͼ��;����ͼ���С��ͬ,ͼ���ƽ�������ʶ�һȦ(3*3)���о���Ȧ(5*5)����Ȼ��ʵ���㿼���Լ۱� 

��Ӧ�Ĺ̶��ľ����g()�ͱ仯��ͼƬ����f()
![����������](D:\numpy%2Bdeep%20learning\����������.png)

����һ����ǰ���ֵ�Ժ���Ҳ����Ӱ�죬��f(x)=��g(t-x)f(x)dt,��ǰ��Ժ���Ӱ�� 
��������ͼ��g(x)��180����ת���������ֳ�**��** �ĺ���

![��άͼ����ʾ��ͼ(3x3)](D:\numpy%2Bdeep%20learning\��άͼ����ʾ��ͼ(3x3).png)

![g�����;����](D:\numpy%2Bdeep%20learning\g�����;����.png)
g����180����ת�õ������(����ȫһ��)

####  ������
������Ҳ��һ�־���ˣ����ò�ͬ�ľ�����о�����Եõ���ͬ�Ľ��(������ˮƽ��Ե���ߴ�ֱ��Ե)���������£�
![ˮƽ�ʹ�ֱ������](D:\numpy%2Bdeep%20learning\ˮƽ�ʹ�ֱ������.png)
ɸѡ��һЩ���������Դ˽��м���
![�������˵õ��������ֵ](D:\numpy%2Bdeep%20learning\�������˵õ��������ֵ.png)
�����Ƕ������������õ��ľ���

>�����1.���ȶ����룬�ȶ��������ϵͳ����
			2.��Χ���ص���β���Ӱ��
			3.һ�����ص����Χ���ص������������ԭҪ��ĳ̶�

�������ά������ӣ��ͺϳ�����ά�������£�
![����õ��ľ�����ӵõ���ά����](D:\numpy%2Bdeep%20learning\����õ��ľ�����ӵõ���ά����.png)
�����������ֵ(è��������������Ե)��


input_channels:����ͨ�������Ҷ�ͼ��1����ɫ��3or4
Kernel_channels�����ͨ������һ������˶�Ӧһ��ͨ����
Kernel_size:����˴�С:3,5,7������������˵ı߳�
stride��ÿ�ξ���Ĳ������ƶ��ĸ�����
padding���ھ���ľ�����Χ�ӵ�0��Ȧ��

H_out��W_outΪͼ��߶ȣ�ͼ���ȣ��ھ������������仯����ʽ���£���Ȼͼ��Ҳ��
H_out = (H_in-H_k+2padding)/stride + 1
W_out = (W_in-W_k+2padding)/stride + 1
![����˲�������](D:\numpy%2Bdeep%20learning\����˲�������.png)
x��b��ͼƬ��3��ͨ��(rgb)��28* 28�����ص�	
������Ŀ��ͼ��ͨ������ͼ��߶ȣ�ͼ����

one k:3_1:��Ӧ����![123](D:\numpy%2Bdeep%20learning\123.gif)ͨ�����õ���Kernel��		3_2��3��Ӧ��3* 3�ľ����

multi-k:	16��Ҫ�۲��������(filter,kernel,weight������)��edge��blue...����16����3�Ķ���ͬ��
Ȩ�ؾ��󣨾���˵ĸ�ʽ:���ͨ����������˵ĸ���������ͨ��������RGBΪ����ÿ��ͨ����Ӧ�Լ���һ��Ȩ�ؾ��󣩾���˸߶ȣ�����˿�ȣ���

biasƫ�ã����ͨ������һ������˶�Ӧһ��ƫ�ã�

out:���֮������ʽ: ������Ŀ��ͼ��ͨ������ͼ��߶ȣ�ͼ���� ��������ά���ھ��֮��ᷢ���仯��
![Kernel�����ֱ淶��](D:\numpy%2Bdeep%20learning\Kernel�����ֱ淶��.png)

��ͼ���������ĸ���ֵ��
![123](D:\numpy%2Bdeep%20learning\123.gif)

[1 0 1
 0 1 0
 1 0 1]��˾�����һ�ξ����������
![x�ͼ��](D:\numpy%2Bdeep%20learning\x�ͼ��.png)

��ͼ�����������convolution���ӳ���subsampling
![���������1](D:\numpy%2Bdeep%20learning\���������1.png)

���������磬�ӵײ�����(��Ե����ɫ)��߲�����(���֣������ȵ�)�Ĳ�����ȡ
![���������2](D:\numpy%2Bdeep%20learning\���������2.png)

```python
layer=nn.Conv2d(1,3,kernel_size=3,stride=1,padding=0)   #�����3��kernel��ͨ����
x=torch.rand(1,1,28,28)  #ģ��ͼƬ������ͨ��������͸�

out=layer.forward(x)
print(layer.size)
##�������һ�κ�Ŀ�͸�

layer=layer=nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1)

out=layer.forward(x)
print(layer.size)
#��һȦ

layer=layer=nn.Conv2d(1,3,kernel_size=3,stride=,padding=1)
out=layer.forward(x)
print(layer.size)
out=layer(x) #��ô�ÿ��Ե���__call__�������Ӷ�����hooks�ķ�����.forward���ò��˵ģ�������ʵ������
print(layer.size)

layer.weight
Parameter containing:
tensor([[[[0.2727,-0.0923,-0.15.0],  #�����ԣ���ͼ



									,requires_grad=True)

print��layer.weight.shape)
print��layer.bias.shape)

w=torch.rand(16,3,5,5)
b=torch.rand(16)

#out=F.conv2d(x,w,b.stride=1,padding=1)  #ʹ���Զ���ľ���˺�ƫ�ý��ж�ά���ֱ���ûᱨ��w��ͨ�������治һ��
x=torch.randn(1,3,28,28)
out=F.conv2d(x,w,b.stride=1,padding=1)
print(out.shape)
out=F.conv2d(x,w,b.stride=2,padding=2)
print(out.shape)
```
�����
[1,3,26,26]
[1,3,28,28]
[1,3,14,14]
[1,3,14,14]

[3,1,3,3]
[3]

[ 1,16,26,26]
[1,16,14,14]
ͨ������stride����

### �ػ��Ͳ���
pooling�ػ�
upsample�ϲ���
 ��Сͼ�񣨻��Ϊ�²�����subsampled���򽵲�����downsampled��������ҪĿ����������1��ʹ��ͼ�������ʾ����Ĵ�С��2�����ɶ�Ӧͼ�������ͼ��

>�Ŵ�ͼ�񣨻��Ϊ�ϲ�����upsampling����ͼ���ֵ��interpolating��������ҪĿ���ǷŴ�ԭͼ��,�Ӷ�������ʾ�ڸ��߷ֱ��ʵ���ʾ�豸�ϡ���ͼ������Ų��������ܴ���������ڸ�ͼ�����Ϣ, ���ͼ������������ɱ�����ܵ�Ӱ�졣Ȼ����ȷʵ��һЩ���ŷ����ܹ�����ͼ�����Ϣ���Ӷ�ʹ�����ź��ͼ����������ԭͼ�����ġ�


��ͼ��һ���²���(�����������٣���ȡ���ֵ )
![���²���(�����ĺ���Ϊmax����)](D:\numpy%2Bdeep%20learning\���²���(�����ĺ���Ϊmax����).png)
ʹ��nn.��F.�����ԣ��������� 
```
print(x.shape)
layer=nn.MaxPool2d(2,stride=2)

out=layer(x)
print(out.shape)

out=F.avg_pool2d(x,2,stride=2)
print(out.shape)

```


![���ֳػ�����](D:\numpy%2Bdeep%20learning\���ֳػ�����.png)
�����
[1,16,14,14]
[1,16,7,7]
[1,16,7,7]

pytorchҲ���Լ����ϲ���(Ϊ����gpu������)
```
print(x.shape)

out=F.interpolate(x,scale_factor=3,mode='nearest')#�˴����ý��ڲ�ֵ
print(out.shape)

```
�����
[1,16,7,7]
[1,16,21,21]

ͬ���������ݽ���ReLU����ȥ������Ӧ��
```python

layer=nn.ReLU(inplace=True)  #�˲����ɽ�ʡ�ڴ�ռ�
out=layer(x)

out=F.relu(x)
```
������API�ȼ�


### Batch-Norm(olization)
![Batch-Norm](D:\numpy%2Bdeep%20learning\Batch-Norm.png)


ѵ����������ʱ�򾭳�����ѵ�����ѵ����⣬��Ϊ��ÿһ�β����������º���һ�������������ݾ�����һ�������������ݵķֲ��ᷢ���仯��Ϊ��һ�������ѧϰ�������ѣ������籾������Ҫѧϰ���ݵķֲ���Ҫ�Ƿֲ�һֱ�ڱ䣬ѧϰ�ͺ����ˣ����������֮ΪInternal Covariate Shift����Ϊ��ʹ��Batch-Norm(���淶)��ʹ�����ݼ�����ĳһ����

����ͼ��ʹ���ݼ������ݶȽϴ�ķ�Χ�ڣ��Ӷ���������ݶ���ɢ

BatchNorm��ѵ���������Ÿ�������Ӱ�죺����ʹ�Ż�����Ľ�ռ����ƽ����������ƽ����ȷ�����ݶȸ���Ԥ���Ժ��ȶ��ԣ���˿���ʹ�ø���Χ��ѧϰ���ʲ���ø��������������
 ���幫ʽ��	  z_i`=(z_i-��)/��
					z_i``=��x_i`+��

�̣���ֵ   �ҷ���(ͨ��ÿһ�����ݼ���õ���ֵ)    �� �����ű��� ��:ƽ�Ʊ���	�����ѧϰ��õ���ֵ��

```
x=torch.rand(100,16,784)
#����28*28��һά�ģ�����ѡ��1d
layer=nn.BatchNorm1d(16)
out=layer(x)

print(layer.running_main)#����ֵ
print(layer.running_var)#������
```
 ����Ϊ�˵����������ģ�һ���С��10^-8
![�淶����batch_norm](D:\numpy%2Bdeep%20learning\�淶����batch_norm.png)
```
x=torch.rand(100,16,784)
#����28*28��һά�ģ�����ѡ��1d
layer=nn.BatchNorm1d(16)
out=layer(x)

print(layer.weight.shape)#��������Ħ�
print(layer.bias.shape)#����Ħ£�������Ҫ�ݶȵ�

```
![ȫ���ݶ���Ϣ](D:\numpy%2Bdeep%20learning\ȫ���ݶ���Ϣ.png)
affine:�ºͦò����Ƿ���Ҫ�Զ�ѧϰ
training������ѧϰģʽ����ѵ��ģʽ
��������testʱ�ǵ��л�ΪFalse

����BN����**���ĵ�ʹ�ô�ѧϰ��**������ʹ����BN���Ͳ���С�ĵĵ����ˣ�**�ϴ��ѧϰ�ʼ���������ѧϰ�ٶȣ�(���ο��Ե���һ��)**
Batchnorm������Ҳ��һ������ķ�ʽ������**������������ʽ��dropout��**
���⣬������Ϊ��**batchnorm����������֮��ľ��Բ��죬��һ��ȥ��ص����ʣ�����Ŀ�����Բ����ԣ�����ڷ��������Ͼ��и��õ�Ч����**
ע�⣺BN��������������������ģ���image-to-image�����������У������ǳ��ֱ����ϣ�ͼ��ľ��Բ����Ե���Ϊ��Ҫ������batchnorm��scale�����ʺϡ�
### �����������
 LeNet-5
![LeNet-5](D:\numpy%2Bdeep%20learning\LeNet-5.png)
��ͳ��������� 

������д����ʶ�����ּ�������ȷ�ʸߣ����ö���

#####  AlexNet
8layers(5+3)
����relu��dropout
��Ϊ�Դ治�������Էֿ������ֽ��м���
![AlexNet](D:\numpy%2Bdeep%20learning\AlexNet.png)

VGG
��VGG11 VGG16 VGG19����汾
��� pooling ȫ����
С���ڴ���ͬ������Ұ��ͬʱ���ܼ���������(5*5->3 *3��������Զ����7 *7�ģ�������Ұ��ͬ)
С���ڵļ������
1 *1С���ڿ����ڵ���channal��������������3 *3������ľ����������channal����ʡʱ�䣬�ռ�Ͳ���
![VGG](D:\numpy%2Bdeep%20learning\VGG.png)

###### GoogleNet
���־����ʽ����õ�same���͵�output��Ȼ����ֱ����������
https://blog.csdn.net/jufengwudi/article/details/79102024

�����м��Ҳ����output�����Է�ֹ�����
### ResNet
�ο����� deeplearning�ıʼ�
�������������¼���ѵ��Ȼ������û�����������⣬ʹ��������Խ�һ������(����������)
![ResNet�Աȡ�](D:\numpy%2Bdeep%20learning\ResNet�Աȡ�.png)

��ͼ���ݾ������ĺô�����ʹ����Ĳ�������ѵ����ɣ�������Ȼ��ͨ��shortcut��ѵ��ǰ��Ĳ����������Ҳ�ܰ�ǰ��Ĳ�ѵ���ã����������һ��ѵ�����ã���ô��ֱ����shortcut������resnet�Ĳ�����������һ�������ǲ��ή��ʶ���ʵ�����

ÿһ��res module���������������-relu��

����shortcut����������˵��ʹ��ģ�͸�ƽ���������ҵ�ȫ�����Ž�(͹�Ż�)

���ļ����е�RESNET(2).py�м�¼�˲��ֱʼǣ����Բο�һ��

### ���б�ʾ����

һ����ԣ���Ȼ���д������Ϣ��ʾ�������й�(�������Ի�����)������һ���ͼƬ���ݲ�ͬ(ͼƬ����x��y��channal)

![�����ź�ʾ��](D:\numpy%2Bdeep%20learning\�����ź�ʾ��.png)
��ͼ��һ������������ź�չʾ���ٲ�ͬʱ��������ź�
Ȼ����pytorch����û��string���͵�(������������)
###### ���б���sequence representation
��string��ʾ������һ����������
�磺[seq_len,feature_len]


1. one-hot ϡ�裬ռ�ÿռ��ά�ȸ�
2. glove (������ word2vec)������������Զ����ֱ������ת��(��������ͬ���֮����ѭһ������ѧ�߼�)

```python
word={"hello":0,"word":1}
lookup=torch.tensor([word["hello"]],dtype=torch.long)
embeds=nn.Embedding(2,5)  #�����ʽΪ[2,5]
hello_embed=embeds(lookup)
print(hello_embed)

```
���ϣ������ֵ��¼����
����ת��Ϊ����
ʹ��[2,5]�ı����ʽ(�������ʣ�ÿ��������������ֱ�ʾ)
��ת���õ�������ʹ��embed����(������Ӧ�ı��) 
���(����ģ���ʱ���ɵ�)

glove������һ�ݱ�񣬹̶������й̶��Ķ�Ӧ����

# RNNѭ��������
1. ����Ȩֵ�����ٸ���
2. һ���ᴩʼ�յ�consistent memory������¼�ﾳ
3.   h_t(memory)����Ų��Ͻ��ܵ���Ϣ���ϸ���(�����������һ�δ����ٸ��£�����ȡ����h_t-1��x_t)�����Ž���һ����Ϣ�͸ı�һ���ﾳ��Ϣ
4. 
h_t=tanh(W_hh *h_(t-1)+W_xh *xt)
y_t=W_hy *h_t

ʹ��tanh��Ϊ����� 

```python
nn.RNN.input_size




```



