## 数据类型
```
numpy.dtype(object, align, copy)
```
object - 要转换为的数据类型对象   （结果是类型而非数字
align - 如果为 true，填充字段使其类似 C 的结构体。  
copy - 复制 dtype 对象 ，如果为 false，则是对内置数据类型对象的引用

##### 将数据类型应用于 ndarray 对象
```
dt = np.dtype([('age',np.int8)]) 
a = np.array([(10,),(20,),(30,)], dtype = dt) 
print(a)
```
结果：[10 20 30]


#####  下面的示例定义一个结构化数据类型 student，包含字符串字段 name，整数字段 age，及浮点字段 marks
```
student = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')]) 
print(student)
```
输出结果为：
[('name', 'S20'), ('age', 'i1'), ('marks', 'f4')]

## 数组属性
NumPy 数组的维数称为秩（rank），秩就是轴的数量，即数组的维度

``` 
a = np.arange(24)  
print (a.ndim)             # a 现只有一个维度

#现在调整其大小
b = a.reshape(2,4,3)  # b 现在拥有三个维度
print (b.ndim)
```
结果：
1
3



#### ndarray.flags
ndarray.flags 返回 ndarray 对象的内存信息，包含以下属性：

属性                                 	描述
C_CONTIGUOUS (C)	数据是在一个单一的C风格的连续段中
F_CONTIGUOUS (F)	数据是在一个单一的Fortran风格的连续段中
OWNDATA (O)	数组拥有它所使用的内存或从另一个对象中借用它
WRITEABLE (W)	数据区域可以被写入，将该值设置为 False，则数据为只读
ALIGNED (A) 	数据和所有元素都适当地对齐到硬件上
UPDATEIFCOPY (U)	这个数组是其它数组的一个副本，当这个数组被释放时，原数组的内容将被更新
实例
```python
import numpy as np 
 
x = np.array([1,2,3,4,5])  
print (x.flags)
输出结果为：

  C_CONTIGUOUS : True
  F_CONTIGUOUS : True
  OWNDATA : True
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
```

## NumPy 创建数组
numpy.empty
numpy.empty 方法用来创建一个指定形状（shape）、数据类型（dtype）且未初始化的数组：

numpy.empty(shape, dtype = float, order = 'C')
参数说明：

参数	描述
shape	数组形状
dtype	数据类型，可选
order	有"C"和"F"两个选项,分别代表，行优先和列优先，在计算机内存中的存储元素的顺序。（默认是C）

```python
import numpy as np 
x = np.empty([3,2], dtype = int) 
print (x)
```

创建的数组元素值随机，因为没有初始化

几乎与上面相同，但是值全为0
numpy.zeros
创建指定大小的数组，数组元素以 0 来填充：
```
numpy.zeros(shape, dtype = float, order = 'C')
```
创建指定大小的数组，数组元素以 1 来填充：
```python
numpy.ones(shape, dtype = None, order = 'C')
```

numpy.asarray 类似 numpy.array，但 numpy.asarray 参数只有三个，比 numpy.array 少两个。
```python
numpy.asarray(a, dtype = None, order = None)

```
a	任意形式的输入参数，可以是，列表, 列表的元组, 元组, 元组的元组, 元组的列表，多维数组


```python
 numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)   创建 Ndarray数组
```
跨度可以是负数，这样会使数组*在内存中后向移动
名称	描述
object	数组或嵌套的数列
dtype	数组元素的数据类型，可选
copy	对象是否需要复制，可选
order	创建数组的样式，C为行方向，F为列方向，A为任意方向（默认）
subok	默认返回一个与基类类型一致的数组
ndmin	指定生成数组的最小维度

```
numpy.arange(start, stop, step, dtype)
```

根据 start 与 stop 指定的范围以及 step 设定的步长，生成一个 ndarray

参数	描述
start	起始值，默认为0
stop	终止值（不包含）
step	步长，默认为1
dtype	返回ndarray的数据类型，如果没有提供，则会使用输入数据的类型。

```
import numpy as np
x = np.arange(10,20,2,data = int)  
print (x)
```
输出结果如下：

[10  12  14  16  18]



numpy.linspace 函数用于创建一个一维数组，数组是一个等差数列构成的，格式如下：
```
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
```


start	序列的起始值
stop	序列的终止值，如果endpoint为true，该值包含于数列中
num	要生成的等步长的样本数量，默认为50
endpoint	该值为 true 时，数列中包含stop值，反之不包含，默认是True。
retstep	如果为 True 时，生成的数组中会显示间距，反之不显示。
dtype	ndarray 的数据类型

```
import numpy as np
a = np.linspace(1,10,10)
print(a)
```
输出结果为：

[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]



###### numpy.linspace和reshape一起用示例

```
import numpy as np
a =np.linspace(1,10,10,retstep= True)
 
print(a)

b =np.linspace(1,10,10).reshape([10,1])
print(b)
输出结果为：

(array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]), 1.0)
[[ 1.]
 [ 2.]
 [ 3.]
 [ 4.]
 [ 5.]
 [ 6.]
 [ 7.]
 [ 8.]
 [ 9.]
 [10.]]
```


###### 等比数列：

```
np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
```
base 参数意思是取对数的时候 log 的下标。

参数	描述
start	序列的起始值为：base ** start
stop	序列的终止值为：base ** stop。如果endpoint为true，该值包含于数列中
num	要生成的等步长的样本数量，默认为50
endpoint	该值为 true 时，数列中中包含stop值，反之不包含，默认是True。
base	对数 log 的底数。
dtype	ndarray 的数据类型


实例
```
import numpy as np
# 默认底数是 10
a = np.logspace(1.0,  2.0, num =  10)  
print (a)
```

输出结果为：

[ 10.           12.91549665     16.68100537      21.5443469  27.82559402      
  35.93813664   46.41588834     59.94842503      77.42636827    100.    ]
(base *start )是初值，(base *stop )是末值，

## 切片
```python
import numpy as np
 
a = np.arange(10)
s = slice(2,7,2)   # 从索引 2(默认0) 开始到索引 7(默认n) 停止，间隔为2(默认1)
b=a[2:7:2]   # 从索引 2 开始到索引 7 停止，间隔为 2（都是有头没尾）
b=a[2:6:2]

print (a[s])
print(b)
print(c)
```

结果：
[2  4  6]
[2  4  6]
[2  4 ]


多维数组同样适用上述索引提取方法：

实例
```python
import numpy as np
 
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a)
```

#### 从某个索引处开始切割
```
import numpy as np
 
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a)
print('从数组索引 a[1:] 处开始切割')
print(a[1:])
```
输出结果为：

[[1 2 3]
 [3 4 5]
 [4 5 6]]
从数组索引 a[1:] 处开始切割
[[3 4 5]
 [4 5 6]]


切片还可以包括省略号 x…，来使选择元组的长度与数组的维度相同。 如果在行位置使用省略号，它将返回包含行中元素的 ndarray。

实例
```
import numpy as np
 
a = np.array([[1,2,3],[3,4,5],[4,5,6]])  
print (a[...,1])   # 第2列元素
print (a[1,...])   # 第2行元素
print (a[...,1:])  # 第2列及剩下的所有元素

```
输出结果为：

[2 4 5]
[3 4 5]
[[2 3]
 [4 5]
 [5 6]]


常规列表没有的

以下实例获取数组中(0,0)，(1,1)和(2,0)位置处的元素。

实例
 ```python
import numpy as np 
a = np.array([[1,  2],  [3,  4],  [5,  6]]) 
b = a[[0,1,2],  [0,1,0]]  #前面为
print (b)
```
输出结果为：

[1  4  5]


以下实例获取了 4X3 数组中的四个角的元素。 行索引是 [0,0] 和 [3,3]，而列索引是 [0,2] 和 [0,2]。

实例
```
import numpy as np 
 
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
print ('我们的数组是：' )
print (x)
print ('\n')
rows = np.array([[0,0],[3,3]]) # (行坐标）
cols = np.array([[0,2],[0,2]]) #（列坐标）
y = x[rows,cols]  
print  ('这个数组的四个角元素是：')
print (y)
```
输出结果为：

我们的数组是：
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]


这个数组的四个角元素是：
[[ 0  2]
 [ 9 11]]


借助切片 : 或 … 与索引数组组合。如下面例子：
```
import numpy as np
a = np.array([[1,2,3], [4,5,6],[7,8,9]])
b = a[1:3, 1:3]
c = a[1:3,[1,2]]
d = a[...,1:]
print(b)
print(c)
print(d)
```

输出结果为：

[[5 6]
 [8 9]]

[[5 6]
 [8 9]]

[[2 3]
 [5 6]
 [8 9]]



2.2 布尔索引
我们可以通过一个布尔数组来索引目标数组。

布尔索引通过布尔运算（如：比较运算符）来获取符合指定条件的元素的数组。

实例：获取大于 5 的元素
```python
import numpy as np 
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
print ('我们的数组是：')
print (x)
print ('\n')
print  ('大于 5 的元素是：')
print (x[x >  5])
```

我们的数组是：
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

大于 5 的元素是：
[ 6  7  8  9 10 11]



```
import numpy as np 
 
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
print ('我们的数组是：')
print (x)
print ('\n')
# 现在我们会打印出大于 5 的元素  
print  ('大于 5 的元素是：')
print (x[x >  5])
```
输出结果为：

-我们的数组是：
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

大于 5 的元素是：
[ 6  7  8  9 10 11]


###### sp：
np.nan是一个float类型的数据 和None有点像，但不完全一样
np.isnan()用于判断此值是否为空值   ！！！不能用None==np.nan判断，因为空值的相等不能被正确识别
np.isnan(np.nan)    这能输出一个bool型的 True，用于判断值是否为空

以下实例使用了 ~（取补运算符）来过滤 NaN。
~表示操作符按位取反             ~5=-6
实例
```python
import numpy as np 
 
a = np.array([np.nan,  1,2,np.nan,3,4,5])  
print (a[~np.isnan(a)])         
```
输出结果为：

[ 1.   2.   3.   4.   5.]




1、传入顺序索引数组

实例
```
import numpy as np 
 
x=np.arange(32).reshape((8,4))
print (x[[4,2,1,7]])   #(含0)，0是第一行
```
输出结果为：

[[16 17 18 19]
 [ 8  9 10 11]
 [ 4  5  6  7]
 [28 29 30 31]]
2、传入倒序索引数组

实例
```
import numpy as np 
 
x=np.arange(32).reshape((8,4))
print (x[[-4,-2,-1,-7]])     #-1就是最后一行
```
输出结果为：

[[16 17 18 19]
 [24 25 26 27]
 [28 29 30 31]
 [ 4  5  6  7]]
**顺序选取是从0开始数的，a[0]代表第一个；而逆序选取是从1开始数的，a[-1]是倒数第一个**


**先按我们要求选取行，再按顺序将列排序，获得一个矩形**
```
arr2 = np.arange(32).reshape((8,4))

print(arr2[[1,5,7,2]][:,[0,3,1,2]])

 

```

输出：
array([[ 4,  7,  5,  6],
       [20, 23, 21, 22],
       [28, 31, 29, 30],
       [ 8, 11,  9, 10]])


3、传入多个索引数组（要使用np.ix_）

np.ix_函数，能把两个一维数组 转换为 一个用于选取方形区域的索引器

实际意思就是，直接往np.ix_()里扔进两个一维数组[1,5,7,2]，[0,3,1,2]，就能先按我们要求选取行，再按顺序将列排序，跟上面得到的结果一样，而不用写“[ : , [0,3,1,2] ]”

实例
```
import numpy as np 
 
x=np.arange(32).reshape((8,4))
print (x[np.ix_([1,5,7,2],[0,3,1,2])])
```
输出结果为：

[[ 4  7  5  6]
 [20 23 21 22]
 [28 31 29 30]
 [ 8 11  9 10]]

## NumPy 广播(Broadcast)
广播(Broadcast)是 numpy 对不同形状(shape)的数组进行数值计算的方式， 对数组的算术运算通常在相应的元素上进行。

如果两个数组 a 和 b 形状相同，即满足 a.shape == b.shape，那么 a*b 的结果就是 a 与 b 数组对应位相乘。这要求维数相同，且各维度的长度相同。

实例
import numpy as np 
 
a = np.array([1,2,3,4]) 
b = np.array([10,20,30,40]) 
c = a * b 
print (c)

输出结果为：

[ 10  40  90 160]




广播的规则:

让所有输入数组都向其中形状最长的数组看齐，形状中不足的部分都通过在前面加 1 补齐。
输出数组的形状是输入数组形状的各个维度上的最大值。
如果输入数组的某个维度和输出数组的对应维度的长度相同或者其长度为 1 时，这个数组能够用来计算，否则出错。
当输入数组的某个维度的长度为 1 时，沿着此维度运算时都用此维度上的第一组值。
简单理解：对两个数组，分别比较他们的每一个维度（若其中一个数组没有当前维度则忽略），满足：

数组拥有相同形状。
当前维度的值相等。
当前维度的值有一个是 1。


## NumPy 迭代数组
NumPy 迭代器对象 numpy.nditer 提供了一种灵活访问一个或者多个数组元素的方式。
```python
import numpy as np
 
a = np.arange(6).reshape(2,3)
print ('原始数组是：')
print (a)
print ('\n')
print ('迭代输出元素：')
for x in np.nditer(a):
    print (x, end=", " )
print ('\n')
```


输出结果为：

原始数组是：
[[0 1 2]
 [3 4 5]]


迭代输出元素：
0, 1, 2, 3, 4, 5, 

**以上实例不是使用标准 C 或者 Fortran 顺序，选择的顺序是和数组内存布局一致的，这样做是为了提升访问的效率，默认是行序优先（row-major order，或者说是 C-order）。
这反映了默认情况下只需访问每个元素，而无需考虑其特定顺序。我们可以通过迭代上述数组的转置来看到这一点，并与以 C 顺序访问数组转置的 copy 方式做对比，如下实例：**

```python
import numpy as np
 
a = np.arange(6).reshape(2,3)
for x in np.nditer(a.T):
    print (x, end=", " )
print ('\n')
 
for x in np.nditer(a.T.copy(order='C')):
    print (x, end=", " )
print ('\n')
输出结果为：

0, 1, 2, 3, 4, 5, 

0, 3, 1, 4, 2, 5, 
```
a 和 a.T 的遍历顺序是一样的，也就是他们在内存中的存储顺序也是一样的，但是 a.T.copy(order = 'C') 的遍历结果是不同的，那是因为它和前两种的存储方式是不一样的，默认是按行访问。


 ### 控制遍历顺序
```python
import numpy as np
 
a = np.arange(0,60,5) 
a = a.reshape(3,4)  
print ('原始数组是：') 
print (a) 
print ('\n') 
print  ('以 F 风格顺序排序：')
c = b.copy(order='F')  
print (c)
for x in np.nditer(c):  
    print (x, end=", " )
```
输出结果为：

原始数组是：
[[ 0  5 10 15]
 [20 25 30 35]
 [40 45 50 55]]


以 F 风格顺序排序：
[[ 0 20 40]
 [ 5 25 45]
 [10 30 50]
 [15 35 55]]


#### 修改数组中元素的值

nditer 对象有另一个可选参数 op_flags。 默认情况下，nditer 将视待迭代遍历的数组为只读对象（read-only），为了在遍历数组的同时，实现对数组元素值得修改，必须指定 read-write 或者 write-only 的模式。

实例
```python
import numpy as np
 
a = np.arange(0,60,5) 
a = a.reshape(3,4)  
print ('原始数组是：')
print (a)
print ('\n')
for x in np.nditer(a, op_flags=['readwrite']): 
    x[...]=2*x 
print ('修改后的数组是：')
print (a)
```


输出结果为：

原始数组是：
[[ 0  5 10 15]
 [20 25 30 35]
 [40 45 50 55]]


修改后的数组是：
[[  0  10  20  30]
 [ 40  50  60  70]
 [ 80  90 100 110]]


使用外部循环
nditer 类的构造器拥有 flags 参数，它可以接受下列值：

参数									描述
c_index							可以跟踪 C 顺序的索引
f_index						可以跟踪 Fortran 顺序的索引
multi_index					每次迭代可以跟踪一种索引类型
external_loop			给出的值是具有多个值的一维数组，而不是零维数组
在下面的实例中，		迭代器遍历对应于每列，并组合为一维数组。

实例
```python
import numpy as np 
a = np.arange(0,60,5) 
a = a.reshape(3,4)  
print ('原始数组是：')
print (a)
print ('\n')
print ('修改后的数组是：')
for x in np.nditer(a, flags =  ['external_loop'], order =  'F'):  
   print (x, end=", " )
```
输出结果为：

原始数组是：
[[ 0  5 10 15]
 [20 25 30 35]
 [40 45 50 55]]


修改后的数组是：
[ 0 20 40], [ 5 25 45], [10 30 50], [15 35 55],


### 广播迭代
如果两个数组是可广播的，nditer 组合对象能够同时迭代它们。 假设数组 a 的维度为 3X4，数组 b 的维度为 1X4 ，则使用以下迭代器（数组 b 被广播到 a 的大小）。

实例
```python
import numpy as np 
 
a = np.arange(0,60,5) 
a = a.reshape(3,4)  
print  ('第一个数组为：')
print (a)
print  ('\n')
print ('第二个数组为：')
b = np.array([1,  2,  3,  4], dtype =  int)  
print (b)
print ('\n')
print ('修改后的数组为：')
for x,y in np.nditer([a,b]):  
    print ("%d:%d"  %  (x,y), end=", " )
```
输出结果为：

第一个数组为：
[[ 0  5 10 15]
 [20 25 30 35]
 [40 45 50 55]]


第二个数组为：
[1 2 3 4]


修改后的数组为：
0:1, 5:2, 10:3, 15:4, 20:1, 25:2, 30:3, 35:4, 40:1, 45:2, 50:3, 55:4,

## Numpy 数组操作


#### numpy.reshape
numpy.reshape 函数可以在不改变数据的条件下修改形状，格式如下：

**numpy.reshape(arr, newshape, order='C')**

arr：要修改形状的数组
newshape：整数或者整数数组，新的形状应当兼容原有形状
order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'k' -- 元素在内存中的出现顺序。

实例
```python
import numpy as np
 
a = np.arange(8)
print ('原始数组：')
print (a)
print ('\n')
 
b = a.reshape(4,2)
print ('修改后的数组：')
print (b)
```

输出结果如下：

原始数组：
[0 1 2 3 4 5 6 7]

修改后的数组：
[[0 1]
 [2 3]
 [4 5]
 [6 7]]



#### numpy.ndarray.flat
numpy.ndarray.flat 是一个数组元素迭代器，实例如下:
实例
```python
import numpy as np
 
a = np.arange(9).reshape(3,3) 
print ('原始数组：')
for row in a:
    print (row)
 
#对数组中每个元素都进行处理，可以使用flat属性，该属性是一个数组元素迭代器：
print ('迭代后的数组：')
for element in a.flat:
    print (element)
```

输出结果如下：

原始数组：
[0 1 2]
[3 4 5]
[6 7 8]
迭代后的数组：
0
1
2
3
4
5
6
7
8



### numpy.ndarray.flatten
numpy.ndarray.flatten 返回一份数组拷贝，对拷贝所做的修改不会影响原始数组，格式如下：

ndarray.flatten(order='C')
参数说明：

order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'K' -- 元素在内存中的出现顺序。
实例
```python
import numpy as np
 
a = np.arange(8).reshape(2,4)
 
print ('原数组：')
print (a)
print ('\n')
# 默认按行
 
print ('展开的数组：')
print (a.flatten())
print ('\n')
 
print ('以 F 风格顺序展开的数组：')
print (a.flatten(order = 'F'))
```

输出结果如下：

原数组：
[[0 1 2 3]
 [4 5 6 7]]


展开的数组：
[0 1 2 3 4 5 6 7]


以 F 风格顺序展开的数组：
[0 4 1 5 2 6 3 7]

#### numpy.ravel
numpy.ravel() 展平的数组元素，顺序通常是"C风格"，返回的是数组视图（view，有点类似 C/C++引用reference的意味），修改会影响原始数组。

该函数接收两个参数：

numpy.ravel(a, order='C')
参数说明：

order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'K' -- 元素在内存中的出现顺序。
实例
```python
import numpy as np
 
a = np.arange(8).reshape(2,4)
 
print ('原数组：')
print (a)
print ('\n')
 
print ('调用 ravel 函数之后：')
print (a.ravel())
print ('\n')
 
print ('以 F 风格顺序调用 ravel 函数之后：')
print (a.ravel(order = 'F'))
输出结果如下：

原数组：
[[0 1 2 3]
 [4 5 6 7]]


调用 ravel 函数之后：
[0 1 2 3 4 5 6 7]


以 F 风格顺序调用 ravel 函数之后：
[0 4 1 5 2 6 3 7]
```

### 翻转数组

**数组转置**
#### numpy.transpose
#### a.T
numpy.transpose 函数用于对换数组的维度，格式如下：

numpy.transpose(arr, axes)
参数说明:

arr：要操作的数组
axes：整数列表，对应维度，通常所有维度都会对换。

实例
```python
import numpy as np
 
a = np.arange(12).reshape(3,4)
 
print ('原数组：')
print (a )
print ('\n')
 
print ('对换数组：')
print (np.transpose(a))
print (a.T)
```
输出结果如下：

原数组：
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]


对换数组：
[[ 0  4  8]
 [ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]]


转置数组：
[[ 0  4  8]
 [ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]]

numpy.ndarray.T 类似 numpy.transpose，都是转置：




#### numpy.rollaxis
numpy.rollaxis 函数向后滚动特定的轴到一个特定位置，格式如下：

numpy.rollaxis(arr, axis, start)
参数说明：

arr：数组
axis：要向后滚动的轴，其它轴的相对位置不会改变
start：默认为零，表示完整的滚动。会滚动到特定位置。

实例
```python
import numpy as np
 
# 创建了三维的 ndarray
a = np.arange(8).reshape(2,2,2)
 
print ('原数组：')
print (a)
print ('获取数组中一个值：')
print(np.where(a==6))   
print(a[1,1,0])  # 为 6
print ('\n')
 
 
# 将轴 2 滚动到轴 0（宽度到深度）
 
print ('调用 rollaxis 函数：')
b = np.rollaxis(a,2,0)
print (b)
# 查看元素 a[1,1,0]，即 6 的坐标，变成 [0, 1, 1]
# 最后一个 0 移动到最前面
print(np.where(b==6))   
print ('\n')
 
# 将轴 2 滚动到轴 1：（宽度到高度）
 
print ('调用 rollaxis 函数：')
c = np.rollaxis(a,2,1)
print (c)
# 查看元素 a[1,1,0]，即 6 的坐标，变成 [1, 0, 1]
# 最后的 0 和 它前面的 1 对换位置
print(np.where(c==6))   
print ('\n')
```
输出结果如下：
```
原数组：
[[[0 1]
  [2 3]]

 [[4 5]
  [6 7]]]
获取数组中一个值：
(array([1]), array([1]), array([0]))
6


调用 rollaxis 函数：
[[[0 2]
  [4 6]]

 [[1 3]
  [5 7]]]
(array([0]), array([1]), array([1]))


调用 rollaxis 函数：
[[[0 2]
  [1 3]]

 [[4 6]
  [5 7]]]
(array([1]), array([0]), array([1]))

#太长了用框括一下
```

1. np.where(condition, x, y)
满足条件(condition)，输出x，不满足输出y,如果是一维数组,输出数组数量的x或者y

2.np.where(condition)
只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)。这里的坐标以tuple的形式给出，通常原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标






