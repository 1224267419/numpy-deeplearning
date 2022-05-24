## ��������
```
numpy.dtype(object, align, copy)
```
object - Ҫת��Ϊ���������Ͷ���   ����������Ͷ�������
align - ���Ϊ true������ֶ�ʹ������ C �Ľṹ�塣  
copy - ���� dtype ���� �����Ϊ false�����Ƕ������������Ͷ��������

##### ����������Ӧ���� ndarray ����
```
dt = np.dtype([('age',np.int8)]) 
a = np.array([(10,),(20,),(30,)], dtype = dt) 
print(a)
```
�����[10 20 30]


#####  �����ʾ������һ���ṹ���������� student�������ַ����ֶ� name�������ֶ� age���������ֶ� marks
```
student = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')]) 
print(student)
```
������Ϊ��
[('name', 'S20'), ('age', 'i1'), ('marks', 'f4')]

## ��������
NumPy �����ά����Ϊ�ȣ�rank�����Ⱦ�������������������ά��

``` 
a = np.arange(24)  
print (a.ndim)             # a ��ֻ��һ��ά��

#���ڵ������С
b = a.reshape(2,4,3)  # b ����ӵ������ά��
print (b.ndim)
```
�����
1
3



#### ndarray.flags
ndarray.flags ���� ndarray ������ڴ���Ϣ�������������ԣ�

����                                 	����
C_CONTIGUOUS (C)	��������һ����һ��C������������
F_CONTIGUOUS (F)	��������һ����һ��Fortran������������
OWNDATA (O)	����ӵ������ʹ�õ��ڴ�����һ�������н�����
WRITEABLE (W)	����������Ա�д�룬����ֵ����Ϊ False��������Ϊֻ��
ALIGNED (A) 	���ݺ�����Ԫ�ض��ʵ��ض��뵽Ӳ����
UPDATEIFCOPY (U)	������������������һ����������������鱻�ͷ�ʱ��ԭ��������ݽ�������
ʵ��
```python
import numpy as np 
 
x = np.array([1,2,3,4,5])  
print (x.flags)
������Ϊ��

  C_CONTIGUOUS : True
  F_CONTIGUOUS : True
  OWNDATA : True
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
```

## NumPy ��������
numpy.empty
numpy.empty ������������һ��ָ����״��shape�����������ͣ�dtype����δ��ʼ�������飺

numpy.empty(shape, dtype = float, order = 'C')
����˵����

����	����
shape	������״
dtype	�������ͣ���ѡ
order	��"C"��"F"����ѡ��,�ֱ���������Ⱥ������ȣ��ڼ�����ڴ��еĴ洢Ԫ�ص�˳�򡣣�Ĭ����C��

```python
import numpy as np 
x = np.empty([3,2], dtype = int) 
print (x)
```

����������Ԫ��ֵ�������Ϊû�г�ʼ��

������������ͬ������ֵȫΪ0
numpy.zeros
����ָ����С�����飬����Ԫ���� 0 ����䣺
```
numpy.zeros(shape, dtype = float, order = 'C')
```
����ָ����С�����飬����Ԫ���� 1 ����䣺
```python
numpy.ones(shape, dtype = None, order = 'C')
```

numpy.asarray ���� numpy.array���� numpy.asarray ����ֻ���������� numpy.array ��������
```python
numpy.asarray(a, dtype = None, order = None)

```
a	������ʽ����������������ǣ��б�, �б��Ԫ��, Ԫ��, Ԫ���Ԫ��, Ԫ����б���ά����


```python
 numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)   ���� Ndarray����
```
��ȿ����Ǹ�����������ʹ����*���ڴ��к����ƶ�
����	����
object	�����Ƕ�׵�����
dtype	����Ԫ�ص��������ͣ���ѡ
copy	�����Ƿ���Ҫ���ƣ���ѡ
order	�����������ʽ��CΪ�з���FΪ�з���AΪ���ⷽ��Ĭ�ϣ�
subok	Ĭ�Ϸ���һ�����������һ�µ�����
ndmin	ָ�������������Сά��

```
numpy.arange(start, stop, step, dtype)
```

���� start �� stop ָ���ķ�Χ�Լ� step �趨�Ĳ���������һ�� ndarray

����	����
start	��ʼֵ��Ĭ��Ϊ0
stop	��ֵֹ����������
step	������Ĭ��Ϊ1
dtype	����ndarray���������ͣ����û���ṩ�����ʹ���������ݵ����͡�

```
import numpy as np
x = np.arange(10,20,2,data = int)  
print (x)
```
���������£�

[10  12  14  16  18]



numpy.linspace �������ڴ���һ��һά���飬������һ���Ȳ����й��ɵģ���ʽ���£�
```
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
```


start	���е���ʼֵ
stop	���е���ֵֹ�����endpointΪtrue����ֵ������������
num	Ҫ���ɵĵȲ���������������Ĭ��Ϊ50
endpoint	��ֵΪ true ʱ�������а���stopֵ����֮��������Ĭ����True��
retstep	���Ϊ True ʱ�����ɵ������л���ʾ��࣬��֮����ʾ��
dtype	ndarray ����������

```
import numpy as np
a = np.linspace(1,10,10)
print(a)
```
������Ϊ��

[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]



###### numpy.linspace��reshapeһ����ʾ��

```
import numpy as np
a =np.linspace(1,10,10,retstep= True)
 
print(a)

b =np.linspace(1,10,10).reshape([10,1])
print(b)
������Ϊ��

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


###### �ȱ����У�

```
np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
```
base ������˼��ȡ������ʱ�� log ���±ꡣ

����	����
start	���е���ʼֵΪ��base ** start
stop	���е���ֵֹΪ��base ** stop�����endpointΪtrue����ֵ������������
num	Ҫ���ɵĵȲ���������������Ĭ��Ϊ50
endpoint	��ֵΪ true ʱ���������а���stopֵ����֮��������Ĭ����True��
base	���� log �ĵ�����
dtype	ndarray ����������


ʵ��
```
import numpy as np
# Ĭ�ϵ����� 10
a = np.logspace(1.0,  2.0, num =  10)  
print (a)
```

������Ϊ��

[ 10.           12.91549665     16.68100537      21.5443469  27.82559402      
  35.93813664   46.41588834     59.94842503      77.42636827    100.    ]
(base *start )�ǳ�ֵ��(base *stop )��ĩֵ��

## ��Ƭ
```python
import numpy as np
 
a = np.arange(10)
s = slice(2,7,2)   # ������ 2(Ĭ��0) ��ʼ������ 7(Ĭ��n) ֹͣ�����Ϊ2(Ĭ��1)
b=a[2:7:2]   # ������ 2 ��ʼ������ 7 ֹͣ�����Ϊ 2��������ͷûβ��
b=a[2:6:2]

print (a[s])
print(b)
print(c)
```

�����
[2  4  6]
[2  4  6]
[2  4 ]


��ά����ͬ����������������ȡ������

ʵ��
```python
import numpy as np
 
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a)
```

#### ��ĳ����������ʼ�и�
```
import numpy as np
 
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a)
print('���������� a[1:] ����ʼ�и�')
print(a[1:])
```
������Ϊ��

[[1 2 3]
 [3 4 5]
 [4 5 6]]
���������� a[1:] ����ʼ�и�
[[3 4 5]
 [4 5 6]]


��Ƭ�����԰���ʡ�Ժ� x������ʹѡ��Ԫ��ĳ����������ά����ͬ�� �������λ��ʹ��ʡ�Ժţ��������ذ�������Ԫ�ص� ndarray��

ʵ��
```
import numpy as np
 
a = np.array([[1,2,3],[3,4,5],[4,5,6]])  
print (a[...,1])   # ��2��Ԫ��
print (a[1,...])   # ��2��Ԫ��
print (a[...,1:])  # ��2�м�ʣ�µ�����Ԫ��

```
������Ϊ��

[2 4 5]
[3 4 5]
[[2 3]
 [4 5]
 [5 6]]


�����б�û�е�

����ʵ����ȡ������(0,0)��(1,1)��(2,0)λ�ô���Ԫ�ء�

ʵ��
 ```python
import numpy as np 
a = np.array([[1,  2],  [3,  4],  [5,  6]]) 
b = a[[0,1,2],  [0,1,0]]  #ǰ��Ϊ
print (b)
```
������Ϊ��

[1  4  5]


����ʵ����ȡ�� 4X3 �����е��ĸ��ǵ�Ԫ�ء� �������� [0,0] �� [3,3]������������ [0,2] �� [0,2]��

ʵ��
```
import numpy as np 
 
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
print ('���ǵ������ǣ�' )
print (x)
print ('\n')
rows = np.array([[0,0],[3,3]]) # (�����꣩
cols = np.array([[0,2],[0,2]]) #�������꣩
y = x[rows,cols]  
print  ('���������ĸ���Ԫ���ǣ�')
print (y)
```
������Ϊ��

���ǵ������ǣ�
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]


���������ĸ���Ԫ���ǣ�
[[ 0  2]
 [ 9 11]]


������Ƭ : �� �� ������������ϡ����������ӣ�
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

������Ϊ��

[[5 6]
 [8 9]]

[[5 6]
 [8 9]]

[[2 3]
 [5 6]
 [8 9]]



2.2 ��������
���ǿ���ͨ��һ����������������Ŀ�����顣

��������ͨ���������㣨�磺�Ƚ������������ȡ����ָ��������Ԫ�ص����顣

ʵ������ȡ���� 5 ��Ԫ��
```python
import numpy as np 
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
print ('���ǵ������ǣ�')
print (x)
print ('\n')
print  ('���� 5 ��Ԫ���ǣ�')
print (x[x >  5])
```

���ǵ������ǣ�
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

���� 5 ��Ԫ���ǣ�
[ 6  7  8  9 10 11]



```
import numpy as np 
 
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
print ('���ǵ������ǣ�')
print (x)
print ('\n')
# �������ǻ��ӡ������ 5 ��Ԫ��  
print  ('���� 5 ��Ԫ���ǣ�')
print (x[x >  5])
```
������Ϊ��

-���ǵ������ǣ�
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

���� 5 ��Ԫ���ǣ�
[ 6  7  8  9 10 11]


###### sp��
np.nan��һ��float���͵����� ��None�е��񣬵�����ȫһ��
np.isnan()�����жϴ�ֵ�Ƿ�Ϊ��ֵ   ������������None==np.nan�жϣ���Ϊ��ֵ����Ȳ��ܱ���ȷʶ��
np.isnan(np.nan)    �������һ��bool�͵� True�������ж�ֵ�Ƿ�Ϊ��

����ʵ��ʹ���� ~��ȡ��������������� NaN��
~��ʾ��������λȡ��             ~5=-6
ʵ��
```python
import numpy as np 
 
a = np.array([np.nan,  1,2,np.nan,3,4,5])  
print (a[~np.isnan(a)])         
```
������Ϊ��

[ 1.   2.   3.   4.   5.]




1������˳����������

ʵ��
```
import numpy as np 
 
x=np.arange(32).reshape((8,4))
print (x[[4,2,1,7]])   #(��0)��0�ǵ�һ��
```
������Ϊ��

[[16 17 18 19]
 [ 8  9 10 11]
 [ 4  5  6  7]
 [28 29 30 31]]
2�����뵹����������

ʵ��
```
import numpy as np 
 
x=np.arange(32).reshape((8,4))
print (x[[-4,-2,-1,-7]])     #-1�������һ��
```
������Ϊ��

[[16 17 18 19]
 [24 25 26 27]
 [28 29 30 31]
 [ 4  5  6  7]]
**˳��ѡȡ�Ǵ�0��ʼ���ģ�a[0]�����һ����������ѡȡ�Ǵ�1��ʼ���ģ�a[-1]�ǵ�����һ��**


**�Ȱ�����Ҫ��ѡȡ�У��ٰ�˳�������򣬻��һ������**
```
arr2 = np.arange(32).reshape((8,4))

print(arr2[[1,5,7,2]][:,[0,3,1,2]])

 

```

�����
array([[ 4,  7,  5,  6],
       [20, 23, 21, 22],
       [28, 31, 29, 30],
       [ 8, 11,  9, 10]])


3���������������飨Ҫʹ��np.ix_��

np.ix_�������ܰ�����һά���� ת��Ϊ һ������ѡȡ���������������

ʵ����˼���ǣ�ֱ����np.ix_()���ӽ�����һά����[1,5,7,2]��[0,3,1,2]�������Ȱ�����Ҫ��ѡȡ�У��ٰ�˳�������򣬸�����õ��Ľ��һ����������д��[ : , [0,3,1,2] ]��

ʵ��
```
import numpy as np 
 
x=np.arange(32).reshape((8,4))
print (x[np.ix_([1,5,7,2],[0,3,1,2])])
```
������Ϊ��

[[ 4  7  5  6]
 [20 23 21 22]
 [28 31 29 30]
 [ 8 11  9 10]]

## NumPy �㲥(Broadcast)
�㲥(Broadcast)�� numpy �Բ�ͬ��״(shape)�����������ֵ����ķ�ʽ�� ���������������ͨ������Ӧ��Ԫ���Ͻ��С�

����������� a �� b ��״��ͬ�������� a.shape == b.shape����ô a*b �Ľ������ a �� b �����Ӧλ��ˡ���Ҫ��ά����ͬ���Ҹ�ά�ȵĳ�����ͬ��

ʵ��
import numpy as np 
 
a = np.array([1,2,3,4]) 
b = np.array([10,20,30,40]) 
c = a * b 
print (c)

������Ϊ��

[ 10  40  90 160]




�㲥�Ĺ���:

�������������鶼��������״������鿴�룬��״�в���Ĳ��ֶ�ͨ����ǰ��� 1 ���롣
����������״������������״�ĸ���ά���ϵ����ֵ��
������������ĳ��ά�Ⱥ��������Ķ�Ӧά�ȵĳ�����ͬ�����䳤��Ϊ 1 ʱ����������ܹ��������㣬�������
�����������ĳ��ά�ȵĳ���Ϊ 1 ʱ�����Ŵ�ά������ʱ���ô�ά���ϵĵ�һ��ֵ��
����⣺���������飬�ֱ�Ƚ����ǵ�ÿһ��ά�ȣ�������һ������û�е�ǰά������ԣ������㣺

����ӵ����ͬ��״��
��ǰά�ȵ�ֵ��ȡ�
��ǰά�ȵ�ֵ��һ���� 1��


## NumPy ��������
NumPy ���������� numpy.nditer �ṩ��һ��������һ�����߶������Ԫ�صķ�ʽ��
```python
import numpy as np
 
a = np.arange(6).reshape(2,3)
print ('ԭʼ�����ǣ�')
print (a)
print ('\n')
print ('�������Ԫ�أ�')
for x in np.nditer(a):
    print (x, end=", " )
print ('\n')
```


������Ϊ��

ԭʼ�����ǣ�
[[0 1 2]
 [3 4 5]]


�������Ԫ�أ�
0, 1, 2, 3, 4, 5, 

**����ʵ������ʹ�ñ�׼ C ���� Fortran ˳��ѡ���˳���Ǻ������ڴ沼��һ�µģ���������Ϊ���������ʵ�Ч�ʣ�Ĭ�����������ȣ�row-major order������˵�� C-order����
�ⷴӳ��Ĭ�������ֻ�����ÿ��Ԫ�أ������迼�����ض�˳�����ǿ���ͨ���������������ת����������һ�㣬������ C ˳���������ת�õ� copy ��ʽ���Աȣ�����ʵ����**

```python
import numpy as np
 
a = np.arange(6).reshape(2,3)
for x in np.nditer(a.T):
    print (x, end=", " )
print ('\n')
 
for x in np.nditer(a.T.copy(order='C')):
    print (x, end=", " )
print ('\n')
������Ϊ��

0, 1, 2, 3, 4, 5, 

0, 3, 1, 4, 2, 5, 
```
a �� a.T �ı���˳����һ���ģ�Ҳ�����������ڴ��еĴ洢˳��Ҳ��һ���ģ����� a.T.copy(order = 'C') �ı�������ǲ�ͬ�ģ�������Ϊ����ǰ���ֵĴ洢��ʽ�ǲ�һ���ģ�Ĭ���ǰ��з��ʡ�


 ### ���Ʊ���˳��
```python
import numpy as np
 
a = np.arange(0,60,5) 
a = a.reshape(3,4)  
print ('ԭʼ�����ǣ�') 
print (a) 
print ('\n') 
print  ('�� F ���˳������')
c = b.copy(order='F')  
print (c)
for x in np.nditer(c):  
    print (x, end=", " )
```
������Ϊ��

ԭʼ�����ǣ�
[[ 0  5 10 15]
 [20 25 30 35]
 [40 45 50 55]]


�� F ���˳������
[[ 0 20 40]
 [ 5 25 45]
 [10 30 50]
 [15 35 55]]


#### �޸�������Ԫ�ص�ֵ

nditer ��������һ����ѡ���� op_flags�� Ĭ������£�nditer ���Ӵ���������������Ϊֻ������read-only����Ϊ���ڱ��������ͬʱ��ʵ�ֶ�����Ԫ��ֵ���޸ģ�����ָ�� read-write ���� write-only ��ģʽ��

ʵ��
```python
import numpy as np
 
a = np.arange(0,60,5) 
a = a.reshape(3,4)  
print ('ԭʼ�����ǣ�')
print (a)
print ('\n')
for x in np.nditer(a, op_flags=['readwrite']): 
    x[...]=2*x 
print ('�޸ĺ�������ǣ�')
print (a)
```


������Ϊ��

ԭʼ�����ǣ�
[[ 0  5 10 15]
 [20 25 30 35]
 [40 45 50 55]]


�޸ĺ�������ǣ�
[[  0  10  20  30]
 [ 40  50  60  70]
 [ 80  90 100 110]]


ʹ���ⲿѭ��
nditer ��Ĺ�����ӵ�� flags �����������Խ�������ֵ��

����									����
c_index							���Ը��� C ˳�������
f_index						���Ը��� Fortran ˳�������
multi_index					ÿ�ε������Ը���һ����������
external_loop			������ֵ�Ǿ��ж��ֵ��һά���飬��������ά����
�������ʵ���У�		������������Ӧ��ÿ�У������Ϊһά���顣

ʵ��
```python
import numpy as np 
a = np.arange(0,60,5) 
a = a.reshape(3,4)  
print ('ԭʼ�����ǣ�')
print (a)
print ('\n')
print ('�޸ĺ�������ǣ�')
for x in np.nditer(a, flags =  ['external_loop'], order =  'F'):  
   print (x, end=", " )
```
������Ϊ��

ԭʼ�����ǣ�
[[ 0  5 10 15]
 [20 25 30 35]
 [40 45 50 55]]


�޸ĺ�������ǣ�
[ 0 20 40], [ 5 25 45], [10 30 50], [15 35 55],


### �㲥����
������������ǿɹ㲥�ģ�nditer ��϶����ܹ�ͬʱ�������ǡ� �������� a ��ά��Ϊ 3X4������ b ��ά��Ϊ 1X4 ����ʹ�����µ����������� b ���㲥�� a �Ĵ�С����

ʵ��
```python
import numpy as np 
 
a = np.arange(0,60,5) 
a = a.reshape(3,4)  
print  ('��һ������Ϊ��')
print (a)
print  ('\n')
print ('�ڶ�������Ϊ��')
b = np.array([1,  2,  3,  4], dtype =  int)  
print (b)
print ('\n')
print ('�޸ĺ������Ϊ��')
for x,y in np.nditer([a,b]):  
    print ("%d:%d"  %  (x,y), end=", " )
```
������Ϊ��

��һ������Ϊ��
[[ 0  5 10 15]
 [20 25 30 35]
 [40 45 50 55]]


�ڶ�������Ϊ��
[1 2 3 4]


�޸ĺ������Ϊ��
0:1, 5:2, 10:3, 15:4, 20:1, 25:2, 30:3, 35:4, 40:1, 45:2, 50:3, 55:4,

## Numpy �������


#### numpy.reshape
numpy.reshape ���������ڲ��ı����ݵ��������޸���״����ʽ���£�

**numpy.reshape(arr, newshape, order='C')**

arr��Ҫ�޸���״������
newshape�����������������飬�µ���״Ӧ������ԭ����״
order��'C' -- ���У�'F' -- ���У�'A' -- ԭ˳��'k' -- Ԫ�����ڴ��еĳ���˳��

ʵ��
```python
import numpy as np
 
a = np.arange(8)
print ('ԭʼ���飺')
print (a)
print ('\n')
 
b = a.reshape(4,2)
print ('�޸ĺ�����飺')
print (b)
```

���������£�

ԭʼ���飺
[0 1 2 3 4 5 6 7]

�޸ĺ�����飺
[[0 1]
 [2 3]
 [4 5]
 [6 7]]



#### numpy.ndarray.flat
numpy.ndarray.flat ��һ������Ԫ�ص�������ʵ������:
ʵ��
```python
import numpy as np
 
a = np.arange(9).reshape(3,3) 
print ('ԭʼ���飺')
for row in a:
    print (row)
 
#��������ÿ��Ԫ�ض����д�������ʹ��flat���ԣ���������һ������Ԫ�ص�������
print ('����������飺')
for element in a.flat:
    print (element)
```

���������£�

ԭʼ���飺
[0 1 2]
[3 4 5]
[6 7 8]
����������飺
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
numpy.ndarray.flatten ����һ�����鿽�����Կ����������޸Ĳ���Ӱ��ԭʼ���飬��ʽ���£�

ndarray.flatten(order='C')
����˵����

order��'C' -- ���У�'F' -- ���У�'A' -- ԭ˳��'K' -- Ԫ�����ڴ��еĳ���˳��
ʵ��
```python
import numpy as np
 
a = np.arange(8).reshape(2,4)
 
print ('ԭ���飺')
print (a)
print ('\n')
# Ĭ�ϰ���
 
print ('չ�������飺')
print (a.flatten())
print ('\n')
 
print ('�� F ���˳��չ�������飺')
print (a.flatten(order = 'F'))
```

���������£�

ԭ���飺
[[0 1 2 3]
 [4 5 6 7]]


չ�������飺
[0 1 2 3 4 5 6 7]


�� F ���˳��չ�������飺
[0 4 1 5 2 6 3 7]

#### numpy.ravel
numpy.ravel() չƽ������Ԫ�أ�˳��ͨ����"C���"�����ص���������ͼ��view���е����� C/C++����reference����ζ�����޸Ļ�Ӱ��ԭʼ���顣

�ú�����������������

numpy.ravel(a, order='C')
����˵����

order��'C' -- ���У�'F' -- ���У�'A' -- ԭ˳��'K' -- Ԫ�����ڴ��еĳ���˳��
ʵ��
```python
import numpy as np
 
a = np.arange(8).reshape(2,4)
 
print ('ԭ���飺')
print (a)
print ('\n')
 
print ('���� ravel ����֮��')
print (a.ravel())
print ('\n')
 
print ('�� F ���˳����� ravel ����֮��')
print (a.ravel(order = 'F'))
���������£�

ԭ���飺
[[0 1 2 3]
 [4 5 6 7]]


���� ravel ����֮��
[0 1 2 3 4 5 6 7]


�� F ���˳����� ravel ����֮��
[0 4 1 5 2 6 3 7]
```

### ��ת����

**����ת��**
#### numpy.transpose
#### a.T
numpy.transpose �������ڶԻ������ά�ȣ���ʽ���£�

numpy.transpose(arr, axes)
����˵��:

arr��Ҫ����������
axes�������б���Ӧά�ȣ�ͨ������ά�ȶ���Ի���

ʵ��
```python
import numpy as np
 
a = np.arange(12).reshape(3,4)
 
print ('ԭ���飺')
print (a )
print ('\n')
 
print ('�Ի����飺')
print (np.transpose(a))
print (a.T)
```
���������£�

ԭ���飺
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]


�Ի����飺
[[ 0  4  8]
 [ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]]


ת�����飺
[[ 0  4  8]
 [ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]]

numpy.ndarray.T ���� numpy.transpose������ת�ã�




#### numpy.rollaxis
numpy.rollaxis �����������ض����ᵽһ���ض�λ�ã���ʽ���£�

numpy.rollaxis(arr, axis, start)
����˵����

arr������
axis��Ҫ���������ᣬ����������λ�ò���ı�
start��Ĭ��Ϊ�㣬��ʾ�����Ĺ�������������ض�λ�á�

ʵ��
```python
import numpy as np
 
# ��������ά�� ndarray
a = np.arange(8).reshape(2,2,2)
 
print ('ԭ���飺')
print (a)
print ('��ȡ������һ��ֵ��')
print(np.where(a==6))   
print(a[1,1,0])  # Ϊ 6
print ('\n')
 
 
# ���� 2 �������� 0����ȵ���ȣ�
 
print ('���� rollaxis ������')
b = np.rollaxis(a,2,0)
print (b)
# �鿴Ԫ�� a[1,1,0]���� 6 �����꣬��� [0, 1, 1]
# ���һ�� 0 �ƶ�����ǰ��
print(np.where(b==6))   
print ('\n')
 
# ���� 2 �������� 1������ȵ��߶ȣ�
 
print ('���� rollaxis ������')
c = np.rollaxis(a,2,1)
print (c)
# �鿴Ԫ�� a[1,1,0]���� 6 �����꣬��� [1, 0, 1]
# ���� 0 �� ��ǰ��� 1 �Ի�λ��
print(np.where(c==6))   
print ('\n')
```
���������£�
```
ԭ���飺
[[[0 1]
  [2 3]]

 [[4 5]
  [6 7]]]
��ȡ������һ��ֵ��
(array([1]), array([1]), array([0]))
6


���� rollaxis ������
[[[0 2]
  [4 6]]

 [[1 3]
  [5 7]]]
(array([0]), array([1]), array([1]))


���� rollaxis ������
[[[0 2]
  [1 3]]

 [[4 6]
  [5 7]]]
(array([1]), array([0]), array([1]))

#̫�����ÿ���һ��
```

1. np.where(condition, x, y)
��������(condition)�����x�����������y,�����һά����,�������������x����y

2.np.where(condition)
ֻ������ (condition)��û��x��y��������������� (����0) Ԫ�ص����� (�ȼ���numpy.nonzero)�������������tuple����ʽ������ͨ��ԭ�����ж���ά�������tuple�оͰ����������飬�ֱ��Ӧ��������Ԫ�صĸ�ά����






