import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

b=torch.FloatTensor(1,2,3,4)
print(b.shape)
print(b.transpose(1,3).shape)
c=b.transpose(1,3).contiguous().view(1,2*3*4).view(1,4,3,2).transpose(1,3)     #不推荐这种方式，会导致数据污染
d=b.transpose(1,3).transpose(1,3)

print(torch.all(torch.eq(b,c)))
print(torch.all(torch.eq(b,d)))


def himmelblau(x):
	return 	((x[0] ** 2+ x[1] -11)**2 +(x[0] + x[1]** 2 -7)**2)
#原函数
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
#输出图像
x=torch.tensor([4.,-1.],requires_grad=True)
optimizer =torch.optim.Adam([x], lr=1e-3)
for step in range(20000):
	pred =himmelblau(x)

	optimizer.zero_grad()
	pred.backward()
	optimizer.step()
	if step%2000 == 0 :
		print('step {} : x={} , f(x)= {}'
			  .format(step,x.tolist(),pred.item()))