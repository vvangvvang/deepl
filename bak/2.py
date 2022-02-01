import net.net as net
import numpy as np
import net.bn as bn
#生成一个线性回归数据集
x=np.random.randn(300,1)
print('x',x)
y=np.random.randn(x.shape[0],x.shape[1])
print('y',y)
#以下是需要学习出来的模型
z=x*6.2+y*1.1+3.3

c=np.concatenate((x,y),axis=1)
print('c',c.shape)
rnd=np.random.randn(c.shape[0],c.shape[1])
c=c+rnd*0.001
rnd=np.random.randn(z.shape[0],z.shape[1])
z=z+rnd*0.01
fc=net.Dense(2,1)
bn=bn.Norm()
for i in range(300000):
    a=bn.forward(c)
    pred=fc.forward(a)
    next_g=pred-z
    error=next_g**2
    print(np.sum(error)/z.shape[0])
    next_g = next_g/c.shape[0]
    next_g=fc.gradient(a,next_g)
    fc.update(lr=0.02)
    next_g=bn.gradient(c,next_g)
print(fc.W,fc.b)

