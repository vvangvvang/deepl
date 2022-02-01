import numpy as np
import net.net as net
import time
conv=net.Conv2D(3,32,3,2)
#a=[i for i in range(18)]
#w=[i for i in range(1,9)]
a=np.random.randn(1024,3,96,96)
conv2=net.Conv2D2(3,32,3,2)
#conv2.fc.W=np.concatenate([w,w],axis=1)
#conv2.fc.b[:]=0.
start=time.time()
c=conv2.forward(a)
end=time.time()
print("conv2d2 time",end-start)
start=time.time()
conv.forward(a)
end=time.time()
print("conv2d time",end-start)
start=time.time()
conv2.back(c)
end=time.time()
print("conv2d2 back time",end-start)
start=time.time()
conv.back(c)
end=time.time()
print("conv2d back time",end-start)

