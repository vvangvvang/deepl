import numpy as np
import net
import relu
import dropout

ce=net.CrossEntropy()

a=np.array([[0.3,0.3,0.4],
    [0.3,0.4,0.3],
    [0.1,0.2,0.7]])
y=np.array([[0,0,1],
    [0,1,0],
    [1,0,0]])
l=ce.forward(a,y)
print(l.sum()/3)
next_g=ce.gradient(a,y)
print(next_g)
import math
b=np.log(0.4)+np.log(0.4)+np.log(0.1)
print(b/3)
