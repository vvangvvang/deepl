import numpy as np
import net
import relu
import dropout

sm=net.Softmax()

a=np.array([[3,1,-3]])
y=np.array([[0,0,1],
    [0,1,0],
    [1,0,0]])
p=sm.forward(a)
print(p)

