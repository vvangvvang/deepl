import numpy as np
b = np.array([[[[5],[6]],[[7],[8]]]])
b.shape
print(b.shape)

"""
(1L, 2L, 2L, 1L)
array([[[[5],
         [6]],

        [[7],
         [8]]]])
"""

b_squeeze = b.squeeze()
print(b_squeeze.shape)
#默认压缩所有为1的维度


b_squeeze0 = b.squeeze(axis=None)   
#调用array实例的方法
print(b_squeeze0.shape)


b_squeeze3 = np.squeeze(b, axis=(0,1,2,3))   
#调用numpy的方法
print(b_squeeze3.shape)

c=(1,2,3)
print(type(c))

