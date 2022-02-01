import lib.base as nn
import bn
import data
import numpy as np
import relu

net1 = net.Dense(784,128)
net2 = net.Dense(128,10)
trainX=np.load("data/trainx.npy")
trainY=np.load("data/trainy.npy")

batch_size = 256
imgs_length = trainX.shape[0]#总长度
index_list = list(range(imgs_length))
values = trainY
n_values = np.max(values) + 1
label = np.eye(n_values)[values]

print("ok")
print(label.shape)
lr_=0.000000005
epoch=100
step=1
for epoch_id in range(epoch):
    np.random.shuffle(index_list)
    for i in range(len(index_list)//batch_size):
        step += 1
        lr=lr_/step
        x,y = [],[]
        for j in range(batch_size):
            x.append(trainX[index_list[i*batch_size+j]])
            y.append(label[index_list[i*batch_size+j]])
        x = np.array(x)
        x = x/128. - 1
        y = np.array(y)
        a = net1.forward(x)
        b = net2.forward(a)
        #计算交叉熵损失
        print(b[0])
        print(y[0])
        next_g=b-y
        cost=next_g**2
        loss = np.mean(cost)
        next_g = net2.gradient(a,next_g)
        net2.update(lr=lr)
        next_g = net1.gradient(x,next_g)
        net1.update(lr=lr)
        if i % 1 == 0:
            print('epoch:',epoch_id,'step:',step,'loss:',loss)
        #save()




