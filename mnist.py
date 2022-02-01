import numpy as np
import lib.base as net
import time

m = net.Sequence()
m.add(net.Conv2D(1,32,5,0,2))
m.add(net.BatchNorm(32))
m.add(net.Relu())
m.add(net.Dropout(0.))
m.add(net.Conv2D(32,128,3,0,2))
m.add(net.BatchNorm(128))
m.add(net.Relu())
m.add(net.Dropout(0.))
m.add(net.Reshape((128*5*5,),'CHW'))
m.add(net.Dense(128*5*5,512))
m.add(net.Relu())
m.add(net.Dropout(0.))
m.add(net.Dense(512,10))
m.load("weight/")
sce = net.Softmax_CrossEntropy()
mean = net.Mean()


trainX=np.load("data/trainx.npy")
batch_size = 1
imgs_length = trainX.shape[0]#总长度
# 定义数据集每个数据的序号，根据序号读取数据
index_list = list(range(imgs_length))
label=np.load("data/label.npy")

epoch = 1
start=time.time()
for epoch_id in range(epoch):
    # 随机打乱训练数据的索引序号
    np.random.shuffle(index_list)
    for i in range(len(index_list)//batch_size):
        x,y,yyy = [],[],[]
        for j in range(batch_size):
            x.append(trainX[index_list[i*batch_size+j]])
            y.append(label[index_list[i*batch_size+j]])
        x = np.array(x)
        x = x / 128. - 1.
        y = np.array(y)
        x = x.reshape([x.shape[0],1,28,28])
        z = m.forward(x)
        sm = net.Softmax()
        ce = net.CrossEntropy()
        loss1=sm.forward(z)
        print("softmax",loss1)
        loss2=ce.forward(loss1,y)
        print(loss2)
        loss,pred = sce.forward(z,y)
        print(loss)
        loss=mean.forward(loss)
        next_g=mean.back(1.0)
        print(next_g)
        print("核对精度")
        g2=ce.back(next_g)
        print("ce back",g2)
        g2=sm.back(g2)
        print(g2)
        next_g=sce.back(next_g)
        print(next_g)
        next_g=m.back(next_g)
        acc=net.acc(pred,y)
        import lib.display as disp
        disp.disp(x[0,0],"@wxrl- ")
        print("标签是:",np.argmax(y))
        print("预测值是:",np.argmax(pred))
        print("epoch",epoch_id,"i",i,"acc",acc,"loss",loss)
        end=time.time()
        print("本批次耗时:",end-start)
        start=end
        break
    break
#m.save("model/")
