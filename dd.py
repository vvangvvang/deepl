import net
import bn
import data
import numpy as np
import relu

conv0=net.Conv2D(in_ch=1,out_ch=32,k_size=3,stride=1)
bn0=bn.BatchNorm()
relu=relu.Relu()
conv1=net.Conv2D(in_ch=32,out_ch=48,k_size=3,stride=2)
bn1=bn.BatchNorm()
conv2=net.Conv2D(in_ch=48,out_ch=64,k_size=3,stride=2)
bn2=bn.BatchNorm()
net1 = net.Dense(64*5*5,128)
net2 = net.Dense(128,10)
sm=net.Softmax()
ce=net.CrossEntropy()

def load():
    conv0.fc.W=np.load('data/conv0.fc.w.npy')
    conv0.fc.b=np.load('data/conv0.fc.b.npy')
    conv1.fc.W=np.load('data/conv1.fc.w.npy')
    conv1.fc.b=np.load('data/conv1.fc.b.npy')
    conv2.fc.W=np.load('data/conv2.fc.w.npy')
    conv2.fc.b=np.load('data/conv2.fc.b.npy')
    net1.W=np.load('data/net1.w.npy')
    net1.b=np.load('data/net1.b.npy')
    net2.W=np.load('data/net2.w.npy')
    net2.b=np.load('data/net2.b.npy')
    bn0.gama=np.load('data/bn0.gama.npy')
    bn0.bate=np.load('data/bn0.bate.npy')
    bn1.gama=np.load('data/bn1.gama.npy')
    bn1.bate=np.load('data/bn1.bate.npy')
    bn2.gama=np.load('data/bn2.gama.npy')
    bn2.bate=np.load('data/bn2.bate.npy')

def save():
    np.save('data/conv0.fc.w',conv0.fc.W)
    np.save('data/conv0.fc.b',conv0.fc.b)
    np.save('data/conv1.fc.w',conv1.fc.W)
    np.save('data/conv1.fc.b',conv1.fc.b)
    np.save('data/conv2.fc.w',conv2.fc.W)
    np.save('data/conv2.fc.b',conv2.fc.b)
    np.save('data/net1.w',net1.W)
    np.save('data/net1.b',net1.b)
    np.save('data/net2.w',net2.W)
    np.save('data/net2.b',net2.b)
    np.save('data/bn0.gama',bn0.gama)
    np.save('data/bn0.bate',bn0.bate)
    np.save('data/bn1.gama',bn1.gama)
    np.save('data/bn1.bate',bn1.bate)
    np.save('data/bn2.gama',bn2.gama)
    np.save('data/bn2.bate',bn2.bate)

#load()
trainX=np.load("data/trainx.npy")
trainY=np.load("data/trainy.npy")

batch_size = 128
imgs_length = trainX.shape[0]#总长度
index_list = list(range(imgs_length))
values = trainY
n_values = np.max(values) + 1
label = np.eye(n_values)[values]

print("ok")
print(label.shape)
lr_=0.0005
epoch=1
step=0
for epoch_id in range(epoch):
    np.random.shuffle(index_list)
    for i in range(len(index_list)//batch_size):
        step += 1
        lr=lr_/step
        x,y = [],[]
        batch=i*batch_size
        for j in range(batch_size):
            index=batch+j
            x.append(trainX[index_list[index]])
            y.append(label[index_list[index]])
        x = np.array(x)
        x = x/128. - 1
        y = np.array(y)
        x = x.reshape([x.shape[0],1,28,28])
        a = conv0.forward(x)
        b = bn0.forward(a)
        c = conv1.forward(b)
        d = bn1.forward(c)
        e = conv2.forward(d)
        print(e.shape,'e')
        f = e.reshape([e.shape[0],e.shape[1]*e.shape[2]*e.shape[3]])
        g = net1.forward(f)
        print(g.shape)
        h=bn2.forward(g)
        p=relu.forward(h)
        q = net2.forward(p)
        r = sm.forward(q)
        #计算交叉熵损失
        cost = ce.forward(r,y)
        print(r)
        loss = np.mean(cost)
        next_g = ce.gradient(r,y)
        next_g = sm.gradient(q,next_g)
        next_g = net2.gradient(p,next_g)
        net2.update(lr=lr)
        next_g =relu.gradient(h,next_g)
        next_g = bn2.gradient(g,next_g)
        bn2.update(lr=lr)
        next_g = net1.gradient(f,next_g)
        net1.update(lr=lr)
        next_g = next_g.reshape([e.shape[0],e.shape[1],e.shape[2],e.shape[3]])
        next_g = conv2.gradient(d,next_g)
        conv2.update(lr=lr)
        next_g = bn1.gradient(c,next_g)
        bn1.update(lr=lr)
        next_g = conv1.gradient(b,next_g)
        conv1.update(lr=lr)
        next_g=bn0.gradient(a,next_g)
        bn0.update(lr=lr)
        next_g = conv0.gradient(x,next_g)
        conv0.update(lr=lr)
        if i % 1 == 0:
            print('epoch:',epoch_id,'step:',step,'loss:',loss)
        #save()




