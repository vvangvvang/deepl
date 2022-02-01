import numpy as np
#基础类
class Layer(object):
    def __init__(self,name="",lr=0.00002):
        self.lr=lr
        self.name="layer_"+name
        self.mode="train" #train or test
    def forward(self,x):
        return x
    def back(self,next_g):
        return next_g
    def save(self,path=""):
        pass
    def load(self,path=""):
        pass
    def set_mode(self,mode="train"):
        self.mode=mode
    def __call__(self,x):
        return self.forward(x)
#主要的数学操作
#相加
class Add(Layer):
    def __init__(self):
        super(Add,self).__init__("add")
    def forward(self,x,y):
        return x + y
    def back(self,next_g):
        return next_g,next_g
#相减
class Sub(Layer):
    def __init__(self):
        super(Sub,self).__init__("sub")
    def forward(self,x,y):
        return x - y
    def back(self,next_g):
        return next_g,-next_g
#求和
class Sum(Layer):
    def __init__(self,axis=None):
        super(Sum,self).__init__("sum")
        self.axis=axis
    def forward(self,x):
        self.g=np.ones_like(x)
        if self.axis:
            m=1
            for i in x.shape[1:]:
                m*=i
        else:
            m=x.shape[self.axis]
        self.g=self.g/m
        xx=np.sum(x,axis=self.axis,keepdims=True)
        return xx
    def back(self,next_g):
        return self.g*next_g

#相乘
class Mul(Layer):
    def __init__(self):
        super(Mul,self).__init__("mul")
    def forward(self,x,y):
        self.x=x
        self.y=y
        #后面梯度要用到
        return x*y
    def back(self,next_g):
        return self.y*next_g,self.x*next_g

#矩阵点积
class Dot(Layer):
    def __init__(self):
        super(Dot,self).__init__("dot")
    def forward(self,x,y):
        self.x,self.y=x,y
        return np.dot(x,y)
    def back(self,next_g):
        gx=np.dot(next_g,self.y.T)
        gy=np.dot(self.x.T,next_g)
        return gx,gy

#倒数
class Recip(Layer):
    def __init__(self):
        super(Recip,self).__init__("recip")
    def forward(self,x):
        self.x=x
        self.g=1/x
        return self.g
    def back(self,next_g):
        self.g=-self.g/self.x
        return self.g*next_g

#最大值
class Max(Layer):
    def __init__(self,axis=None):
        super(Max,self).__init__("max")
        self.axis=axis
    def forward(self,x):
        self.x=x
        axis=self.axis
        self.xm=np.max(x,axis=axis,keepdims=True)
        return self.xm
    def back(self,next_g):
        g=np.zeros_like(self.x)
        g[:,:]=next_g
        k=(self.x==self.xm)
        g[k==False]=0.
        return g

#最小值
class Min(Layer):
    def __init__(self,axis=None):
        super(Min,self).__init__("min")
        self.max=Max(axis)
    def forward(self,x):
        self.x=x
        return -self.max.forward(-x)
    def back(self,next_g):
        return -self.max.gradient(-next_g)
#绝对值
class Abs(Layer):
    def __init__(self):
        super(Abs,self).__init__("abs")
    def forward(self,x):
        self.g=np.ones_like(x)
        self.g[np.where(x<0)]=-1.
        return self.g*x
    def back(self,next_g):
        return self.g*next_g

#求平均值
class Mean(Layer):
    def __init__(self,axis=None):
        super(Mean,self).__init__("mean")
        self.axis=axis
        self.x=np.zeros(1)
    def forward(self,x):
        if self.axis is None:
            self.axis=tuple([i for i in range(len(x.shape))])
        elif type(self.axis)==int:
            self.axis=(self.axis,)
        elif type(self.axis)==tuple:
            pass
        self.m=1
        for i in self.axis:
            self.m *= x.shape[i]
        self.x=x
        return np.mean(x,axis=self.axis,
                            keepdims=True)
    def back(self,next_g):
        self.g=np.ones_like(self.x)/self.m
        return self.g*next_g

#开方
class Sqrt(Layer):
    def __init__(self):
        super(Sqrt,self).__init__("sqrt")
    def forward(self,x):
        self.x=x
        self.sqrt=np.sqrt(x)
        return self.sqrt
    def back(self,next_g):
        self.g=-0.5/self.sqrt
        return self.g*next_g


class Dense(Layer):
    def __init__(self,in_dim,out_dim=1,
            lr=None,name=""):
        super(Dense,self).__init__("fc_"+name)
        self.W = np.random.randn(in_dim,out_dim)
        self.b = np.zeros((1,out_dim))
        if lr:
            self.lr=lr
    def forward(self,x):
        self.x=x
        self.z = np.dot(x,self.W) + self.b
        return self.z
    def back(self,next_g):
        x=self.x
        self.gw = np.dot(x.T,next_g)
        self.gb = np.mean(next_g,axis=0,
                            keepdims=True)
        out=np.dot(next_g,self.W.T)
        self.W = self.W-self.lr*self.gw
        self.b = self.b-self.gb*self.lr
        return out
    def save(self,path=""):
        np.save(path+self.name+"_w",self.W)
        np.save(path+self.name+"_b",self.b)
    def load(self,path=""):
        self.W=np.load(path+self.name+"_w.npy")
        self.b=np.load(path+self.name+"_b.npy")

class Conv2D(Layer):
    #已经与paddle精度对齐
    def __init__(self,in_ch,out_ch,ksize=3,
            padding=None,stride=2,name=""):
        super(Conv2D,self).__init__("conv_"+name)
        self.padding=padding
        if padding and padding!=0:
            self.pad=Pad(padding=padding)
        self.c,self.oc=in_ch,out_ch
        self.ksize=pram2pram(ksize)
        self.stride=pram2pram(stride)
        self.fin=in_ch*self.ksize[0]*self.ksize[1]
        self.fc=Dense(self.fin,out_ch)
    def forward(self,x):
        if self.padding and self.padding!=0:
            x=self.pad.forward(x)
        self.xsize=x.shape[2],x.shape[3]
        n,c,h,w=x.shape
        col=im2col(x,self.ksize,self.stride) #NHWC
        col=np.transpose(col,(0,3,1,2)) #NCHW
        col=col.reshape(col.shape[0],self.fin)
        ofc=self.fc.forward(col)
        oh=(h-self.ksize[0]+1)//self.stride[0]
        ow=(w-self.ksize[1]+1)//self.stride[1]
        ofc=ofc.reshape(n,oh,ow,ofc.shape[-1])
        return np.transpose(ofc,(0,3,1,2))
    def back(self,next_g):
        c=self.c
        n,oc,oh,ow=next_g.shape
        next_g=np.transpose(next_g,(0,2,3,1))
        next_g=next_g.reshape(n*oh*ow,oc)
        next_g=self.fc.back(next_g)
        kh,kw=self.ksize
        next_g=next_g.reshape(n*oh*ow,c,kh,kw)
        next_g=np.transpose(next_g,(0,2,3,1))
        xsize=self.xsize
        ksize=self.ksize
        stride=self.stride
        next_g=col2im(next_g,xsize,ksize,stride)
        if self.padding and self.padding!=0:
            next_g=self.pad.back(next_g)
        return next_g
    def save(self,path=""):
        np.save(path+self.name+"_w",self.fc.W)
        np.save(path+self.name+"_b",self.fc.b)
    def load(self,path=""):
        self.fc.W=np.load(path+self.name+"_w.npy")
        self.fc.b=np.load(path+self.name+"_b.npy")

def im2col(im,ksize,stride,mode="conv"):
    #为了将卷积窗口和池化窗口在图片上滑动的过程
    #转换成一次矩阵点乘运算，需要把图片按两种窗口
    #尺寸和步距裁剪出来，然后拉伸成一维，再把所
    #有这些一维向量拼接成二维向量后，就可以一次
    #矩阵点乘运算，就完成卷积中最花时间的运算过程
    n,c,h,w=im.shape
    sh,sw=stride
    kh,kw=ksize
    #卷积与池化运算的输出尺寸，需要不同的计算公式
    if mode=="conv":
        oh,ow=(h-kh+1)//sh,(w-kw+1)//sw
    elif mode=="pool":
        oh,ow=h//sh,w//sw
    #每张图展开后的形状是(oh*ow,c*kh*kw)
    col_n=n*oh*ow
    slices=[]
    #使用numpy中的步距,对窗口两个方向循环
    for i in range(kh):
        wslice=[]
        ii=h-kh+1+i #切片时下方减去卷积核尺寸
        for j in range(kw):
            jj=w-kw+1+j #切片时右侧减去卷积核尺寸
            ohw=im[:,:,i:ii:sh,j:jj:sw]#NCHW
            out=np.transpose(ohw,(0,2,3,1))#NHWC
            out=out.reshape(col_n,1,1,c)#123维展开
            wslice.append(out) #滑窗w方向合并
        wslices=np.concatenate(wslice,axis=2)
        slices.append(wslices) #滑窗h方向合并
    col=np.concatenate(slices,axis=1)
    return col #NHWC

def col2im(col,imsize,ksize,stride,mode="conv"):
    #与im2col是逆过程,注意点是滑动窗口重叠的部分
    #是相加的,前向计算时，重叠部分多次计算,所以逆
    #向计算时,需要把相应的多次计算梯度值累加起来
    nhw,kh,kw,c=col.shape
    h,w=imsize
    sh,sw=stride
    #根据卷积与池化的不同,输出尺寸需要不同的公式
    if mode=="conv":
        oh=(h-ksize[0]+1)//stride[0]
        #计算前向输出尺寸
        ow=(w-ksize[1]+1)//stride[1]
    elif mode=="pool":
        oh,ow=h//stride[0],w//stride[1]
    n=nhw//oh//ow #计算batch_size
    im=np.zeros((n,h,w,c)) #NHWC
    col=col.reshape(n,oh,ow,kh,kw,c)
    for i in range(kh):
        ii=h-ksize[0]+1+i
        for j in range(kw):
            jj=w-ksize[1]+1+j
            kk=col[:,:,:,i,j,:] #NHWC
            im[:,i:ii:sh,j:jj:sw,:]+=kk #维度对齐
    return np.transpose(im,(0,3,1,2)) #NCHW

def pram2pram(pram):
    #常用于卷积核尺寸,池化核尺寸,卷积和池化时滑
    #动的步距等尺寸参数的转换
    if type(pram)==int:
        pram=(pram,pram)
    elif type(pram)==tuple and len(pram)==2:
        pass
    elif type(pram)==list and len(pram)==2:
        pram=tuple(pram)
    else:
        print("psize类型错误")
    return pram

class Pool2D(Layer):
    #已经与paddle精度对齐
    def __init__(self,psize=2,padding=None,
            stride=2,ptype="max",name=""):
        #psize:池化核尺寸,padding平面填充尺寸
        #stride:滑动步距
        name=ptype+"pool_"+name
        super(Pool2D,self).__init__(name)
        '''
        使用时注意到边对齐，需要填充时向右下方填
        填充尺寸为
        当psize > stride时
        padding=psize - size % stride
        当psize <=stride时
        向上取整(尺寸/步长)再乘以步长再减去原尺寸

        '''
        self.padding=padding
        if padding and padding!=0:
            self.pad=Pad(padding=padding)
        self.psize=pram2pram(psize)
        self.stride=pram2pram(stride)
        self.ptype=ptype
        if ptype=="max":
            self.pool=Max(axis=(2,3))
        elif ptype=="avg":
            self.pool=Mean(axis=(2,3))
        else:
            print("不支持的池化类型")

    def forward(self,x):
        if self.padding and self.padding!=0:
            x=self.pad.forward(x)
        self.x=x
        n,c,h,w=x.shape
        kh,kw=self.psize
        sh,sw=self.stride
        oh,ow=h//sh,w//sw
        col=im2col(x,(kh,kw),(sh,sw),"pool")
        col=np.transpose(col,(0,3,1,2)) #NCHW
        col=self.pool.forward(col)
        out=col.reshape(n,oh,ow,c)
        return np.transpose(out,(0,3,1,2))
    def back(self,next_g):
        n,c,h,w=self.x.shape
        kh,kw=self.psize
        sh,sw=self.stride
        oh,ow=h//sh,w//sw
        next_g=np.transpose(next_g,(0,2,3,1))#NHWC
        next_g=next_g.reshape(n*oh*ow,1,1,c)#NHWC
        next_g=np.transpose(next_g,(0,3,1,2))#NCHW
        next_g=self.pool.back(next_g)
        stride=self.stride
        next_g=np.transpose(next_g,(0,2,3,1))#NCHW
        next_g=col2im(next_g,(h,w),(kh,kw),
                stride,mode="pool")
        if self.padding and self.padding!=0:
            next_g=self.pad.back(next_g)
        return next_g

#填充,NCHW
class Pad(Layer):
    def __init__(self,padding=0,name=""):
        #参数padding=1or[1,1]or[1,2,1,2]上下左右
        #仅支持在平面两个方向填充
        super(Pad,self).__init__("pad_"+name)
        ptype=type(padding)
        if ptype==int:
            self.shape=((padding,padding),
                    (padding,padding))
        elif ptype==list or ptype==tuple:
            print(ptype)
            if len(padding)==4:
                print(len(padding))
                self.shape=((padding[0],padding[1]),
                    (padding[2],padding[3]))
            elif len(padding)==2 and type(padding[0])==int:
                self.shape=((padding[0],padding[0]),
                    (padding[1],padding[1]))
            elif len(padding)==2:
                self.shape=((padding[0][0],padding[0][1]),
                    (padding[1][0],padding[1][1]))

        else:
            print("填充格式错误")
    def forward(self,x):
        print(self.shape[0])
        #获取NCHW数据的高度和宽度
        self.h,self.w=x.shape[2],x.shape[3]
        t,b=self.shape[0]
        r,l=self.shape[1]
        return np.pad(x,((0,0),(0,0),
                    (t,b),(r,l)),'constant')
    def back(self,next_g):
        h0=self.shape[0][0]
        h1=self.shape[0][0]+self.h
        w0=self.shape[1][0]
        w1=self.shape[1][0]+self.w
        return next_g[:,:,h0:h1,w0:w1]
#批规范化
class BatchNorm(Layer):
    #已经与paddle精度对齐
    def __init__(self,ch,lr=None,axis=(0,2,3),
            mode="train",name=""):
        #参数ch为通道数,数据形状为NCHW
        super(BatchNorm,self).__init__("bn_"+name)
        if lr:
            self.lr=lr
        self.mode=mode
        self.eps=0.00001
        self.axis=axis
        shape=np.ones(max(max(axis)+1,2))
        shape=shape.astype("int64")
        shape=shape.tolist()
        for i in axis:
            shape[i]=1
        shape[1]=ch
        shape=tuple(shape) #转化成元组
        self.gama=np.ones((1,ch,1,1))
        self.bate=np.zeros((1,ch,1,1))
        self.run_uB=np.zeros((1,ch,1,1))
        self.run_bB2=np.ones((1,ch,1,1))
        self.monentum=0.9
        
    def forward(self,x):
        self.x=x
        a = self.monentum
        self.m=x.shape[0]*x.shape[2]*x.shape[3]
        if self.mode == 'train':
            self.uB=np.mean(x,axis=self.axis,keepdims=True)
            self.bB2=np.var(x,axis=self.axis,keepdims=True)
            self.y=(x-self.uB)/np.sqrt(self.bB2+self.eps)
            if self.run_uB is None:
                self.run_uB = self.uB
                self.run_bB2 = self.bB2
            self.run_uB=a*self.run_uB+(1-a)*self.uB
            self.run_bB2=a*self.run_bB2+(1-a)*self.bB2
        elif self.mode=="test":
            self.y=(x-self.run_uB)/np.sqrt(self.run_bB2+self.eps)
        else:
            raise ValueError('无效"%s"'%self.mode)
        self.out=self.y*self.gama + self.bate
        return self.out
    def back(self,next_g):
        dgama = np.sum(self.out*next_g,axis=self.axis,keepdims=True)
        dbate = np.sum(next_g,axis=self.axis,keepdims=True)
        dy =next_g*self.gama
        dsigma1=-0.5*np.sum(dy*(self.x-self.uB),axis=self.axis,keepdims=True)
        dsigma2 = np.power(self.bB2+self.eps,-1.5)
        dsigma=dsigma1*dsigma2
        dmu1=-np.sum(dy/np.sqrt(self.bB2+self.eps),axis=self.axis,keepdims=True)
        dmu2=-2*dsigma*np.sum(self.x-self.uB,axis=self.axis,keepdims=True)/self.m
        dmu=dmu1+dmu2
        #dmu = -np.sum(dx_hat / np.sqrt(sample_var + eps), axis=0) - 2 * dsigma*np.sum(x-sample_mean, axis=0)/ N
        dx1=dy/np.sqrt(self.bB2 + self.eps)
        dx2=2.0*dsigma*(self.x-self.uB)/self.m
        dx3=dmu/self.m
        dx=dx1+dx2+dx3
        self.gama -=dgama*self.lr
        self.bate -=dbate*self.lr
        return dx
    def save(self,path=""):
        arr=np.array([self.gama,self.bate,
                    self.run_uB,self.run_bB2])
        np.save(path+self.name,arr)
    def load(self,path=""):
        arr=np.load(path+self.name+".npy")
        self.gama=arr[0]
        self.bate=arr[1]
        self.run_uB=arr[2]
        self.run_bB2=arr[3]

class Relu(Layer):
    def __init__(self,name=""):
        super(Relu,self).__init__("relu_"+name)
    def forward(self,x):
        self.x=x
        x[x<0]=0.
        return x
    def back(self,next_g):
        g=np.zeros_like(self.x)
        g[self.x>0]=1.
        return g*next_g

class Dropout(Layer):
    def __init__(self,r=0.2,name=""):
        super(Dropout,self).__init__("drop_"+name)
        self.r=r
    def forward(self,x):
        if self.mode=="test":
            return x
        a=[0,1]
        p=[self.r,1.-self.r]
        self.g=np.random.choice(a,x.shape,p=p)
        return self.g*x
    def back(self,next_g):
        return self.g*next_g

class Tanh(Layer):
    def __init__(self,name=""):
        super(Tanh,self).__init__("tanh_"+name)
    def forward(self,x):
        self.y=np.nan_to_num(np.tanh(x))
        return self.y
    def back(self,next_g):
        return (1-self.y**2)*next_g

class Sigmoid(Layer):
    def __init__(self,name=""):
        super(Sigmoid,self).__init__("sigmo"+name)
    def s(self,x):
        exps=np.nan_to_num(np.exp(-x))
        return 1/(1+exps)
    def forward(self,x):
        xmax=np.max(x)
        xx=x-xmax
        self.y=self.s(xx)
        return self.y
    def back(self,next_g):
        next_g=self.y*(1-self.y)*next_g
        return next_g

class Softmax(Layer):
    def __init__(self,axis=1,name=""):
        super(Softmax,self).__init__("sm_"+name)
        self.axis=axis
    def forward(self,x):
        xmax=np.max(x,axis=self.axis,keepdims=True)
        self.xx=x-xmax
        exps = np.nan_to_num(np.exp(self.xx))
        exps_sum=np.sum(exps,axis=self.axis,keepdims=True)
        self.y=exps/exps_sum
        return self.y
    def back(self,next_g):
        n=self.xx.shape[self.axis]
        self.a=np.eye(n)
        #i==j
        g1=self.y*np.dot((1-self.y)*next_g,self.a)
        #i!=j
        g2=-self.y*np.dot(self.y*next_g,1-self.a)
        g=g1+g2
        return g


class CrossEntropy(Layer):
    #多分类交叉熵,采用ont_hot形式
    #已经与paddle精度对齐
    def __init__(self,axis=1,name=""):
        super(CrossEntropy,self).__init__("ce_"+
                name)
        self.axis=axis
    def forward(self,x,y):
        self.x,self.y=x,y
        a = y * np.log(x)
        out = -np.nan_to_num(a)
        return np.sum(out,axis=1,keepdims=True)
    def back(self,next_g):
        self.g = -self.y/self.x
        return self.g * next_g

class Softmax_CrossEntropy(Layer):
    #多分类交叉熵,采用ont_hot形式
    #已经与paddle精度对齐
    def __init__(self,axis=(1,),name=""):
        super(Softmax_CrossEntropy,self).__init__("sce_"+name)
        self.axis=axis
        self.sm=Softmax()
    def forward(self,x,y):
        self.x=self.sm.forward(x)
        self.y=y
        out=y*np.log(self.x)
        out=np.nan_to_num(out)
        out=-np.sum(out,axis=self.axis,keepdims=True)
        return out,self.x
    def back(self,next_g):
        g = self.x-self.y
        return g*next_g

#改变形状
class Reshape(Layer):
    #已经与paddle精度对齐
    def __init__(self,shape,m='CHW',name=""):
        super(Reshape,self).__init__("resha"+name)
        self.shape=shape
        self.m=m
    def forward(self,x):
        #NCHW格式
        self.xShape=x.shape#记录x的形状
        #shape=[x.shape[0]].extend(self.reshape)
        if self.m=='CHW':
            shape=tuple([x.shape[0]]+list(self.shape))
        elif self.m=='NCHW':
            shape=self.shape
        #拼接成需要的形状
        return x.reshape(shape)
    def back(self,next_g):
        return next_g.reshape(self.xShape)

#顺序模式
class Sequence(Layer):
    def __init__(self,name="",mode="train"):
        super(Sequence,self).__init__("seq_"+name)
        self.seq=[]
        self.mode=mode
    def add(self,layer):
        self.seq.append(layer)
        self.seq[-1].name+=str(len(self.seq))
        self.seq[-1].mode=self.mode
    def forward(self,x):
        out=x
        for layer in self.seq:
            out=layer.forward(out)
        return out
    def back(self,next_g):
        for layer in reversed(self.seq):
            next_g=layer.back(next_g)
        return next_g
    def save(self,path=""):
        print("保存模型。。。")
        for layer in self.seq:
            layer.save(path)
    def load(self,path=""):
        for layer in self.seq:
            layer.load(path)
    def train(self):
        for layer in self.seq:
            layer.mode="train"
    def infer(self):
        for layer in self.seq:
            layer.mode="test"
    def set_lr(self,lr):
        for layer in self.seq:
            layer.lr=lr
#计算正确率
def acc(pred,y):
    p=np.argmax(pred,axis=1)
    l=np.argmax(y,axis=1)
    zeros=np.zeros(l.shape)
    zeros[p==l]=1.
    return np.mean(zeros)

if __name__ == '__main__':
    #print("Dense,Conv2D...")
    a=np.random.randn(3,2,4,4)
    print("a",a.shape)
    pool=Pool(2,2,"max")
    b=pool.forward(a)
    print("b",b.shape)
    c=pool.back(b)
    print("c",c.shape)

