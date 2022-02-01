import numpy as np
import lib.base as nn
class RNN(nn.Layer):
    def __init__(self,maxlen,xdim,hdim,
            ydim,name=""):
        super(RNN,self).__init__("rnn_"+name)
        #数据格式:批数量N,句长L,词向量宽度W
        self.xdim=xdim
        self.hdim=hdim
        self.ydim=ydim
        self.maxlen=maxlen
        self.wx=np.random.randn(xdim,hdim)
        self.wa=np.random.randn(hdim,hdim)
        self.bxa=np.zeros((1,hdim))
        self.wh=np.random.randn(hdim,ydim)
        self.by=np.zeros((1,ydim))
    def forward(self,x,a0=None):#x.shape(N,dim)
        self.x=x
        n,l,w=x.shape #宽度w=词向量维度xdim
        if a0 is None:
            a0 = np.zeros((n,self.hdim))
        l=min(l,self.maxlen)
        self.l=l
        self.h=np.zeros((n,l,self.hdim))
        self.a=np.zeros((n,l,self.hdim))
        self.y=np.zeros((n,l,self.ydim))
        self.yy=np.zeros((n,l,self.ydim))
        self.y1=np.zeros((n,l,self.ydim))
        for i in range(l):
            #print("x[:,i,:]",x[:,i,:].shape)
            h1=np.dot(x[:,i,:],self.wx)
            h2=np.dot(a0,self.wa)
            #print("h2",h2.shape)
            self.h[:,i,:]=h1+h2+self.bxa
            self.a[:,i,:]=np.tanh(self.h[:,i,:])
            #print("wh",self.wh.shape)
            self.y[:,i,:]=np.dot(self.a[:,i,:],
                                self.wh)
            #print("y",self.y.shape)
            self.y1[:,i,:]=self.y[:,i,:]+self.by
            #softmax
            exps=np.exp(self.y1[:,i,:])
            esum=np.sum(exps,axis=1,keepdims=True)
            self.yy[:,i,:]=exps/esum
        return self.yy,self.a[:,l-1,:]
    def back(self,gy,ga):
        x=self.x
        n,_,w=x.shape
        gx=np.zeros((n,self.l,self.xdim))
        gh=np.zeros((n,self.l,self.hdim))
        a=self.a
        #对角矩阵，用来分配对齐与不对齐的不同导数
        e=np.eye(self.ydim)
        #gy,形状与y一样
        for i in range(self.l):
            j=self.l-i-1
            #print(ga.shape)
            #print("开始softmax梯度反传")
            #print("gy[:,j,:]",gy[:,j,:].shape)
            yy=self.yy[:,j,:]
            #print("out_dim",self.out_dim)
            #print('yy',yy.shape)
            #i==j
            g1=yy*np.dot((1-yy)*gy[:,j,:],e)
            #print('g1',g1.shape)
            #i!=j
            g2=-yy*np.dot(yy*gy[:,j,:],1-e)
            gy[:,j,:]=g1+g2
            #print("gy[:,j,:]",gy[:,j,:].shape)
            #print('更新self.by')
            d=np.sum(gy[:,j,:],axis=0)
            self.by -=self.lr*d
            #print('a与Wh点积反传')
            #print('wh',self.wh.T.shape)
            gh[:,j,:]=np.dot(gy[:,j,:],self.wh.T)
            #print('更新Wh')
            d=np.dot(a[:,j,:].T,gy[:,j,:])
            self.wh -=self.lr*d
            #print('合并梯度')
            #更新self.wy
            #print('ga',ga.shape)
            gh[:,j,:]=gh[:,j,:]+ga
            #print('tanh反传梯度')
            #print(self.a[:,j,:].shape)
            ghh=(1-self.a[:,j,:]**2)*gh[:,j,:]
            #print('更新self.bxa')
            d=np.sum(ghh,axis=0)
            self.bxa-=self.lr*d
            #print('x与Wx点积反传')
            gx[:,j,:]=np.dot(ghh,self.wx.T)
            #print('更新self.wx')
            d=np.dot(x[:,j,:].T,ghh)
            self.wx-=self.lr*d
            #print('a与Wa点积反传')
            ga=np.dot(ghh,self.wa.T)
            #print('更新self.wa')
            d=np.dot(self.a[:,j,:].T,ghh)
            self.wa-=self.lr*d
        return gx,ga
    def __call__(self,x,a0=None):
        return self.forward(x,a0)
    def save(self,path=""):
        np.save(path+self.name+"_wx",self.wx)
        np.save(path+self.name+"_wa",self.wa)
        np.save(path+self.name+"_bxa",self.bxa)
        np.save(path+self.name+"_wh",self.wh)
        np.save(path+self.name+"_by",self.by)
    def load(self,path=""):
        self.wx=np.load(path+self.name+"_wx.npy")
        self.wa=np.load(path+self.name+"_wa.npy")
        self.bxa=np.load(path+self.name+"_bxa.npy")
        self.wh=np.load(path+self.name+"_wh.npy")
        self.by=np.load(path+self.name+"_by.npy")

class Embedding(nn.Layer):
    #初始化时，需要词包长度，词向量维度两个参数
    def __init__(self,cab_len,dim,name=""):
        super(Embedding,self).__init__(
                                "embed"+name)
        self.dim=dim
        self.embed=np.random.randn(cab_len,dim)
    def forward(self,x):
        self.x=x
        self.y=self.embed[x]
        return self.y
    def back(self,next_g):
        d=self.lr*next_g
        self.embed[self.x]-=d
        return next_g
    def save(self,path=""):
        np.save(path+self.name+"_embed",self.embed)
    def load(self,path=""):
        self.embed=np.load(path+self.name+"_embed.npy")

class Attention(nn.Layer):
    def __init__(self,seq_len,xdim,sdim,hdim,name=""):
        super(Attention,self).__init__("atten_"+name)
        self.seq_len=seq_len
        self.w=np.random.randn(xdim+sdim,hdim)
        self.vT=np.random.randn(hdim,1)
    def forward(self,x,s0):
        n,l,w=x.shape
        self.x,self.s0=x,s0
        l=min(l,self.seq_len)
        self.cache=[]
        self.a=np.zeros((n,l,1))
        for i in range(l):
            xi = x[:,i]
            xs=np.concatenate([xi,s0],axis=1)
            h=np.dot(xs,self.w) #(n,xdim+sdim)点乘(xdim+sdim,hdim)
            t=np.tanh(h) #(n,hdim)
            self.a[:,i]=np.dot(t,self.vT) #(n,hdim)点乘(hdim,1)
            self.cache.append((xs,t))
        self.a=np.squeeze(self.a,axis=2)
        exps=np.exp(self.a)
        esum=np.sum(exps,axis=1,keepdims=True)
        self.aa=exps/esum
        return self.aa  #(n,l)
    def __call__(self,x,s0):
        return self.forward(x,s0)
    def back(self,gaa): #(n,l,1)
        n,l,w=self.x.shape
        l=min(l,self.seq_len)
        e=np.eye(l) #序列长度
        g1 = self.aa*np.dot((1-self.aa)*gaa,e)
        g2 = -self.aa*np.dot(self.aa*gaa,1-e)
        ga =g1+g2
        ga = ga.reshape(n,l,1)
        dvT,dw=0,0
        gx=np.zeros_like(self.x)
        gs0=np.zeros_like(self.s0)
        for i in range(l):
            xs,t =self.cache[i]
            gt=np.dot(ga[:,i],self.vT.T)
            dvT+=np.dot(t.T,ga[:,i])
            gh=(1-t**2)*gt
            gxs=np.dot(gh,self.w.T)
            dw+=np.dot(xs.T,gh)
            gx[:,i] = gxs[:,:w]
            gs0 += gxs[:,w:]
        return gx,gs0,dw,dvT
    def update(self,dvT,dw):
        self.vT -=self.lr*dvT
        self.w -=self.lr*dw

class Decode(nn.Layer):
    def __init__(self,seq_len,xdim,sdim,adim,ydim,name=""):
        super(Decode,self).__init__("decode_"+name)
        self.xdim,self.sdim=xdim,sdim
        self.adim,self.ydim=adim,ydim
        self.seq_len=seq_len
        self.att=Attention(seq_len=256,xdim=xdim,sdim=xdim,hdim=128)
        self.w=np.random.randn(xdim*2+adim,adim)
        self.b=np.zeros((1,adim))
        self.wy=np.random.rand(adim,ydim)
        self.by=np.zeros((1,ydim))
    def __call__(self,x,a0):
        return self.forward(x,a0)
    def forward(self,x,a0):
        n,l,w=x.shape
        self.x=x
        x0=x[:,-1]
        self.seq=[]
        self.cache=[]
        self.yy=[]
        for i in range(self.seq_len):
            self.seq.append(self.att)
            alphe=self.seq[-1](x,x0)
            xalphe=np.einsum("nlw,nl->nlw",x,alphe)
            c0=np.sum(xalphe,axis=1)
            xsa=np.concatenate([x0,c0,a0],axis=1)
            h=np.dot(xsa,self.w)
            h=h+self.b
            a0=np.tanh(h)
            y=np.dot(a0,self.wy)
            yb=y+self.by
            exps=np.exp(yb)
            esum=np.sum(exps,axis=1,keepdims=True)
            yy=exps/esum
            self.cache.append((xsa,a0))
            self.yy.append(yy)
            x0=yy
        self.yy=np.array(self.yy)
        return np.transpose(self.yy,(1,0,2)),a0
    def back(self,gyy,ga0):
        n,l,w=self.x.shape
        e=np.eye(w)
        gyy=np.transpose(gyy,(1,0,2))
        dw,db,dwy,dby=0,0,0,0
        dattw,dattvT=0,0
        gx=0
        gx1=0
        for i in range(self.seq_len):
            j = self.seq_len-i-1
            gy0=gyy[j]+gx1
            yy=self.yy[j]
            g1=yy*np.dot((1-yy)*gy0,e)
            g2=-yy*np.dot(yy*gy0,1-e)
            gy=g1+g2
            xsa,a0=self.cache[j]
            dby+=np.sum(gy,axis=0)  #对by的梯度进行累加
            ga0+=np.dot(gy,self.wy.T)
            gwy=np.dot(a0.T,gy)
            dwy+=gwy  #对wy的梯度进行累加
            gh=(1-a0**2)*ga0
            db+=np.sum(gh,axis=0) #对b的梯度进行累加
            gxsa=np.dot(gh,self.w.T)
            gw=np.dot(xsa.T,gh)
            dw+=gw  #对w的梯度进行累加
            gx0=gxsa[:,:self.xdim]
            gc0=gxsa[:,self.xdim:2*self.xdim]
            ga0=gxsa[:,2*self.xdim:]
            galphe=np.einsum("nlw,nw->nlw",self.x,gc0)
            #galphe=self.x*gc0
            galphe=np.sum(galphe,axis=2)
            gx_,gx0_,dattw_,dattvT_=self.seq[j].back(galphe)
            gx+=gx_   #每一次注意力反传时，都对x有影响
            dattw+=dattw_
            dattvT+=dattvT_
            gx1=gx0_
        gx[:,-1]+=gx0_
        self.w-=self.lr*dw
        self.b-=self.lr*db
        self.wy-=self.lr*dwy
        self.by-=self.lr*dby
        self.att.vT-=self.att.lr*dattvT
        self.att.w-=self.att.lr*dattw
        return gx,ga0
    def save(self,path=""):
        np.save(path+self.name+"_w",self.w)
        np.save(path+self.name+"_b",self.b)
        np.save(path+self.name+"_wy",self.wy)
        np.save(path+self.name+"_by",self.by)
        np.save(path+self.name+"_att.vT",self.att.vT)
        np.save(path+self.name+"_att.w",self.att.w)
    def load(self,path=""):
        self.w=np.load(path+self.name+"_w.npy")
        self.b=np.load(path+self.name+"_b.npy")
        self.wy=np.load(path+self.name+"_wy.npy")
        self.by=np.load(path+self.name+"_by.npy")
        self.att.vT=np.load(path+self.name+"_att.vT.npy")
        self.att.w=np.load(path+self.name+"_att.w.npy")

if __name__=='__main__':
    x=np.random.randint(17000, size=(128,200))
    #词典包长度，词向量维度
    embed=Embedding(17000,128)
    #句子长度,词向量维度,隐藏层维度，输出维度，
    rnn=RNN(256,128,200,128)
    decode=Decode(seq_len=11,xdim=128,sdim=128,adim=200,ydim=128)
    b=embed(x)
    print(b.shape)
    c,a=rnn(b)
    print(c.shape)
    y,a0=decode(x=c,a0=a)
    print(y.shape,a0.shape)
    gx,ga0 = decode.back(y,a0)
    b,ga=rnn.back(gx,ga0)
    c=embed.back(b)
    embed.save("model/")
    embed.load("model/")
    rnn.save("model/")
    rnn.load("model/")
    decode.save("model/")
    decode.load("model/")
