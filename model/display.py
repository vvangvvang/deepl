import numpy as np

def disp(x,char=" 123456789"):
    print(x.shape)
    xmax=np.max(x)
    xmin=np.min(x)
    x=x-xmin
    div=len(char)-1
    x=np.ceil(x/(xmax-xmin)*div)
    x=x.astype("int")
    h,w=x.shape[0],x.shape[1]
    string=""
    for i in range(h):
        string+="\n"
        for j in range(min(w,50)):
            string+=char[x[i,j]]
    print(string)
