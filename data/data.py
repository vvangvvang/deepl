#数据准备
import numpy as np
def load():
    with open("train-labels-idx1-ubyte",'rb') as f:
        labels = []
        data = f.read()
        print(len(data))
        for i in range(len(data)):
            if i > 7:
                labels.append(data[i])
    print(len(labels))
    
    
    with open("train-images-idx3-ubyte",'rb') as f:
        train = []
        data = f.read()
        print(len(data))
        print(60000*28*28)
        line = []
        for i in range(len(data)):
            if i > 15:
                line.append(data[i])
                if (i - 15) % 784 == 0 and len(line) == 784:
                    train.append(line)
                    line= []
            
        pass

    return np.array(train),np.array(labels)
