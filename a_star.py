import numpy as np
from turbojpeg import TurboJPEG
jpeg=TurboJPEG()
def imread(fname):
    with open(fname,'rb') as f:
        bgr=jpeg.decode(f.read())
    return bgr

def imwrite(fname,bgr):
    with open(fname,'wb') as f:
        f.write(jpeg.encode(bgr))
width,height=32,64
map_=np.zeros((width,height),dtype="uint8")
start,end=(5,5),(30,62)
barrier =[[2,7,30,7],[0,50,31,55]]
print(start)
map_[start]=160
map_[end]=80
for box in barrier:
    map_[box[0]:box[2],box[1]:box[3]]=255
img = np.zeros((height*8,width*8,3),dtype="uint8")
img[:,::8],img[::8]=127,127

img2 = np.copy(img)
map_[:]=255
img2[:width,:height,0] = np.copy(map_)

img2[:width,:height,1] = np.copy(map_)

img2[:width,:height,2] = np.copy(map_)
i=15
fname="../storage/shared/1/data/%07d.jpg"%i
imwrite(fname,img2)
print(6666)
