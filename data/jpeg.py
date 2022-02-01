from turbojpeg import TurboJPEG
jpeg=TurboJPEG()
def imread(fname):
    with open(fname,'rb') as f:
        bgr=jpeg.decode(f.read())
    return bgr

def imwrite(fname,bgr):
    with open(fname,'wb') as f:
        f.write(jpeg.encode(bgr))

