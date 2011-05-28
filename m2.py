#!/usr/bin/python
import sys
import urllib2
import cv
from PIL import Image
from math import atan,pi as PI,radians as rad,ceil, sin, cos
import time
import os

src = 0
image = 0
dest = 0
element_shape = cv.CV_SHAPE_RECT

def opening(pos,src):
    element = cv.CreateStructuringElementEx(pos*2+1, pos*2+1, pos, pos, element_shape)
    cv.Erode(src, src, element, 1)
    cv.Dilate(src, src, element, 1)
    return src

def get_angel(src):
    pos = 6
    element = cv.CreateStructuringElementEx(pos*2+1, pos*2+1, pos, pos, element_shape)
    ni = cv.CreateImage(cv.GetSize(src),src.depth,src.nChannels) 
    cv.Erode(src, ni, element, 1)
    
    image_gray = cv.CreateImage(cv.GetSize(ni),8,1)  
    cv.CvtColor(ni,image_gray,cv.CV_RGB2GRAY)
    pi = Image.fromstring("L", cv.GetSize(ni), image_gray.tostring())
    first = 0 if pi.getpixel((0,0))<240 else 255
    xstart = xend = ystart = yend = 0
    for x in xrange(1,pi.size[0]):
        v = 0 if pi.getpixel((x,0))<240 else 255
        if first == 0 and v != 0:
            xstart = x
            xend = pi.size[0]
            break
        if first == 255 and v != 255:
            xstart = 0
            xend = x
            break
    
    if first == 255:
        for y in xrange(pi.size[1]):
            v = 0 if pi.getpixel((0,y))<240 else 255
            if v != 255:
                yend = y
                break
    else:
        for y in xrange(pi.size[1]):
            v = 0 if pi.getpixel((pi.size[0]-1,y))<240 else 255
            if v != 255:
                yend = y
                break
                
    a = yend - ystart
    b = xend - xstart or 1
    alpha = atan(a*1.0/b)/(PI*1.0/180)
    if first == 255:
        alpha = -alpha
    
    return (alpha,pi.size[0]*1.0/2,pi.size[1]*1.0/2)

def doRotate(image, alpha, fillval=0, resize=True, interpolation=cv.CV_INTER_CUBIC):
    matrix = cv.CreateMat(2, 3, cv.CV_32FC1)
    w, h = cv.GetSize(image)
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    cv.GetRotationMatrix2D(center, alpha, 1.0, matrix)
    if resize:
        angle = rad(abs(alpha))
        nw = w * cos(angle) + h * sin(angle)
        nh = w * sin(angle) + h * cos(angle)
        ncenter = (nw / 2.0, nh / 2.0)
        matrix[0, 2] += ncenter[0] - center[0]
        matrix[1, 2] += ncenter[1] - center[1]
        size = (int(ceil(nw)), int(ceil(nh)))
    else:
        size = cv.GetSize(image)
    result = cv.CreateImage(size, image.depth, image.nChannels)
    cv.WarpAffine(image, result, matrix, interpolation + cv.CV_WARP_FILL_OUTLIERS, fillval)
    return result
            
def crop(img,x,y,w,h):
    cv.SetImageROI(img, (int(x),int(y),int(w),int(h)))
 
    res = cv.CreateImage(cv.GetSize(img),img.depth,img.nChannels)
 
    cv.Copy(img, res);
 
    cv.ResetImageROI(img);
    
    return res
            
def skew(img,x,y):
    return img.transform((img.size[0],img.size[1]),Image.AFFINE, (1,x,-130*x,y,1,-130*y))
    
def intel_crop(img):
    y1 = x1 = x2 = y2 = 0
    ystop = 0.06
    for y in xrange(img.size[1]):
        count = 0
        for x in xrange(img.size[0]):
            if img.getpixel((x,y)) > 180:
                count += 1
        if count*1.0/img.size[0] >= ystop:
            y1 = y
            break
    
    xstop = 0.05
    for x in xrange(img.size[0]):
        count = 0
        for y in xrange(img.size[1]):
            if img.getpixel((x,y)) > 180:
                count += 1
        if count*1.0/img.size[1] >= xstop:
            x1 = x
            break    
       

    return img.transform((img.size[0]-x1,img.size[1]-y1),Image.EXTENT, (x1,y1,img.size[0],img.size[1]))

def get_only_white(img):
    data = list(img.getdata())
    ld = len(data)
    for i in xrange(ld):
        if data[i] < 245:
            data[i] = 0
            
    img.putdata(data)
    return img


def FindDividingCols(img):
    lst = []
    for i in range(img.size[0]):
        lst.append(sum(1 for j in range(img.size[1]) if img.getpixel((i, j)) != 0))

    centers = [img.size[0]/4,2*(img.size[0]/4),3*(img.size[0]/4)]
    delta = 20
    cols = []
    for c in centers:
        m = 999999
        ind = 0
        for i in xrange(c-delta,c + delta):
            if lst[i] < m:
                m = lst[i] 
                ind = i             
        cols.append(ind)
    """
    ax1 = pylab.subplot(212)
    pylab.imshow(img.transpose(Image.FLIP_TOP_BOTTOM), cmap=pylab.cm.gray, shape=(img.size[0], img.size[1]), interpolation='bilinear')
    pylab.subplot(211, sharex=ax1)
    pylab.plot(lst)
    pylab.show()
    """

    return cols

def DivideDigits(img):
    col = FindDividingCols(img)
    imgs = []
    imgs.append(img.crop((0, 0, col[0], img.size[1])))
    imgs.append(img.crop((col[0], 0, col[1], img.size[1])))
    imgs.append(img.crop((col[1], 0, col[2], img.size[1])))
    imgs.append(img.crop((col[2], 0, img.size[0], img.size[1])))

    return imgs

def CropImages(imgs):
    percent = 0.05
    res = []
    for img in imgs:
        lst = []
        for i in range(img.size[0]):
            lst.append(sum(1 for j in range(img.size[1]) if img.getpixel((i, j)) > 0))
        
        lsth = []
        for i in range(img.size[1]):
            lsth.append(sum(1 for j in range(img.size[0]) if img.getpixel((j, i)) > 0))
        
        bottom = 0
        for i in xrange(len(lsth)):
            if lsth[i]*1.0/img.size[0] >= percent:
                bottom = i
                
        top = 0
        for i in xrange(len(lsth)-1,-1,-1):
            if lsth[i]*1.0/img.size[0] >= percent:
                top = i
        
        left = 0
        for i in xrange(len(lst)):
            if lst[i]*1.0/img.size[1] >= percent:
                left = i
                break  
    
        right = 0
        for i in xrange(len(lst)-1,-1,-1):
            if lst[i]*1.0/img.size[1] >= percent:
                right = i
                break
                
        res.append(img.crop((left, top, right, bottom)))

    return res



if __name__ == "__main__":

    path = sys.argv[1].rstrip('/')
    spath = sys.argv[2].rstrip('/')
    files = os.listdir(path)
    for f in files:
        
        if len(sys.argv) > 1:
            src = cv.LoadImage(path+'/'+f)

        new = opening(1,src)
        angel,cx,cy  = get_angel(new)
        new = doRotate(src,angel)

        x = 0
        y = h = cv.GetSize(new)[1]*1.0/3
        w = cv.GetSize(new)[0]
        
        image_gray = cv.CreateImage(cv.GetSize(new),8,1)  
        cv.CvtColor(new,image_gray,cv.CV_RGB2GRAY)
        img = Image.fromstring("L", cv.GetSize(image_gray), image_gray.tostring())
        sk = 0.7
        if angel >= 0:
            sk = -sk
        img = skew(img,sk,0)
        cv_im = cv.CreateImageHeader(img.size, 8, 1)
        cv.SetData(cv_im, img.tostring())
        cv_im = crop(cv_im,x+30,y,w-50,h)
        
        img = Image.fromstring("L", cv.GetSize(cv_im), cv_im .tostring())
        img = get_only_white(img)
        img = intel_crop(img)
        imgs = DivideDigits(img)
        imgs = CropImages(imgs)
        for i,img in enumerate(imgs):
            img.resize((52,73),Image.BICUBIC).save(spath+'/%s_%s.png' % (f,i),'PNG')
    
    
    
    
