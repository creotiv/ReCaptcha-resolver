import cv
import sys
from PIL import Image, ImageDraw
from math import ceil,pi,cos

def pil2cv(img,t='L'):
    if t == 'L':
        res = cv.CreateImageHeader(img.size, 8, 1)
    else:
        res = cv.CreateImageHeader(img.size, 32, 3)
    cv.SetData(res, img.tostring())
    return res

def cv2pil(img,t='L'):

    img = Image.fromstring(t, cv.GetSize(img), img.tostring())

    return img

def togrey(img):
    image_gray = cv.CreateImage(cv.GetSize(img),8,1)        
    cv.CvtColor(img,image_gray,cv.CV_RGB2GRAY)
    
    return image_gray
    
def torgb(img):
    image = cv.CreateImage(cv.GetSize(img),32,3)        
    cv.CvtColor(img,image,cv.CV_GRAY2RGB)
    
    return image
   
def getCenterLine(img,step=10,block=8):    
    """
        Draw and return center line of text.
    
        Input image must be previously cropped and words must have one baseline
        
    """
    draw = ImageDraw.Draw(img)
    
    MEDIAN_STEP = 5
    
    def lf(num):
        return num if num >= 0 else 0
     
    def rf(num):
        return num if num < img.size[0] else img.size[0]-1 
     
    # Getting top and bottom contour pixels     
    top = []
    bottom = [] 
    for x in xrange(img.size[0]):
        end = start = None
        for y in xrange(img.size[1]):
            p = img.getpixel((x,y))
            p = (p[0]+p[1]+p[2])/3
            if p < 20 and not start:
                start = y
            if p < 20:
                end = y
                e = False
            if p > 200 and not end and start:
                end = y
        if not end or end <= img.size[1]/2:
            end = img.size[1]/2
        if not start or start >= img.size[1]/2:
            start = img.size[1]/2
        top.append(start)
        #draw.point((x,top[x]),fill='#00ff00')  
        bottom.append(end)   
        #draw.point((x,bottom[x]),fill='#0000ff')  
        
        
    # Getting top blocks
    tblocks = []
    for st in xrange(0,img.size[0],block):
        s = []
        for x in xrange(st,st+block):
            s.append(top[rf(x)]*1.0)
        s = min(s)
        tblocks.append([st,s])
         
    # Getting bottom blocks
    bblocks = []
    for st in xrange(0,img.size[0],block):
        s = []
        for x in xrange(st,st+block):
            s.append(bottom[rf(x)]*1.0)
        s = max(s)
        bblocks.append([st,s])

    bblocks = remove_gap_noise(bblocks,top=False)
    bblocks = remove_big_gaps(bblocks,top=False)

    tblocks = remove_gap_noise(tblocks)
    tblocks = remove_big_gaps(tblocks)
    
    center = []
    for i in xrange(len(tblocks)):
        center.append((bblocks[i][0],ceil((tblocks[i][1]+bblocks[i][1])/2)))          
         
    # getting points from blocks
    middle,points = get_sinusoid(center,block)

    # getting all points based on lines that goes through blocks points
    points = get_line_points(points)
    for i in xrange(1,len(points)):
        draw.point((points[i][0],points[i][1]),fill='#0000ff') 

    del draw
        
    return (points,middle)
    
def get_line_points(lps,t='line'):
    res = []
    
    if t == 'line':
        for i in xrange(1,len(lps)):
            x0,y0 = lps[i-1]
            x1,y1 = lps[i]
            delta = int(abs(x1-x0))
            h = y1-y0
            
            for l in xrange(delta):
                # aproximate by line
                ys = y0+l*(h/delta)
        
                res.append((int(x0+l),round(ys)))
                
        res.append((int(x1),round(y1)))
                    
    if t == 'cos':
        # NORMALIZING BY SINE
        # Not good idea for small images
        # if will be used net to be tested, because may contain bugs
        res = []
        for i in xrange(1,len(lps)):
            x0,y0 = lps[i-1]
            x1,y1 = lps[i]
            
            if y0 >= y1:
                t = 1
            else:
                t = 0
                
            delta = int(abs(x1-x0))
            h = abs(y1-y0)/2
            w = delta/pi
            xoffset = x0
            yoffset = -y0
            for l in xrange(delta):
                # aproximate by cos
                xs = (pi/delta)*l - pi/2 + pi*t
                ys = h*cos(xs/w-xoffset/w) - yoffset
        
                res.append((int(x0+l),int(ys)))
        
    return res
    
def get_sinusoid(array,block):
    
    points = []
    sp = 0
    for st,s in array:
        sp += s
        points.append((st+block*1.0/2,s))
        
    
    sp = sp/len(array)
    points.insert(0,(0,sp))
    
    return (sp,points)
    


def normalize_sin(img,points,zero_line):
    
    res = Image.new('RGB', (img.size[0],img.size[1]+30),(255,255,255))
    
    for x in xrange(img.size[0]):
        change = points[x][1]-zero_line
        for y in xrange(img.size[1]):
            c = img.getpixel((x,y))
            res.putpixel((int(x),int(y-change+15)),c)
            
              
    return res


def remove_big_gaps(array,top=True,fstep=3,block=8):
    """
        Finding big gaps beetwen blocks and set average value for gaped block
    """
    p = False
    if top:
        for i in xrange(0,len(array)-1):
            if p:
                p = False
                continue
            
            if (array[i][1] > array[i+1][1]+fstep) or \
                (array[i][1] < array[i+1][1]+fstep):
                if i > 0 and i < len(array)-1:
                    array[i][1] = (array[i-1][1]+array[i+1][1])/2
                    p = True
                else:
                    try:
                        array[i][1] = array[i+1][1]
                        p = True
                    except:
                        pass
                continue
        
        
    else:
        for i in xrange(0,len(array)-1):
            if p:
                p = False
                continue
            
            if (array[i][1] > array[i+1][1]+fstep) or \
                (array[i][1] < array[i+1][1]-fstep):
                try:
                    array[i+1][1] = (array[i][1]+array[i+2][1])/2
                    p = True
                except:
                    pass
                continue
                

    return array

def remove_gap_noise(array,top=True,fstep=6,move_coef=0.8,block=8):
    """
        Filter block for unnormal gaps (leters y,g,j,q and other)
    """
    s = 0
    for i in xrange(0,len(array)-1):
        s += array[i][1]*1.0
    s = s/len(array)
    if top:
        for i in xrange(0,len(array)-1):
            if array[i][1] < s - fstep:
                array[i][1] += (s-array[i][1])*move_coef 

    else:
        for i in xrange(0,len(array)-1):
            if array[i][1] > s + fstep:
                array[i][1] -= (array[i][1] - s)*move_coef 

    return array
    
    
if __name__ == "__main__":

    imgpath = sys.argv[1]
    img = cv.LoadImage(imgpath)
    img = togrey(img)
    cv.ShowImage('asdasd',img)
    cv.WaitKey();
    
    img = cv2pil(img).convert('RGB')
    points,middle = getCenterLine(img)
    img = normalize_sin(img,points,middle)
    img.show()
    
    
    
    
    
