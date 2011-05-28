import cv
import sys
from PIL import Image, ImageDraw, ImageOps
from math import ceil,pi,cos,sin,tan,sqrt,acos
import time

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
   
def rad(x):
    # pi/180 = 0.017453292519943295
    return 0.017453292519943295*x
    
def cart(x):
    # pi/180 = 0.017453292519943295
    return x/0.017453292519943295  
   
################################################################################
######################## Removing disortions ################################### 
################################################################################  
   
def getCenterLine(img,block=8,use="center"):    
    """
        Draw and return center line of text.
    
        Input image must be previously cropped and words must have one baseline
        
    """
    draw = ImageDraw.Draw(img)
    
    def rf(num):
        return num if num < img.size[0] else img.size[0]-1 
     
    bufimg = img.load()
     
    # Getting top and bottom contour pixels     
    top = []
    bottom = [] 
    for x in xrange(img.size[0]):
        end = start = None
        for y in xrange(img.size[1]):
            p = bufimg[x,y]
            p = (p[0]+p[1]+p[2])/3 # for RGB
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
    
    #for i in xrange(len(bblocks)):
    #    draw.line((bblocks[i][0],bblocks[i][1],bblocks[i][0]+block,bblocks[i][1]),fill='#ff0000') 
    #    draw.line((tblocks[i][0],tblocks[i][1],tblocks[i][0]+block,tblocks[i][1]),fill='#00ff00') 
    
    center = []
    for i in xrange(len(tblocks)):
        center.append((bblocks[i][0],ceil((tblocks[i][1]+bblocks[i][1])/2)))   
        
    if use == 'bottom':  
        center = bblocks    
    
    if use == 'top':  
        center = tblocks    
    
    # getting points from blocks
    #print img.size[1],img.size[1]*1.0/50
    middle,points = get_sinusoid(center,block,s=img.size[1])

    # getting all points based on lines that goes through blocks points
    points = get_line_points(points)
    #for i in xrange(1,len(points)):
    #    draw.point((points[i][0],points[i][1]),fill='#0000ff') 

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
    
def get_sinusoid(array,block,s=0,phase=0.6):
    
    points = []
    sp = 0
    for st,s in array:
        sp += s
    sp = sp/len(array)
    if not phase:
        phase = abs(sp - s*1.0/2)/s*1.0/2 
    
    for st,s in array:
        points.append((st+block*1.0/2,s+(sp-s)*phase))
        
    
    points.insert(0,(0,sp))
    
    return (sp,points)
    


def normalize_sin(img,points,zero_line):
    
    res = Image.new('L', (img.size[0],img.size[1]+30),255)
    
    bufimg = img.load()
    
    for x in xrange(img.size[0]):
        try:
            change = points[x][1]-zero_line
            for y in xrange(img.size[1]):
                c = bufimg[x,y]
                res.putpixel((int(x),int(y-change+15)),c)
        except:
            pass         
              
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

## Finding lines ###############################################################

def skew(img,x=0,y=0):
    #(a,b,c,d,e,f)
    # x` = ax + by + c 
    # y` = dx + ey + f
    return img.transform((int(img.size[0]+abs(x)*4*img.size[1]),int(img.size[1]+abs(y)*2*img.size[0])),Image.AFFINE, (1, x, -abs(x)*img.size[1], y, 1, -abs(y)*img.size[0]))   

def find_angles(img,min_alpha=60,max_alpha=140,step_alpha=5,r=None,percent_black=0.8,xblock=6):
    
    if not r:
        r = int(img.size[1]-img.size[1]*0.1)
        
    #tmp = img.copy().convert('RGB')
    #draw = ImageDraw.Draw(tmp) 
        
    bufimg = img.load()
        
    angles = []
    passx = 0
    y0 = img.size[1]-1
    tmp_an = []
    for x0 in xrange(img.size[0]):
        for alpha in xrange(min_alpha,max_alpha,step_alpha):
            s = 0
            used = []
            for ri in xrange(1,r+1):    
                x1 = int(ri*cos(rad(alpha)))+x0
                #x1 = 2*x0 - x1
                y1 = int(img.size[1] - ri*sin(rad(alpha)))
                try:
                    if bufimg[x1,y1] < 40 and (x1,y1) not in used:
                        s += 1
                        used.append((x1,y1))
                except:
                    pass

            if s*1.0/r >= percent_black:    
                tmp_an.append(int(alpha))
                passx = x0+xblock
                #draw.line((x0,y0,x1,y1),fill=(255,0,0))                  
                    
        if x0 >= passx:
            l = len(tmp_an) or 1
            alpha = sum(tmp_an)/l
            if alpha >= min_alpha and alpha < max_alpha : 
                angles.append(alpha)
            tmp_an = []
    
    l = len(tmp_an) or 1
    alpha = sum(tmp_an)/l
    if alpha >= min_alpha and alpha < max_alpha : 
        angles.append(alpha)


    la = [a for a in angles if a > 90]
    ra = [a for a in angles if a < 90]
    ca = [a for a in angles if a == 90]

    #print 'left',la
    #print 'right',ra
    #print 'center',ca
    

    #tmp.show()
    #time.sleep(1)

    if (len(la) < len(ca) and len(ra) < len(ca)) or (not len(ra) and not len(la) and not len(ca)):
        return img
        
    if len(la) > len(ra):
        # skew to the right
        l = len(la) or 1
        skew_angle = sum(la)/l
        #print 'alpha right',skew_angle-90
        coef = tan(rad(skew_angle-90))
    else:
        # skew to the left 
        l = len(ra) or 1
        skew_angle = sum(ra)/l
        #print 'alpha left',-(90-skew_angle)
        coef = tan(rad(-(90-skew_angle)))
              
    img = ImageOps.invert(img)
            
    #print 'tan:',coef
    img = skew(img,coef,0)

    img = ImageOps.invert(img)
    

    #del draw
    #tmp.show()
    
    return img

## Finding breaks ##############################################################
def find_breaks(img,percent_black=0.02):

    draw = ImageDraw.Draw(img) 
        
    bufimg = img.load()
    
    for x in xrange(img.size[0]):
        s = 0
        for y in xrange(img.size[1]):
            c = (bufimg[x,y][0]+bufimg[x,y][1]+bufimg[x,y][2])/3
            if c < 10:
                s += 1
      
        s = s*1.0/img.size[1]
        if s <= percent_black:
            draw.line((x,img.size[1],x,0),fill="#ff0000")
            
    del draw
    
    return img
        

## Croping #####################################################################

def split_words(img,length=3):

    img = intel_crop(img)
    
    bufimg = img.load()
    
    ca = 0
    for x in xrange(img.size[0]):
        count = 0
        for y in xrange(img.size[1]):
            if bufimg[x,y] > 240:
                count += 1
        if count == img.size[1]:
            ca += 1
            if ca >= length:
                break
                
    split_x = x-int(length/2)
    
    img1 = img.transform((split_x,img.size[1]),Image.EXTENT, (0,0,split_x,img.size[1]))
    img2 = img.transform((img.size[0]-split_x,img.size[1]),Image.EXTENT, (split_x,0,img.size[0],img.size[1]))
    
    img1 = intel_crop(img1)
    img2 = intel_crop(img2)

    return img1,img2
    

def intel_crop(img,xstop = 0.02,ystop = 0.02):
    y1 = x1 = x2 = y2 = 0
    
    bufimg = img.load()
    
    for y in xrange(img.size[1]):
        count = 0
        for x in xrange(img.size[0]):
            if bufimg[x,y] < 120:
                count += 1

        if count*1.0/img.size[0] >= ystop:
            y1 = y
            break
            
    for y in xrange(img.size[1]-1,0,-1):
        count = 0
        for x in xrange(img.size[0]):
            if bufimg[x,y] < 120:
                count += 1

        if count*1.0/img.size[0] >= ystop:
            y2 = y
            break
    
    for x in xrange(img.size[0]):
        count = 0
        for y in xrange(img.size[1]):
            if bufimg[x,y] < 120:
                count += 1

        if count*1.0/img.size[1] >= xstop:
            x1 = x
            break
            
    for x in xrange(img.size[0]-1,0,-1):
        count = 0
        for y in xrange(img.size[1]):
            if bufimg[x,y] < 120:
                count += 1
        if count*1.0/img.size[1] >= xstop:
            x2 = x
            break           
      
    return img.transform((x2-x1,y2-y1),Image.EXTENT, (x1,y1,x2,y2))
    
################################################################################
################################################################################ 
################################################################################     



    
if __name__ == "__main__":

    imgpath = sys.argv[1]
    img = cv.LoadImage(imgpath)
    img = togrey(img)
    cv.ShowImage('asdasd',img)
    cv.WaitKey();
    
    img = cv2pil(img).convert('RGB')
    img = img.convert('L')
    imgs = split_words(img)
    
    #imgs[0].show()
    #imgs[1].show()
    #sys.exit()
    
    for img in imgs:
        img = img.convert('RGB')
        points,middle = getCenterLine(img)
        img = img.convert('L')
        img = normalize_sin(img,points,middle)
        
        img.show()
        
        img = intel_crop(img,ystop=0.1)

        find_angles(img)

    #cvim = pil2cv(img)
    
    #cv.ShowImage('asdasd',cvim)
    #cv.WaitKey();
    
    
    
    
    
    
    
