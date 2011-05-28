import sys
import os

from PIL import Image 
from preprocess import intel_crop,normalize_sin,getCenterLine,split_words,find_angles,find_breaks

import time

if __name__ == "__main__":

    path = sys.argv[1].rstrip('/')+'/'
    patho = sys.argv[2].rstrip('/')+'/'
    images = os.listdir(path)
    
    start = time.time()
    
    html = '<html><body>'
    
    for imp in images:
        
        img = Image.open(path+imp)
        img = img.convert('L')
        img.save(patho+imp,'JPEG')
        html += '<img src="%s" />&nbsp&nbsp=&gt;&nbsp&nbsp' % imp
        
        imgs = split_words(img)
        
        for i,img in enumerate(imgs):
        
            #img = intel_crop(img)
            img = img.convert('RGB')
            points,middle = getCenterLine(img,use='center')
            img = img.convert('L')
            img = normalize_sin(img,points,middle)
            
            #img = img.convert('L')
            
            img = intel_crop(img)

            img = find_angles(img)

            img = img.convert('RGB')
            img = find_breaks(img)
            
            img.save(patho+imp+'.'+str(i),'JPEG')
            
            html += '<img src="%s" />' % (imp+'.'+str(i))
        
        html += '<br/>'
        
    html += '</body></html>'
    
    open(patho+'index.html','w').write(html)
    
    print time.time()-start
