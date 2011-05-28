import cv
from cv import *
import sys


src = cv.LoadImage(sys.argv[1], 8) # OpenCV object
cv.ShowImage('asdasd',src)
cv.WaitKey()

storage1 = cv.CreateMemStorage()
seq1 = cv.FindContours(src, storage1, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)

mask = cv.LoadImage(sys.argv[2], 8)
storage2 = cv.CreateMemStorage()
seq2 = cv.FindContours(mask, storage2, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)

a = cv.MatchShapes(seq1,seq2,method=1)

print a

cv.ShowImage('asdasd',mask)
cv.WaitKey()
