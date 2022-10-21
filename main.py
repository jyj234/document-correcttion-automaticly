import cv2 as cv
import numpy as np
from collections import Counter
import math
from matplotlib import pyplot as plt

def swap(a,b):
    t=a
    a=b
    b=t
def cal_dis(x1,y1,x2,y2):
    return (x1-x2)**2+(y1-y2)**2
document="2"
format=".jpg"
im=cv.imread(document+format)
#转化为灰度图
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#二值化
binary = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,51,3)

# cv.imwrite("binary.jpg",binary)
#检测轮廓值
cnts,hierarchy=cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

contour=sorted(cnts, key=cv.contourArea, reverse=True)[0]
cv.drawContours(im,contour,-1,255,3)
# cv.imwrite("contour.jpg",im)
#通过线性规划找到矩形的四个顶点
recsum=np.sum(contour,axis=2)
recdiff=np.diff(contour,axis=2)
lu=contour[np.argmin(recsum),0,:]
rd=contour[np.argmax(recsum),0,:]
ru=contour[np.argmin(recdiff),0,:]
ld=contour[np.argmax(recdiff),0,:]

# print(lu, ld, ru, rd)

if(cal_dis(lu[0],lu[1],ld[0],ld[1])<cal_dis(lu[0],lu[1],ru[0],ru[1])):
    ru,lu,rd,ld=lu,ld,ru,rd
#透视变换
src = np.float32([[lu[0], lu[1]], [ru[0], ru[1]], [ld[0], ld[1]], [rd[0], rd[1]]])
dst = np.float32([[0, 0], [im.shape[1], 0],[0, im.shape[0]] , [im.shape[1], im.shape[0]]])
m = cv.getPerspectiveTransform(src, dst)
result = cv.warpPerspective(im, m, (im.shape[1], im.shape[0]))
#将矫正后的图像转变为灰度图
res_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
#中值滤波去除椒盐噪声
res_median = cv.medianBlur(res_gray, 9)
# res_gauss=cv.GaussianBlur(res_gray,(3,3),0,0)

#自适应阈值
res_binary_median = cv.adaptiveThreshold(res_median,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,31,3)
# res_binary_gauss = cv.adaptiveThreshold(res_gauss,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,5,3)


document_res=document+"_res"+format
cv.imwrite(document_res,res_binary_median)

cv.waitKey(0)