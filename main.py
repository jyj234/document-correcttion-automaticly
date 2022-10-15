import cv2 as cv
import numpy as np
from collections import Counter
import math
from matplotlib import pyplot as plt

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def ssim(y_true , y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom
im=cv.imread("document.jpg")
#转化为灰度图
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#全局二值化
ret,binary=cv.threshold(gray,0,255,cv.THRESH_BINARY| cv.THRESH_OTSU)
#检测轮廓
cnts,hierarchy=cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
maxarea=0
maxi=0
#找到面积最大的轮廓
for i in range(0,len(cnts)):
    area=cv.contourArea(cnts[i])
    if(area>maxarea):
        maxarea=area
        maxi=i
contour=cnts[maxi]

maxx_add=0
maxi_add=0
minx_add=10000
mini_add=0
t_add=0

maxx_sub=0
maxi_sub=0
minx_sub=10000
mini_sub=0
t_sub=0
#通过线性规划找到矩形的四个顶点
for i in range(0,contour.shape[0]):
    t_add=contour[i][0][0]+contour[i][0][1]
    t_sub=contour[i][0][1]-contour[i][0][0]
    if(t_sub>maxx_sub):
        maxx_sub=t_sub
        maxi_sub=i
    if(t_sub<minx_sub):
        minx_sub=t_sub
        mini_sub=i
    if(t_add>maxx_add):
        maxx_add=t_add
        maxi_add=i
    if(t_add<minx_add):
        minx_add=t_add
        mini_add=i
lu=np.array([contour[mini_add][0][0],contour[mini_add][0][1]])
rd=np.array([contour[maxi_add][0][0],contour[maxi_add][0][1]])
ld=np.array([contour[maxi_sub][0][0],contour[maxi_sub][0][1]])
ru=np.array([contour[mini_sub][0][0],contour[mini_sub][0][1]])
#画出四个顶点
# cv.circle(im,(int(lu[0]),int(lu[1])),5,(255,255,255),4)
# cv.circle(im,(int(rd[0]),int(rd[1])),5,(255,255,255),4)
# cv.circle(im,(int(ld[0]),int(ld[1])),5,(255,255,255),4)
# cv.circle(im,(int(ru[0]),int(ru[1])),5,(255,255,255),4)

#透视变换
src = np.float32([[lu[0], lu[1]], [ru[0], ru[1]], [ld[0], ld[1]], [rd[0], rd[1]]])
dst = np.float32([[0, 0], [im.shape[1], 0],[0, im.shape[0]] , [im.shape[1], im.shape[0]]])
m = cv.getPerspectiveTransform(src, dst)
result = cv.warpPerspective(im, m, (im.shape[1], im.shape[0]))

#将矫正后的图像转变为灰度图
res_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
#中值滤波去除椒盐噪声
res_median = cv.medianBlur(res_gray, 9)
res_gauss=cv.GaussianBlur(res_gray,(3,3),0,0)


#自适应阈值
#自己拍
res_binary_median = cv.adaptiveThreshold(res_median,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,31,3)
#作业给的
res_binary_gauss = cv.adaptiveThreshold(res_gauss,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,7,3)
# cv.imshow("res_binary_gauss",res_binary_median)

cv.imwrite("res.jpg",res_binary_gauss)

cv.waitKey(0)