import cv2 as cv
import numpy as np
from collections import Counter
import math
from matplotlib import pyplot as plt

def img_show(img,img_name):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.title(img_name)

def convert_polar_to_two_points(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 10000 * (-b))
    y1 = int(y0 + 10000 * (a))
    x2 = int(x0 - 10000 * (-b))
    y2 = int(y0 - 10000 * (a))
    return x1, y1, x2, y2

def GeneralEquation(x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C

def cal_crossover(rho1, theta1, rho2, theta2):
    x11, y11, x12, y12 = convert_polar_to_two_points(rho1, theta1)
    x21, y21, x22, y22 = convert_polar_to_two_points(rho2, theta2)
    A1, B1, C1 = GeneralEquation(x11, y11, x12, y12)
    A2, B2, C2 = GeneralEquation(x21, y21, x22, y22)
    m = A1 * B2 - A2 * B1
    x = (C2 * B1 - C1 * B2) / m
    y = (C1 * A2 - C2 * A1) / m
    return [x, y]

document="3"
format=".jpg"
im = cv.imread(document+format)

# img = cv.resize(img, (600, 800), interpolation=cv.INTER_AREA)
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# ret,binary=cv.threshold(gray,0,255,cv.THRESH_BINARY| cv.THRESH_OTSU)
# cv.imshow("binary",binary)

# ret,res_binary=cv.threshold(res_gray,0,255,cv.THRESH_BINARY| cv.THRESH_OTSU)
# cv.imshow("res_binary", res_binary)
# print(lines)
# RotatedRect=cv.minAreaRect(cnts[maxi])
# print(RotatedRect[0][0])
# cv.circle(im,(int(RotatedRect[0][0]),int(RotatedRect[0][1])),5,(255,255,255),4)
# box=np.int0(cv.boxPoints(RotatedRect))
# print(box)
# for i in range(0,3):
#     cv.circle(im,(box[i][0],box[i][1]), 20, (255, 255, 255), 4)
# cv.circle(im,(560-514,750-574),5,(255,255,255),4)



# edges = cv.Canny(gray, threshold1=100, threshold2=200,apertureSize =3)
binary = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,3,3)
cv.imwrite(document+"_canny"+format,binary)

lines = cv.HoughLines(binary, rho=1, theta=np.pi/240, threshold=500)
# print(type(lines))
a,b=Counter(lines[:,0,1]).most_common(2)
print(a,b)
theta1=a[0]
theta2=b[0]
l1=lines[lines[:,0,1]==theta1]
r11=min(l1[:,0,0])
r12=max(l1[:,0,0])
l2=lines[lines[:,0,1]==theta2]
r21=min(l2[:,0,0])
r22=max(l2[:,0,0])
print(r11,r12,r21,r22)


# lines = cv.HoughLinesP(edges, 1, np.pi/480, 150,100,10)
# k=(lines[:,0,3]-lines[:,0,1])/(lines[:,0,2]-lines[:,0,0])
# b=lines[:,0,1]-k*lines[:,0,0]
# theta=np.arctan(1/k)
# r=b*np.sin(theta)
# print(theta)
# theta
# a,b=Counter(theta).most_common(2)
# print(a,b)
# theta1=a[0]
# theta2=b[0]
# l1=r[theta==theta1]
# r11=min(l1)
# r12=max(l1)
# l2=r[theta==theta2]
# r21=min(r)
# r22=max(r)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        x1,y1,x2,y2=convert_polar_to_two_points(rho,theta)
        # x1,y1,x2,y2=line[0]
        cv.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # print(line)
x0, y0 = cal_crossover(r11, theta1, r21, theta2)
x1,y1=cal_crossover(r11,theta1,r22 , theta2)
x2,y2=cal_crossover(r12,theta1,r21 , theta2)
x3,y3=cal_crossover(r12,theta1,r22, theta2)

cv.circle(im,(int(x0),int(y0)),5,(255,255,255),4)
cv.circle(im,(int(x1),int(y1)),5,(255,255,255),4)
cv.circle(im,(int(x2),int(y2)),5,(255,255,255),4)
cv.circle(im,(int(x3),int(y3)),5,(255,255,255),4)

cv.imwrite(document+"_line"+format,im)

src = np.float32([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
dst = np.float32([[0, 0], [im.shape[1], 0],[0, im.shape[0]] , [im.shape[1], im.shape[0]]])
m = cv.getPerspectiveTransform(src, dst)
result = cv.warpPerspective(im, m, (im.shape[1], im.shape[0]))
# cv.imwrite(document+"_res"+format,result)

# cv.waitKey(0)