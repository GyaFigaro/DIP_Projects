import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

def getGrayHist(I):
    # 计算灰度直方图
    h = I.shape[0]
    w = I.shape[1]
    grayhist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayhist[I[i][j]] += 1
    return grayhist

def process(img):
    h = img.shape[0]
    w = img.shape[1]
    #得到灰度直方图
    grayhist = getGrayHist(img)
    #累加灰度直方图
    factor = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            factor[p] = grayhist[0]
        else:
            factor[p] = factor[p - 1] + grayhist[p]
    #得到输入和输出之间的线性关系
    out = np.zeros([256], np.uint8)
    k = 256.0 / (h * w)
    for p in range(256):
        q = k * float(factor[p]) - 1
        if q >= 0:
            out[p] = math.floor(q)
        else:
            out[p] = 0
    #均衡化后的图像
    res = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            res[i][j] = out[img[i][j]]
    return res
 
# 使用自己写的函数实现
img = cv.imread("C:\\DIP\\Project_1\\1_1\\face.png",0)
equa = process(img)
cv.imshow("img", img)
cv.imshow("equa", equa)
cv.waitKey()
