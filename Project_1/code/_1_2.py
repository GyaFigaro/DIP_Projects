import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

#最小值滤波器
def medianfilter(img):
    size = 3
    h,w = img.shape[:2]
    p = size//2
    res = np.zeros((h + 2 * p, w + 2 * p, 3), dtype = np.uint8)  
    res[p:p + h, p:p + w] = img.copy()
    tmp = res.copy()
    for j in range(h):
        for i in range(w):
                res[p + j, p + i] = np.amin(tmp[j:j + size,i:i + size])
    res = res[p:p + h, p:p + w].copy()
    return res

#显示
img = cv.imread("C:\\DIP\\Project_1\\1_2\\hit.png")
new = medianfilter(img)
cv.imshow("img",img)
cv.imshow("new",new)
cv.waitKey()