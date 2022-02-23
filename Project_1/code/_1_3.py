import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

n = [1,0,-1,0,1]

im1 = cv.imread("C:\\DIP\\Project_1\\1_3\\rye_catcher_e_1.png")
im2 = cv.imread("C:\\DIP\\Project_1\\1_3\\rye_catcher_c_1.png")

def image_dilate(img):
    H, W = img.shape
    out = img.copy()
    #tmp = img.copy()
    MF = np.array(((0, 1, 0),(1, 1, 1),(0, 1, 0)))
    tmp = np.pad(out, (1, 1), 'edge')
    for y in range(1, H):
        for x in range(1, W):
            print(y,x,out[y,x])
            if  np.amax(MF * tmp[y-1:y+2, x-1:x+2]) == 255:
            #if tmp[y-1,x]==255|tmp[y,x-1]==255|tmp[y+1,x]==255|tmp[y,x+1]==255|tmp[y,x]==255:
                out[y, x] = 255
    return out

def filling(im):
    img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #转化为纯黑白（0，1）图像
    img[img < 120] = 0
    img[img >= 120] = 255
    # 原图取补
    mask = 255 - img
    # 构造膨胀起始图像
    temp = np.zeros_like(img)
    temp[0, :] = 255
    temp[-1, :] = 255
    temp[:, 0] = 255
    temp[:, -1] = 255
    se = cv.getStructuringElement(shape=cv.MORPH_CROSS, ksize=(3, 3))
    while True:
        temp_0 = temp
        dilation = image_dilate(temp,se)
        temp = np.min((dilation, img), axis=0)
        if (temp_0 == temp).all():
            break
    result = 255 - temp
    return result

# 显示
cv.imshow("im1",im1)
cv.imshow("im2",im2)
cv.imshow("new1",filling(im1))
cv.imshow("new2",filling(im2))
cv.waitKey()
