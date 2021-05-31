'''
将一个图像多次卷积，高斯模糊处理
2021.5.25
zhangkaibo
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


# 自定义模糊 (卷积可以处理模糊和锐化)
def custom_blur_demo(image):
    kernel = np.ones([5, 5], np.float64) / 25
    image = cv2.filter2D(image, -1, kernel = kernel)
    # cv2.imshow("mo_hu_iamge", dst)
    # cv2.waitKey()
    return image


'''
锐化图像
    kernel1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float64)
    dst1 = cv2.filter2D(image, -1, kernel = kernel1)
    cv2.imshow("mo_hu_image", dst1)
'''

if __name__ == '__main__':
    src = cv2.imread('D:\cvtest/zhi_en.png')
    cv2.imshow("yuan_tu", src)
    # blur_demo(src)
    for i in range(50):

        src = cv2.GaussianBlur(src, (5, 5), 0)

    cv2.imshow("mo_hu_iamge", src)
    cv2.waitKey()
    # median_blur_demo(src)
