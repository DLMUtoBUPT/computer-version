import cv2
from matplotlib import pyplot as plt
import numpy as np
import time

def histogram_demo(img):
    b, g, r = cv2.split(img)
    for x in (b, g, r):
        plt.hist(x.ravel(), 256, [0, 256])
        plt.title('channel histogram')
            # title内容如何动态变化？
        plt.ylabel('Pixel value')
        plt.show()


if __name__ == '__main__':
    start = time.time_ns()
    img = cv2.imread('D:\cvtest/iu.jpg')
    histogram_demo(img)
    end = time.time_ns()
    print("time is", end - start)
