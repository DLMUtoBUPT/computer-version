# plot the histogram
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

file_name = 'D:\cvtest/2313.jpg'
img_bgr = cv2.imread(file_name)
img_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
gray_levels = np.arange(0,256,1)
#pixel_counts
N_x = np.zeros_like(gray_levels, dtype=np.float64)
for (i,level) in enumerate(gray_levels):
 N_x[i] = np.sum(img_gray==level)
plt.bar(gray_levels, N_x)
plt.xlabel('bins = 256 gray levels')
plt.ylabel('Counted pixel numbers in each level')
plt.title('Gray Histogram')
plt.show()