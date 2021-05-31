import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
# def gauss1d(x, mean, sigma):
#     '''
#     1d gaussian with mean and sigma
#     :param x:
#     :param mean:
#     :param sigma:
#     :return fx:
#     '''
#     dist = (x-mean)**2
#     stand_dist = dist/(2*sigma**2)
#     fx = np.exp(-stand_dist)/(np.sqrt(2*np.pi)*sigma)
#     return fx
#
#
# if __name__=='__main__':
#     from mpl_toolkits.mplot3d import Axes3D
#     from matplotlib import cm
#
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     y = np.linspace(-10, 10, 1000)
#     x = np.linspace(-8, 12, 10000)
#     fx = gauss1d(x, mean=0, sigma=1)
#     print('sum of Gaussian function %.5f'%np.sum(fx))
#     X, Y = np.meshgrid(x, y)
#
#     fX = gauss1d(X, mean=0, sigma=1)
#     fY = gauss1d(Y, mean=2, sigma=1)
#     fp = fX * fY
#     ax.plot_surface(X, Y, fp, cmap=cm.coolwarm)
#     plt.show()

# plot the histogram
file_name = 'D:\cvtest/2313.jpg'
img_bgr   = cv2.imread(file_name)
img_gray  = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)

gray_levels = np.arange(0,256,10)
#pixel_counts
time_start = time.time_ns()
#time.sleep(1)
N_x = np.zeros_like(gray_levels, dtype=np.float64)
for (i,level) in enumerate(gray_levels): #
     if i>=25:
         break#大于26 没有取值
     condition_matrix = (img_gray == level)#
     #N_x[i] = np.sum(img_gray==level)
     condition_matrix_big    = (img_gray > gray_levels[i])
     condition_matrix_little = (img_gray <= gray_levels[i+1])
     condition_matrix = condition_matrix_big & condition_matrix_little#用+不可以，必须同时满足。
     N_x[i] = np.sum(condition_matrix)#计数
time_cost_diy = time.time_ns() - time_start

# time_start = time.time_ns()
# hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
# hist_np = np.histogram(img_gray.ravel(), gray_levels)
# # plt.bar(range(0,256), hist.reshape(256,),color='b')
# # plt.xlim([0, 256])
# time_cost_cv2 = time.time_ns() - time_start
# print('histogram compute time %.7f - %.7f seconds'%(time_cost_diy, time_cost_cv2))


plt.bar(gray_levels, N_x,width = 10)
plt.xlabel('bins = 256 gray levels')
plt.ylabel('Counted pixel numbers in each level')
plt.title('Gray Histogram')
plt.pause(0.1)
plt.show()