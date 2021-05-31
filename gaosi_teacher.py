'''
 注意使用注释
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def gauss1d(x, mean, sigma):
     '''
     1d gaussian with mean and sigma
     :param x:
     :param mean:
     :param sigma:
     :return fx:
     '''
     dist = (x-mean)**2
     stand_dist = dist/(2*sigma**2)
     fx = np.exp(-stand_dist)/(np.sqrt(2*np.pi)*sigma)
     return fx

    if __name__=='__main__':
    #x = np.arange(-10, 10, 0.001)
    x = np.linspace(-10, 10, 1000)
    fx_m0_s1 = gauss1d(x, mean=0, sigma=1) ##对fx积分值远远大于一？？
    fx_m0_s5 = gauss1d(x, mean = 0, sigma = 5)
    plt.plot(x, fx_m0_s1, color='red', label='mu=%i, sigma=%.1f' % (0, 1))
    plt.show()


'''
二维高斯分布

'''
fig = plt.figure()
ax = fig.gca(projection = '3d')
y = np.linspace(-10, 10, 1000)
x = np.linspace(-8, 12, 1000)
X, Y = np.meshgrid(x, y) #合并矩阵，
fX = gauss1d(X, mean = 0, sigma = 1)
fY = gauss1d(Y, mean = 2, sigma = 1)
fp = fX * fY
ax.plot_surface(X, Y, fp, cmap = cm.coolwarm)
plt.show()