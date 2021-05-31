import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

start = time.perf_counter()
img = cv2.imread('D:\cvtest/2313.jpg')
rows, cols, dpt = img.shape
size = rows * cols
color = ('b', 'g', 'r')
colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))

for i, c in enumerate(colors):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    histr = histr * cols / size
    lines = []
    for j, k in enumerate(histr):
        x = int(j * rows / 255)
        y = cols - int(k * 10)
        # cv2.circle(img, (x, y), 2, c, -1)  # 点显示
        ploy = lines.append([x, y])
    pts = np.array([lines], dtype = np.int32)
    cv2.polylines(img, pts, 0, c, 2)
    plt.subplot(122), plt.plot(histr, color = color[i]),
    plt.xlim([0, 256]), plt.title('Histogram')

plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Image2')
plt.ion()
plt.pause(2)
plt.close()
end = time.perf_counter()
print('time is', end - start-2)
