import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_name = 'c:/exp01/EXP01.jpg'
    img = cv2.imread(file_name)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img2)
    plt.show()

    print('type', type(img))
    print('shape{}'.format(img.shape))
    print('image height{}'.format(img.shape[0]))
    print('image width'.format(img.shape[1]))
    print('dimension{}'.format(img.ndim))
    print('maxium RGB value {}'.format(img.max()))
    print('minium RGB value{}', format(img.min))
    print('blue green red{}'.format(img[111, 111, :]))

    ##以下 关键代码

    bluel = np.zeros_like(img)
    greenl = np.zeros_like(img)
    redl = np.zeros_like(img)

    bluel[:, :, 0] = img[:, :, 0]
    greenl[:, :, 1] = img[:, :, 1]
    redl[:, :, 2] = img[:, :, 2]

    print('total g{}'.format(np.sum(greenl)))
    print('total b{}'.format(np.sum(bluel)))
    print('red total{}'.format(np.sum(redl)))

    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    gray = 0.2627 * red + 0.6780 + 0.593 * blue
    gray = gray.astype('uint8')
    gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('yuan_tu', img)
    cv2.imshow('blue_tuceng', bluel)
    cv2.imshow('green_tuceng', greenl)
    # cv2.imshow('red_tuceng',redl)
    # cv2.imshow('gray',gray)
    # cv2.imshow('gray_cv',gray_cv)

    redl=cv2.cvtColor(redl,cv2.COLOR_BGR2GRAY)
    # gray=cv2.cvtColor(gray,cv2.COLOR_BGR2RGB)
    # gray_cv=cv2.cvtColor(gray_cv,cv2.COLOR_BGR2RGB)

    plt.show(redl)
    plt.imshow(gray)
    plt.imshow(gray_cv)
    cv2.waitKey()
