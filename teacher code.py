"""
Split the RGB channels of the input colorful image.
Convert it into Grayscale image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__': #main entrance of the program


    file_name = 'C:\exp01/EXP01.jpg'
    img  = cv2.imread(file_name)

    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(cimg)
    plt.show()

    # check the image properties
    print('Type of the image: ', type(img))
    print('Shape of the image: {}'.format(img.shape))
    print('Image Height {}'.format(img.shape[0]))
    print('Image Width {}'.format(img.shape[1]))
    print('Dimension of Image {}'.format(img.ndim))
    # Find the max and min to check the bit depth of the image
    print('Maximum RGB value {}'.format(img.max()))
    print('Minimum RGB value {}'.format(img.min()))
    # Access to one pixel to observe the Red, Green, Blue value.

    print('Blue, Green, Red {}'.format(img[116,176,:]))

    #Split the RGB channels

    blueI  = np.zeros_like(img)
    greenI = np.zeros_like(img)
    redI   = np.zeros_like(img)

    blueI[:,:,0] = img[:,:,0]
    greenI[:,:,1] = img[:,:,1]
    redI[:,:,2]   = img[:,:,2]

    print('Total green intensities {}'.format(np.sum(greenI)))
    print('Total blue intensities {}'.format(np.sum(blueI)))
    print('Total red intensities {}'.format(np.sum(redI)))
    print('green occupies %f, blue takes %f, red occupies %f' %
          (np.sum(greenI)/np.sum(img), np.sum(blueI)/np.sum(img), np.sum(redI)/np.sum(img)))

    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]

    gray = 0.2627*red + 0.6780*green + 0.0593*blue
    #gray = (red + green + blue)
    #graynorm = 255*cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = gray.astype('uint8')
    gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ##彩色图转换为灰度图
    cv2.imshow('win', img)
    cv2.imshow('blue', blueI)
    cv2.imshow('green', greenI)
    cv2.imshow('red', redI)
    cv2.imshow('gray', gray)
    cv2.imshow('gray_cv', gray_cv)
    cv2.waitKey()


    def odd_even(x):
        if np.mod(x, 2)==0:
            return 'e'
        else:
            return 'o'

    op = ''
    for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        op = op + odd_even(x) + ' '
    print(op)