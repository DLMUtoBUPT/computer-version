import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    file_name = 'c:/exp01/EXP01.jpg'
    img = cv2.imread(file_name)
    # ge shi exchange
    # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img2)
    plt.show()

    print('Type of the image: ', type(img))
    print('shape of the image')

    bluel = np.zeros_like(img)
    greenl = np.zeros_like(img)
    redl = np.zeros_like(img)

    bluel[:,:,0] = img[:, :, 0]
    greenl[:,:,1] = img[:, :, 1]
    redl[:,:,2] = img[:, :, 2]

    print('total blue intensities{} ', format(np.sum(bluel)))
    print('total green intensities{} ', format(np.sum(greenl)))
    print('total red intensities{} ', format(np.sum(redl)))
    print('green occupies:%f,blue occupies:%f,red occupies:%f' %
          (np.sum(greenl) / np.sum(img), np.sum(bluel) / np.sum(img), np.sum(redl) / np.sum(img)))

    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]

    gray = 0.2627 * red + 0.6788 * green + 0.0593 * blue
    #gray = gray.astype("uint8")
    #def cv2.COLOR_BGR2GRAY 2
    gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('yuan_tu', img)
    cv2.imshow('blue', bluel)
    cv2.imshow('green', greenl)
    cv2.imshow('red', redl)
    cv2.imshow('gray', gray)
    cv2.imshow('grayl', gray_cv)

    cv2.waitKey()

# type
# cimg = cv2.cvtcolor(img, cv2.color)
# uint8:unsigned int 8 bit?
