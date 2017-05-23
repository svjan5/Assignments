## PROBLEM 1

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math

def getLaplacianOp(shape=(5,5), sigma=0.5):
    dim = int(shape[0]/2);
    Y = X = np.arange(-dim, dim+1, 1);
    op = np.zeros(shape, np.float64)

    i = 0;
    for x in X:
        j = 0
        for y in Y:
            h = -(x**2 + y**2) / (2.0*sigma ** 2)
            op[i,j]= (-1 / (np.pi * sigma ** 4)) * (1 + h) * np.exp(h)
            j=j+1
        i=i+1

    return op;

img_org = cv.imread('./Images/q1/cameraman.tif', cv.IMREAD_GRAYSCALE)
rows, cols = img_org.shape;

noise = np.ones((rows,cols), np.float64)

noise_levels = np.linspace(0, 80, 6);
counter = 1;

for noise_level in noise_levels:

    # Adding noise to the Image
    noise = cv.randn(noise, 0, noise_level)
    img = img_org + noise;
    
    # Gradient Based method (Sobel)
    filter = np.ones((5,5), np.float32)/25;
    img_blur = cv.filter2D(img, -1, filter);

    sobelX = np.float64([   [-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

    sobelY = np.float64([   [1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])

    Ix = cv.filter2D(img, cv.CV_64F, sobelX);
    Iy = cv.filter2D(img, cv.CV_64F, sobelY);

    out_grad = np.sqrt(Ix*Ix + Iy*Iy)
    
    out_grad = out_grad > 300

    # LAPLACIAN OF GAUSSIAN METHOD
    lapcian_op = getLaplacianOp((9,9), 1.0);


    out_log = cv.filter2D(img, cv.CV_64F, lapcian_op)
    out_log = out_log > 17.5;
    # plt.subplot(2, 3, 1), plt.imshow(out_log, cmap='gray'), plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 3, 2), plt.imshow(out_log > 0, cmap='gray'), plt.title('Threshold = 0'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 3, 3), plt.imshow(out_log > 7.5, cmap='gray'), plt.title('Threshold = 7.5'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 3, 4), plt.imshow(out_log > 15, cmap='gray'), plt.title('Threshold = 15'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 3, 5), plt.imshow(out_log > 20, cmap='gray'), plt.title('Threshold = 20'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 3, 6), plt.imshow(out_log > 45, cmap='gray'), plt.title('Threshold = 45'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # exit(0)

    # DISPLAY OUTPUT
    plt.figure();
    title = 'Sigma value: ' + str(noise_level);
    plt.suptitle(title, fontsize=15)

    # plt.figtext(.5, .72, title, fontsize=16, ha='center')
    plt.subplot(1,3,1), plt.imshow(img, cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    plt.subplot(1,3,2), plt.imshow(out_grad, cmap = 'gray')
    plt.title('Gradient based'), plt.xticks([]), plt.yticks([])

    plt.subplot(1,3,3), plt.imshow(out_log, cmap = 'gray')
    plt.title('LOG'), plt.xticks([]), plt.yticks([])
    plt.show()
    # break;
    # plt.savefig('./Solution_Report/q1/noise_' + str(counter) + '.png',  bbox_inches='tight')
    # counter = counter + 1;
    # break;

cv.waitKey(100000)
cv.destroyAllWindows();