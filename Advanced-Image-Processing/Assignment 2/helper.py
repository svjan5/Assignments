import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.stats as st

def wait(time=0):
    key = cv.waitKey(time);
    while (key == 233):
        key = cv.waitKey(0);
    cv.destroyAllWindows()
    
def viewImage(name, img):
    img = cv.normalize(img, img, alpha=255, beta=0, norm_type = cv.NORM_MINMAX, dtype = cv.CV_16UC1)
    cv.imshow(name, img)

def getMSE(img_new, img_org):
    rows, cols = img_new.shape
    return np.sum((img_org - img_new)*(img_org -  img_new)) / (rows*cols)

def show(img, title='',  size=(8,5)):
    plt.rcParams["figure.figsize"] = size;
    plt.imshow(img, cmap='gray'), plt.title(title), plt.xticks([]), plt.yticks([]);
    plt.show()

def subShow(i, j, k, img, title = ''):
    plt.subplot(i,j,k), plt.imshow(img, cmap='gray'), plt.title(title), plt.xticks([]), plt.yticks([]);

def cvShow(img, title=' '):
    cv.imshow(title, img)
    wait(10000)
    
def gaussianKernel(kernlen=21, nsig=3):
    ker = cv.getGaussianKernel(kernlen, nsig, cv.CV_32F)
    ker = np.matmul(ker, ker.T)
    return ker

def normImg(img):
    return cv.normalize(img, img, alpha=255, beta=0, norm_type = cv.NORM_MINMAX, dtype = cv.CV_8UC1)

def saturate(img):
    img[img > 255] = 255;
    img[img < 0] = 0;
    return img