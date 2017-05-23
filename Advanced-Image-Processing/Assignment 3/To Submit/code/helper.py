import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import helper as hp
plt.style.use('ggplot')
import time
from IPython.display import display, HTML
import os

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

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

# Create BRISK object
