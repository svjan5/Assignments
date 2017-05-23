## PROBLEM 3
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math

def wait():
    key = cv.waitKey(0);
    while (key == 233):
        key = cv.waitKey(0);
    cv.destroyAllWindows()


def addNoise(img, noise_level):
    rows, cols = img.shape;
    noise = np.ones((rows, cols), np.uint8)
    noise = cv.randu(noise, 0, 255)
    img_noise = cv.addWeighted(img, 1 - noise_level, noise, noise_level, 0)
    return img_noise


def getMatch(des1, des2):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2);
    # df = cv.DMatch(des2, des1);
    # matches = cv.matc
    # matches = df.match(des2, des1);
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])
    
    return len(good)


category = ['giraffe', 'cup', 'bottle', 'cat', 'book'];
n = len(category);
m = 5;

all_kps = [];
all_des = [];
imgs = [];

for i in range(n):
    imgs.append([])
    for j in range(1,m+1):
        src_path = './Images/q3/' + category[i]+'/'+str(j)+'.jpg'
        imgs[i].append(cv.imread(src_path, cv.IMREAD_GRAYSCALE))
        

sift = cv.xfeatures2d.SIFT_create();
for i in range(n):
    all_kps.append([])
    all_des.append([])
    for j in range(m):
        kp, des = sift.detectAndCompute(imgs[i][j],None)
        all_kps[i].append(kp)
        all_des[i].append(des)

model_kps = []
model_des = []

for i in range(n):
    model_kps.append([])
    model_des.append([])
    
    model_kps[i].append(all_kps[i][0])
    model_des[i].append(all_des[i][0])
    
    img_n1 = addNoise(imgs[i][0], 0.2)
    kp, des = sift.detectAndCompute(img_n1, None)
    model_kps[i].append(kp)
    model_des[i].append(des)
    
    img_n2 = addNoise(imgs[i][0], 0.6)
    kp, des = sift.detectAndCompute(img_n2, None)
    model_kps[i].append(kp)
    model_des[i].append(des)

    
    img_flip = cv.flip(imgs[i][0], 1);
    kp, des = sift.detectAndCompute(img_flip, None)
    model_kps[i].append(kp)
    model_des[i].append(des)
        
    # plt.subplot(2, 2, 1), plt.imshow(imgs[i][0], cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('Original')
    # plt.subplot(2, 2, 2), plt.imshow(img_n1, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('With uniform Noise - 0.2')
    # plt.subplot(2, 2, 3), plt.imshow(img_n2, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('With uniform Noise - 0.6')
    # plt.subplot(2, 2, 4), plt.imshow(img_flip, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('Flipped Image')
    # plt.show()
    # wait();
    # exit(0)
    
pos_count = neg_count = 0;

for i in range(n):
    for j in range(m):
        des_match_count = [0 for z in range(n)]
        for k in range(n):
            for t in range(len(model_kps[k])):
                des_match_count[k]  += getMatch(model_des[k][t], all_des[i][j]);

        
        pred = max(des_match_count)
        if des_match_count[i] == pred:
            pos_count = pos_count + 1
            print('Correct',i,j,i);
        else:
            neg_count = neg_count + 1;
            print('Incorrect',i,j,des_match_count.index(pred))

acc = pos_count/(pos_count+neg_count) * 100;
print('Corect pred = ', pos_count);
print('Incorrect pred = ', neg_count);

wait();