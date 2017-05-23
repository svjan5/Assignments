## PROBLEM 2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
import math


def addNoise(img, noise_level):
    rows, cols = img.shape;
    noise = np.ones((rows, cols), np.uint8)
    noise = cv.randu(noise, 0, 255)
    img_noise = cv.addWeighted(img, 1 - noise_level, noise, noise_level, 0)
    return img_noise


def toRadians(deg):
    return (deg * math.pi) / 180;


def wait():
    key = cv.waitKey(0);
    while (key == 233):
        key = cv.waitKey(0);
    cv.destroyAllWindows()


def getMatch(img1, kp1, des1, img2, kp2, des2):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2);
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])
    
    return len(good), cv.drawMatchesKnn(img1, kp1, img2, kp2, good, img1, flags=2)


# -----------------------Load Image-------------------------
img1_path = './Images/q2/house.png'
img2_path = './Images/q2/stop_symbol.jpg'
img = cv.imread(img2_path, cv.IMREAD_GRAYSCALE);
rows, cols = img.shape;

# img = np.float64(img);

sift = cv.xfeatures2d.SIFT_create();
kp_org, des_org = sift.detectAndCompute(img, None)

# ----------------------Adding Noise-------------------------
img_noise = addNoise(img, 0.2);
kp_noise, des_noise = sift.detectAndCompute(img_noise, None)

# --------------Applying Affine transformation----------------
theta = toRadians(15);

M = np.float32([[1.2 * math.cos(theta), math.sin(theta), 0],
                [-math.sin(theta), 1.2 * math.cos(theta), 0]])

img_aff = cv.warpAffine(img, M, (cols, rows));
kp_aff, des_aff = sift.detectAndCompute(img_aff, None)

# ----------------Applying both transformation--------------------
img_both = addNoise(img_aff, 0.2)
kp_both, des_both = sift.detectAndCompute(img_both, None)

# ==After log transform==
light = 135
img_log = img.copy();
for i in range(rows):
    for j in range(cols):
        if (img_log[i,j] + light > 255):
            img_log[i,j] = 255;
        else:
            img_log[i,j] = img_log[i,j] + light;
    
#
# cv.imshow('input', img);
# wait()
# exit()
# img_log = np.uint8(np.log(img));
kp_log, des_log = sift.detectAndCompute(img_log, None);

# img_flip
img_flip = cv.flip(img, 1);

kp_flip, des_flip = sift.detectAndCompute(img_flip, None);

# -------------Showing Keypoints and matches------------------
img_k1 = cv.drawKeypoints(img, kp_org, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_k2 = cv.drawKeypoints(img_noise, kp_noise, img_noise, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_k3 = cv.drawKeypoints(img_aff, kp_aff, img_aff, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_k4 = cv.drawKeypoints(img_both, kp_both, img_both, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_k5 = cv.drawKeypoints(img_log, kp_both, img_log, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_k6 = cv.drawKeypoints(img_flip, kp_both, img_flip, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(1);
plt.suptitle('Showing keypoints', fontsize=15)
plt.subplot(2, 3, 1), plt.imshow(img_k1, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('Original ({:d})'.format(len(kp_org)))
plt.subplot(2, 3, 2), plt.imshow(img_k2, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('With Noise ({:d})'.format(len(kp_noise)))
plt.subplot(2, 3, 3), plt.imshow(img_k3, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('With affine and occlusion trans ({:d})'.format(len(kp_aff)))
plt.subplot(2, 3, 4), plt.imshow(img_k4, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('With noise and affine trans ({:d})'.format(len(kp_both)))
plt.subplot(2, 3, 5), plt.imshow(img_k5, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('With changed illumination({:d})'.format(len(kp_log)))
plt.subplot(2, 3, 6), plt.imshow(img_k6, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('With flip transform ({:d})'.format(len(kp_flip)))

no_m1, img_m1 = getMatch(img, kp_org, des_org, img_aff, kp_aff, des_aff);
no_m2, img_m2 = getMatch(img, kp_org, des_org, img_noise, kp_noise, des_noise);
no_m3, img_m3 = getMatch(img, kp_org, des_org, img_both, kp_both, des_both);
no_m4, img_m4 = getMatch(img, kp_org, des_org, img_log, kp_log, des_log);
no_m5, img_m5 = getMatch(img, kp_org, des_org, img_flip, kp_flip, des_flip);

cv.imwrite('./Solution_Report/q2/match2_1.png', img_m1)
cv.imwrite('./Solution_Report/q2/match2_2.png', img_m2)
cv.imwrite('./Solution_Report/q2/match2_3.png', img_m3)
cv.imwrite('./Solution_Report/q2/match2_4.png', img_m4)
cv.imwrite('./Solution_Report/q2/match2_5.png', img_m5)

plt.figure(2);
plt.suptitle('Comparing Original Image with transformed images', fontsize=15)
plt.subplot(5, 1, 1), plt.imshow(img_m1, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('Affine tranformation ({:d})'.format(no_m1))
plt.subplot(5, 1, 2), plt.imshow(img_m2, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('With noise ({:d})'.format(no_m2))
plt.subplot(5, 1, 3), plt.imshow(img_m3, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('With noise & affine trans ({:d})'.format(no_m3))
plt.subplot(5, 1, 4), plt.imshow(img_m4, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('With changed illumination ({:d})'.format(no_m4))
plt.subplot(5, 1, 5), plt.imshow(img_m5, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('With flip transform ({:d})'.format(no_m5))

plt.show();