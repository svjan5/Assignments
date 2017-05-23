clc;clear;
% Load Images
img_org       = imread('Images/einstein.gif');
img_blur	  = imread('Images/blur.gif');
img_contrast  = imread('Images/contrast.gif');
img_impulse	  = imread('Images/impulse.gif');
img_jpg       = imread('Images/jpg.gif');
img_meanshift = imread('Images/meanshift.gif');
[m,n] = size(img_org);

% Part (a): Calculating Mean square error
mse_error = zeros(5,1);

mse_error(1) = sum(sum ((img_org - img_blur) .* (img_org - img_blur))) / (m*n);
mse_error(2) = sum(sum ((img_org - img_contrast) .* (img_org - img_contrast)))/ (m*n);
mse_error(3) = sum(sum ((img_org - img_impulse) .* (img_org - img_impulse)))/ (m*n);
mse_error(4) = sum(sum ((img_org - img_jpg) .* (img_org - img_jpg)))/ (m*n);
mse_error(5) = sum(sum ((img_org - img_meanshift) .* (img_org - img_meanshift)))/ (m*n);

% Part (b):  Single scale structural similarity index
ssim_error = zeros(5,1);

ssim_error(1) = ssim_index(img_org, img_blur);
ssim_error(2) = ssim_index(img_org, img_contrast);
ssim_error(3) = ssim_index(img_org, img_impulse);
ssim_error(4) = ssim_index(img_org, img_jpg);
ssim_error(5) = ssim_index(img_org, img_meanshift);

% Part (c): SSIM without luminance
ssim_error_1c = zeros(5,1);

ssim_error_1c(1) = ssim_index_woLuminance(img_org, img_blur);
ssim_error_1c(2) = ssim_index_woLuminance(img_org, img_contrast);
ssim_error_1c(3) = ssim_index_woLuminance(img_org, img_impulse);
ssim_error_1c(4) = ssim_index_woLuminance(img_org, img_jpg);
ssim_error_1c(5) = ssim_index_woLuminance(img_org, img_meanshift);

% Part (d): New SSIM index
ssim_error_1d = zeros(5,1);

ssim_error_1d(1) = ssim_q1d(img_org, img_blur);
ssim_error_1d(2) = ssim_q1d(img_org, img_contrast);
ssim_error_1d(3) = ssim_q1d(img_org, img_impulse);
ssim_error_1d(4) = ssim_q1d(img_org, img_jpg);
ssim_error_1d(5) = ssim_q1d(img_org, img_meanshift);