clear; clc;

addpath('/home/shikhar/Dropbox/IISC/Data mining/DS Assignment 1/q1')
X = imread('fruit.pgm');
X = double(X);
[m,n] = size(X);
num = min(n,m);
[U,S,V] = svd(X);
k = 20;

fro_norm = zeros(num,1);
values = [1,5,10,15,20,30,50,75,100,150,250];

for iter = 1:length(values);
    k = values(iter);
    X_app = U(:,1:k) * S(1:k, 1:k) * V(:,1:k)';
    diff = X - X_app;
    fro_norm(k) = norm(diff, 'fro');
%     cmap = colormap('gray');
    
%     X_app = imadjust(X_app, []);
    imshow(X_app, []);
    pause
%     X_app = X_app;
%     imwrite(X_app, strcat('./images/img_', int2str(k), '.jpg'), 'jpg','BitDepth', 8);
end

% % plot(1:num, fro_norm);
% % xlabel('rank k');
