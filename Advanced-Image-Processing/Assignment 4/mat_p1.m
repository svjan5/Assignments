clc; clear;
fnum = 5;
fname = strcat('./Images/', num2str(fnum), '.jpg');
im = imread(fname);
[m,n,c] = size(im);

im_org = im;
im = double(im);
X_fea = zeros(m*n,c);
X_pos = zeros(m*n,2);

hsv_im = rgb2hsv(im); 

counter = 1;
for i = 1:m
    for j = 1:n
        X_fea(counter,1) = hsv_im(i,j,3);
        X_fea(counter,2) = hsv_im(i,j,3) * hsv_im(i,j,2) * sin(2*pi*hsv_im(i,j,1));
        X_fea(counter,3) = hsv_im(i,j,3) * hsv_im(i,j,2) * cos(2*pi*hsv_im(i,j,1));
        X_pos(counter,:) = [i,j];
        counter = counter + 1;
    end
end

%%% Compute distances
clc
sig_f = 5;
sig_p = 1;

dist_fea = pdist2(X_fea, X_fea, 'euclidean');
dist_fea = exp(-dist_fea / sig_f);
dist_pos = pdist2(X_pos, X_pos, 'euclidean');
dist_pos = exp(-dist_pos / sig_p);

r = mean(mean(dist_pos));
dist_mask = dist_pos < r;

W = dist_fea .* dist_pos .* dist_mask;
D = diag(sum(W,2));

display('real game');
[V, ~] = eigs(D-W, D, 2, 'sa');
display('done');

% Segment Images
clc
eig_v2 = V(:,2);

tol = mean(eig_v2)
% tol = 0

s1 = eig_v2 < tol;
seg1 = reshape(s1, m, n); seg1 = uint8(seg1)';
im_s1 = zeros(m,n,3);
im_s1 = uint8(im_s1);
im_s1(:,:,1) = seg1 .* im_org(:,:,1);
im_s1(:,:,2) = seg1 .* im_org(:,:,2);
im_s1(:,:,3) = seg1 .* im_org(:,:,3);

s2 = eig_v2 >= tol;
seg2 = reshape(s2, m, n); seg2 = uint8(seg2)';

im_s2 = zeros(m,n,3);
im_s2 = uint8(im_s2);
im_s2(:,:,1) = seg2 .* im_org(:,:,1);
im_s2(:,:,2) = seg2 .* im_org(:,:,2);
im_s2(:,:,3) = seg2 .* im_org(:,:,3);

% im = int(im);
fig = figure();

ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 
1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
text(0.5, 1,'\bf Do you like this title?','HorizontalAlignment' ,'center','VerticalAlignment', 'top')
subplot(1,3,1); imshow(im_org, []);% title('Original Image'); 
subplot(1,3,3); imshow(im_s1, []);% title('Segment 1'); 
subplot(1,3,2); imshow(im_s2, []);% title('Segment 2');

name = strcat('b', num2str(fnum), '_f_', num2str(sig_f), '_p_', num2str(sig_p), '_r_', num2str(r), '_tol_', num2str(tol),'.jpg');
print(fig,name,'-djpeg')

[sound_y, sound_Fs] = audioread('/home/shikhar/Dropbox/alert_signal.wav');
sound(sound_y, sound_Fs);