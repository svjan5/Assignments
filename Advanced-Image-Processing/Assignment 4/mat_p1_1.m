% Part(a) and (b)

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
% r = 5
dist_mask = dist_pos < r;

W = dist_fea .* dist_pos .* dist_mask;
D = diag(sum(W,2));


display('real game');
[V, ~] = eigs(D-W, D, 2, 'sa');
display('done');

%nara Segment Images
clc
eig_v2 = V(:,2);
% tol = 0;
tol = mean(eig_v2);

s1 = eig_v2 < tol;
seg1 = reshape(s1, n, m); seg1 = uint8(seg1)';
im_s1 = zeros(m,n,3);
im_s1 = uint8(im_s1);
im_s1(:,:,1) = seg1 .* im_org(:,:,1);
im_s1(:,:,2) = seg1 .* im_org(:,:,2);
im_s1(:,:,3) = seg1 .* im_org(:,:,3);

s2 = eig_v2 >= tol;
seg2 = reshape(s2, n, m); seg2 = uint8(seg2)';

im_s2 = zeros(m,n,3);
im_s2 = uint8(im_s2);
im_s2(:,:,1) = seg2 .* im_org(:,:,1);
im_s2(:,:,2) = seg2 .* im_org(:,:,2);
im_s2(:,:,3) = seg2 .* im_org(:,:,3);

% im = int(im);
fig = figure();
subplot(1,3,1); imshow(im_org, []); title('Original Image'); 
subplot(1,3,2); imshow(im_s1, []); title('Segment 1');
subplot(1,3,3); imshow(im_s2, []); title('Segment 2');

set(gcf,'PaperPositionMode','auto')
name = strcat('b', num2str(fnum), '_f_', num2str(sig_f), '_p_', num2str(sig_p), '_r_', num2str(r), '_tol_', num2str(tol),'.jpg');
print(fig,name,'-djpeg')

%% Part (c): Applying 2-cut twice 
clc;
total_num = sum(sum(s1));
X2_fea = zeros(total_num, c);
X2_pos = zeros(total_num, 2);

slog = reshape(s1, n , m);
slog = slog';

counter = 1;
for i = 1:m
    for j = 1:n
        if(slog(i,j) == 1)
            X2_fea(counter,1) = hsv_im(i,j,3);
            X2_fea(counter,2) = hsv_im(i,j,3) * hsv_im(i,j,2) * sin(2*pi*hsv_im(i,j,1));
            X2_fea(counter,3) = hsv_im(i,j,3) * hsv_im(i,j,2) * cos(2*pi*hsv_im(i,j,1));
            X2_pos(counter,:) = [i,j];
            counter = counter + 1;
        end
    end
end


sig_f = 50;
sig_p = 30;

dist2_fea = pdist2(X2_fea, X2_fea, 'euclidean');
dist2_fea = exp(-dist2_fea / sig_f);
dist2_pos = pdist2(X2_pos, X2_pos, 'euclidean');
dist2_pos = exp(-dist2_pos / sig_p);
r = mean(mean(dist2_pos));
dist2_mask = dist2_pos < r;

W2 = dist2_fea .* dist2_pos .* dist2_mask;
D2 = diag(sum(W2,2));

display('real game');
[V2, ~] = eigs(D2-W2, D2, 2, 'sa');
display('done');



eig2_v2 = V2(:,2);
tol = mean(eig2_v2);
% tol = 0;

s12 = eig2_v2 < tol;
seg12 = zeros(m,n);

counter = 1;
for i = 1:m
    for j = 1:n
        if(slog(i,j) == 1)
            seg12(i,j) = s12(counter);
            counter = counter + 1;
        end
    end
end

% seg12 = uint8(seg12);

s22 = eig2_v2 >= tol;
seg22 = zeros(m,n);

counter = 1;
for i = 1:m
    for j = 1:n
        if(slog(i,j) == 1)
            seg22(i,j) = s22(counter);
            counter = counter + 1;
        end
    end
end

seg12 = uint8(seg12);
seg22 = uint8(seg22);
% Create the final segments
im_s12 = zeros(m,n,3);
im_s12 = uint8(im_s12);

for i = 1:m
    for j = 1:n
        if(slog(i,j) == 1)
            im_s12(:,:,1) = seg12 .* im_org(:,:,1);
            im_s12(:,:,2) = seg12 .* im_org(:,:,2);
            im_s12(:,:,3) = seg12 .* im_org(:,:,3);
        end
    end
end

im_s22 = zeros(m,n,3);
im_s22 = uint8(im_s22);

for i = 1:m
    for j = 1:n
        if(slog(i,j) == 1)
            im_s22(:,:,1) = seg22 .* im_org(:,:,1);
            im_s22(:,:,2) = seg22 .* im_org(:,:,2);
            im_s22(:,:,3) = seg22 .* im_org(:,:,3);
        end
    end
end

% im = int(im);
fig = figure();
subplot(1,3,1); imshow(im_s1, []);% title('Original Image'); 
subplot(1,3,3); imshow(im_s12, []);% title('Segment 1'); 
subplot(1,3,2); imshow(im_s22, []);% title('Segment 2');

name = strcat('seg2b', num2str(fnum), '_f_', num2str(sig_f), '_p_', num2str(sig_p), '_r_', num2str(r), '_tol_', num2str(tol),'.jpg');
print(fig,name,'-djpeg')

[sound_y, sound_Fs] = audioread('/home/shikhar/Music/alert_signal.wav');
sound(sound_y, sound_Fs);