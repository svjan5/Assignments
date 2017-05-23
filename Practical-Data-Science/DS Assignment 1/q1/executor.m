clear;clc;
addpath('./PROPACK/')
X = imread('fruit.pgm');
[n1,n2] = size(X);

X = double(X);
X = X / max(max(X));


sparsity_values = [.1 .2 .3 .4 .5 .6 .7 .8 .9];
svt_fro = zeros(length(sparsity_values),1);
isvd_fro = zeros(length(sparsity_values),1);

% 
% for k = 1:length(sparsity_values)
%     Mask = double(rand(n1,n2) > sparsity);
%     X_temp = X .* Mask;
%     tau = 15;
%     sparsity = sparsity_values(k);
%     tic();
%     X_svt = svt(X_temp, Mask);
%     time_svt = toc();
%     tic();
%     X_isvd = isvd(X_temp, Mask);
%     time_isvd = toc();
%     fprintf('Time Sparsity:%f svt: %f, isvd: %f\n',sparsity, time_svt, time_isvd );
%     svt_fro(k) = norm(X-X_svt, 'fro');
%     isvd_fro(k) = norm(X-X_isvd, 'fro');
% end
% 
% plot(10:10:90, svt_fro, 10:10:90, isvd_fro);
% title('Comparing SVT and ISVD');
% xlabel('% of density of data Matrix');
% ylabel('Frobenius Norm');
% ylabel('||X_{org}-X_{apprx}||_F', 'interpreter', 'tex');
% legend('SVT', 'ISVD');
% 
% [sound_y, sound_Fs] = audioread('/home/shikhar/Music/alert_signal.wav');
% sound(sound_y, sound_Fs);
% 
% return;

sparsity_values = [.9];
for k = 1:length(sparsity_values)
    sparsity = sparsity_values(k);
    Mask = double(rand(n1,n2) > sparsity);
    X_temp = X .* Mask;
    tau = 15;
    sparsity = sparsity_values(k);
    X_svt = svt(X_temp, Mask);
    X_isvd = isvd(X_temp, Mask);
    figure();
    
    subplot(1,3,1); imshow(X_temp); title('Original')
    subplot(1,3,2); imshow(X_svt); title('SVT');
    subplot(1,3,3); imshow(X_isvd); title('ISVD');
    pause;
end
