function ssim_index = ssim_q1d(img1, img2)

% Check if the size of images are same
if (size(img1) ~= size(img2))
   ssim_index = -Inf;
   return;
end


window = fspecial('gaussian', 11, 1.5);
L = 255;
K = [0.01; 0.03];
[M,N] = size(img1);

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

sz = 5;
size(img1);
size(mu1);

mse_new = zeros(M,N);
count = 0;
for i = sz+1:M-sz
	for j = sz+1:N-sz
		num_temp = 0;
		for k = -sz:sz
			for l = -sz:sz
                		a = img1(i+k,j+l) - mu1(i-sz,j-sz) - img2(i+k, j+l) + mu2(i-sz,j-sz);
				num_temp = num_temp + window(k+sz+1,l+sz+1) * (a*a);
			end
		end
		den = mu1_sq(i-sz,j-sz) + mu2_sq(i-sz,j-sz) + C2;
		mse_new(i,j) = num_temp / den;
		count = count + 1;
% 		i,j
	end
end
% count
% mse_new
ssim_index = mse_new
% ssim_index = mse_new / count
return;