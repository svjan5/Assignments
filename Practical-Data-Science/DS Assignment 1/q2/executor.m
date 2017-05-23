clc; clear;

load Features.mat
X = X';
[m,n] = size(X);

num = 5;

% Load word list
vocab_list = cell(m,1); 
fid   = fopen('./vocab_list.txt');
for i = 1:1:m
    tline = fgetl(fid);
    vocab_list{i} = tline;
end
fclose(fid);

taus = [1000 1 1/1000];
tol = 0.000001;

for iter = 1:length(taus)
    tau = taus(iter);
    
    tic();
    is_nndsvd = 0;
    maxIters = 100;
    [W,H] = nmr_2(X, num, maxIters, tau, is_nndsvd);
    toc()
    
    if (is_nndsvd == 0) fprintf('Initalization: Random\n');
    else fprintf('Initalization: SVD\n'); end
    
    App_X = W*H;
    diff = norm(X- App_X, 'fro');
    fprintf('Tau: %f, MaxIters: %d, Frobenius: %f \n', tau, maxIters, diff);

    fprintf('sparsity W: %d, H: %d\n',  sum(sum(W < tol)) * 100.0 / (size(W,1)*size(W,2)), sum(sum(H < tol)) * 100.0 / (size(H,1)*size(H,2)));

    % Print Topic Matrix
    Topic_mat = cell(10,num);
    
    for k = 1:num
        % Sort weights
        w = W(:,k);
        [sortvals, sortidx] = sort(w,'descend');

        % Print top 10 influential features
        for i = 1:1:10
            ind = sortidx(i);
            Topic_mat(i,k) = vocab_list(ind);  
        end
    end
    
    Topic_mat
end

[sound_y, sound_Fs] = audioread('/home/shikhar/Music/alert_signal.wav');
sound(sound_y, sound_Fs);