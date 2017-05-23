% Train Perceptron
run('../part-b/part_5b.m')
clc; close;
clearvars -except w n

%% Get influential features

clc;
% Load word list
vocab_list = cell(n); 
fid   = fopen('../../data/imdb_vocab.csv');
for i = 1:1:n
    tline = fgetl(fid);
    vocab_list{i} = tline;
end
fclose(fid);
        
% Sort weights
w_abs = abs(w);
[sortvals, sortidx] = sort(w_abs,'descend');

% Print top 10 influential features
fprintf('Most Influential Features:\n');
for i = 1:1:10
    ind = sortidx(i);
    fprintf('%d. %s\n', i, vocab_list{ind});
end