clear; clc;
X = importdata('../data/synthetic/syndata2.txt');
K = importdata('../data/synthetic/syndata2_kernel.txt');
Y = importdata('../data/synthetic/syndata2_lab.txt');

n = size(X,1);
k = 5;

S = zeros(n,k);
D = zeros(n,k);

% Initialize
rand_num = randsample(n, k);
S(1:100, 1) = 1;
S(101:200, 2) = 1;
S(201:300, 3) = 1;
S(301:400, 4) = 1;
S(401:500, 5) = 1;

count = 0;
while(count <= 1000)
    S_old = S;
    for i = 1:n
        for j = 1:k
            D(i,j) = K(i,i);
            ind = find(S(:,j) == 1);
            sz = sum(S(:,j));
            sz_sq = sz*sz;
            
            for p = ind'
                for q = ind'
                    D(i,j) = D(i,j) + K(p,q)/ sz_sq;
                end
                D(i,j) = D(i,j) - 2*K(i,p) / sz;
            end
        end
        
    end
    
    [~, new_labels] = min(D, [], 2);
    
    S = zeros(n,k);
    for i = 1:n
        S(i, new_labels(i)) = 1;
    end
    
    if( norm(S-S_old, 'fro') == 0)
        break
    end
    
    count = count+1
end

labels = zeros(n,1);
for i = 1:k
    labels(S(:,i) == 1) = i;
end
scatter(X(:,1), X(:,2), [],labels, 'filled')

% Computing RAND score
rand = RAND(labels,Y);
fprintf('RAND score kernel k-means: %f', rand)