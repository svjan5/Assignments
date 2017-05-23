clear; clc;
X = importdata('../data/synthetic/syndata2.txt');
Y = importdata('../data/synthetic/syndata2_lab.txt');
k = 5;

[n,d] = size(X);

c_centers = rand(k,d);
new_c_centers = zeros(k,d);

prev_labels = ones(n,1);
prev_dist = 0;

while(1)
    dist_mat = pdist2(X, c_centers, 'euclidean');
    [~, new_labels] = min(dist_mat, [], 2);
     
    for i = 1:k
        Xi = X(new_labels == i, :);
        new_c_centers(i,:) = sum(Xi)/size(Xi,1);
    end
    
    dist = pdist2(c_centers, new_c_centers, 'euclidean');
    new_dist = trace(dist);
    abs(prev_dist - new_dist)
    
    if( abs(prev_dist - new_dist) < 0.01 )
        break;
    else
        prev_dist = new_dist;
    end
    
end

scatter(X(:,1), X(:,2), [],new_labels, 'filled')

% Computing RAND score
rand = RAND(new_labels,Y);
fprintf('RAND score K-means: %f', rand)