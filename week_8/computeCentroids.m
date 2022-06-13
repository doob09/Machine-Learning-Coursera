function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

cluster_idx = [];
cluster_label = [];

[r c] = size(centroids);
[r_idx c_idx] = size(idx);

#extract data points having same label
for i = 1:r
    cluster_label = ones(m,1);
    cluster_label = cluster_label * i ;
    cluster_idx = [cluster_idx idx==cluster_label];
end 

avg_C = []
for j= 1:r
    cluster_x = find(cluster_idx(:,j));
    cluster_x = X(cluster_x,:);
    avg_cluster = mean(cluster_x);
    avg_C = [avg_C ; avg_cluster];
end

centroids = avg_C;






% =============================================================


end

