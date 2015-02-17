function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% KMeans states norm^2 however, this additional squaring step does not
% change anything because norm is always semi positive.

%for i = 1:size(X,1)
%     mat = zeros(K,1);
%     for j = 1:K
%         mat(j) = norm(centroids(j,:)-X(i,:));
%     end
%    [~,idx(i)] = min(mat);
%end

for i = 1:size(X,1)
    [~,idx(i)] = min(sum((centroids - repmat(X(i,:), K, 1)) .^ 2, 2));
end

% =============================================================

end

