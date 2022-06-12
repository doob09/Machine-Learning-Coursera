function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
#z = X * theta . output of linear into sigmoid function 
#return value close to 1 if z=X* theta  is large postive
# close to 0 if z is large negative 
g = 1 ./ (1 + exp(-z));

% =============================================================

end
