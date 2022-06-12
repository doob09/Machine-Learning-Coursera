function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
#vectorization way
# y shape: 100x1
#log will help to blame for mistake
#log(sigmoid(theta' * X))

#X * theta : 100x3 * 3x1 = 100x1
raw_output = X * theta;
# squiz the result in between 0 and 1 as probability. 100x1
prob_output = sigmoid(raw_output);
#1*100 * 100x1 = 1x1
left_term = -y' * log(prob_output);

#100x1
one_minus_prob = 1 .- prob_output;
# 1x100 *   100x1 = 1x1
right_term = (1 .-y)' * log(one_minus_prob);

J = (1/m) * (left_term - right_term); 

#hypothesis will include the sigmoid as probability
# 100x1
error_term = prob_output .- y;

# 3x100  * 100x1
dot_product = X' * error_term ;
#output shape: 3x1
grad = 1/m * dot_product;
% =============================================================

end
