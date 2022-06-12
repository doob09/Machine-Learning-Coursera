function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

preds = X * theta;
errors = preds - y;
#total errors
sum_errors = sum(errors.^2);
#average errors
avg_errors = sum_errors / (2*m);
#do not count theta_0
reg_term = [lambda/(2*m)] * sum(theta(2:end,:).^2);
J = avg_errors + reg_term;

#calculate partial derivatie of regularized linear regression'cost function
#let theta_0 is 0 since we do not count theta_0 
#octave count from index 1 not 0
theta(1,:)  = 0;

reg_grad = (lambda/m) .* theta;
grad  = [(1/m) * (X' * errors) ]+ reg_grad;



% =========================================================================

grad = grad(:);

end
