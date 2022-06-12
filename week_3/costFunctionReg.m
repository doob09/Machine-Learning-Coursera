function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

#X* theta = 118
prob_output = sigmoid(X * theta);

#1x118 * 118x1 = 1x1
left_term = -y' * log(prob_output) 

right_term = (1.-y)' * log(1 .-prob_output)

theta_from_second = theta(2:end,:) ;
lambda_term = lambda/(2*m) * (theta_from_second' * theta_from_second )

J = [ (left_term - right_term) / m] + lambda_term

#mx1
error_term = prob_output .-y;
#118x1
bias_term = X(:,1);
#1x118 * 118x1
theta_0 = 1/m * (error_term' * bias_term)


#shape: 27x1
reg_term = lambda/m * theta(2:end,:);
#X without bias_term shape: 128x27
#error_term' : 1x128
theta_others =  [ 1/m * ( error_term' * X(:,2:end) )]' + reg_term;

grad = vertcat(theta_0,theta_others)

% =============================================================

end
