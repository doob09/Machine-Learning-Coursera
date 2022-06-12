function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
#cost function
#X * theta = 5000x400 * 400x1
#y = 5000x1 => y'= 1x5000
prob_output = sigmoid(X * theta);
#get the entropy
left_term = -y' * log(prob_output)
right_term = (1 .-y)' * log(1 .- prob_output)
# does not count bias 
reg_term = (lambda/(2*m)) * [ theta(2:end,:)' * theta(2:end,:) ]
J =  [ (1/m) * (left_term - right_term)  ] + reg_term

#mx1
error_term = prob_output .-y;
#118x1
bias_term = X(:,1);
#1x118 * 118x1
theta_0 = 1/m * (error_term' * bias_term);


#shape: 27x1
reg_dri_term = lambda/m * theta(2:end,:);
#X without bias_term shape: 128x27
#error_term' : 1x128
theta_others =  [ 1/m * ( error_term' * X(:,2:end) )]' + reg_dri_term;

grad = vertcat(theta_0,theta_others)




% =============================================================

grad = grad(:);

end
