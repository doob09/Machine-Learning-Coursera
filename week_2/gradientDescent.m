function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    #theta size : 2x1
    #preds shape: m x2 with 1 colums added for bias
    predictions = X * theta;
    #without square it will show the sign + or -
    #errors shape: mx1
    errors = predictions - y;
    
    #errors is a vector mx1 
    # X transpose: 2xm
    # internal term : sum as dot product. each error * coressponding data point
    internal = X' * errors ;
    #gradient shape: 2x1 = 2xm @ mx1 
    derivatives = 1/m * internal;
    
    #theta shape 2x1
    theta = theta - ( alpha * derivatives );

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
