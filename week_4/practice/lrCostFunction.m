function [J grad] = lrCostFunction(theta,X, y, lambda)

#5000 data points
m = length(y);

J= 0;
grad = zeros(size(theta));


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

error_term = prob_output .- y;
theta_0 = (1/m)  * [ X(:,1)' * error_term ]
thetas = [(1/m) * [ X(:,2:end)' * error_term] ] + [(lambda/m) .* theta(2:end,:)]

grad = vertcat(theta_0,thetas);
end