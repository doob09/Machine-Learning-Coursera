function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

#add bias term -> shape : 5000x401
X = [ones(m,1) X];

# 25x401
size(Theta1);
#10x26
size(Theta2);

#need one-hot y
encoded_y = (1:num_labels) == y;

###### Forward pass
#layer_2
#5000x401 * 401x25 = 5000x25
z_2= X * Theta1';
a_2= sigmoid(z_2);

#require to add bias term at output layer 2
#5000x26
a_2 = [ones(m,1) a_2];

#final layer
# 5000x26 * 26x10 = 5000x 10 -> each data point: 10 output precdiction 
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

#to do element wise since out put is 10 not 1
left_term = -encoded_y .* log(a_3);
right_term = (1 .-encoded_y) .* log(1 .- a_3);

#without regularization term
#first sum all rows : 5000 rows to 10
first_sum = sum(left_term - right_term);
#sencond sum all columns 10 to 1
J = (1/m) * [sum(first_sum)];


## with regularization 
#group list of theta at layer 1->2 - do not count theta_0 as bias term  
#group list of theta at layer 2-> output - do not count theta_0 as bias term
#since it is a matrix not vector => square elements-wise

#shape: 25x400 ( cut 1 as first column) -> double sum : do sum column first and then rows
reg_layer_2 = sum(sum(Theta1(:,2:end).^2));
#shape: 10x25 -> sum => 1
reg_layer_3 = sum(sum(Theta2(:,2:end).^2)) ;
reg_term = (lambda/(2*m)) * (reg_layer_2 + reg_layer_3);

#final J as scalar
J = J + reg_term


###### BACK PROP
#calculate Error at each layer

#error at final output layer
#a_3 shape: 5000x10  - encoded_y: 5000x10 = 5000x10
delta_3 = a_3 .- encoded_y;


#g prime shape: 5000 x25 if using z_2
# using a_2 will have  5000x26 with bias term
g_prime = a_2 .* (1-a_2);
#deta_3 shape: 5000x10 * Theta2 shape: 10x26 = 5000x26
#need to remove bias term from a_2 !! 

delta_2 = (delta_3 * Theta2) .* g_prime;


######gradients - 

#reg_grad_2 shape: 10x26 
#reg_term with and without Theta_0 =0
Theta_2= [zeros(size(Theta2),1) Theta2(:,2:end)];
reg_grad_2 = (lambda/m) .* Theta_2; 
#delta_3': 10x5000 * a_2 shape: 5000x26  = 10x26
# NOT SURE SHAPE !!!! 26x10 or 10x26
# 10 list of [--theta--]= > help to unroll
Theta2_grad = [ (1/m) .* (delta_3' * a_2) ].+ reg_grad_2;


#25 list of Thetas. Each Theta list have 401
#reg_term with Theta_0 =0
Theta1 = [zeros(size(Theta1),1) Theta1(:,2:end)];
reg_grad_1 = (lambda/m) .* Theta1;
% Remove bias term: 26 nodes -> 25 nodes
#delta_2': 25x5000 * X shape: 5000 x 401 = 25*401
Theta1_grad = [ (1/m) .* delta_2(:,2:end)' * X ] .+ reg_grad_1 ;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
