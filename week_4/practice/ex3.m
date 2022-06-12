#set up 
#input size
input_layer_size = 400;
#10 classes
num_labels = 10;


#######load data 
#size 5000x400
load("ex3data1.mat");

#m data points 5000
m = size(X,1);

#choose random data to show
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

% displayData(sel);

% fprintf('Program paused. Press enter to continue.\n');
% pause;

#-----------------Cost function ------------------------
% Test case for lrCostFunction
fprintf('\nTesting lrCostFunction() with regularization\n');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;

[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2b: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

% ================ Part 3: Predict for One-Vs-All ===========
pred = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);