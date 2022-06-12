function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

cross_eval = [];
for i = 1: length(C_list) 
    for j=1:length(sigma_list)
        # @to define anonymous,callback function
        model = svmTrain(X,y,C_list(i),@(x1,x2) gaussianKernel(x1,x2,sigma_list(j)) );
        predictions = svmPredict(model,Xval);
        cross_eval(i,j) = mean(double(predictions ~= yval));
    end
end

disp(cross_eval)
#find the smallet value in cross validation result matrix 8x8

% #get the col
[val,col_idx] = min(min(cross_eval))
#search for that col to get row index
[val,row_idx] = min(cross_eval(:,col_idx))


C = C_list(row_idx)
sigma = sigma_list(col_idx)

% =========================================================================

end
