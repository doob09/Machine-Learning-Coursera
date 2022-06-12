function  p = predictOneVsAll(all_theta,X)

m = size(X,1);
num_labels = size(all_theta,1);

#size: 5000x1
p = zeros(size(X,1),1);

#add bias
X = [ones(m,1) X];

#X: 5000x401  * theta: 401x10
raw_output = X * all_theta';
prob_output = sigmoid(raw_output);
[prob,p] = max(prob_output,[],2);



end