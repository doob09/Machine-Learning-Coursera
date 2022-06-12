function [all_theta] = oneVsAll(X,y,num_labels,lambda)

#mxn 
[m,n] = size(X)

#size: 10x401
all_theta = zeros(num_labels, n+1);
[s,k] = size(all_theta)

#add bias term to X
#size 5000x 401
X = [ones(m,1) X];

#set initial theta
initial_theta = zeros(n+1, 1);

#set options for fminunc
options = optimset('GradObj','on','MaxIter',50);

#get theta
for c = 1:num_labels
    theta(c,:)= ...
        fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                initial_theta, options);
end

end