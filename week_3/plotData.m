function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

#find positive and negative in label
#find function will return the index in form of vector where condition is true 
pos = find(y==1);
neg = find(y==0);

#get all the data point with label 1  using index from pos 
% t = [X(pos,1), X(pos,2) , y(pos,1)]

#show all admitted data point or label as 1
#plot x-axis: ft_1 , y-axis:ft_2 points which is label as 1
#k+: black crosshair
#bd: blue diamond

plot(  X(pos,1) , X(pos,2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);

#show all unadmitted points with label 0
#using vector neg containing index of all 0
#MarkerFaceColor : give color to data point as yellow
plot( X(neg,1) , X(neg,2), 'ko', 'MarkerFaceColor','y','MarkerSize',7);






% =========================================================================



hold off;

end
