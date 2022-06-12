data = load('ex1data1.txt');
%load feature
X = data(:,1);
%load label
y = data(:,2);
%count how many data points
m = length(y);
% disp(m)
# show or plot the data
#rx as red cross marker, r:red , x:cross
#o circle , or : circle and red color
% plot(X,y,'dr','MarkerSize',10);
% ylabel('Profit in $10,000s');
% xlabel('Population of City in 10,000s');

##### 
#Computing the Cost or Loss with  initialized random values of theta 
#Learning rate, iteration as Hyperparameter
#####

#add column theta_0 or bias term to the data
#create a vector m rows filled with one
# new size mx2
X = [ones(m,1),data(:,1)];

#initialize theta start 0
# both theta_0, theta_1 are 0 
# size : 2x1
theta = zeros(2,1);

iterations= 1500;
#learning rate
#and learning rate will try to hellp reduce the Loss in next step
#as velocity in update phase
alpha = 0.01;

#Compute the Cost:
J = 0;

#get all hypothesis can be thought as sum
# mx2 @ 2x1 = mx1 matech size of y as mx1
preds = X * theta ;
disp(size(preds));

#vector mx1
sqrErrors = (preds - y).^2;
#first time calculate the loss
# average loss
J = 1/(2*m) * sum(sqrErrors)

#update the prama or weight to have new Loss