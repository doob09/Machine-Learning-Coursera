data = load('ex1data1.txt');
%load feature
X = data(:,1);
%load label
y = data(:,2);
%count how many data points
m = length(y);

# show or plot the data
#rx as red cross marker, r:red , x:cross
#o circle , or : circle and red color
plot(X,y,'dr','MarkerSize',10);
ylabel('Profit in $10,000s');
xlabel('Population of City in 10,000s');


